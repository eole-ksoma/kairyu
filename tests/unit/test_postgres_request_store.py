"""Real-PostgreSQL tests for the durable asynchronous request store."""

from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from kairyu.async_requests import (
    AsyncRequestError,
    AsyncRequestState,
    AsyncRequestSubmission,
    IdempotencyConflictError,
    RequestCapacityError,
    RequestStoreProtocol,
    StaleRequestClaimError,
)
from kairyu.async_requests.postgres_store import PostgresRequestStore
from kairyu.async_requests.worker import AsyncRequestWorker
from kairyu.engine.mock import MockBackend

_POSTGRES_DSN = os.environ.get("KAIRYU_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.postgres

if _POSTGRES_DSN:
    psycopg = pytest.importorskip("psycopg")
else:  # Keep core-only test collection importable.
    psycopg = None


@pytest.fixture
def store_factory():
    assert _POSTGRES_DSN is not None
    store_id = f"pytest-requests-{uuid.uuid4().hex}"
    stores: list[PostgresRequestStore] = []

    def create(**kwargs) -> PostgresRequestStore:
        store = PostgresRequestStore(_POSTGRES_DSN, store_id=store_id, **kwargs)
        stores.append(store)
        return store

    yield create, store_id

    for store in reversed(stores):
        store.close()
    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            "DELETE FROM async_request_store_registry WHERE store_id = %s",
            (store_id,),
        )


def submission(**updates) -> AsyncRequestSubmission:
    values = {
        "owner": "tenant-a",
        "endpoint": "/v1/chat/completions",
        "body": {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
    }
    values.update(updates)
    return AsyncRequestSubmission(**values)


def test_cross_instance_idempotency_tenant_scope_and_listing(store_factory) -> None:
    create, _store_id = store_factory
    first = create()
    second = create()
    intent = submission(idempotency_key="order-42", metadata={"source": "test"})

    with ThreadPoolExecutor(max_workers=2) as executor:
        submitted = [
            future.result()
            for future in (
                executor.submit(first.submit, intent),
                executor.submit(second.submit, intent),
            )
        ]

    assert isinstance(first, RequestStoreProtocol)
    assert submitted[0] == submitted[1]
    request = submitted[0]
    assert second.get(request.id, owner="tenant-a") == request
    with pytest.raises(KeyError):
        second.get(request.id, owner="tenant-b")
    with pytest.raises(IdempotencyConflictError):
        second.submit(
            submission(
                idempotency_key="order-42",
                body={"model": "different", "messages": []},
            )
        )

    other_owner = second.submit(submission(owner="tenant-b", idempotency_key="order-42"))
    assert other_owner.id != request.id
    assert [item.id for item in first.list(owner="tenant-a")] == [request.id]
    assert [item.id for item in first.list(owner="tenant-b")] == [other_owner.id]
    status = first.get_status(request.id, owner="tenant-a")
    assert status.id == request.id
    assert status.has_result is False
    assert "body" not in status.model_dump()
    listed_statuses = first.list_statuses(owner="tenant-a")
    assert [item.id for item in listed_statuses] == [request.id]
    assert all("result" not in item.model_dump() for item in listed_statuses)
    pending_status, pending_result = first.get_result(request.id, owner="tenant-a")
    assert pending_status.state is AsyncRequestState.QUEUED
    assert pending_result is None


def test_metrics_snapshot_is_shared_aggregate_and_totals_are_durable(
    store_factory,
) -> None:
    create, store_id = store_factory
    first = create()
    second = create()
    running_request = first.submit(submission(idempotency_key="running"))
    expiring_request = first.submit(
        submission(
            owner="tenant-b",
            idempotency_key="expired",
            deadline_at=datetime.now(UTC) + timedelta(minutes=5),
        )
    )
    first.submit(submission(owner="tenant-c", idempotency_key="queued"))
    claim = second.claim_next("worker-a", lease_seconds=30)
    assert claim is not None
    assert claim.request_id == running_request.id
    second.mark_running(claim)

    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            """
            UPDATE async_requests
            SET deadline_at = clock_timestamp() - interval '1 second'
            WHERE store_id = %s AND request_id = %s
            """,
            (store_id, expiring_request.id),
        )

    snapshot = first.metrics_snapshot()

    assert snapshot.queue_depth == 1
    assert snapshot.state_counts[AsyncRequestState.RUNNING] == 1
    assert snapshot.state_counts[AsyncRequestState.EXPIRED] == 1
    assert snapshot.oldest_queued_age_seconds >= 0
    assert snapshot.transition_counts["claim"] == 1
    assert snapshot.transition_counts["running"] == 1
    assert snapshot.attempts_total == 1

    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            "DELETE FROM async_request_claim_audit WHERE store_id = %s",
            (store_id,),
        )
    after_audit_retention = second.metrics_snapshot()
    assert after_audit_retention.transition_counts["claim"] == 1
    assert after_audit_retention.transition_counts["running"] == 1
    first.migrate_metrics_during_maintenance()
    after_rebuild = first.metrics_snapshot()
    assert after_rebuild.transition_counts["claim"] == 1
    assert after_rebuild.transition_counts["running"] == 1
    restarted = create()
    after_restart = restarted.metrics_snapshot()
    assert after_restart.transition_counts["claim"] == 1
    assert after_restart.transition_counts["running"] == 1


def test_metrics_totals_backfill_requires_explicit_maintenance(store_factory) -> None:
    create, store_id = store_factory
    first = create()
    first.submit(submission())
    claim = first.claim_next("worker-a", lease_seconds=30)
    assert claim is not None

    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            "DELETE FROM async_request_metric_migrations WHERE store_id = %s",
            (store_id,),
        )
        connection.execute(
            "DELETE FROM async_request_metric_shards WHERE store_id = %s",
            (store_id,),
        )
        connection.execute(
            "DELETE FROM async_request_state_shards WHERE store_id = %s",
            (store_id,),
        )

    restarted = create()

    with pytest.raises(RuntimeError, match="maintenance migration"):
        restarted.metrics_snapshot()
    restarted.migrate_metrics_during_maintenance()
    assert restarted.metrics_snapshot().transition_counts["claim"] == 1


def test_maintenance_preserves_legacy_totals_after_all_history_is_deleted(
    store_factory,
) -> None:
    create, store_id = store_factory
    first = create()
    request = first.submit(submission())
    claim = first.claim_next("worker-a", lease_seconds=30)
    assert claim is not None

    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            "DELETE FROM async_request_claim_audit WHERE store_id = %s",
            (store_id,),
        )
        connection.execute(
            "DELETE FROM async_requests WHERE store_id = %s AND request_id = %s",
            (store_id, request.id),
        )
        connection.execute(
            "DELETE FROM async_request_metric_shards WHERE store_id = %s",
            (store_id,),
        )
        connection.execute(
            """
            INSERT INTO async_request_metric_totals (store_id, event, total)
            VALUES (%s, 'claim', 7)
            ON CONFLICT (store_id, event) DO UPDATE SET total = EXCLUDED.total
            """,
            (store_id,),
        )
        connection.execute(
            "DELETE FROM async_request_metric_migrations WHERE store_id = %s",
            (store_id,),
        )

    restarted = create()
    with pytest.raises(RuntimeError, match="maintenance migration"):
        restarted.metrics_snapshot()
    restarted.migrate_metrics_during_maintenance()
    assert restarted.metrics_snapshot().transition_counts["claim"] == 7


def test_unmigrated_store_transitions_remain_available(store_factory) -> None:
    create, store_id = store_factory
    first = create()
    request = first.submit(submission())
    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            "DELETE FROM async_request_metric_migrations WHERE store_id = %s",
            (store_id,),
        )
        connection.execute(
            "DELETE FROM async_request_metric_shards WHERE store_id = %s",
            (store_id,),
        )
        connection.execute(
            "DELETE FROM async_request_state_shards WHERE store_id = %s",
            (store_id,),
        )

    claim = first.claim_next("worker-a", lease_seconds=30)
    assert claim is not None
    assert claim.request_id == request.id
    first.mark_running(claim)
    first.cancel(request.id)
    first.migrate_metrics_during_maintenance()
    snapshot = first.metrics_snapshot()
    assert snapshot.state_counts[AsyncRequestState.CANCELLED] == 1
    assert snapshot.transition_counts["cancel"] == 1


def test_request_delete_decrements_state_metrics(store_factory) -> None:
    create, store_id = store_factory
    store = create()
    request = store.submit(submission())
    assert store.metrics_snapshot().queue_depth == 1

    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            "DELETE FROM async_requests WHERE store_id = %s AND request_id = %s",
            (store_id, request.id),
        )

    assert store.metrics_snapshot().queue_depth == 0


def test_cross_instance_owner_capacity_is_atomic_and_replays_survive_limit(
    store_factory,
) -> None:
    create, _store_id = store_factory
    first = create(max_records_per_owner=1)
    second = create(max_records_per_owner=1)
    intent = submission(idempotency_key="one")
    created = first.submit(intent)

    assert second.submit(intent) == created
    with pytest.raises(RequestCapacityError):
        second.submit(submission())
    assert second.submit(submission(owner="tenant-b")).owner == "tenant-b"


def test_defer_cools_down_owner_and_allows_another_tenant_to_claim(
    store_factory,
) -> None:
    create, _store_id = store_factory
    store = create()
    blocked = store.submit(submission(owner="tenant-a", priority=0))
    runnable = store.submit(submission(owner="tenant-b", priority=1))
    claim = store.claim_next("worker-a", lease_seconds=30)
    assert claim is not None
    assert claim.request_id == blocked.id
    store.mark_running(claim)

    deferred = store.defer(claim, delay_seconds=30)
    next_claim = store.claim_next("worker-b", lease_seconds=30)

    assert deferred.state is AsyncRequestState.QUEUED
    assert next_claim is not None
    assert next_claim.request_id == runnable.id
    assert "defer" in [row["event"] for row in store.export_claim_audit(blocked.id)]


def test_cross_instance_claim_is_atomic_and_terminal_publication_is_fenced(
    store_factory,
    monkeypatch,
) -> None:
    create, _store_id = store_factory
    submitter = create()
    worker_a = create()
    worker_b = create()
    request = submitter.submit(submission())

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(store.claim_next, worker_id, lease_seconds=30)
            for store, worker_id in (
                (worker_a, "worker-a"),
                (worker_b, "worker-b"),
            )
        ]
        claims = [future.result() for future in futures]

    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    claim = winners[0]
    owner = worker_a if claim.worker_id == "worker-a" else worker_b
    assert claim.request_id == request.id
    assert claim.request.attempt == 1
    running = owner.mark_running(claim)
    assert running.state is AsyncRequestState.RUNNING
    renewed = owner.renew_claim(claim, lease_seconds=60)
    assert renewed.request.state is AsyncRequestState.RUNNING
    finish_once = owner._finish_once

    def commit_then_lose_ack(*args, **kwargs):
        finish_once(*args, **kwargs)
        raise RuntimeError("simulated lost COMMIT acknowledgement")

    monkeypatch.setattr(owner, "_finish_once", commit_then_lose_ack)
    completed = owner.succeed(renewed, {"choices": [{"text": "done"}]})

    assert submitter.get(request.id) == completed
    assert completed.state is AsyncRequestState.SUCCEEDED
    with pytest.raises(StaleRequestClaimError):
        owner.fail(renewed, AsyncRequestError(code="late", message="late failure"))
    assert [row["event"] for row in submitter.export_claim_audit(request.id)] == [
        "claim",
        "running",
        "renew",
        "succeed",
    ]


def test_expired_lease_is_reclaimed_and_old_fence_cannot_publish(store_factory) -> None:
    create, store_id = store_factory
    first = create()
    second = create()
    request = first.submit(submission())
    old_claim = first.claim_next("worker-old", lease_seconds=30)
    assert old_claim is not None
    first.mark_running(old_claim)

    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            """
            UPDATE async_requests
            SET claimed_at = clock_timestamp() - interval '2 seconds',
                lease_until = clock_timestamp() - interval '1 second'
            WHERE store_id = %s AND request_id = %s
            """,
            (store_id, request.id),
        )

    new_claim = second.claim_next("worker-new", lease_seconds=30)
    assert new_claim is not None
    assert new_claim.request_id == request.id
    assert new_claim.fencing_token == old_claim.fencing_token + 1
    assert new_claim.request.attempt == 2
    with pytest.raises(StaleRequestClaimError):
        first.succeed(old_claim, {"stale": True})

    second.mark_running(new_claim)
    failed = second.fail(
        new_claim,
        AsyncRequestError(code="upstream_timeout", message="timed out", retryable=True),
    )
    assert failed.state is AsyncRequestState.FAILED
    assert [row["event"] for row in first.export_claim_audit(request.id)] == [
        "claim",
        "running",
        "reclaim",
        "running",
        "fail",
    ]


def test_deadline_precedes_replay_and_cancel_and_invalidates_claim(store_factory) -> None:
    create, store_id = store_factory
    first = create()
    second = create()
    intent = submission(
        idempotency_key="deadline",
        deadline_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    request = first.submit(intent)
    claim = first.claim_next("worker-a", lease_seconds=30)
    assert claim is not None

    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            """
            UPDATE async_requests
            SET deadline_at = clock_timestamp() - interval '1 second'
            WHERE store_id = %s AND request_id = %s
            """,
            (store_id, request.id),
        )

    with pytest.raises(IdempotencyConflictError):
        second.submit(
            submission(
                idempotency_key="deadline",
                deadline_at=intent.deadline_at,
                body={"model": "different", "messages": []},
            )
        )
    expired = second.get(request.id)
    assert expired.state is AsyncRequestState.EXPIRED
    assert second.cancel(request.id).state is AsyncRequestState.EXPIRED
    assert second.submit(intent).state is AsyncRequestState.EXPIRED
    with pytest.raises(StaleRequestClaimError):
        first.mark_running(claim)
    assert [row["event"] for row in first.export_claim_audit(request.id)] == [
        "claim",
        "expire",
    ]


def test_stale_claim_call_commits_deadline_expiry(store_factory) -> None:
    create, store_id = store_factory
    store = create()
    request = store.submit(
        submission(deadline_at=datetime.now(UTC) + timedelta(minutes=5))
    )
    claim = store.claim_next("worker-a", lease_seconds=30)
    assert claim is not None

    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            """
            UPDATE async_requests
            SET deadline_at = clock_timestamp() - interval '1 second'
            WHERE store_id = %s AND request_id = %s
            """,
            (store_id, request.id),
        )

    with pytest.raises(StaleRequestClaimError):
        store.renew_claim(claim, lease_seconds=30)

    with psycopg.connect(_POSTGRES_DSN) as connection:
        state = connection.execute(
            """
            SELECT state FROM async_requests
            WHERE store_id = %s AND request_id = %s
            """,
            (store_id, request.id),
        ).fetchone()[0]
    assert state == "expired"
    assert [row["event"] for row in store.export_claim_audit(request.id)] == [
        "claim",
        "expire",
    ]


def test_locked_due_request_does_not_block_other_tenants(store_factory) -> None:
    create, store_id = store_factory
    first = create()
    second = create()
    locked = first.submit(
        submission(deadline_at=datetime.now(UTC) + timedelta(minutes=5))
    )
    eligible = first.submit(submission(owner="tenant-b"))

    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as locker:
        locker.execute(
            """
            UPDATE async_requests
            SET deadline_at = clock_timestamp() - interval '1 second'
            WHERE store_id = %s AND request_id = %s
            """,
            (store_id, locked.id),
        )
        with locker.transaction():
            locker.execute(
                """
                SELECT request_id FROM async_requests
                WHERE store_id = %s AND request_id = %s
                FOR UPDATE
                """,
                (store_id, locked.id),
            )
            claim = second.claim_next("worker-b", lease_seconds=30)

    assert claim is not None
    assert claim.request_id == eligible.id
    assert first.get(locked.id).state is AsyncRequestState.EXPIRED


def test_list_projects_and_persists_deadlines_beyond_sweeper_batch(store_factory) -> None:
    create, store_id = store_factory
    store = create(max_records_per_owner=200)
    requests = [
        store.submit(
            submission(
                idempotency_key=f"expired-{index}",
                deadline_at=datetime.now(UTC) + timedelta(minutes=5),
            )
        )
        for index in range(105)
    ]

    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            """
            UPDATE async_requests
            SET deadline_at = clock_timestamp() - interval '1 second'
            WHERE store_id = %s
            """,
            (store_id,),
        )

    assert store.list(state=AsyncRequestState.QUEUED, limit=200) == []
    expired = store.list(state=AsyncRequestState.EXPIRED, limit=200)
    assert {request.id for request in expired} == {request.id for request in requests}

    with psycopg.connect(_POSTGRES_DSN) as connection:
        expired_count = connection.execute(
            """
            SELECT count(*) FROM async_requests
            WHERE store_id = %s AND state = 'expired'
            """,
            (store_id,),
        ).fetchone()[0]
    assert expired_count == 105


def test_list_includes_locked_rows_and_projects_expiry_without_blocking(
    store_factory,
) -> None:
    create, store_id = store_factory
    store = create()
    older = store.submit(submission(idempotency_key="older"))
    locked = store.submit(
        submission(
            idempotency_key="locked",
            deadline_at=datetime.now(UTC) + timedelta(minutes=5),
        )
    )

    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as locker:
        locker.execute(
            """
            UPDATE async_requests
            SET deadline_at = clock_timestamp() - interval '1 second'
            WHERE store_id = %s AND request_id = %s
            """,
            (store_id, locked.id),
        )
        with locker.transaction():
            locker.execute(
                """
                SELECT request_id FROM async_requests
                WHERE store_id = %s AND request_id = %s
                FOR UPDATE
                """,
                (store_id, locked.id),
            )
            listed = store.list(limit=2)
            expired = store.list(state=AsyncRequestState.EXPIRED, limit=2)

    assert [request.id for request in listed] == [locked.id, older.id]
    assert listed[0].state is AsyncRequestState.EXPIRED
    assert [request.id for request in expired] == [locked.id]


def test_closed_connections_are_reestablished(store_factory) -> None:
    create, _store_id = store_factory
    store = create()
    request = store.submit(submission())

    store._connection.close()
    assert store.get(request.id) == request
    store._lease_connection.close()
    claim = store.claim_next("worker-a", lease_seconds=30)
    assert claim is not None
    assert claim.request_id == request.id
    assert store._metrics_connection is None
    assert store.metrics_snapshot().transition_counts["claim"] == 1
    store._metrics_connection.close()
    assert store.metrics_snapshot().transition_counts["claim"] == 1


def test_metric_updates_are_sharded_across_request_ids(store_factory) -> None:
    create, store_id = store_factory
    store = create(max_records_per_owner=64)
    for index in range(32):
        store.submit(submission(owner=f"tenant-{index:02d}"))
    for index in range(32):
        claim = store.claim_next(f"worker-{index:02d}", lease_seconds=30)
        assert claim is not None

    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        row = connection.execute(
            """
            SELECT count(*), sum(total)
            FROM async_request_metric_shards
            WHERE store_id = %s AND event = 'claim'
            """,
            (store_id,),
        ).fetchone()

    assert row is not None
    assert int(row[0]) > 1
    assert int(row[1]) == 32
    assert store.metrics_snapshot().transition_counts["claim"] == 32


def test_new_schema_preserves_legacy_counter_during_rolling_upgrade(
    store_factory,
) -> None:
    create, store_id = store_factory
    first = create()
    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            "DROP TRIGGER IF EXISTS async_request_claim_audit_metric_total "
            "ON async_request_claim_audit"
        )
        connection.execute(
            """
            CREATE OR REPLACE FUNCTION increment_async_request_metric_total()
            RETURNS TRIGGER
            LANGUAGE plpgsql
            AS $$
            BEGIN
                INSERT INTO async_request_metric_shards (
                    store_id, event, shard, total
                ) VALUES (
                    NEW.store_id,
                    NEW.event,
                    mod(
                        hashtextextended(NEW.request_id, 0)
                            & 9223372036854775807,
                        64
                    )::smallint,
                    1
                )
                ON CONFLICT (store_id, event, shard)
                DO UPDATE SET total = async_request_metric_shards.total + 1;
                RETURN NEW;
            END;
            $$
            """
        )
        connection.execute(
            """
            CREATE TRIGGER async_request_claim_audit_metric_total
            AFTER INSERT ON async_request_claim_audit
            FOR EACH ROW
            EXECUTE FUNCTION increment_async_request_metric_total()
            """
        )
    try:
        # Starting the new code repairs the old same-name function before any
        # mixed-version event, while its separately named shard trigger remains.
        second = create()
        first.submit(submission(owner="tenant-a"))
        assert first.claim_next("worker-a", lease_seconds=30) is not None
        second.submit(submission(owner="tenant-b"))
        assert second.claim_next("worker-b", lease_seconds=30) is not None

        with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
            legacy_total = connection.execute(
                """
                SELECT total
                FROM async_request_metric_totals
                WHERE store_id = %s AND event = 'claim'
                """,
                (store_id,),
            ).fetchone()
        assert legacy_total is not None
        assert int(legacy_total[0]) == 2
        assert second.metrics_snapshot().transition_counts["claim"] == 2
    finally:
        with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
            connection.execute(
                "DROP TRIGGER IF EXISTS async_request_claim_audit_metric_total "
                "ON async_request_claim_audit"
            )


def test_schema_cutover_blocks_old_store_initializers_globally(store_factory) -> None:
    create, _store_id = store_factory
    first = create()
    legacy_store_id = f"legacy-{uuid.uuid4().hex}"
    assert psycopg is not None
    try:
        with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
            connection.execute(
                """
                INSERT INTO async_request_store_registry (store_id, schema_version)
                VALUES (%s, 1)
                """,
                (legacy_store_id,),
            )

        create()

        with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
            row = connection.execute(
                """
                SELECT schema_version
                FROM async_request_store_registry
                WHERE store_id = %s
                """,
                (legacy_store_id,),
            ).fetchone()
        assert row is not None
        assert int(row[0]) == 2
        assert first.metrics_snapshot().queue_depth == 0
    finally:
        with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
            connection.execute(
                "DELETE FROM async_request_store_registry WHERE store_id = %s",
                (legacy_store_id,),
            )


def test_cancel_and_success_race_commits_only_one_terminal_state(store_factory) -> None:
    create, _store_id = store_factory
    worker = create()
    canceller = create()
    request = worker.submit(submission())
    claim = worker.claim_next("worker-a", lease_seconds=30)
    assert claim is not None
    worker.mark_running(claim)
    barrier = threading.Barrier(2)

    def complete():
        barrier.wait()
        try:
            return worker.succeed(claim, {"answer": 42})
        except StaleRequestClaimError as error:
            return error

    def cancel():
        barrier.wait()
        return canceller.cancel(request.id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        complete_future = executor.submit(complete)
        cancel_future = executor.submit(cancel)
        outcomes = (
            complete_future.result(),
            cancel_future.result(),
        )

    stored = worker.get(request.id)
    assert stored.state in {AsyncRequestState.SUCCEEDED, AsyncRequestState.CANCELLED}
    terminal_states = {
        outcome.state for outcome in outcomes if not isinstance(outcome, Exception)
    }
    assert terminal_states == {stored.state}
    terminal_events = [
        row["event"]
        for row in worker.export_claim_audit(request.id)
        if row["event"] in {"succeed", "cancel"}
    ]
    assert len(terminal_events) == 1


async def test_async_worker_renews_and_publishes_through_postgres(store_factory) -> None:
    create, _store_id = store_factory
    store = create()
    backend = MockBackend(responses={"hello": "postgres result"}, latency_s=0.25)
    worker = AsyncRequestWorker(
        store,
        {"m": backend},
        lease_seconds=0.15,
        legacy_chat_models={"m"},
    )
    request = store.submit(
        submission(
            body={
                "model": "m",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    )

    assert await worker.process_next() is True
    completed = store.get(request.id)
    assert completed.state is AsyncRequestState.SUCCEEDED
    assert completed.result is not None
    assert completed.result["choices"][0]["message"]["content"] == "postgres result"
    events = [row["event"] for row in store.export_claim_audit(request.id)]
    assert events[:2] == ["claim", "running"]
    assert events[-1] == "succeed"
    assert "renew" in events[2:-1]
