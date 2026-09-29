from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from kairyu.async_requests import (
    AsyncRequestError,
    AsyncRequestState,
    AsyncRequestSubmission,
    IdempotencyConflictError,
    InMemoryRequestStore,
    InvalidRequestTransitionError,
    RequestCapacityError,
    RequestStoreProtocol,
    StaleRequestClaimError,
)


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 7, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(clock: Clock) -> InMemoryRequestStore:
    ids = iter(f"req-{index}" for index in range(100))
    return InMemoryRequestStore(store_id="test", clock=clock, id_factory=lambda: next(ids))


def submission(**updates) -> AsyncRequestSubmission:
    values = {
        "owner": "tenant-a",
        "endpoint": "/v1/chat/completions",
        "body": {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
    }
    values.update(updates)
    return AsyncRequestSubmission(**values)


def test_store_satisfies_runtime_protocol(store: InMemoryRequestStore) -> None:
    assert isinstance(store, RequestStoreProtocol)


def test_metrics_snapshot_reports_bounded_queue_state_and_transitions(
    store: InMemoryRequestStore,
    clock: Clock,
) -> None:
    first = store.submit(submission())
    store.submit(submission(owner="tenant-b"))
    clock.advance(3)
    claim = store.claim_next("worker-a", lease_seconds=30)
    assert claim is not None
    assert claim.request_id == first.id
    store.mark_running(claim)
    store.succeed(claim, {"answer": 42})
    clock.advance(2)

    snapshot = store.metrics_snapshot()

    assert snapshot.queue_depth == 1
    assert snapshot.state_counts[AsyncRequestState.QUEUED] == 1
    assert snapshot.state_counts[AsyncRequestState.SUCCEEDED] == 1
    assert snapshot.oldest_queued_age_seconds == 5
    assert snapshot.transition_counts["claim"] == 1
    assert snapshot.transition_counts["running"] == 1
    assert snapshot.transition_counts["succeed"] == 1
    assert snapshot.attempts_total == 1
    assert set(snapshot.transition_counts) == {
        "claim",
        "reclaim",
        "renew",
        "defer",
        "running",
        "succeed",
        "fail",
        "cancel",
        "expire",
    }


def test_metrics_snapshot_projects_due_deadlines_without_transition_side_effect(
    store: InMemoryRequestStore,
    clock: Clock,
) -> None:
    store.submit(submission(deadline_at=clock.now + timedelta(seconds=1)))
    clock.advance(2)

    snapshot = store.metrics_snapshot()

    assert snapshot.queue_depth == 0
    assert snapshot.state_counts[AsyncRequestState.EXPIRED] == 1
    assert snapshot.transition_counts["expire"] == 0

    request_id = store.list()[0].id
    assert store.get(request_id).state is AsyncRequestState.EXPIRED
    snapshot = store.metrics_snapshot()
    assert snapshot.transition_counts["expire"] == 1


def test_retention_is_disabled_by_default_and_releases_terminal_capacity(
    store: InMemoryRequestStore,
    clock: Clock,
) -> None:
    intent = submission(idempotency_key="retained-key")
    terminal = store.submit(intent)
    claim = store.claim_next("worker-a", lease_seconds=30)
    assert claim is not None
    store.mark_running(claim)
    store.succeed(claim, {"answer": 42})
    active = store.submit(submission(owner="tenant-b"))

    disabled = store.purge_retained_data(
        request_retention_seconds=None,
        audit_retention_seconds=None,
    )
    assert disabled.terminal_requests_deleted == 0
    assert store.get(terminal.id).state is AsyncRequestState.SUCCEEDED

    clock.advance(60)
    preview = store.purge_retained_data(
        request_retention_seconds=60,
        audit_retention_seconds=None,
        batch_size=1,
        dry_run=True,
    )
    assert preview.applied is False
    assert preview.terminal_requests_deleted == 1
    assert store.get(terminal.id).state is AsyncRequestState.SUCCEEDED

    applied = store.purge_retained_data(
        request_retention_seconds=60,
        audit_retention_seconds=None,
        batch_size=1,
    )
    assert applied.applied is True
    assert applied.terminal_requests_deleted == 1
    with pytest.raises(KeyError):
        store.get(terminal.id)
    assert store.get(active.id).state is AsyncRequestState.QUEUED
    replay_after_retention = store.submit(intent)
    assert replay_after_retention.id != terminal.id
    assert store.metrics_snapshot().transition_counts["succeed"] == 1


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"request_retention_seconds": 0}, "request_retention_seconds"),
        ({"audit_retention_seconds": float("nan")}, "audit_retention_seconds"),
        ({"dry_run": "false"}, "dry_run"),
        ({"batch_size": True}, "batch_size"),
        ({"batch_size": 10_001}, "batch_size"),
    ],
)
def test_retention_rejects_unbounded_policy_values(
    store: InMemoryRequestStore,
    updates,
    message: str,
) -> None:
    values = {
        "request_retention_seconds": None,
        "audit_retention_seconds": None,
        "batch_size": 500,
    }
    values.update(updates)
    with pytest.raises(ValueError, match=message):
        store.purge_retained_data(**values)


def test_retention_cutoff_includes_exact_boundary_only_after_ttl(
    store: InMemoryRequestStore,
    clock: Clock,
) -> None:
    before = store.submit(submission(deadline_at=clock.now))
    clock.advance(1)
    boundary = store.submit(submission(deadline_at=clock.now))
    clock.advance(1)
    after = store.submit(submission(deadline_at=clock.now))

    result = store.purge_retained_data(
        request_retention_seconds=1,
        audit_retention_seconds=None,
    )

    assert result.terminal_requests_deleted == 2
    with pytest.raises(KeyError):
        store.get(before.id)
    with pytest.raises(KeyError):
        store.get(boundary.id)
    assert store.get(after.id).state is AsyncRequestState.EXPIRED


def test_submit_get_and_list_are_tenant_scoped(store: InMemoryRequestStore) -> None:
    first = store.submit(submission())
    second = store.submit(submission(owner="tenant-b", priority=10))

    assert first.state is AsyncRequestState.QUEUED
    assert first.attempt == 0
    assert store.get(first.id, owner="tenant-a") == first
    with pytest.raises(KeyError):
        store.get(first.id, owner="tenant-b")
    assert [item.id for item in store.list(owner="tenant-a")] == [first.id]
    assert [item.id for item in store.list(owner="tenant-b")] == [second.id]


def test_idempotency_is_owner_scoped_and_rejects_payload_conflicts(
    store: InMemoryRequestStore,
) -> None:
    first = store.submit(submission(idempotency_key="checkout-42"))
    replay = store.submit(submission(idempotency_key="checkout-42"))
    other_owner = store.submit(
        submission(owner="tenant-b", idempotency_key="checkout-42")
    )

    assert replay == first
    assert other_owner.id != first.id
    with pytest.raises(IdempotencyConflictError):
        store.submit(
            submission(
                idempotency_key="checkout-42",
                body={"model": "different", "messages": []},
            )
        )


def test_owner_record_capacity_is_bounded_but_allows_idempotent_replay() -> None:
    bounded = InMemoryRequestStore(max_records_per_owner=1)
    intent = submission(idempotency_key="stable")
    first = bounded.submit(intent)

    assert bounded.submit(intent) == first
    with pytest.raises(RequestCapacityError):
        bounded.submit(submission())
    assert bounded.submit(submission(owner="tenant-b")).owner == "tenant-b"


def test_store_state_is_isolated_from_nested_input_and_output_mutation(
    store: InMemoryRequestStore,
) -> None:
    original = submission()
    request = store.submit(original)
    original.body["messages"][0]["content"] = "mutated after submit"

    claim = store.claim_next("worker-a", lease_seconds=30)
    assert claim is not None
    store.mark_running(claim)
    result = {"choices": [{"message": {"content": "stable"}}]}
    completed = store.succeed(claim, result)
    result["choices"][0]["message"]["content"] = "mutated after success"
    completed.body["messages"][0]["content"] = "mutated returned snapshot"

    stored = store.get(request.id)
    assert stored.body["messages"][0]["content"] == "hi"
    assert stored.result == {"choices": [{"message": {"content": "stable"}}]}


def test_concurrent_idempotent_submit_and_claim_have_single_winner(
    store: InMemoryRequestStore,
) -> None:
    intent = submission(idempotency_key="same")
    with ThreadPoolExecutor(max_workers=4) as executor:
        submitted = list(executor.map(lambda _index: store.submit(intent), range(4)))

    assert len({request.id for request in submitted}) == 1
    with ThreadPoolExecutor(max_workers=4) as executor:
        claims = list(
            executor.map(
                lambda index: store.claim_next(f"worker-{index}", lease_seconds=30),
                range(4),
            )
        )
    assert len([claim for claim in claims if claim is not None]) == 1


def test_claim_orders_by_priority_then_age_and_completes_once(
    store: InMemoryRequestStore,
) -> None:
    high = store.submit(submission(priority=0))
    low = store.submit(submission(priority=10))

    claim = store.claim_next("worker-a", lease_seconds=30)
    assert claim is not None
    assert claim.request_id == high.id
    assert claim.request.state is AsyncRequestState.CLAIMED
    assert claim.request.attempt == 1

    with pytest.raises(InvalidRequestTransitionError):
        store.succeed(claim, {"premature": True})

    running = store.mark_running(claim)
    assert running.state is AsyncRequestState.RUNNING
    completed = store.succeed(claim, {"answer": 42})
    assert completed.state is AsyncRequestState.SUCCEEDED
    assert completed.result == {"answer": 42}
    assert completed.completed_at is not None
    with pytest.raises(StaleRequestClaimError):
        store.succeed(claim, {"answer": 43})

    next_claim = store.claim_next("worker-a", lease_seconds=30)
    assert next_claim is not None
    assert next_claim.request_id == low.id


def test_expired_lease_is_reclaimed_with_a_new_fencing_token(
    store: InMemoryRequestStore,
    clock: Clock,
) -> None:
    request = store.submit(submission())
    first = store.claim_next("worker-a", lease_seconds=5)
    assert first is not None
    store.mark_running(first)
    clock.advance(6)

    second = store.claim_next("worker-b", lease_seconds=5)
    assert second is not None
    assert second.request_id == request.id
    assert second.fencing_token == first.fencing_token + 1
    assert second.request.attempt == 2
    with pytest.raises(StaleRequestClaimError):
        store.fail(first, AsyncRequestError(code="old", message="stale"))


def test_renewed_claim_extends_from_store_clock(
    store: InMemoryRequestStore,
    clock: Clock,
) -> None:
    store.submit(submission())
    claim = store.claim_next("worker-a", lease_seconds=5)
    assert claim is not None
    clock.advance(2)

    renewed = store.renew_claim(claim, lease_seconds=10)

    assert renewed.fencing_token == claim.fencing_token
    assert renewed.lease_until == clock.now + timedelta(seconds=10)

    store.mark_running(renewed)
    clock.advance(1)
    running_renewal = store.renew_claim(renewed, lease_seconds=10)
    assert running_renewal.request.state is AsyncRequestState.RUNNING


def test_cancel_invalidates_an_active_claim(store: InMemoryRequestStore) -> None:
    request = store.submit(submission())
    claim = store.claim_next("worker-a", lease_seconds=30)
    assert claim is not None

    cancelled = store.cancel(request.id, owner="tenant-a")

    assert cancelled.state is AsyncRequestState.CANCELLED
    assert store.cancel(request.id) == cancelled
    with pytest.raises(StaleRequestClaimError):
        store.mark_running(claim)


def test_cancel_and_success_race_publishes_one_terminal_state(
    store: InMemoryRequestStore,
) -> None:
    request = store.submit(submission())
    claim = store.claim_next("worker-a", lease_seconds=30)
    assert claim is not None
    store.mark_running(claim)
    barrier = threading.Barrier(2)

    def complete():
        barrier.wait()
        try:
            return store.succeed(claim, {"answer": 42})
        except StaleRequestClaimError as error:
            return error

    def cancel():
        barrier.wait()
        return store.cancel(request.id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        complete_future = executor.submit(complete)
        cancel_future = executor.submit(cancel)
        outcomes = (complete_future.result(), cancel_future.result())

    stored = store.get(request.id)
    assert stored.state in {AsyncRequestState.SUCCEEDED, AsyncRequestState.CANCELLED}
    terminal_states = {
        outcome.state for outcome in outcomes if not isinstance(outcome, Exception)
    }
    assert terminal_states == {stored.state}


def test_deadline_expires_queued_and_claimed_requests(
    store: InMemoryRequestStore,
    clock: Clock,
) -> None:
    already_expired = store.submit(submission(deadline_at=clock.now))
    claimed_request = store.submit(submission(deadline_at=clock.now + timedelta(seconds=3)))
    claim = store.claim_next("worker-a", lease_seconds=30)
    assert claim is not None
    assert claim.request_id == claimed_request.id
    clock.advance(4)

    assert store.claim_next("worker-b", lease_seconds=30) is None
    assert store.get(already_expired.id).state is AsyncRequestState.EXPIRED
    assert store.get(claimed_request.id).state is AsyncRequestState.EXPIRED
    with pytest.raises(StaleRequestClaimError):
        store.mark_running(claim)


def test_deadline_wins_over_idempotent_replay_and_cancel(
    store: InMemoryRequestStore,
    clock: Clock,
) -> None:
    intent = submission(
        idempotency_key="deadline",
        deadline_at=clock.now + timedelta(seconds=3),
    )
    request = store.submit(intent)
    clock.advance(4)

    replay = store.submit(intent)
    cancelled = store.cancel(request.id)

    assert replay.state is AsyncRequestState.EXPIRED
    assert cancelled.state is AsyncRequestState.EXPIRED


def test_status_reads_expire_deadlines_and_claim_lease_is_capped(
    store: InMemoryRequestStore,
    clock: Clock,
) -> None:
    deadline = clock.now + timedelta(seconds=3)
    request = store.submit(submission(deadline_at=deadline))
    claim = store.claim_next("worker-a", lease_seconds=30)
    assert claim is not None
    assert claim.lease_until == deadline

    clock.advance(4)

    assert store.get(request.id).state is AsyncRequestState.EXPIRED
    assert store.list(state=AsyncRequestState.EXPIRED)[0].id == request.id


def test_failed_request_carries_only_structured_error(store: InMemoryRequestStore) -> None:
    request = store.submit(submission())
    claim = store.claim_next("worker-a", lease_seconds=30)
    assert claim is not None
    store.mark_running(claim)

    failed = store.fail(
        claim,
        AsyncRequestError(code="upstream_timeout", message="The model timed out", retryable=True),
    )

    assert failed.id == request.id
    assert failed.state is AsyncRequestState.FAILED
    assert failed.result is None
    assert failed.error is not None
    assert failed.error.retryable is True


def test_public_error_fields_are_size_bounded() -> None:
    with pytest.raises(ValidationError):
        AsyncRequestError(code="x" * 129, message="bounded")
    with pytest.raises(ValidationError):
        AsyncRequestError(code="bounded", message="x" * 1025)


def test_non_json_input_and_result_are_rejected(store: InMemoryRequestStore) -> None:
    with pytest.raises(ValidationError):
        submission(body={"invalid": object()})
    with pytest.raises(ValidationError, match="non-finite"):
        submission(body={"invalid": [float("nan")]})

    request = store.submit(submission())
    claim = store.claim_next("worker-a", lease_seconds=30)
    assert claim is not None
    store.mark_running(claim)
    with pytest.raises(ValidationError):
        store.succeed(claim, {"invalid": object()})  # type: ignore[dict-item]
    with pytest.raises(ValidationError, match="non-finite"):
        store.succeed(claim, {"invalid": float("inf")})
    assert store.get(request.id).state is AsyncRequestState.RUNNING


@pytest.mark.parametrize("lease_seconds", [0, -1, float("inf"), float("nan")])
def test_invalid_lease_is_rejected(store: InMemoryRequestStore, lease_seconds: float) -> None:
    store.submit(submission())
    with pytest.raises(ValueError, match="lease_seconds"):
        store.claim_next("worker", lease_seconds=lease_seconds)
