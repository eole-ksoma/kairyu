"""Real-PostgreSQL tests for the append-only autoscaler decision log."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from kairyu.runners import (
    PostgresScalingDecisionLog,
    ScalingDecisionAction,
    ScalingDecisionCapacityError,
    ScalingDecisionConflictError,
    ScalingDecisionGenerationError,
    ScalingDecisionLog,
    ScalingDecisionReason,
    ScalingDecisionRecord,
    ScalingObservation,
    ScalingObservationWindow,
    ScalingPolicy,
    ScalingQueueSnapshot,
    ScalingRunnerSnapshot,
)

_POSTGRES_DSN = os.environ.get("KAIRYU_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.postgres

if _POSTGRES_DSN:
    psycopg = pytest.importorskip("psycopg")
else:  # Keep core-only test collection importable.
    psycopg = None

NOW = datetime(2026, 9, 15, 6, 0, tzinfo=UTC)


def _record(
    decision_id: str = "decision-1",
    *,
    decided_at: datetime | None = None,
    model_class: str = "interactive-14b",
    reason: ScalingDecisionReason = ScalingDecisionReason.NO_CHANGE,
    action: ScalingDecisionAction = ScalingDecisionAction.HOLD,
    decision_generation: int | None = None,
) -> ScalingDecisionRecord:
    observed_at = (decided_at or NOW + timedelta(seconds=1)) - timedelta(seconds=1)
    policy = ScalingPolicy(
        model_class=model_class,
        policy_revision=7,
        min_replicas=2,
        max_replicas=50,
        warm_buffer_replicas=3,
        warm_buffer_ratio=0.25,
        max_scale_up_step=8,
        max_scale_down_step=2,
    )
    observation = ScalingObservation(
        observation_id=f"observation-{decision_id}",
        model_class=model_class,
        observed_at=observed_at,
        queue=ScalingQueueSnapshot(
            observed_at=observed_at,
            queue_depth=6,
            interactive_queue_depth=4,
            batch_queue_depth=2,
            oldest_queue_age_seconds=2.5,
            arrival_rate_per_second=3.25,
            deadline_remaining_p50_seconds=10,
            deadline_remaining_p95_seconds=30,
            predicted_ttft_seconds=0.8,
            goodput_slo_ratio=0.97,
        ),
        runners=ScalingRunnerSnapshot(
            observed_at=observed_at,
            current_replicas=3,
            busy_replicas=1,
            ready_replicas=1,
            loading_replicas=1,
            unhealthy_replicas=0,
        ),
    )
    return ScalingDecisionRecord(
        decision_id=decision_id,
        decision_generation=decision_generation,
        decided_at=decided_at or NOW + timedelta(seconds=1),
        catalog_revision=9,
        policy=policy,
        window=ScalingObservationWindow(
            window_id=f"window-{decision_id}",
            model_class=model_class,
            started_at=observed_at - timedelta(seconds=30),
            ended_at=observed_at,
            observations=(observation,),
        ),
        action=action,
        reason=reason,
        demand_replicas=3,
        buffered_target_replicas=6,
        desired_replicas=4 if action is ScalingDecisionAction.SCALE_UP else 3,
        target_delta=1 if action is ScalingDecisionAction.SCALE_UP else 0,
    )


@pytest.fixture
def store_factory():
    assert _POSTGRES_DSN is not None
    store_id = f"pytest-runner-scaling-{uuid.uuid4().hex}"
    stores: list[PostgresScalingDecisionLog] = []

    def create(**kwargs) -> PostgresScalingDecisionLog:
        store = PostgresScalingDecisionLog(
            _POSTGRES_DSN,
            store_id=store_id,
            **kwargs,
        )
        stores.append(store)
        return store

    yield create, store_id

    for store in reversed(stores):
        store.close()
    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            "DELETE FROM public.runner_scaling_log_registry WHERE store_id = %s",
            (store_id,),
        )


@pytest.fixture
def isolated_database():
    assert _POSTGRES_DSN is not None
    assert psycopg is not None
    database_name = f"pytest_runner_scaling_{uuid.uuid4().hex}"
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            psycopg.sql.SQL("CREATE DATABASE {}").format(
                psycopg.sql.Identifier(database_name)
            )
        )
    database_dsn = psycopg.conninfo.make_conninfo(
        _POSTGRES_DSN,
        dbname=database_name,
    )
    yield database_dsn
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            psycopg.sql.SQL("DROP DATABASE {} WITH (FORCE)").format(
                psycopg.sql.Identifier(database_name)
            )
        )


def test_cross_instance_append_replay_get_and_filtered_list(store_factory) -> None:
    create, _store_id = store_factory
    first = create()
    second = create()
    oldest = _record("decision-1")
    newest = _record(
        "decision-2",
        decided_at=NOW + timedelta(seconds=2),
    )
    batch = _record(
        "decision-3",
        decided_at=NOW + timedelta(seconds=3),
        model_class="batch-14b",
    )

    assert isinstance(first, ScalingDecisionLog)
    assert first.append(oldest) == oldest
    assert second.append(oldest) == oldest
    assert second.get(oldest.decision_id) == oldest
    assert second.append(newest) == newest
    assert first.append(batch) == batch
    assert first.list(limit=2) == (batch, newest)
    assert second.list(model_class="interactive-14b") == (newest, oldest)
    assert first.list(since=NOW + timedelta(seconds=2)) == (batch, newest)


def test_pre_wp34_record_fingerprint_remains_readable() -> None:
    record = _record()
    legacy_payload = record.model_dump(mode="json")
    legacy_payload.pop("decision_generation")
    legacy_payload.pop("target_revision")
    legacy_payload.pop("quota_admission")
    legacy_payload.pop("prewarm_plan")
    legacy_fingerprint = hashlib.sha256(
        json.dumps(
            legacy_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    row = (
        record.decision_id,
        record.policy.model_class,
        record.decided_at,
        record.window.started_at,
        record.window.ended_at,
        record.catalog_revision,
        record.policy.policy_revision,
        record.action.value,
        legacy_fingerprint,
        legacy_payload,
    )

    assert record.fingerprint == legacy_fingerprint
    assert PostgresScalingDecisionLog._record(row) == record


def test_changed_replay_conflicts_across_instances(store_factory) -> None:
    create, _store_id = store_factory
    first = create()
    second = create()
    first.append(_record())

    with pytest.raises(ScalingDecisionConflictError):
        second.append(_record(reason=ScalingDecisionReason.HYSTERESIS))


def test_capacity_is_shared_while_exact_replay_remains_allowed(store_factory) -> None:
    create, _store_id = store_factory
    first = create(max_records=1)
    second = create(max_records=1)
    record = _record()

    assert first.append(record) == record
    assert second.append(record) == record
    with pytest.raises(ScalingDecisionCapacityError):
        second.append(_record("decision-2"))


def test_store_rejects_a_different_capacity_configuration(store_factory) -> None:
    create, _store_id = store_factory
    create(max_records=1)

    with pytest.raises(RuntimeError, match="capacity is 1; configured 2"):
        create(max_records=2)


def test_store_capacity_fits_postgres_bigint() -> None:
    assert _POSTGRES_DSN is not None
    with pytest.raises(ValueError, match="signed 64-bit"):
        PostgresScalingDecisionLog(
            _POSTGRES_DSN,
            store_id="scaling-test",
            max_records=2**63,
        )


def test_concurrent_cross_instance_append_is_exactly_once(store_factory) -> None:
    create, store_id = store_factory
    stores = tuple(create() for _ in range(8))
    record = _record()

    with ThreadPoolExecutor(max_workers=len(stores)) as pool:
        results = tuple(pool.map(lambda store: store.append(record), stores))

    assert results == (record,) * len(stores)
    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN) as connection:
        count = connection.execute(
            "SELECT count(*) FROM public.runner_scaling_decisions "
            "WHERE store_id = %s",
            (store_id,),
        ).fetchone()
    assert count == (1,)


def test_cross_instance_mutation_generation_is_durable_and_atomic(store_factory) -> None:
    create, _store_id = store_factory
    stores = tuple(create() for _ in range(8))
    drafts = tuple(
        _record(
            f"scale-{index}",
            action=ScalingDecisionAction.SCALE_UP,
            reason=ScalingDecisionReason.QUEUE_PRESSURE,
        )
        for index in range(8)
    )
    with ThreadPoolExecutor(max_workers=len(stores)) as pool:
        results = tuple(
            pool.map(
                lambda pair: pair[0].append(pair[1]),
                zip(stores, drafts, strict=True),
            )
        )

    assert {record.decision_generation for record in results} == set(range(1, 9))
    replay = stores[0].append(drafts[0])
    assert replay == stores[1].get(drafts[0].decision_id)


def test_postgres_rejects_forged_generation_for_new_decision(store_factory) -> None:
    create, _store_id = store_factory
    store = create()
    forged = _record(
        "scale-forged",
        action=ScalingDecisionAction.SCALE_UP,
        reason=ScalingDecisionReason.QUEUE_PRESSURE,
        decision_generation=4,
    )

    with pytest.raises(ScalingDecisionGenerationError, match="must not supply"):
        store.append(forged)


def test_startup_without_initialization_rejects_missing_tables(
    isolated_database,
) -> None:
    with pytest.raises(RuntimeError, match="missing required tables"):
        PostgresScalingDecisionLog(
            isolated_database,
            store_id="scaling-test",
            initialize_schema=False,
        )


@pytest.mark.parametrize(
    "schema_mutation",
    [
        "ALTER TABLE runner_scaling_decisions "
        "ADD CONSTRAINT unexpected_action CHECK (action <> '')",
        "ALTER TABLE runner_scaling_decisions "
        "DROP CONSTRAINT runner_scaling_decisions_pkey",
        "ALTER TABLE runner_scaling_decisions "
        "DROP CONSTRAINT runner_scaling_decisions_store_id_fkey",
    ],
    ids=["check", "primary-key", "foreign-key"],
)
@pytest.mark.parametrize("initialize_schema", [False, True])
def test_startup_rejects_incompatible_constraints(
    isolated_database,
    initialize_schema: bool,
    schema_mutation: str,
) -> None:
    bootstrap = PostgresScalingDecisionLog(
        isolated_database,
        store_id="scaling-test",
    )
    bootstrap.close()
    assert psycopg is not None
    with psycopg.connect(isolated_database, autocommit=True) as connection:
        connection.execute(schema_mutation)

    with pytest.raises(RuntimeError, match="incompatible constraints"):
        PostgresScalingDecisionLog(
            isolated_database,
            store_id="scaling-test",
            initialize_schema=initialize_schema,
        )


def test_startup_rejects_incompatible_column_and_index(isolated_database) -> None:
    bootstrap = PostgresScalingDecisionLog(
        isolated_database,
        store_id="scaling-test",
    )
    bootstrap.close()
    assert psycopg is not None
    with psycopg.connect(isolated_database, autocommit=True) as connection:
        connection.execute(
            "ALTER TABLE runner_scaling_decisions "
            "ALTER COLUMN action DROP NOT NULL"
        )

    with pytest.raises(RuntimeError, match="incompatible columns"):
        PostgresScalingDecisionLog(
            isolated_database,
            store_id="scaling-test",
            initialize_schema=False,
        )


def test_startup_rejects_replaced_decision_index(isolated_database) -> None:
    bootstrap = PostgresScalingDecisionLog(
        isolated_database,
        store_id="scaling-test",
    )
    bootstrap.close()
    assert psycopg is not None
    with psycopg.connect(isolated_database, autocommit=True) as connection:
        connection.execute("DROP INDEX runner_scaling_decisions_model_time_idx")
        connection.execute(
            "CREATE INDEX runner_scaling_decisions_model_time_idx "
            "ON runner_scaling_decisions (store_id, decided_at DESC)"
        )

    with pytest.raises(RuntimeError, match="index is incompatible"):
        PostgresScalingDecisionLog(
            isolated_database,
            store_id="scaling-test",
            initialize_schema=False,
        )


def test_startup_rejects_unlogged_decision_table(isolated_database) -> None:
    bootstrap = PostgresScalingDecisionLog(
        isolated_database,
        store_id="scaling-test",
    )
    bootstrap.close()
    assert psycopg is not None
    with psycopg.connect(isolated_database, autocommit=True) as connection:
        connection.execute("ALTER TABLE runner_scaling_decisions SET UNLOGGED")

    with pytest.raises(RuntimeError, match="permanent ordinary tables"):
        PostgresScalingDecisionLog(
            isolated_database,
            store_id="scaling-test",
            initialize_schema=False,
        )


def test_reconnect_rejects_recreated_control_namespace(isolated_database) -> None:
    original = PostgresScalingDecisionLog(
        isolated_database,
        store_id="scaling-test",
    )
    assert psycopg is not None
    with psycopg.connect(isolated_database, autocommit=True) as connection:
        connection.execute("DROP SCHEMA public CASCADE")
        connection.execute("CREATE SCHEMA public")
    replacement = PostgresScalingDecisionLog(
        isolated_database,
        store_id="scaling-test",
    )
    replacement.close()
    assert original._connection is not None
    original._connection.close()
    try:
        with pytest.raises(RuntimeError, match="namespace identity changed"):
            original.list()
    finally:
        original.close()
