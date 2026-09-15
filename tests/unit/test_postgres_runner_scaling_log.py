"""Real-PostgreSQL tests for the append-only autoscaler decision log."""

from __future__ import annotations

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
        action=ScalingDecisionAction.HOLD,
        reason=reason,
        demand_replicas=3,
        buffered_target_replicas=6,
        desired_replicas=3,
        target_delta=0,
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
