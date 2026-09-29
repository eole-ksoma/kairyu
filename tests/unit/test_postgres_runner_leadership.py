"""Real-PostgreSQL tests for Runner leader election and fencing."""

from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from kairyu.runners.leadership import (
    RunnerLeaderLeaseStore,
    StaleRunnerLeaderLeaseError,
)
from kairyu.runners.postgres_leadership import PostgresRunnerLeaderLeaseStore

_POSTGRES_DSN = os.environ.get("KAIRYU_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.postgres

if _POSTGRES_DSN:
    psycopg = pytest.importorskip("psycopg")
else:  # Keep core-only test collection importable.
    psycopg = None


@pytest.fixture
def store_factory():
    assert _POSTGRES_DSN is not None
    store_id = f"pytest-runner-leader-{uuid.uuid4().hex}"
    stores: list[PostgresRunnerLeaderLeaseStore] = []

    def create(**kwargs) -> PostgresRunnerLeaderLeaseStore:
        store = PostgresRunnerLeaderLeaseStore(
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
            "DELETE FROM public.runner_leader_store_registry WHERE store_id = %s",
            (store_id,),
        )


@pytest.fixture
def isolated_database():
    assert _POSTGRES_DSN is not None
    assert psycopg is not None
    database_name = f"pytest_runner_leader_{uuid.uuid4().hex}"
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


def test_startup_without_initialization_rejects_missing_lease_table(
    isolated_database,
) -> None:
    store_id = f"pytest-runner-leader-{uuid.uuid4().hex}"
    bootstrap = PostgresRunnerLeaderLeaseStore(isolated_database, store_id=store_id)
    bootstrap.close()
    assert psycopg is not None
    with psycopg.connect(isolated_database, autocommit=True) as connection:
        connection.execute("DROP TABLE runner_leader_leases")

    with pytest.raises(RuntimeError, match="missing required tables"):
        PostgresRunnerLeaderLeaseStore(
            isolated_database,
            store_id=store_id,
            initialize_schema=False,
        )


@pytest.mark.parametrize(
    "schema_mutation",
    [
        "ALTER TABLE runner_leader_leases "
        "ADD CONSTRAINT unexpected_lease_window "
        "CHECK (lease_until >= acquired_at)",
        "ALTER TABLE runner_leader_leases "
        "DROP CONSTRAINT runner_leader_leases_pkey",
        "ALTER TABLE runner_leader_leases "
        "DROP CONSTRAINT runner_leader_leases_store_id_fkey",
    ],
    ids=["check", "primary-key", "foreign-key"],
)
@pytest.mark.parametrize("initialize_schema", [False, True])
def test_startup_rejects_incompatible_lease_constraints(
    isolated_database,
    initialize_schema: bool,
    schema_mutation: str,
) -> None:
    store_id = f"pytest-runner-leader-{uuid.uuid4().hex}"
    bootstrap = PostgresRunnerLeaderLeaseStore(isolated_database, store_id=store_id)
    bootstrap.close()
    assert psycopg is not None
    with psycopg.connect(isolated_database, autocommit=True) as connection:
        connection.execute(schema_mutation)

    with pytest.raises(RuntimeError, match="incompatible constraints"):
        PostgresRunnerLeaderLeaseStore(
            isolated_database,
            store_id=store_id,
            initialize_schema=initialize_schema,
        )


def test_startup_rejects_incompatible_column_nullability(isolated_database) -> None:
    store_id = f"pytest-runner-leader-{uuid.uuid4().hex}"
    bootstrap = PostgresRunnerLeaderLeaseStore(isolated_database, store_id=store_id)
    bootstrap.close()
    assert psycopg is not None
    with psycopg.connect(isolated_database, autocommit=True) as connection:
        connection.execute(
            "ALTER TABLE runner_leader_leases ALTER COLUMN holder_id DROP NOT NULL"
        )

    with pytest.raises(RuntimeError, match="incompatible columns"):
        PostgresRunnerLeaderLeaseStore(
            isolated_database,
            store_id=store_id,
            initialize_schema=False,
        )


def test_startup_rejects_incompatible_column_type(isolated_database) -> None:
    store_id = f"pytest-runner-leader-{uuid.uuid4().hex}"
    bootstrap = PostgresRunnerLeaderLeaseStore(isolated_database, store_id=store_id)
    bootstrap.close()
    assert psycopg is not None
    with psycopg.connect(isolated_database, autocommit=True) as connection:
        connection.execute(
            "ALTER TABLE runner_leader_store_registry "
            "ALTER COLUMN schema_version TYPE BIGINT"
        )

    with pytest.raises(RuntimeError, match="incompatible columns"):
        PostgresRunnerLeaderLeaseStore(
            isolated_database,
            store_id=store_id,
            initialize_schema=False,
        )


def test_startup_rejects_table_moved_out_of_control_namespace(isolated_database) -> None:
    store_id = f"pytest-runner-leader-{uuid.uuid4().hex}"
    bootstrap = PostgresRunnerLeaderLeaseStore(isolated_database, store_id=store_id)
    bootstrap.close()
    assert psycopg is not None
    with psycopg.connect(isolated_database, autocommit=True) as connection:
        connection.execute("CREATE SCHEMA runner_split")
        connection.execute("ALTER TABLE runner_leader_leases SET SCHEMA runner_split")
        connection.execute(
            psycopg.sql.SQL(
                "ALTER DATABASE {} SET search_path = runner_split, public"
            ).format(psycopg.sql.Identifier(connection.info.dbname))
        )

    with pytest.raises(RuntimeError, match="missing required tables"):
        PostgresRunnerLeaderLeaseStore(
            isolated_database,
            store_id=store_id,
            initialize_schema=False,
        )


def test_startup_rejects_unlogged_lease_table(isolated_database) -> None:
    store_id = f"pytest-runner-leader-{uuid.uuid4().hex}"
    bootstrap = PostgresRunnerLeaderLeaseStore(isolated_database, store_id=store_id)
    bootstrap.close()
    assert psycopg is not None
    with psycopg.connect(isolated_database, autocommit=True) as connection:
        connection.execute("ALTER TABLE runner_leader_leases SET UNLOGGED")

    with pytest.raises(RuntimeError, match="permanent ordinary tables"):
        PostgresRunnerLeaderLeaseStore(
            isolated_database,
            store_id=store_id,
            initialize_schema=False,
        )


def test_different_search_paths_cannot_split_the_election(isolated_database) -> None:
    store_id = f"pytest-runner-leader-{uuid.uuid4().hex}"
    bootstrap = PostgresRunnerLeaderLeaseStore(isolated_database, store_id=store_id)
    bootstrap.close()
    assert psycopg is not None
    with psycopg.connect(isolated_database, autocommit=True) as connection:
        for schema_name in ("contender_a", "contender_b"):
            registry = psycopg.sql.Identifier(
                schema_name, "runner_leader_store_registry"
            )
            leases = psycopg.sql.Identifier(schema_name, "runner_leader_leases")
            connection.execute(
                psycopg.sql.SQL("CREATE SCHEMA {}").format(
                    psycopg.sql.Identifier(schema_name)
                )
            )
            connection.execute(
                psycopg.sql.SQL(
                    "CREATE TABLE {} "
                    "(LIKE public.runner_leader_store_registry INCLUDING ALL)"
                ).format(registry)
            )
            connection.execute(
                psycopg.sql.SQL(
                    "CREATE TABLE {} "
                    "(LIKE public.runner_leader_leases INCLUDING ALL)"
                ).format(leases)
            )
            connection.execute(
                psycopg.sql.SQL(
                    "ALTER TABLE {} ADD FOREIGN KEY (store_id) "
                    "REFERENCES {} (store_id) ON DELETE CASCADE"
                ).format(
                    leases,
                    registry,
                )
            )
            connection.execute(
                psycopg.sql.SQL(
                    "INSERT INTO {} (store_id, schema_version) VALUES (%s, 1)"
                ).format(registry),
                (store_id,),
            )

    contender_dsns = tuple(
        psycopg.conninfo.make_conninfo(
            isolated_database,
            options=f"-c search_path={schema_name},public",
        )
        for schema_name in ("contender_a", "contender_b")
    )
    first = PostgresRunnerLeaderLeaseStore(
        contender_dsns[0], store_id=store_id, initialize_schema=False
    )
    second = PostgresRunnerLeaderLeaseStore(
        contender_dsns[1], store_id=store_id, initialize_schema=False
    )
    try:
        lease = first.acquire("runner-control-plane", "controller-a", lease_seconds=5)
        assert lease is not None
        assert second.acquire(
            "runner-control-plane", "controller-b", lease_seconds=5
        ) is None
        assert first.authorize(lease).fencing_token == 1
    finally:
        second.close()
        first.close()


def test_reconnect_rejects_recreated_control_namespace(isolated_database) -> None:
    store_id = f"pytest-runner-leader-{uuid.uuid4().hex}"
    original = PostgresRunnerLeaderLeaseStore(isolated_database, store_id=store_id)
    assert psycopg is not None
    with psycopg.connect(isolated_database, autocommit=True) as connection:
        connection.execute("DROP SCHEMA public CASCADE")
        connection.execute("CREATE SCHEMA public")
    replacement = PostgresRunnerLeaderLeaseStore(isolated_database, store_id=store_id)
    replacement.close()
    assert original._connection is not None
    original._connection.close()
    try:
        with pytest.raises(RuntimeError, match="namespace identity changed"):
            original.acquire(
                "runner-control-plane",
                "controller-a",
                lease_seconds=5,
            )
    finally:
        original.close()


def test_cross_instance_election_renewal_release_and_fencing(store_factory) -> None:
    create, _store_id = store_factory
    first = create()
    second = create()
    lease = first.acquire("runner-control-plane", "controller-a", lease_seconds=5)
    assert lease is not None
    assert isinstance(first, RunnerLeaderLeaseStore)
    assert lease.fencing_token == 1
    assert second.acquire("runner-control-plane", "controller-b", lease_seconds=5) is None

    renewed = first.renew(lease, lease_seconds=5)
    assert renewed.tenure == lease.tenure
    assert renewed.renewed_at >= lease.renewed_at
    assert first.authorize(lease).lease_until == renewed.lease_until
    first.release(lease)

    successor = second.acquire(
        "runner-control-plane",
        "controller-b",
        lease_seconds=5,
    )
    assert successor is not None
    assert successor.fencing_token == lease.fencing_token + 1
    with pytest.raises(StaleRunnerLeaderLeaseError):
        first.authorize(lease)


def test_expired_lease_is_reclaimed_with_a_higher_token(store_factory) -> None:
    create, _store_id = store_factory
    first = create()
    second = create()
    expired = first.acquire(
        "runner-control-plane",
        "controller-a",
        lease_seconds=0.1,
    )
    assert expired is not None
    time.sleep(0.15)

    successor = second.acquire(
        "runner-control-plane",
        "controller-b",
        lease_seconds=1,
    )
    assert successor is not None
    assert successor.fencing_token == expired.fencing_token + 1
    with pytest.raises(StaleRunnerLeaderLeaseError):
        first.renew(expired, lease_seconds=1)
    with pytest.raises(StaleRunnerLeaderLeaseError):
        first.release(expired)


def test_released_equal_timestamp_tombstone_remains_reclaimable(store_factory) -> None:
    create, store_id = store_factory
    first = create()
    second = create()
    released = first.acquire(
        "runner-control-plane",
        "controller-a",
        lease_seconds=5,
    )
    assert released is not None
    first.release(released)
    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            """
            UPDATE runner_leader_leases
            SET lease_until = renewed_at
            WHERE store_id = %s AND election_id = %s
            """,
            (store_id, "runner-control-plane"),
        )

    with pytest.raises(StaleRunnerLeaderLeaseError):
        first.authorize(released)
    successor = second.acquire(
        "runner-control-plane",
        "controller-b",
        lease_seconds=5,
    )
    assert successor is not None
    assert successor.fencing_token == released.fencing_token + 1


def test_concurrent_cross_instance_campaign_has_one_winner(store_factory) -> None:
    create, _store_id = store_factory
    stores = tuple(create() for _ in range(8))

    def acquire(item: tuple[int, PostgresRunnerLeaderLeaseStore]):
        index, store = item
        return store.acquire(
            "runner-control-plane",
            f"controller-{index}",
            lease_seconds=5,
        )

    with ThreadPoolExecutor(max_workers=len(stores)) as pool:
        leases = tuple(pool.map(acquire, enumerate(stores)))
    acquired = tuple(lease for lease in leases if lease is not None)
    assert len(acquired) == 1
    assert acquired[0].fencing_token == 1
