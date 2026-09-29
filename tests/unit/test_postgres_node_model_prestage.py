"""Real-PostgreSQL tests for durable node model pre-stage state."""

from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kairyu.artifacts import NodeModelCacheFillResult
from kairyu.runners import (
    InMemoryNodeModelPrestageStore,
    ModelCachePlacement,
    ModelCachePlacementState,
    NodeModelPrestageCapacityError,
    NodeModelPrestageConflictError,
    NodeModelPrestageStore,
    PostgresNodeModelPrestageStore,
    RunnerWriterAuthority,
    ScalingPrewarmSnapshot,
    build_node_model_prestage_commands,
    build_node_model_prestage_release_command,
    plan_cache_aware_scale_up,
)

_POSTGRES_DSN = os.environ.get("KAIRYU_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.postgres
_NOW = datetime(2026, 9, 29, 3, 0, tzinfo=UTC)
_DIGEST = "a" * 64

if _POSTGRES_DSN:
    psycopg = pytest.importorskip("psycopg")
else:  # Keep core-only test collection importable.
    psycopg = None


def _authority(*, token: int = 4) -> RunnerWriterAuthority:
    return RunnerWriterAuthority(
        election_id="runner-controller/production",
        holder_id="controller-a",
        fencing_token=token,
        validated_at=_NOW - timedelta(seconds=1),
        lease_until=_NOW + timedelta(minutes=3),
    )


def _command(
    placement_id: str = "placement-00",
    *,
    generation: int = 20,
    target_revision: int = 8,
    token: int = 4,
    node_id: str = "gpu-node-00",
):
    snapshot = ScalingPrewarmSnapshot(
        snapshot_id=f"cache-snapshot-{placement_id}",
        cache_revision=10,
        observed_at=_NOW,
        model_class="interactive-h100",
        model_revision="release-1",
        artifact_digest=_DIGEST,
        placement_binding_id="binding-prestage-h100",
        placements=(
            ModelCachePlacement(
                placement_id=placement_id,
                node_name=node_id,
                resource_flavor="h100-sxm",
                profile_id="h100-sxm",
                compatibility_approval_id="compat-h100-v1",
                state=ModelCachePlacementState.ABSENT,
            ),
        ),
    )
    plan = plan_cache_aware_scale_up(
        snapshot,
        current_replicas=0,
        quota_target_replicas=1,
        resource_flavor="h100-sxm",
    )
    return build_node_model_prestage_commands(
        plan,
        authority=_authority(token=token),
        decision_id=f"scale-decision-{generation}",
        decision_fingerprint="d" * 64,
        target_id="statefulset/production/prestage-test",
        target_revision=target_revision,
        deployment_id="production/prestage-test",
        model_id="org/prestage-test",
        command_generations={placement_id: generation},
        issued_at=_NOW,
        ttl_seconds=120,
    )[0]


def _fill_result() -> NodeModelCacheFillResult:
    return NodeModelCacheFillResult(
        deployment_id="production/prestage-test",
        manifest_digest=_DIGEST,
        artifact_path=Path("/mnt/nvme/kairyu-model-cache/artifacts") / _DIGEST,
        cache_hit=False,
        resumed_bytes=0,
        downloaded_bytes=128,
        file_count=1,
        total_bytes=128,
    )


@pytest.fixture
def store_factory():
    if _POSTGRES_DSN is None or psycopg is None:
        pytest.skip("KAIRYU_TEST_POSTGRES_DSN is not configured")
    store_id = f"pytest-node-prestage-{uuid.uuid4().hex}"
    stores: list[PostgresNodeModelPrestageStore] = []

    def create(**kwargs) -> PostgresNodeModelPrestageStore:
        store = PostgresNodeModelPrestageStore(
            _POSTGRES_DSN,
            store_id=store_id,
            node_id="gpu-node-00",
            **kwargs,
        )
        stores.append(store)
        return store

    yield create, store_id

    for store in reversed(stores):
        store.close()
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            "DELETE FROM public.node_model_prestage_store_registry WHERE store_id = %s",
            (store_id,),
        )


@pytest.fixture
def isolated_database():
    if _POSTGRES_DSN is None or psycopg is None:
        pytest.skip("KAIRYU_TEST_POSTGRES_DSN is not configured")
    database_name = f"pytest_node_prestage_{uuid.uuid4().hex}"
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


def test_cross_instance_claim_complete_release_and_replay(store_factory) -> None:
    create, _store_id = store_factory
    first = create()
    second = create()
    command = _command()

    assert isinstance(first, NodeModelPrestageStore)
    claimed = first.claim(command, claim_id="b" * 64, now=_NOW + timedelta(seconds=1))
    assert second.claim(
        command, claim_id="b" * 64, now=_NOW + timedelta(seconds=1)
    ) == claimed
    with pytest.raises(NodeModelPrestageConflictError, match="another attempt"):
        second.claim(command, claim_id="c" * 64, now=_NOW + timedelta(seconds=1))

    ready = second.complete(
        command,
        claim_id="b" * 64,
        fill_result=_fill_result(),
        now=_NOW + timedelta(seconds=2),
    )
    assert first.list_records() == (ready,)
    assert first.claim(
        command,
        claim_id=b"ignored".hex().ljust(64, "0"),
        now=_NOW + timedelta(seconds=3),
    ) == ready

    release = build_node_model_prestage_release_command(
        command,
        authority=_authority(token=5),
        decision_id="scale-decision-21",
        decision_fingerprint="e" * 64,
        target_revision=9,
        command_generation=21,
        issued_at=_NOW + timedelta(seconds=4),
        ttl_seconds=60,
    )
    absent = second.release(release, now=_NOW + timedelta(seconds=5))
    assert absent.state is ModelCachePlacementState.ABSENT
    assert first.release(release, now=_NOW + timedelta(seconds=6)) == absent


def test_concurrent_cross_instance_claim_has_one_winner(store_factory) -> None:
    create, _store_id = store_factory
    stores = (create(), create())
    command = _command()
    barrier = threading.Barrier(2)

    def claim(index: int):
        barrier.wait(timeout=5)
        return stores[index].claim(
            command,
            claim_id=("b" if index == 0 else "c") * 64,
            now=_NOW + timedelta(seconds=1),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(claim, index) for index in range(2)]
    outcomes: list[object] = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except NodeModelPrestageConflictError as exc:
            outcomes.append(exc)

    assert sum(isinstance(value, NodeModelPrestageConflictError) for value in outcomes) == 1
    assert stores[0].list_records()[0].state is ModelCachePlacementState.FILLING


def test_failed_exact_retry_and_shared_capacity(store_factory) -> None:
    create, _store_id = store_factory
    first = create(max_placements=1)
    second = create(max_placements=1)
    command = _command()
    first.claim(command, claim_id="b" * 64, now=_NOW + timedelta(seconds=1))
    failed = second.fail(
        command,
        claim_id="b" * 64,
        failure="object store unavailable",
        now=_NOW + timedelta(seconds=2),
    )
    retried = first.claim(command, claim_id="c" * 64, now=_NOW + timedelta(seconds=3))
    assert failed.attempt == 1
    assert retried.attempt == 2

    with pytest.raises(NodeModelPrestageCapacityError):
        second.claim(
            _command("placement-01", generation=21),
            claim_id="d" * 64,
            now=_NOW + timedelta(seconds=4),
        )


def test_store_configuration_mismatch_fails_closed(store_factory) -> None:
    create, store_id = store_factory
    create(max_placements=4)
    with pytest.raises(RuntimeError, match="configuration changed"):
        PostgresNodeModelPrestageStore(
            _POSTGRES_DSN,
            store_id=store_id,
            node_id="gpu-node-00",
            max_placements=5,
        )


def test_startup_without_initialization_rejects_missing_schema(isolated_database) -> None:
    store = PostgresNodeModelPrestageStore(
        isolated_database,
        store_id="missing-schema",
        node_id="gpu-node-00",
        eager_connect=False,
        initialize_schema=False,
    )
    with pytest.raises(RuntimeError, match="missing required tables"):
        store.startup()
    store.close()


def test_startup_rejects_changed_state_constraint(isolated_database) -> None:
    first = PostgresNodeModelPrestageStore(
        isolated_database,
        store_id="changed-constraint",
        node_id="gpu-node-00",
    )
    first.close()
    with psycopg.connect(isolated_database, autocommit=True) as connection:
        connection.execute(
            """
            ALTER TABLE public.node_model_prestage_records
            DROP CONSTRAINT node_model_prestage_records_state_check
            """
        )
        connection.execute(
            """
            ALTER TABLE public.node_model_prestage_records
            ADD CONSTRAINT node_model_prestage_records_state_check
            CHECK (state IN ('absent', 'filling', 'ready', 'failed', 'corrupt'))
            """
        )
    with pytest.raises(RuntimeError, match="incompatible constraints"):
        PostgresNodeModelPrestageStore(
            isolated_database,
            store_id="changed-constraint",
            node_id="gpu-node-00",
            initialize_schema=False,
        )


def test_stored_metadata_mismatch_fails_closed() -> None:
    command = _command()
    memory = InMemoryNodeModelPrestageStore(node_id="gpu-node-00")
    record = memory.claim(
        command,
        claim_id="b" * 64,
        now=_NOW + timedelta(seconds=1),
    )
    row = (
        "another-placement",
        command.command_id,
        command.command_generation,
        command.authority.fencing_token,
        command.target_revision,
        record.state.value,
        record.updated_at,
        record.model_dump(mode="json"),
    )

    with pytest.raises(RuntimeError, match="metadata is inconsistent"):
        PostgresNodeModelPrestageStore._record(row, node_id="gpu-node-00")


def test_stored_record_for_another_node_fails_closed() -> None:
    command = _command(node_id="gpu-node-01")
    memory = InMemoryNodeModelPrestageStore(node_id="gpu-node-01")
    record = memory.claim(
        command,
        claim_id="b" * 64,
        now=_NOW + timedelta(seconds=1),
    )
    row = (
        command.placement_id,
        command.command_id,
        command.command_generation,
        command.authority.fencing_token,
        command.target_revision,
        record.state.value,
        record.updated_at,
        record.model_dump(mode="json"),
    )

    with pytest.raises(RuntimeError, match="targets another node"):
        PostgresNodeModelPrestageStore._record(row, node_id="gpu-node-00")
