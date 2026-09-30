"""Real-PostgreSQL tests for shared Runner cache placement admission."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest

from kairyu.runners import (
    PostgresRunnerCachePlacementAdmissionStore,
    RunnerCachePlacementAdmissionConflictError,
    RunnerCachePlacementAdmissionPlan,
    RunnerCachePlacementAdmissionStore,
    RunnerCacheStartupBinding,
    RunnerCacheStartupPlacement,
)

_POSTGRES_DSN = os.environ.get("KAIRYU_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.postgres
_DIGEST = "a" * 64
_TARGET = "deployment/model-serving/qwen-runners"

if _POSTGRES_DSN:
    psycopg = pytest.importorskip("psycopg")
else:  # Keep core-only test collection importable.
    psycopg = None


def _binding(
    *,
    suffix: str = "a",
    nodes: tuple[str, ...] = ("gpu-a", "gpu-b"),
    bound_at: datetime | None = None,
    valid_for: timedelta = timedelta(minutes=5),
    target_id: str = _TARGET,
) -> RunnerCacheStartupBinding:
    if bound_at is None:
        bound_at = _database_now() - timedelta(seconds=1)
    placements = tuple(
        RunnerCacheStartupPlacement(
            placement_id=f"placement-{index}-{suffix}",
            node_name=node,
            resource_flavor="h100-sxm",
            profile_id="h100-sxm-tp1",
            compatibility_approval_id="compat-qwen-h100",
            manifest_digest=_DIGEST,
            pin_owner=f"prestage/model-serving/qwen/placement-{index}-{suffix}",
            prestage_command_id=hashlib.sha256(f"command-{index}-{suffix}".encode()).hexdigest(),
            prestage_command_generation=index + 1,
            hint_index_revision=10 + index,
            resident_record_generation=20 + index,
            hint_observed_at=bound_at - timedelta(seconds=1),
            hint_valid_until=bound_at + valid_for,
        )
        for index, node in enumerate(nodes)
    )
    payload = {
        "schema_version": "runner-cache-startup-binding-v1",
        "decision_id": f"decision-{suffix}",
        "decision_fingerprint": hashlib.sha256(f"decision-{suffix}".encode()).hexdigest(),
        "target_id": target_id,
        "target_revision": 7,
        "deployment_id": "model-serving/qwen",
        "model_class": "qwen-14b",
        "model_id": "org/qwen",
        "model_revision": "revision-a",
        "manifest_digest": _DIGEST,
        "placement_binding_id": f"placement-binding-{suffix}",
        "prewarm_snapshot_id": f"snapshot-{suffix}",
        "prewarm_cache_revision": 9,
        "bound_at": bound_at,
        "valid_until": bound_at + valid_for,
        "placements": placements,
    }
    unsigned = RunnerCacheStartupBinding.model_construct(binding_id="0" * 64, **payload)
    encoded = json.dumps(
        unsigned.model_dump(mode="json", exclude={"binding_id"}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return RunnerCacheStartupBinding(
        binding_id=hashlib.sha256(encoded).hexdigest(),
        **payload,
    )


def _plan(binding: RunnerCacheStartupBinding) -> RunnerCachePlacementAdmissionPlan:
    return RunnerCachePlacementAdmissionPlan(
        binding=binding,
        release_id="release-a",
        namespace="model-serving",
        owner_api_version="apps/v1",
        owner_kind="StatefulSet",
        owner_name="qwen-runners",
        owner_uid="workload-uid-a",
        creator_username="system:serviceaccount:kairyu:statefulset-controller",
        registered_at=binding.bound_at
        + min(timedelta(seconds=1), (binding.valid_until - binding.bound_at) / 2),
    )


def _database_now() -> datetime:
    assert _POSTGRES_DSN is not None
    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        row = connection.execute("SELECT clock_timestamp()").fetchone()
    assert row is not None
    return row[0]


@pytest.fixture
def store_factory():
    if _POSTGRES_DSN is None or psycopg is None:
        pytest.skip("KAIRYU_TEST_POSTGRES_DSN is not configured")
    store_id = f"pytest-runner-cache-admission-{uuid.uuid4().hex}"
    stores: list[PostgresRunnerCachePlacementAdmissionStore] = []

    def create(**kwargs) -> PostgresRunnerCachePlacementAdmissionStore:
        store = PostgresRunnerCachePlacementAdmissionStore(
            _POSTGRES_DSN,
            store_id=store_id,
            **kwargs,
        )
        stores.append(store)
        return store

    yield create, store_id

    for store in reversed(stores):
        store.close()
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            """
            DELETE FROM public.runner_cache_placement_admission_store_registry
            WHERE store_id = %s
            """,
            (store_id,),
        )


def _claim(
    store: PostgresRunnerCachePlacementAdmissionStore,
    binding: RunnerCacheStartupBinding,
    *,
    name: str,
    admission_uid: str,
    claimed_at: datetime | None = None,
):
    return store.claim(
        target_id=binding.target_id,
        binding_id=binding.binding_id,
        pod_key=f"model-serving/{name}",
        admission_uid=admission_uid,
        claimed_at=claimed_at or (binding.bound_at + timedelta(seconds=2)),
    )


def test_cross_instance_claim_replay_release_and_capacity(store_factory) -> None:
    create, _store_id = store_factory
    first = create()
    second = create()
    binding = _binding()
    plan = _plan(binding)

    assert isinstance(first, RunnerCachePlacementAdmissionStore)
    first.register(plan)
    second.register(plan)
    assert second.resolve(_TARGET) == plan

    created = _claim(first, binding, name="qwen-7", admission_uid="admission-a")
    replay = _claim(
        second,
        binding,
        name="qwen-7",
        admission_uid="admission-retry",
        claimed_at=binding.bound_at + timedelta(seconds=3),
    )
    second_claim = _claim(
        first,
        binding,
        name="qwen-8",
        admission_uid="admission-b",
    )

    assert created.created is True
    assert replay.created is False
    assert replay.claim == created.claim
    assert {created.claim.placement_id, second_claim.claim.placement_id} == {
        "placement-0-a",
        "placement-1-a",
    }
    second.release(created.claim)
    with pytest.raises(RunnerCachePlacementAdmissionConflictError, match="exhausted"):
        _claim(first, binding, name="qwen-9", admission_uid="admission-c")


def test_unobserved_creator_claim_can_be_released(store_factory) -> None:
    create, _store_id = store_factory
    first = create()
    second = create()
    binding = _binding()
    first.register(_plan(binding))

    created = _claim(first, binding, name="qwen-7", admission_uid="admission-a")
    second.release(created.claim)
    replacement = _claim(
        second,
        binding,
        name="qwen-8",
        admission_uid="admission-b",
    )
    assert replacement.claim.placement_id == "placement-0-a"


def test_cross_instance_concurrent_claims_are_unique(store_factory) -> None:
    create, _store_id = store_factory
    stores = tuple(create() for _ in range(4))
    binding = _binding(nodes=("gpu-a", "gpu-b", "gpu-c", "gpu-d"))
    stores[0].register(_plan(binding))

    def allocate(index: int) -> str:
        return _claim(
            stores[index],
            binding,
            name=f"qwen-{index}",
            admission_uid=f"admission-{index}",
        ).claim.placement_id

    with ThreadPoolExecutor(max_workers=4) as pool:
        placements = tuple(pool.map(allocate, range(4)))

    assert len(set(placements)) == 4


def test_claimed_plan_replacement_waits_for_replay_window(store_factory) -> None:
    create, _store_id = store_factory
    replay_window = timedelta(milliseconds=300)
    first = create(replay_safety_window=replay_window)
    second = create(replay_safety_window=replay_window)
    database_now = _database_now()
    binding = _binding(
        bound_at=database_now - timedelta(milliseconds=50),
        valid_for=timedelta(milliseconds=500),
    )
    first.register(_plan(binding))
    _claim(
        first,
        binding,
        name="qwen-7",
        admission_uid="admission-a",
        claimed_at=binding.bound_at + timedelta(milliseconds=100),
    )

    too_early = _binding(suffix="b")
    with pytest.raises(RunnerCachePlacementAdmissionConflictError, match="safety window"):
        second.register(_plan(too_early))

    time.sleep(0.9)
    successor = _binding(suffix="c")
    second.register(_plan(successor))
    allocated = _claim(
        first,
        successor,
        name="qwen-8",
        admission_uid="admission-successor",
    )
    assert allocated.claim.placement_id == "placement-0-c"


def test_target_capacity_and_registry_configuration_are_fixed(store_factory) -> None:
    create, _store_id = store_factory
    store = create(max_targets=1)
    store.register(_plan(_binding()))
    other = _binding(suffix="other", target_id="deployment/model-serving/other")
    with pytest.raises(RunnerCachePlacementAdmissionConflictError, match="capacity"):
        store.register(_plan(other))

    with pytest.raises(RuntimeError, match="configuration changed"):
        create(max_targets=2)


def test_validate_only_startup_and_projection_corruption_fail_closed(
    store_factory,
) -> None:
    create, store_id = store_factory
    first = create()
    binding = _binding()
    first.register(_plan(binding))
    validate_only = create(initialize_schema=False)
    assert validate_only.check_ready() is None

    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            """
            UPDATE public.runner_cache_placement_admission_plans
            SET binding_id = %s
            WHERE store_id = %s AND target_id = %s
            """,
            ("f" * 64, store_id, _TARGET),
        )
    with pytest.raises(RuntimeError, match="metadata is inconsistent"):
        validate_only.resolve(_TARGET)


def test_deadline_bounded_resolve_and_readiness(store_factory) -> None:
    create, _store_id = store_factory
    store = create()
    plan = _plan(_binding())
    store.register(plan)

    assert store.connect_timeout_s == 10
    assert store.resolve_with_timeout(_TARGET, timeout_s=0.5) == plan
    assert store.check_ready_with_timeout(timeout_s=0.5) is None

    namespace_oid = store._namespace_oid
    assert namespace_oid is not None
    store._namespace_oid = -1
    with pytest.raises(RuntimeError, match="namespace identity changed"):
        store.check_ready_with_timeout(timeout_s=0.5)
    store._namespace_oid = namespace_oid

    with pytest.raises(
        RunnerCachePlacementAdmissionConflictError,
        match="no active",
    ):
        store.resolve_with_timeout("deployment/model-serving/missing", timeout_s=0.5)
    with pytest.raises(ValueError, match="greater than zero"):
        store.resolve_with_timeout(_TARGET, timeout_s=0)

    assert store._connection is not None
    store._connection.close()
    with pytest.raises(TimeoutError, match="do not reconnect"):
        store.resolve_with_timeout(_TARGET, timeout_s=0.5)
    assert store.check_ready() is None
    assert store.resolve_with_timeout(_TARGET, timeout_s=2.0) == plan

    with ThreadPoolExecutor(max_workers=1) as pool:
        with store._lock:
            blocked = pool.submit(
                store.resolve_with_timeout,
                _TARGET,
                timeout_s=0.05,
            )
            with pytest.raises(TimeoutError, match="lock budget"):
                blocked.result(timeout=1)

    with psycopg.connect(_POSTGRES_DSN) as blocker:
        with blocker.transaction():
            blocker.execute(
                "LOCK TABLE public.runner_cache_placement_admission_plans IN ACCESS EXCLUSIVE MODE"
            )
            started_at = time.monotonic()
            with pytest.raises(psycopg.Error):
                store.resolve_with_timeout(_TARGET, timeout_s=0.05)
            assert time.monotonic() - started_at < 0.5

    with psycopg.connect(_POSTGRES_DSN) as blocker:
        with blocker.transaction():
            blocker.execute(
                "LOCK TABLE public.runner_cache_placement_admission_store_registry "
                "IN ACCESS EXCLUSIVE MODE"
            )
            assert store._connection is not None
            store._connection.close()
            started_at = time.monotonic()
            with pytest.raises(TimeoutError, match="do not reconnect"):
                store.resolve_with_timeout(_TARGET, timeout_s=1.25)
            assert time.monotonic() - started_at < 0.5
    assert store.check_ready() is None
    assert store.resolve_with_timeout(_TARGET, timeout_s=2.0) == plan


def test_claim_from_another_binding_fails_closed(store_factory) -> None:
    create, store_id = store_factory
    store = create()
    binding = _binding()
    store.register(_plan(binding))
    _claim(store, binding, name="qwen-7", admission_uid="admission-a")

    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            """
            UPDATE public.runner_cache_placement_admission_claims
            SET binding_id = %s,
                claim = jsonb_set(claim, '{binding_id}', to_jsonb(%s::text), false)
            WHERE store_id = %s AND target_id = %s
            """,
            ("f" * 64, "f" * 64, store_id, _TARGET),
        )
    with pytest.raises(RuntimeError, match="another binding"):
        _claim(
            store,
            binding,
            name="qwen-7",
            admission_uid="admission-retry",
        )


def test_database_clock_rejects_future_plan_and_expired_backdated_claim(
    store_factory,
) -> None:
    create, _store_id = store_factory
    store = create()
    database_now = _database_now()
    future = _binding(
        suffix="future",
        bound_at=database_now + timedelta(minutes=1),
    )
    with pytest.raises(RunnerCachePlacementAdmissionConflictError, match="database clock"):
        store.register(_plan(future))

    expiring = _binding(
        suffix="expiring",
        bound_at=database_now - timedelta(milliseconds=50),
        valid_for=timedelta(milliseconds=300),
    )
    store.register(_plan(expiring))
    time.sleep(0.4)
    with pytest.raises(RunnerCachePlacementAdmissionConflictError, match="not live"):
        _claim(
            store,
            expiring,
            name="qwen-backdated",
            admission_uid="admission-backdated",
            claimed_at=expiring.bound_at + timedelta(milliseconds=100),
        )


@pytest.mark.parametrize("field", ["creator_username", "owner_uid", "release_id"])
def test_full_plan_digest_rejects_authorization_json_corruption(
    store_factory,
    field: str,
) -> None:
    create, store_id = store_factory
    store = create()
    binding = _binding()
    store.register(_plan(binding))

    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            """
            UPDATE public.runner_cache_placement_admission_plans
            SET plan = jsonb_set(plan, %s, to_jsonb(%s::text), false)
            WHERE store_id = %s AND target_id = %s
            """,
            ([field], f"changed-{field}", store_id, _TARGET),
        )
    with pytest.raises(RuntimeError, match="plan digest is inconsistent"):
        store.resolve(_TARGET)


def test_replay_window_must_fit_postgresql_bigint(store_factory) -> None:
    assert _POSTGRES_DSN is not None
    create, _store_id = store_factory
    maximum = create(
        replay_safety_window=timedelta(microseconds=2**63 - 1),
    )
    binding = _binding()
    maximum.register(_plan(binding))
    _claim(maximum, binding, name="qwen-7", admission_uid="admission-a")
    with pytest.raises(RunnerCachePlacementAdmissionConflictError, match="safety window"):
        maximum.register(_plan(_binding(suffix="successor")))

    with pytest.raises(ValueError, match="BIGINT"):
        PostgresRunnerCachePlacementAdmissionStore(
            _POSTGRES_DSN,
            store_id="oversized-window",
            replay_safety_window=timedelta(microseconds=2**63),
            eager_connect=False,
        )


def test_release_fails_closed_and_retains_claim_when_parent_plan_is_corrupt(
    store_factory,
) -> None:
    create, store_id = store_factory
    store = create()
    binding = _binding()
    store.register(_plan(binding))
    created = _claim(store, binding, name="qwen-7", admission_uid="admission-a")

    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            """
            UPDATE public.runner_cache_placement_admission_plans
            SET plan = jsonb_set(
                plan,
                '{creator_username}',
                to_jsonb('changed-creator'::text),
                false
            )
            WHERE store_id = %s AND target_id = %s
            """,
            (store_id, _TARGET),
        )
    with pytest.raises(RuntimeError, match="plan digest is inconsistent"):
        store.release(created.claim)

    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        row = connection.execute(
            """
            SELECT count(*)
            FROM public.runner_cache_placement_admission_claims
            WHERE store_id = %s AND target_id = %s AND pod_key = %s
            """,
            (store_id, _TARGET, created.claim.pod_key),
        ).fetchone()
    assert row == (1,)
