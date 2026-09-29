"""Live scaling-controller placement-binding authorization coverage."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from kairyu.artifacts import (
    NodeModelCacheFillResult,
    NodeModelCachePlacementHintSnapshot,
    NodeModelCacheResidentHint,
)
from kairyu.runners import (
    ComposedRunnerCachePlacementBindingLiveStateSource,
    InMemoryRunnerLeaderLeaseStore,
    LeaderFencedRunnerController,
    RunnerCachePlacementBindingAuthorizationDeniedError,
    RunnerCachePlacementBindingCacheState,
    RunnerCachePlacementBindingLiveState,
    RunnerCachePlacementBindingPinEvidence,
    RunnerCachePlacementBindingPrestageEvidence,
    RunnerCachePlacementBindingTargetState,
    RunnerCacheStartupBinding,
    RunnerCacheStartupPlacement,
    RunnerLeaderElector,
    RunnerNotLeaderError,
    RunnerStatusReconciler,
    RunnerWriterAuthority,
    ScalingControllerPlacementBindingAuthority,
    build_runner_start_prestage_commands,
    create_runner_cache_placement_binding_authority_app,
)
from kairyu.runners.prestage import NodeModelPrestageRecord
from kairyu.runners.prewarm import (
    ModelCachePlacement,
    ModelCachePlacementState,
    ScalingPrewarmSnapshot,
    plan_cache_aware_scale_up,
)
from kairyu.runners.scaling import ScalingPolicy
from kairyu.runners.scaling_log import (
    ScalingDecisionAction,
    ScalingDecisionReason,
    ScalingDecisionRecord,
    ScalingDecisionTargetRevision,
    ScalingObservation,
    ScalingObservationWindow,
    ScalingQueueSnapshot,
    ScalingRunnerSnapshot,
)
from kairyu.runners.scaling_quota import (
    KueueScalingAdmission,
    ScalingQuotaLimit,
    ScalingQuotaScope,
    ScalingQuotaSnapshot,
    admit_scaling_quota,
    kueue_scaling_workload_name,
)

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
LIVE = NOW + timedelta(seconds=2)
TOKEN = "a" * 32
NONCE = "b" * 64


def _quota(*, observed_at: datetime = NOW, admitted: bool = True):
    admitted_pods = 1 if admitted else 0
    admitted_gpus = 1 if admitted else 0
    snapshot = ScalingQuotaSnapshot(
        snapshot_id=f"quota-{observed_at.isoformat()}-{admitted}",
        quota_revision=2 if observed_at > NOW else 1,
        observed_at=observed_at,
        tenant_id="tenant-a",
        model_class="qwen-14b",
        model_family="qwen",
        gpus_per_replica=1,
        target_reserved_gpus=admitted_gpus,
        limits=tuple(
            ScalingQuotaLimit(
                scope=scope,
                quota_name=f"{scope.value}-quota",
                hard_limit_gpus=10 if admitted else 0,
                used_gpus_excluding_target=0,
            )
            for scope in ScalingQuotaScope
        ),
        kueue=KueueScalingAdmission(
            api_version="kueue.x-k8s.io/v1beta2",
            namespace="model-serving",
            workload_name=kueue_scaling_workload_name(
                target_kind="Deployment",
                target_namespace="model-serving",
                target_name="qwen-runners",
                target_uid="workload-uid",
            ),
            workload_uid="kueue-workload-uid",
            workload_generation=3,
            resource_version="42" if admitted else "43",
            target_kind="Deployment",
            target_namespace="model-serving",
            target_name="qwen-runners",
            target_uid="workload-uid",
            local_queue="serving",
            cluster_queue="tenant-a-gpu" if admitted else None,
            pod_set_name="runners",
            resource_flavor="h100-sxm" if admitted else None,
            priority=100,
            admitted=admitted,
            admitted_pods=admitted_pods,
            admitted_gpus=admitted_gpus,
        ),
    )
    return admit_scaling_quota(snapshot, current_replicas=0, requested_replicas=1)


def _prewarm(
    *,
    observed_at: datetime = NOW,
    state: ModelCachePlacementState = ModelCachePlacementState.READY,
):
    revision = 2 if observed_at > NOW else 1
    snapshot = ScalingPrewarmSnapshot(
        snapshot_id="cache-original" if observed_at == NOW else "cache-refreshed",
        cache_revision=revision,
        observed_at=observed_at,
        model_class="qwen-14b",
        model_revision="model-revision-a",
        artifact_digest="a" * 64,
        placement_binding_id="placement-binding-a",
        placements=(
            ModelCachePlacement(
                placement_id="placement-a",
                node_name="gpu-a",
                resource_flavor="h100-sxm",
                profile_id="h100-sxm-tp1",
                compatibility_approval_id="compat-qwen-h100",
                state=state,
                cache_hint_observed_at=(
                    observed_at if state is ModelCachePlacementState.READY else None
                ),
                cache_hint_valid_until=(
                    observed_at + timedelta(minutes=5)
                    if state is ModelCachePlacementState.READY
                    else None
                ),
                cache_hint_index_revision=(
                    revision if state is ModelCachePlacementState.READY else None
                ),
            ),
        ),
    )
    return plan_cache_aware_scale_up(
        snapshot,
        current_replicas=0,
        quota_target_replicas=1,
        resource_flavor="h100-sxm",
    )


def _decision() -> ScalingDecisionRecord:
    observation = ScalingObservation(
        observation_id="observation-a",
        model_class="qwen-14b",
        observed_at=NOW,
        queue=ScalingQueueSnapshot(
            observed_at=NOW,
            queue_depth=1,
            interactive_queue_depth=1,
            batch_queue_depth=0,
            oldest_queue_age_seconds=1,
            arrival_rate_per_second=1,
        ),
        runners=ScalingRunnerSnapshot(
            observed_at=NOW,
            current_replicas=0,
            busy_replicas=0,
            ready_replicas=0,
            loading_replicas=0,
            unhealthy_replicas=0,
        ),
    )
    return ScalingDecisionRecord(
        decision_id="decision-a",
        decision_generation=1,
        decided_at=NOW,
        catalog_revision=1,
        policy=ScalingPolicy(
            model_class="qwen-14b",
            policy_revision=1,
            min_replicas=0,
            max_replicas=2,
            warm_buffer_replicas=0,
            scale_to_zero=True,
            scale_to_zero_approval_id="approved-scale-to-zero",
            max_scale_up_step=1,
            max_scale_down_step=1,
            max_observation_age_seconds=30,
        ),
        window=ScalingObservationWindow(
            window_id="window-a",
            model_class="qwen-14b",
            started_at=NOW,
            ended_at=NOW,
            observations=(observation,),
        ),
        target_revision=ScalingDecisionTargetRevision(
            target_kind="Deployment",
            namespace="model-serving",
            name="qwen-runners",
            election_id="runner-control-plane",
            fencing_token=1,
            workload_uid="workload-uid",
            workload_generation=7,
            release_id="release-a",
            model_revision="model-revision-a",
        ),
        quota_admission=_quota(),
        prewarm_plan=_prewarm(),
        action=ScalingDecisionAction.SCALE_UP,
        reason=ScalingDecisionReason.QUEUE_PRESSURE,
        demand_replicas=1,
        buffered_target_replicas=1,
        desired_replicas=1,
        target_delta=1,
    )


def _prestage_command(
    decision: ScalingDecisionRecord,
    *,
    command_generation: int = 1,
):
    assert decision.prewarm_plan is not None
    return build_runner_start_prestage_commands(
        decision.prewarm_plan,
        authority=RunnerWriterAuthority(
            election_id="runner-control-plane",
            holder_id="controller-a",
            fencing_token=1,
            validated_at=NOW,
            lease_until=NOW + timedelta(minutes=1),
        ),
        decision_id=decision.decision_id,
        decision_fingerprint=decision.fingerprint,
        target_id="deployment/model-serving/qwen-runners",
        target_revision=7,
        deployment_id="model-serving/qwen",
        model_id="org/qwen",
        command_generations={"placement-a": command_generation},
        issued_at=NOW,
        ttl_seconds=30,
    )[0]


def _binding(decision: ScalingDecisionRecord, *, command_id: str) -> RunnerCacheStartupBinding:
    placement = RunnerCacheStartupPlacement(
        placement_id="placement-a",
        node_name="gpu-a",
        resource_flavor="h100-sxm",
        profile_id="h100-sxm-tp1",
        compatibility_approval_id="compat-qwen-h100",
        manifest_digest="a" * 64,
        pin_owner="prestage/model-serving/qwen/placement-a",
        prestage_command_id=command_id,
        prestage_command_generation=1,
        hint_index_revision=1,
        resident_record_generation=1,
        hint_observed_at=NOW,
        hint_valid_until=NOW + timedelta(minutes=5),
    )
    payload = {
        "schema_version": "runner-cache-startup-binding-v1",
        "decision_id": decision.decision_id,
        "decision_fingerprint": decision.fingerprint,
        "target_id": "deployment/model-serving/qwen-runners",
        "target_revision": 7,
        "deployment_id": "model-serving/qwen",
        "model_class": "qwen-14b",
        "model_id": "org/qwen",
        "model_revision": "model-revision-a",
        "manifest_digest": "a" * 64,
        "placement_binding_id": "placement-binding-a",
        "prewarm_snapshot_id": "cache-original",
        "prewarm_cache_revision": 1,
        "bound_at": NOW + timedelta(seconds=1),
        "valid_until": NOW + timedelta(minutes=5),
        "placements": (placement,),
    }
    unsigned = RunnerCacheStartupBinding.model_construct(binding_id="0" * 64, **payload)
    encoded = json.dumps(
        unsigned.model_dump(mode="json", exclude={"binding_id"}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return RunnerCacheStartupBinding(binding_id=hashlib.sha256(encoded).hexdigest(), **payload)


def _live_cache_evidence(
    decision: ScalingDecisionRecord,
    *,
    observed_at: datetime = LIVE,
    command_generation: int = 1,
    resident_generation: int = 1,
    pinned: bool = True,
    hint_index_revision: int = 2,
):
    command = _prestage_command(decision, command_generation=command_generation)
    record = NodeModelPrestageRecord(
        command=command,
        state=ModelCachePlacementState.READY,
        attempt=1,
        fill_result=NodeModelCacheFillResult(
            deployment_id="model-serving/qwen",
            manifest_digest="a" * 64,
            artifact_path=Path("/mnt/nvme/kairyu-model-cache/artifacts") / ("a" * 64),
            cache_hit=True,
            resumed_bytes=0,
            downloaded_bytes=0,
            file_count=1,
            total_bytes=1024,
        ),
        pin_record_generation=resident_generation,
        updated_at=NOW + timedelta(seconds=1),
    )
    hint = NodeModelCachePlacementHintSnapshot(
        node_id="gpu-a",
        index_revision=hint_index_revision,
        observed_at=observed_at,
        valid_until=observed_at + timedelta(minutes=5),
        residents=(
            NodeModelCacheResidentHint(
                node_id="gpu-a",
                manifest_digest="a" * 64,
                model_id="org/qwen",
                model_revision="model-revision-a",
                total_bytes=1024,
                file_count=1,
                verified_at_ns=1,
                last_access_at_ns=2,
                pinned=pinned,
                record_generation=resident_generation,
            ),
        ),
    )
    return record, hint


def _target_state(
    decision: ScalingDecisionRecord,
    binding: RunnerCacheStartupBinding,
    *,
    decision_id: str | None = None,
) -> RunnerCachePlacementBindingTargetState:
    assert decision.target_revision is not None
    assert decision.decision_generation is not None
    target = decision.target_revision
    return RunnerCachePlacementBindingTargetState(
        observed_at=LIVE,
        target_kind=target.target_kind,
        namespace=target.namespace,
        name=target.name,
        workload_uid=target.workload_uid,
        workload_generation=target.workload_generation + 1,
        release_id=target.release_id,
        model_id=binding.model_id,
        model_revision=target.model_revision,
        placement_binding_id=binding.placement_binding_id,
        decision_generation=decision.decision_generation,
        decision_id=decision_id or decision.decision_id,
        decision_fingerprint=decision.fingerprint,
        replicas=decision.desired_replicas,
    )


class _Source:
    def __init__(self, state: RunnerCachePlacementBindingLiveState) -> None:
        self.state = state
        self.read_calls: list[tuple[float, float]] = []
        self.ready_calls: list[tuple[float, float]] = []
        self.on_read = None

    def read(self, _binding, *, deadline_monotonic: float, backend_timeout_s: float):
        self.read_calls.append((deadline_monotonic, backend_timeout_s))
        if self.on_read is not None:
            self.on_read()
        return self.state

    def readiness(self, *, deadline_monotonic: float, backend_timeout_s: float) -> None:
        self.ready_calls.append((deadline_monotonic, backend_timeout_s))


class _ComponentReaders:
    def __init__(self, state: RunnerCachePlacementBindingLiveState) -> None:
        self.state = state
        self.current_values = [state.binding, state.binding]
        self.calls: list[tuple[str, float, float]] = []
        self.read_hook = None
        self.ready_calls = 0

    def _record(self, name, deadline_monotonic, backend_timeout_s):
        self.calls.append((name, deadline_monotonic, backend_timeout_s))
        if self.read_hook is not None:
            self.read_hook(name)

    def read_current(self, _candidate, *, deadline_monotonic: float, backend_timeout_s: float):
        self._record("current", deadline_monotonic, backend_timeout_s)
        return self.current_values.pop(0)

    def read_decision(self, _binding, *, deadline_monotonic: float, backend_timeout_s: float):
        self._record("decision", deadline_monotonic, backend_timeout_s)
        return self.state.decision

    def read_target(
        self,
        _binding,
        _decision,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ):
        self._record("target", deadline_monotonic, backend_timeout_s)
        return self.state.target

    def read_quota(
        self,
        _binding,
        _decision,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ):
        self._record("quota", deadline_monotonic, backend_timeout_s)
        return self.state.quota_admission

    def read_cache(
        self,
        _binding,
        _decision,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ):
        self._record("cache", deadline_monotonic, backend_timeout_s)
        return RunnerCachePlacementBindingCacheState(
            prewarm_plan=self.state.prewarm_plan,
            prestage_records=self.state.prestage_records,
            placement_hints=self.state.placement_hints,
            pin_evidence=self.state.pin_evidence,
        )

    def readiness(self, *, deadline_monotonic: float, backend_timeout_s: float) -> None:
        self.ready_calls += 1
        self._record("ready", deadline_monotonic, backend_timeout_s)


def _live_state(
    decision: ScalingDecisionRecord,
    binding: RunnerCacheStartupBinding,
    *,
    observed_at: datetime = LIVE,
    quota_admission=None,
    prewarm_plan=None,
    target=None,
    command_generation: int = 1,
    resident_generation: int = 1,
    pinned: bool = True,
    hint_index_revision: int = 2,
    evidence_observed_at: datetime | None = None,
    pin_owners: tuple[str, ...] = ("prestage/model-serving/qwen/placement-a",),
) -> RunnerCachePlacementBindingLiveState:
    record, hint = _live_cache_evidence(
        decision,
        observed_at=evidence_observed_at or observed_at,
        command_generation=command_generation,
        resident_generation=resident_generation,
        pinned=pinned,
        hint_index_revision=hint_index_revision,
    )
    return RunnerCachePlacementBindingLiveState(
        observed_at=observed_at,
        binding=binding,
        decision=decision,
        target=target or _target_state(decision, binding),
        quota_admission=quota_admission or _quota(observed_at=observed_at),
        prewarm_plan=prewarm_plan or _prewarm(observed_at=observed_at),
        prestage_records=(RunnerCachePlacementBindingPrestageEvidence.from_record(record),),
        placement_hints=(hint,),
        pin_evidence=(
            RunnerCachePlacementBindingPinEvidence(
                node_id="gpu-a",
                index_revision=hint.index_revision,
                observed_at=hint.observed_at,
                manifest_digest="a" * 64,
                model_id="org/qwen",
                model_revision="model-revision-a",
                record_generation=resident_generation,
                pin_owners=pin_owners,
            ),
        ),
    )


def _authority(*, monotonic=None, leader_time: datetime = LIVE):
    decision = _decision()
    command = _prestage_command(decision)
    binding = _binding(decision, command_id=command.command_id)
    state = _live_state(decision, binding)
    leader_clock = [leader_time]
    store = InMemoryRunnerLeaderLeaseStore(clock=lambda: leader_clock[0])
    elector = RunnerLeaderElector(
        store,
        election_id="runner-control-plane",
        holder_id="controller-a",
        lease_seconds=30,
    )
    assert elector.campaign() is not None
    controller = LeaderFencedRunnerController(elector, RunnerStatusReconciler())
    source = _Source(state)
    clock = monotonic or (lambda: 10.0)
    authority = ScalingControllerPlacementBindingAuthority(
        controller,
        source,
        monotonic_clock=clock,
    )
    return authority, source, binding, decision, leader_clock, store


def _app(authority: ScalingControllerPlacementBindingAuthority):
    return create_runner_cache_placement_binding_authority_app(
        reauthorize=authority.reauthorize,
        readiness_check=authority.readiness,
        bearer_token=TOKEN,
        request_timeout_s=1.5,
        backend_timeout_s=1.0,
    )


def _payload(binding: RunnerCacheStartupBinding):
    return {
        "schema_version": "kairyu-runner-cache-placement-binding-authorization-request-v1",
        "nonce": NONCE,
        "binding": binding.model_dump(mode="json"),
    }


def _binding_with_deployment(
    binding: RunnerCacheStartupBinding, deployment_id: str
) -> RunnerCacheStartupBinding:
    payload = binding.model_dump(mode="json", exclude={"binding_id"})
    payload["deployment_id"] = deployment_id
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return RunnerCacheStartupBinding(
        binding_id=hashlib.sha256(encoded).hexdigest(),
        **payload,
    )


@pytest.mark.asyncio
async def test_live_controller_authority_serves_only_fully_reauthorized_binding() -> None:
    authority, source, binding, _decision_value, _leader_clock, _store = _authority()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(authority)),
        base_url="https://authority.test",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        ready = await client.get("/readyz")
        response = await client.post("/v1/reauthorize", json=_payload(binding))

    assert ready.status_code == 204
    assert response.status_code == 200
    assert response.json()["binding"] == binding.model_dump(mode="json")
    assert len(source.ready_calls) == 1
    assert len(source.read_calls) == 1
    assert source.read_calls[0][1] == 1.0


@pytest.mark.asyncio
async def test_live_controller_authority_denies_quota_or_cache_drift() -> None:
    authority, source, binding, decision, _leader_clock, _store = _authority()
    cases = (
        _live_state(
            decision,
            binding,
            quota_admission=_quota(observed_at=LIVE, admitted=False),
        ),
        _live_state(
            decision,
            binding,
            prewarm_plan=_prewarm(
                observed_at=LIVE,
                state=ModelCachePlacementState.ABSENT,
            ),
        ),
        _live_state(
            decision,
            binding,
            target=_target_state(decision, binding, decision_id="superseding-decision"),
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(authority)),
        base_url="https://authority.test",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        for state in cases:
            source.state = state
            response = await client.post("/v1/reauthorize", json=_payload(binding))
            assert response.status_code == 409
            assert response.json() == {"detail": "binding is not currently authorized"}


@pytest.mark.asyncio
async def test_live_controller_authority_denies_pin_lineage_and_hint_rollback() -> None:
    authority, source, binding, decision, _leader_clock, _store = _authority()
    rollback_time = NOW - timedelta(seconds=1)
    cases = (
        _live_state(decision, binding, pinned=False),
        _live_state(decision, binding, resident_generation=2),
        _live_state(decision, binding, command_generation=2),
        _live_state(
            decision,
            binding,
            pin_owners=("prestage/model-serving/qwen/other-placement",),
        ),
        _live_state(
            decision,
            binding,
            prewarm_plan=_prewarm(observed_at=rollback_time),
            evidence_observed_at=rollback_time,
            hint_index_revision=1,
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(authority)),
        base_url="https://authority.test",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        for state in cases:
            source.state = state
            response = await client.post("/v1/reauthorize", json=_payload(binding))
            assert response.status_code == 409
            assert response.json() == {"detail": "binding is not currently authorized"}


def test_live_controller_authority_accepts_snapshot_observed_during_read() -> None:
    authority, source, binding, _decision_value, leader_clock, _store = _authority(
        leader_time=NOW + timedelta(seconds=1)
    )
    source.on_read = lambda: leader_clock.__setitem__(0, LIVE)

    assert (
        authority.reauthorize(
            binding,
            deadline_monotonic=20.0,
            backend_timeout_s=1.0,
        )
        == binding
    )


def test_live_controller_authority_fails_closed_on_deadline_and_lost_leadership() -> None:
    monotonic = [10.0]
    authority, source, binding, _decision_value, leader_clock, _store = _authority(
        monotonic=lambda: monotonic[0]
    )

    original_read = source.read

    def late_read(*args, **kwargs):
        state = original_read(*args, **kwargs)
        monotonic[0] = 20.0
        return state

    source.read = late_read  # type: ignore[method-assign]
    with pytest.raises(TimeoutError):
        authority.reauthorize(
            binding,
            deadline_monotonic=20.0,
            backend_timeout_s=1.0,
        )

    monotonic[0] = 10.0
    source.read = original_read  # type: ignore[method-assign]
    leader_clock[0] = LIVE + timedelta(seconds=31)
    with pytest.raises(RunnerNotLeaderError, match="leader"):
        authority.reauthorize(
            binding,
            deadline_monotonic=20.0,
            backend_timeout_s=1.0,
        )


def test_live_controller_authority_rejects_takeover_during_source_read() -> None:
    authority, source, binding, _decision_value, leader_clock, store = _authority()

    def replace_leader() -> None:
        leader_clock[0] = LIVE + timedelta(seconds=31)
        successor = RunnerLeaderElector(
            store,
            election_id="runner-control-plane",
            holder_id="controller-b",
            lease_seconds=30,
        )
        assert successor.campaign() is not None

    source.on_read = replace_leader
    with pytest.raises(RunnerNotLeaderError, match="leader"):
        authority.reauthorize(
            binding,
            deadline_monotonic=20.0,
            backend_timeout_s=1.0,
        )


def test_live_controller_authority_checks_deadline_after_final_validation() -> None:
    moments = iter((10.0, 10.0, 10.0, 10.0, 20.0))
    authority, _source, binding, _decision_value, _leader_clock, _store = _authority(
        monotonic=lambda: next(moments)
    )

    with pytest.raises(TimeoutError, match="deadline"):
        authority.reauthorize(
            binding,
            deadline_monotonic=20.0,
            backend_timeout_s=1.0,
        )


def test_composed_live_source_reads_all_backends_and_fences_current_binding() -> None:
    _authority_value, source, binding, _decision_value, _leader_clock, _store = _authority()
    readers = _ComponentReaders(source.state)
    composed = ComposedRunnerCachePlacementBindingLiveStateSource(
        current=readers,
        decisions=readers,
        targets=readers,
        quotas=readers,
        cache=readers,
        monotonic_clock=lambda: 10.0,
        wall_clock=lambda: LIVE,
    )

    assert (
        composed.read(
            binding,
            deadline_monotonic=20.0,
            backend_timeout_s=1.0,
        )
        == source.state
    )
    assert [name for name, _deadline, _timeout in readers.calls] == [
        "current",
        "decision",
        "target",
        "quota",
        "cache",
        "current",
    ]
    assert all(timeout == 1.0 for _name, _deadline, timeout in readers.calls)

    composed.readiness(deadline_monotonic=20.0, backend_timeout_s=1.0)
    assert readers.ready_calls == 1


def test_composed_live_source_denies_binding_replacement_during_reads() -> None:
    _authority_value, source, binding, _decision_value, _leader_clock, _store = _authority()
    readers = _ComponentReaders(source.state)
    readers.current_values[-1] = _binding_with_deployment(binding, "model-serving/other")
    composed = ComposedRunnerCachePlacementBindingLiveStateSource(
        current=readers,
        decisions=readers,
        targets=readers,
        quotas=readers,
        cache=readers,
        monotonic_clock=lambda: 10.0,
        wall_clock=lambda: LIVE,
    )

    with pytest.raises(
        RunnerCachePlacementBindingAuthorizationDeniedError,
        match="changed",
    ):
        composed.read(
            binding,
            deadline_monotonic=20.0,
            backend_timeout_s=1.0,
        )


def test_composed_live_source_stops_when_shared_deadline_expires() -> None:
    _authority_value, source, binding, _decision_value, _leader_clock, _store = _authority()
    monotonic = [10.0]
    readers = _ComponentReaders(source.state)

    def expire_after_target(name: str) -> None:
        if name == "target":
            monotonic[0] = 20.0

    readers.read_hook = expire_after_target
    composed = ComposedRunnerCachePlacementBindingLiveStateSource(
        current=readers,
        decisions=readers,
        targets=readers,
        quotas=readers,
        cache=readers,
        monotonic_clock=lambda: monotonic[0],
        wall_clock=lambda: LIVE,
    )

    with pytest.raises(TimeoutError, match="deadline"):
        composed.read(
            binding,
            deadline_monotonic=20.0,
            backend_timeout_s=15.0,
        )
    assert [name for name, _deadline, _timeout in readers.calls] == [
        "current",
        "decision",
        "target",
    ]


def test_composed_live_source_checks_deadline_after_snapshot_validation() -> None:
    _authority_value, source, binding, _decision_value, _leader_clock, _store = _authority()
    readers = _ComponentReaders(source.state)
    moments = iter((*([10.0] * 13), 20.0))
    composed = ComposedRunnerCachePlacementBindingLiveStateSource(
        current=readers,
        decisions=readers,
        targets=readers,
        quotas=readers,
        cache=readers,
        monotonic_clock=lambda: next(moments),
        wall_clock=lambda: LIVE,
    )

    with pytest.raises(TimeoutError, match="deadline"):
        composed.read(
            binding,
            deadline_monotonic=20.0,
            backend_timeout_s=1.0,
        )
