"""Live scaling-controller placement-binding authorization coverage."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
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
    AggregatingRunnerCachePlacementBindingCacheReader,
    AuthenticatedNodeModelCacheLiveEvidenceClient,
    ComposedRunnerCachePlacementBindingLiveStateSource,
    InMemoryRunnerLeaderLeaseStore,
    InvalidKubernetesPlacementBindingLiveResponseError,
    KubernetesKueueRunnerCachePlacementBindingReader,
    KubernetesPlacementBindingLiveTarget,
    LeaderFencedRunnerController,
    NodeModelCacheAgentEndpoint,
    NodeModelCacheLiveEvidenceRequest,
    NodeModelCacheLiveEvidenceResponse,
    PostgresRunnerCachePlacementBindingReader,
    RunnerCachePlacementAdmissionConflictError,
    RunnerCachePlacementAdmissionPlan,
    RunnerCachePlacementBindingAuthorizationDeniedError,
    RunnerCachePlacementBindingAuthorizationError,
    RunnerCachePlacementBindingCacheState,
    RunnerCachePlacementBindingInventory,
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
    ModelCachePlacementCandidate,
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


class _TimedAdmissionPlanStore:
    def __init__(self, plan: RunnerCachePlacementAdmissionPlan) -> None:
        self.plan = plan
        self.error: BaseException | None = None
        self.calls: list[tuple[str, object, float]] = []
        self.on_call = None

    def resolve_with_timeout(self, target_id: str, *, timeout_s: float):
        self.calls.append(("resolve", target_id, timeout_s))
        if self.on_call is not None:
            self.on_call()
        if self.error is not None:
            raise self.error
        return self.plan

    def check_ready_with_timeout(self, *, timeout_s: float) -> None:
        self.calls.append(("ready", None, timeout_s))
        if self.on_call is not None:
            self.on_call()
        if self.error is not None:
            raise self.error


class _TimedDecisionLog:
    def __init__(self, decision: ScalingDecisionRecord) -> None:
        self.decision = decision
        self.error: BaseException | None = None
        self.calls: list[tuple[str, object, float]] = []
        self.on_call = None

    def get_with_timeout(self, decision_id: str, *, timeout_s: float):
        self.calls.append(("get", decision_id, timeout_s))
        if self.on_call is not None:
            self.on_call()
        if self.error is not None:
            raise self.error
        return self.decision

    def check_ready_with_timeout(self, *, timeout_s: float) -> None:
        self.calls.append(("ready", None, timeout_s))
        if self.on_call is not None:
            self.on_call()
        if self.error is not None:
            raise self.error


def _admission_plan(
    binding: RunnerCacheStartupBinding,
) -> RunnerCachePlacementAdmissionPlan:
    return RunnerCachePlacementAdmissionPlan(
        binding=binding,
        release_id="release-a",
        namespace="model-serving",
        owner_api_version="apps/v1",
        owner_kind="Deployment",
        owner_name="qwen-runners",
        owner_uid="workload-uid",
        creator_username="system:serviceaccount:kairyu:statefulset-controller",
        registered_at=binding.bound_at,
    )


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


def _node_live_evidence_response(
    decision: ScalingDecisionRecord,
    binding: RunnerCacheStartupBinding,
) -> NodeModelCacheLiveEvidenceResponse:
    state = _live_state(decision, binding)
    return NodeModelCacheLiveEvidenceResponse(
        node_id="gpu-a",
        prestage_record=state.prestage_records[0],
        placement_hint=state.placement_hints[0],
        pin_evidence=state.pin_evidence[0],
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


def test_postgres_live_reader_reads_current_decision_and_readiness() -> None:
    decision = _decision()
    command = _prestage_command(decision)
    binding = _binding(decision, command_id=command.command_id)
    admission = _TimedAdmissionPlanStore(_admission_plan(binding))
    decisions = _TimedDecisionLog(decision)
    reader = PostgresRunnerCachePlacementBindingReader(
        admission,
        decisions,
        monotonic_clock=lambda: 10.0,
    )

    assert (
        reader.read_current(
            binding,
            deadline_monotonic=20.0,
            backend_timeout_s=2.0,
        )
        == binding
    )
    assert (
        reader.read_decision(
            binding,
            deadline_monotonic=20.0,
            backend_timeout_s=2.0,
        )
        == decision
    )
    reader.readiness(deadline_monotonic=20.0, backend_timeout_s=2.0)

    assert admission.calls == [
        ("resolve", binding.target_id, 2.0),
        ("ready", None, 2.0),
    ]
    assert decisions.calls == [
        ("get", binding.decision_id, 2.0),
        ("ready", None, 2.0),
    ]


def test_postgres_live_reader_maps_missing_authority_to_denial() -> None:
    decision = _decision()
    command = _prestage_command(decision)
    binding = _binding(decision, command_id=command.command_id)
    admission = _TimedAdmissionPlanStore(_admission_plan(binding))
    decisions = _TimedDecisionLog(decision)
    reader = PostgresRunnerCachePlacementBindingReader(
        admission,
        decisions,
        monotonic_clock=lambda: 10.0,
    )

    admission.error = RunnerCachePlacementAdmissionConflictError("missing")
    with pytest.raises(
        RunnerCachePlacementBindingAuthorizationDeniedError,
        match="current placement binding",
    ):
        reader.read_current(
            binding,
            deadline_monotonic=20.0,
            backend_timeout_s=1.0,
        )

    admission.error = None
    decisions.error = KeyError("missing")
    with pytest.raises(
        RunnerCachePlacementBindingAuthorizationDeniedError,
        match="durable scaling decision",
    ):
        reader.read_decision(
            binding,
            deadline_monotonic=20.0,
            backend_timeout_s=1.0,
        )


def test_postgres_live_reader_caps_timeout_and_rejects_late_result() -> None:
    decision = _decision()
    command = _prestage_command(decision)
    binding = _binding(decision, command_id=command.command_id)
    admission = _TimedAdmissionPlanStore(_admission_plan(binding))
    decisions = _TimedDecisionLog(decision)
    monotonic = [10.0]
    reader = PostgresRunnerCachePlacementBindingReader(
        admission,
        decisions,
        monotonic_clock=lambda: monotonic[0],
    )

    assert (
        reader.read_current(
            binding,
            deadline_monotonic=10.25,
            backend_timeout_s=1.0,
        )
        == binding
    )
    assert admission.calls[-1] == ("resolve", binding.target_id, 0.25)

    admission.on_call = lambda: monotonic.__setitem__(0, 10.25)
    with pytest.raises(TimeoutError, match="deadline"):
        reader.read_current(
            binding,
            deadline_monotonic=10.25,
            backend_timeout_s=1.0,
        )


def test_authenticated_node_live_evidence_client_binds_transport_and_readiness() -> None:
    decision = _decision()
    command = _prestage_command(decision)
    binding = _binding(decision, command_id=command.command_id)
    evidence = _node_live_evidence_response(decision, binding)
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.path == "/readyz":
            return httpx.Response(200, json={"status": "ready", "node_id": "gpu-a"})
        assert request.url.path == "/v1/cache/live-evidence"
        return httpx.Response(
            200,
            json=evidence.model_dump(mode="json"),
            headers={"Cache-Control": "private, no-store"},
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
        trust_env=False,
    )
    client = AuthenticatedNodeModelCacheLiveEvidenceClient(
        http_client,
        endpoints=(NodeModelCacheAgentEndpoint(node_id="gpu-a", base_url="https://gpu-a.test"),),
        bearer_token=TOKEN,
        monotonic_clock=lambda: 10.0,
    )
    request = NodeModelCacheLiveEvidenceRequest(
        placement=binding.placements[0],
        model_id=binding.model_id,
        model_revision=binding.model_revision,
    )
    try:
        assert (
            client.read(
                request,
                deadline_monotonic=20.0,
                backend_timeout_s=2.0,
            )
            == evidence
        )
        client.readiness(deadline_monotonic=20.0, backend_timeout_s=2.0)
    finally:
        client.close()

    assert [request.method for request in observed] == ["POST", "GET"]
    assert all(request.headers["authorization"] == f"Bearer {TOKEN}" for request in observed)
    assert observed[0].headers["content-type"] == "application/json"
    assert json.loads(observed[0].content) == request.model_dump(mode="json")


@pytest.mark.parametrize("failure", ["duplicate", "cacheable", "wrong-node", "conflict"])
def test_authenticated_node_live_evidence_client_fails_closed(failure: str) -> None:
    decision = _decision()
    command = _prestage_command(decision)
    binding = _binding(decision, command_id=command.command_id)
    evidence = _node_live_evidence_response(decision, binding)

    def handler(_request: httpx.Request) -> httpx.Response:
        if failure == "duplicate":
            return httpx.Response(
                200,
                content=b'{"node_id":"gpu-a","node_id":"gpu-b"}',
                headers={
                    "Content-Type": "application/json",
                    "Cache-Control": "no-store",
                },
            )
        if failure == "cacheable":
            return httpx.Response(200, json=evidence.model_dump(mode="json"))
        if failure == "wrong-node":
            return httpx.Response(
                200,
                json=evidence.model_copy(update={"node_id": "gpu-b"}).model_dump(mode="json"),
                headers={"Cache-Control": "no-store"},
            )
        return httpx.Response(409, json={"error": {"code": "prestage_conflict"}})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    client = AuthenticatedNodeModelCacheLiveEvidenceClient(
        http_client,
        endpoints=(NodeModelCacheAgentEndpoint(node_id="gpu-a", base_url="https://gpu-a.test"),),
        bearer_token=TOKEN,
        monotonic_clock=lambda: 10.0,
    )
    request = NodeModelCacheLiveEvidenceRequest(
        placement=binding.placements[0],
        model_id=binding.model_id,
        model_revision=binding.model_revision,
    )
    expected = (
        RunnerCachePlacementBindingAuthorizationDeniedError
        if failure in {"wrong-node", "conflict"}
        else RunnerCachePlacementBindingAuthorizationError
    )
    try:
        with pytest.raises(expected):
            client.read(
                request,
                deadline_monotonic=20.0,
                backend_timeout_s=2.0,
            )
    finally:
        client.close()


def test_authenticated_node_live_evidence_client_enforces_size_and_deadline() -> None:
    decision = _decision()
    command = _prestage_command(decision)
    binding = _binding(decision, command_id=command.command_id)
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            content=b"{" + b"x" * 128 + b"}",
            headers={
                "Content-Type": "application/json",
                "Cache-Control": "no-store",
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    request = NodeModelCacheLiveEvidenceRequest(
        placement=binding.placements[0],
        model_id=binding.model_id,
        model_revision=binding.model_revision,
    )
    clock = [10.0]
    limited = AuthenticatedNodeModelCacheLiveEvidenceClient(
        http_client,
        endpoints=(NodeModelCacheAgentEndpoint(node_id="gpu-a", base_url="https://gpu-a.test"),),
        bearer_token=TOKEN,
        response_limit_bytes=64,
        monotonic_clock=lambda: clock[0],
    )
    try:
        with pytest.raises(RunnerCachePlacementBindingAuthorizationError, match="limit"):
            limited.read(
                request,
                deadline_monotonic=20.0,
                backend_timeout_s=2.0,
            )
        clock[0] = 20.0
        with pytest.raises(TimeoutError, match="deadline"):
            limited.read(
                request,
                deadline_monotonic=20.0,
                backend_timeout_s=2.0,
            )
    finally:
        limited.close()

    assert calls == 1


def test_live_cache_reader_aggregates_node_evidence_then_current_inventory() -> None:
    decision = _decision()
    command = _prestage_command(decision)
    binding = _binding(decision, command_id=command.command_id)
    response = _node_live_evidence_response(decision, binding)
    calls: list[str] = []

    class EvidenceReader:
        def read(self, request, *, deadline_monotonic, backend_timeout_s):
            calls.append("evidence")
            assert request.placement == binding.placements[0]
            assert deadline_monotonic == 20.0
            assert backend_timeout_s == 2.0
            return response

        def read_many(self, requests, **kwargs):
            return tuple(self.read(request, **kwargs) for request in requests)

        def readiness(self, *, deadline_monotonic, backend_timeout_s):
            calls.append("evidence-ready")

    class InventoryReader:
        def read_inventory(self, current_binding, current_decision, **kwargs):
            calls.append("inventory")
            assert current_binding == binding
            assert current_decision == decision
            return RunnerCachePlacementBindingInventory(
                snapshot_id="live-cache-2",
                cache_revision=2,
                observed_at=LIVE + timedelta(seconds=1),
                candidates=(
                    ModelCachePlacementCandidate(
                        placement_id="placement-a",
                        node_name="gpu-a",
                        resource_flavor="h100-sxm",
                        profile_id="h100-sxm-tp1",
                        compatibility_approval_id="compat-qwen-h100",
                    ),
                ),
            )

        def readiness(self, *, deadline_monotonic, backend_timeout_s):
            calls.append("inventory-ready")

    reader = AggregatingRunnerCachePlacementBindingCacheReader(
        evidence=EvidenceReader(),
        inventory=InventoryReader(),
        monotonic_clock=lambda: 10.0,
    )

    state = reader.read_cache(
        binding,
        decision,
        deadline_monotonic=20.0,
        backend_timeout_s=2.0,
    )
    reader.readiness(deadline_monotonic=20.0, backend_timeout_s=2.0)

    assert calls == ["evidence", "inventory", "evidence-ready", "inventory-ready"]
    assert state.prewarm_plan.runner_start_placement_ids == ("placement-a",)
    assert state.prewarm_plan.snapshot.snapshot_id == "live-cache-2"
    assert state.prewarm_plan.snapshot.cache_revision == 2
    assert state.prestage_records == (response.prestage_record,)
    assert state.placement_hints == (response.placement_hint,)
    assert state.pin_evidence == (response.pin_evidence,)


def test_live_cache_reader_rejects_inventory_observed_before_node_evidence() -> None:
    decision = _decision()
    command = _prestage_command(decision)
    binding = _binding(decision, command_id=command.command_id)
    response = _node_live_evidence_response(decision, binding)

    class EvidenceReader:
        def read(self, *_args, **_kwargs):
            return response

        def read_many(self, requests, **kwargs):
            return tuple(self.read(request, **kwargs) for request in requests)

        def readiness(self, **_kwargs):
            return None

    class StaleInventoryReader:
        def read_inventory(self, *_args, **_kwargs):
            return RunnerCachePlacementBindingInventory(
                snapshot_id="stale-cache",
                cache_revision=2,
                observed_at=LIVE - timedelta(seconds=1),
                candidates=(
                    ModelCachePlacementCandidate(
                        placement_id="placement-a",
                        node_name="gpu-a",
                        resource_flavor="h100-sxm",
                        profile_id="h100-sxm-tp1",
                        compatibility_approval_id="compat-qwen-h100",
                    ),
                ),
            )

        def readiness(self, **_kwargs):
            return None

    reader = AggregatingRunnerCachePlacementBindingCacheReader(
        evidence=EvidenceReader(),
        inventory=StaleInventoryReader(),
        monotonic_clock=lambda: 10.0,
    )

    with pytest.raises(
        RunnerCachePlacementBindingAuthorizationDeniedError,
        match="inventory",
    ):
        reader.read_cache(
            binding,
            decision,
            deadline_monotonic=20.0,
            backend_timeout_s=2.0,
        )


def test_node_endpoint_rejects_non_https_and_non_origin_urls() -> None:
    with pytest.raises(ValueError, match="HTTPS origin"):
        NodeModelCacheAgentEndpoint(node_id="gpu-a", base_url="http://gpu-a.test")
    with pytest.raises(ValueError, match="HTTPS origin"):
        NodeModelCacheAgentEndpoint(
            node_id="gpu-a",
            base_url="https://gpu-a.test/untrusted/path",
        )


def test_node_live_evidence_client_rejects_environment_proxy_inheritance() -> None:
    unsafe = httpx.AsyncClient()
    try:
        with pytest.raises(ValueError, match="environment trust"):
            AuthenticatedNodeModelCacheLiveEvidenceClient(
                unsafe,
                endpoints=(
                    NodeModelCacheAgentEndpoint(
                        node_id="gpu-a",
                        base_url="https://gpu-a.test",
                    ),
                ),
                bearer_token=TOKEN,
            )
    finally:
        asyncio.run(unsafe.aclose())


def test_node_live_evidence_client_cancels_slow_drip_without_worker_leak() -> None:
    decision = _decision()
    command = _prestage_command(decision)
    binding = _binding(decision, command_id=command.command_id)
    request = NodeModelCacheLiveEvidenceRequest(
        placement=binding.placements[0],
        model_id=binding.model_id,
        model_revision=binding.model_revision,
    )
    stopped = 0
    stopped_lock = threading.Lock()

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            nonlocal stopped
            try:
                while True:
                    await asyncio.sleep(0.01)
                    yield b" "
            finally:
                with stopped_lock:
                    stopped += 1

        async def aclose(self) -> None:
            return None

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=SlowStream(),
            headers={
                "Content-Type": "application/json",
                "Cache-Control": "no-store",
            },
        )

    baseline = sum(
        thread.name == "kairyu-node-cache-live-evidence" for thread in threading.enumerate()
    )
    client = AuthenticatedNodeModelCacheLiveEvidenceClient(
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            trust_env=False,
        ),
        endpoints=(NodeModelCacheAgentEndpoint(node_id="gpu-a", base_url="https://gpu-a.test"),),
        bearer_token=TOKEN,
    )
    try:
        for _ in range(3):
            with pytest.raises(TimeoutError):
                client.read(
                    request,
                    deadline_monotonic=time.monotonic() + 0.04,
                    backend_timeout_s=1.0,
                )
            assert (
                sum(
                    thread.name == "kairyu-node-cache-live-evidence"
                    for thread in threading.enumerate()
                )
                == baseline + 1
            )
    finally:
        client.close()

    assert stopped == 3
    assert (
        sum(thread.name == "kairyu-node-cache-live-evidence" for thread in threading.enumerate())
        == baseline
    )


def test_node_live_evidence_client_cancels_sibling_after_partial_failure() -> None:
    decision = _decision()
    command = _prestage_command(decision)
    binding = _binding(decision, command_id=command.command_id)
    first = NodeModelCacheLiveEvidenceRequest(
        placement=binding.placements[0],
        model_id=binding.model_id,
        model_revision=binding.model_revision,
    )
    second = NodeModelCacheLiveEvidenceRequest(
        placement=binding.placements[0].model_copy(update={"node_name": "gpu-b"}),
        model_id=binding.model_id,
        model_revision=binding.model_revision,
    )
    sibling_started = asyncio.Event()
    sibling_stopped = threading.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "gpu-a.test":
            await sibling_started.wait()
            return httpx.Response(409, json={"error": {"code": "prestage_conflict"}})
        sibling_started.set()
        try:
            await asyncio.sleep(10)
        finally:
            sibling_stopped.set()
        raise AssertionError("cancelled sibling resumed")

    client = AuthenticatedNodeModelCacheLiveEvidenceClient(
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            trust_env=False,
        ),
        endpoints=(
            NodeModelCacheAgentEndpoint(node_id="gpu-a", base_url="https://gpu-a.test"),
            NodeModelCacheAgentEndpoint(node_id="gpu-b", base_url="https://gpu-b.test"),
        ),
        bearer_token=TOKEN,
        max_parallel_requests=2,
    )
    try:
        with pytest.raises(RunnerCachePlacementBindingAuthorizationDeniedError):
            client.read_many(
                (first, second),
                deadline_monotonic=time.monotonic() + 1.0,
                backend_timeout_s=1.0,
            )
        assert sibling_stopped.wait(timeout=0.1)
    finally:
        client.close()


def test_node_live_evidence_client_bounds_pending_tasks_for_large_fanout() -> None:
    decision = _decision()
    command = _prestage_command(decision)
    binding = _binding(decision, command_id=command.command_id)
    request = NodeModelCacheLiveEvidenceRequest(
        placement=binding.placements[0],
        model_id=binding.model_id,
        model_revision=binding.model_revision,
    )
    started = 0
    stopped = 0
    task_counts: list[int] = []

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal started, stopped
        started += 1
        task_counts.append(len(asyncio.all_tasks()))
        try:
            await asyncio.sleep(10)
        finally:
            stopped += 1
        raise AssertionError("cancelled request resumed")

    client = AuthenticatedNodeModelCacheLiveEvidenceClient(
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            trust_env=False,
        ),
        endpoints=(NodeModelCacheAgentEndpoint(node_id="gpu-a", base_url="https://gpu-a.test"),),
        bearer_token=TOKEN,
        max_parallel_requests=3,
    )
    try:
        with pytest.raises(TimeoutError):
            client.read_many(
                (request,) * 100,
                deadline_monotonic=time.monotonic() + 0.04,
                backend_timeout_s=1.0,
            )
    finally:
        client.close()

    assert started == 3
    assert stopped == 3
    assert task_counts and max(task_counts) <= 5


def _kubernetes_live_target() -> KubernetesPlacementBindingLiveTarget:
    return KubernetesPlacementBindingLiveTarget(
        model_class="qwen-14b",
        target_kind="Deployment",
        namespace="model-serving",
        name="qwen-runners",
        authority_namespace="scaling-system",
        kueue_namespace="model-serving",
        quota_snapshot_name="qwen-quota",
        inventory_name="qwen-inventory",
        pod_set_name="runners",
    )


def _kubernetes_workload_payload(
    decision: ScalingDecisionRecord,
    binding: RunnerCacheStartupBinding,
) -> dict[str, object]:
    assert decision.target_revision is not None
    assert decision.decision_generation is not None
    target = decision.target_revision
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": target.name,
            "namespace": target.namespace,
            "uid": target.workload_uid,
            "generation": target.workload_generation + 1,
            "resourceVersion": "100",
            "annotations": {
                "kairyu.ai/release-id": target.release_id,
                "kairyu.ai/model-id": binding.model_id,
                "kairyu.ai/model-revision": target.model_revision,
                "kairyu.ai/cache-placement-binding": binding.placement_binding_id,
                "kairyu.ai/scale-decision-generation": str(decision.decision_generation),
                "kairyu.ai/scale-decision-id": decision.decision_id,
                "kairyu.ai/scale-decision-fingerprint": decision.fingerprint,
            },
        },
        "spec": {"replicas": decision.desired_replicas},
    }


def _kueue_workload_payload(
    decision: ScalingDecisionRecord,
    *,
    admitted: bool = True,
    resource_version: str = "42",
) -> dict[str, object]:
    assert decision.quota_admission is not None
    kueue = decision.quota_admission.snapshot.kueue
    status: dict[str, object] = {
        "conditions": [
            {
                "type": "Admitted",
                "status": "True" if admitted else "False",
                "observedGeneration": kueue.workload_generation,
            }
        ]
    }
    if admitted:
        status["admission"] = {
            "clusterQueue": kueue.cluster_queue,
            "podSetAssignments": [
                {
                    "name": kueue.pod_set_name,
                    "flavors": {kueue.resource_name: kueue.resource_flavor},
                    "resourceUsage": {kueue.resource_name: 1},
                    "count": 1,
                }
            ],
        }
    return {
        "apiVersion": kueue.api_version,
        "kind": "Workload",
        "metadata": {
            "name": kueue.workload_name,
            "namespace": kueue.namespace,
            "uid": kueue.workload_uid,
            "generation": kueue.workload_generation,
            "resourceVersion": resource_version,
            "annotations": {
                "kairyu.ai/scale-target-kind": kueue.target_kind,
                "kairyu.ai/scale-target-namespace": kueue.target_namespace,
                "kairyu.ai/scale-target-name": kueue.target_name,
                "kairyu.ai/scale-target-uid": kueue.target_uid,
            },
        },
        "spec": {
            "queueName": kueue.local_queue,
            "priority": kueue.priority,
            "active": True,
            "podSets": [{"name": kueue.pod_set_name, "count": 1}],
        },
        "status": status,
    }


def _target_reference(decision: ScalingDecisionRecord) -> dict[str, object]:
    assert decision.target_revision is not None
    target = decision.target_revision
    return {
        "apiVersion": "apps/v1",
        "kind": target.target_kind,
        "namespace": target.namespace,
        "name": target.name,
        "uid": target.workload_uid,
    }


def _quota_snapshot_payload(
    decision: ScalingDecisionRecord,
    *,
    ready: bool = True,
    kueue_resource_version: str = "42",
) -> dict[str, object]:
    assert decision.quota_admission is not None
    quota = decision.quota_admission.snapshot
    kueue = quota.kueue
    return {
        "apiVersion": "autoscaling.kairyu.ai/v1alpha1",
        "kind": "RunnerScalingQuotaSnapshot",
        "metadata": {
            "name": "qwen-quota",
            "namespace": "scaling-system",
            "uid": "quota-uid",
            "generation": 5,
            "resourceVersion": "78",
        },
        "spec": {
            "targetRef": _target_reference(decision),
            "tenantId": quota.tenant_id,
            "modelClass": quota.model_class,
            "modelFamily": quota.model_family,
            "gpusPerReplica": quota.gpus_per_replica,
            "kueueWorkloadRef": {
                "apiVersion": kueue.api_version,
                "namespace": kueue.namespace,
                "name": kueue.workload_name,
                "uid": kueue.workload_uid,
            },
        },
        "status": {
            "observedGeneration": 5,
            "conditions": [
                {"type": "Ready", "status": "True" if ready else "False", "observedGeneration": 5}
            ],
            "snapshotId": "quota-live",
            "quotaRevision": 2,
            "observedAt": LIVE.isoformat(),
            "kueueResourceVersion": kueue_resource_version,
            "limits": [
                {
                    "scope": scope.value,
                    "quotaName": f"{scope.value}-quota",
                    "hardLimitGpus": 10,
                    "usedGpusExcludingTarget": 0,
                    "reservedGpusForHigherPriority": 0,
                }
                for scope in ScalingQuotaScope
            ],
        },
    }


def _inventory_payload(
    decision: ScalingDecisionRecord,
    binding: RunnerCacheStartupBinding,
    *,
    ready: bool = True,
) -> dict[str, object]:
    placement = binding.placements[0]
    return {
        "apiVersion": "autoscaling.kairyu.ai/v1alpha1",
        "kind": "RunnerCachePlacementInventory",
        "metadata": {
            "name": "qwen-inventory",
            "namespace": "scaling-system",
            "uid": "inventory-uid",
            "generation": 6,
            "resourceVersion": "79",
        },
        "spec": {
            "targetRef": _target_reference(decision),
            "bindingId": binding.binding_id,
            "decisionId": decision.decision_id,
            "decisionFingerprint": decision.fingerprint,
            "modelClass": binding.model_class,
            "modelId": binding.model_id,
            "modelRevision": binding.model_revision,
            "manifestDigest": binding.manifest_digest,
            "placementBindingId": binding.placement_binding_id,
        },
        "status": {
            "observedGeneration": 6,
            "conditions": [
                {"type": "Ready", "status": "True" if ready else "False", "observedGeneration": 6}
            ],
            "snapshotId": "inventory-live",
            "cacheRevision": 2,
            "observedAt": LIVE.isoformat(),
            "candidates": [
                {
                    "placementId": placement.placement_id,
                    "nodeName": placement.node_name,
                    "resourceFlavor": placement.resource_flavor,
                    "profileId": placement.profile_id,
                    "compatibilityApprovalId": placement.compatibility_approval_id,
                    "assigned": False,
                    "healthy": True,
                    "schedulable": True,
                }
            ],
        },
    }


def _kubernetes_live_reader(
    tmp_path: Path,
    handler,
) -> KubernetesKueueRunnerCachePlacementBindingReader:
    token = tmp_path / "token"
    token.write_text("service-account-token\n", encoding="ascii")

    async def async_handler(request: httpx.Request) -> httpx.Response:
        response = handler(request)
        if asyncio.iscoroutine(response):
            return await response
        return response

    return KubernetesKueueRunnerCachePlacementBindingReader(
        targets=(_kubernetes_live_target(),),
        api_server="https://kubernetes.example",
        token_path=token,
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(async_handler),
            trust_env=False,
            follow_redirects=False,
        ),
        close_client=True,
        wall_clock=lambda: LIVE,
    )


def _kubernetes_live_routes(
    decision: ScalingDecisionRecord,
    binding: RunnerCacheStartupBinding,
) -> dict[str, object]:
    assert decision.quota_admission is not None
    workload_name = decision.quota_admission.snapshot.kueue.workload_name
    return {
        "/apis/apps/v1/namespaces/model-serving/deployments/qwen-runners": (
            _kubernetes_workload_payload(decision, binding)
        ),
        f"/apis/kueue.x-k8s.io/v1beta2/namespaces/model-serving/workloads/{workload_name}": (
            _kueue_workload_payload(decision)
        ),
        "/apis/autoscaling.kairyu.ai/v1alpha1/namespaces/scaling-system/"
        "runnerscalingquotasnapshots/qwen-quota": _quota_snapshot_payload(decision),
        "/apis/autoscaling.kairyu.ai/v1alpha1/namespaces/scaling-system/"
        "runnercacheplacementinventories/qwen-inventory": _inventory_payload(
            decision,
            binding,
        ),
        "/apis/kueue.x-k8s.io/v1beta2/namespaces/model-serving/workloads": {
            "apiVersion": "kueue.x-k8s.io/v1beta2",
            "kind": "WorkloadList",
            "items": [],
        },
    }


def test_kubernetes_kueue_live_reader_reads_target_quota_and_inventory(tmp_path: Path) -> None:
    decision = _decision()
    binding = _binding(decision, command_id=_prestage_command(decision).command_id)
    routes = _kubernetes_live_routes(decision, binding)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=routes[request.url.path])

    reader = _kubernetes_live_reader(tmp_path, handler)
    deadline = time.monotonic() + 1
    try:
        target = reader.read_target(
            binding,
            decision,
            deadline_monotonic=deadline,
            backend_timeout_s=0.5,
        )
        quota = reader.read_quota(
            binding,
            decision,
            deadline_monotonic=deadline,
            backend_timeout_s=0.5,
        )
        inventory = reader.read_inventory(
            binding,
            decision,
            deadline_monotonic=deadline,
            backend_timeout_s=0.5,
        )
    finally:
        reader.close()

    assert target == _target_state(decision, binding)
    assert quota.snapshot.snapshot_id == "quota-live"
    assert quota.snapshot.quota_revision == 2
    assert quota.admitted_replicas == 1
    assert inventory.snapshot_id == "inventory-live"
    assert inventory.cache_revision == 2
    assert inventory.candidates[0].placement_id == "placement-a"
    assert len(requests) == 4
    assert all(
        request.headers["authorization"] == "Bearer service-account-token" for request in requests
    )


def test_kubernetes_kueue_live_reader_readiness_checks_all_authority_routes(
    tmp_path: Path,
) -> None:
    decision = _decision()
    binding = _binding(decision, command_id=_prestage_command(decision).command_id)
    routes = _kubernetes_live_routes(decision, binding)
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json=routes[request.url.path])

    reader = _kubernetes_live_reader(tmp_path, handler)
    try:
        reader.readiness(
            deadline_monotonic=time.monotonic() + 1,
            backend_timeout_s=0.5,
        )
    finally:
        reader.close()

    assert paths == [
        "/apis/apps/v1/namespaces/model-serving/deployments/qwen-runners",
        "/apis/autoscaling.kairyu.ai/v1alpha1/namespaces/scaling-system/"
        "runnerscalingquotasnapshots/qwen-quota",
        "/apis/autoscaling.kairyu.ai/v1alpha1/namespaces/scaling-system/"
        "runnercacheplacementinventories/qwen-inventory",
        "/apis/kueue.x-k8s.io/v1beta2/namespaces/model-serving/workloads",
    ]


def test_kubernetes_kueue_live_reader_maps_missing_authority_to_denial(tmp_path: Path) -> None:
    decision = _decision()
    binding = _binding(decision, command_id=_prestage_command(decision).command_id)
    reader = _kubernetes_live_reader(
        tmp_path,
        lambda _request: httpx.Response(404, json={"kind": "Status"}),
    )
    try:
        with pytest.raises(
            RunnerCachePlacementBindingAuthorizationDeniedError,
            match="unavailable",
        ):
            reader.read_target(
                binding,
                decision,
                deadline_monotonic=time.monotonic() + 1,
                backend_timeout_s=0.5,
            )
    finally:
        reader.close()


def test_kubernetes_kueue_live_reader_keeps_api_failures_as_dependency_errors(
    tmp_path: Path,
) -> None:
    decision = _decision()
    binding = _binding(decision, command_id=_prestage_command(decision).command_id)
    reader = _kubernetes_live_reader(
        tmp_path,
        lambda _request: httpx.Response(503, json={"kind": "Status"}),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            reader.read_target(
                binding,
                decision,
                deadline_monotonic=time.monotonic() + 1,
                backend_timeout_s=0.5,
            )
    finally:
        reader.close()


def test_kubernetes_kueue_live_reader_rejects_stale_quota_projection(tmp_path: Path) -> None:
    decision = _decision()
    binding = _binding(decision, command_id=_prestage_command(decision).command_id)
    routes = _kubernetes_live_routes(decision, binding)
    quota_path = (
        "/apis/autoscaling.kairyu.ai/v1alpha1/namespaces/scaling-system/"
        "runnerscalingquotasnapshots/qwen-quota"
    )
    routes[quota_path] = _quota_snapshot_payload(
        decision,
        kueue_resource_version="old",
    )
    reader = _kubernetes_live_reader(
        tmp_path,
        lambda request: httpx.Response(200, json=routes[request.url.path]),
    )
    try:
        with pytest.raises(
            RunnerCachePlacementBindingAuthorizationDeniedError,
            match="current Kueue",
        ):
            reader.read_quota(
                binding,
                decision,
                deadline_monotonic=time.monotonic() + 1,
                backend_timeout_s=0.5,
            )
    finally:
        reader.close()


def test_kubernetes_kueue_live_reader_returns_revoked_kueue_admission(tmp_path: Path) -> None:
    decision = _decision()
    binding = _binding(decision, command_id=_prestage_command(decision).command_id)
    routes = _kubernetes_live_routes(decision, binding)
    assert decision.quota_admission is not None
    workload_name = decision.quota_admission.snapshot.kueue.workload_name
    routes[f"/apis/kueue.x-k8s.io/v1beta2/namespaces/model-serving/workloads/{workload_name}"] = (
        _kueue_workload_payload(decision, admitted=False, resource_version="44")
    )
    quota_path = (
        "/apis/autoscaling.kairyu.ai/v1alpha1/namespaces/scaling-system/"
        "runnerscalingquotasnapshots/qwen-quota"
    )
    routes[quota_path] = _quota_snapshot_payload(
        decision,
        kueue_resource_version="44",
    )
    reader = _kubernetes_live_reader(
        tmp_path,
        lambda request: httpx.Response(200, json=routes[request.url.path]),
    )
    try:
        quota = reader.read_quota(
            binding,
            decision,
            deadline_monotonic=time.monotonic() + 1,
            backend_timeout_s=0.5,
        )
    finally:
        reader.close()

    assert not quota.snapshot.kueue.admitted
    assert quota.admitted_replicas == 0


def test_kubernetes_kueue_live_reader_rejects_inventory_binding_drift(tmp_path: Path) -> None:
    decision = _decision()
    binding = _binding(decision, command_id=_prestage_command(decision).command_id)
    routes = _kubernetes_live_routes(decision, binding)
    inventory_path = (
        "/apis/autoscaling.kairyu.ai/v1alpha1/namespaces/scaling-system/"
        "runnercacheplacementinventories/qwen-inventory"
    )
    inventory = _inventory_payload(decision, binding)
    inventory["spec"]["bindingId"] = "0" * 64  # type: ignore[index]
    routes[inventory_path] = inventory
    reader = _kubernetes_live_reader(
        tmp_path,
        lambda request: httpx.Response(200, json=routes[request.url.path]),
    )
    try:
        with pytest.raises(
            RunnerCachePlacementBindingAuthorizationDeniedError,
            match="current decision",
        ):
            reader.read_inventory(
                binding,
                decision,
                deadline_monotonic=time.monotonic() + 1,
                backend_timeout_s=0.5,
            )
    finally:
        reader.close()


def test_kubernetes_kueue_live_reader_rejects_unready_projection(tmp_path: Path) -> None:
    decision = _decision()
    binding = _binding(decision, command_id=_prestage_command(decision).command_id)
    routes = _kubernetes_live_routes(decision, binding)
    inventory_path = (
        "/apis/autoscaling.kairyu.ai/v1alpha1/namespaces/scaling-system/"
        "runnercacheplacementinventories/qwen-inventory"
    )
    routes[inventory_path] = _inventory_payload(decision, binding, ready=False)
    reader = _kubernetes_live_reader(
        tmp_path,
        lambda request: httpx.Response(200, json=routes[request.url.path]),
    )
    try:
        with pytest.raises(
            RunnerCachePlacementBindingAuthorizationDeniedError,
            match="not current and ready",
        ):
            reader.read_inventory(
                binding,
                decision,
                deadline_monotonic=time.monotonic() + 1,
                backend_timeout_s=0.5,
            )
    finally:
        reader.close()


def test_kubernetes_kueue_live_reader_rejects_duplicate_json(tmp_path: Path) -> None:
    decision = _decision()
    binding = _binding(decision, command_id=_prestage_command(decision).command_id)
    reader = _kubernetes_live_reader(
        tmp_path,
        lambda _request: httpx.Response(
            200,
            content=b'{"apiVersion":"apps/v1","apiVersion":"apps/v1"}',
            headers={"content-type": "application/json"},
        ),
    )
    try:
        with pytest.raises(
            InvalidKubernetesPlacementBindingLiveResponseError,
            match="strict JSON",
        ):
            reader.read_target(
                binding,
                decision,
                deadline_monotonic=time.monotonic() + 1,
                backend_timeout_s=0.5,
            )
    finally:
        reader.close()


def test_kubernetes_kueue_live_reader_uses_one_backend_budget_for_multi_get(
    tmp_path: Path,
) -> None:
    decision = _decision()
    binding = _binding(decision, command_id=_prestage_command(decision).command_id)
    routes = _kubernetes_live_routes(decision, binding)

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.03)
        return httpx.Response(200, json=routes[request.url.path])

    reader = _kubernetes_live_reader(tmp_path, handler)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            reader.read_quota(
                binding,
                decision,
                deadline_monotonic=started + 1,
                backend_timeout_s=0.05,
            )
    finally:
        reader.close()
    assert time.monotonic() - started < 0.15


def test_kubernetes_kueue_live_reader_bounds_python_lock_wait(tmp_path: Path) -> None:
    decision = _decision()
    binding = _binding(decision, command_id=_prestage_command(decision).command_id)
    routes = _kubernetes_live_routes(decision, binding)
    reader = _kubernetes_live_reader(
        tmp_path,
        lambda request: httpx.Response(200, json=routes[request.url.path]),
    )
    acquired = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with reader._lock:  # noqa: SLF001 - deadline lock regression coverage.
            acquired.set()
            release.wait(timeout=1)

    thread = threading.Thread(target=hold_lock)
    thread.start()
    assert acquired.wait(timeout=0.2)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="lock"):
            reader.read_target(
                binding,
                decision,
                deadline_monotonic=started + 0.04,
                backend_timeout_s=0.5,
            )
    finally:
        release.set()
        thread.join(timeout=0.2)
        reader.close()
    assert time.monotonic() - started < 0.15


def test_kubernetes_kueue_live_reader_joins_cancelled_request_before_close(
    tmp_path: Path,
) -> None:
    decision = _decision()
    binding = _binding(decision, command_id=_prestage_command(decision).command_id)
    started = threading.Event()
    stopped = threading.Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        started.set()
        try:
            await asyncio.sleep(10)
        finally:
            stopped.set()
        raise AssertionError("cancelled Kubernetes request resumed")

    reader = _kubernetes_live_reader(tmp_path, handler)
    loop_thread = reader._loop_thread  # noqa: SLF001 - lifecycle regression coverage.
    with pytest.raises(TimeoutError):
        reader.read_target(
            binding,
            decision,
            deadline_monotonic=time.monotonic() + 0.04,
            backend_timeout_s=0.5,
        )
    assert started.is_set()
    assert stopped.wait(timeout=0.1)
    reader.close()
    assert not loop_thread.is_alive()


def test_kubernetes_kueue_live_reader_bounds_service_account_token_read(
    tmp_path: Path,
) -> None:
    decision = _decision()
    binding = _binding(decision, command_id=_prestage_command(decision).command_id)
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    reader = _kubernetes_live_reader(tmp_path, handler)
    reader._token_path.write_bytes(b"a" * (64 * 1024 + 1))  # noqa: SLF001
    try:
        with pytest.raises(ValueError, match="token exceeds"):
            reader.read_target(
                binding,
                decision,
                deadline_monotonic=time.monotonic() + 1,
                backend_timeout_s=0.5,
            )
    finally:
        reader.close()
    assert not called


def test_kubernetes_kueue_live_reader_keeps_malformed_crd_type_as_dependency_failure(
    tmp_path: Path,
) -> None:
    decision = _decision()
    binding = _binding(decision, command_id=_prestage_command(decision).command_id)
    routes = _kubernetes_live_routes(decision, binding)
    quota_path = (
        "/apis/autoscaling.kairyu.ai/v1alpha1/namespaces/scaling-system/"
        "runnerscalingquotasnapshots/qwen-quota"
    )
    quota = _quota_snapshot_payload(decision)
    quota["spec"]["tenantId"] = ["tenant-a"]  # type: ignore[index]
    routes[quota_path] = quota
    reader = _kubernetes_live_reader(
        tmp_path,
        lambda request: httpx.Response(200, json=routes[request.url.path]),
    )
    try:
        with pytest.raises(ValueError, match="tenantId"):
            reader.read_quota(
                binding,
                decision,
                deadline_monotonic=time.monotonic() + 1,
                backend_timeout_s=0.5,
            )
    finally:
        reader.close()


def test_kubernetes_kueue_live_readiness_checks_each_api_version_per_namespace(
    tmp_path: Path,
) -> None:
    token = tmp_path / "token"
    token.write_text("service-account-token\n", encoding="ascii")
    targets = (
        _kubernetes_live_target(),
        KubernetesPlacementBindingLiveTarget(
            model_class="qwen-32b",
            target_kind="Deployment",
            namespace="model-serving",
            name="qwen-32b-runners",
            authority_namespace="scaling-system",
            kueue_namespace="model-serving",
            kueue_api_version="kueue.x-k8s.io/v1beta1",
            quota_snapshot_name="qwen-32b-quota",
            inventory_name="qwen-32b-inventory",
            pod_set_name="runners",
        ),
    )
    paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/workloads"):
            version = request.url.path.split("/")[3]
            return httpx.Response(
                200,
                json={
                    "apiVersion": f"kueue.x-k8s.io/{version}",
                    "kind": "WorkloadList",
                    "items": [],
                },
            )
        if "runnerscalingquotasnapshots" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "apiVersion": "autoscaling.kairyu.ai/v1alpha1",
                    "kind": "RunnerScalingQuotaSnapshot",
                },
            )
        if "runnercacheplacementinventories" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "apiVersion": "autoscaling.kairyu.ai/v1alpha1",
                    "kind": "RunnerCachePlacementInventory",
                },
            )
        return httpx.Response(200, json={"apiVersion": "apps/v1", "kind": "Deployment"})

    reader = KubernetesKueueRunnerCachePlacementBindingReader(
        targets=targets,
        api_server="https://kubernetes.example",
        token_path=token,
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            trust_env=False,
            follow_redirects=False,
        ),
    )
    try:
        reader.readiness(
            deadline_monotonic=time.monotonic() + 1,
            backend_timeout_s=0.5,
        )
    finally:
        reader.close()

    assert "/apis/kueue.x-k8s.io/v1beta1/namespaces/model-serving/workloads" in paths
    assert "/apis/kueue.x-k8s.io/v1beta2/namespaces/model-serving/workloads" in paths
