"""Atomic Kubernetes CAS for D3.2 Runner cache scheduling bindings."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from kairyu.runners import (
    InMemoryScalingDecisionLog,
    ModelCachePlacement,
    ModelCachePlacementState,
    RunnerCacheStartupBinding,
    RunnerCacheStartupPlacement,
    RunnerWriterAuthority,
    ScalingPrewarmSnapshot,
    admit_scaling_quota,
    bind_runner_cache_to_pod_template,
    plan_cache_aware_scale_up,
)
from kairyu.runners.kubernetes import (
    MODEL_ID_ANNOTATION,
    MODEL_REVISION_ANNOTATION,
    RELEASE_ID_ANNOTATION,
)
from kairyu.runners.scale_actuator import (
    CACHE_PLACEMENT_BINDING_ANNOTATION,
    SCALE_DECISION_FINGERPRINT_ANNOTATION,
    SCALE_DECISION_GENERATION_ANNOTATION,
    SCALE_DECISION_ID_ANNOTATION,
    SCALE_ELECTION_ID_ANNOTATION,
    SCALE_FENCING_TOKEN_ANNOTATION,
    InvalidKubernetesScaleResponseError,
    KubernetesScalableKind,
    KubernetesScaleActuator,
    KubernetesScaleConflictError,
    KubernetesScaleFence,
    KubernetesScaleTarget,
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
    kueue_scaling_workload_name,
)
from kairyu.runners.startup_scheduling import (
    RUNNER_CACHE_STARTUP_BINDING_ID_ANNOTATION,
)

NOW = datetime(2026, 9, 29, 10, 0, tzinfo=UTC)
DIGEST = "a" * 64


def _target() -> KubernetesScaleTarget:
    return KubernetesScaleTarget(
        model_class="qwen-14b",
        namespace="model-serving",
        name="qwen-runners",
        kind=KubernetesScalableKind.DEPLOYMENT,
    )


def _fence() -> KubernetesScaleFence:
    return KubernetesScaleFence(
        workload_uid="workload-uid",
        workload_generation=7,
        release_id="release-a",
        model_revision="revision-a",
    )


def _authority(*, validated_at: datetime = NOW) -> RunnerWriterAuthority:
    return RunnerWriterAuthority(
        election_id="runner-controller",
        holder_id="controller-a",
        fencing_token=3,
        validated_at=validated_at,
        lease_until=validated_at + timedelta(minutes=20),
    )


def _prewarm_plan():
    snapshot = ScalingPrewarmSnapshot(
        snapshot_id="cache-snapshot-a",
        cache_revision=9,
        observed_at=NOW,
        model_class="qwen-14b",
        model_revision="revision-a",
        artifact_digest=DIGEST,
        placement_binding_id="placement-binding-a",
        placements=tuple(
            ModelCachePlacement(
                placement_id=f"placement-{index}",
                node_name=f"gpu-{index}",
                resource_flavor="h100-sxm",
                profile_id="h100-sxm-tp1",
                compatibility_approval_id="compat-qwen-h100",
                state=ModelCachePlacementState.READY,
                cache_hint_observed_at=NOW,
                cache_hint_valid_until=NOW + timedelta(minutes=5),
                cache_hint_index_revision=10 + index,
            )
            for index in range(2)
        ),
    )
    return plan_cache_aware_scale_up(
        snapshot,
        current_replicas=0,
        quota_target_replicas=2,
        resource_flavor="h100-sxm",
    )


def _quota():
    snapshot = ScalingQuotaSnapshot(
        snapshot_id="quota-a",
        quota_revision=4,
        observed_at=NOW,
        tenant_id="tenant-a",
        model_class="qwen-14b",
        model_family="qwen",
        gpus_per_replica=1,
        target_reserved_gpus=2,
        limits=tuple(
            ScalingQuotaLimit(
                scope=scope,
                quota_name=f"{scope.value}-quota",
                hard_limit_gpus=10,
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
            workload_uid="kueue-uid",
            workload_generation=2,
            resource_version="22",
            target_kind="Deployment",
            target_namespace="model-serving",
            target_name="qwen-runners",
            target_uid="workload-uid",
            local_queue="serving",
            cluster_queue="tenant-a-gpu",
            pod_set_name="runners",
            resource_flavor="h100-sxm",
            priority_class_name="interactive-serving",
            priority_class_group="kueue.x-k8s.io",
            priority_class_kind="WorkloadPriorityClass",
            priority=1000,
            admitted=True,
            admitted_pods=2,
            admitted_gpus=2,
        ),
    )
    return admit_scaling_quota(snapshot, current_replicas=0, requested_replicas=2)


def _decision(log: InMemoryScalingDecisionLog) -> ScalingDecisionRecord:
    observation = ScalingObservation(
        observation_id="observation-a",
        model_class="qwen-14b",
        observed_at=NOW,
        queue=ScalingQueueSnapshot(
            observed_at=NOW,
            queue_depth=2,
            interactive_queue_depth=2,
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
    record = ScalingDecisionRecord(
        decision_id="decision-a",
        decided_at=NOW,
        catalog_revision=1,
        policy=ScalingPolicy(
            model_class="qwen-14b",
            policy_revision=1,
            min_replicas=0,
            max_replicas=10,
            warm_buffer_replicas=0,
            scale_to_zero=True,
            scale_to_zero_approval_id="approval-scale-zero-a",
            max_scale_up_step=2,
            max_scale_down_step=1,
            max_observation_age_seconds=600,
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
            election_id="runner-controller",
            fencing_token=3,
            workload_uid="workload-uid",
            workload_generation=7,
            release_id="release-a",
            model_revision="revision-a",
        ),
        quota_admission=_quota(),
        prewarm_plan=_prewarm_plan(),
        action=ScalingDecisionAction.SCALE_UP,
        reason=ScalingDecisionReason.QUEUE_PRESSURE,
        demand_replicas=2,
        buffered_target_replicas=2,
        desired_replicas=2,
        target_delta=2,
    )
    return log.append(record)


def _binding(decision: ScalingDecisionRecord, *, fingerprint: str | None = None):
    placements = tuple(
        RunnerCacheStartupPlacement(
            placement_id=f"placement-{index}",
            node_name=f"gpu-{index}",
            resource_flavor="h100-sxm",
            profile_id="h100-sxm-tp1",
            compatibility_approval_id="compat-qwen-h100",
            manifest_digest=DIGEST,
            pin_owner=f"prestage/model-serving/qwen/placement-{index}",
            prestage_command_id=hashlib.sha256(f"command-{index}".encode()).hexdigest(),
            prestage_command_generation=index + 1,
            hint_index_revision=10 + index,
            resident_record_generation=20 + index,
            hint_observed_at=NOW,
            hint_valid_until=NOW + timedelta(minutes=5),
        )
        for index in range(2)
    )
    payload = {
        "schema_version": "runner-cache-startup-binding-v1",
        "decision_id": decision.decision_id,
        "decision_fingerprint": fingerprint or decision.fingerprint,
        "target_id": "deployment/model-serving/qwen-runners",
        "target_revision": 7,
        "deployment_id": "model-serving/qwen",
        "model_class": "qwen-14b",
        "model_id": "org/qwen",
        "model_revision": "revision-a",
        "manifest_digest": DIGEST,
        "placement_binding_id": "placement-binding-a",
        "prewarm_snapshot_id": "cache-snapshot-a",
        "prewarm_cache_revision": 9,
        "bound_at": NOW,
        "valid_until": NOW + timedelta(minutes=5),
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
    return RunnerCacheStartupBinding(binding_id=hashlib.sha256(encoded).hexdigest(), **payload)


def _annotations(decision: ScalingDecisionRecord | None = None) -> dict[str, str]:
    result = {
        RELEASE_ID_ANNOTATION: "release-a",
        MODEL_ID_ANNOTATION: "org/qwen",
        MODEL_REVISION_ANNOTATION: "revision-a",
        CACHE_PLACEMENT_BINDING_ANNOTATION: "placement-binding-a",
        SCALE_ELECTION_ID_ANNOTATION: "runner-controller",
        SCALE_FENCING_TOKEN_ANNOTATION: "3",
    }
    if decision is not None:
        assert decision.decision_generation is not None
        result.update(
            {
                SCALE_DECISION_GENERATION_ANNOTATION: str(decision.decision_generation),
                SCALE_DECISION_ID_ANNOTATION: decision.decision_id,
                SCALE_DECISION_FINGERPRINT_ANNOTATION: decision.fingerprint,
            }
        )
    return result


def _template() -> dict:
    return {
        "metadata": {"labels": {"app": "qwen"}},
        "spec": {"containers": [{"name": "runner", "image": "runner@sha256:a"}]},
    }


def _workload(
    *,
    decision: ScalingDecisionRecord | None = None,
    template: dict | None = None,
    replicas: int = 0,
    generation: int = 7,
    resource_version: str = "10",
) -> dict:
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "qwen-runners",
            "namespace": "model-serving",
            "uid": "workload-uid",
            "resourceVersion": resource_version,
            "generation": generation,
            "annotations": _annotations(decision),
        },
        "spec": {"replicas": replicas, "template": template or _template()},
    }


def _actuator(tmp_path: Path, handler, log: InMemoryScalingDecisionLog):
    token = tmp_path / "token"
    token.write_text("token", encoding="utf-8")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return KubernetesScaleActuator(
        api_server="https://kubernetes.example",
        token_path=token,
        client=client,
        decision_log=log,
    ), client


def test_scale_up_atomically_cas_binds_template_and_replicas(tmp_path: Path) -> None:
    log = InMemoryScalingDecisionLog()
    decision = _decision(log)
    binding = _binding(decision)
    state = _workload()
    patches: list[list[dict]] = []
    authority_checks: list[datetime] = []

    def reauthorize() -> RunnerWriterAuthority:
        authority_checks.append(NOW)
        return _authority()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal state
        if request.method == "GET":
            return httpx.Response(200, json=state)
        patch = json.loads(request.content)
        patches.append(patch)
        values = {operation["path"]: operation.get("value") for operation in patch}
        state = _workload(
            decision=decision,
            template=values["/spec/template"],
            replicas=2,
            generation=8,
            resource_version="11",
        )
        return httpx.Response(200, json=state)

    actuator, client = _actuator(tmp_path, handler, log)
    result = actuator.apply_fenced(
        decision,
        _target(),
        authority=_authority(),
        fence=_fence(),
        reauthorize=reauthorize,
        reauthorize_quota=lambda: decision.quota_admission,
        reauthorize_prewarm=lambda: decision.prewarm_plan,
        startup_binding=binding,
        reauthorize_startup_binding=lambda: binding,
    )

    assert result.scale.applied is True
    assert authority_checks == [NOW, NOW]
    replacements = [
        operation["path"] for operation in patches[0] if operation["op"] == "replace"
    ]
    assert replacements.index("/spec/template") < replacements.index("/spec/replicas")
    template_operations = {
        operation["op"]
        for operation in patches[0]
        if operation["path"] == "/spec/template"
    }
    assert template_operations == {
        "test",
        "replace",
    }
    assert (
        state["spec"]["template"]["metadata"]["annotations"]
        [RUNNER_CACHE_STARTUP_BINDING_ID_ANNOTATION]
        == binding.binding_id
    )
    client.close()


def test_mismatched_binding_is_rejected_before_kubernetes_write(tmp_path: Path) -> None:
    log = InMemoryScalingDecisionLog()
    decision = _decision(log)
    binding = _binding(decision, fingerprint="f" * 64)
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200, json=_workload())

    actuator, client = _actuator(tmp_path, handler, log)
    with pytest.raises(KubernetesScaleConflictError, match="durable scaling decision"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(),
            startup_binding=binding,
            reauthorize_startup_binding=lambda: binding,
        )
    assert methods == ["GET"]
    client.close()


def test_shared_template_binding_rejects_incremental_scale_up(tmp_path: Path) -> None:
    log = InMemoryScalingDecisionLog()
    decision = _decision(log)
    binding = _binding(decision)
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200, json=_workload(replicas=1))

    actuator, client = _actuator(tmp_path, handler, log)
    with pytest.raises(KubernetesScaleConflictError, match="scale-from-zero"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(),
            startup_binding=binding,
            reauthorize_startup_binding=lambda: binding,
        )
    assert methods == ["GET"]
    client.close()


def test_binding_expiry_at_final_authorization_prevents_patch(tmp_path: Path) -> None:
    log = InMemoryScalingDecisionLog()
    decision = _decision(log)
    binding = _binding(decision)
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200, json=_workload())

    actuator, client = _actuator(tmp_path, handler, log)
    assert decision.prewarm_plan is not None
    refreshed_at = NOW + timedelta(minutes=6)
    refreshed_snapshot = ScalingPrewarmSnapshot.model_validate(
        decision.prewarm_plan.snapshot.model_copy(
            update={
                "snapshot_id": "cache-snapshot-refreshed",
                "cache_revision": 10,
                "observed_at": refreshed_at,
                "placements": tuple(
                    placement.model_copy(
                        update={
                            "cache_hint_observed_at": refreshed_at,
                            "cache_hint_valid_until": refreshed_at + timedelta(minutes=5),
                            "cache_hint_index_revision": 20 + index,
                        }
                    )
                    for index, placement in enumerate(
                        decision.prewarm_plan.snapshot.placements
                    )
                ),
            }
        ).model_dump()
    )
    refreshed_prewarm = plan_cache_aware_scale_up(
        refreshed_snapshot,
        current_replicas=0,
        quota_target_replicas=2,
        resource_flavor="h100-sxm",
    )
    with pytest.raises(KubernetesScaleConflictError, match="not live"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(validated_at=NOW + timedelta(minutes=6)),
            reauthorize_quota=lambda: decision.quota_admission,
            reauthorize_prewarm=lambda: refreshed_prewarm,
            startup_binding=binding,
            reauthorize_startup_binding=lambda: binding,
        )
    assert methods == ["GET"]
    client.close()


def test_exact_retry_requires_the_bound_template_to_remain_intact(tmp_path: Path) -> None:
    log = InMemoryScalingDecisionLog()
    decision = _decision(log)
    binding = _binding(decision)
    bound = bind_runner_cache_to_pod_template(
        _template(),
        binding,
        release_id="release-a",
    )
    state = _workload(
        decision=decision,
        template=bound,
        replicas=2,
        generation=8,
        resource_version="11",
    )
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200, json=state)

    actuator, client = _actuator(tmp_path, handler, log)
    result = actuator.apply_fenced(
        decision,
        _target(),
        authority=_authority(),
        fence=_fence(),
        reauthorize=lambda: _authority(),
        startup_binding=binding,
        reauthorize_startup_binding=lambda: binding,
    )
    assert result.scale.applied is False
    assert methods == ["GET"]

    state["spec"]["template"]["metadata"]["annotations"].pop(
        RUNNER_CACHE_STARTUP_BINDING_ID_ANNOTATION
    )
    with pytest.raises((KubernetesScaleConflictError, RuntimeError)):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(),
            startup_binding=binding,
            reauthorize_startup_binding=lambda: binding,
        )
    client.close()


def test_patch_response_cannot_drop_the_bound_template(tmp_path: Path) -> None:
    log = InMemoryScalingDecisionLog()
    decision = _decision(log)
    binding = _binding(decision)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_workload())
        return httpx.Response(
            200,
            json=_workload(
                decision=decision,
                replicas=2,
                generation=8,
                resource_version="11",
            ),
        )

    actuator, client = _actuator(tmp_path, handler, log)
    with pytest.raises(InvalidKubernetesScaleResponseError, match="mutation contract"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(),
            reauthorize_quota=lambda: decision.quota_admission,
            reauthorize_prewarm=lambda: decision.prewarm_plan,
            startup_binding=binding,
            reauthorize_startup_binding=lambda: binding,
        )
    client.close()
