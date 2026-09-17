"""Leader claim and durable-decision fencing for Kubernetes scale writes."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from kairyu.runners import (
    InMemoryRunnerLeaderLeaseStore,
    InMemoryScalingDecisionLog,
    LeaderFencedRunnerController,
    RunnerLeaderElector,
    RunnerNotLeaderError,
    RunnerStatusReconciler,
    RunnerWriterAuthority,
)
from kairyu.runners.kubernetes import MODEL_REVISION_ANNOTATION, RELEASE_ID_ANNOTATION
from kairyu.runners.prewarm import (
    ModelCachePlacement,
    ModelCachePlacementState,
    ScalingPrewarmSnapshot,
    plan_cache_aware_scale_up,
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
    admit_scaling_quota,
    kueue_scaling_workload_name,
)

NOW = datetime(2026, 9, 17, 8, 0, tzinfo=UTC)


def _quota_admission(*, current: int, requested: int):
    reserved_gpus = requested * 2
    snapshot = ScalingQuotaSnapshot(
        snapshot_id=f"quota-{current}-{requested}",
        quota_revision=7,
        observed_at=NOW,
        tenant_id="tenant-a",
        model_class="qwen-14b",
        model_family="qwen",
        gpus_per_replica=2,
        target_reserved_gpus=reserved_gpus,
        limits=tuple(
            ScalingQuotaLimit(
                scope=scope,
                quota_name=f"{scope.value}-quota",
                hard_limit_gpus=100,
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
                target_name="qwen-14b-runners",
                target_uid="workload-uid",
            ),
            workload_uid="kueue-workload-uid",
            workload_generation=3,
            resource_version="41",
            target_kind="Deployment",
            target_namespace="model-serving",
            target_name="qwen-14b-runners",
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
            admitted_pods=requested,
            admitted_gpus=reserved_gpus,
        ),
    )
    return admit_scaling_quota(
        snapshot,
        current_replicas=current,
        requested_replicas=requested,
    )


def _prewarm_plan(
    *,
    current: int,
    requested: int,
    observed_at: datetime = NOW,
    cache_revision: int = 7,
    states: tuple[ModelCachePlacementState, ...] | None = None,
):
    if states is None:
        states = (ModelCachePlacementState.READY,) * (requested - current)
    snapshot = ScalingPrewarmSnapshot(
        snapshot_id=f"cache-{current}-{requested}-{cache_revision}",
        cache_revision=cache_revision,
        observed_at=observed_at,
        model_class="qwen-14b",
        model_revision="model-revision-a",
        artifact_digest="sha256:model-artifact-a",
        placement_binding_id="binding-qwen-h100-a",
        placements=tuple(
            ModelCachePlacement(
                placement_id=f"placement-{index:03d}",
                node_name=f"gpu-node-{index:03d}",
                resource_flavor="h100-sxm",
                profile_id="h100-sxm-tp1",
                compatibility_approval_id="compat-h100-qwen-v1",
                state=state,
            )
            for index, state in enumerate(states)
        ),
    )
    return plan_cache_aware_scale_up(
        snapshot,
        current_replicas=current,
        quota_target_replicas=requested,
        resource_flavor="h100-sxm",
    )


def _refresh_prewarm(
    decision: ScalingDecisionRecord,
    *,
    state: ModelCachePlacementState = ModelCachePlacementState.READY,
    observed_at: datetime = NOW + timedelta(seconds=1),
    cache_revision: int = 8,
    artifact_digest: str = "sha256:model-artifact-a",
    node_name: str | None = None,
):
    original = decision.prewarm_plan
    assert original is not None
    snapshot = ScalingPrewarmSnapshot.model_validate(
        original.snapshot.model_copy(
            update={
                "snapshot_id": f"cache-refresh-{cache_revision}",
                "cache_revision": cache_revision,
                "observed_at": observed_at,
                "artifact_digest": artifact_digest,
                "placements": tuple(
                    placement.model_copy(
                        update={
                            "state": state,
                            **({"node_name": node_name} if node_name is not None else {}),
                        }
                    )
                    for placement in original.snapshot.placements
                ),
            }
        ).model_dump()
    )
    return plan_cache_aware_scale_up(
        snapshot,
        current_replicas=original.current_replicas,
        quota_target_replicas=original.quota_target_replicas,
        resource_flavor=original.resource_flavor,
    )


def _decision(
    *,
    action: ScalingDecisionAction = ScalingDecisionAction.SCALE_UP,
    desired: int = 4,
    current: int = 2,
    decision_id: str = "decision-a",
    target_revision: ScalingDecisionTargetRevision | None = None,
) -> ScalingDecisionRecord:
    observation = ScalingObservation(
        observation_id=f"observation-{decision_id}",
        model_class="qwen-14b",
        observed_at=NOW,
        queue=ScalingQueueSnapshot(
            observed_at=NOW,
            queue_depth=6,
            interactive_queue_depth=4,
            batch_queue_depth=2,
            oldest_queue_age_seconds=1.5,
            arrival_rate_per_second=3.0,
        ),
        runners=ScalingRunnerSnapshot(
            observed_at=NOW,
            current_replicas=current,
            busy_replicas=min(2, current),
            ready_replicas=0,
            loading_replicas=0,
            unhealthy_replicas=0,
        ),
    )
    return ScalingDecisionRecord(
        decision_id=decision_id,
        decided_at=NOW,
        catalog_revision=1,
        policy=ScalingPolicy(
            model_class="qwen-14b",
            policy_revision=1,
            min_replicas=1,
            max_replicas=10,
            warm_buffer_replicas=1,
            max_scale_up_step=3,
            max_scale_down_step=2,
        ),
        window=ScalingObservationWindow(
            window_id=f"window-{decision_id}",
            model_class="qwen-14b",
            started_at=NOW,
            ended_at=NOW,
            observations=(observation,),
        ),
        target_revision=target_revision,
        quota_admission=(
            _quota_admission(current=current, requested=desired)
            if action is ScalingDecisionAction.SCALE_UP
            else None
        ),
        prewarm_plan=(
            _prewarm_plan(current=current, requested=desired)
            if action is ScalingDecisionAction.SCALE_UP
            else None
        ),
        action=action,
        reason=(
            ScalingDecisionReason.NO_CHANGE
            if action is ScalingDecisionAction.HOLD
            else ScalingDecisionReason.QUEUE_PRESSURE
        ),
        demand_replicas=3,
        buffered_target_replicas=4,
        desired_replicas=desired,
        target_delta=desired - current,
    )


def _target_revision(
    *, target_kind: str = "Deployment", token: int = 1, generation: int = 7
) -> ScalingDecisionTargetRevision:
    return ScalingDecisionTargetRevision(
        target_kind=target_kind,
        namespace="model-serving",
        name="qwen-14b-runners",
        election_id="runner-control-plane",
        fencing_token=token,
        workload_uid="workload-uid",
        workload_generation=generation,
        release_id="release-a",
        model_revision="model-revision-a",
    )


def _persist(
    log: InMemoryScalingDecisionLog,
    decision: ScalingDecisionRecord,
) -> ScalingDecisionRecord:
    if decision.target_revision is None:
        decision = ScalingDecisionRecord.model_validate(
            decision.model_copy(
                update={"target_revision": _target_revision()}
            ).model_dump()
        )
    return log.append(decision)


def _target(
    kind: KubernetesScalableKind = KubernetesScalableKind.DEPLOYMENT,
) -> KubernetesScaleTarget:
    return KubernetesScaleTarget(
        model_class="qwen-14b",
        namespace="model-serving",
        name="qwen-14b-runners",
        kind=kind,
    )


def _authority(
    *,
    token: int = 1,
    validated_at: datetime = NOW,
) -> RunnerWriterAuthority:
    return RunnerWriterAuthority(
        election_id="runner-control-plane",
        holder_id=f"controller-{token}",
        fencing_token=token,
        validated_at=validated_at,
        lease_until=validated_at + timedelta(minutes=1),
    )


def _fence(*, workload_generation: int = 7) -> KubernetesScaleFence:
    return KubernetesScaleFence(
        workload_uid="workload-uid",
        workload_generation=workload_generation,
        release_id="release-a",
        model_revision="model-revision-a",
    )


def _annotations(
    *,
    token: int | None = None,
    decision: ScalingDecisionRecord | None = None,
) -> dict[str, str]:
    annotations = {
        RELEASE_ID_ANNOTATION: "release-a",
        MODEL_REVISION_ANNOTATION: "model-revision-a",
        CACHE_PLACEMENT_BINDING_ANNOTATION: "binding-qwen-h100-a",
    }
    if token is not None:
        annotations.update(
            {
                SCALE_ELECTION_ID_ANNOTATION: "runner-control-plane",
                SCALE_FENCING_TOKEN_ANNOTATION: str(token),
            }
        )
    if decision is not None:
        assert decision.decision_generation is not None
        annotations.update(
            {
                SCALE_DECISION_GENERATION_ANNOTATION: str(
                    decision.decision_generation
                ),
                SCALE_DECISION_ID_ANNOTATION: decision.decision_id,
                SCALE_DECISION_FINGERPRINT_ANNOTATION: decision.fingerprint,
            }
        )
    return annotations


def _workload_payload(
    *,
    kind: str = "Deployment",
    replicas: int = 2,
    resource_version: str = "10",
    generation: int = 7,
    uid: str = "workload-uid",
    annotations: dict[str, str] | None = None,
) -> dict:
    return {
        "apiVersion": "apps/v1",
        "kind": kind,
        "metadata": {
            "name": "qwen-14b-runners",
            "namespace": "model-serving",
            "uid": uid,
            "resourceVersion": resource_version,
            "generation": generation,
            "annotations": _annotations() if annotations is None else annotations,
        },
        "spec": {"replicas": replicas},
    }


def _actuator(
    tmp_path: Path,
    handler,
    *,
    decision_log: InMemoryScalingDecisionLog | None = None,
) -> tuple[KubernetesScaleActuator, httpx.Client, InMemoryScalingDecisionLog]:
    token_path = tmp_path / "token"
    token_path.write_text("service-account-token", encoding="utf-8")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    decision_log = decision_log or InMemoryScalingDecisionLog()
    return (
        KubernetesScaleActuator(
            api_server="https://kubernetes.example",
            token_path=token_path,
            client=client,
            decision_log=decision_log,
        ),
        client,
        decision_log,
    )


def test_claim_then_fenced_scale_persists_full_decision_identity_and_retries(
    tmp_path: Path,
) -> None:
    decision: ScalingDecisionRecord | None = None
    state = _workload_payload()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal state
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=state)
        patch = json.loads(request.content)
        if SCALE_ELECTION_ID_ANNOTATION not in state["metadata"]["annotations"]:
            assert patch[:2] == [
                {"op": "test", "path": "/metadata/resourceVersion", "value": "10"},
                {"op": "test", "path": "/metadata/uid", "value": "workload-uid"},
            ]
            state = _workload_payload(
                resource_version="11", annotations=_annotations(token=1)
            )
        else:
            assert decision is not None
            values = {operation["path"]: operation.get("value") for operation in patch}
            assert values["/metadata/annotations/kairyu.ai~1scale-fencing-token"] == "1"
            assert (
                values[
                    "/metadata/annotations/kairyu.ai~1cache-placement-binding"
                ]
                == "binding-qwen-h100-a"
            )
            assert values["/metadata/annotations/kairyu.ai~1scale-decision-generation"] == "1"
            assert values["/metadata/annotations/kairyu.ai~1scale-decision-id"] == "decision-a"
            assert (
                values["/metadata/annotations/kairyu.ai~1scale-decision-fingerprint"]
                == decision.fingerprint
            )
            state = _workload_payload(
                replicas=4,
                resource_version="12",
                generation=8,
                annotations=_annotations(token=1, decision=decision),
            )
        return httpx.Response(200, json=state)

    actuator, client, log = _actuator(tmp_path, handler)
    claim = actuator.claim_authority(
        _target(), _authority(), reauthorize=lambda: _authority()
    )
    decision = log.append(
        _decision(target_revision=claim.target_revision)
    )
    result = actuator.apply_fenced(
        decision,
        _target(),
        authority=_authority(),
        fence=_fence(),
        reauthorize=lambda: _authority(),
        reauthorize_quota=lambda: decision.quota_admission,
        reauthorize_prewarm=lambda: decision.prewarm_plan,
    )
    retry = actuator.apply_fenced(
        decision,
        _target(),
        authority=_authority(),
        fence=_fence(),
        reauthorize=lambda: _authority(),
        reauthorize_quota=lambda: decision.quota_admission,
        reauthorize_prewarm=lambda: decision.prewarm_plan,
    )

    assert claim.applied is True
    assert result.scale.applied is True
    assert result.decision == decision
    assert retry.scale.applied is False
    assert [request.method for request in requests] == [
        "GET",
        "PATCH",
        "GET",
        "PATCH",
        "GET",
    ]
    client.close()


def test_successor_claim_blocks_stale_leader_before_decision(tmp_path: Path) -> None:
    state = _workload_payload(annotations=_annotations(token=1))

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal state
        if request.method == "GET":
            return httpx.Response(200, json=state)
        state = _workload_payload(
            resource_version="11", annotations=_annotations(token=2)
        )
        return httpx.Response(200, json=state)

    actuator, client, log = _actuator(tmp_path, handler)
    claim = actuator.claim_authority(
        _target(),
        _authority(token=2),
        reauthorize=lambda: _authority(token=2),
    )
    assert claim.applied is True
    with pytest.raises(KubernetesScaleConflictError, match="superseded"):
        actuator.apply_fenced(
            _persist(log, _decision()),
            _target(),
            authority=_authority(token=1),
            fence=_fence(),
            reauthorize=lambda: _authority(token=1),
        )
    client.close()


def test_successor_cannot_relabel_predecessor_decision_with_its_token(
    tmp_path: Path,
) -> None:
    actuator, client, log = _actuator(
        tmp_path,
        lambda _request: pytest.fail("mismatched durable authority must fail locally"),
    )
    predecessor = _persist(log, _decision())

    with pytest.raises(ValueError, match="target revision"):
        actuator.apply_fenced(
            predecessor,
            _target(),
            authority=_authority(token=2),
            fence=_fence(),
            reauthorize=lambda: _authority(token=2),
        )
    client.close()


def test_successor_claim_between_old_read_and_patch_invalidates_old_cas(
    tmp_path: Path,
) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "GET":
            return httpx.Response(
                200, json=_workload_payload(annotations=_annotations(token=1))
            )
        return httpx.Response(422, json={"kind": "Status", "reason": "Invalid"})

    actuator, client, log = _actuator(tmp_path, handler)
    decision = _persist(log, _decision())
    with pytest.raises(KubernetesScaleConflictError, match="changed during"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(token=1),
            fence=_fence(),
            reauthorize=lambda: _authority(token=1),
            reauthorize_quota=lambda: decision.quota_admission,
            reauthorize_prewarm=lambda: decision.prewarm_plan,
        )
    assert methods == ["GET", "PATCH"]
    client.close()


def test_expired_authority_between_read_and_patch_prevents_write(tmp_path: Path) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(
            200, json=_workload_payload(annotations=_annotations(token=1))
        )

    actuator, client, log = _actuator(tmp_path, handler)
    decision = _persist(log, _decision())

    def expired() -> RunnerWriterAuthority:
        raise RunnerNotLeaderError("lease expired before Kubernetes write")

    with pytest.raises(RunnerNotLeaderError, match="expired"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=expired,
        )
    assert methods == ["GET"]
    client.close()


def test_revoked_quota_between_decision_and_patch_prevents_scale_up(
    tmp_path: Path,
) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(
            200,
            json=_workload_payload(annotations=_annotations(token=1)),
        )

    actuator, client, log = _actuator(tmp_path, handler)
    decision = _persist(log, _decision())
    assert decision.quota_admission is not None
    original = decision.quota_admission
    revoked_kueue = original.snapshot.kueue.model_copy(
        update={
            "resource_version": "42",
            "cluster_queue": None,
            "resource_flavor": None,
            "admitted": False,
            "admitted_pods": 0,
            "admitted_gpus": 0,
        }
    )
    refreshed_snapshot = ScalingQuotaSnapshot.model_validate(
        original.snapshot.model_copy(
            update={
                "snapshot_id": "quota-revoked",
                "quota_revision": 2,
                "observed_at": NOW + timedelta(seconds=1),
                "target_reserved_gpus": 0,
                "kueue": revoked_kueue,
            }
        ).model_dump()
    )
    revoked = admit_scaling_quota(
        refreshed_snapshot,
        current_replicas=original.current_replicas,
        requested_replicas=original.requested_replicas,
    )

    with pytest.raises(KubernetesScaleConflictError, match="quota authority"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(),
            reauthorize_quota=lambda: revoked,
            reauthorize_prewarm=lambda: decision.prewarm_plan,
        )
    assert methods == ["GET"]
    client.close()


def test_scale_up_requires_final_quota_reauthorization(tmp_path: Path) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(
            200,
            json=_workload_payload(annotations=_annotations(token=1)),
        )

    actuator, client, log = _actuator(tmp_path, handler)
    decision = _persist(log, _decision())

    with pytest.raises(TypeError, match="reauthorize_quota"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(),
        )
    assert methods == ["GET"]
    client.close()


def test_scale_up_requires_final_prewarm_reauthorization(tmp_path: Path) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(
            200,
            json=_workload_payload(annotations=_annotations(token=1)),
        )

    actuator, client, log = _actuator(tmp_path, handler)
    decision = _persist(log, _decision())

    with pytest.raises(TypeError, match="reauthorize_prewarm"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(),
            reauthorize_quota=lambda: decision.quota_admission,
        )
    assert methods == ["GET"]
    client.close()


def test_revoked_ready_cache_between_decision_and_patch_prevents_scale_up(
    tmp_path: Path,
) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(
            200,
            json=_workload_payload(annotations=_annotations(token=1)),
        )

    actuator, client, log = _actuator(tmp_path, handler)
    decision = _persist(log, _decision())
    revoked = _refresh_prewarm(
        decision,
        state=ModelCachePlacementState.FAILED,
    )

    with pytest.raises(KubernetesScaleConflictError, match="ready cache capacity"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(validated_at=NOW + timedelta(seconds=1)),
            reauthorize_quota=lambda: decision.quota_admission,
            reauthorize_prewarm=lambda: revoked,
        )
    assert methods == ["GET"]
    client.close()


def test_prewarm_revision_rollback_prevents_scale_up(tmp_path: Path) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(
            200,
            json=_workload_payload(annotations=_annotations(token=1)),
        )

    actuator, client, log = _actuator(tmp_path, handler)
    decision = _persist(log, _decision())
    rolled_back = _refresh_prewarm(decision, cache_revision=6)

    with pytest.raises(KubernetesScaleConflictError, match="prewarm authority changed"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(validated_at=NOW + timedelta(seconds=1)),
            reauthorize_quota=lambda: decision.quota_admission,
            reauthorize_prewarm=lambda: rolled_back,
        )
    assert methods == ["GET"]
    client.close()


def test_prewarm_artifact_change_prevents_scale_up(tmp_path: Path) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(
            200,
            json=_workload_payload(annotations=_annotations(token=1)),
        )

    actuator, client, log = _actuator(tmp_path, handler)
    decision = _persist(log, _decision())
    changed = _refresh_prewarm(
        decision,
        artifact_digest="sha256:other-model-artifact",
    )

    with pytest.raises(KubernetesScaleConflictError, match="prewarm authority changed"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(validated_at=NOW + timedelta(seconds=1)),
            reauthorize_quota=lambda: decision.quota_admission,
            reauthorize_prewarm=lambda: changed,
        )
    assert methods == ["GET"]
    client.close()


def test_prewarm_placement_rebinding_prevents_scale_up(tmp_path: Path) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(
            200,
            json=_workload_payload(annotations=_annotations(token=1)),
        )

    actuator, client, log = _actuator(tmp_path, handler)
    decision = _persist(log, _decision())
    rebound = _refresh_prewarm(decision, node_name="replacement-gpu-node")

    with pytest.raises(KubernetesScaleConflictError, match="ready cache capacity"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(validated_at=NOW + timedelta(seconds=1)),
            reauthorize_quota=lambda: decision.quota_admission,
            reauthorize_prewarm=lambda: rebound,
        )
    assert methods == ["GET"]
    client.close()


def test_workload_cache_binding_mismatch_prevents_scale_up(tmp_path: Path) -> None:
    methods: list[str] = []
    annotations = _annotations(token=1)
    annotations[CACHE_PLACEMENT_BINDING_ANNOTATION] = "other-binding"

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(
            200,
            json=_workload_payload(annotations=annotations),
        )

    actuator, client, log = _actuator(tmp_path, handler)
    decision = _persist(log, _decision())

    with pytest.raises(KubernetesScaleConflictError, match="placement binding"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(),
            reauthorize_quota=lambda: decision.quota_admission,
            reauthorize_prewarm=lambda: decision.prewarm_plan,
        )
    assert methods == ["GET"]
    client.close()


def test_final_quota_and_mixed_prewarm_target_must_still_agree(
    tmp_path: Path,
) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(
            200,
            json=_workload_payload(
                replicas=3,
                annotations=_annotations(token=1),
            ),
        )

    original_quota = _quota_admission(current=3, requested=6)
    original_prewarm = _prewarm_plan(
        current=3,
        requested=6,
        states=(
            ModelCachePlacementState.READY,
            ModelCachePlacementState.READY,
            ModelCachePlacementState.ABSENT,
        ),
    )
    draft = ScalingDecisionRecord.model_validate(
        _decision(current=3, desired=5, target_revision=_target_revision())
        .model_copy(
            update={
                "quota_admission": original_quota,
                "prewarm_plan": original_prewarm,
            }
        )
        .model_dump()
    )
    original_kueue = original_quota.snapshot.kueue
    reduced_snapshot = ScalingQuotaSnapshot.model_validate(
        original_quota.snapshot.model_copy(
            update={
                "snapshot_id": "quota-reduced-final",
                "quota_revision": 8,
                "observed_at": NOW + timedelta(seconds=1),
                "target_reserved_gpus": 10,
                "kueue": original_kueue.model_copy(
                    update={
                        "resource_version": "42",
                        "admitted_pods": 5,
                        "admitted_gpus": 10,
                    }
                ),
            }
        ).model_dump()
    )
    reduced_quota = admit_scaling_quota(
        reduced_snapshot,
        current_replicas=3,
        requested_replicas=6,
    )
    log = InMemoryScalingDecisionLog()
    decision = log.append(draft)
    actuator, client, _unused = _actuator(
        tmp_path,
        handler,
        decision_log=log,
    )

    with pytest.raises(KubernetesScaleConflictError, match="disagree"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(validated_at=NOW + timedelta(seconds=1)),
            reauthorize_quota=lambda: reduced_quota,
            reauthorize_prewarm=lambda: original_prewarm,
        )
    assert methods == ["GET"]
    client.close()


def test_stale_prewarm_at_final_authorization_prevents_scale_up(
    tmp_path: Path,
) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(
            200,
            json=_workload_payload(annotations=_annotations(token=1)),
        )

    actuator, client, log = _actuator(tmp_path, handler)
    decision = _persist(log, _decision())
    original_quota = decision.quota_admission
    assert original_quota is not None
    fresh_quota_snapshot = ScalingQuotaSnapshot.model_validate(
        original_quota.snapshot.model_copy(
            update={
                "snapshot_id": "quota-final-auth",
                "quota_revision": 8,
                "observed_at": NOW + timedelta(seconds=1),
            }
        ).model_dump()
    )
    fresh_quota = admit_scaling_quota(
        fresh_quota_snapshot,
        current_replicas=original_quota.current_replicas,
        requested_replicas=original_quota.requested_replicas,
    )

    with pytest.raises(KubernetesScaleConflictError, match="prewarm authority is not fresh"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(validated_at=NOW + timedelta(seconds=31)),
            reauthorize_quota=lambda: fresh_quota,
            reauthorize_prewarm=lambda: decision.prewarm_plan,
        )
    assert methods == ["GET"]
    client.close()


def test_quota_reservation_must_be_bound_to_decision_target() -> None:
    mismatched = _target_revision().model_copy(update={"name": "other-runners"})

    with pytest.raises(ValidationError, match="quota reservation target"):
        _decision(target_revision=mismatched)


def test_stale_quota_at_final_authorization_prevents_scale_up(tmp_path: Path) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(
            200,
            json=_workload_payload(annotations=_annotations(token=1)),
        )

    actuator, client, log = _actuator(tmp_path, handler)
    decision = _persist(log, _decision())
    late_authority = _authority(validated_at=NOW + timedelta(seconds=31))

    with pytest.raises(KubernetesScaleConflictError, match="not fresh"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: late_authority,
            reauthorize_quota=lambda: decision.quota_admission,
            reauthorize_prewarm=lambda: decision.prewarm_plan,
        )
    assert methods == ["GET"]
    client.close()


def test_quota_revision_rollback_prevents_scale_up(tmp_path: Path) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(
            200,
            json=_workload_payload(annotations=_annotations(token=1)),
        )

    actuator, client, log = _actuator(tmp_path, handler)
    decision = _persist(log, _decision())
    assert decision.quota_admission is not None
    original = decision.quota_admission
    rolled_back_snapshot = ScalingQuotaSnapshot.model_validate(
        original.snapshot.model_copy(
            update={
                "snapshot_id": "quota-rollback",
                "quota_revision": 6,
                "observed_at": NOW + timedelta(seconds=1),
            }
        ).model_dump()
    )
    rolled_back = admit_scaling_quota(
        rolled_back_snapshot,
        current_replicas=original.current_replicas,
        requested_replicas=original.requested_replicas,
    )

    with pytest.raises(KubernetesScaleConflictError, match="quota authority changed"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(validated_at=NOW + timedelta(seconds=1)),
            reauthorize_quota=lambda: rolled_back,
            reauthorize_prewarm=lambda: decision.prewarm_plan,
        )
    assert methods == ["GET"]
    client.close()


def test_newer_generation_skips_abandoned_decision_and_delayed_one_is_rejected(
    tmp_path: Path,
) -> None:
    log = InMemoryScalingDecisionLog()
    first = _persist(log, _decision(decision_id="decision-1"))
    revision = _target_revision(generation=8)
    abandoned = _persist(
        log,
        _decision(
            decision_id="decision-2",
            current=4,
            desired=5,
            target_revision=revision,
        ),
    )
    current = _persist(
        log,
        _decision(
            decision_id="decision-3",
            current=4,
            desired=6,
            target_revision=revision,
        ),
    )
    state = _workload_payload(
        replicas=4,
        resource_version="12",
        generation=8,
        annotations=_annotations(token=1, decision=first),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal state
        if request.method == "GET":
            return httpx.Response(200, json=state)
        state = _workload_payload(
            replicas=6,
            resource_version="13",
            generation=9,
            annotations=_annotations(token=1, decision=current),
        )
        return httpx.Response(200, json=state)

    actuator, client, _unused = _actuator(
        tmp_path, handler, decision_log=log
    )
    applied = actuator.apply_fenced(
        current,
        _target(),
        authority=_authority(),
        fence=_fence(workload_generation=8),
        reauthorize=lambda: _authority(),
        reauthorize_quota=lambda: current.quota_admission,
        reauthorize_prewarm=lambda: current.prewarm_plan,
    )
    assert applied.scale.applied is True
    assert current.decision_generation == 3
    with pytest.raises(KubernetesScaleConflictError, match="workload generation"):
        actuator.apply_fenced(
            abandoned,
            _target(),
            authority=_authority(),
            fence=_fence(workload_generation=8),
            reauthorize=lambda: _authority(),
            reauthorize_quota=lambda: abandoned.quota_admission,
            reauthorize_prewarm=lambda: abandoned.prewarm_plan,
        )
    client.close()


def test_hold_is_generation_free_but_requires_claimed_authority(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=_workload_payload(
                kind="StatefulSet", annotations=_annotations(token=1)
            ),
        )

    actuator, client, log = _actuator(tmp_path, handler)
    decision = _decision(
        action=ScalingDecisionAction.HOLD,
        desired=2,
        target_revision=_target_revision(target_kind="StatefulSet"),
    )
    decision = log.append(decision)
    result = actuator.apply_fenced(
        decision,
        _target(KubernetesScalableKind.STATEFUL_SET),
        authority=_authority(),
        fence=_fence(),
        reauthorize=lambda: _authority(),
    )
    assert decision.decision_generation is None
    assert result.scale.applied is False
    assert [request.method for request in requests] == ["GET"]
    client.close()


def test_forged_generation_is_rejected_before_kubernetes_io(tmp_path: Path) -> None:
    actuator, client, _log = _actuator(
        tmp_path,
        lambda _request: pytest.fail("Kubernetes must not be called"),
    )
    forged = ScalingDecisionRecord.model_validate(
        _decision(target_revision=_target_revision())
        .model_copy(update={"decision_generation": 1})
        .model_dump()
    )
    with pytest.raises(ValueError, match="durably appended"):
        actuator.apply_fenced(
            forged,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(),
        )
    client.close()


def test_fenced_scale_up_requires_durable_quota_admission_before_io(
    tmp_path: Path,
) -> None:
    log = InMemoryScalingDecisionLog()
    without_quota = ScalingDecisionRecord.model_validate(
        _decision(target_revision=_target_revision())
        .model_copy(update={"quota_admission": None, "prewarm_plan": None})
        .model_dump()
    )
    decision = log.append(without_quota)
    actuator, client, _unused = _actuator(
        tmp_path,
        lambda _request: pytest.fail("Kubernetes must not be called"),
        decision_log=log,
    )

    with pytest.raises(ValueError, match="durable quota admission"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(),
            reauthorize_quota=lambda: decision.quota_admission,
            reauthorize_prewarm=lambda: decision.prewarm_plan,
        )
    client.close()


def test_fenced_scale_up_requires_durable_prewarm_plan_before_io(
    tmp_path: Path,
) -> None:
    log = InMemoryScalingDecisionLog()
    without_prewarm = ScalingDecisionRecord.model_validate(
        _decision(target_revision=_target_revision())
        .model_copy(update={"prewarm_plan": None})
        .model_dump()
    )
    decision = log.append(without_prewarm)
    actuator, client, _unused = _actuator(
        tmp_path,
        lambda _request: pytest.fail("Kubernetes must not be called"),
        decision_log=log,
    )

    with pytest.raises(ValueError, match="durable prewarm plan"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(),
            reauthorize_quota=lambda: decision.quota_admission,
            reauthorize_prewarm=lambda: decision.prewarm_plan,
        )
    client.close()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            _workload_payload(uid="replacement", annotations=_annotations(token=1)),
            "UID",
        ),
        (
            _workload_payload(generation=8, annotations=_annotations(token=1)),
            "workload generation",
        ),
        (
            _workload_payload(
                annotations={
                    **_annotations(token=1),
                    RELEASE_ID_ANNOTATION: "release-b",
                }
            ),
            "release",
        ),
    ],
)
def test_stale_workload_identity_is_rejected(
    tmp_path: Path, payload: dict, message: str
) -> None:
    actuator, client, log = _actuator(
        tmp_path, lambda _request: httpx.Response(200, json=payload)
    )
    with pytest.raises(KubernetesScaleConflictError, match=message):
        actuator.apply_fenced(
            _persist(log, _decision()),
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(),
        )
    client.close()


def test_incomplete_authority_annotations_fail_closed(tmp_path: Path) -> None:
    annotations = _annotations()
    annotations[SCALE_FENCING_TOKEN_ANNOTATION] = "1"
    actuator, client, log = _actuator(
        tmp_path,
        lambda _request: httpx.Response(
            200, json=_workload_payload(annotations=annotations)
        ),
    )
    with pytest.raises(InvalidKubernetesScaleResponseError, match="incomplete"):
        actuator.apply_fenced(
            _persist(log, _decision()),
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(),
        )
    client.close()


def test_exact_retry_rejects_same_id_and_generation_with_other_fingerprint(
    tmp_path: Path,
) -> None:
    log = InMemoryScalingDecisionLog()
    original = _persist(log, _decision())
    changed = ScalingDecisionRecord.model_validate(
        original.model_copy(update={"reason_detail": "different"}).model_dump()
    )
    payload = _workload_payload(
        replicas=4,
        resource_version="12",
        generation=8,
        annotations=_annotations(token=1, decision=original),
    )
    actuator, client, _unused_log = _actuator(
        tmp_path,
        lambda _request: httpx.Response(200, json=payload),
        decision_log=log,
    )
    with pytest.raises(KubernetesScaleConflictError, match="durable decision log"):
        actuator.apply_fenced(
            changed,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(),
        )
    client.close()


def test_fenced_response_requires_exact_authority_decision_and_generation(
    tmp_path: Path,
) -> None:
    log = InMemoryScalingDecisionLog()
    decision = _persist(log, _decision())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200, json=_workload_payload(annotations=_annotations(token=1))
            )
        return httpx.Response(
            200,
            json=_workload_payload(
                replicas=4,
                resource_version="11",
                generation=7,
                annotations=_annotations(token=1, decision=decision),
            ),
        )

    actuator, client, _unused_log = _actuator(
        tmp_path, handler, decision_log=log
    )
    with pytest.raises(InvalidKubernetesScaleResponseError, match="mutation contract"):
        actuator.apply_fenced(
            decision,
            _target(),
            authority=_authority(),
            fence=_fence(),
            reauthorize=lambda: _authority(),
            reauthorize_quota=lambda: decision.quota_admission,
            reauthorize_prewarm=lambda: decision.prewarm_plan,
        )
    client.close()


def test_fenced_result_revalidates_cross_field_identity(tmp_path: Path) -> None:
    log = InMemoryScalingDecisionLog()
    decision = _persist(log, _decision())
    state = _workload_payload(annotations=_annotations(token=1))

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal state
        if request.method == "GET":
            return httpx.Response(200, json=state)
        state = _workload_payload(
            replicas=4,
            resource_version="11",
            generation=8,
            annotations=_annotations(token=1, decision=decision),
        )
        return httpx.Response(200, json=state)

    actuator, client, _unused_log = _actuator(
        tmp_path, handler, decision_log=log
    )
    result = actuator.apply_fenced(
        decision,
        _target(),
        authority=_authority(),
        fence=_fence(),
        reauthorize=lambda: _authority(),
        reauthorize_quota=lambda: decision.quota_admission,
        reauthorize_prewarm=lambda: decision.prewarm_plan,
    )
    bypass = result.model_copy(
        update={"decision": decision.model_copy(update={"decision_id": "other"})}
    )
    with pytest.raises(ValidationError, match="decision_id"):
        type(result).model_validate(bypass.model_dump())
    client.close()


def test_leader_gate_claims_before_fenced_scale(tmp_path: Path) -> None:
    decision: ScalingDecisionRecord | None = None
    state = _workload_payload()
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal state
        methods.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json=state)
        if SCALE_ELECTION_ID_ANNOTATION not in state["metadata"]["annotations"]:
            state = _workload_payload(
                resource_version="11", annotations=_annotations(token=1)
            )
        else:
            assert decision is not None
            state = _workload_payload(
                replicas=4,
                resource_version="12",
                generation=8,
                annotations=_annotations(token=1, decision=decision),
            )
        return httpx.Response(200, json=state)

    actuator, client, log = _actuator(tmp_path, handler)
    elector = RunnerLeaderElector(
        InMemoryRunnerLeaderLeaseStore(clock=lambda: NOW),
        election_id="runner-control-plane",
        holder_id="controller-a",
        lease_seconds=10,
    )
    gate = LeaderFencedRunnerController(elector, RunnerStatusReconciler())
    with pytest.raises(RunnerNotLeaderError):
        gate.mutate_autoscaler(
            lambda authority: actuator.claim_authority(
                _target(), authority, reauthorize=elector.authority
            )
        )
    assert methods == []

    assert elector.campaign() is not None
    claim = gate.mutate_autoscaler(
        lambda authority: actuator.claim_authority(
            _target(), authority, reauthorize=elector.authority
        )
    )
    decision = log.append(_decision(target_revision=claim.target_revision))
    result = gate.mutate_autoscaler(
        lambda authority: actuator.apply_fenced(
            decision,
            _target(),
            authority=authority,
            fence=_fence(),
            reauthorize=elector.authority,
            reauthorize_quota=lambda: decision.quota_admission,
            reauthorize_prewarm=lambda: decision.prewarm_plan,
        )
    )
    assert claim.applied is True
    assert result.scale.applied is True
    assert methods == ["GET", "PATCH", "GET", "PATCH"]
    client.close()
