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
from kairyu.runners.scale_actuator import (
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

NOW = datetime(2026, 9, 17, 8, 0, tzinfo=UTC)


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


def _authority(*, token: int = 1) -> RunnerWriterAuthority:
    return RunnerWriterAuthority(
        election_id="runner-control-plane",
        holder_id=f"controller-{token}",
        fencing_token=token,
        validated_at=NOW,
        lease_until=NOW + timedelta(minutes=1),
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
    )
    retry = actuator.apply_fenced(
        decision,
        _target(),
        authority=_authority(),
        fence=_fence(),
        reauthorize=lambda: _authority(),
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
        InMemoryRunnerLeaderLeaseStore(),
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
        )
    )
    assert claim.applied is True
    assert result.scale.applied is True
    assert methods == ["GET", "PATCH", "GET", "PATCH"]
    client.close()
