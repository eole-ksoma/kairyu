"""Executable contract for the Kubernetes Runner scale actuator."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from kairyu.runners.leadership import (
    InMemoryRunnerLeaderLeaseStore,
    LeaderFencedRunnerController,
    RunnerLeaderElector,
    RunnerNotLeaderError,
)
from kairyu.runners.reconciler import RunnerStatusReconciler
from kairyu.runners.scale_actuator import (
    InvalidKubernetesScaleResponseError,
    KubernetesScalableKind,
    KubernetesScaleActuator,
    KubernetesScaleConflictError,
    KubernetesScaleTarget,
)
from kairyu.runners.scaling import ScalingPolicy
from kairyu.runners.scaling_log import (
    ScalingDecisionAction,
    ScalingDecisionReason,
    ScalingDecisionRecord,
    ScalingObservation,
    ScalingObservationWindow,
    ScalingQueueSnapshot,
    ScalingRunnerSnapshot,
)

NOW = datetime(2026, 9, 17, 3, 0, tzinfo=UTC)


def _decision(
    *,
    action: ScalingDecisionAction = ScalingDecisionAction.SCALE_UP,
    desired: int = 4,
    current: int = 2,
    decision_id: str = "decision-a",
) -> ScalingDecisionRecord:
    observation = ScalingObservation(
        observation_id="observation-a",
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
            busy_replicas=2,
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
            window_id="window-a",
            model_class="qwen-14b",
            started_at=NOW,
            ended_at=NOW,
            observations=(observation,),
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


def _target(
    kind: KubernetesScalableKind = KubernetesScalableKind.DEPLOYMENT,
) -> KubernetesScaleTarget:
    return KubernetesScaleTarget(
        model_class="qwen-14b",
        namespace="model-serving",
        name="qwen-14b-runners",
        kind=kind,
    )


def _scale_payload(*, replicas: int, resource_version: str = "10") -> dict:
    return {
        "apiVersion": "autoscaling/v1",
        "kind": "Scale",
        "metadata": {
            "name": "qwen-14b-runners",
            "namespace": "model-serving",
            "resourceVersion": resource_version,
        },
        "spec": {"replicas": replicas},
        "status": {"replicas": replicas},
    }


def _actuator(
    tmp_path: Path,
    handler,
) -> tuple[KubernetesScaleActuator, httpx.Client]:
    token_path = tmp_path / "token"
    token_path.write_text("token-one\n", encoding="utf-8")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    actuator = KubernetesScaleActuator(
        api_server="https://kubernetes.example",
        token_path=token_path,
        client=client,
        allow_unfenced=True,
    )
    return actuator, client


def test_unfenced_scale_is_disabled_by_default(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("token-one\n", encoding="utf-8")
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _request: pytest.fail("must not perform I/O"))
    )
    actuator = KubernetesScaleActuator(
        api_server="https://kubernetes.example",
        token_path=token_path,
        client=client,
    )

    with pytest.raises(RuntimeError, match="unfenced scale writes are disabled"):
        actuator.apply(_decision(), _target())
    client.close()


def test_deployment_scale_uses_resource_version_cas_and_is_idempotent(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []
    replicas = 2
    version = 10

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal replicas, version
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json=_scale_payload(replicas=replicas, resource_version=str(version)),
            )
        body = json.loads(request.content)
        assert body["metadata"]["resourceVersion"] == str(version)
        assert body["spec"]["replicas"] == 4
        replicas = 4
        version += 1
        return httpx.Response(
            200,
            json=_scale_payload(replicas=replicas, resource_version=str(version)),
        )

    actuator, client = _actuator(tmp_path, handler)
    first = actuator.apply(_decision(), _target())
    second = actuator.apply(_decision(), _target())

    expected_path = "/apis/apps/v1/namespaces/model-serving/deployments/qwen-14b-runners/scale"
    assert [request.url.path for request in requests] == [
        expected_path,
        expected_path,
        expected_path,
    ]
    assert [request.method for request in requests] == ["GET", "PUT", "GET"]
    assert first.applied is True
    assert first.previous_replicas == 2
    assert first.requested_replicas == 4
    assert first.resulting_replicas == 4
    assert first.resource_version_before == "10"
    assert first.resource_version_after == "11"
    assert second.applied is False
    assert second.previous_replicas == 4
    assert second.requested_replicas == 4
    assert second.resulting_replicas == 4
    assert all(request.headers["authorization"] == "Bearer token-one" for request in requests)
    client.close()


def test_statefulset_target_uses_its_scale_subresource(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_scale_payload(replicas=4))

    actuator, client = _actuator(tmp_path, handler)
    result = actuator.apply(_decision(), _target(KubernetesScalableKind.STATEFUL_SET))

    assert result.applied is False
    assert seen[0].url.path.endswith("/statefulsets/qwen-14b-runners/scale")
    client.close()


def test_hold_observes_but_never_restores_a_changed_live_replica_count(
    tmp_path: Path,
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_scale_payload(replicas=7))

    actuator, client = _actuator(tmp_path, handler)
    result = actuator.apply(
        _decision(action=ScalingDecisionAction.HOLD, desired=2),
        _target(),
    )

    assert result.applied is False
    assert result.previous_replicas == 7
    assert result.requested_replicas == 2
    assert result.resulting_replicas == 7
    assert [request.method for request in seen] == ["GET"]
    client.close()


def test_token_rotates_between_decisions(tmp_path: Path) -> None:
    authorizations: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        authorizations.append(request.headers["authorization"])
        return httpx.Response(200, json=_scale_payload(replicas=4))

    actuator, client = _actuator(tmp_path, handler)
    actuator.apply(_decision(), _target())
    (tmp_path / "token").write_text("token-two", encoding="utf-8")
    actuator.apply(_decision(decision_id="decision-b"), _target())

    assert authorizations == ["Bearer token-one", "Bearer token-two"]
    client.close()


def test_scale_write_is_entered_through_fresh_leader_authority(tmp_path: Path) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200, json=_scale_payload(replicas=4))

    actuator, client = _actuator(tmp_path, handler)
    store = InMemoryRunnerLeaderLeaseStore()
    elector = RunnerLeaderElector(
        store,
        election_id="runner-control-plane",
        holder_id="controller-a",
        lease_seconds=10,
    )
    gate = LeaderFencedRunnerController(elector, RunnerStatusReconciler())

    with pytest.raises(RunnerNotLeaderError):
        gate.mutate_autoscaler(lambda _authority: actuator.apply(_decision(), _target()))
    assert methods == []

    assert elector.campaign() is not None
    result = gate.mutate_autoscaler(lambda _authority: actuator.apply(_decision(), _target()))
    assert result.applied is False
    assert methods == ["GET"]
    client.close()


def test_conflict_fails_closed_without_unbounded_retry(tmp_path: Path) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json=_scale_payload(replicas=2))
        return httpx.Response(409, json={"kind": "Status", "reason": "Conflict"})

    actuator, client = _actuator(tmp_path, handler)
    with pytest.raises(KubernetesScaleConflictError, match="concurrently"):
        actuator.apply(_decision(), _target())
    assert methods == ["GET", "PUT"]
    client.close()


@pytest.mark.parametrize(
    ("decision", "live_replicas"),
    [
        (_decision(action=ScalingDecisionAction.SCALE_UP, current=2, desired=4), 7),
        (_decision(action=ScalingDecisionAction.SCALE_DOWN, current=4, desired=2), 1),
    ],
)
def test_live_replicas_diverging_from_decision_fail_as_conflict(
    tmp_path: Path,
    decision: ScalingDecisionRecord,
    live_replicas: int,
) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200, json=_scale_payload(replicas=live_replicas))

    actuator, client = _actuator(tmp_path, handler)
    with pytest.raises(KubernetesScaleConflictError, match="decision observation"):
        actuator.apply(decision, _target())
    assert methods == ["GET"]
    client.close()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"kind": "Deployment"}),
        lambda payload: payload["metadata"].update({"name": "other"}),
        lambda payload: payload["metadata"].pop("resourceVersion"),
        lambda payload: payload["spec"].update({"replicas": True}),
    ],
)
def test_malformed_scale_responses_fail_closed(tmp_path: Path, mutate) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        payload = _scale_payload(replicas=2)
        mutate(payload)
        return httpx.Response(200, json=payload)

    actuator, client = _actuator(tmp_path, handler)
    with pytest.raises(InvalidKubernetesScaleResponseError):
        actuator.apply(_decision(), _target())
    client.close()


def test_non_json_scale_response_fails_closed(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json")

    actuator, client = _actuator(tmp_path, handler)
    with pytest.raises(InvalidKubernetesScaleResponseError, match="valid JSON"):
        actuator.apply(_decision(), _target())
    client.close()


def test_write_response_must_confirm_requested_replicas(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_scale_payload(replicas=2))
        return httpx.Response(
            200,
            json=_scale_payload(replicas=3, resource_version="11"),
        )

    actuator, client = _actuator(tmp_path, handler)
    with pytest.raises(
        InvalidKubernetesScaleResponseError,
        match="requested replica count",
    ):
        actuator.apply(_decision(desired=4), _target())
    client.close()


def test_write_response_must_advance_resource_version(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        replicas = 2 if request.method == "GET" else 4
        return httpx.Response(
            200,
            json=_scale_payload(replicas=replicas, resource_version="10"),
        )

    actuator, client = _actuator(tmp_path, handler)
    with pytest.raises(
        InvalidKubernetesScaleResponseError,
        match="advance resourceVersion",
    ):
        actuator.apply(_decision(desired=4), _target())
    client.close()


def test_unchecked_copies_and_closed_state_are_enforced(tmp_path: Path) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_scale_payload(replicas=4))

    actuator, client = _actuator(tmp_path, handler)
    mismatched = _target().model_copy(update={"model_class": "other"})
    with pytest.raises(ValueError, match="model_class"):
        actuator.apply(_decision(), mismatched)
    invalid_target = _target().model_copy(update={"name": ""})
    with pytest.raises(ValueError, match="name"):
        actuator.apply(_decision(), invalid_target)
    invalid_decision = _decision().model_copy(update={"desired_replicas": True})
    with pytest.raises(ValueError, match="desired_replicas"):
        actuator.apply(invalid_decision, _target())
    assert calls == 0

    actuator.close()
    with pytest.raises(RuntimeError, match="closed"):
        actuator.apply(_decision(), _target())
    assert client.is_closed is False
    client.close()
