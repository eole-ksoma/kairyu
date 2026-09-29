"""AdmissionReview v1 HTTP boundary for incremental Runner placement."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from kairyu.runners import (
    RUNNER_CACHE_STARTUP_PLACEMENT_ANNOTATION,
    RUNNER_CACHE_STARTUP_SCHEDULING_GATE,
    RUNNER_CACHE_STARTUP_TARGET_ANNOTATION,
    InMemoryRunnerCachePlacementAdmissionStore,
    RunnerCachePlacementAdmissionController,
    RunnerCachePlacementAdmissionPlan,
    RunnerCachePlacementAdmissionTimeoutError,
    RunnerCacheStartupBinding,
    RunnerCacheStartupPlacement,
    create_runner_cache_placement_admission_app,
)

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
TARGET = "statefulset/model-serving/qwen-runners"
USERNAME = "system:serviceaccount:kairyu:statefulset-controller"


def _binding() -> RunnerCacheStartupBinding:
    placement = RunnerCacheStartupPlacement(
        placement_id="placement-a",
        node_name="gpu-a",
        resource_flavor="h100-sxm",
        profile_id="h100-sxm-tp1",
        compatibility_approval_id="compat-qwen-h100",
        manifest_digest="a" * 64,
        pin_owner="prestage/model-serving/qwen/placement-a",
        prestage_command_id=hashlib.sha256(b"command-a").hexdigest(),
        prestage_command_generation=1,
        hint_index_revision=10,
        resident_record_generation=20,
        hint_observed_at=NOW - timedelta(seconds=1),
        hint_valid_until=NOW + timedelta(minutes=5),
    )
    payload = {
        "schema_version": "runner-cache-startup-binding-v1",
        "decision_id": "decision-a",
        "decision_fingerprint": hashlib.sha256(b"decision-a").hexdigest(),
        "target_id": TARGET,
        "target_revision": 7,
        "deployment_id": "model-serving/qwen",
        "model_class": "qwen-14b",
        "model_id": "org/qwen",
        "model_revision": "revision-a",
        "manifest_digest": "a" * 64,
        "placement_binding_id": "placement-binding-a",
        "prewarm_snapshot_id": "snapshot-a",
        "prewarm_cache_revision": 9,
        "bound_at": NOW,
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
    return RunnerCacheStartupBinding(
        binding_id=hashlib.sha256(encoded).hexdigest(),
        **payload,
    )


def _pod() -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "namespace": "model-serving",
            "name": "qwen-7",
            "ownerReferences": [
                {
                    "apiVersion": "apps/v1",
                    "kind": "StatefulSet",
                    "name": "qwen-runners",
                    "uid": "workload-uid-a",
                    "controller": True,
                }
            ],
            "annotations": {
                RUNNER_CACHE_STARTUP_TARGET_ANNOTATION: TARGET,
                "example.com/key~part": "preserved",
            },
            "labels": {"app": "qwen"},
        },
        "spec": {
            "schedulingGates": [
                {"name": RUNNER_CACHE_STARTUP_SCHEDULING_GATE},
                {"name": "example.com/other-gate"},
            ],
            "containers": [{"name": "runner", "image": "runner@sha256:deadbeef"}],
        },
    }


def _review(*, pod: dict[str, Any] | None = None, uid: str = "admission-a") -> dict:
    pod = _pod() if pod is None else pod
    return {
        "apiVersion": "admission.k8s.io/v1",
        "kind": "AdmissionReview",
        "request": {
            "uid": uid,
            "kind": {"group": "", "version": "v1", "kind": "Pod"},
            "resource": {"group": "", "version": "v1", "resource": "pods"},
            "requestKind": {"group": "", "version": "v1", "kind": "Pod"},
            "requestResource": {
                "group": "",
                "version": "v1",
                "resource": "pods",
            },
            "name": "qwen-7",
            "namespace": "model-serving",
            "operation": "CREATE",
            "userInfo": {
                "username": USERNAME,
                "uid": "service-account-uid",
                "groups": ["system:serviceaccounts"],
                "extra": {"authentication.kubernetes.io/pod-name": ["controller-0"]},
            },
            "object": pod,
            "oldObject": None,
            "dryRun": False,
            "options": {"apiVersion": "meta.k8s.io/v1", "kind": "CreateOptions"},
        },
    }


def _app(
    *,
    ready: bool = True,
    body_limit: int = 1024 * 1024,
    active_limit: int = 16,
    total_limit: int = 64,
    queue_wait_timeout_s: float = 0.5,
    controller_type: type[RunnerCachePlacementAdmissionController] = (
        RunnerCachePlacementAdmissionController
    ),
):
    binding = _binding()
    store = InMemoryRunnerCachePlacementAdmissionStore()
    store.register(
        RunnerCachePlacementAdmissionPlan(
            binding=binding,
            release_id="release-a",
            namespace="model-serving",
            owner_api_version="apps/v1",
            owner_kind="StatefulSet",
            owner_name="qwen-runners",
            owner_uid="workload-uid-a",
            creator_username=USERNAME,
            registered_at=NOW + timedelta(seconds=1),
        )
    )
    controller = controller_type(
        store,
        reauthorize=lambda candidate: candidate,
    )

    def readiness_check() -> None:
        if not ready:
            raise RuntimeError("secret database failure")

    app = create_runner_cache_placement_admission_app(
        controller=controller,
        readiness_check=readiness_check,
        clock=lambda: NOW + timedelta(seconds=2),
        request_body_limit_bytes=body_limit,
        active_request_limit=active_limit,
        total_request_limit=total_limit,
        queue_wait_timeout_s=queue_wait_timeout_s,
    )
    return app, controller


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="https://admission.test",
    )


def _pointer_tokens(path: str) -> list[str]:
    if not path:
        return []
    return [part.replace("~1", "/").replace("~0", "~") for part in path[1:].split("/")]


def _apply_patch(document: dict[str, Any], patch: list[dict[str, Any]]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    for operation in patch:
        tokens = _pointer_tokens(operation["path"])
        parent: Any = result
        for token in tokens[:-1]:
            parent = parent[int(token)] if isinstance(parent, list) else parent[token]
        key = tokens[-1]
        if operation["op"] == "remove":
            if isinstance(parent, list):
                parent.pop(int(key))
            else:
                del parent[key]
        elif operation["op"] in {"add", "replace"}:
            if isinstance(parent, list):
                parent[int(key)] = operation["value"]
            else:
                parent[key] = operation["value"]
        else:  # pragma: no cover - catches accidental implementation expansion
            raise AssertionError(f"unsupported operation: {operation['op']}")
    return result


@pytest.mark.asyncio
async def test_pod_create_returns_uid_bound_json_patch() -> None:
    app, controller = _app()
    original = _pod()
    review = _review(pod=copy.deepcopy(original))

    async with _client(app) as client:
        response = await client.post("/v1/admit", json=review)

    assert response.status_code == 200
    envelope = response.json()
    assert envelope["apiVersion"] == "admission.k8s.io/v1"
    assert envelope["kind"] == "AdmissionReview"
    assert envelope["response"]["uid"] == "admission-a"
    assert envelope["response"]["allowed"] is True
    assert envelope["response"]["patchType"] == "JSONPatch"
    patch = json.loads(base64.b64decode(envelope["response"]["patch"]))
    patched = _apply_patch(original, patch)
    expected, claim = controller.admit(
        original,
        admission_uid="admission-replay",
        request_username=USERNAME,
        observed_at=NOW + timedelta(seconds=3),
    )
    assert patched == expected
    assert claim.placement_id == "placement-a"
    assert (
        patched["metadata"]["annotations"][RUNNER_CACHE_STARTUP_PLACEMENT_ANNOTATION]
        == "placement-a"
    )
    assert patched["metadata"]["annotations"]["example.com/key~part"] == "preserved"
    assert patched["spec"]["schedulingGates"] == [{"name": "example.com/other-gate"}]


@pytest.mark.asyncio
async def test_exact_admitted_replay_is_allowed_without_empty_patch_fields() -> None:
    app, controller = _app()
    admitted, _claim = controller.admit(
        _pod(),
        admission_uid="first",
        request_username=USERNAME,
        observed_at=NOW + timedelta(seconds=2),
    )

    async with _client(app) as client:
        response = await client.post("/v1/admit", json=_review(pod=admitted, uid="retry"))

    result = response.json()["response"]
    assert result == {"uid": "retry", "allowed": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda request: request.__setitem__("operation", "UPDATE"), "only Pod CREATE"),
        (lambda request: request["kind"].__setitem__("kind", "Secret"), "only core/v1 Pod"),
        (lambda request: request.__setitem__("subResource", "status"), "subresource"),
        (lambda request: request.__setitem__("dryRun", True), "dry-run"),
        (lambda request: request.__setitem__("oldObject", {}), "oldObject"),
        (lambda request: request.__setitem__("name", "qwen-8"), "names must match"),
    ],
)
async def test_semantically_invalid_reviews_are_uid_bound_denials(mutate, message: str) -> None:
    app, _controller = _app()
    review = _review()
    mutate(review["request"])

    async with _client(app) as client:
        response = await client.post("/v1/admit", json=review)

    assert response.status_code == 200
    decision = response.json()["response"]
    assert decision["uid"] == "admission-a"
    assert decision["allowed"] is False
    assert decision["status"]["code"] == 422
    assert message in decision["status"]["message"]
    assert "patch" not in decision


@pytest.mark.asyncio
async def test_authorization_failure_is_sanitized() -> None:
    app, _controller = _app()
    review = _review()
    review["request"]["userInfo"]["username"] = "attacker-with-secret-name"

    async with _client(app) as client:
        response = await client.post("/v1/admit", json=review)

    decision = response.json()["response"]
    assert decision["allowed"] is False
    assert decision["status"] == {
        "code": 422,
        "message": "Pod is not eligible for cache placement admission",
    }
    assert "attacker" not in response.text
    assert "placement-a" not in response.text


@pytest.mark.asyncio
async def test_strict_json_content_type_schema_and_body_limit() -> None:
    app, _controller = _app()
    duplicate = (
        b'{"apiVersion":"admission.k8s.io/v1",'
        b'"apiVersion":"admission.k8s.io/v1","kind":"AdmissionReview"}'
    )
    nonfinite = b'{"apiVersion":NaN}'
    overflow_review = _review()
    overflow_review["request"]["object"]["spec"]["schedulingGates"][1]["weight"] = "__OVERFLOW__"
    overflow = json.dumps(overflow_review, separators=(",", ":")).replace('"__OVERFLOW__"', "1e999")
    async with _client(app) as client:
        wrong_type = await client.post(
            "/v1/admit", content=b"{}", headers={"Content-Type": "text/plain"}
        )
        duplicate_response = await client.post(
            "/v1/admit", content=duplicate, headers={"Content-Type": "application/json"}
        )
        nonfinite_response = await client.post(
            "/v1/admit", content=nonfinite, headers={"Content-Type": "application/json"}
        )
        overflow_response = await client.post(
            "/v1/admit",
            content=overflow,
            headers={"Content-Type": "application/json"},
        )
        unknown_field = _review()
        unknown_field["request"]["unexpected"] = True
        unknown_response = await client.post("/v1/admit", json=unknown_field)
    limited_app, _limited_controller = _app(body_limit=512)
    async with _client(limited_app) as client:
        oversized = await client.post("/v1/admit", json=_review())

    assert wrong_type.status_code == 415
    assert duplicate_response.status_code == 400
    assert nonfinite_response.status_code == 400
    assert overflow_response.status_code == 400
    assert unknown_response.status_code == 422
    assert oversized.status_code == 413
    for response in (
        duplicate_response,
        nonfinite_response,
        overflow_response,
        unknown_response,
    ):
        assert "admission-a" not in response.text
        assert "input" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("wire_name", "python_name"),
    [("apiVersion", "api_version"), ("userInfo", "user_info")],
)
async def test_python_field_names_are_not_wire_aliases(wire_name: str, python_name: str) -> None:
    app, _controller = _app()
    review = _review()
    container = review if wire_name == "apiVersion" else review["request"]
    container[python_name] = container.pop(wire_name)

    async with _client(app) as client:
        response = await client.post("/v1/admit", json=review)

    assert response.status_code == 422
    assert response.json() == {"detail": "request body does not match AdmissionReview v1"}


@pytest.mark.asyncio
async def test_chunked_body_is_bounded_without_content_length() -> None:
    app, _controller = _app(body_limit=512)
    body = json.dumps(_review(), separators=(",", ":")).encode()

    async def chunks():
        for start in range(0, len(body), 100):
            yield body[start : start + 100]
            await asyncio.sleep(0)

    async with _client(app) as client:
        response = await client.post(
            "/v1/admit",
            content=chunks(),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 413


class _BlockingController(RunnerCachePlacementAdmissionController):
    entered = threading.Event()
    release = threading.Event()

    def admit(self, *args, **kwargs):
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test controller timed out")
        return super().admit(*args, **kwargs)


@pytest.mark.asyncio
async def test_active_queue_and_overflow_are_bounded() -> None:
    _BlockingController.entered.clear()
    _BlockingController.release.clear()
    app, _controller = _app(
        active_limit=1,
        total_limit=2,
        queue_wait_timeout_s=1.0,
        controller_type=_BlockingController,
    )

    async with _client(app) as client:
        active = asyncio.create_task(client.post("/v1/admit", json=_review(uid="one")))
        assert await asyncio.to_thread(_BlockingController.entered.wait, 2)
        queued = asyncio.create_task(client.post("/v1/admit", json=_review(uid="two")))
        await asyncio.sleep(0.05)
        overflow = await client.post("/v1/admit", json=_review(uid="three"))
        _BlockingController.release.set()
        active_response, queued_response = await asyncio.gather(active, queued)

    assert overflow.status_code == 429
    assert active_response.status_code == 200
    assert queued_response.status_code == 200


class _ExplodingController(RunnerCachePlacementAdmissionController):
    def admit(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("postgresql host and password must stay private")


class _TimeoutController(RunnerCachePlacementAdmissionController):
    def admit(self, *args, **kwargs):
        del args, kwargs
        raise RunnerCachePlacementAdmissionTimeoutError("private deadline detail")


@pytest.mark.asyncio
async def test_unexpected_backend_failure_is_a_sanitized_uid_bound_denial() -> None:
    store = InMemoryRunnerCachePlacementAdmissionStore()
    controller = _ExplodingController(store, reauthorize=lambda candidate: candidate)
    app = create_runner_cache_placement_admission_app(
        controller=controller,
        readiness_check=lambda: None,
        clock=lambda: NOW,
    )

    async with _client(app) as client:
        response = await client.post("/v1/admit", json=_review())

    decision = response.json()["response"]
    assert decision == {
        "uid": "admission-a",
        "allowed": False,
        "status": {
            "code": 503,
            "message": "cache placement admission is temporarily unavailable",
        },
    }
    assert "postgresql" not in response.text
    assert "password" not in response.text


@pytest.mark.asyncio
async def test_internal_deadline_is_a_sanitized_uid_bound_denial() -> None:
    store = InMemoryRunnerCachePlacementAdmissionStore()
    controller = _TimeoutController(store, reauthorize=lambda candidate: candidate)
    app = create_runner_cache_placement_admission_app(
        controller=controller,
        readiness_check=lambda: None,
        clock=lambda: NOW,
    )

    async with _client(app) as client:
        response = await client.post("/v1/admit", json=_review())

    decision = response.json()["response"]
    assert decision["uid"] == "admission-a"
    assert decision["allowed"] is False
    assert decision["status"] == {
        "code": 503,
        "message": "cache placement admission exceeded its internal deadline",
    }
    assert "private" not in response.text


@pytest.mark.asyncio
async def test_health_and_readiness_disclose_no_backend_detail() -> None:
    app, _controller = _app(ready=False)

    async with _client(app) as client:
        health = await client.get("/health")
        ready = await client.get("/readyz")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code == 503
    assert ready.json() == {"status": "not_ready"}
    assert "database" not in ready.text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"request_body_limit_bytes": 0},
        {"active_request_limit": 0},
        {"active_request_limit": 2, "total_request_limit": 1},
        {"queue_wait_timeout_s": float("nan")},
        {"request_timeout_s": 0},
    ],
)
def test_app_configuration_is_validated(kwargs: dict[str, Any]) -> None:
    store = InMemoryRunnerCachePlacementAdmissionStore()
    controller = RunnerCachePlacementAdmissionController(
        store, reauthorize=lambda candidate: candidate
    )
    with pytest.raises(ValueError):
        create_runner_cache_placement_admission_app(
            controller=controller,
            readiness_check=lambda: None,
            **kwargs,
        )
