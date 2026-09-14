"""Kubernetes LIST watcher parsing and read-only I/O contract."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from kairyu.runners import (
    GPU_UUIDS_ANNOTATION,
    MODEL_ID_ANNOTATION,
    MODEL_REVISION_ANNOTATION,
    RELEASE_ID_ANNOTATION,
    RUNNER_CONTAINER_ANNOTATION,
    KubernetesPodPhase,
    KubernetesRunnerWatcher,
    RunnerRuntimeObservation,
    parse_ready_endpoint_uids,
    parse_runner_pods,
)

NOW = datetime(2026, 9, 14, 4, 0, tzinfo=UTC)


def _pod_payload() -> dict:
    return {
        "apiVersion": "v1",
        "kind": "PodList",
        "metadata": {"resourceVersion": "101"},
        "items": [
            {
                "metadata": {
                    "name": "runner-b",
                    "uid": "uid-b",
                    "annotations": {
                        RELEASE_ID_ANNOTATION: "release-b",
                        MODEL_ID_ANNOTATION: "qwen",
                        MODEL_REVISION_ANNOTATION: "revision-b",
                    },
                },
                "spec": {"containers": [{"name": "runner"}]},
                "status": {
                    "phase": "Pending",
                    "conditions": [],
                    "containerStatuses": [
                        {
                            "name": "runner",
                            "state": {"waiting": {"reason": "ContainerCreating"}},
                        }
                    ],
                },
            },
            {
                "metadata": {
                    "name": "runner-a",
                    "uid": "uid-a",
                    "deletionTimestamp": "2026-09-14T04:00:00Z",
                    "annotations": {
                        RELEASE_ID_ANNOTATION: "release-a",
                        MODEL_ID_ANNOTATION: "qwen",
                        MODEL_REVISION_ANNOTATION: "revision-a",
                        GPU_UUIDS_ANNOTATION: '["GPU-a", "GPU-b"]',
                    },
                },
                "spec": {
                    "nodeName": "gpu-node-a",
                    "containers": [{"name": "runner"}],
                },
                "status": {
                    "phase": "Running",
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "containerStatuses": [
                        {"name": "runner", "state": {"running": {}}}
                    ],
                },
            },
        ],
    }


def _endpoint_payload() -> dict:
    return {
        "apiVersion": "discovery.k8s.io/v1",
        "kind": "EndpointSliceList",
        "metadata": {"resourceVersion": "202"},
        "items": [
            {
                "endpoints": [
                    {
                        "conditions": {"ready": True, "terminating": False},
                        "targetRef": {"kind": "Pod", "uid": "uid-a"},
                    },
                    {
                        "conditions": {"ready": False},
                        "targetRef": {"kind": "Pod", "uid": "uid-b"},
                    },
                    {
                        "conditions": {"ready": True, "terminating": True},
                        "targetRef": {"kind": "Pod", "uid": "uid-c"},
                    },
                    {
                        "conditions": {"ready": True},
                        "targetRef": {"kind": "Service", "uid": "service-a"},
                    },
                ]
            }
        ],
    }


def test_parsers_preserve_stable_identity_and_independent_gates() -> None:
    pods = parse_runner_pods(_pod_payload())
    assert set(pods) == {"uid-a", "uid-b"}
    assert pods["uid-a"].release_id == "release-a"
    assert pods["uid-a"].pod.phase is KubernetesPodPhase.RUNNING
    assert pods["uid-a"].pod.node_name == "gpu-node-a"
    assert pods["uid-a"].pod.ready is True
    assert pods["uid-a"].pod.deleting is True
    assert pods["uid-a"].pod.gpu_uuids == ("GPU-a", "GPU-b")
    assert pods["uid-b"].pod.waiting_reason == "ContainerCreating"
    assert parse_ready_endpoint_uids(_endpoint_payload()) == frozenset({"uid-a"})


def test_parser_preserves_fatal_main_container_state_over_completed_init() -> None:
    payload = _pod_payload()
    running = payload["items"][1]["status"]
    running["initContainerStatuses"] = [
        {"state": {"terminated": {"reason": "Completed", "exitCode": 0}}}
    ]
    running["containerStatuses"] = [
        {
            "name": "runner",
            "state": {"terminated": {"reason": "OOMKilled", "exitCode": 137}},
        }
    ]
    pod = parse_runner_pods(payload)["uid-a"].pod
    assert pod.terminated_reason == "OOMKilled"
    assert pod.exit_code == 137


@pytest.mark.parametrize(
    ("reason", "exit_code"),
    [("OOMKilled", 137), ("NvidiaGPUXid", 17)],
)
def test_parser_preserves_last_termination_cause_during_crash_loop(
    reason: str,
    exit_code: int,
) -> None:
    payload = _pod_payload()
    running = payload["items"][1]["status"]
    running["containerStatuses"] = [
        {
            "name": "runner",
            "state": {"waiting": {"reason": "CrashLoopBackOff"}},
            "lastState": {
                "terminated": {"reason": reason, "exitCode": exit_code}
            },
        }
    ]
    pod = parse_runner_pods(payload)["uid-a"].pod
    assert pod.waiting_reason == "CrashLoopBackOff"
    assert pod.terminated_reason == reason
    assert pod.exit_code == exit_code


def test_runner_container_identity_is_explicit_for_multi_container_pods() -> None:
    payload = _pod_payload()
    pending = payload["items"][0]
    pending["spec"]["containers"].append({"name": "metrics"})
    with pytest.raises(ValueError, match="multi-container"):
        parse_runner_pods(payload)

    pending["metadata"]["annotations"][RUNNER_CONTAINER_ANNOTATION] = "runner"
    pending["status"]["containerStatuses"] = []
    parsed = parse_runner_pods(payload)["uid-b"].pod
    assert parsed.waiting_reason is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["items"][0]["metadata"]["annotations"].pop(
            RELEASE_ID_ANNOTATION
        ),
        lambda payload: payload["items"][0]["metadata"]["annotations"].update(
            {GPU_UUIDS_ANNOTATION: "not-json"}
        ),
        lambda payload: payload["items"].append(payload["items"][0]),
    ],
)
def test_pod_parser_rejects_missing_identity_bad_gpu_and_duplicate_uid(mutate) -> None:
    payload = _pod_payload()
    mutate(payload)
    with pytest.raises(ValueError):
        parse_runner_pods(payload)


class _RuntimeSource:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    async def poll(self, runner_ids: tuple[str, ...]):
        self.calls.append(runner_ids)
        return {
            "uid-a": RunnerRuntimeObservation(
                runner_id="uid-a",
                observed_at=NOW,
                ready=False,
                active_requests=0,
            ),
        }


@pytest.mark.asyncio
async def test_watcher_reads_only_lists_rotates_token_and_joins_by_uid(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/pods"):
            return httpx.Response(200, json=_pod_payload())
        return httpx.Response(200, json=_endpoint_payload())

    token_path = tmp_path / "token"
    token_path.write_text("token-one\n", encoding="utf-8")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    runtime_source = _RuntimeSource()
    watcher = KubernetesRunnerWatcher(
        service="kairyu-runners",
        namespace="model-serving",
        pod_label_selector="app.kubernetes.io/name=kairyu-runner",
        runtime_source=runtime_source,
        api_server="https://kubernetes.example",
        token_path=token_path,
        client=client,
        now=lambda: NOW,
    )

    batch = await watcher.poll()
    token_path.write_text("token-two", encoding="utf-8")
    second_batch = await watcher.poll()

    assert batch.source_started_at == NOW
    assert batch.observed_at == NOW
    assert batch.source_epoch == 1
    assert second_batch.source_epoch == 2
    assert batch.pod_resource_version == "101"
    assert batch.endpoint_slice_resource_version == "202"
    assert tuple(item.runner_id for item in batch.runners) == ("uid-a", "uid-b")
    assert batch.runners[0].endpoint_ready is True
    assert batch.runners[0].runtime is not None
    assert batch.runners[1].endpoint_ready is False
    assert batch.runners[1].runtime is None
    assert runtime_source.calls == [("uid-a", "uid-b"), ("uid-a", "uid-b")]
    assert {request.method for request in requests} == {"GET"}
    assert [request.headers["authorization"] for request in requests] == [
        "Bearer token-one",
        "Bearer token-one",
        "Bearer token-two",
        "Bearer token-two",
    ]
    pod_requests = [request for request in requests if request.url.path.endswith("/pods")]
    assert all(
        request.url.params["labelSelector"]
        == "app.kubernetes.io/name=kairyu-runner"
        for request in pod_requests
    )
    slice_requests = [request for request in requests if "endpointslices" in request.url.path]
    assert all(
        request.url.params["labelSelector"]
        == "kubernetes.io/service-name=kairyu-runners"
        for request in slice_requests
    )
    await watcher.aclose()
    assert client.is_closed is False
    await client.aclose()


@pytest.mark.asyncio
async def test_watcher_collects_request_store_runtime_for_disappeared_runner(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = _pod_payload() if request.url.path.endswith("/pods") else _endpoint_payload()
        return httpx.Response(200, json=payload)

    class TrackedRuntimeSource:
        async def poll(self, runner_ids: tuple[str, ...]):
            assert runner_ids == ("gone-uid", "uid-a", "uid-b")
            return {
                "gone-uid": RunnerRuntimeObservation(
                    runner_id="gone-uid",
                    observed_at=NOW,
                    ready=False,
                    active_requests=0,
                )
            }

    token_path = tmp_path / "token"
    token_path.write_text("token", encoding="utf-8")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    watcher = KubernetesRunnerWatcher(
        service="runners",
        namespace="default",
        pod_label_selector="app=runners",
        runtime_source=TrackedRuntimeSource(),
        api_server="https://kubernetes.example",
        token_path=token_path,
        client=client,
        now=lambda: NOW,
    )
    batch = await watcher.poll(("gone-uid",))
    assert tuple(item.runner_id for item in batch.missing_runner_runtime) == (
        "gone-uid",
    )
    await client.aclose()


@pytest.mark.asyncio
async def test_watcher_bounds_runtime_poll_latency(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = _pod_payload() if request.url.path.endswith("/pods") else _endpoint_payload()
        return httpx.Response(200, json=payload)

    class HangingRuntimeSource:
        async def poll(self, runner_ids: tuple[str, ...]):
            await asyncio.Event().wait()
            return {}

    token_path = tmp_path / "token"
    token_path.write_text("token", encoding="utf-8")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    watcher = KubernetesRunnerWatcher(
        service="runners",
        namespace="default",
        pod_label_selector="app=runners",
        runtime_source=HangingRuntimeSource(),
        api_server="https://kubernetes.example",
        token_path=token_path,
        client=client,
        runtime_timeout_s=0.01,
        now=lambda: NOW,
    )
    with pytest.raises(TimeoutError):
        await watcher.poll()
    await client.aclose()


@pytest.mark.asyncio
async def test_failed_list_cancels_sibling_before_next_poll(tmp_path: Path) -> None:
    slice_started = asyncio.Event()
    slice_cancelled = asyncio.Event()
    release_first_slice = asyncio.Event()
    pod_calls = 0
    slice_calls = 0
    in_flight_slices = 0
    max_in_flight_slices = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal pod_calls, slice_calls, in_flight_slices, max_in_flight_slices
        if request.url.path.endswith("/pods"):
            pod_calls += 1
            if pod_calls == 1:
                await slice_started.wait()
                raise httpx.ConnectError("Pod LIST failed", request=request)
            return httpx.Response(200, json=_pod_payload())
        slice_calls += 1
        in_flight_slices += 1
        max_in_flight_slices = max(max_in_flight_slices, in_flight_slices)
        try:
            if slice_calls == 1:
                slice_started.set()
                await release_first_slice.wait()
            return httpx.Response(200, json=_endpoint_payload())
        finally:
            in_flight_slices -= 1
            if slice_calls == 1:
                slice_cancelled.set()

    token_path = tmp_path / "token"
    token_path.write_text("token", encoding="utf-8")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    watcher = KubernetesRunnerWatcher(
        service="runners",
        namespace="default",
        pod_label_selector="app=runners",
        runtime_source=_RuntimeSource(),
        api_server="https://kubernetes.example",
        token_path=token_path,
        client=client,
        now=lambda: NOW,
    )
    with pytest.raises(httpx.ConnectError):
        await watcher.poll()
    assert slice_cancelled.is_set()
    assert in_flight_slices == 0
    await watcher.poll()
    assert max_in_flight_slices == 1
    await client.aclose()


def test_internal_client_cannot_be_declared_caller_owned(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="injected client"):
        KubernetesRunnerWatcher(
            service="runners",
            namespace="default",
            pod_label_selector="app=runners",
            runtime_source=_RuntimeSource(),
            api_server="https://kubernetes.example",
            token_path=tmp_path / "token",
            close_client=False,
        )


@pytest.mark.asyncio
async def test_watcher_rejects_runtime_rows_outside_selected_pods(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = _pod_payload() if request.url.path.endswith("/pods") else _endpoint_payload()
        return httpx.Response(200, json=payload)

    class UnexpectedRuntimeSource:
        async def poll(self, runner_ids: tuple[str, ...]):
            return {
                "not-selected": RunnerRuntimeObservation(
                    runner_id="not-selected",
                    observed_at=NOW,
                    ready=False,
                    active_requests=0,
                )
            }

    token_path = tmp_path / "token"
    token_path.write_text("token", encoding="utf-8")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    watcher = KubernetesRunnerWatcher(
        service="runners",
        namespace="default",
        pod_label_selector="app=runners",
        runtime_source=UnexpectedRuntimeSource(),
        api_server="https://kubernetes.example",
        token_path=token_path,
        client=client,
        now=lambda: NOW,
    )
    with pytest.raises(ValueError, match="unexpected Runner IDs"):
        await watcher.poll()
    await client.aclose()
