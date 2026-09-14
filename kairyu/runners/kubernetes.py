"""Read-only Kubernetes LIST watcher for managed Runner observations."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from kairyu.runners.observation import (
    KubernetesPodPhase,
    RunnerObservation,
    RunnerObservationBatch,
    RunnerPodObservation,
    RunnerRuntimeObservation,
)

RELEASE_ID_ANNOTATION = "kairyu.ai/release-id"
MODEL_ID_ANNOTATION = "kairyu.ai/model-id"
MODEL_REVISION_ANNOTATION = "kairyu.ai/model-revision"
GPU_UUIDS_ANNOTATION = "kairyu.ai/gpu-uuids"
RUNNER_CONTAINER_ANNOTATION = "kairyu.ai/runner-container"


class RunnerRuntimeSource(Protocol):
    """Read-only seam for request-store/startup/readiness observations."""

    async def poll(
        self,
        runner_ids: tuple[str, ...],
    ) -> Mapping[str, RunnerRuntimeObservation]:
        """Return observations keyed by the requested stable Pod UID."""
        ...


@dataclass(frozen=True)
class KubernetesRunnerPodSnapshot:
    """Strict identity plus Pod evidence parsed from one Kubernetes object."""

    release_id: str
    model_id: str
    model_revision: str
    pod: RunnerPodObservation


def _required_string(mapping: Mapping[str, Any], key: str, *, owner: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} requires non-empty {key!r}")
    return value


def _resource_version(payload: Any, *, owner: str) -> str:
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"{owner} response metadata must be an object")
    return _required_string(payload["metadata"], "resourceVersion", owner=owner)


def _gpu_uuids(annotations: Mapping[str, Any]) -> tuple[str, ...]:
    raw = annotations.get(GPU_UUIDS_ANNOTATION)
    if raw is None:
        return ()
    if not isinstance(raw, str):
        raise ValueError(f"{GPU_UUIDS_ANNOTATION!r} must be a JSON string")
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{GPU_UUIDS_ANNOTATION!r} must contain a JSON array"
        ) from error
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{GPU_UUIDS_ANNOTATION!r} must contain a JSON string array")
    return tuple(values)


def _status_entries(status: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    entries = status.get(key, [])
    if not isinstance(entries, list):
        raise ValueError(f"Pod status {key} must be a list")
    if not all(isinstance(entry, dict) for entry in entries):
        raise ValueError(f"Pod status {key} entries must be objects")
    return entries


def _container_state(
    status: Mapping[str, Any],
    runner_container: str | None,
) -> tuple[str | None, str | None, int | None]:
    waiting_reason: str | None = None
    terminated_reason: str | None = None
    exit_code: int | None = None
    containers = _status_entries(status, "containerStatuses")
    init_containers = _status_entries(status, "initContainerStatuses")
    if runner_container is not None:
        selected = [entry for entry in containers if entry.get("name") == runner_container]
        if len(selected) > 1:
            raise ValueError(
                "Runner container annotation matched duplicate container statuses"
            )
    else:
        selected = containers[:1]

    # Init containers may block image/startup progress, but only the selected
    # serving container owns terminal Runner failure evidence.
    for entry in (*init_containers, *selected):
        state = entry.get("state", {})
        if not isinstance(state, dict):
            raise ValueError("Pod container state must be an object")
        waiting = state.get("waiting")
        if isinstance(waiting, dict):
            reason = waiting.get("reason")
            if isinstance(reason, str) and reason and (
                waiting_reason is None
                or reason.lower()
                in {
                    "crashloopbackoff",
                    "errimagepull",
                    "imagepullbackoff",
                    "invalidimagename",
                }
            ):
                waiting_reason = reason

    if selected:
        state = selected[0].get("state", {})
        assert isinstance(state, dict)
        terminated = state.get("terminated")
        if not isinstance(terminated, dict) and isinstance(state.get("waiting"), dict):
            last_state = selected[0].get("lastState", {})
            if not isinstance(last_state, dict):
                raise ValueError("Pod container lastState must be an object")
            terminated = last_state.get("terminated")
        if isinstance(terminated, dict):
            reason = terminated.get("reason")
            code = terminated.get("exitCode")
            terminated_reason = reason if isinstance(reason, str) and reason else None
            exit_code = code if type(code) is int else None
    return waiting_reason, terminated_reason, exit_code


def parse_runner_pods(payload: Any) -> dict[str, KubernetesRunnerPodSnapshot]:
    """Parse a PodList selected exclusively for one managed Runner fleet."""

    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("Pod response must contain an items list")
    result: dict[str, KubernetesRunnerPodSnapshot] = {}
    for item in payload["items"]:
        if not isinstance(item, dict):
            raise ValueError("Pod items must be objects")
        metadata = item.get("metadata")
        spec = item.get("spec")
        status = item.get("status")
        if not all(isinstance(value, dict) for value in (metadata, spec, status)):
            raise ValueError("Pod metadata, spec, and status must be objects")
        assert isinstance(metadata, dict)
        assert isinstance(spec, dict)
        assert isinstance(status, dict)
        uid = _required_string(metadata, "uid", owner="Pod metadata")
        annotations = metadata.get("annotations", {})
        if not isinstance(annotations, dict):
            raise ValueError("Pod metadata annotations must be an object")
        release_id = _required_string(
            annotations,
            RELEASE_ID_ANNOTATION,
            owner=f"Pod {uid!r} annotations",
        )
        model_id = _required_string(
            annotations,
            MODEL_ID_ANNOTATION,
            owner=f"Pod {uid!r} annotations",
        )
        model_revision = _required_string(
            annotations,
            MODEL_REVISION_ANNOTATION,
            owner=f"Pod {uid!r} annotations",
        )
        conditions = status.get("conditions", [])
        if not isinstance(conditions, list):
            raise ValueError("Pod status conditions must be a list")
        ready = any(
            isinstance(condition, dict)
            and condition.get("type") == "Ready"
            and condition.get("status") == "True"
            for condition in conditions
        )
        runner_container = annotations.get(RUNNER_CONTAINER_ANNOTATION)
        if runner_container is not None and (
            not isinstance(runner_container, str) or not runner_container
        ):
            raise ValueError(
                f"{RUNNER_CONTAINER_ANNOTATION!r} must be a non-empty string"
            )
        containers = spec.get("containers")
        if not isinstance(containers, list) or not containers:
            raise ValueError("Pod spec containers must be a non-empty list")
        container_names = tuple(
            container.get("name") if isinstance(container, dict) else None
            for container in containers
        )
        if not all(isinstance(name, str) and name for name in container_names):
            raise ValueError("Pod spec containers require non-empty names")
        if runner_container is None:
            if len(container_names) != 1:
                raise ValueError(
                    f"multi-container Pods require {RUNNER_CONTAINER_ANNOTATION!r}"
                )
            runner_container = container_names[0]
        elif container_names.count(runner_container) != 1:
            raise ValueError(
                "Runner container annotation must match exactly one spec container"
            )
        waiting_reason, terminated_reason, exit_code = _container_state(
            status,
            runner_container,
        )
        pod = RunnerPodObservation(
            uid=uid,
            phase=KubernetesPodPhase(
                _required_string(status, "phase", owner=f"Pod {uid!r} status")
            ),
            node_name=spec.get("nodeName"),
            ready=ready,
            deleting=metadata.get("deletionTimestamp") is not None,
            waiting_reason=waiting_reason,
            terminated_reason=terminated_reason,
            exit_code=exit_code,
            gpu_uuids=_gpu_uuids(annotations),
        )
        if uid in result:
            raise ValueError(f"Pod response contains duplicate uid {uid!r}")
        result[uid] = KubernetesRunnerPodSnapshot(
            release_id,
            model_id,
            model_revision,
            pod,
        )
    return result


def parse_ready_endpoint_uids(payload: Any) -> frozenset[str]:
    """Return ready, non-terminating Pod UIDs from EndpointSlices."""

    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("EndpointSlice response must contain an items list")
    ready_uids: set[str] = set()
    for item in payload["items"]:
        if not isinstance(item, dict):
            raise ValueError("EndpointSlice items must be objects")
        endpoints = item.get("endpoints")
        if not isinstance(endpoints, list):
            raise ValueError("EndpointSlice endpoints must be a list")
        for endpoint in endpoints:
            if not isinstance(endpoint, dict):
                raise ValueError("EndpointSlice endpoints must be objects")
            conditions = endpoint.get("conditions")
            if not isinstance(conditions, dict):
                continue
            if conditions.get("ready") is not True or conditions.get("terminating") is True:
                continue
            target = endpoint.get("targetRef")
            if not isinstance(target, dict) or target.get("kind", "Pod") != "Pod":
                continue
            uid = target.get("uid")
            if isinstance(uid, str) and uid:
                ready_uids.add(uid)
    return frozenset(ready_uids)


class KubernetesRunnerWatcher:
    """Poll Pod and EndpointSlice LIST APIs without mutating cluster state."""

    _SERVICE_ACCOUNT_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")

    def __init__(
        self,
        *,
        service: str,
        namespace: str,
        pod_label_selector: str,
        runtime_source: RunnerRuntimeSource,
        api_server: str | None = None,
        token_path: str | Path | None = None,
        ca_path: str | Path | None = None,
        client: httpx.AsyncClient | None = None,
        close_client: bool | None = None,
        timeout_s: float = 10.0,
        runtime_timeout_s: float | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        for name, value in (
            ("service", service),
            ("namespace", namespace),
            ("pod_label_selector", pod_label_selector),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        if runtime_timeout_s is None:
            runtime_timeout_s = timeout_s
        if runtime_timeout_s <= 0:
            raise ValueError(
                f"runtime_timeout_s must be > 0, got {runtime_timeout_s}"
            )
        host = os.environ.get("KUBERNETES_SERVICE_HOST")
        port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        if api_server is None:
            api_host = f"[{host}]" if host and ":" in host else host
            api_server = (
                "https://kubernetes.default.svc"
                if not api_host
                else f"https://{api_host}:{port}"
            )
        base = api_server.rstrip("/")
        encoded_namespace = quote(namespace, safe="")
        self._pods_url = f"{base}/api/v1/namespaces/{encoded_namespace}/pods"
        self._slices_url = (
            f"{base}/apis/discovery.k8s.io/v1/namespaces/"
            f"{encoded_namespace}/endpointslices"
        )
        self._service = service
        self._pod_label_selector = pod_label_selector
        self._runtime_source = runtime_source
        self._token_path = Path(token_path or self._SERVICE_ACCOUNT_DIR / "token")
        resolved_ca = Path(ca_path or self._SERVICE_ACCOUNT_DIR / "ca.crt")
        if client is None:
            if close_client is False:
                raise ValueError("close_client=False requires an injected client")
            self._client = httpx.AsyncClient(verify=str(resolved_ca), timeout=timeout_s)
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False if close_client is None else close_client
        self._now = now
        self._runtime_timeout_s = runtime_timeout_s
        self._source_id = uuid.uuid4().hex
        self._source_epoch = 0
        self._poll_lock = asyncio.Lock()
        self._closed = False

    async def poll(
        self,
        tracked_runner_ids: tuple[str, ...] = (),
    ) -> RunnerObservationBatch:
        async with self._poll_lock:
            return await self._poll_once(tracked_runner_ids)

    async def _poll_once(
        self,
        tracked_runner_ids: tuple[str, ...],
    ) -> RunnerObservationBatch:
        if self._closed:
            raise RuntimeError("KubernetesRunnerWatcher is closed")
        source_started_at = self._now()
        token = self._token_path.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("Kubernetes service-account token is empty")
        headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
        pods_task = asyncio.create_task(
            self._client.get(
                self._pods_url,
                params={"labelSelector": self._pod_label_selector},
                headers=headers,
            )
        )
        slices_task = asyncio.create_task(
            self._client.get(
                self._slices_url,
                params={"labelSelector": f"kubernetes.io/service-name={self._service}"},
                headers=headers,
            )
        )
        try:
            pods_response, slices_response = await asyncio.gather(
                pods_task,
                slices_task,
            )
        except BaseException:
            pods_task.cancel()
            slices_task.cancel()
            await asyncio.gather(pods_task, slices_task, return_exceptions=True)
            raise
        pods_response.raise_for_status()
        slices_response.raise_for_status()
        pods_payload = pods_response.json()
        slices_payload = slices_response.json()
        pod_resource_version = _resource_version(pods_payload, owner="PodList")
        endpoint_resource_version = _resource_version(
            slices_payload,
            owner="EndpointSliceList",
        )
        pods = parse_runner_pods(pods_payload)
        ready_uids = parse_ready_endpoint_uids(slices_payload)
        runner_ids = tuple(sorted(pods))
        runtime_ids = tuple(sorted(set(runner_ids) | set(tracked_runner_ids)))
        async with asyncio.timeout(self._runtime_timeout_s):
            runtime = await self._runtime_source.poll(runtime_ids)
        for runner_id, item in runtime.items():
            if runner_id != item.runner_id:
                raise ValueError("runtime source keys must match runtime Runner IDs")
        unexpected = set(runtime) - set(runtime_ids)
        if unexpected:
            raise ValueError(
                f"runtime source returned unexpected Runner IDs: {sorted(unexpected)!r}"
            )
        observed_at = self._now()
        batch = RunnerObservationBatch(
            source_id=self._source_id,
            source_epoch=self._source_epoch + 1,
            source_started_at=source_started_at,
            observed_at=observed_at,
            pod_resource_version=pod_resource_version,
            endpoint_slice_resource_version=endpoint_resource_version,
            runners=tuple(
                RunnerObservation(
                    runner_id=runner_id,
                    release_id=pods[runner_id].release_id,
                    model_id=pods[runner_id].model_id,
                    model_revision=pods[runner_id].model_revision,
                    observed_at=observed_at,
                    pod=pods[runner_id].pod,
                    endpoint_ready=runner_id in ready_uids,
                    runtime=runtime.get(runner_id),
                )
                for runner_id in runner_ids
            ),
            missing_runner_runtime=tuple(
                runtime[runner_id]
                for runner_id in sorted(set(runtime) - set(runner_ids))
            ),
        )
        self._source_epoch = batch.source_epoch
        return batch

    async def aclose(self) -> None:
        async with self._poll_lock:
            if self._closed:
                return
            if self._owns_client:
                await self._client.aclose()
            self._closed = True

    async def __aenter__(self) -> KubernetesRunnerWatcher:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()
