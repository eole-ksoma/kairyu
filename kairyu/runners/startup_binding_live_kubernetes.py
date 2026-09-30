"""Deadline-bounded Kubernetes and Kueue placement-binding readers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import math
import os
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from kairyu.runners.kubernetes import (
    MODEL_ID_ANNOTATION,
    MODEL_REVISION_ANNOTATION,
    RELEASE_ID_ANNOTATION,
)
from kairyu.runners.prewarm import ModelCachePlacementCandidate
from kairyu.runners.scale_actuator import (
    CACHE_PLACEMENT_BINDING_ANNOTATION,
    SCALE_DECISION_FINGERPRINT_ANNOTATION,
    SCALE_DECISION_GENERATION_ANNOTATION,
    SCALE_DECISION_ID_ANNOTATION,
)
from kairyu.runners.scaling_log import ScalingDecisionRecord
from kairyu.runners.scaling_quota import (
    ScalingQuotaAdmission,
    ScalingQuotaLimit,
    ScalingQuotaSnapshot,
    admit_scaling_quota,
    kueue_scaling_workload_name,
    parse_kueue_scaling_admission,
)
from kairyu.runners.startup_binding import RunnerCacheStartupBinding
from kairyu.runners.startup_binding_authority import (
    RunnerCachePlacementBindingAuthorizationDeniedError,
)
from kairyu.runners.startup_binding_live_authority import (
    RunnerCachePlacementBindingTargetState,
)
from kairyu.runners.startup_binding_live_cache import (
    RunnerCachePlacementBindingInventory,
)

_AUTHORITY_API_VERSION = "autoscaling.kairyu.ai/v1alpha1"
_QUOTA_KIND = "RunnerScalingQuotaSnapshot"
_QUOTA_PLURAL = "runnerscalingquotasnapshots"
_INVENTORY_KIND = "RunnerCachePlacementInventory"
_INVENTORY_PLURAL = "runnercacheplacementinventories"
_MAX_SIGNED_BIGINT = 2**63 - 1


class InvalidKubernetesPlacementBindingLiveResponseError(RuntimeError):
    """A Kubernetes authority response violated the live-reader contract."""


def _text(value: object, *, name: str, max_length: int = 255) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
        or len(value) > max_length
    ):
        raise ValueError(f"{name} must be a bounded non-empty string without NUL")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _MAX_SIGNED_BIGINT:
        raise InvalidKubernetesPlacementBindingLiveResponseError(
            f"{name} must be an integer in [{minimum}, 2^63-1]"
        )
    return value


def _aware(value: object, *, name: str) -> datetime:
    if not isinstance(value, str):
        raise InvalidKubernetesPlacementBindingLiveResponseError(
            f"{name} must be an RFC 3339 timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidKubernetesPlacementBindingLiveResponseError(
            f"{name} must be an RFC 3339 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise InvalidKubernetesPlacementBindingLiveResponseError(f"{name} must be timezone-aware")
    return parsed


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise InvalidKubernetesPlacementBindingLiveResponseError(f"{name} must be an object")
    return value


def _string_map(value: object, *, name: str) -> dict[str, str]:
    mapping = _mapping(value, name=name)
    if any(not isinstance(item, str) for item in mapping.values()):
        raise InvalidKubernetesPlacementBindingLiveResponseError(
            f"{name} must contain string pairs"
        )
    return mapping  # type: ignore[return-value]


def _strict_keys(
    value: dict[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    if set(value) != expected:
        raise InvalidKubernetesPlacementBindingLiveResponseError(
            f"{name} fields do not match the authority contract"
        )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number is not allowed")
    return parsed


def _utc_now() -> datetime:
    return datetime.now(UTC)


class KubernetesPlacementBindingLiveTarget(BaseModel):
    """Trusted routing for one model class and its authority CRDs."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-kubernetes-placement-binding-live-target-v1"] = (
        "runner-kubernetes-placement-binding-live-target-v1"
    )
    model_class: str = Field(max_length=128)
    target_kind: Literal["Deployment", "StatefulSet"]
    namespace: str = Field(max_length=253)
    name: str = Field(max_length=253)
    authority_namespace: str = Field(max_length=253)
    kueue_namespace: str = Field(max_length=253)
    kueue_api_version: Literal[
        "kueue.x-k8s.io/v1beta1",
        "kueue.x-k8s.io/v1beta2",
    ] = "kueue.x-k8s.io/v1beta2"
    quota_snapshot_name: str = Field(max_length=253)
    inventory_name: str = Field(max_length=253)
    pod_set_name: str = Field(max_length=253)
    resource_name: str = Field(default="nvidia.com/gpu", max_length=253)

    @field_validator(
        "model_class",
        "namespace",
        "name",
        "authority_namespace",
        "kueue_namespace",
        "quota_snapshot_name",
        "inventory_name",
        "pod_set_name",
        "resource_name",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        maximum = 128 if info.field_name == "model_class" else 253
        return _text(value, name=info.field_name, max_length=maximum)


class _ObjectMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    namespace: str
    uid: str
    generation: int
    resource_version: str


class KubernetesKueueRunnerCachePlacementBindingReader:
    """Read target, quota, and placement inventory from Kubernetes authority.

    Standard workload and Kueue objects are joined with two controller-owned
    status CRDs. The CRDs retain the monotonic quota/cache revision domains and
    the scheduler facts that standard Kubernetes resources cannot represent.
    All routing is supplied by trusted configuration, never by the candidate
    binding.
    """

    _SERVICE_ACCOUNT_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")

    def __init__(
        self,
        *,
        targets: tuple[KubernetesPlacementBindingLiveTarget, ...],
        api_server: str | None = None,
        token_path: str | Path | None = None,
        ca_path: str | Path | None = None,
        client: httpx.AsyncClient | None = None,
        close_client: bool | None = None,
        response_limit_bytes: int = 2 * 1024 * 1024,
        monotonic_clock=time.monotonic,
        wall_clock=_utc_now,
    ) -> None:
        if not isinstance(targets, tuple) or not targets:
            raise TypeError("targets must be a non-empty tuple")
        if len(targets) > 1024:
            raise ValueError("targets exceed the supported model-class count")
        validated = tuple(
            KubernetesPlacementBindingLiveTarget.model_validate(target.model_dump())
            if isinstance(target, KubernetesPlacementBindingLiveTarget)
            else None
            for target in targets
        )
        if any(target is None for target in validated):
            raise TypeError("targets must contain KubernetesPlacementBindingLiveTarget values")
        values = tuple(target for target in validated if target is not None)
        model_classes = tuple(target.model_class for target in values)
        if model_classes != tuple(sorted(model_classes)) or len(set(model_classes)) != len(
            model_classes
        ):
            raise ValueError("targets must use unique canonical model classes")
        if (
            type(response_limit_bytes) is not int
            or not 1 <= response_limit_bytes <= 16 * 1024 * 1024
        ):
            raise ValueError("response_limit_bytes must be an integer in [1, 16777216]")
        if not callable(monotonic_clock) or not callable(wall_clock):
            raise TypeError("clocks must be callable")

        host = os.environ.get("KUBERNETES_SERVICE_HOST")
        port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        if api_server is None:
            api_host = f"[{host}]" if host and ":" in host else host
            api_server = (
                "https://kubernetes.default.svc" if not api_host else f"https://{api_host}:{port}"
            )
        api_server = _text(api_server, name="api_server", max_length=2048)
        parsed = urlsplit(api_server)
        try:
            parsed_port = parsed.port
        except ValueError as exc:
            raise ValueError("api_server contains an invalid port") from exc
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
            or (parsed_port is not None and not 1 <= parsed_port <= 65535)
        ):
            raise ValueError("api_server must be one credential-free HTTPS origin")

        resolved_ca = Path(ca_path or self._SERVICE_ACCOUNT_DIR / "ca.crt")
        if client is None:
            if close_client is False:
                raise ValueError("Kubernetes live reader must own its async client")
            client = httpx.AsyncClient(
                verify=str(resolved_ca),
                follow_redirects=False,
                trust_env=False,
            )
        else:
            if close_client is False:
                raise ValueError("Kubernetes live reader must adopt its async client")
            if client.follow_redirects:
                raise ValueError("Kubernetes live reader must not follow redirects")
            if client.trust_env:
                raise ValueError("Kubernetes live reader must disable environment trust")
        self._api_server = api_server.rstrip("/")
        self._token_path = Path(token_path or self._SERVICE_ACCOUNT_DIR / "token")
        self._client = client
        self._targets = {target.model_class: target for target in values}
        self._response_limit_bytes = response_limit_bytes
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._lock = threading.RLock()
        self._closed = False
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            name="kairyu-kubernetes-placement-binding-live",
            daemon=True,
        )
        self._inflight: set[concurrent.futures.Future[Any]] = set()
        self._loop_thread.start()

    @staticmethod
    def _validate_budget(deadline_monotonic: float, backend_timeout_s: float) -> None:
        for name, value in (
            ("deadline_monotonic", deadline_monotonic),
            ("backend_timeout_s", backend_timeout_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TypeError(f"{name} must be a finite number")
        if backend_timeout_s <= 0:
            raise ValueError("backend_timeout_s must be positive")

    def _remaining(self, operation_deadline: float) -> float:
        remaining = operation_deadline - self._monotonic_clock()
        if remaining <= 0:
            raise TimeoutError("Kubernetes placement-binding live read deadline expired")
        return remaining

    @contextmanager
    def _operation(self, *, deadline_monotonic: float, backend_timeout_s: float):
        self._validate_budget(deadline_monotonic, backend_timeout_s)
        started_at = self._monotonic_clock()
        operation_deadline = min(deadline_monotonic, started_at + backend_timeout_s)
        remaining = operation_deadline - started_at
        if remaining <= 0:
            raise TimeoutError("Kubernetes placement-binding live read deadline expired")
        if not self._lock.acquire(timeout=remaining):
            raise TimeoutError("Kubernetes placement-binding live reader lock timed out")
        try:
            if self._closed:
                raise RuntimeError("Kubernetes placement-binding live reader is closed")
            self._remaining(operation_deadline)
            yield operation_deadline
            self._remaining(operation_deadline)
        finally:
            self._lock.release()

    def _headers(self, *, operation_deadline: float) -> dict[str, str]:
        self._remaining(operation_deadline)
        descriptor = os.open(
            self._token_path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK,
        )
        try:
            raw = os.read(descriptor, 64 * 1024 + 1)
        finally:
            os.close(descriptor)
        self._remaining(operation_deadline)
        if len(raw) > 64 * 1024:
            raise ValueError("Kubernetes service-account token exceeds the supported size")
        try:
            token = raw.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("Kubernetes service-account token must be ASCII") from exc
        if not token or any(ord(character) < 0x21 or ord(character) > 0x7E for character in token):
            raise ValueError("Kubernetes service-account token is empty or malformed")
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }

    async def _request_json_async(
        self,
        url: str,
        *,
        operation_deadline: float,
        missing_is_denial: bool,
        params: dict[str, str] | None = None,
    ) -> Any:
        headers = self._headers(operation_deadline=operation_deadline)
        timeout = self._remaining(operation_deadline)
        try:
            async with asyncio.timeout(timeout):
                async with self._client.stream(
                    "GET",
                    url,
                    params=params,
                    headers=headers,
                    timeout=httpx.Timeout(timeout),
                ) as response:
                    if response.status_code == 404 and missing_is_denial:
                        raise RunnerCachePlacementBindingAuthorizationDeniedError(
                            "Kubernetes placement-binding authority is unavailable"
                        )
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    if content_type.split(";", 1)[0].strip().lower() != "application/json":
                        raise InvalidKubernetesPlacementBindingLiveResponseError(
                            "Kubernetes authority response must use application/json"
                        )
                    raw_length = response.headers.get("content-length")
                    if raw_length is not None:
                        try:
                            content_length = int(raw_length)
                        except ValueError as exc:
                            raise InvalidKubernetesPlacementBindingLiveResponseError(
                                "Kubernetes authority response has invalid content-length"
                            ) from exc
                        if content_length < 0 or content_length > self._response_limit_bytes:
                            raise InvalidKubernetesPlacementBindingLiveResponseError(
                                "Kubernetes authority response exceeds the configured limit"
                            )
                    chunks: list[bytes] = []
                    received = 0
                    async for chunk in response.aiter_bytes():
                        self._remaining(operation_deadline)
                        received += len(chunk)
                        if received > self._response_limit_bytes:
                            raise InvalidKubernetesPlacementBindingLiveResponseError(
                                "Kubernetes authority response exceeds the configured limit"
                            )
                        chunks.append(chunk)
        except (httpx.TimeoutException, httpx.PoolTimeout) as exc:
            raise TimeoutError("Kubernetes placement-binding request timed out") from exc
        self._remaining(operation_deadline)
        try:
            parsed = json.loads(
                b"".join(chunks),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
                parse_float=_finite_float,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise InvalidKubernetesPlacementBindingLiveResponseError(
                "Kubernetes authority response must be strict JSON"
            ) from exc
        self._remaining(operation_deadline)
        return parsed

    def _request_json(
        self,
        url: str,
        *,
        operation_deadline: float,
        missing_is_denial: bool,
        params: dict[str, str] | None = None,
    ) -> Any:
        coroutine = self._request_json_async(
            url,
            operation_deadline=operation_deadline,
            missing_is_denial=missing_is_denial,
            params=params,
        )
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        self._inflight.add(future)
        try:
            return future.result()
        finally:
            self._inflight.discard(future)

    def _target_config(
        self,
        binding: RunnerCacheStartupBinding,
        decision: ScalingDecisionRecord,
    ) -> KubernetesPlacementBindingLiveTarget:
        if not isinstance(binding, RunnerCacheStartupBinding):
            raise TypeError("binding must be a RunnerCacheStartupBinding")
        if not isinstance(decision, ScalingDecisionRecord):
            raise TypeError("decision must be a ScalingDecisionRecord")
        target = decision.target_revision
        configured = self._targets.get(binding.model_class)
        if (
            configured is None
            or decision.policy.model_class != binding.model_class
            or target is None
            or (
                target.target_kind,
                target.namespace,
                target.name,
            )
            != (
                configured.target_kind,
                configured.namespace,
                configured.name,
            )
        ):
            raise RunnerCachePlacementBindingAuthorizationDeniedError(
                "binding target is not configured for Kubernetes live authority"
            )
        return configured

    def _target_url(self, target: KubernetesPlacementBindingLiveTarget) -> str:
        plural = "deployments" if target.target_kind == "Deployment" else "statefulsets"
        return (
            f"{self._api_server}/apis/apps/v1/namespaces/"
            f"{quote(target.namespace, safe='')}/{plural}/{quote(target.name, safe='')}"
        )

    def _authority_url(
        self,
        target: KubernetesPlacementBindingLiveTarget,
        *,
        plural: str,
        name: str,
    ) -> str:
        return (
            f"{self._api_server}/apis/{_AUTHORITY_API_VERSION}/namespaces/"
            f"{quote(target.authority_namespace, safe='')}/{plural}/{quote(name, safe='')}"
        )

    def _kueue_url(
        self,
        target: KubernetesPlacementBindingLiveTarget,
        workload_name: str,
        *,
        api_version: str,
    ) -> str:
        version = api_version.rsplit("/", 1)[1]
        return (
            f"{self._api_server}/apis/kueue.x-k8s.io/{version}/namespaces/"
            f"{quote(target.kueue_namespace, safe='')}/workloads/"
            f"{quote(workload_name, safe='')}"
        )

    @staticmethod
    def _root(payload: Any, *, api_version: str, kind: str) -> dict[str, Any]:
        root = _mapping(payload, name=f"{kind} response")
        if root.get("apiVersion") != api_version or root.get("kind") != kind:
            raise InvalidKubernetesPlacementBindingLiveResponseError(
                f"Kubernetes authority response must use {api_version} kind {kind}"
            )
        return root

    @staticmethod
    def _metadata(root: dict[str, Any], *, name: str) -> _ObjectMetadata:
        metadata = _mapping(root.get("metadata"), name=f"{name} metadata")
        return _ObjectMetadata(
            name=_text(metadata.get("name"), name=f"{name} metadata.name", max_length=253),
            namespace=_text(
                metadata.get("namespace"),
                name=f"{name} metadata.namespace",
                max_length=253,
            ),
            uid=_text(metadata.get("uid"), name=f"{name} metadata.uid"),
            generation=_integer(
                metadata.get("generation"),
                name=f"{name} metadata.generation",
                minimum=1,
            ),
            resource_version=_text(
                metadata.get("resourceVersion"),
                name=f"{name} metadata.resourceVersion",
            ),
        )

    @staticmethod
    def _ready_status(
        root: dict[str, Any],
        *,
        metadata: _ObjectMetadata,
        name: str,
    ) -> dict[str, Any]:
        status = _mapping(root.get("status"), name=f"{name} status")
        observed_generation = _integer(
            status.get("observedGeneration"),
            name=f"{name} status.observedGeneration",
            minimum=1,
        )
        conditions = status.get("conditions")
        if not isinstance(conditions, list) or not all(
            isinstance(item, dict) for item in conditions
        ):
            raise InvalidKubernetesPlacementBindingLiveResponseError(
                f"{name} status.conditions must be an object list"
            )
        for condition in conditions:
            condition_type = condition.get("type")
            condition_status = condition.get("status")
            if not isinstance(condition_type, str) or condition_status not in {
                "True",
                "False",
                "Unknown",
            }:
                raise InvalidKubernetesPlacementBindingLiveResponseError(
                    f"{name} status.conditions contain malformed state"
                )
            if "observedGeneration" in condition:
                _integer(
                    condition["observedGeneration"],
                    name=f"{name} condition observedGeneration",
                    minimum=1,
                )
        ready = [condition for condition in conditions if condition.get("type") == "Ready"]
        if (
            observed_generation != metadata.generation
            or len(ready) != 1
            or ready[0].get("status") != "True"
            or ready[0].get("observedGeneration") != metadata.generation
        ):
            raise RunnerCachePlacementBindingAuthorizationDeniedError(
                f"{name} authority status is not current and ready"
            )
        return status

    @staticmethod
    def _target_reference(
        value: object,
        *,
        configured: KubernetesPlacementBindingLiveTarget,
        workload_uid: str,
        name: str,
    ) -> None:
        reference = _mapping(value, name=name)
        _strict_keys(
            reference,
            {"apiVersion", "kind", "namespace", "name", "uid"},
            name=name,
        )
        normalized = {
            "apiVersion": _text(reference["apiVersion"], name=f"{name}.apiVersion"),
            "kind": _text(reference["kind"], name=f"{name}.kind"),
            "namespace": _text(
                reference["namespace"],
                name=f"{name}.namespace",
                max_length=253,
            ),
            "name": _text(reference["name"], name=f"{name}.name", max_length=253),
            "uid": _text(reference["uid"], name=f"{name}.uid"),
        }
        if normalized != {
            "apiVersion": "apps/v1",
            "kind": configured.target_kind,
            "namespace": configured.namespace,
            "name": configured.name,
            "uid": workload_uid,
        }:
            raise RunnerCachePlacementBindingAuthorizationDeniedError(
                "controller authority references a different Kubernetes target"
            )

    def read_target(
        self,
        binding: RunnerCacheStartupBinding,
        decision: ScalingDecisionRecord,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> RunnerCachePlacementBindingTargetState:
        if not isinstance(binding, RunnerCacheStartupBinding):
            raise TypeError("binding must be a RunnerCacheStartupBinding")
        if not isinstance(decision, ScalingDecisionRecord):
            raise TypeError("decision must be a ScalingDecisionRecord")
        binding = RunnerCacheStartupBinding.model_validate(binding.model_dump())
        decision = ScalingDecisionRecord.model_validate(decision.model_dump())
        configured = self._target_config(binding, decision)
        with self._operation(
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=backend_timeout_s,
        ) as operation_deadline:
            payload = self._request_json(
                self._target_url(configured),
                operation_deadline=operation_deadline,
                missing_is_denial=True,
            )
            root = self._root(payload, api_version="apps/v1", kind=configured.target_kind)
            metadata = self._metadata(root, name="workload")
            if (metadata.namespace, metadata.name) != (configured.namespace, configured.name):
                raise InvalidKubernetesPlacementBindingLiveResponseError(
                    "workload response identity does not match configured target"
                )
            spec = _mapping(root.get("spec"), name="workload spec")
            replicas = _integer(spec.get("replicas"), name="workload spec.replicas")
            if replicas > 100_000:
                raise InvalidKubernetesPlacementBindingLiveResponseError(
                    "workload spec.replicas exceeds the supported limit"
                )
            annotations = _string_map(
                _mapping(root.get("metadata"), name="workload metadata").get(
                    "annotations",
                    {},
                ),
                name="workload metadata.annotations",
            )
            required = {
                RELEASE_ID_ANNOTATION,
                MODEL_ID_ANNOTATION,
                MODEL_REVISION_ANNOTATION,
                CACHE_PLACEMENT_BINDING_ANNOTATION,
                SCALE_DECISION_GENERATION_ANNOTATION,
                SCALE_DECISION_ID_ANNOTATION,
                SCALE_DECISION_FINGERPRINT_ANNOTATION,
            }
            if not required.issubset(annotations):
                raise RunnerCachePlacementBindingAuthorizationDeniedError(
                    "Kubernetes target lacks the placement-binding decision fence"
                )
            try:
                decision_generation = int(annotations[SCALE_DECISION_GENERATION_ANNOTATION])
            except ValueError as exc:
                raise RunnerCachePlacementBindingAuthorizationDeniedError(
                    "Kubernetes target carries an invalid decision generation"
                ) from exc
            if (
                not 1 <= decision_generation <= _MAX_SIGNED_BIGINT
                or str(decision_generation) != annotations[SCALE_DECISION_GENERATION_ANNOTATION]
            ):
                raise RunnerCachePlacementBindingAuthorizationDeniedError(
                    "Kubernetes target carries a non-canonical decision generation"
                )
            observed_at = self._wall_clock()
            if (
                not isinstance(observed_at, datetime)
                or observed_at.tzinfo is None
                or observed_at.utcoffset() is None
            ):
                raise ValueError("wall clock must return a timezone-aware datetime")
            return RunnerCachePlacementBindingTargetState(
                observed_at=observed_at,
                target_kind=configured.target_kind,
                namespace=metadata.namespace,
                name=metadata.name,
                workload_uid=metadata.uid,
                workload_generation=metadata.generation,
                release_id=annotations[RELEASE_ID_ANNOTATION],
                model_id=annotations[MODEL_ID_ANNOTATION],
                model_revision=annotations[MODEL_REVISION_ANNOTATION],
                placement_binding_id=annotations[CACHE_PLACEMENT_BINDING_ANNOTATION],
                decision_generation=decision_generation,
                decision_id=annotations[SCALE_DECISION_ID_ANNOTATION],
                decision_fingerprint=annotations[SCALE_DECISION_FINGERPRINT_ANNOTATION],
                replicas=replicas,
            )

    def _parse_quota_snapshot(
        self,
        payload: Any,
        *,
        configured: KubernetesPlacementBindingLiveTarget,
        decision: ScalingDecisionRecord,
        kueue,
    ) -> ScalingQuotaAdmission:
        original = decision.quota_admission
        target = decision.target_revision
        assert original is not None and target is not None
        root = self._root(payload, api_version=_AUTHORITY_API_VERSION, kind=_QUOTA_KIND)
        metadata = self._metadata(root, name="quota snapshot")
        if (metadata.namespace, metadata.name) != (
            configured.authority_namespace,
            configured.quota_snapshot_name,
        ):
            raise InvalidKubernetesPlacementBindingLiveResponseError(
                "quota snapshot identity does not match configured authority"
            )
        spec = _mapping(root.get("spec"), name="quota snapshot spec")
        _strict_keys(
            spec,
            {
                "targetRef",
                "tenantId",
                "modelClass",
                "modelFamily",
                "gpusPerReplica",
                "kueueWorkloadRef",
            },
            name="quota snapshot spec",
        )
        self._target_reference(
            spec["targetRef"],
            configured=configured,
            workload_uid=target.workload_uid,
            name="quota snapshot spec.targetRef",
        )
        kueue_ref = _mapping(
            spec["kueueWorkloadRef"],
            name="quota snapshot spec.kueueWorkloadRef",
        )
        tenant_id = _text(spec["tenantId"], name="quota snapshot spec.tenantId")
        model_class = _text(
            spec["modelClass"],
            name="quota snapshot spec.modelClass",
            max_length=128,
        )
        model_family = _text(
            spec["modelFamily"],
            name="quota snapshot spec.modelFamily",
            max_length=128,
        )
        gpus_per_replica = _integer(
            spec["gpusPerReplica"],
            name="quota snapshot spec.gpusPerReplica",
            minimum=1,
        )
        if gpus_per_replica > 1024:
            raise InvalidKubernetesPlacementBindingLiveResponseError(
                "quota snapshot spec.gpusPerReplica exceeds the supported limit"
            )
        _strict_keys(
            kueue_ref,
            {"apiVersion", "namespace", "name", "uid"},
            name="quota snapshot spec.kueueWorkloadRef",
        )
        normalized_kueue_ref = {
            "apiVersion": _text(
                kueue_ref["apiVersion"],
                name="quota snapshot spec.kueueWorkloadRef.apiVersion",
            ),
            "namespace": _text(
                kueue_ref["namespace"],
                name="quota snapshot spec.kueueWorkloadRef.namespace",
                max_length=253,
            ),
            "name": _text(
                kueue_ref["name"],
                name="quota snapshot spec.kueueWorkloadRef.name",
                max_length=253,
            ),
            "uid": _text(
                kueue_ref["uid"],
                name="quota snapshot spec.kueueWorkloadRef.uid",
            ),
        }
        original_snapshot = original.snapshot
        original_kueue = original_snapshot.kueue
        if (
            tenant_id,
            model_class,
            model_family,
            gpus_per_replica,
        ) != (
            original_snapshot.tenant_id,
            original_snapshot.model_class,
            original_snapshot.model_family,
            original_snapshot.gpus_per_replica,
        ) or normalized_kueue_ref != {
            "apiVersion": original_kueue.api_version,
            "namespace": original_kueue.namespace,
            "name": original_kueue.workload_name,
            "uid": original_kueue.workload_uid,
        }:
            raise RunnerCachePlacementBindingAuthorizationDeniedError(
                "quota snapshot identity changed from the durable decision"
            )
        status = self._ready_status(root, metadata=metadata, name="quota snapshot")
        _strict_keys(
            status,
            {
                "observedGeneration",
                "conditions",
                "snapshotId",
                "quotaRevision",
                "observedAt",
                "kueueResourceVersion",
                "limits",
            },
            name="quota snapshot status",
        )
        kueue_resource_version = _text(
            status["kueueResourceVersion"],
            name="quota snapshot status.kueueResourceVersion",
        )
        if kueue_resource_version != kueue.resource_version:
            raise RunnerCachePlacementBindingAuthorizationDeniedError(
                "quota snapshot does not observe the current Kueue Workload"
            )
        limits_payload = status["limits"]
        if not isinstance(limits_payload, list):
            raise InvalidKubernetesPlacementBindingLiveResponseError(
                "quota snapshot status.limits must be a list"
            )
        limits: list[ScalingQuotaLimit] = []
        for item in limits_payload:
            limit = _mapping(item, name="quota snapshot limit")
            _strict_keys(
                limit,
                {
                    "scope",
                    "quotaName",
                    "hardLimitGpus",
                    "usedGpusExcludingTarget",
                    "reservedGpusForHigherPriority",
                },
                name="quota snapshot limit",
            )
            limits.append(
                ScalingQuotaLimit(
                    scope=limit["scope"],
                    quota_name=limit["quotaName"],
                    hard_limit_gpus=limit["hardLimitGpus"],
                    used_gpus_excluding_target=limit["usedGpusExcludingTarget"],
                    reserved_gpus_for_higher_priority=limit["reservedGpusForHigherPriority"],
                )
            )
        snapshot = ScalingQuotaSnapshot(
            snapshot_id=status["snapshotId"],
            quota_revision=status["quotaRevision"],
            observed_at=_aware(status["observedAt"], name="quota snapshot status.observedAt"),
            tenant_id=tenant_id,
            model_class=model_class,
            model_family=model_family,
            gpus_per_replica=gpus_per_replica,
            target_reserved_gpus=kueue.admitted_gpus if kueue.admitted else 0,
            limits=tuple(limits),
            kueue=kueue,
        )
        return admit_scaling_quota(
            snapshot,
            current_replicas=original.current_replicas,
            requested_replicas=original.requested_replicas,
        )

    def read_quota(
        self,
        binding: RunnerCacheStartupBinding,
        decision: ScalingDecisionRecord,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> ScalingQuotaAdmission:
        if not isinstance(binding, RunnerCacheStartupBinding):
            raise TypeError("binding must be a RunnerCacheStartupBinding")
        if not isinstance(decision, ScalingDecisionRecord):
            raise TypeError("decision must be a ScalingDecisionRecord")
        binding = RunnerCacheStartupBinding.model_validate(binding.model_dump())
        decision = ScalingDecisionRecord.model_validate(decision.model_dump())
        configured = self._target_config(binding, decision)
        original = decision.quota_admission
        target = decision.target_revision
        if original is None or target is None:
            raise RunnerCachePlacementBindingAuthorizationDeniedError(
                "binding decision has no quota authority"
            )
        original_kueue = original.snapshot.kueue
        expected_name = kueue_scaling_workload_name(
            target_kind=configured.target_kind,
            target_namespace=configured.namespace,
            target_name=configured.name,
            target_uid=target.workload_uid,
        )
        if (
            original_kueue.namespace != configured.kueue_namespace
            or original_kueue.api_version != configured.kueue_api_version
            or original_kueue.workload_name != expected_name
            or original_kueue.pod_set_name != configured.pod_set_name
            or original_kueue.resource_name != configured.resource_name
        ):
            raise RunnerCachePlacementBindingAuthorizationDeniedError(
                "durable Kueue identity does not match trusted configuration"
            )
        with self._operation(
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=backend_timeout_s,
        ) as operation_deadline:
            kueue_payload = self._request_json(
                self._kueue_url(
                    configured,
                    expected_name,
                    api_version=original_kueue.api_version,
                ),
                operation_deadline=operation_deadline,
                missing_is_denial=True,
            )
            try:
                kueue = parse_kueue_scaling_admission(
                    kueue_payload,
                    pod_set_name=configured.pod_set_name,
                    resource_name=configured.resource_name,
                )
            except ValueError as exc:
                raise InvalidKubernetesPlacementBindingLiveResponseError(
                    "Kueue Workload violates the scaling admission contract"
                ) from exc
            quota_payload = self._request_json(
                self._authority_url(
                    configured,
                    plural=_QUOTA_PLURAL,
                    name=configured.quota_snapshot_name,
                ),
                operation_deadline=operation_deadline,
                missing_is_denial=True,
            )
            return self._parse_quota_snapshot(
                quota_payload,
                configured=configured,
                decision=decision,
                kueue=kueue,
            )

    def _parse_inventory(
        self,
        payload: Any,
        *,
        configured: KubernetesPlacementBindingLiveTarget,
        binding: RunnerCacheStartupBinding,
        decision: ScalingDecisionRecord,
    ) -> RunnerCachePlacementBindingInventory:
        target = decision.target_revision
        assert target is not None
        root = self._root(payload, api_version=_AUTHORITY_API_VERSION, kind=_INVENTORY_KIND)
        metadata = self._metadata(root, name="placement inventory")
        if (metadata.namespace, metadata.name) != (
            configured.authority_namespace,
            configured.inventory_name,
        ):
            raise InvalidKubernetesPlacementBindingLiveResponseError(
                "placement inventory identity does not match configured authority"
            )
        spec = _mapping(root.get("spec"), name="placement inventory spec")
        _strict_keys(
            spec,
            {
                "targetRef",
                "bindingId",
                "decisionId",
                "decisionFingerprint",
                "modelClass",
                "modelId",
                "modelRevision",
                "manifestDigest",
                "placementBindingId",
            },
            name="placement inventory spec",
        )
        self._target_reference(
            spec["targetRef"],
            configured=configured,
            workload_uid=target.workload_uid,
            name="placement inventory spec.targetRef",
        )
        normalized_identity = (
            _text(spec["bindingId"], name="placement inventory spec.bindingId"),
            _text(spec["decisionId"], name="placement inventory spec.decisionId"),
            _text(
                spec["decisionFingerprint"],
                name="placement inventory spec.decisionFingerprint",
            ),
            _text(
                spec["modelClass"],
                name="placement inventory spec.modelClass",
                max_length=128,
            ),
            _text(spec["modelId"], name="placement inventory spec.modelId"),
            _text(spec["modelRevision"], name="placement inventory spec.modelRevision"),
            _text(spec["manifestDigest"], name="placement inventory spec.manifestDigest"),
            _text(
                spec["placementBindingId"],
                name="placement inventory spec.placementBindingId",
            ),
        )
        if (*normalized_identity,) != (
            binding.binding_id,
            decision.decision_id,
            decision.fingerprint,
            binding.model_class,
            binding.model_id,
            binding.model_revision,
            binding.manifest_digest,
            binding.placement_binding_id,
        ):
            raise RunnerCachePlacementBindingAuthorizationDeniedError(
                "placement inventory does not bind the current decision and artifact"
            )
        status = self._ready_status(root, metadata=metadata, name="placement inventory")
        _strict_keys(
            status,
            {
                "observedGeneration",
                "conditions",
                "snapshotId",
                "cacheRevision",
                "observedAt",
                "candidates",
            },
            name="placement inventory status",
        )
        candidates_payload = status["candidates"]
        if not isinstance(candidates_payload, list):
            raise InvalidKubernetesPlacementBindingLiveResponseError(
                "placement inventory status.candidates must be a list"
            )
        candidates: list[ModelCachePlacementCandidate] = []
        for item in candidates_payload:
            candidate = _mapping(item, name="placement inventory candidate")
            _strict_keys(
                candidate,
                {
                    "placementId",
                    "nodeName",
                    "resourceFlavor",
                    "profileId",
                    "compatibilityApprovalId",
                    "assigned",
                    "healthy",
                    "schedulable",
                },
                name="placement inventory candidate",
            )
            candidates.append(
                ModelCachePlacementCandidate(
                    placement_id=candidate["placementId"],
                    node_name=candidate["nodeName"],
                    resource_flavor=candidate["resourceFlavor"],
                    profile_id=candidate["profileId"],
                    compatibility_approval_id=candidate["compatibilityApprovalId"],
                    assigned=candidate["assigned"],
                    healthy=candidate["healthy"],
                    schedulable=candidate["schedulable"],
                )
            )
        return RunnerCachePlacementBindingInventory(
            snapshot_id=status["snapshotId"],
            cache_revision=status["cacheRevision"],
            observed_at=_aware(
                status["observedAt"],
                name="placement inventory status.observedAt",
            ),
            candidates=tuple(candidates),
        )

    def read_inventory(
        self,
        binding: RunnerCacheStartupBinding,
        decision: ScalingDecisionRecord,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> RunnerCachePlacementBindingInventory:
        if not isinstance(binding, RunnerCacheStartupBinding):
            raise TypeError("binding must be a RunnerCacheStartupBinding")
        if not isinstance(decision, ScalingDecisionRecord):
            raise TypeError("decision must be a ScalingDecisionRecord")
        binding = RunnerCacheStartupBinding.model_validate(binding.model_dump())
        decision = ScalingDecisionRecord.model_validate(decision.model_dump())
        configured = self._target_config(binding, decision)
        with self._operation(
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=backend_timeout_s,
        ) as operation_deadline:
            payload = self._request_json(
                self._authority_url(
                    configured,
                    plural=_INVENTORY_PLURAL,
                    name=configured.inventory_name,
                ),
                operation_deadline=operation_deadline,
                missing_is_denial=True,
            )
            return self._parse_inventory(
                payload,
                configured=configured,
                binding=binding,
                decision=decision,
            )

    def readiness(
        self,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> None:
        """Verify target and authority-resource access under one budget."""

        with self._operation(
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=backend_timeout_s,
        ) as operation_deadline:
            checked_kueue_namespaces: set[tuple[str, str]] = set()
            for configured in self._targets.values():
                target = self._request_json(
                    self._target_url(configured),
                    operation_deadline=operation_deadline,
                    missing_is_denial=False,
                )
                self._root(target, api_version="apps/v1", kind=configured.target_kind)
                quota = self._request_json(
                    self._authority_url(
                        configured,
                        plural=_QUOTA_PLURAL,
                        name=configured.quota_snapshot_name,
                    ),
                    operation_deadline=operation_deadline,
                    missing_is_denial=False,
                )
                self._root(quota, api_version=_AUTHORITY_API_VERSION, kind=_QUOTA_KIND)
                inventory = self._request_json(
                    self._authority_url(
                        configured,
                        plural=_INVENTORY_PLURAL,
                        name=configured.inventory_name,
                    ),
                    operation_deadline=operation_deadline,
                    missing_is_denial=False,
                )
                self._root(
                    inventory,
                    api_version=_AUTHORITY_API_VERSION,
                    kind=_INVENTORY_KIND,
                )
                kueue_route = (
                    configured.kueue_api_version,
                    configured.kueue_namespace,
                )
                if kueue_route not in checked_kueue_namespaces:
                    checked_kueue_namespaces.add(kueue_route)
                    version = configured.kueue_api_version.rsplit("/", 1)[1]
                    url = (
                        f"{self._api_server}/apis/kueue.x-k8s.io/{version}/namespaces/"
                        f"{quote(configured.kueue_namespace, safe='')}/workloads"
                    )
                    listing = self._request_json(
                        url,
                        operation_deadline=operation_deadline,
                        missing_is_denial=False,
                        params={"limit": "1"},
                    )
                    root = self._root(
                        listing,
                        api_version=configured.kueue_api_version,
                        kind="WorkloadList",
                    )
                    if not isinstance(root.get("items"), list):
                        raise InvalidKubernetesPlacementBindingLiveResponseError(
                            "Kueue WorkloadList response requires an items list"
                        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            inflight = tuple(self._inflight)
        for request in inflight:
            request.cancel()
        for request in inflight:
            try:
                request.result()
            except BaseException:
                pass
        try:
            future = asyncio.run_coroutine_threadsafe(self._client.aclose(), self._loop)
            future.result()
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join()
            self._loop.close()

    def __enter__(self) -> KubernetesKueueRunnerCachePlacementBindingReader:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


__all__ = [
    "InvalidKubernetesPlacementBindingLiveResponseError",
    "KubernetesKueueRunnerCachePlacementBindingReader",
    "KubernetesPlacementBindingLiveTarget",
]
