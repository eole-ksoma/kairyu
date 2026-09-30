"""Leader-fenced Kubernetes publishers for placement-binding authority CRDs."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.runners.leadership import RunnerWriterAuthority
from kairyu.runners.scale_actuator import (
    SCALE_DECISION_GENERATION_ANNOTATION,
    SCALE_ELECTION_ID_ANNOTATION,
    SCALE_FENCING_TOKEN_ANNOTATION,
)
from kairyu.runners.scaling_quota import ScalingQuotaSnapshot
from kairyu.runners.startup_binding_live_cache import (
    RunnerCachePlacementBindingInventory,
)
from kairyu.runners.startup_binding_live_kubernetes import (
    KubernetesPlacementBindingLiveTarget,
)

AUTHORITY_API_VERSION = "autoscaling.kairyu.ai/v1alpha1"
QUOTA_SNAPSHOT_KIND = "RunnerScalingQuotaSnapshot"
QUOTA_SNAPSHOT_PLURAL = "runnerscalingquotasnapshots"
PLACEMENT_INVENTORY_KIND = "RunnerCachePlacementInventory"
PLACEMENT_INVENTORY_PLURAL = "runnercacheplacementinventories"
AUTHORITY_HOLDER_ID_ANNOTATION = "kairyu.ai/authority-holder-id"
AUTHORITY_SOURCE_REVISION_ANNOTATION = "kairyu.ai/authority-source-revision"
AUTHORITY_SOURCE_DIGEST_ANNOTATION = "kairyu.ai/authority-source-digest"
_MAX_SIGNED_BIGINT = 2**63 - 1


class KubernetesAuthorityReconcileConflictError(RuntimeError):
    """A concurrent or newer authority publication prevents this mutation."""


class InvalidKubernetesAuthorityReconcileResponseError(RuntimeError):
    """The Kubernetes API response violated the reconciler contract."""


def _text(value: str, *, name: str, max_length: int = 255) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
        or len(value) > max_length
    ):
        raise ValueError(f"{name} must be a bounded non-empty string without NUL")
    return value


def _aware(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _timestamp(value: datetime) -> str:
    rendered = value.isoformat()
    return rendered[:-6] + "Z" if rendered.endswith("+00:00") else rendered


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


class RunnerScalingQuotaSnapshotPublication(BaseModel):
    """Controller-owned quota state to publish for one immutable target."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-scaling-quota-snapshot-publication-v1"] = (
        "runner-scaling-quota-snapshot-publication-v1"
    )
    snapshot: ScalingQuotaSnapshot

    @model_validator(mode="after")
    def validate_snapshot(self) -> RunnerScalingQuotaSnapshotPublication:
        ScalingQuotaSnapshot.model_validate(self.snapshot.model_dump())
        return self


class RunnerCachePlacementInventoryPublication(BaseModel):
    """Decision-bound scheduler inventory to publish for live authorization."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-cache-placement-inventory-publication-v1"] = (
        "runner-cache-placement-inventory-publication-v1"
    )
    model_class: str = Field(max_length=128)
    target_uid: str = Field(max_length=255)
    binding_id: str = Field(min_length=64, max_length=64)
    decision_id: str = Field(max_length=255)
    decision_generation: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    decision_fingerprint: str = Field(min_length=64, max_length=64)
    model_id: str = Field(max_length=255)
    model_revision: str = Field(max_length=255)
    manifest_digest: str = Field(min_length=64, max_length=64)
    placement_binding_id: str = Field(max_length=255)
    inventory: RunnerCachePlacementBindingInventory

    @field_validator(
        "model_class",
        "target_uid",
        "binding_id",
        "decision_id",
        "decision_fingerprint",
        "model_id",
        "model_revision",
        "manifest_digest",
        "placement_binding_id",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        maximum = 128 if info.field_name == "model_class" else 255
        return _text(value, name=info.field_name, max_length=maximum)

    @field_validator("decision_generation", mode="before")
    @classmethod
    def validate_generation(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("decision_generation must be an integer")
        return value

    @model_validator(mode="after")
    def validate_inventory(self) -> RunnerCachePlacementInventoryPublication:
        RunnerCachePlacementBindingInventory.model_validate(self.inventory.model_dump())
        return self


class KubernetesAuthorityReconcileResult(BaseModel):
    """Auditable outcome of one CRD spec/status reconciliation."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-kubernetes-authority-reconcile-result-v1"] = (
        "runner-kubernetes-authority-reconcile-result-v1"
    )
    api_version: Literal["autoscaling.kairyu.ai/v1alpha1"] = AUTHORITY_API_VERSION
    kind: Literal["RunnerScalingQuotaSnapshot", "RunnerCachePlacementInventory"]
    namespace: str = Field(max_length=253)
    name: str = Field(max_length=253)
    uid: str = Field(max_length=255)
    generation: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    resource_version: str = Field(max_length=255)
    spec_applied: bool
    status_applied: bool

    @field_validator("namespace", "name", "uid", "resource_version")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        maximum = 253 if info.field_name in {"namespace", "name"} else 255
        return _text(value, name=info.field_name, max_length=maximum)

    @field_validator("generation", mode="before")
    @classmethod
    def validate_generation(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("generation must be an integer")
        return value

    @field_validator("spec_applied", "status_applied", mode="before")
    @classmethod
    def validate_boolean(cls, value: object, info) -> object:
        if type(value) is not bool:
            raise ValueError(f"{info.field_name} must be a boolean")
        return value


@dataclass(frozen=True)
class _ObservedResource:
    uid: str
    generation: int
    resource_version: str
    spec: dict[str, Any]
    status: dict[str, Any] | None
    annotations: dict[str, str] | None


@dataclass(frozen=True)
class _DesiredResource:
    kind: str
    plural: str
    namespace: str
    name: str
    spec: dict[str, Any]
    status_without_generation: dict[str, Any]
    revision_field: str
    revision: int
    decision_generation: int | None


class KubernetesPlacementBindingAuthorityReconciler:
    """Publish reader-compatible CRDs through optimistic, leader-fenced CAS."""

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
    ) -> None:
        if not isinstance(targets, tuple) or not targets:
            raise TypeError("targets must be a non-empty tuple")
        if len(targets) > 1024:
            raise ValueError("targets exceed the supported model-class count")
        validated: list[KubernetesPlacementBindingLiveTarget] = []
        for target in targets:
            if not isinstance(target, KubernetesPlacementBindingLiveTarget):
                raise TypeError("targets must contain KubernetesPlacementBindingLiveTarget values")
            validated.append(
                KubernetesPlacementBindingLiveTarget.model_validate(target.model_dump())
            )
        model_classes = tuple(target.model_class for target in validated)
        if model_classes != tuple(sorted(model_classes)) or len(set(model_classes)) != len(
            model_classes
        ):
            raise ValueError("targets must use unique canonical model classes")
        if (
            type(response_limit_bytes) is not int
            or not 1 <= response_limit_bytes <= 16 * 1024 * 1024
        ):
            raise ValueError("response_limit_bytes must be an integer in [1, 16777216]")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")

        host = os.environ.get("KUBERNETES_SERVICE_HOST")
        port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        if api_server is None:
            api_host = f"[{host}]" if host and ":" in host else host
            api_server = (
                "https://kubernetes.default.svc" if not api_host else f"https://{api_host}:{port}"
            )
        api_server = _text(api_server, name="api_server", max_length=2048)
        if any(character.isspace() or ord(character) < 0x20 for character in api_server):
            raise ValueError("api_server must not contain whitespace or controls")
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
                raise ValueError("Kubernetes authority reconciler must own its client")
            client = httpx.AsyncClient(
                verify=str(resolved_ca),
                follow_redirects=False,
                trust_env=False,
            )
        else:
            if close_client is False:
                raise ValueError("Kubernetes authority reconciler must adopt its client")
            if client.follow_redirects:
                raise ValueError("Kubernetes authority reconciler must not follow redirects")
            if client.trust_env:
                raise ValueError("Kubernetes authority reconciler must disable environment trust")
        self._targets = {target.model_class: target for target in validated}
        self._api_server = api_server.rstrip("/")
        self._token_path = Path(token_path or self._SERVICE_ACCOUNT_DIR / "token")
        self._client = client
        self._response_limit_bytes = response_limit_bytes
        self._monotonic_clock = monotonic_clock
        self._lock = threading.RLock()
        self._closed = False
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            name="kairyu-kubernetes-authority-reconciler",
            daemon=True,
        )
        self._inflight: set[concurrent.futures.Future[Any]] = set()
        try:
            self._loop_thread.start()
        except BaseException:
            try:
                self._loop.run_until_complete(self._client.aclose())
            finally:
                self._loop.close()
            raise

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
            raise TimeoutError("Kubernetes authority reconciliation deadline expired")
        return remaining

    @contextmanager
    def _operation(self, *, deadline_monotonic: float, backend_timeout_s: float):
        self._validate_budget(deadline_monotonic, backend_timeout_s)
        started_at = self._monotonic_clock()
        operation_deadline = min(deadline_monotonic, started_at + backend_timeout_s)
        remaining = operation_deadline - started_at
        if remaining <= 0:
            raise TimeoutError("Kubernetes authority reconciliation deadline expired")
        if not self._lock.acquire(timeout=remaining):
            raise TimeoutError("Kubernetes authority reconciler lock timed out")
        try:
            if self._closed:
                raise RuntimeError("Kubernetes authority reconciler is closed")
            yield operation_deadline
            self._remaining(operation_deadline)
        finally:
            self._lock.release()

    def _headers(self, *, operation_deadline: float) -> dict[str, str]:
        self._remaining(operation_deadline)
        descriptor = os.open(self._token_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
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
        return {"Accept": "application/json", "Authorization": f"Bearer {token}"}

    async def _request_json_async(
        self,
        method: str,
        url: str,
        *,
        operation_deadline: float,
        body: object | None = None,
        content_type: str | None = None,
        allow_missing: bool = False,
    ) -> Any | None:
        headers = self._headers(operation_deadline=operation_deadline)
        if content_type is not None:
            headers["Content-Type"] = content_type
        encoded = None
        if body is not None:
            encoded = json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            if len(encoded) > self._response_limit_bytes:
                raise ValueError("Kubernetes authority request exceeds the configured limit")
        timeout = self._remaining(operation_deadline)
        try:
            async with asyncio.timeout(timeout):
                async with self._client.stream(
                    method,
                    url,
                    content=encoded,
                    headers=headers,
                    timeout=httpx.Timeout(timeout),
                ) as response:
                    if response.status_code == 404 and allow_missing:
                        return None
                    if response.status_code in {409, 422}:
                        raise KubernetesAuthorityReconcileConflictError(
                            "Kubernetes authority resource changed concurrently"
                        )
                    response.raise_for_status()
                    content_type_value = response.headers.get("content-type", "")
                    if content_type_value.split(";", 1)[0].strip().lower() != "application/json":
                        raise InvalidKubernetesAuthorityReconcileResponseError(
                            "Kubernetes authority response must use application/json"
                        )
                    raw_length = response.headers.get("content-length")
                    if raw_length is not None:
                        try:
                            content_length = int(raw_length)
                        except ValueError as exc:
                            raise InvalidKubernetesAuthorityReconcileResponseError(
                                "Kubernetes authority response has invalid content-length"
                            ) from exc
                        if content_length < 0 or content_length > self._response_limit_bytes:
                            raise InvalidKubernetesAuthorityReconcileResponseError(
                                "Kubernetes authority response exceeds the configured limit"
                            )
                    chunks: list[bytes] = []
                    received = 0
                    async for chunk in response.aiter_bytes():
                        self._remaining(operation_deadline)
                        received += len(chunk)
                        if received > self._response_limit_bytes:
                            raise InvalidKubernetesAuthorityReconcileResponseError(
                                "Kubernetes authority response exceeds the configured limit"
                            )
                        chunks.append(chunk)
        except httpx.TimeoutException as exc:
            raise TimeoutError("Kubernetes authority reconciliation request timed out") from exc
        self._remaining(operation_deadline)
        try:
            return json.loads(
                b"".join(chunks),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
                parse_float=_finite_float,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise InvalidKubernetesAuthorityReconcileResponseError(
                "Kubernetes authority response must be strict JSON"
            ) from exc

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        operation_deadline: float,
        body: object | None = None,
        content_type: str | None = None,
        allow_missing: bool = False,
    ) -> Any | None:
        coroutine = self._request_json_async(
            method,
            url,
            operation_deadline=operation_deadline,
            body=body,
            content_type=content_type,
            allow_missing=allow_missing,
        )
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        self._inflight.add(future)
        try:
            return future.result()
        finally:
            self._inflight.discard(future)

    def _collection_url(self, desired: _DesiredResource) -> str:
        return (
            f"{self._api_server}/apis/{AUTHORITY_API_VERSION}/namespaces/"
            f"{quote(desired.namespace, safe='')}/{desired.plural}"
        )

    def _resource_url(self, desired: _DesiredResource) -> str:
        return f"{self._collection_url(desired)}/{quote(desired.name, safe='')}"

    @staticmethod
    def _parse_resource(payload: Any, desired: _DesiredResource) -> _ObservedResource:
        if not isinstance(payload, dict):
            raise InvalidKubernetesAuthorityReconcileResponseError(
                "Kubernetes authority resource must be an object"
            )
        if (
            payload.get("apiVersion") != AUTHORITY_API_VERSION
            or payload.get("kind") != desired.kind
        ):
            raise InvalidKubernetesAuthorityReconcileResponseError(
                "Kubernetes authority resource has an unexpected apiVersion or kind"
            )
        metadata = payload.get("metadata")
        spec = payload.get("spec")
        status = payload.get("status")
        if not isinstance(metadata, dict) or not isinstance(spec, dict):
            raise InvalidKubernetesAuthorityReconcileResponseError(
                "Kubernetes authority metadata and spec must be objects"
            )
        if status is not None and not isinstance(status, dict):
            raise InvalidKubernetesAuthorityReconcileResponseError(
                "Kubernetes authority status must be an object"
            )
        if (metadata.get("namespace"), metadata.get("name")) != (
            desired.namespace,
            desired.name,
        ):
            raise InvalidKubernetesAuthorityReconcileResponseError(
                "Kubernetes authority response identity does not match trusted routing"
            )
        uid = metadata.get("uid")
        generation = metadata.get("generation")
        resource_version = metadata.get("resourceVersion")
        if not isinstance(uid, str) or not uid:
            raise InvalidKubernetesAuthorityReconcileResponseError(
                "Kubernetes authority response requires a UID"
            )
        if type(generation) is not int or not 1 <= generation <= _MAX_SIGNED_BIGINT:
            raise InvalidKubernetesAuthorityReconcileResponseError(
                "Kubernetes authority response requires a positive generation"
            )
        if not isinstance(resource_version, str) or not resource_version:
            raise InvalidKubernetesAuthorityReconcileResponseError(
                "Kubernetes authority response requires a resourceVersion"
            )
        annotations_payload = metadata.get("annotations")
        annotations: dict[str, str] | None
        if annotations_payload is None:
            annotations = None
        elif not isinstance(annotations_payload, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in annotations_payload.items()
        ):
            raise InvalidKubernetesAuthorityReconcileResponseError(
                "Kubernetes authority annotations must contain string pairs"
            )
        else:
            annotations = dict(annotations_payload)
        return _ObservedResource(
            uid=uid,
            generation=generation,
            resource_version=resource_version,
            spec=dict(spec),
            status=None if status is None else dict(status),
            annotations=annotations,
        )

    @staticmethod
    def _authority_annotations(
        authority: RunnerWriterAuthority,
        *,
        desired: _DesiredResource,
        decision_generation: int | None,
    ) -> dict[str, str]:
        source_digest = hashlib.sha256(
            json.dumps(
                desired.status_without_generation,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        annotations = {
            SCALE_ELECTION_ID_ANNOTATION: authority.election_id,
            AUTHORITY_HOLDER_ID_ANNOTATION: authority.holder_id,
            SCALE_FENCING_TOKEN_ANNOTATION: str(authority.fencing_token),
            AUTHORITY_SOURCE_REVISION_ANNOTATION: str(desired.revision),
            AUTHORITY_SOURCE_DIGEST_ANNOTATION: source_digest,
        }
        if decision_generation is not None:
            annotations[SCALE_DECISION_GENERATION_ANNOTATION] = str(decision_generation)
        return annotations

    @staticmethod
    def _validate_existing_fence(
        observed: _ObservedResource,
        authority: RunnerWriterAuthority,
        *,
        desired: _DesiredResource,
        spec_changes: bool,
    ) -> None:
        annotations = observed.annotations
        if annotations is None:
            raise KubernetesAuthorityReconcileConflictError(
                "existing Kubernetes authority resource is not controller-owned"
            )
        election_id = annotations.get(SCALE_ELECTION_ID_ANNOTATION)
        holder_id = annotations.get(AUTHORITY_HOLDER_ID_ANNOTATION)
        token_text = annotations.get(SCALE_FENCING_TOKEN_ANNOTATION)
        if election_id != authority.election_id or holder_id is None or token_text is None:
            raise KubernetesAuthorityReconcileConflictError(
                "existing Kubernetes authority resource has an incompatible leader fence"
            )
        try:
            token = int(token_text)
        except ValueError as exc:
            raise KubernetesAuthorityReconcileConflictError(
                "existing Kubernetes authority resource has a malformed leader fence"
            ) from exc
        if str(token) != token_text or not 1 <= token <= _MAX_SIGNED_BIGINT:
            raise KubernetesAuthorityReconcileConflictError(
                "existing Kubernetes authority resource has a malformed leader fence"
            )
        if token > authority.fencing_token or (
            token == authority.fencing_token and holder_id != authority.holder_id
        ):
            raise KubernetesAuthorityReconcileConflictError(
                "a newer Kubernetes authority writer already owns the resource"
            )
        source_revision_text = annotations.get(AUTHORITY_SOURCE_REVISION_ANNOTATION)
        source_digest = annotations.get(AUTHORITY_SOURCE_DIGEST_ANNOTATION)
        if source_revision_text is None or source_digest is None:
            raise KubernetesAuthorityReconcileConflictError(
                "existing Kubernetes authority resource lacks a source fence"
            )
        try:
            source_revision = int(source_revision_text)
        except ValueError as exc:
            raise KubernetesAuthorityReconcileConflictError(
                "existing Kubernetes authority resource has a malformed source revision"
            ) from exc
        if (
            str(source_revision) != source_revision_text
            or not 1 <= source_revision <= _MAX_SIGNED_BIGINT
        ):
            raise KubernetesAuthorityReconcileConflictError(
                "existing Kubernetes authority resource has a malformed source revision"
            )
        expected_source_digest = hashlib.sha256(
            json.dumps(
                desired.status_without_generation,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if source_revision > desired.revision:
            raise KubernetesAuthorityReconcileConflictError(
                "Kubernetes authority source revision would regress"
            )
        if source_revision == desired.revision and source_digest != expected_source_digest:
            raise KubernetesAuthorityReconcileConflictError(
                "Kubernetes authority source revision was reused for different content"
            )
        decision_generation = desired.decision_generation
        if decision_generation is not None:
            current_text = annotations.get(SCALE_DECISION_GENERATION_ANNOTATION)
            if current_text is None:
                raise KubernetesAuthorityReconcileConflictError(
                    "existing placement inventory lacks a decision generation"
                )
            try:
                current = int(current_text)
            except ValueError as exc:
                raise KubernetesAuthorityReconcileConflictError(
                    "existing placement inventory has a malformed decision generation"
                ) from exc
            if str(current) != current_text or not 1 <= current <= _MAX_SIGNED_BIGINT:
                raise KubernetesAuthorityReconcileConflictError(
                    "existing placement inventory has a malformed decision generation"
                )
            if current > decision_generation or (spec_changes and current >= decision_generation):
                raise KubernetesAuthorityReconcileConflictError(
                    "placement inventory decision generation did not advance"
                )
            if not spec_changes and current != decision_generation:
                raise KubernetesAuthorityReconcileConflictError(
                    "placement inventory decision generation changed without its spec"
                )

    @staticmethod
    def _validate_written_fence(
        observed: _ObservedResource,
        expected: dict[str, str],
    ) -> None:
        annotations = observed.annotations or {}
        if any(annotations.get(key) != value for key, value in expected.items()):
            raise InvalidKubernetesAuthorityReconcileResponseError(
                "Kubernetes authority write did not persist the requested leader fence"
            )

    @staticmethod
    def _reauthorize(
        authority: RunnerWriterAuthority,
        callback: Callable[[], RunnerWriterAuthority],
    ) -> None:
        if not callable(callback):
            raise TypeError("reauthorize must be callable")
        refreshed = callback()
        if not isinstance(refreshed, RunnerWriterAuthority):
            raise TypeError("reauthorize must return RunnerWriterAuthority")
        refreshed = RunnerWriterAuthority.model_validate(refreshed.model_dump())
        if refreshed.tenure != authority.tenure:
            raise KubernetesAuthorityReconcileConflictError(
                "leader authority changed during Kubernetes authority reconciliation"
            )

    @staticmethod
    def _target_reference(target: KubernetesPlacementBindingLiveTarget, uid: str) -> dict[str, Any]:
        return {
            "apiVersion": "apps/v1",
            "kind": target.target_kind,
            "namespace": target.namespace,
            "name": target.name,
            "uid": uid,
        }

    def _quota_desired(
        self,
        publication: RunnerScalingQuotaSnapshotPublication,
    ) -> _DesiredResource:
        publication = RunnerScalingQuotaSnapshotPublication.model_validate(publication.model_dump())
        snapshot = publication.snapshot
        target = self._targets.get(snapshot.model_class)
        if target is None:
            raise ValueError("quota snapshot model_class has no trusted Kubernetes target")
        kueue = snapshot.kueue
        if (
            (kueue.target_kind, kueue.target_namespace, kueue.target_name)
            != (target.target_kind, target.namespace, target.name)
            or kueue.namespace != target.kueue_namespace
            or kueue.api_version != target.kueue_api_version
            or kueue.pod_set_name != target.pod_set_name
            or kueue.resource_name != target.resource_name
        ):
            raise ValueError("quota snapshot does not match trusted Kubernetes/Kueue routing")
        spec = {
            "targetRef": self._target_reference(target, kueue.target_uid),
            "tenantId": snapshot.tenant_id,
            "modelClass": snapshot.model_class,
            "modelFamily": snapshot.model_family,
            "gpusPerReplica": snapshot.gpus_per_replica,
            "kueueWorkloadRef": {
                "apiVersion": kueue.api_version,
                "namespace": kueue.namespace,
                "name": kueue.workload_name,
                "uid": kueue.workload_uid,
            },
        }
        status = {
            "snapshotId": snapshot.snapshot_id,
            "quotaRevision": snapshot.quota_revision,
            "observedAt": _timestamp(snapshot.observed_at),
            "kueueResourceVersion": kueue.resource_version,
            "limits": [
                {
                    "scope": limit.scope.value,
                    "quotaName": limit.quota_name,
                    "hardLimitGpus": limit.hard_limit_gpus,
                    "usedGpusExcludingTarget": limit.used_gpus_excluding_target,
                    "reservedGpusForHigherPriority": limit.reserved_gpus_for_higher_priority,
                }
                for limit in snapshot.limits
            ],
        }
        return _DesiredResource(
            kind=QUOTA_SNAPSHOT_KIND,
            plural=QUOTA_SNAPSHOT_PLURAL,
            namespace=target.authority_namespace,
            name=target.quota_snapshot_name,
            spec=spec,
            status_without_generation=status,
            revision_field="quotaRevision",
            revision=snapshot.quota_revision,
            decision_generation=None,
        )

    def _inventory_desired(
        self,
        publication: RunnerCachePlacementInventoryPublication,
    ) -> _DesiredResource:
        publication = RunnerCachePlacementInventoryPublication.model_validate(
            publication.model_dump()
        )
        target = self._targets.get(publication.model_class)
        if target is None:
            raise ValueError("placement inventory model_class has no trusted Kubernetes target")
        inventory = publication.inventory
        spec = {
            "targetRef": self._target_reference(target, publication.target_uid),
            "bindingId": publication.binding_id,
            "decisionId": publication.decision_id,
            "decisionFingerprint": publication.decision_fingerprint,
            "modelClass": publication.model_class,
            "modelId": publication.model_id,
            "modelRevision": publication.model_revision,
            "manifestDigest": publication.manifest_digest,
            "placementBindingId": publication.placement_binding_id,
        }
        status = {
            "snapshotId": inventory.snapshot_id,
            "cacheRevision": inventory.cache_revision,
            "observedAt": _timestamp(inventory.observed_at),
            "candidates": [
                {
                    "placementId": candidate.placement_id,
                    "nodeName": candidate.node_name,
                    "resourceFlavor": candidate.resource_flavor,
                    "profileId": candidate.profile_id,
                    "compatibilityApprovalId": candidate.compatibility_approval_id,
                    "assigned": candidate.assigned,
                    "healthy": candidate.healthy,
                    "schedulable": candidate.schedulable,
                }
                for candidate in inventory.candidates
            ],
        }
        return _DesiredResource(
            kind=PLACEMENT_INVENTORY_KIND,
            plural=PLACEMENT_INVENTORY_PLURAL,
            namespace=target.authority_namespace,
            name=target.inventory_name,
            spec=spec,
            status_without_generation=status,
            revision_field="cacheRevision",
            revision=inventory.cache_revision,
            decision_generation=publication.decision_generation,
        )

    @staticmethod
    def _desired_status(desired: _DesiredResource, generation: int) -> dict[str, Any]:
        return {
            "observedGeneration": generation,
            "conditions": [
                {
                    "type": "Ready",
                    "status": "True",
                    "observedGeneration": generation,
                }
            ],
            **desired.status_without_generation,
        }

    @staticmethod
    def _validate_revision(
        observed: _ObservedResource,
        desired: _DesiredResource,
        expected_status: dict[str, Any],
    ) -> None:
        if observed.status is None or not observed.status:
            return
        current = observed.status.get(desired.revision_field)
        if type(current) is not int or not 1 <= current <= _MAX_SIGNED_BIGINT:
            raise KubernetesAuthorityReconcileConflictError(
                "existing Kubernetes authority status has an invalid source revision"
            )
        if current > desired.revision:
            raise KubernetesAuthorityReconcileConflictError(
                "Kubernetes authority source revision would regress"
            )
        if current == desired.revision:
            current_payload = {
                key: value
                for key, value in observed.status.items()
                if key not in {"observedGeneration", "conditions"}
            }
            expected_payload = {
                key: value
                for key, value in expected_status.items()
                if key not in {"observedGeneration", "conditions"}
            }
            if current_payload != expected_payload:
                raise KubernetesAuthorityReconcileConflictError(
                    "Kubernetes authority source revision was reused for different content"
                )

    def _reconcile(
        self,
        desired: _DesiredResource,
        authority: RunnerWriterAuthority,
        *,
        reauthorize: Callable[[], RunnerWriterAuthority],
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> KubernetesAuthorityReconcileResult:
        if not isinstance(authority, RunnerWriterAuthority):
            raise TypeError("authority must be a RunnerWriterAuthority")
        authority = RunnerWriterAuthority.model_validate(authority.model_dump())
        with self._operation(
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=backend_timeout_s,
        ) as operation_deadline:
            resource_url = self._resource_url(desired)
            payload = self._request_json(
                "GET",
                resource_url,
                operation_deadline=operation_deadline,
                allow_missing=True,
            )
            spec_applied = False
            annotations = self._authority_annotations(
                authority,
                desired=desired,
                decision_generation=desired.decision_generation,
            )
            if payload is None:
                self._reauthorize(authority, reauthorize)
                created = self._request_json(
                    "POST",
                    self._collection_url(desired),
                    operation_deadline=operation_deadline,
                    body={
                        "apiVersion": AUTHORITY_API_VERSION,
                        "kind": desired.kind,
                        "metadata": {
                            "namespace": desired.namespace,
                            "name": desired.name,
                            "annotations": annotations,
                        },
                        "spec": desired.spec,
                    },
                    content_type="application/json",
                )
                observed = self._parse_resource(created, desired)
                spec_applied = True
            else:
                observed = self._parse_resource(payload, desired)
                spec_changes = observed.spec != desired.spec
                self._validate_existing_fence(
                    observed,
                    authority,
                    desired=desired,
                    spec_changes=spec_changes,
                )
                self._validate_revision(
                    observed,
                    desired,
                    self._desired_status(desired, observed.generation),
                )
                current_annotations = observed.annotations or {}
                fence_changes = any(
                    current_annotations.get(key) != value for key, value in annotations.items()
                )
                if spec_changes or fence_changes:
                    previous = observed
                    self._reauthorize(authority, reauthorize)
                    patch: list[dict[str, Any]] = [
                        {
                            "op": "test",
                            "path": "/metadata/resourceVersion",
                            "value": observed.resource_version,
                        }
                    ]
                    if observed.annotations is None:
                        patch.append(
                            {
                                "op": "add",
                                "path": "/metadata/annotations",
                                "value": annotations,
                            }
                        )
                    else:
                        for key, value in annotations.items():
                            escaped = key.replace("~", "~0").replace("/", "~1")
                            patch.append(
                                {
                                    "op": "add",
                                    "path": f"/metadata/annotations/{escaped}",
                                    "value": value,
                                }
                            )
                    if spec_changes:
                        patch.append({"op": "replace", "path": "/spec", "value": desired.spec})
                    updated = self._request_json(
                        "PATCH",
                        resource_url,
                        operation_deadline=operation_deadline,
                        body=patch,
                        content_type="application/json-patch+json",
                    )
                    observed = self._parse_resource(updated, desired)
                    if observed.uid != previous.uid:
                        raise InvalidKubernetesAuthorityReconcileResponseError(
                            "Kubernetes authority resource UID changed during update"
                        )
                    if observed.resource_version == previous.resource_version:
                        raise InvalidKubernetesAuthorityReconcileResponseError(
                            "Kubernetes authority update did not advance resourceVersion"
                        )
                    if spec_changes and observed.generation <= previous.generation:
                        raise InvalidKubernetesAuthorityReconcileResponseError(
                            "Kubernetes authority spec update did not advance generation"
                        )
                    if not spec_changes and observed.generation != previous.generation:
                        raise InvalidKubernetesAuthorityReconcileResponseError(
                            "Kubernetes authority metadata update changed generation"
                        )
                    spec_applied = True
            if observed.spec != desired.spec:
                raise InvalidKubernetesAuthorityReconcileResponseError(
                    "Kubernetes authority write did not persist the requested spec"
                )
            self._validate_written_fence(observed, annotations)
            expected_status = self._desired_status(desired, observed.generation)
            self._validate_revision(observed, desired, expected_status)
            status_applied = observed.status != expected_status
            if status_applied:
                previous = observed
                self._reauthorize(authority, reauthorize)
                status_payload = self._request_json(
                    "PUT",
                    f"{resource_url}/status",
                    operation_deadline=operation_deadline,
                    body={
                        "apiVersion": AUTHORITY_API_VERSION,
                        "kind": desired.kind,
                        "metadata": {
                            "namespace": desired.namespace,
                            "name": desired.name,
                            "resourceVersion": observed.resource_version,
                        },
                        "spec": desired.spec,
                        "status": expected_status,
                    },
                    content_type="application/json",
                )
                observed = self._parse_resource(status_payload, desired)
                if (
                    observed.uid != previous.uid
                    or observed.generation != previous.generation
                    or observed.resource_version == previous.resource_version
                    or observed.spec != desired.spec
                    or observed.status != expected_status
                ):
                    raise InvalidKubernetesAuthorityReconcileResponseError(
                        "Kubernetes authority status write did not persist the requested state"
                    )
                self._validate_written_fence(observed, annotations)
            elif not spec_applied:
                self._reauthorize(authority, reauthorize)
            return KubernetesAuthorityReconcileResult(
                kind=desired.kind,
                namespace=desired.namespace,
                name=desired.name,
                uid=observed.uid,
                generation=observed.generation,
                resource_version=observed.resource_version,
                spec_applied=spec_applied,
                status_applied=status_applied,
            )

    def reconcile_quota_snapshot(
        self,
        publication: RunnerScalingQuotaSnapshotPublication,
        authority: RunnerWriterAuthority,
        *,
        reauthorize: Callable[[], RunnerWriterAuthority],
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> KubernetesAuthorityReconcileResult:
        """Publish one monotonic quota snapshot to its trusted configured name."""

        if not isinstance(publication, RunnerScalingQuotaSnapshotPublication):
            raise TypeError("publication must be a RunnerScalingQuotaSnapshotPublication")
        return self._reconcile(
            self._quota_desired(publication),
            authority,
            reauthorize=reauthorize,
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=backend_timeout_s,
        )

    def reconcile_placement_inventory(
        self,
        publication: RunnerCachePlacementInventoryPublication,
        authority: RunnerWriterAuthority,
        *,
        reauthorize: Callable[[], RunnerWriterAuthority],
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> KubernetesAuthorityReconcileResult:
        """Publish one decision-bound placement inventory to trusted routing."""

        if not isinstance(publication, RunnerCachePlacementInventoryPublication):
            raise TypeError("publication must be a RunnerCachePlacementInventoryPublication")
        return self._reconcile(
            self._inventory_desired(publication),
            authority,
            reauthorize=reauthorize,
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=backend_timeout_s,
        )

    def close(self) -> None:
        """Close the adopted HTTP client after in-flight reconciliation completes."""

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

    def __enter__(self) -> KubernetesPlacementBindingAuthorityReconciler:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
