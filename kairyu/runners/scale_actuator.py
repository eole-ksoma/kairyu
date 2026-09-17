"""Idempotent Kubernetes scale-subresource actuator for Runner workloads."""

from __future__ import annotations

import math
import os
import threading
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.runners.scaling_log import ScalingDecisionAction, ScalingDecisionRecord


class KubernetesScaleConflictError(RuntimeError):
    """The workload changed concurrently with a scale-subresource write."""


class InvalidKubernetesScaleResponseError(RuntimeError):
    """The Kubernetes Scale response violated the actuator contract."""


def _identity(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL")
    return value


class KubernetesScalableKind(StrEnum):
    """Workload kinds with the stable apps/v1 Scale subresource."""

    DEPLOYMENT = "Deployment"
    STATEFUL_SET = "StatefulSet"

    @property
    def plural(self) -> str:
        if self is KubernetesScalableKind.DEPLOYMENT:
            return "deployments"
        return "statefulsets"


class KubernetesScaleTarget(BaseModel):
    """One model class bound to one namespaced scalable workload."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-kubernetes-scale-target-v1"] = (
        "runner-kubernetes-scale-target-v1"
    )
    model_class: str = Field(max_length=128)
    namespace: str = Field(max_length=253)
    name: str = Field(max_length=253)
    kind: KubernetesScalableKind

    @field_validator("model_class", "namespace", "name")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identity(value, name=info.field_name)


class KubernetesScaleResult(BaseModel):
    """Auditable outcome of applying one scaling decision."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-kubernetes-scale-result-v1"] = (
        "runner-kubernetes-scale-result-v1"
    )
    decision_id: str = Field(max_length=255)
    model_class: str = Field(max_length=128)
    target: KubernetesScaleTarget
    action: ScalingDecisionAction
    previous_replicas: int = Field(ge=0, le=100_000)
    requested_replicas: int = Field(ge=0, le=100_000)
    resulting_replicas: int = Field(ge=0, le=100_000)
    applied: bool
    resource_version_before: str = Field(max_length=255)
    resource_version_after: str = Field(max_length=255)

    @field_validator(
        "previous_replicas",
        "requested_replicas",
        "resulting_replicas",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        return value

    @field_validator("applied", mode="before")
    @classmethod
    def validate_boolean(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("applied must be a boolean")
        return value

    @field_validator(
        "decision_id",
        "model_class",
        "resource_version_before",
        "resource_version_after",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identity(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_result(self) -> KubernetesScaleResult:
        if self.target.model_class != self.model_class:
            raise ValueError("scale result target must match model_class")
        should_apply = (
            self.action is not ScalingDecisionAction.HOLD
            and self.previous_replicas != self.requested_replicas
        )
        if self.applied != should_apply:
            raise ValueError("applied must match the requested replica mutation")
        if self.action is ScalingDecisionAction.HOLD and self.applied:
            raise ValueError("hold decisions cannot mutate the scale subresource")
        expected_result = self.requested_replicas if self.applied else self.previous_replicas
        if self.resulting_replicas != expected_result:
            raise ValueError("resulting_replicas must match the applied outcome")
        if not self.applied and self.resource_version_after != self.resource_version_before:
            raise ValueError("a no-op must preserve the observed resource version")
        if self.applied and self.resource_version_after == self.resource_version_before:
            raise ValueError("an applied mutation must advance the resource version")
        return self


class KubernetesScaleActuator:
    """Apply validated decisions through apps/v1 Scale with resourceVersion CAS.

    Leadership is intentionally supplied by the caller through
    ``LeaderFencedRunnerController.mutate_autoscaler``. WP3.4 will propagate
    that fencing token into the mutation target; this slice only performs the
    bounded, idempotent scale-subresource write. It must not be wired into a
    production reconciliation loop until that structural fence exists.
    """

    _SERVICE_ACCOUNT_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")

    def __init__(
        self,
        *,
        api_server: str | None = None,
        token_path: str | Path | None = None,
        ca_path: str | Path | None = None,
        client: httpx.Client | None = None,
        close_client: bool | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ValueError("timeout_s must be a number")
        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_s must be finite and > 0")
        host = os.environ.get("KUBERNETES_SERVICE_HOST")
        port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        if api_server is None:
            api_host = f"[{host}]" if host and ":" in host else host
            api_server = (
                "https://kubernetes.default.svc" if not api_host else f"https://{api_host}:{port}"
            )
        self._api_server = api_server.rstrip("/")
        self._token_path = Path(token_path or self._SERVICE_ACCOUNT_DIR / "token")
        resolved_ca = Path(ca_path or self._SERVICE_ACCOUNT_DIR / "ca.crt")
        if client is None:
            if close_client is False:
                raise ValueError("close_client=False requires an injected client")
            self._client = httpx.Client(verify=str(resolved_ca), timeout=timeout)
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False if close_client is None else close_client
        self._lock = threading.RLock()
        self._closed = False

    def _url(self, target: KubernetesScaleTarget) -> str:
        namespace = quote(target.namespace, safe="")
        name = quote(target.name, safe="")
        return (
            f"{self._api_server}/apis/apps/v1/namespaces/{namespace}/"
            f"{target.kind.plural}/{name}/scale"
        )

    def _headers(self) -> dict[str, str]:
        token = self._token_path.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("Kubernetes service-account token is empty")
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }

    @staticmethod
    def _parse_scale(
        payload: Any,
        *,
        target: KubernetesScaleTarget,
    ) -> tuple[int, str]:
        if not isinstance(payload, dict):
            raise InvalidKubernetesScaleResponseError("Scale response must be an object")
        if payload.get("apiVersion") != "autoscaling/v1" or payload.get("kind") != "Scale":
            raise InvalidKubernetesScaleResponseError(
                "Scale response must use autoscaling/v1 kind Scale"
            )
        metadata = payload.get("metadata")
        spec = payload.get("spec")
        if not isinstance(metadata, dict) or not isinstance(spec, dict):
            raise InvalidKubernetesScaleResponseError(
                "Scale response metadata and spec must be objects"
            )
        if metadata.get("name") != target.name or metadata.get("namespace") != target.namespace:
            raise InvalidKubernetesScaleResponseError(
                "Scale response identity does not match the requested target"
            )
        resource_version = metadata.get("resourceVersion")
        if not isinstance(resource_version, str) or not resource_version:
            raise InvalidKubernetesScaleResponseError(
                "Scale response requires a non-empty resourceVersion"
            )
        replicas = spec.get("replicas")
        if type(replicas) is not int or not 0 <= replicas <= 100_000:
            raise InvalidKubernetesScaleResponseError(
                "Scale response spec.replicas must be an integer in [0, 100000]"
            )
        return replicas, resource_version

    @staticmethod
    def _response_payload(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as error:
            raise InvalidKubernetesScaleResponseError(
                "Scale response body must be valid JSON"
            ) from error

    def apply(
        self,
        decision: ScalingDecisionRecord,
        target: KubernetesScaleTarget,
    ) -> KubernetesScaleResult:
        """Apply one decision once; exact retries become read-only no-ops."""

        if not isinstance(decision, ScalingDecisionRecord):
            raise TypeError("decision must be a ScalingDecisionRecord")
        if not isinstance(target, KubernetesScaleTarget):
            raise TypeError("target must be a KubernetesScaleTarget")
        decision = ScalingDecisionRecord.model_validate(decision.model_dump())
        target = KubernetesScaleTarget.model_validate(target.model_dump())
        if decision.window.model_class != target.model_class:
            raise ValueError("decision and Kubernetes target model_class must match")

        with self._lock:
            if self._closed:
                raise RuntimeError("KubernetesScaleActuator is closed")
            url = self._url(target)
            headers = self._headers()
            observed_response = self._client.get(url, headers=headers)
            observed_response.raise_for_status()
            previous, resource_version = self._parse_scale(
                self._response_payload(observed_response),
                target=target,
            )
            if (
                decision.action is ScalingDecisionAction.HOLD
                or previous == decision.desired_replicas
            ):
                return KubernetesScaleResult(
                    decision_id=decision.decision_id,
                    model_class=target.model_class,
                    target=target,
                    action=decision.action,
                    previous_replicas=previous,
                    requested_replicas=decision.desired_replicas,
                    resulting_replicas=previous,
                    applied=False,
                    resource_version_before=resource_version,
                    resource_version_after=resource_version,
                )

            observed_replicas = decision.window.observations[-1].runners.current_replicas
            if previous != observed_replicas:
                raise KubernetesScaleConflictError(
                    "live replicas changed since the scaling decision observation"
                )

            body = {
                "apiVersion": "autoscaling/v1",
                "kind": "Scale",
                "metadata": {
                    "name": target.name,
                    "namespace": target.namespace,
                    "resourceVersion": resource_version,
                },
                "spec": {"replicas": decision.desired_replicas},
            }
            response = self._client.put(
                url,
                headers={**headers, "Content-Type": "application/json"},
                json=body,
            )
            if response.status_code == 409:
                raise KubernetesScaleConflictError(
                    "Kubernetes Scale resourceVersion changed concurrently"
                )
            response.raise_for_status()
            current, updated_resource_version = self._parse_scale(
                self._response_payload(response),
                target=target,
            )
            if current != decision.desired_replicas:
                raise InvalidKubernetesScaleResponseError(
                    "Scale response did not confirm the requested replica count"
                )
            if updated_resource_version == resource_version:
                raise InvalidKubernetesScaleResponseError(
                    "Scale response did not advance resourceVersion"
                )
            return KubernetesScaleResult(
                decision_id=decision.decision_id,
                model_class=target.model_class,
                target=target,
                action=decision.action,
                previous_replicas=previous,
                requested_replicas=decision.desired_replicas,
                resulting_replicas=current,
                applied=True,
                resource_version_before=resource_version,
                resource_version_after=updated_resource_version,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._owns_client:
                self._client.close()
            self._closed = True

    def __enter__(self) -> KubernetesScaleActuator:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
