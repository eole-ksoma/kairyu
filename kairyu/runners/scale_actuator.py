"""Idempotent Kubernetes scale-subresource actuator for Runner workloads."""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.runners.kubernetes import MODEL_REVISION_ANNOTATION, RELEASE_ID_ANNOTATION
from kairyu.runners.leadership import RunnerWriterAuthority
from kairyu.runners.prewarm import (
    ModelCachePlacementState,
    ScalingPrewarmPlan,
)
from kairyu.runners.scaling_drain import ScalingDrainCandidate, ScalingDrainPlan
from kairyu.runners.scaling_log import (
    ScalingDecisionAction,
    ScalingDecisionLog,
    ScalingDecisionRecord,
    ScalingDecisionTargetRevision,
)
from kairyu.runners.scaling_quota import ScalingQuotaAdmission

SCALE_ELECTION_ID_ANNOTATION = "kairyu.ai/scale-election-id"
SCALE_FENCING_TOKEN_ANNOTATION = "kairyu.ai/scale-fencing-token"
SCALE_DECISION_GENERATION_ANNOTATION = "kairyu.ai/scale-decision-generation"
SCALE_DECISION_ID_ANNOTATION = "kairyu.ai/scale-decision-id"
SCALE_DECISION_FINGERPRINT_ANNOTATION = "kairyu.ai/scale-decision-fingerprint"
CACHE_PLACEMENT_BINDING_ANNOTATION = "kairyu.ai/cache-placement-binding"
SCALE_DOWN_DRAIN_FINALIZER = "kairyu.ai/scale-down-drain"
_SCALE_AUTHORITY_ANNOTATIONS = (
    SCALE_ELECTION_ID_ANNOTATION,
    SCALE_FENCING_TOKEN_ANNOTATION,
)
_SCALE_DECISION_ANNOTATIONS = (
    SCALE_DECISION_GENERATION_ANNOTATION,
    SCALE_DECISION_ID_ANNOTATION,
    SCALE_DECISION_FINGERPRINT_ANNOTATION,
)


class KubernetesScaleConflictError(RuntimeError):
    """The workload changed concurrently with a scale-subresource write."""


class KubernetesScaleCleanupPendingError(KubernetesScaleConflictError):
    """An applied ordered scale-down still has a terminating higher ordinal."""


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


class KubernetesScaleFence(BaseModel):
    """Immutable workload revision observed after authority was claimed."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-kubernetes-scale-fence-v1"] = "runner-kubernetes-scale-fence-v1"
    workload_uid: str = Field(max_length=255)
    workload_generation: int = Field(ge=1, le=2**63 - 1)
    release_id: str = Field(max_length=255)
    model_revision: str = Field(max_length=255)

    @field_validator("workload_generation", mode="before")
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        return value

    @field_validator("workload_uid", "release_id", "model_revision")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identity(value, name=info.field_name)


class KubernetesFencedScaleResult(BaseModel):
    """Auditable result of a generation- and leader-fenced mutation."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-kubernetes-fenced-scale-result-v1"] = (
        "runner-kubernetes-fenced-scale-result-v1"
    )
    scale: KubernetesScaleResult
    decision: ScalingDecisionRecord
    authority: RunnerWriterAuthority
    fence: KubernetesScaleFence
    successor_cleanup: bool = False
    workload_generation_before: int = Field(ge=1, le=2**63 - 1)
    workload_generation_after: int = Field(ge=1, le=2**63 - 1)

    @field_validator(
        "workload_generation_before",
        "workload_generation_after",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        return value

    @field_validator("successor_cleanup", mode="before")
    @classmethod
    def validate_boolean(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("successor_cleanup must be a boolean")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> KubernetesFencedScaleResult:
        if self.decision.decision_id != self.scale.decision_id:
            raise ValueError("fenced result decision must match scale decision_id")
        if self.decision.policy.model_class != self.scale.model_class:
            raise ValueError("fenced result decision must match scale model_class")
        if self.decision.action is not self.scale.action:
            raise ValueError("fenced result decision must match scale action")
        if self.decision.desired_replicas != self.scale.requested_replicas:
            raise ValueError("fenced result decision must match requested replicas")
        revision = self.decision.target_revision
        if revision is None:
            raise ValueError("fenced result decision must persist its target revision")
        if (
            revision.target_kind != self.scale.target.kind.value
            or revision.namespace != self.scale.target.namespace
            or revision.name != self.scale.target.name
            or revision.workload_uid != self.fence.workload_uid
            or revision.workload_generation != self.fence.workload_generation
            or revision.release_id != self.fence.release_id
            or revision.model_revision != self.fence.model_revision
        ):
            raise ValueError("fenced result decision target revision is inconsistent")
        authority_matches_decision = (
            revision.election_id == self.authority.election_id
            and revision.fencing_token == self.authority.fencing_token
        )
        if self.successor_cleanup:
            if (
                self.scale.applied
                or self.decision.action is not ScalingDecisionAction.SCALE_DOWN
                or revision.election_id != self.authority.election_id
                or self.authority.fencing_token <= revision.fencing_token
            ):
                raise ValueError(
                    "successor cleanup requires a newer leader and a previously applied scale-down"
                )
        elif not authority_matches_decision:
            raise ValueError("fenced result authority must match the scale decision")
        if self.decision.action is ScalingDecisionAction.HOLD:
            if self.decision.decision_generation is not None:
                raise ValueError("hold result cannot consume a decision generation")
            if self.fence.workload_generation != self.workload_generation_before:
                raise ValueError("hold result must match the fenced workload generation")
        elif self.decision.decision_generation is None:
            raise ValueError("mutating result requires a durable decision generation")
        if self.scale.applied:
            if self.fence.workload_generation != self.workload_generation_before:
                raise ValueError("applied result must start from the fenced generation")
            if self.workload_generation_after <= self.workload_generation_before:
                raise ValueError("an applied mutation must advance workload generation")
        else:
            if self.workload_generation_after != self.workload_generation_before:
                raise ValueError("a no-op must preserve workload generation")
            if (
                self.decision.action is not ScalingDecisionAction.HOLD
                and self.workload_generation_before != self.fence.workload_generation + 1
            ):
                raise ValueError("an exact retry must observe the applied generation")
        return self


class KubernetesScaleAuthorityClaim(BaseModel):
    """Durable leader claim that must precede observation and decision."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-kubernetes-scale-authority-claim-v1"] = (
        "runner-kubernetes-scale-authority-claim-v1"
    )
    target: KubernetesScaleTarget
    authority: RunnerWriterAuthority
    applied: bool
    replicas: int = Field(ge=0, le=100_000)
    workload_uid: str = Field(max_length=255)
    workload_generation: int = Field(ge=1, le=2**63 - 1)
    release_id: str = Field(max_length=255)
    model_revision: str = Field(max_length=255)
    resource_version_before: str = Field(max_length=255)
    resource_version_after: str = Field(max_length=255)

    @field_validator("replicas", "workload_generation", mode="before")
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
        "workload_uid",
        "release_id",
        "model_revision",
        "resource_version_before",
        "resource_version_after",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identity(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_claim(self) -> KubernetesScaleAuthorityClaim:
        changed = self.resource_version_before != self.resource_version_after
        if self.applied != changed:
            raise ValueError("authority claim applied must match resourceVersion advancement")
        return self

    @property
    def target_revision(self) -> ScalingDecisionTargetRevision:
        """Return the immutable decision input captured after the claim."""

        return ScalingDecisionTargetRevision(
            target_kind=self.target.kind.value,
            namespace=self.target.namespace,
            name=self.target.name,
            election_id=self.authority.election_id,
            fencing_token=self.authority.fencing_token,
            workload_uid=self.workload_uid,
            workload_generation=self.workload_generation,
            release_id=self.release_id,
            model_revision=self.model_revision,
        )


@dataclass(frozen=True)
class _WorkloadSnapshot:
    replicas: int
    statefulset_start_ordinal: int | None
    resource_version: str
    uid: str
    generation: int
    annotations: dict[str, str]
    annotations_present: bool


@dataclass(frozen=True)
class _DrainPodSnapshot:
    name: str
    uid: str
    finalizers: tuple[str, ...]
    deleting: bool


class KubernetesScaleActuator:
    """Apply validated decisions through Kubernetes optimistic concurrency.

    ``apply`` is the WP3.3 scale-subresource primitive and remains suitable for
    isolated verification only when ``allow_unfenced=True`` is explicitly set.
    Production callers enter through
    ``LeaderFencedRunnerController.mutate_autoscaler`` and pass its authority to
    ``apply_fenced``. That path atomically patches replicas and monotonic fence
    annotations on the parent workload, so a superseded leader cannot write
    after a successor has advanced the token.
    """

    _SERVICE_ACCOUNT_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    _DRAIN_DELETE_WAIT_SECONDS = 30.0
    _DRAIN_DELETE_POLL_SECONDS = 0.25

    def __init__(
        self,
        *,
        api_server: str | None = None,
        token_path: str | Path | None = None,
        ca_path: str | Path | None = None,
        client: httpx.Client | None = None,
        close_client: bool | None = None,
        decision_log: ScalingDecisionLog | None = None,
        allow_unfenced: bool = False,
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
        if decision_log is not None and not isinstance(decision_log, ScalingDecisionLog):
            raise TypeError("decision_log must implement ScalingDecisionLog")
        if type(allow_unfenced) is not bool:
            raise TypeError("allow_unfenced must be a boolean")
        self._decision_log = decision_log
        self._allow_unfenced = allow_unfenced

    def _url(self, target: KubernetesScaleTarget) -> str:
        namespace = quote(target.namespace, safe="")
        name = quote(target.name, safe="")
        return (
            f"{self._api_server}/apis/apps/v1/namespaces/{namespace}/"
            f"{target.kind.plural}/{name}/scale"
        )

    def _workload_url(self, target: KubernetesScaleTarget) -> str:
        namespace = quote(target.namespace, safe="")
        name = quote(target.name, safe="")
        return f"{self._api_server}/apis/apps/v1/namespaces/{namespace}/{target.kind.plural}/{name}"

    def _pod_url(self, *, namespace: str, name: str) -> str:
        namespace = quote(namespace, safe="")
        name = quote(name, safe="")
        return f"{self._api_server}/api/v1/namespaces/{namespace}/pods/{name}"

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
                "Kubernetes response body must be valid JSON"
            ) from error

    @staticmethod
    def _reauthorize(
        authority: RunnerWriterAuthority,
        reauthorize: Callable[[], RunnerWriterAuthority],
    ) -> RunnerWriterAuthority:
        if not callable(reauthorize):
            raise TypeError("reauthorize must be callable")
        refreshed = reauthorize()
        if not isinstance(refreshed, RunnerWriterAuthority):
            raise TypeError("reauthorize must return RunnerWriterAuthority")
        refreshed = RunnerWriterAuthority.model_validate(refreshed.model_dump())
        if refreshed.tenure != authority.tenure:
            raise KubernetesScaleConflictError(
                "leader authority changed during Kubernetes mutation"
            )
        return refreshed

    @staticmethod
    def _reauthorize_quota(
        decision: ScalingDecisionRecord,
        reauthorize_quota: Callable[[], ScalingQuotaAdmission] | None,
        *,
        authority: RunnerWriterAuthority,
    ) -> ScalingQuotaAdmission:
        original = decision.quota_admission
        if original is None:
            raise ValueError("fenced scale-up requires a durable quota admission")
        if not callable(reauthorize_quota):
            raise TypeError("scale-up reauthorize_quota must be callable")
        refreshed = reauthorize_quota()
        if not isinstance(refreshed, ScalingQuotaAdmission):
            raise TypeError("reauthorize_quota must return ScalingQuotaAdmission")
        refreshed = ScalingQuotaAdmission.model_validate(refreshed.model_dump())
        quota_age = (
            authority.validated_at - refreshed.snapshot.observed_at
        ).total_seconds()
        if not 0 <= quota_age <= decision.policy.max_observation_age_seconds:
            raise KubernetesScaleConflictError(
                "quota authority is not fresh at final scale authorization"
            )
        original_kueue = original.snapshot.kueue
        refreshed_kueue = refreshed.snapshot.kueue
        original_kueue_identity = (
            original_kueue.api_version,
            original_kueue.namespace,
            original_kueue.workload_name,
            original_kueue.workload_uid,
            original_kueue.workload_generation,
            original_kueue.target_kind,
            original_kueue.target_namespace,
            original_kueue.target_name,
            original_kueue.target_uid,
            original_kueue.local_queue,
            original_kueue.cluster_queue,
            original_kueue.pod_set_name,
            original_kueue.resource_flavor,
            original_kueue.resource_name,
            original_kueue.priority_class_name,
            original_kueue.priority_class_group,
            original_kueue.priority_class_kind,
            original_kueue.priority_class_source,
            original_kueue.priority,
        )
        refreshed_kueue_identity = (
            refreshed_kueue.api_version,
            refreshed_kueue.namespace,
            refreshed_kueue.workload_name,
            refreshed_kueue.workload_uid,
            refreshed_kueue.workload_generation,
            refreshed_kueue.target_kind,
            refreshed_kueue.target_namespace,
            refreshed_kueue.target_name,
            refreshed_kueue.target_uid,
            refreshed_kueue.local_queue,
            refreshed_kueue.cluster_queue,
            refreshed_kueue.pod_set_name,
            refreshed_kueue.resource_flavor,
            refreshed_kueue.resource_name,
            refreshed_kueue.priority_class_name,
            refreshed_kueue.priority_class_group,
            refreshed_kueue.priority_class_kind,
            refreshed_kueue.priority_class_source,
            refreshed_kueue.priority,
        )
        if (
            refreshed.snapshot.tenant_id != original.snapshot.tenant_id
            or refreshed.snapshot.model_class != original.snapshot.model_class
            or refreshed.snapshot.model_family != original.snapshot.model_family
            or refreshed.snapshot.gpus_per_replica
            != original.snapshot.gpus_per_replica
            or refreshed.current_replicas != original.current_replicas
            or refreshed.requested_replicas != original.requested_replicas
            or refreshed.snapshot.observed_at < original.snapshot.observed_at
            or refreshed.snapshot.quota_revision
            < original.snapshot.quota_revision
            or refreshed_kueue_identity != original_kueue_identity
        ):
            raise KubernetesScaleConflictError(
                "quota authority changed during Kubernetes mutation"
            )
        if (
            not refreshed_kueue.admitted
            or refreshed.admitted_replicas < decision.desired_replicas
        ):
            raise KubernetesScaleConflictError(
                "quota authority no longer admits the scaling decision"
            )
        return refreshed

    @staticmethod
    def _reauthorize_prewarm(
        decision: ScalingDecisionRecord,
        reauthorize_prewarm: Callable[[], ScalingPrewarmPlan] | None,
        *,
        authority: RunnerWriterAuthority,
    ) -> ScalingPrewarmPlan:
        original = decision.prewarm_plan
        if original is None:
            raise ValueError("fenced scale-up requires a durable prewarm plan")
        if not callable(reauthorize_prewarm):
            raise TypeError("scale-up reauthorize_prewarm must be callable")
        refreshed = reauthorize_prewarm()
        if not isinstance(refreshed, ScalingPrewarmPlan):
            raise TypeError("reauthorize_prewarm must return ScalingPrewarmPlan")
        refreshed = ScalingPrewarmPlan.model_validate(refreshed.model_dump())
        cache_age = (
            authority.validated_at - refreshed.snapshot.observed_at
        ).total_seconds()
        if not 0 <= cache_age <= decision.policy.max_observation_age_seconds:
            raise KubernetesScaleConflictError(
                "prewarm authority is not fresh at final scale authorization"
            )
        if (
            refreshed.snapshot.model_class != original.snapshot.model_class
            or refreshed.snapshot.model_revision != original.snapshot.model_revision
            or refreshed.snapshot.artifact_digest != original.snapshot.artifact_digest
            or refreshed.snapshot.placement_binding_id
            != original.snapshot.placement_binding_id
            or refreshed.resource_flavor != original.resource_flavor
            or refreshed.current_replicas != original.current_replicas
            or refreshed.quota_target_replicas != original.quota_target_replicas
            or refreshed.snapshot.observed_at < original.snapshot.observed_at
            or refreshed.snapshot.cache_revision < original.snapshot.cache_revision
        ):
            raise KubernetesScaleConflictError(
                "prewarm authority changed during Kubernetes mutation"
            )
        original_placements = {
            placement.placement_id: placement
            for placement in original.snapshot.placements
            if placement.placement_id in original.runner_start_placement_ids
        }
        refreshed_placements = {
            placement.placement_id: placement
            for placement in refreshed.snapshot.placements
        }
        retained_ready_capacity = all(
            placement_id in refreshed_placements
            and (
                refreshed_placements[placement_id].node_name,
                refreshed_placements[placement_id].resource_flavor,
                refreshed_placements[placement_id].profile_id,
                refreshed_placements[placement_id].compatibility_approval_id,
            )
            == (
                original_placement.node_name,
                original_placement.resource_flavor,
                original_placement.profile_id,
                original_placement.compatibility_approval_id,
            )
            and not refreshed_placements[placement_id].assigned
            and refreshed_placements[placement_id].healthy
            and refreshed_placements[placement_id].schedulable
            and refreshed_placements[placement_id].state
            is ModelCachePlacementState.READY
            for placement_id, original_placement in original_placements.items()
        )
        if (
            refreshed.runner_target_replicas < decision.desired_replicas
            or not retained_ready_capacity
        ):
            raise KubernetesScaleConflictError(
                "prewarm authority no longer provides ready cache capacity"
            )
        return refreshed

    @staticmethod
    def _reauthorize_drain(
        decision: ScalingDecisionRecord,
        reauthorize_drain: Callable[[], ScalingDrainPlan] | None,
        *,
        authority: RunnerWriterAuthority,
    ) -> ScalingDrainPlan:
        original = decision.drain_plan
        if original is None:
            raise ValueError("fenced scale-down requires a durable drain plan")
        if not callable(reauthorize_drain):
            raise TypeError("scale-down reauthorize_drain must be callable")
        refreshed = reauthorize_drain()
        if not isinstance(refreshed, ScalingDrainPlan):
            raise TypeError("reauthorize_drain must return ScalingDrainPlan")
        refreshed = ScalingDrainPlan.model_validate(refreshed.model_dump())
        drain_age = (
            authority.validated_at - refreshed.source_observed_at
        ).total_seconds()
        if not 0 <= drain_age <= decision.policy.max_observation_age_seconds:
            raise KubernetesScaleConflictError(
                "drain authority is not fresh at final scale authorization"
            )
        original_identity = (
            original.snapshot.model_class,
            original.snapshot.namespace,
            original.snapshot.statefulset_name,
            original.snapshot.workload_uid,
            original.snapshot.workload_generation,
            original.snapshot.release_id,
            original.snapshot.model_revision,
            original.current_replicas,
            original.desired_replicas,
            original.candidate_runner_ids,
            original.candidate_pod_uids,
        )
        refreshed_identity = (
            refreshed.snapshot.model_class,
            refreshed.snapshot.namespace,
            refreshed.snapshot.statefulset_name,
            refreshed.snapshot.workload_uid,
            refreshed.snapshot.workload_generation,
            refreshed.snapshot.release_id,
            refreshed.snapshot.model_revision,
            refreshed.current_replicas,
            refreshed.desired_replicas,
            refreshed.candidate_runner_ids,
            refreshed.candidate_pod_uids,
        )
        if (
            refreshed_identity != original_identity
            or refreshed.snapshot.observed_at < original.snapshot.observed_at
            or refreshed.snapshot.drain_revision < original.snapshot.drain_revision
            or refreshed.source_observed_at < original.source_observed_at
        ):
            raise KubernetesScaleConflictError(
                "drain authority changed during Kubernetes mutation"
            )
        original_candidates = {
            candidate.workload_ordinal: candidate
            for candidate in original.snapshot.candidates
            if candidate.status.pod_uid in original.candidate_pod_uids
        }
        refreshed_candidates = {
            candidate.workload_ordinal: candidate
            for candidate in refreshed.snapshot.candidates
            if candidate.status.pod_uid in refreshed.candidate_pod_uids
        }
        evidence_retained = original_candidates.keys() == refreshed_candidates.keys() and all(
            (
                refreshed_candidates[ordinal].pod_name,
                refreshed_candidates[ordinal].status.runner_id,
                refreshed_candidates[ordinal].status.pod_uid,
                refreshed_candidates[ordinal].status.termination_authorization,
            )
            == (
                candidate.pod_name,
                candidate.status.runner_id,
                candidate.status.pod_uid,
                candidate.status.termination_authorization,
            )
            and refreshed_candidates[ordinal].status.observed_at
            >= candidate.status.observed_at
            for ordinal, candidate in original_candidates.items()
        )
        if not evidence_retained:
            raise KubernetesScaleConflictError(
                "drain authority no longer authorizes the removed StatefulSet ordinals"
            )
        return refreshed

    @staticmethod
    def _parse_workload(
        payload: Any,
        *,
        target: KubernetesScaleTarget,
    ) -> _WorkloadSnapshot:
        if not isinstance(payload, dict):
            raise InvalidKubernetesScaleResponseError("workload response must be an object")
        if payload.get("apiVersion") != "apps/v1" or payload.get("kind") != target.kind:
            raise InvalidKubernetesScaleResponseError(
                f"workload response must use apps/v1 kind {target.kind.value}"
            )
        metadata = payload.get("metadata")
        spec = payload.get("spec")
        if not isinstance(metadata, dict) or not isinstance(spec, dict):
            raise InvalidKubernetesScaleResponseError(
                "workload response metadata and spec must be objects"
            )
        if metadata.get("name") != target.name or metadata.get("namespace") != target.namespace:
            raise InvalidKubernetesScaleResponseError(
                "workload response identity does not match the requested target"
            )
        resource_version = metadata.get("resourceVersion")
        uid = metadata.get("uid")
        generation = metadata.get("generation")
        replicas = spec.get("replicas")
        if not isinstance(resource_version, str) or not resource_version:
            raise InvalidKubernetesScaleResponseError(
                "workload response requires a non-empty resourceVersion"
            )
        if not isinstance(uid, str) or not uid:
            raise InvalidKubernetesScaleResponseError("workload response requires a non-empty UID")
        if type(generation) is not int or not 1 <= generation <= 2**63 - 1:
            raise InvalidKubernetesScaleResponseError(
                "workload metadata.generation must be a positive integer"
            )
        if type(replicas) is not int or not 0 <= replicas <= 100_000:
            raise InvalidKubernetesScaleResponseError(
                "workload spec.replicas must be an integer in [0, 100000]"
            )
        statefulset_start_ordinal: int | None = None
        if target.kind is KubernetesScalableKind.STATEFUL_SET:
            ordinals = spec.get("ordinals")
            if ordinals is None:
                statefulset_start_ordinal = 0
            elif not isinstance(ordinals, dict):
                raise InvalidKubernetesScaleResponseError(
                    "StatefulSet spec.ordinals must be an object"
                )
            else:
                statefulset_start_ordinal = ordinals.get("start", 0)
                if (
                    type(statefulset_start_ordinal) is not int
                    or not 0 <= statefulset_start_ordinal <= 100_000
                ):
                    raise InvalidKubernetesScaleResponseError(
                        "StatefulSet start ordinal must be an integer in [0, 100000]"
                    )
        annotations_present = "annotations" in metadata
        annotations_payload = metadata.get("annotations", {})
        if not isinstance(annotations_payload, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in annotations_payload.items()
        ):
            raise InvalidKubernetesScaleResponseError(
                "workload metadata.annotations must contain string pairs"
            )
        return _WorkloadSnapshot(
            replicas=replicas,
            statefulset_start_ordinal=statefulset_start_ordinal,
            resource_version=resource_version,
            uid=uid,
            generation=generation,
            annotations=dict(annotations_payload),
            annotations_present=annotations_present,
        )

    @staticmethod
    def _parse_drain_pod(
        payload: Any,
        *,
        candidate: ScalingDrainCandidate,
        plan: ScalingDrainPlan,
    ) -> _DrainPodSnapshot:
        if not isinstance(payload, dict):
            raise InvalidKubernetesScaleResponseError(
                "drain Pod response must be an object"
            )
        if payload.get("apiVersion") != "v1" or payload.get("kind") != "Pod":
            raise InvalidKubernetesScaleResponseError(
                "drain Pod response must use v1 kind Pod"
            )
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise InvalidKubernetesScaleResponseError(
                "drain Pod response metadata must be an object"
            )
        name = metadata.get("name")
        namespace = metadata.get("namespace")
        uid = metadata.get("uid")
        if (
            name != candidate.pod_name
            or namespace != plan.snapshot.namespace
            or uid != candidate.status.pod_uid
        ):
            raise KubernetesScaleConflictError(
                "drain Pod identity changed before StatefulSet scale-down"
            )
        finalizers = metadata.get("finalizers", [])
        if (
            not isinstance(finalizers, list)
            or any(not isinstance(value, str) or not value for value in finalizers)
            or len(set(finalizers)) != len(finalizers)
        ):
            raise InvalidKubernetesScaleResponseError(
                "drain Pod finalizers must be unique non-empty strings"
            )
        owner_references = metadata.get("ownerReferences", [])
        if not isinstance(owner_references, list) or not any(
            isinstance(owner, dict)
            and owner.get("apiVersion") == "apps/v1"
            and owner.get("kind") == "StatefulSet"
            and owner.get("name") == plan.snapshot.statefulset_name
            and owner.get("uid") == plan.snapshot.workload_uid
            and owner.get("controller") is True
            for owner in owner_references
        ):
            raise KubernetesScaleConflictError(
                "drain Pod is not controlled by the planned StatefulSet"
            )
        deletion_timestamp = metadata.get("deletionTimestamp")
        if deletion_timestamp is not None and not isinstance(deletion_timestamp, str):
            raise InvalidKubernetesScaleResponseError(
                "drain Pod deletionTimestamp must be a string"
            )
        return _DrainPodSnapshot(
            name=name,
            uid=uid,
            finalizers=tuple(finalizers),
            deleting=deletion_timestamp is not None,
        )

    def _observe_drain_pods(
        self,
        plan: ScalingDrainPlan,
        *,
        headers: dict[str, str],
        allow_absent: bool,
        allow_deleting: bool,
        allow_released: bool,
    ) -> tuple[_DrainPodSnapshot, ...]:
        observed: list[_DrainPodSnapshot] = []
        for candidate in plan.selected_candidates:
            response = self._client.get(
                self._pod_url(
                    namespace=plan.snapshot.namespace,
                    name=candidate.pod_name,
                ),
                headers=headers,
            )
            if response.status_code == 404 and allow_absent:
                continue
            if response.status_code == 404:
                raise KubernetesScaleConflictError(
                    "drain Pod disappeared before StatefulSet scale-down"
                )
            response.raise_for_status()
            pod = self._parse_drain_pod(
                self._response_payload(response),
                candidate=candidate,
                plan=plan,
            )
            if SCALE_DOWN_DRAIN_FINALIZER not in pod.finalizers:
                if allow_released and pod.deleting:
                    observed.append(pod)
                    continue
                raise KubernetesScaleConflictError(
                    "drain Pod is missing its scale-down deletion hold"
                )
            if pod.deleting and not allow_deleting:
                raise KubernetesScaleConflictError(
                    "drain Pod deletion started before StatefulSet scale-down"
                )
            observed.append(pod)
        return tuple(observed)

    def _release_drain_pods(
        self,
        plan: ScalingDrainPlan,
        pods: tuple[_DrainPodSnapshot, ...],
        *,
        authority: RunnerWriterAuthority,
        reauthorize: Callable[[], RunnerWriterAuthority],
        headers: dict[str, str],
    ) -> RunnerWriterAuthority:
        pods_by_name = {pod.name: pod for pod in pods}
        for candidate in reversed(plan.selected_candidates):
            pod = pods_by_name.get(candidate.pod_name)
            if pod is None:
                continue
            if SCALE_DOWN_DRAIN_FINALIZER not in pod.finalizers:
                self._wait_for_drain_pod_absence(
                    candidate,
                    plan,
                    headers=headers,
                )
                continue
            authority = self._reauthorize(authority, reauthorize)
            url = self._pod_url(namespace=plan.snapshot.namespace, name=pod.name)
            response = self._client.request(
                "DELETE",
                url,
                headers={**headers, "Content-Type": "application/json"},
                json={
                    "apiVersion": "v1",
                    "kind": "DeleteOptions",
                    "gracePeriodSeconds": 0,
                    "preconditions": {"uid": pod.uid},
                },
            )
            if response.status_code in {409, 422}:
                raise KubernetesScaleConflictError(
                    "drain Pod changed before preconditioned deletion"
                )
            if response.status_code == 404:
                raise KubernetesScaleConflictError(
                    "held drain Pod disappeared before preconditioned deletion"
                )
            response.raise_for_status()
            finalizer_index = pod.finalizers.index(SCALE_DOWN_DRAIN_FINALIZER)
            response = self._client.patch(
                url,
                headers={**headers, "Content-Type": "application/json-patch+json"},
                json=[
                    {"op": "test", "path": "/metadata/uid", "value": pod.uid},
                    {
                        "op": "test",
                        "path": "/metadata/finalizers",
                        "value": list(pod.finalizers),
                    },
                    {
                        "op": "remove",
                        "path": f"/metadata/finalizers/{finalizer_index}",
                    },
                ],
            )
            if response.status_code in {404, 409, 422}:
                raise KubernetesScaleConflictError(
                    "drain Pod deletion hold changed before release"
                )
            response.raise_for_status()
            self._wait_for_drain_pod_absence(
                candidate,
                plan,
                headers=headers,
            )
        return authority

    def _wait_for_drain_pod_absence(
        self,
        candidate: ScalingDrainCandidate,
        plan: ScalingDrainPlan,
        *,
        headers: dict[str, str],
    ) -> None:
        deadline = time.monotonic() + self._DRAIN_DELETE_WAIT_SECONDS
        url = self._pod_url(
            namespace=plan.snapshot.namespace,
            name=candidate.pod_name,
        )
        while True:
            response = self._client.get(url, headers=headers)
            if response.status_code == 404:
                return
            response.raise_for_status()
            pod = self._parse_drain_pod(
                self._response_payload(response),
                candidate=candidate,
                plan=plan,
            )
            if not pod.deleting or SCALE_DOWN_DRAIN_FINALIZER in pod.finalizers:
                raise KubernetesScaleConflictError(
                    "drain Pod did not retain preconditioned deletion progress"
                )
            if time.monotonic() >= deadline:
                raise KubernetesScaleCleanupPendingError(
                    "ordered scale-down cleanup remains pending; retry the decision"
                )
            time.sleep(self._DRAIN_DELETE_POLL_SECONDS)

    @staticmethod
    def _stored_authority(snapshot: _WorkloadSnapshot) -> tuple[str, int] | None:
        values = tuple(snapshot.annotations.get(key) for key in _SCALE_AUTHORITY_ANNOTATIONS)
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise InvalidKubernetesScaleResponseError(
                "workload has incomplete scale authority annotations"
            )
        election_id, token_text = values
        assert election_id is not None
        assert token_text is not None
        try:
            token = int(token_text)
        except ValueError as error:
            raise InvalidKubernetesScaleResponseError(
                "workload scale authority token must be an integer"
            ) from error
        if not election_id or token <= 0 or str(token) != token_text:
            raise InvalidKubernetesScaleResponseError(
                "workload scale authority annotations are not canonical"
            )
        return election_id, token

    @staticmethod
    def _stored_decision(snapshot: _WorkloadSnapshot) -> tuple[int, str, str] | None:
        values = tuple(snapshot.annotations.get(key) for key in _SCALE_DECISION_ANNOTATIONS)
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise InvalidKubernetesScaleResponseError(
                "workload has incomplete scale decision annotations"
            )
        generation_text, decision_id, fingerprint = values
        assert generation_text is not None
        assert decision_id is not None
        assert fingerprint is not None
        try:
            generation = int(generation_text)
        except ValueError as error:
            raise InvalidKubernetesScaleResponseError(
                "workload scale decision generation must be an integer"
            ) from error
        if (
            generation <= 0
            or str(generation) != generation_text
            or not decision_id
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise InvalidKubernetesScaleResponseError(
                "workload scale decision annotations are not canonical"
            )
        return generation, decision_id, fingerprint

    @staticmethod
    def _annotation_path(name: str) -> str:
        escaped = name.replace("~", "~0").replace("/", "~1")
        return f"/metadata/annotations/{escaped}"

    @staticmethod
    def _scale_result(
        *,
        decision: ScalingDecisionRecord,
        target: KubernetesScaleTarget,
        previous: int,
        resulting: int,
        applied: bool,
        resource_version_before: str,
        resource_version_after: str,
    ) -> KubernetesScaleResult:
        return KubernetesScaleResult(
            decision_id=decision.decision_id,
            model_class=target.model_class,
            target=target,
            action=decision.action,
            previous_replicas=previous,
            requested_replicas=decision.desired_replicas,
            resulting_replicas=resulting,
            applied=applied,
            resource_version_before=resource_version_before,
            resource_version_after=resource_version_after,
        )

    def apply(
        self,
        decision: ScalingDecisionRecord,
        target: KubernetesScaleTarget,
    ) -> KubernetesScaleResult:
        """Apply one decision once; exact retries become read-only no-ops."""

        if not self._allow_unfenced:
            raise RuntimeError(
                "unfenced scale writes are disabled; use apply_fenced in production"
            )

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

    def claim_authority(
        self,
        target: KubernetesScaleTarget,
        authority: RunnerWriterAuthority,
        *,
        reauthorize: Callable[[], RunnerWriterAuthority],
    ) -> KubernetesScaleAuthorityClaim:
        """Persist a new leader token before observing inputs or deciding."""

        if not isinstance(target, KubernetesScaleTarget):
            raise TypeError("target must be a KubernetesScaleTarget")
        if not isinstance(authority, RunnerWriterAuthority):
            raise TypeError("authority must be a RunnerWriterAuthority")
        target = KubernetesScaleTarget.model_validate(target.model_dump())
        authority = RunnerWriterAuthority.model_validate(authority.model_dump())
        with self._lock:
            if self._closed:
                raise RuntimeError("KubernetesScaleActuator is closed")
            url = self._workload_url(target)
            headers = self._headers()
            response = self._client.get(url, headers=headers)
            response.raise_for_status()
            observed = self._parse_workload(self._response_payload(response), target=target)
            release_id = observed.annotations.get(RELEASE_ID_ANNOTATION)
            model_revision = observed.annotations.get(MODEL_REVISION_ANNOTATION)
            if not release_id or not model_revision:
                raise InvalidKubernetesScaleResponseError(
                    "workload requires release and model revision annotations"
                )
            stored = self._stored_authority(observed)
            requested = (authority.election_id, authority.fencing_token)
            if stored == requested:
                authority = self._reauthorize(authority, reauthorize)
                return KubernetesScaleAuthorityClaim(
                    target=target,
                    authority=authority,
                    applied=False,
                    replicas=observed.replicas,
                    workload_uid=observed.uid,
                    workload_generation=observed.generation,
                    release_id=release_id,
                    model_revision=model_revision,
                    resource_version_before=observed.resource_version,
                    resource_version_after=observed.resource_version,
                )
            if stored is not None:
                election_id, token = stored
                if election_id != authority.election_id:
                    raise KubernetesScaleConflictError(
                        "workload scale authority belongs to another election"
                    )
                if token >= authority.fencing_token:
                    raise KubernetesScaleConflictError(
                        "workload was already claimed by an equal or newer leader"
                    )
            authority = self._reauthorize(authority, reauthorize)
            annotations = {
                SCALE_ELECTION_ID_ANNOTATION: authority.election_id,
                SCALE_FENCING_TOKEN_ANNOTATION: str(authority.fencing_token),
            }
            patch: list[dict[str, object]] = [
                {
                    "op": "test",
                    "path": "/metadata/resourceVersion",
                    "value": observed.resource_version,
                },
                {"op": "test", "path": "/metadata/uid", "value": observed.uid},
            ]
            if observed.annotations_present:
                patch.extend(
                    {
                        "op": "add",
                        "path": self._annotation_path(name),
                        "value": value,
                    }
                    for name, value in annotations.items()
                )
            else:
                patch.append(
                    {"op": "add", "path": "/metadata/annotations", "value": annotations}
                )
            response = self._client.patch(
                url,
                headers={**headers, "Content-Type": "application/json-patch+json"},
                json=patch,
            )
            if response.status_code in {409, 422}:
                raise KubernetesScaleConflictError(
                    "Kubernetes workload changed during authority claim"
                )
            response.raise_for_status()
            updated = self._parse_workload(self._response_payload(response), target=target)
            if (
                updated.uid != observed.uid
                or updated.replicas != observed.replicas
                or updated.generation != observed.generation
                or updated.resource_version == observed.resource_version
                or self._stored_authority(updated) != requested
                or updated.annotations.get(RELEASE_ID_ANNOTATION) != release_id
                or updated.annotations.get(MODEL_REVISION_ANNOTATION) != model_revision
            ):
                raise InvalidKubernetesScaleResponseError(
                    "authority claim response violated the workload contract"
                )
            return KubernetesScaleAuthorityClaim(
                target=target,
                authority=authority,
                applied=True,
                replicas=updated.replicas,
                workload_uid=updated.uid,
                workload_generation=updated.generation,
                release_id=release_id,
                model_revision=model_revision,
                resource_version_before=observed.resource_version,
                resource_version_after=updated.resource_version,
            )

    def apply_fenced(
        self,
        decision: ScalingDecisionRecord,
        target: KubernetesScaleTarget,
        *,
        authority: RunnerWriterAuthority,
        fence: KubernetesScaleFence,
        reauthorize: Callable[[], RunnerWriterAuthority],
        reauthorize_quota: Callable[[], ScalingQuotaAdmission] | None = None,
        reauthorize_prewarm: Callable[[], ScalingPrewarmPlan] | None = None,
        reauthorize_drain: Callable[[], ScalingDrainPlan] | None = None,
    ) -> KubernetesFencedScaleResult:
        """Apply a durable decision only under a previously claimed leader token."""

        if not isinstance(decision, ScalingDecisionRecord):
            raise TypeError("decision must be a ScalingDecisionRecord")
        if not isinstance(target, KubernetesScaleTarget):
            raise TypeError("target must be a KubernetesScaleTarget")
        if not isinstance(authority, RunnerWriterAuthority):
            raise TypeError("authority must be a RunnerWriterAuthority")
        if not isinstance(fence, KubernetesScaleFence):
            raise TypeError("fence must be a KubernetesScaleFence")
        decision = ScalingDecisionRecord.model_validate(decision.model_dump())
        target = KubernetesScaleTarget.model_validate(target.model_dump())
        authority = RunnerWriterAuthority.model_validate(authority.model_dump())
        fence = KubernetesScaleFence.model_validate(fence.model_dump())
        if decision.window.model_class != target.model_class:
            raise ValueError("decision and Kubernetes target model_class must match")
        if self._decision_log is None:
            raise RuntimeError("fenced scale requires a durable decision log")
        try:
            durable_decision = self._decision_log.get(decision.decision_id)
        except KeyError as error:
            raise ValueError("scaling decision was not durably appended") from error
        if durable_decision.fingerprint != decision.fingerprint:
            raise KubernetesScaleConflictError(
                "scaling decision does not match the durable decision log"
            )
        decision = durable_decision
        target_revision = decision.target_revision
        if target_revision is None:
            raise ValueError("fenced decision must persist its target revision")
        target_matches_fence = (
            target_revision.target_kind == target.kind.value
            and target_revision.namespace == target.namespace
            and target_revision.name == target.name
            and target_revision.workload_uid == fence.workload_uid
            and target_revision.workload_generation == fence.workload_generation
            and target_revision.release_id == fence.release_id
            and target_revision.model_revision == fence.model_revision
        )
        if not target_matches_fence:
            raise ValueError("decision target revision must match target and fence")
        authority_matches_decision = (
            target_revision.election_id == authority.election_id
            and target_revision.fencing_token == authority.fencing_token
        )
        successor_cleanup = False
        if not authority_matches_decision:
            successor_cleanup = (
                decision.action is ScalingDecisionAction.SCALE_DOWN
                and target_revision.election_id == authority.election_id
                and authority.fencing_token > target_revision.fencing_token
            )
            if not successor_cleanup:
                raise ValueError("decision target revision must match leader authority")
        if (
            decision.action is not ScalingDecisionAction.HOLD
            and decision.decision_generation is None
        ):
            raise ValueError("mutating decision must be appended before fenced apply")
        if (
            decision.action is ScalingDecisionAction.SCALE_UP
            and decision.quota_admission is None
        ):
            raise ValueError("fenced scale-up requires a durable quota admission")
        if (
            decision.action is ScalingDecisionAction.SCALE_UP
            and decision.prewarm_plan is None
        ):
            raise ValueError("fenced scale-up requires a durable prewarm plan")
        if decision.action is ScalingDecisionAction.SCALE_DOWN:
            if target.kind is not KubernetesScalableKind.STATEFUL_SET:
                raise ValueError(
                    "fenced scale-down requires deterministic StatefulSet ordinals"
                )
            if decision.drain_plan is None:
                raise ValueError("fenced scale-down requires a durable drain plan")

        with self._lock:
            if self._closed:
                raise RuntimeError("KubernetesScaleActuator is closed")
            url = self._workload_url(target)
            headers = self._headers()
            response = self._client.get(url, headers=headers)
            response.raise_for_status()
            observed = self._parse_workload(self._response_payload(response), target=target)
            if self._stored_authority(observed) != (
                authority.election_id,
                authority.fencing_token,
            ):
                raise KubernetesScaleConflictError(
                    "leader authority was not claimed or has been superseded"
                )
            if observed.uid != fence.workload_uid:
                raise KubernetesScaleConflictError(
                    "workload UID changed since the scaling decision"
                )
            if observed.annotations.get(RELEASE_ID_ANNOTATION) != fence.release_id:
                raise KubernetesScaleConflictError(
                    "workload release changed since the scaling decision"
                )
            if observed.annotations.get(MODEL_REVISION_ANNOTATION) != fence.model_revision:
                raise KubernetesScaleConflictError(
                    "workload model revision changed since the scaling decision"
                )
            if (
                decision.action is ScalingDecisionAction.SCALE_DOWN
                and observed.statefulset_start_ordinal != 0
            ):
                raise KubernetesScaleConflictError(
                    "scale-down requires the standard zero StatefulSet start ordinal"
                )
            if (
                decision.action is ScalingDecisionAction.SCALE_UP
                and decision.prewarm_plan is not None
                and observed.annotations.get(CACHE_PLACEMENT_BINDING_ANNOTATION)
                != decision.prewarm_plan.snapshot.placement_binding_id
            ):
                raise KubernetesScaleConflictError(
                    "workload cache placement binding does not match the prewarm plan"
                )

            if decision.action is ScalingDecisionAction.HOLD:
                if observed.generation != fence.workload_generation:
                    raise KubernetesScaleConflictError(
                        "workload generation changed since the scaling decision"
                    )
                scale_result = self._scale_result(
                    decision=decision,
                    target=target,
                    previous=observed.replicas,
                    resulting=observed.replicas,
                    applied=False,
                    resource_version_before=observed.resource_version,
                    resource_version_after=observed.resource_version,
                )
                return KubernetesFencedScaleResult(
                    scale=scale_result,
                    decision=decision,
                    authority=authority,
                    fence=fence,
                    workload_generation_before=observed.generation,
                    workload_generation_after=observed.generation,
                )

            assert decision.decision_generation is not None
            stored_decision = self._stored_decision(observed)
            expected_decision = (
                decision.decision_generation,
                decision.decision_id,
                decision.fingerprint,
            )
            if stored_decision == expected_decision:
                if observed.generation != fence.workload_generation + 1:
                    raise KubernetesScaleConflictError(
                        "workload generation changed after the recorded scaling decision"
                    )
                if observed.replicas != decision.desired_replicas:
                    raise KubernetesScaleConflictError(
                        "recorded scaling decision no longer matches live replicas"
                    )
                if decision.action is ScalingDecisionAction.SCALE_DOWN:
                    assert decision.drain_plan is not None
                    remaining_pods = self._observe_drain_pods(
                        decision.drain_plan,
                        headers=headers,
                        allow_absent=True,
                        allow_deleting=True,
                        allow_released=True,
                    )
                    if remaining_pods:
                        authority = self._release_drain_pods(
                            decision.drain_plan,
                            remaining_pods,
                            authority=authority,
                            reauthorize=reauthorize,
                            headers=headers,
                        )
                scale_result = self._scale_result(
                    decision=decision,
                    target=target,
                    previous=observed.replicas,
                    resulting=observed.replicas,
                    applied=False,
                    resource_version_before=observed.resource_version,
                    resource_version_after=observed.resource_version,
                )
                return KubernetesFencedScaleResult(
                    scale=scale_result,
                    decision=decision,
                    authority=authority,
                    fence=fence,
                    successor_cleanup=successor_cleanup,
                    workload_generation_before=observed.generation,
                    workload_generation_after=observed.generation,
                )
            if successor_cleanup:
                raise KubernetesScaleConflictError(
                    "successor leader may only finish an already applied scale-down"
                )
            if (
                stored_decision is not None
                and stored_decision[0] == decision.decision_generation
            ):
                raise KubernetesScaleConflictError(
                    "decision generation is bound to a different decision fingerprint"
                )
            if observed.generation != fence.workload_generation:
                raise KubernetesScaleConflictError(
                    "workload generation changed since the scaling decision"
                )
            if (
                stored_decision is not None
                and decision.decision_generation <= stored_decision[0]
            ):
                raise KubernetesScaleConflictError(
                    "scaling decision generation must advance monotonically"
                )
            observed_replicas = decision.window.observations[-1].runners.current_replicas
            if observed.replicas != observed_replicas:
                raise KubernetesScaleConflictError(
                    "live replicas changed since the scaling decision observation"
                )
            if observed.replicas == decision.desired_replicas:
                raise KubernetesScaleConflictError(
                    "desired replicas were reached without the matching decision fence"
                )

            annotations = {
                SCALE_DECISION_GENERATION_ANNOTATION: str(decision.decision_generation),
                SCALE_DECISION_ID_ANNOTATION: decision.decision_id,
                SCALE_DECISION_FINGERPRINT_ANNOTATION: decision.fingerprint,
            }
            patch: list[dict[str, object]] = [
                {
                    "op": "test",
                    "path": "/metadata/resourceVersion",
                    "value": observed.resource_version,
                },
                {"op": "test", "path": "/metadata/uid", "value": observed.uid},
                {
                    "op": "test",
                    "path": "/metadata/generation",
                    "value": observed.generation,
                },
                {
                    "op": "test",
                    "path": self._annotation_path(SCALE_ELECTION_ID_ANNOTATION),
                    "value": authority.election_id,
                },
                {
                    "op": "test",
                    "path": self._annotation_path(SCALE_FENCING_TOKEN_ANNOTATION),
                    "value": str(authority.fencing_token),
                },
                *(
                    [
                        {
                            "op": "test",
                            "path": self._annotation_path(
                                CACHE_PLACEMENT_BINDING_ANNOTATION
                            ),
                            "value": decision.prewarm_plan.snapshot.placement_binding_id,
                        }
                    ]
                    if decision.action is ScalingDecisionAction.SCALE_UP
                    and decision.prewarm_plan is not None
                    else []
                ),
                {
                    "op": "test",
                    "path": "/spec/replicas",
                    "value": observed.replicas,
                },
                {
                    "op": "replace",
                    "path": "/spec/replicas",
                    "value": decision.desired_replicas,
                },
            ]
            patch.extend(
                {
                    "op": "add",
                    "path": self._annotation_path(name),
                    "value": value,
                }
                for name, value in annotations.items()
            )
            held_drain_pods: tuple[_DrainPodSnapshot, ...] = ()
            authority = self._reauthorize(authority, reauthorize)
            if decision.action is ScalingDecisionAction.SCALE_UP:
                refreshed_quota = self._reauthorize_quota(
                    decision,
                    reauthorize_quota,
                    authority=authority,
                )
                refreshed_prewarm = self._reauthorize_prewarm(
                    decision,
                    reauthorize_prewarm,
                    authority=authority,
                )
                if (
                    refreshed_prewarm.quota_target_replicas
                    != refreshed_quota.admitted_replicas
                ):
                    raise KubernetesScaleConflictError(
                        "final quota and prewarm authorities disagree"
                    )
            elif decision.action is ScalingDecisionAction.SCALE_DOWN:
                refreshed_drain = self._reauthorize_drain(
                    decision,
                    reauthorize_drain,
                    authority=authority,
                )
                held_drain_pods = self._observe_drain_pods(
                    refreshed_drain,
                    headers=headers,
                    allow_absent=False,
                    allow_deleting=False,
                    allow_released=False,
                )
                authority = self._reauthorize(authority, reauthorize)
                final_drain = self._reauthorize_drain(
                    decision,
                    reauthorize_drain,
                    authority=authority,
                )
                refreshed_candidate_times = {
                    candidate.workload_ordinal: candidate.status.observed_at
                    for candidate in refreshed_drain.selected_candidates
                }
                final_candidate_times = {
                    candidate.workload_ordinal: candidate.status.observed_at
                    for candidate in final_drain.selected_candidates
                }
                if (
                    final_drain.snapshot.observed_at
                    < refreshed_drain.snapshot.observed_at
                    or final_drain.source_observed_at
                    < refreshed_drain.source_observed_at
                    or final_drain.snapshot.drain_revision
                    < refreshed_drain.snapshot.drain_revision
                    or any(
                        final_candidate_times[ordinal] < observed_at
                        for ordinal, observed_at in refreshed_candidate_times.items()
                    )
                ):
                    raise KubernetesScaleConflictError(
                        "drain authority rolled back during Pod hold verification"
                    )
            response = self._client.patch(
                url,
                headers={**headers, "Content-Type": "application/json-patch+json"},
                json=patch,
            )
            if response.status_code in {409, 422}:
                raise KubernetesScaleConflictError(
                    "Kubernetes workload changed during fenced scale mutation"
                )
            response.raise_for_status()
            updated = self._parse_workload(self._response_payload(response), target=target)
            if (
                updated.uid != observed.uid
                or updated.replicas != decision.desired_replicas
                or updated.statefulset_start_ordinal
                != observed.statefulset_start_ordinal
                or updated.resource_version == observed.resource_version
                or updated.generation != observed.generation + 1
                or self._stored_authority(updated)
                != (authority.election_id, authority.fencing_token)
                or self._stored_decision(updated) != expected_decision
                or updated.annotations.get(RELEASE_ID_ANNOTATION) != fence.release_id
                or updated.annotations.get(MODEL_REVISION_ANNOTATION) != fence.model_revision
                or (
                    decision.action is ScalingDecisionAction.SCALE_UP
                    and decision.prewarm_plan is not None
                    and updated.annotations.get(CACHE_PLACEMENT_BINDING_ANNOTATION)
                    != decision.prewarm_plan.snapshot.placement_binding_id
                )
            ):
                raise InvalidKubernetesScaleResponseError(
                    "fenced scale response violated the mutation contract"
                )
            if decision.action is ScalingDecisionAction.SCALE_DOWN:
                assert decision.drain_plan is not None
                authority = self._release_drain_pods(
                    decision.drain_plan,
                    held_drain_pods,
                    authority=authority,
                    reauthorize=reauthorize,
                    headers=headers,
                )
            scale_result = self._scale_result(
                decision=decision,
                target=target,
                previous=observed.replicas,
                resulting=updated.replicas,
                applied=True,
                resource_version_before=observed.resource_version,
                resource_version_after=updated.resource_version,
            )
            return KubernetesFencedScaleResult(
                scale=scale_result,
                decision=decision,
                authority=authority,
                fence=fence,
                workload_generation_before=observed.generation,
                workload_generation_after=updated.generation,
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
