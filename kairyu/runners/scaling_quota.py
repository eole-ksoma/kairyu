"""Quota and Kueue admission contracts for Runner scale-up decisions."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.runners.scaling import _MAX_SIGNED_BIGINT

_MAX_GPU_CAPACITY = 10_000_000
_MAX_SIGNED_INT32 = 2**31 - 1
_TARGET_KIND_ANNOTATION = "kairyu.ai/scale-target-kind"
_TARGET_NAMESPACE_ANNOTATION = "kairyu.ai/scale-target-namespace"
_TARGET_NAME_ANNOTATION = "kairyu.ai/scale-target-name"
_TARGET_UID_ANNOTATION = "kairyu.ai/scale-target-uid"


def _non_empty(value: str, *, name: str) -> str:
    if not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL")
    return value


def _integer(value: object, *, name: str) -> object:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _aware(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def kueue_scaling_workload_name(
    *,
    target_kind: str,
    target_namespace: str,
    target_name: str,
    target_uid: str,
) -> str:
    """Derive the immutable Kueue reservation name for one target lifetime."""

    target_kind = _non_empty(target_kind, name="target_kind")
    if target_kind not in {"Deployment", "StatefulSet"}:
        raise ValueError("target_kind must be Deployment or StatefulSet")
    identity = (
        target_kind,
        _non_empty(target_namespace, name="target_namespace"),
        _non_empty(target_name, name="target_name"),
        _non_empty(target_uid, name="target_uid"),
    )
    digest = hashlib.sha256("\x00".join(identity).encode()).hexdigest()
    return f"kairyu-scale-{digest}"


class ScalingQuotaScope(StrEnum):
    """Nested GPU budgets that every scale-up must satisfy."""

    CLUSTER = "cluster"
    MODEL_FAMILY = "model_family"
    TENANT_MODEL = "tenant_model"


class ScalingQuotaConstraint(StrEnum):
    """Ordered reasons that constrained an admitted replica count."""

    CLUSTER = "cluster"
    MODEL_FAMILY = "model_family"
    TENANT_MODEL = "tenant_model"
    KUEUE_ADMISSION = "kueue_admission"


class ScalingQuotaLimit(BaseModel):
    """One nested GPU budget after protecting higher-priority reservations.

    ``used_gpus_excluding_target`` excludes the target workload so the derived
    ceiling is the target's total permitted allocation, not merely an increment.
    ``reserved_gpus_for_higher_priority`` is computed by the quota controller
    from pending/running higher-priority tenants and is unavailable to this
    target even when the physical GPUs are currently idle.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    scope: ScalingQuotaScope
    quota_name: str = Field(max_length=255)
    hard_limit_gpus: int = Field(ge=0, le=_MAX_GPU_CAPACITY)
    used_gpus_excluding_target: int = Field(ge=0, le=_MAX_GPU_CAPACITY)
    reserved_gpus_for_higher_priority: int = Field(
        default=0,
        ge=0,
        le=_MAX_GPU_CAPACITY,
    )

    @field_validator("quota_name")
    @classmethod
    def validate_quota_name(cls, value: str) -> str:
        return _non_empty(value, name="quota_name")

    @field_validator(
        "hard_limit_gpus",
        "used_gpus_excluding_target",
        "reserved_gpus_for_higher_priority",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        return _integer(value, name=info.field_name)

    @property
    def available_target_gpus(self) -> int:
        validated = type(self).model_validate(self.model_dump())
        return max(
            0,
            validated.hard_limit_gpus
            - validated.used_gpus_excluding_target
            - validated.reserved_gpus_for_higher_priority,
        )


class KueueScalingAdmission(BaseModel):
    """Exact Kueue Workload admission used as a scale-up capacity fence."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-kueue-scaling-admission-v1"] = (
        "runner-kueue-scaling-admission-v1"
    )
    api_version: Literal[
        "kueue.x-k8s.io/v1beta1",
        "kueue.x-k8s.io/v1beta2",
    ]
    namespace: str = Field(max_length=253)
    workload_name: str = Field(max_length=253)
    workload_uid: str = Field(max_length=255)
    workload_generation: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    resource_version: str = Field(max_length=255)
    target_kind: Literal["Deployment", "StatefulSet"]
    target_namespace: str = Field(max_length=253)
    target_name: str = Field(max_length=253)
    target_uid: str = Field(max_length=255)
    local_queue: str = Field(max_length=253)
    cluster_queue: str | None = Field(default=None, max_length=253)
    pod_set_name: str = Field(max_length=253)
    resource_flavor: str | None = Field(default=None, max_length=253)
    resource_name: str = Field(default="nvidia.com/gpu", max_length=253)
    priority_class_name: str | None = Field(default=None, max_length=253)
    priority_class_group: str | None = Field(default=None, max_length=253)
    priority_class_kind: str | None = Field(default=None, max_length=253)
    priority_class_source: str | None = Field(default=None, max_length=253)
    priority: int = Field(ge=-(2**31), le=_MAX_SIGNED_INT32)
    admitted: bool
    admitted_pods: int = Field(ge=0, le=100_000)
    admitted_gpus: int = Field(ge=0, le=_MAX_GPU_CAPACITY)

    @field_validator(
        "namespace",
        "workload_name",
        "workload_uid",
        "resource_version",
        "target_namespace",
        "target_name",
        "target_uid",
        "local_queue",
        "pod_set_name",
        "resource_name",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator(
        "cluster_queue",
        "resource_flavor",
        "priority_class_name",
        "priority_class_group",
        "priority_class_kind",
        "priority_class_source",
    )
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _non_empty(value, name=info.field_name)

    @field_validator(
        "workload_generation",
        "priority",
        "admitted_pods",
        "admitted_gpus",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        return _integer(value, name=info.field_name)

    @field_validator("admitted", mode="before")
    @classmethod
    def validate_boolean(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("admitted must be a boolean")
        return value

    @model_validator(mode="after")
    def validate_admission(self) -> KueueScalingAdmission:
        expected_workload_name = kueue_scaling_workload_name(
            target_kind=self.target_kind,
            target_namespace=self.target_namespace,
            target_name=self.target_name,
            target_uid=self.target_uid,
        )
        if self.workload_name != expected_workload_name:
            raise ValueError(
                "Kueue Workload name must bind the immutable scale target identity"
            )
        if self.admitted != (self.admitted_gpus > 0 and self.admitted_pods > 0):
            raise ValueError(
                "admitted must be true exactly when pod and GPU capacity are positive"
            )
        if not self.admitted and (self.admitted_gpus != 0 or self.admitted_pods != 0):
            raise ValueError("non-admitted Kueue capacity must be zero")
        if self.admitted and (self.cluster_queue is None or self.resource_flavor is None):
            raise ValueError("admitted Kueue capacity requires queue and flavor assignment")
        if not self.admitted and (
            self.cluster_queue is not None or self.resource_flavor is not None
        ):
            raise ValueError("non-admitted Kueue capacity cannot carry an assignment")
        if self.priority_class_name is None:
            if any(
                value is not None
                for value in (
                    self.priority_class_group,
                    self.priority_class_kind,
                    self.priority_class_source,
                )
            ):
                raise ValueError("priority class metadata requires a priority class name")
        elif self.api_version == "kueue.x-k8s.io/v1beta2":
            if (
                self.priority_class_group is None
                or self.priority_class_kind is None
                or self.priority_class_source is not None
            ):
                raise ValueError("v1beta2 priority requires group and kind")
        elif (
            self.priority_class_source is None
            or self.priority_class_group is not None
            or self.priority_class_kind is not None
        ):
            raise ValueError("v1beta1 priority requires priorityClassSource")
        return self


def _mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be an object")
    return value


def _required_string(mapping: Mapping[str, Any], key: str, *, owner: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{owner} requires string {key!r}")
    return _non_empty(value, name=f"{owner}.{key}")


def _gpu_quantity(value: Any, *, resource_name: str) -> int:
    if type(value) is int:
        quantity = value
    elif isinstance(value, str) and value.isascii() and value.isdigit():
        quantity = int(value)
    else:
        raise ValueError(f"Kueue {resource_name!r} usage must be a whole GPU quantity")
    if not 0 <= quantity <= _MAX_GPU_CAPACITY:
        raise ValueError(f"Kueue {resource_name!r} usage is outside the supported range")
    return quantity


def parse_kueue_scaling_admission(
    payload: Any,
    *,
    pod_set_name: str,
    resource_name: str = "nvidia.com/gpu",
) -> KueueScalingAdmission:
    """Parse one Kueue Workload into the exact GPU admission decision input."""

    pod_set_name = _non_empty(pod_set_name, name="pod_set_name")
    resource_name = _non_empty(resource_name, name="resource_name")
    root = _mapping(payload, owner="Kueue Workload")
    api_version = root.get("apiVersion")
    if api_version not in {
        "kueue.x-k8s.io/v1beta1",
        "kueue.x-k8s.io/v1beta2",
    }:
        raise ValueError("Kueue Workload uses an unsupported apiVersion")
    if root.get("kind") != "Workload":
        raise ValueError("Kueue admission input must be kind Workload")
    metadata = _mapping(root.get("metadata"), owner="Workload metadata")
    annotations = _mapping(
        metadata.get("annotations"),
        owner="Workload metadata.annotations",
    )
    spec = _mapping(root.get("spec"), owner="Workload spec")
    status = _mapping(root.get("status", {}), owner="Workload status")
    generation = metadata.get("generation")
    priority = spec.get("priority")
    active = spec.get("active", True)
    if type(generation) is not int or not 1 <= generation <= _MAX_SIGNED_BIGINT:
        raise ValueError("Workload metadata.generation must be a positive integer")
    if type(priority) is not int or not -(2**31) <= priority <= _MAX_SIGNED_INT32:
        raise ValueError("Workload spec.priority must be a signed 32-bit integer")
    if type(active) is not bool:
        raise ValueError("Workload spec.active must be a boolean")
    if api_version == "kueue.x-k8s.io/v1beta2":
        priority_ref = spec.get("priorityClassRef")
        if priority_ref is None:
            priority_class_name = None
            priority_class_group = None
            priority_class_kind = None
        else:
            priority_ref = _mapping(priority_ref, owner="Workload priorityClassRef")
            priority_class_name = _required_string(
                priority_ref,
                "name",
                owner="Workload priorityClassRef",
            )
            priority_class_group = _required_string(
                priority_ref,
                "group",
                owner="Workload priorityClassRef",
            )
            priority_class_kind = _required_string(
                priority_ref,
                "kind",
                owner="Workload priorityClassRef",
            )
        priority_class_source = None
    else:
        priority_name = spec.get("priorityClassName")
        if priority_name is not None and not isinstance(priority_name, str):
            raise ValueError("Workload spec.priorityClassName must be a string")
        priority_class_name = (
            _non_empty(priority_name, name="priorityClassName")
            if priority_name
            else None
        )
        priority_source = spec.get("priorityClassSource")
        if priority_source is not None and not isinstance(priority_source, str):
            raise ValueError("Workload spec.priorityClassSource must be a string")
        priority_class_source = (
            _non_empty(priority_source, name="priorityClassSource")
            if priority_source
            else None
        )
        priority_class_group = None
        priority_class_kind = None
    conditions = status.get("conditions", [])
    if not isinstance(conditions, list) or not all(
        isinstance(condition, dict) for condition in conditions
    ):
        raise ValueError("Workload status.conditions must be an object list")
    admitted_conditions = [
        condition for condition in conditions if condition.get("type") == "Admitted"
    ]
    if len(admitted_conditions) > 1:
        raise ValueError("Workload status has duplicate Admitted conditions")
    if admitted_conditions and admitted_conditions[0].get("status") == "True":
        observed_generation = admitted_conditions[0].get("observedGeneration")
        if type(observed_generation) is not int or observed_generation != generation:
            raise ValueError(
                "admitted Workload condition must observe metadata.generation"
            )
    admitted_condition = active and bool(
        admitted_conditions and admitted_conditions[0].get("status") == "True"
    )
    admission_payload = status.get("admission")
    if not admitted_condition or admission_payload is None:
        return KueueScalingAdmission(
            api_version=api_version,
            namespace=_required_string(metadata, "namespace", owner="Workload metadata"),
            workload_name=_required_string(metadata, "name", owner="Workload metadata"),
            workload_uid=_required_string(metadata, "uid", owner="Workload metadata"),
            workload_generation=generation,
            resource_version=_required_string(
                metadata,
                "resourceVersion",
                owner="Workload metadata",
            ),
            target_kind=_required_string(
                annotations,
                _TARGET_KIND_ANNOTATION,
                owner="Workload metadata.annotations",
            ),
            target_namespace=_required_string(
                annotations,
                _TARGET_NAMESPACE_ANNOTATION,
                owner="Workload metadata.annotations",
            ),
            target_name=_required_string(
                annotations,
                _TARGET_NAME_ANNOTATION,
                owner="Workload metadata.annotations",
            ),
            target_uid=_required_string(
                annotations,
                _TARGET_UID_ANNOTATION,
                owner="Workload metadata.annotations",
            ),
            local_queue=_required_string(spec, "queueName", owner="Workload spec"),
            pod_set_name=pod_set_name,
            priority_class_name=priority_class_name,
            priority_class_group=priority_class_group,
            priority_class_kind=priority_class_kind,
            priority_class_source=priority_class_source,
            priority=priority,
            admitted=False,
            admitted_pods=0,
            admitted_gpus=0,
            resource_name=resource_name,
        )
    admission = _mapping(admission_payload, owner="Workload status.admission")
    assignments = admission.get("podSetAssignments")
    if not isinstance(assignments, list) or not all(
        isinstance(assignment, dict) for assignment in assignments
    ):
        raise ValueError("Workload admission podSetAssignments must be an object list")
    selected = [
        assignment for assignment in assignments if assignment.get("name") == pod_set_name
    ]
    if len(selected) != 1:
        raise ValueError("Kueue admission must assign the requested pod set exactly once")
    assignment = selected[0]
    flavors = _mapping(assignment.get("flavors"), owner="PodSetAssignment flavors")
    usage = _mapping(
        assignment.get("resourceUsage"),
        owner="PodSetAssignment resourceUsage",
    )
    flavor = _required_string(flavors, resource_name, owner="PodSetAssignment flavors")
    if resource_name not in usage:
        raise ValueError(f"PodSetAssignment resourceUsage requires {resource_name!r}")
    admitted_gpus = _gpu_quantity(usage[resource_name], resource_name=resource_name)
    admitted_pods = assignment.get("count")
    if admitted_pods is None:
        pod_sets = spec.get("podSets")
        if not isinstance(pod_sets, list) or not all(
            isinstance(pod_set, dict) for pod_set in pod_sets
        ):
            raise ValueError("Workload spec.podSets must be an object list")
        matching_pod_sets = [
            pod_set for pod_set in pod_sets if pod_set.get("name") == pod_set_name
        ]
        if len(matching_pod_sets) != 1:
            raise ValueError("Workload spec must define the requested pod set exactly once")
        admitted_pods = matching_pod_sets[0].get("count")
    if type(admitted_pods) is not int or not 1 <= admitted_pods <= 100_000:
        raise ValueError("Kueue admitted pod count must be a positive integer")
    return KueueScalingAdmission(
        api_version=api_version,
        namespace=_required_string(metadata, "namespace", owner="Workload metadata"),
        workload_name=_required_string(metadata, "name", owner="Workload metadata"),
        workload_uid=_required_string(metadata, "uid", owner="Workload metadata"),
        workload_generation=generation,
        resource_version=_required_string(
            metadata,
            "resourceVersion",
            owner="Workload metadata",
        ),
        target_kind=_required_string(
            annotations,
            _TARGET_KIND_ANNOTATION,
            owner="Workload metadata.annotations",
        ),
        target_namespace=_required_string(
            annotations,
            _TARGET_NAMESPACE_ANNOTATION,
            owner="Workload metadata.annotations",
        ),
        target_name=_required_string(
            annotations,
            _TARGET_NAME_ANNOTATION,
            owner="Workload metadata.annotations",
        ),
        target_uid=_required_string(
            annotations,
            _TARGET_UID_ANNOTATION,
            owner="Workload metadata.annotations",
        ),
        local_queue=_required_string(spec, "queueName", owner="Workload spec"),
        cluster_queue=_required_string(
            admission,
            "clusterQueue",
            owner="Workload status.admission",
        ),
        pod_set_name=pod_set_name,
        resource_flavor=flavor,
        resource_name=resource_name,
        priority_class_name=priority_class_name,
        priority_class_group=priority_class_group,
        priority_class_kind=priority_class_kind,
        priority_class_source=priority_class_source,
        priority=priority,
        admitted=True,
        admitted_pods=admitted_pods,
        admitted_gpus=admitted_gpus,
    )


class ScalingQuotaSnapshot(BaseModel):
    """Source-timestamped capacity evidence for one tenant/model target."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-scaling-quota-snapshot-v1"] = (
        "runner-scaling-quota-snapshot-v1"
    )
    snapshot_id: str = Field(max_length=255)
    quota_revision: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    observed_at: datetime
    tenant_id: str = Field(max_length=255)
    model_class: str = Field(max_length=128)
    model_family: str = Field(max_length=128)
    gpus_per_replica: int = Field(ge=1, le=1024)
    target_reserved_gpus: int = Field(default=0, ge=0, le=_MAX_GPU_CAPACITY)
    limits: tuple[ScalingQuotaLimit, ...] = Field(min_length=3, max_length=3)
    kueue: KueueScalingAdmission

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="observed_at")

    @field_validator("snapshot_id", "tenant_id", "model_class", "model_family")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator(
        "quota_revision",
        "gpus_per_replica",
        "target_reserved_gpus",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        return _integer(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_limits(self) -> ScalingQuotaSnapshot:
        expected = tuple(ScalingQuotaScope)
        actual = tuple(limit.scope for limit in self.limits)
        if actual != expected:
            raise ValueError(
                "quota limits must contain cluster, model_family, and tenant_model "
                "in canonical order"
            )
        if len({limit.quota_name for limit in self.limits}) != len(self.limits):
            raise ValueError("quota limits must use unique quota names")
        if self.kueue.admitted:
            if self.target_reserved_gpus != self.kueue.admitted_gpus:
                raise ValueError(
                    "target reservation must equal the authoritative Kueue admission"
                )
        elif self.target_reserved_gpus != 0:
            raise ValueError("non-admitted Kueue workload cannot reserve target GPUs")
        if any(
            self.target_reserved_gpus > limit.available_target_gpus
            for limit in self.limits
        ):
            raise ValueError(
                "the Kueue target reservation must fit every nested hard limit"
            )
        if (
            self.kueue.admitted
            and self.kueue.admitted_pods * self.gpus_per_replica
            != self.kueue.admitted_gpus
        ):
            raise ValueError(
                "Kueue pod and GPU admission must match gpus_per_replica"
            )
        return self

    def replica_ceilings(self) -> tuple[tuple[ScalingQuotaConstraint, int], ...]:
        """Return all independently auditable total-replica ceilings."""

        validated = type(self).model_validate(self.model_dump())
        scope_ceilings = tuple(
            (
                ScalingQuotaConstraint(limit.scope.value),
                limit.available_target_gpus // validated.gpus_per_replica,
            )
            for limit in validated.limits
        )
        kueue_ceiling = (
            validated.kueue.admitted_gpus // validated.gpus_per_replica
            if validated.kueue.admitted
            else 0
        )
        return (*scope_ceilings, (ScalingQuotaConstraint.KUEUE_ADMISSION, kueue_ceiling))


class ScalingQuotaAdmission(BaseModel):
    """Deterministic result of applying all quota fences to one scale-up intent."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-scaling-quota-admission-v1"] = (
        "runner-scaling-quota-admission-v1"
    )
    snapshot: ScalingQuotaSnapshot
    current_replicas: int = Field(ge=0, le=100_000)
    requested_replicas: int = Field(ge=0, le=100_000)
    quota_ceiling_replicas: int = Field(ge=0, le=100_000)
    admitted_replicas: int = Field(ge=0, le=100_000)
    constrained_by: tuple[ScalingQuotaConstraint, ...] = Field(
        default=(),
        max_length=4,
    )

    @field_validator(
        "current_replicas",
        "requested_replicas",
        "quota_ceiling_replicas",
        "admitted_replicas",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        return _integer(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_result(self) -> ScalingQuotaAdmission:
        if self.requested_replicas <= self.current_replicas:
            raise ValueError("quota admission only accepts scale-up intents")
        ceilings = self.snapshot.replica_ceilings()
        expected_ceiling = min(value for _, value in ceilings)
        expected_admitted = min(
            self.requested_replicas,
            max(self.current_replicas, expected_ceiling),
        )
        expected_constraints = tuple(
            constraint
            for constraint, ceiling in ceilings
            if ceiling < self.requested_replicas
        )
        if self.quota_ceiling_replicas != expected_ceiling:
            raise ValueError("quota_ceiling_replicas must match all quota inputs")
        if self.admitted_replicas != expected_admitted:
            raise ValueError("admitted_replicas must be the fail-safe quota clamp")
        if self.constrained_by != expected_constraints:
            raise ValueError("constrained_by must list every limiting quota in order")
        return self

    @property
    def constrained(self) -> bool:
        validated = type(self).model_validate(self.model_dump())
        return bool(validated.constrained_by)


def admit_scaling_quota(
    snapshot: ScalingQuotaSnapshot,
    *,
    current_replicas: int,
    requested_replicas: int,
) -> ScalingQuotaAdmission:
    """Clamp a scale-up intent without ever turning quota pressure into scale-down."""

    if not isinstance(snapshot, ScalingQuotaSnapshot):
        raise TypeError("snapshot must be a ScalingQuotaSnapshot")
    current_replicas = _integer(current_replicas, name="current_replicas")
    requested_replicas = _integer(requested_replicas, name="requested_replicas")
    assert isinstance(current_replicas, int)
    assert isinstance(requested_replicas, int)
    if not 0 <= current_replicas <= 100_000:
        raise ValueError("current_replicas must be in [0, 100000]")
    if not 0 <= requested_replicas <= 100_000:
        raise ValueError("requested_replicas must be in [0, 100000]")
    if requested_replicas <= current_replicas:
        raise ValueError("quota admission only accepts scale-up intents")
    snapshot = ScalingQuotaSnapshot.model_validate(snapshot.model_dump())
    ceilings = snapshot.replica_ceilings()
    quota_ceiling = min(value for _, value in ceilings)
    admitted = min(requested_replicas, max(current_replicas, quota_ceiling))
    constraints = tuple(
        constraint
        for constraint, ceiling in ceilings
        if ceiling < requested_replicas
    )
    return ScalingQuotaAdmission(
        snapshot=snapshot,
        current_replicas=current_replicas,
        requested_replicas=requested_replicas,
        quota_ceiling_replicas=quota_ceiling,
        admitted_replicas=admitted,
        constrained_by=constraints,
    )
