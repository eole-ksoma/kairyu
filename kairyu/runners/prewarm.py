"""Cache-aware prewarm contracts for staged Runner scale-out."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.artifacts.placement_hint import NodeModelCachePlacementHintSnapshot
from kairyu.runners.scaling import _MAX_SIGNED_BIGINT

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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


class ModelCachePlacementState(StrEnum):
    """Observed state of one compatible replica-sized cache placement."""

    ABSENT = "absent"
    FILLING = "filling"
    READY = "ready"
    FAILED = "failed"


class ScalingPrewarmAction(StrEnum):
    """Immediate action derived from a final quota-admitted scale target."""

    CACHE_FILL = "cache_fill"
    RUNNER_START = "runner_start"
    HOLD = "hold"


class ModelCachePlacement(BaseModel):
    """One replica-sized placement from an explicitly approved GPU profile."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    placement_id: str = Field(max_length=255)
    node_name: str = Field(max_length=253)
    resource_flavor: str = Field(max_length=253)
    profile_id: str = Field(max_length=255)
    compatibility_approval_id: str = Field(max_length=255)
    state: ModelCachePlacementState
    assigned: bool = False
    healthy: bool = True
    schedulable: bool = True

    @field_validator(
        "placement_id",
        "node_name",
        "resource_flavor",
        "profile_id",
        "compatibility_approval_id",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator("assigned", "healthy", "schedulable", mode="before")
    @classmethod
    def validate_boolean(cls, value: object, info) -> object:
        if type(value) is not bool:
            raise ValueError(f"{info.field_name} must be a boolean")
        return value


class ModelCachePlacementCandidate(BaseModel):
    """Controller-owned placement facts before advisory cache state is joined."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    placement_id: str = Field(max_length=255)
    node_name: str = Field(max_length=253)
    resource_flavor: str = Field(max_length=253)
    profile_id: str = Field(max_length=255)
    compatibility_approval_id: str = Field(max_length=255)
    assigned: bool = False
    healthy: bool = True
    schedulable: bool = True

    @field_validator(
        "placement_id",
        "node_name",
        "resource_flavor",
        "profile_id",
        "compatibility_approval_id",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator("assigned", "healthy", "schedulable", mode="before")
    @classmethod
    def validate_boolean(cls, value: object, info) -> object:
        if type(value) is not bool:
            raise ValueError(f"{info.field_name} must be a boolean")
        return value


class ScalingPrewarmSnapshot(BaseModel):
    """Source-versioned cache inventory for one immutable model artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-scaling-prewarm-snapshot-v1"] = (
        "runner-scaling-prewarm-snapshot-v1"
    )
    snapshot_id: str = Field(max_length=255)
    cache_revision: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    observed_at: datetime
    model_class: str = Field(max_length=128)
    model_revision: str = Field(max_length=255)
    artifact_digest: str = Field(max_length=255)
    placement_binding_id: str = Field(max_length=255)
    placements: tuple[ModelCachePlacement, ...] = Field(
        default=(),
        max_length=100_000,
    )

    @field_validator(
        "snapshot_id",
        "model_class",
        "model_revision",
        "artifact_digest",
        "placement_binding_id",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator("cache_revision", mode="before")
    @classmethod
    def validate_revision(cls, value: object) -> object:
        return _integer(value, name="cache_revision")

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="observed_at")

    @model_validator(mode="after")
    def validate_placements(self) -> ScalingPrewarmSnapshot:
        placement_ids = tuple(placement.placement_id for placement in self.placements)
        if len(set(placement_ids)) != len(placement_ids):
            raise ValueError("cache placements must use unique placement IDs")
        if placement_ids != tuple(sorted(placement_ids)):
            raise ValueError("cache placements must use canonical placement-ID order")
        return self


def _derive_plan(
    snapshot: ScalingPrewarmSnapshot,
    *,
    current_replicas: int,
    quota_target_replicas: int,
    resource_flavor: str,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    int,
    ScalingPrewarmAction,
]:
    needed = quota_target_replicas - current_replicas
    eligible = tuple(
        placement
        for placement in snapshot.placements
        if placement.resource_flavor == resource_flavor
        and not placement.assigned
        and placement.healthy
        and placement.schedulable
    )
    ready = tuple(
        placement.placement_id
        for placement in eligible
        if placement.state is ModelCachePlacementState.READY
    )[:needed]
    remaining = needed - len(ready)
    pending = tuple(
        placement.placement_id
        for placement in eligible
        if placement.state is ModelCachePlacementState.FILLING
    )[:remaining]
    remaining -= len(pending)
    cache_fill = tuple(
        placement.placement_id
        for placement in eligible
        if placement.state is ModelCachePlacementState.ABSENT
    )[:remaining]
    unplanned = remaining - len(cache_fill)
    if ready:
        action = ScalingPrewarmAction.RUNNER_START
    elif pending or cache_fill:
        action = ScalingPrewarmAction.CACHE_FILL
    else:
        action = ScalingPrewarmAction.HOLD
    return ready, cache_fill, pending, unplanned, action


class ScalingPrewarmPlan(BaseModel):
    """Auditable split between cache fill and immediately safe Runner start."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-scaling-prewarm-plan-v1"] = "runner-scaling-prewarm-plan-v1"
    snapshot: ScalingPrewarmSnapshot
    resource_flavor: str = Field(max_length=253)
    current_replicas: int = Field(ge=0, le=100_000)
    quota_target_replicas: int = Field(ge=0, le=100_000)
    runner_target_replicas: int = Field(ge=0, le=100_000)
    runner_start_placement_ids: tuple[str, ...] = Field(default=(), max_length=100_000)
    cache_fill_placement_ids: tuple[str, ...] = Field(default=(), max_length=100_000)
    pending_fill_placement_ids: tuple[str, ...] = Field(default=(), max_length=100_000)
    unplanned_replicas: int = Field(default=0, ge=0, le=100_000)
    action: ScalingPrewarmAction

    @field_validator("resource_flavor")
    @classmethod
    def validate_resource_flavor(cls, value: str) -> str:
        return _non_empty(value, name="resource_flavor")

    @field_validator(
        "current_replicas",
        "quota_target_replicas",
        "runner_target_replicas",
        "unplanned_replicas",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        return _integer(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_plan(self) -> ScalingPrewarmPlan:
        if self.quota_target_replicas <= self.current_replicas:
            raise ValueError("prewarm planning requires a scale-up target")
        ready, cache_fill, pending, unplanned, action = _derive_plan(
            self.snapshot,
            current_replicas=self.current_replicas,
            quota_target_replicas=self.quota_target_replicas,
            resource_flavor=self.resource_flavor,
        )
        expected_runner_target = self.current_replicas + len(ready)
        if self.runner_start_placement_ids != ready:
            raise ValueError("runner-start placements must match ready cache capacity")
        if self.cache_fill_placement_ids != cache_fill:
            raise ValueError("cache-fill placements must match absent cache capacity")
        if self.pending_fill_placement_ids != pending:
            raise ValueError("pending-fill placements must match filling cache capacity")
        if self.runner_target_replicas != expected_runner_target:
            raise ValueError("runner_target_replicas must use only ready cache placements")
        if self.unplanned_replicas != unplanned:
            raise ValueError("unplanned_replicas must match eligible placement capacity")
        if self.action is not action:
            raise ValueError("prewarm action must match the staged cache plan")
        return self

    @property
    def cache_ready_for_quota_target(self) -> bool:
        validated = type(self).model_validate(self.model_dump())
        return validated.runner_target_replicas == validated.quota_target_replicas


def plan_cache_aware_scale_up(
    snapshot: ScalingPrewarmSnapshot,
    *,
    current_replicas: int,
    quota_target_replicas: int,
    resource_flavor: str,
) -> ScalingPrewarmPlan:
    """Split a quota-admitted scale-out into cache fill and Runner start."""

    if not isinstance(snapshot, ScalingPrewarmSnapshot):
        raise TypeError("snapshot must be a ScalingPrewarmSnapshot")
    current_replicas = _integer(current_replicas, name="current_replicas")
    quota_target_replicas = _integer(
        quota_target_replicas,
        name="quota_target_replicas",
    )
    assert isinstance(current_replicas, int)
    assert isinstance(quota_target_replicas, int)
    if not 0 <= current_replicas <= 100_000:
        raise ValueError("current_replicas must be in [0, 100000]")
    if not 0 <= quota_target_replicas <= 100_000:
        raise ValueError("quota_target_replicas must be in [0, 100000]")
    if quota_target_replicas <= current_replicas:
        raise ValueError("prewarm planning requires a scale-up target")
    resource_flavor = _non_empty(resource_flavor, name="resource_flavor")
    snapshot = ScalingPrewarmSnapshot.model_validate(snapshot.model_dump())
    ready, cache_fill, pending, unplanned, action = _derive_plan(
        snapshot,
        current_replicas=current_replicas,
        quota_target_replicas=quota_target_replicas,
        resource_flavor=resource_flavor,
    )
    return ScalingPrewarmPlan(
        snapshot=snapshot,
        resource_flavor=resource_flavor,
        current_replicas=current_replicas,
        quota_target_replicas=quota_target_replicas,
        runner_target_replicas=current_replicas + len(ready),
        runner_start_placement_ids=ready,
        cache_fill_placement_ids=cache_fill,
        pending_fill_placement_ids=pending,
        unplanned_replicas=unplanned,
        action=action,
    )


def build_cache_placement_snapshot(
    hints: tuple[NodeModelCachePlacementHintSnapshot, ...],
    candidates: tuple[ModelCachePlacementCandidate, ...],
    *,
    snapshot_id: str,
    cache_revision: int,
    observed_at: datetime,
    model_class: str,
    model_id: str,
    model_revision: str,
    artifact_digest: str,
    placement_binding_id: str,
) -> ScalingPrewarmSnapshot:
    """Join fresh exact-residency hints to controller-owned placement facts."""

    snapshot_id = _non_empty(snapshot_id, name="snapshot_id")
    model_class = _non_empty(model_class, name="model_class")
    model_id = _non_empty(model_id, name="model_id")
    model_revision = _non_empty(model_revision, name="model_revision")
    artifact_digest = _non_empty(artifact_digest, name="artifact_digest")
    placement_binding_id = _non_empty(
        placement_binding_id,
        name="placement_binding_id",
    )
    if len(model_id) > 255:
        raise ValueError("model_id exceeds maximum length")
    if not _SHA256_PATTERN.fullmatch(artifact_digest):
        raise ValueError("artifact_digest must be a lowercase SHA-256 digest")
    observed_at = _aware(observed_at, name="observed_at")
    cache_revision = _integer(cache_revision, name="cache_revision")
    assert isinstance(cache_revision, int)
    if not 1 <= cache_revision <= _MAX_SIGNED_BIGINT:
        raise ValueError("cache_revision must be in [1, 2^63-1]")
    if not isinstance(hints, tuple):
        raise TypeError("hints must be a tuple")
    if not isinstance(candidates, tuple):
        raise TypeError("candidates must be a tuple")
    if len(hints) > 100_000:
        raise ValueError("hints exceed maximum node count")
    if len(candidates) > 100_000:
        raise ValueError("candidates exceed maximum placement count")

    validated_hints_list = []
    for hint in hints:
        if not isinstance(hint, NodeModelCachePlacementHintSnapshot):
            raise TypeError("hints must contain NodeModelCachePlacementHintSnapshot values")
        validated_hints_list.append(
            NodeModelCachePlacementHintSnapshot.model_validate(hint.model_dump())
        )
    validated_hints = tuple(validated_hints_list)
    hint_nodes = tuple(hint.node_id for hint in validated_hints)
    if len(set(hint_nodes)) != len(hint_nodes):
        raise ValueError("placement hints must use unique node IDs")
    hints_by_node = {hint.node_id: hint for hint in validated_hints}

    validated_candidates_list = []
    for candidate in candidates:
        if not isinstance(candidate, ModelCachePlacementCandidate):
            raise TypeError("candidates must contain ModelCachePlacementCandidate values")
        validated_candidates_list.append(
            ModelCachePlacementCandidate.model_validate(candidate.model_dump())
        )
    validated_candidates = tuple(validated_candidates_list)
    placement_ids = tuple(candidate.placement_id for candidate in validated_candidates)
    if len(set(placement_ids)) != len(placement_ids):
        raise ValueError("placement candidates must use unique placement IDs")

    placements = []
    for candidate in sorted(validated_candidates, key=lambda value: value.placement_id):
        hint = hints_by_node.get(candidate.node_name)
        fresh = (
            hint is not None and hint.observed_at <= observed_at and observed_at < hint.valid_until
        )
        resident = (
            hint.resident_for(
                manifest_digest=artifact_digest,
                model_id=model_id,
                model_revision=model_revision,
            )
            if fresh and hint is not None
            else None
        )
        placements.append(
            ModelCachePlacement(
                placement_id=candidate.placement_id,
                node_name=candidate.node_name,
                resource_flavor=candidate.resource_flavor,
                profile_id=candidate.profile_id,
                compatibility_approval_id=candidate.compatibility_approval_id,
                state=(
                    ModelCachePlacementState.READY
                    if resident is not None
                    else ModelCachePlacementState.ABSENT
                ),
                assigned=candidate.assigned,
                healthy=candidate.healthy,
                schedulable=candidate.schedulable,
            )
        )

    return ScalingPrewarmSnapshot(
        snapshot_id=snapshot_id,
        cache_revision=cache_revision,
        observed_at=observed_at,
        model_class=model_class,
        model_revision=model_revision,
        artifact_digest=artifact_digest,
        placement_binding_id=placement_binding_id,
        placements=tuple(placements),
    )
