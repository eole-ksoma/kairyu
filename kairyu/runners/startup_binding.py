"""Immutable cache-to-scheduler evidence for one fenced Runner scale-up."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.artifacts.placement_hint import NodeModelCachePlacementHintSnapshot
from kairyu.runners.prewarm import ModelCachePlacementState, ScalingPrewarmPlan

if TYPE_CHECKING:
    from kairyu.runners.prestage import NodeModelPrestageRecord

_MAX_SIGNED_BIGINT = 2**63 - 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class RunnerCacheStartupBindingError(RuntimeError):
    """Cache, pin, or placement evidence cannot authorize Runner startup."""


def _text(value: str, *, name: str, max_length: int = 255) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL")
    if len(value) > max_length:
        raise ValueError(f"{name} exceeds maximum length")
    return value


def _digest(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _aware(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    assert isinstance(value, int)
    return value


def _canonical_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=lambda value: value.isoformat() if isinstance(value, datetime) else str(value),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RunnerCacheStartupPlacement(BaseModel):
    """One scheduler placement backed by a published deployment-owned cache pin."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    placement_id: str = Field(max_length=255)
    node_name: str = Field(max_length=253)
    resource_flavor: str = Field(max_length=253)
    profile_id: str = Field(max_length=255)
    compatibility_approval_id: str = Field(max_length=255)
    manifest_digest: str = Field(min_length=64, max_length=64)
    pin_owner: str = Field(max_length=255)
    prestage_command_id: str = Field(min_length=64, max_length=64)
    prestage_command_generation: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    hint_index_revision: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    resident_record_generation: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    hint_observed_at: datetime
    hint_valid_until: datetime

    @field_validator(
        "placement_id",
        "node_name",
        "resource_flavor",
        "profile_id",
        "compatibility_approval_id",
        "pin_owner",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        maximum = 253 if info.field_name in {"node_name", "resource_flavor"} else 255
        return _text(value, name=info.field_name, max_length=maximum)

    @field_validator("manifest_digest", "prestage_command_id")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _digest(value, name=info.field_name)

    @field_validator(
        "prestage_command_generation",
        "hint_index_revision",
        "resident_record_generation",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> int:
        return _integer(value, name=info.field_name)

    @field_validator("hint_observed_at", "hint_valid_until")
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        return _aware(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_hint_window(self) -> RunnerCacheStartupPlacement:
        if self.hint_observed_at >= self.hint_valid_until:
            raise ValueError("placement hint expiry must follow its observation")
        return self


class RunnerCacheStartupBinding(BaseModel):
    """Hash-bound decision input consumed by the Kubernetes scheduling step."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-cache-startup-binding-v1"] = "runner-cache-startup-binding-v1"
    binding_id: str = Field(min_length=64, max_length=64)
    decision_id: str = Field(max_length=255)
    decision_fingerprint: str = Field(min_length=64, max_length=64)
    target_id: str = Field(max_length=255)
    target_revision: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    deployment_id: str = Field(max_length=255)
    model_class: str = Field(max_length=128)
    model_id: str = Field(max_length=255)
    model_revision: str = Field(max_length=255)
    manifest_digest: str = Field(min_length=64, max_length=64)
    placement_binding_id: str = Field(max_length=255)
    prewarm_snapshot_id: str = Field(max_length=255)
    prewarm_cache_revision: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    bound_at: datetime
    valid_until: datetime
    placements: tuple[RunnerCacheStartupPlacement, ...] = Field(
        min_length=1,
        max_length=100_000,
    )

    @field_validator(
        "decision_id",
        "target_id",
        "deployment_id",
        "model_class",
        "model_id",
        "model_revision",
        "placement_binding_id",
        "prewarm_snapshot_id",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        maximum = 128 if info.field_name == "model_class" else 255
        return _text(value, name=info.field_name, max_length=maximum)

    @field_validator("binding_id", "decision_fingerprint", "manifest_digest")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _digest(value, name=info.field_name)

    @field_validator("target_revision", "prewarm_cache_revision", mode="before")
    @classmethod
    def validate_integer(cls, value: object, info) -> int:
        return _integer(value, name=info.field_name)

    @field_validator("bound_at", "valid_until")
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        return _aware(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_binding(self) -> RunnerCacheStartupBinding:
        placement_ids = tuple(placement.placement_id for placement in self.placements)
        if placement_ids != tuple(sorted(placement_ids)):
            raise ValueError("startup placements must use canonical placement-ID order")
        if len(set(placement_ids)) != len(placement_ids):
            raise ValueError("startup placements must use unique placement IDs")
        if len({placement.node_name for placement in self.placements}) != len(self.placements):
            raise ValueError("startup placements must use unique nodes")
        if any(placement.manifest_digest != self.manifest_digest for placement in self.placements):
            raise ValueError("startup placements must use the binding artifact")
        expected_expiry = min(placement.hint_valid_until for placement in self.placements)
        if self.valid_until != expected_expiry or not self.bound_at < self.valid_until:
            raise ValueError("binding expiry must equal the earliest live placement hint")
        if any(
            not placement.hint_observed_at <= self.bound_at < placement.hint_valid_until
            for placement in self.placements
        ):
            raise ValueError("every startup placement hint must be live at binding time")
        expected_id = _canonical_digest(self.model_dump(mode="json", exclude={"binding_id"}))
        if self.binding_id != expected_id:
            raise ValueError("binding_id must match the canonical binding payload")
        return self


def build_runner_cache_startup_binding(
    plan: ScalingPrewarmPlan,
    records: tuple[NodeModelPrestageRecord, ...],
    hints: tuple[NodeModelCachePlacementHintSnapshot, ...],
    *,
    decision_id: str,
    decision_fingerprint: str,
    target_id: str,
    target_revision: int,
    deployment_id: str,
    model_id: str,
    bound_at: datetime,
) -> RunnerCacheStartupBinding:
    """Join ready plan slots to exact pin completions and later physical hints."""

    from kairyu.runners.prestage import NodeModelPrestageRecord

    if not isinstance(plan, ScalingPrewarmPlan):
        raise TypeError("plan must be a ScalingPrewarmPlan")
    plan = ScalingPrewarmPlan.model_validate(plan.model_dump())
    if not plan.runner_start_placement_ids:
        raise RunnerCacheStartupBindingError("Runner startup requires ready placements")
    if not isinstance(records, tuple):
        raise TypeError("records must be a tuple")
    if not isinstance(hints, tuple):
        raise TypeError("hints must be a tuple")
    decision_id = _text(decision_id, name="decision_id")
    decision_fingerprint = _digest(decision_fingerprint, name="decision_fingerprint")
    target_id = _text(target_id, name="target_id")
    target_revision = _integer(target_revision, name="target_revision")
    if not 1 <= target_revision <= _MAX_SIGNED_BIGINT:
        raise ValueError("target_revision must be in [1, 2^63-1]")
    deployment_id = _text(deployment_id, name="deployment_id")
    model_id = _text(model_id, name="model_id")
    bound_at = _aware(bound_at, name="bound_at")
    manifest_digest = _digest(plan.snapshot.artifact_digest, name="artifact_digest")

    validated_records = []
    for record in records:
        if not isinstance(record, NodeModelPrestageRecord):
            raise TypeError("records must contain NodeModelPrestageRecord values")
        validated_records.append(NodeModelPrestageRecord.model_validate(record.model_dump()))
    records_by_placement = {record.command.placement_id: record for record in validated_records}
    if len(records_by_placement) != len(validated_records):
        raise RunnerCacheStartupBindingError("pin records must use unique placement IDs")

    validated_hints = []
    for hint in hints:
        if not isinstance(hint, NodeModelCachePlacementHintSnapshot):
            raise TypeError("hints must contain NodeModelCachePlacementHintSnapshot values")
        validated_hints.append(
            NodeModelCachePlacementHintSnapshot.model_validate(hint.model_dump())
        )
    hints_by_node = {hint.node_id: hint for hint in validated_hints}
    if len(hints_by_node) != len(validated_hints):
        raise RunnerCacheStartupBindingError("placement hints must use unique nodes")

    placements_by_id = {placement.placement_id: placement for placement in plan.snapshot.placements}
    bound_placements = []
    for placement_id in plan.runner_start_placement_ids:
        placement = placements_by_id[placement_id]
        record = records_by_placement.get(placement_id)
        hint = hints_by_node.get(placement.node_name)
        if record is None or hint is None:
            raise RunnerCacheStartupBindingError(
                "every Runner-start placement requires a pin record and node hint"
            )
        command = record.command
        expected_pin_owner = f"prestage/{deployment_id}/{placement_id}"
        if record.state is not ModelCachePlacementState.READY:
            raise RunnerCacheStartupBindingError("Runner-start pin record is not ready")
        if record.pin_record_generation is None:
            raise RunnerCacheStartupBindingError(
                "Runner-start pin record lacks cache generation evidence"
            )
        if (
            command.action,
            command.decision_id,
            command.decision_fingerprint,
            command.target_id,
            command.target_revision,
            command.deployment_id,
            command.placement_binding_id,
            command.placement_id,
            command.node_id,
            command.resource_flavor,
            command.profile_id,
            command.compatibility_approval_id,
            command.model_id,
            command.model_revision,
            command.manifest_digest,
            command.pin_owner,
        ) != (
            "ensure",
            decision_id,
            decision_fingerprint,
            target_id,
            target_revision,
            deployment_id,
            plan.snapshot.placement_binding_id,
            placement.placement_id,
            placement.node_name,
            placement.resource_flavor,
            placement.profile_id,
            placement.compatibility_approval_id,
            model_id,
            plan.snapshot.model_revision,
            manifest_digest,
            expected_pin_owner,
        ):
            raise RunnerCacheStartupBindingError(
                "Runner-start pin record does not match the scheduling decision"
            )
        if not record.updated_at <= hint.observed_at <= bound_at < hint.valid_until:
            raise RunnerCacheStartupBindingError(
                "placement hint must publish the completed pin and remain live"
            )
        resident = hint.resident_for(
            manifest_digest=manifest_digest,
            model_id=model_id,
            model_revision=plan.snapshot.model_revision,
        )
        if (
            resident is None
            or not resident.pinned
            or resident.record_generation != record.pin_record_generation
        ):
            raise RunnerCacheStartupBindingError(
                "placement hint must publish the exact owned pin generation"
            )
        if (
            placement.state is not ModelCachePlacementState.READY
            or placement.cache_hint_observed_at != hint.observed_at
            or placement.cache_hint_valid_until != hint.valid_until
            or placement.cache_hint_index_revision != hint.index_revision
        ):
            raise RunnerCacheStartupBindingError(
                "prewarm placement does not match the physical hint used for binding"
            )
        bound_placements.append(
            RunnerCacheStartupPlacement(
                placement_id=placement.placement_id,
                node_name=placement.node_name,
                resource_flavor=placement.resource_flavor,
                profile_id=placement.profile_id,
                compatibility_approval_id=placement.compatibility_approval_id,
                manifest_digest=manifest_digest,
                pin_owner=command.pin_owner,
                prestage_command_id=command.command_id,
                prestage_command_generation=command.command_generation,
                hint_index_revision=hint.index_revision,
                resident_record_generation=resident.record_generation,
                hint_observed_at=hint.observed_at,
                hint_valid_until=hint.valid_until,
            )
        )

    payload = {
        "schema_version": "runner-cache-startup-binding-v1",
        "decision_id": decision_id,
        "decision_fingerprint": decision_fingerprint,
        "target_id": target_id,
        "target_revision": target_revision,
        "deployment_id": deployment_id,
        "model_class": plan.snapshot.model_class,
        "model_id": model_id,
        "model_revision": plan.snapshot.model_revision,
        "manifest_digest": manifest_digest,
        "placement_binding_id": plan.snapshot.placement_binding_id,
        "prewarm_snapshot_id": plan.snapshot.snapshot_id,
        "prewarm_cache_revision": plan.snapshot.cache_revision,
        "bound_at": bound_at,
        "valid_until": min(placement.hint_valid_until for placement in bound_placements),
        "placements": tuple(bound_placements),
    }
    unsigned = RunnerCacheStartupBinding.model_construct(binding_id="0" * 64, **payload)
    binding_id = _canonical_digest(unsigned.model_dump(mode="json", exclude={"binding_id"}))
    return RunnerCacheStartupBinding(binding_id=binding_id, **payload)
