"""Runner-owned full-digest proof for one cache-bound Pod startup."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.artifacts.node_cache import NodeModelCacheRunnerStartDecision
from kairyu.runners.startup_binding import RunnerCacheStartupBinding

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_RUNNER_CLOCK_SKEW_TOLERANCE = timedelta(seconds=2)


class RunnerCacheStartupAttestationError(RuntimeError):
    """Pod, binding, and node verification evidence do not agree."""


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


def _clock_skew_tolerance(value: timedelta | None) -> timedelta | None:
    if value is not None and (
        not isinstance(value, timedelta) or value < timedelta(0)
    ):
        raise ValueError("clock_skew_tolerance must be a non-negative timedelta")
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


class RunnerCacheStartupProof(BaseModel):
    """Hash-bound proof that one scheduled Pod re-hashed its exact artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-cache-startup-proof-v1"] = (
        "runner-cache-startup-proof-v1"
    )
    proof_id: str = Field(min_length=64, max_length=64)
    runner_id: str = Field(max_length=255)
    node_name: str = Field(max_length=253)
    binding_id: str = Field(min_length=64, max_length=64)
    placement_id: str = Field(max_length=255)
    decision_id: str = Field(max_length=255)
    decision_fingerprint: str = Field(min_length=64, max_length=64)
    model_id: str = Field(max_length=255)
    model_revision: str = Field(max_length=255)
    manifest_digest: str = Field(min_length=64, max_length=64)
    prestage_command_id: str = Field(min_length=64, max_length=64)
    prestage_command_generation: int = Field(ge=1, le=2**63 - 1)
    resident_record_generation: int = Field(ge=1, le=2**63 - 1)
    verification_reason: Literal["verified"] = "verified"
    verified_at: datetime

    @field_validator(
        "runner_id",
        "placement_id",
        "decision_id",
        "model_id",
        "model_revision",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _text(value, name=info.field_name)

    @field_validator("node_name")
    @classmethod
    def validate_node_name(cls, value: str) -> str:
        return _text(value, name="node_name", max_length=253)

    @field_validator(
        "proof_id",
        "binding_id",
        "decision_fingerprint",
        "manifest_digest",
        "prestage_command_id",
    )
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _digest(value, name=info.field_name)

    @field_validator(
        "prestage_command_generation",
        "resident_record_generation",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> int:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        assert isinstance(value, int)
        return value

    @field_validator("verified_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _aware(value, name="verified_at")

    @model_validator(mode="after")
    def validate_proof_id(self) -> RunnerCacheStartupProof:
        expected = _canonical_digest(self.model_dump(mode="json", exclude={"proof_id"}))
        if self.proof_id != expected:
            raise ValueError("proof_id must match the canonical startup proof")
        return self


def build_runner_cache_startup_proof(
    binding: RunnerCacheStartupBinding,
    decision: NodeModelCacheRunnerStartDecision,
    *,
    runner_id: str,
    node_name: str,
    verified_at: datetime,
    clock_skew_tolerance: timedelta = DEFAULT_RUNNER_CLOCK_SKEW_TOLERANCE,
) -> RunnerCacheStartupProof:
    """Bind a successful local WP4.6 verification to one scheduled Pod."""

    if not isinstance(binding, RunnerCacheStartupBinding):
        raise TypeError("binding must be a RunnerCacheStartupBinding")
    if not isinstance(decision, NodeModelCacheRunnerStartDecision):
        raise TypeError("decision must be a NodeModelCacheRunnerStartDecision")
    binding = RunnerCacheStartupBinding.model_validate(binding.model_dump())
    decision = NodeModelCacheRunnerStartDecision.model_validate(decision.model_dump())
    runner_id = _text(runner_id, name="runner_id")
    node_name = _text(node_name, name="node_name", max_length=253)
    verified_at = _aware(verified_at, name="verified_at")
    clock_skew_tolerance = _clock_skew_tolerance(clock_skew_tolerance)
    assert clock_skew_tolerance is not None
    placements = tuple(
        placement for placement in binding.placements if placement.node_name == node_name
    )
    if len(placements) != 1:
        raise RunnerCacheStartupAttestationError(
            "scheduled node must identify exactly one startup placement"
        )
    if (
        not decision.runner_start_allowed
        or decision.reason != "verified"
        or decision.artifact_path is None
        or decision.manifest_digest != binding.manifest_digest
    ):
        raise RunnerCacheStartupAttestationError(
            "Runner startup requires an allowed full-digest cache decision"
        )
    if (
        verified_at < binding.bound_at
        and binding.bound_at - verified_at > clock_skew_tolerance
    ):
        raise RunnerCacheStartupAttestationError(
            "Runner cache verification predates the startup binding beyond clock skew"
        )
    placement = placements[0]
    payload = {
        "schema_version": "runner-cache-startup-proof-v1",
        "runner_id": runner_id,
        "node_name": node_name,
        "binding_id": binding.binding_id,
        "placement_id": placement.placement_id,
        "decision_id": binding.decision_id,
        "decision_fingerprint": binding.decision_fingerprint,
        "model_id": binding.model_id,
        "model_revision": binding.model_revision,
        "manifest_digest": binding.manifest_digest,
        "prestage_command_id": placement.prestage_command_id,
        "prestage_command_generation": placement.prestage_command_generation,
        "resident_record_generation": placement.resident_record_generation,
        "verification_reason": "verified",
        "verified_at": verified_at,
    }
    unsigned = RunnerCacheStartupProof.model_construct(proof_id="0" * 64, **payload)
    proof_id = _canonical_digest(unsigned.model_dump(mode="json", exclude={"proof_id"}))
    return RunnerCacheStartupProof(proof_id=proof_id, **payload)


def validate_runner_cache_startup_proof(
    binding: RunnerCacheStartupBinding,
    proof: RunnerCacheStartupProof,
    *,
    runner_id: str,
    node_name: str,
    model_id: str,
    model_revision: str,
    observed_at: datetime,
    clock_skew_tolerance: timedelta | None = DEFAULT_RUNNER_CLOCK_SKEW_TOLERANCE,
) -> None:
    """Fail closed unless a runtime proof exactly attests one observed Pod."""

    if not isinstance(binding, RunnerCacheStartupBinding):
        raise TypeError("binding must be a RunnerCacheStartupBinding")
    if not isinstance(proof, RunnerCacheStartupProof):
        raise TypeError("proof must be a RunnerCacheStartupProof")
    binding = RunnerCacheStartupBinding.model_validate(binding.model_dump())
    proof = RunnerCacheStartupProof.model_validate(proof.model_dump())
    observed_at = _aware(observed_at, name="observed_at")
    clock_skew_tolerance = _clock_skew_tolerance(clock_skew_tolerance)
    placements = {
        placement.placement_id: placement for placement in binding.placements
    }
    placement = placements.get(proof.placement_id)
    if proof.verified_at > observed_at:
        raise RunnerCacheStartupAttestationError(
            "Runner cache verification exceeds its runtime observation"
        )
    if (
        clock_skew_tolerance is not None
        and proof.verified_at < binding.bound_at
        and binding.bound_at - proof.verified_at > clock_skew_tolerance
    ):
        raise RunnerCacheStartupAttestationError(
            "Runner cache verification predates the binding beyond clock skew"
        )
    if (
        placement is None
        or proof.runner_id != runner_id
        or proof.node_name != node_name
        or proof.binding_id != binding.binding_id
        or proof.decision_id != binding.decision_id
        or proof.decision_fingerprint != binding.decision_fingerprint
        or proof.model_id != model_id
        or proof.model_id != binding.model_id
        or proof.model_revision != model_revision
        or proof.model_revision != binding.model_revision
        or proof.manifest_digest != binding.manifest_digest
        or proof.prestage_command_id != placement.prestage_command_id
        or proof.prestage_command_generation != placement.prestage_command_generation
        or proof.resident_record_generation != placement.resident_record_generation
        or placement.node_name != node_name
    ):
        raise RunnerCacheStartupAttestationError(
            "Runner cache startup proof does not match the observed Pod"
        )
