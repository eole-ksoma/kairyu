"""Versioned, model-class scaling policy contracts for Runner autoscaling."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_MAX_SIGNED_BIGINT = 2**63 - 1


def _non_empty(value: str, *, name: str) -> str:
    if not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL")
    return value


class ScalingPolicy(BaseModel):
    """Immutable autoscaling limits and timing for one model class.

    The safe default is a single warm replica with no scale-to-zero and no
    request multiplexing. Decision inputs and actuation deliberately remain in
    later WP3 slices.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    schema_version: Literal["runner-scaling-policy-v1"] = "runner-scaling-policy-v1"
    model_class: str = Field(max_length=128)
    policy_revision: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    min_replicas: int = Field(default=1, ge=0, le=100_000)
    max_replicas: int = Field(default=1, ge=1, le=100_000)
    warm_buffer_replicas: int = Field(default=0, ge=0, le=100_000)
    warm_buffer_ratio: float = Field(default=0.0, ge=0, le=10)
    scale_up_delay_seconds: float = Field(default=0.0, ge=0, le=86_400)
    keep_alive_seconds: float = Field(default=60.0, ge=0, le=604_800)
    cooldown_seconds: float = Field(default=60.0, ge=0, le=604_800)
    max_observation_age_seconds: float = Field(default=30.0, gt=0, le=86_400)
    scale_to_zero: bool = False
    scale_to_zero_approval_id: str | None = Field(default=None, max_length=255)
    max_multiplexing: int = Field(default=1, ge=1, le=1024)
    max_scale_up_step: int = Field(default=1, ge=1, le=100_000)
    max_scale_down_step: int = Field(default=1, ge=1, le=100_000)

    @field_validator("model_class")
    @classmethod
    def validate_model_class(cls, value: str) -> str:
        return _non_empty(value, name="model_class")

    @field_validator("scale_to_zero_approval_id")
    @classmethod
    def validate_approval_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _non_empty(value, name="scale_to_zero_approval_id")

    @field_validator(
        "policy_revision",
        "min_replicas",
        "max_replicas",
        "warm_buffer_replicas",
        "max_multiplexing",
        "max_scale_up_step",
        "max_scale_down_step",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        return value

    @field_validator("scale_to_zero", mode="before")
    @classmethod
    def validate_boolean(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("scale_to_zero must be a boolean")
        return value

    @field_validator(
        "warm_buffer_ratio",
        "scale_up_delay_seconds",
        "keep_alive_seconds",
        "cooldown_seconds",
        "max_observation_age_seconds",
        mode="before",
    )
    @classmethod
    def validate_finite_number(cls, value: object, info) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{info.field_name} must be a number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{info.field_name} must be finite")
        return value

    @model_validator(mode="after")
    def validate_policy_bounds(self) -> ScalingPolicy:
        if self.min_replicas > self.max_replicas:
            raise ValueError("min_replicas cannot exceed max_replicas")
        if self.warm_buffer_replicas > self.max_replicas:
            raise ValueError("warm_buffer_replicas cannot exceed max_replicas")
        if self.max_scale_up_step > self.max_replicas:
            raise ValueError("max_scale_up_step cannot exceed max_replicas")
        if self.max_scale_down_step > self.max_replicas:
            raise ValueError("max_scale_down_step cannot exceed max_replicas")
        if self.scale_to_zero:
            if self.min_replicas != 0:
                raise ValueError("scale-to-zero requires min_replicas=0")
            if self.scale_to_zero_approval_id is None:
                raise ValueError("scale-to-zero requires an approval ID")
            if self.warm_buffer_replicas != 0:
                raise ValueError(
                    "scale-to-zero requires warm_buffer_replicas=0; "
                    "use the demand-relative buffer"
                )
        else:
            if self.min_replicas == 0:
                raise ValueError("min_replicas=0 requires scale-to-zero")
            if self.scale_to_zero_approval_id is not None:
                raise ValueError("scale-to-zero approval requires scale_to_zero=true")
        return self

    @property
    def identity(self) -> tuple[str, int]:
        """Stable policy identity persisted by the later decision log."""

        validated = type(self).model_validate(self.model_dump())
        return (validated.model_class, validated.policy_revision)

    def warm_buffer_for(self, demand_replicas: int) -> int:
        """Return max(absolute, ceil(demand replicas * ratio))."""

        if type(demand_replicas) is not int or demand_replicas < 0:
            raise ValueError("demand_replicas must be a non-negative integer")
        validated = type(self).model_validate(self.model_dump())
        bounded_demand = min(demand_replicas, validated.max_replicas)
        ratio_buffer = math.ceil(bounded_demand * validated.warm_buffer_ratio)
        return max(validated.warm_buffer_replicas, ratio_buffer)


class ScalingPolicyCatalog(BaseModel):
    """Bounded, uniquely keyed collection of model-class scaling policies."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    schema_version: Literal["runner-scaling-policy-catalog-v1"] = (
        "runner-scaling-policy-catalog-v1"
    )
    catalog_revision: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    policies: tuple[ScalingPolicy, ...] = Field(min_length=1, max_length=4096)

    @field_validator("catalog_revision", mode="before")
    @classmethod
    def validate_catalog_revision(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("catalog_revision must be an integer")
        return value

    @model_validator(mode="after")
    def validate_unique_model_classes(self) -> ScalingPolicyCatalog:
        model_classes = tuple(policy.model_class for policy in self.policies)
        if len(set(model_classes)) != len(model_classes):
            raise ValueError("scaling policies must use unique model classes")
        return self

    def policy_for(self, model_class: str) -> ScalingPolicy:
        model_class = _non_empty(model_class, name="model_class")
        validated = type(self).model_validate(self.model_dump())
        for policy in validated.policies:
            if policy.model_class == model_class:
                return policy
        raise KeyError(f"unknown scaling model class {model_class!r}")
