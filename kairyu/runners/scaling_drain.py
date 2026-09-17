"""Drain-authorized StatefulSet scale-down planning contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.runners.models import RunnerState, RunnerStatus
from kairyu.runners.scaling import _MAX_SIGNED_BIGINT


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


class ScalingDrainCandidate(BaseModel):
    """One exact StatefulSet ordinal authorized for controller deletion."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    pod_name: str = Field(max_length=253)
    workload_ordinal: int = Field(ge=0, le=100_000)
    status: RunnerStatus

    @field_validator("pod_name")
    @classmethod
    def validate_pod_name(cls, value: str) -> str:
        return _non_empty(value, name="pod_name")

    @field_validator("workload_ordinal", mode="before")
    @classmethod
    def validate_ordinal(cls, value: object) -> object:
        return _integer(value, name="workload_ordinal")

    @model_validator(mode="after")
    def validate_status(self) -> ScalingDrainCandidate:
        if self.status.state is not RunnerState.TERMINATING:
            raise ValueError("scale-down candidates must be termination-authorized")
        if self.status.active_requests != 0:
            raise ValueError("scale-down candidates must have zero active requests")
        if self.status.pod_uid is None or self.status.termination_authorization is None:
            raise ValueError("scale-down candidates require stable termination evidence")
        return self


class ScalingDrainSnapshot(BaseModel):
    """Source-versioned drain inventory for one StatefulSet generation."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-scaling-drain-snapshot-v1"] = (
        "runner-scaling-drain-snapshot-v1"
    )
    snapshot_id: str = Field(max_length=255)
    drain_revision: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    observed_at: datetime
    model_class: str = Field(max_length=128)
    namespace: str = Field(max_length=253)
    statefulset_name: str = Field(max_length=253)
    workload_uid: str = Field(max_length=255)
    workload_generation: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    release_id: str = Field(max_length=255)
    model_revision: str = Field(max_length=255)
    candidates: tuple[ScalingDrainCandidate, ...] = Field(
        default=(),
        max_length=100_000,
    )

    @field_validator(
        "snapshot_id",
        "model_class",
        "namespace",
        "statefulset_name",
        "workload_uid",
        "release_id",
        "model_revision",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator("drain_revision", "workload_generation", mode="before")
    @classmethod
    def validate_revision(cls, value: object, info) -> object:
        return _integer(value, name=info.field_name)

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="observed_at")

    @model_validator(mode="after")
    def validate_candidates(self) -> ScalingDrainSnapshot:
        ordinals = tuple(candidate.workload_ordinal for candidate in self.candidates)
        if len(set(ordinals)) != len(ordinals):
            raise ValueError("drain candidates must use unique StatefulSet ordinals")
        if ordinals != tuple(sorted(ordinals)):
            raise ValueError("drain candidates must use canonical ordinal order")
        runner_ids = tuple(candidate.status.runner_id for candidate in self.candidates)
        pod_uids = tuple(candidate.status.pod_uid for candidate in self.candidates)
        if len(set(runner_ids)) != len(runner_ids) or len(set(pod_uids)) != len(pod_uids):
            raise ValueError("drain candidates must use unique Runner and Pod identities")
        for candidate in self.candidates:
            if candidate.pod_name != f"{self.statefulset_name}-{candidate.workload_ordinal}":
                raise ValueError("drain candidate Pod name must match its StatefulSet ordinal")
            status = candidate.status
            if status.release_id != self.release_id:
                raise ValueError("drain candidate release must match the target release")
            if status.model_revision != self.model_revision:
                raise ValueError("drain candidate model revision must match the target revision")
            if status.observed_at > self.observed_at:
                raise ValueError("drain candidate observation cannot postdate its snapshot")
        return self


def _selected_candidates(
    snapshot: ScalingDrainSnapshot,
    *,
    current_replicas: int,
    desired_replicas: int,
) -> tuple[ScalingDrainCandidate, ...]:
    candidates = {candidate.workload_ordinal: candidate for candidate in snapshot.candidates}
    required_ordinals = range(desired_replicas, current_replicas)
    try:
        return tuple(candidates[ordinal] for ordinal in required_ordinals)
    except KeyError as error:
        raise ValueError(
            "every removed StatefulSet ordinal requires termination authorization"
        ) from error


class ScalingDrainPlan(BaseModel):
    """Auditable proof for one deterministic StatefulSet replica reduction."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-scaling-drain-plan-v1"] = (
        "runner-scaling-drain-plan-v1"
    )
    snapshot: ScalingDrainSnapshot
    current_replicas: int = Field(ge=1, le=100_000)
    desired_replicas: int = Field(ge=0, le=100_000)
    candidate_runner_ids: tuple[str, ...] = Field(min_length=1, max_length=100_000)
    candidate_pod_uids: tuple[str, ...] = Field(min_length=1, max_length=100_000)

    @field_validator("current_replicas", "desired_replicas", mode="before")
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        return _integer(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_plan(self) -> ScalingDrainPlan:
        if self.desired_replicas >= self.current_replicas:
            raise ValueError("drain planning requires a scale-down target")
        selected = _selected_candidates(
            self.snapshot,
            current_replicas=self.current_replicas,
            desired_replicas=self.desired_replicas,
        )
        expected_runners = tuple(candidate.status.runner_id for candidate in selected)
        expected_pods = tuple(candidate.status.pod_uid for candidate in selected)
        if self.candidate_runner_ids != expected_runners:
            raise ValueError("candidate Runner IDs must match removed StatefulSet ordinals")
        if self.candidate_pod_uids != expected_pods:
            raise ValueError("candidate Pod UIDs must match removed StatefulSet ordinals")
        return self

    @property
    def selected_candidates(self) -> tuple[ScalingDrainCandidate, ...]:
        """Return only the exact ordinal suffix removed by this plan."""

        return _selected_candidates(
            self.snapshot,
            current_replicas=self.current_replicas,
            desired_replicas=self.desired_replicas,
        )

    @property
    def source_observed_at(self) -> datetime:
        """Return the oldest source time that authorizes the replica reduction."""

        return min(
            self.snapshot.observed_at,
            *(candidate.status.observed_at for candidate in self.selected_candidates),
        )


def plan_statefulset_scale_down(
    snapshot: ScalingDrainSnapshot,
    *,
    current_replicas: int,
    desired_replicas: int,
) -> ScalingDrainPlan:
    """Select the exact highest StatefulSet ordinals removed by scale-down."""

    if not isinstance(snapshot, ScalingDrainSnapshot):
        raise TypeError("snapshot must be a ScalingDrainSnapshot")
    current_replicas = _integer(current_replicas, name="current_replicas")
    desired_replicas = _integer(desired_replicas, name="desired_replicas")
    assert isinstance(current_replicas, int)
    assert isinstance(desired_replicas, int)
    if not 1 <= current_replicas <= 100_000:
        raise ValueError("current_replicas must be in [1, 100000]")
    if not 0 <= desired_replicas <= 100_000:
        raise ValueError("desired_replicas must be in [0, 100000]")
    if desired_replicas >= current_replicas:
        raise ValueError("drain planning requires a scale-down target")
    snapshot = ScalingDrainSnapshot.model_validate(snapshot.model_dump())
    selected = _selected_candidates(
        snapshot,
        current_replicas=current_replicas,
        desired_replicas=desired_replicas,
    )
    return ScalingDrainPlan(
        snapshot=snapshot,
        current_replicas=current_replicas,
        desired_replicas=desired_replicas,
        candidate_runner_ids=tuple(candidate.status.runner_id for candidate in selected),
        candidate_pod_uids=tuple(candidate.status.pod_uid for candidate in selected),
    )
