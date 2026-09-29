"""Read-only observations consumed by the Runner status reconciler."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.runners.models import RunnerStartupReport


def _non_empty(value: str, *, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{name} cannot contain NUL")
    return value


def _aware(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


class KubernetesPodPhase(StrEnum):
    """Pod phases used as evidence, never as the Runner state itself."""

    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    UNKNOWN = "Unknown"


class RunnerRuntimeObservation(BaseModel):
    """One bounded response from the Runner-owned status/readiness surface."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runner_id: str = Field(max_length=255)
    observed_at: datetime
    ready: bool
    active_requests: int = Field(ge=0)
    startup: RunnerStartupReport | None = None
    fatal: bool = False
    detail: str = Field(default="", max_length=256)

    @field_validator("runner_id")
    @classmethod
    def validate_runner_id(cls, value: str) -> str:
        return _non_empty(value, name="runner_id")

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="observed_at")

    @field_validator("active_requests", mode="before")
    @classmethod
    def validate_counter_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("active_requests must be an integer")
        return value

    @field_validator("detail")
    @classmethod
    def validate_detail(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("detail cannot contain NUL")
        return value

    @model_validator(mode="after")
    def validate_fatal_readiness(self) -> RunnerRuntimeObservation:
        if self.ready and self.fatal:
            raise ValueError("a ready runtime cannot report a fatal condition")
        if self.startup is not None and self.startup.observed_at > self.observed_at:
            raise ValueError("startup observed_at cannot exceed runtime observed_at")
        return self


class RunnerPodObservation(BaseModel):
    """Relevant immutable projection of a Kubernetes Pod."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    uid: str = Field(max_length=255)
    phase: KubernetesPodPhase
    node_name: str | None = Field(default=None, max_length=253)
    ready: bool = False
    deleting: bool = False
    waiting_reason: str | None = Field(default=None, max_length=128)
    terminated_reason: str | None = Field(default=None, max_length=128)
    exit_code: int | None = None
    gpu_uuids: tuple[str, ...] = Field(default=(), max_length=64)

    @field_validator("uid")
    @classmethod
    def validate_uid(cls, value: str) -> str:
        return _non_empty(value, name="uid")

    @field_validator("node_name", "waiting_reason", "terminated_reason")
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _non_empty(value, name=info.field_name)

    @field_validator("exit_code", mode="before")
    @classmethod
    def validate_exit_code_type(cls, value: object) -> object:
        if value is not None and type(value) is not int:
            raise ValueError("exit_code must be an integer")
        return value

    @field_validator("gpu_uuids")
    @classmethod
    def validate_gpu_uuids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(_non_empty(value, name="gpu_uuids") for value in values)
        if len(set(validated)) != len(validated):
            raise ValueError("gpu_uuids must be unique")
        return validated


class RunnerObservation(BaseModel):
    """One coherent control-plane observation for a logical Runner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runner_id: str = Field(max_length=255)
    release_id: str = Field(max_length=512)
    model_id: str = Field(max_length=512)
    model_revision: str = Field(max_length=512)
    observed_at: datetime
    pod: RunnerPodObservation | None = None
    endpoint_ready: bool = False
    runtime: RunnerRuntimeObservation | None = None

    @field_validator("runner_id", "release_id", "model_id", "model_revision")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="observed_at")

    @model_validator(mode="after")
    def validate_consistency(self) -> RunnerObservation:
        if self.pod is not None and self.pod.uid != self.runner_id:
            raise ValueError("Pod uid must match runner_id")
        if self.runtime is not None:
            if self.runtime.runner_id != self.runner_id:
                raise ValueError("runtime runner_id must match observation runner_id")
        if self.runtime is not None and self.runtime.startup is not None:
            startup = self.runtime.startup
            if startup.runner_id != self.runner_id:
                raise ValueError("startup runner_id must match observation runner_id")
        return self


class RunnerObservationBatch(BaseModel):
    """One full-list observation epoch, including the empty-fleet case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(max_length=255)
    source_epoch: int = Field(ge=1)
    source_started_at: datetime
    observed_at: datetime
    pod_resource_version: str | None = Field(default=None, max_length=255)
    endpoint_slice_resource_version: str | None = Field(default=None, max_length=255)
    runners: tuple[RunnerObservation, ...] = ()
    missing_runner_runtime: tuple[RunnerRuntimeObservation, ...] = ()

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _non_empty(value, name="source_id")

    @field_validator("source_epoch", mode="before")
    @classmethod
    def validate_source_epoch_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("source_epoch must be an integer")
        return value

    @field_validator("pod_resource_version", "endpoint_slice_resource_version")
    @classmethod
    def validate_resource_version(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _non_empty(value, name=info.field_name)

    @field_validator("source_started_at", "observed_at")
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        return _aware(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_epoch(self) -> RunnerObservationBatch:
        if self.observed_at < self.source_started_at:
            raise ValueError("batch completion cannot predate source start")
        runner_ids: set[str] = set()
        for runner in self.runners:
            if runner.observed_at != self.observed_at:
                raise ValueError("Runner observations must use the batch observation time")
            if runner.runner_id in runner_ids:
                raise ValueError("Runner observations must have unique runner IDs")
            runner_ids.add(runner.runner_id)
        missing_ids: set[str] = set()
        for runtime in self.missing_runner_runtime:
            if runtime.runner_id in runner_ids or runtime.runner_id in missing_ids:
                raise ValueError(
                    "missing Runner runtime rows must be unique and disjoint from Pods"
                )
            missing_ids.add(runtime.runner_id)
        return self
