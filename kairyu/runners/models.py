"""Backend-neutral lifecycle snapshots for managed inference runners."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RunnerState(StrEnum):
    """Logical serving state; deliberately independent of Kubernetes Pod phase."""

    REQUESTED = "requested"
    SCHEDULING = "scheduling"
    IMAGE_PULL = "image_pull"
    MODEL_LOADING = "model_loading"
    WARMING = "warming"
    READY = "ready"
    BUSY = "busy"
    DRAINING = "draining"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    UNHEALTHY = "unhealthy"


class RunnerStartupPhase(StrEnum):
    """Ordered phases whose latency composes one Runner cold start."""

    IMAGE_PULL = "image_pull"
    MODEL_FETCH = "model_fetch"
    MODEL_LOAD = "model_load"
    GRAPH_COMPILE = "graph_compile"
    WARMUP = "warmup"


RUNNER_STARTUP_PHASES = tuple(RunnerStartupPhase)
OPTIONAL_RUNNER_STARTUP_PHASES = frozenset(
    {
        RunnerStartupPhase.MODEL_FETCH,
        RunnerStartupPhase.GRAPH_COMPILE,
        RunnerStartupPhase.WARMUP,
    }
)


class RunnerStartupPhaseOutcome(StrEnum):
    """Terminal outcome of one startup phase; None means still running."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunnerFailureDomainKind(StrEnum):
    """Failure scope used by bounded restart backoff and quarantine."""

    REVISION = "revision"
    NODE = "node"
    GPU = "gpu"


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


class RunnerFailure(BaseModel):
    """Bounded, sanitized lifecycle failure suitable for status and audit output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(max_length=128)
    message: str = Field(max_length=1024)
    retryable: bool = False
    domain: RunnerFailureDomainKind | None = None

    @field_validator("code", "message")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)


class RunnerStartupPhaseReport(BaseModel):
    """One immutable phase observation; an absent outcome is in progress."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: RunnerStartupPhase
    started_at: datetime
    completed_at: datetime | None = None
    outcome: RunnerStartupPhaseOutcome | None = None
    failure: RunnerFailure | None = None

    @field_validator("started_at", "completed_at")
    @classmethod
    def validate_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return _aware(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_completion(self) -> RunnerStartupPhaseReport:
        if self.completed_at is None:
            if self.outcome is not None or self.failure is not None:
                raise ValueError(
                    "an in-progress startup phase cannot have outcome or failure"
                )
            return self
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be earlier than started_at")
        if self.outcome is None:
            raise ValueError("a completed startup phase requires an outcome")
        if self.outcome is RunnerStartupPhaseOutcome.FAILED:
            if self.failure is None:
                raise ValueError("a failed startup phase requires failure")
        elif self.failure is not None:
            raise ValueError("only a failed startup phase can carry failure")
        if (
            self.outcome is RunnerStartupPhaseOutcome.SKIPPED
            and self.phase not in OPTIONAL_RUNNER_STARTUP_PHASES
        ):
            raise ValueError(f"startup phase {self.phase.value} cannot be skipped")
        return self

    @property
    def duration_seconds(self) -> float | None:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()


class RunnerStartupReport(BaseModel):
    """Ordered, gap-free startup phase snapshot for one Runner attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["runner-startup-v1"] = "runner-startup-v1"
    runner_id: str = Field(max_length=255)
    attempt: int = Field(default=1, ge=1)
    observed_at: datetime
    phases: tuple[RunnerStartupPhaseReport, ...] = ()

    @field_validator("runner_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return _non_empty(value, name="runner_id")

    @field_validator("attempt", mode="before")
    @classmethod
    def validate_attempt_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("attempt must be an integer")
        return value

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="observed_at")

    @model_validator(mode="after")
    def validate_phase_sequence(self) -> RunnerStartupReport:
        observed_phases = tuple(report.phase for report in self.phases)
        expected_phases = RUNNER_STARTUP_PHASES[: len(observed_phases)]
        if observed_phases != expected_phases:
            raise ValueError(
                "startup phases must be unique, gap-free, and in canonical order"
            )
        previous_completed_at: datetime | None = None
        for index, report in enumerate(self.phases):
            if report.started_at > self.observed_at or (
                report.completed_at is not None
                and report.completed_at > self.observed_at
            ):
                raise ValueError("startup phase timestamps cannot exceed observed_at")
            if previous_completed_at is not None and report.started_at < previous_completed_at:
                raise ValueError("startup phases cannot overlap")
            if report.completed_at is None and index != len(self.phases) - 1:
                raise ValueError("only the last startup phase may be in progress")
            if (
                report.outcome is RunnerStartupPhaseOutcome.FAILED
                and index != len(self.phases) - 1
            ):
                raise ValueError("no startup phase may follow a failed phase")
            previous_completed_at = report.completed_at
        return self

    @property
    def current_phase(self) -> RunnerStartupPhase | None:
        if self.phases and self.phases[-1].completed_at is None:
            return self.phases[-1].phase
        return None

    @property
    def completed(self) -> bool:
        return len(self.phases) == len(RUNNER_STARTUP_PHASES) and all(
            report.outcome
            in {
                RunnerStartupPhaseOutcome.SUCCEEDED,
                RunnerStartupPhaseOutcome.SKIPPED,
            }
            for report in self.phases
        )

    @property
    def failure(self) -> RunnerFailure | None:
        if not self.phases:
            return None
        return self.phases[-1].failure


class RunnerTerminationAuthorization(BaseModel):
    """Persisted proof that a drained Runner may enter termination."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["runner-termination-authorization-v1"] = (
        "runner-termination-authorization-v1"
    )
    runner_id: str = Field(max_length=255)
    pod_uid: str = Field(max_length=255)
    fence_id: str = Field(max_length=255)
    fence_sequence: int = Field(gt=0)
    drain_state_version: int = Field(ge=0)
    replica_generation: str = Field(max_length=255)
    dispatch_stopped_at: datetime
    routing_excluded_at: datetime
    activity_observed_at: datetime
    authorized_at: datetime

    @field_validator("runner_id", "pod_uid", "fence_id", "replica_generation")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator("fence_sequence", "drain_state_version", mode="before")
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        return value

    @field_validator(
        "dispatch_stopped_at",
        "routing_excluded_at",
        "activity_observed_at",
        "authorized_at",
    )
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        return _aware(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_timeline(self) -> RunnerTerminationAuthorization:
        if self.routing_excluded_at < self.dispatch_stopped_at:
            raise ValueError("routing exclusion cannot predate dispatch stop")
        if self.activity_observed_at < self.routing_excluded_at:
            raise ValueError("activity observation must be post-fence")
        if self.authorized_at < self.activity_observed_at:
            raise ValueError("authorization cannot predate activity observation")
        return self


class RunnerStatus(BaseModel):
    """Immutable controller-facing view of one logical inference Runner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["runner-status-v1"] = "runner-status-v1"
    runner_id: str = Field(max_length=255)
    release_id: str = Field(max_length=512)
    model_id: str = Field(max_length=512)
    model_revision: str = Field(max_length=512)
    state: RunnerState
    state_version: int = Field(default=0, ge=0)
    state_changed_at: datetime
    observed_at: datetime
    node_name: str | None = Field(default=None, max_length=253)
    pod_uid: str | None = Field(default=None, max_length=255)
    gpu_uuids: tuple[str, ...] = Field(default=(), max_length=64)
    active_requests: int = Field(default=0, ge=0)
    runtime_observed_at: datetime | None = None
    runtime_observation_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    termination_authorization: RunnerTerminationAuthorization | None = None
    startup: RunnerStartupReport | None = None
    failure: RunnerFailure | None = None

    @field_validator("runner_id", "release_id", "model_id", "model_revision")
    @classmethod
    def validate_required_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator("state_version", "active_requests", mode="before")
    @classmethod
    def validate_counter_type(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        return value

    @field_validator("node_name", "pod_uid")
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _non_empty(value, name=info.field_name)

    @field_validator("gpu_uuids")
    @classmethod
    def validate_gpu_uuids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(_non_empty(value, name="gpu_uuids") for value in values)
        if len(set(validated)) != len(validated):
            raise ValueError("gpu_uuids must be unique")
        return validated

    @field_validator("state_changed_at", "observed_at", "runtime_observed_at")
    @classmethod
    def validate_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return _aware(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_snapshot(self) -> RunnerStatus:
        if self.observed_at < self.state_changed_at:
            raise ValueError("observed_at cannot be earlier than state_changed_at")
        if self.startup is not None and self.startup.runner_id != self.runner_id:
            raise ValueError("startup runner_id must match status runner_id")
        if (
            self.runtime_observation_fingerprint is not None
            and self.runtime_observed_at is None
        ):
            raise ValueError(
                "runtime observation fingerprint requires runtime_observed_at"
            )
        if self.state in {RunnerState.TERMINATING, RunnerState.TERMINATED}:
            authorization = self.termination_authorization
            if authorization is None:
                raise ValueError(
                    "terminating or terminated runners require authorization"
                )
            if authorization.runner_id != self.runner_id:
                raise ValueError("termination authorization runner_id must match status")
            if authorization.pod_uid != self.pod_uid:
                raise ValueError("termination authorization pod_uid must match status")
            version_delta = (
                1 if self.state is RunnerState.TERMINATING else 2
            )
            if self.state_version != authorization.drain_state_version + version_delta:
                raise ValueError(
                    "termination state version must directly follow its drain state"
                )
            if authorization.authorized_at > self.observed_at:
                raise ValueError("termination authorization cannot exceed observed_at")
        elif self.termination_authorization is not None:
            raise ValueError(
                "only terminating or terminated runners may carry authorization"
            )
        if self.state is RunnerState.UNHEALTHY:
            if self.failure is None:
                raise ValueError("unhealthy runners require failure")
        elif self.failure is not None:
            raise ValueError("only unhealthy runners can carry failure")
        if self.state is RunnerState.BUSY:
            if self.active_requests == 0:
                raise ValueError("busy runners require at least one active request")
        elif self.state in {
            RunnerState.REQUESTED,
            RunnerState.SCHEDULING,
            RunnerState.IMAGE_PULL,
            RunnerState.MODEL_LOADING,
            RunnerState.WARMING,
            RunnerState.READY,
            RunnerState.TERMINATING,
            RunnerState.TERMINATED,
        } and self.active_requests != 0:
            raise ValueError(f"{self.state.value} runners cannot have active requests")
        if self.state in {RunnerState.READY, RunnerState.BUSY} and (
            self.startup is None or not self.startup.completed
        ):
            raise ValueError("ready or busy runners require completed startup phases")
        if self.active_requests > 0 and (
            self.startup is None or not self.startup.completed
        ):
            raise ValueError("active requests require completed startup phases")
        return self
