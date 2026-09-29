"""Versioned autoscaler observation windows and append-only decision records."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.runners.models import RunnerStartupPhase
from kairyu.runners.scaling import _MAX_SIGNED_BIGINT, ScalingPolicy

_MAX_BUFFERED_TARGET_REPLICAS = 11_000_000


def _non_empty(value: str, *, name: str) -> str:
    if not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL")
    return value


def _aware(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _integer(value: object, *, name: str) -> object:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _finite(value: object, *, name: str) -> object:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return value


class ScalingDecisionAction(StrEnum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    HOLD = "hold"


class ScalingDecisionReason(StrEnum):
    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
    STALE_OBSERVATIONS = "stale_observations"
    QUEUE_PRESSURE = "queue_pressure"
    DEADLINE_PRESSURE = "deadline_pressure"
    HIGH_UTILIZATION = "high_utilization"
    LOW_UTILIZATION = "low_utilization"
    WARM_BUFFER = "warm_buffer"
    MIN_REPLICAS = "min_replicas"
    MAX_REPLICAS = "max_replicas"
    SCALE_UP_DELAY = "scale_up_delay"
    KEEP_ALIVE = "keep_alive"
    COOLDOWN = "cooldown"
    FAILURE_QUARANTINE = "failure_quarantine"
    BUDGET_LIMIT = "budget_limit"
    HYSTERESIS = "hysteresis"
    NO_CHANGE = "no_change"


class ScalingQueueSnapshot(BaseModel):
    """Request-store inputs captured at one source-owned timestamp."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    observed_at: datetime
    queue_depth: int = Field(ge=0, le=10_000_000)
    interactive_queue_depth: int = Field(ge=0, le=10_000_000)
    batch_queue_depth: int = Field(ge=0, le=10_000_000)
    oldest_queue_age_seconds: float = Field(ge=0, le=604_800)
    arrival_rate_per_second: float = Field(ge=0, le=10_000_000)
    deadline_remaining_p50_seconds: float | None = Field(default=None, ge=0)
    deadline_remaining_p95_seconds: float | None = Field(default=None, ge=0)
    predicted_ttft_seconds: float | None = Field(default=None, ge=0)
    goodput_slo_ratio: float | None = Field(default=None, ge=0, le=1)

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="observed_at")

    @field_validator(
        "queue_depth",
        "interactive_queue_depth",
        "batch_queue_depth",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        return _integer(value, name=info.field_name)

    @field_validator(
        "oldest_queue_age_seconds",
        "arrival_rate_per_second",
        "deadline_remaining_p50_seconds",
        "deadline_remaining_p95_seconds",
        "predicted_ttft_seconds",
        "goodput_slo_ratio",
        mode="before",
    )
    @classmethod
    def validate_number(cls, value: object, info) -> object:
        if value is None:
            return value
        return _finite(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_queue_shape(self) -> ScalingQueueSnapshot:
        if self.interactive_queue_depth + self.batch_queue_depth != self.queue_depth:
            raise ValueError("request-class queue depths must sum to queue_depth")
        percentiles = (
            self.deadline_remaining_p50_seconds,
            self.deadline_remaining_p95_seconds,
        )
        if (percentiles[0] is None) != (percentiles[1] is None):
            raise ValueError("deadline percentiles must be both present or both absent")
        if (
            percentiles[0] is not None
            and percentiles[1] is not None
            and percentiles[1] < percentiles[0]
        ):
            raise ValueError("deadline p95 cannot be below p50")
        return self


class ScalingRunnerSnapshot(BaseModel):
    """Logical Runner-state counts for one model class."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    observed_at: datetime
    current_replicas: int = Field(ge=0, le=100_000)
    busy_replicas: int = Field(ge=0, le=100_000)
    ready_replicas: int = Field(ge=0, le=100_000)
    loading_replicas: int = Field(ge=0, le=100_000)
    unhealthy_replicas: int = Field(ge=0, le=100_000)
    draining_replicas: int = Field(default=0, ge=0, le=100_000)

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="observed_at")

    @field_validator(
        "current_replicas",
        "busy_replicas",
        "ready_replicas",
        "loading_replicas",
        "unhealthy_replicas",
        "draining_replicas",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        return _integer(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_counts(self) -> ScalingRunnerSnapshot:
        classified = (
            self.busy_replicas
            + self.ready_replicas
            + self.loading_replicas
            + self.unhealthy_replicas
            + self.draining_replicas
        )
        if classified > self.current_replicas:
            raise ValueError("classified Runner counts cannot exceed current_replicas")
        return self


class ScalingResourceSnapshot(BaseModel):
    """Optional accelerator and scheduler metrics from one scrape epoch."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    observed_at: datetime
    gpu_utilization: float | None = Field(default=None, ge=0, le=1)
    hbm_utilization: float | None = Field(default=None, ge=0, le=1)
    kv_utilization: float | None = Field(default=None, ge=0, le=1)
    multiplexing_occupancy: float | None = Field(default=None, ge=0, le=1)
    batch_occupancy: float | None = Field(default=None, ge=0, le=1)

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="observed_at")

    @field_validator(
        "gpu_utilization",
        "hbm_utilization",
        "kv_utilization",
        "multiplexing_occupancy",
        "batch_occupancy",
        mode="before",
    )
    @classmethod
    def validate_number(cls, value: object, info) -> object:
        if value is None:
            return value
        return _finite(value, name=info.field_name)


class ScalingStartupPhaseMetrics(BaseModel):
    """Bounded cold-start aggregate for one canonical startup phase."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    phase: RunnerStartupPhase
    sample_count: int = Field(ge=1, le=1_000_000)
    ema_seconds: float = Field(ge=0, le=86_400)
    p95_seconds: float = Field(ge=0, le=86_400)

    @field_validator("sample_count", mode="before")
    @classmethod
    def validate_sample_count(cls, value: object) -> object:
        return _integer(value, name="sample_count")

    @field_validator("ema_seconds", "p95_seconds", mode="before")
    @classmethod
    def validate_number(cls, value: object, info) -> object:
        return _finite(value, name=info.field_name)


class ScalingStartupSnapshot(BaseModel):
    """Model residency and cold-start phase metrics at one observation time."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    observed_at: datetime
    model_cache_resident_replicas: int = Field(ge=0, le=100_000)
    phases: tuple[ScalingStartupPhaseMetrics, ...] = Field(default=(), max_length=5)

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="observed_at")

    @field_validator("model_cache_resident_replicas", mode="before")
    @classmethod
    def validate_integer(cls, value: object) -> object:
        return _integer(value, name="model_cache_resident_replicas")

    @model_validator(mode="after")
    def validate_phases(self) -> ScalingStartupSnapshot:
        phases = tuple(metric.phase for metric in self.phases)
        if len(set(phases)) != len(phases):
            raise ValueError("startup phase metrics must be unique")
        if phases != tuple(sorted(phases, key=lambda phase: list(RunnerStartupPhase).index(phase))):
            raise ValueError("startup phase metrics must use canonical order")
        return self


class ScalingObservation(BaseModel):
    """One coherent, source-timestamped autoscaler input snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-scaling-observation-v1"] = (
        "runner-scaling-observation-v1"
    )
    observation_id: str = Field(max_length=255)
    model_class: str = Field(max_length=128)
    observed_at: datetime
    queue: ScalingQueueSnapshot
    runners: ScalingRunnerSnapshot
    resources: ScalingResourceSnapshot | None = None
    startup: ScalingStartupSnapshot | None = None

    @field_validator("observation_id", "model_class")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="observed_at")

    @model_validator(mode="after")
    def validate_source_times(self) -> ScalingObservation:
        source_times = [self.queue.observed_at, self.runners.observed_at]
        if self.resources is not None:
            source_times.append(self.resources.observed_at)
        if self.startup is not None:
            source_times.append(self.startup.observed_at)
        if any(source_time > self.observed_at for source_time in source_times):
            raise ValueError("source observation times cannot exceed observed_at")
        if self.startup is not None and (
            self.startup.model_cache_resident_replicas > self.runners.current_replicas
        ):
            raise ValueError("cache-resident replicas cannot exceed current_replicas")
        return self


class ScalingObservationWindow(BaseModel):
    """Ordered, bounded decision input window for one model class."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-scaling-window-v1"] = "runner-scaling-window-v1"
    window_id: str = Field(max_length=255)
    model_class: str = Field(max_length=128)
    started_at: datetime
    ended_at: datetime
    observations: tuple[ScalingObservation, ...] = Field(
        min_length=1,
        max_length=2048,
    )

    @field_validator("window_id", "model_class")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator("started_at", "ended_at")
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        return _aware(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_window(self) -> ScalingObservationWindow:
        if self.ended_at < self.started_at:
            raise ValueError("window ended_at cannot precede started_at")
        previous: datetime | None = None
        observation_ids: set[str] = set()
        for observation in self.observations:
            if observation.model_class != self.model_class:
                raise ValueError("window observations must match model_class")
            if not self.started_at <= observation.observed_at <= self.ended_at:
                raise ValueError("observation time must be inside the window")
            if previous is not None and observation.observed_at <= previous:
                raise ValueError("window observations must be strictly time ordered")
            if observation.observation_id in observation_ids:
                raise ValueError("window observation IDs must be unique")
            previous = observation.observed_at
            observation_ids.add(observation.observation_id)
        return self

    @property
    def fingerprint(self) -> str:
        validated = type(self).model_validate(self.model_dump())
        payload = json.dumps(
            validated.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class ScalingDecisionRecord(BaseModel):
    """Append-only decision with the exact input window and policy snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-scaling-decision-v1"] = (
        "runner-scaling-decision-v1"
    )
    decision_id: str = Field(max_length=255)
    decided_at: datetime
    catalog_revision: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    policy: ScalingPolicy
    window: ScalingObservationWindow
    action: ScalingDecisionAction
    reason: ScalingDecisionReason
    reason_detail: str = Field(default="", max_length=512)
    inputs_stale: bool = False
    demand_replicas: int = Field(ge=0, le=10_000_000)
    buffered_target_replicas: int = Field(
        ge=0,
        le=_MAX_BUFFERED_TARGET_REPLICAS,
    )
    desired_replicas: int = Field(ge=0, le=100_000)
    target_delta: int = Field(ge=-100_000, le=100_000)

    @field_validator("decision_id")
    @classmethod
    def validate_decision_id(cls, value: str) -> str:
        return _non_empty(value, name="decision_id")

    @field_validator("reason_detail")
    @classmethod
    def validate_reason_detail(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("reason_detail cannot contain NUL")
        return value

    @field_validator("decided_at")
    @classmethod
    def validate_decided_at(cls, value: datetime) -> datetime:
        return _aware(value, name="decided_at")

    @field_validator(
        "catalog_revision",
        "demand_replicas",
        "buffered_target_replicas",
        "desired_replicas",
        "target_delta",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        return _integer(value, name=info.field_name)

    @field_validator("inputs_stale", mode="before")
    @classmethod
    def validate_inputs_stale(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("inputs_stale must be a boolean")
        return value

    @model_validator(mode="after")
    def validate_decision(self) -> ScalingDecisionRecord:
        if self.decided_at < self.window.ended_at:
            raise ValueError("decision cannot predate its observation window")
        if self.policy.model_class != self.window.model_class:
            raise ValueError("decision policy and window model_class must match")
        latest = self.window.observations[-1]
        latest_source_times = [
            latest.observed_at,
            latest.queue.observed_at,
            latest.runners.observed_at,
        ]
        if latest.resources is not None:
            latest_source_times.append(latest.resources.observed_at)
        if latest.startup is not None:
            latest_source_times.append(latest.startup.observed_at)
        derived_stale = (
            self.decided_at - min(latest_source_times)
        ).total_seconds() > self.policy.max_observation_age_seconds
        if self.inputs_stale != derived_stale:
            raise ValueError(
                "inputs_stale must match the policy freshness limit and source times"
            )
        expected_buffered_target = self.demand_replicas + self.policy.warm_buffer_for(
            self.demand_replicas
        )
        if self.buffered_target_replicas != expected_buffered_target:
            raise ValueError(
                "buffered_target_replicas must equal demand plus the policy buffer"
            )
        current = latest.runners.current_replicas
        if self.desired_replicas - current != self.target_delta:
            raise ValueError("target_delta must equal desired minus current replicas")
        if self.target_delta != 0:
            if current > self.policy.max_replicas and not (
                self.policy.max_replicas <= self.desired_replicas < current
            ):
                raise ValueError(
                    "an over-max replica count must converge toward the policy max"
                )
            if current < self.policy.min_replicas and not (
                current < self.desired_replicas <= self.policy.min_replicas
            ):
                raise ValueError(
                    "an under-min replica count must converge toward the policy min"
                )
            if (
                self.policy.min_replicas <= current <= self.policy.max_replicas
                and not self.policy.min_replicas
                <= self.desired_replicas
                <= self.policy.max_replicas
            ):
                raise ValueError("desired_replicas must satisfy policy min/max")
        if self.action is ScalingDecisionAction.HOLD and self.target_delta != 0:
            raise ValueError("hold decisions require target_delta=0")
        if self.action is ScalingDecisionAction.SCALE_UP:
            if not 0 < self.target_delta <= self.policy.max_scale_up_step:
                raise ValueError("scale-up delta must satisfy the policy step bound")
        if self.action is ScalingDecisionAction.SCALE_DOWN:
            if not -self.policy.max_scale_down_step <= self.target_delta < 0:
                raise ValueError("scale-down delta must satisfy the policy step bound")
        stale_reason = self.reason is ScalingDecisionReason.STALE_OBSERVATIONS
        if stale_reason != self.inputs_stale:
            raise ValueError("stale observations reason must match inputs_stale")
        if self.inputs_stale and self.action is ScalingDecisionAction.SCALE_DOWN:
            raise ValueError("stale observations cannot authorize scale-down")
        return self

    @property
    def fingerprint(self) -> str:
        validated = type(self).model_validate(self.model_dump())
        payload = json.dumps(
            validated.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class ScalingDecisionConflictError(RuntimeError):
    """A decision ID was replayed with different immutable content."""


class ScalingDecisionCapacityError(RuntimeError):
    """The bounded decision log cannot accept another unique record."""


@runtime_checkable
class ScalingDecisionLog(Protocol):
    def append(self, record: ScalingDecisionRecord) -> ScalingDecisionRecord: ...

    def get(self, decision_id: str) -> ScalingDecisionRecord: ...

    def list(
        self,
        *,
        model_class: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> tuple[ScalingDecisionRecord, ...]: ...


class InMemoryScalingDecisionLog:
    """Thread-safe bounded reference backend; production uses shared storage."""

    def __init__(self, *, max_records: int = 10_000) -> None:
        if type(max_records) is not int or max_records <= 0:
            raise ValueError("max_records must be a positive integer")
        self._max_records = max_records
        self._records: dict[str, tuple[str, ScalingDecisionRecord]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _validated(record: ScalingDecisionRecord) -> ScalingDecisionRecord:
        if not isinstance(record, ScalingDecisionRecord):
            raise TypeError("record must be a ScalingDecisionRecord")
        return ScalingDecisionRecord.model_validate(record.model_dump())

    def append(self, record: ScalingDecisionRecord) -> ScalingDecisionRecord:
        record = self._validated(record)
        fingerprint = record.fingerprint
        with self._lock:
            existing = self._records.get(record.decision_id)
            if existing is not None:
                if existing[0] != fingerprint:
                    raise ScalingDecisionConflictError(
                        "decision ID was already used with different content"
                    )
                return existing[1].model_copy(deep=True)
            if len(self._records) >= self._max_records:
                raise ScalingDecisionCapacityError("decision log capacity is exhausted")
            self._records[record.decision_id] = (fingerprint, record)
            return record.model_copy(deep=True)

    def get(self, decision_id: str) -> ScalingDecisionRecord:
        decision_id = _non_empty(decision_id, name="decision_id")
        with self._lock:
            try:
                record = self._records[decision_id][1]
            except KeyError:
                raise KeyError(f"unknown scaling decision {decision_id!r}") from None
            return record.model_copy(deep=True)

    def list(
        self,
        *,
        model_class: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> tuple[ScalingDecisionRecord, ...]:
        if model_class is not None:
            model_class = _non_empty(model_class, name="model_class")
        if since is not None:
            since = _aware(since, name="since")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer from 1 to 1000")
        with self._lock:
            records = (
                record
                for _, record in self._records.values()
                if (model_class is None or record.policy.model_class == model_class)
                and (since is None or record.decided_at >= since)
            )
            ordered = sorted(
                records,
                key=lambda record: (record.decided_at, record.decision_id),
                reverse=True,
            )
            return tuple(record.model_copy(deep=True) for record in ordered[:limit])
