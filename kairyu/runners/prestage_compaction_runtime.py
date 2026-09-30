"""Lifecycle-owned scheduled compaction for released pre-stage records."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.runners.prestage import (
    NodeModelPrestageCompactionMonitoringStore,
    NodeModelPrestageCompactionStore,
    NodeModelPrestageHighWaterMark,
)

_MAX_COMPACTED_RECORDS_PER_CYCLE = 10_000


def _aware(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


class NodeModelPrestageCompactionRuntimeConfig(BaseModel):
    """Bounded schedule and readiness policy for one node-scoped store."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    retirement_age_seconds: float = Field(ge=1.0, le=10 * 365 * 24 * 60 * 60)
    interval_seconds: float = Field(default=60.0, ge=0.01, le=24 * 60 * 60)
    batch_size: int = Field(default=100, ge=1, le=1000)
    max_batches_per_cycle: int = Field(default=10, ge=1, le=1000)
    readiness_max_staleness_seconds: float = Field(
        default=300.0,
        ge=0.01,
        le=7 * 24 * 60 * 60,
    )
    readiness_failure_threshold: int = Field(default=3, ge=1, le=1000)
    shutdown_timeout_seconds: float = Field(default=30.0, ge=0.01, le=300.0)

    @field_validator(
        "retirement_age_seconds",
        "interval_seconds",
        "readiness_max_staleness_seconds",
        "shutdown_timeout_seconds",
        mode="before",
    )
    @classmethod
    def validate_finite_float(cls, value: object, info) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{info.field_name} must be a finite number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{info.field_name} must be a finite number")
        return value

    @field_validator(
        "batch_size", "max_batches_per_cycle", "readiness_failure_threshold", mode="before"
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        return value

    @model_validator(mode="after")
    def validate_readiness_window(self) -> NodeModelPrestageCompactionRuntimeConfig:
        if self.readiness_max_staleness_seconds < self.interval_seconds:
            raise ValueError(
                "readiness_max_staleness_seconds cannot be shorter than interval_seconds"
            )
        if self.batch_size * self.max_batches_per_cycle > _MAX_COMPACTED_RECORDS_PER_CYCLE:
            raise ValueError("batch_size times max_batches_per_cycle cannot exceed 10000")
        return self


class NodeModelPrestageCompactionCycle(BaseModel):
    """Bounded, monitorable result of one completed compaction cycle."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    node_id: str = Field(min_length=1, max_length=253)
    started_at: datetime
    completed_at: datetime
    retired_before: datetime
    batch_calls: int = Field(ge=1, le=1000)
    batch_budget_exhausted: bool
    compacted: tuple[NodeModelPrestageHighWaterMark, ...]

    @field_validator("started_at", "completed_at", "retired_before")
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        return _aware(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_times(self) -> NodeModelPrestageCompactionCycle:
        if self.retired_before > self.started_at:
            raise ValueError("retired_before cannot follow started_at")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot predate started_at")
        return self


class NodeModelPrestageCompactionRuntimeStatus(BaseModel):
    """Low-disclosure lifecycle and cumulative process-local counters."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    node_id: str = Field(min_length=1, max_length=253)
    state: Literal["new", "running", "stopping", "stopped", "failed"]
    ready: bool
    cycles_started: int = Field(ge=0)
    cycles_succeeded: int = Field(ge=0)
    cycles_failed: int = Field(ge=0)
    batch_calls: int = Field(ge=0)
    compacted_records: int = Field(ge=0)
    consecutive_failures: int = Field(ge=0)
    last_cycle: NodeModelPrestageCompactionCycle | None = None
    last_failure_at: datetime | None = None

    @field_validator("last_failure_at")
    @classmethod
    def validate_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware(value, name="last_failure_at")


class NodeModelPrestageCompactionRuntime:
    """Run cutoff-stable, bounded compaction cycles on one owned worker.

    The injected store remains caller-owned. ``close`` stops only this runtime;
    it deliberately does not close the shared store or its database connection.
    Runtime counters are process-local, while every returned high-water mark is
    durable in the store and remains available through
    :meth:`list_high_water_marks_page` when the store implements the bounded
    monitoring extension.
    """

    def __init__(
        self,
        *,
        store: NodeModelPrestageCompactionStore,
        config: NodeModelPrestageCompactionRuntimeConfig,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(store, NodeModelPrestageCompactionStore):
            raise TypeError("store must implement NodeModelPrestageCompactionStore")
        if not isinstance(config, NodeModelPrestageCompactionRuntimeConfig):
            raise TypeError("config must be a NodeModelPrestageCompactionRuntimeConfig")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(monotonic):
            raise TypeError("monotonic must be callable")
        self._store = store
        self._config = config
        self._clock = clock
        self._monotonic = monotonic
        self._state_lock = threading.Lock()
        self._cycle_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: Literal["new", "running", "stopping", "stopped", "failed"] = "new"
        self._cycles_started = 0
        self._cycles_succeeded = 0
        self._cycles_failed = 0
        self._batch_calls = 0
        self._compacted_records = 0
        self._consecutive_failures = 0
        self._last_cycle: NodeModelPrestageCompactionCycle | None = None
        self._last_success_monotonic: float | None = None
        self._last_failure_at: datetime | None = None

    @property
    def node_id(self) -> str:
        return self._store.node_id

    @property
    def config(self) -> NodeModelPrestageCompactionRuntimeConfig:
        return self._config

    def _timestamp(self, *, name: str) -> datetime:
        return _aware(self._clock(), name=name)

    def _readiness_locked(self, now_monotonic: float) -> bool:
        thread = self._thread
        if self._state != "running" or thread is None or not thread.is_alive():
            return False
        if self._last_success_monotonic is None:
            return False
        if self._consecutive_failures >= self._config.readiness_failure_threshold:
            return False
        return (
            now_monotonic - self._last_success_monotonic
            <= self._config.readiness_max_staleness_seconds
        )

    def status(self) -> NodeModelPrestageCompactionRuntimeStatus:
        now_monotonic = self._monotonic()
        if not math.isfinite(now_monotonic):
            raise RuntimeError("monotonic clock returned a non-finite value")
        with self._state_lock:
            return NodeModelPrestageCompactionRuntimeStatus(
                node_id=self.node_id,
                state=self._state,
                ready=self._readiness_locked(now_monotonic),
                cycles_started=self._cycles_started,
                cycles_succeeded=self._cycles_succeeded,
                cycles_failed=self._cycles_failed,
                batch_calls=self._batch_calls,
                compacted_records=self._compacted_records,
                consecutive_failures=self._consecutive_failures,
                last_cycle=self._last_cycle,
                last_failure_at=self._last_failure_at,
            )

    def check_ready(self) -> None:
        """Fail closed without exposing backend exception details."""

        if not self.status().ready:
            raise RuntimeError("pre-stage compaction runtime is not ready")

    def _validate_marks(
        self,
        marks: object,
        *,
        retired_before: datetime,
        compacted_at: datetime,
        seen_placement_ids: set[str],
    ) -> tuple[NodeModelPrestageHighWaterMark, ...]:
        if not isinstance(marks, tuple):
            raise RuntimeError("pre-stage compaction store returned an invalid batch")
        if len(marks) > self._config.batch_size:
            raise RuntimeError("pre-stage compaction store exceeded the requested batch size")
        validated: list[NodeModelPrestageHighWaterMark] = []
        for raw_mark in marks:
            try:
                mark = NodeModelPrestageHighWaterMark.model_validate(raw_mark)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "pre-stage compaction store returned an invalid high-water mark"
                ) from error
            if mark.updated_at > retired_before or mark.compacted_at != compacted_at:
                raise RuntimeError(
                    "pre-stage compaction store returned a mark outside the cycle cutoff"
                )
            if mark.placement_id in seen_placement_ids:
                raise RuntimeError("pre-stage compaction store returned a duplicate placement")
            seen_placement_ids.add(mark.placement_id)
            validated.append(mark)
        return tuple(validated)

    def run_once(self) -> NodeModelPrestageCompactionCycle:
        """Run one bounded cycle, suitable for explicit jobs and tests."""

        if not self._cycle_lock.acquire(blocking=False):
            raise RuntimeError("pre-stage compaction cycle is already in progress")
        try:
            with self._state_lock:
                if self._state in {"stopping", "stopped", "failed"}:
                    raise RuntimeError("pre-stage compaction runtime is not executable")
                self._cycles_started += 1
            try:
                started_at = self._timestamp(name="compaction cycle start")
                retired_before = started_at - timedelta(seconds=self._config.retirement_age_seconds)
                compacted: list[NodeModelPrestageHighWaterMark] = []
                seen_placement_ids: set[str] = set()
                batch_calls = 0
                batch_budget_exhausted = False
                for batch_index in range(self._config.max_batches_per_cycle):
                    marks = self._store.compact_absent_records(
                        retired_before=retired_before,
                        compacted_at=started_at,
                        limit=self._config.batch_size,
                    )
                    batch_calls += 1
                    validated = self._validate_marks(
                        marks,
                        retired_before=retired_before,
                        compacted_at=started_at,
                        seen_placement_ids=seen_placement_ids,
                    )
                    compacted.extend(validated)
                    with self._state_lock:
                        self._batch_calls += 1
                        self._compacted_records += len(validated)
                    if len(validated) < self._config.batch_size:
                        break
                    if batch_index + 1 == self._config.max_batches_per_cycle:
                        batch_budget_exhausted = True
                completed_at = self._timestamp(name="compaction cycle completion")
                cycle = NodeModelPrestageCompactionCycle(
                    node_id=self.node_id,
                    started_at=started_at,
                    completed_at=completed_at,
                    retired_before=retired_before,
                    batch_calls=batch_calls,
                    batch_budget_exhausted=batch_budget_exhausted,
                    compacted=tuple(compacted),
                )
                success_monotonic = self._monotonic()
                if not math.isfinite(success_monotonic):
                    raise RuntimeError("monotonic clock returned a non-finite value")
            except Exception:
                try:
                    failed_at = self._timestamp(name="compaction cycle failure")
                except Exception:
                    failed_at = datetime.now(UTC)
                with self._state_lock:
                    self._cycles_failed += 1
                    self._consecutive_failures += 1
                    self._last_failure_at = failed_at
                raise
            with self._state_lock:
                self._cycles_succeeded += 1
                self._consecutive_failures = 0
                self._last_cycle = cycle
                self._last_success_monotonic = success_monotonic
            return cycle
        finally:
            self._cycle_lock.release()

    def list_high_water_marks_page(
        self,
        *,
        after_placement_id: str | None = None,
        limit: int = 100,
    ) -> tuple[NodeModelPrestageHighWaterMark, ...]:
        """Read one bounded, deep-validated durable monitoring page."""

        if after_placement_id is not None and (
            not isinstance(after_placement_id, str)
            or not after_placement_id.strip()
            or "\x00" in after_placement_id
            or len(after_placement_id) > 255
        ):
            raise ValueError("after_placement_id must be a non-empty string without NUL")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer in [1, 1000]")
        if not isinstance(self._store, NodeModelPrestageCompactionMonitoringStore):
            raise RuntimeError("pre-stage compaction store has no bounded monitoring view")
        marks = self._store.list_high_water_marks_page(
            after_placement_id=after_placement_id,
            limit=limit,
        )
        if not isinstance(marks, tuple):
            raise RuntimeError("pre-stage compaction store returned an invalid monitoring view")
        if len(marks) > limit:
            raise RuntimeError("pre-stage compaction store exceeded the monitoring page limit")
        validated: list[NodeModelPrestageHighWaterMark] = []
        seen: set[str] = set()
        for raw_mark in marks:
            try:
                mark = NodeModelPrestageHighWaterMark.model_validate(raw_mark)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "pre-stage compaction store returned an invalid monitoring mark"
                ) from error
            if mark.placement_id in seen:
                raise RuntimeError("pre-stage compaction store returned duplicate monitoring marks")
            seen.add(mark.placement_id)
            validated.append(mark)
        return tuple(validated)

    def _run(self) -> None:
        fatal = False
        try:
            while not self._stop.is_set():
                try:
                    self.run_once()
                except Exception:
                    # Failure counters and the low-disclosure readiness state are
                    # recorded by run_once. The next scheduled cycle retries.
                    pass
                if self._stop.wait(self._config.interval_seconds):
                    break
        except BaseException:
            fatal = True
            try:
                failed_at = self._timestamp(name="compaction worker failure")
            except BaseException:
                failed_at = datetime.now(UTC)
            with self._state_lock:
                self._cycles_failed += 1
                self._consecutive_failures += 1
                self._last_failure_at = failed_at
                self._state = "failed"
        finally:
            if not fatal:
                with self._state_lock:
                    if self._state == "running":
                        self._state = "failed"
                        self._last_failure_at = datetime.now(UTC)

    def start(self) -> None:
        with self._state_lock:
            if self._state == "running":
                return
            if self._state != "new":
                raise RuntimeError("pre-stage compaction runtime cannot be restarted")
            self._stop.clear()
            thread = threading.Thread(
                target=self._run,
                name="kairyu-prestage-compaction",
                daemon=True,
            )
            self._thread = thread
            self._state = "running"
            try:
                thread.start()
            except BaseException:
                self._thread = None
                self._state = "failed"
                self._last_failure_at = datetime.now(UTC)
                raise

    def close(self) -> None:
        shutdown_deadline = time.monotonic() + self._config.shutdown_timeout_seconds
        with self._state_lock:
            if self._state == "stopped":
                return
            if self._state == "new":
                self._state = "stopping"
            thread = self._thread
            if self._state != "failed":
                self._state = "stopping"
            self._stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, shutdown_deadline - time.monotonic()))
            if thread.is_alive():
                raise RuntimeError("pre-stage compaction worker did not stop before deadline")
        remaining = max(0.0, shutdown_deadline - time.monotonic())
        if not self._cycle_lock.acquire(timeout=remaining):
            raise RuntimeError("pre-stage compaction cycle did not stop before deadline")
        self._cycle_lock.release()
        with self._state_lock:
            self._state = "stopped"

    def __enter__(self) -> NodeModelPrestageCompactionRuntime:
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


__all__ = [
    "NodeModelPrestageCompactionCycle",
    "NodeModelPrestageCompactionRuntime",
    "NodeModelPrestageCompactionRuntimeConfig",
    "NodeModelPrestageCompactionRuntimeStatus",
]
