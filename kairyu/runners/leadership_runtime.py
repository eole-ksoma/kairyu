"""Lifecycle-owned leader election for the Runner control plane."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.runners.leadership import (
    InvalidRunnerLeadershipError,
    LeaderFencedRunnerController,
    RunnerLeaderElector,
    RunnerLeaderLease,
    RunnerLeaderLeaseStore,
    RunnerNotLeaderError,
    RunnerWriterAuthority,
)
from kairyu.runners.reconciler import RunnerStatusReconciler

_MutationResult = TypeVar("_MutationResult")


def _aware(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


class RunnerLeaderElectionRuntimeConfig(BaseModel):
    """Bounded election, renewal, readiness, and shutdown policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    election_id: str = Field(min_length=1, max_length=255)
    holder_id: str = Field(min_length=1, max_length=255)
    lease_seconds: float = Field(default=15.0, gt=0, le=86_400)
    campaign_interval_seconds: float = Field(default=1.0, ge=0.01, le=300)
    renew_interval_seconds: float = Field(default=5.0, ge=0.01, le=300)
    readiness_max_staleness_seconds: float = Field(default=10.0, ge=0.01, le=300)
    readiness_failure_threshold: int = Field(default=3, ge=1, le=1000, strict=True)
    shutdown_timeout_seconds: float = Field(default=30.0, ge=0.01, le=300)

    @field_validator("election_id", "holder_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        if value != value.strip() or "\x00" in value:
            raise ValueError(f"{info.field_name} must be trimmed and contain no NUL")
        return value

    @field_validator(
        "lease_seconds",
        "campaign_interval_seconds",
        "renew_interval_seconds",
        "readiness_max_staleness_seconds",
        "shutdown_timeout_seconds",
        mode="before",
    )
    @classmethod
    def validate_finite_number(cls, value: object, info) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{info.field_name} must be a finite number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{info.field_name} must be a finite number")
        return value

    @model_validator(mode="after")
    def validate_intervals(self) -> RunnerLeaderElectionRuntimeConfig:
        if self.renew_interval_seconds >= self.lease_seconds:
            raise ValueError("renew_interval_seconds must be shorter than lease_seconds")
        if self.readiness_max_staleness_seconds >= self.lease_seconds:
            raise ValueError("readiness_max_staleness_seconds must be shorter than lease_seconds")
        if self.readiness_max_staleness_seconds < self.renew_interval_seconds:
            raise ValueError(
                "readiness_max_staleness_seconds cannot be shorter than renew_interval_seconds"
            )
        return self


class RunnerLeaderElectionRuntimeStatus(BaseModel):
    """Low-disclosure process-local election status."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    election_id: str = Field(min_length=1, max_length=255)
    holder_id: str = Field(min_length=1, max_length=255)
    state: Literal["new", "running", "stopping", "stopped", "failed"]
    ready: bool
    leader: bool
    operations_started: int = Field(ge=0)
    operations_succeeded: int = Field(ge=0)
    operations_failed: int = Field(ge=0)
    campaigns_won: int = Field(ge=0)
    renewals_succeeded: int = Field(ge=0)
    consecutive_failures: int = Field(ge=0)
    active_mutations: int = Field(ge=0)
    current_lease: RunnerLeaderLease | None = None
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None

    @field_validator("last_success_at", "last_failure_at")
    @classmethod
    def validate_timestamp(cls, value: datetime | None, info) -> datetime | None:
        return None if value is None else _aware(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_leadership(self) -> RunnerLeaderElectionRuntimeStatus:
        if self.leader != (self.current_lease is not None):
            raise ValueError("leader must match current_lease presence")
        if self.ready and (self.state != "running" or not self.leader):
            raise ValueError("ready election runtime must be running leader")
        if self.operations_succeeded + self.operations_failed > self.operations_started:
            raise ValueError("completed election operations exceed started operations")
        return self


class _LifecycleFencedRunnerController(LeaderFencedRunnerController):
    """Serialize mutation admission with runtime lifecycle transitions."""

    def __init__(
        self,
        elector: RunnerLeaderElector,
        reconciler: RunnerStatusReconciler,
        *,
        precheck: Callable[[], None],
        begin_mutation: Callable[[], None],
        end_mutation: Callable[[], None],
        revoke_snapshot: Callable[[], None],
    ) -> None:
        super().__init__(elector, reconciler)
        self._execution_lock = threading.Lock()
        self._precheck = precheck
        self._begin_mutation = begin_mutation
        self._end_mutation = end_mutation
        self._revoke_snapshot = revoke_snapshot

    def mutate(
        self,
        operation: Callable[[RunnerWriterAuthority], _MutationResult],
    ) -> _MutationResult:
        self._precheck()
        with self._execution_lock:
            self._begin_mutation()
            try:
                return super().mutate(operation)
            except (RunnerNotLeaderError, InvalidRunnerLeadershipError):
                self._revoke_snapshot()
                raise
            finally:
                self._end_mutation()


class RunnerLeaderElectionRuntime:
    """Continuously campaign and renew one store-fenced leader tenure.

    The injected store remains caller-owned. Every control-plane mutation still
    calls the store through :class:`RunnerLeaderElector.authority`; process-local
    readiness is only an additional liveness signal and never grants authority.
    """

    def __init__(
        self,
        *,
        store: RunnerLeaderLeaseStore,
        config: RunnerLeaderElectionRuntimeConfig,
        reconciler: RunnerStatusReconciler | None = None,
        clock=lambda: datetime.now(UTC),
        monotonic=time.monotonic,
    ) -> None:
        if not isinstance(store, RunnerLeaderLeaseStore):
            raise TypeError("store must implement RunnerLeaderLeaseStore")
        if not isinstance(config, RunnerLeaderElectionRuntimeConfig):
            raise TypeError("config must be a RunnerLeaderElectionRuntimeConfig")
        if reconciler is not None and not isinstance(reconciler, RunnerStatusReconciler):
            raise TypeError("reconciler must be a RunnerStatusReconciler")
        if not callable(clock) or not callable(monotonic):
            raise TypeError("clocks must be callable")
        self._config = config
        self._clock = clock
        self._monotonic = monotonic
        self._elector = RunnerLeaderElector(
            store,
            election_id=config.election_id,
            holder_id=config.holder_id,
            lease_seconds=config.lease_seconds,
        )
        self._state_lock = threading.RLock()
        self._state_changed = threading.Condition(self._state_lock)
        self._operation_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: Literal["new", "running", "stopping", "stopped", "failed"] = "new"
        self._operations_started = 0
        self._operations_succeeded = 0
        self._operations_failed = 0
        self._campaigns_won = 0
        self._renewals_succeeded = 0
        self._consecutive_failures = 0
        self._last_success_at: datetime | None = None
        self._last_success_monotonic: float | None = None
        self._last_failure_at: datetime | None = None
        self._current_lease: RunnerLeaderLease | None = None
        self._active_mutations = 0
        self._controller = _LifecycleFencedRunnerController(
            self._elector,
            reconciler or RunnerStatusReconciler(),
            precheck=self._precheck_mutation,
            begin_mutation=self._begin_mutation,
            end_mutation=self._end_mutation,
            revoke_snapshot=self._revoke_lease_snapshot,
        )

    @property
    def config(self) -> RunnerLeaderElectionRuntimeConfig:
        return self._config

    @property
    def controller(self) -> LeaderFencedRunnerController:
        return self._controller

    def _assert_mutation_allowed_locked(self) -> None:
        thread = self._thread
        if self._state != "running" or thread is None or not thread.is_alive():
            raise RunnerNotLeaderError(
                "leader election runtime is not accepting controller mutations"
            )

    def _precheck_mutation(self) -> None:
        with self._state_lock:
            self._assert_mutation_allowed_locked()

    def _begin_mutation(self) -> None:
        with self._state_lock:
            self._assert_mutation_allowed_locked()
            self._active_mutations += 1

    def _end_mutation(self) -> None:
        with self._state_changed:
            self._active_mutations -= 1
            self._state_changed.notify_all()

    def _revoke_lease_snapshot(self) -> None:
        with self._state_lock:
            self._current_lease = None

    def _sync_lease_locked(self) -> None:
        self._current_lease = self._elector.lease

    def _timestamp(self, *, name: str) -> datetime:
        return _aware(self._clock(), name=name)

    def _monotonic_now(self) -> float:
        value = self._monotonic()
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RuntimeError("monotonic clock returned an invalid value")
        value = float(value)
        if not math.isfinite(value):
            raise RuntimeError("monotonic clock returned a non-finite value")
        return value

    def _record_failure(self, *, count_operation: bool = True) -> None:
        try:
            failed_at = self._timestamp(name="leader election failure")
        except Exception:
            failed_at = datetime.now(UTC)
        with self._state_lock:
            if count_operation:
                self._operations_failed += 1
            self._consecutive_failures += 1
            self._last_failure_at = failed_at

    def run_once(self) -> RunnerLeaderLease | None:
        """Perform one campaign or renewal without overlapping another call."""

        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("leader election operation is already in progress")
        try:
            with self._state_lock:
                if self._state in {"stopping", "stopped", "failed"}:
                    raise RuntimeError("leader election runtime is not executable")
                self._operations_started += 1
            renewing = self._elector.lease is not None
            try:
                lease = self._elector.renew() if renewing else self._elector.campaign()
                succeeded_at = self._timestamp(name="leader election success")
                succeeded_monotonic = self._monotonic_now()
            except Exception:
                with self._state_lock:
                    self._sync_lease_locked()
                self._record_failure()
                raise
            with self._state_lock:
                self._current_lease = lease
                self._operations_succeeded += 1
                self._consecutive_failures = 0
                self._last_success_at = succeeded_at
                self._last_success_monotonic = succeeded_monotonic
                if lease is not None:
                    if renewing:
                        self._renewals_succeeded += 1
                    else:
                        self._campaigns_won += 1
            return lease
        finally:
            self._operation_lock.release()

    def _ready_locked(
        self,
        *,
        lease: RunnerLeaderLease | None,
        now_monotonic: float,
    ) -> bool:
        thread = self._thread
        if (
            self._state != "running"
            or thread is None
            or not thread.is_alive()
            or lease is None
            or self._last_success_monotonic is None
            or self._consecutive_failures >= self._config.readiness_failure_threshold
        ):
            return False
        return (
            now_monotonic - self._last_success_monotonic
            <= self._config.readiness_max_staleness_seconds
        )

    def status(self) -> RunnerLeaderElectionRuntimeStatus:
        now_monotonic = self._monotonic_now()
        with self._state_lock:
            lease = self._current_lease
            return RunnerLeaderElectionRuntimeStatus(
                election_id=self._config.election_id,
                holder_id=self._config.holder_id,
                state=self._state,
                ready=self._ready_locked(lease=lease, now_monotonic=now_monotonic),
                leader=lease is not None,
                operations_started=self._operations_started,
                operations_succeeded=self._operations_succeeded,
                operations_failed=self._operations_failed,
                campaigns_won=self._campaigns_won,
                renewals_succeeded=self._renewals_succeeded,
                consecutive_failures=self._consecutive_failures,
                active_mutations=self._active_mutations,
                current_lease=lease,
                last_success_at=self._last_success_at,
                last_failure_at=self._last_failure_at,
            )

    def check_ready(self) -> None:
        if not self.status().ready:
            raise RuntimeError("leader election runtime is not ready")

    def _run(self) -> None:
        fatal = False
        try:
            while not self._stop.is_set():
                try:
                    self.run_once()
                except Exception:
                    pass
                with self._state_lock:
                    leader = self._current_lease is not None
                interval = (
                    self._config.renew_interval_seconds
                    if leader
                    else (self._config.campaign_interval_seconds)
                )
                if self._stop.wait(interval):
                    break
        except BaseException:
            fatal = True
            self._record_failure(count_operation=False)
            with self._state_lock:
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
                raise RuntimeError("leader election runtime cannot be restarted")
            self._stop.clear()
            thread = threading.Thread(
                target=self._run,
                name="kairyu-runner-leader-election",
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
            thread = self._thread
            if self._state != "failed":
                self._state = "stopping"
            self._stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, shutdown_deadline - time.monotonic()))
            if thread.is_alive():
                raise RuntimeError("leader election worker did not stop before deadline")
        with self._state_changed:
            while self._active_mutations:
                remaining = shutdown_deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("controller mutations did not stop before shutdown deadline")
                self._state_changed.wait(timeout=remaining)
        remaining = max(0.0, shutdown_deadline - time.monotonic())
        if not self._operation_lock.acquire(timeout=remaining):
            raise RuntimeError("leader election operation did not stop before deadline")
        release_error: Exception | None = None
        try:
            try:
                self._elector.resign()
            except RunnerNotLeaderError:
                pass
            except Exception as exc:
                release_error = exc
                self._elector.abandon()
                self._record_failure(count_operation=False)
        finally:
            self._operation_lock.release()
        with self._state_lock:
            self._sync_lease_locked()
            self._state = "stopped"
        if release_error is not None:
            raise RuntimeError("leader election lease release failed") from release_error

    def __enter__(self) -> RunnerLeaderElectionRuntime:
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


__all__ = [
    "RunnerLeaderElectionRuntime",
    "RunnerLeaderElectionRuntimeConfig",
    "RunnerLeaderElectionRuntimeStatus",
]
