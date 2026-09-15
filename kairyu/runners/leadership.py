"""Lease-fenced single-writer authority for Runner control-plane mutations."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.runners.drain import RunnerDispatchFence, RunnerDrainController
from kairyu.runners.models import RunnerStatus
from kairyu.runners.observation import RunnerObservationBatch
from kairyu.runners.reconciler import RunnerStatusReconciler

_MAX_LEASE_SECONDS = 86_400.0
_MAX_FENCING_TOKEN = 2**63 - 1
MutationResult = TypeVar("MutationResult")


class InvalidRunnerLeadershipError(RuntimeError):
    """Leadership evidence or backend behavior violates the contract."""


class StaleRunnerLeaderLeaseError(RuntimeError):
    """A lease no longer identifies the active leadership tenure."""


class RunnerNotLeaderError(RuntimeError):
    """A contender attempted a mutation without current authority."""


class RunnerLeaderCapacityError(RuntimeError):
    """The bounded election store cannot admit another election identity."""


def _identity(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL")
    return value


def _aware(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _lease_duration(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("lease_seconds must be a number")
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0 or duration > _MAX_LEASE_SECONDS:
        raise ValueError(f"lease_seconds must be in (0, {_MAX_LEASE_SECONDS:g}]")
    return duration


class RunnerLeaderLease(BaseModel):
    """One immutable leadership tenure backed by a store-owned clock."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["runner-leader-lease-v1"] = "runner-leader-lease-v1"
    election_id: str = Field(max_length=255)
    holder_id: str = Field(max_length=255)
    fencing_token: int = Field(ge=1, le=_MAX_FENCING_TOKEN)
    acquired_at: datetime
    renewed_at: datetime
    lease_until: datetime

    @field_validator("election_id", "holder_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identity(value, name=info.field_name)

    @field_validator("fencing_token", mode="before")
    @classmethod
    def validate_fencing_token(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("fencing_token must be an integer")
        return value

    @field_validator("acquired_at", "renewed_at", "lease_until")
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        return _aware(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_interval(self) -> RunnerLeaderLease:
        if self.renewed_at < self.acquired_at:
            raise ValueError("renewed_at cannot precede acquired_at")
        if self.lease_until <= self.renewed_at:
            raise ValueError("lease_until must follow renewed_at")
        return self

    @property
    def tenure(self) -> tuple[str, str, int]:
        return (self.election_id, self.holder_id, self.fencing_token)


class RunnerWriterAuthority(BaseModel):
    """Store-validated permission for one immediate control-plane mutation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["runner-writer-authority-v1"] = "runner-writer-authority-v1"
    election_id: str = Field(max_length=255)
    holder_id: str = Field(max_length=255)
    fencing_token: int = Field(ge=1, le=_MAX_FENCING_TOKEN)
    validated_at: datetime
    lease_until: datetime

    @field_validator("election_id", "holder_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identity(value, name=info.field_name)

    @field_validator("fencing_token", mode="before")
    @classmethod
    def validate_fencing_token(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("fencing_token must be an integer")
        return value

    @field_validator("validated_at", "lease_until")
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        return _aware(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_window(self) -> RunnerWriterAuthority:
        if self.lease_until <= self.validated_at:
            raise ValueError("writer authority must be validated before lease expiry")
        return self

    @property
    def tenure(self) -> tuple[str, str, int]:
        return (self.election_id, self.holder_id, self.fencing_token)


@runtime_checkable
class RunnerLeaderLeaseStore(Protocol):
    """Shared, linearizable lease store used by every controller contender."""

    def acquire(
        self,
        election_id: str,
        holder_id: str,
        *,
        lease_seconds: float,
    ) -> RunnerLeaderLease | None: ...

    def renew(
        self,
        lease: RunnerLeaderLease,
        *,
        lease_seconds: float,
    ) -> RunnerLeaderLease: ...

    def authorize(self, lease: RunnerLeaderLease) -> RunnerWriterAuthority: ...

    def release(self, lease: RunnerLeaderLease) -> None: ...


def _validated_lease(lease: RunnerLeaderLease) -> RunnerLeaderLease:
    if not isinstance(lease, RunnerLeaderLease):
        raise TypeError("lease must be a RunnerLeaderLease")
    return RunnerLeaderLease.model_validate(lease.model_dump())


class InMemoryRunnerLeaderLeaseStore:
    """Thread-safe executable specification for shared lease semantics.

    This store is process-local and intended for tests and development. A
    production backend must provide the same compare-and-swap behavior over a
    shared durable store and use its authoritative clock.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        max_elections: int = 64,
    ) -> None:
        if type(max_elections) is not int or max_elections < 1 or max_elections > 100_000:
            raise ValueError("max_elections must be an integer in [1, 100000]")
        self._clock = clock
        self._max_elections = max_elections
        self._leases: dict[str, RunnerLeaderLease] = {}
        self._tokens: dict[str, int] = {}
        self._last_observed_at: datetime | None = None
        self._lock = threading.Lock()

    def _now(self) -> datetime:
        now = _aware(self._clock(), name="store clock")
        if self._last_observed_at is not None and now < self._last_observed_at:
            raise InvalidRunnerLeadershipError("leader store clock cannot move backwards")
        self._last_observed_at = now
        return now

    @staticmethod
    def _copy(lease: RunnerLeaderLease) -> RunnerLeaderLease:
        return RunnerLeaderLease.model_validate(lease.model_dump())

    def acquire(
        self,
        election_id: str,
        holder_id: str,
        *,
        lease_seconds: float,
    ) -> RunnerLeaderLease | None:
        election_id = _identity(election_id, name="election_id")
        holder_id = _identity(holder_id, name="holder_id")
        duration = _lease_duration(lease_seconds)
        with self._lock:
            now = self._now()
            current = self._leases.get(election_id)
            if current is not None and current.lease_until > now:
                if current.holder_id == holder_id:
                    return self._copy(current)
                return None
            if election_id not in self._tokens and len(self._tokens) >= self._max_elections:
                raise RunnerLeaderCapacityError("leader election capacity is exhausted")
            previous_token = self._tokens.get(election_id, 0)
            if previous_token >= _MAX_FENCING_TOKEN:
                raise RunnerLeaderCapacityError("leader fencing token is exhausted")
            lease = RunnerLeaderLease(
                election_id=election_id,
                holder_id=holder_id,
                fencing_token=previous_token + 1,
                acquired_at=now,
                renewed_at=now,
                lease_until=now + timedelta(seconds=duration),
            )
            self._tokens[election_id] = lease.fencing_token
            self._leases[election_id] = lease
            return self._copy(lease)

    def renew(
        self,
        lease: RunnerLeaderLease,
        *,
        lease_seconds: float,
    ) -> RunnerLeaderLease:
        lease = _validated_lease(lease)
        duration = _lease_duration(lease_seconds)
        with self._lock:
            now = self._now()
            current = self._leases.get(lease.election_id)
            if current is None or current.lease_until <= now or current.tenure != lease.tenure:
                raise StaleRunnerLeaderLeaseError("leader lease cannot be renewed")
            renewed = RunnerLeaderLease(
                election_id=current.election_id,
                holder_id=current.holder_id,
                fencing_token=current.fencing_token,
                acquired_at=current.acquired_at,
                renewed_at=now,
                lease_until=now + timedelta(seconds=duration),
            )
            self._leases[current.election_id] = renewed
            return self._copy(renewed)

    def authorize(self, lease: RunnerLeaderLease) -> RunnerWriterAuthority:
        lease = _validated_lease(lease)
        with self._lock:
            now = self._now()
            current = self._leases.get(lease.election_id)
            if current is None or current.lease_until <= now or current.tenure != lease.tenure:
                raise StaleRunnerLeaderLeaseError("leader lease is not current")
            return RunnerWriterAuthority(
                election_id=current.election_id,
                holder_id=current.holder_id,
                fencing_token=current.fencing_token,
                validated_at=now,
                lease_until=current.lease_until,
            )

    def release(self, lease: RunnerLeaderLease) -> None:
        lease = _validated_lease(lease)
        with self._lock:
            now = self._now()
            current = self._leases.get(lease.election_id)
            if current is None or current.lease_until <= now or current.tenure != lease.tenure:
                raise StaleRunnerLeaderLeaseError("leader lease cannot be released")
            del self._leases[lease.election_id]


class RunnerLeaderElector:
    """One contender's local view of a shared leader election."""

    def __init__(
        self,
        store: RunnerLeaderLeaseStore,
        *,
        election_id: str,
        holder_id: str,
        lease_seconds: float = 15.0,
    ) -> None:
        if not isinstance(store, RunnerLeaderLeaseStore):
            raise TypeError("store must implement RunnerLeaderLeaseStore")
        self._store = store
        self._election_id = _identity(election_id, name="election_id")
        self._holder_id = _identity(holder_id, name="holder_id")
        self._lease_seconds = _lease_duration(lease_seconds)
        self._lease: RunnerLeaderLease | None = None
        self._lock = threading.RLock()

    @property
    def election_id(self) -> str:
        return self._election_id

    @property
    def holder_id(self) -> str:
        return self._holder_id

    @property
    def lease(self) -> RunnerLeaderLease | None:
        with self._lock:
            return None if self._lease is None else _validated_lease(self._lease)

    def campaign(self) -> RunnerLeaderLease | None:
        with self._lock:
            lease = self._store.acquire(
                self._election_id,
                self._holder_id,
                lease_seconds=self._lease_seconds,
            )
            if lease is None:
                self._lease = None
                return None
            try:
                validated = _validated_lease(lease)
            except (TypeError, ValueError) as error:
                self._lease = None
                raise InvalidRunnerLeadershipError(
                    "leader store returned an invalid lease"
                ) from error
            if validated.election_id != self._election_id or validated.holder_id != self._holder_id:
                self._lease = None
                raise InvalidRunnerLeadershipError(
                    "leader store returned authority for another contender"
                )
            self._lease = validated
            return _validated_lease(validated)

    def renew(self) -> RunnerLeaderLease:
        with self._lock:
            if self._lease is None:
                raise RunnerNotLeaderError("contender has no leader lease to renew")
            try:
                renewed = self._store.renew(
                    self._lease,
                    lease_seconds=self._lease_seconds,
                )
            except StaleRunnerLeaderLeaseError as error:
                self._lease = None
                raise RunnerNotLeaderError("contender lost its leader lease") from error
            try:
                validated = _validated_lease(renewed)
            except (TypeError, ValueError) as error:
                self._lease = None
                raise InvalidRunnerLeadershipError(
                    "leader store returned an invalid renewal"
                ) from error
            if validated.tenure != self._lease.tenure:
                self._lease = None
                raise InvalidRunnerLeadershipError("leader renewal changed the active tenure")
            self._lease = validated
            return _validated_lease(validated)

    def authority(self) -> RunnerWriterAuthority:
        with self._lock:
            if self._lease is None:
                raise RunnerNotLeaderError("contender is not the elected leader")
            try:
                authority = self._store.authorize(self._lease)
            except StaleRunnerLeaderLeaseError as error:
                self._lease = None
                raise RunnerNotLeaderError("contender lost its leader lease") from error
            if not isinstance(authority, RunnerWriterAuthority):
                self._lease = None
                raise InvalidRunnerLeadershipError(
                    "leader store returned an invalid writer authority"
                )
            try:
                validated = RunnerWriterAuthority.model_validate(authority.model_dump())
            except (TypeError, ValueError) as error:
                self._lease = None
                raise InvalidRunnerLeadershipError(
                    "leader store returned an invalid writer authority"
                ) from error
            if (
                validated.tenure != self._lease.tenure
                or validated.lease_until != self._lease.lease_until
                or validated.validated_at < self._lease.renewed_at
            ):
                self._lease = None
                raise InvalidRunnerLeadershipError(
                    "leader store returned stale or conflicting writer authority"
                )
            return validated

    def resign(self) -> bool:
        with self._lock:
            if self._lease is None:
                return False
            lease = self._lease
            try:
                self._store.release(lease)
            except StaleRunnerLeaderLeaseError as error:
                self._lease = None
                raise RunnerNotLeaderError("contender lost its leader lease") from error
            self._lease = None
            return True


class LeaderFencedRunnerController:
    """Allow Runner and future autoscaler mutations only under live authority."""

    def __init__(
        self,
        elector: RunnerLeaderElector,
        reconciler: RunnerStatusReconciler,
    ) -> None:
        if not isinstance(elector, RunnerLeaderElector):
            raise TypeError("elector must be a RunnerLeaderElector")
        if not isinstance(reconciler, RunnerStatusReconciler):
            raise TypeError("reconciler must be a RunnerStatusReconciler")
        self._elector = elector
        self._reconciler = reconciler
        self._last_fencing_token = 0
        self._lock = threading.RLock()

    @property
    def reconciler(self) -> RunnerStatusReconciler:
        return self._reconciler

    @property
    def last_fencing_token(self) -> int:
        with self._lock:
            return self._last_fencing_token

    def mutate(
        self,
        operation: Callable[[RunnerWriterAuthority], MutationResult],
    ) -> MutationResult:
        """Run one synchronous mutation after a fresh store-owned lease check."""

        if not callable(operation):
            raise TypeError("operation must be callable")
        with self._lock:
            authority = self._elector.authority()
            if authority.fencing_token < self._last_fencing_token:
                raise InvalidRunnerLeadershipError("writer fencing token cannot move backwards")
            self._last_fencing_token = authority.fencing_token
            return operation(authority)

    def reconcile(self, batch: RunnerObservationBatch) -> dict[str, RunnerStatus]:
        return self.mutate(lambda _authority: self._reconciler.reconcile(batch))

    def authorize_termination(
        self,
        fence: RunnerDispatchFence,
        *,
        controller: RunnerDrainController,
        at: datetime,
    ) -> RunnerStatus:
        return self.mutate(
            lambda _authority: self._reconciler.authorize_termination(
                fence,
                controller=controller,
                at=at,
            )
        )

    def mutate_autoscaler(
        self,
        operation: Callable[[RunnerWriterAuthority], MutationResult],
    ) -> MutationResult:
        """Fenced seam for the Kubernetes scale actuator implemented in WP3."""

        return self.mutate(operation)
