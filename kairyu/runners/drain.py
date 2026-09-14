"""Dispatch fencing and post-fence termination authorization for Runners."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.orchestration.replica import DrainLease, ReplicaPool
from kairyu.runners.lifecycle import _transition_runner_status
from kairyu.runners.models import (
    RunnerState,
    RunnerStatus,
    RunnerTerminationAuthorization,
)


class InvalidRunnerDrainEvidenceError(RuntimeError):
    """Drain evidence is stale, mismatched, or insufficient for termination."""


def _non_empty(value: str, *, name: str) -> str:
    if not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL")
    return value


def _aware(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


class RunnerDispatchFence(BaseModel):
    """Receipt proving that one Runner generation accepts no new placements."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["runner-dispatch-fence-v1"] = "runner-dispatch-fence-v1"
    runner_id: str = Field(max_length=255)
    pod_uid: str = Field(max_length=255)
    fence_id: str = Field(max_length=255)
    fence_sequence: int = Field(gt=0)
    drain_state_version: int = Field(ge=0)
    replica_generation: str = Field(max_length=255)
    dispatch_stopped_at: datetime
    routing_excluded_at: datetime
    active_requests_at_fence: int = Field(ge=0)

    @field_validator("runner_id", "pod_uid", "fence_id", "replica_generation")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator(
        "fence_sequence",
        "drain_state_version",
        "active_requests_at_fence",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        return value

    @field_validator("dispatch_stopped_at", "routing_excluded_at")
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        return _aware(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_timeline(self) -> RunnerDispatchFence:
        if self.routing_excluded_at < self.dispatch_stopped_at:
            raise ValueError("routing exclusion cannot predate dispatch stop")
        return self


class RunnerDrainActivityObservation(BaseModel):
    """Authoritative active count observed after a specific dispatch fence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["runner-drain-activity-v1"] = "runner-drain-activity-v1"
    runner_id: str = Field(max_length=255)
    pod_uid: str = Field(max_length=255)
    fence_id: str = Field(max_length=255)
    fence_sequence: int = Field(gt=0)
    drain_state_version: int = Field(ge=0)
    replica_generation: str = Field(max_length=255)
    observed_at: datetime
    active_requests: int = Field(ge=0)

    @field_validator("runner_id", "pod_uid", "fence_id", "replica_generation")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator(
        "fence_sequence",
        "drain_state_version",
        "active_requests",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        return value

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="observed_at")


@runtime_checkable
class RunnerDrainController(Protocol):
    """Backend-neutral seam for fencing and atomic termination authorization."""

    def begin(
        self,
        status: RunnerStatus,
        *,
        at: datetime | None = None,
    ) -> RunnerDispatchFence: ...

    def observe(
        self,
        fence: RunnerDispatchFence,
        *,
        at: datetime | None = None,
    ) -> RunnerDrainActivityObservation: ...

    def commit_termination(
        self,
        status: RunnerStatus,
        fence: RunnerDispatchFence,
        *,
        at: datetime,
    ) -> RunnerStatus:
        """Atomically revalidate the fence, observe zero, and authorize."""
        ...


@dataclass(frozen=True)
class _OwnedFence:
    receipt: RunnerDispatchFence
    lease: DrainLease


class ReplicaPoolDrainController:
    """Same-event-loop adapter from Runner drains to ``ReplicaPool`` fencing."""

    def __init__(
        self,
        pool: ReplicaPool,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        fence_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._pool = pool
        self._now = now
        self._fence_id_factory = fence_id_factory
        self._sequence = 0
        self._owned: dict[str, _OwnedFence] = {}

    def begin(
        self,
        status: RunnerStatus,
        *,
        at: datetime | None = None,
    ) -> RunnerDispatchFence:
        """Atomically remove a draining Runner from new pool placements."""

        if not isinstance(status, RunnerStatus):
            raise TypeError("status must be a RunnerStatus")
        if status.state is not RunnerState.DRAINING:
            raise InvalidRunnerDrainEvidenceError("dispatch fencing requires a draining Runner")
        if status.pod_uid is None:
            raise InvalidRunnerDrainEvidenceError("dispatch fencing requires a stable Pod UID")
        if status.runner_id not in self._pool.replica_ids:
            raise InvalidRunnerDrainEvidenceError("Runner is absent from the dispatch pool")
        generation = self._pool.entry_generation(status.runner_id)
        existing = self._owned.get(status.runner_id)
        if existing is not None:
            receipt = existing.receipt
            if (
                receipt.pod_uid != status.pod_uid
                or receipt.drain_state_version != status.state_version
                or receipt.replica_generation != generation
            ):
                raise InvalidRunnerDrainEvidenceError(
                    "existing dispatch fence belongs to another Runner generation"
                )
            return receipt

        observed_at = _aware(self._now() if at is None else at, name="at")
        if observed_at < status.observed_at:
            raise InvalidRunnerDrainEvidenceError("dispatch fence cannot predate the Runner status")

        lease = self._pool.acquire_drain(status.runner_id)
        try:
            if not self._pool.is_draining(status.runner_id):
                raise InvalidRunnerDrainEvidenceError(
                    "dispatch pool did not retain the drain fence"
                )
            self._sequence += 1
            receipt = RunnerDispatchFence(
                runner_id=status.runner_id,
                pod_uid=status.pod_uid,
                fence_id=self._fence_id_factory(),
                fence_sequence=self._sequence,
                drain_state_version=status.state_version,
                replica_generation=generation,
                dispatch_stopped_at=observed_at,
                routing_excluded_at=observed_at,
                active_requests_at_fence=self._pool.outstanding_by_id()[status.runner_id],
            )
        except BaseException:
            self._pool.release_drain(status.runner_id, lease)
            raise
        self._owned[status.runner_id] = _OwnedFence(receipt, lease)
        return receipt

    def observe(
        self,
        fence: RunnerDispatchFence,
        *,
        at: datetime | None = None,
    ) -> RunnerDrainActivityObservation:
        """Read a count only while the exact local dispatch fence is retained."""

        if not isinstance(fence, RunnerDispatchFence):
            raise TypeError("fence must be a RunnerDispatchFence")
        owned = self._owned.get(fence.runner_id)
        if owned is None or owned.receipt != fence:
            raise InvalidRunnerDrainEvidenceError("dispatch fence is not owned by this controller")
        if fence.runner_id not in self._pool.replica_ids:
            raise InvalidRunnerDrainEvidenceError(
                "Runner disappeared before its active count was confirmed"
            )
        if self._pool.entry_generation(fence.runner_id) != fence.replica_generation:
            raise InvalidRunnerDrainEvidenceError(
                "Runner generation changed after dispatch fencing"
            )
        if not self._pool.is_draining(fence.runner_id):
            raise InvalidRunnerDrainEvidenceError("dispatch fence is no longer active")
        observed_at = _aware(self._now() if at is None else at, name="at")
        if observed_at < fence.routing_excluded_at:
            raise InvalidRunnerDrainEvidenceError("active count observation must be post-fence")
        return RunnerDrainActivityObservation(
            runner_id=fence.runner_id,
            pod_uid=fence.pod_uid,
            fence_id=fence.fence_id,
            fence_sequence=fence.fence_sequence,
            drain_state_version=fence.drain_state_version,
            replica_generation=fence.replica_generation,
            observed_at=observed_at,
            active_requests=self._pool.outstanding_by_id()[fence.runner_id],
        )

    def commit_termination(
        self,
        status: RunnerStatus,
        fence: RunnerDispatchFence,
        *,
        at: datetime,
    ) -> RunnerStatus:
        """Commit authorization without yielding between revalidation and zero."""

        authorized_at = _aware(at, name="at")
        if status.state in {RunnerState.TERMINATING, RunnerState.TERMINATED}:
            return _authorize_runner_termination_evidence(
                status,
                fence,
                None,
                at=authorized_at,
            )
        activity = self.observe(fence, at=authorized_at)
        return _authorize_runner_termination_evidence(
            status,
            fence,
            activity,
            at=authorized_at,
        )


def _authorize_runner_termination_evidence(
    status: RunnerStatus,
    fence: RunnerDispatchFence,
    activity: RunnerDrainActivityObservation | None,
    *,
    at: datetime,
) -> RunnerStatus:
    """Validate evidence supplied by one atomic drain-controller commit."""

    if not isinstance(status, RunnerStatus):
        raise TypeError("status must be a RunnerStatus")
    if not isinstance(fence, RunnerDispatchFence):
        raise TypeError("fence must be a RunnerDispatchFence")
    authorized_at = _aware(at, name="at")
    identity = (
        fence.runner_id,
        fence.pod_uid,
        fence.fence_id,
        fence.fence_sequence,
        fence.drain_state_version,
        fence.replica_generation,
    )
    if status.runner_id != fence.runner_id or status.pod_uid != fence.pod_uid:
        raise InvalidRunnerDrainEvidenceError("dispatch fence does not match the Runner status")
    if status.state in {RunnerState.TERMINATING, RunnerState.TERMINATED}:
        existing = status.termination_authorization
        assert existing is not None
        existing_identity = (
            existing.runner_id,
            existing.pod_uid,
            existing.fence_id,
            existing.fence_sequence,
            existing.drain_state_version,
            existing.replica_generation,
        )
        if existing_identity != identity:
            raise InvalidRunnerDrainEvidenceError("Runner was authorized by another dispatch fence")
        return status

    if not isinstance(activity, RunnerDrainActivityObservation):
        raise TypeError("activity must be a RunnerDrainActivityObservation")
    activity_identity = (
        activity.runner_id,
        activity.pod_uid,
        activity.fence_id,
        activity.fence_sequence,
        activity.drain_state_version,
        activity.replica_generation,
    )
    if activity_identity != identity:
        raise InvalidRunnerDrainEvidenceError(
            "activity observation does not match the dispatch fence"
        )
    if activity.observed_at < fence.routing_excluded_at:
        raise InvalidRunnerDrainEvidenceError("active count observation must be post-fence")
    if activity.active_requests != 0:
        raise InvalidRunnerDrainEvidenceError(
            "termination requires a post-fence active request count of zero"
        )
    if status.runtime_observed_at is not None and activity.observed_at < status.runtime_observed_at:
        raise InvalidRunnerDrainEvidenceError(
            "termination evidence predates the latest runtime observation"
        )
    if authorized_at < status.observed_at or authorized_at < activity.observed_at:
        raise InvalidRunnerDrainEvidenceError(
            "termination authorization cannot predate its evidence"
        )

    if status.state is not RunnerState.DRAINING:
        raise InvalidRunnerDrainEvidenceError(
            "termination authorization requires a draining Runner"
        )
    if fence.drain_state_version != status.state_version:
        raise InvalidRunnerDrainEvidenceError("dispatch fence references a stale drain state")
    if fence.dispatch_stopped_at < status.state_changed_at:
        raise InvalidRunnerDrainEvidenceError("dispatch fence predates the draining transition")

    authorization = RunnerTerminationAuthorization(
        runner_id=status.runner_id,
        pod_uid=fence.pod_uid,
        fence_id=fence.fence_id,
        fence_sequence=fence.fence_sequence,
        drain_state_version=fence.drain_state_version,
        replica_generation=fence.replica_generation,
        dispatch_stopped_at=fence.dispatch_stopped_at,
        routing_excluded_at=fence.routing_excluded_at,
        activity_observed_at=activity.observed_at,
        authorized_at=authorized_at,
    )
    return _transition_runner_status(
        status,
        RunnerState.TERMINATING,
        at=authorized_at,
        active_requests=0,
        termination_authorization=authorization,
    )


def authorize_runner_termination(
    status: RunnerStatus,
    fence: RunnerDispatchFence,
    *,
    controller: RunnerDrainController,
    at: datetime,
) -> RunnerStatus:
    """Use the controller's atomic commit boundary to authorize termination."""

    if not isinstance(status, RunnerStatus):
        raise TypeError("status must be a RunnerStatus")
    if not isinstance(fence, RunnerDispatchFence):
        raise TypeError("fence must be a RunnerDispatchFence")
    if not isinstance(controller, RunnerDrainController):
        raise TypeError("controller must implement RunnerDrainController")
    if status.state not in {
        RunnerState.DRAINING,
        RunnerState.TERMINATING,
        RunnerState.TERMINATED,
    }:
        raise InvalidRunnerDrainEvidenceError(
            "termination authorization requires a draining Runner"
        )
    authorized_at = _aware(at, name="at")
    committed = controller.commit_termination(status, fence, at=authorized_at)
    if not isinstance(committed, RunnerStatus):
        raise InvalidRunnerDrainEvidenceError("drain controller returned an invalid Runner status")
    if status.state in {RunnerState.TERMINATING, RunnerState.TERMINATED}:
        if committed != status:
            raise InvalidRunnerDrainEvidenceError(
                "drain controller changed an authorized Runner replay"
            )
        return committed

    authorization = committed.termination_authorization
    fence_identity = (
        fence.runner_id,
        fence.pod_uid,
        fence.fence_id,
        fence.fence_sequence,
        fence.drain_state_version,
        fence.replica_generation,
    )
    authorization_identity = (
        (
            authorization.runner_id,
            authorization.pod_uid,
            authorization.fence_id,
            authorization.fence_sequence,
            authorization.drain_state_version,
            authorization.replica_generation,
        )
        if authorization is not None
        else None
    )
    if (
        committed.runner_id != status.runner_id
        or committed.pod_uid != status.pod_uid
        or committed.state is not RunnerState.TERMINATING
        or committed.active_requests != 0
        or committed.state_version != status.state_version + 1
        or committed.state_changed_at != authorized_at
        or committed.observed_at != authorized_at
        or authorization_identity != fence_identity
        or authorization is None
        or authorization.authorized_at != authorized_at
        or authorization.dispatch_stopped_at != fence.dispatch_stopped_at
        or authorization.routing_excluded_at != fence.routing_excluded_at
        or authorization.activity_observed_at < fence.routing_excluded_at
        or (
            status.runtime_observed_at is not None
            and authorization.activity_observed_at < status.runtime_observed_at
        )
    ):
        raise InvalidRunnerDrainEvidenceError(
            "drain controller returned mismatched termination authorization"
        )
    preserved_fields = (
        "release_id",
        "model_id",
        "model_revision",
        "node_name",
        "gpu_uuids",
        "runtime_observed_at",
        "runtime_observation_fingerprint",
        "startup",
    )
    if any(getattr(committed, field) != getattr(status, field) for field in preserved_fields):
        raise InvalidRunnerDrainEvidenceError("drain controller changed immutable Runner evidence")
    return committed
