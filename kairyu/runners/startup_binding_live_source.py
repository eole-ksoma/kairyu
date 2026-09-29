"""Deadline-bounded assembly of placement-binding live authorization state."""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from kairyu.artifacts.placement_hint import NodeModelCachePlacementHintSnapshot
from kairyu.runners.prewarm import ScalingPrewarmPlan
from kairyu.runners.scaling_log import ScalingDecisionRecord
from kairyu.runners.scaling_quota import ScalingQuotaAdmission
from kairyu.runners.startup_binding import RunnerCacheStartupBinding
from kairyu.runners.startup_binding_authority import (
    RunnerCachePlacementBindingAuthorizationDeniedError,
)
from kairyu.runners.startup_binding_live_authority import (
    RunnerCachePlacementBindingLiveState,
    RunnerCachePlacementBindingPinEvidence,
    RunnerCachePlacementBindingPrestageEvidence,
    RunnerCachePlacementBindingTargetState,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


class RunnerCachePlacementBindingCacheState(BaseModel):
    """One coherent cache/pre-stage read from the controller's node sources."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    prewarm_plan: ScalingPrewarmPlan
    prestage_records: tuple[RunnerCachePlacementBindingPrestageEvidence, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    placement_hints: tuple[NodeModelCachePlacementHintSnapshot, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    pin_evidence: tuple[RunnerCachePlacementBindingPinEvidence, ...] = Field(
        min_length=1,
        max_length=100_000,
    )


class _DeadlineBoundReader(Protocol):
    def readiness(
        self,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> None: ...


@runtime_checkable
class RunnerCachePlacementBindingCurrentReader(_DeadlineBoundReader, Protocol):
    """Read the scaling controller's current binding for one candidate target."""

    def read_current(
        self,
        candidate: RunnerCacheStartupBinding,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> RunnerCacheStartupBinding: ...


@runtime_checkable
class RunnerCachePlacementBindingDecisionReader(_DeadlineBoundReader, Protocol):
    """Read the exact durable scaling decision named by a current binding."""

    def read_decision(
        self,
        binding: RunnerCacheStartupBinding,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> ScalingDecisionRecord: ...


@runtime_checkable
class RunnerCachePlacementBindingTargetReader(_DeadlineBoundReader, Protocol):
    """Read the live Kubernetes scale target and applied decision annotations."""

    def read_target(
        self,
        binding: RunnerCacheStartupBinding,
        decision: ScalingDecisionRecord,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> RunnerCachePlacementBindingTargetState: ...


@runtime_checkable
class RunnerCachePlacementBindingQuotaReader(_DeadlineBoundReader, Protocol):
    """Read and derive current Kueue/tenant quota admission."""

    def read_quota(
        self,
        binding: RunnerCacheStartupBinding,
        decision: ScalingDecisionRecord,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> ScalingQuotaAdmission: ...


@runtime_checkable
class RunnerCachePlacementBindingCacheReader(_DeadlineBoundReader, Protocol):
    """Read current prewarm, prestage, hint, and owner-pin evidence coherently."""

    def read_cache(
        self,
        binding: RunnerCacheStartupBinding,
        decision: ScalingDecisionRecord,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> RunnerCachePlacementBindingCacheState: ...


class ComposedRunnerCachePlacementBindingLiveStateSource:
    """Assemble live state from explicit production backend boundaries.

    The current binding is read before and after all other dependencies.  This
    turns replacement into a fail-closed authorization denial and prevents a
    mixed snapshot from being attributed to a superseded binding.  Each backend
    receives the same absolute deadline and a timeout capped by the remaining
    request budget.
    """

    def __init__(
        self,
        *,
        current: RunnerCachePlacementBindingCurrentReader,
        decisions: RunnerCachePlacementBindingDecisionReader,
        targets: RunnerCachePlacementBindingTargetReader,
        quotas: RunnerCachePlacementBindingQuotaReader,
        cache: RunnerCachePlacementBindingCacheReader,
        monotonic_clock=time.monotonic,
        wall_clock=_utc_now,
    ) -> None:
        readers = (
            ("current", current, RunnerCachePlacementBindingCurrentReader),
            ("decisions", decisions, RunnerCachePlacementBindingDecisionReader),
            ("targets", targets, RunnerCachePlacementBindingTargetReader),
            ("quotas", quotas, RunnerCachePlacementBindingQuotaReader),
            ("cache", cache, RunnerCachePlacementBindingCacheReader),
        )
        for name, reader, protocol in readers:
            if not isinstance(reader, protocol):
                raise TypeError(f"{name} must implement {protocol.__name__}")
        if not callable(monotonic_clock) or not callable(wall_clock):
            raise TypeError("clocks must be callable")
        self._current = current
        self._decisions = decisions
        self._targets = targets
        self._quotas = quotas
        self._cache = cache
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock

    def _remaining_timeout(
        self,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> float:
        now = self._monotonic_clock()
        remaining = deadline_monotonic - now
        if remaining <= 0:
            raise TimeoutError("placement-binding live-state deadline expired")
        return min(backend_timeout_s, remaining)

    @staticmethod
    def _validate_budget(deadline_monotonic: float, backend_timeout_s: float) -> None:
        for name, value in (
            ("deadline_monotonic", deadline_monotonic),
            ("backend_timeout_s", backend_timeout_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TypeError(f"{name} must be a finite number")
        if backend_timeout_s <= 0:
            raise ValueError("backend_timeout_s must be positive")

    def _read(self, method, *args, deadline_monotonic: float, backend_timeout_s: float):
        timeout = self._remaining_timeout(deadline_monotonic, backend_timeout_s)
        value = method(
            *args,
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=timeout,
        )
        self._remaining_timeout(deadline_monotonic, backend_timeout_s)
        return value

    @staticmethod
    def _deny_changed_binding() -> None:
        raise RunnerCachePlacementBindingAuthorizationDeniedError(
            "current placement binding changed during live-state assembly"
        )

    def read(
        self,
        binding: RunnerCacheStartupBinding,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> RunnerCachePlacementBindingLiveState:
        """Read every dependency and return one deeply validated live snapshot."""

        if not isinstance(binding, RunnerCacheStartupBinding):
            raise TypeError("binding must be a RunnerCacheStartupBinding")
        candidate = RunnerCacheStartupBinding.model_validate(binding.model_dump())
        self._validate_budget(deadline_monotonic, backend_timeout_s)

        current = self._read(
            self._current.read_current,
            candidate,
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=backend_timeout_s,
        )
        if not isinstance(current, RunnerCacheStartupBinding):
            raise TypeError("current binding reader returned an invalid binding")
        current = RunnerCacheStartupBinding.model_validate(current.model_dump())
        if current != candidate:
            self._deny_changed_binding()

        decision = self._read(
            self._decisions.read_decision,
            current,
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=backend_timeout_s,
        )
        if not isinstance(decision, ScalingDecisionRecord):
            raise TypeError("decision reader returned an invalid decision")
        decision = ScalingDecisionRecord.model_validate(decision.model_dump())

        target = self._read(
            self._targets.read_target,
            current,
            decision,
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=backend_timeout_s,
        )
        if not isinstance(target, RunnerCachePlacementBindingTargetState):
            raise TypeError("target reader returned an invalid target state")
        target = RunnerCachePlacementBindingTargetState.model_validate(target.model_dump())
        quota = self._read(
            self._quotas.read_quota,
            current,
            decision,
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=backend_timeout_s,
        )
        if not isinstance(quota, ScalingQuotaAdmission):
            raise TypeError("quota reader returned an invalid admission")
        quota = ScalingQuotaAdmission.model_validate(quota.model_dump())
        cache = self._read(
            self._cache.read_cache,
            current,
            decision,
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=backend_timeout_s,
        )
        if not isinstance(cache, RunnerCachePlacementBindingCacheState):
            raise TypeError("cache reader returned invalid cache state")
        cache = RunnerCachePlacementBindingCacheState.model_validate(cache.model_dump())

        final_current = self._read(
            self._current.read_current,
            candidate,
            deadline_monotonic=deadline_monotonic,
            backend_timeout_s=backend_timeout_s,
        )
        if not isinstance(final_current, RunnerCacheStartupBinding):
            raise TypeError("current binding reader returned an invalid binding")
        final_current = RunnerCacheStartupBinding.model_validate(final_current.model_dump())
        if final_current != current:
            self._deny_changed_binding()

        observed_at = _aware(self._wall_clock(), name="wall clock result")
        self._remaining_timeout(deadline_monotonic, backend_timeout_s)
        state = RunnerCachePlacementBindingLiveState(
            observed_at=observed_at,
            binding=final_current,
            decision=decision,
            target=target,
            quota_admission=quota,
            prewarm_plan=cache.prewarm_plan,
            prestage_records=cache.prestage_records,
            placement_hints=cache.placement_hints,
            pin_evidence=cache.pin_evidence,
        )
        self._remaining_timeout(deadline_monotonic, backend_timeout_s)
        return state

    def readiness(
        self,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> None:
        """Verify every distinct backend dependency inside the shared budget."""

        self._validate_budget(deadline_monotonic, backend_timeout_s)
        seen: set[int] = set()
        for reader in (
            self._current,
            self._decisions,
            self._targets,
            self._quotas,
            self._cache,
        ):
            if id(reader) in seen:
                continue
            seen.add(id(reader))
            self._read(
                reader.readiness,
                deadline_monotonic=deadline_monotonic,
                backend_timeout_s=backend_timeout_s,
            )


__all__ = [
    "ComposedRunnerCachePlacementBindingLiveStateSource",
    "RunnerCachePlacementBindingCacheReader",
    "RunnerCachePlacementBindingCacheState",
    "RunnerCachePlacementBindingCurrentReader",
    "RunnerCachePlacementBindingDecisionReader",
    "RunnerCachePlacementBindingQuotaReader",
    "RunnerCachePlacementBindingTargetReader",
]
