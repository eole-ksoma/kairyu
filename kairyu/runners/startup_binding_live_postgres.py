"""PostgreSQL readers for live placement-binding authorization."""

from __future__ import annotations

import math
import time
from typing import Protocol, runtime_checkable

from kairyu.runners.scaling_log import ScalingDecisionRecord
from kairyu.runners.startup_admission import (
    RunnerCachePlacementAdmissionConflictError,
    RunnerCachePlacementAdmissionPlan,
)
from kairyu.runners.startup_binding import RunnerCacheStartupBinding
from kairyu.runners.startup_binding_authority import (
    RunnerCachePlacementBindingAuthorizationDeniedError,
)


@runtime_checkable
class _TimedAdmissionPlanStore(Protocol):
    def resolve_with_timeout(
        self,
        target_id: str,
        *,
        timeout_s: float,
    ) -> RunnerCachePlacementAdmissionPlan: ...

    def check_ready_with_timeout(self, *, timeout_s: float) -> None: ...


@runtime_checkable
class _TimedScalingDecisionLog(Protocol):
    def get_with_timeout(
        self,
        decision_id: str,
        *,
        timeout_s: float,
    ) -> ScalingDecisionRecord: ...

    def check_ready_with_timeout(self, *, timeout_s: float) -> None: ...


class PostgresRunnerCachePlacementBindingReader:
    """Read current bindings and exact decisions from bounded PostgreSQL stores.

    One instance implements both D3.12 reader protocols. Passing the same
    instance as ``current`` and ``decisions`` also makes composed readiness
    de-duplicate the shared PostgreSQL dependency boundary.
    """

    def __init__(
        self,
        admission_store: _TimedAdmissionPlanStore,
        decision_log: _TimedScalingDecisionLog,
        *,
        monotonic_clock=time.monotonic,
    ) -> None:
        if not isinstance(admission_store, _TimedAdmissionPlanStore):
            raise TypeError("admission_store must implement bounded PostgreSQL reads")
        if not isinstance(decision_log, _TimedScalingDecisionLog):
            raise TypeError("decision_log must implement bounded PostgreSQL reads")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        self._admission_store = admission_store
        self._decision_log = decision_log
        self._monotonic_clock = monotonic_clock

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

    def _remaining_timeout(
        self,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> float:
        remaining = deadline_monotonic - self._monotonic_clock()
        if remaining <= 0:
            raise TimeoutError("placement-binding PostgreSQL deadline expired")
        return min(backend_timeout_s, remaining)

    def _budget(
        self,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> float:
        self._validate_budget(deadline_monotonic, backend_timeout_s)
        return self._remaining_timeout(deadline_monotonic, backend_timeout_s)

    def read_current(
        self,
        candidate: RunnerCacheStartupBinding,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> RunnerCacheStartupBinding:
        """Read the target's exact current admission binding."""

        if not isinstance(candidate, RunnerCacheStartupBinding):
            raise TypeError("candidate must be a RunnerCacheStartupBinding")
        candidate = RunnerCacheStartupBinding.model_validate(candidate.model_dump())
        timeout = self._budget(deadline_monotonic, backend_timeout_s)
        try:
            plan = self._admission_store.resolve_with_timeout(
                candidate.target_id,
                timeout_s=timeout,
            )
        except RunnerCachePlacementAdmissionConflictError as exc:
            raise RunnerCachePlacementBindingAuthorizationDeniedError(
                "current placement binding is unavailable"
            ) from exc
        self._remaining_timeout(deadline_monotonic, backend_timeout_s)
        if not isinstance(plan, RunnerCachePlacementAdmissionPlan):
            raise TypeError("admission store returned an invalid plan")
        plan = RunnerCachePlacementAdmissionPlan.model_validate(plan.model_dump())
        self._remaining_timeout(deadline_monotonic, backend_timeout_s)
        return RunnerCacheStartupBinding.model_validate(plan.binding.model_dump())

    def read_decision(
        self,
        binding: RunnerCacheStartupBinding,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> ScalingDecisionRecord:
        """Read the exact durable decision named by the current binding."""

        if not isinstance(binding, RunnerCacheStartupBinding):
            raise TypeError("binding must be a RunnerCacheStartupBinding")
        binding = RunnerCacheStartupBinding.model_validate(binding.model_dump())
        timeout = self._budget(deadline_monotonic, backend_timeout_s)
        try:
            decision = self._decision_log.get_with_timeout(
                binding.decision_id,
                timeout_s=timeout,
            )
        except KeyError as exc:
            raise RunnerCachePlacementBindingAuthorizationDeniedError(
                "durable scaling decision is unavailable"
            ) from exc
        self._remaining_timeout(deadline_monotonic, backend_timeout_s)
        if not isinstance(decision, ScalingDecisionRecord):
            raise TypeError("decision log returned an invalid decision")
        decision = ScalingDecisionRecord.model_validate(decision.model_dump())
        self._remaining_timeout(deadline_monotonic, backend_timeout_s)
        return decision

    def readiness(
        self,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> None:
        """Check both durable stores sequentially inside one absolute deadline."""

        timeout = self._budget(deadline_monotonic, backend_timeout_s)
        self._admission_store.check_ready_with_timeout(timeout_s=timeout)
        timeout = self._remaining_timeout(deadline_monotonic, backend_timeout_s)
        self._decision_log.check_ready_with_timeout(timeout_s=timeout)
        self._remaining_timeout(deadline_monotonic, backend_timeout_s)


__all__ = ["PostgresRunnerCachePlacementBindingReader"]
