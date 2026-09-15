"""Deterministic aggregation of Kubernetes and Runner-owned observations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timedelta

from kairyu.runners.backoff import RunnerFailureGuard
from kairyu.runners.drain import (
    RunnerDispatchFence,
    RunnerDrainController,
    authorize_runner_termination,
)
from kairyu.runners.lifecycle import (
    InvalidRunnerStartupReportError,
    transition_runner_status,
    validate_startup_report_update,
)
from kairyu.runners.models import (
    RunnerFailure,
    RunnerFailureDomainKind,
    RunnerStartupPhase,
    RunnerStartupReport,
    RunnerState,
    RunnerStatus,
)
from kairyu.runners.observation import (
    KubernetesPodPhase,
    RunnerObservation,
    RunnerObservationBatch,
    RunnerRuntimeObservation,
)


class InvalidRunnerObservationError(RuntimeError):
    """The observed inputs are stale, contradictory, or change identity."""


_STARTUP_STATE = {
    RunnerStartupPhase.IMAGE_PULL: RunnerState.IMAGE_PULL,
    RunnerStartupPhase.MODEL_FETCH: RunnerState.MODEL_LOADING,
    RunnerStartupPhase.MODEL_LOAD: RunnerState.MODEL_LOADING,
    RunnerStartupPhase.GRAPH_COMPILE: RunnerState.WARMING,
    RunnerStartupPhase.WARMUP: RunnerState.WARMING,
}

_NORMAL_NEXT = {
    RunnerState.REQUESTED: RunnerState.SCHEDULING,
    RunnerState.SCHEDULING: RunnerState.IMAGE_PULL,
    RunnerState.IMAGE_PULL: RunnerState.MODEL_LOADING,
    RunnerState.MODEL_LOADING: RunnerState.WARMING,
    RunnerState.WARMING: RunnerState.READY,
    RunnerState.READY: RunnerState.BUSY,
    RunnerState.BUSY: RunnerState.READY,
}

_NORMAL_RANK = {
    RunnerState.REQUESTED: 0,
    RunnerState.SCHEDULING: 1,
    RunnerState.IMAGE_PULL: 2,
    RunnerState.MODEL_LOADING: 3,
    RunnerState.WARMING: 4,
    RunnerState.READY: 5,
    RunnerState.BUSY: 5,
}


def runner_is_routing_eligible(
    status: RunnerStatus,
    *,
    serving_gates_ready: bool,
    serving_gates_observed_at: datetime,
    now: datetime,
    max_observation_age: timedelta,
) -> bool:
    """Require both a serving state and a caller-selected freshness lease."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if serving_gates_observed_at.tzinfo is None or serving_gates_observed_at.utcoffset() is None:
        raise ValueError("serving_gates_observed_at must be timezone-aware")
    if max_observation_age <= timedelta(0):
        raise ValueError("max_observation_age must be positive")
    status_age = now - status.observed_at
    gate_age = now - serving_gates_observed_at
    if status_age < timedelta(0) or gate_age < timedelta(0):
        return False
    return (
        status.state in {RunnerState.READY, RunnerState.BUSY}
        and serving_gates_ready
        and status_age <= max_observation_age
        and gate_age <= max_observation_age
    )


def _failure(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    domain: RunnerFailureDomainKind | None = None,
) -> RunnerFailure:
    return RunnerFailure(
        code=code,
        message=message,
        retryable=retryable,
        domain=domain,
    )


def _runtime_fingerprint(runtime: RunnerRuntimeObservation) -> str:
    payload = json.dumps(
        runtime.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _observed_failure(observation: RunnerObservation) -> RunnerFailure | None:
    pod = observation.pod
    runtime = observation.runtime
    if runtime is not None and runtime.startup is not None:
        if runtime.startup.failure is not None:
            return runtime.startup.failure
    if pod is None:
        return None

    terminated = (pod.terminated_reason or "").lower()
    if terminated == "oomkilled":
        return _failure(
            "runner_oom",
            "container terminated: OOMKilled",
            domain=RunnerFailureDomainKind.REVISION,
        )
    if "xid" in terminated:
        return _failure(
            "gpu_xid",
            "container terminated after a GPU Xid error",
            domain=RunnerFailureDomainKind.GPU,
        )
    if pod.exit_code is not None and pod.exit_code != 0:
        return _failure(
            "container_exit_nonzero",
            f"Runner container exited with code {pod.exit_code}",
            domain=RunnerFailureDomainKind.REVISION,
        )
    waiting = (pod.waiting_reason or "").lower()
    if waiting in {"errimagepull", "imagepullbackoff", "invalidimagename"}:
        return _failure(
            "image_pull_failed",
            f"container waiting reason: {pod.waiting_reason}",
            retryable=True,
            domain=RunnerFailureDomainKind.REVISION,
        )
    if waiting == "crashloopbackoff":
        return _failure(
            "container_crash_loop",
            "Runner container entered CrashLoopBackOff",
            retryable=True,
            domain=RunnerFailureDomainKind.REVISION,
        )
    if pod.phase is KubernetesPodPhase.FAILED:
        reason = pod.terminated_reason or "PodFailed"
        return _failure(
            "pod_failed",
            f"Pod entered Failed phase: {reason}",
            domain=RunnerFailureDomainKind.REVISION,
        )
    if pod.phase is KubernetesPodPhase.UNKNOWN:
        return _failure(
            "pod_unknown",
            "Kubernetes could not determine the Pod phase",
            retryable=True,
            domain=RunnerFailureDomainKind.NODE,
        )
    if runtime is not None and runtime.fatal:
        return _failure(
            "readiness_fatal",
            runtime.detail or "Runner reported a fatal readiness condition",
            domain=RunnerFailureDomainKind.REVISION,
        )
    return None


def _startup_state(startup: RunnerStartupReport | None) -> RunnerState:
    if startup is None or not startup.phases:
        return RunnerState.MODEL_LOADING
    if startup.current_phase is not None:
        return _STARTUP_STATE[startup.current_phase]
    if startup.completed:
        return RunnerState.WARMING
    last_phase = startup.phases[-1].phase
    return _STARTUP_STATE[last_phase]


def _missing_serving_gate(observation: RunnerObservation) -> RunnerFailure:
    pod = observation.pod
    runtime = observation.runtime
    if pod is None:
        return _failure("pod_missing", "Runner Pod is absent", retryable=True)
    if not pod.gpu_uuids:
        return _failure(
            "gpu_assignment_missing",
            "Runner has no observed GPU UUID assignment",
            retryable=True,
        )
    if not pod.ready:
        return _failure("pod_not_ready", "Runner Pod readiness is false", retryable=True)
    if not observation.endpoint_ready:
        return _failure(
            "endpoint_not_ready",
            "Runner is absent from ready EndpointSlice targets",
            retryable=True,
        )
    if runtime is None:
        return _failure(
            "runtime_observation_missing",
            "Runner readiness observation is unavailable",
            retryable=True,
        )
    return _failure(
        "readiness_failed",
        runtime.detail or "Runner readiness is false",
        retryable=not runtime.fatal,
    )


def _serving_gates_ready(
    observation: RunnerObservation,
    startup: RunnerStartupReport | None,
) -> bool:
    pod = observation.pod
    runtime = observation.runtime
    return (
        pod is not None
        and pod.phase is KubernetesPodPhase.RUNNING
        and bool(pod.gpu_uuids)
        and pod.ready
        and observation.endpoint_ready
        and runtime is not None
        and runtime.ready
        and startup is not None
        and startup.completed
    )


def _desired_state(
    previous: RunnerStatus | None,
    observation: RunnerObservation,
    *,
    active_requests: int,
    startup: RunnerStartupReport | None,
    serving_gate_loss_confirmed: bool,
) -> tuple[RunnerState, RunnerFailure | None]:
    pod = observation.pod
    if previous is not None:
        if previous.state is RunnerState.TERMINATED:
            return RunnerState.TERMINATED, None
        if previous.state is RunnerState.TERMINATING:
            if pod is None:
                return RunnerState.TERMINATED, None
            return RunnerState.TERMINATING, None
        if previous.state is RunnerState.DRAINING:
            return RunnerState.DRAINING, None
        if previous.state is RunnerState.UNHEALTHY:
            if pod is None or pod.deleting:
                return RunnerState.DRAINING, None
            return RunnerState.UNHEALTHY, previous.failure

    observed_failure = _observed_failure(observation)
    if observed_failure is not None:
        return RunnerState.UNHEALTHY, observed_failure

    if pod is None:
        if previous is None:
            return RunnerState.REQUESTED, None
        return RunnerState.DRAINING, None
    if pod.deleting:
        return RunnerState.DRAINING, None
    if pod.phase is KubernetesPodPhase.SUCCEEDED:
        return RunnerState.UNHEALTHY, _failure(
            "pod_exited",
            "Serving Pod exited without a termination handshake",
            domain=RunnerFailureDomainKind.REVISION,
        )
    if pod.phase is KubernetesPodPhase.PENDING:
        return (
            RunnerState.IMAGE_PULL if pod.node_name is not None else RunnerState.SCHEDULING,
            None,
        )

    if pod.phase is KubernetesPodPhase.RUNNING and (startup is None or not startup.completed):
        return _startup_state(startup), None

    all_serving_gates = _serving_gates_ready(observation, startup)
    if all_serving_gates:
        state = RunnerState.BUSY if active_requests > 0 else RunnerState.READY
        return state, None

    if previous is not None and previous.state in {
        RunnerState.READY,
        RunnerState.BUSY,
    }:
        if not serving_gate_loss_confirmed:
            state = RunnerState.BUSY if active_requests > 0 else RunnerState.READY
            return state, None
        return RunnerState.UNHEALTHY, _missing_serving_gate(observation)
    return RunnerState.WARMING, None


def _validate_identity(previous: RunnerStatus, observation: RunnerObservation) -> None:
    for name in ("runner_id", "release_id", "model_id", "model_revision"):
        if getattr(previous, name) != getattr(observation, name):
            raise InvalidRunnerObservationError(f"Runner {name} cannot change")
    if observation.observed_at < previous.observed_at:
        raise InvalidRunnerObservationError("Runner observations cannot move backwards")
    if observation.pod is not None:
        if previous.node_name is not None and observation.pod.node_name != previous.node_name:
            raise InvalidRunnerObservationError("Runner node_name cannot change")
        if previous.gpu_uuids and observation.pod.gpu_uuids != previous.gpu_uuids:
            raise InvalidRunnerObservationError("Runner gpu_uuids cannot change")
    startup = None if observation.runtime is None else observation.runtime.startup
    if previous.startup is not None and observation.runtime is not None and startup is None:
        raise InvalidRunnerObservationError("startup evidence cannot disappear")
    if previous.startup is not None and startup is not None:
        try:
            validate_startup_report_update(previous.startup, startup)
        except InvalidRunnerStartupReportError as error:
            raise InvalidRunnerObservationError(str(error)) from error


def _advance(
    status: RunnerStatus,
    target: RunnerState,
    *,
    observation: RunnerObservation,
    active_requests: int,
    startup: RunnerStartupReport | None,
    failure: RunnerFailure | None,
) -> RunnerStatus:
    while status.state is not target:
        current = status.state
        if target is RunnerState.UNHEALTHY:
            next_state = RunnerState.UNHEALTHY
        elif target is RunnerState.DRAINING:
            next_state = RunnerState.DRAINING
        elif target is RunnerState.TERMINATING:
            next_state = (
                RunnerState.TERMINATING
                if current in {RunnerState.DRAINING, RunnerState.UNHEALTHY}
                else RunnerState.DRAINING
            )
        elif target is RunnerState.TERMINATED:
            next_state = RunnerState.TERMINATED
        else:
            current_rank = _NORMAL_RANK.get(current)
            target_rank = _NORMAL_RANK[target]
            if current_rank is None or current_rank > target_rank:
                next_state = RunnerState.UNHEALTHY
                failure = _failure(
                    "observation_regressed",
                    f"observed state regressed from {current.value} to {target.value}",
                )
                target = next_state
            else:
                next_state = _NORMAL_NEXT[current]

        next_active = (
            active_requests
            if next_state
            in {
                RunnerState.BUSY,
                RunnerState.DRAINING,
                RunnerState.UNHEALTHY,
            }
            else 0
        )
        status = transition_runner_status(
            status,
            next_state,
            at=observation.observed_at,
            active_requests=next_active,
            startup=startup,
            failure=failure if next_state is RunnerState.UNHEALTHY else None,
        )
    if status.observed_at != observation.observed_at or (
        target in {RunnerState.BUSY, RunnerState.DRAINING, RunnerState.UNHEALTHY}
        and status.active_requests != active_requests
    ):
        status = transition_runner_status(
            status,
            target,
            at=observation.observed_at,
            active_requests=active_requests,
            startup=startup,
            failure=failure if target is RunnerState.UNHEALTHY else None,
        )
    return status


def _effective_observation(
    previous: RunnerStatus | None,
    observation: RunnerObservation,
    max_runtime_observation_age: timedelta,
    runtime_clock_skew_tolerance: timedelta,
) -> RunnerObservation:
    runtime = observation.runtime
    if (
        runtime is not None
        and previous is not None
        and previous.runtime_observed_at == runtime.observed_at
    ):
        fingerprint = previous.runtime_observation_fingerprint
        if fingerprint is None:
            return observation.model_copy(update={"runtime": None})
        if _runtime_fingerprint(runtime) != fingerprint:
            raise InvalidRunnerObservationError(
                "runtime payload cannot change at an equal observation timestamp"
            )
    if runtime is not None and (
        observation.observed_at - runtime.observed_at > max_runtime_observation_age
        or runtime.observed_at - observation.observed_at > runtime_clock_skew_tolerance
        or (
            previous is not None
            and previous.runtime_observed_at is not None
            and runtime.observed_at < previous.runtime_observed_at
        )
    ):
        return observation.model_copy(update={"runtime": None})
    return observation


def reconcile_runner_status(
    previous: RunnerStatus | None,
    observation: RunnerObservation,
    *,
    max_runtime_observation_age: timedelta = timedelta(seconds=5),
    runtime_clock_skew_tolerance: timedelta = timedelta(seconds=2),
    serving_gate_loss_confirmed: bool = True,
) -> RunnerStatus:
    """Combine one snapshot without changing Kubernetes or routing state."""

    if max_runtime_observation_age <= timedelta(0):
        raise ValueError("max_runtime_observation_age must be positive")
    if runtime_clock_skew_tolerance < timedelta(0):
        raise ValueError("runtime_clock_skew_tolerance cannot be negative")
    observation = _effective_observation(
        previous,
        observation,
        max_runtime_observation_age,
        runtime_clock_skew_tolerance,
    )
    runtime = observation.runtime
    if previous is not None:
        _validate_identity(previous, observation)

    startup = None if runtime is None else runtime.startup
    if startup is None and previous is not None:
        startup = previous.startup
    if runtime is not None:
        active_requests = runtime.active_requests
    elif previous is not None:
        active_requests = previous.active_requests
    else:
        active_requests = 0
    if active_requests > 0 and (startup is None or not startup.completed):
        raise InvalidRunnerObservationError("active requests require completed startup evidence")

    target, failure = _desired_state(
        previous,
        observation,
        active_requests=active_requests,
        startup=startup,
        serving_gate_loss_confirmed=serving_gate_loss_confirmed,
    )
    if previous is None:
        previous = RunnerStatus(
            runner_id=observation.runner_id,
            release_id=observation.release_id,
            model_id=observation.model_id,
            model_revision=observation.model_revision,
            state=RunnerState.REQUESTED,
            state_changed_at=observation.observed_at,
            observed_at=observation.observed_at,
            node_name=None if observation.pod is None else observation.pod.node_name,
            pod_uid=None if observation.pod is None else observation.pod.uid,
            gpu_uuids=() if observation.pod is None else observation.pod.gpu_uuids,
            runtime_observed_at=None if runtime is None else runtime.observed_at,
            runtime_observation_fingerprint=(
                None if runtime is None else _runtime_fingerprint(runtime)
            ),
            startup=startup,
        )

    status = _advance(
        previous,
        target,
        observation=observation,
        active_requests=active_requests,
        startup=startup,
        failure=failure,
    )
    values = status.model_dump()
    if observation.pod is not None:
        values.update(
            {
                "node_name": observation.pod.node_name,
                "pod_uid": observation.pod.uid,
                "gpu_uuids": observation.pod.gpu_uuids,
            }
        )
    if runtime is not None:
        values["runtime_observed_at"] = runtime.observed_at
        values["runtime_observation_fingerprint"] = _runtime_fingerprint(runtime)
    return RunnerStatus.model_validate(values)


class RunnerStatusReconciler:
    """Retain Runner snapshots and reconcile complete read-only watch epochs."""

    def __init__(
        self,
        statuses: Mapping[str, RunnerStatus] | None = None,
        *,
        serving_gate_failure_grace: timedelta = timedelta(seconds=5),
        max_runtime_observation_age: timedelta = timedelta(seconds=5),
        runtime_clock_skew_tolerance: timedelta = timedelta(seconds=2),
        failure_guard: RunnerFailureGuard | None = None,
    ) -> None:
        if serving_gate_failure_grace <= timedelta(0):
            raise ValueError("serving_gate_failure_grace must be positive")
        if max_runtime_observation_age <= timedelta(0):
            raise ValueError("max_runtime_observation_age must be positive")
        if runtime_clock_skew_tolerance < timedelta(0):
            raise ValueError("runtime_clock_skew_tolerance cannot be negative")
        initial = {} if statuses is None else dict(statuses)
        for runner_id, status in initial.items():
            if runner_id != status.runner_id:
                raise ValueError("initial status keys must match Runner IDs")
        if failure_guard is not None and not isinstance(failure_guard, RunnerFailureGuard):
            raise TypeError("failure_guard must be a RunnerFailureGuard")
        self._statuses = initial
        self._failure_guard = failure_guard
        self._serving_gate_failure_grace = serving_gate_failure_grace
        self._max_runtime_observation_age = max_runtime_observation_age
        self._runtime_clock_skew_tolerance = runtime_clock_skew_tolerance
        self._gate_loss_started_at: dict[str, datetime] = {}
        self._serving_gates: dict[str, bool] = {runner_id: False for runner_id in initial}
        self._serving_gate_observed_at: dict[str, datetime] = {}
        self._last_observed_at: datetime | None = None
        self._last_source_started_at: datetime | None = None
        self._source_epochs: dict[str, int] = {}

    @property
    def statuses(self) -> dict[str, RunnerStatus]:
        return dict(self._statuses)

    @property
    def failure_guard(self) -> RunnerFailureGuard | None:
        return self._failure_guard

    def routing_eligible(
        self,
        runner_id: str,
        *,
        now: datetime,
        max_observation_age: timedelta,
    ) -> bool:
        status = self._statuses.get(runner_id)
        if status is None:
            return False
        gate_observed_at = self._serving_gate_observed_at.get(runner_id)
        if gate_observed_at is None:
            return False
        return runner_is_routing_eligible(
            status,
            serving_gates_ready=self._serving_gates.get(runner_id, False),
            serving_gates_observed_at=gate_observed_at,
            now=now,
            max_observation_age=max_observation_age,
        )

    def authorize_termination(
        self,
        fence: RunnerDispatchFence,
        *,
        controller: RunnerDrainController,
        at: datetime,
    ) -> RunnerStatus:
        """Persist one fence-bound transition in the in-memory status view."""

        status = self._statuses.get(fence.runner_id)
        if status is None:
            raise InvalidRunnerObservationError("termination fence references an unknown Runner")
        authorized = authorize_runner_termination(
            status,
            fence,
            controller=controller,
            at=at,
        )
        updated = dict(self._statuses)
        updated[fence.runner_id] = authorized
        self._statuses = updated
        self._serving_gates[fence.runner_id] = False
        self._gate_loss_started_at.pop(fence.runner_id, None)
        return authorized

    def reconcile(self, batch: RunnerObservationBatch) -> dict[str, RunnerStatus]:
        """Apply one complete epoch atomically and infer disappeared Pods."""

        if self._last_observed_at is not None and batch.observed_at < self._last_observed_at:
            raise InvalidRunnerObservationError("observation batches cannot move backwards")
        if (
            self._last_source_started_at is not None
            and batch.source_started_at < self._last_source_started_at
        ):
            raise InvalidRunnerObservationError("observation source times cannot move backwards")
        previous_epoch = self._source_epochs.get(batch.source_id)
        if previous_epoch is not None and batch.source_epoch <= previous_epoch:
            raise InvalidRunnerObservationError("source epochs must increase monotonically")
        observed = {item.runner_id: item for item in batch.runners}
        missing_runtime = {item.runner_id: item for item in batch.missing_runner_runtime}
        unknown_runtime = set(missing_runtime) - set(self._statuses)
        if unknown_runtime:
            raise InvalidRunnerObservationError(
                f"runtime rows reference unknown missing Runners: {sorted(unknown_runtime)!r}"
            )
        updated: dict[str, RunnerStatus] = {}
        gate_loss_started_at = dict(self._gate_loss_started_at)
        serving_gates: dict[str, bool] = {}
        serving_gate_observed_at: dict[str, datetime] = {}
        for runner_id in sorted(set(self._statuses) | set(observed)):
            previous = self._statuses.get(runner_id)
            observation = observed.get(runner_id)
            if observation is None:
                assert previous is not None
                observation = RunnerObservation(
                    runner_id=previous.runner_id,
                    release_id=previous.release_id,
                    model_id=previous.model_id,
                    model_revision=previous.model_revision,
                    observed_at=batch.observed_at,
                    runtime=missing_runtime.get(runner_id),
                )
            effective = _effective_observation(
                previous,
                observation,
                self._max_runtime_observation_age,
                self._runtime_clock_skew_tolerance,
            )
            startup = (
                previous.startup
                if effective.runtime is None and previous is not None
                else None
                if effective.runtime is None
                else effective.runtime.startup
            )
            gate_ready = _serving_gates_ready(effective, startup)
            serving_gates[runner_id] = gate_ready
            serving_gate_observed_at[runner_id] = min(
                batch.source_started_at,
                (
                    batch.source_started_at
                    if effective.runtime is None
                    else effective.runtime.observed_at
                ),
            )
            previously_serving = previous is not None and previous.state in {
                RunnerState.READY,
                RunnerState.BUSY,
            }
            if previously_serving and not gate_ready and effective.pod is not None:
                gate_loss_started_at.setdefault(runner_id, batch.source_started_at)
            else:
                gate_loss_started_at.pop(runner_id, None)
            gate_loss_confirmed = (
                runner_id in gate_loss_started_at
                and batch.source_started_at - gate_loss_started_at[runner_id]
                >= self._serving_gate_failure_grace
            )
            updated[runner_id] = reconcile_runner_status(
                previous,
                effective,
                max_runtime_observation_age=self._max_runtime_observation_age,
                runtime_clock_skew_tolerance=self._runtime_clock_skew_tolerance,
                serving_gate_loss_confirmed=gate_loss_confirmed,
            )
        if self._failure_guard is not None:
            self._failure_guard.record_many(updated.values(), at=batch.observed_at)
        self._statuses = updated
        self._gate_loss_started_at = gate_loss_started_at
        self._serving_gates = serving_gates
        self._serving_gate_observed_at = serving_gate_observed_at
        self._last_observed_at = batch.observed_at
        self._last_source_started_at = batch.source_started_at
        self._source_epochs[batch.source_id] = batch.source_epoch
        return dict(updated)
