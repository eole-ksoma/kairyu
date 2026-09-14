"""Pure transition and startup-report operations for managed runners."""

from __future__ import annotations

from datetime import datetime

from kairyu.runners.models import (
    RUNNER_STARTUP_PHASES,
    RunnerFailure,
    RunnerStartupPhase,
    RunnerStartupPhaseOutcome,
    RunnerStartupPhaseReport,
    RunnerStartupReport,
    RunnerState,
    RunnerStatus,
    RunnerTerminationAuthorization,
)


class InvalidRunnerTransitionError(RuntimeError):
    """The requested logical Runner transition is not permitted."""


class InvalidRunnerStartupReportError(RuntimeError):
    """A startup phase update is stale, out of order, or already terminal."""


_ALLOWED_RUNNER_TRANSITIONS = {
    RunnerState.REQUESTED: frozenset(
        {RunnerState.SCHEDULING, RunnerState.DRAINING, RunnerState.UNHEALTHY}
    ),
    RunnerState.SCHEDULING: frozenset(
        {RunnerState.IMAGE_PULL, RunnerState.DRAINING, RunnerState.UNHEALTHY}
    ),
    RunnerState.IMAGE_PULL: frozenset(
        {RunnerState.MODEL_LOADING, RunnerState.DRAINING, RunnerState.UNHEALTHY}
    ),
    RunnerState.MODEL_LOADING: frozenset(
        {RunnerState.WARMING, RunnerState.DRAINING, RunnerState.UNHEALTHY}
    ),
    RunnerState.WARMING: frozenset(
        {RunnerState.READY, RunnerState.DRAINING, RunnerState.UNHEALTHY}
    ),
    RunnerState.READY: frozenset(
        {RunnerState.BUSY, RunnerState.DRAINING, RunnerState.UNHEALTHY}
    ),
    RunnerState.BUSY: frozenset(
        {RunnerState.READY, RunnerState.DRAINING, RunnerState.UNHEALTHY}
    ),
    RunnerState.DRAINING: frozenset({RunnerState.TERMINATING}),
    RunnerState.UNHEALTHY: frozenset({RunnerState.DRAINING}),
    RunnerState.TERMINATING: frozenset({RunnerState.TERMINATED}),
    RunnerState.TERMINATED: frozenset(),
}


def _require_aware(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def validate_runner_transition(current: RunnerState, target: RunnerState) -> None:
    """Accept a legal transition or idempotent observation, otherwise raise."""

    if not isinstance(current, RunnerState) or not isinstance(target, RunnerState):
        raise TypeError("current and target must be RunnerState values")
    if current is target:
        return
    if target not in _ALLOWED_RUNNER_TRANSITIONS[current]:
        raise InvalidRunnerTransitionError(
            f"invalid Runner transition: {current.value} -> {target.value}"
        )


def validate_startup_report_update(
    current: RunnerStartupReport,
    target: RunnerStartupReport,
) -> None:
    """Reject stale, truncated, or forked startup evidence."""

    if not isinstance(current, RunnerStartupReport) or not isinstance(
        target, RunnerStartupReport
    ):
        raise TypeError("current and target must be RunnerStartupReport values")
    if target.runner_id != current.runner_id:
        raise InvalidRunnerStartupReportError("startup runner_id cannot change")
    if target.attempt != current.attempt:
        raise InvalidRunnerStartupReportError(
            "startup attempt changes require a new Runner generation"
        )
    if target.observed_at < current.observed_at:
        raise InvalidRunnerStartupReportError("startup observations cannot move backwards")
    if len(target.phases) < len(current.phases):
        raise InvalidRunnerStartupReportError("startup phase history cannot be truncated")
    for index, current_phase in enumerate(current.phases):
        target_phase = target.phases[index]
        if current_phase.completed_at is not None:
            if target_phase != current_phase:
                raise InvalidRunnerStartupReportError(
                    f"completed startup phase {current_phase.phase.value} is immutable"
                )
            continue
        if (
            target_phase.phase is not current_phase.phase
            or target_phase.started_at != current_phase.started_at
        ):
            raise InvalidRunnerStartupReportError(
                f"in-progress startup phase {current_phase.phase.value} cannot be replaced"
            )
        if target_phase.completed_at is None and target_phase != current_phase:
            raise InvalidRunnerStartupReportError(
                f"in-progress startup phase {current_phase.phase.value} forked"
            )
    if current.phases and current.phases[-1].completed_at is None:
        completed_at = target.phases[len(current.phases) - 1].completed_at
        if completed_at is not None and completed_at < current.observed_at:
            raise InvalidRunnerStartupReportError(
                "startup phase completion cannot predate the prior observation"
            )
    if len(target.phases) > len(current.phases):
        first_appended = target.phases[len(current.phases)]
        if first_appended.started_at < current.observed_at:
            raise InvalidRunnerStartupReportError(
                "new startup phase cannot predate the prior observation"
            )


def _transition_runner_status(
    status: RunnerStatus,
    target: RunnerState,
    *,
    at: datetime,
    active_requests: int | None = None,
    startup: RunnerStartupReport | None = None,
    failure: RunnerFailure | None = None,
    termination_authorization: RunnerTerminationAuthorization | None = None,
) -> RunnerStatus:
    """Core transition path; termination authorization stays module-private."""

    if not isinstance(status, RunnerStatus):
        raise TypeError("status must be a RunnerStatus")
    validate_runner_transition(status.state, target)
    at = _require_aware(at, name="at")
    if at < status.observed_at:
        raise InvalidRunnerTransitionError("Runner observations cannot move backwards")
    if failure is not None and target is not RunnerState.UNHEALTHY:
        raise InvalidRunnerTransitionError(
            "failure can only be supplied for an unhealthy Runner"
        )
    if termination_authorization is not None and target is not RunnerState.TERMINATING:
        raise InvalidRunnerTransitionError(
            "termination authorization can only enter the terminating state"
        )
    authorization = status.termination_authorization
    if target is RunnerState.TERMINATING:
        if status.state is RunnerState.TERMINATING:
            if (
                termination_authorization is not None
                and termination_authorization != authorization
            ):
                raise InvalidRunnerTransitionError(
                    "termination authorization is immutable"
                )
        else:
            authorization = termination_authorization
        if authorization is None:
            raise InvalidRunnerTransitionError(
                "terminating requires drain/dispatch-fence authorization"
            )
        if status.state is RunnerState.DRAINING:
            if active_requests != 0:
                raise InvalidRunnerTransitionError(
                    "terminating requires an explicit active request count of zero"
                )
            if (
                authorization.runner_id != status.runner_id
                or authorization.pod_uid != status.pod_uid
            ):
                raise InvalidRunnerTransitionError(
                    "termination authorization does not match the Runner"
                )
            if authorization.drain_state_version != status.state_version:
                raise InvalidRunnerTransitionError(
                    "termination authorization references a stale drain state"
                )
            if authorization.dispatch_stopped_at < status.state_changed_at:
                raise InvalidRunnerTransitionError(
                    "termination authorization predates the draining transition"
                )
            if (
                status.runtime_observed_at is not None
                and authorization.activity_observed_at < status.runtime_observed_at
            ):
                raise InvalidRunnerTransitionError(
                    "termination authorization predates runtime activity"
                )
            if authorization.authorized_at != at:
                raise InvalidRunnerTransitionError(
                    "termination transition time must match its authorization"
                )
    if startup is not None and status.startup is not None:
        validate_startup_report_update(status.startup, startup)
    same_state = target is status.state
    next_failure = failure
    if target is RunnerState.UNHEALTHY and next_failure is None and same_state:
        next_failure = status.failure
    next_active_requests = active_requests
    if next_active_requests is None:
        # Never manufacture an empty Runner. READY and TERMINATING require a
        # caller-observed zero when the preceding snapshot still owns work.
        next_active_requests = status.active_requests
    values = status.model_dump()
    values.update(
        {
            "state": target,
            "state_version": status.state_version + (not same_state),
            "state_changed_at": status.state_changed_at if same_state else at,
            "observed_at": at,
            "active_requests": next_active_requests,
            "startup": status.startup if startup is None else startup,
            "failure": next_failure if target is RunnerState.UNHEALTHY else None,
            "termination_authorization": (
                authorization
                if target is RunnerState.TERMINATING
                else status.termination_authorization
                if target is RunnerState.TERMINATED
                else None
            ),
        }
    )
    return RunnerStatus.model_validate(values)


def transition_runner_status(
    status: RunnerStatus,
    target: RunnerState,
    *,
    at: datetime,
    active_requests: int | None = None,
    startup: RunnerStartupReport | None = None,
    failure: RunnerFailure | None = None,
) -> RunnerStatus:
    """Return a validated snapshot; drain authorization owns termination entry."""

    return _transition_runner_status(
        status,
        target,
        at=at,
        active_requests=active_requests,
        startup=startup,
        failure=failure,
    )


def start_startup_phase(
    report: RunnerStartupReport,
    phase: RunnerStartupPhase,
    *,
    at: datetime,
) -> RunnerStartupReport:
    """Append the next canonical in-progress startup phase."""

    if not isinstance(report, RunnerStartupReport):
        raise TypeError("report must be a RunnerStartupReport")
    if not isinstance(phase, RunnerStartupPhase):
        raise TypeError("phase must be a RunnerStartupPhase")
    at = _require_aware(at, name="at")
    if at < report.observed_at:
        raise InvalidRunnerStartupReportError("startup observations cannot move backwards")
    if report.current_phase is not None:
        raise InvalidRunnerStartupReportError(
            f"startup phase {report.current_phase.value} is still in progress"
        )
    if report.failure is not None:
        raise InvalidRunnerStartupReportError("failed startup reports are terminal")
    if len(report.phases) == len(RUNNER_STARTUP_PHASES):
        raise InvalidRunnerStartupReportError("startup report is already complete")
    expected = RUNNER_STARTUP_PHASES[len(report.phases)]
    if phase is not expected:
        raise InvalidRunnerStartupReportError(
            f"expected startup phase {expected.value}, got {phase.value}"
        )
    return RunnerStartupReport(
        runner_id=report.runner_id,
        attempt=report.attempt,
        observed_at=at,
        phases=(
            *report.phases,
            RunnerStartupPhaseReport(phase=phase, started_at=at),
        ),
    )


def complete_startup_phase(
    report: RunnerStartupReport,
    phase: RunnerStartupPhase,
    *,
    at: datetime,
    outcome: RunnerStartupPhaseOutcome = RunnerStartupPhaseOutcome.SUCCEEDED,
    failure: RunnerFailure | None = None,
) -> RunnerStartupReport:
    """Complete the current phase and return a new validated report."""

    if not isinstance(report, RunnerStartupReport):
        raise TypeError("report must be a RunnerStartupReport")
    if not isinstance(phase, RunnerStartupPhase):
        raise TypeError("phase must be a RunnerStartupPhase")
    if not isinstance(outcome, RunnerStartupPhaseOutcome):
        raise TypeError("outcome must be a RunnerStartupPhaseOutcome")
    at = _require_aware(at, name="at")
    if report.current_phase is not phase:
        current = None if report.current_phase is None else report.current_phase.value
        raise InvalidRunnerStartupReportError(
            f"cannot complete startup phase {phase.value}; current phase is {current!r}"
        )
    if at < report.observed_at:
        raise InvalidRunnerStartupReportError("startup observations cannot move backwards")
    current_report = report.phases[-1]
    completed = RunnerStartupPhaseReport(
        phase=phase,
        started_at=current_report.started_at,
        completed_at=at,
        outcome=outcome,
        failure=failure,
    )
    return RunnerStartupReport(
        runner_id=report.runner_id,
        attempt=report.attempt,
        observed_at=at,
        phases=(*report.phases[:-1], completed),
    )


def skip_startup_phase(
    report: RunnerStartupReport,
    phase: RunnerStartupPhase,
    *,
    at: datetime,
) -> RunnerStartupReport:
    """Record an explicitly skipped optional phase without hiding a gap."""

    started = start_startup_phase(report, phase, at=at)
    return complete_startup_phase(
        started,
        phase,
        at=at,
        outcome=RunnerStartupPhaseOutcome.SKIPPED,
    )
