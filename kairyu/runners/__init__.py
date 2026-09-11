"""Logical Runner lifecycle contracts used by serving controllers."""

from kairyu.runners.lifecycle import (
    InvalidRunnerStartupReportError,
    InvalidRunnerTransitionError,
    complete_startup_phase,
    skip_startup_phase,
    start_startup_phase,
    transition_runner_status,
    validate_runner_transition,
    validate_startup_report_update,
)
from kairyu.runners.models import (
    OPTIONAL_RUNNER_STARTUP_PHASES,
    RUNNER_STARTUP_PHASES,
    RunnerFailure,
    RunnerStartupPhase,
    RunnerStartupPhaseOutcome,
    RunnerStartupPhaseReport,
    RunnerStartupReport,
    RunnerState,
    RunnerStatus,
)

__all__ = [
    "RUNNER_STARTUP_PHASES",
    "OPTIONAL_RUNNER_STARTUP_PHASES",
    "InvalidRunnerStartupReportError",
    "InvalidRunnerTransitionError",
    "RunnerFailure",
    "RunnerStartupPhase",
    "RunnerStartupPhaseOutcome",
    "RunnerStartupPhaseReport",
    "RunnerStartupReport",
    "RunnerState",
    "RunnerStatus",
    "complete_startup_phase",
    "skip_startup_phase",
    "start_startup_phase",
    "transition_runner_status",
    "validate_runner_transition",
    "validate_startup_report_update",
]
