"""Executable contract for logical Runner state and startup reporting."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest
from pydantic import ValidationError

from kairyu.runners import (
    RUNNER_STARTUP_PHASES,
    InvalidRunnerStartupReportError,
    InvalidRunnerTransitionError,
    RunnerFailure,
    RunnerStartupPhase,
    RunnerStartupPhaseOutcome,
    RunnerStartupPhaseReport,
    RunnerStartupReport,
    RunnerState,
    RunnerStatus,
    complete_startup_phase,
    skip_startup_phase,
    start_startup_phase,
    transition_runner_status,
    validate_runner_transition,
    validate_startup_report_update,
)

NOW = datetime(2026, 9, 11, 5, 0, tzinfo=UTC)


def _complete_startup(*, runner_id: str = "runner-a") -> RunnerStartupReport:
    report = RunnerStartupReport(runner_id=runner_id, observed_at=NOW)
    at = NOW
    for phase in RUNNER_STARTUP_PHASES:
        at += timedelta(seconds=1)
        if phase is RunnerStartupPhase.GRAPH_COMPILE:
            report = skip_startup_phase(report, phase, at=at)
        else:
            report = start_startup_phase(report, phase, at=at)
            at += timedelta(seconds=1)
            report = complete_startup_phase(report, phase, at=at)
    return report


def _status(
    state: RunnerState = RunnerState.REQUESTED,
    **updates,
) -> RunnerStatus:
    values = {
        "runner_id": "runner-a",
        "release_id": "release-sha256:a",
        "model_id": "qwen",
        "model_revision": "model-sha256:b",
        "state": state,
        "state_changed_at": NOW,
        "observed_at": NOW,
    }
    startup = updates.get("startup")
    if isinstance(startup, RunnerStartupReport) and "observed_at" not in updates:
        values["observed_at"] = startup.observed_at
    values.update(updates)
    return RunnerStatus(**values)


def _warmup_in_progress() -> RunnerStartupReport:
    completed = _complete_startup()
    warmup = completed.phases[-1]
    in_progress = RunnerStartupPhaseReport(
        phase=RunnerStartupPhase.WARMUP,
        started_at=warmup.started_at,
    )
    return RunnerStartupReport(
        runner_id=completed.runner_id,
        attempt=completed.attempt,
        observed_at=warmup.started_at,
        phases=(*completed.phases[:-1], in_progress),
    )


def test_normal_runner_lifecycle_and_busy_ready_loop_are_legal() -> None:
    path = (
        RunnerState.REQUESTED,
        RunnerState.SCHEDULING,
        RunnerState.IMAGE_PULL,
        RunnerState.MODEL_LOADING,
        RunnerState.WARMING,
        RunnerState.READY,
        RunnerState.BUSY,
        RunnerState.READY,
        RunnerState.DRAINING,
        RunnerState.TERMINATING,
        RunnerState.TERMINATED,
    )
    for current, target in pairwise(path):
        validate_runner_transition(current, target)
    for state in RunnerState:
        validate_runner_transition(state, state)


@pytest.mark.parametrize(
    "state",
    [
        RunnerState.REQUESTED,
        RunnerState.SCHEDULING,
        RunnerState.IMAGE_PULL,
        RunnerState.MODEL_LOADING,
        RunnerState.WARMING,
        RunnerState.READY,
        RunnerState.BUSY,
    ],
)
def test_active_runner_states_can_fail_closed(state: RunnerState) -> None:
    validate_runner_transition(state, RunnerState.UNHEALTHY)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (RunnerState.REQUESTED, RunnerState.READY),
        (RunnerState.SCHEDULING, RunnerState.MODEL_LOADING),
        (RunnerState.WARMING, RunnerState.BUSY),
        (RunnerState.DRAINING, RunnerState.READY),
        (RunnerState.DRAINING, RunnerState.UNHEALTHY),
        (RunnerState.UNHEALTHY, RunnerState.READY),
        (RunnerState.UNHEALTHY, RunnerState.TERMINATING),
        (RunnerState.TERMINATING, RunnerState.READY),
        (RunnerState.TERMINATED, RunnerState.REQUESTED),
    ],
)
def test_runner_lifecycle_rejects_skips_and_resurrection(
    current: RunnerState,
    target: RunnerState,
) -> None:
    with pytest.raises(
        InvalidRunnerTransitionError,
        match=f"{current.value} -> {target.value}",
    ):
        validate_runner_transition(current, target)


def test_runner_transition_requires_typed_states() -> None:
    with pytest.raises(TypeError, match="RunnerState"):
        validate_runner_transition("ready", RunnerState.BUSY)  # type: ignore[arg-type]


def test_startup_report_records_ordered_phases_without_mutating_history() -> None:
    initial = RunnerStartupReport(runner_id="runner-a", observed_at=NOW)
    started = start_startup_phase(
        initial,
        RunnerStartupPhase.IMAGE_PULL,
        at=NOW + timedelta(seconds=1),
    )
    assert initial.phases == ()
    assert started.current_phase is RunnerStartupPhase.IMAGE_PULL
    assert started.phases[0].duration_seconds is None

    completed = complete_startup_phase(
        started,
        RunnerStartupPhase.IMAGE_PULL,
        at=NOW + timedelta(seconds=4),
    )
    assert started.current_phase is RunnerStartupPhase.IMAGE_PULL
    assert completed.current_phase is None
    assert completed.phases[0].duration_seconds == 3
    assert completed.phases[0].outcome is RunnerStartupPhaseOutcome.SUCCEEDED


def test_complete_startup_report_supports_explicit_optional_skip() -> None:
    report = _complete_startup()
    assert report.completed is True
    assert report.failure is None
    assert tuple(phase.phase for phase in report.phases) == RUNNER_STARTUP_PHASES
    graph = report.phases[RUNNER_STARTUP_PHASES.index(RunnerStartupPhase.GRAPH_COMPILE)]
    assert graph.outcome is RunnerStartupPhaseOutcome.SKIPPED
    assert graph.duration_seconds == 0


def test_required_startup_phases_cannot_be_skipped() -> None:
    report = RunnerStartupReport(runner_id="runner-a", observed_at=NOW)
    started = start_startup_phase(
        report,
        RunnerStartupPhase.IMAGE_PULL,
        at=NOW + timedelta(seconds=1),
    )
    with pytest.raises(ValidationError, match="cannot be skipped"):
        complete_startup_phase(
            started,
            RunnerStartupPhase.IMAGE_PULL,
            at=NOW + timedelta(seconds=1),
            outcome=RunnerStartupPhaseOutcome.SKIPPED,
        )


def test_failed_startup_phase_requires_sanitized_failure_and_is_terminal() -> None:
    report = RunnerStartupReport(runner_id="runner-a", observed_at=NOW)
    report = start_startup_phase(
        report,
        RunnerStartupPhase.IMAGE_PULL,
        at=NOW + timedelta(seconds=1),
    )
    failure = RunnerFailure(
        code="image_pull_failed",
        message="registry unavailable",
        retryable=True,
    )
    report = complete_startup_phase(
        report,
        RunnerStartupPhase.IMAGE_PULL,
        at=NOW + timedelta(seconds=2),
        outcome=RunnerStartupPhaseOutcome.FAILED,
        failure=failure,
    )
    assert report.failure == failure
    assert report.completed is False
    with pytest.raises(InvalidRunnerStartupReportError, match="terminal"):
        start_startup_phase(
            report,
            RunnerStartupPhase.MODEL_FETCH,
            at=NOW + timedelta(seconds=3),
        )


def test_startup_helpers_reject_gaps_overlap_and_stale_observations() -> None:
    report = RunnerStartupReport(runner_id="runner-a", observed_at=NOW)
    with pytest.raises(InvalidRunnerStartupReportError, match="expected"):
        start_startup_phase(
            report,
            RunnerStartupPhase.MODEL_FETCH,
            at=NOW + timedelta(seconds=1),
        )
    started = start_startup_phase(
        report,
        RunnerStartupPhase.IMAGE_PULL,
        at=NOW + timedelta(seconds=1),
    )
    with pytest.raises(InvalidRunnerStartupReportError, match="in progress"):
        start_startup_phase(
            started,
            RunnerStartupPhase.MODEL_FETCH,
            at=NOW + timedelta(seconds=2),
        )
    with pytest.raises(InvalidRunnerStartupReportError, match="backwards"):
        complete_startup_phase(
            started,
            RunnerStartupPhase.IMAGE_PULL,
            at=NOW,
        )


def test_startup_phase_completion_shape_is_validated() -> None:
    with pytest.raises(ValidationError, match="in-progress"):
        RunnerStartupPhaseReport(
            phase=RunnerStartupPhase.IMAGE_PULL,
            started_at=NOW,
            outcome=RunnerStartupPhaseOutcome.SUCCEEDED,
        )
    phase = RunnerStartupPhaseReport.model_construct(
        phase=RunnerStartupPhase.IMAGE_PULL,
        started_at=NOW,
        completed_at=NOW + timedelta(seconds=1),
        outcome=None,
        failure=None,
    )
    with pytest.raises(ValidationError):
        RunnerStartupPhaseReport.model_validate(phase.model_dump())


def test_runner_status_is_frozen_bounded_and_identity_consistent() -> None:
    status = _status()
    with pytest.raises(ValidationError, match="frozen"):
        status.state = RunnerState.SCHEDULING  # type: ignore[misc]
    with pytest.raises(ValidationError, match="Extra inputs"):
        RunnerStatus(**status.model_dump(), secret="do-not-accept")
    with pytest.raises(ValidationError, match="startup runner_id"):
        _status(startup=RunnerStartupReport(runner_id="other", observed_at=NOW))
    skewed = _status(
        startup=RunnerStartupReport(
            runner_id="runner-a",
            observed_at=NOW + timedelta(seconds=1),
        ),
        observed_at=NOW,
    )
    assert skewed.startup is not None
    with pytest.raises(ValidationError, match="gpu_uuids must be unique"):
        _status(gpu_uuids=("GPU-a", "GPU-a"))
    with pytest.raises(ValidationError, match="state_version must be an integer"):
        _status(state_version=True)
    with pytest.raises(ValidationError, match="requires runtime_observed_at"):
        _status(runtime_observation_fingerprint="0" * 64)
    with pytest.raises(ValidationError, match="String should match pattern"):
        _status(
            runtime_observed_at=NOW,
            runtime_observation_fingerprint="not-a-sha256",
        )
    with pytest.raises(ValidationError, match="at most 255 characters"):
        _status(runner_id="r" * 256)
    with pytest.raises(ValidationError, match="attempt must be an integer"):
        RunnerStartupReport(runner_id="runner-a", observed_at=NOW, attempt=True)


def test_ready_busy_and_unhealthy_snapshot_invariants() -> None:
    startup = _complete_startup()
    with pytest.raises(ValidationError, match="completed startup"):
        _status(RunnerState.READY)
    with pytest.raises(ValidationError, match="at least one active"):
        _status(RunnerState.BUSY, startup=startup)
    with pytest.raises(ValidationError, match="cannot have active"):
        _status(RunnerState.READY, startup=startup, active_requests=1)
    with pytest.raises(ValidationError, match="require failure"):
        _status(RunnerState.UNHEALTHY)
    with pytest.raises(ValidationError, match="only unhealthy"):
        _status(
            failure=RunnerFailure(code="unexpected", message="not unhealthy")
        )
    failure = RunnerFailure(code="runner_failed", message="failed")
    for state in (RunnerState.DRAINING, RunnerState.UNHEALTHY):
        with pytest.raises(ValidationError, match="active requests require"):
            _status(
                state,
                active_requests=1,
                failure=failure if state is RunnerState.UNHEALTHY else None,
            )


def test_transition_runner_status_is_monotonic_versioned_and_pure() -> None:
    initial = _status()
    scheduling = transition_runner_status(
        initial,
        RunnerState.SCHEDULING,
        at=NOW + timedelta(seconds=1),
    )
    observed = transition_runner_status(
        scheduling,
        RunnerState.SCHEDULING,
        at=NOW + timedelta(seconds=2),
    )
    assert initial.state is RunnerState.REQUESTED
    assert scheduling.state is RunnerState.SCHEDULING
    assert scheduling.state_version == 1
    assert observed.state_version == 1
    assert observed.state_changed_at == scheduling.state_changed_at
    assert observed.observed_at == NOW + timedelta(seconds=2)
    with pytest.raises(InvalidRunnerTransitionError, match="backwards"):
        transition_runner_status(
            observed,
            RunnerState.SCHEDULING,
            at=NOW + timedelta(seconds=1),
        )


def test_transition_to_ready_busy_and_failure_uses_explicit_evidence() -> None:
    startup = _complete_startup()
    warming = _status(RunnerState.WARMING, startup=startup)
    ready = transition_runner_status(
        warming,
        RunnerState.READY,
        at=startup.observed_at + timedelta(seconds=1),
    )
    with pytest.raises(ValidationError, match="at least one active"):
        transition_runner_status(
            ready,
            RunnerState.BUSY,
            at=ready.observed_at + timedelta(seconds=1),
        )
    busy = transition_runner_status(
        ready,
        RunnerState.BUSY,
        at=ready.observed_at + timedelta(seconds=1),
        active_requests=2,
    )
    failure = RunnerFailure(code="gpu_xid", message="GPU unavailable")
    with pytest.raises(InvalidRunnerTransitionError, match="only be supplied"):
        transition_runner_status(
            ready,
            RunnerState.DRAINING,
            at=ready.observed_at + timedelta(seconds=1),
            failure=failure,
        )
    unhealthy = transition_runner_status(
        busy,
        RunnerState.UNHEALTHY,
        at=busy.observed_at + timedelta(seconds=1),
        failure=failure,
    )
    assert unhealthy.active_requests == 2
    assert unhealthy.failure == failure


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (RunnerState.BUSY, RunnerState.READY),
    ],
)
def test_transition_never_erases_inflight_work_without_an_explicit_zero(
    source: RunnerState,
    target: RunnerState,
) -> None:
    startup = _complete_startup()
    values = {
        "startup": startup,
        "active_requests": 2,
    }
    if source is RunnerState.UNHEALTHY:
        values["failure"] = RunnerFailure(code="runner_failed", message="failed")
    status = _status(source, **values)

    with pytest.raises(ValidationError, match="cannot have active"):
        transition_runner_status(
            status,
            target,
            at=status.observed_at + timedelta(seconds=1),
        )

    transitioned = transition_runner_status(
        status,
        target,
        at=status.observed_at + timedelta(seconds=1),
        active_requests=0,
    )
    assert transitioned.state is target
    assert transitioned.active_requests == 0


def test_transition_to_terminating_requires_handshake_authorization() -> None:
    draining = _status(RunnerState.DRAINING, pod_uid="runner-a")
    with pytest.raises(InvalidRunnerTransitionError, match="dispatch-fence"):
        transition_runner_status(
            draining,
            RunnerState.TERMINATING,
            at=NOW + timedelta(seconds=1),
            active_requests=0,
        )


def test_startup_report_update_is_monotonic_and_completed_history_is_immutable() -> None:
    current = _warmup_in_progress()
    shorter = RunnerStartupReport(
        runner_id=current.runner_id,
        attempt=current.attempt,
        observed_at=current.observed_at + timedelta(seconds=1),
        phases=current.phases[:-1],
    )
    with pytest.raises(InvalidRunnerStartupReportError, match="truncated"):
        validate_startup_report_update(current, shorter)

    changed_attempt = RunnerStartupReport(
        runner_id=current.runner_id,
        attempt=current.attempt + 1,
        observed_at=current.observed_at + timedelta(seconds=1),
        phases=current.phases,
    )
    with pytest.raises(InvalidRunnerStartupReportError, match="new Runner generation"):
        validate_startup_report_update(current, changed_attempt)

    completed = _complete_startup()
    first = completed.phases[0]
    changed_first = RunnerStartupPhaseReport(
        phase=first.phase,
        started_at=first.started_at,
        completed_at=first.completed_at - timedelta(milliseconds=1),
        outcome=first.outcome,
    )
    forked = RunnerStartupReport(
        runner_id=completed.runner_id,
        attempt=completed.attempt,
        observed_at=completed.observed_at + timedelta(seconds=1),
        phases=(changed_first, *completed.phases[1:]),
    )
    with pytest.raises(InvalidRunnerStartupReportError, match="immutable"):
        validate_startup_report_update(completed, forked)


def test_transition_accepts_only_a_monotonic_startup_phase_completion() -> None:
    current = _warmup_in_progress()
    status = _status(RunnerState.WARMING, startup=current)
    completed = complete_startup_phase(
        current,
        RunnerStartupPhase.WARMUP,
        at=current.observed_at + timedelta(seconds=1),
    )
    ready = transition_runner_status(
        status,
        RunnerState.READY,
        at=completed.observed_at,
        startup=completed,
    )
    assert ready.startup == completed

    truncated = RunnerStartupReport(
        runner_id=current.runner_id,
        attempt=current.attempt,
        observed_at=completed.observed_at,
        phases=current.phases[:-1],
    )
    with pytest.raises(InvalidRunnerStartupReportError, match="truncated"):
        transition_runner_status(
            status,
            RunnerState.WARMING,
            at=completed.observed_at,
            startup=truncated,
        )


def test_startup_update_rejects_completion_before_prior_heartbeat() -> None:
    in_progress = _warmup_in_progress()
    heartbeat_at = in_progress.observed_at + timedelta(seconds=10)
    heartbeat = RunnerStartupReport(
        runner_id=in_progress.runner_id,
        attempt=in_progress.attempt,
        observed_at=heartbeat_at,
        phases=in_progress.phases,
    )
    warmup = in_progress.phases[-1]
    retroactive_phase = RunnerStartupPhaseReport(
        phase=warmup.phase,
        started_at=warmup.started_at,
        completed_at=heartbeat_at - timedelta(seconds=1),
        outcome=RunnerStartupPhaseOutcome.SUCCEEDED,
    )
    retroactive = RunnerStartupReport(
        runner_id=heartbeat.runner_id,
        attempt=heartbeat.attempt,
        observed_at=heartbeat_at + timedelta(seconds=1),
        phases=(*heartbeat.phases[:-1], retroactive_phase),
    )
    with pytest.raises(InvalidRunnerStartupReportError, match="completion cannot predate"):
        validate_startup_report_update(heartbeat, retroactive)


def test_startup_update_rejects_new_phase_before_prior_observation() -> None:
    observed_at = NOW + timedelta(seconds=10)
    empty = RunnerStartupReport(runner_id="runner-a", observed_at=observed_at)
    backdated_first = RunnerStartupReport(
        runner_id="runner-a",
        observed_at=observed_at + timedelta(seconds=1),
        phases=(
            RunnerStartupPhaseReport(
                phase=RunnerStartupPhase.IMAGE_PULL,
                started_at=observed_at - timedelta(seconds=1),
            ),
        ),
    )
    with pytest.raises(InvalidRunnerStartupReportError, match="phase cannot predate"):
        validate_startup_report_update(empty, backdated_first)

    first = RunnerStartupPhaseReport(
        phase=RunnerStartupPhase.IMAGE_PULL,
        started_at=NOW,
        completed_at=NOW + timedelta(seconds=1),
        outcome=RunnerStartupPhaseOutcome.SUCCEEDED,
    )
    completed_prefix = RunnerStartupReport(
        runner_id="runner-a",
        observed_at=observed_at,
        phases=(first,),
    )
    backdated_next = RunnerStartupReport(
        runner_id="runner-a",
        observed_at=observed_at + timedelta(seconds=1),
        phases=(
            first,
            RunnerStartupPhaseReport(
                phase=RunnerStartupPhase.MODEL_FETCH,
                started_at=observed_at - timedelta(seconds=1),
            ),
        ),
    )
    with pytest.raises(InvalidRunnerStartupReportError, match="phase cannot predate"):
        validate_startup_report_update(completed_prefix, backdated_next)
