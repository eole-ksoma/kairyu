"""Read-only Runner observation and reconciliation contract."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from kairyu.runners import (
    RUNNER_STARTUP_PHASES,
    InvalidRunnerObservationError,
    KubernetesPodPhase,
    RunnerBackoffPolicy,
    RunnerFailure,
    RunnerFailureCapacityError,
    RunnerFailureDomainKind,
    RunnerFailureGuard,
    RunnerObservation,
    RunnerObservationBatch,
    RunnerPodObservation,
    RunnerRuntimeObservation,
    RunnerStartupPhase,
    RunnerStartupPhaseOutcome,
    RunnerStartupReport,
    RunnerState,
    RunnerStatus,
    RunnerStatusReconciler,
    RunnerTerminationAuthorization,
    complete_startup_phase,
    reconcile_runner_status,
    revision_failure_domain,
    runner_is_routing_eligible,
    skip_startup_phase,
    start_startup_phase,
    transition_runner_status,
)

NOW = datetime(2026, 9, 14, 3, 0, tzinfo=UTC)


def _complete_startup() -> RunnerStartupReport:
    report = RunnerStartupReport(runner_id="pod-uid-a", observed_at=NOW)
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


def _pod(
    *,
    phase: KubernetesPodPhase = KubernetesPodPhase.RUNNING,
    ready: bool = True,
    deleting: bool = False,
    node_name: str | None = "gpu-node-a",
    gpu_uuids: tuple[str, ...] = ("GPU-a",),
    waiting_reason: str | None = None,
    terminated_reason: str | None = None,
    exit_code: int | None = None,
) -> RunnerPodObservation:
    return RunnerPodObservation(
        uid="pod-uid-a",
        phase=phase,
        ready=ready,
        deleting=deleting,
        node_name=node_name,
        gpu_uuids=gpu_uuids,
        waiting_reason=waiting_reason,
        terminated_reason=terminated_reason,
        exit_code=exit_code,
    )


def _observation(
    *,
    at: datetime = NOW + timedelta(seconds=20),
    pod: RunnerPodObservation | None = None,
    endpoint_ready: bool = False,
    runtime: RunnerRuntimeObservation | None = None,
    **identity,
) -> RunnerObservation:
    values = {
        "runner_id": "pod-uid-a",
        "release_id": "release-sha256:a",
        "model_id": "qwen",
        "model_revision": "model-sha256:b",
        "observed_at": at,
        "pod": pod,
        "endpoint_ready": endpoint_ready,
        "runtime": runtime,
    }
    values.update(identity)
    return RunnerObservation(**values)


def _runtime(
    *,
    at: datetime = NOW + timedelta(seconds=20),
    **values,
) -> RunnerRuntimeObservation:
    return RunnerRuntimeObservation(
        runner_id="pod-uid-a",
        observed_at=at,
        **values,
    )


def _eligible(
    status,
    *,
    now: datetime | None = None,
    serving_gates_ready: bool = True,
) -> bool:
    return runner_is_routing_eligible(
        status,
        serving_gates_ready=serving_gates_ready,
        serving_gates_observed_at=status.observed_at,
        now=status.observed_at if now is None else now,
        max_observation_age=timedelta(seconds=5),
    )


def _batch(
    epoch: int,
    *,
    at: datetime,
    runners: tuple[RunnerObservation, ...] = (),
    missing_runner_runtime: tuple[RunnerRuntimeObservation, ...] = (),
    source_started_at: datetime | None = None,
    source_id: str = "watcher-a",
) -> RunnerObservationBatch:
    return RunnerObservationBatch(
        source_id=source_id,
        source_epoch=epoch,
        source_started_at=at if source_started_at is None else source_started_at,
        observed_at=at,
        runners=runners,
        missing_runner_runtime=missing_runner_runtime,
    )


def _ready_observation(
    *,
    active_requests: int = 0,
    at: datetime = NOW + timedelta(seconds=20),
) -> RunnerObservation:
    return _observation(
        at=at,
        pod=_pod(),
        endpoint_ready=True,
        runtime=_runtime(
            at=at,
            ready=True,
            active_requests=active_requests,
            startup=_complete_startup(),
        ),
    )


def _termination_authorization(
    status,
    *,
    at: datetime,
) -> RunnerTerminationAuthorization:
    assert status.pod_uid is not None
    return RunnerTerminationAuthorization(
        runner_id=status.runner_id,
        pod_uid=status.pod_uid,
        fence_id="fence-a",
        fence_sequence=1,
        drain_state_version=status.state_version,
        replica_generation="generation-a",
        dispatch_stopped_at=at,
        routing_excluded_at=at,
        activity_observed_at=at,
        authorized_at=at,
    )


def test_running_pod_is_not_ready_without_startup_evidence() -> None:
    status = reconcile_runner_status(
        None,
        _observation(
            pod=_pod(),
            endpoint_ready=True,
            runtime=_runtime(ready=True, active_requests=0),
        ),
    )
    assert status.state is RunnerState.MODEL_LOADING
    assert status.state_version == 3
    assert _eligible(status) is False


def test_every_serving_gate_is_required_before_ready() -> None:
    startup = _complete_startup()
    for observation in (
        _observation(
            pod=_pod(gpu_uuids=()),
            endpoint_ready=True,
            runtime=_runtime(
                ready=True,
                active_requests=0,
                startup=startup,
            ),
        ),
        _observation(
            pod=_pod(ready=False),
            endpoint_ready=True,
            runtime=_runtime(
                ready=True,
                active_requests=0,
                startup=startup,
            ),
        ),
        _observation(
            pod=_pod(),
            endpoint_ready=False,
            runtime=_runtime(
                ready=True,
                active_requests=0,
                startup=startup,
            ),
        ),
        _observation(
            pod=_pod(),
            endpoint_ready=True,
            runtime=_runtime(
                ready=False,
                active_requests=0,
                startup=startup,
            ),
        ),
    ):
        status = reconcile_runner_status(None, observation)
        assert status.state is RunnerState.WARMING
        assert _eligible(status) is False


def test_complete_coherent_observation_reconstructs_ready_and_busy() -> None:
    ready = reconcile_runner_status(None, _ready_observation())
    assert ready.state is RunnerState.READY
    assert ready.state_version == 5
    assert ready.node_name == "gpu-node-a"
    assert ready.pod_uid == "pod-uid-a"
    assert ready.gpu_uuids == ("GPU-a",)
    assert _eligible(ready) is True

    busy = reconcile_runner_status(
        ready,
        _ready_observation(
            active_requests=2,
            at=NOW + timedelta(seconds=21),
        ),
    )
    assert busy.state is RunnerState.BUSY
    assert busy.state_version == 6
    assert busy.active_requests == 2
    assert _eligible(busy) is True

    idle = reconcile_runner_status(
        busy,
        _ready_observation(at=NOW + timedelta(seconds=22)),
    )
    assert idle.state is RunnerState.READY
    assert idle.active_requests == 0


@pytest.mark.parametrize(
    ("pod", "runtime", "code", "domain"),
    [
        (
            _pod(
                phase=KubernetesPodPhase.PENDING,
                ready=False,
                waiting_reason="ImagePullBackOff",
            ),
            None,
            "image_pull_failed",
            RunnerFailureDomainKind.REVISION,
        ),
        (
            _pod(
                phase=KubernetesPodPhase.FAILED,
                ready=False,
                terminated_reason="OOMKilled",
            ),
            None,
            "runner_oom",
            RunnerFailureDomainKind.REVISION,
        ),
        (
            _pod(
                phase=KubernetesPodPhase.FAILED,
                ready=False,
                terminated_reason="NvidiaGPUXid",
            ),
            None,
            "gpu_xid",
            RunnerFailureDomainKind.GPU,
        ),
        (
            _pod(
                phase=KubernetesPodPhase.RUNNING,
                ready=False,
                waiting_reason="CrashLoopBackOff",
            ),
            None,
            "container_crash_loop",
            RunnerFailureDomainKind.REVISION,
        ),
        (
            _pod(
                phase=KubernetesPodPhase.RUNNING,
                ready=False,
                terminated_reason="Error",
                exit_code=17,
            ),
            None,
            "container_exit_nonzero",
            RunnerFailureDomainKind.REVISION,
        ),
        (
            _pod(),
            _runtime(
                ready=False,
                fatal=True,
                detail="engine worker stopped",
                active_requests=0,
                startup=_complete_startup(),
            ),
            "readiness_fatal",
            RunnerFailureDomainKind.REVISION,
        ),
    ],
)
def test_failures_have_distinct_bounded_reason_codes(
    pod: RunnerPodObservation,
    runtime: RunnerRuntimeObservation | None,
    code: str,
    domain: RunnerFailureDomainKind,
) -> None:
    status = reconcile_runner_status(
        None,
        _observation(pod=pod, endpoint_ready=True, runtime=runtime),
    )
    assert status.state is RunnerState.UNHEALTHY
    assert status.failure is not None
    assert status.failure.code == code
    assert status.failure.domain is domain


def test_startup_failure_is_preserved_as_authoritative_reason() -> None:
    report = RunnerStartupReport(runner_id="pod-uid-a", observed_at=NOW)
    report = start_startup_phase(report, RunnerStartupPhase.IMAGE_PULL, at=NOW)
    report = complete_startup_phase(
        report,
        RunnerStartupPhase.IMAGE_PULL,
        at=NOW + timedelta(seconds=1),
        outcome=RunnerStartupPhaseOutcome.FAILED,
        failure=RunnerFailure(
            code="registry_denied",
            message="image registry denied the pull",
        ),
    )
    status = reconcile_runner_status(
        None,
        _observation(
            pod=_pod(ready=False),
            runtime=_runtime(
                ready=False,
                active_requests=0,
                startup=report,
            ),
        ),
    )
    assert status.failure is not None
    assert status.failure.code == "registry_denied"


def test_stateful_reconciler_records_failure_guard_once() -> None:
    guard = RunnerFailureGuard()
    reconciler = RunnerStatusReconciler(failure_guard=guard)
    observation = _observation(
        pod=_pod(
            phase=KubernetesPodPhase.RUNNING,
            ready=False,
            waiting_reason="CrashLoopBackOff",
        ),
        endpoint_ready=False,
        runtime=None,
    )
    observed_at = observation.observed_at
    status = reconciler.reconcile(_batch(1, at=observed_at, runners=(observation,)))["pod-uid-a"]
    assert status.state is RunnerState.UNHEALTHY
    domain = revision_failure_domain(
        release_id=status.release_id,
        model_id=status.model_id,
        model_revision=status.model_revision,
    )
    assert guard.decision(domain, at=observed_at).failure_count == 1

    replay_at = observed_at + timedelta(seconds=1)
    replay = observation.model_copy(update={"observed_at": replay_at})
    reconciler.reconcile(
        _batch(
            2,
            at=replay_at,
            source_started_at=replay_at,
            runners=(replay,),
        )
    )
    assert guard.decision(domain, at=replay_at).failure_count == 1
    assert len(guard.snapshot(at=replay_at).observations) == 1


def test_failure_guard_capacity_failure_does_not_commit_reconciler_state() -> None:
    guard = RunnerFailureGuard(RunnerBackoffPolicy(max_observations=1))
    reconciler = RunnerStatusReconciler(failure_guard=guard)
    first = _observation(
        pod=_pod(
            phase=KubernetesPodPhase.RUNNING,
            ready=False,
            waiting_reason="CrashLoopBackOff",
        ),
        endpoint_ready=False,
        runtime=None,
    )
    assert first.pod is not None
    second = first.model_copy(
        update={
            "runner_id": "pod-uid-b",
            "pod": first.pod.model_copy(update={"uid": "pod-uid-b"}),
        }
    )
    with pytest.raises(RunnerFailureCapacityError, match="observations"):
        reconciler.reconcile(
            _batch(
                1,
                at=first.observed_at,
                runners=(first, second),
            )
        )
    assert reconciler.statuses == {}
    assert guard.snapshot(at=first.observed_at).observations == ()


def test_ready_gate_loss_fails_closed_and_cannot_self_recover() -> None:
    ready = reconcile_runner_status(None, _ready_observation())
    lost = reconcile_runner_status(
        ready,
        _observation(
            at=NOW + timedelta(seconds=21),
            pod=_pod(),
            endpoint_ready=False,
            runtime=_runtime(
                at=NOW + timedelta(seconds=21),
                ready=True,
                active_requests=0,
                startup=_complete_startup(),
            ),
        ),
    )
    assert lost.state is RunnerState.UNHEALTHY
    assert lost.failure is not None
    assert lost.failure.code == "endpoint_not_ready"
    observed_again = reconcile_runner_status(
        lost,
        _ready_observation(at=NOW + timedelta(seconds=22)),
    )
    assert observed_again.state is RunnerState.UNHEALTHY
    assert _eligible(observed_again) is False


def test_missing_runtime_observation_preserves_startup_and_fails_closed() -> None:
    ready = reconcile_runner_status(None, _ready_observation())
    lost = reconcile_runner_status(
        ready,
        _observation(
            at=NOW + timedelta(seconds=21),
            pod=_pod(),
            endpoint_ready=True,
        ),
    )
    assert lost.state is RunnerState.UNHEALTHY
    assert lost.startup == ready.startup
    assert lost.failure is not None
    assert lost.failure.code == "runtime_observation_missing"


def test_stateful_reconciler_debounces_cross_resource_gate_skew() -> None:
    reconciler = RunnerStatusReconciler(serving_gate_failure_grace=timedelta(seconds=5))
    ready_observation = _ready_observation()
    ready = reconciler.reconcile(
        _batch(1, at=ready_observation.observed_at, runners=(ready_observation,))
    )["pod-uid-a"]
    assert ready.state is RunnerState.READY
    assert (
        reconciler.routing_eligible(
            "pod-uid-a",
            now=ready.observed_at,
            max_observation_age=timedelta(seconds=5),
        )
        is True
    )

    skewed = _observation(
        at=NOW + timedelta(seconds=21),
        pod=_pod(),
        endpoint_ready=False,
        runtime=_runtime(
            at=NOW + timedelta(seconds=21),
            ready=True,
            active_requests=0,
            startup=_complete_startup(),
        ),
    )
    held = reconciler.reconcile(_batch(2, at=skewed.observed_at, runners=(skewed,)))["pod-uid-a"]
    assert held.state is RunnerState.READY
    assert (
        reconciler.routing_eligible(
            "pod-uid-a",
            now=held.observed_at,
            max_observation_age=timedelta(seconds=5),
        )
        is False
    )

    recovered_observation = _ready_observation(at=NOW + timedelta(seconds=22))
    recovered = reconciler.reconcile(
        _batch(3, at=recovered_observation.observed_at, runners=(recovered_observation,))
    )["pod-uid-a"]
    assert recovered.state is RunnerState.READY
    assert (
        reconciler.routing_eligible(
            "pod-uid-a",
            now=recovered.observed_at,
            max_observation_age=timedelta(seconds=5),
        )
        is True
    )

    first_loss_at = NOW + timedelta(seconds=23)
    for epoch, offset in ((4, 0), (5, 0.02)):
        at = first_loss_at + timedelta(seconds=offset)
        rapid_loss = _observation(
            at=at,
            pod=_pod(),
            endpoint_ready=False,
            runtime=_runtime(
                at=at,
                ready=True,
                active_requests=0,
                startup=_complete_startup(),
            ),
        )
        held = reconciler.reconcile(_batch(epoch, at=at, runners=(rapid_loss,)))["pod-uid-a"]
        assert held.state is RunnerState.READY

    confirmed_at = first_loss_at + timedelta(seconds=5)
    confirmed_loss = _observation(
        at=confirmed_at,
        pod=_pod(),
        endpoint_ready=False,
        runtime=_runtime(
            at=confirmed_at,
            ready=True,
            active_requests=0,
            startup=_complete_startup(),
        ),
    )
    unhealthy = reconciler.reconcile(_batch(6, at=confirmed_at, runners=(confirmed_loss,)))[
        "pod-uid-a"
    ]
    assert unhealthy.state is RunnerState.UNHEALTHY


def test_routing_lease_expires_when_observation_loop_stops() -> None:
    reconciler = RunnerStatusReconciler()
    observation = _ready_observation()
    reconciler.reconcile(_batch(1, at=observation.observed_at, runners=(observation,)))
    assert (
        reconciler.routing_eligible(
            "pod-uid-a",
            now=observation.observed_at + timedelta(seconds=6),
            max_observation_age=timedelta(seconds=5),
        )
        is False
    )


def test_routing_fails_closed_when_caller_clock_is_behind_observations() -> None:
    status = reconcile_runner_status(None, _ready_observation())
    assert (
        runner_is_routing_eligible(
            status,
            serving_gates_ready=True,
            serving_gates_observed_at=status.observed_at,
            now=status.observed_at - timedelta(milliseconds=1),
            max_observation_age=timedelta(seconds=5),
        )
        is False
    )


def test_runtime_freshness_is_not_extended_by_the_routing_lease() -> None:
    reconciler = RunnerStatusReconciler(max_runtime_observation_age=timedelta(seconds=5))
    runtime_at = NOW + timedelta(seconds=20)
    source_started_at = runtime_at + timedelta(seconds=4, milliseconds=900)
    completed_at = source_started_at + timedelta(milliseconds=100)
    observation = _observation(
        at=completed_at,
        pod=_pod(),
        endpoint_ready=True,
        runtime=_runtime(
            at=runtime_at,
            ready=True,
            active_requests=0,
            startup=_complete_startup(),
        ),
    )
    status = reconciler.reconcile(
        _batch(
            1,
            at=completed_at,
            source_started_at=source_started_at,
            runners=(observation,),
        )
    )["pod-uid-a"]
    assert status.state is RunnerState.READY
    assert (
        reconciler.routing_eligible(
            "pod-uid-a",
            now=runtime_at + timedelta(seconds=5, milliseconds=1),
            max_observation_age=timedelta(seconds=5),
        )
        is False
    )


def test_equal_runtime_timestamp_requires_an_identical_payload() -> None:
    reconciler = RunnerStatusReconciler()
    first_observation = _ready_observation(active_requests=2)
    initial = reconciler.reconcile(
        _batch(
            1,
            at=first_observation.observed_at,
            runners=(first_observation,),
        )
    )
    assert first_observation.runtime is not None
    assert (
        reconcile_runner_status(
            initial["pod-uid-a"],
            first_observation,
        )
        == initial["pod-uid-a"]
    )
    replay = _observation(
        at=NOW + timedelta(seconds=21),
        pod=_pod(),
        endpoint_ready=True,
        runtime=first_observation.runtime,
    )
    repeated = reconciler.reconcile(_batch(2, at=replay.observed_at, runners=(replay,)))
    assert repeated["pod-uid-a"].state is RunnerState.BUSY
    assert repeated["pod-uid-a"].active_requests == 2

    changed = replay.model_copy(
        update={
            "observed_at": NOW + timedelta(seconds=22),
            "runtime": first_observation.runtime.model_copy(update={"active_requests": 0}),
        }
    )
    restarted = RunnerStatusReconciler(repeated)
    with pytest.raises(InvalidRunnerObservationError, match="payload cannot change"):
        restarted.reconcile(_batch(3, at=changed.observed_at, runners=(changed,)))
    assert restarted.statuses == repeated
    assert initial["pod-uid-a"].active_requests == 2


def test_slow_kubernetes_poll_does_not_refresh_old_serving_gates() -> None:
    reconciler = RunnerStatusReconciler()
    source_started_at = NOW + timedelta(seconds=20)
    completed_at = source_started_at + timedelta(seconds=9)
    observation = _ready_observation(at=completed_at)
    status = reconciler.reconcile(
        _batch(
            1,
            at=completed_at,
            source_started_at=source_started_at,
            runners=(observation,),
        )
    )["pod-uid-a"]
    assert status.state is RunnerState.READY
    assert (
        reconciler.routing_eligible(
            "pod-uid-a",
            now=completed_at,
            max_observation_age=timedelta(seconds=5),
        )
        is False
    )


def test_stale_runtime_zero_cannot_authorize_termination() -> None:
    busy = reconcile_runner_status(None, _ready_observation(active_requests=2))
    stale_zero = _observation(
        at=NOW + timedelta(seconds=30),
        pod=_pod(deleting=True),
        runtime=_runtime(
            at=NOW + timedelta(seconds=21),
            ready=False,
            active_requests=0,
            startup=_complete_startup(),
        ),
    )
    draining = reconcile_runner_status(busy, stale_zero)
    assert draining.state is RunnerState.DRAINING
    assert draining.active_requests == 2


def test_runtime_clock_skew_is_bounded_without_falsifying_gate_age() -> None:
    observed_at = NOW + timedelta(seconds=20)
    within_tolerance = _observation(
        at=observed_at,
        pod=_pod(),
        endpoint_ready=True,
        runtime=_runtime(
            at=observed_at + timedelta(seconds=1),
            ready=True,
            active_requests=0,
            startup=_complete_startup().model_copy(
                update={"observed_at": observed_at + timedelta(seconds=1)}
            ),
        ),
    )
    ready = reconcile_runner_status(None, within_tolerance)
    assert ready.state is RunnerState.READY
    assert ready.runtime_observed_at == observed_at + timedelta(seconds=1)

    too_far_ahead = within_tolerance.model_copy(
        update={
            "runtime": _runtime(
                at=observed_at + timedelta(seconds=3),
                ready=True,
                active_requests=0,
                startup=_complete_startup(),
            )
        }
    )
    not_ready = reconcile_runner_status(None, too_far_ahead)
    assert not_ready.state is RunnerState.MODEL_LOADING


def test_deletion_drains_active_work_before_termination() -> None:
    busy = reconcile_runner_status(None, _ready_observation(active_requests=2))
    draining = reconcile_runner_status(
        busy,
        _observation(
            at=NOW + timedelta(seconds=21),
            pod=_pod(deleting=True),
            runtime=_runtime(
                at=NOW + timedelta(seconds=21),
                ready=False,
                active_requests=2,
                startup=_complete_startup(),
            ),
        ),
    )
    assert draining.state is RunnerState.DRAINING
    assert draining.active_requests == 2
    assert _eligible(draining) is False

    drained_zero = reconcile_runner_status(
        draining,
        _observation(
            at=NOW + timedelta(seconds=22),
            pod=_pod(deleting=True),
            runtime=_runtime(
                at=NOW + timedelta(seconds=22),
                ready=False,
                active_requests=0,
                startup=_complete_startup(),
            ),
        ),
    )
    assert drained_zero.state is RunnerState.DRAINING
    assert drained_zero.active_requests == 0


@pytest.mark.parametrize(
    "pod",
    [
        _pod(phase=KubernetesPodPhase.SUCCEEDED, ready=False),
        _pod(
            phase=KubernetesPodPhase.FAILED,
            ready=False,
            terminated_reason="OOMKilled",
            exit_code=137,
        ),
    ],
)
def test_terminating_state_ignores_late_container_failure_evidence(
    pod: RunnerPodObservation,
) -> None:
    ready = reconcile_runner_status(None, _ready_observation())
    draining = transition_runner_status(
        ready,
        RunnerState.DRAINING,
        at=NOW + timedelta(seconds=21),
        active_requests=0,
    )
    authorization = _termination_authorization(
        draining,
        at=NOW + timedelta(seconds=22),
    )
    values = draining.model_dump()
    values.update(
        state=RunnerState.TERMINATING,
        state_version=draining.state_version + 1,
        state_changed_at=authorization.authorized_at,
        observed_at=authorization.authorized_at,
        active_requests=0,
        termination_authorization=authorization,
    )
    terminating = RunnerStatus.model_validate(values)
    observed = reconcile_runner_status(
        terminating,
        _observation(
            at=NOW + timedelta(seconds=23),
            pod=pod,
            runtime=_runtime(
                at=NOW + timedelta(seconds=23),
                ready=False,
                fatal=True,
                active_requests=0,
                startup=_complete_startup(),
            ),
        ),
    )
    assert observed.state is RunnerState.TERMINATING


def test_unhealthy_runner_can_progress_to_draining() -> None:
    unhealthy = reconcile_runner_status(
        None,
        _observation(
            pod=_pod(
                phase=KubernetesPodPhase.PENDING,
                ready=False,
                waiting_reason="ImagePullBackOff",
            )
        ),
    )
    draining = reconcile_runner_status(
        unhealthy,
        _observation(
            at=NOW + timedelta(seconds=21),
            pod=_pod(deleting=True),
            runtime=_runtime(
                at=NOW + timedelta(seconds=21),
                ready=False,
                active_requests=0,
            ),
        ),
    )
    assert draining.state is RunnerState.DRAINING


def test_incomplete_startup_cannot_coexist_with_observed_active_work() -> None:
    with pytest.raises(InvalidRunnerObservationError, match="active requests"):
        reconcile_runner_status(
            None,
            _observation(
                pod=_pod(),
                runtime=_runtime(ready=False, active_requests=1),
            ),
        )


def test_identity_time_and_startup_evidence_are_monotonic() -> None:
    ready = reconcile_runner_status(None, _ready_observation())
    with pytest.raises(InvalidRunnerObservationError, match="model_revision"):
        reconcile_runner_status(
            ready,
            _ready_observation(
                at=NOW + timedelta(seconds=21),
            ).model_copy(update={"model_revision": "other"}),
        )
    with pytest.raises(InvalidRunnerObservationError, match="backwards"):
        reconcile_runner_status(
            ready,
            _ready_observation(at=NOW + timedelta(seconds=19)),
        )
    with pytest.raises(InvalidRunnerObservationError, match="cannot disappear"):
        reconcile_runner_status(
            ready,
            _observation(
                at=NOW + timedelta(seconds=21),
                pod=_pod(),
                runtime=_runtime(
                    at=NOW + timedelta(seconds=21),
                    ready=False,
                    active_requests=0,
                ),
            ),
        )
    for pod, message in (
        (_pod(node_name="gpu-node-b"), "node_name"),
        (_pod(gpu_uuids=("GPU-b",)), "gpu_uuids"),
        (_pod(gpu_uuids=()), "gpu_uuids"),
    ):
        with pytest.raises(InvalidRunnerObservationError, match=message):
            reconcile_runner_status(
                ready,
                _observation(
                    at=NOW + timedelta(seconds=21),
                    pod=pod,
                    endpoint_ready=True,
                    runtime=_runtime(
                        at=NOW + timedelta(seconds=21),
                        ready=True,
                        active_requests=0,
                        startup=_complete_startup(),
                    ),
                ),
            )


def test_full_epoch_reconciler_tracks_disappearance_without_kubernetes_writes() -> None:
    reconciler = RunnerStatusReconciler()
    first = reconciler.reconcile(
        RunnerObservationBatch(
            source_id="test",
            source_epoch=1,
            source_started_at=NOW + timedelta(seconds=20),
            observed_at=NOW + timedelta(seconds=20),
            runners=(_ready_observation(),),
        )
    )
    assert first["pod-uid-a"].state is RunnerState.READY

    second = reconciler.reconcile(
        RunnerObservationBatch(
            source_id="test",
            source_epoch=2,
            source_started_at=NOW + timedelta(seconds=21),
            observed_at=NOW + timedelta(seconds=21),
        )
    )
    assert second["pod-uid-a"].state is RunnerState.DRAINING
    assert second["pod-uid-a"].startup == first["pod-uid-a"].startup

    third = reconciler.reconcile(
        RunnerObservationBatch(
            source_id="test",
            source_epoch=3,
            source_started_at=NOW + timedelta(seconds=22),
            observed_at=NOW + timedelta(seconds=22),
        )
    )
    assert third["pod-uid-a"].state is RunnerState.DRAINING


def test_full_epoch_reconcile_is_atomic_when_one_observation_is_invalid() -> None:
    first_observation = _ready_observation()
    reconciler = RunnerStatusReconciler()
    initial = reconciler.reconcile(
        RunnerObservationBatch(
            source_id="test",
            source_epoch=1,
            source_started_at=first_observation.observed_at,
            observed_at=first_observation.observed_at,
            runners=(first_observation,),
        )
    )
    invalid = _ready_observation(at=NOW + timedelta(seconds=19))
    with pytest.raises(InvalidRunnerObservationError, match="backwards"):
        reconciler.reconcile(
            RunnerObservationBatch(
                source_id="test",
                source_epoch=2,
                source_started_at=invalid.observed_at,
                observed_at=invalid.observed_at,
                runners=(invalid,),
            )
        )
    assert reconciler.statuses == initial


def test_full_epoch_rejects_stale_source_time_and_duplicate_epoch() -> None:
    reconciler = RunnerStatusReconciler()
    reconciler.reconcile(
        _batch(
            1,
            at=NOW + timedelta(seconds=20),
        )
    )
    with pytest.raises(InvalidRunnerObservationError, match="source times"):
        reconciler.reconcile(
            _batch(
                1,
                source_id="watcher-b",
                source_started_at=NOW + timedelta(seconds=10),
                at=NOW + timedelta(seconds=21),
            )
        )
    with pytest.raises(InvalidRunnerObservationError, match="source epochs"):
        reconciler.reconcile(_batch(1, at=NOW + timedelta(seconds=22)))


def test_missing_pod_runtime_zero_is_retained_for_later_drain_handshake() -> None:
    reconciler = RunnerStatusReconciler()
    busy_observation = _ready_observation(active_requests=2)
    reconciler.reconcile(_batch(1, at=busy_observation.observed_at, runners=(busy_observation,)))
    zero = _runtime(
        at=NOW + timedelta(seconds=21),
        ready=False,
        active_requests=0,
        startup=_complete_startup(),
    )
    draining = reconciler.reconcile(
        _batch(
            2,
            at=NOW + timedelta(seconds=21),
            missing_runner_runtime=(zero,),
        )
    )["pod-uid-a"]
    assert draining.state is RunnerState.DRAINING
    assert draining.active_requests == 0
