"""Drain fencing and termination authorization contract."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from kairyu.engine.backend import GenerationRequest, GenerationResult, SamplingParams
from kairyu.orchestration.replica import ReplicaPool
from kairyu.runners import (
    RUNNER_STARTUP_PHASES,
    InvalidRunnerDrainEvidenceError,
    ReplicaPoolDrainController,
    RunnerDispatchFence,
    RunnerDrainActivityObservation,
    RunnerObservationBatch,
    RunnerStartupPhase,
    RunnerStartupReport,
    RunnerState,
    RunnerStatus,
    RunnerStatusReconciler,
    authorize_runner_termination,
    complete_startup_phase,
    skip_startup_phase,
    start_startup_phase,
    transition_runner_status,
)

NOW = datetime(2026, 9, 14, 6, 0, tzinfo=UTC)


def _complete_startup() -> RunnerStartupReport:
    report = RunnerStartupReport(runner_id="pod-uid-a", observed_at=NOW)
    at = NOW
    for phase in RUNNER_STARTUP_PHASES:
        at += timedelta(milliseconds=1)
        if phase is RunnerStartupPhase.GRAPH_COMPILE:
            report = skip_startup_phase(report, phase, at=at)
        else:
            report = start_startup_phase(report, phase, at=at)
            at += timedelta(milliseconds=1)
            report = complete_startup_phase(report, phase, at=at)
    return report


def _draining(*, active_requests: int = 0) -> RunnerStatus:
    return RunnerStatus(
        runner_id="pod-uid-a",
        release_id="release-a",
        model_id="qwen",
        model_revision="revision-a",
        state=RunnerState.DRAINING,
        state_version=6,
        state_changed_at=NOW + timedelta(seconds=1),
        observed_at=NOW + timedelta(seconds=1),
        node_name="gpu-node-a",
        pod_uid="pod-uid-a",
        gpu_uuids=("GPU-a",),
        active_requests=active_requests,
        startup=_complete_startup(),
    )


def _request(request_id: str) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt="hello",
        sampling_params=SamplingParams(),
    )


class _BlockingBackend:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return GenerationResult(
            request_id=request.request_id,
            prompt="hello",
            completions=(),
            finished=True,
        )

    async def stream(self, request: GenerationRequest):
        yield await self.generate(request)

    async def shutdown(self) -> None:
        return None


class _MalformedDrainController:
    def __init__(
        self,
        delegate: ReplicaPoolDrainController,
        committed: RunnerStatus,
    ) -> None:
        self._delegate = delegate
        self._committed = committed

    def begin(
        self,
        status: RunnerStatus,
        *,
        at: datetime | None = None,
    ) -> RunnerDispatchFence:
        return self._delegate.begin(status, at=at)

    def observe(
        self,
        fence: RunnerDispatchFence,
        *,
        at: datetime | None = None,
    ) -> RunnerDrainActivityObservation:
        return self._delegate.observe(fence, at=at)

    def commit_termination(
        self,
        status: RunnerStatus,
        fence: RunnerDispatchFence,
        *,
        at: datetime,
    ) -> RunnerStatus:
        return self._committed


@pytest.mark.asyncio
async def test_dispatch_fence_blocks_new_work_and_waits_for_post_fence_zero() -> None:
    backend = _BlockingBackend()
    pool = ReplicaPool({"pod-uid-a": backend})
    controller = ReplicaPoolDrainController(
        pool,
        now=lambda: NOW + timedelta(seconds=2),
        fence_id_factory=lambda: "fence-a",
    )
    status = _draining(active_requests=1)
    preflighted = _request("request-b")
    await pool.prepare_request(preflighted)
    in_flight = asyncio.create_task(pool.generate(_request("request-a")))
    await backend.started.wait()

    fence = controller.begin(status)
    assert fence.active_requests_at_fence == 1
    assert controller.begin(status) == fence
    assert pool.eligible_ids == ()
    with pytest.raises(RuntimeError, match="remains eligible"):
        await pool.generate(preflighted)
    assert backend.calls == 1

    active = controller.observe(fence, at=NOW + timedelta(seconds=3))
    assert active.active_requests == 1
    with pytest.raises(InvalidRunnerDrainEvidenceError, match="count of zero"):
        authorize_runner_termination(
            status,
            fence,
            controller=controller,
            at=NOW + timedelta(seconds=3),
        )

    backend.release.set()
    await in_flight
    drained = controller.observe(fence, at=NOW + timedelta(seconds=4))
    assert drained.active_requests == 0
    reconciler = RunnerStatusReconciler({status.runner_id: status})
    terminating = reconciler.authorize_termination(
        fence,
        controller=controller,
        at=NOW + timedelta(seconds=4),
    )
    assert terminating.state is RunnerState.TERMINATING
    assert terminating.active_requests == 0
    assert terminating.termination_authorization is not None
    assert terminating.termination_authorization.fence_id == "fence-a"
    assert (
        reconciler.routing_eligible(
            status.runner_id,
            now=NOW + timedelta(seconds=4),
            max_observation_age=timedelta(seconds=5),
        )
        is False
    )

    terminated = reconciler.reconcile(
        RunnerObservationBatch(
            source_id="watcher-a",
            source_epoch=1,
            source_started_at=NOW + timedelta(seconds=5),
            observed_at=NOW + timedelta(seconds=5),
        )
    )[status.runner_id]
    assert terminated.state is RunnerState.TERMINATED
    assert terminated.termination_authorization == (terminating.termination_authorization)


def test_authorization_revalidates_owned_fence_and_drain_state() -> None:
    pool = ReplicaPool({"pod-uid-a": _BlockingBackend()})
    controller = ReplicaPoolDrainController(
        pool,
        fence_id_factory=lambda: "fence-a",
    )
    status = _draining()
    fence = controller.begin(status, at=NOW + timedelta(seconds=2))
    with pytest.raises(InvalidRunnerDrainEvidenceError, match="post-fence"):
        authorize_runner_termination(
            status,
            fence,
            controller=controller,
            at=NOW + timedelta(seconds=1),
        )

    mismatched_fence = fence.model_copy(update={"fence_id": "other"})
    with pytest.raises(InvalidRunnerDrainEvidenceError, match="not owned"):
        authorize_runner_termination(
            status,
            mismatched_fence,
            controller=controller,
            at=NOW + timedelta(seconds=4),
        )

    newer_status = status.model_copy(update={"state_version": 7})
    with pytest.raises(InvalidRunnerDrainEvidenceError, match="stale drain"):
        authorize_runner_termination(
            newer_status,
            fence,
            controller=controller,
            at=NOW + timedelta(seconds=4),
        )


@pytest.mark.asyncio
async def test_zero_observation_cannot_authorize_a_readded_replica_generation() -> None:
    pool = ReplicaPool({"pod-uid-a": _BlockingBackend()})
    controller = ReplicaPoolDrainController(
        pool,
        fence_id_factory=lambda: "fence-a",
    )
    status = _draining()
    fence = controller.begin(status, at=NOW + timedelta(seconds=2))
    zero = controller.observe(fence, at=NOW + timedelta(seconds=3))
    assert zero.active_requests == 0
    await pool.remove_replica("pod-uid-a")
    replacement = _BlockingBackend()
    replacement.release.set()
    pool.add_replica("pod-uid-a", replacement)
    await pool.generate(_request("request-after-readd"))
    assert replacement.calls == 1
    with pytest.raises(InvalidRunnerDrainEvidenceError, match="generation changed"):
        authorize_runner_termination(
            status,
            fence,
            controller=controller,
            at=NOW + timedelta(seconds=4),
        )


def test_authorization_is_idempotent_for_the_same_fence() -> None:
    pool = ReplicaPool({"pod-uid-a": _BlockingBackend()})
    controller = ReplicaPoolDrainController(
        pool,
        fence_id_factory=lambda: "fence-a",
    )
    status = _draining()
    fence = controller.begin(status, at=NOW + timedelta(seconds=2))
    heartbeat = transition_runner_status(
        status,
        RunnerState.DRAINING,
        at=NOW + timedelta(seconds=3),
        active_requests=0,
    )
    assert controller.begin(heartbeat, at=NOW + timedelta(seconds=2)) == fence
    terminating = authorize_runner_termination(
        heartbeat,
        fence,
        controller=controller,
        at=NOW + timedelta(seconds=3),
    )
    terminating_heartbeat = transition_runner_status(
        terminating,
        RunnerState.TERMINATING,
        at=NOW + timedelta(seconds=4),
        active_requests=0,
    )
    assert (
        authorize_runner_termination(
            terminating_heartbeat,
            fence,
            controller=controller,
            at=NOW + timedelta(seconds=3),
        )
        == terminating_heartbeat
    )
    terminated = transition_runner_status(
        terminating_heartbeat,
        RunnerState.TERMINATED,
        at=NOW + timedelta(seconds=5),
        active_requests=0,
    )
    assert (
        authorize_runner_termination(
            terminated,
            fence,
            controller=controller,
            at=NOW + timedelta(seconds=3),
        )
        == terminated
    )


def test_authorizer_rejects_malformed_controller_commit() -> None:
    status = _draining()
    pool = ReplicaPool({"pod-uid-a": _BlockingBackend()})
    controller = ReplicaPoolDrainController(
        pool,
        fence_id_factory=lambda: "fence-a",
    )
    fence = controller.begin(status, at=NOW + timedelta(seconds=2))

    with pytest.raises(InvalidRunnerDrainEvidenceError, match="mismatched"):
        authorize_runner_termination(
            status,
            fence,
            controller=_MalformedDrainController(controller, status),
            at=NOW + timedelta(seconds=3),
        )

    valid_terminating = authorize_runner_termination(
        status,
        fence,
        controller=controller,
        at=NOW + timedelta(seconds=3),
    )
    valid_authorization = valid_terminating.termination_authorization
    assert valid_authorization is not None
    altered_authorization = type(valid_authorization).model_validate(
        {
            **valid_authorization.model_dump(),
            "dispatch_stopped_at": NOW,
            "routing_excluded_at": NOW,
            "activity_observed_at": NOW,
        }
    )
    altered_values = valid_terminating.model_dump()
    altered_values["termination_authorization"] = altered_authorization
    altered_terminating = RunnerStatus.model_validate(altered_values)
    with pytest.raises(InvalidRunnerDrainEvidenceError, match="mismatched"):
        authorize_runner_termination(
            status,
            fence,
            controller=_MalformedDrainController(controller, altered_terminating),
            at=NOW + timedelta(seconds=3),
        )

    runtime_status_values = status.model_dump()
    runtime_status_values.update(
        observed_at=NOW + timedelta(seconds=2, milliseconds=750),
        runtime_observed_at=NOW + timedelta(seconds=2, milliseconds=750),
    )
    runtime_status = RunnerStatus.model_validate(runtime_status_values)
    stale_runtime_authorization = type(valid_authorization).model_validate(
        {
            **valid_authorization.model_dump(),
            "activity_observed_at": NOW + timedelta(seconds=2, milliseconds=500),
        }
    )
    stale_runtime_values = valid_terminating.model_dump()
    stale_runtime_values["termination_authorization"] = stale_runtime_authorization
    stale_runtime_terminating = RunnerStatus.model_validate(stale_runtime_values)
    with pytest.raises(InvalidRunnerDrainEvidenceError, match="mismatched"):
        authorize_runner_termination(
            runtime_status,
            fence,
            controller=_MalformedDrainController(
                controller,
                stale_runtime_terminating,
            ),
            at=NOW + timedelta(seconds=3),
        )

    other_pool = ReplicaPool({"pod-uid-a": _BlockingBackend()})
    other_controller = ReplicaPoolDrainController(
        other_pool,
        fence_id_factory=lambda: "fence-b",
    )
    other_fence = other_controller.begin(
        status,
        at=NOW + timedelta(seconds=2),
    )
    other_terminating = authorize_runner_termination(
        status,
        other_fence,
        controller=other_controller,
        at=NOW + timedelta(seconds=3),
    )
    with pytest.raises(InvalidRunnerDrainEvidenceError, match="mismatched"):
        authorize_runner_termination(
            status,
            fence,
            controller=_MalformedDrainController(controller, other_terminating),
            at=NOW + timedelta(seconds=3),
        )


def test_activity_model_rejects_boolean_counter() -> None:
    with pytest.raises(ValueError, match="active_requests must be an integer"):
        RunnerDrainActivityObservation(
            runner_id="pod-uid-a",
            pod_uid="pod-uid-a",
            fence_id="fence-a",
            fence_sequence=1,
            drain_state_version=6,
            replica_generation="generation-a",
            observed_at=NOW,
            active_requests=True,
        )
