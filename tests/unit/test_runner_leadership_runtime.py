"""D3.19 lifecycle coverage for durable Runner leader election."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

import pytest

from kairyu.runners import (
    InMemoryRunnerLeaderLeaseStore,
    RunnerLeaderElectionRuntime,
    RunnerLeaderElectionRuntimeConfig,
    RunnerLeaderElector,
    RunnerLeaderLease,
    RunnerLeaderLeaseStore,
    RunnerNotLeaderError,
    RunnerWriterAuthority,
)

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)


def _config(**updates: object) -> RunnerLeaderElectionRuntimeConfig:
    values: dict[str, object] = {
        "election_id": "runner-control-plane",
        "holder_id": "controller-a",
        "lease_seconds": 1.0,
        "campaign_interval_seconds": 0.01,
        "renew_interval_seconds": 0.01,
        "readiness_max_staleness_seconds": 0.1,
        "readiness_failure_threshold": 2,
        "shutdown_timeout_seconds": 1.0,
    }
    values.update(updates)
    return RunnerLeaderElectionRuntimeConfig.model_validate(values)


def _wait_until(predicate, *, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true before timeout")


class _FlakyLeaseStore:
    def __init__(self, failures: int) -> None:
        self.delegate = InMemoryRunnerLeaderLeaseStore(clock=lambda: NOW)
        self.failures = failures
        self.acquire_calls = 0

    def acquire(
        self,
        election_id: str,
        holder_id: str,
        *,
        lease_seconds: float,
    ) -> RunnerLeaderLease | None:
        self.acquire_calls += 1
        if self.acquire_calls <= self.failures:
            raise RuntimeError("backend detail must not escape status")
        return self.delegate.acquire(
            election_id,
            holder_id,
            lease_seconds=lease_seconds,
        )

    def renew(
        self,
        lease: RunnerLeaderLease,
        *,
        lease_seconds: float,
    ) -> RunnerLeaderLease:
        return self.delegate.renew(lease, lease_seconds=lease_seconds)

    def authorize(self, lease: RunnerLeaderLease) -> RunnerWriterAuthority:
        return self.delegate.authorize(lease)

    def release(self, lease: RunnerLeaderLease) -> None:
        self.delegate.release(lease)


class _ReleaseFailureStore(_FlakyLeaseStore):
    def __init__(self) -> None:
        super().__init__(failures=0)

    def release(self, lease: RunnerLeaderLease) -> None:
        del lease
        raise RuntimeError("release failed")


class _FatalLeaseError(BaseException):
    pass


class _FatalRenewStore(_FlakyLeaseStore):
    def __init__(self) -> None:
        super().__init__(failures=0)
        self.renew_calls = 0

    def renew(
        self,
        lease: RunnerLeaderLease,
        *,
        lease_seconds: float,
    ) -> RunnerLeaderLease:
        del lease, lease_seconds
        self.renew_calls += 1
        raise _FatalLeaseError


def test_config_rejects_unsafe_timing_and_identity() -> None:
    with pytest.raises(ValueError, match="trimmed"):
        _config(holder_id=" controller-a")
    with pytest.raises(ValueError, match="renew_interval_seconds"):
        _config(lease_seconds=1.0, renew_interval_seconds=1.0)
    with pytest.raises(ValueError, match="readiness_max_staleness_seconds"):
        _config(
            lease_seconds=1.0,
            renew_interval_seconds=0.2,
            readiness_max_staleness_seconds=0.1,
        )
    with pytest.raises(ValueError, match="finite"):
        _config(lease_seconds=float("nan"))
    with pytest.raises(ValueError):
        _config(readiness_failure_threshold=True)


def test_run_once_campaigns_renews_and_controller_reauthorizes() -> None:
    store = InMemoryRunnerLeaderLeaseStore(clock=lambda: NOW)
    runtime = RunnerLeaderElectionRuntime(store=store, config=_config(), clock=lambda: NOW)

    acquired = runtime.run_once()
    renewed = runtime.run_once()
    assert acquired is not None
    assert renewed is not None
    assert renewed.tenure == acquired.tenure
    with pytest.raises(RunnerNotLeaderError, match="not accepting"):
        runtime.controller.mutate_autoscaler(lambda value: value)

    status = runtime.status()
    assert status.state == "new"
    assert not status.ready
    assert status.leader
    assert status.operations_started == 2
    assert status.operations_succeeded == 2
    assert status.operations_failed == 0
    assert status.campaigns_won == 1
    assert status.renewals_succeeded == 1
    runtime.close()


def test_worker_retries_backend_failures_then_becomes_ready() -> None:
    store = _FlakyLeaseStore(failures=2)
    assert isinstance(store, RunnerLeaderLeaseStore)
    runtime = RunnerLeaderElectionRuntime(store=store, config=_config(), clock=lambda: NOW)
    runtime.start()
    try:
        _wait_until(lambda: runtime.status().ready)
        status = runtime.status()
        assert status.state == "running"
        assert status.leader
        assert status.operations_failed == 2
        assert status.operations_succeeded >= 1
        assert status.campaigns_won == 1
        assert status.consecutive_failures == 0
        assert status.last_failure_at == NOW
        assert status.last_success_at == NOW
    finally:
        runtime.close()
    assert runtime.status().state == "stopped"
    assert not runtime.status().ready


def test_controller_allows_same_thread_nested_reauthorization() -> None:
    runtime = RunnerLeaderElectionRuntime(
        store=InMemoryRunnerLeaderLeaseStore(clock=lambda: NOW),
        config=_config(),
        clock=lambda: NOW,
    )
    runtime.start()
    _wait_until(lambda: runtime.status().ready)
    try:
        authority = runtime.controller.mutate_autoscaler(
            lambda initial: runtime.controller.mutate_autoscaler(
                lambda refreshed: (initial, refreshed)
            )
        )
        assert authority[0].tenure == authority[1].tenure
        assert runtime.status().active_mutations == 0
    finally:
        runtime.close()


def test_follower_stays_unready_and_can_take_over_after_release() -> None:
    store = InMemoryRunnerLeaderLeaseStore(clock=lambda: NOW)
    incumbent = RunnerLeaderElector(
        store,
        election_id="runner-control-plane",
        holder_id="controller-incumbent",
        lease_seconds=1.0,
    )
    assert incumbent.campaign() is not None
    runtime = RunnerLeaderElectionRuntime(store=store, config=_config(), clock=lambda: NOW)
    runtime.start()
    try:
        _wait_until(lambda: runtime.status().operations_succeeded >= 1)
        status = runtime.status()
        assert not status.ready
        assert not status.leader
        with pytest.raises(RunnerNotLeaderError):
            runtime.controller.mutate_autoscaler(lambda authority: authority)

        assert incumbent.resign()
        _wait_until(lambda: runtime.status().ready)
        assert runtime.status().current_lease is not None
        assert runtime.status().current_lease.fencing_token == 2
    finally:
        runtime.close()


def test_close_releases_once_and_is_idempotent() -> None:
    store = InMemoryRunnerLeaderLeaseStore(clock=lambda: NOW)
    runtime = RunnerLeaderElectionRuntime(store=store, config=_config(), clock=lambda: NOW)
    runtime.start()
    _wait_until(lambda: runtime.status().ready)
    runtime.close()
    runtime.close()
    assert not hasattr(runtime, "elector")
    with pytest.raises(RunnerNotLeaderError, match="not accepting"):
        runtime.controller.mutate_autoscaler(lambda authority: authority)

    successor = RunnerLeaderElector(
        store,
        election_id="runner-control-plane",
        holder_id="controller-b",
        lease_seconds=1.0,
    )
    lease = successor.campaign()
    assert lease is not None
    assert lease.fencing_token == 2


def test_close_reports_release_failure_but_stops_runtime() -> None:
    runtime = RunnerLeaderElectionRuntime(
        store=_ReleaseFailureStore(),
        config=_config(),
        clock=lambda: NOW,
    )
    runtime.start()
    _wait_until(lambda: runtime.status().ready)
    with pytest.raises(RuntimeError, match="lease release failed"):
        runtime.close()
    status = runtime.status()
    assert status.state == "stopped"
    assert not status.ready
    assert not status.leader
    assert status.last_failure_at == NOW
    with pytest.raises(RunnerNotLeaderError):
        runtime.controller.mutate_autoscaler(lambda authority: authority)


def test_worker_fatal_state_revokes_runtime_controller_access() -> None:
    store = _FatalRenewStore()
    runtime = RunnerLeaderElectionRuntime(store=store, config=_config(), clock=lambda: NOW)
    runtime.start()
    _wait_until(lambda: runtime.status().state == "failed")

    status = runtime.status()
    assert status.leader
    assert store.renew_calls == 1
    with pytest.raises(RunnerNotLeaderError, match="not accepting"):
        runtime.controller.mutate_autoscaler(lambda authority: authority)
    runtime.close()
    assert not runtime.status().leader


def test_close_timeout_revokes_mutations_before_delayed_renewal_finishes() -> None:
    renew_entered = threading.Event()
    release_renewal = threading.Event()

    class BlockingRenewStore(_FlakyLeaseStore):
        def renew(self, lease, *, lease_seconds):
            renew_entered.set()
            assert release_renewal.wait(1.0)
            return super().renew(lease, lease_seconds=lease_seconds)

    runtime = RunnerLeaderElectionRuntime(
        store=BlockingRenewStore(failures=0),
        config=_config(shutdown_timeout_seconds=0.01),
        clock=lambda: NOW,
    )
    runtime.start()
    _wait_until(lambda: runtime.status().ready)
    assert renew_entered.wait(1.0)

    with pytest.raises(RuntimeError, match="worker did not stop"):
        runtime.close()
    assert runtime.status().state == "stopping"
    with pytest.raises(RunnerNotLeaderError, match="not accepting"):
        runtime.controller.mutate_autoscaler(lambda authority: authority)

    release_renewal.set()
    _wait_until(lambda: runtime.status().renewals_succeeded == 1)
    with pytest.raises(RunnerNotLeaderError, match="not accepting"):
        runtime.controller.mutate_autoscaler(lambda authority: authority)
    runtime.close()
    assert runtime.status().state == "stopped"


def test_blocked_mutation_does_not_block_status_or_shutdown_deadline() -> None:
    mutation_entered = threading.Event()
    release_mutation = threading.Event()
    runtime = RunnerLeaderElectionRuntime(
        store=InMemoryRunnerLeaderLeaseStore(clock=lambda: NOW),
        config=_config(shutdown_timeout_seconds=0.02),
        clock=lambda: NOW,
    )
    runtime.start()
    _wait_until(lambda: runtime.status().ready)

    def block_mutation(_authority: RunnerWriterAuthority) -> None:
        mutation_entered.set()
        assert release_mutation.wait(1.0)

    mutation = threading.Thread(target=lambda: runtime.controller.mutate_autoscaler(block_mutation))
    mutation.start()
    assert mutation_entered.wait(1.0)

    status_started = time.monotonic()
    assert runtime.status().active_mutations == 1
    assert time.monotonic() - status_started < 0.1

    close_started = time.monotonic()
    with pytest.raises(RuntimeError, match="controller mutations did not stop"):
        runtime.close()
    assert time.monotonic() - close_started < 0.5
    assert runtime.status().state == "stopping"

    rejection_started = time.monotonic()
    with pytest.raises(RunnerNotLeaderError, match="not accepting"):
        runtime.controller.mutate_autoscaler(lambda authority: authority)
    assert time.monotonic() - rejection_started < 0.1

    release_mutation.set()
    mutation.join(1.0)
    assert not mutation.is_alive()
    runtime.close()
    assert runtime.status().state == "stopped"


def test_concurrent_run_once_is_rejected_without_starting_an_operation() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingStore(_FlakyLeaseStore):
        def acquire(self, *args, **kwargs):
            entered.set()
            assert release.wait(1.0)
            return super().acquire(*args, **kwargs)

    runtime = RunnerLeaderElectionRuntime(
        store=BlockingStore(failures=0),
        config=_config(),
        clock=lambda: NOW,
    )
    thread = threading.Thread(target=runtime.run_once)
    thread.start()
    assert entered.wait(1.0)
    with pytest.raises(RuntimeError, match="already in progress"):
        runtime.run_once()
    release.set()
    thread.join(1.0)
    assert not thread.is_alive()
    assert runtime.status().operations_started == 1
    runtime.close()
