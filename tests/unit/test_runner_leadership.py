"""Lease-fenced single-writer contract for Runner control-plane mutations."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from kairyu.runners.leadership import (
    InMemoryRunnerLeaderLeaseStore,
    InvalidRunnerLeadershipError,
    LeaderFencedRunnerController,
    RunnerLeaderCapacityError,
    RunnerLeaderElector,
    RunnerLeaderLease,
    RunnerNotLeaderError,
    RunnerWriterAuthority,
    StaleRunnerLeaderLeaseError,
)
from kairyu.runners.observation import RunnerObservation, RunnerObservationBatch
from kairyu.runners.reconciler import RunnerStatusReconciler

NOW = datetime(2026, 9, 15, 7, 0, tzinfo=UTC)


class MutableClock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class MalformedAuthorityStore(InMemoryRunnerLeaderLeaseStore):
    def __init__(self, *, clock: MutableClock, mode: str) -> None:
        super().__init__(clock=clock)
        self._mode = mode

    def authorize(self, lease: RunnerLeaderLease) -> RunnerWriterAuthority:
        authority = super().authorize(lease)
        if self._mode == "cached_time":
            return authority.model_copy(
                update={"validated_at": lease.renewed_at - timedelta(microseconds=1)}
            )
        return authority.model_copy(
            update={"lease_until": authority.lease_until - timedelta(seconds=1)}
        )


def _elector(
    store: InMemoryRunnerLeaderLeaseStore,
    holder_id: str,
    *,
    lease_seconds: float = 10.0,
) -> RunnerLeaderElector:
    return RunnerLeaderElector(
        store,
        election_id="runner-control-plane",
        holder_id=holder_id,
        lease_seconds=lease_seconds,
    )


def _batch(epoch: int, *, at: datetime) -> RunnerObservationBatch:
    observation = RunnerObservation(
        runner_id="runner-a",
        release_id="release-a",
        model_id="qwen",
        model_revision="revision-a",
        observed_at=at,
        endpoint_ready=False,
    )
    return RunnerObservationBatch(
        source_id="watcher-a",
        source_epoch=epoch,
        source_started_at=at,
        observed_at=at,
        runners=(observation,),
    )


def test_only_one_contender_acquires_and_renewal_keeps_the_tenure() -> None:
    clock = MutableClock()
    store = InMemoryRunnerLeaderLeaseStore(clock=clock)
    leader = _elector(store, "controller-a")
    follower = _elector(store, "controller-b")

    acquired = leader.campaign()
    assert acquired is not None
    assert acquired.fencing_token == 1
    assert follower.campaign() is None
    assert follower.lease is None

    clock.advance(2)
    same = leader.campaign()
    assert same == acquired
    renewed = leader.renew()
    assert renewed.tenure == acquired.tenure
    assert renewed.acquired_at == acquired.acquired_at
    assert renewed.renewed_at == NOW + timedelta(seconds=2)
    assert renewed.lease_until == NOW + timedelta(seconds=12)
    authority = leader.authority()
    assert authority.tenure == renewed.tenure
    assert authority.validated_at == NOW + timedelta(seconds=2)


def test_expiry_takeover_increments_fence_and_rejects_stale_operations() -> None:
    clock = MutableClock()
    store = InMemoryRunnerLeaderLeaseStore(clock=clock)
    previous = _elector(store, "controller-a", lease_seconds=5)
    successor = _elector(store, "controller-b", lease_seconds=5)
    stale = previous.campaign()
    assert stale is not None

    clock.advance(5)
    current = successor.campaign()
    assert current is not None
    assert current.fencing_token == stale.fencing_token + 1
    with pytest.raises(RunnerNotLeaderError, match="lost"):
        previous.authority()
    with pytest.raises(StaleRunnerLeaderLeaseError, match="renewed"):
        store.renew(stale, lease_seconds=5)
    with pytest.raises(StaleRunnerLeaderLeaseError, match="released"):
        store.release(stale)


def test_resignation_is_idempotent_locally_and_next_tenure_is_fenced() -> None:
    store = InMemoryRunnerLeaderLeaseStore(clock=MutableClock())
    first = _elector(store, "controller-a")
    second = _elector(store, "controller-b")
    first_lease = first.campaign()
    assert first_lease is not None
    assert first.resign() is True
    assert first.resign() is False

    second_lease = second.campaign()
    assert second_lease is not None
    assert second_lease.fencing_token == first_lease.fencing_token + 1


def test_fenced_controller_rejects_follower_and_expired_leader_before_mutation() -> None:
    clock = MutableClock()
    store = InMemoryRunnerLeaderLeaseStore(clock=clock)
    first = _elector(store, "controller-a", lease_seconds=5)
    second = _elector(store, "controller-b", lease_seconds=5)
    reconciler = RunnerStatusReconciler()
    first_gate = LeaderFencedRunnerController(first, reconciler)
    second_gate = LeaderFencedRunnerController(second, reconciler)

    with pytest.raises(RunnerNotLeaderError, match="not the elected"):
        second_gate.reconcile(_batch(1, at=NOW))
    assert reconciler.statuses == {}

    lease = first.campaign()
    assert lease is not None
    initial = first_gate.reconcile(_batch(1, at=NOW))["runner-a"]
    assert initial.observed_at == NOW
    assert first_gate.last_fencing_token == lease.fencing_token

    clock.advance(5)
    next_at = NOW + timedelta(seconds=5)
    with pytest.raises(RunnerNotLeaderError, match="lost"):
        first_gate.reconcile(_batch(2, at=next_at))
    assert reconciler.statuses["runner-a"].observed_at == NOW

    successor = second.campaign()
    assert successor is not None
    updated = second_gate.reconcile(_batch(2, at=next_at))["runner-a"]
    assert updated.observed_at == next_at
    assert second_gate.last_fencing_token == successor.fencing_token


def test_autoscaler_mutation_receives_current_fencing_authority() -> None:
    clock = MutableClock()
    store = InMemoryRunnerLeaderLeaseStore(clock=clock)
    elector = _elector(store, "controller-a")
    gate = LeaderFencedRunnerController(elector, RunnerStatusReconciler())
    called: list[int] = []

    with pytest.raises(RunnerNotLeaderError):
        gate.mutate_autoscaler(lambda authority: called.append(authority.fencing_token))
    assert called == []

    lease = elector.campaign()
    assert lease is not None
    result = gate.mutate_autoscaler(
        lambda authority: (called.append(authority.fencing_token), "scaled")[1]
    )
    assert result == "scaled"
    assert called == [lease.fencing_token]


def test_takeover_does_not_wait_for_an_old_callback_after_expiry() -> None:
    clock = MutableClock()
    store = InMemoryRunnerLeaderLeaseStore(clock=clock)
    leader = _elector(store, "controller-a", lease_seconds=5)
    successor = _elector(store, "controller-b", lease_seconds=5)
    gate = LeaderFencedRunnerController(leader, RunnerStatusReconciler())
    lease = leader.campaign()
    assert lease is not None
    entered = threading.Event()
    release = threading.Event()
    campaign_started = threading.Event()

    def mutation(authority):
        entered.set()
        assert release.wait(timeout=2)
        return authority.fencing_token

    def campaign():
        campaign_started.set()
        return successor.campaign()

    with ThreadPoolExecutor(max_workers=2) as pool:
        mutation_result = pool.submit(gate.mutate_autoscaler, mutation)
        assert entered.wait(timeout=2)
        clock.advance(5)
        takeover_result = pool.submit(campaign)
        assert campaign_started.wait(timeout=2)
        takeover = takeover_result.result(timeout=2)
        assert takeover is not None
        assert takeover.fencing_token == lease.fencing_token + 1
        assert mutation_result.done() is False
        release.set()
        assert mutation_result.result(timeout=2) == lease.fencing_token


def test_concurrent_campaign_elects_exactly_one_holder() -> None:
    store = InMemoryRunnerLeaderLeaseStore(clock=MutableClock())
    contenders = tuple(_elector(store, f"controller-{index}") for index in range(32))
    with ThreadPoolExecutor(max_workers=16) as pool:
        leases = tuple(pool.map(lambda contender: contender.campaign(), contenders))
    acquired = tuple(lease for lease in leases if lease is not None)
    assert len(acquired) == 1
    assert acquired[0].fencing_token == 1


def test_capacity_and_fencing_token_tombstone_are_fail_closed() -> None:
    store = InMemoryRunnerLeaderLeaseStore(clock=MutableClock(), max_elections=1)
    first = store.acquire("election-a", "controller-a", lease_seconds=5)
    assert first is not None
    store.release(first)
    with pytest.raises(RunnerLeaderCapacityError, match="capacity"):
        store.acquire("election-b", "controller-b", lease_seconds=5)
    reacquired = store.acquire("election-a", "controller-a", lease_seconds=5)
    assert reacquired is not None
    assert reacquired.fencing_token == 2


@pytest.mark.parametrize("lease_seconds", [True, 0, -1, float("inf"), float("nan")])
def test_invalid_lease_durations_are_rejected(lease_seconds: object) -> None:
    store = InMemoryRunnerLeaderLeaseStore(clock=MutableClock())
    with pytest.raises(ValueError, match="lease_seconds"):
        store.acquire(
            "runner-control-plane",
            "controller-a",
            lease_seconds=lease_seconds,  # type: ignore[arg-type]
        )


def test_store_clock_rollback_and_naive_time_fail_closed() -> None:
    clock = MutableClock()
    store = InMemoryRunnerLeaderLeaseStore(clock=clock)
    assert store.acquire("election-a", "controller-a", lease_seconds=5) is not None
    clock.now = NOW - timedelta(microseconds=1)
    with pytest.raises(InvalidRunnerLeadershipError, match="backwards"):
        store.acquire("election-a", "controller-a", lease_seconds=5)

    naive_store = InMemoryRunnerLeaderLeaseStore(clock=MutableClock(NOW.replace(tzinfo=None)))
    with pytest.raises(ValueError, match="timezone-aware"):
        naive_store.acquire("election-a", "controller-a", lease_seconds=5)


def test_models_reject_invalid_intervals_and_validator_bypass() -> None:
    with pytest.raises(ValidationError, match="lease_until"):
        RunnerLeaderLease(
            election_id="runner-control-plane",
            holder_id="controller-a",
            fencing_token=1,
            acquired_at=NOW,
            renewed_at=NOW,
            lease_until=NOW,
        )

    store = InMemoryRunnerLeaderLeaseStore(clock=MutableClock())
    lease = store.acquire("election-a", "controller-a", lease_seconds=5)
    assert lease is not None
    bypass = lease.model_copy(update={"fencing_token": True})
    with pytest.raises(ValidationError, match="fencing_token"):
        store.authorize(bypass)


@pytest.mark.parametrize("mode", ["cached_time", "wrong_deadline"])
def test_elector_rejects_cached_or_conflicting_backend_authority(mode: str) -> None:
    clock = MutableClock()
    store = MalformedAuthorityStore(clock=clock, mode=mode)
    elector = _elector(store, "controller-a")
    lease = elector.campaign()
    assert lease is not None
    with pytest.raises(InvalidRunnerLeadershipError, match="stale or conflicting"):
        elector.authority()
    assert elector.lease is None
