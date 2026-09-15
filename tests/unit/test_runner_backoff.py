"""Crash backoff and revision/node/GPU quarantine contract."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from kairyu.runners import (
    InvalidRunnerFailureEvidenceError,
    RunnerBackoffPolicy,
    RunnerFailure,
    RunnerFailureCapacityError,
    RunnerFailureDomainKind,
    RunnerFailureGuard,
    RunnerStartupPhase,
    RunnerStartupPhaseOutcome,
    RunnerStartupPhaseReport,
    RunnerStartupReport,
    RunnerState,
    RunnerStatus,
    gpu_failure_domain,
    node_failure_domain,
    revision_failure_domain,
)

NOW = datetime(2026, 9, 15, 3, 0, tzinfo=UTC)


def _failed(
    runner_id: str,
    *,
    at: datetime,
    domain: RunnerFailureDomainKind | None = RunnerFailureDomainKind.REVISION,
    code: str = "container_crash_loop",
    release_id: str = "release-a",
    model_revision: str = "revision-a",
    node_name: str | None = "node-a",
    gpu_uuids: tuple[str, ...] = ("GPU-a",),
    state_version: int = 3,
    startup: RunnerStartupReport | None = None,
) -> RunnerStatus:
    return RunnerStatus(
        runner_id=runner_id,
        release_id=release_id,
        model_id="qwen",
        model_revision=model_revision,
        state=RunnerState.UNHEALTHY,
        state_version=state_version,
        state_changed_at=at,
        observed_at=at,
        node_name=node_name,
        pod_uid=runner_id,
        gpu_uuids=gpu_uuids,
        startup=startup,
        failure=RunnerFailure(
            code=code,
            message="sanitized failure",
            domain=domain,
        ),
    )


def _policy(**updates: object) -> RunnerBackoffPolicy:
    values = {
        "base_delay_seconds": 5.0,
        "backoff_factor": 2,
        "max_delay_seconds": 12.0,
        "failure_window_seconds": 60.0,
        "quarantine_threshold": 3,
        "quarantine_seconds": 30.0,
        "max_events_per_domain": 8,
        "max_domains": 32,
    }
    values.update(updates)
    return RunnerBackoffPolicy.model_validate(values)


def test_revision_failures_back_off_then_quarantine_exact_revision() -> None:
    guard = RunnerFailureGuard(_policy())
    domain = revision_failure_domain(
        release_id="release-a",
        model_id="qwen",
        model_revision="revision-a",
    )

    first = guard.record(_failed("runner-1", at=NOW), at=NOW)[0]
    assert first.failure_count == 1
    assert first.backoff_until == NOW + timedelta(seconds=5)
    assert first.blocked is True
    assert first.quarantined is False

    second_at = NOW + timedelta(seconds=10)
    second = guard.record(_failed("runner-2", at=second_at), at=second_at)[0]
    assert second.failure_count == 2
    assert second.backoff_until == NOW + timedelta(seconds=20)

    third_at = NOW + timedelta(seconds=20)
    third = guard.record(_failed("runner-3", at=third_at), at=third_at)[0]
    assert third.failure_count == 3
    assert third.backoff_until == NOW + timedelta(seconds=32)
    assert third.quarantine_until == NOW + timedelta(seconds=50)
    assert third.eligible_at == third.quarantine_until
    assert third.quarantined is True

    blocked = guard.blocked_domains(
        release_id="release-a",
        model_id="qwen",
        model_revision="revision-a",
        at=NOW + timedelta(seconds=21),
    )
    assert tuple(decision.domain for decision in blocked) == (domain,)
    assert (
        guard.blocked_domains(
            release_id="release-b",
            model_id="qwen",
            model_revision="revision-b",
            at=NOW + timedelta(seconds=21),
        )
        == ()
    )
    expired_at = NOW + timedelta(seconds=81)
    expired = guard.decision(domain, at=expired_at)
    assert expired.failure_count == 0
    assert expired.blocked is False
    assert guard.snapshot(at=expired_at).domains == ()


def test_gpu_failure_blocks_each_physical_gpu_without_blocking_node() -> None:
    guard = RunnerFailureGuard(_policy(quarantine_threshold=2))
    status = _failed(
        "runner-gpu",
        at=NOW,
        domain=RunnerFailureDomainKind.GPU,
        code="gpu_xid",
        gpu_uuids=("GPU-b", "GPU-a"),
    )
    decisions = guard.record(status, at=NOW)
    assert tuple(decision.domain for decision in decisions) == (
        gpu_failure_domain("GPU-a"),
        gpu_failure_domain("GPU-b"),
    )
    assert guard.blocked_domains(
        release_id="other-release",
        model_id="other-model",
        model_revision="other-revision",
        node_name="node-a",
        gpu_uuids=("GPU-b",),
        at=NOW + timedelta(seconds=1),
    )[0].domain == gpu_failure_domain("GPU-b")
    assert (
        guard.decision(
            node_failure_domain("node-a"),
            at=NOW + timedelta(seconds=1),
        ).blocked
        is False
    )


@pytest.mark.parametrize("gpu_uuids", ["GPU-a", b"GPU-a"])
def test_candidate_gpu_domains_reject_scalar_text(gpu_uuids: object) -> None:
    guard = RunnerFailureGuard(_policy())
    with pytest.raises(TypeError, match="iterable of UUID strings"):
        guard.blocked_domains(
            release_id="release-a",
            model_id="qwen",
            model_revision="revision-a",
            gpu_uuids=gpu_uuids,  # type: ignore[arg-type]
            at=NOW,
        )


def test_node_failure_blocks_node_across_revisions() -> None:
    guard = RunnerFailureGuard(_policy())
    guard.record(
        _failed(
            "runner-node",
            at=NOW,
            domain=RunnerFailureDomainKind.NODE,
            code="pod_unknown",
        ),
        at=NOW,
    )
    blocked = guard.blocked_domains(
        release_id="unrelated-release",
        model_id="llama",
        model_revision="unrelated-revision",
        node_name="node-a",
        at=NOW + timedelta(seconds=1),
    )
    assert tuple(decision.domain for decision in blocked) == (node_failure_domain("node-a"),)


def test_failure_replay_is_idempotent_and_conflicting_replay_fails_closed() -> None:
    guard = RunnerFailureGuard(_policy())
    status = _failed("runner-a", at=NOW)
    first = guard.record(status, at=NOW)
    snapshot = guard.snapshot(at=NOW)
    assert guard.record(status, at=NOW) == first
    assert guard.snapshot(at=NOW) == snapshot

    conflicting = _failed(
        "runner-a",
        at=NOW,
        code="different_failure",
        state_version=status.state_version,
    )
    with pytest.raises(
        InvalidRunnerFailureEvidenceError,
        match="changed its evidence",
    ):
        guard.record(conflicting, at=NOW)


def test_global_tombstone_rejects_domain_changes_after_none_or_event_eviction() -> None:
    policy = _policy(
        quarantine_threshold=2,
        max_events_per_domain=2,
        max_observations=16,
    )
    unscoped_guard = RunnerFailureGuard(policy)
    unscoped = _failed("runner-unscoped", at=NOW, domain=None)
    assert unscoped_guard.record(unscoped, at=NOW) == ()
    with pytest.raises(
        InvalidRunnerFailureEvidenceError,
        match="changed its evidence",
    ):
        unscoped_guard.record(
            _failed("runner-unscoped", at=NOW),
            at=NOW,
        )

    evicted_guard = RunnerFailureGuard(policy)
    original = _failed("runner-0", at=NOW)
    evicted_guard.record(original, at=NOW)
    for index in (1, 2):
        at = NOW + timedelta(seconds=index)
        evicted_guard.record(_failed(f"runner-{index}", at=at), at=at)
    snapshot = evicted_guard.snapshot(at=NOW + timedelta(seconds=2))
    assert all(event.runner_id != original.runner_id for event in snapshot.domains[0].events)
    assert any(observation.runner_id == original.runner_id for observation in snapshot.observations)
    with pytest.raises(
        InvalidRunnerFailureEvidenceError,
        match="changed its evidence",
    ):
        evicted_guard.record(
            _failed(
                "runner-0",
                at=NOW,
                domain=RunnerFailureDomainKind.NODE,
                code="pod_unknown",
            ),
            at=NOW + timedelta(seconds=2),
        )


def test_multi_gpu_exact_replay_survives_asymmetric_event_eviction() -> None:
    guard = RunnerFailureGuard(
        _policy(
            quarantine_threshold=2,
            max_events_per_domain=2,
            max_observations=16,
        )
    )
    shared = _failed(
        "runner-shared",
        at=NOW,
        domain=RunnerFailureDomainKind.GPU,
        code="gpu_xid",
        gpu_uuids=("GPU-a", "GPU-b"),
    )
    guard.record(shared, at=NOW)
    for index in (1, 2):
        at = NOW + timedelta(seconds=index)
        guard.record(
            _failed(
                f"runner-a-{index}",
                at=at,
                domain=RunnerFailureDomainKind.GPU,
                code="gpu_xid",
                gpu_uuids=("GPU-a",),
            ),
            at=at,
        )
    replay = guard.record(shared, at=NOW + timedelta(seconds=2))
    assert tuple(decision.domain for decision in replay) == (
        gpu_failure_domain("GPU-a"),
        gpu_failure_domain("GPU-b"),
    )


def test_multi_gpu_replay_survives_one_expired_domain_retained_by_quarantine() -> None:
    guard = RunnerFailureGuard(
        _policy(
            failure_window_seconds=5.0,
            quarantine_threshold=2,
            quarantine_seconds=30.0,
        )
    )
    shared = _failed(
        "runner-shared",
        at=NOW,
        domain=RunnerFailureDomainKind.GPU,
        code="gpu_xid",
        gpu_uuids=("GPU-a", "GPU-b"),
    )
    guard.record(shared, at=NOW)
    guard.record(
        _failed(
            "runner-b",
            at=NOW + timedelta(seconds=1),
            domain=RunnerFailureDomainKind.GPU,
            code="gpu_xid",
            gpu_uuids=("GPU-b",),
        ),
        at=NOW + timedelta(seconds=1),
    )

    snapshot_at = NOW + timedelta(seconds=10)
    snapshot = guard.snapshot(at=snapshot_at)
    assert tuple(state.domain for state in snapshot.domains) == (gpu_failure_domain("GPU-b"),)
    restored = RunnerFailureGuard.from_snapshot(snapshot)
    replay = restored.record(shared, at=snapshot_at)
    assert replay[0].domain == gpu_failure_domain("GPU-a")
    assert replay[0].blocked is False
    assert replay[1].domain == gpu_failure_domain("GPU-b")
    assert replay[1].quarantined is True


def test_snapshot_restore_preserves_backoff_and_quarantine() -> None:
    guard = RunnerFailureGuard(_policy(quarantine_threshold=2))
    guard.record(_failed("runner-1", at=NOW), at=NOW)
    second_at = NOW + timedelta(seconds=10)
    guard.record(_failed("runner-2", at=second_at), at=second_at)
    snapshot = guard.snapshot(at=second_at)
    snapshot = type(snapshot).model_validate_json(snapshot.model_dump_json())

    restored = RunnerFailureGuard.from_snapshot(snapshot)
    domain = revision_failure_domain(
        release_id="release-a",
        model_id="qwen",
        model_revision="revision-a",
    )
    evaluated_at = NOW + timedelta(seconds=11)
    assert restored.decision(domain, at=evaluated_at) == guard.decision(
        domain,
        at=evaluated_at,
    )
    assert restored.snapshot(at=evaluated_at).model_dump(mode="json") == (
        guard.snapshot(at=evaluated_at).model_dump(mode="json")
    )

    malformed = snapshot.model_dump()
    malformed["domains"][0]["quarantine_until"] = NOW + timedelta(days=365)
    with pytest.raises(ValueError, match="deadline does not match"):
        type(snapshot).model_validate(malformed)

    missing_quarantine = snapshot.model_dump()
    missing_quarantine["domains"][0]["quarantine_until"] = None
    with pytest.raises(ValueError, match="deadline does not match"):
        type(snapshot).model_validate(missing_quarantine)

    missing_ledger = snapshot.model_dump()
    missing_ledger["domains"] = ()
    with pytest.raises(ValueError, match="ledger conflicts with its observations"):
        type(snapshot).model_validate(missing_ledger)

    validator_bypass = snapshot.model_copy(update={"domains": ()})
    with pytest.raises(ValueError, match="ledger conflicts with its observations"):
        RunnerFailureGuard.from_snapshot(validator_bypass)


def test_active_quarantine_retains_evidence_beyond_failure_window() -> None:
    policy = _policy(
        failure_window_seconds=5.0,
        quarantine_threshold=2,
        quarantine_seconds=30.0,
    )
    guard = RunnerFailureGuard(policy)
    guard.record(_failed("runner-1", at=NOW), at=NOW)
    second_at = NOW + timedelta(seconds=1)
    guard.record(_failed("runner-2", at=second_at), at=second_at)
    snapshot_at = NOW + timedelta(seconds=10)
    snapshot = guard.snapshot(at=snapshot_at)
    assert len(snapshot.domains[0].events) == 2
    assert len(snapshot.observations) == 2
    restored = RunnerFailureGuard.from_snapshot(snapshot)
    domain = revision_failure_domain(
        release_id="release-a",
        model_id="qwen",
        model_revision="revision-a",
    )
    assert restored.decision(domain, at=snapshot_at).quarantine_until == (
        NOW + timedelta(seconds=31)
    )


def test_event_and_domain_capacity_are_bounded_without_partial_commit() -> None:
    policy = _policy(
        quarantine_threshold=2,
        max_events_per_domain=3,
        max_domains=1,
    )
    guard = RunnerFailureGuard(policy)
    for index in range(5):
        at = NOW + timedelta(seconds=index)
        guard.record(_failed(f"runner-{index}", at=at), at=at)
    snapshot = guard.snapshot(at=NOW + timedelta(seconds=4))
    assert len(snapshot.domains) == 1
    assert len(snapshot.domains[0].events) == 3
    before = snapshot.model_dump(mode="json")

    with pytest.raises(RunnerFailureCapacityError, match="capacity"):
        guard.record(
            _failed(
                "runner-other",
                at=NOW + timedelta(seconds=5),
                release_id="release-b",
                model_revision="revision-b",
            ),
            at=NOW + timedelta(seconds=5),
        )
    assert guard.snapshot(at=NOW + timedelta(seconds=4)).model_dump(mode="json") == before
    recovered_at = NOW + timedelta(seconds=100)
    recovered = guard.record(
        _failed(
            "runner-recovered",
            at=recovered_at,
            release_id="release-b",
            model_revision="revision-b",
        ),
        at=recovered_at,
    )
    assert recovered[0].domain.release_id == "release-b"
    assert len(guard.snapshot(at=recovered_at).domains) == 1


def test_live_observation_capacity_fails_without_dropping_dedupe_state() -> None:
    guard = RunnerFailureGuard(_policy(max_observations=2))
    for index in range(2):
        at = NOW + timedelta(seconds=index)
        assert (
            guard.record(
                _failed(f"runner-{index}", at=at, domain=None),
                at=at,
            )
            == ()
        )
    before = guard.snapshot(at=NOW + timedelta(seconds=1))
    with pytest.raises(RunnerFailureCapacityError, match="observations"):
        guard.record(
            _failed(
                "runner-overflow",
                at=NOW + timedelta(seconds=2),
                domain=None,
            ),
            at=NOW + timedelta(seconds=2),
        )
    assert guard.snapshot(at=NOW + timedelta(seconds=1)).observations == (before.observations)


def test_record_many_is_atomic_when_one_observation_exceeds_capacity() -> None:
    guard = RunnerFailureGuard(_policy(max_observations=1))
    before = guard.snapshot(at=NOW)
    with pytest.raises(RunnerFailureCapacityError, match="observations"):
        guard.record_many(
            (
                _failed("runner-a", at=NOW),
                _failed("runner-b", at=NOW),
            ),
            at=NOW,
        )
    assert guard.snapshot(at=NOW) == before


def test_stale_failure_replay_cannot_retrigger_backoff_or_quarantine() -> None:
    guard = RunnerFailureGuard(_policy(failure_window_seconds=10.0, quarantine_threshold=2))
    old = _failed("runner-old", at=NOW)
    guard.record(old, at=NOW)
    evaluated_at = NOW + timedelta(seconds=11)
    assert guard.record(old, at=evaluated_at) == ()
    domain = revision_failure_domain(
        release_id="release-a",
        model_id="qwen",
        model_revision="revision-a",
    )
    decision = guard.decision(domain, at=evaluated_at)
    assert decision.failure_count == 0
    assert decision.quarantined is False


def test_unkeyable_and_future_failure_evidence_fails_closed() -> None:
    guard = RunnerFailureGuard(_policy())
    with pytest.raises(InvalidRunnerFailureEvidenceError, match="node identity"):
        guard.record(
            _failed(
                "runner-node",
                at=NOW,
                domain=RunnerFailureDomainKind.NODE,
                node_name=None,
            ),
            at=NOW,
        )
    with pytest.raises(InvalidRunnerFailureEvidenceError, match="GPU identity"):
        guard.record(
            _failed(
                "runner-gpu",
                at=NOW,
                domain=RunnerFailureDomainKind.GPU,
                gpu_uuids=(),
            ),
            at=NOW,
        )
    with pytest.raises(InvalidRunnerFailureEvidenceError, match="future"):
        guard.record(
            _failed("runner-future", at=NOW + timedelta(seconds=1)),
            at=NOW,
        )
    guard.record(_failed("runner-now", at=NOW), at=NOW)
    with pytest.raises(InvalidRunnerFailureEvidenceError, match="backwards"):
        guard.decision(
            revision_failure_domain(
                release_id="release-a",
                model_id="qwen",
                model_revision="revision-a",
            ),
            at=NOW - timedelta(seconds=1),
        )


def test_startup_failure_without_explicit_scope_uses_revision_domain() -> None:
    failure = RunnerFailure(code="model_digest_mismatch", message="digest mismatch")
    phase = RunnerStartupPhaseReport(
        phase=RunnerStartupPhase.IMAGE_PULL,
        started_at=NOW,
        completed_at=NOW,
        outcome=RunnerStartupPhaseOutcome.FAILED,
        failure=failure,
    )
    startup = RunnerStartupReport(
        runner_id="runner-startup",
        observed_at=NOW,
        phases=(phase,),
    )
    status = _failed(
        "runner-startup",
        at=NOW,
        domain=None,
        code=failure.code,
        startup=startup,
    )
    values = status.model_dump()
    values["failure"] = failure
    status = RunnerStatus.model_validate(values)
    decision = RunnerFailureGuard(_policy()).record(status, at=NOW)[0]
    assert decision.domain == revision_failure_domain(
        release_id="release-a",
        model_id="qwen",
        model_revision="revision-a",
    )
