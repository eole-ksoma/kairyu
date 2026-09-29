"""Executable contract for autoscaler observations and decision logging."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from kairyu.runners import (
    InMemoryScalingDecisionLog,
    RunnerStartupPhase,
    ScalingDecisionAction,
    ScalingDecisionCapacityError,
    ScalingDecisionConflictError,
    ScalingDecisionLog,
    ScalingDecisionReason,
    ScalingDecisionRecord,
    ScalingObservation,
    ScalingObservationWindow,
    ScalingPolicy,
    ScalingQueueSnapshot,
    ScalingResourceSnapshot,
    ScalingRunnerSnapshot,
    ScalingStartupPhaseMetrics,
    ScalingStartupSnapshot,
)

NOW = datetime(2026, 9, 15, 6, 0, tzinfo=UTC)


def _policy(**updates) -> ScalingPolicy:
    values = {
        "model_class": "interactive-14b",
        "policy_revision": 7,
        "min_replicas": 2,
        "max_replicas": 50,
        "warm_buffer_replicas": 3,
        "warm_buffer_ratio": 0.25,
        "max_scale_up_step": 8,
        "max_scale_down_step": 2,
    }
    values.update(updates)
    return ScalingPolicy(**values)


def _observation(
    suffix: str,
    *,
    observed_at: datetime,
    current_replicas: int = 3,
    model_class: str = "interactive-14b",
) -> ScalingObservation:
    source_at = observed_at - timedelta(seconds=1)
    return ScalingObservation(
        observation_id=f"observation-{suffix}",
        model_class=model_class,
        observed_at=observed_at,
        queue=ScalingQueueSnapshot(
            observed_at=source_at,
            queue_depth=6,
            interactive_queue_depth=4,
            batch_queue_depth=2,
            oldest_queue_age_seconds=2.5,
            arrival_rate_per_second=3.25,
            deadline_remaining_p50_seconds=10,
            deadline_remaining_p95_seconds=30,
            predicted_ttft_seconds=0.8,
            goodput_slo_ratio=0.97,
        ),
        runners=ScalingRunnerSnapshot(
            observed_at=source_at,
            current_replicas=current_replicas,
            busy_replicas=int(current_replicas >= 1),
            ready_replicas=int(current_replicas >= 2),
            loading_replicas=int(current_replicas >= 3),
            unhealthy_replicas=0,
        ),
        resources=ScalingResourceSnapshot(
            observed_at=source_at,
            gpu_utilization=0.9,
            hbm_utilization=0.75,
            kv_utilization=0.6,
            multiplexing_occupancy=0.5,
            batch_occupancy=0.4,
        ),
        startup=ScalingStartupSnapshot(
            observed_at=source_at,
            model_cache_resident_replicas=min(2, current_replicas),
            phases=(
                ScalingStartupPhaseMetrics(
                    phase=RunnerStartupPhase.IMAGE_PULL,
                    sample_count=20,
                    ema_seconds=1.5,
                    p95_seconds=3,
                ),
                ScalingStartupPhaseMetrics(
                    phase=RunnerStartupPhase.MODEL_FETCH,
                    sample_count=20,
                    ema_seconds=4,
                    p95_seconds=9,
                ),
            ),
        ),
    )


def _window(*, model_class: str = "interactive-14b") -> ScalingObservationWindow:
    return ScalingObservationWindow(
        window_id="window-1",
        model_class=model_class,
        started_at=NOW - timedelta(seconds=30),
        ended_at=NOW,
        observations=(
            _observation(
                "1",
                observed_at=NOW - timedelta(seconds=20),
                model_class=model_class,
            ),
            _observation("2", observed_at=NOW, model_class=model_class),
        ),
    )


def _record(
    decision_id: str = "decision-1",
    **updates,
) -> ScalingDecisionRecord:
    values = {
        "decision_id": decision_id,
        "decided_at": NOW + timedelta(seconds=1),
        "catalog_revision": 9,
        "policy": _policy(),
        "window": _window(),
        "action": ScalingDecisionAction.HOLD,
        "reason": ScalingDecisionReason.NO_CHANGE,
        "demand_replicas": 3,
        "buffered_target_replicas": 6,
        "desired_replicas": 3,
        "target_delta": 0,
    }
    values.update(updates)
    return ScalingDecisionRecord(**values)


def test_observation_captures_all_planned_input_families() -> None:
    observation = _observation("1", observed_at=NOW)

    assert observation.queue.interactive_queue_depth == 4
    assert observation.queue.deadline_remaining_p95_seconds == 30
    assert observation.runners.loading_replicas == 1
    assert observation.resources is not None
    assert observation.resources.kv_utilization == 0.6
    assert observation.startup is not None
    assert observation.startup.phases[1].phase is RunnerStartupPhase.MODEL_FETCH
    assert ScalingObservation.model_validate_json(observation.model_dump_json()) == observation


def test_queue_snapshot_requires_consistent_classes_and_deadline_percentiles() -> None:
    observation = _observation("1", observed_at=NOW)
    values = observation.queue.model_dump()
    values["batch_queue_depth"] = 1
    with pytest.raises(ValidationError, match="sum to queue_depth"):
        ScalingQueueSnapshot(**values)

    values = observation.queue.model_dump()
    values["deadline_remaining_p95_seconds"] = None
    with pytest.raises(ValidationError, match="both present"):
        ScalingQueueSnapshot(**values)

    values = observation.queue.model_dump()
    values["deadline_remaining_p95_seconds"] = 5
    with pytest.raises(ValidationError, match="p95"):
        ScalingQueueSnapshot(**values)


def test_runner_snapshot_rejects_overclassified_replicas() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        ScalingRunnerSnapshot(
            observed_at=NOW,
            current_replicas=2,
            busy_replicas=1,
            ready_replicas=1,
            loading_replicas=1,
            unhealthy_replicas=0,
        )


def test_observation_rejects_future_sources_and_impossible_cache_count() -> None:
    observation = _observation("1", observed_at=NOW)
    values = observation.model_dump()
    values["queue"]["observed_at"] = NOW + timedelta(seconds=1)
    with pytest.raises(ValidationError, match="cannot exceed"):
        ScalingObservation(**values)

    values = observation.model_dump()
    values["startup"]["model_cache_resident_replicas"] = 4
    with pytest.raises(ValidationError, match="cache-resident"):
        ScalingObservation(**values)


def test_startup_phases_are_unique_and_canonically_ordered() -> None:
    phase = ScalingStartupPhaseMetrics(
        phase=RunnerStartupPhase.MODEL_FETCH,
        sample_count=1,
        ema_seconds=1,
        p95_seconds=2,
    )
    with pytest.raises(ValidationError, match="unique"):
        ScalingStartupSnapshot(
            observed_at=NOW,
            model_cache_resident_replicas=0,
            phases=(phase, phase),
        )
    with pytest.raises(ValidationError, match="canonical order"):
        ScalingStartupSnapshot(
            observed_at=NOW,
            model_cache_resident_replicas=0,
            phases=(
                phase,
                ScalingStartupPhaseMetrics(
                    phase=RunnerStartupPhase.IMAGE_PULL,
                    sample_count=1,
                    ema_seconds=1,
                    p95_seconds=2,
                ),
            ),
        )


def test_window_is_ordered_bounded_and_fingerprinted() -> None:
    window = _window()

    assert len(window.fingerprint) == 64
    assert ScalingObservationWindow.model_validate_json(window.model_dump_json()) == window
    reversed_observations = tuple(reversed(window.observations))
    with pytest.raises(ValidationError, match="strictly time ordered"):
        ScalingObservationWindow(
            window_id=window.window_id,
            model_class=window.model_class,
            started_at=window.started_at,
            ended_at=window.ended_at,
            observations=reversed_observations,
        )
    with pytest.raises(ValidationError, match="match model_class"):
        ScalingObservationWindow(
            window_id=window.window_id,
            model_class="batch-14b",
            started_at=window.started_at,
            ended_at=window.ended_at,
            observations=window.observations,
        )


def test_decision_persists_exact_window_policy_revision_and_reason() -> None:
    record = _record()

    assert record.catalog_revision == 9
    assert record.policy.identity == ("interactive-14b", 7)
    assert record.window.fingerprint
    assert record.reason is ScalingDecisionReason.NO_CHANGE
    assert len(record.fingerprint) == 64
    assert ScalingDecisionRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"decided_at": NOW - timedelta(seconds=1)}, "cannot predate"),
        ({"desired_replicas": 4}, "target_delta"),
        ({"desired_replicas": 51, "target_delta": 48}, "policy min/max"),
        (
            {
                "action": ScalingDecisionAction.SCALE_UP,
                "desired_replicas": 12,
                "target_delta": 9,
            },
            "scale-up delta",
        ),
        (
            {
                "action": ScalingDecisionAction.SCALE_DOWN,
                "desired_replicas": 0,
                "target_delta": -3,
            },
            "policy min/max",
        ),
    ],
)
def test_decision_rejects_inconsistent_targets(updates, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _record(**updates)


def test_decision_rejects_buffer_not_derived_from_policy() -> None:
    with pytest.raises(ValidationError, match="demand plus the policy buffer"):
        _record(buffered_target_replicas=5)


def test_decision_can_converge_from_outside_new_policy_bounds() -> None:
    over_max = _record(
        policy=_policy(max_scale_down_step=20),
        window=ScalingObservationWindow(
            window_id="window-over-max",
            model_class="interactive-14b",
            started_at=NOW - timedelta(seconds=30),
            ended_at=NOW,
            observations=(_observation("over-max", observed_at=NOW, current_replicas=60),),
        ),
        action=ScalingDecisionAction.SCALE_DOWN,
        reason=ScalingDecisionReason.MAX_REPLICAS,
        desired_replicas=50,
        target_delta=-10,
    )
    assert over_max.desired_replicas == 50

    under_min = _record(
        policy=_policy(min_replicas=10, max_scale_up_step=8),
        window=ScalingObservationWindow(
            window_id="window-under-min",
            model_class="interactive-14b",
            started_at=NOW - timedelta(seconds=30),
            ended_at=NOW,
            observations=(_observation("under-min", observed_at=NOW, current_replicas=0),),
        ),
        action=ScalingDecisionAction.SCALE_UP,
        reason=ScalingDecisionReason.MIN_REPLICAS,
        desired_replicas=8,
        target_delta=8,
    )
    assert under_min.desired_replicas == 8


@pytest.mark.parametrize(
    ("current", "desired", "action", "message"),
    [
        (60, 49, ScalingDecisionAction.SCALE_DOWN, "policy max"),
        (0, 11, ScalingDecisionAction.SCALE_UP, "policy min"),
        (60, 61, ScalingDecisionAction.SCALE_UP, "policy max"),
    ],
)
def test_out_of_range_decisions_cannot_move_away_or_overshoot(
    current: int,
    desired: int,
    action: ScalingDecisionAction,
    message: str,
) -> None:
    policy = _policy(
        min_replicas=10,
        max_scale_up_step=20,
        max_scale_down_step=20,
    )
    window = ScalingObservationWindow(
        window_id=f"window-{current}-{desired}",
        model_class="interactive-14b",
        started_at=NOW - timedelta(seconds=30),
        ended_at=NOW,
        observations=(
            _observation("outside", observed_at=NOW, current_replicas=current),
        ),
    )
    with pytest.raises(ValidationError, match=message):
        _record(
            policy=policy,
            window=window,
            action=action,
            reason=ScalingDecisionReason.MAX_REPLICAS,
            desired_replicas=desired,
            target_delta=desired - current,
        )


def test_buffered_target_accepts_the_documented_maximum() -> None:
    policy = _policy(
        max_replicas=100_000,
        warm_buffer_replicas=100_000,
        warm_buffer_ratio=10,
        max_scale_up_step=100_000,
        max_scale_down_step=100_000,
    )
    record = _record(
        policy=policy,
        demand_replicas=10_000_000,
        buffered_target_replicas=11_000_000,
    )
    assert record.buffered_target_replicas == 11_000_000


def test_stale_inputs_require_reason_and_never_allow_scale_down() -> None:
    stale = _record(
        policy=_policy(max_observation_age_seconds=1),
        inputs_stale=True,
        reason=ScalingDecisionReason.STALE_OBSERVATIONS,
    )
    assert stale.action is ScalingDecisionAction.HOLD

    with pytest.raises(ValidationError, match="must match"):
        _record(
            policy=_policy(max_observation_age_seconds=1),
            inputs_stale=True,
        )
    with pytest.raises(ValidationError, match="cannot authorize scale-down"):
        _record(
            policy=_policy(max_observation_age_seconds=1),
            inputs_stale=True,
            reason=ScalingDecisionReason.STALE_OBSERVATIONS,
            action=ScalingDecisionAction.SCALE_DOWN,
            desired_replicas=2,
            target_delta=-1,
        )


def test_staleness_is_derived_from_all_latest_source_times() -> None:
    stale_policy = _policy(max_observation_age_seconds=1)
    with pytest.raises(ValidationError, match="policy freshness limit"):
        _record(
            policy=stale_policy,
            action=ScalingDecisionAction.SCALE_DOWN,
            reason=ScalingDecisionReason.LOW_UTILIZATION,
            desired_replicas=2,
            target_delta=-1,
        )

    fresh_window = _window()
    latest = fresh_window.observations[-1]
    old_resources = latest.resources.model_copy(
        update={"observed_at": NOW - timedelta(seconds=60)}
    )
    stale_observation = latest.model_copy(update={"resources": old_resources})
    stale_window = fresh_window.model_copy(
        update={
            "observations": (fresh_window.observations[0], stale_observation),
        }
    )
    with pytest.raises(ValidationError, match="policy freshness limit"):
        _record(window=stale_window)


def test_catalog_revision_fits_postgres_bigint() -> None:
    with pytest.raises(ValidationError, match="less than or equal"):
        _record(catalog_revision=2**63)


def test_policy_revision_fits_postgres_bigint() -> None:
    with pytest.raises(ValidationError, match="less than or equal"):
        _policy(policy_revision=2**63)


def test_in_memory_log_is_idempotent_and_detects_conflicts() -> None:
    log = InMemoryScalingDecisionLog(max_records=2)
    assert isinstance(log, ScalingDecisionLog)
    record = _record()

    assert log.append(record) == record
    assert log.append(record) == record
    assert log.get(record.decision_id) == record
    with pytest.raises(ScalingDecisionConflictError):
        log.append(_record(reason=ScalingDecisionReason.HYSTERESIS))


def test_in_memory_log_lists_newest_with_filters_and_limit() -> None:
    log = InMemoryScalingDecisionLog()
    first = _record("decision-1")
    second = _record(
        "decision-2",
        decided_at=NOW + timedelta(seconds=2),
    )
    batch_policy = _policy(model_class="batch-14b")
    batch = _record(
        "decision-3",
        decided_at=NOW + timedelta(seconds=3),
        policy=batch_policy,
        window=_window(model_class="batch-14b"),
    )
    for record in (first, second, batch):
        log.append(record)

    assert log.list(limit=2) == (batch, second)
    assert log.list(model_class="interactive-14b") == (second, first)
    assert log.list(since=NOW + timedelta(seconds=2)) == (batch, second)


def test_in_memory_log_is_bounded_and_concurrent_replay_is_exact() -> None:
    log = InMemoryScalingDecisionLog(max_records=1)
    record = _record()
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = tuple(pool.map(log.append, (record,) * 32))
    assert results == (record,) * 32
    assert log.list() == (record,)
    with pytest.raises(ScalingDecisionCapacityError):
        log.append(_record("decision-2"))


def test_log_public_boundaries_reject_model_copy_bypass() -> None:
    record = _record()
    bypassed = record.model_copy(update={"desired_replicas": 51})
    log = InMemoryScalingDecisionLog()

    with pytest.raises(ValidationError):
        _ = bypassed.fingerprint
    with pytest.raises(ValidationError):
        log.append(bypassed)


@pytest.mark.parametrize("limit", [True, 0, 1001, 1.5])
def test_log_list_rejects_invalid_limits(limit: object) -> None:
    with pytest.raises(ValueError, match="limit"):
        InMemoryScalingDecisionLog().list(limit=limit)  # type: ignore[arg-type]
