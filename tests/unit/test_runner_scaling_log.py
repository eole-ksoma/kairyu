"""Executable contract for autoscaler observations and decision logging."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from kairyu.runners import (
    InMemoryScalingDecisionLog,
    KueueScalingAdmission,
    ModelCachePlacement,
    ModelCachePlacementState,
    RunnerStartupPhase,
    RunnerState,
    RunnerStatus,
    RunnerTerminationAuthorization,
    ScalingDecisionAction,
    ScalingDecisionCapacityError,
    ScalingDecisionConflictError,
    ScalingDecisionGenerationError,
    ScalingDecisionLog,
    ScalingDecisionReason,
    ScalingDecisionRecord,
    ScalingDecisionTargetRevision,
    ScalingDrainCandidate,
    ScalingDrainSnapshot,
    ScalingObservation,
    ScalingObservationWindow,
    ScalingPolicy,
    ScalingPrewarmSnapshot,
    ScalingQueueSnapshot,
    ScalingQuotaLimit,
    ScalingQuotaScope,
    ScalingQuotaSnapshot,
    ScalingResourceSnapshot,
    ScalingRunnerSnapshot,
    ScalingStartupPhaseMetrics,
    ScalingStartupSnapshot,
    admit_scaling_quota,
    kueue_scaling_workload_name,
    plan_cache_aware_scale_up,
    plan_statefulset_scale_down,
)
from kairyu.runners.postgres_scaling_log import PostgresScalingDecisionLog

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


def _quota_admission(
    *,
    requested: int,
    tenant_limit: int = 50,
    observed_at: datetime = NOW,
):
    snapshot = ScalingQuotaSnapshot(
        snapshot_id="quota-snapshot-1",
        quota_revision=7,
        observed_at=observed_at,
        tenant_id="tenant-a",
        model_class="interactive-14b",
        model_family="qwen",
        gpus_per_replica=1,
        target_reserved_gpus=tenant_limit,
        limits=(
            ScalingQuotaLimit(
                scope=ScalingQuotaScope.CLUSTER,
                quota_name="cluster",
                hard_limit_gpus=100,
                used_gpus_excluding_target=0,
            ),
            ScalingQuotaLimit(
                scope=ScalingQuotaScope.MODEL_FAMILY,
                quota_name="qwen",
                hard_limit_gpus=50,
                used_gpus_excluding_target=0,
            ),
            ScalingQuotaLimit(
                scope=ScalingQuotaScope.TENANT_MODEL,
                quota_name="tenant-a/interactive-14b",
                hard_limit_gpus=tenant_limit,
                used_gpus_excluding_target=0,
            ),
        ),
        kueue=KueueScalingAdmission(
            api_version="kueue.x-k8s.io/v1beta2",
            namespace="tenant-a",
            workload_name=kueue_scaling_workload_name(
                target_kind="Deployment",
                target_namespace="model-serving",
                target_name="interactive-14b-runners",
                target_uid="target-workload-uid",
            ),
            workload_uid="kueue-workload-uid",
            workload_generation=2,
            resource_version="99",
            target_kind="Deployment",
            target_namespace="model-serving",
            target_name="interactive-14b-runners",
            target_uid="target-workload-uid",
            local_queue="serving",
            cluster_queue="tenant-a-gpu",
            pod_set_name="runners",
            resource_flavor="h100-sxm",
            priority_class_name="interactive-serving",
            priority_class_group="kueue.x-k8s.io",
            priority_class_kind="WorkloadPriorityClass",
            priority=1000,
            admitted=True,
            admitted_pods=tenant_limit,
            admitted_gpus=tenant_limit,
        ),
    )
    return admit_scaling_quota(
        snapshot,
        current_replicas=3,
        requested_replicas=requested,
    )


def _prewarm_plan(
    *,
    quota_target: int,
    states: tuple[ModelCachePlacementState, ...],
    observed_at: datetime = NOW,
    resource_flavor: str = "h100-sxm",
):
    placements = tuple(
        ModelCachePlacement(
            placement_id=f"placement-{index:02d}",
            node_name=f"gpu-node-{index:02d}",
            resource_flavor=resource_flavor,
            profile_id="h100-sxm-tp1",
            compatibility_approval_id="compat-h100-qwen-v1",
            state=state,
        )
        for index, state in enumerate(states)
    )
    snapshot = ScalingPrewarmSnapshot(
        snapshot_id="prewarm-snapshot-1",
        cache_revision=3,
        observed_at=observed_at,
        model_class="interactive-14b",
        model_revision="model-revision-a",
        artifact_digest="sha256:model-artifact-a",
        placement_binding_id="binding-interactive-h100-a",
        placements=placements,
    )
    return plan_cache_aware_scale_up(
        snapshot,
        current_replicas=3,
        quota_target_replicas=quota_target,
        resource_flavor=resource_flavor,
    )


def _target_revision(**updates) -> ScalingDecisionTargetRevision:
    values = {
        "target_kind": "StatefulSet",
        "namespace": "model-serving",
        "name": "interactive-14b-runners",
        "election_id": "election-a",
        "fencing_token": 11,
        "workload_uid": "target-workload-uid",
        "workload_generation": 7,
        "release_id": "release-a",
        "model_revision": "model-revision-a",
    }
    values.update(updates)
    return ScalingDecisionTargetRevision(**values)


def _drain_plan(
    *,
    current: int = 3,
    desired: int = 2,
    observed_at: datetime = NOW,
    status_observed_at: datetime | None = None,
):
    status_observed_at = status_observed_at or observed_at
    candidates = []
    for ordinal in range(desired, current):
        authorization = RunnerTerminationAuthorization(
            runner_id=f"runner-{ordinal}",
            pod_uid=f"pod-uid-{ordinal}",
            fence_id=f"fence-{ordinal}",
            fence_sequence=ordinal + 1,
            drain_state_version=6,
            replica_generation=f"replica-generation-{ordinal}",
            dispatch_stopped_at=status_observed_at - timedelta(seconds=2),
            routing_excluded_at=status_observed_at - timedelta(seconds=2),
            activity_observed_at=status_observed_at - timedelta(seconds=1),
            authorized_at=status_observed_at - timedelta(seconds=1),
        )
        status = RunnerStatus(
            runner_id=f"runner-{ordinal}",
            release_id="release-a",
            model_id="qwen",
            model_revision="model-revision-a",
            state=RunnerState.TERMINATING,
            state_version=7,
            state_changed_at=status_observed_at - timedelta(seconds=1),
            observed_at=status_observed_at,
            pod_uid=f"pod-uid-{ordinal}",
            active_requests=0,
            termination_authorization=authorization,
        )
        candidates.append(
            ScalingDrainCandidate(
                pod_name=f"interactive-14b-runners-{ordinal}",
                workload_ordinal=ordinal,
                status=status,
            )
        )
    snapshot = ScalingDrainSnapshot(
        snapshot_id="drain-snapshot-1",
        drain_revision=7,
        observed_at=observed_at,
        model_class="interactive-14b",
        namespace="model-serving",
        statefulset_name="interactive-14b-runners",
        workload_uid="target-workload-uid",
        workload_generation=7,
        release_id="release-a",
        model_revision="model-revision-a",
        candidates=tuple(candidates),
    )
    return plan_statefulset_scale_down(
        snapshot,
        current_replicas=current,
        desired_replicas=desired,
    )


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


def test_legacy_schema_v1_fingerprint_and_postgres_row_remain_readable() -> None:
    record = _record()
    legacy_payload = record.model_dump(mode="json")
    for optional_field in (
        "decision_generation",
        "target_revision",
        "quota_admission",
        "prewarm_plan",
        "drain_plan",
    ):
        legacy_payload.pop(optional_field)
    legacy_fingerprint = hashlib.sha256(
        json.dumps(
            legacy_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    row = (
        record.decision_id,
        record.policy.model_class,
        record.decided_at,
        record.window.started_at,
        record.window.ended_at,
        record.catalog_revision,
        record.policy.policy_revision,
        record.action.value,
        legacy_fingerprint,
        legacy_payload,
    )

    assert record.fingerprint == legacy_fingerprint
    assert PostgresScalingDecisionLog._record(row) == record


def test_scale_down_persists_exact_drain_plan_and_target_binding() -> None:
    drain = _drain_plan()
    decision = _record(
        target_revision=_target_revision(),
        drain_plan=drain,
        action=ScalingDecisionAction.SCALE_DOWN,
        reason=ScalingDecisionReason.LOW_UTILIZATION,
        desired_replicas=2,
        target_delta=-1,
    )

    assert decision.drain_plan == drain
    assert decision.drain_plan.candidate_runner_ids == ("runner-2",)
    assert decision.drain_plan.snapshot.workload_uid == (
        decision.target_revision.workload_uid
    )


@pytest.mark.parametrize(
    ("record_updates", "message"),
    [
        (
            {"drain_plan": _drain_plan(current=4, desired=2)},
            "observed current replicas",
        ),
        (
            {
                "action": ScalingDecisionAction.HOLD,
                "reason": ScalingDecisionReason.NO_CHANGE,
                "desired_replicas": 3,
                "target_delta": 0,
            },
            "scale-down decisions",
        ),
        (
            {"target_revision": _target_revision(workload_uid="other-uid")},
            "decision target revision",
        ),
    ],
)
def test_drain_plan_rejects_wrong_decision_or_target_binding(
    record_updates: dict[str, object],
    message: str,
) -> None:
    values = {
        "target_revision": _target_revision(),
        "drain_plan": _drain_plan(),
        "action": ScalingDecisionAction.SCALE_DOWN,
        "reason": ScalingDecisionReason.LOW_UTILIZATION,
        "desired_replicas": 2,
        "target_delta": -1,
    }
    values.update(record_updates)

    with pytest.raises(ValidationError, match=message):
        _record(**values)


def test_stale_drain_evidence_is_a_decision_source_and_cannot_scale_down() -> None:
    stale_drain = _drain_plan(observed_at=NOW - timedelta(seconds=31))

    with pytest.raises(ValidationError, match="policy freshness limit"):
        _record(
            target_revision=_target_revision(),
            drain_plan=stale_drain,
            action=ScalingDecisionAction.SCALE_DOWN,
            reason=ScalingDecisionReason.LOW_UTILIZATION,
            desired_replicas=2,
            target_delta=-1,
        )

    with pytest.raises(ValidationError, match="cannot authorize scale-down"):
        _record(
            target_revision=_target_revision(),
            drain_plan=stale_drain,
            action=ScalingDecisionAction.SCALE_DOWN,
            reason=ScalingDecisionReason.STALE_OBSERVATIONS,
            inputs_stale=True,
            desired_replicas=2,
            target_delta=-1,
        )


def test_fresh_drain_snapshot_cannot_launder_stale_runner_evidence() -> None:
    stale_inner = _drain_plan(
        observed_at=NOW,
        status_observed_at=NOW - timedelta(seconds=31),
    )

    assert stale_inner.snapshot.observed_at == NOW
    assert stale_inner.source_observed_at == NOW - timedelta(seconds=31)
    with pytest.raises(ValidationError, match="policy freshness limit"):
        _record(
            target_revision=_target_revision(),
            drain_plan=stale_inner,
            action=ScalingDecisionAction.SCALE_DOWN,
            reason=ScalingDecisionReason.LOW_UTILIZATION,
            desired_replicas=2,
            target_delta=-1,
        )

def test_decision_persists_unconstrained_quota_and_kueue_admission() -> None:
    quota = _quota_admission(requested=6)
    decision = _record(
        action=ScalingDecisionAction.SCALE_UP,
        reason=ScalingDecisionReason.QUEUE_PRESSURE,
        quota_admission=quota,
        desired_replicas=6,
        target_delta=3,
    )

    assert decision.quota_admission == quota
    assert decision.quota_admission.snapshot.kueue.cluster_queue == "tenant-a-gpu"


def test_cache_fill_stage_holds_runner_start_and_persists_full_plan() -> None:
    quota = _quota_admission(requested=6)
    prewarm = _prewarm_plan(
        quota_target=quota.admitted_replicas,
        states=(
            ModelCachePlacementState.FILLING,
            ModelCachePlacementState.ABSENT,
            ModelCachePlacementState.ABSENT,
        ),
    )

    decision = _record(
        quota_admission=quota,
        prewarm_plan=prewarm,
        action=ScalingDecisionAction.HOLD,
        reason=ScalingDecisionReason.CACHE_PREWARM,
    )

    assert decision.desired_replicas == 3
    assert decision.prewarm_plan is not None
    assert decision.prewarm_plan.cache_fill_placement_ids == (
        "placement-01",
        "placement-02",
    )


def test_runner_start_stage_uses_only_ready_cache_capacity() -> None:
    quota = _quota_admission(requested=6)
    prewarm = _prewarm_plan(
        quota_target=quota.admitted_replicas,
        states=(
            ModelCachePlacementState.READY,
            ModelCachePlacementState.READY,
            ModelCachePlacementState.ABSENT,
        ),
    )

    decision = _record(
        quota_admission=quota,
        prewarm_plan=prewarm,
        action=ScalingDecisionAction.SCALE_UP,
        reason=ScalingDecisionReason.QUEUE_PRESSURE,
        desired_replicas=5,
        target_delta=2,
    )

    assert decision.prewarm_plan is not None
    assert decision.prewarm_plan.quota_target_replicas == 6
    assert decision.desired_replicas == 5
    assert decision.prewarm_plan.runner_start_placement_ids == (
        "placement-00",
        "placement-01",
    )


def test_prewarm_plan_requires_quota_flavor_and_current_replica_binding() -> None:
    quota = _quota_admission(requested=6)
    wrong_flavor = _prewarm_plan(
        quota_target=quota.admitted_replicas,
        states=(ModelCachePlacementState.READY,),
        resource_flavor="l40s-pcie",
    )

    with pytest.raises(ValidationError, match="Kueue resource flavor"):
        _record(
            quota_admission=quota,
            prewarm_plan=wrong_flavor,
            action=ScalingDecisionAction.SCALE_UP,
            reason=ScalingDecisionReason.QUEUE_PRESSURE,
            desired_replicas=4,
            target_delta=1,
        )


def test_stale_prewarm_evidence_cannot_hide_behind_cache_hold() -> None:
    quota = _quota_admission(requested=6)
    prewarm = _prewarm_plan(
        quota_target=quota.admitted_replicas,
        states=(ModelCachePlacementState.ABSENT,) * 3,
        observed_at=NOW - timedelta(seconds=31),
    )

    decision = _record(
        quota_admission=quota,
        prewarm_plan=prewarm,
        action=ScalingDecisionAction.HOLD,
        reason=ScalingDecisionReason.STALE_OBSERVATIONS,
        inputs_stale=True,
    )

    assert decision.inputs_stale is True


def test_budget_limit_reason_is_derived_from_quota_clamp() -> None:
    quota = _quota_admission(requested=8, tenant_limit=5)
    decision = _record(
        action=ScalingDecisionAction.SCALE_UP,
        reason=ScalingDecisionReason.BUDGET_LIMIT,
        quota_admission=quota,
        demand_replicas=5,
        buffered_target_replicas=8,
        desired_replicas=5,
        target_delta=2,
    )

    assert decision.quota_admission is not None
    assert decision.quota_admission.requested_replicas == 8
    assert decision.desired_replicas == 5
    with pytest.raises(ValidationError, match="budget-limit reason"):
        _record(
            action=ScalingDecisionAction.SCALE_UP,
            reason=ScalingDecisionReason.QUEUE_PRESSURE,
            quota_admission=quota,
            demand_replicas=5,
            buffered_target_replicas=8,
            desired_replicas=5,
            target_delta=2,
        )


def test_stale_quota_evidence_cannot_authorize_scale_up() -> None:
    quota = _quota_admission(
        requested=6,
        observed_at=NOW - timedelta(seconds=31),
    )

    with pytest.raises(ValidationError, match="stale quota evidence"):
        _record(
            action=ScalingDecisionAction.SCALE_UP,
            reason=ScalingDecisionReason.QUEUE_PRESSURE,
            quota_admission=quota,
            desired_replicas=6,
            target_delta=3,
        )


def test_stale_observation_reason_takes_precedence_over_quota_constraint() -> None:
    quota = _quota_admission(
        requested=8,
        tenant_limit=5,
        observed_at=NOW + timedelta(seconds=1),
    )

    decision = _record(
        decided_at=NOW + timedelta(seconds=31),
        action=ScalingDecisionAction.SCALE_UP,
        reason=ScalingDecisionReason.STALE_OBSERVATIONS,
        inputs_stale=True,
        quota_admission=quota,
        demand_replicas=5,
        buffered_target_replicas=8,
        desired_replicas=5,
        target_delta=2,
    )

    assert decision.quota_admission is not None
    assert decision.quota_admission.constrained is True
    assert decision.reason is ScalingDecisionReason.STALE_OBSERVATIONS


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


def test_log_allocates_mutation_generations_and_hold_does_not_consume() -> None:
    log = InMemoryScalingDecisionLog()
    first_draft = _record(
        "scale-1",
        action=ScalingDecisionAction.SCALE_UP,
        reason=ScalingDecisionReason.QUEUE_PRESSURE,
        desired_replicas=4,
        target_delta=1,
    )
    first = log.append(first_draft)
    hold = log.append(_record("hold-1"))
    second = log.append(
        _record(
            "scale-2",
            action=ScalingDecisionAction.SCALE_UP,
            reason=ScalingDecisionReason.QUEUE_PRESSURE,
            desired_replicas=4,
            target_delta=1,
        )
    )

    assert first.decision_generation == 1
    assert hold.decision_generation is None
    assert second.decision_generation == 2
    assert log.append(first_draft) == first
    assert log.append(first) == first


def test_new_decision_cannot_forge_a_durable_generation() -> None:
    log = InMemoryScalingDecisionLog()
    draft = _record(
        "scale-1",
        action=ScalingDecisionAction.SCALE_UP,
        reason=ScalingDecisionReason.QUEUE_PRESSURE,
        desired_replicas=4,
        target_delta=1,
    )
    forged = ScalingDecisionRecord.model_validate(
        draft.model_copy(update={"decision_generation": 9}).model_dump()
    )

    with pytest.raises(ScalingDecisionGenerationError, match="must not supply"):
        log.append(forged)


def test_mutation_generation_allocation_is_atomic_per_model_class() -> None:
    log = InMemoryScalingDecisionLog()
    drafts = tuple(
        _record(
            f"scale-{index}",
            action=ScalingDecisionAction.SCALE_UP,
            reason=ScalingDecisionReason.QUEUE_PRESSURE,
            desired_replicas=4,
            target_delta=1,
        )
        for index in range(1, 17)
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(log.append, drafts))

    assert {record.decision_generation for record in results} == set(range(1, 17))


def test_hold_cannot_be_labeled_with_a_mutation_generation() -> None:
    with pytest.raises(ValidationError, match="cannot consume"):
        _record(decision_generation=1)


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
