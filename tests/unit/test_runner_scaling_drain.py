"""Executable contract for drain-authorized StatefulSet scale-down."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from kairyu.runners.models import (
    RunnerState,
    RunnerStatus,
    RunnerTerminationAuthorization,
)
from kairyu.runners.scaling_drain import (
    ScalingDrainCandidate,
    ScalingDrainPlan,
    ScalingDrainSnapshot,
    plan_statefulset_scale_down,
)

NOW = datetime(2026, 9, 17, 10, 0, tzinfo=UTC)


def _status(ordinal: int) -> RunnerStatus:
    authorization = RunnerTerminationAuthorization(
        runner_id=f"runner-{ordinal}",
        pod_uid=f"pod-uid-{ordinal}",
        fence_id=f"fence-{ordinal}",
        fence_sequence=ordinal + 1,
        drain_state_version=6,
        replica_generation=f"replica-generation-{ordinal}",
        dispatch_stopped_at=NOW,
        routing_excluded_at=NOW,
        activity_observed_at=NOW + timedelta(seconds=1),
        authorized_at=NOW + timedelta(seconds=1),
    )
    return RunnerStatus(
        runner_id=f"runner-{ordinal}",
        release_id="release-a",
        model_id="qwen",
        model_revision="model-revision-a",
        state=RunnerState.TERMINATING,
        state_version=7,
        state_changed_at=NOW + timedelta(seconds=1),
        observed_at=NOW + timedelta(seconds=1),
        node_name=f"gpu-node-{ordinal}",
        pod_uid=f"pod-uid-{ordinal}",
        gpu_uuids=(f"GPU-{ordinal}",),
        active_requests=0,
        termination_authorization=authorization,
    )


def _candidate(ordinal: int, *, status: RunnerStatus | None = None):
    return ScalingDrainCandidate(
        pod_name=f"qwen-runners-{ordinal}",
        workload_ordinal=ordinal,
        status=_status(ordinal) if status is None else status,
    )


def _snapshot(
    candidates: tuple[ScalingDrainCandidate, ...],
    *,
    revision: int = 7,
    observed_at: datetime = NOW + timedelta(seconds=2),
) -> ScalingDrainSnapshot:
    return ScalingDrainSnapshot(
        snapshot_id=f"drain-{revision}",
        drain_revision=revision,
        observed_at=observed_at,
        model_class="qwen-14b",
        namespace="model-serving",
        statefulset_name="qwen-runners",
        workload_uid="workload-uid",
        workload_generation=7,
        release_id="release-a",
        model_revision="model-revision-a",
        candidates=candidates,
    )


def test_plan_selects_exact_highest_statefulset_ordinals() -> None:
    plan = plan_statefulset_scale_down(
        _snapshot((_candidate(2), _candidate(3))),
        current_replicas=4,
        desired_replicas=2,
    )

    assert plan.candidate_runner_ids == ("runner-2", "runner-3")
    assert plan.candidate_pod_uids == ("pod-uid-2", "pod-uid-3")


def test_plan_requires_every_removed_ordinal_to_be_authorized() -> None:
    with pytest.raises(ValueError, match="every removed StatefulSet ordinal"):
        plan_statefulset_scale_down(
            _snapshot((_candidate(3),)),
            current_replicas=4,
            desired_replicas=2,
        )


def test_candidate_requires_termination_authorization_and_zero_activity() -> None:
    draining = RunnerStatus(
        runner_id="runner-2",
        release_id="release-a",
        model_id="qwen",
        model_revision="model-revision-a",
        state=RunnerState.DRAINING,
        state_version=6,
        state_changed_at=NOW,
        observed_at=NOW,
        pod_uid="pod-uid-2",
        active_requests=0,
    )
    with pytest.raises(ValidationError, match="termination-authorized"):
        _candidate(2, status=draining)


def test_snapshot_binds_names_release_revision_and_canonical_identity() -> None:
    with pytest.raises(ValidationError, match="canonical ordinal"):
        _snapshot((_candidate(3), _candidate(2)))
    with pytest.raises(ValidationError, match="StatefulSet ordinal"):
        ScalingDrainSnapshot.model_validate(
            _snapshot((_candidate(2),))
            .model_copy(
                update={
                    "candidates": (
                        _candidate(2).model_copy(update={"pod_name": "other-2"}),
                    )
                }
            )
            .model_dump()
        )
    changed_status = _status(2).model_copy(update={"release_id": "release-b"})
    with pytest.raises(ValidationError, match="release"):
        _snapshot((_candidate(2, status=changed_status),))


def test_plan_outputs_cannot_be_forged() -> None:
    plan = plan_statefulset_scale_down(
        _snapshot((_candidate(2), _candidate(3))),
        current_replicas=4,
        desired_replicas=2,
    )
    payload = plan.model_dump()
    payload["candidate_pod_uids"] = ("other", "pod-uid-3")

    with pytest.raises(ValidationError, match="candidate Pod UIDs"):
        ScalingDrainPlan.model_validate(payload)


@pytest.mark.parametrize(
    ("current", "desired", "message"),
    [
        (0, 0, "current_replicas"),
        (2, 2, "scale-down target"),
        (2, -1, "desired_replicas"),
    ],
)
def test_plan_rejects_invalid_replica_bounds(
    current: int,
    desired: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        plan_statefulset_scale_down(
            _snapshot(()),
            current_replicas=current,
            desired_replicas=desired,
        )
