"""Executable contract for cache-aware staged Runner scale-out."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from kairyu.runners.prewarm import (
    ModelCachePlacement,
    ModelCachePlacementState,
    ScalingPrewarmAction,
    ScalingPrewarmPlan,
    ScalingPrewarmSnapshot,
    plan_cache_aware_scale_up,
)

NOW = datetime(2026, 9, 17, 9, 0, tzinfo=UTC)


def _placement(
    suffix: str,
    state: ModelCachePlacementState,
    **updates,
) -> ModelCachePlacement:
    values = {
        "placement_id": f"placement-{suffix}",
        "node_name": f"gpu-node-{suffix}",
        "resource_flavor": "h100-sxm",
        "profile_id": "h100-sxm-tp1",
        "compatibility_approval_id": "compat-h100-qwen-v1",
        "state": state,
    }
    values.update(updates)
    return ModelCachePlacement(**values)


def _snapshot(
    placements: tuple[ModelCachePlacement, ...],
    *,
    revision: int = 7,
) -> ScalingPrewarmSnapshot:
    return ScalingPrewarmSnapshot(
        snapshot_id=f"cache-{revision}",
        cache_revision=revision,
        observed_at=NOW,
        model_class="qwen-14b",
        model_revision="qwen-revision-a",
        artifact_digest="sha256:model-artifact-a",
        placement_binding_id="binding-qwen-h100-a",
        placements=placements,
    )


def test_plan_starts_only_ready_slots_and_prefetches_absent_slots() -> None:
    snapshot = _snapshot(
        (
            _placement("a", ModelCachePlacementState.READY),
            _placement("b", ModelCachePlacementState.FILLING),
            _placement("c", ModelCachePlacementState.ABSENT),
            _placement("d", ModelCachePlacementState.ABSENT),
        )
    )

    plan = plan_cache_aware_scale_up(
        snapshot,
        current_replicas=2,
        quota_target_replicas=6,
        resource_flavor="h100-sxm",
    )

    assert plan.action is ScalingPrewarmAction.RUNNER_START
    assert plan.runner_target_replicas == 3
    assert plan.runner_start_placement_ids == ("placement-a",)
    assert plan.pending_fill_placement_ids == ("placement-b",)
    assert plan.cache_fill_placement_ids == ("placement-c", "placement-d")
    assert plan.unplanned_replicas == 0
    assert plan.cache_ready_for_quota_target is False


def test_plan_holds_runner_scale_while_cache_fill_is_required() -> None:
    plan = plan_cache_aware_scale_up(
        _snapshot(
            (
                _placement("a", ModelCachePlacementState.FILLING),
                _placement("b", ModelCachePlacementState.ABSENT),
            )
        ),
        current_replicas=0,
        quota_target_replicas=2,
        resource_flavor="h100-sxm",
    )

    assert plan.action is ScalingPrewarmAction.CACHE_FILL
    assert plan.runner_target_replicas == 0
    assert plan.pending_fill_placement_ids == ("placement-a",)
    assert plan.cache_fill_placement_ids == ("placement-b",)


def test_plan_excludes_wrong_flavor_assigned_unhealthy_and_failed_slots() -> None:
    plan = plan_cache_aware_scale_up(
        _snapshot(
            (
                _placement("a", ModelCachePlacementState.READY, assigned=True),
                _placement("b", ModelCachePlacementState.READY, healthy=False),
                _placement(
                    "c",
                    ModelCachePlacementState.READY,
                    resource_flavor="l40s-pcie",
                ),
                _placement("d", ModelCachePlacementState.FAILED),
                _placement("e", ModelCachePlacementState.READY),
            )
        ),
        current_replicas=2,
        quota_target_replicas=5,
        resource_flavor="h100-sxm",
    )

    assert plan.runner_start_placement_ids == ("placement-e",)
    assert plan.runner_target_replicas == 3
    assert plan.unplanned_replicas == 2


def test_all_ready_slots_allow_full_quota_target() -> None:
    plan = plan_cache_aware_scale_up(
        _snapshot(
            tuple(
                _placement(suffix, ModelCachePlacementState.READY)
                for suffix in ("a", "b", "c")
            )
        ),
        current_replicas=2,
        quota_target_replicas=5,
        resource_flavor="h100-sxm",
    )

    assert plan.runner_target_replicas == 5
    assert plan.cache_ready_for_quota_target is True
    assert plan.cache_fill_placement_ids == ()


def test_snapshot_requires_canonical_unique_placements() -> None:
    with pytest.raises(ValidationError, match="canonical"):
        _snapshot(
            (
                _placement("b", ModelCachePlacementState.READY),
                _placement("a", ModelCachePlacementState.READY),
            )
        )
    with pytest.raises(ValidationError, match="unique"):
        _snapshot(
            (
                _placement("a", ModelCachePlacementState.READY),
                _placement("a", ModelCachePlacementState.ABSENT),
            )
        )


def test_compatibility_approval_and_boolean_fields_fail_closed() -> None:
    with pytest.raises(ValidationError, match="compatibility_approval_id"):
        _placement(
            "a",
            ModelCachePlacementState.READY,
            compatibility_approval_id="",
        )
    with pytest.raises(ValidationError, match="healthy must be a boolean"):
        _placement("a", ModelCachePlacementState.READY, healthy=1)


def test_plan_outputs_cannot_be_forged() -> None:
    plan = plan_cache_aware_scale_up(
        _snapshot((_placement("a", ModelCachePlacementState.READY),)),
        current_replicas=1,
        quota_target_replicas=2,
        resource_flavor="h100-sxm",
    )
    payload = plan.model_dump()
    payload["runner_start_placement_ids"] = ()

    with pytest.raises(ValidationError, match="runner-start placements"):
        ScalingPrewarmPlan.model_validate(payload)


@pytest.mark.parametrize(
    ("current", "target", "message"),
    [(-1, 2, "current_replicas"), (1, 1, "scale-up target"), (1, 100_001, "quota")],
)
def test_plan_rejects_invalid_replica_bounds(
    current: int,
    target: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        plan_cache_aware_scale_up(
            _snapshot(()),
            current_replicas=current,
            quota_target_replicas=target,
            resource_flavor="h100-sxm",
        )
