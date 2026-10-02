"""D3.1 gates for binding cache pins to exact Runner scheduler placements."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from kairyu.artifacts import (
    NodeModelCacheFillResult,
    NodeModelCachePlacementHintSnapshot,
    NodeModelCacheResidentHint,
)
from kairyu.runners import (
    ModelCachePlacement,
    ModelCachePlacementState,
    NodeModelPrestageRecord,
    RunnerCacheStartupBinding,
    RunnerCacheStartupBindingError,
    RunnerWriterAuthority,
    ScalingPrewarmSnapshot,
    build_runner_cache_startup_binding,
    build_runner_start_prestage_commands,
    plan_cache_aware_scale_up,
)

NOW = datetime(2026, 9, 29, 9, 0, tzinfo=UTC)
DIGEST = "a" * 64
DECISION_FINGERPRINT = "b" * 64


def _authority() -> RunnerWriterAuthority:
    return RunnerWriterAuthority(
        election_id="runner-controller",
        holder_id="controller-a",
        fencing_token=7,
        validated_at=NOW,
        lease_until=NOW + timedelta(minutes=5),
    )


def _hint(
    *,
    observed_at: datetime,
    pinned: bool = True,
    record_generation: int = 4,
):
    return NodeModelCachePlacementHintSnapshot(
        node_id="gpu-node-a",
        index_revision=11,
        observed_at=observed_at,
        valid_until=observed_at + timedelta(minutes=1),
        residents=(
            NodeModelCacheResidentHint(
                node_id="gpu-node-a",
                manifest_digest=DIGEST,
                model_id="org/model-a",
                model_revision="model-revision-a",
                total_bytes=1024,
                file_count=1,
                verified_at_ns=1,
                last_access_at_ns=2,
                pinned=pinned,
                record_generation=record_generation,
            ),
        ),
    )


def _plan(hint: NodeModelCachePlacementHintSnapshot):
    snapshot = ScalingPrewarmSnapshot(
        snapshot_id=f"cache-{hint.index_revision}-{hint.observed_at.second}",
        cache_revision=hint.index_revision,
        observed_at=hint.observed_at,
        model_class="qwen-14b",
        model_revision="model-revision-a",
        artifact_digest=DIGEST,
        placement_binding_id="binding-qwen-a",
        placements=(
            ModelCachePlacement(
                placement_id="placement-a",
                node_name="gpu-node-a",
                resource_flavor="rtx-pro-6000-blackwell",
                profile_id="rtx-pro-6000-blackwell-tp1",
                compatibility_approval_id="compat-qwen-a",
                state=ModelCachePlacementState.READY,
                cache_hint_observed_at=hint.observed_at,
                cache_hint_valid_until=hint.valid_until,
                cache_hint_index_revision=hint.index_revision,
            ),
        ),
    )
    return plan_cache_aware_scale_up(
        snapshot,
        current_replicas=0,
        quota_target_replicas=1,
        resource_flavor="rtx-pro-6000-blackwell",
    )


def _command(plan, *, decision_id: str = "decision-a"):
    return build_runner_start_prestage_commands(
        plan,
        authority=_authority(),
        decision_id=decision_id,
        decision_fingerprint=DECISION_FINGERPRINT,
        target_id="statefulset/model-serving/qwen-runners",
        target_revision=3,
        deployment_id="model-serving/qwen",
        model_id="org/model-a",
        command_generations={"placement-a": 1},
        issued_at=NOW + timedelta(seconds=11),
        ttl_seconds=60,
    )[0]


def _record(command) -> NodeModelPrestageRecord:
    return NodeModelPrestageRecord(
        command=command,
        state=ModelCachePlacementState.READY,
        attempt=1,
        fill_result=NodeModelCacheFillResult(
            deployment_id="model-serving/qwen",
            manifest_digest=DIGEST,
            artifact_path=Path(f"/mnt/nvme/kairyu-model-cache/artifacts/{DIGEST}"),
            cache_hit=True,
            resumed_bytes=0,
            downloaded_bytes=0,
            file_count=1,
            total_bytes=1024,
        ),
        pin_record_generation=4,
        updated_at=NOW + timedelta(seconds=12),
    )


def _binding(*, pinned: bool = True, decision_id: str = "decision-a"):
    initial = _plan(_hint(observed_at=NOW + timedelta(seconds=10)))
    command = _command(initial, decision_id=decision_id)
    published_hint = _hint(
        observed_at=NOW + timedelta(seconds=13),
        pinned=pinned,
    )
    final = _plan(published_hint)
    binding = build_runner_cache_startup_binding(
        final,
        (_record(command),),
        (published_hint,),
        decision_id="decision-a",
        decision_fingerprint=DECISION_FINGERPRINT,
        target_id="statefulset/model-serving/qwen-runners",
        target_revision=3,
        deployment_id="model-serving/qwen",
        model_id="org/model-a",
        bound_at=NOW + timedelta(seconds=14),
    )
    return binding, command


def test_runner_start_commands_pin_ready_placements_without_refill_intent() -> None:
    plan = _plan(_hint(observed_at=NOW + timedelta(seconds=10)))

    commands = build_runner_start_prestage_commands(
        plan,
        authority=_authority(),
        decision_id="decision-a",
        decision_fingerprint=DECISION_FINGERPRINT,
        target_id="statefulset/model-serving/qwen-runners",
        target_revision=3,
        deployment_id="model-serving/qwen",
        model_id="org/model-a",
        command_generations={"placement-a": 1},
        issued_at=NOW + timedelta(seconds=11),
        ttl_seconds=60,
    )

    assert len(commands) == 1
    assert commands[0].placement_id == plan.runner_start_placement_ids[0]
    assert commands[0].pin_owner == "prestage/model-serving/qwen/placement-a/1"
    assert plan.cache_fill_placement_ids == ()


def test_binding_joins_decision_pin_hint_node_and_artifact_identity() -> None:
    binding, command = _binding()

    assert binding.placement_binding_id == "binding-qwen-a"
    assert binding.manifest_digest == DIGEST
    assert binding.valid_until == NOW + timedelta(seconds=73)
    assert binding.placements[0].node_name == "gpu-node-a"
    assert binding.placements[0].pin_owner == command.pin_owner
    assert binding.placements[0].prestage_command_id == command.command_id
    assert RunnerCacheStartupBinding.model_validate(binding.model_dump()) == binding


def test_binding_rejects_unpinned_physical_residency() -> None:
    with pytest.raises(RunnerCacheStartupBindingError, match="owned pin generation"):
        _binding(pinned=False)


def test_binding_rejects_other_owner_after_deployment_pin_release() -> None:
    initial = _plan(_hint(observed_at=NOW + timedelta(seconds=10)))
    command = _command(initial)
    other_owner_hint = _hint(
        observed_at=NOW + timedelta(seconds=13),
        pinned=True,
        record_generation=5,
    )

    with pytest.raises(RunnerCacheStartupBindingError, match="owned pin generation"):
        build_runner_cache_startup_binding(
            _plan(other_owner_hint),
            (_record(command),),
            (other_owner_hint,),
            decision_id="decision-a",
            decision_fingerprint=DECISION_FINGERPRINT,
            target_id="statefulset/model-serving/qwen-runners",
            target_revision=3,
            deployment_id="model-serving/qwen",
            model_id="org/model-a",
            bound_at=NOW + timedelta(seconds=14),
        )


def test_binding_rejects_pin_from_another_scaling_decision() -> None:
    with pytest.raises(RunnerCacheStartupBindingError, match="scheduling decision"):
        _binding(decision_id="decision-other")


def test_binding_rejects_hint_that_predates_pin_completion() -> None:
    initial = _plan(_hint(observed_at=NOW + timedelta(seconds=10)))
    command = _command(initial)
    stale_hint = _hint(observed_at=NOW + timedelta(seconds=11))

    with pytest.raises(RunnerCacheStartupBindingError, match="publish the completed pin"):
        build_runner_cache_startup_binding(
            _plan(stale_hint),
            (_record(command),),
            (stale_hint,),
            decision_id="decision-a",
            decision_fingerprint=DECISION_FINGERPRINT,
            target_id="statefulset/model-serving/qwen-runners",
            target_revision=3,
            deployment_id="model-serving/qwen",
            model_id="org/model-a",
            bound_at=NOW + timedelta(seconds=14),
        )


def test_binding_id_detects_serialized_contract_tampering() -> None:
    binding, _ = _binding()
    payload = binding.model_dump()
    payload["target_revision"] = 4

    with pytest.raises(ValidationError, match="binding_id"):
        RunnerCacheStartupBinding.model_validate(payload)


def test_runner_start_command_generations_must_exactly_cover_ready_slots() -> None:
    plan = _plan(_hint(observed_at=NOW + timedelta(seconds=10)))

    with pytest.raises(ValueError, match="Runner-start placements"):
        build_runner_start_prestage_commands(
            plan,
            authority=_authority(),
            decision_id="decision-a",
            decision_fingerprint=DECISION_FINGERPRINT,
            target_id="statefulset/model-serving/qwen-runners",
            target_revision=3,
            deployment_id="model-serving/qwen",
            model_id="org/model-a",
            command_generations={},
            issued_at=NOW + timedelta(seconds=11),
            ttl_seconds=60,
        )
