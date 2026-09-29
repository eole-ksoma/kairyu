"""D3.3 cache-bound Runner startup attestation gates."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from kairyu.artifacts import NodeModelCacheRunnerStartDecision
from kairyu.runners import (
    RUNNER_STARTUP_PHASES,
    InvalidRunnerObservationError,
    KubernetesPodPhase,
    RunnerCacheStartupAttestationError,
    RunnerCacheStartupBinding,
    RunnerCacheStartupPlacement,
    RunnerObservation,
    RunnerPodObservation,
    RunnerRuntimeObservation,
    RunnerStartupPhase,
    RunnerStartupReport,
    RunnerState,
    build_runner_cache_startup_proof,
    complete_startup_phase,
    reconcile_runner_status,
    skip_startup_phase,
    start_startup_phase,
    validate_runner_cache_startup_proof,
)

NOW = datetime(2026, 9, 29, 11, 0, tzinfo=UTC)
DIGEST = "a" * 64


def _binding() -> RunnerCacheStartupBinding:
    placement = RunnerCacheStartupPlacement(
        placement_id="placement-a",
        node_name="gpu-node-a",
        resource_flavor="h100-sxm",
        profile_id="h100-sxm-tp1",
        compatibility_approval_id="compat-qwen-h100",
        manifest_digest=DIGEST,
        pin_owner="prestage/model-serving/qwen/placement-a",
        prestage_command_id=hashlib.sha256(b"command-a").hexdigest(),
        prestage_command_generation=3,
        hint_index_revision=11,
        resident_record_generation=17,
        hint_observed_at=NOW,
        hint_valid_until=NOW + timedelta(minutes=5),
    )
    payload = {
        "schema_version": "runner-cache-startup-binding-v1",
        "decision_id": "decision-a",
        "decision_fingerprint": hashlib.sha256(b"decision-a").hexdigest(),
        "target_id": "deployment/model-serving/qwen-runners",
        "target_revision": 7,
        "deployment_id": "model-serving/qwen",
        "model_class": "qwen-14b",
        "model_id": "org/qwen",
        "model_revision": "revision-a",
        "manifest_digest": DIGEST,
        "placement_binding_id": "placement-binding-a",
        "prewarm_snapshot_id": "snapshot-a",
        "prewarm_cache_revision": 9,
        "bound_at": NOW + timedelta(seconds=1),
        "valid_until": NOW + timedelta(minutes=5),
        "placements": (placement,),
    }
    unsigned = RunnerCacheStartupBinding.model_construct(binding_id="0" * 64, **payload)
    encoded = json.dumps(
        unsigned.model_dump(mode="json", exclude={"binding_id"}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return RunnerCacheStartupBinding(
        binding_id=hashlib.sha256(encoded).hexdigest(),
        **payload,
    )


def _allowed_decision() -> NodeModelCacheRunnerStartDecision:
    return NodeModelCacheRunnerStartDecision(
        runner_start_allowed=True,
        manifest_digest=DIGEST,
        artifact_path=Path(f"/mnt/cache/artifacts/{DIGEST}/tree"),
        corruption_detected=False,
        refetched=False,
        reason="verified",
    )


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


def _observation(*, include_proof: bool) -> RunnerObservation:
    binding = _binding()
    observed_at = NOW + timedelta(seconds=30)
    proof = (
        build_runner_cache_startup_proof(
            binding,
            _allowed_decision(),
            runner_id="pod-uid-a",
            node_name="gpu-node-a",
            verified_at=NOW + timedelta(seconds=20),
        )
        if include_proof
        else None
    )
    return RunnerObservation(
        runner_id="pod-uid-a",
        release_id="release-a",
        model_id="org/qwen",
        model_revision="revision-a",
        observed_at=observed_at,
        pod=RunnerPodObservation(
            uid="pod-uid-a",
            phase=KubernetesPodPhase.RUNNING,
            node_name="gpu-node-a",
            ready=True,
            gpu_uuids=("GPU-a",),
        ),
        cache_startup_binding=binding,
        endpoint_ready=True,
        runtime=RunnerRuntimeObservation(
            runner_id="pod-uid-a",
            observed_at=observed_at,
            ready=True,
            active_requests=0,
            startup=_complete_startup(),
            cache_startup_proof=proof,
        ),
    )


def test_full_digest_decision_is_bound_to_exact_pod_node_and_generation() -> None:
    binding = _binding()
    verified_at = NOW + timedelta(seconds=20)

    proof = build_runner_cache_startup_proof(
        binding,
        _allowed_decision(),
        runner_id="pod-uid-a",
        node_name="gpu-node-a",
        verified_at=verified_at,
    )

    placement = binding.placements[0]
    assert proof.binding_id == binding.binding_id
    assert proof.node_name == placement.node_name
    assert proof.prestage_command_id == placement.prestage_command_id
    assert proof.resident_record_generation == placement.resident_record_generation
    assert "artifact_path" not in proof.model_dump()
    validate_runner_cache_startup_proof(
        binding,
        proof,
        runner_id="pod-uid-a",
        node_name="gpu-node-a",
        model_id="org/qwen",
        model_revision="revision-a",
        observed_at=verified_at,
    )


def test_proof_builder_rejects_unselected_node_and_denied_verification() -> None:
    binding = _binding()
    with pytest.raises(RunnerCacheStartupAttestationError, match="scheduled node"):
        build_runner_cache_startup_proof(
            binding,
            _allowed_decision(),
            runner_id="pod-uid-a",
            node_name="gpu-node-b",
            verified_at=NOW + timedelta(seconds=20),
        )

    denied = NodeModelCacheRunnerStartDecision(
        runner_start_allowed=False,
        manifest_digest=DIGEST,
        corruption_detected=True,
        refetched=True,
        quarantine_path=Path("/mnt/cache/quarantine/a"),
        reason="digest_mismatch",
    )
    with pytest.raises(RunnerCacheStartupAttestationError, match="full-digest"):
        build_runner_cache_startup_proof(
            binding,
            denied,
            runner_id="pod-uid-a",
            node_name="gpu-node-a",
            verified_at=NOW + timedelta(seconds=20),
        )


def test_cross_clock_order_uses_the_reconciler_skew_policy() -> None:
    binding = _binding()
    proof = build_runner_cache_startup_proof(
        binding,
        _allowed_decision(),
        runner_id="pod-uid-a",
        node_name="gpu-node-a",
        verified_at=NOW,
    )
    validate_runner_cache_startup_proof(
        binding,
        proof,
        runner_id="pod-uid-a",
        node_name="gpu-node-a",
        model_id="org/qwen",
        model_revision="revision-a",
        observed_at=NOW + timedelta(seconds=30),
    )

    base = _observation(include_proof=False).model_dump()
    base["runtime"]["cache_startup_proof"] = proof.model_dump()
    observation = RunnerObservation.model_validate(base)
    assert reconcile_runner_status(None, observation).state is RunnerState.READY
    assert (
        reconcile_runner_status(
            None,
            observation,
            runtime_clock_skew_tolerance=timedelta(0),
        ).state
        is RunnerState.WARMING
    )

    with pytest.raises(RunnerCacheStartupAttestationError, match="clock skew"):
        validate_runner_cache_startup_proof(
            binding,
            proof,
            runner_id="pod-uid-a",
            node_name="gpu-node-a",
            model_id="org/qwen",
            model_revision="revision-a",
            observed_at=NOW + timedelta(seconds=30),
            clock_skew_tolerance=timedelta(0),
        )
    with pytest.raises(RunnerCacheStartupAttestationError, match="clock skew"):
        build_runner_cache_startup_proof(
            binding,
            _allowed_decision(),
            runner_id="pod-uid-a",
            node_name="gpu-node-a",
            verified_at=NOW - timedelta(seconds=2),
        )


def test_proof_hash_and_observation_identity_fail_closed_on_tamper() -> None:
    binding = _binding()
    proof = build_runner_cache_startup_proof(
        binding,
        _allowed_decision(),
        runner_id="pod-uid-a",
        node_name="gpu-node-a",
        verified_at=NOW + timedelta(seconds=20),
    )
    tampered = proof.model_dump()
    tampered["resident_record_generation"] += 1
    with pytest.raises(ValidationError, match="proof_id"):
        type(proof).model_validate(tampered)

    wrong_runner_proof = build_runner_cache_startup_proof(
        binding,
        _allowed_decision(),
        runner_id="pod-uid-b",
        node_name="gpu-node-a",
        verified_at=NOW + timedelta(seconds=20),
    )
    observation = _observation(include_proof=False).model_dump()
    observation["runtime"]["cache_startup_proof"] = wrong_runner_proof.model_dump()
    with pytest.raises(ValidationError, match="does not match the observed Pod"):
        RunnerObservation.model_validate(observation)


def test_cache_bound_runner_stays_warming_until_exact_proof_arrives() -> None:
    missing = reconcile_runner_status(None, _observation(include_proof=False))
    attested = reconcile_runner_status(None, _observation(include_proof=True))

    assert missing.state is RunnerState.WARMING
    assert attested.state is RunnerState.READY


def test_observed_pod_cannot_downgrade_by_removing_its_binding() -> None:
    initial = _observation(include_proof=True)
    ready = reconcile_runner_status(None, initial)
    downgraded = initial.model_copy(update={"cache_startup_binding": None})
    assert ready.cache_startup_binding_id == initial.cache_startup_binding.binding_id

    with pytest.raises(InvalidRunnerObservationError, match="cannot disappear"):
        reconcile_runner_status(ready, downgraded)


def test_incremental_pod_cannot_remove_its_selected_placement() -> None:
    initial = _observation(include_proof=True).model_copy(
        update={"cache_startup_placement_id": "placement-a"}
    )
    initial = RunnerObservation.model_validate(initial.model_dump())
    ready = reconcile_runner_status(None, initial)
    downgraded = initial.model_copy(update={"cache_startup_placement_id": None})
    assert ready.cache_startup_placement_id == "placement-a"

    with pytest.raises(InvalidRunnerObservationError, match="cannot disappear"):
        reconcile_runner_status(ready, downgraded)


def test_legacy_runner_without_binding_keeps_existing_readiness_contract() -> None:
    observation = _observation(include_proof=False).model_copy(
        update={"cache_startup_binding": None}
    )

    status = reconcile_runner_status(None, observation)

    assert status.state is RunnerState.READY
