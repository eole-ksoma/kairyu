"""WP4.7 fenced model pre-staging coverage."""

from __future__ import annotations

import base64
import hashlib
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from kairyu.artifacts import (
    ModelArtifactAdmissionRequest,
    ModelArtifactBlob,
    ModelArtifactDownloadError,
    ModelArtifactManifest,
    ModelArtifactQuantization,
    ModelArtifactResourceEstimate,
    ModelArtifactTokenizer,
    ModelArtifactTrustStore,
    NodeModelCacheAgent,
    NodeModelCacheIndex,
    NodeModelCachePlacementHintPublisher,
    SignedModelArtifactManifest,
    TrustedModelSigner,
    model_file_tree_digest,
    sign_model_artifact_manifest,
)
from kairyu.runners import (
    InMemoryNodeModelPrestageStore,
    ModelCachePlacement,
    ModelCachePlacementCandidate,
    ModelCachePlacementState,
    NodeModelPrestageConflictError,
    NodeModelPrestageExecutor,
    NodeModelPrestageExpiredError,
    RunnerWriterAuthority,
    ScalingPrewarmSnapshot,
    apply_node_model_prestage_records,
    build_cache_placement_snapshot,
    build_node_model_prestage_commands,
    build_node_model_prestage_release_command,
    plan_cache_aware_scale_up,
)

_NOW = datetime(2026, 9, 28, 3, 0, tzinfo=UTC)
_DATA = {
    "LICENSE": b"Apache-2.0\n",
    "weights/model.bin": b"prestage-model-weights",
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class Source:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.failure = failure

    def iter_blob(
        self,
        *,
        manifest_digest: str,
        blob: ModelArtifactBlob,
        offset: int,
        chunk_size: int,
    ) -> Iterable[bytes]:
        del manifest_digest, chunk_size
        self.calls.append((blob.path, offset))
        if self.failure is not None:
            raise self.failure
        yield _DATA[blob.path][offset:]


def _artifact() -> tuple[
    SignedModelArtifactManifest,
    ModelArtifactTrustStore,
    ModelArtifactAdmissionRequest,
]:
    blobs = tuple(
        ModelArtifactBlob(path=path, size_bytes=len(content), sha256=_sha256(content))
        for path, content in sorted(_DATA.items())
    )
    manifest = ModelArtifactManifest(
        model_id="org/prestage-test",
        model_revision="release-1",
        upstream_repository="https://example.invalid/org/prestage-test",
        upstream_revision="1" * 40,
        architecture="PrestageTestForCausalLM",
        quantization=ModelArtifactQuantization(method="none", format="safetensors"),
        tokenizer=ModelArtifactTokenizer(
            repository="https://example.invalid/org/prestage-test",
            revision="2" * 40,
            sha256="3" * 64,
        ),
        license_id="Apache-2.0",
        license_files=("LICENSE",),
        required_gpu_profiles=("h100-sxm",),
        approved_environments=("production",),
        signer_key_id="release-key",
        resources=ModelArtifactResourceEstimate(
            disk_bytes=sum(map(len, _DATA.values())),
            ram_bytes=1024,
            vram_bytes=2048,
        ),
        files=blobs,
        file_tree_sha256=model_file_tree_digest(blobs),
    )
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    envelope = sign_model_artifact_manifest(manifest, private_key)
    trust_store = ModelArtifactTrustStore(
        signers=(
            TrustedModelSigner(
                key_id="release-key",
                public_key_base64=base64.b64encode(public_key).decode("ascii"),
                approved_environments=("production",),
            ),
        )
    )
    request = ModelArtifactAdmissionRequest(
        deployment_id="production/prestage-test",
        manifest_digest=envelope.manifest_digest,
        model_id=manifest.model_id,
        model_revision=manifest.model_revision,
        environment="production",
        gpu_profile="h100-sxm",
    )
    return envelope, trust_store, request


def _authority(*, token: int = 4, holder_id: str = "controller-a") -> RunnerWriterAuthority:
    return RunnerWriterAuthority(
        election_id="runner-controller/production",
        holder_id=holder_id,
        fencing_token=token,
        validated_at=_NOW - timedelta(seconds=1),
        lease_until=_NOW + timedelta(minutes=2),
    )


def _plan(digest: str, *, count: int = 1):
    snapshot = ScalingPrewarmSnapshot(
        snapshot_id="cache-snapshot-10",
        cache_revision=10,
        observed_at=_NOW,
        model_class="interactive-h100",
        model_revision="release-1",
        artifact_digest=digest,
        placement_binding_id="binding-prestage-h100",
        placements=tuple(
            ModelCachePlacement(
                placement_id=f"placement-{index:02d}",
                node_name=f"gpu-node-{index:02d}",
                resource_flavor="h100-sxm",
                profile_id="h100-sxm",
                compatibility_approval_id="compat-h100-v1",
                state=ModelCachePlacementState.ABSENT,
            )
            for index in range(count)
        ),
    )
    return plan_cache_aware_scale_up(
        snapshot,
        current_replicas=0,
        quota_target_replicas=count,
        resource_flavor="h100-sxm",
    )


def _commands(
    digest: str,
    *,
    count: int = 1,
    token: int = 4,
    generation_start: int = 20,
    target_revision: int = 8,
):
    return build_node_model_prestage_commands(
        _plan(digest, count=count),
        authority=_authority(token=token),
        decision_id="scale-decision-17",
        decision_fingerprint="d" * 64,
        target_id="statefulset/production/prestage-test",
        target_revision=target_revision,
        deployment_id="production/prestage-test",
        model_id="org/prestage-test",
        command_generations={
            f"placement-{index:02d}": generation_start + index for index in range(count)
        },
        issued_at=_NOW,
        ttl_seconds=60,
    )


def _executor(tmp_path: Path, source: Source):
    root = tmp_path / "cache"
    index = NodeModelCacheIndex(root / "cache-index.sqlite3", node_id="gpu-node-00")
    store = InMemoryNodeModelPrestageStore(node_id="gpu-node-00")
    agent = NodeModelCacheAgent(root, source, index=index, chunk_size_bytes=4)
    executor = NodeModelPrestageExecutor(
        node_id="gpu-node-00",
        agent=agent,
        index=index,
        store=store,
        clock=lambda: _NOW + timedelta(seconds=2),
    )
    return executor, index, store


def test_builds_exact_deterministic_command_per_absent_placement() -> None:
    envelope, _, _ = _artifact()

    first = _commands(envelope.manifest_digest, count=2)
    second = _commands(envelope.manifest_digest, count=2)

    assert first == second
    assert tuple(command.placement_id for command in first) == (
        "placement-00",
        "placement-01",
    )
    assert tuple(command.node_id for command in first) == ("gpu-node-00", "gpu-node-01")
    assert first[0].command_generation == 20
    assert first[0].authority.fencing_token == 4
    assert first[0].pin_owner == "prestage/production/prestage-test/placement-00"


def test_command_builder_requires_exact_generation_allocation() -> None:
    envelope, _, _ = _artifact()
    plan = _plan(envelope.manifest_digest)

    with pytest.raises(ValueError, match="exactly cover"):
        build_node_model_prestage_commands(
            plan,
            authority=_authority(),
            decision_id="decision",
            decision_fingerprint="d" * 64,
            target_id="target",
            target_revision=1,
            deployment_id="production/prestage-test",
            model_id="org/prestage-test",
            command_generations={},
            issued_at=_NOW,
            ttl_seconds=60,
        )


def test_canonical_command_id_rejects_payload_forgery() -> None:
    envelope, _, _ = _artifact()
    command = _commands(envelope.manifest_digest)[0]
    payload = command.model_dump()
    payload["target_revision"] = 9

    with pytest.raises(ValidationError, match="canonical command payload"):
        type(command).model_validate(payload)


def test_execute_fills_pins_completes_and_replays_without_download(tmp_path: Path) -> None:
    envelope, trust_store, request = _artifact()
    source = Source()
    executor, index, store = _executor(tmp_path, source)
    command = _commands(envelope.manifest_digest)[0]

    ready = executor.execute(
        command,
        claim_id="a" * 64,
        envelope=envelope,
        trust_store=trust_store,
        request=request,
    )
    calls = tuple(source.calls)
    replay = executor.execute(
        command,
        claim_id="b" * 64,
        envelope=envelope,
        trust_store=trust_store,
        request=request,
    )

    assert ready.state is ModelCachePlacementState.READY
    assert ready.fill_result is not None and not ready.fill_result.cache_hit
    assert replay == ready
    assert tuple(source.calls) == calls
    assert command.pin_owner in index.get(command.manifest_digest).pin_owners
    assert store.list_records() == (ready,)


def test_exact_claim_can_complete_after_command_expiry(tmp_path: Path) -> None:
    envelope, trust_store, request = _artifact()
    root = tmp_path / "cache"
    index = NodeModelCacheIndex(root / "cache-index.sqlite3", node_id="gpu-node-00")
    store = InMemoryNodeModelPrestageStore(node_id="gpu-node-00")
    clock_values = iter((_NOW + timedelta(seconds=2), _NOW + timedelta(seconds=61)))
    executor = NodeModelPrestageExecutor(
        node_id="gpu-node-00",
        agent=NodeModelCacheAgent(root, Source(), index=index, chunk_size_bytes=4),
        index=index,
        store=store,
        clock=lambda: next(clock_values),
    )

    command = _commands(envelope.manifest_digest)[0]
    ready = executor.execute(
        command,
        claim_id="a" * 64,
        envelope=envelope,
        trust_store=trust_store,
        request=request,
    )

    assert ready.state is ModelCachePlacementState.READY
    assert (
        store.claim(
            command,
            claim_id="b" * 64,
            now=_NOW + timedelta(minutes=5),
        )
        == ready
    )


def test_execute_rejects_admission_binding_before_claim(tmp_path: Path) -> None:
    envelope, trust_store, request = _artifact()
    executor, _, store = _executor(tmp_path, Source())
    command = _commands(envelope.manifest_digest)[0]
    wrong = request.model_copy(update={"deployment_id": "production/other"})

    with pytest.raises(NodeModelPrestageConflictError, match="admission request"):
        executor.execute(
            command,
            claim_id="a" * 64,
            envelope=envelope,
            trust_store=trust_store,
            request=wrong,
        )

    assert store.list_records() == ()


def test_execute_rejects_fill_path_that_differs_from_pinned_index(tmp_path: Path) -> None:
    envelope, trust_store, request = _artifact()
    root = tmp_path / "cache"
    index = NodeModelCacheIndex(root / "cache-index.sqlite3", node_id="gpu-node-00")
    real_agent = NodeModelCacheAgent(root, Source(), index=index, chunk_size_bytes=4)

    class WrongPathAgent:
        def ensure_cached(self, *args, **kwargs):
            result = real_agent.ensure_cached(*args, **kwargs)
            return result.model_copy(update={"artifact_path": root / "wrong/tree"})

    store = InMemoryNodeModelPrestageStore(node_id="gpu-node-00")
    executor = NodeModelPrestageExecutor(
        node_id="gpu-node-00",
        agent=WrongPathAgent(),
        index=index,
        store=store,
        clock=lambda: _NOW + timedelta(seconds=2),
    )

    with pytest.raises(NodeModelPrestageConflictError, match="pin does not match"):
        executor.execute(
            _commands(envelope.manifest_digest)[0],
            claim_id="a" * 64,
            envelope=envelope,
            trust_store=trust_store,
            request=request,
        )

    assert store.list_records()[0].state is ModelCachePlacementState.FAILED


def test_fill_failure_is_recorded_and_exact_command_can_retry(tmp_path: Path) -> None:
    envelope, trust_store, request = _artifact()
    source = Source(failure=OSError("object store unavailable"))
    executor, index, store = _executor(tmp_path, source)
    command = _commands(envelope.manifest_digest)[0]

    with pytest.raises(ModelArtifactDownloadError, match="download failed"):
        executor.execute(
            command,
            claim_id="a" * 64,
            envelope=envelope,
            trust_store=trust_store,
            request=request,
        )
    failed = store.list_records()[0]
    assert failed.state is ModelCachePlacementState.FAILED
    assert failed.attempt == 1
    assert index.get(command.manifest_digest) is None

    source.failure = None
    ready = executor.execute(
        command,
        claim_id="b" * 64,
        envelope=envelope,
        trust_store=trust_store,
        request=request,
    )
    assert ready.state is ModelCachePlacementState.READY
    assert ready.attempt == 2


def test_failed_command_must_retry_exactly_or_release_before_successor(tmp_path: Path) -> None:
    envelope, trust_store, request = _artifact()
    executor, index, store = _executor(
        tmp_path,
        Source(failure=OSError("object store unavailable")),
    )
    command = _commands(envelope.manifest_digest)[0]
    with pytest.raises(ModelArtifactDownloadError):
        executor.execute(
            command,
            claim_id="a" * 64,
            envelope=envelope,
            trust_store=trust_store,
            request=request,
        )
    successor = _commands(
        envelope.manifest_digest,
        token=5,
        generation_start=21,
        target_revision=9,
    )[0]
    with pytest.raises(NodeModelPrestageConflictError, match="released absent"):
        store.claim(successor, claim_id="b" * 64, now=_NOW + timedelta(seconds=3))

    release = build_node_model_prestage_release_command(
        command,
        authority=_authority(token=5),
        decision_id="scale-decision-18",
        decision_fingerprint="e" * 64,
        target_revision=9,
        command_generation=21,
        issued_at=_NOW + timedelta(seconds=1),
        ttl_seconds=60,
    )
    assert executor.release(release).state is ModelCachePlacementState.ABSENT
    assert executor.release(release).state is ModelCachePlacementState.ABSENT
    assert index.get(command.manifest_digest) is None


def test_claim_cas_allows_only_one_concurrent_attempt() -> None:
    envelope, _, _ = _artifact()
    command = _commands(envelope.manifest_digest)[0]
    store = InMemoryNodeModelPrestageStore(node_id="gpu-node-00")
    barrier = threading.Barrier(2)

    def claim(claim_id: str):
        barrier.wait(timeout=5)
        return store.claim(command, claim_id=claim_id, now=_NOW + timedelta(seconds=1))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(claim, value * 64) for value in ("a", "b")]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except NodeModelPrestageConflictError as exc:
            outcomes.append(exc)

    assert sum(isinstance(value, NodeModelPrestageConflictError) for value in outcomes) == 1
    assert store.list_records()[0].state is ModelCachePlacementState.FILLING


def test_state_mutation_rejects_clock_rollback() -> None:
    envelope, _, _ = _artifact()
    command = _commands(envelope.manifest_digest)[0]
    store = InMemoryNodeModelPrestageStore(node_id="gpu-node-00")
    store.claim(command, claim_id="a" * 64, now=_NOW + timedelta(seconds=2))

    with pytest.raises(NodeModelPrestageConflictError, match="time regressed"):
        store.fail(
            command,
            claim_id="a" * 64,
            failure="injected failure",
            now=_NOW + timedelta(seconds=1),
        )


def test_expired_or_wrong_node_command_is_fail_closed() -> None:
    envelope, _, _ = _artifact()
    command = _commands(envelope.manifest_digest)[0]
    store = InMemoryNodeModelPrestageStore(node_id="gpu-node-00")

    with pytest.raises(NodeModelPrestageExpiredError):
        store.claim(command, claim_id="a" * 64, now=_NOW + timedelta(seconds=60))
    with pytest.raises(NodeModelPrestageConflictError, match="another node"):
        InMemoryNodeModelPrestageStore(node_id="gpu-node-99").claim(
            command,
            claim_id="a" * 64,
            now=_NOW + timedelta(seconds=1),
        )


def test_release_removes_only_prestage_owner_pin(tmp_path: Path) -> None:
    envelope, trust_store, request = _artifact()
    executor, index, store = _executor(tmp_path, Source())
    command = _commands(envelope.manifest_digest)[0]
    executor.execute(
        command,
        claim_id="a" * 64,
        envelope=envelope,
        trust_store=trust_store,
        request=request,
    )
    index.pin(command.manifest_digest, owner="rollback/release-1", reason="rollback")

    stale_release = build_node_model_prestage_release_command(
        command,
        authority=_authority(token=3),
        decision_id="scale-decision-18",
        decision_fingerprint="e" * 64,
        target_revision=9,
        command_generation=21,
        issued_at=_NOW + timedelta(seconds=1),
        ttl_seconds=60,
    )
    with pytest.raises(NodeModelPrestageConflictError, match="fencing token regressed"):
        executor.release(stale_release)
    assert command.pin_owner in index.get(command.manifest_digest).pin_owners
    assert store.list_records()[0].state is ModelCachePlacementState.READY

    changed_holder = build_node_model_prestage_release_command(
        command,
        authority=_authority(token=4, holder_id="controller-b"),
        decision_id="scale-decision-18",
        decision_fingerprint="e" * 64,
        target_revision=9,
        command_generation=21,
        issued_at=_NOW + timedelta(seconds=1),
        ttl_seconds=60,
    )
    with pytest.raises(NodeModelPrestageConflictError, match="holder changed"):
        executor.release(changed_holder)

    release = build_node_model_prestage_release_command(
        command,
        authority=_authority(token=5),
        decision_id="scale-decision-18",
        decision_fingerprint="e" * 64,
        target_revision=9,
        command_generation=22,
        issued_at=_NOW + timedelta(seconds=1),
        ttl_seconds=60,
    )
    released = executor.release(release)
    replay = executor.release(release)

    assert released.state is ModelCachePlacementState.ABSENT
    assert replay == released
    assert index.get(command.manifest_digest).pin_owners == ("rollback/release-1",)


def test_release_replay_succeeds_after_unpinned_entry_is_evicted(tmp_path: Path) -> None:
    envelope, trust_store, request = _artifact()
    executor, index, _ = _executor(tmp_path, Source())
    command = _commands(envelope.manifest_digest)[0]
    executor.execute(
        command,
        claim_id="a" * 64,
        envelope=envelope,
        trust_store=trust_store,
        request=request,
    )
    release = build_node_model_prestage_release_command(
        command,
        authority=_authority(token=5),
        decision_id="scale-decision-18",
        decision_fingerprint="e" * 64,
        target_revision=9,
        command_generation=21,
        issued_at=_NOW + timedelta(seconds=1),
        ttl_seconds=60,
    )
    executor.release(release)
    record = index.get(command.manifest_digest)
    snapshot = index.snapshot()
    with index.fenced_eviction(
        command.manifest_digest,
        expected_index_revision=snapshot.revision,
        expected_generation=record.generation,
    ):
        pass

    assert executor.release(release).state is ModelCachePlacementState.ABSENT


def test_ready_release_converges_after_node_index_is_recreated(tmp_path: Path) -> None:
    envelope, trust_store, request = _artifact()
    executor, _, store = _executor(tmp_path / "old", Source())
    command = _commands(envelope.manifest_digest)[0]
    executor.execute(
        command,
        claim_id="a" * 64,
        envelope=envelope,
        trust_store=trust_store,
        request=request,
    )
    replacement_root = tmp_path / "replacement"
    replacement_index = NodeModelCacheIndex(
        replacement_root / "cache-index.sqlite3",
        node_id="gpu-node-00",
    )
    replacement = NodeModelPrestageExecutor(
        node_id="gpu-node-00",
        agent=NodeModelCacheAgent(replacement_root, Source(), index=replacement_index),
        index=replacement_index,
        store=store,
        clock=lambda: _NOW + timedelta(seconds=2),
    )
    release = build_node_model_prestage_release_command(
        command,
        authority=_authority(token=5),
        decision_id="scale-decision-18",
        decision_fingerprint="e" * 64,
        target_revision=9,
        command_generation=21,
        issued_at=_NOW + timedelta(seconds=1),
        ttl_seconds=60,
    )

    assert replacement.release(release).state is ModelCachePlacementState.ABSENT
    assert replacement.release(release).state is ModelCachePlacementState.ABSENT


def test_successor_requires_release_and_monotonic_generation() -> None:
    envelope, _, _ = _artifact()
    original = _commands(envelope.manifest_digest)[0]
    store = InMemoryNodeModelPrestageStore(node_id="gpu-node-00")
    store.claim(original, claim_id="a" * 64, now=_NOW + timedelta(seconds=1))

    newer_payload = original.model_dump(mode="json", exclude={"command_id"})
    newer_payload["command_generation"] = original.command_generation + 1
    newer_payload["command_id"] = "0" * 64
    with pytest.raises(ValidationError):
        type(original).model_validate(newer_payload)

    with pytest.raises(NodeModelPrestageConflictError, match="another attempt"):
        store.claim(original, claim_id="b" * 64, now=_NOW + timedelta(seconds=1))

    with pytest.raises(ValueError, match="target_revision must advance"):
        build_node_model_prestage_release_command(
            original,
            authority=_authority(token=5),
            decision_id="scale-decision-18",
            decision_fingerprint="e" * 64,
            target_revision=7,
            command_generation=21,
            issued_at=_NOW + timedelta(seconds=1),
            ttl_seconds=60,
        )


def test_overlay_publishes_filling_ready_failed_and_preserves_released_residency(
    tmp_path: Path,
) -> None:
    envelope, trust_store, request = _artifact()
    plan = _plan(envelope.manifest_digest, count=2)
    first, second = _commands(envelope.manifest_digest, count=2)
    executor, _, store = _executor(tmp_path, Source())
    ready = executor.execute(
        first,
        claim_id="a" * 64,
        envelope=envelope,
        trust_store=trust_store,
        request=request,
    )
    other_store = InMemoryNodeModelPrestageStore(node_id="gpu-node-01")
    filling = other_store.claim(second, claim_id="b" * 64, now=_NOW + timedelta(seconds=1))

    fresh_placements = (
        plan.snapshot.placements[0].model_copy(
            update={
                "state": ModelCachePlacementState.READY,
                "cache_hint_observed_at": _NOW + timedelta(seconds=3),
                "cache_hint_valid_until": _NOW + timedelta(minutes=1),
                "cache_hint_index_revision": 11,
            }
        ),
        plan.snapshot.placements[1],
    )
    fresh_snapshot = plan.snapshot.model_copy(
        update={
            "observed_at": _NOW + timedelta(seconds=3),
            "placements": fresh_placements,
        }
    )
    updated = apply_node_model_prestage_records(
        fresh_snapshot,
        (ready, filling),
        snapshot_id="cache-snapshot-11",
        cache_revision=11,
        observed_at=_NOW + timedelta(seconds=4),
    )

    assert tuple(placement.state for placement in updated.placements) == (
        ModelCachePlacementState.READY,
        ModelCachePlacementState.FILLING,
    )
    assert store.list_records()[0] == ready


def test_overlay_does_not_publish_ready_without_physical_hint_evidence(
    tmp_path: Path,
) -> None:
    envelope, trust_store, request = _artifact()
    executor, _, _ = _executor(tmp_path, Source())
    command = _commands(envelope.manifest_digest)[0]
    ready = executor.execute(
        command,
        claim_id="a" * 64,
        envelope=envelope,
        trust_store=trust_store,
        request=request,
    )
    unproven = _plan(envelope.manifest_digest).snapshot.model_copy(
        update={
            "observed_at": _NOW + timedelta(seconds=3),
            "placements": (
                _plan(envelope.manifest_digest)
                .snapshot.placements[0]
                .model_copy(update={"state": ModelCachePlacementState.READY}),
            ),
        }
    )

    without_record = apply_node_model_prestage_records(
        unproven,
        (),
        snapshot_id="cache-snapshot-11",
        cache_revision=11,
        observed_at=_NOW + timedelta(seconds=4),
    )

    updated = apply_node_model_prestage_records(
        unproven,
        (ready,),
        snapshot_id="cache-snapshot-12",
        cache_revision=12,
        observed_at=_NOW + timedelta(seconds=4),
    )

    assert without_record.placements[0].state is ModelCachePlacementState.ABSENT
    assert updated.placements[0].state is ModelCachePlacementState.FILLING


def test_snapshot_rejects_future_dated_physical_hint_evidence() -> None:
    envelope, _, _ = _artifact()
    placement = (
        _plan(envelope.manifest_digest)
        .snapshot.placements[0]
        .model_copy(
            update={
                "state": ModelCachePlacementState.READY,
                "cache_hint_observed_at": _NOW + timedelta(seconds=1),
                "cache_hint_valid_until": _NOW + timedelta(minutes=1),
                "cache_hint_index_revision": 11,
            }
        )
    )

    with pytest.raises(ValidationError, match="newer than the snapshot"):
        ScalingPrewarmSnapshot(
            snapshot_id="future-hint",
            cache_revision=11,
            observed_at=_NOW,
            model_class="interactive-h100",
            model_revision="release-1",
            artifact_digest=envelope.manifest_digest,
            placement_binding_id="binding-prestage-h100",
            placements=(placement,),
        )


def test_overlay_does_not_extend_physical_hint_ttl() -> None:
    envelope, _, _ = _artifact()
    placement = (
        _plan(envelope.manifest_digest)
        .snapshot.placements[0]
        .model_copy(
            update={
                "state": ModelCachePlacementState.READY,
                "cache_hint_observed_at": _NOW,
                "cache_hint_valid_until": _NOW + timedelta(seconds=4),
                "cache_hint_index_revision": 11,
            }
        )
    )
    physical = ScalingPrewarmSnapshot(
        snapshot_id="physical-before-expiry",
        cache_revision=11,
        observed_at=_NOW + timedelta(seconds=3),
        model_class="interactive-h100",
        model_revision="release-1",
        artifact_digest=envelope.manifest_digest,
        placement_binding_id="binding-prestage-h100",
        placements=(placement,),
    )

    updated = apply_node_model_prestage_records(
        physical,
        (),
        snapshot_id="overlay-after-expiry",
        cache_revision=12,
        observed_at=_NOW + timedelta(seconds=5),
    )

    assert updated.placements[0].state is ModelCachePlacementState.ABSENT


def test_fresh_absent_cache_snapshot_cannot_resurrect_old_ready_record(tmp_path: Path) -> None:
    envelope, trust_store, request = _artifact()
    executor, index, _ = _executor(tmp_path, Source())
    stale_hint = NodeModelCachePlacementHintPublisher(
        index,
        clock=lambda: _NOW + timedelta(seconds=1),
    ).snapshot()
    command = _commands(envelope.manifest_digest)[0]
    ready = executor.execute(
        command,
        claim_id="a" * 64,
        envelope=envelope,
        trust_store=trust_store,
        request=request,
    )
    candidate = ModelCachePlacementCandidate(
        placement_id="placement-00",
        node_name="gpu-node-00",
        resource_flavor="h100-sxm",
        profile_id="h100-sxm",
        compatibility_approval_id="compat-h100-v1",
    )
    old_physical = build_cache_placement_snapshot(
        (stale_hint,),
        (candidate,),
        snapshot_id="physical-11",
        cache_revision=11,
        observed_at=_NOW + timedelta(seconds=3),
        model_class="interactive-h100",
        model_id="org/prestage-test",
        model_revision="release-1",
        artifact_digest=envelope.manifest_digest,
        placement_binding_id="binding-prestage-h100",
    )
    pending = apply_node_model_prestage_records(
        old_physical,
        (ready,),
        snapshot_id="physical-overlay-12",
        cache_revision=12,
        observed_at=_NOW + timedelta(seconds=4),
    )
    assert pending.placements[0].state is ModelCachePlacementState.FILLING

    index.mark_unverified(command.manifest_digest, reason="injected corruption")
    fresh_hint = NodeModelCachePlacementHintPublisher(
        index,
        clock=lambda: _NOW + timedelta(seconds=5),
    ).snapshot()
    fresh_physical = build_cache_placement_snapshot(
        (fresh_hint,),
        (candidate,),
        snapshot_id="physical-13",
        cache_revision=13,
        observed_at=_NOW + timedelta(seconds=6),
        model_class="interactive-h100",
        model_id="org/prestage-test",
        model_revision="release-1",
        artifact_digest=envelope.manifest_digest,
        placement_binding_id="binding-prestage-h100",
    )
    updated = apply_node_model_prestage_records(
        fresh_physical,
        (ready,),
        snapshot_id="physical-overlay-14",
        cache_revision=14,
        observed_at=_NOW + timedelta(seconds=7),
    )

    assert updated.placements[0].state is ModelCachePlacementState.FAILED


def test_overlay_rejects_cross_binding_record() -> None:
    envelope, _, _ = _artifact()
    command = _commands(envelope.manifest_digest)[0]
    store = InMemoryNodeModelPrestageStore(node_id="gpu-node-00")
    filling = store.claim(command, claim_id="a" * 64, now=_NOW + timedelta(seconds=1))
    other = _plan(envelope.manifest_digest).snapshot.model_copy(
        update={"placement_binding_id": "binding-other"}
    )

    with pytest.raises(NodeModelPrestageConflictError, match="identity"):
        apply_node_model_prestage_records(
            other,
            (filling,),
            snapshot_id="cache-snapshot-11",
            cache_revision=11,
            observed_at=_NOW + timedelta(seconds=3),
        )
