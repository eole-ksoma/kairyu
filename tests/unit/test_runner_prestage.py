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
    NodeModelCacheCapacityPolicy,
    NodeModelCacheIndex,
    NodeModelCachePlacementHintPublisher,
    SignedModelArtifactManifest,
    TrustedModelSigner,
    model_file_tree_digest,
    plan_node_model_cache_eviction,
    sign_model_artifact_manifest,
)
from kairyu.runners import (
    InMemoryNodeModelPrestageStore,
    LocalNodeModelCacheLiveEvidenceSource,
    ModelCachePlacement,
    ModelCachePlacementCandidate,
    ModelCachePlacementState,
    NodeModelCacheLiveEvidenceRequest,
    NodeModelPrestageCapacityError,
    NodeModelPrestageCompactionMonitoringStore,
    NodeModelPrestageCompactionStore,
    NodeModelPrestageConflictError,
    NodeModelPrestageExecutor,
    NodeModelPrestageExpiredError,
    NodeModelPrestageLookupStore,
    NodeModelPrestageStore,
    RunnerCacheStartupPlacement,
    RunnerWriterAuthority,
    ScalingPrewarmSnapshot,
    apply_node_model_prestage_records,
    build_cache_placement_snapshot,
    build_node_model_prestage_commands,
    build_node_model_prestage_release_command,
    plan_cache_aware_scale_up,
)
from kairyu.runners.prestage import _canonical_digest

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
    assert first[0].pin_owner == "prestage/production/prestage-test/placement-00/20"


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
    cached = index.get(command.manifest_digest)
    assert cached is not None
    assert command.pin_owner in cached.pin_owners
    assert ready.pin_record_generation == cached.generation
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


def test_compaction_reclaims_active_capacity_and_preserves_all_fences() -> None:
    envelope, _, _ = _artifact()
    original = _commands(envelope.manifest_digest)[0]
    store = InMemoryNodeModelPrestageStore(
        node_id="gpu-node-00",
        max_placements=1,
    )
    store.claim(original, claim_id="a" * 64, now=_NOW + timedelta(seconds=1))
    store.fail(
        original,
        claim_id="a" * 64,
        failure="retire this placement",
        now=_NOW + timedelta(seconds=2),
    )
    release = build_node_model_prestage_release_command(
        original,
        authority=_authority(token=5),
        decision_id="scale-decision-18",
        decision_fingerprint="e" * 64,
        target_revision=9,
        command_generation=21,
        issued_at=_NOW + timedelta(seconds=3),
        ttl_seconds=60,
    )
    absent = store.release(release, now=_NOW + timedelta(seconds=4))

    other = _commands(
        envelope.manifest_digest,
        generation_start=30,
    )[0].model_copy(update={"placement_id": "placement-other"})
    other_payload = other.model_dump(mode="json", exclude={"command_id"})
    other_payload["command_id"] = _canonical_digest(other_payload)
    other = type(original).model_validate(other_payload)
    with pytest.raises(NodeModelPrestageCapacityError):
        store.claim(other, claim_id="b" * 64, now=_NOW + timedelta(seconds=5))

    compacted = store.compact_absent_records(
        retired_before=_NOW + timedelta(seconds=4),
        compacted_at=_NOW + timedelta(seconds=6),
    )

    assert len(compacted) == 1
    assert compacted[0].command_generation == 21
    assert compacted[0].fencing_token == 5
    assert compacted[0].target_revision == 9
    assert store.list_records() == ()
    assert store.list_high_water_marks() == compacted
    assert store.release(release, now=_NOW + timedelta(seconds=7)) == absent
    assert store.list_records() == ()
    store.claim(other, claim_id="b" * 64, now=_NOW + timedelta(seconds=5))

    stale_generation = _commands(
        envelope.manifest_digest,
        token=6,
        generation_start=21,
        target_revision=10,
    )[0]
    with pytest.raises(NodeModelPrestageConflictError, match="generation did not advance"):
        store.claim(
            stale_generation,
            claim_id="c" * 64,
            now=_NOW + timedelta(seconds=5),
        )

    changed_election_payload = stale_generation.model_dump(mode="json", exclude={"command_id"})
    changed_election_payload["command_generation"] = 22
    changed_election_payload["authority"]["election_id"] = "other-election"
    changed_election_payload["command_id"] = _canonical_digest(changed_election_payload)
    changed_election = type(original).model_validate(changed_election_payload)
    with pytest.raises(NodeModelPrestageConflictError, match="election identity changed"):
        store.claim(
            changed_election,
            claim_id="c" * 64,
            now=_NOW + timedelta(seconds=5),
        )

    changed_holder_payload = stale_generation.model_dump(mode="json", exclude={"command_id"})
    changed_holder_payload["command_generation"] = 22
    changed_holder_payload["authority"]["fencing_token"] = 5
    changed_holder_payload["authority"]["holder_id"] = "controller-b"
    changed_holder_payload["command_id"] = _canonical_digest(changed_holder_payload)
    changed_holder = type(original).model_validate(changed_holder_payload)
    with pytest.raises(NodeModelPrestageConflictError, match="holder changed"):
        store.claim(
            changed_holder,
            claim_id="c" * 64,
            now=_NOW + timedelta(seconds=5),
        )

    store.compact_absent_records(
        retired_before=_NOW + timedelta(seconds=10),
        compacted_at=_NOW + timedelta(seconds=10),
    )
    stale_fence = _commands(
        envelope.manifest_digest,
        token=4,
        generation_start=22,
        target_revision=10,
    )[0]
    with pytest.raises(NodeModelPrestageConflictError, match="fencing token regressed"):
        store.claim(stale_fence, claim_id="d" * 64, now=_NOW + timedelta(seconds=5))

    stale_target = _commands(
        envelope.manifest_digest,
        token=6,
        generation_start=22,
        target_revision=8,
    )[0]
    with pytest.raises(NodeModelPrestageConflictError, match="target revision regressed"):
        store.claim(stale_target, claim_id="e" * 64, now=_NOW + timedelta(seconds=5))


def test_compaction_is_cutoff_limited_and_never_moves_live_records() -> None:
    envelope, _, _ = _artifact()
    first, second = _commands(envelope.manifest_digest, count=2)
    store = InMemoryNodeModelPrestageStore(node_id="gpu-node-00")
    for index, command in enumerate((first, second), start=1):
        if command.node_id != "gpu-node-00":
            payload = command.model_dump(mode="json", exclude={"command_id"})
            payload["node_id"] = "gpu-node-00"
            payload["command_id"] = _canonical_digest(payload)
            command = type(command).model_validate(payload)
        claim_id = str(index) * 64
        store.claim(command, claim_id=claim_id, now=_NOW + timedelta(seconds=index))
        store.fail(
            command,
            claim_id=claim_id,
            failure="retired",
            now=_NOW + timedelta(seconds=index + 2),
        )
        release = build_node_model_prestage_release_command(
            command,
            authority=_authority(token=5),
            decision_id=f"retire-{index}",
            decision_fingerprint=str(index) * 64,
            target_revision=9,
            command_generation=30 + index,
            issued_at=_NOW + timedelta(seconds=index + 4),
            ttl_seconds=60,
        )
        store.release(release, now=_NOW + timedelta(seconds=index + 5))

    live = _commands(
        envelope.manifest_digest,
        generation_start=40,
    )[0]
    live_payload = live.model_dump(mode="json", exclude={"command_id"})
    live_payload["placement_id"] = "placement-live"
    live_payload["command_id"] = _canonical_digest(live_payload)
    live = type(live).model_validate(live_payload)
    store.claim(live, claim_id="f" * 64, now=_NOW + timedelta(seconds=1))

    first_batch = store.compact_absent_records(
        retired_before=_NOW + timedelta(seconds=20),
        compacted_at=_NOW + timedelta(seconds=20),
        limit=1,
    )
    second_batch = store.compact_absent_records(
        retired_before=_NOW + timedelta(seconds=20),
        compacted_at=_NOW + timedelta(seconds=20),
        limit=1,
    )

    assert len(first_batch) == len(second_batch) == 1
    assert {mark.placement_id for mark in first_batch + second_batch} == {
        "placement-00",
        "placement-01",
    }
    assert store.list_high_water_marks_page(limit=1) == first_batch
    assert (
        store.list_high_water_marks_page(
            after_placement_id=first_batch[0].placement_id,
            limit=1,
        )
        == second_batch
    )
    with pytest.raises(ValueError, match=r"\[1, 1000\]"):
        store.list_high_water_marks_page(limit=0)
    assert tuple(record.command.placement_id for record in store.list_records()) == (
        "placement-live",
    )
    assert (
        store.compact_absent_records(
            retired_before=_NOW + timedelta(seconds=20),
            compacted_at=_NOW + timedelta(seconds=20),
        )
        == ()
    )

    with pytest.raises(ValueError, match="cannot predate"):
        store.compact_absent_records(
            retired_before=_NOW + timedelta(seconds=20),
            compacted_at=_NOW + timedelta(seconds=19),
        )


def test_compacted_mark_accepts_only_an_authorized_successor_lineage() -> None:
    envelope, _, _ = _artifact()
    original = _commands(envelope.manifest_digest)[0]
    store = InMemoryNodeModelPrestageStore(node_id="gpu-node-00")
    store.claim(original, claim_id="a" * 64, now=_NOW + timedelta(seconds=1))
    store.fail(
        original,
        claim_id="a" * 64,
        failure="retired",
        now=_NOW + timedelta(seconds=2),
    )
    release = build_node_model_prestage_release_command(
        original,
        authority=_authority(token=5),
        decision_id="retire-original",
        decision_fingerprint="e" * 64,
        target_revision=9,
        command_generation=21,
        issued_at=_NOW + timedelta(seconds=3),
        ttl_seconds=60,
    )
    store.release(release, now=_NOW + timedelta(seconds=4))
    store.compact_absent_records(
        retired_before=_NOW + timedelta(seconds=4),
        compacted_at=_NOW + timedelta(seconds=5),
    )

    forged_payload = release.model_dump(mode="json", exclude={"command_id"})
    forged_payload["deployment_id"] = "production/other"
    forged_payload["command_generation"] = 22
    forged_payload["target_revision"] = 10
    forged_payload["command_id"] = _canonical_digest(forged_payload)
    forged = type(release).model_validate(forged_payload)
    with pytest.raises(NodeModelPrestageConflictError, match="identity does not match"):
        store.release(forged, now=_NOW + timedelta(seconds=6))

    successor = _commands(
        envelope.manifest_digest,
        token=6,
        generation_start=22,
        target_revision=10,
    )[0]
    claimed = store.claim(
        successor,
        claim_id="b" * 64,
        now=_NOW + timedelta(seconds=6),
    )
    assert claimed.command == successor
    assert claimed.attempt == 1


def test_compaction_extension_preserves_legacy_store_runtime_compatibility() -> None:
    class LegacyStore:
        node_id = "gpu-node-00"

        def claim(self, *args, **kwargs):
            raise NotImplementedError

        def complete(self, *args, **kwargs):
            raise NotImplementedError

        def fail(self, *args, **kwargs):
            raise NotImplementedError

        def release(self, *args, **kwargs):
            raise NotImplementedError

        def list_records(self):
            return ()

        def list_records_page(self, *args, **kwargs):
            return ()

    legacy = LegacyStore()

    assert isinstance(legacy, NodeModelPrestageStore)
    assert not isinstance(legacy, NodeModelPrestageLookupStore)
    assert not isinstance(legacy, NodeModelPrestageCompactionStore)
    assert isinstance(
        InMemoryNodeModelPrestageStore(node_id="gpu-node-00"),
        NodeModelPrestageCompactionMonitoringStore,
    )


def test_lookup_extension_returns_an_isolated_exact_record() -> None:
    command = _commands("a" * 64)[0]
    store = InMemoryNodeModelPrestageStore(node_id="gpu-node-00")
    claimed = store.claim(
        command,
        claim_id="b" * 64,
        now=_NOW + timedelta(seconds=1),
    )

    first = store.get_record(command.placement_id)
    second = store.get_record(command.placement_id)

    assert isinstance(store, NodeModelPrestageLookupStore)
    assert first == claimed
    assert second == claimed
    assert first is not second
    assert store.get_record("missing-placement") is None


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


class _NoAuditSink:
    def emit(self, event) -> None:
        raise AssertionError(f"unexpected cache audit event: {event}")


def _ticking_node(tmp_path: Path):
    ticks = iter(range(1, 1_000))
    root = tmp_path / "cache"
    index = NodeModelCacheIndex(
        root / "cache-index.sqlite3",
        node_id="gpu-node-00",
        clock_ns=lambda: next(ticks) * 1_000_000,
    )
    store = InMemoryNodeModelPrestageStore(node_id="gpu-node-00")
    agent = NodeModelCacheAgent(root, Source(), index=index, chunk_size_bytes=4)
    executor = NodeModelPrestageExecutor(
        node_id="gpu-node-00",
        agent=agent,
        index=index,
        store=store,
        clock=lambda: _NOW + timedelta(seconds=2),
    )
    return executor, agent, index, store


def _read_live_evidence(store, index, command, record):
    evidence = LocalNodeModelCacheLiveEvidenceSource(
        node_id="gpu-node-00",
        store=store,
        index=index,
        hint_ttl_seconds=30,
        clock=lambda: _NOW + timedelta(seconds=3),
    )
    return evidence.read(
        NodeModelCacheLiveEvidenceRequest(
            placement=RunnerCacheStartupPlacement(
                placement_id=command.placement_id,
                node_name=command.node_id,
                resource_flavor=command.resource_flavor,
                profile_id=command.profile_id,
                compatibility_approval_id=command.compatibility_approval_id,
                manifest_digest=command.manifest_digest,
                pin_owner=command.pin_owner,
                prestage_command_id=command.command_id,
                prestage_command_generation=command.command_generation,
                hint_index_revision=1,
                resident_record_generation=record.pin_record_generation,
                hint_observed_at=_NOW,
                hint_valid_until=_NOW + timedelta(minutes=1),
            ),
            model_id=command.model_id,
            model_revision=command.model_revision,
        )
    )


def test_runner_start_verification_keeps_prestage_binding_evidence_live(
    tmp_path: Path,
) -> None:
    envelope, trust_store, request = _artifact()
    executor, agent, index, store = _ticking_node(tmp_path)
    command = _commands(envelope.manifest_digest)[0]
    record = executor.execute(
        command,
        claim_id="a" * 64,
        envelope=envelope,
        trust_store=trust_store,
        request=request,
    )
    _read_live_evidence(store, index, command, record)
    accessed_before = index.get(command.manifest_digest).last_access_at_ns

    decision = agent.verify_for_runner_start(
        envelope,
        trust_store,
        request,
        audit_sink=_NoAuditSink(),
    )

    assert decision.runner_start_allowed is True
    assert index.get(command.manifest_digest).last_access_at_ns > accessed_before
    assert _read_live_evidence(store, index, command, record).pin_evidence.record_generation == (
        record.pin_record_generation
    )


def test_in_flight_duplicate_ensure_keeps_completed_prestage_evidence_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope, trust_store, request = _artifact()
    executor, agent, index, store = _ticking_node(tmp_path)
    command = _commands(envelope.manifest_digest)[0]
    fill = agent.ensure_cached
    completed = []

    def duplicate_completes_while_this_request_waits(*args, **kwargs):
        if not completed:
            completed.append(None)
            completed[0] = executor.execute(
                command,
                claim_id="a" * 64,
                envelope=envelope,
                trust_store=trust_store,
                request=request,
            )
        return fill(*args, **kwargs)

    monkeypatch.setattr(agent, "ensure_cached", duplicate_completes_while_this_request_waits)

    with pytest.raises(NodeModelPrestageConflictError, match="stale"):
        executor.execute(
            command,
            claim_id="a" * 64,
            envelope=envelope,
            trust_store=trust_store,
            request=request,
        )

    record = completed[0]
    assert record.state is ModelCachePlacementState.READY
    assert store.list_records() == (record,)
    assert _read_live_evidence(store, index, command, record).pin_evidence.record_generation == (
        record.pin_record_generation
    )


def test_stale_release_unpin_cannot_remove_successor_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope, trust_store, request = _artifact()
    executor, index, store = _executor(tmp_path, Source())
    first = _commands(envelope.manifest_digest)[0]
    executor.execute(
        first,
        claim_id="a" * 64,
        envelope=envelope,
        trust_store=trust_store,
        request=request,
    )
    release = build_node_model_prestage_release_command(
        first,
        authority=_authority(token=5),
        decision_id="scale-decision-18",
        decision_fingerprint="e" * 64,
        target_revision=9,
        command_generation=21,
        issued_at=_NOW + timedelta(seconds=1),
        ttl_seconds=60,
    )
    successor = _commands(
        envelope.manifest_digest,
        token=5,
        generation_start=22,
        target_revision=10,
    )[0]
    commit_release = store.release

    def release_then_successor_completes(command, *, now):
        record = commit_release(command, now=now)
        executor.execute(
            successor,
            claim_id="b" * 64,
            envelope=envelope,
            trust_store=trust_store,
            request=request,
        )
        return record

    monkeypatch.setattr(store, "release", release_then_successor_completes)

    executor.release(release)

    record = store.list_records()[0]
    cached = index.get(envelope.manifest_digest)
    assert record.state is ModelCachePlacementState.READY
    assert record.command.command_generation == 22
    assert cached.pin_owners == (successor.pin_owner,)
    assert cached.generation == record.pin_record_generation
    plan = plan_node_model_cache_eviction(
        index.snapshot(),
        NodeModelCacheCapacityPolicy(high_watermark_bytes=1, low_watermark_bytes=0),
    )
    assert plan.victims == ()
