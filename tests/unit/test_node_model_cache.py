"""WP4.2 resumable, verified node-local model cache coverage."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import multiprocessing
import os
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kairyu.artifacts import (
    HttpRangeModelArtifactBlobSource,
    InvalidNodeModelCacheEntryError,
    LocalModelArtifactBlobSource,
    ModelArtifactAdmissionError,
    ModelArtifactAdmissionRequest,
    ModelArtifactBlob,
    ModelArtifactDigestMismatchError,
    ModelArtifactDownloadError,
    ModelArtifactManifest,
    ModelArtifactQuantization,
    ModelArtifactResourceEstimate,
    ModelArtifactTokenizer,
    ModelArtifactTrustStore,
    NodeModelCacheAgent,
    NodeModelCacheAuditError,
    NodeModelCacheCorruptionAuditEvent,
    NodeModelCacheIndex,
    NodeModelCacheIndexError,
    NodeModelCacheIndexUnverifiedError,
    NodeModelCacheLockTimeoutError,
    NodeModelCacheRecoveryError,
    SignedModelArtifactManifest,
    TrustedModelSigner,
    model_file_tree_digest,
    sign_model_artifact_manifest,
)

_DATA = {
    "LICENSE": b"Apache-2.0\n",
    "weights/model.bin": b"0123456789abcdef",
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _signed_artifact() -> tuple[
    SignedModelArtifactManifest,
    ModelArtifactTrustStore,
    ModelArtifactAdmissionRequest,
]:
    blobs = tuple(
        ModelArtifactBlob(path=path, size_bytes=len(content), sha256=_sha256(content))
        for path, content in sorted(_DATA.items())
    )
    manifest = ModelArtifactManifest(
        model_id="org/cache-test",
        model_revision="release-1",
        upstream_repository="https://example.invalid/org/cache-test",
        upstream_revision="1" * 40,
        architecture="CacheTestForCausalLM",
        quantization=ModelArtifactQuantization(
            method="none",
            format="safetensors",
        ),
        tokenizer=ModelArtifactTokenizer(
            repository="https://example.invalid/org/cache-test",
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
        deployment_id="production/cache-test",
        manifest_digest=envelope.manifest_digest,
        model_id=manifest.model_id,
        model_revision=manifest.model_revision,
        environment="production",
        gpu_profile="h100-sxm",
    )
    return envelope, trust_store, request


class RecordingSource:
    def __init__(self, *, fail_once_path: str | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self._fail_once_path = fail_once_path
        self._failed = False
        self._lock = threading.Lock()

    def iter_blob(
        self,
        *,
        manifest_digest: str,
        blob: ModelArtifactBlob,
        offset: int,
        chunk_size: int,
    ) -> Iterable[bytes]:
        del manifest_digest
        with self._lock:
            self.calls.append((blob.path, offset))
        content = _DATA[blob.path][offset:]
        for start in range(0, len(content), chunk_size):
            yield content[start : start + chunk_size]
            if blob.path == self._fail_once_path and not self._failed:
                self._failed = True
                raise OSError("simulated source interruption")


class BlockingSource(RecordingSource):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self._blocked = False

    def iter_blob(self, **kwargs) -> Iterable[bytes]:
        with self._lock:
            if not self._blocked:
                self._blocked = True
                should_block = True
            else:
                should_block = False
        if should_block:
            self.started.set()
            assert self.release.wait(timeout=5)
        yield from super().iter_blob(**kwargs)


class RecordingAuditSink:
    def __init__(
        self,
        *,
        fail: bool = False,
        fail_on_event: str | None = None,
        index: NodeModelCacheIndex | None = None,
    ) -> None:
        self.events: list[NodeModelCacheCorruptionAuditEvent] = []
        self.fail = fail
        self.fail_on_event = fail_on_event
        self.index = index
        self.verified_at_emit: list[bool] = []

    def emit(self, event: NodeModelCacheCorruptionAuditEvent) -> None:
        if self.fail or event.event == self.fail_on_event:
            raise OSError("simulated audit failure")
        if self.index is not None:
            record = self.index.get(event.manifest_digest)
            assert record is not None
            self.verified_at_emit.append(record.verified)
        self.events.append(event)


def _assert_published_content(cache_root: Path, envelope: SignedModelArtifactManifest) -> None:
    published = cache_root / "artifacts" / envelope.manifest_digest
    assert published.is_dir()
    for path, content in _DATA.items():
        assert (published / "tree" / path).read_bytes() == content
    assert not (cache_root / ".staging" / envelope.manifest_digest).exists()


def _write_local_source(source_root: Path, manifest_digest: str) -> None:
    for path, content in _DATA.items():
        target = source_root / manifest_digest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _process_fill(cache_root: str, source_root: str, start, output) -> None:
    try:
        envelope, trust_store, request = _signed_artifact()
        start.wait()
        result = NodeModelCacheAgent(
            Path(cache_root),
            LocalModelArtifactBlobSource(Path(source_root)),
            chunk_size_bytes=4,
        ).ensure_cached(envelope, trust_store, request)
        output.put(("ok", result.cache_hit))
    except Exception as exc:  # pragma: no cover - failure is asserted in the parent
        output.put(("error", repr(exc)))


def _hold_process_lock(path: str, ready, release) -> None:
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        ready.set()
        release.wait()
    finally:
        os.close(descriptor)


def test_cold_fill_verifies_and_atomically_publishes_then_hits(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    source = RecordingSource()
    cache_root = tmp_path / "cache"
    agent = NodeModelCacheAgent(cache_root, source, chunk_size_bytes=4)

    cold = agent.ensure_cached(envelope, trust_store, request)
    first_calls = list(source.calls)
    hit = agent.ensure_cached(envelope, trust_store, request)

    _assert_published_content(cache_root, envelope)
    assert cold.cache_hit is False
    assert cold.downloaded_bytes == sum(map(len, _DATA.values()))
    assert cold.resumed_bytes == 0
    assert hit.cache_hit is True
    assert hit.downloaded_bytes == 0
    assert source.calls == first_calls


def test_cache_agent_records_fill_and_hit_in_durable_index(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    cache_root = tmp_path / "cache"
    now = [100]
    index = NodeModelCacheIndex(
        cache_root / "cache-index.sqlite3",
        node_id="node-a",
        clock_ns=lambda: now[0],
    )
    source = RecordingSource()
    agent = NodeModelCacheAgent(cache_root, source, index=index)

    now[0] = 200
    agent.ensure_cached(envelope, trust_store, request)
    filled = index.get(envelope.manifest_digest)
    now[0] = 300
    agent.ensure_cached(envelope, trust_store, request)
    hit = index.get(envelope.manifest_digest)

    assert filled is not None
    assert filled.model_id == envelope.manifest.model_id
    assert filled.model_revision == envelope.manifest.model_revision
    assert filled.verification_source == "filled"
    assert filled.verified_at_ns == 200
    assert hit is not None
    assert hit.verification_source == "filled"
    assert hit.verified_at_ns == 200
    assert hit.last_access_at_ns == 300
    assert hit.generation == filled.generation


def test_interrupted_fill_does_not_create_residency_record(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    cache_root = tmp_path / "cache"
    index = NodeModelCacheIndex(
        cache_root / "cache-index.sqlite3",
        node_id="node-a",
    )
    agent = NodeModelCacheAgent(
        cache_root,
        RecordingSource(fail_once_path="weights/model.bin"),
        index=index,
        chunk_size_bytes=4,
    )

    with pytest.raises(ModelArtifactDownloadError, match="download failed"):
        agent.ensure_cached(envelope, trust_store, request)

    assert index.get(envelope.manifest_digest) is None


def test_index_failure_after_publish_is_repaired_by_hit_without_redownload(
    tmp_path: Path,
):
    class FailingOnceIndex(NodeModelCacheIndex):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.fail_next_record = True

        def record_verified(self, **kwargs):
            if self.fail_next_record:
                self.fail_next_record = False
                raise NodeModelCacheIndexError("injected index failure")
            return super().record_verified(**kwargs)

    envelope, trust_store, request = _signed_artifact()
    cache_root = tmp_path / "cache"
    index = FailingOnceIndex(
        cache_root / "cache-index.sqlite3",
        node_id="node-a",
    )
    source = RecordingSource()
    agent = NodeModelCacheAgent(cache_root, source, index=index)

    with pytest.raises(NodeModelCacheIndexError, match="injected index failure"):
        agent.ensure_cached(envelope, trust_store, request)

    calls_after_publish = list(source.calls)
    _assert_published_content(cache_root, envelope)
    assert index.get(envelope.manifest_digest) is None

    repaired = agent.ensure_cached(envelope, trust_store, request)
    record = index.get(envelope.manifest_digest)

    assert repaired.cache_hit is True
    assert source.calls == calls_after_publish
    assert record is not None
    assert record.verification_source == "published_marker"


def test_index_unverified_state_blocks_structural_cache_hit(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    cache_root = tmp_path / "cache"
    index = NodeModelCacheIndex(
        cache_root / "cache-index.sqlite3",
        node_id="node-a",
    )
    source = RecordingSource()
    agent = NodeModelCacheAgent(cache_root, source, index=index)
    agent.ensure_cached(envelope, trust_store, request)
    calls = list(source.calls)
    index.mark_unverified(envelope.manifest_digest, reason="operator corruption report")

    with pytest.raises(NodeModelCacheIndexUnverifiedError, match="verified refill"):
        agent.ensure_cached(envelope, trust_store, request)

    assert source.calls == calls


def test_runner_start_full_digest_check_allows_verified_residency(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    cache_root = tmp_path / "cache"
    index = NodeModelCacheIndex(
        cache_root / "cache-index.sqlite3",
        node_id="node-a",
    )
    agent = NodeModelCacheAgent(cache_root, RecordingSource(), index=index)
    agent.ensure_cached(envelope, trust_store, request)
    audit = RecordingAuditSink(index=index)

    decision = agent.verify_for_runner_start(
        envelope,
        trust_store,
        request,
        audit_sink=audit,
    )

    assert decision.runner_start_allowed is True
    assert decision.corruption_detected is False
    assert decision.refetched is False
    assert decision.artifact_path == cache_root / "artifacts" / envelope.manifest_digest / "tree"
    assert audit.events == []


def test_same_size_corruption_is_quarantined_audited_refetched_and_start_denied(
    tmp_path: Path,
):
    envelope, trust_store, request = _signed_artifact()
    cache_root = tmp_path / "cache"
    index = NodeModelCacheIndex(
        cache_root / "cache-index.sqlite3",
        node_id="node-a",
    )
    source = RecordingSource()
    agent = NodeModelCacheAgent(cache_root, source, index=index)
    agent.ensure_cached(envelope, trust_store, request)
    corrupt = cache_root / "artifacts" / envelope.manifest_digest / "tree/weights/model.bin"
    corrupt.write_bytes(b"x" * len(_DATA["weights/model.bin"]))
    audit = RecordingAuditSink(index=index)

    decision = agent.verify_for_runner_start(
        envelope,
        trust_store,
        request,
        audit_sink=audit,
    )

    assert decision.runner_start_allowed is False
    assert decision.reason == "digest_mismatch"
    assert decision.refetched is True
    assert decision.quarantine_path is not None
    assert (decision.quarantine_path / "tree/weights/model.bin").read_bytes() == b"x" * len(
        _DATA["weights/model.bin"]
    )
    assert [event.event for event in audit.events] == [
        "corruption_quarantined",
        "corruption_refetched",
    ]
    assert audit.verified_at_emit == [False, False]
    assert index.get(envelope.manifest_digest).verified is True
    _assert_published_content(cache_root, envelope)

    allowed = agent.verify_for_runner_start(
        envelope,
        trust_store,
        request,
        audit_sink=audit,
    )
    assert allowed.runner_start_allowed is True


def test_corruption_refetch_failure_keeps_index_unverified_and_runner_blocked(
    tmp_path: Path,
):
    envelope, trust_store, request = _signed_artifact()
    cache_root = tmp_path / "cache"
    index = NodeModelCacheIndex(
        cache_root / "cache-index.sqlite3",
        node_id="node-a",
    )
    agent = NodeModelCacheAgent(cache_root, RecordingSource(), index=index)
    agent.ensure_cached(envelope, trust_store, request)
    corrupt = cache_root / "artifacts" / envelope.manifest_digest / "tree/weights/model.bin"
    corrupt.write_bytes(b"x" * len(_DATA["weights/model.bin"]))
    audit = RecordingAuditSink()
    original = _DATA["weights/model.bin"]
    _DATA["weights/model.bin"] = b"y" * len(original)
    try:
        with pytest.raises(NodeModelCacheRecoveryError, match="verified refill failed"):
            agent.verify_for_runner_start(
                envelope,
                trust_store,
                request,
                audit_sink=audit,
            )
    finally:
        _DATA["weights/model.bin"] = original

    record = index.get(envelope.manifest_digest)
    assert record is not None and record.verified is False
    assert not (cache_root / "artifacts" / envelope.manifest_digest).exists()
    assert [event.event for event in audit.events] == [
        "corruption_quarantined",
        "corruption_refetch_failed",
    ]


def test_audit_failure_leaves_corruption_quarantined_without_refetch(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    cache_root = tmp_path / "cache"
    index = NodeModelCacheIndex(
        cache_root / "cache-index.sqlite3",
        node_id="node-a",
    )
    agent = NodeModelCacheAgent(cache_root, RecordingSource(), index=index)
    agent.ensure_cached(envelope, trust_store, request)
    corrupt = cache_root / "artifacts" / envelope.manifest_digest / "tree/weights/model.bin"
    corrupt.write_bytes(b"x" * len(_DATA["weights/model.bin"]))

    with pytest.raises(NodeModelCacheAuditError, match="audit sink rejected"):
        agent.verify_for_runner_start(
            envelope,
            trust_store,
            request,
            audit_sink=RecordingAuditSink(fail=True),
        )

    record = index.get(envelope.manifest_digest)
    assert record is not None and record.verified is False
    assert not (cache_root / "artifacts" / envelope.manifest_digest).exists()
    assert len(tuple((cache_root / ".quarantine").iterdir())) == 1

    with pytest.raises(NodeModelCacheIndexUnverifiedError, match="fenced audit completion"):
        agent.ensure_cached(envelope, trust_store, request)
    pending = index.get(envelope.manifest_digest)
    assert pending is not None and pending.recovery_id == record.recovery_id

    retry_audit = RecordingAuditSink()
    decision = agent.verify_for_runner_start(
        envelope,
        trust_store,
        request,
        audit_sink=retry_audit,
    )
    assert decision.runner_start_allowed is False
    assert decision.refetched is True
    assert [event.event for event in retry_audit.events] == [
        "corruption_quarantined",
        "corruption_refetched",
    ]


def test_refetch_audit_failure_marks_replacement_unverified(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    cache_root = tmp_path / "cache"
    index = NodeModelCacheIndex(
        cache_root / "cache-index.sqlite3",
        node_id="node-a",
    )
    agent = NodeModelCacheAgent(cache_root, RecordingSource(), index=index)
    agent.ensure_cached(envelope, trust_store, request)
    corrupt = cache_root / "artifacts" / envelope.manifest_digest / "tree/weights/model.bin"
    corrupt.write_bytes(b"x" * len(_DATA["weights/model.bin"]))

    with pytest.raises(NodeModelCacheAuditError, match="audit sink rejected"):
        agent.verify_for_runner_start(
            envelope,
            trust_store,
            request,
            audit_sink=RecordingAuditSink(fail_on_event="corruption_refetched"),
        )

    record = index.get(envelope.manifest_digest)
    assert record is not None and record.verified is False
    assert record.verification_failure == "digest_mismatch"
    assert record.recovery_id is not None
    _assert_published_content(cache_root, envelope)

    retry_audit = RecordingAuditSink()
    decision = agent.verify_for_runner_start(
        envelope,
        trust_store,
        request,
        audit_sink=retry_audit,
    )
    assert decision.runner_start_allowed is False
    assert decision.refetched is True
    assert [event.event for event in retry_audit.events] == [
        "corruption_quarantined",
        "corruption_refetched",
    ]
    assert retry_audit.events[0].recovery_id == retry_audit.events[1].recovery_id


def test_missing_published_tree_is_recorded_and_refetched_before_start(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    cache_root = tmp_path / "cache"
    index = NodeModelCacheIndex(
        cache_root / "cache-index.sqlite3",
        node_id="node-a",
    )
    agent = NodeModelCacheAgent(cache_root, RecordingSource(), index=index)
    agent.ensure_cached(envelope, trust_store, request)
    published = cache_root / "artifacts" / envelope.manifest_digest
    os.rename(published, cache_root / "missing-tree-simulated")
    audit = RecordingAuditSink(index=index)

    decision = agent.verify_for_runner_start(
        envelope,
        trust_store,
        request,
        audit_sink=audit,
    )

    assert decision.runner_start_allowed is False
    assert decision.reason == "structural_invalid"
    assert decision.refetched is True
    assert [event.event for event in audit.events] == [
        "corruption_quarantined",
        "corruption_refetched",
    ]
    assert audit.verified_at_emit == [False, False]
    recovered = index.get(envelope.manifest_digest)
    assert recovered is not None and recovered.verified is True


def test_interrupted_fill_is_not_published_and_retry_resumes(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    source = RecordingSource(fail_once_path="weights/model.bin")
    cache_root = tmp_path / "cache"
    agent = NodeModelCacheAgent(cache_root, source, chunk_size_bytes=4)

    with pytest.raises(ModelArtifactDownloadError, match="download failed"):
        agent.ensure_cached(envelope, trust_store, request)

    published = cache_root / "artifacts" / envelope.manifest_digest
    partial = cache_root / ".staging" / envelope.manifest_digest / "tree/weights/model.bin"
    assert not published.exists()
    assert partial.read_bytes() == _DATA["weights/model.bin"][:4]

    result = agent.ensure_cached(envelope, trust_store, request)

    _assert_published_content(cache_root, envelope)
    assert result.resumed_bytes == len(_DATA["LICENSE"]) + 4
    assert ("weights/model.bin", 4) in source.calls


def test_digest_mismatch_discards_partial_and_never_publishes(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    source = RecordingSource()
    source_data = _DATA["weights/model.bin"]
    original = _DATA["weights/model.bin"]
    _DATA["weights/model.bin"] = b"x" * len(source_data)
    cache_root = tmp_path / "cache"
    try:
        with pytest.raises(ModelArtifactDigestMismatchError, match="does not match"):
            NodeModelCacheAgent(cache_root, source).ensure_cached(envelope, trust_store, request)
    finally:
        _DATA["weights/model.bin"] = original

    published = cache_root / "artifacts" / envelope.manifest_digest
    partial = cache_root / ".staging" / envelope.manifest_digest / "tree/weights/model.bin"
    assert not published.exists()
    assert not partial.exists()


def test_digest_lock_allows_one_concurrent_fill(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    source = BlockingSource()
    agent = NodeModelCacheAgent(tmp_path / "cache", source, chunk_size_bytes=4)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(agent.ensure_cached, envelope, trust_store, request)
        assert source.started.wait(timeout=5)
        second = pool.submit(agent.ensure_cached, envelope, trust_store, request)
        source.release.set()
        results = (first.result(timeout=5), second.result(timeout=5))

    assert sorted(result.cache_hit for result in results) == [False, True]
    assert len(source.calls) == len(envelope.manifest.files)


def test_digest_lock_timeout_fails_without_second_writer(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    source = BlockingSource()
    cache_root = tmp_path / "cache"
    filling_agent = NodeModelCacheAgent(cache_root, source, chunk_size_bytes=4)
    impatient_agent = NodeModelCacheAgent(
        cache_root,
        source,
        chunk_size_bytes=4,
        lock_timeout_seconds=0,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        filling = pool.submit(
            filling_agent.ensure_cached,
            envelope,
            trust_store,
            request,
        )
        assert source.started.wait(timeout=5)
        try:
            with pytest.raises(NodeModelCacheLockTimeoutError, match="timed out"):
                impatient_agent.ensure_cached(envelope, trust_store, request)
        finally:
            source.release.set()
        filling.result(timeout=5)

    assert len(source.calls) == len(envelope.manifest.files)


def test_digest_lock_serializes_fills_across_processes(tmp_path: Path):
    envelope, _, _ = _signed_artifact()
    cache_root = tmp_path / "cache"
    source_root = tmp_path / "source"
    _write_local_source(source_root, envelope.manifest_digest)
    context = multiprocessing.get_context("fork")
    start = context.Event()
    output = context.Queue()
    processes = [
        context.Process(
            target=_process_fill,
            args=(str(cache_root), str(source_root), start, output),
        )
        for _ in range(2)
    ]

    for process in processes:
        process.start()
    start.set()
    results = [output.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    output.close()
    output.join_thread()

    assert sorted(results) == [("ok", False), ("ok", True)]
    _assert_published_content(cache_root, envelope)


def test_process_crash_releases_digest_lock(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    cache_root = tmp_path / "cache"
    source_root = tmp_path / "source"
    _write_local_source(source_root, envelope.manifest_digest)
    locks = cache_root / ".locks"
    locks.mkdir(mode=0o700, parents=True)
    lock_path = locks / f"{envelope.manifest_digest}.lock"
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_process_lock,
        args=(str(lock_path), ready, release),
    )
    holder.start()
    assert ready.wait(timeout=5)
    source = LocalModelArtifactBlobSource(source_root)

    with pytest.raises(NodeModelCacheLockTimeoutError, match="timed out"):
        NodeModelCacheAgent(
            cache_root,
            source,
            lock_timeout_seconds=0,
        ).ensure_cached(envelope, trust_store, request)

    holder.terminate()
    holder.join(timeout=5)
    assert holder.exitcode is not None
    result = NodeModelCacheAgent(
        cache_root,
        source,
        lock_timeout_seconds=1,
    ).ensure_cached(envelope, trust_store, request)
    assert result.cache_hit is False


def test_modified_published_tree_fails_closed_without_refill(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    source = RecordingSource()
    cache_root = tmp_path / "cache"
    agent = NodeModelCacheAgent(cache_root, source)
    result = agent.ensure_cached(envelope, trust_store, request)
    calls = list(source.calls)
    (result.artifact_path / "LICENSE").write_bytes(b"short")

    with pytest.raises(InvalidNodeModelCacheEntryError, match="does not match"):
        agent.ensure_cached(envelope, trust_store, request)

    assert source.calls == calls


def test_invalid_admission_fails_before_creating_cache_paths(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    source = RecordingSource()
    cache_root = tmp_path / "cache"
    invalid_request = request.model_copy(update={"manifest_digest": "0" * 64})

    with pytest.raises(ModelArtifactAdmissionError, match="digest does not match"):
        NodeModelCacheAgent(cache_root, source).ensure_cached(
            envelope, trust_store, invalid_request
        )

    assert not cache_root.exists()
    assert source.calls == []


def test_invalid_staging_tree_is_reset_before_fill(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    source = RecordingSource()
    cache_root = tmp_path / "cache"
    staging = cache_root / ".staging" / envelope.manifest_digest
    staging.mkdir(parents=True)
    (staging / "manifest.json").write_bytes(
        json.dumps(
            envelope.manifest.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    (staging / "tree").write_text("not a directory")

    NodeModelCacheAgent(cache_root, source).ensure_cached(envelope, trust_store, request)

    _assert_published_content(cache_root, envelope)


def test_unexpected_staging_file_is_never_published(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    source = RecordingSource()
    cache_root = tmp_path / "cache"
    staging = cache_root / ".staging" / envelope.manifest_digest
    tree = staging / "tree"
    tree.mkdir(parents=True)
    (staging / "manifest.json").write_bytes(
        json.dumps(
            envelope.manifest.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    (tree / "unexpected.bin").write_bytes(b"untrusted")

    with pytest.raises(InvalidNodeModelCacheEntryError, match="does not match"):
        NodeModelCacheAgent(cache_root, source).ensure_cached(envelope, trust_store, request)

    assert not (cache_root / "artifacts" / envelope.manifest_digest).exists()
    assert not staging.exists()

    NodeModelCacheAgent(cache_root, source).ensure_cached(envelope, trust_store, request)
    _assert_published_content(cache_root, envelope)


def test_insecure_preexisting_cache_root_is_rejected(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    cache_root = tmp_path / "cache"
    cache_root.mkdir(mode=0o777)
    cache_root.chmod(0o777)

    with pytest.raises(InvalidNodeModelCacheEntryError, match="group/world writable"):
        NodeModelCacheAgent(cache_root, RecordingSource()).ensure_cached(
            envelope, trust_store, request
        )


def test_hardlinked_partial_blob_is_rejected(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    cache_root = tmp_path / "cache"
    staging = cache_root / ".staging" / envelope.manifest_digest
    tree = staging / "tree"
    tree.mkdir(mode=0o700, parents=True)
    (staging / "manifest.json").write_bytes(
        json.dumps(
            envelope.manifest.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    external = tmp_path / "external"
    external.write_bytes(_DATA["LICENSE"][:4])
    os.link(external, tree / "LICENSE")

    with pytest.raises(InvalidNodeModelCacheEntryError, match="unsafe link count"):
        NodeModelCacheAgent(cache_root, RecordingSource()).ensure_cached(
            envelope, trust_store, request
        )

    assert not (cache_root / "artifacts" / envelope.manifest_digest).exists()


def test_symlinked_staging_path_component_is_rejected_and_reset(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    cache_root = tmp_path / "cache"
    staging = cache_root / ".staging" / envelope.manifest_digest
    tree = staging / "tree"
    tree.mkdir(mode=0o700, parents=True)
    (staging / "manifest.json").write_bytes(
        json.dumps(
            envelope.manifest.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (tree / "weights").symlink_to(outside, target_is_directory=True)

    with pytest.raises(InvalidNodeModelCacheEntryError, match="not a directory"):
        NodeModelCacheAgent(cache_root, RecordingSource()).ensure_cached(
            envelope, trust_store, request
        )

    assert not staging.exists()
    assert list(outside.iterdir()) == []


def test_publish_barrier_resyncs_recovered_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    envelope, trust_store, request = _signed_artifact()
    cache_root = tmp_path / "cache"
    source = RecordingSource()
    agent = NodeModelCacheAgent(cache_root, source)
    original = NodeModelCacheAgent._fsync_file
    failed = False

    def fail_completion_once(path: Path) -> None:
        nonlocal failed
        original(path)
        if path.name == "completion.json" and not failed:
            failed = True
            raise InvalidNodeModelCacheEntryError("simulated durability barrier failure")

    monkeypatch.setattr(NodeModelCacheAgent, "_fsync_file", fail_completion_once)
    with pytest.raises(InvalidNodeModelCacheEntryError, match="durability barrier"):
        agent.ensure_cached(envelope, trust_store, request)
    assert not (cache_root / "artifacts" / envelope.manifest_digest).exists()

    synced: list[str] = []

    def record_sync(path: Path) -> None:
        synced.append(path.name)
        original(path)

    monkeypatch.setattr(NodeModelCacheAgent, "_fsync_file", record_sync)
    result = agent.ensure_cached(envelope, trust_store, request)

    assert result.resumed_bytes == sum(map(len, _DATA.values()))
    assert {"LICENSE", "model.bin", "manifest.json", "completion.json"} <= set(synced)


def test_atomic_rename_failure_leaves_recoverable_unpublished_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    envelope, trust_store, request = _signed_artifact()
    cache_root = tmp_path / "cache"
    agent = NodeModelCacheAgent(cache_root, RecordingSource())
    original = os.rename

    def fail_rename(source, target) -> None:
        del source, target
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "rename", fail_rename)
    with pytest.raises(InvalidNodeModelCacheEntryError, match="atomically publish"):
        agent.ensure_cached(envelope, trust_store, request)

    published = cache_root / "artifacts" / envelope.manifest_digest
    staging = cache_root / ".staging" / envelope.manifest_digest
    assert not published.exists()
    assert staging.is_dir()

    monkeypatch.setattr(os, "rename", original)
    result = agent.ensure_cached(envelope, trust_store, request)
    assert result.resumed_bytes == sum(map(len, _DATA.values()))
    _assert_published_content(cache_root, envelope)


def test_duplicate_completion_marker_keys_are_rejected(tmp_path: Path):
    envelope, trust_store, request = _signed_artifact()
    source = RecordingSource()
    cache_root = tmp_path / "cache"
    NodeModelCacheAgent(cache_root, source).ensure_cached(envelope, trust_store, request)
    completion = cache_root / "artifacts" / envelope.manifest_digest / "completion.json"
    completion.write_text(
        '{"schema_version":"kairyu-node-model-cache-completion-v1",'
        f'"manifest_digest":"{envelope.manifest_digest}",'
        f'"manifest_digest":"{envelope.manifest_digest}",'
        f'"file_tree_sha256":"{envelope.manifest.file_tree_sha256}",'
        '"file_count":2,"total_bytes":27}'
    )

    with pytest.raises(InvalidNodeModelCacheEntryError, match="invalid"):
        NodeModelCacheAgent(cache_root, source).ensure_cached(envelope, trust_store, request)


def test_local_source_reads_digest_namespaced_tree(tmp_path: Path):
    envelope, _, _ = _signed_artifact()
    source_root = tmp_path / "source"
    _write_local_source(source_root, envelope.manifest_digest)
    blob = envelope.manifest.files[1]

    content = b"".join(
        LocalModelArtifactBlobSource(source_root).iter_blob(
            manifest_digest=envelope.manifest_digest,
            blob=blob,
            offset=4,
            chunk_size=3,
        )
    )

    assert content == _DATA[blob.path][4:]


def test_local_source_rejects_digest_path_traversal(tmp_path: Path):
    envelope, _, _ = _signed_artifact()
    blob = envelope.manifest.files[0]

    with pytest.raises(ModelArtifactDownloadError, match="manifest digest is invalid"):
        b"".join(
            LocalModelArtifactBlobSource(tmp_path).iter_blob(
                manifest_digest="..",
                blob=blob,
                offset=0,
                chunk_size=4,
            )
        )


def test_http_source_requires_valid_range_response():
    envelope, _, _ = _signed_artifact()
    blob = envelope.manifest.files[1]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Range"] == "bytes=4-"
        assert request.url.path.endswith(f"/{envelope.manifest_digest}/weights/model.bin")
        content = _DATA[blob.path][4:]
        return httpx.Response(
            206,
            headers={
                "Content-Range": f"bytes 4-{blob.size_bytes - 1}/{blob.size_bytes}",
                "Content-Length": str(len(content)),
            },
            content=content,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        source = HttpRangeModelArtifactBlobSource("https://objects.invalid/models", client=client)
        content = b"".join(
            source.iter_blob(
                manifest_digest=envelope.manifest_digest,
                blob=blob,
                offset=4,
                chunk_size=3,
            )
        )

    assert content == _DATA[blob.path][4:]


def test_http_source_rejects_server_ignoring_resume_range():
    envelope, _, _ = _signed_artifact()
    blob = envelope.manifest.files[1]
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=_DATA[blob.path], request=request)
    )

    with httpx.Client(transport=transport) as client:
        source = HttpRangeModelArtifactBlobSource("https://objects.invalid", client=client)
        with pytest.raises(ModelArtifactDownloadError, match="did not honor"):
            b"".join(
                source.iter_blob(
                    manifest_digest=envelope.manifest_digest,
                    blob=blob,
                    offset=4,
                    chunk_size=3,
                )
            )
