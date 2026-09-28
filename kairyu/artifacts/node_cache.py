"""Node-local verified model cache fill and recovery agent (WP4.2/WP4.6).

Only a fully downloaded and digest-verified staging directory is atomically
renamed into the published cache namespace. Runner-start verification can
quarantine later corruption and replace it from the authoritative source.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self
from urllib.parse import quote

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from kairyu.artifacts.manifest import (
    ModelArtifactAdmission,
    ModelArtifactAdmissionRequest,
    ModelArtifactBlob,
    ModelArtifactTrustStore,
    SignedModelArtifactManifest,
    admit_model_artifact,
    canonical_model_manifest_bytes,
)

if TYPE_CHECKING:
    from kairyu.artifacts.cache_index import NodeModelCacheIndex

_MAX_SIGNED_BIGINT = 2**63 - 1
_MAX_COMPLETION_BYTES = 64 * 1024
_DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
_CONTENT_RANGE_PATTERN = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+)$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _validate_blob_read(
    *,
    manifest_digest: str,
    blob: ModelArtifactBlob,
    offset: int,
    chunk_size: int,
) -> None:
    if not _SHA256_PATTERN.fullmatch(manifest_digest):
        raise ModelArtifactDownloadError("artifact manifest digest is invalid")
    if type(offset) is not int or offset < 0 or offset > blob.size_bytes:
        raise ModelArtifactDownloadError("artifact resume offset is invalid")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ModelArtifactDownloadError("artifact chunk size is invalid")


class NodeModelCacheError(RuntimeError):
    """Base class for node-cache fill failures."""


class NodeModelCacheLockTimeoutError(NodeModelCacheError):
    """The digest-specific inter-process fill lock could not be acquired."""


class InvalidNodeModelCacheEntryError(NodeModelCacheError):
    """A staging or published cache entry violates the cache contract."""


class ModelArtifactDownloadError(NodeModelCacheError):
    """The artifact source could not provide the declared remaining bytes."""


class ModelArtifactDigestMismatchError(NodeModelCacheError):
    """A downloaded blob did not match its signed digest."""


class NodeModelCacheAuditError(NodeModelCacheError):
    """A corruption transition could not be written to the required audit sink."""


class NodeModelCacheRecoveryError(NodeModelCacheError):
    """A quarantined artifact could not be replaced by a verified refill."""


def _require_secure_cache_directory(path_stat: os.stat_result, *, name: str) -> None:
    if not stat.S_ISDIR(path_stat.st_mode):
        raise InvalidNodeModelCacheEntryError(f"{name} is not a directory")
    if path_stat.st_uid != os.geteuid():
        raise InvalidNodeModelCacheEntryError(f"{name} is not owned by the cache user")
    if stat.S_IMODE(path_stat.st_mode) & 0o022:
        raise InvalidNodeModelCacheEntryError(f"{name} is group/world writable")


def _require_secure_cache_file(path_stat: os.stat_result, *, name: str) -> None:
    if not stat.S_ISREG(path_stat.st_mode):
        raise InvalidNodeModelCacheEntryError(f"{name} is not a regular file")
    if path_stat.st_uid != os.geteuid():
        raise InvalidNodeModelCacheEntryError(f"{name} is not owned by the cache user")
    if path_stat.st_nlink != 1:
        raise InvalidNodeModelCacheEntryError(f"{name} has an unsafe link count")
    if stat.S_IMODE(path_stat.st_mode) & 0o022:
        raise InvalidNodeModelCacheEntryError(f"{name} is group/world writable")


class ModelArtifactBlobSource(Protocol):
    """Resumable immutable blob source used by :class:`NodeModelCacheAgent`."""

    def iter_blob(
        self,
        *,
        manifest_digest: str,
        blob: ModelArtifactBlob,
        offset: int,
        chunk_size: int,
    ) -> Iterable[bytes]:
        """Yield bytes starting exactly at ``offset`` until EOF."""


class NodeModelCacheCompletion(BaseModel):
    """Durable marker written before the staging directory is published."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-cache-completion-v1"] = (
        "kairyu-node-model-cache-completion-v1"
    )
    manifest_digest: str = Field(min_length=64, max_length=64)
    file_tree_sha256: str = Field(min_length=64, max_length=64)
    file_count: int = Field(ge=1, le=100_000)
    total_bytes: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)

    @field_validator("file_count", "total_bytes", mode="before")
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        return value

    @field_validator("manifest_digest", "file_tree_sha256")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest")
        return value


class NodeModelCacheFillResult(BaseModel):
    """Evidence returned after a verified published tree is available."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-cache-fill-result-v1"] = (
        "kairyu-node-model-cache-fill-result-v1"
    )
    deployment_id: str
    manifest_digest: str
    artifact_path: Path
    cache_hit: bool
    resumed_bytes: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    downloaded_bytes: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    file_count: int = Field(ge=1, le=100_000)
    total_bytes: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)

    @field_validator("deployment_id")
    @classmethod
    def validate_deployment_id(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("deployment_id must be a non-empty string without NUL")
        return value

    @field_validator("manifest_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("manifest_digest must be a lowercase SHA-256 digest")
        return value

    @field_validator("cache_hit", mode="before")
    @classmethod
    def validate_cache_hit(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("cache_hit must be a boolean")
        return value

    @field_validator(
        "resumed_bytes",
        "downloaded_bytes",
        "file_count",
        "total_bytes",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        return value


class NodeModelCacheCorruptionAuditEvent(BaseModel):
    """Bounded audit evidence for one corruption recovery transition."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-cache-corruption-audit-v1"] = (
        "kairyu-node-model-cache-corruption-audit-v1"
    )
    event: Literal[
        "corruption_quarantined",
        "corruption_refetched",
        "corruption_refetch_failed",
    ]
    at_ns: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    node_id: str = Field(min_length=1, max_length=255)
    deployment_id: str = Field(min_length=1, max_length=255)
    manifest_digest: str = Field(min_length=64, max_length=64)
    model_id: str = Field(min_length=1, max_length=255)
    model_revision: str = Field(min_length=1, max_length=255)
    recovery_id: str = Field(min_length=64, max_length=64)
    record_generation: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    reason: Literal[
        "digest_mismatch",
        "structural_invalid",
        "index_unverified",
        "verified_refetch",
        "refetch_failed",
    ]
    quarantine_path: Path

    @field_validator("at_ns", "record_generation", mode="before")
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        return value

    @field_validator("node_id", "deployment_id", "model_id", "model_revision")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError(f"{info.field_name} must be a non-empty string without NUL")
        return value

    @field_validator("manifest_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("manifest_digest must be a lowercase SHA-256 digest")
        return value

    @field_validator("recovery_id")
    @classmethod
    def validate_recovery_id(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("recovery_id must be a lowercase 256-bit identifier")
        return value

    @field_validator("quarantine_path", mode="before")
    @classmethod
    def validate_quarantine_path(cls, value: object) -> object:
        path = Path(value) if isinstance(value, (str, Path)) else None
        if path is None or not path.is_absolute() or "\x00" in str(path):
            raise ValueError("quarantine_path must be an absolute path without NUL")
        return path


class NodeModelCacheAuditSink(Protocol):
    """Required synchronous sink for corruption state transitions."""

    def emit(self, event: NodeModelCacheCorruptionAuditEvent) -> None:
        """Persist one event or raise; silent drops are forbidden."""


class NodeModelCacheRunnerStartDecision(BaseModel):
    """Fail-closed result of full cache verification before Runner startup."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-cache-runner-start-v1"] = (
        "kairyu-node-model-cache-runner-start-v1"
    )
    runner_start_allowed: bool
    manifest_digest: str = Field(min_length=64, max_length=64)
    artifact_path: Path | None = None
    corruption_detected: bool
    refetched: bool
    quarantine_path: Path | None = None
    reason: Literal["verified", "digest_mismatch", "structural_invalid", "index_unverified"]

    @field_validator(
        "runner_start_allowed",
        "corruption_detected",
        "refetched",
        mode="before",
    )
    @classmethod
    def validate_boolean(cls, value: object, info) -> object:
        if type(value) is not bool:
            raise ValueError(f"{info.field_name} must be a boolean")
        return value

    @field_validator("manifest_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("manifest_digest must be a lowercase SHA-256 digest")
        return value

    @field_validator("artifact_path", "quarantine_path", mode="before")
    @classmethod
    def validate_optional_path(cls, value: object, info) -> object:
        if value is None:
            return value
        path = Path(value) if isinstance(value, (str, Path)) else None
        if path is None or not path.is_absolute() or "\x00" in str(path):
            raise ValueError(f"{info.field_name} must be an absolute path without NUL")
        return path

    @model_validator(mode="after")
    def validate_decision(self) -> NodeModelCacheRunnerStartDecision:
        if self.runner_start_allowed:
            if (
                self.corruption_detected
                or self.refetched
                or self.quarantine_path is not None
                or self.artifact_path is None
                or self.reason != "verified"
            ):
                raise ValueError("allowed Runner start must contain only verified evidence")
        elif (
            not self.corruption_detected
            or self.quarantine_path is None
            or self.artifact_path is not None
            or self.reason == "verified"
        ):
            raise ValueError("denied Runner start must contain quarantine evidence")
        return self


class LocalModelArtifactBlobSource:
    """Read an immutable ``<root>/<digest>/<blob path>`` source tree.

    This adapter is useful for tests, offline staging, and a read-only mounted
    object-store gateway.  It never follows a final-component symlink.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def iter_blob(
        self,
        *,
        manifest_digest: str,
        blob: ModelArtifactBlob,
        offset: int,
        chunk_size: int,
    ) -> Iterator[bytes]:
        _validate_blob_read(
            manifest_digest=manifest_digest,
            blob=blob,
            offset=offset,
            chunk_size=chunk_size,
        )
        path = self._root / manifest_digest / Path(*blob.path.split("/"))
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as exc:
            raise ModelArtifactDownloadError("cannot open artifact source blob") from exc
        try:
            source_stat = os.fstat(descriptor)
            if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_size != blob.size_bytes:
                raise ModelArtifactDownloadError("artifact source blob size is invalid")
            if offset < 0 or offset > source_stat.st_size:
                raise ModelArtifactDownloadError("artifact resume offset is invalid")
            os.lseek(descriptor, offset, os.SEEK_SET)
            while True:
                chunk = os.read(descriptor, chunk_size)
                if not chunk:
                    return
                yield chunk
        finally:
            os.close(descriptor)


class HttpRangeModelArtifactBlobSource:
    """HTTP Range adapter for an S3-compatible immutable object layout.

    Objects are addressed as ``<base_url>/<manifest digest>/<blob path>``.
    Supply a preconfigured ``httpx.Client`` for authentication, TLS policy, or
    transport customization.  Resumed reads require a valid 206 Content-Range.
    """

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        if not base_url.strip() or "\x00" in base_url:
            raise ValueError("base_url must be a non-empty string without NUL")
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool):
            raise ValueError("timeout_seconds must be a number")
        if not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=float(timeout_seconds))

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _blob_url(self, manifest_digest: str, blob: ModelArtifactBlob) -> str:
        encoded_path = "/".join(quote(part, safe="") for part in blob.path.split("/"))
        return f"{self._base_url}/{quote(manifest_digest, safe='')}/{encoded_path}"

    def iter_blob(
        self,
        *,
        manifest_digest: str,
        blob: ModelArtifactBlob,
        offset: int,
        chunk_size: int,
    ) -> Iterator[bytes]:
        _validate_blob_read(
            manifest_digest=manifest_digest,
            blob=blob,
            offset=offset,
            chunk_size=chunk_size,
        )
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with self._client.stream(
                "GET",
                self._blob_url(manifest_digest, blob),
                headers=headers,
            ) as response:
                if offset:
                    self._validate_range_response(response, blob=blob, offset=offset)
                elif response.status_code != 200:
                    raise ModelArtifactDownloadError("artifact source rejected full download")
                self._validate_content_length(response, expected=blob.size_bytes - offset)
                for chunk in response.iter_bytes(chunk_size=chunk_size):
                    if chunk:
                        yield chunk
        except ModelArtifactDownloadError:
            raise
        except (httpx.HTTPError, OSError) as exc:
            raise ModelArtifactDownloadError("artifact source request failed") from exc

    @staticmethod
    def _validate_range_response(
        response: httpx.Response,
        *,
        blob: ModelArtifactBlob,
        offset: int,
    ) -> None:
        if response.status_code != 206:
            raise ModelArtifactDownloadError("artifact source did not honor resume range")
        match = _CONTENT_RANGE_PATTERN.fullmatch(response.headers.get("Content-Range", ""))
        if match is None:
            raise ModelArtifactDownloadError("artifact source range metadata is invalid")
        start, end, total = (int(value) for value in match.groups())
        if start != offset or end < start or total != blob.size_bytes:
            raise ModelArtifactDownloadError("artifact source range metadata does not match blob")
        if end - start + 1 != blob.size_bytes - offset:
            raise ModelArtifactDownloadError("artifact source range length does not match blob")

    @staticmethod
    def _validate_content_length(response: httpx.Response, *, expected: int) -> None:
        value = response.headers.get("Content-Length")
        if value is None:
            return
        try:
            actual = int(value)
        except ValueError as exc:
            raise ModelArtifactDownloadError("artifact source content length is invalid") from exc
        if actual != expected:
            raise ModelArtifactDownloadError("artifact source content length does not match blob")


class NodeModelCacheAgent:
    """Fill and atomically publish one verified model tree per manifest digest."""

    def __init__(
        self,
        root: Path,
        source: ModelArtifactBlobSource,
        *,
        index: NodeModelCacheIndex | None = None,
        chunk_size_bytes: int = _DEFAULT_CHUNK_BYTES,
        lock_timeout_seconds: float | None = None,
    ) -> None:
        if type(chunk_size_bytes) is not int or not 1 <= chunk_size_bytes <= 64 * 1024 * 1024:
            raise ValueError("chunk_size_bytes must be an integer from 1 to 67108864")
        if lock_timeout_seconds is not None:
            if isinstance(lock_timeout_seconds, bool) or not isinstance(
                lock_timeout_seconds, (int, float)
            ):
                raise ValueError("lock_timeout_seconds must be a number")
            if not math.isfinite(float(lock_timeout_seconds)) or lock_timeout_seconds < 0:
                raise ValueError("lock_timeout_seconds must be finite and non-negative")
        self._root = root
        self._source = source
        self._index = index
        self._chunk_size_bytes = chunk_size_bytes
        self._lock_timeout_seconds = (
            None if lock_timeout_seconds is None else float(lock_timeout_seconds)
        )

    def ensure_cached(
        self,
        envelope: SignedModelArtifactManifest,
        trust_store: ModelArtifactTrustStore,
        request: ModelArtifactAdmissionRequest,
    ) -> NodeModelCacheFillResult:
        """Verify admission, resume any staging tree, and atomically publish it."""

        admission = admit_model_artifact(envelope, trust_store, request)
        manifest = envelope.manifest
        digest = admission.manifest_digest
        self._prepare_root()
        lock_path = self._root / ".locks" / f"{digest}.lock"
        with self._exclusive_lock(lock_path):
            # Revalidate the immutable inputs after waiting for another filler.
            admission = admit_model_artifact(envelope, trust_store, request)
            published = self._root / "artifacts" / digest
            if published.exists() or published.is_symlink():
                self._validate_published(published, envelope)
                result = self._result(
                    admission=admission,
                    envelope=envelope,
                    artifact_path=published / "tree",
                    cache_hit=True,
                    resumed_bytes=0,
                    downloaded_bytes=0,
                )
                self._record_index(
                    result,
                    envelope=envelope,
                    verification_source="published_marker",
                )
                return result

            staging = self._prepare_staging(envelope)
            tree = staging / "tree"
            resumed_bytes = 0
            downloaded_bytes = 0
            try:
                for blob in manifest.files:
                    resumed, downloaded = self._fill_blob(
                        tree=tree,
                        manifest_digest=digest,
                        blob=blob,
                    )
                    resumed_bytes += resumed
                    downloaded_bytes += downloaded
            except InvalidNodeModelCacheEntryError:
                self._discard_staging(staging)
                raise

            try:
                self._validate_artifact_tree(tree, manifest.files)
            except InvalidNodeModelCacheEntryError:
                self._discard_staging(staging)
                raise
            completion = self._completion(envelope)
            try:
                self._write_or_validate_bytes(
                    staging / "completion.json",
                    self._canonical_json(completion.model_dump(mode="json")),
                )
            except InvalidNodeModelCacheEntryError:
                self._discard_staging(staging)
                raise
            self._fsync_staging(staging, manifest.files)
            try:
                os.rename(staging, published)
            except OSError as exc:
                raise InvalidNodeModelCacheEntryError(
                    "cannot atomically publish verified artifact"
                ) from exc
            self._fsync_directory(published.parent)
            self._validate_published(published, envelope)
            result = self._result(
                admission=admission,
                envelope=envelope,
                artifact_path=published / "tree",
                cache_hit=False,
                resumed_bytes=resumed_bytes,
                downloaded_bytes=downloaded_bytes,
            )
            self._record_index(
                result,
                envelope=envelope,
                verification_source="filled",
            )
            return result

    def verify_for_runner_start(
        self,
        envelope: SignedModelArtifactManifest,
        trust_store: ModelArtifactTrustStore,
        request: ModelArtifactAdmissionRequest,
        *,
        audit_sink: NodeModelCacheAuditSink,
    ) -> NodeModelCacheRunnerStartDecision:
        """Hash every resident blob; quarantine/refetch corruption but deny this start."""

        if self._index is None:
            raise NodeModelCacheError("Runner-start verification requires a cache index")
        emitter = getattr(audit_sink, "emit", None)
        if not callable(emitter):
            raise TypeError("audit_sink must provide a callable emit method")
        admission = admit_model_artifact(envelope, trust_store, request)
        digest = admission.manifest_digest
        published = self._root / "artifacts" / digest
        self._prepare_root()
        quarantine_path: Path | None = None
        reason: Literal[
            "digest_mismatch",
            "structural_invalid",
            "index_unverified",
        ]
        recovery_id = ""
        recovery_generation = 0
        replacement_ready = False
        with self._exclusive_lock(self._root / ".locks" / f"{digest}.lock"):
            admission = admit_model_artifact(envelope, trust_store, request)
            record = self._index.get(digest)
            if record is None:
                from kairyu.artifacts.cache_index import (
                    NodeModelCacheIndexEntryNotFoundError,
                )

                raise NodeModelCacheIndexEntryNotFoundError(
                    "Runner-start verification requires indexed residency"
                )
            expected_identity = (
                envelope.manifest.model_id,
                envelope.manifest.model_revision,
                published / "tree",
                sum(blob.size_bytes for blob in envelope.manifest.files),
                len(envelope.manifest.files),
            )
            actual_identity = (
                record.model_id,
                record.model_revision,
                record.artifact_path,
                record.total_bytes,
                record.file_count,
            )
            if actual_identity != expected_identity:
                from kairyu.artifacts.cache_index import (
                    NodeModelCacheIndexIdentityError,
                )

                raise NodeModelCacheIndexIdentityError(
                    "indexed residency does not match Runner-start manifest"
                )
            if not record.verified:
                reason = (
                    record.verification_failure
                    if record.verification_failure
                    in {"digest_mismatch", "structural_invalid", "index_unverified"}
                    else "index_unverified"
                )
            else:
                try:
                    self._validate_published(published, envelope)
                    self._verify_artifact_digests(
                        published / "tree",
                        envelope.manifest.files,
                    )
                except ModelArtifactDigestMismatchError:
                    reason = "digest_mismatch"
                except InvalidNodeModelCacheEntryError:
                    reason = "structural_invalid"
                else:
                    self._index.touch(digest)
                    return NodeModelCacheRunnerStartDecision(
                        runner_start_allowed=True,
                        manifest_digest=digest,
                        artifact_path=published / "tree",
                        corruption_detected=False,
                        refetched=False,
                        reason="verified",
                    )

            recovery = self._index.begin_recovery(digest, reason=reason)
            if recovery.recovery_id is None:
                raise NodeModelCacheRecoveryError(
                    "cache index did not persist a recovery identifier"
                )
            recovery_id = recovery.recovery_id
            recovery_generation = recovery.generation
            quarantine_root = self._root / ".quarantine"
            try:
                quarantine_root.mkdir(mode=0o700, exist_ok=True)
                _require_secure_cache_directory(
                    quarantine_root.stat(follow_symlinks=False),
                    name="cache quarantine path",
                )
            except (OSError, InvalidNodeModelCacheEntryError) as exc:
                raise InvalidNodeModelCacheEntryError(
                    "cannot prepare cache quarantine path"
                ) from exc
            incident_path = quarantine_root / f"{digest}.{recovery_id}"
            if published.exists() or published.is_symlink():
                if record.recovery_id == recovery_id and (
                    incident_path.exists() or incident_path.is_symlink()
                ):
                    try:
                        self._validate_published(published, envelope)
                        self._verify_artifact_digests(
                            published / "tree",
                            envelope.manifest.files,
                        )
                    except (
                        ModelArtifactDigestMismatchError,
                        InvalidNodeModelCacheEntryError,
                    ):
                        pass
                    else:
                        replacement_ready = True
                        quarantine_path = incident_path
                if not replacement_ready:
                    quarantine_path = incident_path
                    if quarantine_path.exists() or quarantine_path.is_symlink():
                        quarantine_path = quarantine_root / (
                            f"{digest}.{recovery_id}.retry.{secrets.token_hex(16)}"
                        )
                    try:
                        os.rename(published, quarantine_path)
                    except OSError as exc:
                        raise InvalidNodeModelCacheEntryError(
                            "cannot quarantine corrupt cache entry"
                        ) from exc
                    self._fsync_directory(published.parent)
                    self._fsync_directory(quarantine_root)
            else:
                quarantine_path = incident_path
                if not quarantine_path.exists() and not quarantine_path.is_symlink():
                    try:
                        quarantine_path.mkdir(mode=0o700)
                    except OSError as exc:
                        raise InvalidNodeModelCacheEntryError(
                            "cannot persist missing-residency quarantine marker"
                        ) from exc
                    self._fsync_directory(quarantine_root)
            self._emit_corruption_audit(
                audit_sink,
                event="corruption_quarantined",
                reason=reason,
                admission=admission,
                recovery_id=recovery_id,
                generation=recovery_generation,
                quarantine_path=quarantine_path,
            )

        assert quarantine_path is not None
        if not replacement_ready:
            try:
                refill_agent = NodeModelCacheAgent(
                    self._root,
                    self._source,
                    index=None,
                    chunk_size_bytes=self._chunk_size_bytes,
                    lock_timeout_seconds=self._lock_timeout_seconds,
                )
                refill_agent.ensure_cached(envelope, trust_store, request)
            except Exception as exc:
                self._emit_corruption_audit(
                    audit_sink,
                    event="corruption_refetch_failed",
                    reason="refetch_failed",
                    admission=admission,
                    recovery_id=recovery_id,
                    generation=recovery_generation,
                    quarantine_path=quarantine_path,
                )
                raise NodeModelCacheRecoveryError(
                    "corrupt cache entry was quarantined but verified refill failed"
                ) from exc

        with self._exclusive_lock(self._root / ".locks" / f"{digest}.lock"):
            current = self._index.get(digest)
            if (
                current is None
                or current.verified
                or current.recovery_id != recovery_id
                or current.generation != recovery_generation
            ):
                raise NodeModelCacheRecoveryError(
                    "cache recovery fence changed before audit completion"
                )
            try:
                self._validate_published(published, envelope)
                self._verify_artifact_digests(
                    published / "tree",
                    envelope.manifest.files,
                )
            except (
                ModelArtifactDigestMismatchError,
                InvalidNodeModelCacheEntryError,
            ) as exc:
                self._emit_corruption_audit(
                    audit_sink,
                    event="corruption_refetch_failed",
                    reason="refetch_failed",
                    admission=admission,
                    recovery_id=recovery_id,
                    generation=recovery_generation,
                    quarantine_path=quarantine_path,
                )
                raise NodeModelCacheRecoveryError(
                    "replacement failed final Runner-start verification"
                ) from exc
            self._emit_corruption_audit(
                audit_sink,
                event="corruption_refetched",
                reason="verified_refetch",
                admission=admission,
                recovery_id=recovery_id,
                generation=recovery_generation,
                quarantine_path=quarantine_path,
            )
            self._index.complete_recovery(
                manifest_digest=digest,
                recovery_id=recovery_id,
                expected_generation=recovery_generation,
                model_id=envelope.manifest.model_id,
                model_revision=envelope.manifest.model_revision,
                artifact_path=published / "tree",
                total_bytes=sum(blob.size_bytes for blob in envelope.manifest.files),
                file_count=len(envelope.manifest.files),
            )
        return NodeModelCacheRunnerStartDecision(
            runner_start_allowed=False,
            manifest_digest=digest,
            corruption_detected=True,
            refetched=True,
            quarantine_path=quarantine_path,
            reason=reason,
        )

    @staticmethod
    def _verify_artifact_digests(
        tree: Path,
        blobs: tuple[ModelArtifactBlob, ...],
    ) -> None:
        for blob in blobs:
            path = tree / Path(*blob.path.split("/"))
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(path, flags)
            except OSError as exc:
                raise InvalidNodeModelCacheEntryError(
                    "cannot open artifact blob for Runner-start verification"
                ) from exc
            digest = hashlib.sha256()
            try:
                blob_stat = os.fstat(descriptor)
                _require_secure_cache_file(blob_stat, name="artifact blob")
                if blob_stat.st_size != blob.size_bytes:
                    raise InvalidNodeModelCacheEntryError(
                        "artifact blob size changed before Runner start"
                    )
                while True:
                    chunk = os.read(descriptor, _DEFAULT_CHUNK_BYTES)
                    if not chunk:
                        break
                    digest.update(chunk)
            except InvalidNodeModelCacheEntryError:
                raise
            except OSError as exc:
                raise InvalidNodeModelCacheEntryError(
                    "cannot hash artifact blob before Runner start"
                ) from exc
            finally:
                os.close(descriptor)
            if digest.hexdigest() != blob.sha256:
                raise ModelArtifactDigestMismatchError(
                    "resident blob digest does not match signed manifest"
                )

    def _emit_corruption_audit(
        self,
        sink: NodeModelCacheAuditSink,
        *,
        event: Literal[
            "corruption_quarantined",
            "corruption_refetched",
            "corruption_refetch_failed",
        ],
        reason: Literal[
            "digest_mismatch",
            "structural_invalid",
            "index_unverified",
            "verified_refetch",
            "refetch_failed",
        ],
        admission: ModelArtifactAdmission,
        recovery_id: str,
        generation: int,
        quarantine_path: Path,
    ) -> None:
        assert self._index is not None
        try:
            sink.emit(
                NodeModelCacheCorruptionAuditEvent(
                    event=event,
                    at_ns=time.time_ns(),
                    node_id=self._index.node_id,
                    deployment_id=admission.deployment_id,
                    manifest_digest=admission.manifest_digest,
                    model_id=admission.model_id,
                    model_revision=admission.model_revision,
                    recovery_id=recovery_id,
                    record_generation=generation,
                    reason=reason,
                    quarantine_path=quarantine_path,
                )
            )
        except Exception as exc:
            raise NodeModelCacheAuditError("cache corruption audit sink rejected an event") from exc

    def _prepare_root(self) -> None:
        try:
            self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
            root_stat = self._root.stat(follow_symlinks=False)
            _require_secure_cache_directory(root_stat, name="cache root")
            for name in ("artifacts", ".staging", ".locks"):
                path = self._root / name
                path.mkdir(mode=0o700, exist_ok=True)
                path_stat = path.stat(follow_symlinks=False)
                _require_secure_cache_directory(path_stat, name="cache control path")
        except InvalidNodeModelCacheEntryError:
            raise
        except OSError as exc:
            raise InvalidNodeModelCacheEntryError("cannot prepare node cache root") from exc

    @contextlib.contextmanager
    def _exclusive_lock(self, path: Path) -> Iterator[None]:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise InvalidNodeModelCacheEntryError("cannot open cache fill lock") from exc
        try:
            lock_stat = os.fstat(descriptor)
            _require_secure_cache_file(lock_stat, name="cache fill lock")
            if self._lock_timeout_seconds is None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            else:
                deadline = time.monotonic() + self._lock_timeout_seconds
                while True:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError as exc:
                        if time.monotonic() >= deadline:
                            raise NodeModelCacheLockTimeoutError(
                                "timed out waiting for cache fill lock"
                            ) from exc
                        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _prepare_staging(self, envelope: SignedModelArtifactManifest) -> Path:
        digest = envelope.manifest_digest
        staging = self._root / ".staging" / digest
        expected_manifest = canonical_model_manifest_bytes(envelope.manifest)
        manifest_path = staging / "manifest.json"
        if staging.exists() or staging.is_symlink():
            reset = staging.is_symlink() or not staging.is_dir()
            if not reset:
                try:
                    staging_stat = staging.stat(follow_symlinks=False)
                    _require_secure_cache_directory(staging_stat, name="staging directory")
                except (OSError, InvalidNodeModelCacheEntryError):
                    reset = True
            if not reset:
                try:
                    existing = self._read_bounded(
                        manifest_path,
                        max_bytes=8 * 1024 * 1024,
                    )
                except InvalidNodeModelCacheEntryError:
                    reset = True
                else:
                    reset = existing != expected_manifest
            if not reset:
                tree = staging / "tree"
                try:
                    tree_stat = tree.stat(follow_symlinks=False)
                except OSError:
                    reset = True
                else:
                    try:
                        _require_secure_cache_directory(tree_stat, name="staging tree")
                    except InvalidNodeModelCacheEntryError:
                        reset = True
            if reset:
                self._discard_staging(staging)
        if not staging.exists():
            try:
                (staging / "tree").mkdir(mode=0o700, parents=True)
            except OSError as exc:
                raise InvalidNodeModelCacheEntryError("cannot create staging tree") from exc
            self._write_bytes(manifest_path, expected_manifest)
        tree = staging / "tree"
        try:
            tree_stat = tree.stat(follow_symlinks=False)
        except OSError as exc:
            raise InvalidNodeModelCacheEntryError("staging tree is unavailable") from exc
        _require_secure_cache_directory(tree_stat, name="staging tree")
        return staging

    @staticmethod
    def _discard_staging(path: Path) -> None:
        try:
            if path.is_symlink():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)
        except OSError as exc:
            raise InvalidNodeModelCacheEntryError("cannot reset invalid staging tree") from exc

    def _fill_blob(
        self,
        *,
        tree: Path,
        manifest_digest: str,
        blob: ModelArtifactBlob,
    ) -> tuple[int, int]:
        target = self._safe_blob_target(tree, blob.path)
        descriptor, existing_size, digest = self._open_and_hash_partial(target, blob)
        if existing_size == blob.size_bytes:
            if digest.hexdigest() == blob.sha256:
                try:
                    os.fsync(descriptor)
                except OSError as exc:
                    raise InvalidNodeModelCacheEntryError(
                        "cannot sync recovered staging blob"
                    ) from exc
                finally:
                    os.close(descriptor)
                return (existing_size, 0)
            os.close(descriptor)
            self._unlink_partial(target)
            descriptor, existing_size, digest = self._open_and_hash_partial(target, blob)

        resumed_bytes = existing_size
        downloaded_bytes = 0
        try:
            try:
                if blob.size_bytes:
                    chunks = self._source.iter_blob(
                        manifest_digest=manifest_digest,
                        blob=blob,
                        offset=existing_size,
                        chunk_size=self._chunk_size_bytes,
                    )
                    for chunk in chunks:
                        if not isinstance(chunk, bytes):
                            raise ModelArtifactDownloadError(
                                "artifact source yielded a non-bytes chunk"
                            )
                        if not chunk:
                            continue
                        if existing_size + downloaded_bytes + len(chunk) > blob.size_bytes:
                            raise ModelArtifactDownloadError(
                                "artifact source exceeded declared blob size"
                            )
                        self._write_all(descriptor, chunk)
                        digest.update(chunk)
                        downloaded_bytes += len(chunk)
                os.fsync(descriptor)
            except ModelArtifactDownloadError:
                raise
            except Exception as exc:
                raise ModelArtifactDownloadError("artifact source download failed") from exc
        finally:
            os.close(descriptor)

        final_size = existing_size + downloaded_bytes
        if final_size != blob.size_bytes:
            raise ModelArtifactDownloadError("artifact source ended before declared blob size")
        if digest.hexdigest() != blob.sha256:
            self._unlink_partial(target)
            raise ModelArtifactDigestMismatchError("downloaded blob digest does not match manifest")
        return (resumed_bytes, downloaded_bytes)

    def _safe_blob_target(self, tree: Path, relative: str) -> Path:
        components = relative.split("/")
        parent = tree
        for component in components[:-1]:
            parent = parent / component
            try:
                parent.mkdir(mode=0o700)
            except FileExistsError:
                pass
            except OSError as exc:
                raise InvalidNodeModelCacheEntryError("cannot create staging directory") from exc
            try:
                parent_stat = parent.stat(follow_symlinks=False)
            except OSError as exc:
                raise InvalidNodeModelCacheEntryError("cannot inspect staging directory") from exc
            _require_secure_cache_directory(
                parent_stat,
                name="staging path component",
            )
        return parent / components[-1]

    @staticmethod
    def _open_and_hash_partial(
        path: Path,
        blob: ModelArtifactBlob,
    ) -> tuple[int, int, Any]:
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise InvalidNodeModelCacheEntryError("cannot open staging blob") from exc
        digest = hashlib.sha256()
        try:
            partial_stat = os.fstat(descriptor)
            _require_secure_cache_file(partial_stat, name="partial staging blob")
            if partial_stat.st_size > blob.size_bytes:
                os.close(descriptor)
                NodeModelCacheAgent._unlink_partial(path)
                return NodeModelCacheAgent._open_and_hash_partial(path, blob)
            os.lseek(descriptor, 0, os.SEEK_SET)
            while True:
                chunk = os.read(descriptor, _DEFAULT_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
            return (descriptor, partial_stat.st_size, digest)
        except InvalidNodeModelCacheEntryError:
            with contextlib.suppress(OSError):
                os.close(descriptor)
            raise
        except OSError as exc:
            with contextlib.suppress(OSError):
                os.close(descriptor)
            raise InvalidNodeModelCacheEntryError("cannot read partial staging blob") from exc

    @staticmethod
    def _write_all(descriptor: int, content: bytes) -> None:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(errno.EIO, "short write")
            view = view[written:]

    @staticmethod
    def _unlink_partial(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise InvalidNodeModelCacheEntryError("cannot discard invalid partial blob") from exc

    def _validate_published(
        self,
        published: Path,
        envelope: SignedModelArtifactManifest,
    ) -> None:
        try:
            published_stat = published.stat(follow_symlinks=False)
        except OSError as exc:
            raise InvalidNodeModelCacheEntryError("published cache entry is unavailable") from exc
        _require_secure_cache_directory(published_stat, name="published cache entry")
        manifest_bytes = self._read_bounded(
            published / "manifest.json",
            max_bytes=8 * 1024 * 1024,
        )
        if manifest_bytes != canonical_model_manifest_bytes(envelope.manifest):
            raise InvalidNodeModelCacheEntryError(
                "published manifest does not match requested digest"
            )
        completion = self._load_completion(published / "completion.json")
        if completion != self._completion(envelope):
            raise InvalidNodeModelCacheEntryError(
                "published completion marker does not match manifest"
            )
        self._validate_artifact_tree(published / "tree", envelope.manifest.files)

    @staticmethod
    def _validate_artifact_tree(tree: Path, blobs: tuple[ModelArtifactBlob, ...]) -> None:
        try:
            tree_stat = tree.stat(follow_symlinks=False)
        except OSError as exc:
            raise InvalidNodeModelCacheEntryError("published artifact tree is unavailable") from exc
        _require_secure_cache_directory(tree_stat, name="artifact tree")
        expected = {blob.path: blob for blob in blobs}
        observed: set[str] = set()

        def raise_walk_error(error: OSError) -> None:
            raise error

        try:
            for directory, dirnames, filenames in os.walk(
                tree,
                followlinks=False,
                onerror=raise_walk_error,
            ):
                directory_path = Path(directory)
                for name in dirnames:
                    child = directory_path / name
                    _require_secure_cache_directory(
                        child.stat(follow_symlinks=False),
                        name="artifact path component",
                    )
                for name in filenames:
                    child = directory_path / name
                    relative = child.relative_to(tree).as_posix()
                    blob = expected.get(relative)
                    child_stat = child.stat(follow_symlinks=False)
                    if blob is None:
                        raise InvalidNodeModelCacheEntryError(
                            "published artifact tree does not match manifest"
                        )
                    _require_secure_cache_file(child_stat, name="artifact blob")
                    if child_stat.st_size != blob.size_bytes:
                        raise InvalidNodeModelCacheEntryError(
                            "published artifact tree does not match manifest"
                        )
                    observed.add(relative)
        except InvalidNodeModelCacheEntryError:
            raise
        except OSError as exc:
            raise InvalidNodeModelCacheEntryError("cannot inspect published artifact tree") from exc
        if observed != set(expected):
            raise InvalidNodeModelCacheEntryError("published artifact tree is incomplete")

    @staticmethod
    def _completion(envelope: SignedModelArtifactManifest) -> NodeModelCacheCompletion:
        manifest = envelope.manifest
        return NodeModelCacheCompletion(
            manifest_digest=envelope.manifest_digest,
            file_tree_sha256=manifest.file_tree_sha256,
            file_count=len(manifest.files),
            total_bytes=sum(blob.size_bytes for blob in manifest.files),
        )

    @staticmethod
    def _canonical_json(value: object) -> bytes:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    @staticmethod
    def _read_bounded(path: Path, *, max_bytes: int) -> bytes:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise InvalidNodeModelCacheEntryError("cannot read cache metadata") from exc
        try:
            metadata_stat = os.fstat(descriptor)
            _require_secure_cache_file(metadata_stat, name="cache metadata")
            content = bytearray()
            while len(content) <= max_bytes:
                chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - len(content)))
                if not chunk:
                    break
                content.extend(chunk)
        except OSError as exc:
            raise InvalidNodeModelCacheEntryError("cannot read cache metadata") from exc
        finally:
            os.close(descriptor)
        if len(content) > max_bytes:
            raise InvalidNodeModelCacheEntryError("cache metadata exceeds the size limit")
        return bytes(content)

    @classmethod
    def _load_completion(cls, path: Path) -> NodeModelCacheCompletion:
        content = cls._read_bounded(path, max_bytes=_MAX_COMPLETION_BYTES)
        try:
            value = json.loads(
                content,
                object_pairs_hook=cls._reject_duplicate_keys,
                parse_constant=cls._reject_non_finite_json,
            )
            return NodeModelCacheCompletion.model_validate(value)
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            ValidationError,
            InvalidNodeModelCacheEntryError,
        ) as exc:
            raise InvalidNodeModelCacheEntryError("cache completion marker is invalid") from exc

    @staticmethod
    def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise InvalidNodeModelCacheEntryError(
                    "cache completion marker contains duplicate keys"
                )
            value[key] = item
        return value

    @staticmethod
    def _reject_non_finite_json(value: str) -> Any:
        raise InvalidNodeModelCacheEntryError(
            f"cache completion marker contains invalid constant {value}"
        )

    @staticmethod
    def _write_bytes(path: Path, content: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise InvalidNodeModelCacheEntryError("cannot create cache metadata") from exc
        try:
            NodeModelCacheAgent._write_all(descriptor, content)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _write_or_validate_bytes(cls, path: Path, content: bytes) -> None:
        if path.exists() or path.is_symlink():
            if cls._read_bounded(path, max_bytes=_MAX_COMPLETION_BYTES) != content:
                raise InvalidNodeModelCacheEntryError("existing cache metadata does not match")
            return
        cls._write_bytes(path, content)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise InvalidNodeModelCacheEntryError("cannot open cache directory") from exc
        try:
            _require_secure_cache_directory(os.fstat(descriptor), name="cache directory")
            os.fsync(descriptor)
        except OSError as exc:
            raise InvalidNodeModelCacheEntryError("cannot sync cache directory") from exc
        finally:
            os.close(descriptor)

    @classmethod
    def _fsync_tree(cls, tree: Path) -> None:
        def raise_walk_error(error: OSError) -> None:
            raise error

        try:
            directories = [
                Path(directory) for directory, _, _ in os.walk(tree, onerror=raise_walk_error)
            ]
        except OSError as exc:
            raise InvalidNodeModelCacheEntryError("cannot enumerate staging tree") from exc
        for directory in reversed(directories):
            cls._fsync_directory(directory)

    @staticmethod
    def _fsync_file(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise InvalidNodeModelCacheEntryError("cannot open cache file for sync") from exc
        try:
            _require_secure_cache_file(os.fstat(descriptor), name="cache file")
            os.fsync(descriptor)
        except OSError as exc:
            raise InvalidNodeModelCacheEntryError("cannot sync cache file") from exc
        finally:
            os.close(descriptor)

    @classmethod
    def _fsync_staging(
        cls,
        staging: Path,
        blobs: tuple[ModelArtifactBlob, ...],
    ) -> None:
        tree = staging / "tree"
        for blob in blobs:
            cls._fsync_file(tree / Path(*blob.path.split("/")))
        cls._fsync_file(staging / "manifest.json")
        cls._fsync_file(staging / "completion.json")
        cls._fsync_tree(tree)
        cls._fsync_directory(staging)

    @staticmethod
    def _result(
        *,
        admission: ModelArtifactAdmission,
        envelope: SignedModelArtifactManifest,
        artifact_path: Path,
        cache_hit: bool,
        resumed_bytes: int,
        downloaded_bytes: int,
    ) -> NodeModelCacheFillResult:
        manifest = envelope.manifest
        return NodeModelCacheFillResult(
            deployment_id=admission.deployment_id,
            manifest_digest=admission.manifest_digest,
            artifact_path=artifact_path,
            cache_hit=cache_hit,
            resumed_bytes=resumed_bytes,
            downloaded_bytes=downloaded_bytes,
            file_count=len(manifest.files),
            total_bytes=sum(blob.size_bytes for blob in manifest.files),
        )

    def _record_index(
        self,
        result: NodeModelCacheFillResult,
        *,
        envelope: SignedModelArtifactManifest,
        verification_source: Literal["filled", "published_marker"],
    ) -> None:
        if self._index is None:
            return
        self._index.record_verified(
            manifest_digest=result.manifest_digest,
            model_id=envelope.manifest.model_id,
            model_revision=envelope.manifest.model_revision,
            artifact_path=result.artifact_path,
            total_bytes=result.total_bytes,
            file_count=result.file_count,
            verification_source=verification_source,
        )
