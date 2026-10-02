"""Durable node-local model cache residency index (WP4.3)."""

from __future__ import annotations

import contextlib
import os
import re
import secrets
import sqlite3
import stat
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_LEGACY_SCHEMA_VERSION = "kairyu-node-model-cache-index-v1"
_PREVIOUS_SCHEMA_VERSION = "kairyu-node-model-cache-index-v2"
_SCHEMA_VERSION = "kairyu-node-model-cache-index-v3"
_APPLICATION_ID = 0x4B414943  # "KAIC"
_LEGACY_USER_VERSION = 1
_PREVIOUS_USER_VERSION = 2
_USER_VERSION = 3
_MAX_SIGNED_BIGINT = 2**63 - 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RECOVERY_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class NodeModelCacheIndexError(RuntimeError):
    """Base class for durable cache-index failures."""


class NodeModelCacheIndexIdentityError(NodeModelCacheIndexError):
    """The index, node, or digest identity conflicts with existing state."""


class NodeModelCacheIndexEntryNotFoundError(NodeModelCacheIndexError):
    """The requested resident digest is absent from the index."""


class NodeModelCacheIndexUnverifiedError(NodeModelCacheIndexError):
    """An operation requires residency that is still marked verified."""


class NodeModelCacheIndexEvictionConflictError(NodeModelCacheIndexError):
    """An eviction fence no longer matches the current residency state."""


class NodeModelCacheIndexPinnedError(NodeModelCacheIndexError):
    """An eviction attempted to remove owner-pinned residency."""


class NodeModelCacheRecord(BaseModel):
    """One immutable artifact identity plus mutable node-local residency state."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-cache-record-v2"] = (
        "kairyu-node-model-cache-record-v2"
    )
    node_id: str = Field(max_length=255)
    manifest_digest: str = Field(min_length=64, max_length=64)
    model_id: str = Field(max_length=255)
    model_revision: str = Field(max_length=255)
    artifact_path: Path
    total_bytes: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    file_count: int = Field(ge=1, le=100_000)
    verified: bool
    verification_source: Literal["filled", "published_marker"]
    verification_failure: str | None = Field(default=None, max_length=1024)
    recovery_id: str | None = Field(default=None, min_length=64, max_length=64)
    verified_at_ns: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    last_access_at_ns: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    pin_owners: tuple[str, ...] = Field(default=(), max_length=10_000)
    pinned: bool
    generation: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)

    @field_validator("node_id", "model_id", "model_revision")
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
    def validate_recovery_id(cls, value: str | None) -> str | None:
        if value is not None and not _RECOVERY_ID_PATTERN.fullmatch(value):
            raise ValueError("recovery_id must be a lowercase 256-bit identifier")
        return value

    @field_validator("artifact_path", mode="before")
    @classmethod
    def validate_artifact_path(cls, value: object) -> object:
        path = Path(value) if isinstance(value, (str, Path)) else None
        if path is None or not path.is_absolute() or "\x00" in str(path):
            raise ValueError("artifact_path must be an absolute path without NUL")
        return path

    @field_validator(
        "total_bytes",
        "file_count",
        "verified_at_ns",
        "last_access_at_ns",
        "generation",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        return value

    @field_validator("verified", "pinned", mode="before")
    @classmethod
    def validate_boolean(cls, value: object, info) -> object:
        if type(value) is not bool:
            raise ValueError(f"{info.field_name} must be a boolean")
        return value

    @field_validator("pin_owners")
    @classmethod
    def validate_pin_owners(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for owner in value:
            if not owner.strip() or "\x00" in owner or len(owner) > 255:
                raise ValueError("pin owner must be a non-empty bounded string without NUL")
        if value != tuple(sorted(set(value))):
            raise ValueError("pin_owners must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_consistency(self) -> NodeModelCacheRecord:
        if self.verified == (self.verification_failure is not None):
            raise ValueError("verification failure must be present only for unverified residency")
        if self.verified and self.recovery_id is not None:
            raise ValueError("verified residency cannot have a recovery identifier")
        if self.pinned != bool(self.pin_owners):
            raise ValueError("pinned must match pin_owners")
        if self.last_access_at_ns < self.verified_at_ns:
            raise ValueError("last access cannot precede verification")
        return self


class NodeModelCacheIndexSnapshot(BaseModel):
    """One transactionally consistent cache-index publication source."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-cache-index-snapshot-v2"] = (
        "kairyu-node-model-cache-index-snapshot-v2"
    )
    node_id: str = Field(max_length=255)
    revision: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    records: tuple[NodeModelCacheRecord, ...] = Field(default=(), max_length=100_000)

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("node_id must be a non-empty string without NUL")
        return value

    @field_validator("revision", mode="before")
    @classmethod
    def validate_revision(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("revision must be an integer")
        return value

    @model_validator(mode="after")
    def validate_records(self) -> NodeModelCacheIndexSnapshot:
        identities = tuple(
            (record.model_id, record.model_revision, record.manifest_digest)
            for record in self.records
        )
        if identities != tuple(sorted(identities)):
            raise ValueError("cache records must use canonical identity order")
        if len({record.manifest_digest for record in self.records}) != len(self.records):
            raise ValueError("cache records must use unique manifest digests")
        if any(record.node_id != self.node_id for record in self.records):
            raise ValueError("cache records must belong to the snapshot node")
        return self


class NodeModelCacheIndex:
    """SQLite-backed node-local cache index with owner-scoped pins."""

    def __init__(
        self,
        path: Path,
        *,
        node_id: str,
        cache_root: Path | None = None,
        busy_timeout_seconds: float = 5.0,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self._path = path
        self._node_id = self._validate_text(node_id, name="node_id", max_length=255)
        if not path.is_absolute() or "\x00" in str(path):
            raise ValueError("cache index path must be absolute and contain no NUL")
        cache_root = path.parent if cache_root is None else cache_root
        if not cache_root.is_absolute() or "\x00" in str(cache_root):
            raise ValueError("cache root must be absolute and contain no NUL")
        if isinstance(busy_timeout_seconds, bool) or not isinstance(
            busy_timeout_seconds, (int, float)
        ):
            raise ValueError("busy_timeout_seconds must be a number")
        if not 0 <= float(busy_timeout_seconds) <= 300:
            raise ValueError("busy_timeout_seconds must be between 0 and 300")
        self._busy_timeout_ms = int(float(busy_timeout_seconds) * 1000)
        self._clock_ns = clock_ns
        self._prepare_parent()
        self._cache_root = cache_root
        self._initialize()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def cache_root(self) -> Path:
        return self._cache_root

    def record_verified(
        self,
        *,
        manifest_digest: str,
        model_id: str,
        model_revision: str,
        artifact_path: Path,
        total_bytes: int,
        file_count: int,
        verification_source: Literal["filled", "published_marker"],
    ) -> NodeModelCacheRecord:
        """Insert verified residency or touch an identical existing identity."""

        digest = self._validate_digest(manifest_digest)
        model_id = self._validate_text(model_id, name="model_id", max_length=255)
        model_revision = self._validate_text(
            model_revision,
            name="model_revision",
            max_length=255,
        )
        artifact_path = self._validate_artifact_path(artifact_path, digest=digest)
        total_bytes = self._validate_integer(
            total_bytes,
            name="total_bytes",
            minimum=0,
            maximum=_MAX_SIGNED_BIGINT,
        )
        file_count = self._validate_integer(
            file_count,
            name="file_count",
            minimum=1,
            maximum=100_000,
        )
        if verification_source not in {"filled", "published_marker"}:
            raise ValueError("verification_source is invalid")
        now = self._now_ns()
        try:
            with self._write_transaction() as connection:
                row = connection.execute(
                    "SELECT * FROM cache_entries WHERE manifest_digest = ?",
                    (digest,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO cache_entries (
                            manifest_digest, model_id, model_revision, artifact_path,
                            total_bytes, file_count, verified, verification_source,
                            verified_at_ns, last_access_at_ns, generation
                        ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, 1)
                        """,
                        (
                            digest,
                            model_id,
                            model_revision,
                            str(artifact_path),
                            total_bytes,
                            file_count,
                            verification_source,
                            now,
                            now,
                        ),
                    )
                else:
                    self._require_matching_identity(
                        row,
                        model_id=model_id,
                        model_revision=model_revision,
                        artifact_path=artifact_path,
                        total_bytes=total_bytes,
                        file_count=file_count,
                    )
                    if not row["verified"]:
                        if row["recovery_id"] is not None:
                            raise NodeModelCacheIndexUnverifiedError(
                                "recovery-required residency needs fenced audit completion"
                            )
                        if verification_source != "filled":
                            raise NodeModelCacheIndexUnverifiedError(
                                "unverified residency requires a digest-verified refill"
                            )
                    source = row["verification_source"]
                    verified_at = row["verified_at_ns"]
                    verification_failure = row["verification_failure"]
                    verified = row["verified"]
                    if not verified:
                        verified = 1
                        source = "filled"
                        verified_at = now
                        verification_failure = None
                    elif source == "published_marker" and verification_source == "filled":
                        source = "filled"
                    last_access = max(row["last_access_at_ns"], now)
                    changed = (
                        last_access != row["last_access_at_ns"]
                        or source != row["verification_source"]
                        or verified != row["verified"]
                    )
                    if changed:
                        connection.execute(
                            """
                            UPDATE cache_entries
                            SET last_access_at_ns = ?, verification_source = ?, verified = ?,
                                verified_at_ns = ?, verification_failure = ?,
                                generation = generation + 1
                            WHERE manifest_digest = ?
                            """,
                            (
                                last_access,
                                source,
                                verified,
                                verified_at,
                                verification_failure,
                                digest,
                            ),
                        )
                return self._get_required(connection, digest)
        except sqlite3.Error as exc:
            raise NodeModelCacheIndexError("cannot record verified cache residency") from exc

    def touch(self, manifest_digest: str) -> NodeModelCacheRecord:
        """Advance last-access time without changing identity, pins, or generation.

        Access recency advances only the index revision, so placement bindings that
        pin a residency generation survive a successful Runner-start verification.
        """

        digest = self._validate_digest(manifest_digest)
        now = self._now_ns()
        try:
            with self._write_transaction() as connection:
                row = connection.execute(
                    """
                    SELECT last_access_at_ns, verified FROM cache_entries
                    WHERE manifest_digest = ?
                    """,
                    (digest,),
                ).fetchone()
                if row is None:
                    raise NodeModelCacheIndexEntryNotFoundError("cache index entry does not exist")
                if not row["verified"]:
                    raise NodeModelCacheIndexUnverifiedError(
                        "unverified residency cannot be touched"
                    )
                last_access = max(row["last_access_at_ns"], now)
                if last_access != row["last_access_at_ns"]:
                    connection.execute(
                        """
                        UPDATE cache_entries SET last_access_at_ns = ?
                        WHERE manifest_digest = ?
                        """,
                        (last_access, digest),
                    )
                return self._get_required(connection, digest)
        except sqlite3.Error as exc:
            raise NodeModelCacheIndexError("cannot touch cache residency") from exc

    def pin(
        self,
        manifest_digest: str,
        *,
        owner: str,
        reason: str,
    ) -> NodeModelCacheRecord:
        """Create or update one owner's durable pin without replacing other pins."""

        digest = self._validate_digest(manifest_digest)
        owner = self._validate_text(owner, name="pin owner", max_length=255)
        reason = self._validate_text(reason, name="pin reason", max_length=1024)
        now = self._now_ns()
        try:
            with self._write_transaction() as connection:
                self._require_entry(connection, digest)
                verified = connection.execute(
                    "SELECT verified FROM cache_entries WHERE manifest_digest = ?",
                    (digest,),
                ).fetchone()["verified"]
                if not verified:
                    raise NodeModelCacheIndexUnverifiedError(
                        "unverified residency cannot be pinned"
                    )
                existing = connection.execute(
                    """
                    SELECT reason FROM cache_pins
                    WHERE manifest_digest = ? AND owner = ?
                    """,
                    (digest, owner),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO cache_pins (
                            manifest_digest, owner, reason, pinned_at_ns
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (digest, owner, reason, now),
                    )
                    self._advance_generation(connection, digest)
                elif existing["reason"] != reason:
                    connection.execute(
                        """
                        UPDATE cache_pins SET reason = ?, pinned_at_ns = ?
                        WHERE manifest_digest = ? AND owner = ?
                        """,
                        (reason, now, digest, owner),
                    )
                    self._advance_generation(connection, digest)
                return self._get_required(connection, digest)
        except sqlite3.Error as exc:
            raise NodeModelCacheIndexError("cannot pin cache residency") from exc

    def unpin(self, manifest_digest: str, *, owner: str) -> NodeModelCacheRecord:
        """Release only the named owner's pin; repeated release is idempotent."""

        digest = self._validate_digest(manifest_digest)
        owner = self._validate_text(owner, name="pin owner", max_length=255)
        try:
            with self._write_transaction() as connection:
                self._require_entry(connection, digest)
                cursor = connection.execute(
                    "DELETE FROM cache_pins WHERE manifest_digest = ? AND owner = ?",
                    (digest, owner),
                )
                if cursor.rowcount:
                    self._advance_generation(connection, digest)
                return self._get_required(connection, digest)
        except sqlite3.Error as exc:
            raise NodeModelCacheIndexError("cannot unpin cache residency") from exc

    def mark_unverified(
        self,
        manifest_digest: str,
        *,
        reason: str,
    ) -> NodeModelCacheRecord:
        """Persist a fail-closed verification state for later WP4.6 recovery."""

        digest = self._validate_digest(manifest_digest)
        reason = self._validate_text(reason, name="verification failure", max_length=1024)
        try:
            with self._write_transaction() as connection:
                row = connection.execute(
                    """
                    SELECT verified, verification_failure, recovery_id FROM cache_entries
                    WHERE manifest_digest = ?
                    """,
                    (digest,),
                ).fetchone()
                if row is None:
                    raise NodeModelCacheIndexEntryNotFoundError("cache index entry does not exist")
                if row["recovery_id"] is not None:
                    raise NodeModelCacheIndexUnverifiedError(
                        "recovery-required residency cannot be replaced by a generic failure"
                    )
                if row["verified"] or row["verification_failure"] != reason:
                    connection.execute(
                        """
                        UPDATE cache_entries
                        SET verified = 0, verification_failure = ?,
                            generation = generation + 1
                        WHERE manifest_digest = ?
                        """,
                        (reason, digest),
                    )
                return self._get_required(connection, digest)
        except sqlite3.Error as exc:
            raise NodeModelCacheIndexError("cannot mark cache residency unverified") from exc

    def begin_recovery(
        self,
        manifest_digest: str,
        *,
        reason: str,
    ) -> NodeModelCacheRecord:
        """Create or resume one durable, globally unique corruption incident."""

        digest = self._validate_digest(manifest_digest)
        reason = self._validate_text(reason, name="recovery reason", max_length=900)
        try:
            with self._write_transaction() as connection:
                row = connection.execute(
                    "SELECT * FROM cache_entries WHERE manifest_digest = ?",
                    (digest,),
                ).fetchone()
                if row is None:
                    raise NodeModelCacheIndexEntryNotFoundError("cache index entry does not exist")
                recovery_id = row["recovery_id"]
                if recovery_id is None:
                    recovery_id = secrets.token_hex(32)
                    connection.execute(
                        """
                        UPDATE cache_entries
                        SET verified = 0, verification_failure = ?, recovery_id = ?,
                            generation = generation + 1
                        WHERE manifest_digest = ?
                        """,
                        (reason, recovery_id, digest),
                    )
                return self._get_required(connection, digest)
        except sqlite3.Error as exc:
            raise NodeModelCacheIndexError("cannot begin cache recovery") from exc

    def complete_recovery(
        self,
        *,
        manifest_digest: str,
        recovery_id: str,
        expected_generation: int,
        model_id: str,
        model_revision: str,
        artifact_path: Path,
        total_bytes: int,
        file_count: int,
    ) -> NodeModelCacheRecord:
        """CAS one audited, digest-verified replacement back to verified."""

        digest = self._validate_digest(manifest_digest)
        recovery_id = self._validate_recovery_id(recovery_id)
        generation = self._validate_integer(
            expected_generation,
            name="expected_generation",
            minimum=1,
            maximum=_MAX_SIGNED_BIGINT,
        )
        model_id = self._validate_text(model_id, name="model_id", max_length=255)
        model_revision = self._validate_text(
            model_revision,
            name="model_revision",
            max_length=255,
        )
        artifact_path = self._validate_artifact_path(artifact_path, digest=digest)
        total_bytes = self._validate_integer(
            total_bytes,
            name="total_bytes",
            minimum=0,
            maximum=_MAX_SIGNED_BIGINT,
        )
        file_count = self._validate_integer(
            file_count,
            name="file_count",
            minimum=1,
            maximum=100_000,
        )
        now = self._now_ns()
        try:
            with self._write_transaction() as connection:
                row = connection.execute(
                    "SELECT * FROM cache_entries WHERE manifest_digest = ?",
                    (digest,),
                ).fetchone()
                if row is None:
                    raise NodeModelCacheIndexEntryNotFoundError("cache index entry does not exist")
                self._require_matching_identity(
                    row,
                    model_id=model_id,
                    model_revision=model_revision,
                    artifact_path=artifact_path,
                    total_bytes=total_bytes,
                    file_count=file_count,
                )
                if (
                    row["verified"]
                    or row["generation"] != generation
                    or row["recovery_id"] != recovery_id
                ):
                    raise NodeModelCacheIndexUnverifiedError(
                        "cache recovery fence no longer matches"
                    )
                connection.execute(
                    """
                    UPDATE cache_entries
                    SET verified = 1, verification_source = 'filled',
                        verification_failure = NULL, verified_at_ns = ?,
                        recovery_id = NULL, last_access_at_ns = ?,
                        generation = generation + 1
                    WHERE manifest_digest = ? AND generation = ?
                        AND verified = 0 AND recovery_id = ?
                    """,
                    (
                        now,
                        max(row["last_access_at_ns"], now),
                        digest,
                        generation,
                        recovery_id,
                    ),
                )
                return self._get_required(connection, digest)
        except sqlite3.Error as exc:
            raise NodeModelCacheIndexError("cannot complete cache recovery") from exc

    def get(self, manifest_digest: str) -> NodeModelCacheRecord | None:
        digest = self._validate_digest(manifest_digest)
        try:
            with self._read_transaction() as connection:
                row = connection.execute(
                    "SELECT * FROM cache_entries WHERE manifest_digest = ?",
                    (digest,),
                ).fetchone()
                return None if row is None else self._record_from_row(connection, row)
        except sqlite3.Error as exc:
            raise NodeModelCacheIndexError("cannot read cache residency") from exc

    def list_records(self) -> tuple[NodeModelCacheRecord, ...]:
        """Return a stable node/model/revision/digest ordered residency snapshot."""

        try:
            with self._read_transaction() as connection:
                rows = connection.execute(
                    """
                    SELECT * FROM cache_entries
                    ORDER BY model_id, model_revision, manifest_digest
                    """
                ).fetchall()
                return tuple(self._record_from_row(connection, row) for row in rows)
        except sqlite3.Error as exc:
            raise NodeModelCacheIndexError("cannot list cache residency") from exc

    def snapshot(self) -> NodeModelCacheIndexSnapshot:
        """Read the global revision and all records from one SQLite snapshot."""

        try:
            with self._read_transaction() as connection:
                revision_row = connection.execute(
                    "SELECT revision FROM cache_index_revision WHERE singleton = 1"
                ).fetchone()
                if revision_row is None:
                    raise NodeModelCacheIndexIdentityError(
                        "cache index revision metadata is absent"
                    )
                rows = connection.execute(
                    """
                    SELECT * FROM cache_entries
                    ORDER BY model_id, model_revision, manifest_digest
                    """
                ).fetchall()
                return NodeModelCacheIndexSnapshot(
                    node_id=self._node_id,
                    revision=revision_row["revision"],
                    records=tuple(self._record_from_row(connection, row) for row in rows),
                )
        except sqlite3.Error as exc:
            raise NodeModelCacheIndexError("cannot snapshot cache residency") from exc

    def snapshot_record(self, manifest_digest: str) -> NodeModelCacheIndexSnapshot:
        """Read the global revision and one exact record from one SQLite snapshot."""

        digest = self._validate_digest(manifest_digest)
        try:
            with self._read_transaction() as connection:
                revision_row = connection.execute(
                    "SELECT revision FROM cache_index_revision WHERE singleton = 1"
                ).fetchone()
                if revision_row is None:
                    raise NodeModelCacheIndexIdentityError(
                        "cache index revision metadata is absent"
                    )
                row = connection.execute(
                    "SELECT * FROM cache_entries WHERE manifest_digest = ?",
                    (digest,),
                ).fetchone()
                records = () if row is None else (self._record_from_row(connection, row),)
                return NodeModelCacheIndexSnapshot(
                    node_id=self._node_id,
                    revision=revision_row["revision"],
                    records=records,
                )
        except sqlite3.Error as exc:
            raise NodeModelCacheIndexError("cannot snapshot cache residency") from exc

    @contextlib.contextmanager
    def fenced_eviction(
        self,
        manifest_digest: str,
        *,
        expected_index_revision: int,
        expected_generation: int,
    ) -> Iterator[NodeModelCacheRecord]:
        """Hold a write fence while a caller atomically detaches one artifact tree.

        The caller may perform the same-filesystem rename while the transaction is
        open.  Returning from the context deletes the exact unpinned generation;
        raising rolls the database transaction back so the caller can restore the
        detached tree before releasing its digest lock.
        """

        digest = self._validate_digest(manifest_digest)
        index_revision = self._validate_integer(
            expected_index_revision,
            name="expected_index_revision",
            minimum=1,
            maximum=_MAX_SIGNED_BIGINT,
        )
        generation = self._validate_integer(
            expected_generation,
            name="expected_generation",
            minimum=1,
            maximum=_MAX_SIGNED_BIGINT,
        )
        try:
            with self._write_transaction() as connection:
                revision_row = connection.execute(
                    "SELECT revision FROM cache_index_revision WHERE singleton = 1"
                ).fetchone()
                if revision_row is None or revision_row["revision"] != index_revision:
                    raise NodeModelCacheIndexEvictionConflictError(
                        "cache index revision changed before eviction"
                    )
                row = connection.execute(
                    "SELECT * FROM cache_entries WHERE manifest_digest = ?",
                    (digest,),
                ).fetchone()
                if row is None or row["generation"] != generation:
                    raise NodeModelCacheIndexEvictionConflictError(
                        "cache residency generation changed before eviction"
                    )
                if row["recovery_id"] is not None:
                    raise NodeModelCacheIndexEvictionConflictError(
                        "recovery-required residency cannot be evicted"
                    )
                pin = connection.execute(
                    "SELECT 1 FROM cache_pins WHERE manifest_digest = ? LIMIT 1",
                    (digest,),
                ).fetchone()
                if pin is not None:
                    raise NodeModelCacheIndexPinnedError("pinned cache residency cannot be evicted")
                record = self._record_from_row(connection, row)
                yield record
                cursor = connection.execute(
                    """
                    DELETE FROM cache_entries
                    WHERE manifest_digest = ? AND generation = ?
                      AND recovery_id IS NULL
                      AND NOT EXISTS (
                          SELECT 1 FROM cache_pins WHERE manifest_digest = ?
                      )
                    """,
                    (digest, generation, digest),
                )
                if cursor.rowcount != 1:
                    raise NodeModelCacheIndexEvictionConflictError(
                        "cache residency changed while eviction was fenced"
                    )
        except sqlite3.Error as exc:
            raise NodeModelCacheIndexError("cannot evict cache residency") from exc

    def _prepare_parent(self) -> None:
        try:
            self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            parent_stat = self._path.parent.stat(follow_symlinks=False)
        except OSError as exc:
            raise NodeModelCacheIndexError("cannot prepare cache index directory") from exc
        self._require_secure_directory(parent_stat, name="cache index directory")

    def _initialize(self) -> None:
        try:
            with self._connection(initialize=True) as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    application_id = connection.execute("PRAGMA application_id").fetchone()[0]
                    user_version = connection.execute("PRAGMA user_version").fetchone()[0]
                    if application_id not in {0, _APPLICATION_ID}:
                        raise NodeModelCacheIndexIdentityError(
                            "SQLite file belongs to another application"
                        )
                    if user_version not in {
                        0,
                        _LEGACY_USER_VERSION,
                        _PREVIOUS_USER_VERSION,
                        _USER_VERSION,
                    }:
                        raise NodeModelCacheIndexIdentityError(
                            "cache index schema version is unsupported"
                        )
                    if user_version == 0:
                        self._create_current_schema(connection)
                    elif user_version == _LEGACY_USER_VERSION:
                        if application_id != _APPLICATION_ID:
                            raise NodeModelCacheIndexIdentityError(
                                "legacy cache index has no application identity"
                            )
                        self._migrate_v1_to_v2(connection)
                        self._migrate_v2_to_v3(connection)
                    elif user_version == _PREVIOUS_USER_VERSION:
                        if application_id != _APPLICATION_ID:
                            raise NodeModelCacheIndexIdentityError(
                                "previous cache index has no application identity"
                            )
                        self._migrate_v2_to_v3(connection)
                    else:
                        if application_id != _APPLICATION_ID:
                            raise NodeModelCacheIndexIdentityError(
                                "cache index has no application identity"
                            )
                        self._require_meta_binding(
                            connection,
                            schema_version=_SCHEMA_VERSION,
                        )
                        self._require_index_revision(connection)
                        self._create_revision_triggers(
                            connection,
                            if_not_exists=True,
                        )
                        self._create_recovery_guard_triggers(
                            connection,
                            if_not_exists=True,
                        )
                        self._require_revision_triggers(connection)
                        self._require_recovery_guard_triggers(connection)
                    connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
                    connection.execute(f"PRAGMA user_version = {_USER_VERSION}")
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise
        except sqlite3.Error as exc:
            raise NodeModelCacheIndexError("cannot initialize cache index") from exc

    def _create_current_schema(self, connection: sqlite3.Connection) -> None:
        schema_statements = (
            """
            CREATE TABLE IF NOT EXISTS cache_index_meta (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                schema_version TEXT NOT NULL,
                node_id TEXT NOT NULL,
                cache_root TEXT NOT NULL,
                created_at_ns INTEGER NOT NULL CHECK (created_at_ns >= 0)
            )
            """,
            self._entries_table_statement("cache_entries", if_not_exists=True),
            self._pins_table_statement(
                "cache_pins",
                entries_table="cache_entries",
                if_not_exists=True,
            ),
            """
            CREATE INDEX IF NOT EXISTS cache_entries_lru
                ON cache_entries(last_access_at_ns, manifest_digest)
            """,
            """
            CREATE INDEX IF NOT EXISTS cache_entries_model
                ON cache_entries(model_id, model_revision, manifest_digest)
            """,
            self._revision_table_statement(if_not_exists=True),
        )
        for statement in schema_statements:
            connection.execute(statement)
        meta = connection.execute(
            "SELECT schema_version, node_id, cache_root FROM cache_index_meta WHERE singleton = 1"
        ).fetchone()
        if meta is None:
            connection.execute(
                """
                INSERT INTO cache_index_meta (
                    singleton, schema_version, node_id, cache_root, created_at_ns
                ) VALUES (1, ?, ?, ?, ?)
                """,
                (
                    _SCHEMA_VERSION,
                    self._node_id,
                    str(self._cache_root),
                    self._now_ns(),
                ),
            )
        else:
            self._validate_meta_binding(meta, schema_version=_SCHEMA_VERSION)
        connection.execute(
            "INSERT OR IGNORE INTO cache_index_revision (singleton, revision) VALUES (1, 1)"
        )
        self._require_index_revision(connection)
        self._create_revision_triggers(connection, if_not_exists=True)
        self._create_recovery_guard_triggers(connection, if_not_exists=True)
        self._require_revision_triggers(connection)
        self._require_recovery_guard_triggers(connection)

    def _migrate_v1_to_v2(self, connection: sqlite3.Connection) -> None:
        self._require_meta_binding(connection, schema_version=_LEGACY_SCHEMA_VERSION)
        revision = 1
        generations = connection.execute("SELECT generation FROM cache_entries").fetchall()
        for row in generations:
            generation = row["generation"]
            if (
                type(generation) is not int
                or not 1 <= generation <= _MAX_SIGNED_BIGINT
                or revision > _MAX_SIGNED_BIGINT - generation
            ):
                raise NodeModelCacheIndexIdentityError(
                    "legacy cache index revision cannot be represented safely"
                )
            revision += generation
        connection.execute(
            self._v2_entries_table_statement("cache_entries_v2", if_not_exists=False)
        )
        connection.execute(
            """
            INSERT INTO cache_entries_v2 SELECT
                manifest_digest, model_id, model_revision, artifact_path,
                total_bytes, file_count, verified, verification_source,
                verification_failure, verified_at_ns, last_access_at_ns, generation
            FROM cache_entries
            """
        )
        connection.execute(
            self._pins_table_statement(
                "cache_pins_v2",
                entries_table="cache_entries_v2",
                if_not_exists=False,
            )
        )
        connection.execute(
            """
            INSERT INTO cache_pins_v2
            SELECT manifest_digest, owner, reason, pinned_at_ns FROM cache_pins
            """
        )
        connection.execute("DROP TABLE cache_pins")
        connection.execute("DROP TABLE cache_entries")
        connection.execute("ALTER TABLE cache_entries_v2 RENAME TO cache_entries")
        connection.execute("ALTER TABLE cache_pins_v2 RENAME TO cache_pins")
        connection.execute(
            """
            CREATE INDEX cache_entries_lru
                ON cache_entries(last_access_at_ns, manifest_digest)
            """
        )
        connection.execute(
            """
            CREATE INDEX cache_entries_model
                ON cache_entries(model_id, model_revision, manifest_digest)
            """
        )
        connection.execute(self._revision_table_statement(if_not_exists=False))
        connection.execute(
            "INSERT INTO cache_index_revision (singleton, revision) VALUES (1, ?)",
            (revision,),
        )
        self._create_v2_revision_triggers(connection)
        connection.execute(
            "UPDATE cache_index_meta SET schema_version = ? WHERE singleton = 1",
            (_PREVIOUS_SCHEMA_VERSION,),
        )
        self._require_revision_triggers(connection)

    def _migrate_v2_to_v3(self, connection: sqlite3.Connection) -> None:
        self._require_meta_binding(connection, schema_version=_PREVIOUS_SCHEMA_VERSION)
        self._require_index_revision(connection)
        connection.execute(
            """
            ALTER TABLE cache_entries ADD COLUMN recovery_id TEXT
                CHECK (recovery_id IS NULL OR length(recovery_id) = 64)
            """
        )
        for trigger in (
            "cache_entries_revision_insert",
            "cache_entries_revision_update",
            "cache_entries_revision_delete",
            "cache_pins_revision_insert",
            "cache_pins_revision_update",
            "cache_pins_revision_delete",
        ):
            connection.execute(f"DROP TRIGGER {trigger}")
        self._create_revision_triggers(connection, if_not_exists=False)
        self._create_recovery_guard_triggers(connection, if_not_exists=False)
        connection.execute(
            "UPDATE cache_index_meta SET schema_version = ? WHERE singleton = 1",
            (_SCHEMA_VERSION,),
        )
        self._require_revision_triggers(connection)
        self._require_recovery_guard_triggers(connection)

    @staticmethod
    def _v2_entries_table_statement(name: str, *, if_not_exists: bool) -> str:
        qualifier = " IF NOT EXISTS" if if_not_exists else ""
        return f"""
            CREATE TABLE{qualifier} {name} (
                manifest_digest TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                total_bytes INTEGER NOT NULL CHECK (total_bytes >= 0),
                file_count INTEGER NOT NULL CHECK (file_count >= 1),
                verified INTEGER NOT NULL CHECK (verified IN (0, 1)),
                verification_source TEXT NOT NULL CHECK (
                    verification_source IN ('filled', 'published_marker')
                ),
                verification_failure TEXT,
                verified_at_ns INTEGER NOT NULL CHECK (verified_at_ns >= 0),
                last_access_at_ns INTEGER NOT NULL CHECK (
                    last_access_at_ns >= verified_at_ns
                ),
                generation INTEGER NOT NULL CHECK (
                    generation BETWEEN 1 AND {_MAX_SIGNED_BIGINT}
                )
            )
        """

    @staticmethod
    def _entries_table_statement(name: str, *, if_not_exists: bool) -> str:
        qualifier = " IF NOT EXISTS" if if_not_exists else ""
        return f"""
            CREATE TABLE{qualifier} {name} (
                manifest_digest TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                total_bytes INTEGER NOT NULL CHECK (total_bytes >= 0),
                file_count INTEGER NOT NULL CHECK (file_count >= 1),
                verified INTEGER NOT NULL CHECK (verified IN (0, 1)),
                verification_source TEXT NOT NULL CHECK (
                    verification_source IN ('filled', 'published_marker')
                ),
                verification_failure TEXT,
                recovery_id TEXT CHECK (
                    recovery_id IS NULL OR length(recovery_id) = 64
                ),
                verified_at_ns INTEGER NOT NULL CHECK (verified_at_ns >= 0),
                last_access_at_ns INTEGER NOT NULL CHECK (
                    last_access_at_ns >= verified_at_ns
                ),
                generation INTEGER NOT NULL CHECK (
                    generation BETWEEN 1 AND {_MAX_SIGNED_BIGINT}
                )
            )
        """

    @staticmethod
    def _pins_table_statement(
        name: str,
        *,
        entries_table: str,
        if_not_exists: bool,
    ) -> str:
        qualifier = " IF NOT EXISTS" if if_not_exists else ""
        return f"""
            CREATE TABLE{qualifier} {name} (
                manifest_digest TEXT NOT NULL REFERENCES {entries_table}(
                    manifest_digest
                ) ON DELETE CASCADE,
                owner TEXT NOT NULL,
                reason TEXT NOT NULL,
                pinned_at_ns INTEGER NOT NULL CHECK (pinned_at_ns >= 0),
                PRIMARY KEY (manifest_digest, owner)
            )
        """

    @staticmethod
    def _revision_table_statement(*, if_not_exists: bool) -> str:
        qualifier = " IF NOT EXISTS" if if_not_exists else ""
        return f"""
            CREATE TABLE{qualifier} cache_index_revision (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                revision INTEGER NOT NULL CHECK (
                    revision BETWEEN 1 AND {_MAX_SIGNED_BIGINT}
                )
            )
        """

    @staticmethod
    def _create_v2_revision_triggers(connection: sqlite3.Connection) -> None:
        events = {
            "cache_entries_revision_insert": "AFTER INSERT ON cache_entries",
            "cache_entries_revision_update": (
                "AFTER UPDATE OF last_access_at_ns, verification_source, verified, "
                "verification_failure, verified_at_ns ON cache_entries"
            ),
            "cache_entries_revision_delete": "AFTER DELETE ON cache_entries",
            "cache_pins_revision_insert": "AFTER INSERT ON cache_pins",
            "cache_pins_revision_update": "AFTER UPDATE ON cache_pins",
            "cache_pins_revision_delete": "AFTER DELETE ON cache_pins",
        }
        for name, event in events.items():
            connection.execute(
                f"""
                CREATE TRIGGER {name}
                {event}
                BEGIN
                    SELECT CASE
                        WHEN NOT EXISTS (
                            SELECT 1 FROM cache_index_revision
                            WHERE singleton = 1 AND revision < {_MAX_SIGNED_BIGINT}
                        )
                        THEN RAISE(ABORT, 'cache index revision exhausted or absent')
                    END;
                    UPDATE cache_index_revision
                    SET revision = revision + 1 WHERE singleton = 1;
                END
                """
            )

    @staticmethod
    def _create_revision_triggers(
        connection: sqlite3.Connection,
        *,
        if_not_exists: bool,
    ) -> None:
        qualifier = " IF NOT EXISTS" if if_not_exists else ""
        events = {
            "cache_entries_revision_insert": "AFTER INSERT ON cache_entries",
            "cache_entries_revision_update": (
                "AFTER UPDATE OF last_access_at_ns, verification_source, verified, "
                "verification_failure, recovery_id, verified_at_ns ON cache_entries"
            ),
            "cache_entries_revision_delete": "AFTER DELETE ON cache_entries",
            "cache_pins_revision_insert": "AFTER INSERT ON cache_pins",
            "cache_pins_revision_update": "AFTER UPDATE ON cache_pins",
            "cache_pins_revision_delete": "AFTER DELETE ON cache_pins",
        }
        for name, event in events.items():
            connection.execute(
                f"""
                CREATE TRIGGER{qualifier} {name}
                {event}
                BEGIN
                    SELECT CASE
                        WHEN NOT EXISTS (
                            SELECT 1 FROM cache_index_revision
                            WHERE singleton = 1 AND revision < {_MAX_SIGNED_BIGINT}
                        )
                        THEN RAISE(ABORT, 'cache index revision exhausted or absent')
                    END;
                    UPDATE cache_index_revision
                    SET revision = revision + 1 WHERE singleton = 1;
                END
                """
            )

    @staticmethod
    def _create_recovery_guard_triggers(
        connection: sqlite3.Connection,
        *,
        if_not_exists: bool,
    ) -> None:
        qualifier = " IF NOT EXISTS" if if_not_exists else ""
        connection.execute(
            f"""
            CREATE TRIGGER{qualifier} cache_recovery_update_guard
            BEFORE UPDATE ON cache_entries
            WHEN OLD.recovery_id IS NOT NULL AND NOT (
                (
                    NEW.recovery_id = OLD.recovery_id
                    AND NEW.verified = 0
                    AND NEW.verification_failure = OLD.verification_failure
                )
                OR (
                    NEW.recovery_id IS NULL
                    AND NEW.verified = 1
                    AND NEW.verification_failure IS NULL
                    AND NEW.generation = OLD.generation + 1
                )
            )
            BEGIN
                SELECT RAISE(ABORT, 'recovery-required cache row update rejected');
            END
            """
        )
        connection.execute(
            f"""
            CREATE TRIGGER{qualifier} cache_recovery_delete_guard
            BEFORE DELETE ON cache_entries
            WHEN OLD.recovery_id IS NOT NULL
            BEGIN
                SELECT RAISE(ABORT, 'recovery-required cache row delete rejected');
            END
            """
        )

    @staticmethod
    def _require_recovery_guard_triggers(connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT name, sql FROM sqlite_master
            WHERE type = 'trigger' AND name IN (
                'cache_recovery_update_guard', 'cache_recovery_delete_guard'
            )
            """
        ).fetchall()
        definitions = {row["name"]: row["sql"] or "" for row in rows}
        update_sql = definitions.get("cache_recovery_update_guard", "")
        delete_sql = definitions.get("cache_recovery_delete_guard", "")
        if not (
            "OLD.recovery_id IS NOT NULL" in update_sql
            and "NEW.recovery_id IS NULL" in update_sql
            and "NEW.generation = OLD.generation + 1" in update_sql
            and "RAISE(ABORT" in update_sql
            and "OLD.recovery_id IS NOT NULL" in delete_sql
            and "RAISE(ABORT" in delete_sql
        ):
            raise NodeModelCacheIndexIdentityError(
                "cache recovery guard triggers are absent or invalid"
            )

    @staticmethod
    def _require_revision_triggers(connection: sqlite3.Connection) -> None:
        expected = {
            "cache_entries_revision_insert",
            "cache_entries_revision_update",
            "cache_entries_revision_delete",
            "cache_pins_revision_insert",
            "cache_pins_revision_update",
            "cache_pins_revision_delete",
        }
        present = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }
        if not expected <= present:
            raise NodeModelCacheIndexIdentityError("cache index revision triggers are absent")

    def _require_meta_binding(
        self,
        connection: sqlite3.Connection,
        *,
        schema_version: str,
    ) -> None:
        meta = connection.execute(
            "SELECT schema_version, node_id, cache_root FROM cache_index_meta WHERE singleton = 1"
        ).fetchone()
        if meta is None:
            raise NodeModelCacheIndexIdentityError("cache index metadata is absent")
        self._validate_meta_binding(meta, schema_version=schema_version)

    def _validate_meta_binding(
        self,
        meta: sqlite3.Row,
        *,
        schema_version: str,
    ) -> None:
        if (
            meta["schema_version"] != schema_version
            or meta["node_id"] != self._node_id
            or meta["cache_root"] != str(self._cache_root)
        ):
            raise NodeModelCacheIndexIdentityError(
                "cache index is bound to another schema, node, or cache root"
            )

    @staticmethod
    def _require_index_revision(connection: sqlite3.Connection) -> int:
        row = connection.execute(
            "SELECT revision FROM cache_index_revision WHERE singleton = 1"
        ).fetchone()
        if (
            row is None
            or type(row["revision"]) is not int
            or not 1 <= row["revision"] <= _MAX_SIGNED_BIGINT
        ):
            raise NodeModelCacheIndexIdentityError(
                "cache index revision metadata is absent or invalid"
            )
        return row["revision"]

    @contextlib.contextmanager
    def _connection(self, *, initialize: bool = False) -> Iterator[sqlite3.Connection]:
        existed = self._path.exists() or self._path.is_symlink()
        if existed:
            self._validate_index_file()
        try:
            connection = sqlite3.connect(
                self._path,
                timeout=self._busy_timeout_ms / 1000,
                isolation_level=None,
            )
        except sqlite3.Error as exc:
            raise NodeModelCacheIndexError("cannot open cache index") from exc
        try:
            if not existed:
                os.chmod(self._path, 0o600)
            self._validate_index_file()
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
            connection.execute("PRAGMA synchronous = FULL")
            if initialize:
                mode = self._enable_wal_mode(connection)
                if str(mode).lower() != "wal":
                    raise NodeModelCacheIndexError("cache index could not enable WAL mode")
            else:
                application_id = connection.execute("PRAGMA application_id").fetchone()[0]
                user_version = connection.execute("PRAGMA user_version").fetchone()[0]
                if application_id != _APPLICATION_ID or user_version != _USER_VERSION:
                    raise NodeModelCacheIndexIdentityError(
                        "cache index identity or schema version changed"
                    )
                meta = connection.execute(
                    """
                    SELECT schema_version, node_id, cache_root
                    FROM cache_index_meta WHERE singleton = 1
                    """
                ).fetchone()
                if (
                    meta is None
                    or meta["schema_version"] != _SCHEMA_VERSION
                    or meta["node_id"] != self._node_id
                    or meta["cache_root"] != str(self._cache_root)
                ):
                    raise NodeModelCacheIndexIdentityError(
                        "cache index is bound to another schema, node, or cache root"
                    )
            yield connection
        finally:
            connection.close()

    def _enable_wal_mode(self, connection: sqlite3.Connection) -> object:
        """Enable WAL despite SQLite's non-waiting concurrent PRAGMA behavior."""

        deadline_ns = time.monotonic_ns() + self._busy_timeout_ms * 1_000_000
        while True:
            try:
                return connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    raise
                time.sleep(min(0.01, remaining_ns / 1_000_000_000))

    @contextlib.contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    @contextlib.contextmanager
    def _read_transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN")
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def _get_required(
        self,
        connection: sqlite3.Connection,
        digest: str,
    ) -> NodeModelCacheRecord:
        row = connection.execute(
            "SELECT * FROM cache_entries WHERE manifest_digest = ?",
            (digest,),
        ).fetchone()
        if row is None:
            raise NodeModelCacheIndexEntryNotFoundError("cache index entry does not exist")
        return self._record_from_row(connection, row)

    @staticmethod
    def _require_entry(connection: sqlite3.Connection, digest: str) -> None:
        row = connection.execute(
            "SELECT 1 FROM cache_entries WHERE manifest_digest = ?",
            (digest,),
        ).fetchone()
        if row is None:
            raise NodeModelCacheIndexEntryNotFoundError("cache index entry does not exist")

    def _record_from_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> NodeModelCacheRecord:
        pins = connection.execute(
            """
            SELECT owner FROM cache_pins
            WHERE manifest_digest = ? ORDER BY owner
            """,
            (row["manifest_digest"],),
        ).fetchall()
        owners = tuple(pin["owner"] for pin in pins)
        try:
            artifact_path = self._validate_artifact_path(
                Path(row["artifact_path"]),
                digest=row["manifest_digest"],
            )
            return NodeModelCacheRecord(
                node_id=self._node_id,
                manifest_digest=row["manifest_digest"],
                model_id=row["model_id"],
                model_revision=row["model_revision"],
                artifact_path=artifact_path,
                total_bytes=row["total_bytes"],
                file_count=row["file_count"],
                verified=bool(row["verified"]),
                verification_source=row["verification_source"],
                verification_failure=row["verification_failure"],
                recovery_id=row["recovery_id"],
                verified_at_ns=row["verified_at_ns"],
                last_access_at_ns=row["last_access_at_ns"],
                pin_owners=owners,
                pinned=bool(owners),
                generation=row["generation"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise NodeModelCacheIndexError("cache index record is invalid") from exc

    @staticmethod
    def _require_matching_identity(
        row: sqlite3.Row,
        *,
        model_id: str,
        model_revision: str,
        artifact_path: Path,
        total_bytes: int,
        file_count: int,
    ) -> None:
        expected = (
            model_id,
            model_revision,
            str(artifact_path),
            total_bytes,
            file_count,
        )
        actual = (
            row["model_id"],
            row["model_revision"],
            row["artifact_path"],
            row["total_bytes"],
            row["file_count"],
        )
        if actual != expected:
            raise NodeModelCacheIndexIdentityError(
                "existing digest is bound to different cache metadata"
            )

    @staticmethod
    def _advance_generation(connection: sqlite3.Connection, digest: str) -> None:
        connection.execute(
            """
            UPDATE cache_entries SET generation = generation + 1
            WHERE manifest_digest = ?
            """,
            (digest,),
        )

    def _validate_index_file(self) -> None:
        try:
            path_stat = self._path.stat(follow_symlinks=False)
        except OSError as exc:
            raise NodeModelCacheIndexError("cannot inspect cache index file") from exc
        if not stat.S_ISREG(path_stat.st_mode):
            raise NodeModelCacheIndexError("cache index is not a regular file")
        if path_stat.st_uid != os.geteuid():
            raise NodeModelCacheIndexError("cache index is not owned by the cache user")
        if path_stat.st_nlink != 1:
            raise NodeModelCacheIndexError("cache index has an unsafe link count")
        if stat.S_IMODE(path_stat.st_mode) & 0o022:
            raise NodeModelCacheIndexError("cache index is group/world writable")

    @staticmethod
    def _require_secure_directory(path_stat: os.stat_result, *, name: str) -> None:
        if not stat.S_ISDIR(path_stat.st_mode):
            raise NodeModelCacheIndexError(f"{name} is not a directory")
        if path_stat.st_uid != os.geteuid():
            raise NodeModelCacheIndexError(f"{name} is not owned by the cache user")
        if stat.S_IMODE(path_stat.st_mode) & 0o022:
            raise NodeModelCacheIndexError(f"{name} is group/world writable")

    def _now_ns(self) -> int:
        value = self._clock_ns()
        return self._validate_integer(
            value,
            name="clock_ns",
            minimum=0,
            maximum=_MAX_SIGNED_BIGINT,
        )

    @staticmethod
    def _validate_digest(value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("manifest_digest must be a lowercase SHA-256 digest")
        return value

    @staticmethod
    def _validate_recovery_id(value: str) -> str:
        if not isinstance(value, str) or not _RECOVERY_ID_PATTERN.fullmatch(value):
            raise ValueError("recovery_id must be a lowercase 256-bit identifier")
        return value

    @staticmethod
    def _validate_text(value: str, *, name: str, max_length: int) -> str:
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"{name} must be a non-empty string without NUL")
        if len(value) > max_length:
            raise ValueError(f"{name} exceeds maximum length")
        return value

    def _validate_artifact_path(self, value: Path, *, digest: str) -> Path:
        if not isinstance(value, Path) or not value.is_absolute() or "\x00" in str(value):
            raise ValueError("artifact_path must be an absolute path without NUL")
        expected = self._cache_root / "artifacts" / digest / "tree"
        if value != expected:
            raise NodeModelCacheIndexIdentityError(
                "artifact_path does not match the index cache root and digest"
            )
        return value

    @staticmethod
    def _validate_integer(
        value: int,
        *,
        name: str,
        minimum: int,
        maximum: int,
    ) -> int:
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
        return value
