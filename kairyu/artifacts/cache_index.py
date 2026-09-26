"""Durable node-local model cache residency index (WP4.3)."""

from __future__ import annotations

import contextlib
import os
import re
import sqlite3
import stat
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SCHEMA_VERSION = "kairyu-node-model-cache-index-v1"
_APPLICATION_ID = 0x4B414943  # "KAIC"
_USER_VERSION = 1
_MAX_SIGNED_BIGINT = 2**63 - 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class NodeModelCacheIndexError(RuntimeError):
    """Base class for durable cache-index failures."""


class NodeModelCacheIndexIdentityError(NodeModelCacheIndexError):
    """The index, node, or digest identity conflicts with existing state."""


class NodeModelCacheIndexEntryNotFoundError(NodeModelCacheIndexError):
    """The requested resident digest is absent from the index."""


class NodeModelCacheIndexUnverifiedError(NodeModelCacheIndexError):
    """An operation requires residency that is still marked verified."""


class NodeModelCacheRecord(BaseModel):
    """One immutable artifact identity plus mutable node-local residency state."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-cache-record-v1"] = (
        "kairyu-node-model-cache-record-v1"
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
        if self.pinned != bool(self.pin_owners):
            raise ValueError("pinned must match pin_owners")
        if self.last_access_at_ns < self.verified_at_ns:
            raise ValueError("last access cannot precede verification")
        return self


class NodeModelCacheIndex:
    """SQLite-backed node-local cache index with owner-scoped pins."""

    def __init__(
        self,
        path: Path,
        *,
        node_id: str,
        busy_timeout_seconds: float = 5.0,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self._path = path
        self._node_id = self._validate_text(node_id, name="node_id", max_length=255)
        if not path.is_absolute() or "\x00" in str(path):
            raise ValueError("cache index path must be absolute and contain no NUL")
        if isinstance(busy_timeout_seconds, bool) or not isinstance(
            busy_timeout_seconds, (int, float)
        ):
            raise ValueError("busy_timeout_seconds must be a number")
        if not 0 <= float(busy_timeout_seconds) <= 300:
            raise ValueError("busy_timeout_seconds must be between 0 and 300")
        self._busy_timeout_ms = int(float(busy_timeout_seconds) * 1000)
        self._clock_ns = clock_ns
        self._prepare_parent()
        self._cache_root = self._path.parent
        self._initialize()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def node_id(self) -> str:
        return self._node_id

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
                    if not row["verified"] and verification_source != "filled":
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
        """Advance last-access time without changing immutable identity or pins."""

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
                        UPDATE cache_entries
                        SET last_access_at_ns = ?, generation = generation + 1
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
                    SELECT verified, verification_failure FROM cache_entries
                    WHERE manifest_digest = ?
                    """,
                    (digest,),
                ).fetchone()
                if row is None:
                    raise NodeModelCacheIndexEntryNotFoundError("cache index entry does not exist")
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
                    if user_version not in {0, _USER_VERSION}:
                        raise NodeModelCacheIndexIdentityError(
                            "cache index schema version is unsupported"
                        )
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
                        """
                        CREATE TABLE IF NOT EXISTS cache_entries (
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
                            generation INTEGER NOT NULL CHECK (generation >= 1)
                        )
                        """,
                        """
                        CREATE TABLE IF NOT EXISTS cache_pins (
                            manifest_digest TEXT NOT NULL REFERENCES cache_entries(
                                manifest_digest
                            ) ON DELETE CASCADE,
                            owner TEXT NOT NULL,
                            reason TEXT NOT NULL,
                            pinned_at_ns INTEGER NOT NULL CHECK (pinned_at_ns >= 0),
                            PRIMARY KEY (manifest_digest, owner)
                        )
                        """,
                        """
                        CREATE INDEX IF NOT EXISTS cache_entries_lru
                            ON cache_entries(last_access_at_ns, manifest_digest)
                        """,
                        """
                        CREATE INDEX IF NOT EXISTS cache_entries_model
                            ON cache_entries(model_id, model_revision, manifest_digest)
                        """,
                    )
                    for statement in schema_statements:
                        connection.execute(statement)
                    connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
                    connection.execute(f"PRAGMA user_version = {_USER_VERSION}")
                    meta = connection.execute(
                        """
                        SELECT schema_version, node_id, cache_root
                        FROM cache_index_meta WHERE singleton = 1
                        """
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
                    elif (
                        meta["schema_version"] != _SCHEMA_VERSION
                        or meta["node_id"] != self._node_id
                        or meta["cache_root"] != str(self._cache_root)
                    ):
                        raise NodeModelCacheIndexIdentityError(
                            "cache index is bound to another schema, node, or cache root"
                        )
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise
        except sqlite3.Error as exc:
            raise NodeModelCacheIndexError("cannot initialize cache index") from exc

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
                mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
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
