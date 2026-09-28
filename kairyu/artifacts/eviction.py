"""Capacity watermarks and generation-fenced node-model cache eviction."""

from __future__ import annotations

import contextlib
import fcntl
import math
import os
import re
import shutil
import stat
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.artifacts.cache_index import (
    NodeModelCacheIndex,
    NodeModelCacheIndexSnapshot,
)

_MAX_SIGNED_BIGINT = 2**63 - 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TOMBSTONE_PATTERN = re.compile(r"^([0-9a-f]{64})\.([1-9][0-9]*)\.([1-9][0-9]*)$")


class NodeModelCacheEvictionError(RuntimeError):
    """A capacity plan could not be applied safely."""


class NodeModelCacheEvictionPlanStaleError(NodeModelCacheEvictionError):
    """The source index revision changed before eviction execution."""


def _integer(value: object, *, name: str) -> object:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _bounded_sum(values: tuple[int, ...], *, name: str) -> int:
    total = 0
    for value in values:
        if total > _MAX_SIGNED_BIGINT - value:
            raise ValueError(f"{name} exceeds signed 64-bit capacity")
        total += value
    return total


class NodeModelCacheCapacityPolicy(BaseModel):
    """Absolute byte watermarks for one node-local cache filesystem."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-cache-capacity-policy-v1"] = (
        "kairyu-node-model-cache-capacity-policy-v1"
    )
    high_watermark_bytes: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    low_watermark_bytes: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)

    @field_validator("high_watermark_bytes", "low_watermark_bytes", mode="before")
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        return _integer(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_watermarks(self) -> NodeModelCacheCapacityPolicy:
        if self.low_watermark_bytes >= self.high_watermark_bytes:
            raise ValueError("low watermark must be lower than high watermark")
        return self


class NodeModelCacheEvictionVictim(BaseModel):
    """One exact unpinned cache generation selected in LRU order."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    manifest_digest: str = Field(min_length=64, max_length=64)
    model_id: str = Field(min_length=1, max_length=255)
    model_revision: str = Field(min_length=1, max_length=255)
    total_bytes: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    last_access_at_ns: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    expected_generation: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)

    @field_validator("manifest_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("manifest_digest must be a lowercase SHA-256 digest")
        return value

    @field_validator("model_id", "model_revision")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError(f"{info.field_name} must be a non-empty string without NUL")
        return value

    @field_validator(
        "total_bytes",
        "last_access_at_ns",
        "expected_generation",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        return _integer(value, name=info.field_name)


class NodeModelCacheEvictionPlan(BaseModel):
    """Revision-bound LRU work needed to return below the low watermark."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-cache-eviction-plan-v1"] = (
        "kairyu-node-model-cache-eviction-plan-v1"
    )
    node_id: str = Field(min_length=1, max_length=255)
    index_revision: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    policy: NodeModelCacheCapacityPolicy
    observed_used_bytes: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    target_reclaim_bytes: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    planned_reclaim_bytes: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    blocked_reclaim_bytes: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    victims: tuple[NodeModelCacheEvictionVictim, ...] = Field(
        default=(),
        max_length=100_000,
    )

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("node_id must be a non-empty string without NUL")
        return value

    @field_validator(
        "index_revision",
        "observed_used_bytes",
        "target_reclaim_bytes",
        "planned_reclaim_bytes",
        "blocked_reclaim_bytes",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        return _integer(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_plan(self) -> NodeModelCacheEvictionPlan:
        triggered = self.observed_used_bytes > self.policy.high_watermark_bytes
        target = self.observed_used_bytes - self.policy.low_watermark_bytes if triggered else 0
        if self.target_reclaim_bytes != target:
            raise ValueError("target reclaim bytes do not match the watermarks")
        order = tuple((victim.last_access_at_ns, victim.manifest_digest) for victim in self.victims)
        if order != tuple(sorted(order)):
            raise ValueError("eviction victims must use canonical LRU order")
        digests = tuple(victim.manifest_digest for victim in self.victims)
        if len(set(digests)) != len(digests):
            raise ValueError("eviction victims must use unique manifest digests")
        planned = _bounded_sum(
            tuple(victim.total_bytes for victim in self.victims),
            name="planned reclaim bytes",
        )
        if self.planned_reclaim_bytes != planned:
            raise ValueError("planned reclaim bytes do not match victims")
        if self.blocked_reclaim_bytes != max(0, target - planned):
            raise ValueError("blocked reclaim bytes do not match eligible capacity")
        if not triggered and self.victims:
            raise ValueError("eviction victims require pressure above the high watermark")
        return self

    @property
    def triggered(self) -> bool:
        return self.observed_used_bytes > self.policy.high_watermark_bytes


class NodeModelCacheEvictionResult(BaseModel):
    """Durable evidence of exact generations removed from the cache."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-cache-eviction-result-v1"] = (
        "kairyu-node-model-cache-eviction-result-v1"
    )
    node_id: str = Field(min_length=1, max_length=255)
    source_index_revision: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    ending_index_revision: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    evicted: tuple[NodeModelCacheEvictionVictim, ...] = Field(default=(), max_length=100_000)
    reclaimed_bytes: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("node_id must be a non-empty string without NUL")
        return value

    @field_validator(
        "source_index_revision",
        "ending_index_revision",
        "reclaimed_bytes",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        return _integer(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_result(self) -> NodeModelCacheEvictionResult:
        reclaimed = _bounded_sum(
            tuple(victim.total_bytes for victim in self.evicted),
            name="reclaimed bytes",
        )
        if self.reclaimed_bytes != reclaimed:
            raise ValueError("reclaimed bytes do not match evicted victims")
        if self.ending_index_revision < self.source_index_revision:
            raise ValueError("ending index revision cannot move backward")
        return self


def plan_node_model_cache_eviction(
    snapshot: NodeModelCacheIndexSnapshot,
    policy: NodeModelCacheCapacityPolicy,
) -> NodeModelCacheEvictionPlan:
    """Select oldest unpinned generations only after crossing the high watermark."""

    if not isinstance(snapshot, NodeModelCacheIndexSnapshot):
        raise TypeError("snapshot must be a NodeModelCacheIndexSnapshot")
    if not isinstance(policy, NodeModelCacheCapacityPolicy):
        raise TypeError("policy must be a NodeModelCacheCapacityPolicy")
    snapshot = NodeModelCacheIndexSnapshot.model_validate(snapshot.model_dump())
    policy = NodeModelCacheCapacityPolicy.model_validate(policy.model_dump())
    used = _bounded_sum(
        tuple(record.total_bytes for record in snapshot.records),
        name="observed used bytes",
    )
    target = used - policy.low_watermark_bytes if used > policy.high_watermark_bytes else 0
    victims: list[NodeModelCacheEvictionVictim] = []
    planned = 0
    if target:
        eligible = sorted(
            (
                record
                for record in snapshot.records
                if not record.pinned and record.recovery_id is None
            ),
            key=lambda record: (record.last_access_at_ns, record.manifest_digest),
        )
        for record in eligible:
            if planned >= target:
                break
            victims.append(
                NodeModelCacheEvictionVictim(
                    manifest_digest=record.manifest_digest,
                    model_id=record.model_id,
                    model_revision=record.model_revision,
                    total_bytes=record.total_bytes,
                    last_access_at_ns=record.last_access_at_ns,
                    expected_generation=record.generation,
                )
            )
            planned += record.total_bytes
            if planned > _MAX_SIGNED_BIGINT:
                raise ValueError("planned reclaim bytes exceeds signed 64-bit capacity")
    return NodeModelCacheEvictionPlan(
        node_id=snapshot.node_id,
        index_revision=snapshot.revision,
        policy=policy,
        observed_used_bytes=used,
        target_reclaim_bytes=target,
        planned_reclaim_bytes=planned,
        blocked_reclaim_bytes=max(0, target - planned),
        victims=tuple(victims),
    )


class NodeModelCacheEvictor:
    """Apply one revision-bound plan under per-digest filesystem and DB fences."""

    def __init__(
        self,
        root: Path,
        index: NodeModelCacheIndex,
        *,
        lock_timeout_seconds: float | None = None,
    ) -> None:
        if not isinstance(root, Path) or not root.is_absolute() or "\x00" in str(root):
            raise ValueError("root must be an absolute Path without NUL")
        if not isinstance(index, NodeModelCacheIndex):
            raise TypeError("index must be a NodeModelCacheIndex")
        if root != index.path.parent:
            raise ValueError("root must match the cache index root")
        if lock_timeout_seconds is not None:
            if isinstance(lock_timeout_seconds, bool) or not isinstance(
                lock_timeout_seconds,
                (int, float),
            ):
                raise ValueError("lock_timeout_seconds must be a number")
            if not math.isfinite(float(lock_timeout_seconds)) or lock_timeout_seconds < 0:
                raise ValueError("lock_timeout_seconds must be finite and non-negative")
        self._root = root
        self._index = index
        self._lock_timeout_seconds = (
            None if lock_timeout_seconds is None else float(lock_timeout_seconds)
        )

    def execute(self, plan: NodeModelCacheEvictionPlan) -> NodeModelCacheEvictionResult:
        """Evict exact planned generations, stopping safely on the first conflict."""

        if not isinstance(plan, NodeModelCacheEvictionPlan):
            raise TypeError("plan must be a NodeModelCacheEvictionPlan")
        plan = NodeModelCacheEvictionPlan.model_validate(plan.model_dump())
        self._prepare_root()
        with self._exclusive_lock("eviction"):
            self._recover_interrupted()
            source = self._index.snapshot()
            if source.node_id != plan.node_id or source.revision != plan.index_revision:
                raise NodeModelCacheEvictionPlanStaleError(
                    "cache index revision changed before eviction"
                )
            canonical = plan_node_model_cache_eviction(source, plan.policy)
            if canonical != plan:
                raise NodeModelCacheEvictionPlanStaleError(
                    "eviction plan is not canonical for its source snapshot"
                )
            evicted: list[NodeModelCacheEvictionVictim] = []
            for victim in plan.victims:
                expected_revision = plan.index_revision + len(evicted)
                if self._index.snapshot().revision != expected_revision:
                    raise NodeModelCacheEvictionPlanStaleError(
                        "cache index revision changed during eviction"
                    )
                self._evict_one(
                    victim,
                    expected_index_revision=expected_revision,
                )
                evicted.append(victim)
            ending_revision = self._index.snapshot().revision
        return NodeModelCacheEvictionResult(
            node_id=plan.node_id,
            source_index_revision=plan.index_revision,
            ending_index_revision=ending_revision,
            evicted=tuple(evicted),
            reclaimed_bytes=sum(victim.total_bytes for victim in evicted),
        )

    def recover_interrupted(self) -> None:
        """Restore rolled-back detaches or remove tombstones already absent from the index."""

        self._prepare_root()
        with self._exclusive_lock("eviction"):
            self._recover_interrupted()

    def _recover_interrupted(self) -> None:
        tombstone_root = self._root / ".evicting"
        try:
            entries = tuple(sorted(tombstone_root.iterdir(), key=lambda path: path.name))
        except OSError as exc:
            raise NodeModelCacheEvictionError("cannot inspect eviction tombstones") from exc
        for tombstone in entries:
            match = _TOMBSTONE_PATTERN.fullmatch(tombstone.name)
            if match is None:
                raise NodeModelCacheEvictionError("eviction tombstone identity is invalid")
            digest, revision_text, generation_text = match.groups()
            index_revision = int(revision_text)
            generation = int(generation_text)
            if index_revision > _MAX_SIGNED_BIGINT or generation > _MAX_SIGNED_BIGINT:
                raise NodeModelCacheEvictionError("eviction tombstone fence is invalid")
            with self._exclusive_lock(digest):
                self._recover_one(
                    tombstone,
                    digest=digest,
                    index_revision=index_revision,
                    generation=generation,
                )

    def _evict_one(
        self,
        victim: NodeModelCacheEvictionVictim,
        *,
        expected_index_revision: int,
    ) -> None:
        digest = victim.manifest_digest
        published = self._root / "artifacts" / digest
        tombstone = (
            self._root
            / ".evicting"
            / f"{digest}.{expected_index_revision}.{victim.expected_generation}"
        )
        moved = False
        with self._exclusive_lock(digest):
            try:
                with self._index.fenced_eviction(
                    digest,
                    expected_index_revision=expected_index_revision,
                    expected_generation=victim.expected_generation,
                ) as record:
                    if record.artifact_path.parent != published:
                        raise NodeModelCacheEvictionError(
                            "indexed artifact path does not match the eviction target"
                        )
                    self._require_secure_directory(published, name="published cache entry")
                    if tombstone.exists() or tombstone.is_symlink():
                        raise NodeModelCacheEvictionError("eviction tombstone already exists")
                    os.rename(published, tombstone)
                    moved = True
                    self._fsync_directory(published.parent)
                    self._fsync_directory(tombstone.parent)
            except Exception as exc:
                if moved:
                    try:
                        os.rename(tombstone, published)
                        self._fsync_directory(published.parent)
                        self._fsync_directory(tombstone.parent)
                    except OSError as restore_exc:
                        raise NodeModelCacheEvictionError(
                            "cannot restore artifact after eviction rollback"
                        ) from restore_exc
                if isinstance(exc, OSError):
                    raise NodeModelCacheEvictionError(
                        "cannot detach artifact for eviction"
                    ) from exc
                raise
            try:
                shutil.rmtree(tombstone)
                self._fsync_directory(tombstone.parent)
            except OSError as exc:
                raise NodeModelCacheEvictionError(
                    "eviction committed but tombstone cleanup failed"
                ) from exc

    def _recover_one(
        self,
        tombstone: Path,
        *,
        digest: str,
        index_revision: int,
        generation: int,
    ) -> None:
        self._require_secure_directory(tombstone, name="eviction tombstone")
        published = self._root / "artifacts" / digest
        snapshot = self._index.snapshot()
        record = next(
            (candidate for candidate in snapshot.records if candidate.manifest_digest == digest),
            None,
        )
        try:
            published_exists = published.exists() or published.is_symlink()
            if published_exists:
                self._require_secure_directory(
                    published,
                    name="published cache entry",
                )
            if record is None:
                shutil.rmtree(tombstone)
            elif (
                not published_exists
                and snapshot.revision == index_revision
                and record.generation == generation
            ):
                os.rename(tombstone, published)
            elif published_exists and snapshot.revision > index_revision:
                shutil.rmtree(tombstone)
            else:
                raise NodeModelCacheEvictionError(
                    "eviction tombstone conflicts with current cache fence"
                )
            self._fsync_directory(published.parent)
            self._fsync_directory(tombstone.parent)
        except NodeModelCacheEvictionError:
            raise
        except OSError as exc:
            raise NodeModelCacheEvictionError("cannot recover interrupted cache eviction") from exc

    def _prepare_root(self) -> None:
        try:
            self._require_secure_directory(self._root, name="cache root")
            for name in ("artifacts", ".locks", ".evicting"):
                path = self._root / name
                path.mkdir(mode=0o700, exist_ok=True)
                self._require_secure_directory(path, name="cache control path")
        except NodeModelCacheEvictionError:
            raise
        except OSError as exc:
            raise NodeModelCacheEvictionError("cannot prepare eviction paths") from exc

    @staticmethod
    def _require_secure_directory(path: Path, *, name: str) -> None:
        try:
            path_stat = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise NodeModelCacheEvictionError(f"cannot inspect {name}") from exc
        if not stat.S_ISDIR(path_stat.st_mode):
            raise NodeModelCacheEvictionError(f"{name} is not a directory")
        if path_stat.st_uid != os.geteuid():
            raise NodeModelCacheEvictionError(f"{name} is not owned by the cache user")
        if stat.S_IMODE(path_stat.st_mode) & 0o022:
            raise NodeModelCacheEvictionError(f"{name} is group/world writable")

    @contextlib.contextmanager
    def _exclusive_lock(self, digest: str) -> Iterator[None]:
        path = self._root / ".locks" / f"{digest}.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise NodeModelCacheEvictionError("cannot open cache eviction lock") from exc
        try:
            lock_stat = os.fstat(descriptor)
            if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
                raise NodeModelCacheEvictionError("cache eviction lock is unsafe")
            if lock_stat.st_uid != os.geteuid() or stat.S_IMODE(lock_stat.st_mode) & 0o022:
                raise NodeModelCacheEvictionError("cache eviction lock is unsafe")
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
                            raise NodeModelCacheEvictionError(
                                "timed out waiting for cache eviction lock"
                            ) from exc
                        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0)
        try:
            descriptor = os.open(path, flags)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise NodeModelCacheEvictionError("cannot sync eviction directory") from exc
