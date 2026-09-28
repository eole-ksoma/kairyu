"""Fresh, advisory node-model residency hints for scheduler/controller use."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.artifacts.cache_index import NodeModelCacheIndex, NodeModelCacheRecord

_MAX_SIGNED_BIGINT = 2**63 - 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_HINT_TTL_SECONDS = 300


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _text(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL")
    return value


def _aware(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


class NodeModelCacheResidentHint(BaseModel):
    """Path-free evidence that one verified artifact is resident on one node."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-cache-resident-hint-v1"] = (
        "kairyu-node-model-cache-resident-hint-v1"
    )
    node_id: str = Field(max_length=255)
    manifest_digest: str = Field(min_length=64, max_length=64)
    model_id: str = Field(max_length=255)
    model_revision: str = Field(max_length=255)
    total_bytes: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    file_count: int = Field(ge=1, le=100_000)
    verified: Literal[True] = True
    verified_at_ns: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    last_access_at_ns: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    pinned: bool
    record_generation: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)

    @field_validator("node_id", "model_id", "model_revision")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _text(value, name=info.field_name)

    @field_validator("manifest_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("manifest_digest must be a lowercase SHA-256 digest")
        return value

    @field_validator(
        "total_bytes",
        "file_count",
        "verified_at_ns",
        "last_access_at_ns",
        "record_generation",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        return value

    @field_validator("pinned", mode="before")
    @classmethod
    def validate_pinned(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("pinned must be a boolean")
        return value

    @field_validator("verified", mode="before")
    @classmethod
    def validate_verified(cls, value: object) -> object:
        if type(value) is not bool or value is not True:
            raise ValueError("verified must be true")
        return value

    @model_validator(mode="after")
    def validate_timestamps(self) -> NodeModelCacheResidentHint:
        if self.last_access_at_ns < self.verified_at_ns:
            raise ValueError("last access cannot precede verification")
        return self

    @classmethod
    def from_record(cls, record: NodeModelCacheRecord) -> NodeModelCacheResidentHint:
        """Project a verified index record without leaking its local path or pin owners."""

        record = NodeModelCacheRecord.model_validate(record.model_dump())
        if not record.verified:
            raise ValueError("placement hints require verified cache records")
        return cls(
            node_id=record.node_id,
            manifest_digest=record.manifest_digest,
            model_id=record.model_id,
            model_revision=record.model_revision,
            total_bytes=record.total_bytes,
            file_count=record.file_count,
            verified_at_ns=record.verified_at_ns,
            last_access_at_ns=record.last_access_at_ns,
            pinned=record.pinned,
            record_generation=record.generation,
        )


class NodeModelCachePlacementHintSnapshot(BaseModel):
    """Short-lived publication of one node's verified cache residency."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-cache-placement-hints-v1"] = (
        "kairyu-node-model-cache-placement-hints-v1"
    )
    node_id: str = Field(max_length=255)
    index_revision: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    observed_at: datetime
    valid_until: datetime
    residents: tuple[NodeModelCacheResidentHint, ...] = Field(
        default=(),
        max_length=100_000,
    )

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, value: str) -> str:
        return _text(value, name="node_id")

    @field_validator("index_revision", mode="before")
    @classmethod
    def validate_revision(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("index_revision must be an integer")
        return value

    @field_validator("observed_at", "valid_until")
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        return _aware(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_snapshot(self) -> NodeModelCachePlacementHintSnapshot:
        lifetime = (self.valid_until - self.observed_at).total_seconds()
        if not 0 < lifetime <= _MAX_HINT_TTL_SECONDS:
            raise ValueError("placement hint lifetime must be in (0, 300] seconds")
        identities = tuple(
            (resident.model_id, resident.model_revision, resident.manifest_digest)
            for resident in self.residents
        )
        if identities != tuple(sorted(identities)):
            raise ValueError("resident hints must use canonical identity order")
        if len({resident.manifest_digest for resident in self.residents}) != len(self.residents):
            raise ValueError("resident hints must use unique manifest digests")
        if any(resident.node_id != self.node_id for resident in self.residents):
            raise ValueError("resident hints must belong to the snapshot node")
        return self

    def resident_for(
        self,
        *,
        manifest_digest: str,
        model_id: str,
        model_revision: str,
    ) -> NodeModelCacheResidentHint | None:
        """Return an exact immutable artifact match, never a revision fallback."""

        for resident in self.residents:
            if resident.manifest_digest == manifest_digest:
                if resident.model_id == model_id and resident.model_revision == model_revision:
                    return resident
                return None
        return None


class NodeModelCachePlacementHintPublisher:
    """Project one cache index into a bounded-lifetime controller payload."""

    def __init__(
        self,
        index: NodeModelCacheIndex,
        *,
        ttl_seconds: int = 60,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not isinstance(index, NodeModelCacheIndex):
            raise TypeError("index must be a NodeModelCacheIndex")
        if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= _MAX_HINT_TTL_SECONDS:
            raise ValueError("ttl_seconds must be an integer from 1 to 300")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._index = index
        self._ttl_seconds = ttl_seconds
        self._clock = clock

    def snapshot(self) -> NodeModelCachePlacementHintSnapshot:
        """Publish only verified records from one transactionally consistent index read."""

        source = self._index.snapshot()
        observed_at = _aware(self._clock(), name="clock result")
        residents = tuple(
            NodeModelCacheResidentHint.from_record(record)
            for record in source.records
            if record.verified
        )
        return NodeModelCachePlacementHintSnapshot(
            node_id=source.node_id,
            index_revision=source.revision,
            observed_at=observed_at,
            valid_until=observed_at + timedelta(seconds=self._ttl_seconds),
            residents=residents,
        )
