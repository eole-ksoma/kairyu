"""Owner-scoped live cache evidence served by one authenticated node agent."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kairyu.artifacts.cache_index import NodeModelCacheIndex, NodeModelCacheRecord
from kairyu.artifacts.placement_hint import (
    NodeModelCachePlacementHintSnapshot,
    NodeModelCacheResidentHint,
)
from kairyu.runners.prestage import (
    NodeModelPrestageConflictError,
    NodeModelPrestageLookupStore,
    NodeModelPrestageRecord,
)
from kairyu.runners.startup_binding import RunnerCacheStartupPlacement
from kairyu.runners.startup_binding_live_authority import (
    RunnerCachePlacementBindingPinEvidence,
    RunnerCachePlacementBindingPrestageEvidence,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _text(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL")
    return value


class NodeModelCacheLiveEvidenceRequest(BaseModel):
    """Exact binding placement whose current node-local lineage is requested."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-cache-live-evidence-request-v1"] = (
        "kairyu-node-model-cache-live-evidence-request-v1"
    )
    placement: RunnerCacheStartupPlacement
    model_id: str = Field(max_length=255)
    model_revision: str = Field(max_length=255)

    @field_validator("model_id", "model_revision")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _text(value, name=info.field_name)


class NodeModelCacheLiveEvidenceResponse(BaseModel):
    """Path-free evidence derived from one node-index snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-cache-live-evidence-response-v1"] = (
        "kairyu-node-model-cache-live-evidence-response-v1"
    )
    node_id: str = Field(max_length=253)
    prestage_record: RunnerCachePlacementBindingPrestageEvidence
    placement_hint: NodeModelCachePlacementHintSnapshot
    pin_evidence: RunnerCachePlacementBindingPinEvidence

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, value: str) -> str:
        return _text(value, name="node_id")


@runtime_checkable
class NodeModelCacheLiveEvidenceSource(Protocol):
    def read(
        self,
        request: NodeModelCacheLiveEvidenceRequest,
    ) -> NodeModelCacheLiveEvidenceResponse: ...


class LocalNodeModelCacheLiveEvidenceSource:
    """Join a durable pre-stage record to one transactionally read cache index."""

    def __init__(
        self,
        *,
        node_id: str,
        store: NodeModelPrestageLookupStore,
        index: NodeModelCacheIndex,
        hint_ttl_seconds: int = 60,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._node_id = _text(node_id, name="node_id")
        if len(self._node_id) > 253:
            raise ValueError("node_id exceeds maximum length")
        if not isinstance(store, NodeModelPrestageLookupStore):
            raise TypeError("store must implement NodeModelPrestageLookupStore")
        if store.node_id != self._node_id:
            raise ValueError("pre-stage store belongs to another node")
        if not isinstance(index, NodeModelCacheIndex):
            raise TypeError("index must be a NodeModelCacheIndex")
        if index.node_id != self._node_id:
            raise ValueError("cache index belongs to another node")
        if type(hint_ttl_seconds) is not int or not 1 <= hint_ttl_seconds <= 300:
            raise ValueError("hint_ttl_seconds must be an integer from 1 to 300")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._store = store
        self._index = index
        self._hint_ttl_seconds = hint_ttl_seconds
        self._clock = clock

    @staticmethod
    def _record_for(
        record: NodeModelPrestageRecord | None,
        placement_id: str,
    ) -> NodeModelPrestageRecord:
        if record is None or record.command.placement_id != placement_id:
            raise NodeModelPrestageConflictError(
                "live evidence requires one current pre-stage record"
            )
        return NodeModelPrestageRecord.model_validate(record.model_dump())

    @staticmethod
    def _cache_record_for(
        records: tuple[NodeModelCacheRecord, ...],
        manifest_digest: str,
    ) -> NodeModelCacheRecord:
        matches = tuple(record for record in records if record.manifest_digest == manifest_digest)
        if len(matches) != 1:
            raise NodeModelPrestageConflictError(
                "live evidence requires one current cache residency"
            )
        return NodeModelCacheRecord.model_validate(matches[0].model_dump())

    @staticmethod
    def _validate_lineage(
        request: NodeModelCacheLiveEvidenceRequest,
        record: NodeModelPrestageRecord,
        cached: NodeModelCacheRecord,
    ) -> None:
        placement = request.placement
        command = record.command
        if (
            command.action != "ensure"
            or command.placement_id != placement.placement_id
            or command.node_id != placement.node_name
            or command.command_id != placement.prestage_command_id
            or command.command_generation != placement.prestage_command_generation
            or command.pin_owner != placement.pin_owner
            or command.manifest_digest != placement.manifest_digest
            or command.model_id != request.model_id
            or command.model_revision != request.model_revision
            or record.pin_record_generation != placement.resident_record_generation
            or cached.node_id != placement.node_name
            or cached.manifest_digest != placement.manifest_digest
            or cached.model_id != request.model_id
            or cached.model_revision != request.model_revision
            or not cached.verified
            or placement.pin_owner not in cached.pin_owners
            or cached.generation != placement.resident_record_generation
        ):
            raise NodeModelPrestageConflictError(
                "live cache evidence does not match the requested binding placement"
            )

    def read(
        self,
        request: NodeModelCacheLiveEvidenceRequest,
    ) -> NodeModelCacheLiveEvidenceResponse:
        if not isinstance(request, NodeModelCacheLiveEvidenceRequest):
            raise TypeError("request must be a NodeModelCacheLiveEvidenceRequest")
        request = NodeModelCacheLiveEvidenceRequest.model_validate(request.model_dump())
        if request.placement.node_name != self._node_id:
            raise NodeModelPrestageConflictError("live evidence targets another node")

        before = self._record_for(
            self._store.get_record(request.placement.placement_id),
            request.placement.placement_id,
        )
        snapshot = self._index.snapshot_record(request.placement.manifest_digest)
        cached = self._cache_record_for(
            snapshot.records,
            request.placement.manifest_digest,
        )
        after = self._record_for(
            self._store.get_record(request.placement.placement_id),
            request.placement.placement_id,
        )
        if after != before:
            raise NodeModelPrestageConflictError(
                "pre-stage lineage changed during live evidence read"
            )
        self._validate_lineage(request, after, cached)
        observed_at = self._clock()
        if (
            not isinstance(observed_at, datetime)
            or observed_at.tzinfo is None
            or observed_at.utcoffset() is None
        ):
            raise ValueError("clock result must be timezone-aware")
        if after.updated_at > observed_at:
            raise NodeModelPrestageConflictError(
                "pre-stage record is newer than live evidence observation"
            )
        resident = NodeModelCacheResidentHint.from_record(cached)
        hint = NodeModelCachePlacementHintSnapshot(
            node_id=self._node_id,
            index_revision=snapshot.revision,
            observed_at=observed_at,
            valid_until=observed_at + timedelta(seconds=self._hint_ttl_seconds),
            residents=(resident,),
        )
        return NodeModelCacheLiveEvidenceResponse(
            node_id=self._node_id,
            prestage_record=RunnerCachePlacementBindingPrestageEvidence.from_record(after),
            placement_hint=hint,
            pin_evidence=RunnerCachePlacementBindingPinEvidence(
                node_id=self._node_id,
                index_revision=snapshot.revision,
                observed_at=observed_at,
                manifest_digest=cached.manifest_digest,
                model_id=cached.model_id,
                model_revision=cached.model_revision,
                record_generation=cached.generation,
                pin_owners=cached.pin_owners,
            ),
        )


__all__ = [
    "LocalNodeModelCacheLiveEvidenceSource",
    "NodeModelCacheLiveEvidenceRequest",
    "NodeModelCacheLiveEvidenceResponse",
    "NodeModelCacheLiveEvidenceSource",
]
