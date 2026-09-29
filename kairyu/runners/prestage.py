"""Fenced node-local execution for controller-issued model pre-staging."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from itertools import islice
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.artifacts.cache_index import NodeModelCacheIndex, NodeModelCacheRecord
from kairyu.artifacts.manifest import (
    ModelArtifactAdmissionRequest,
    ModelArtifactTrustStore,
    SignedModelArtifactManifest,
)
from kairyu.artifacts.node_cache import NodeModelCacheAgent, NodeModelCacheFillResult
from kairyu.runners.leadership import RunnerWriterAuthority
from kairyu.runners.prewarm import (
    ModelCachePlacementState,
    ScalingPrewarmPlan,
    ScalingPrewarmSnapshot,
)

_MAX_SIGNED_BIGINT = 2**63 - 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class NodeModelPrestageError(RuntimeError):
    """Base class for fenced pre-stage failures."""


class NodeModelPrestageConflictError(NodeModelPrestageError):
    """A command or claim conflicts with already accepted placement work."""


class NodeModelPrestageExpiredError(NodeModelPrestageError):
    """A command is outside its leader-authorized execution window."""


class NodeModelPrestageCapacityError(NodeModelPrestageError):
    """The bounded node-local command ledger is full."""


def _text(value: str, *, name: str, max_length: int = 255) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL")
    if len(value) > max_length:
        raise ValueError(f"{name} exceeds maximum length")
    return value


def _digest(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    assert isinstance(value, int)
    return value


def _aware(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _canonical_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=lambda value: value.isoformat() if isinstance(value, datetime) else str(value),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class NodeModelPrestageCommand(BaseModel):
    """One exact placement mutation authorized by a live controller leader."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-prestage-command-v1"] = (
        "kairyu-node-model-prestage-command-v1"
    )
    action: Literal["ensure", "release"]
    command_id: str = Field(min_length=64, max_length=64)
    command_generation: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    decision_id: str = Field(max_length=255)
    decision_fingerprint: str = Field(min_length=64, max_length=64)
    target_id: str = Field(max_length=255)
    target_revision: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    deployment_id: str = Field(max_length=255)
    placement_binding_id: str = Field(max_length=255)
    snapshot_id: str = Field(max_length=255)
    cache_revision: int = Field(ge=1, le=_MAX_SIGNED_BIGINT)
    placement_id: str = Field(max_length=255)
    node_id: str = Field(max_length=253)
    resource_flavor: str = Field(max_length=253)
    profile_id: str = Field(max_length=255)
    compatibility_approval_id: str = Field(max_length=255)
    model_id: str = Field(max_length=255)
    model_revision: str = Field(max_length=255)
    manifest_digest: str = Field(min_length=64, max_length=64)
    pin_owner: str = Field(max_length=255)
    authority: RunnerWriterAuthority
    issued_at: datetime
    expires_at: datetime

    @field_validator(
        "decision_id",
        "target_id",
        "deployment_id",
        "placement_binding_id",
        "snapshot_id",
        "placement_id",
        "node_id",
        "resource_flavor",
        "profile_id",
        "compatibility_approval_id",
        "model_id",
        "model_revision",
        "pin_owner",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        maximum = 253 if info.field_name in {"node_id", "resource_flavor"} else 255
        return _text(value, name=info.field_name, max_length=maximum)

    @field_validator("command_id", "decision_fingerprint", "manifest_digest")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _digest(value, name=info.field_name)

    @field_validator(
        "command_generation",
        "target_revision",
        "cache_revision",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> int:
        return _integer(value, name=info.field_name)

    @field_validator("issued_at", "expires_at")
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        return _aware(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_fence(self) -> NodeModelPrestageCommand:
        if not self.authority.validated_at <= self.issued_at < self.authority.lease_until:
            raise ValueError("command must be issued inside the validated leader lease")
        if not self.issued_at < self.expires_at <= self.authority.lease_until:
            raise ValueError("command expiry must remain inside the leader lease")
        expected = _canonical_digest(self.model_dump(mode="json", exclude={"command_id"}))
        if self.command_id != expected:
            raise ValueError("command_id must match the canonical command payload")
        return self


class NodeModelPrestageRecord(BaseModel):
    """Serializable placement-level claim and completion evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-prestage-record-v1"] = (
        "kairyu-node-model-prestage-record-v1"
    )
    command: NodeModelPrestageCommand
    state: ModelCachePlacementState
    attempt: int = Field(ge=0, le=_MAX_SIGNED_BIGINT)
    claim_id: str | None = Field(default=None, min_length=64, max_length=64)
    failure: str | None = Field(default=None, max_length=1024)
    fill_result: NodeModelCacheFillResult | None = None
    updated_at: datetime

    @field_validator("attempt", mode="before")
    @classmethod
    def validate_attempt(cls, value: object) -> int:
        return _integer(value, name="attempt")

    @field_validator("claim_id")
    @classmethod
    def validate_claim_id(cls, value: str | None) -> str | None:
        return None if value is None else _digest(value, name="claim_id")

    @field_validator("failure")
    @classmethod
    def validate_failure(cls, value: str | None) -> str | None:
        return None if value is None else _text(value, name="failure", max_length=1024)

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, value: datetime) -> datetime:
        return _aware(value, name="updated_at")

    @model_validator(mode="after")
    def validate_state(self) -> NodeModelPrestageRecord:
        if self.state is ModelCachePlacementState.FILLING:
            if self.attempt < 1 or self.claim_id is None or self.failure is not None:
                raise ValueError("filling pre-stage state requires only an active claim")
        elif self.state is ModelCachePlacementState.READY:
            if self.attempt < 1 or self.claim_id is not None or self.fill_result is None:
                raise ValueError("ready pre-stage state requires fill evidence and no claim")
            if self.failure is not None:
                raise ValueError("ready pre-stage state cannot retain failure evidence")
            if (
                self.fill_result.manifest_digest != self.command.manifest_digest
                or self.fill_result.deployment_id != self.command.deployment_id
            ):
                raise ValueError("fill evidence must match the command artifact")
        elif self.state is ModelCachePlacementState.FAILED:
            if self.attempt < 1 or self.claim_id is not None or self.failure is None:
                raise ValueError("failed pre-stage state requires failure evidence")
            if self.fill_result is not None:
                raise ValueError("failed pre-stage state cannot retain fill evidence")
        else:
            if (
                self.claim_id is not None
                or self.failure is not None
                or self.fill_result is not None
            ):
                raise ValueError("absent pre-stage state cannot retain execution evidence")
        return self


@runtime_checkable
class NodeModelPrestageStore(Protocol):
    """Atomic placement claim/CAS contract for a durable production backend."""

    @property
    def node_id(self) -> str: ...

    def claim(
        self,
        command: NodeModelPrestageCommand,
        *,
        claim_id: str,
        now: datetime,
    ) -> NodeModelPrestageRecord: ...

    def complete(
        self,
        command: NodeModelPrestageCommand,
        *,
        claim_id: str,
        fill_result: NodeModelCacheFillResult,
        now: datetime,
    ) -> NodeModelPrestageRecord: ...

    def fail(
        self,
        command: NodeModelPrestageCommand,
        *,
        claim_id: str,
        failure: str,
        now: datetime,
    ) -> NodeModelPrestageRecord: ...

    def release(
        self,
        command: NodeModelPrestageCommand,
        *,
        now: datetime,
    ) -> NodeModelPrestageRecord: ...

    def list_records(self) -> tuple[NodeModelPrestageRecord, ...]: ...

    def list_records_page(
        self,
        *,
        after_placement_id: str | None = None,
        limit: int = 100,
    ) -> tuple[NodeModelPrestageRecord, ...]: ...


def _copy_prestage_record(record: NodeModelPrestageRecord) -> NodeModelPrestageRecord:
    return NodeModelPrestageRecord.model_validate(record.model_dump())


class _NodeModelPrestageTransitions:
    """Pure state machine shared by volatile and durable store adapters."""

    def __init__(self, *, node_id: str) -> None:
        self.node_id = _text(node_id, name="node_id", max_length=253)

    def validate_command(
        self,
        command: NodeModelPrestageCommand,
        *,
        now: datetime,
        require_active: bool,
    ) -> NodeModelPrestageCommand:
        if not isinstance(command, NodeModelPrestageCommand):
            raise TypeError("command must be a NodeModelPrestageCommand")
        command = NodeModelPrestageCommand.model_validate(command.model_dump())
        now = _aware(now, name="now")
        if command.node_id != self.node_id:
            raise NodeModelPrestageConflictError("command targets another node")
        if require_active and not command.issued_at <= now < command.expires_at:
            raise NodeModelPrestageExpiredError("pre-stage command is not currently valid")
        return command

    @staticmethod
    def _same_command(
        existing: NodeModelPrestageRecord,
        command: NodeModelPrestageCommand,
    ) -> bool:
        return existing.command.model_dump() == command.model_dump()

    @staticmethod
    def _require_monotonic_time(
        existing: NodeModelPrestageRecord,
        *,
        now: datetime,
    ) -> None:
        if now < existing.updated_at:
            raise NodeModelPrestageConflictError("pre-stage record time regressed")

    @staticmethod
    def _validate_successor(
        existing: NodeModelPrestageRecord,
        command: NodeModelPrestageCommand,
    ) -> None:
        previous = existing.command
        if command.command_generation <= previous.command_generation:
            raise NodeModelPrestageConflictError("command generation did not advance")
        if command.authority.election_id != previous.authority.election_id:
            raise NodeModelPrestageConflictError("leader election identity changed")
        if command.authority.fencing_token < previous.authority.fencing_token:
            raise NodeModelPrestageConflictError("leader fencing token regressed")
        if (
            command.authority.fencing_token == previous.authority.fencing_token
            and command.authority.holder_id != previous.authority.holder_id
        ):
            raise NodeModelPrestageConflictError("leader holder changed without a new fence")
        if (
            command.target_id == previous.target_id
            and command.target_revision < previous.target_revision
        ):
            raise NodeModelPrestageConflictError("target revision regressed")

    def claim(
        self,
        existing: NodeModelPrestageRecord | None,
        *,
        record_count: int,
        max_placements: int,
        command: NodeModelPrestageCommand,
        claim_id: str,
        now: datetime,
    ) -> NodeModelPrestageRecord:
        command = self.validate_command(command, now=now, require_active=False)
        if command.action != "ensure":
            raise NodeModelPrestageConflictError("release command cannot claim a cache fill")
        claim_id = _digest(claim_id, name="claim_id")
        now = _aware(now, name="now")
        if existing is None:
            if record_count >= max_placements:
                raise NodeModelPrestageCapacityError(
                    "pre-stage placement capacity is exhausted"
                )
            attempt = 1
        elif existing.command.command_id == command.command_id:
            if not self._same_command(existing, command):
                raise NodeModelPrestageConflictError("command ID payload conflict")
            if existing.state is ModelCachePlacementState.READY:
                return _copy_prestage_record(existing)
            if existing.state is ModelCachePlacementState.FILLING:
                if existing.claim_id != claim_id:
                    raise NodeModelPrestageConflictError(
                        "placement is claimed by another attempt"
                    )
                return _copy_prestage_record(existing)
            if existing.state is ModelCachePlacementState.ABSENT:
                raise NodeModelPrestageConflictError("released command cannot be reactivated")
            if not command.issued_at <= now < command.expires_at:
                raise NodeModelPrestageExpiredError("pre-stage command is not currently valid")
            self._require_monotonic_time(existing, now=now)
            attempt = existing.attempt + 1
        else:
            if not command.issued_at <= now < command.expires_at:
                raise NodeModelPrestageExpiredError("pre-stage command is not currently valid")
            if existing.state is not ModelCachePlacementState.ABSENT:
                raise NodeModelPrestageConflictError(
                    "replacement ensure requires a released absent placement"
                )
            self._validate_successor(existing, command)
            self._require_monotonic_time(existing, now=now)
            attempt = 1
        if existing is None and not command.issued_at <= now < command.expires_at:
            raise NodeModelPrestageExpiredError("pre-stage command is not currently valid")
        return NodeModelPrestageRecord(
            command=command,
            state=ModelCachePlacementState.FILLING,
            attempt=attempt,
            claim_id=claim_id,
            updated_at=now,
        )

    def complete(
        self,
        existing: NodeModelPrestageRecord | None,
        *,
        command: NodeModelPrestageCommand,
        claim_id: str,
        fill_result: NodeModelCacheFillResult,
        now: datetime,
    ) -> NodeModelPrestageRecord:
        command = self.validate_command(command, now=now, require_active=False)
        claim_id = _digest(claim_id, name="claim_id")
        if not isinstance(fill_result, NodeModelCacheFillResult):
            raise TypeError("fill_result must be a NodeModelCacheFillResult")
        fill_result = NodeModelCacheFillResult.model_validate(fill_result.model_dump())
        now = _aware(now, name="now")
        existing = self._require_claim(existing, command=command, claim_id=claim_id)
        self._require_monotonic_time(existing, now=now)
        return NodeModelPrestageRecord(
            command=command,
            state=ModelCachePlacementState.READY,
            attempt=existing.attempt,
            fill_result=fill_result,
            updated_at=now,
        )

    def fail(
        self,
        existing: NodeModelPrestageRecord | None,
        *,
        command: NodeModelPrestageCommand,
        claim_id: str,
        failure: str,
        now: datetime,
    ) -> NodeModelPrestageRecord:
        command = self.validate_command(command, now=now, require_active=False)
        claim_id = _digest(claim_id, name="claim_id")
        failure = _text(failure, name="failure", max_length=1024)
        now = _aware(now, name="now")
        existing = self._require_claim(existing, command=command, claim_id=claim_id)
        self._require_monotonic_time(existing, now=now)
        return NodeModelPrestageRecord(
            command=command,
            state=ModelCachePlacementState.FAILED,
            attempt=existing.attempt,
            failure=failure,
            updated_at=now,
        )

    def release(
        self,
        existing: NodeModelPrestageRecord | None,
        *,
        command: NodeModelPrestageCommand,
        now: datetime,
    ) -> NodeModelPrestageRecord:
        command = self.validate_command(command, now=now, require_active=True)
        if command.action != "release":
            raise NodeModelPrestageConflictError("owner pin release requires a release command")
        now = _aware(now, name="now")
        if existing is None:
            raise NodeModelPrestageConflictError("release has no accepted placement")
        if existing.command.command_id == command.command_id:
            if not self._same_command(existing, command):
                raise NodeModelPrestageConflictError("release command ID payload conflict")
            if existing.state is not ModelCachePlacementState.ABSENT:
                raise NodeModelPrestageConflictError("release replay has inconsistent state")
            return _copy_prestage_record(existing)
        if existing.state is ModelCachePlacementState.FILLING:
            raise NodeModelPrestageConflictError("cannot release an active fill claim")
        self._validate_successor(existing, command)
        self._validate_release_identity(existing.command, command)
        self._require_monotonic_time(existing, now=now)
        return NodeModelPrestageRecord(
            command=command,
            state=ModelCachePlacementState.ABSENT,
            attempt=existing.attempt,
            updated_at=now,
        )

    @staticmethod
    def _validate_release_identity(
        previous: NodeModelPrestageCommand,
        release: NodeModelPrestageCommand,
    ) -> None:
        fields = (
            "target_id",
            "deployment_id",
            "placement_binding_id",
            "placement_id",
            "node_id",
            "resource_flavor",
            "profile_id",
            "compatibility_approval_id",
            "model_id",
            "model_revision",
            "manifest_digest",
            "pin_owner",
        )
        if any(getattr(previous, field) != getattr(release, field) for field in fields):
            raise NodeModelPrestageConflictError("release identity does not match pinned placement")

    def _require_claim(
        self,
        existing: NodeModelPrestageRecord | None,
        *,
        command: NodeModelPrestageCommand,
        claim_id: str,
    ) -> NodeModelPrestageRecord:
        if (
            existing is None
            or not self._same_command(existing, command)
            or existing.state is not ModelCachePlacementState.FILLING
            or existing.claim_id != claim_id
        ):
            raise NodeModelPrestageConflictError("pre-stage completion claim is stale")
        return existing


class InMemoryNodeModelPrestageStore:
    """Thread-safe executable specification for pre-stage store semantics."""

    def __init__(self, *, node_id: str, max_placements: int = 100_000) -> None:
        if type(max_placements) is not int or not 1 <= max_placements <= 100_000:
            raise ValueError("max_placements must be an integer in [1, 100000]")
        self._transitions = _NodeModelPrestageTransitions(node_id=node_id)
        self._max_placements = max_placements
        self._records: dict[str, NodeModelPrestageRecord] = {}
        self._lock = threading.Lock()

    @property
    def node_id(self) -> str:
        return self._transitions.node_id

    def claim(
        self,
        command: NodeModelPrestageCommand,
        *,
        claim_id: str,
        now: datetime,
    ) -> NodeModelPrestageRecord:
        command = self._transitions.validate_command(
            command, now=now, require_active=False
        )
        placement_id = command.placement_id
        with self._lock:
            record = self._transitions.claim(
                self._records.get(placement_id),
                record_count=len(self._records),
                max_placements=self._max_placements,
                command=command,
                claim_id=claim_id,
                now=now,
            )
            self._records[placement_id] = record
            return _copy_prestage_record(record)

    def complete(
        self,
        command: NodeModelPrestageCommand,
        *,
        claim_id: str,
        fill_result: NodeModelCacheFillResult,
        now: datetime,
    ) -> NodeModelPrestageRecord:
        command = self._transitions.validate_command(
            command, now=now, require_active=False
        )
        placement_id = command.placement_id
        with self._lock:
            record = self._transitions.complete(
                self._records.get(placement_id),
                command=command,
                claim_id=claim_id,
                fill_result=fill_result,
                now=now,
            )
            self._records[placement_id] = record
            return _copy_prestage_record(record)

    def fail(
        self,
        command: NodeModelPrestageCommand,
        *,
        claim_id: str,
        failure: str,
        now: datetime,
    ) -> NodeModelPrestageRecord:
        command = self._transitions.validate_command(
            command, now=now, require_active=False
        )
        placement_id = command.placement_id
        with self._lock:
            record = self._transitions.fail(
                self._records.get(placement_id),
                command=command,
                claim_id=claim_id,
                failure=failure,
                now=now,
            )
            self._records[placement_id] = record
            return _copy_prestage_record(record)

    def release(
        self,
        command: NodeModelPrestageCommand,
        *,
        now: datetime,
    ) -> NodeModelPrestageRecord:
        command = self._transitions.validate_command(
            command, now=now, require_active=True
        )
        placement_id = command.placement_id
        with self._lock:
            record = self._transitions.release(
                self._records.get(placement_id),
                command=command,
                now=now,
            )
            self._records[placement_id] = record
            return _copy_prestage_record(record)

    def list_records(self) -> tuple[NodeModelPrestageRecord, ...]:
        with self._lock:
            return tuple(
                _copy_prestage_record(self._records[key]) for key in sorted(self._records)
            )

    def list_records_page(
        self,
        *,
        after_placement_id: str | None = None,
        limit: int = 100,
    ) -> tuple[NodeModelPrestageRecord, ...]:
        if after_placement_id is not None:
            after_placement_id = _text(
                after_placement_id, name="after_placement_id"
            )
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer in [1, 1000]")
        with self._lock:
            keys = (
                key
                for key in sorted(self._records)
                if after_placement_id is None or key > after_placement_id
            )
            selected = tuple(islice(keys, limit))
            return tuple(_copy_prestage_record(self._records[key]) for key in selected)


class NodeModelPrestageExecutor:
    """Execute one fenced command through verified fill and owner-scoped pinning."""

    def __init__(
        self,
        *,
        node_id: str,
        agent: NodeModelCacheAgent,
        index: NodeModelCacheIndex,
        store: NodeModelPrestageStore,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._node_id = _text(node_id, name="node_id", max_length=253)
        if index.node_id != self._node_id:
            raise ValueError("cache index belongs to another node")
        if not isinstance(store, NodeModelPrestageStore):
            raise TypeError("store must implement NodeModelPrestageStore")
        if store.node_id != self._node_id:
            raise ValueError("pre-stage store belongs to another node")
        self._agent = agent
        self._index = index
        self._store = store
        self._clock = clock

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def store(self) -> NodeModelPrestageStore:
        return self._store

    def execute(
        self,
        command: NodeModelPrestageCommand,
        *,
        claim_id: str,
        envelope: SignedModelArtifactManifest,
        trust_store: ModelArtifactTrustStore,
        request: ModelArtifactAdmissionRequest,
    ) -> NodeModelPrestageRecord:
        """Claim, fill, pin, and CAS-complete one exact node placement."""

        command = NodeModelPrestageCommand.model_validate(command.model_dump())
        if command.action != "ensure":
            raise NodeModelPrestageConflictError("executor requires an ensure command")
        if command.node_id != self._node_id:
            raise NodeModelPrestageConflictError("executor command targets another node")
        self._validate_artifact_binding(command, request=request)
        now = _aware(self._clock(), name="executor clock")
        claimed = self._store.claim(command, claim_id=claim_id, now=now)
        if claimed.state is ModelCachePlacementState.READY:
            return claimed
        try:
            result = self._agent.ensure_cached(envelope, trust_store, request)
            if result.manifest_digest != command.manifest_digest:
                raise NodeModelPrestageConflictError("cache fill returned another artifact")
            pinned = self._index.pin(
                command.manifest_digest,
                owner=command.pin_owner,
                reason=f"prestage:{command.command_id}",
            )
            self._validate_pinned(command, pinned, fill_result=result)
            return self._store.complete(
                command,
                claim_id=claim_id,
                fill_result=result,
                now=_aware(self._clock(), name="executor clock"),
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"[:1024]
            try:
                self._store.fail(
                    command,
                    claim_id=claim_id,
                    failure=failure,
                    now=_aware(self._clock(), name="executor clock"),
                )
            except NodeModelPrestageError:
                pass
            raise

    def release(self, command: NodeModelPrestageCommand) -> NodeModelPrestageRecord:
        """Release desired work first, then idempotently remove only its cache pin."""

        command = NodeModelPrestageCommand.model_validate(command.model_dump())
        if command.action != "release":
            raise NodeModelPrestageConflictError("release requires a fenced release command")
        if command.node_id != self._node_id:
            raise NodeModelPrestageConflictError("executor command targets another node")
        cached = self._index.get(command.manifest_digest)
        record = self._store.release(
            command,
            now=_aware(self._clock(), name="executor clock"),
        )
        if cached is not None:
            self._index.unpin(command.manifest_digest, owner=command.pin_owner)
        return record

    @staticmethod
    def _validate_artifact_binding(
        command: NodeModelPrestageCommand,
        *,
        request: ModelArtifactAdmissionRequest,
    ) -> None:
        if not isinstance(request, ModelArtifactAdmissionRequest):
            raise TypeError("request must be a ModelArtifactAdmissionRequest")
        request = ModelArtifactAdmissionRequest.model_validate(request.model_dump())
        if (
            request.deployment_id,
            request.manifest_digest,
            request.model_id,
            request.model_revision,
            request.gpu_profile,
        ) != (
            command.deployment_id,
            command.manifest_digest,
            command.model_id,
            command.model_revision,
            command.profile_id,
        ):
            raise NodeModelPrestageConflictError(
                "artifact admission request does not match pre-stage command"
            )

    @staticmethod
    def _validate_pinned(
        command: NodeModelPrestageCommand,
        record: NodeModelCacheRecord,
        *,
        fill_result: NodeModelCacheFillResult,
    ) -> None:
        if (
            not record.verified
            or record.manifest_digest != command.manifest_digest
            or record.model_id != command.model_id
            or record.model_revision != command.model_revision
            or record.artifact_path != fill_result.artifact_path
            or command.pin_owner not in record.pin_owners
        ):
            raise NodeModelPrestageConflictError("verified cache pin does not match command")


def build_node_model_prestage_commands(
    plan: ScalingPrewarmPlan,
    *,
    authority: RunnerWriterAuthority,
    decision_id: str,
    decision_fingerprint: str,
    target_id: str,
    target_revision: int,
    deployment_id: str,
    model_id: str,
    command_generations: Mapping[str, int],
    issued_at: datetime,
    ttl_seconds: float,
) -> tuple[NodeModelPrestageCommand, ...]:
    """Convert durable WP3.6 absent-placement intent into fenced node commands."""

    if not isinstance(plan, ScalingPrewarmPlan):
        raise TypeError("plan must be a ScalingPrewarmPlan")
    plan = ScalingPrewarmPlan.model_validate(plan.model_dump())
    if not isinstance(authority, RunnerWriterAuthority):
        raise TypeError("authority must be a RunnerWriterAuthority")
    authority = RunnerWriterAuthority.model_validate(authority.model_dump())
    decision_id = _text(decision_id, name="decision_id")
    decision_fingerprint = _digest(decision_fingerprint, name="decision_fingerprint")
    target_id = _text(target_id, name="target_id")
    deployment_id = _text(deployment_id, name="deployment_id")
    model_id = _text(model_id, name="model_id")
    target_revision = _integer(target_revision, name="target_revision")
    if not 1 <= target_revision <= _MAX_SIGNED_BIGINT:
        raise ValueError("target_revision must be in [1, 2^63-1]")
    issued_at = _aware(issued_at, name="issued_at")
    if issued_at < plan.snapshot.observed_at:
        raise ValueError("issued_at cannot precede the cache snapshot")
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
        raise ValueError("ttl_seconds must be a number")
    if not math.isfinite(float(ttl_seconds)) or ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be finite and positive")
    expires_at = issued_at + timedelta(seconds=float(ttl_seconds))
    placement_ids = plan.cache_fill_placement_ids
    if set(command_generations) != set(placement_ids):
        raise ValueError("command_generations must exactly cover cache-fill placements")
    placements = {placement.placement_id: placement for placement in plan.snapshot.placements}
    commands = []
    for placement_id in placement_ids:
        placement = placements[placement_id]
        generation = _integer(
            command_generations[placement_id],
            name=f"command_generations[{placement_id!r}]",
        )
        if not 1 <= generation <= _MAX_SIGNED_BIGINT:
            raise ValueError("command generation must be in [1, 2^63-1]")
        pin_owner = f"prestage/{deployment_id}/{placement_id}"
        _text(pin_owner, name="pin_owner")
        payload = {
            "schema_version": "kairyu-node-model-prestage-command-v1",
            "action": "ensure",
            "command_generation": generation,
            "decision_id": decision_id,
            "decision_fingerprint": decision_fingerprint,
            "target_id": target_id,
            "target_revision": target_revision,
            "deployment_id": deployment_id,
            "placement_binding_id": plan.snapshot.placement_binding_id,
            "snapshot_id": plan.snapshot.snapshot_id,
            "cache_revision": plan.snapshot.cache_revision,
            "placement_id": placement.placement_id,
            "node_id": placement.node_name,
            "resource_flavor": placement.resource_flavor,
            "profile_id": placement.profile_id,
            "compatibility_approval_id": placement.compatibility_approval_id,
            "model_id": model_id,
            "model_revision": plan.snapshot.model_revision,
            "manifest_digest": plan.snapshot.artifact_digest,
            "pin_owner": pin_owner,
            "authority": authority,
            "issued_at": issued_at,
            "expires_at": expires_at,
        }
        unsigned = NodeModelPrestageCommand.model_construct(command_id="0" * 64, **payload)
        command_id = _canonical_digest(unsigned.model_dump(mode="json", exclude={"command_id"}))
        commands.append(NodeModelPrestageCommand(command_id=command_id, **payload))
    return tuple(commands)


def build_node_model_prestage_release_command(
    previous: NodeModelPrestageCommand,
    *,
    authority: RunnerWriterAuthority,
    decision_id: str,
    decision_fingerprint: str,
    target_revision: int,
    command_generation: int,
    issued_at: datetime,
    ttl_seconds: float,
) -> NodeModelPrestageCommand:
    """Authorize owner-pin removal with a fresh leader and generation fence."""

    if not isinstance(previous, NodeModelPrestageCommand):
        raise TypeError("previous must be a NodeModelPrestageCommand")
    previous = NodeModelPrestageCommand.model_validate(previous.model_dump())
    if not isinstance(authority, RunnerWriterAuthority):
        raise TypeError("authority must be a RunnerWriterAuthority")
    authority = RunnerWriterAuthority.model_validate(authority.model_dump())
    decision_id = _text(decision_id, name="decision_id")
    decision_fingerprint = _digest(decision_fingerprint, name="decision_fingerprint")
    target_revision = _integer(target_revision, name="target_revision")
    command_generation = _integer(command_generation, name="command_generation")
    if not 1 <= target_revision <= _MAX_SIGNED_BIGINT:
        raise ValueError("target_revision must be in [1, 2^63-1]")
    if target_revision <= previous.target_revision:
        raise ValueError("release target_revision must advance")
    if not 1 <= command_generation <= _MAX_SIGNED_BIGINT:
        raise ValueError("command_generation must be in [1, 2^63-1]")
    if command_generation <= previous.command_generation:
        raise ValueError("release command generation must advance")
    issued_at = _aware(issued_at, name="issued_at")
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
        raise ValueError("ttl_seconds must be a number")
    if not math.isfinite(float(ttl_seconds)) or ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be finite and positive")
    payload = previous.model_dump(exclude={"command_id"})
    payload.update(
        action="release",
        command_generation=command_generation,
        decision_id=decision_id,
        decision_fingerprint=decision_fingerprint,
        target_revision=target_revision,
        authority=authority,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(seconds=float(ttl_seconds)),
    )
    unsigned = NodeModelPrestageCommand.model_construct(command_id="0" * 64, **payload)
    command_id = _canonical_digest(unsigned.model_dump(mode="json", exclude={"command_id"}))
    return NodeModelPrestageCommand(command_id=command_id, **payload)


def apply_node_model_prestage_records(
    snapshot: ScalingPrewarmSnapshot,
    records: tuple[NodeModelPrestageRecord, ...],
    *,
    snapshot_id: str,
    cache_revision: int,
    observed_at: datetime,
) -> ScalingPrewarmSnapshot:
    """Overlay exact command progress onto a later controller cache snapshot."""

    if not isinstance(snapshot, ScalingPrewarmSnapshot):
        raise TypeError("snapshot must be a ScalingPrewarmSnapshot")
    snapshot = ScalingPrewarmSnapshot.model_validate(snapshot.model_dump())
    snapshot_id = _text(snapshot_id, name="snapshot_id")
    cache_revision = _integer(cache_revision, name="cache_revision")
    if not snapshot.cache_revision < cache_revision <= _MAX_SIGNED_BIGINT:
        raise ValueError("cache_revision must advance monotonically")
    observed_at = _aware(observed_at, name="observed_at")
    if observed_at < snapshot.observed_at:
        raise ValueError("observed_at cannot move backwards")
    if not isinstance(records, tuple):
        raise TypeError("records must be a tuple")
    by_placement: dict[str, NodeModelPrestageRecord] = {}
    placements = {placement.placement_id: placement for placement in snapshot.placements}
    for value in records:
        if not isinstance(value, NodeModelPrestageRecord):
            raise TypeError("records must contain NodeModelPrestageRecord values")
        record = NodeModelPrestageRecord.model_validate(value.model_dump())
        command = record.command
        placement = placements.get(command.placement_id)
        if placement is None or command.placement_id in by_placement:
            raise NodeModelPrestageConflictError(
                "pre-stage records must map uniquely to placements"
            )
        if record.updated_at > observed_at or command.cache_revision > snapshot.cache_revision:
            raise NodeModelPrestageConflictError(
                "pre-stage record is future-dated relative to the source snapshot"
            )
        if (
            command.placement_binding_id,
            command.model_revision,
            command.manifest_digest,
            command.node_id,
            command.resource_flavor,
            command.profile_id,
            command.compatibility_approval_id,
        ) != (
            snapshot.placement_binding_id,
            snapshot.model_revision,
            snapshot.artifact_digest,
            placement.node_name,
            placement.resource_flavor,
            placement.profile_id,
            placement.compatibility_approval_id,
        ):
            raise NodeModelPrestageConflictError(
                "pre-stage record identity does not match snapshot"
            )
        by_placement[command.placement_id] = record
    updated = []
    for placement in snapshot.placements:
        record = by_placement.get(placement.placement_id)
        state = placement.state
        if state is ModelCachePlacementState.READY and (
            placement.cache_hint_observed_at is None
            or placement.cache_hint_valid_until is None
            or placement.cache_hint_index_revision is None
            or observed_at >= placement.cache_hint_valid_until
        ):
            state = ModelCachePlacementState.ABSENT
        if record is not None:
            if record.state is ModelCachePlacementState.READY:
                if (
                    placement.state is ModelCachePlacementState.READY
                    and placement.cache_hint_observed_at is not None
                    and placement.cache_hint_valid_until is not None
                    and placement.cache_hint_index_revision is not None
                    and observed_at < placement.cache_hint_valid_until
                ):
                    state = ModelCachePlacementState.READY
                elif (
                    placement.cache_hint_observed_at is None
                    or placement.cache_hint_observed_at < record.updated_at
                ):
                    state = ModelCachePlacementState.FILLING
                else:
                    state = ModelCachePlacementState.FAILED
            elif record.state is not ModelCachePlacementState.ABSENT:
                state = record.state
        updated.append(placement.model_copy(update={"state": state}))
    return ScalingPrewarmSnapshot(
        snapshot_id=snapshot_id,
        cache_revision=cache_revision,
        observed_at=observed_at,
        model_class=snapshot.model_class,
        model_revision=snapshot.model_revision,
        artifact_digest=snapshot.artifact_digest,
        placement_binding_id=snapshot.placement_binding_id,
        placements=tuple(updated),
    )
