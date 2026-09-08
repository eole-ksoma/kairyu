"""Backend-neutral models for asynchronous online inference requests."""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class AsyncRequestState(StrEnum):
    """Lifecycle states owned by a request store, not by a worker."""

    QUEUED = "queued"
    CLAIMED = "claimed"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_REQUEST_STATES = frozenset(
    {
        AsyncRequestState.SUCCEEDED,
        AsyncRequestState.FAILED,
        AsyncRequestState.CANCELLED,
        AsyncRequestState.EXPIRED,
    }
)


def _non_empty(value: str, *, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{name} cannot contain NUL")
    return value


def _aware(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _require_finite_json(value: JsonValue, *, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} cannot contain non-finite numbers")
    if isinstance(value, list):
        for item in value:
            _require_finite_json(item, name=name)
    elif isinstance(value, dict):
        for item in value.values():
            _require_finite_json(item, name=name)


class AsyncRequestError(BaseModel):
    """Sanitized terminal error safe to return through a public API."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(max_length=128)
    message: str = Field(max_length=1024)
    retryable: bool = False

    @field_validator("code", "message")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)


class AsyncRequestSubmission(BaseModel):
    """Top-level-frozen caller intent used for idempotent request creation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    owner: str = "default"
    endpoint: str
    body: dict[str, JsonValue]
    priority: int = Field(default=0, ge=-(2**63), le=2**63 - 1)
    idempotency_key: str | None = Field(default=None, max_length=255)
    metadata: dict[str, str] | None = None
    deadline_at: datetime | None = None

    @field_validator("owner", "endpoint")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _non_empty(value, name="idempotency_key")

    @field_validator("deadline_at")
    @classmethod
    def validate_deadline(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _aware(value, name="deadline_at")

    @field_validator("body")
    @classmethod
    def validate_body(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _require_finite_json(value, name="body")
        return value


class AsyncRequest(BaseModel):
    """Public request snapshot. Lease ownership stays in ``RequestClaim``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    object: Literal["async.request"] = "async.request"
    owner: str
    endpoint: str
    body: dict[str, JsonValue]
    priority: int = Field(ge=-(2**63), le=2**63 - 1)
    idempotency_key: str | None = None
    metadata: dict[str, str] | None = None
    state: AsyncRequestState
    attempt: int = Field(default=0, ge=0)
    created_at: datetime
    updated_at: datetime
    deadline_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, JsonValue] | None = None
    error: AsyncRequestError | None = None

    @field_validator("id", "owner", "endpoint")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator("created_at", "updated_at", "deadline_at", "completed_at")
    @classmethod
    def validate_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return _aware(value, name=info.field_name)

    @field_validator("body", "result")
    @classmethod
    def validate_json_objects(
        cls,
        value: dict[str, JsonValue] | None,
        info,
    ) -> dict[str, JsonValue] | None:
        if value is None:
            return None
        _require_finite_json(value, name=info.field_name)
        return value

    @model_validator(mode="after")
    def validate_terminal_payload(self) -> AsyncRequest:
        if self.state is AsyncRequestState.SUCCEEDED:
            if self.result is None or self.error is not None:
                raise ValueError("succeeded requests require result and forbid error")
        elif self.state is AsyncRequestState.FAILED:
            if self.error is None or self.result is not None:
                raise ValueError("failed requests require error and forbid result")
        elif self.result is not None or self.error is not None:
            raise ValueError("only succeeded or failed requests can carry terminal payloads")
        if self.state in TERMINAL_REQUEST_STATES:
            if self.completed_at is None:
                raise ValueError("terminal requests require completed_at")
        elif self.completed_at is not None:
            raise ValueError("non-terminal requests cannot set completed_at")
        return self


class AsyncRequestStatus(BaseModel):
    """Bounded public projection that never exposes persisted input or output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    object: Literal["async.request"] = "async.request"
    owner: str
    endpoint: str
    priority: int = Field(ge=-(2**63), le=2**63 - 1)
    idempotency_key: str | None = None
    metadata: dict[str, str] | None = None
    state: AsyncRequestState
    attempt: int = Field(default=0, ge=0)
    created_at: datetime
    updated_at: datetime
    deadline_at: datetime | None = None
    completed_at: datetime | None = None
    error: AsyncRequestError | None = None
    has_result: bool = False

    @field_validator("id", "owner", "endpoint")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator("created_at", "updated_at", "deadline_at", "completed_at")
    @classmethod
    def validate_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return _aware(value, name=info.field_name)


def status_of(request: AsyncRequest) -> AsyncRequestStatus:
    """Create the public, size-bounded status view of a full store record."""

    return AsyncRequestStatus(
        id=request.id,
        owner=request.owner,
        endpoint=request.endpoint,
        priority=request.priority,
        idempotency_key=request.idempotency_key,
        metadata=request.metadata,
        state=request.state,
        attempt=request.attempt,
        created_at=request.created_at,
        updated_at=request.updated_at,
        deadline_at=request.deadline_at,
        completed_at=request.completed_at,
        error=request.error,
        has_result=request.result is not None,
    ).model_copy(deep=True)


class RequestClaim(BaseModel):
    """A fenced, expiring worker lease over one request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    store_id: str
    request: AsyncRequest
    worker_id: str
    fencing_token: int = Field(gt=0)
    claimed_at: datetime
    lease_until: datetime

    @field_validator("store_id", "worker_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator("claimed_at", "lease_until")
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        return _aware(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_lease_window(self) -> RequestClaim:
        if self.lease_until <= self.claimed_at:
            raise ValueError("lease_until must be later than claimed_at")
        if self.request.state not in {AsyncRequestState.CLAIMED, AsyncRequestState.RUNNING}:
            raise ValueError("claims require a claimed or running request snapshot")
        return self

    @property
    def request_id(self) -> str:
        return self.request.id
