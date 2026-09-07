"""Request-store contract and deterministic in-memory reference backend."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from kairyu.async_requests.models import (
    TERMINAL_REQUEST_STATES,
    AsyncRequest,
    AsyncRequestError,
    AsyncRequestState,
    AsyncRequestSubmission,
    RequestClaim,
)


class IdempotencyConflictError(RuntimeError):
    """An owner reused an idempotency key with different caller intent."""


class InvalidRequestTransitionError(RuntimeError):
    """The requested lifecycle transition is not valid from the current state."""


class StaleRequestClaimError(RuntimeError):
    """A worker no longer owns the fenced request lease."""


@runtime_checkable
class RequestStoreProtocol(Protocol):
    """Backend-neutral persistence and claim surface for online async work."""

    def submit(self, submission: AsyncRequestSubmission) -> AsyncRequest: ...

    def get(self, request_id: str, *, owner: str | None = None) -> AsyncRequest: ...

    def list(
        self,
        *,
        owner: str | None = None,
        state: AsyncRequestState | None = None,
        limit: int = 20,
    ) -> list[AsyncRequest]: ...

    def claim_next(self, worker_id: str, *, lease_seconds: float) -> RequestClaim | None: ...

    def renew_claim(self, claim: RequestClaim, *, lease_seconds: float) -> RequestClaim: ...

    def mark_running(self, claim: RequestClaim) -> AsyncRequest: ...

    def succeed(self, claim: RequestClaim, result: dict[str, JsonValue]) -> AsyncRequest: ...

    def fail(self, claim: RequestClaim, error: AsyncRequestError) -> AsyncRequest: ...

    def cancel(self, request_id: str, *, owner: str | None = None) -> AsyncRequest: ...


@dataclass(frozen=True)
class _Lease:
    worker_id: str
    fencing_token: int
    claimed_at: datetime
    lease_until: datetime


class InMemoryRequestStore:
    """Thread-safe executable specification for RequestStore semantics.

    This backend is intentionally process-local and is suitable for unit tests
    and development only. Production HA deployments must provide the same
    protocol over shared durable storage.
    """

    def __init__(
        self,
        *,
        store_id: str = "memory",
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not store_id.strip() or "\x00" in store_id:
            raise ValueError("store_id must be a non-empty string without NUL")
        self._store_id = store_id
        self._clock = clock or (lambda: datetime.now(UTC))
        self._id_factory = id_factory or (lambda: f"req-{uuid.uuid4().hex[:24]}")
        self._requests: dict[str, AsyncRequest] = {}
        self._leases: dict[str, _Lease] = {}
        self._fencing_tokens: dict[str, int] = {}
        self._idempotency: dict[tuple[str, str], tuple[str, str]] = {}
        self._lock = threading.RLock()

    @property
    def store_id(self) -> str:
        return self._store_id

    def submit(self, submission: AsyncRequestSubmission) -> AsyncRequest:
        submission = submission.model_copy(deep=True)
        fingerprint = self._fingerprint(submission)
        with self._lock:
            now = self._now()
            self._expire_due(now)
            if submission.idempotency_key is not None:
                key = (submission.owner, submission.idempotency_key)
                existing = self._idempotency.get(key)
                if existing is not None:
                    existing_fingerprint, request_id = existing
                    if existing_fingerprint != fingerprint:
                        raise IdempotencyConflictError(
                            "idempotency key was already used with different request data"
                        )
                    return self._copy(self._requests[request_id])

            request_id = self._id_factory()
            if request_id in self._requests:
                raise RuntimeError(f"request ID collision: {request_id!r}")
            expired = submission.deadline_at is not None and submission.deadline_at <= now
            request = AsyncRequest(
                id=request_id,
                owner=submission.owner,
                endpoint=submission.endpoint,
                body=submission.body,
                priority=submission.priority,
                idempotency_key=submission.idempotency_key,
                metadata=submission.metadata,
                state=(AsyncRequestState.EXPIRED if expired else AsyncRequestState.QUEUED),
                created_at=now,
                updated_at=now,
                deadline_at=submission.deadline_at,
                completed_at=(now if expired else None),
            )
            self._requests[request_id] = request
            self._fencing_tokens[request_id] = 0
            if submission.idempotency_key is not None:
                self._idempotency[(submission.owner, submission.idempotency_key)] = (
                    fingerprint,
                    request_id,
                )
            return self._copy(request)

    def get(self, request_id: str, *, owner: str | None = None) -> AsyncRequest:
        with self._lock:
            self._expire_due(self._now())
            request = self._load(request_id, owner=owner)
            return self._copy(request)

    def list(
        self,
        *,
        owner: str | None = None,
        state: AsyncRequestState | None = None,
        limit: int = 20,
    ) -> list[AsyncRequest]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        with self._lock:
            self._expire_due(self._now())
            requests = (
                request
                for request in self._requests.values()
                if (owner is None or request.owner == owner)
                and (state is None or request.state is state)
            )
            ordered = sorted(
                requests,
                key=lambda request: (request.created_at, request.id),
                reverse=True,
            )
            return [self._copy(request) for request in ordered[:limit]]

    def claim_next(self, worker_id: str, *, lease_seconds: float) -> RequestClaim | None:
        self._validate_worker(worker_id)
        lease_seconds = self._validate_lease_seconds(lease_seconds)
        with self._lock:
            now = self._now()
            self._expire_due(now)
            candidates = [
                request
                for request in self._requests.values()
                if request.state is AsyncRequestState.QUEUED
                or (
                    request.state in {AsyncRequestState.CLAIMED, AsyncRequestState.RUNNING}
                    and self._leases[request.id].lease_until <= now
                )
            ]
            if not candidates:
                return None
            request = min(
                candidates,
                key=lambda item: (-item.priority, item.created_at, item.id),
            )
            token = self._fencing_tokens[request.id] + 1
            claimed = request.model_copy(
                update={
                    "state": AsyncRequestState.CLAIMED,
                    "attempt": request.attempt + 1,
                    "updated_at": now,
                },
                deep=True,
            )
            lease = _Lease(
                worker_id=worker_id,
                fencing_token=token,
                claimed_at=now,
                lease_until=self._lease_until(request, now, lease_seconds),
            )
            self._requests[request.id] = claimed
            self._leases[request.id] = lease
            self._fencing_tokens[request.id] = token
            return self._claim(claimed, lease)

    def renew_claim(self, claim: RequestClaim, *, lease_seconds: float) -> RequestClaim:
        lease_seconds = self._validate_lease_seconds(lease_seconds)
        with self._lock:
            now = self._now()
            request, lease = self._validate_claim(claim, now=now)
            renewed = replace(
                lease,
                lease_until=self._lease_until(request, now, lease_seconds),
            )
            self._leases[request.id] = renewed
            return self._claim(request, renewed)

    def mark_running(self, claim: RequestClaim) -> AsyncRequest:
        with self._lock:
            now = self._now()
            request, _lease = self._validate_claim(claim, now=now)
            if request.state is not AsyncRequestState.CLAIMED:
                raise InvalidRequestTransitionError(
                    f"cannot mark {request.state.value} request as running"
                )
            running = request.model_copy(
                update={"state": AsyncRequestState.RUNNING, "updated_at": now},
                deep=True,
            )
            self._requests[request.id] = running
            return self._copy(running)

    def succeed(self, claim: RequestClaim, result: dict[str, JsonValue]) -> AsyncRequest:
        return self._finish(
            claim,
            state=AsyncRequestState.SUCCEEDED,
            result=result,
            error=None,
        )

    def fail(self, claim: RequestClaim, error: AsyncRequestError) -> AsyncRequest:
        return self._finish(
            claim,
            state=AsyncRequestState.FAILED,
            result=None,
            error=error,
        )

    def cancel(self, request_id: str, *, owner: str | None = None) -> AsyncRequest:
        with self._lock:
            now = self._now()
            self._expire_due(now)
            request = self._load(request_id, owner=owner)
            if request.state in TERMINAL_REQUEST_STATES:
                return self._copy(request)
            cancelled = request.model_copy(
                update={
                    "state": AsyncRequestState.CANCELLED,
                    "updated_at": now,
                    "completed_at": now,
                },
                deep=True,
            )
            self._requests[request.id] = cancelled
            self._leases.pop(request.id, None)
            self._fencing_tokens[request.id] += 1
            return self._copy(cancelled)

    def _finish(
        self,
        claim: RequestClaim,
        *,
        state: AsyncRequestState,
        result: dict[str, JsonValue] | None,
        error: AsyncRequestError | None,
    ) -> AsyncRequest:
        with self._lock:
            now = self._now()
            request, _lease = self._validate_claim(claim, now=now)
            if request.state is not AsyncRequestState.RUNNING:
                raise InvalidRequestTransitionError(
                    f"cannot finish {request.state.value} request; mark it running first"
                )
            finished = AsyncRequest.model_validate(
                {
                    **request.model_dump(),
                    "state": state,
                    "updated_at": now,
                    "completed_at": now,
                    "result": deepcopy(result),
                    "error": error,
                }
            )
            self._requests[request.id] = finished
            self._leases.pop(request.id, None)
            return self._copy(finished)

    def _validate_claim(
        self,
        claim: RequestClaim,
        *,
        now: datetime,
    ) -> tuple[AsyncRequest, _Lease]:
        if claim.store_id != self._store_id:
            raise ValueError(
                f"claim belongs to store_id {claim.store_id!r}, not {self._store_id!r}"
            )
        request = self._requests.get(claim.request_id)
        lease = self._leases.get(claim.request_id)
        if (
            request is None
            or lease is None
            or lease.worker_id != claim.worker_id
            or lease.fencing_token != claim.fencing_token
            or lease.lease_until <= now
            or request.state not in {AsyncRequestState.CLAIMED, AsyncRequestState.RUNNING}
        ):
            raise StaleRequestClaimError(f"request claim for {claim.request_id!r} is stale")
        if request.deadline_at is not None and request.deadline_at <= now:
            self._expire_request(request, now)
            raise StaleRequestClaimError(f"request claim for {claim.request_id!r} is stale")
        return request, lease

    def _expire_due(self, now: datetime) -> None:
        for request in tuple(self._requests.values()):
            if (
                request.state not in TERMINAL_REQUEST_STATES
                and request.deadline_at is not None
                and request.deadline_at <= now
            ):
                self._expire_request(request, now)

    def _expire_request(self, request: AsyncRequest, now: datetime) -> None:
        expired = request.model_copy(
            update={
                "state": AsyncRequestState.EXPIRED,
                "updated_at": now,
                "completed_at": now,
            },
            deep=True,
        )
        self._requests[request.id] = expired
        self._leases.pop(request.id, None)
        self._fencing_tokens[request.id] += 1

    def _load(self, request_id: str, *, owner: str | None) -> AsyncRequest:
        request = self._requests.get(request_id)
        if request is None or (owner is not None and request.owner != owner):
            raise KeyError(request_id)
        return request

    def _claim(self, request: AsyncRequest, lease: _Lease) -> RequestClaim:
        return RequestClaim(
            store_id=self._store_id,
            request=self._copy(request),
            worker_id=lease.worker_id,
            fencing_token=lease.fencing_token,
            claimed_at=lease.claimed_at,
            lease_until=lease.lease_until,
        )

    @staticmethod
    def _lease_until(
        request: AsyncRequest,
        now: datetime,
        lease_seconds: float,
    ) -> datetime:
        lease_until = now + timedelta(seconds=lease_seconds)
        if request.deadline_at is not None:
            return min(lease_until, request.deadline_at)
        return lease_until

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return now

    @staticmethod
    def _copy(request: AsyncRequest) -> AsyncRequest:
        return request.model_copy(deep=True)

    @staticmethod
    def _fingerprint(submission: AsyncRequestSubmission) -> str:
        payload = submission.model_dump(mode="json", exclude={"idempotency_key"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _validate_worker(worker_id: str) -> None:
        if not worker_id.strip() or "\x00" in worker_id:
            raise ValueError("worker_id must be a non-empty string without NUL")

    @staticmethod
    def _validate_lease_seconds(value: float) -> float:
        lease_seconds = float(value)
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be finite and greater than zero")
        return lease_seconds
