"""PostgreSQL-backed durable store for asynchronous online requests.

All ownership decisions use PostgreSQL row locks and its clock. Redis or a
process-local signal may wake workers later, but neither is authoritative.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import threading
import uuid
from copy import deepcopy
from datetime import datetime
from types import TracebackType
from typing import Any, Literal, Self

from pydantic import JsonValue

from kairyu.async_requests.models import (
    TERMINAL_REQUEST_STATES,
    AsyncRequest,
    AsyncRequestError,
    AsyncRequestState,
    AsyncRequestStatus,
    AsyncRequestSubmission,
    RequestClaim,
)
from kairyu.async_requests.store import (
    IdempotencyConflictError,
    InvalidRequestTransitionError,
    RequestCapacityError,
    StaleRequestClaimError,
)

try:  # Optional deployment dependency; construction reports a useful error.
    import psycopg
except ModuleNotFoundError:  # pragma: no cover - core-only installation.
    psycopg = None  # type: ignore[assignment]


_SCHEMA_VERSION = 1
_EXPIRY_SWEEP_LIMIT = 100
_CLAIM_STATES = frozenset({AsyncRequestState.CLAIMED, AsyncRequestState.RUNNING})
_REQUEST_COLUMNS = """
    request_id, owner, endpoint, body, priority, idempotency_key, metadata,
    state, attempt, created_at, updated_at, deadline_at, completed_at, result,
    error
"""
_STATUS_COLUMNS = """
    request_id, owner, endpoint, priority, idempotency_key, metadata,
    state, attempt, created_at, updated_at, deadline_at, completed_at, error,
    result IS NOT NULL
"""
logger = logging.getLogger("kairyu.async_requests.postgres")

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS async_request_store_registry (
        store_id TEXT PRIMARY KEY CHECK (store_id <> ''),
        schema_version INTEGER NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS async_requests (
        store_id TEXT NOT NULL
            REFERENCES async_request_store_registry(store_id) ON DELETE CASCADE,
        request_id TEXT NOT NULL,
        owner TEXT NOT NULL CHECK (owner <> ''),
        endpoint TEXT NOT NULL CHECK (endpoint <> ''),
        body JSONB NOT NULL CHECK (jsonb_typeof(body) = 'object'),
        priority BIGINT NOT NULL,
        idempotency_key TEXT,
        intent_fingerprint TEXT NOT NULL,
        metadata JSONB,
        state TEXT NOT NULL CHECK (
            state IN ('queued', 'claimed', 'running', 'succeeded', 'failed',
                      'cancelled', 'expired')
        ),
        attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        deadline_at TIMESTAMPTZ,
        completed_at TIMESTAMPTZ,
        result JSONB,
        error JSONB,
        claim_worker TEXT,
        fencing_token BIGINT NOT NULL DEFAULT 0 CHECK (fencing_token >= 0),
        claimed_at TIMESTAMPTZ,
        lease_until TIMESTAMPTZ,
        PRIMARY KEY (store_id, request_id),
        CHECK (idempotency_key IS NULL OR idempotency_key <> ''),
        CHECK (metadata IS NULL OR jsonb_typeof(metadata) = 'object'),
        CHECK (result IS NULL OR jsonb_typeof(result) = 'object'),
        CHECK (error IS NULL OR jsonb_typeof(error) = 'object'),
        CHECK (
            (
                state IN ('claimed', 'running')
                AND claim_worker IS NOT NULL
                AND claimed_at IS NOT NULL
                AND lease_until IS NOT NULL
                AND lease_until > claimed_at
                AND fencing_token > 0
            )
            OR (
                state NOT IN ('claimed', 'running')
                AND claim_worker IS NULL
                AND claimed_at IS NULL
                AND lease_until IS NULL
            )
        ),
        CHECK (
            (state = 'succeeded' AND result IS NOT NULL AND error IS NULL
             AND completed_at IS NOT NULL)
            OR (state = 'failed' AND result IS NULL AND error IS NOT NULL
                AND completed_at IS NOT NULL)
            OR (state IN ('cancelled', 'expired') AND result IS NULL
                AND error IS NULL AND completed_at IS NOT NULL)
            OR (state IN ('queued', 'claimed', 'running') AND result IS NULL
                AND error IS NULL AND completed_at IS NULL)
        )
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS async_requests_idempotency_idx
        ON async_requests (store_id, owner, idempotency_key)
        WHERE idempotency_key IS NOT NULL
    """,
    """
    CREATE INDEX IF NOT EXISTS async_requests_owner_capacity_idx
        ON async_requests (store_id, owner)
    """,
    """
    CREATE INDEX IF NOT EXISTS async_requests_claimable_v2_idx
        ON async_requests (
            store_id, state, priority, lease_until, created_at, request_id
        )
    """,
    """
    CREATE INDEX IF NOT EXISTS async_requests_deadline_idx
        ON async_requests (store_id, deadline_at, request_id)
        WHERE state IN ('queued', 'claimed', 'running') AND deadline_at IS NOT NULL
    """,
    """
    CREATE TABLE IF NOT EXISTS async_request_owner_deferrals (
        store_id TEXT NOT NULL
            REFERENCES async_request_store_registry(store_id) ON DELETE CASCADE,
        owner TEXT NOT NULL,
        not_before TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (store_id, owner)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS async_request_claim_audit (
        sequence BIGSERIAL PRIMARY KEY,
        store_id TEXT NOT NULL,
        request_id TEXT NOT NULL,
        worker_id TEXT,
        fencing_token BIGINT NOT NULL,
        at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        event TEXT NOT NULL CHECK (
            event IN ('claim', 'reclaim', 'renew', 'defer', 'running', 'succeed',
                      'fail', 'cancel', 'expire')
        ),
        lease_until TIMESTAMPTZ,
        details JSONB NOT NULL DEFAULT '{}'::jsonb,
        FOREIGN KEY (store_id, request_id)
            REFERENCES async_requests(store_id, request_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS async_request_claim_audit_request_idx
        ON async_request_claim_audit (store_id, request_id, sequence)
    """,
)


def _validate_identity(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{name} cannot contain NUL")
    return value


def _validate_lease_seconds(value: float) -> float:
    lease_seconds = float(value)
    if not math.isfinite(lease_seconds) or lease_seconds <= 0:
        raise ValueError("lease_seconds must be finite and greater than zero")
    return lease_seconds


def _json_object(value: Any, *, name: str) -> dict[str, Any]:
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, dict):
        raise RuntimeError(f"corrupt PostgreSQL async request {name}: expected an object")
    return decoded


def _optional_json_object(value: Any, *, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _json_object(value, name=name)


def _submission_fingerprint(submission: AsyncRequestSubmission) -> str:
    payload = submission.model_dump(mode="json", exclude={"idempotency_key"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class PostgresRequestStore:
    """Shared durable RequestStore with DB-clock leases and fenced writes."""

    def __init__(
        self,
        dsn: str,
        *,
        store_id: str,
        connect_timeout_s: float = 10.0,
        eager_connect: bool = True,
        max_records_per_owner: int = 64,
    ) -> None:
        if psycopg is None:
            raise RuntimeError(
                "PostgresRequestStore requires psycopg; install the fleet dependency"
            )
        self._store_id = _validate_identity(store_id, name="store_id")
        timeout = float(connect_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("connect_timeout_s must be finite and greater than zero")
        self._dsn = dsn
        self._connect_timeout = max(1, math.ceil(timeout))
        if max_records_per_owner <= 0:
            raise ValueError("max_records_per_owner must be positive")
        self._max_records_per_owner = max_records_per_owner
        self._lock = threading.RLock()
        self._lease_lock = threading.RLock()
        self._closed = False
        self._connection = None
        self._lease_connection = None
        if eager_connect:
            self._open()

    def _open(self) -> None:
        with self._lock:
            self._require_open(allow_unstarted=True)
            if self._connection is not None:
                return
            self._connection = self._connect()
            try:
                self._initialize_schema()
                self._lease_connection = self._connect()
            except BaseException:
                self._connection.close()
                self._connection = None
                raise

    async def startup(self) -> None:
        """Connect during application startup so failures use lifecycle cleanup."""
        await asyncio.to_thread(self._open)

    @property
    def store_id(self) -> str:
        return self._store_id

    def close(self) -> None:
        with self._lock:
            with self._lease_lock:
                if self._closed:
                    return
                if self._lease_connection is not None:
                    self._lease_connection.close()
                if self._connection is not None:
                    self._connection.close()
                self._closed = True

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False

    def _connect(self):
        assert psycopg is not None
        timeout_ms = self._connect_timeout * 1000
        return psycopg.connect(
            self._dsn,
            autocommit=True,
            connect_timeout=self._connect_timeout,
            options=(
                f"-c statement_timeout={timeout_ms} "
                f"-c lock_timeout={timeout_ms} "
                f"-c idle_in_transaction_session_timeout={timeout_ms}"
            ),
            keepalives=1,
            keepalives_idle=self._connect_timeout,
            keepalives_interval=self._connect_timeout,
            keepalives_count=1,
        )

    def _require_open(self, *, allow_unstarted: bool = False) -> None:
        if self._closed:
            raise RuntimeError("PostgresRequestStore is closed")
        if not allow_unstarted and self._connection is None:
            raise RuntimeError("PostgresRequestStore has not started")

    def _ensure_general_connection(self) -> None:
        assert self._connection is not None
        if self._connection.closed or self._connection.broken:
            self._connection.close()
            self._connection = self._connect()

    def _ensure_lease_connection(self) -> None:
        assert self._lease_connection is not None
        if self._lease_connection.closed or self._lease_connection.broken:
            self._lease_connection.close()
            self._lease_connection = self._connect()

    def _initialize_schema(self) -> None:
        with self._lock:
            self._require_open()
            assert self._connection is not None
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    cursor.execute("SELECT pg_advisory_xact_lock(%s, %s)", (1_261_587_810, 1))
                    for statement in _SCHEMA_STATEMENTS:
                        cursor.execute(statement)
                    cursor.execute(
                        """
                        INSERT INTO async_request_store_registry (store_id, schema_version)
                        VALUES (%s, %s)
                        ON CONFLICT (store_id) DO NOTHING
                        """,
                        (self._store_id, _SCHEMA_VERSION),
                    )
                    cursor.execute(
                        """
                        SELECT schema_version
                        FROM async_request_store_registry
                        WHERE store_id = %s
                        """,
                        (self._store_id,),
                    )
                    row = cursor.fetchone()
                    if row is None or int(row[0]) != _SCHEMA_VERSION:
                        observed = None if row is None else row[0]
                        raise RuntimeError(
                            "incompatible PostgreSQL async request schema for "
                            f"store_id {self._store_id!r}: expected "
                            f"{_SCHEMA_VERSION}, got {observed!r}"
                        )

    def submit(self, submission: AsyncRequestSubmission) -> AsyncRequest:
        submission = submission.model_copy(deep=True)
        fingerprint = _submission_fingerprint(submission)
        idempotency_conflict = False
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (json.dumps([self._store_id, submission.owner]),),
                    )
                    cursor.execute("SELECT clock_timestamp()")
                    now = cursor.fetchone()[0]
                    replay_exists = False
                    if submission.idempotency_key is not None:
                        cursor.execute(
                            """
                            SELECT EXISTS (
                                SELECT 1 FROM async_requests
                                WHERE store_id = %s AND owner = %s
                                  AND idempotency_key = %s
                            )
                            """,
                            (
                                self._store_id,
                                submission.owner,
                                submission.idempotency_key,
                            ),
                        )
                        replay_exists = bool(cursor.fetchone()[0])
                    if not replay_exists:
                        cursor.execute(
                            """
                            SELECT count(*) FROM async_requests
                            WHERE store_id = %s AND owner = %s
                            """,
                            (self._store_id, submission.owner),
                        )
                        if int(cursor.fetchone()[0]) >= self._max_records_per_owner:
                            raise RequestCapacityError(
                                f"owner {submission.owner!r} reached durable request capacity"
                            )
                    expired = submission.deadline_at is not None and submission.deadline_at <= now
                    state = AsyncRequestState.EXPIRED if expired else AsyncRequestState.QUEUED
                    request = AsyncRequest(
                        id=f"req-{uuid.uuid4().hex[:24]}",
                        owner=submission.owner,
                        endpoint=submission.endpoint,
                        body=submission.body,
                        priority=submission.priority,
                        idempotency_key=submission.idempotency_key,
                        metadata=submission.metadata,
                        state=state,
                        created_at=now,
                        updated_at=now,
                        deadline_at=submission.deadline_at,
                        completed_at=now if expired else None,
                    )
                    inserted = self._insert_request(cursor, request, fingerprint)
                    if inserted is not None:
                        if expired:
                            self._audit(
                                cursor,
                                request_id=request.id,
                                worker_id=None,
                                fencing_token=0,
                                event="expire",
                                lease_until=None,
                                details={"reason": "deadline_at_submission"},
                            )
                        return inserted
                    assert submission.idempotency_key is not None
                    cursor.execute(
                        f"""
                        SELECT {_REQUEST_COLUMNS}, intent_fingerprint,
                               claim_worker, fencing_token, claimed_at, lease_until,
                               clock_timestamp()
                        FROM async_requests
                        WHERE store_id = %s AND owner = %s AND idempotency_key = %s
                        FOR UPDATE
                        """,
                        (self._store_id, submission.owner, submission.idempotency_key),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise RuntimeError("idempotency conflict row disappeared")
                    existing = self._request_from_row(row)
                    existing = self._expire_locked(
                        cursor,
                        existing,
                        worker_id=row[16],
                        fencing_token=int(row[17]),
                        claimed_at=row[18],
                        lease_until=row[19],
                        now=row[20],
                    )
                    if str(row[15]) != fingerprint:
                        # Let deadline expirations performed at the start of this
                        # transaction commit before reporting the caller conflict.
                        idempotency_conflict = True
                    else:
                        return existing
        if idempotency_conflict:
            raise IdempotencyConflictError(
                "idempotency key was already used with different request data"
            )
        raise RuntimeError("PostgreSQL async request submission produced no result")

    def _insert_request(
        self,
        cursor: Any,
        request: AsyncRequest,
        fingerprint: str,
    ) -> AsyncRequest | None:
        conflict = """
            ON CONFLICT (store_id, owner, idempotency_key)
            WHERE idempotency_key IS NOT NULL DO NOTHING
        """ if request.idempotency_key is not None else ""
        cursor.execute(
            f"""
            INSERT INTO async_requests (
                store_id, request_id, owner, endpoint, body, priority,
                idempotency_key, intent_fingerprint, metadata, state, attempt,
                created_at, updated_at, deadline_at, completed_at, result, error
            ) VALUES (
                %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s, %s,
                %s, %s, %s, %s, %s::jsonb, %s::jsonb
            )
            {conflict}
            RETURNING {_REQUEST_COLUMNS}
            """,
            (
                self._store_id,
                request.id,
                request.owner,
                request.endpoint,
                json.dumps(request.body),
                request.priority,
                request.idempotency_key,
                fingerprint,
                json.dumps(request.metadata) if request.metadata is not None else None,
                request.state.value,
                request.attempt,
                request.created_at,
                request.updated_at,
                request.deadline_at,
                request.completed_at,
                None,
                None,
            ),
        )
        row = cursor.fetchone()
        return None if row is None else self._request_from_row(row)

    def get(self, request_id: str, *, owner: str | None = None) -> AsyncRequest:
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    return self._load_request_locked(cursor, request_id, owner=owner)

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
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    self._sweep_expired(cursor, limit=_EXPIRY_SWEEP_LIMIT)
                    predicates = ["store_id = %s"]
                    parameters: list[Any] = [self._store_id]
                    if owner is not None:
                        predicates.append("owner = %s")
                        parameters.append(owner)
                    if state is AsyncRequestState.EXPIRED:
                        predicates.append(
                            """
                            (
                                state = 'expired'
                                OR (
                                    state IN ('queued', 'claimed', 'running')
                                    AND deadline_at <= list_clock.now
                                )
                            )
                            """
                        )
                    elif state in _CLAIM_STATES or state is AsyncRequestState.QUEUED:
                        assert state is not None
                        predicates.append("state = %s")
                        predicates.append(
                            "(deadline_at IS NULL OR deadline_at > list_clock.now)"
                        )
                        parameters.append(state.value)
                    elif state is not None:
                        predicates.append("state = %s")
                        parameters.append(state.value)
                    parameters.append(limit)
                    cursor.execute(
                        f"""
                        WITH list_clock AS MATERIALIZED (
                            SELECT statement_timestamp() AS now
                        )
                        SELECT {_REQUEST_COLUMNS}, list_clock.now,
                               state IN ('queued', 'claimed', 'running')
                                   AND deadline_at <= list_clock.now
                        FROM async_requests
                        CROSS JOIN list_clock
                        WHERE {' AND '.join(predicates)}
                        ORDER BY created_at DESC, request_id DESC
                        LIMIT %s
                        """,
                        parameters,
                    )
                    rows = cursor.fetchall()
                    requests = [
                        (
                            self._expired_projection(self._request_from_row(row), row[15])
                            if bool(row[16])
                            else self._request_from_row(row)
                        )
                        for row in rows
                    ]
                    for row in rows:
                        if bool(row[16]):
                            self._try_persist_expiry(cursor, str(row[0]), now=row[15])
                    return requests

    def get_status(
        self, request_id: str, *, owner: str | None = None
    ) -> AsyncRequestStatus:
        """Read one public status without materializing its body or result."""
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    cursor.execute(
                        f"""
                        WITH status_clock AS MATERIALIZED (
                            SELECT statement_timestamp() AS now
                        )
                        SELECT {_STATUS_COLUMNS}, status_clock.now,
                               state IN ('queued', 'claimed', 'running')
                                   AND deadline_at <= status_clock.now
                        FROM async_requests
                        CROSS JOIN status_clock
                        WHERE store_id = %s AND request_id = %s
                          AND (%s::text IS NULL OR owner = %s)
                        """,
                        (self._store_id, request_id, owner, owner),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise KeyError(request_id)
                    logically_expired = bool(row[15])
                    status = self._status_from_row(
                        row,
                        logically_expired=logically_expired,
                        now=row[14],
                    )
                    if logically_expired:
                        self._try_persist_expiry(cursor, str(row[0]), now=row[14])
                    return status

    def list_statuses(
        self,
        *,
        owner: str | None = None,
        state: AsyncRequestState | None = None,
        limit: int = 20,
    ) -> list[AsyncRequestStatus]:
        """List using a narrow SELECT so aggregate payloads stay bounded."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    self._sweep_expired(cursor, limit=_EXPIRY_SWEEP_LIMIT)
                    predicates = ["store_id = %s"]
                    parameters: list[Any] = [self._store_id]
                    if owner is not None:
                        predicates.append("owner = %s")
                        parameters.append(owner)
                    if state is AsyncRequestState.EXPIRED:
                        predicates.append(
                            """
                            (
                                state = 'expired'
                                OR (
                                    state IN ('queued', 'claimed', 'running')
                                    AND deadline_at <= list_clock.now
                                )
                            )
                            """
                        )
                    elif state in _CLAIM_STATES or state is AsyncRequestState.QUEUED:
                        assert state is not None
                        predicates.append("state = %s")
                        predicates.append(
                            "(deadline_at IS NULL OR deadline_at > list_clock.now)"
                        )
                        parameters.append(state.value)
                    elif state is not None:
                        predicates.append("state = %s")
                        parameters.append(state.value)
                    parameters.append(limit)
                    cursor.execute(
                        f"""
                        WITH list_clock AS MATERIALIZED (
                            SELECT statement_timestamp() AS now
                        )
                        SELECT {_STATUS_COLUMNS}, list_clock.now,
                               state IN ('queued', 'claimed', 'running')
                                   AND deadline_at <= list_clock.now
                        FROM async_requests
                        CROSS JOIN list_clock
                        WHERE {' AND '.join(predicates)}
                        ORDER BY created_at DESC, request_id DESC
                        LIMIT %s
                        """,
                        parameters,
                    )
                    rows = cursor.fetchall()
                    statuses = [
                        self._status_from_row(
                            row,
                            logically_expired=bool(row[15]),
                            now=row[14],
                        )
                        for row in rows
                    ]
                    for row in rows:
                        if bool(row[15]):
                            self._try_persist_expiry(cursor, str(row[0]), now=row[14])
                    return statuses

    def get_result(
        self, request_id: str, *, owner: str | None = None
    ) -> tuple[AsyncRequestStatus, dict[str, JsonValue] | None]:
        """Read status plus result only for a successful terminal row."""
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    cursor.execute(
                        f"""
                        WITH result_clock AS MATERIALIZED (
                            SELECT statement_timestamp() AS now
                        )
                        SELECT {_STATUS_COLUMNS},
                               CASE WHEN state = 'succeeded' THEN result END,
                               result_clock.now,
                               state IN ('queued', 'claimed', 'running')
                                   AND deadline_at <= result_clock.now
                        FROM async_requests
                        CROSS JOIN result_clock
                        WHERE store_id = %s AND request_id = %s
                          AND (%s::text IS NULL OR owner = %s)
                        """,
                        (self._store_id, request_id, owner, owner),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise KeyError(request_id)
                    logically_expired = bool(row[16])
                    status = self._status_from_row(
                        row,
                        logically_expired=logically_expired,
                        now=row[15],
                    )
                    if logically_expired:
                        self._try_persist_expiry(cursor, str(row[0]), now=row[15])
                    result = (
                        None
                        if logically_expired
                        else _optional_json_object(row[14], name="result")
                    )
                    return status, result

    def claim_next(self, worker_id: str, *, lease_seconds: float) -> RequestClaim | None:
        worker_id = _validate_identity(worker_id, name="worker_id")
        lease_seconds = _validate_lease_seconds(lease_seconds)
        with self._lease_lock:
            self._require_open()
            self._ensure_lease_connection()
            with self._lease_connection.transaction():
                with self._lease_connection.cursor() as cursor:
                    self._sweep_expired(cursor, limit=_EXPIRY_SWEEP_LIMIT)
                    while True:
                        cursor.execute(
                            """
                            SELECT request_id, state, claim_worker, fencing_token,
                                   claimed_at, lease_until
                            FROM async_requests
                            WHERE store_id = %s
                              AND (deadline_at IS NULL OR deadline_at > clock_timestamp())
                              AND NOT EXISTS (
                                  SELECT 1 FROM async_request_owner_deferrals deferral
                                  WHERE deferral.store_id = async_requests.store_id
                                    AND deferral.owner = async_requests.owner
                                    AND deferral.not_before > clock_timestamp()
                              )
                              AND (
                                  state = 'queued'
                                  OR (
                                      state IN ('claimed', 'running')
                                      AND lease_until <= clock_timestamp()
                                  )
                              )
                            ORDER BY priority, created_at, request_id
                            FOR UPDATE SKIP LOCKED
                            LIMIT 1
                            """,
                            (self._store_id,),
                        )
                        previous = cursor.fetchone()
                        if previous is None:
                            return None
                        request_id = str(previous[0])
                        previous_state = AsyncRequestState(str(previous[1]))
                        previous_worker = previous[2]
                        previous_token = int(previous[3])
                        previous_claimed_at = previous[4]
                        previous_lease = previous[5]
                        cursor.execute("SELECT clock_timestamp()")
                        claimed_at = cursor.fetchone()[0]
                        cursor.execute(
                            f"""
                            UPDATE async_requests
                            SET state = 'claimed', attempt = attempt + 1,
                                updated_at = %s, claim_worker = %s,
                                fencing_token = fencing_token + 1,
                                claimed_at = %s,
                                lease_until = LEAST(
                                    %s + (%s * interval '1 second'),
                                    COALESCE(deadline_at, 'infinity'::timestamptz)
                                )
                            WHERE store_id = %s AND request_id = %s
                              AND (deadline_at IS NULL OR deadline_at > clock_timestamp())
                              AND NOT EXISTS (
                                  SELECT 1 FROM async_request_owner_deferrals deferral
                                  WHERE deferral.store_id = async_requests.store_id
                                    AND deferral.owner = async_requests.owner
                                    AND deferral.not_before > clock_timestamp()
                              )
                              AND (
                                  state = 'queued'
                                  OR (
                                      state IN ('claimed', 'running')
                                      AND lease_until <= clock_timestamp()
                                  )
                              )
                            RETURNING {_REQUEST_COLUMNS}, fencing_token,
                                      claimed_at, lease_until
                            """,
                            (
                                claimed_at,
                                worker_id,
                                claimed_at,
                                claimed_at,
                                lease_seconds,
                                self._store_id,
                                request_id,
                            ),
                        )
                        row = cursor.fetchone()
                        if row is not None:
                            break
                        self._expire_request_by_id(cursor, request_id)
                    request = self._request_from_row(row)
                    token = int(row[15])
                    lease_until = row[17]
                    event: Literal["claim", "reclaim"] = (
                        "claim" if previous_state is AsyncRequestState.QUEUED else "reclaim"
                    )
                    self._audit(
                        cursor,
                        request_id=request.id,
                        worker_id=worker_id,
                        fencing_token=token,
                        event=event,
                        at=claimed_at,
                        lease_until=lease_until,
                        details={
                            "previous_state": previous_state.value,
                            "previous_worker_id": previous_worker,
                            "previous_fencing_token": previous_token,
                            "previous_claimed_at": (
                                previous_claimed_at.isoformat()
                                if previous_claimed_at is not None
                                else None
                            ),
                            "previous_lease_until": (
                                previous_lease.isoformat() if previous_lease is not None else None
                            ),
                        },
                    )
                    return RequestClaim(
                        store_id=self._store_id,
                        request=request,
                        worker_id=worker_id,
                        fencing_token=token,
                        claimed_at=claimed_at,
                        lease_until=lease_until,
                    )

    def renew_claim(self, claim: RequestClaim, *, lease_seconds: float) -> RequestClaim:
        self._validate_claim_identity(claim)
        lease_seconds = _validate_lease_seconds(lease_seconds)
        stale_after_commit = False
        with self._lease_lock:
            self._require_open()
            self._ensure_lease_connection()
            with self._lease_connection.transaction():
                with self._lease_connection.cursor() as cursor:
                    loaded = self._load_claim(cursor, claim)
                    if loaded is None:
                        stale_after_commit = True
                    else:
                        request, claimed_at, _lease_until, _now = loaded
                        cursor.execute(
                            f"""
                            UPDATE async_requests
                            SET lease_until = LEAST(
                                    clock_timestamp() + (%s * interval '1 second'),
                                    COALESCE(deadline_at, 'infinity'::timestamptz)
                                ),
                                updated_at = clock_timestamp()
                            WHERE store_id = %s AND request_id = %s
                              AND state IN ('claimed', 'running')
                              AND claim_worker = %s AND fencing_token = %s
                              AND lease_until > clock_timestamp()
                              AND (deadline_at IS NULL OR deadline_at > clock_timestamp())
                            RETURNING {_REQUEST_COLUMNS}, lease_until, updated_at
                            """,
                            (
                                lease_seconds,
                                self._store_id,
                                claim.request_id,
                                claim.worker_id,
                                claim.fencing_token,
                            ),
                        )
                        row = cursor.fetchone()
                        if row is None:
                            self._expire_request_by_id(cursor, claim.request_id)
                            stale_after_commit = True
                        else:
                            renewed_request = self._request_from_row(row)
                            renewed_until = row[15]
                            renewed_at = row[16]
                            self._audit(
                                cursor,
                                request_id=request.id,
                                worker_id=claim.worker_id,
                                fencing_token=claim.fencing_token,
                                event="renew",
                                at=renewed_at,
                                lease_until=renewed_until,
                            )
                            return RequestClaim(
                                store_id=self._store_id,
                                request=renewed_request,
                                worker_id=claim.worker_id,
                                fencing_token=claim.fencing_token,
                                claimed_at=claimed_at,
                                lease_until=renewed_until,
                            )
        if stale_after_commit:
            raise StaleRequestClaimError(f"request claim for {claim.request_id!r} is stale")
        raise RuntimeError("PostgreSQL claim renewal produced no result")

    def mark_running(self, claim: RequestClaim) -> AsyncRequest:
        self._validate_claim_identity(claim)
        stale_after_commit = False
        with self._lease_lock:
            self._require_open()
            self._ensure_lease_connection()
            with self._lease_connection.transaction():
                with self._lease_connection.cursor() as cursor:
                    loaded = self._load_claim(cursor, claim)
                    if loaded is None:
                        stale_after_commit = True
                        request = None
                    else:
                        request, _claimed_at, lease_until, _now = loaded
                    if request is None:
                        pass
                    elif request.state is not AsyncRequestState.CLAIMED:
                        raise InvalidRequestTransitionError(
                            f"cannot mark {request.state.value} request as running"
                        )
                    else:
                        cursor.execute(
                            f"""
                            UPDATE async_requests
                            SET state = 'running', updated_at = clock_timestamp()
                            WHERE store_id = %s AND request_id = %s
                              AND state = 'claimed'
                              AND claim_worker = %s AND fencing_token = %s
                              AND lease_until > clock_timestamp()
                              AND (deadline_at IS NULL OR deadline_at > clock_timestamp())
                            RETURNING {_REQUEST_COLUMNS}
                            """,
                            (
                                self._store_id,
                                request.id,
                                claim.worker_id,
                                claim.fencing_token,
                            ),
                        )
                        row = cursor.fetchone()
                        if row is None:
                            self._expire_request_by_id(cursor, claim.request_id)
                            stale_after_commit = True
                        else:
                            running = self._request_from_row(row)
                            self._audit(
                                cursor,
                                request_id=request.id,
                                worker_id=claim.worker_id,
                                fencing_token=claim.fencing_token,
                                event="running",
                                at=running.updated_at,
                                lease_until=lease_until,
                            )
                            return running
        if stale_after_commit:
            raise StaleRequestClaimError(f"request claim for {claim.request_id!r} is stale")
        raise RuntimeError("PostgreSQL running transition produced no result")

    def defer(self, claim: RequestClaim, *, delay_seconds: float) -> AsyncRequest:
        """Release a fence and cool down its tenant without occupying a consumer."""
        self._validate_claim_identity(claim)
        delay_seconds = _validate_lease_seconds(delay_seconds)
        stale_after_commit = False
        with self._lease_lock:
            self._require_open()
            self._ensure_lease_connection()
            with self._lease_connection.transaction():
                with self._lease_connection.cursor() as cursor:
                    loaded = self._load_claim(cursor, claim)
                    if loaded is None:
                        stale_after_commit = True
                    else:
                        request, claimed_at, lease_until, now = loaded
                        cursor.execute(
                            f"""
                            UPDATE async_requests
                            SET state = 'queued', updated_at = %s,
                                claim_worker = NULL,
                                fencing_token = fencing_token + 1,
                                claimed_at = NULL, lease_until = NULL
                            WHERE store_id = %s AND request_id = %s
                              AND state IN ('claimed', 'running')
                              AND claim_worker = %s AND fencing_token = %s
                              AND lease_until > %s
                            RETURNING {_REQUEST_COLUMNS}, fencing_token
                            """,
                            (
                                now,
                                self._store_id,
                                claim.request_id,
                                claim.worker_id,
                                claim.fencing_token,
                                now,
                            ),
                        )
                        row = cursor.fetchone()
                        if row is None:
                            stale_after_commit = True
                        else:
                            deferred = self._request_from_row(row)
                            next_token = int(row[15])
                            cursor.execute(
                                """
                                INSERT INTO async_request_owner_deferrals (
                                    store_id, owner, not_before
                                ) VALUES (
                                    %s, %s, %s + (%s * interval '1 second')
                                )
                                ON CONFLICT (store_id, owner) DO UPDATE
                                SET not_before = GREATEST(
                                    async_request_owner_deferrals.not_before,
                                    EXCLUDED.not_before
                                )
                                """,
                                (self._store_id, request.owner, now, delay_seconds),
                            )
                            self._audit(
                                cursor,
                                request_id=request.id,
                                worker_id=claim.worker_id,
                                fencing_token=next_token,
                                event="defer",
                                at=now,
                                lease_until=None,
                                details={
                                    "previous_claimed_at": claimed_at.isoformat(),
                                    "previous_lease_until": lease_until.isoformat(),
                                },
                            )
                            return deferred
        if stale_after_commit:
            raise StaleRequestClaimError(f"request claim for {claim.request_id!r} is stale")
        raise RuntimeError("PostgreSQL defer transition produced no result")

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

    def _finish(
        self,
        claim: RequestClaim,
        *,
        state: Literal[AsyncRequestState.SUCCEEDED, AsyncRequestState.FAILED],
        result: dict[str, JsonValue] | None,
        error: AsyncRequestError | None,
    ) -> AsyncRequest:
        self._validate_claim_identity(claim)
        try:
            return self._finish_once(
                claim,
                state=state,
                result=result,
                error=error,
            )
        except (StaleRequestClaimError, InvalidRequestTransitionError, ValueError):
            raise
        except Exception:
            try:
                recovered = self._recover_committed_finalization(
                    claim,
                    state=state,
                    result=result,
                    error=error,
                )
            except Exception:
                logger.warning(
                    "could not resolve PostgreSQL async request terminal commit outcome",
                    extra={
                        "request_id": claim.request_id,
                        "worker_id": claim.worker_id,
                        "fencing_token": claim.fencing_token,
                    },
                    exc_info=True,
                )
            else:
                if recovered is not None:
                    logger.warning(
                        "recovered acknowledged PostgreSQL async request terminal commit",
                        extra={
                            "request_id": claim.request_id,
                            "worker_id": claim.worker_id,
                            "fencing_token": claim.fencing_token,
                        },
                    )
                    return recovered
            raise

    def _finish_once(
        self,
        claim: RequestClaim,
        *,
        state: Literal[AsyncRequestState.SUCCEEDED, AsyncRequestState.FAILED],
        result: dict[str, JsonValue] | None,
        error: AsyncRequestError | None,
    ) -> AsyncRequest:
        stale_after_commit = False
        with self._lease_lock:
            self._require_open()
            self._ensure_lease_connection()
            with self._lease_connection.transaction():
                with self._lease_connection.cursor() as cursor:
                    loaded = self._load_claim(cursor, claim)
                    if loaded is None:
                        stale_after_commit = True
                        request = None
                    else:
                        request, _claimed_at, _lease_until, now = loaded
                    if request is None:
                        pass
                    elif request.state is not AsyncRequestState.RUNNING:
                        raise InvalidRequestTransitionError(
                            f"cannot finish {request.state.value} request; mark it running first"
                        )
                    else:
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
                        cursor.execute(
                            f"""
                            UPDATE async_requests
                            SET state = %s, updated_at = clock_timestamp(),
                                completed_at = clock_timestamp(),
                                result = %s::jsonb, error = %s::jsonb,
                                claim_worker = NULL, claimed_at = NULL, lease_until = NULL
                            WHERE store_id = %s AND request_id = %s
                              AND state = 'running'
                              AND claim_worker = %s AND fencing_token = %s
                              AND lease_until > clock_timestamp()
                              AND (deadline_at IS NULL OR deadline_at > clock_timestamp())
                            RETURNING {_REQUEST_COLUMNS}
                            """,
                            (
                                state.value,
                                (
                                    json.dumps(finished.result)
                                    if finished.result is not None
                                    else None
                                ),
                                (
                                    json.dumps(finished.error.model_dump(mode="json"))
                                    if finished.error is not None
                                    else None
                                ),
                                self._store_id,
                                request.id,
                                claim.worker_id,
                                claim.fencing_token,
                            ),
                        )
                        row = cursor.fetchone()
                        if row is None:
                            self._expire_request_by_id(cursor, claim.request_id)
                            stale_after_commit = True
                        else:
                            stored = self._request_from_row(row)
                            event: Literal["succeed", "fail"] = (
                                "succeed" if state is AsyncRequestState.SUCCEEDED else "fail"
                            )
                            self._audit(
                                cursor,
                                request_id=request.id,
                                worker_id=claim.worker_id,
                                fencing_token=claim.fencing_token,
                                event=event,
                                at=stored.updated_at,
                                lease_until=None,
                            )
                            return stored
        if stale_after_commit:
            raise StaleRequestClaimError(f"request claim for {claim.request_id!r} is stale")
        raise RuntimeError("PostgreSQL terminal publication produced no result")

    def _recover_committed_finalization(
        self,
        claim: RequestClaim,
        *,
        state: Literal[AsyncRequestState.SUCCEEDED, AsyncRequestState.FAILED],
        result: dict[str, JsonValue] | None,
        error: AsyncRequestError | None,
    ) -> AsyncRequest | None:
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    cursor.execute(
                        f"""
                        SELECT {_REQUEST_COLUMNS}, claim_worker, fencing_token,
                               claimed_at, lease_until
                        FROM async_requests
                        WHERE store_id = %s AND request_id = %s
                        """,
                        (self._store_id, claim.request_id),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        return None
                    observed = self._request_from_row(row)
                    expected_error = error.model_dump(mode="json") if error is not None else None
                    if (
                        observed.state is not state
                        or observed.result != result
                        or (
                            observed.error.model_dump(mode="json")
                            if observed.error is not None
                            else None
                        )
                        != expected_error
                        or row[15] is not None
                        or int(row[16]) != claim.fencing_token
                        or row[17] is not None
                        or row[18] is not None
                    ):
                        return None
                    event = "succeed" if state is AsyncRequestState.SUCCEEDED else "fail"
                    cursor.execute(
                        """
                        SELECT count(*)
                        FROM async_request_claim_audit
                        WHERE store_id = %s AND request_id = %s
                          AND worker_id = %s AND fencing_token = %s AND event = %s
                        """,
                        (
                            self._store_id,
                            claim.request_id,
                            claim.worker_id,
                            claim.fencing_token,
                            event,
                        ),
                    )
                    if int(cursor.fetchone()[0]) != 1:
                        return None
                    return observed

    def cancel(self, request_id: str, *, owner: str | None = None) -> AsyncRequest:
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    cursor.execute(
                        f"""
                        SELECT {_REQUEST_COLUMNS}, claim_worker, fencing_token,
                               claimed_at, lease_until, clock_timestamp()
                        FROM async_requests
                        WHERE store_id = %s AND request_id = %s
                          AND (%s::text IS NULL OR owner = %s)
                        FOR UPDATE
                        """,
                        (self._store_id, request_id, owner, owner),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise KeyError(request_id)
                    request = self._request_from_row(row)
                    request = self._expire_locked(
                        cursor,
                        request,
                        worker_id=row[15],
                        fencing_token=int(row[16]),
                        claimed_at=row[17],
                        lease_until=row[18],
                        now=row[19],
                    )
                    if request.state in TERMINAL_REQUEST_STATES:
                        return request
                    previous_worker = row[15]
                    previous_token = int(row[16])
                    previous_claimed_at = row[17]
                    previous_lease = row[18]
                    now = row[19]
                    next_token = previous_token + 1
                    cursor.execute(
                        f"""
                        UPDATE async_requests
                        SET state = 'cancelled', updated_at = %s, completed_at = %s,
                            claim_worker = NULL, fencing_token = %s,
                            claimed_at = NULL, lease_until = NULL
                        WHERE store_id = %s AND request_id = %s
                        RETURNING {_REQUEST_COLUMNS}
                        """,
                        (now, now, next_token, self._store_id, request_id),
                    )
                    cancelled = self._request_from_row(cursor.fetchone())
                    self._audit(
                        cursor,
                        request_id=request_id,
                        worker_id=previous_worker,
                        fencing_token=next_token,
                        event="cancel",
                        at=now,
                        lease_until=None,
                        details={
                            "previous_state": request.state.value,
                            "previous_fencing_token": previous_token,
                            "previous_claimed_at": (
                                previous_claimed_at.isoformat()
                                if previous_claimed_at is not None
                                else None
                            ),
                            "previous_lease_until": (
                                previous_lease.isoformat() if previous_lease is not None else None
                            ),
                        },
                    )
                    return cancelled

    def _load_request_locked(
        self,
        cursor: Any,
        request_id: str,
        *,
        owner: str | None,
    ) -> AsyncRequest:
        cursor.execute(
            f"""
            SELECT {_REQUEST_COLUMNS}, claim_worker, fencing_token,
                   claimed_at, lease_until, clock_timestamp()
            FROM async_requests
            WHERE store_id = %s AND request_id = %s
              AND (%s::text IS NULL OR owner = %s)
            FOR UPDATE
            """,
            (self._store_id, request_id, owner, owner),
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(request_id)
        return self._expire_locked(
            cursor,
            self._request_from_row(row),
            worker_id=row[15],
            fencing_token=int(row[16]),
            claimed_at=row[17],
            lease_until=row[18],
            now=row[19],
        )

    def _load_claim(
        self,
        cursor: Any,
        claim: RequestClaim,
    ) -> tuple[AsyncRequest, datetime, datetime, datetime] | None:
        cursor.execute(
            f"""
            SELECT {_REQUEST_COLUMNS}, claim_worker, fencing_token,
                   claimed_at, lease_until, clock_timestamp()
            FROM async_requests
            WHERE store_id = %s AND request_id = %s
            FOR UPDATE
            """,
            (self._store_id, claim.request_id),
        )
        row = cursor.fetchone()
        if row is None:
            raise StaleRequestClaimError(f"request {claim.request_id!r} no longer exists")
        request = self._request_from_row(row)
        worker_id = row[15]
        fencing_token = int(row[16])
        claimed_at = row[17]
        lease_until = row[18]
        now = row[19]
        expired = self._expire_locked(
            cursor,
            request,
            worker_id=worker_id,
            fencing_token=fencing_token,
            claimed_at=claimed_at,
            lease_until=lease_until,
            now=now,
        )
        if expired.state is AsyncRequestState.EXPIRED:
            return None
        if (
            request.state not in _CLAIM_STATES
            or worker_id != claim.worker_id
            or fencing_token != claim.fencing_token
            or claimed_at is None
            or lease_until is None
            or lease_until <= now
        ):
            raise StaleRequestClaimError(f"request claim for {claim.request_id!r} is stale")
        return request, claimed_at, lease_until, now

    def _sweep_expired(self, cursor: Any, *, limit: int) -> None:
        cursor.execute(
            """
            SELECT request_id, state, claim_worker, fencing_token,
                   claimed_at, lease_until, clock_timestamp()
            FROM async_requests
            WHERE store_id = %s
              AND state IN ('queued', 'claimed', 'running')
              AND deadline_at <= clock_timestamp()
            ORDER BY deadline_at, request_id
            FOR UPDATE SKIP LOCKED
            LIMIT %s
            """,
            (self._store_id, limit),
        )
        for row in cursor.fetchall():
            self._expire_request_by_id(cursor, str(row[0]), locked_row=row)

    def _expire_request_by_id(
        self,
        cursor: Any,
        request_id: str,
        *,
        locked_row: Any | None = None,
    ) -> AsyncRequest:
        if locked_row is None:
            cursor.execute(
                f"""
                SELECT {_REQUEST_COLUMNS}, claim_worker, fencing_token,
                       claimed_at, lease_until, clock_timestamp()
                FROM async_requests
                WHERE store_id = %s AND request_id = %s
                FOR UPDATE
                """,
                (self._store_id, request_id),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(request_id)
            return self._expire_locked(
                cursor,
                self._request_from_row(row),
                worker_id=row[15],
                fencing_token=int(row[16]),
                claimed_at=row[17],
                lease_until=row[18],
                now=row[19],
            )

        previous_state = AsyncRequestState(str(locked_row[1]))
        previous_worker = locked_row[2]
        previous_token = int(locked_row[3])
        previous_claimed_at = locked_row[4]
        previous_lease = locked_row[5]
        now = locked_row[6]
        return self._commit_expiry(
            cursor,
            request_id=request_id,
            previous_state=previous_state,
            worker_id=previous_worker,
            fencing_token=previous_token,
            claimed_at=previous_claimed_at,
            lease_until=previous_lease,
            now=now,
        )

    def _try_persist_expiry(
        self,
        cursor: Any,
        request_id: str,
        *,
        now: datetime,
    ) -> None:
        """Persist a list projection when available without delaying the read."""
        cursor.execute(
            """
            SELECT request_id, state, claim_worker, fencing_token,
                   claimed_at, lease_until
            FROM async_requests
            WHERE store_id = %s AND request_id = %s
              AND state IN ('queued', 'claimed', 'running')
              AND deadline_at <= %s
            FOR UPDATE SKIP LOCKED
            """,
            (self._store_id, request_id, now),
        )
        row = cursor.fetchone()
        if row is None:
            return
        self._commit_expiry(
            cursor,
            request_id=request_id,
            previous_state=AsyncRequestState(str(row[1])),
            worker_id=row[2],
            fencing_token=int(row[3]),
            claimed_at=row[4],
            lease_until=row[5],
            now=now,
        )

    def _expired_projection(
        self,
        request: AsyncRequest,
        now: datetime,
    ) -> AsyncRequest:
        return AsyncRequest.model_validate(
            {
                **request.model_dump(),
                "state": AsyncRequestState.EXPIRED,
                "updated_at": now,
                "completed_at": now,
                "result": None,
                "error": None,
            }
        )

    def _expire_locked(
        self,
        cursor: Any,
        request: AsyncRequest,
        *,
        worker_id: str | None,
        fencing_token: int,
        claimed_at: datetime | None,
        lease_until: datetime | None,
        now: datetime,
    ) -> AsyncRequest:
        if (
            request.state in TERMINAL_REQUEST_STATES
            or request.deadline_at is None
            or request.deadline_at > now
        ):
            return request
        return self._commit_expiry(
            cursor,
            request_id=request.id,
            previous_state=request.state,
            worker_id=worker_id,
            fencing_token=fencing_token,
            claimed_at=claimed_at,
            lease_until=lease_until,
            now=now,
        )

    def _commit_expiry(
        self,
        cursor: Any,
        *,
        request_id: str,
        previous_state: AsyncRequestState,
        worker_id: str | None,
        fencing_token: int,
        claimed_at: datetime | None,
        lease_until: datetime | None,
        now: datetime,
    ) -> AsyncRequest:
        next_token = fencing_token + 1
        cursor.execute(
            f"""
            UPDATE async_requests
            SET state = 'expired', updated_at = %s, completed_at = %s,
                claim_worker = NULL, fencing_token = %s,
                claimed_at = NULL, lease_until = NULL
            WHERE store_id = %s AND request_id = %s
              AND state IN ('queued', 'claimed', 'running')
              AND deadline_at <= %s
            RETURNING {_REQUEST_COLUMNS}
            """,
            (now, now, next_token, self._store_id, request_id, now),
        )
        row = cursor.fetchone()
        if row is None:
            return self._load_request_locked(cursor, request_id, owner=None)
        expired = self._request_from_row(row)
        self._audit(
            cursor,
            request_id=request_id,
            worker_id=worker_id,
            fencing_token=next_token,
            event="expire",
            at=now,
            lease_until=None,
            details={
                "previous_state": previous_state.value,
                "previous_fencing_token": fencing_token,
                "previous_claimed_at": claimed_at.isoformat() if claimed_at is not None else None,
                "previous_lease_until": (
                    lease_until.isoformat() if lease_until is not None else None
                ),
            },
        )
        return expired

    def _request_from_row(self, row: Any) -> AsyncRequest:
        return AsyncRequest(
            id=str(row[0]),
            owner=str(row[1]),
            endpoint=str(row[2]),
            body=_json_object(row[3], name="body"),
            priority=int(row[4]),
            idempotency_key=row[5],
            metadata=_optional_json_object(row[6], name="metadata"),
            state=AsyncRequestState(str(row[7])),
            attempt=int(row[8]),
            created_at=row[9],
            updated_at=row[10],
            deadline_at=row[11],
            completed_at=row[12],
            result=_optional_json_object(row[13], name="result"),
            error=_optional_json_object(row[14], name="error"),
        )

    def _status_from_row(
        self,
        row: Any,
        *,
        logically_expired: bool = False,
        now: datetime | None = None,
    ) -> AsyncRequestStatus:
        state = AsyncRequestState.EXPIRED if logically_expired else AsyncRequestState(str(row[6]))
        return AsyncRequestStatus(
            id=str(row[0]),
            owner=str(row[1]),
            endpoint=str(row[2]),
            priority=int(row[3]),
            idempotency_key=row[4],
            metadata=_optional_json_object(row[5], name="metadata"),
            state=state,
            attempt=int(row[7]),
            created_at=row[8],
            updated_at=now if logically_expired else row[9],
            deadline_at=row[10],
            completed_at=now if logically_expired else row[11],
            error=(
                None
                if logically_expired
                else _optional_json_object(row[12], name="error")
            ),
            has_result=False if logically_expired else bool(row[13]),
        )

    def _validate_claim_identity(self, claim: RequestClaim) -> None:
        if claim.store_id != self._store_id:
            raise ValueError(
                f"claim belongs to store_id {claim.store_id!r}, not {self._store_id!r}"
            )
        _validate_identity(claim.worker_id, name="claim.worker_id")
        if claim.fencing_token <= 0:
            raise ValueError("claim.fencing_token must be positive")

    def _audit(
        self,
        cursor: Any,
        *,
        request_id: str,
        worker_id: str | None,
        fencing_token: int,
        event: Literal[
            "claim", "reclaim", "renew", "defer", "running", "succeed", "fail", "cancel", "expire"
        ],
        lease_until: datetime | None,
        at: datetime | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        cursor.execute(
            """
            INSERT INTO async_request_claim_audit (
                store_id, request_id, worker_id, fencing_token, at, event,
                lease_until, details
            ) VALUES (
                %s, %s, %s, %s, COALESCE(%s, clock_timestamp()), %s, %s, %s::jsonb
            )
            """,
            (
                self._store_id,
                request_id,
                worker_id,
                fencing_token,
                at,
                event,
                lease_until,
                json.dumps(details or {}),
            ),
        )

    def export_claim_audit(
        self,
        request_id: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    if request_id is None:
                        cursor.execute(
                            """
                            SELECT sequence, store_id, request_id, worker_id,
                                   fencing_token, at, event, lease_until, details
                            FROM async_request_claim_audit
                            WHERE store_id = %s
                            ORDER BY sequence
                            """,
                            (self._store_id,),
                        )
                    else:
                        cursor.execute(
                            """
                            SELECT sequence, store_id, request_id, worker_id,
                                   fencing_token, at, event, lease_until, details
                            FROM async_request_claim_audit
                            WHERE store_id = %s AND request_id = %s
                            ORDER BY sequence
                            """,
                            (self._store_id, request_id),
                        )
                    return tuple(
                        {
                            "sequence": int(row[0]),
                            "store_id": str(row[1]),
                            "request_id": str(row[2]),
                            "worker_id": row[3],
                            "fencing_token": int(row[4]),
                            "at": row[5].isoformat(),
                            "event": str(row[6]),
                            "lease_until": row[7].isoformat() if row[7] is not None else None,
                            "details": _json_object(row[8], name="audit details"),
                        }
                        for row in cursor
                    )
