"""Authenticated node-local HTTP transport for fenced model pre-staging."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime
from typing import Annotated, Literal

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from kairyu.artifacts.cache_index import NodeModelCacheIndexError
from kairyu.artifacts.manifest import (
    InvalidModelArtifactError,
    ModelArtifactAdmissionRequest,
    ModelArtifactTrustStore,
    SignedModelArtifactManifest,
)
from kairyu.artifacts.node_cache import NodeModelCacheError
from kairyu.entrypoints.server.middleware import AuthMiddleware, ConcurrencyLimitMiddleware
from kairyu.runners.cache_agent_live_evidence import (
    NodeModelCacheLiveEvidenceRequest,
    NodeModelCacheLiveEvidenceResponse,
    NodeModelCacheLiveEvidenceSource,
)
from kairyu.runners.prestage import (
    NodeModelPrestageCapacityError,
    NodeModelPrestageCommand,
    NodeModelPrestageConflictError,
    NodeModelPrestageExecutor,
    NodeModelPrestageExpiredError,
    NodeModelPrestageRecord,
    NodeModelPrestageStore,
    _digest,
    _text,
)
from kairyu.runners.prewarm import ModelCachePlacementState

try:
    import psycopg as _psycopg
except ModuleNotFoundError:  # pragma: no cover - exercised by core-only packaging
    _psycopg = None

_ASGIApp = Callable[..., Awaitable[None]]
_ENSURE_PATH = "/v1/prestage/ensure"
_RELEASE_PATH = "/v1/prestage/release"
_LIVE_EVIDENCE_PATH = "/v1/cache/live-evidence"


class NodeModelPrestageEnsureRequest(BaseModel):
    """Authenticated wire request for one exact ensure command."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-prestage-ensure-request-v1"] = (
        "kairyu-node-model-prestage-ensure-request-v1"
    )
    command: NodeModelPrestageCommand
    claim_id: str = Field(min_length=64, max_length=64)
    manifest: SignedModelArtifactManifest
    admission_request: ModelArtifactAdmissionRequest

    @field_validator("claim_id")
    @classmethod
    def validate_claim_id(cls, value: str) -> str:
        return _digest(value, name="claim_id")


class NodeModelPrestageReleaseRequest(BaseModel):
    """Authenticated wire request for one exact release command."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-prestage-release-request-v1"] = (
        "kairyu-node-model-prestage-release-request-v1"
    )
    command: NodeModelPrestageCommand


class NodeModelPrestageStatus(BaseModel):
    """Path- and failure-detail-free controller view of durable progress."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-prestage-status-v1"] = (
        "kairyu-node-model-prestage-status-v1"
    )
    command: NodeModelPrestageCommand
    state: ModelCachePlacementState
    attempt: int = Field(ge=0)
    updated_at: datetime


def _public_status(record: NodeModelPrestageRecord) -> NodeModelPrestageStatus:
    return NodeModelPrestageStatus(
        command=record.command,
        state=record.state,
        attempt=record.attempt,
        updated_at=record.updated_at,
    )


class NodeModelPrestageRecordsResponse(BaseModel):
    """Bounded durable state returned to the trusted controller."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["kairyu-node-model-prestage-records-response-v1"] = (
        "kairyu-node-model-prestage-records-response-v1"
    )
    node_id: str
    records: tuple[NodeModelPrestageStatus, ...]
    next_cursor: str | None = None


class NodeModelCacheAgentHealth(BaseModel):
    """Low-disclosure liveness/readiness response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["ok", "ready", "not_ready"]
    node_id: str


class _InvalidCacheAgentRequest(ValueError):
    def __init__(self, *, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("JSON object keys must be unique")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON constant {value!r} is not permitted")


def _parse_wire_model(body: bytes, model: type[BaseModel]) -> BaseModel:
    try:
        value = json.loads(
            body,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise _InvalidCacheAgentRequest(
            status_code=400, message="request body must be strict JSON"
        ) from exc
    try:
        return model.model_validate(value)
    except ValidationError as exc:
        raise _InvalidCacheAgentRequest(
            status_code=422, message="request body does not match the command schema"
        ) from exc


def _parse_ensure_request(body: bytes) -> NodeModelPrestageEnsureRequest:
    value = _parse_wire_model(body, NodeModelPrestageEnsureRequest)
    assert isinstance(value, NodeModelPrestageEnsureRequest)
    return value


def _parse_release_request(body: bytes) -> NodeModelPrestageReleaseRequest:
    value = _parse_wire_model(body, NodeModelPrestageReleaseRequest)
    assert isinstance(value, NodeModelPrestageReleaseRequest)
    return value


def _parse_live_evidence_request(body: bytes) -> NodeModelCacheLiveEvidenceRequest:
    value = _parse_wire_model(body, NodeModelCacheLiveEvidenceRequest)
    assert isinstance(value, NodeModelCacheLiveEvidenceRequest)
    return value


class _RequestBodyLimitMiddleware:
    """Reject oversized cache-agent commands before FastAPI materializes JSON."""

    def __init__(self, app: _ASGIApp, *, limit: int, paths: Iterable[str]) -> None:
        self.app = app
        self._limit = limit
        self._paths = frozenset(paths)

    @staticmethod
    async def _send_rejection(send: Callable) -> None:
        body = b'{"detail":"request body exceeds the configured limit"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": (
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ),
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in self._paths
        ):
            await self.app(scope, receive, send)
            return
        raw_length = dict(scope.get("headers") or ()).get(b"content-length")
        if raw_length is not None:
            try:
                content_length = int(raw_length)
            except ValueError:
                content_length = None
            if content_length is not None and content_length > self._limit:
                await self._send_rejection(send)
                return

        received = 0
        rejected = False

        async def limited_receive() -> dict:
            nonlocal received, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._limit:
                    rejected = True
                    await self._send_rejection(send)
                    return {"type": "http.disconnect"}
            return message

        async def limited_send(message: dict) -> None:
            if not rejected:
                await send(message)

        try:
            await self.app(scope, limited_receive, limited_send)
        except ClientDisconnect:
            if not rejected:
                raise


def _validated_api_keys(values: Iterable[str]) -> tuple[str, ...]:
    keys = tuple(values)
    if not keys:
        raise ValueError("api_keys must contain at least one key")
    for key in keys:
        if (
            not isinstance(key, str)
            or not key.isascii()
            or not 32 <= len(key) <= 4096
            or not key.strip()
        ):
            raise ValueError("API keys must be 32-4096 non-empty ASCII characters")
    if len(set(keys)) != len(keys):
        raise ValueError("api_keys must not contain duplicates")
    return keys


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _backend_unavailable() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        headers={"Retry-After": "1"},
        content={
            "error": {
                "code": "backend_unavailable",
                "message": "pre-stage state backend is temporarily unavailable",
            }
        },
    )


def create_node_model_cache_agent_app(
    *,
    node_id: str,
    executor: NodeModelPrestageExecutor,
    store: NodeModelPrestageStore,
    trust_store: ModelArtifactTrustStore,
    api_keys: Iterable[str],
    readiness_check: Callable[[], None],
    live_evidence_source: NodeModelCacheLiveEvidenceSource | None = None,
    request_body_limit_bytes: int = 64 * 1024 * 1024,
    active_request_limit: int = 2,
    total_request_limit: int = 8,
    queue_wait_timeout_s: float = 1.0,
) -> FastAPI:
    """Build a closed-by-default node agent app around the WP4.7 executor."""

    node_id = _text(node_id, name="node_id", max_length=253)
    if not isinstance(store, NodeModelPrestageStore):
        raise TypeError("store must implement NodeModelPrestageStore")
    if store.node_id != node_id:
        raise ValueError("store belongs to another node")
    if getattr(executor, "node_id", None) != node_id:
        raise ValueError("executor belongs to another node")
    if getattr(executor, "store", None) is not store:
        raise ValueError("executor and status API must share one store instance")
    if not isinstance(trust_store, ModelArtifactTrustStore):
        raise TypeError("trust_store must be a ModelArtifactTrustStore")
    trust_store = ModelArtifactTrustStore.model_validate(trust_store.model_dump())
    keys = _validated_api_keys(api_keys)
    if not callable(readiness_check):
        raise TypeError("readiness_check must be callable")
    if live_evidence_source is not None and not isinstance(
        live_evidence_source, NodeModelCacheLiveEvidenceSource
    ):
        raise TypeError("live_evidence_source must implement NodeModelCacheLiveEvidenceSource")
    if (
        type(request_body_limit_bytes) is not int
        or not 1 <= request_body_limit_bytes <= 128 * 1024 * 1024
    ):
        raise ValueError("request_body_limit_bytes must be in [1, 134217728]")
    if type(active_request_limit) is not int or active_request_limit < 1:
        raise ValueError("active_request_limit must be a positive integer")
    if type(total_request_limit) is not int or total_request_limit < active_request_limit:
        raise ValueError("total_request_limit must be at least active_request_limit")
    if (
        isinstance(queue_wait_timeout_s, bool)
        or not isinstance(queue_wait_timeout_s, (int, float))
        or not math.isfinite(float(queue_wait_timeout_s))
        or queue_wait_timeout_s <= 0
    ):
        raise ValueError("queue_wait_timeout_s must be finite and positive")

    app = FastAPI(
        title="kairyu-node-model-cache-agent",
        version="1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(
        _RequestBodyLimitMiddleware,
        limit=request_body_limit_bytes,
        paths=(_ENSURE_PATH, _RELEASE_PATH, _LIVE_EVIDENCE_PATH),
    )
    app.add_middleware(
        ConcurrencyLimitMiddleware,
        limit=active_request_limit,
        total_limit=total_request_limit,
        wait_timeout_s=queue_wait_timeout_s,
    )
    app.add_middleware(AuthMiddleware, api_keys=keys, protect_metrics=True)
    readiness_lock = asyncio.Lock()
    readiness_cached_at = 0.0
    readiness_cached_result = False

    @app.exception_handler(NodeModelPrestageExpiredError)
    async def expired_handler(_request: Request, _exc: Exception) -> JSONResponse:
        return _error(409, "command_expired", "pre-stage command is not currently valid")

    @app.exception_handler(NodeModelPrestageConflictError)
    async def conflict_handler(_request: Request, _exc: Exception) -> JSONResponse:
        return _error(409, "prestage_conflict", "pre-stage command conflicts with node state")

    @app.exception_handler(NodeModelPrestageCapacityError)
    async def capacity_handler(_request: Request, _exc: Exception) -> JSONResponse:
        return _error(507, "prestage_capacity", "pre-stage placement capacity is exhausted")

    @app.exception_handler(InvalidModelArtifactError)
    async def admission_handler(_request: Request, _exc: Exception) -> JSONResponse:
        return _error(422, "artifact_not_admitted", "model artifact admission failed")

    @app.exception_handler(NodeModelCacheError)
    async def cache_handler(_request: Request, _exc: Exception) -> JSONResponse:
        return _error(502, "cache_fill_failed", "verified model cache fill failed")

    @app.exception_handler(NodeModelCacheIndexError)
    async def cache_index_handler(_request: Request, _exc: Exception) -> JSONResponse:
        return _backend_unavailable()

    @app.exception_handler(_InvalidCacheAgentRequest)
    async def invalid_request_handler(
        _request: Request, exc: _InvalidCacheAgentRequest
    ) -> JSONResponse:
        return _error(exc.status_code, "invalid_request", exc.message)

    if _psycopg is not None:

        @app.exception_handler(_psycopg.Error)
        async def postgres_handler(_request: Request, _exc: Exception) -> JSONResponse:
            return _backend_unavailable()

    def require_node(command: NodeModelPrestageCommand) -> None:
        if command.node_id != node_id:
            raise NodeModelPrestageConflictError("command targets another node")

    @app.get("/health", response_model=NodeModelCacheAgentHealth)
    async def health() -> NodeModelCacheAgentHealth:
        return NodeModelCacheAgentHealth(status="ok", node_id=node_id)

    @app.get("/readyz", response_model=NodeModelCacheAgentHealth)
    async def ready():
        nonlocal readiness_cached_at, readiness_cached_result
        now = time.monotonic()
        if now - readiness_cached_at >= 0.5:
            async with readiness_lock:
                now = time.monotonic()
                if now - readiness_cached_at >= 0.5:
                    try:
                        await run_in_threadpool(readiness_check)
                    except Exception:
                        readiness_cached_result = False
                    else:
                        readiness_cached_result = True
                    readiness_cached_at = time.monotonic()
        if not readiness_cached_result:
            return JSONResponse(
                status_code=503,
                content=NodeModelCacheAgentHealth(status="not_ready", node_id=node_id).model_dump(
                    mode="json"
                ),
            )
        return NodeModelCacheAgentHealth(status="ready", node_id=node_id)

    @app.get("/v1/prestage/records", response_model=NodeModelPrestageRecordsResponse)
    async def records(
        after: Annotated[str | None, Query(min_length=1, max_length=255)] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 100,
    ) -> NodeModelPrestageRecordsResponse:
        if after is not None:
            try:
                after = _text(after, name="after")
            except ValueError as exc:
                raise _InvalidCacheAgentRequest(
                    status_code=422, message="after cursor is invalid"
                ) from exc
        page = await run_in_threadpool(
            store.list_records_page,
            after_placement_id=after,
            limit=limit + 1,
        )
        values = page[:limit]
        if any(record.command.node_id != node_id for record in values):
            raise NodeModelPrestageConflictError("stored record targets another node")
        next_cursor = values[-1].command.placement_id if len(page) > limit else None
        return NodeModelPrestageRecordsResponse(
            node_id=node_id,
            records=tuple(_public_status(record) for record in values),
            next_cursor=next_cursor,
        )

    @app.post(_ENSURE_PATH, response_model=NodeModelPrestageStatus)
    async def ensure(request: Request) -> NodeModelPrestageStatus:
        if request.headers.get("content-type", "").partition(";")[0].lower() != (
            "application/json"
        ):
            raise _InvalidCacheAgentRequest(
                status_code=415, message="content type must be application/json"
            )
        payload = await run_in_threadpool(_parse_ensure_request, await request.body())
        require_node(payload.command)
        return _public_status(
            await run_in_threadpool(
                executor.execute,
                payload.command,
                claim_id=payload.claim_id,
                envelope=payload.manifest,
                trust_store=trust_store,
                request=payload.admission_request,
            )
        )

    @app.post(_RELEASE_PATH, response_model=NodeModelPrestageStatus)
    async def release(request: Request) -> NodeModelPrestageStatus:
        if request.headers.get("content-type", "").partition(";")[0].lower() != (
            "application/json"
        ):
            raise _InvalidCacheAgentRequest(
                status_code=415, message="content type must be application/json"
            )
        payload = await run_in_threadpool(_parse_release_request, await request.body())
        require_node(payload.command)
        return _public_status(await run_in_threadpool(executor.release, payload.command))

    @app.post(
        _LIVE_EVIDENCE_PATH,
        response_model=NodeModelCacheLiveEvidenceResponse,
    )
    async def live_evidence(request: Request):
        if live_evidence_source is None:
            return _backend_unavailable()
        if request.headers.get("content-type", "").partition(";")[0].lower() != (
            "application/json"
        ):
            raise _InvalidCacheAgentRequest(
                status_code=415, message="content type must be application/json"
            )
        payload = await run_in_threadpool(
            _parse_live_evidence_request,
            await request.body(),
        )
        result = await run_in_threadpool(live_evidence_source.read, payload)
        if not isinstance(result, NodeModelCacheLiveEvidenceResponse):
            raise TypeError("live evidence source returned an invalid response")
        result = NodeModelCacheLiveEvidenceResponse.model_validate(result.model_dump())
        return JSONResponse(
            content=result.model_dump(mode="json"),
            headers={"Cache-Control": "no-store"},
        )

    return app
