"""Authenticated HTTP boundary for live cache-startup binding authorization."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import secrets
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.requests import ClientDisconnect

from kairyu.entrypoints.server.middleware import ConcurrencyLimitMiddleware
from kairyu.runners.startup_binding import RunnerCacheStartupBinding
from kairyu.runners.startup_binding_authority import (
    RunnerCachePlacementBindingAuthorityReadiness,
    RunnerCachePlacementBindingAuthorizationDeniedError,
    RunnerCachePlacementBindingAuthorizationRequest,
    RunnerCachePlacementBindingAuthorizationResponse,
    RunnerCachePlacementBindingLiveAuthority,
    validate_runner_cache_placement_bearer_token,
)

_AUTHORIZE_PATH = "/v1/reauthorize"
_ASGIApp = Callable[..., Awaitable[None]]
_logger = logging.getLogger(__name__)


class _InvalidAuthorizationRequest(ValueError):
    def __init__(self, *, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class _AuthorizationResponseTooLarge(RuntimeError):
    pass


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("JSON object keys must be unique")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON constant {value!r} is not permitted")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON numbers must be finite")
    return parsed


def _parse_request(body: bytes) -> RunnerCachePlacementBindingAuthorizationRequest:
    try:
        value = json.loads(
            body,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise _InvalidAuthorizationRequest(
            status_code=400,
            message="request body must be strict JSON",
        ) from exc
    try:
        return RunnerCachePlacementBindingAuthorizationRequest.model_validate(value)
    except ValidationError as exc:
        raise _InvalidAuthorizationRequest(
            status_code=422,
            message="request body does not match binding authorization v1",
        ) from exc


class _RequestBodyLimitMiddleware:
    """Reject an oversized authorization request before JSON materialization."""

    def __init__(self, app: _ASGIApp, *, limit: int) -> None:
        self.app = app
        self._limit = limit

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
                    (b"cache-control", b"no-store"),
                ),
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != _AUTHORIZE_PATH
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


class _NoStoreResponseMiddleware:
    """Apply the authority's no-store contract to middleware responses too."""

    def __init__(self, app: _ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        async def no_store_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    header
                    for header in message.get("headers", ())
                    if header[0].lower() != b"cache-control"
                ]
                headers.append((b"cache-control", b"no-store"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, no_store_send)


def _serialize_authorized_response(
    candidate: RunnerCachePlacementBindingAuthorizationRequest,
    current: object,
    *,
    response_body_limit_bytes: int,
) -> bytes:
    if not isinstance(current, RunnerCacheStartupBinding):
        raise TypeError("reauthorize must return RunnerCacheStartupBinding")
    current = RunnerCacheStartupBinding.model_validate(current.model_dump())
    if current != candidate.binding:
        raise RunnerCachePlacementBindingAuthorizationDeniedError(
            "candidate binding is not current"
        )
    response = RunnerCachePlacementBindingAuthorizationResponse(
        nonce=candidate.nonce,
        binding=current,
    )
    response_body = response.model_dump_json().encode("utf-8")
    if len(response_body) > response_body_limit_bytes:
        raise _AuthorizationResponseTooLarge
    return response_body


def create_runner_cache_placement_binding_authority_app(
    *,
    reauthorize: RunnerCachePlacementBindingLiveAuthority,
    readiness_check: RunnerCachePlacementBindingAuthorityReadiness,
    bearer_token: str,
    request_body_limit_bytes: int = 1024 * 1024,
    response_body_limit_bytes: int = 1024 * 1024 + 1,
    active_request_limit: int = 16,
    total_request_limit: int = 64,
    queue_wait_timeout_s: float = 0.5,
    request_timeout_s: float = 1.5,
    backend_timeout_s: float = 1.0,
) -> FastAPI:
    """Build the authenticated, fail-closed live binding authority API."""

    if not callable(reauthorize):
        raise TypeError("reauthorize must be callable")
    if not callable(readiness_check):
        raise TypeError("readiness_check must be callable")
    bearer_token = validate_runner_cache_placement_bearer_token(bearer_token)
    if (
        type(request_body_limit_bytes) is not int
        or not 1 <= request_body_limit_bytes <= 16 * 1024 * 1024
    ):
        raise ValueError("request_body_limit_bytes must be in [1, 16777216]")
    if (
        type(response_body_limit_bytes) is not int
        or not 2 <= response_body_limit_bytes <= 16 * 1024 * 1024
        or response_body_limit_bytes <= request_body_limit_bytes
    ):
        raise ValueError(
            "response_body_limit_bytes must exceed request_body_limit_bytes and be at most 16777216"
        )
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
    for name, value in (
        ("request_timeout_s", request_timeout_s),
        ("backend_timeout_s", backend_timeout_s),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 < value <= 30
        ):
            raise ValueError(f"{name} must be finite and in (0, 30]")
    if backend_timeout_s >= request_timeout_s:
        raise ValueError("backend_timeout_s must be less than request_timeout_s")

    app = FastAPI(
        title="kairyu-runner-cache-placement-binding-authority",
        version="1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(_RequestBodyLimitMiddleware, limit=request_body_limit_bytes)
    app.add_middleware(
        ConcurrencyLimitMiddleware,
        limit=active_request_limit,
        total_limit=total_request_limit,
        wait_timeout_s=queue_wait_timeout_s,
    )
    app.add_middleware(_NoStoreResponseMiddleware)

    def authenticated(request: Request) -> bool:
        supplied = request.headers.get("authorization", "")
        return supplied.isascii() and secrets.compare_digest(supplied, f"Bearer {bearer_token}")

    @app.exception_handler(_InvalidAuthorizationRequest)
    async def invalid_request_handler(
        _request: Request,
        exc: _InvalidAuthorizationRequest,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/health")
    async def health() -> Response:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    @app.get("/readyz")
    async def ready(request: Request) -> Response:
        if not authenticated(request):
            return Response(
                status_code=401,
                headers={
                    "Cache-Control": "no-store",
                    "WWW-Authenticate": "Bearer",
                },
            )
        try:
            deadline_monotonic = time.monotonic() + request_timeout_s
            async with asyncio.timeout(request_timeout_s):
                await asyncio.to_thread(
                    readiness_check,
                    deadline_monotonic=deadline_monotonic,
                    backend_timeout_s=backend_timeout_s,
                )
        except TimeoutError:
            return Response(status_code=503, headers={"Cache-Control": "no-store"})
        except Exception:
            return Response(status_code=503, headers={"Cache-Control": "no-store"})
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    @app.post(_AUTHORIZE_PATH)
    async def authorize(request: Request) -> Response:
        deadline_monotonic = time.monotonic() + request_timeout_s
        if not authenticated(request):
            return Response(
                status_code=401,
                headers={
                    "Cache-Control": "no-store",
                    "WWW-Authenticate": "Bearer",
                },
            )
        if request.headers.get("content-type", "").partition(";")[0].lower() != (
            "application/json"
        ):
            raise _InvalidAuthorizationRequest(
                status_code=415,
                message="content type must be application/json",
            )
        try:
            async with asyncio.timeout(request_timeout_s):
                body = await request.body()
                candidate = await asyncio.to_thread(_parse_request, body)
                current = await asyncio.to_thread(
                    reauthorize,
                    candidate.binding,
                    deadline_monotonic=deadline_monotonic,
                    backend_timeout_s=backend_timeout_s,
                )
                response_body = await asyncio.to_thread(
                    _serialize_authorized_response,
                    candidate,
                    current,
                    response_body_limit_bytes=response_body_limit_bytes,
                )
            if time.monotonic() >= deadline_monotonic:
                raise TimeoutError
        except _InvalidAuthorizationRequest:
            raise
        except RunnerCachePlacementBindingAuthorizationDeniedError:
            return JSONResponse(
                status_code=409,
                content={"detail": "binding is not currently authorized"},
                headers={"Cache-Control": "no-store"},
            )
        except TimeoutError:
            return JSONResponse(
                status_code=503,
                content={"detail": "binding authority exceeded its internal deadline"},
                headers={"Cache-Control": "no-store"},
            )
        except _AuthorizationResponseTooLarge:
            return JSONResponse(
                status_code=503,
                content={"detail": "binding authority response exceeds its configured limit"},
                headers={"Cache-Control": "no-store"},
            )
        except Exception:
            _logger.exception("runner cache placement binding authority failed")
            return JSONResponse(
                status_code=503,
                content={"detail": "binding authority is temporarily unavailable"},
                headers={"Cache-Control": "no-store"},
            )
        return Response(
            status_code=200,
            content=response_body,
            media_type="application/json",
            headers={"Cache-Control": "no-store"},
        )

    return app


__all__ = ["create_runner_cache_placement_binding_authority_app"]
