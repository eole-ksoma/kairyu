"""Pure-ASGI middleware: auth, concurrency guard, metrics, JSON access log.

Pure ASGI (not ``BaseHTTPMiddleware``) so the concurrency guard holds its slot
until the last body byte of an SSE stream is sent, and metrics measure the
full streamed response (design m7 D4/D5/D8).
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from functools import lru_cache

from starlette.requests import ClientDisconnect

from kairyu.entrypoints.server.messages_protocol import (
    anthropic_error_payload,
    anthropic_error_type_for_status,
    wants_anthropic_envelope,
)

_ASGIApp = Callable[..., Awaitable[None]]

# /backends is an open introspection endpoint (kept out of /v1/ so the
# concurrency cap and per-model metrics labeling don't apply); its disclosure
# level matches the already-open /readyz and /metrics.
# /api/hello is Claude Code's best-effort connection-warming probe (issue
# #508): a credential-free empty 200 whose disclosure level matches /health.
_OPEN_PATHS = ("/health", "/readyz", "/backends", "/api/hello")
_GUARDED_PREFIX = "/v1/"
_SLO_ADMISSION_LEASE_STATE_KEY = "slo_admission_lease"
# In-process sentinel (#573): the Anthropic Messages adapter sets this on the
# request state before an AUTO chat dispatch so tool-bearing requests stream
# raw; it is unreachable from the wire, so the public /v1/chat/completions
# contract (including its pre-SSE tool gates) is unchanged.
_ANTHROPIC_INTERNAL_TOOL_STREAM_STATE_KEY = "kairyu_internal_anthropic_tool_stream"

# collapse per-object id path segments (file-…, batch_…, uuids, long hex/digits)
# to {id} so a Prometheus path label cannot explode in cardinality (M1)
_ID_SEGMENT = re.compile(
    r"^(file-|batch_|req-|resp_|chatcmpl-|cmpl-|[0-9a-f-]{16,}|\d{6,})"
)


@lru_cache(maxsize=1024)
def _template_path(path: str) -> str:
    segments = path.split("/")
    if len(segments) >= 4 and segments[1:3] == ["v1", "requests"]:
        segments[3] = "{id}"
        return "/".join(segments)
    return "/".join(
        "{id}" if _ID_SEGMENT.match(segment) else segment for segment in path.split("/")
    )

access_logger = logging.getLogger("kairyu.access")


async def _send_json(send: Callable, status: int, payload: dict, headers: dict[str, str]) -> None:
    body = json.dumps(payload).encode()
    raw_headers = [(k.encode(), v.encode()) for k, v in headers.items()]
    raw_headers.append((b"content-type", b"application/json"))
    raw_headers.append((b"content-length", str(len(body)).encode()))
    await send({"type": "http.response.start", "status": status, "headers": raw_headers})
    await send({"type": "http.response.body", "body": body})


def _state(scope: dict) -> dict:
    return scope.setdefault("state", {})


async def _send_error(
    send: Callable,
    scope: dict,
    *,
    status: int,
    message: str,
    openai_type: str,
    code: str,
    headers: dict[str, str] | None = None,
) -> None:
    """Render one middleware error in the envelope the route's dialect expects.

    ``/v1/messages`` (Anthropic Messages, issue #508) must never receive the
    OpenAI ``{"error": ...}`` envelope; every other route keeps it unchanged.
    """

    if wants_anthropic_envelope(scope.get("path", "")):
        payload = anthropic_error_payload(
            message,
            error_type=anthropic_error_type_for_status(status),
            request_id=_state(scope).get("request_id"),
        )
    else:
        payload = {"error": {"message": message, "type": openai_type, "code": code}}
    await _send_json(send, status, payload, headers or {})


class AuthMiddleware:
    """Static API keys (env-sourced), constant-time compare; /health and /readyz open."""

    def __init__(
        self,
        app: _ASGIApp,
        *,
        api_keys: Iterable[str],
        admin_keys: Iterable[str] = (),
        protect_metrics: bool = False,
    ) -> None:
        self.app = app
        self._api_keys = tuple(api_keys)
        self._admin_keys = tuple(admin_keys)
        self._protect_metrics = protect_metrics

    def _authorized(self, scope: dict) -> bool:
        headers = dict(scope.get("headers") or ())
        header = headers.get(b"authorization", b"")
        prefix, _, token = header.decode("latin-1").partition(" ")
        if prefix.lower() != "bearer" or not token:
            # Anthropic-format clients (Claude Code, issue #508) send the same
            # gateway credential in x-api-key; both headers feed the identical
            # key sets and tenant policy. Bearer wins when both are present.
            token = headers.get(b"x-api-key", b"").decode("latin-1")
        if not token:
            return False
        # hmac.compare_digest rejects non-ASCII strings with TypeError; a
        # non-ASCII token can never match an ASCII key, so 401 not 500 (M5)
        if not token.isascii():
            return False
        is_data_plane = False
        for key in self._api_keys:
            is_data_plane |= hmac.compare_digest(token, key)
        is_admin = False
        for key in self._admin_keys:
            is_admin |= hmac.compare_digest(token, key)
        if not (is_data_plane or is_admin):
            return False
        state = _state(scope)
        state["api_key"] = token
        state["is_data_plane"] = is_data_plane
        state["is_admin"] = is_admin
        return True

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        exempt = path in _OPEN_PATHS or (path == "/metrics" and not self._protect_metrics)
        if exempt:
            await self.app(scope, receive, send)
            return
        if self._authorized(scope):
            if path.startswith(_GUARDED_PREFIX) and not _state(scope)["is_data_plane"]:
                await _send_error(
                    send,
                    scope,
                    status=403,
                    message="data-plane API key required",
                    openai_type="invalid_request_error",
                    code="data_plane_required",
                )
                return
            await self.app(scope, receive, send)
            return
        await _send_error(
            send,
            scope,
            status=401,
            message="missing or invalid API key",
            openai_type="invalid_request_error",
            code="invalid_api_key",
            headers={"www-authenticate": "Bearer"},
        )


class ConcurrencyLimitMiddleware:
    """Bound active and queued /v1/* work; saturation returns 429 (m7 D5).

    Fine-grained per-client rate limiting is the edge WAF/LB's job — this guard
    only protects the process from overload.
    """

    def __init__(
        self,
        app: _ASGIApp,
        *,
        limit: int,
        total_limit: int,
        wait_timeout_s: float | None,
        metrics=None,
    ) -> None:
        if not 1 <= limit <= total_limit:
            raise ValueError("active limit must be within the total concurrency limit")
        if wait_timeout_s is not None and wait_timeout_s <= 0:
            raise ValueError("admission wait timeout must be positive")
        if total_limit > limit and wait_timeout_s is None:
            raise ValueError("an admission queue requires a wait timeout")
        self.app = app
        self._limit = limit
        self._total_limit = total_limit
        self._queue_limit = total_limit - limit
        self._wait_timeout_s = wait_timeout_s
        self._metrics = metrics
        self._active = 0
        self._waiters: deque[asyncio.Future[None]] = deque()

    def _publish_depth(self) -> None:
        if self._metrics is not None:
            self._metrics.set_admission_depth(
                active=self._active,
                waiting=len(self._waiters),
            )

    def _record_rejection(self, reason: str) -> None:
        if self._metrics is not None:
            self._metrics.record_admission_rejection(reason)

    def _discard_waiter(self, waiter: asyncio.Future[None]) -> None:
        try:
            self._waiters.remove(waiter)
        except ValueError:
            return
        self._publish_depth()

    def _release_slot(self) -> None:
        while self._waiters:
            waiter = self._waiters.popleft()
            if waiter.done():
                continue
            self._publish_depth()
            waiter.set_result(None)
            return
        self._active -= 1
        self._publish_depth()

    async def _reject(self, scope: dict, send: Callable, message: str) -> None:
        await _send_error(
            send,
            scope,
            status=429,
            message=message,
            openai_type="rate_limit_error",
            code="concurrency_exceeded",
            headers={"retry-after": "1"},
        )

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(_GUARDED_PREFIX):
            await self.app(scope, receive, send)
            return
        acquired = False
        if self._active >= self._limit or self._waiters:
            if len(self._waiters) >= self._queue_limit:
                message = (
                    f"server is at max concurrency ({self._limit})"
                    if self._queue_limit == 0
                    else f"server admission queue is full ({self._total_limit})"
                )
                self._record_rejection("overflow")
                await self._reject(
                    scope,
                    send,
                    message,
                )
                return
            waiter = asyncio.get_running_loop().create_future()
            self._waiters.append(waiter)
            self._publish_depth()
            try:
                assert self._wait_timeout_s is not None
                await asyncio.wait_for(
                    waiter,
                    timeout=self._wait_timeout_s,
                )
                acquired = True
            except TimeoutError:
                if waiter.done() and not waiter.cancelled():
                    acquired = True
                else:
                    self._discard_waiter(waiter)
                    self._record_rejection("timeout")
                    await self._reject(
                        scope,
                        send,
                        "server admission queue wait timed out",
                    )
                    return
            except asyncio.CancelledError:
                if waiter.done() and not waiter.cancelled():
                    self._release_slot()
                else:
                    waiter.cancel()
                    self._discard_waiter(waiter)
                raise
        else:
            self._active += 1
            acquired = True
            self._publish_depth()
        assert acquired
        try:
            await self.app(scope, receive, send)
        finally:
            self._release_slot()


class ChatBodyLimitMiddleware:
    """Bound chat-shaped JSON before Starlette materializes the body.

    Covers both dialect front doors intentionally (issue #508): the OpenAI
    ``/v1/chat/completions`` route and the Anthropic ``/v1/messages`` route.
    """

    _PATHS = (
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/messages/count_tokens",
    )

    def __init__(
        self,
        app: _ASGIApp,
        *,
        limit: int,
        paths: Iterable[str] | None = None,
    ) -> None:
        self.app = app
        self._limit = limit
        self._paths = frozenset(self._PATHS if paths is None else paths)

    async def _reject(self, scope: dict, send: Callable) -> None:
        await _send_error(
            send,
            scope,
            status=413,
            message=(
                "chat request body exceeds the configured "
                f"{self._limit}-byte limit"
            ),
            openai_type="invalid_request_error",
            code="request_too_large",
        )

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in self._paths
        ):
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or ())
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                content_length = int(raw_length)
            except ValueError:
                content_length = None
            if content_length is not None and content_length > self._limit:
                await self._reject(scope, send)
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
                    await self._reject(scope, send)
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


class MetricsMiddleware:
    """Record kairyu_requests_total{model,code} and duration; model set by the handler."""

    def __init__(self, app: _ASGIApp, *, metrics) -> None:
        self.app = app
        self._metrics = metrics

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started_ns = time.perf_counter_ns()
        # RequestIngressMiddleware owns the true outer-boundary timestamp.
        # Retain this fallback for direct middleware tests and embedded stacks.
        _state(scope).setdefault("placement_started_ns", started_ns)
        status = {"code": 500}

        async def wrapped_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, wrapped_send)
        finally:
            path = _template_path(scope["path"])  # bounded-cardinality label (M1)
            self._metrics.request_duration_seconds.labels(path=path).observe(
                (time.perf_counter_ns() - started_ns) / 1_000_000_000
            )
            if scope["path"].startswith(_GUARDED_PREFIX):
                # an unknown model (404) collapses to "unknown" so an attacker
                # looping random model names can't grow the timeseries (M1)
                model = _state(scope).get("model", "-")
                if status["code"] == 404:
                    model = "unknown"
                self._metrics.requests_total.labels(
                    model=model, code=str(status["code"])
                ).inc()


class RequestIngressMiddleware:
    """Own request-boundary timing and direct-chat SLO lease cleanup."""

    def __init__(self, app: _ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] == "http":
            _state(scope)["placement_started_ns"] = time.perf_counter_ns()
        try:
            await self.app(scope, receive, send)
        finally:
            if scope["type"] == "http":
                lease = _state(scope).pop(_SLO_ADMISSION_LEASE_STATE_KEY, None)
                if lease is not None and lease.active:
                    lease.completed()


class AccessLogMiddleware:
    """One JSON line per request; assigns and echoes X-Request-ID."""

    def __init__(self, app: _ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex[:16]
        _state(scope)["request_id"] = request_id
        started = time.perf_counter()
        status = {"code": 500}

        async def wrapped_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                headers = list(message.get("headers") or [])
                headers.append((b"x-request-id", request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, wrapped_send)
        finally:
            access_logger.info(
                "request",
                extra={
                    "request_id": request_id,
                    "method": scope["method"],
                    "path": scope["path"],
                    "code": status["code"],
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )


class JsonLogFormatter(logging.Formatter):
    """Stdlib JSON formatter (m7 D8 — no structlog/OTel dependency)."""

    _RESERVED = frozenset(
        logging.LogRecord("", 0, "", 0, "", (), None).__dict__
    ) | {"message", "asctime", "taskName"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_json_logging(level: int = logging.INFO) -> None:
    """Route root logging through the JSON formatter (used by `kairyu serve`)."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # Successful outbound requests are already represented by Kairyu's
    # authoritative structured access line. Keep dependency warnings/errors
    # without serializing another INFO line for every replica dispatch.
    logging.getLogger("httpx").setLevel(logging.WARNING)


class TracingMiddleware:
    """Gateway request span (m10a D4): one span per /v1/* request.

    Pure ASGI like the rest of this file — streaming responses must not be
    buffered by a BaseHTTPMiddleware."""

    def __init__(self, app: _ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith("/v1/"):
            await self.app(scope, receive, send)
            return
        from kairyu.telemetry import (
            extract_trace_context,
            mark_span_error,
            traced_span,
        )

        status = {"code": 500, "complete": False, "disconnected": False}

        async def wrapped_receive() -> dict:
            message = await receive()
            if message["type"] == "http.disconnect":
                status["disconnected"] = True
            return message

        async def wrapped_send(message: dict) -> None:
            try:
                await send(message)
            except OSError:
                # ASGI 2.4+ reports a disconnected client from send(); older
                # servers expose the same condition through http.disconnect.
                status["disconnected"] = True
                raise
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            elif (
                message["type"] == "http.response.body"
                and not message.get("more_body", False)
            ):
                # Complete means the downstream server accepted the final
                # body, not merely that the application attempted to send it.
                status["complete"] = True

        attributes: dict[str, str] = {
            # Keep the legacy names for existing dashboards while also
            # emitting the current HTTP semantic-convention spelling.
            "http.route": scope["path"],
            "http.method": scope.get("method", ""),
            "http.request.method": scope.get("method", ""),
        }
        request_id = _state(scope).get("request_id")
        if isinstance(request_id, str):
            attributes["kairyu.request_id"] = request_id
        carrier = {
            key.decode("latin-1"): value.decode("latin-1")
            for key, value in scope.get("headers", ())
        }
        with traced_span(
            "kairyu.request",
            attributes,
            context=extract_trace_context(carrier),
            kind="server",
        ) as span:
            try:
                await self.app(scope, wrapped_receive, wrapped_send)
            finally:
                if span is not None:
                    span.set_attribute("http.status_code", status["code"])
                    span.set_attribute("http.response.status_code", status["code"])
                    span.set_attribute(
                        "kairyu.response.complete",
                        status["complete"],
                    )
                    if not status["complete"]:
                        mark_span_error(
                            span,
                            error_type=(
                                "ClientDisconnect"
                                if status["disconnected"]
                                else "incomplete_response"
                            ),
                            cancelled=status["disconnected"],
                        )
                    elif status["code"] >= 500:
                        mark_span_error(span, error_type=str(status["code"]))
