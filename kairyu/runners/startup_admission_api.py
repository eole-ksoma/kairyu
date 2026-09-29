"""Kubernetes AdmissionReview transport for cache-aware Runner placement."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from kairyu.entrypoints.server.middleware import ConcurrencyLimitMiddleware
from kairyu.runners.startup_admission import (
    RunnerCachePlacementAdmissionConflictError,
    RunnerCachePlacementAdmissionController,
    RunnerCachePlacementAdmissionError,
    RunnerCachePlacementAdmissionTimeoutError,
)
from kairyu.runners.startup_scheduling import RunnerCacheSchedulingError

_ADMIT_PATH = "/v1/admit"
_ASGIApp = Callable[..., Awaitable[None]]
_logger = logging.getLogger(__name__)


def _strict_model_config() -> ConfigDict:
    return ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=False,
        strict=True,
        revalidate_instances="always",
    )


class KubernetesGroupVersionKind(BaseModel):
    """Group/version/kind identity supplied by the API server."""

    model_config = _strict_model_config()

    group: str = Field(max_length=253)
    version: str = Field(min_length=1, max_length=63)
    kind: str = Field(min_length=1, max_length=63)


class KubernetesGroupVersionResource(BaseModel):
    """Group/version/resource identity supplied by the API server."""

    model_config = _strict_model_config()

    group: str = Field(max_length=253)
    version: str = Field(min_length=1, max_length=63)
    resource: str = Field(min_length=1, max_length=63)


class KubernetesAdmissionUserInfo(BaseModel):
    """Authenticated API request identity used by placement authorization."""

    model_config = _strict_model_config()

    username: str = Field(min_length=1, max_length=255)
    uid: str | None = Field(default=None, max_length=255)
    groups: list[str] | None = None
    extra: dict[str, list[str]] | None = None

    @field_validator("username", "uid")
    @classmethod
    def validate_identity(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or "\x00" in value):
            raise ValueError("identity fields must be non-empty and contain no NUL")
        return value


class KubernetesAdmissionRequest(BaseModel):
    """AdmissionRequest fields emitted for an admission.k8s.io/v1 webhook."""

    model_config = _strict_model_config()

    uid: str = Field(min_length=1, max_length=255)
    kind: KubernetesGroupVersionKind
    resource: KubernetesGroupVersionResource
    sub_resource: str = Field(default="", alias="subResource", max_length=253)
    request_kind: KubernetesGroupVersionKind | None = Field(default=None, alias="requestKind")
    request_resource: KubernetesGroupVersionResource | None = Field(
        default=None, alias="requestResource"
    )
    request_sub_resource: str | None = Field(
        default=None, alias="requestSubResource", max_length=253
    )
    name: str = Field(min_length=1, max_length=253)
    namespace: str = Field(min_length=1, max_length=253)
    operation: str = Field(min_length=1, max_length=32)
    user_info: KubernetesAdmissionUserInfo = Field(alias="userInfo")
    object: dict[str, Any] | None
    old_object: dict[str, Any] | None = Field(default=None, alias="oldObject")
    dry_run: bool | None = Field(default=None, alias="dryRun")
    options: dict[str, Any] | None = None

    @field_validator("uid", "name", "namespace", "operation")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("identity fields must be non-empty and contain no NUL")
        return value


class KubernetesAdmissionReview(BaseModel):
    """Strict AdmissionReview envelope accepted by the webhook."""

    model_config = _strict_model_config()

    api_version: Literal["admission.k8s.io/v1"] = Field(alias="apiVersion")
    kind: Literal["AdmissionReview"]
    request: KubernetesAdmissionRequest


class RunnerCachePlacementAdmissionHealth(BaseModel):
    """Low-disclosure liveness/readiness response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["ok", "ready", "not_ready"]


class _InvalidAdmissionReview(ValueError):
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


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON numbers must be finite")
    return parsed


def _parse_admission_review(body: bytes) -> KubernetesAdmissionReview:
    try:
        value = json.loads(
            body,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise _InvalidAdmissionReview(
            status_code=400,
            message="request body must be strict JSON",
        ) from exc
    try:
        return KubernetesAdmissionReview.model_validate(value)
    except ValidationError as exc:
        raise _InvalidAdmissionReview(
            status_code=422,
            message="request body does not match AdmissionReview v1",
        ) from exc


class _RequestBodyLimitMiddleware:
    """Reject oversized admission reviews before materializing request JSON."""

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


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _json_patch(before: Any, after: Any, *, path: str = "") -> list[dict[str, Any]]:
    """Create a deterministic RFC 6902 patch without trusting an extra package."""

    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        patch: list[dict[str, Any]] = []
        before_keys = set(before)
        after_keys = set(after)
        for key in sorted(before_keys - after_keys, reverse=True):
            patch.append({"op": "remove", "path": f"{path}/{_json_pointer_token(key)}"})
        for key in sorted(before_keys & after_keys):
            patch.extend(
                _json_patch(
                    before[key],
                    after[key],
                    path=f"{path}/{_json_pointer_token(key)}",
                )
            )
        for key in sorted(after_keys - before_keys):
            patch.append(
                {
                    "op": "add",
                    "path": f"{path}/{_json_pointer_token(key)}",
                    "value": copy.deepcopy(after[key]),
                }
            )
        return patch
    return [{"op": "replace", "path": path, "value": copy.deepcopy(after)}]


def _review_response(
    uid: str,
    *,
    allowed: bool,
    status_code: int | None = None,
    message: str | None = None,
    patch: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {"uid": uid, "allowed": allowed}
    if status_code is not None or message is not None:
        response["status"] = {
            "code": status_code if status_code is not None else 500,
            "message": message or "admission failed",
        }
    if patch:
        raw_patch = json.dumps(
            patch,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        response["patchType"] = "JSONPatch"
        response["patch"] = base64.b64encode(raw_patch).decode("ascii")
    return {
        "apiVersion": "admission.k8s.io/v1",
        "kind": "AdmissionReview",
        "response": response,
    }


def _deny(uid: str, *, code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content=_review_response(uid, allowed=False, status_code=code, message=message),
    )


def _validate_pod_create(request: KubernetesAdmissionRequest) -> str | None:
    pod_kind = KubernetesGroupVersionKind(group="", version="v1", kind="Pod")
    pod_resource = KubernetesGroupVersionResource(group="", version="v1", resource="pods")
    if request.operation != "CREATE":
        return "only Pod CREATE admission is supported"
    if request.kind != pod_kind or request.resource != pod_resource:
        return "only core/v1 Pod admission is supported"
    if request.sub_resource or request.request_sub_resource not in (None, ""):
        return "Pod subresource admission is not supported"
    if request.request_kind is not None and request.request_kind != pod_kind:
        return "requestKind must identify a core/v1 Pod"
    if request.request_resource is not None and request.request_resource != pod_resource:
        return "requestResource must identify core/v1 pods"
    if request.dry_run is True:
        return "dry-run placement admission is not supported"
    if request.old_object is not None:
        return "oldObject must be null for Pod CREATE admission"
    if request.object is None:
        return "Pod object is required"
    pod = request.object
    if pod.get("apiVersion") != "v1" or pod.get("kind") != "Pod":
        return "object must identify a core/v1 Pod"
    metadata = pod.get("metadata")
    if not isinstance(metadata, Mapping):
        return "Pod metadata must be an object"
    if metadata.get("name") != request.name:
        return "AdmissionRequest and Pod names must match"
    if metadata.get("namespace") != request.namespace:
        return "AdmissionRequest and Pod namespaces must match"
    return None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def create_runner_cache_placement_admission_app(
    *,
    controller: RunnerCachePlacementAdmissionController,
    readiness_check: Callable[[], None],
    clock: Callable[[], datetime] = _utc_now,
    request_body_limit_bytes: int = 1024 * 1024,
    active_request_limit: int = 16,
    total_request_limit: int = 64,
    queue_wait_timeout_s: float = 0.5,
    request_timeout_s: float = 4.0,
) -> FastAPI:
    """Build the fail-closed AdmissionReview v1 transport for Pod CREATE."""

    if not isinstance(controller, RunnerCachePlacementAdmissionController):
        raise TypeError("controller must be a RunnerCachePlacementAdmissionController")
    if not callable(readiness_check):
        raise TypeError("readiness_check must be callable")
    if not callable(clock):
        raise TypeError("clock must be callable")
    if (
        type(request_body_limit_bytes) is not int
        or not 1 <= request_body_limit_bytes <= 16 * 1024 * 1024
    ):
        raise ValueError("request_body_limit_bytes must be in [1, 16777216]")
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
    if (
        isinstance(request_timeout_s, bool)
        or not isinstance(request_timeout_s, (int, float))
        or not math.isfinite(float(request_timeout_s))
        or not 0 < request_timeout_s <= 30
    ):
        raise ValueError("request_timeout_s must be finite and in (0, 30]")

    app = FastAPI(
        title="kairyu-runner-cache-placement-admission",
        version="1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(
        _RequestBodyLimitMiddleware,
        limit=request_body_limit_bytes,
        paths=(_ADMIT_PATH,),
    )
    app.add_middleware(
        ConcurrencyLimitMiddleware,
        limit=active_request_limit,
        total_limit=total_request_limit,
        wait_timeout_s=queue_wait_timeout_s,
    )
    readiness_lock = asyncio.Lock()
    readiness_cached_at = 0.0
    readiness_cached_result = False

    @app.exception_handler(_InvalidAdmissionReview)
    async def invalid_review_handler(
        _request: Request, exc: _InvalidAdmissionReview
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
        )

    @app.get("/health", response_model=RunnerCachePlacementAdmissionHealth)
    async def health() -> RunnerCachePlacementAdmissionHealth:
        return RunnerCachePlacementAdmissionHealth(status="ok")

    @app.get("/readyz", response_model=RunnerCachePlacementAdmissionHealth)
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
                content=RunnerCachePlacementAdmissionHealth(status="not_ready").model_dump(
                    mode="json"
                ),
            )
        return RunnerCachePlacementAdmissionHealth(status="ready")

    @app.post(_ADMIT_PATH)
    async def admit(request: Request) -> JSONResponse:
        deadline_monotonic = time.monotonic() + request_timeout_s
        if request.headers.get("content-type", "").partition(";")[0].lower() != (
            "application/json"
        ):
            raise _InvalidAdmissionReview(
                status_code=415,
                message="content type must be application/json",
            )
        review = await run_in_threadpool(_parse_admission_review, await request.body())
        admission_request = review.request
        invalid = _validate_pod_create(admission_request)
        if invalid is not None:
            return _deny(admission_request.uid, code=422, message=invalid)

        try:
            observed_at = clock()
            if (
                not isinstance(observed_at, datetime)
                or observed_at.tzinfo is None
                or observed_at.utcoffset() is None
            ):
                raise RuntimeError("clock returned an invalid timestamp")
            assert admission_request.object is not None
            admitted, _claim = await run_in_threadpool(
                controller.admit,
                admission_request.object,
                admission_uid=admission_request.uid,
                request_username=admission_request.user_info.username,
                observed_at=observed_at,
                deadline_monotonic=deadline_monotonic,
            )
            patch = _json_patch(admission_request.object, admitted)
            return JSONResponse(
                status_code=200,
                content=_review_response(
                    admission_request.uid,
                    allowed=True,
                    patch=patch,
                ),
            )
        except RunnerCachePlacementAdmissionTimeoutError:
            return _deny(
                admission_request.uid,
                code=503,
                message="cache placement admission exceeded its internal deadline",
            )
        except RunnerCachePlacementAdmissionConflictError:
            return _deny(
                admission_request.uid,
                code=409,
                message="cache placement admission conflicts with current state",
            )
        except (RunnerCachePlacementAdmissionError, RunnerCacheSchedulingError):
            return _deny(
                admission_request.uid,
                code=422,
                message="Pod is not eligible for cache placement admission",
            )
        except Exception:
            _logger.exception("runner cache placement admission backend failed")
            return _deny(
                admission_request.uid,
                code=503,
                message="cache placement admission is temporarily unavailable",
            )

    return app


__all__ = [
    "KubernetesAdmissionRequest",
    "KubernetesAdmissionReview",
    "KubernetesAdmissionUserInfo",
    "KubernetesGroupVersionKind",
    "KubernetesGroupVersionResource",
    "RunnerCachePlacementAdmissionHealth",
    "create_runner_cache_placement_admission_app",
]
