"""Tenant-scoped durable asynchronous Chat Completions routes."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Annotated

from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from kairyu.async_requests.models import (
    AsyncRequest,
    AsyncRequestState,
    AsyncRequestSubmission,
    status_of,
)
from kairyu.async_requests.store import (
    IdempotencyConflictError,
    RequestCapacityError,
    RequestStoreProtocol,
)
from kairyu.async_requests.worker import AsyncRequestWorker
from kairyu.entrypoints.server.errors import model_not_found
from kairyu.entrypoints.server.protocol import ChatCompletionRequest


def _tenant_of(request: Request) -> str:
    return request.scope.get("state", {}).get("tenant", "default")


def _not_found(request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": f"request {request_id!r} not found",
                "type": "invalid_request_error",
                "code": "request_not_found",
            }
        },
    )


def _receipt(request: AsyncRequest) -> dict:
    root = f"/v1/requests/{request.id}"
    return {
        "object": "async.request.receipt",
        "request": status_of(request).model_dump(mode="json"),
        "status_url": root,
        "result_url": f"{root}/result",
        "cancel_url": f"{root}/cancel",
    }


def _build_submission(
    request: ChatCompletionRequest,
    *,
    owner: str,
    priority: int,
    idempotency_key: str | None,
    deadline_at: datetime | None,
) -> AsyncRequestSubmission:
    return AsyncRequestSubmission(
        owner=owner,
        endpoint="/v1/chat/completions",
        body=request.model_dump(mode="json"),
        priority=priority,
        idempotency_key=idempotency_key,
        metadata={"model": request.model},
        deadline_at=deadline_at,
    )


def _encode_json(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def add_async_request_routes(
    app: FastAPI,
    store: RequestStoreProtocol,
    worker: AsyncRequestWorker,
) -> None:
    metrics = getattr(app.state, "metrics", None)
    if metrics is not None:
        metrics.track_async_request_store(store)

    async def store_call(function, *args, **kwargs):
        return await asyncio.to_thread(function, *args, **kwargs)

    @app.post("/v1/async/chat/completions", status_code=202)
    async def create_async_chat_completion(
        request: ChatCompletionRequest,
        http_request: Request,
        idempotency_key: Annotated[
            str | None,
            Header(alias="Idempotency-Key", min_length=1, max_length=255),
        ] = None,
        deadline_at: Annotated[
            datetime | None,
            Header(alias="X-Kairyu-Deadline-At"),
        ] = None,
    ):
        if not worker.supports_model(request.model):
            return model_not_found(request.model)
        if request.stream:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "async chat completions do not support stream=true",
                        "type": "invalid_request_error",
                        "code": "stream_not_supported",
                    }
                },
            )
        if deadline_at is not None and (
            deadline_at.tzinfo is None or deadline_at.utcoffset() is None
        ):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "X-Kairyu-Deadline-At must include a timezone",
                        "type": "invalid_request_error",
                        "code": "invalid_deadline",
                    }
                },
            )
        try:
            owner = _tenant_of(http_request)
            submission = await asyncio.to_thread(
                _build_submission,
                request,
                owner=owner,
                priority=worker.queue_priority(owner),
                idempotency_key=idempotency_key,
                deadline_at=deadline_at,
            )
            created = await store_call(store.submit, submission)
        except ValidationError:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "async request metadata failed validation",
                        "type": "invalid_request_error",
                        "code": "invalid_async_request",
                    }
                },
            )
        except IdempotencyConflictError:
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "message": "idempotency key was already used with different request data",
                        "type": "invalid_request_error",
                        "code": "idempotency_conflict",
                    }
                },
            )
        except RequestCapacityError:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "tenant durable request capacity is exhausted",
                        "type": "rate_limit_error",
                        "code": "request_capacity_exhausted",
                    }
                },
                headers={"Retry-After": "60"},
            )
        worker.submit(created.id)
        return JSONResponse(
            status_code=202,
            content=_receipt(created),
            headers={"Location": f"/v1/requests/{created.id}"},
        )

    @app.get("/v1/requests")
    async def list_async_requests(
        http_request: Request,
        state: AsyncRequestState | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 20,
    ):
        requests = await store_call(
            store.list_statuses,
            owner=_tenant_of(http_request),
            state=state,
            limit=limit,
        )
        return {
            "object": "list",
            "data": [request.model_dump(mode="json") for request in requests],
        }

    @app.get("/v1/requests/{request_id}")
    async def get_async_request(http_request: Request, request_id: str):
        try:
            return await store_call(
                store.get_status,
                request_id,
                owner=_tenant_of(http_request),
            )
        except KeyError:
            return _not_found(request_id)

    @app.get("/v1/requests/{request_id}/result")
    async def get_async_request_result(http_request: Request, request_id: str):
        try:
            status, result = await store_call(
                store.get_result,
                request_id,
                owner=_tenant_of(http_request),
            )
        except KeyError:
            return _not_found(request_id)
        if status.state is AsyncRequestState.SUCCEEDED:
            encoded = await asyncio.to_thread(_encode_json, result)
            return Response(content=encoded, media_type="application/json")
        if status.state not in {
            AsyncRequestState.FAILED,
            AsyncRequestState.CANCELLED,
            AsyncRequestState.EXPIRED,
        }:
            return JSONResponse(
                status_code=202,
                content=status.model_dump(mode="json"),
                headers={"Retry-After": "1"},
            )
        error = (
            status.error.model_dump(mode="json")
            if status.error is not None
            else {
                "code": f"request_{status.state.value}",
                "message": f"request is {status.state.value}",
                "retryable": False,
            }
        )
        return JSONResponse(
            status_code=409,
            content={
                "error": error,
                "request": status.model_dump(mode="json"),
            },
        )

    @app.post("/v1/requests/{request_id}/cancel")
    async def cancel_async_request(http_request: Request, request_id: str):
        try:
            cancelled = await store_call(
                store.cancel,
                request_id,
                owner=_tenant_of(http_request),
            )
        except KeyError:
            return _not_found(request_id)
        worker.notify_cancel(request_id)
        return status_of(cancelled)
