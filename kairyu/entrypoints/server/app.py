"""OpenAI-compatible FastAPI app: /v1/models, /v1/chat/completions (+SSE, tools).

Model name ``kairyu-auto`` routes the request through the Orchestrator behind
the same endpoint (design doc D6).
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import email.message
import inspect
import json
import logging
import math
import threading
import time
import uuid
import weakref
from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import EndpointContext, RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from pydantic import BaseModel
from starlette.exceptions import HTTPException

from kairyu.async_thread import (
    run_prompt_work,
    run_request_body_work,
    run_serialized_prompt_work,
)
from kairyu.engine.backend import (
    AdmissionUpperBound,
    CacheHint,
    EngineBackend,
    GenerationRequest,
    GenerationResult,
    GenerationStageMetric,
    GenerationUsage,
    UpstreamClientError,
    backend_admission_upper_bound_async,
    backend_max_model_len,
    backend_sequence_budget,
    backend_supports_slo_defer,
    prepare_backend_request,
    validate_backend_request_before_prepare,
)
from kairyu.engine.prompt import PromptInput, TokensPrompt
from kairyu.entrypoints.chat_template import ChatTemplate
from kairyu.entrypoints.server.chat_service import (
    ChatRequestError,
    ReasoningDeltaParser,
    _logprob_entries,
    _wire_usage,
    chat_error_from_upstream_client_error,
    completion_response,
    execute_chat,
    parallel_tool_calls_is_satisfied,
    sampling_params_from,
    tool_choice_is_satisfied,
    validate_chat_input_async,
    validate_chat_policy,
    validate_chat_request_async,
    validate_orchestration_chat_input_async,
)
from kairyu.entrypoints.server.chat_service import (
    render_prompt as render_prompt,
)
from kairyu.entrypoints.server.errors import (
    invalid_request,
    model_not_found,
    sanitize_backend_error,
    upstream_error,
)
from kairyu.entrypoints.server.health import add_health_routes
from kairyu.entrypoints.server.metering import (
    StreamUsageOwner,
    record_state_usage,
    resolve_cached_tokens,
    resolve_usage_counts,
    stream_usage_owner_from_state,
)
from kairyu.entrypoints.server.metrics import ServerMetrics
from kairyu.entrypoints.server.middleware import (
    _ANTHROPIC_INTERNAL_TOOL_STREAM_STATE_KEY,
    _SLO_ADMISSION_LEASE_STATE_KEY,
    AccessLogMiddleware,
    AuthMiddleware,
    ChatBodyLimitMiddleware,
    ConcurrencyLimitMiddleware,
    MetricsMiddleware,
    RequestIngressMiddleware,
)
from kairyu.entrypoints.server.protocol import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceLogprobs,
    ChunkChoice,
    ChunkDelta,
    ChunkToolCall,
    CompletionChoice,
    CompletionChunk,
    CompletionLogprobs,
    CompletionRequest,
    CompletionResponse,
    LogLikelihoodCompletionChoice,
    LogLikelihoodCompletionResponse,
    ModelCard,
    ModelList,
    PromptTokensDetails,
    RouteDecisionPayload,
    RoutePreviewRequest,
    RoutePreviewResponse,
    RoutingResponse,
    Usage,
)
from kairyu.entrypoints.server.request_body import reuse_prevalidated_model
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.slo import AdmissionController, AdmissionLease
from kairyu.entrypoints.server.sse_encode import (
    ChatContentSSEEncoder,
    CompletionTextSSEEncoder,
)
from kairyu.entrypoints.server.sse_response import sse_response
from kairyu.orchestration.orchestrator import (
    Orchestrator,
    OrchestratorExecutionError,
    PreviewNotSupportedError,
)
from kairyu.orchestration.replica import ReplicaPool
from kairyu.orchestration.request import OrchestrationRequest
from kairyu.orchestration.trace import StructuredTrace, TraceEvent, utc_now_iso
from kairyu.outputs import CompletionOutput, _lazy_text
from kairyu.pricing import InvoiceExportError, PriceSheet, export_invoice_csv
from kairyu.sampling_params import (
    GENERATION_CONFIG_SAMPLING_FIELDS,
    SamplingParams,
)
from kairyu.sse import escape_json_line_separators

if TYPE_CHECKING:
    from kairyu.entrypoints.server.extra_routes import EmbeddingBackend

logger = logging.getLogger(__name__)
_LOWEST_SCHEDULER_PRIORITY = 2**63 - 1
_SLO_INTERACTIVE_PRIORITY_CEILING = _LOWEST_SCHEDULER_PRIORITY - 1


@dataclass(frozen=True, slots=True)
class _PreparedRequestBody:
    """A body joined, decoded, and validated on the bounded ingress lane."""

    body: bytes
    value: BaseModel | None = None
    json_value: object = None
    error_response: Response | None = None
    validation_errors: list[dict[str, Any]] | None = None
    validation_body: object = None
    validation_exception: Exception | None = None
    body_parse_exception: Exception | None = None


def _validation_error_response(errors: list[dict[str, Any]]) -> Response:
    """Render FastAPI's standard 422 body without returning work to the loop."""

    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(errors)},
    )


def _prepare_request_body(
    chunks: tuple[bytes, ...],
    body_field: Any,
    *,
    parse_json: bool,
) -> _PreparedRequestBody:
    """Mirror FastAPI body parsing while keeping request-sized CPU off-loop."""

    body = b"".join(chunks)
    value: object = None
    if body:
        if parse_json:
            try:
                value = json.loads(body)
            except json.JSONDecodeError as error:
                errors = [
                    {
                        "type": "json_invalid",
                        "loc": ("body", error.pos),
                        "msg": "JSON decode error",
                        "input": {},
                        "ctx": {"error": error.msg},
                    }
                ]
                return _PreparedRequestBody(
                    body=body,
                    error_response=_validation_error_response(errors),
                    validation_errors=errors,
                    validation_body=error.doc,
                    validation_exception=error,
                )
            except Exception as error:
                return _PreparedRequestBody(
                    body=body,
                    body_parse_exception=error,
                )
        else:
            value = body

    if value is None:
        errors = [
            {
                "type": "missing",
                "loc": ("body",),
                "msg": "Field required",
                "input": None,
            }
        ]
        return _PreparedRequestBody(
            body=body,
            error_response=_validation_error_response(errors),
            validation_errors=errors,
            validation_body=None,
        )

    validated, errors = body_field.validate(value, {}, loc=("body",))
    if errors:
        return _PreparedRequestBody(
            body=body,
            error_response=_validation_error_response(errors),
            validation_errors=errors,
            validation_body=value,
        )
    if not isinstance(validated, BaseModel):
        raise RuntimeError("offloaded request body validator returned a non-model")
    return _PreparedRequestBody(
        body=body,
        value=validated,
        json_value=value,
    )


def _strict_content_type_enabled(value: object) -> bool:
    return bool(getattr(value, "value", value))


def _request_body_is_json(request: Request, *, strict: bool) -> bool:
    content_type = request.headers.get("content-type")
    if not content_type:
        return not strict
    message = email.message.Message()
    message["content-type"] = content_type
    if message.get_content_maintype() != "application":
        return False
    subtype = message.get_content_subtype()
    return subtype == "json" or subtype.endswith("+json")


def _endpoint_context(func: Any) -> EndpointContext:
    # FastAPI's _extract_endpoint_context caches by id(func) without holding a
    # reference, so a garbage-collected endpoint can poison the cache for a new
    # endpoint reusing the same address; compute from the live function instead.
    ctx = EndpointContext()
    try:
        if (source_file := inspect.getsourcefile(func)) is not None:
            ctx["file"] = source_file
        if (line_number := inspect.getsourcelines(func)[1]) is not None:
            ctx["line"] = line_number
        if (func_name := getattr(func, "__name__", None)) is not None:
            ctx["function"] = func_name
    except Exception:
        ctx = EndpointContext()
    return ctx


class _OffloadedRequestBodyRoute(APIRoute):
    """Pre-cache large chat/route models without changing their public schema."""

    def get_route_handler(self):
        original = super().get_route_handler()
        body_field = self.body_field
        dependant = self.dependant
        base_endpoint_ctx = _endpoint_context(dependant.call)
        body_model = (
            None
            if body_field is None or not dependant.body_params
            else dependant.body_params[0].field_info.annotation
        )
        supported = (
            body_field is not None
            and len(dependant.body_params) == 1
            and not self._embed_body_fields
            and body_model in {ChatCompletionRequest, RoutePreviewRequest}
        )
        strict = _strict_content_type_enabled(self.strict_content_type)

        async def route_handler(request: Request) -> Response:
            if not supported:
                return await original(request)

            chunks: list[bytes] = []
            try:
                async for chunk in request.stream():
                    chunks.append(chunk)
            except HTTPException:
                raise
            except Exception as error:
                raise HTTPException(
                    status_code=400,
                    detail="There was an error parsing the body",
                ) from error
            prepared = await run_request_body_work(
                _prepare_request_body,
                tuple(chunks),
                body_field,
                parse_json=_request_body_is_json(request, strict=strict),
            )
            # Preserve Starlette's body/json cache contract for FastAPI's
            # typed dependency path; the request-local validator reuses the
            # exact model without reconstructing its nested message graph.
            request._body = prepared.body
            if prepared.body_parse_exception is not None:
                parse_error = HTTPException(
                    status_code=400,
                    detail="There was an error parsing the body",
                )
                # FastAPI raises from inside the parse ``except`` block. The
                # worker boundary preserves both links for custom handlers.
                parse_error.__context__ = prepared.body_parse_exception
                raise parse_error from prepared.body_parse_exception
            if prepared.error_response is not None:
                handler = request.app.exception_handlers.get(RequestValidationError)
                if handler is request_validation_exception_handler:
                    return prepared.error_response
                assert prepared.validation_errors is not None
                endpoint_ctx = EndpointContext(base_endpoint_ctx)
                if dependant.path:
                    mount_path = request.scope.get("root_path", "").rstrip("/")
                    endpoint_ctx["path"] = f"{request.method} {mount_path}{dependant.path}"
                validation_error = RequestValidationError(
                    prepared.validation_errors,
                    body=prepared.validation_body,
                    endpoint_ctx=endpoint_ctx,
                )
                if prepared.validation_exception is not None:
                    validation_error.__context__ = prepared.validation_exception
                    raise validation_error from prepared.validation_exception
                raise validation_error
            assert prepared.value is not None
            request._json = prepared.json_value
            with reuse_prevalidated_model(
                prepared.json_value,
                prepared.value,
            ):
                return await original(request)

        return route_handler


_loglikelihood_gates: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[int, weakref.ReferenceType[asyncio.Lock]],
] = weakref.WeakKeyDictionary()
_loglikelihood_gates_lock = threading.Lock()


def _loglikelihood_gate_for_running_loop(engine: object) -> asyncio.Lock:
    """Return a loop-local gate for one compatibility backend tokenizer."""

    loop = asyncio.get_running_loop()
    key = id(engine)
    with _loglikelihood_gates_lock:
        loop_gates = _loglikelihood_gates.setdefault(loop, {})
        gate_ref = loop_gates.get(key)
        gate = gate_ref() if gate_ref is not None else None
        if gate is None:
            gate = asyncio.Lock()
            # Locks bind to a loop when contended; retaining only a weak value
            # prevents the global weak-key registry from retaining that loop.
            loop_gates[key] = weakref.ref(gate)
    return gate


AUTO_MODEL = "kairyu-auto"
_KAIRYU_CHAT_EXTENSION_FIELDS = frozenset(
    {
        "kairyu_trace",
        "kairyu_trace_v2",
        "kairyu_route",
    }
)
_USAGE_ORCHESTRATION_FIELDS = frozenset(
    {"orchestration_input_tokens", "orchestration_output_tokens"}
)
_USAGE_INPUT_FIELD = frozenset({"orchestration_input_tokens"})
_USAGE_OUTPUT_FIELD = frozenset({"orchestration_output_tokens"})
_CHAT_CHUNK_EXCLUDE_USAGE_FIELDS = _KAIRYU_CHAT_EXTENSION_FIELDS | frozenset({"usage"})
_COMPLETION_CHUNK_EXCLUDE_USAGE_FIELDS = frozenset({"usage"})


def _route_payload(decision) -> RouteDecisionPayload:
    return RouteDecisionPayload(
        target=decision.target,
        confidence=decision.confidence,
        reason=decision.reason,
        features=decision.features.as_dict(),
    )


def _unset_orchestration_usage_fields(usage: Usage) -> frozenset[str]:
    if usage.orchestration_input_tokens is None:
        if usage.orchestration_output_tokens is None:
            return _USAGE_ORCHESTRATION_FIELDS
        return _USAGE_INPUT_FIELD
    if usage.orchestration_output_tokens is None:
        return _USAGE_OUTPUT_FIELD
    return frozenset()


def _with_usage_exclude(
    exclude: frozenset[str] | set[str] | None,
    usage: Usage,
) -> frozenset[str] | set[str] | dict[str, object] | None:
    """Omit unset AUTO counters without a per-Usage serializer callback."""

    fields = _unset_orchestration_usage_fields(usage)
    if not fields or (exclude is not None and "usage" in exclude):
        return exclude
    nested: dict[str, object] = dict.fromkeys(exclude or (), True)
    nested["usage"] = fields
    return nested


def _chat_response_payload(response: ChatCompletionResponse) -> dict:
    """Preserve the OpenAI wire shape while omitting unset Kairyu extensions."""
    return response.model_dump(
        mode="json",
        exclude=_with_usage_exclude(
            _KAIRYU_CHAT_EXTENSION_FIELDS,
            response.usage,
        ),
    )


def _orchestration_wire_usage(
    prompt: str,
    completions: Sequence[CompletionOutput],
    *,
    prompt_tokens: int,
    completion_tokens: int,
    public_prompt_tokens: int | None = None,
    public_completion_tokens: int | None = None,
) -> Usage:
    """OpenAI usage describes the public request/response; AUTO totals are exact.

    Standard ``prompt_tokens``/``completion_tokens`` keep their documented
    OpenAI meaning (issue #496): the caller-visible prompt and completion,
    from backend truth when the public stages reported it and otherwise the
    m9 wire approximation. Cumulative internal-call totals — every stage,
    retry, verifier — stay in ``orchestration_input_tokens`` /
    ``orchestration_output_tokens`` (m11 #196) and in metering.
    """

    usage = _wire_usage(prompt, completions, None)
    if public_prompt_tokens is not None:
        usage.prompt_tokens = public_prompt_tokens
    if public_completion_tokens is not None:
        usage.completion_tokens = public_completion_tokens
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
    usage.orchestration_input_tokens = prompt_tokens
    usage.orchestration_output_tokens = completion_tokens
    return usage


def _with_usage_ledger_cleanup(lifespan):
    """Make the app-created ledger the outermost lifespan-owned resource."""

    @contextlib.asynccontextmanager
    async def wrapped(app: FastAPI):
        try:
            if lifespan is None:
                yield
            else:
                async with lifespan(app):
                    yield
        finally:
            ledger = getattr(app.state, "usage_ledger", None)
            if ledger is not None:
                ledger.close()

    return wrapped


def _validate_generation_request(
    engine: EngineBackend, request: GenerationRequest
) -> JSONResponse | None:
    try:
        validate_backend_request_before_prepare(engine, request)
    except ValueError as error:
        return invalid_request(str(error))
    return None


def _parse_trace_timestamp(value: str | None) -> datetime.datetime | None:
    if value is None:
        return None
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _conductor_stage_metrics_observer(metrics, model: str):
    """Feed conductor trace events into the per-stage histogram (issue #495)."""

    def observe(event) -> None:
        timing = getattr(event, "timing", None)
        started = _parse_trace_timestamp(getattr(timing, "started_at", None))
        completed = _parse_trace_timestamp(getattr(timing, "completed_at", None))
        if started is None or completed is None:
            return
        seconds = (completed - started).total_seconds()
        if seconds < 0:
            return
        metrics.record_conductor_stage(
            model,
            event.node,
            event.operation,
            event.status,
            seconds,
        )

    return observe


async def _watch_client_disconnect(http_request: Request) -> None:
    """Resolve when the HTTP client drops the connection."""

    while True:
        message = await http_request.receive()
        if message["type"] == "http.disconnect":
            return


async def _run_until_client_disconnect(
    http_request: Request,
    work: Awaitable[Any],
) -> tuple[Any, Response | None]:
    """Await ``work`` unless the client disconnects first (issue #495).

    A dropped client can no longer read the response, so the in-flight
    orchestration is cancelled to release replicas, the admission slot, and
    the tenant reservation immediately instead of running to completion.
    Returns ``(result, None)`` normally, or ``(None, response)`` after a
    disconnect-triggered cancellation; ``work``'s exceptions propagate.
    """

    work_task = asyncio.ensure_future(work)
    disconnect_task = asyncio.ensure_future(_watch_client_disconnect(http_request))
    try:
        done, _ = await asyncio.wait(
            (work_task, disconnect_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if work_task not in done:
            work_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await work_task
            logger.info("client disconnected; cancelled in-flight orchestration")
            return None, Response(status_code=499)
    finally:
        disconnect_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await disconnect_task
    return work_task.result(), None


def _tenant_reservation(http_request: Request):
    return getattr(http_request.state, "tenant_admission", None)


def _tenant_limit_response(tenant: str, reason: str) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": "1"},
        content={
            "error": {
                "message": (f"tenant {tenant!r} admission limit exceeded ({reason})"),
                "type": "rate_limit_error",
                "code": "tenant_rate_limited",
            }
        },
    )


def _slo_shed_response() -> JSONResponse:
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": "1"},
        content={
            "error": {
                "message": "predicted TTFT exceeds the configured SLO",
                "type": "rate_limit_error",
                "code": "slo_admission_shed",
            }
        },
    )


def _slo_defer_to_shed_response(
    http_request: Request,
    lease: AdmissionLease,
) -> JSONResponse:
    state = http_request.scope.setdefault("state", {})
    if state.get(_SLO_ADMISSION_LEASE_STATE_KEY) is lease:
        state.pop(_SLO_ADMISSION_LEASE_STATE_KEY)
    if lease.active:
        lease.completed()
    return _slo_shed_response()


def _reserve_tenant_work(
    http_request: Request,
    bound: AdmissionUpperBound,
) -> JSONResponse | None:
    admission = _tenant_reservation(http_request)
    if admission is None:
        return None
    tenant = getattr(http_request.state, "tenant", None) or "default"
    admitted = admission.reserve_tokens(
        bound.tokens,
        refundable_on_exact_usage=bound.refundable_on_exact_usage,
    )
    metrics = getattr(http_request.app.state, "metrics", None)
    if metrics is not None:
        metrics.record_tenant_admission(
            tenant,
            source="http",
            admitted=admitted,
            reason=admission.reason,
        )
    if admitted:
        http_request.state.tenant_metric_admitted = True
        return None
    return _tenant_limit_response(tenant, admission.reason)


def _mark_tenant_dispatched(http_request: Request) -> None:
    admission = _tenant_reservation(http_request)
    if admission is not None:
        admission.mark_dispatched()


def _stream_usage_owner(
    http_request: Request,
    model: str,
    prompt: PromptInput,
    *,
    usage_exact: bool = True,
) -> StreamUsageOwner:
    tenant = getattr(http_request.state, "tenant", None) or "default"
    return stream_usage_owner_from_state(
        http_request.app.state,
        tenant=tenant,
        model=model,
        prompt=prompt,
        reservation=_tenant_reservation(http_request),
        usage_exact=usage_exact,
    )


def _sse_chunk(
    response_id: str,
    created: int,
    model: str,
    index: int,
    delta: ChunkDelta,
    finish_reason: str | None = None,
    include_usage: bool = False,
    usage: Usage | None = None,
    logprobs: ChoiceLogprobs | None = None,
) -> str:
    payload = ChatCompletionChunk(
        id=response_id,
        created=created,
        model=model,
        choices=[
            ChunkChoice(index=index, delta=delta, finish_reason=finish_reason, logprobs=logprobs)
        ],
        usage=usage,
    )
    # OpenAI contract: usage key omitted unless include_usage; then explicit
    # null on non-final chunks, populated on the final choices-less chunk
    exclude = _KAIRYU_CHAT_EXTENSION_FIELDS if include_usage else _CHAT_CHUNK_EXCLUDE_USAGE_FIELDS
    if usage is not None:
        exclude = _with_usage_exclude(exclude, usage)
    serialized = escape_json_line_separators(payload.model_dump_json(exclude=exclude))
    return f"data: {serialized}\n\n"


def _chat_content_sse_chunk(
    encoder: ChatContentSSEEncoder,
    response_id: str,
    created: int,
    model: str,
    index: int,
    content: str,
    *,
    is_first: bool,
    include_usage: bool,
    logprobs: ChoiceLogprobs | None,
) -> str | bytes:
    """Use the byte encoder only for the exact repeated content shape."""

    if not is_first and type(index) is int and type(content) is str and logprobs is None:
        return encoder.encode(index, content)
    return _sse_chunk(
        response_id,
        created,
        model,
        index,
        ChunkDelta(role="assistant" if is_first else None, content=content),
        include_usage=include_usage,
        logprobs=logprobs,
    )


def _usage_chunk(response_id: str, created: int, model: str, usage: Usage) -> str:
    payload = ChatCompletionChunk(
        id=response_id, created=created, model=model, choices=[], usage=usage
    )
    serialized = escape_json_line_separators(
        payload.model_dump_json(
            exclude=_with_usage_exclude(
                _KAIRYU_CHAT_EXTENSION_FIELDS,
                usage,
            )
        )
    )
    return f"data: {serialized}\n\n"


def _orchestrator_metadata_chunk(
    response_id: str,
    created: int,
    model: str,
    *,
    usage: Usage | None,
    result,
    include_usage: bool,
    want_trace: bool,
) -> str | None:
    """One terminal AUTO chunk carrying requested usage and/or trace data."""

    if not include_usage and not want_trace:
        return None
    trace = (
        result.structured_trace.as_dict(request_id=response_id)
        if want_trace and result.structured_trace is not None
        else None
    )
    payload = ChatCompletionChunk(
        id=response_id,
        created=created,
        model=model,
        choices=[],
        usage=usage if include_usage else None,
        kairyu_trace=list(result.trace) if want_trace else None,
        kairyu_trace_v2=trace,
        kairyu_route=_route_payload(result.route) if want_trace else None,
    )
    exclude: set[str] = set()
    if not include_usage:
        exclude.add("usage")
    if not want_trace:
        exclude.update(_KAIRYU_CHAT_EXTENSION_FIELDS)
    elif trace is None:
        exclude.add("kairyu_trace_v2")
    if usage is not None:
        exclude = _with_usage_exclude(exclude, usage)
    serialized = escape_json_line_separators(payload.model_dump_json(exclude=exclude))
    return f"data: {serialized}\n\n"


def _direct_stage_trace(
    response_id: str,
    *,
    started_at: str,
    stage_metrics: Sequence[GenerationStageMetric],
) -> StructuredTrace:
    """Build the privacy-safe v2 envelope for one direct engine request."""

    stage_metrics = _merge_stage_metrics(stage_metrics)
    events = tuple(
        TraceEvent(
            node=metric.stage,
            kind="stage",
            operation="stage",
            status="success",
            role="engine",
            metadata={
                "stage": metric.stage,
                "duration_ns": metric.duration_ns,
                "occurrences": metric.occurrences,
                "aggregation": "sum",
                "scope": "request-observed",
            },
        )
        for metric in stage_metrics
    )
    return StructuredTrace(
        request_id=response_id,
        started_at=started_at,
        completed_at=utc_now_iso(),
        events=events,
    )


def _merge_stage_metrics(
    *groups: Sequence[GenerationStageMetric],
) -> tuple[GenerationStageMetric, ...]:
    """Combine same-stage observations across nested Kairyu transport hops."""

    order: list[str] = []
    totals: dict[str, tuple[int, int]] = {}
    for group in groups:
        for metric in group:
            if metric.stage not in totals:
                order.append(metric.stage)
                totals[metric.stage] = (0, 0)
            duration_ns, occurrences = totals[metric.stage]
            totals[metric.stage] = (
                duration_ns + metric.duration_ns,
                occurrences + metric.occurrences,
            )
    return tuple(
        GenerationStageMetric(
            stage=stage,
            duration_ns=totals[stage][0],
            occurrences=totals[stage][1],
        )
        for stage in order
    )


def _direct_trace_lines(
    stage_metrics: Sequence[GenerationStageMetric],
) -> list[str]:
    stage_metrics = _merge_stage_metrics(stage_metrics)
    return [
        f"stage:{metric.stage} duration_ns={metric.duration_ns} occurrences={metric.occurrences}"
        for metric in stage_metrics
    ]


def _direct_metadata_chunk(
    response_id: str,
    created: int,
    model: str,
    *,
    started_at: str,
    stage_metrics: Sequence[GenerationStageMetric],
) -> str:
    trace = _direct_stage_trace(
        response_id,
        started_at=started_at,
        stage_metrics=stage_metrics,
    ).as_dict()
    payload = ChatCompletionChunk(
        id=response_id,
        created=created,
        model=model,
        choices=[],
        kairyu_trace=_direct_trace_lines(stage_metrics),
        kairyu_trace_v2=trace,
    )
    serialized = escape_json_line_separators(
        payload.model_dump_json(exclude={"usage", "kairyu_route"})
    )
    return f"data: {serialized}\n\n"


async def _stream_engine(
    engine: EngineBackend,
    generation_request: GenerationRequest,
    model: str,
    request: ChatCompletionRequest,
    http_request: Request,
    *,
    trace_started_at: str | None = None,
) -> AsyncIterator[str | bytes]:
    response_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    include_usage = bool(request.stream_options and request.stream_options.include_usage)
    content_encoder = ChatContentSSEEncoder(
        response_id,
        created,
        model,
        include_usage=include_usage,
    )
    sent: dict[int, int] = {}
    reasoning_sent: dict[int, int] = {}
    wire_started: set[int] = set()
    logprobs_sent: dict[int, int] = {}
    last = None
    sse_write_ns = 0
    sse_write_count = 0
    first_token_observed = False
    want_trace = generation_request.trace_requested
    if want_trace and trace_started_at is None:
        trace_started_at = utc_now_iso()

    reasoning_parsers: dict[int, ReasoningDeltaParser] = {}

    def stage_metrics() -> tuple[GenerationStageMetric, ...]:
        metrics = tuple(last.stage_metrics) if last is not None else ()
        if sse_write_count:
            metrics = _merge_stage_metrics(
                metrics,
                (
                    GenerationStageMetric(
                        stage="sse_write",
                        duration_ns=sse_write_ns,
                        occurrences=sse_write_count,
                    ),
                ),
            )
        return metrics

    owner = _stream_usage_owner(http_request, model, generation_request.prompt)
    try:
        try:
            owner.mark_dispatched()
            async for partial in engine.stream(generation_request):
                last = partial
                owner.observe(partial.usage, partial.completions)
                for completion in partial.completions:
                    previous_offset = sent.get(completion.index, 0)
                    delta_text, next_offset = completion.delta_after(previous_offset)
                    remote_reasoning, next_reasoning_offset = completion.reasoning_delta_after(
                        reasoning_sent.get(completion.index, 0)
                    )
                    reasoning_sent[completion.index] = next_reasoning_offset
                    if request.reasoning_effort is not None and not remote_reasoning:
                        parser = reasoning_parsers.setdefault(
                            completion.index, ReasoningDeltaParser()
                        )
                        parsed_reasoning, delta_text = parser.feed(
                            delta_text,
                            final=partial.finished,
                        )
                    else:
                        parsed_reasoning = remote_reasoning
                    if not delta_text and not parsed_reasoning and not partial.finished:
                        continue
                    sent[completion.index] = next_offset
                    chunk_logprobs = None
                    if request.logprobs and completion.logprob_content is not None:
                        seen = logprobs_sent.get(completion.index, 0)
                        fresh = completion.logprob_content[seen:]
                        logprobs_sent[completion.index] = len(completion.logprob_content)
                        if fresh:
                            chunk_logprobs = ChoiceLogprobs(content=_logprob_entries(fresh))
                    write_started_ns = time.perf_counter_ns() if want_trace else None
                    wrote_chunk = False
                    if parsed_reasoning:
                        is_first = completion.index not in wire_started
                        yield _sse_chunk(
                            response_id,
                            created,
                            model,
                            completion.index,
                            ChunkDelta(
                                role="assistant" if is_first else None,
                                reasoning_content=parsed_reasoning,
                            ),
                            include_usage=include_usage,
                        )
                        wrote_chunk = True
                        wire_started.add(completion.index)
                    if delta_text or chunk_logprobs is not None:
                        is_first = completion.index not in wire_started
                        chunk = _chat_content_sse_chunk(
                            content_encoder,
                            response_id,
                            created,
                            model,
                            completion.index,
                            delta_text,
                            is_first=is_first,
                            include_usage=include_usage,
                            logprobs=chunk_logprobs,
                        )
                        yield chunk
                        wrote_chunk = True
                        wire_started.add(completion.index)
                    if (delta_text or parsed_reasoning) and not first_token_observed:
                        lease = getattr(
                            http_request.state,
                            _SLO_ADMISSION_LEASE_STATE_KEY,
                            None,
                        )
                        if lease is not None:
                            lease.finished_first_token()
                        first_token_observed = True
                    if write_started_ns is not None and wrote_chunk:
                        sse_write_ns += time.perf_counter_ns() - write_started_ns
                        sse_write_count += 1
        except Exception as error:  # surface backend failures inside the SSE stream
            logger.exception("upstream backend error")
            if want_trace:
                assert trace_started_at is not None
                yield _direct_metadata_chunk(
                    response_id,
                    created,
                    model,
                    started_at=trace_started_at,
                    stage_metrics=stage_metrics(),
                )
            payload = {  # M3: only the class name, no raw backend message
                "error": {
                    "message": f"upstream backend error ({type(error).__name__})",
                    "type": "upstream_error",
                }
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
            return
        owner.mark_completed()
        for completion in last.completions if last else ():
            write_started_ns = time.perf_counter_ns() if want_trace else None
            yield _sse_chunk(
                response_id,
                created,
                model,
                completion.index,
                ChunkDelta(),
                finish_reason=completion.finish_reason or "stop",
                include_usage=include_usage,
            )
            if write_started_ns is not None:
                sse_write_ns += time.perf_counter_ns() - write_started_ns
                sse_write_count += 1
        if include_usage and last is not None:
            write_started_ns = time.perf_counter_ns() if want_trace else None
            yield _usage_chunk(
                response_id,
                created,
                model,
                _wire_usage(
                    generation_request.prompt,
                    last.completions,
                    owner.latest_usage,
                ),
            )
            if write_started_ns is not None:
                sse_write_ns += time.perf_counter_ns() - write_started_ns
                sse_write_count += 1
        if want_trace:
            assert trace_started_at is not None
            yield _direct_metadata_chunk(
                response_id,
                created,
                model,
                started_at=trace_started_at,
                stage_metrics=stage_metrics(),
            )
        yield "data: [DONE]\n\n"
    finally:
        owner.finalize()


async def _stream_orchestrator(
    orchestrator,
    call: OrchestrationRequest | str,
    prepared,
    request: ChatCompletionRequest,
    include_usage: bool,
    want_trace: bool,
    http_request: Request,
) -> AsyncIterator[str | bytes]:
    """AUTO-model SSE (m11 D1/A2): status keep-alives ride SSE COMMENT lines
    (the OpenAI SDK parses every data: payload as a chunk), deltas and the
    final chunk are standard chat chunks."""
    if isinstance(call, str):
        call = OrchestrationRequest(
            prompt=call,
            sampling_params=sampling_params_from(request),
            tools=tuple(request.tools or ()),
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            response_format=request.response_format,
            tool_call_protocol="generic",
            reasoning_effort=request.reasoning_effort,
        )
    response_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    content_encoder = ChatContentSSEEncoder(
        response_id,
        created,
        request.model,
        include_usage=include_usage,
    )
    started_choices: set[int] = set()
    sent: dict[int, int] = {}
    logprobs_sent: dict[int, int] = {}
    final_result = None
    completion_text = ""
    completion_parts: list[str] = []
    completion_length = 0
    completions: tuple[CompletionOutput, ...] = ()
    reported_usage: GenerationUsage | None = None
    terminal_error_type: str | None = None
    prompt = call.prompt
    owner = _stream_usage_owner(
        http_request,
        request.model,
        prompt,
        usage_exact=False,
    )

    def observe_internal_usage(usage: GenerationUsage) -> None:
        owner.observe(usage, completions)

    def fresh_chunks(
        snapshot: Sequence[CompletionOutput],
    ):
        for completion in sorted(snapshot, key=lambda item: item.index):
            seen_text = sent.get(completion.index, 0)
            delta_text, next_text = completion.delta_after(seen_text)
            seen_logprobs = logprobs_sent.get(completion.index, 0)
            fresh_logprobs = (
                completion.logprob_content[seen_logprobs:]
                if completion.logprob_content is not None
                else ()
            )
            if not delta_text and not fresh_logprobs and completion.index in sent:
                continue
            is_first = completion.index not in started_choices
            started_choices.add(completion.index)
            sent[completion.index] = next_text
            if completion.logprob_content is not None:
                logprobs_sent[completion.index] = len(completion.logprob_content)
            yield (
                completion.index,
                delta_text,
                is_first,
                (
                    ChoiceLogprobs(content=_logprob_entries(fresh_logprobs))
                    if fresh_logprobs
                    else None
                ),
            )

    try:
        try:
            owner.mark_dispatched()
            stream = await orchestrator.run_chat(
                call,
                stream=True,
                usage_observer=observe_internal_usage,
                prepared=prepared,
            )
            async for event in stream:
                if event.kind == "status":
                    yield f": status {event.text}\n\n"  # SSE comment (A2)
                elif event.kind == "reasoning":
                    if not event.text:
                        continue
                    is_first = 0 not in started_choices
                    started_choices.add(0)
                    yield _sse_chunk(
                        response_id,
                        created,
                        request.model,
                        0,
                        ChunkDelta(
                            role="assistant" if is_first else None,
                            reasoning_content=event.text,
                        ),
                        include_usage=include_usage,
                    )
                elif event.kind == "delta":
                    if event.completions:
                        completions = event.completions
                        owner.observe(None, completions)
                        for index, delta_text, is_first, chunk_logprobs in fresh_chunks(
                            completions
                        ):
                            yield _chat_content_sse_chunk(
                                content_encoder,
                                response_id,
                                created,
                                request.model,
                                index,
                                delta_text,
                                is_first=is_first,
                                include_usage=include_usage,
                                logprobs=chunk_logprobs,
                            )
                    else:
                        completion_parts.append(event.text)
                        completions = (
                            CompletionOutput(
                                index=0,
                                text=_lazy_text(completion_parts),
                                token_ids=(),
                                finish_reason=None,
                                text_delta=event.text,
                                text_offset=completion_length,
                            ),
                        )
                        completion_length += len(event.text)
                        # Keep the per-choice offset in step so the final
                        # result's cumulative completion emits only a genuine
                        # tail, never the whole already-streamed text again.
                        sent[0] = completion_length
                        owner.observe(None, completions)
                        is_first = 0 not in started_choices
                        started_choices.add(0)
                        yield _chat_content_sse_chunk(
                            content_encoder,
                            response_id,
                            created,
                            request.model,
                            0,
                            event.text,
                            is_first=is_first,
                            include_usage=include_usage,
                            logprobs=None,
                        )
                elif event.kind in {"result", "error"}:
                    final_result = event.result
                    if final_result is not None:
                        completion_text = final_result.text or "".join(completion_parts)
                        completions = final_result.completions or (
                            CompletionOutput(
                                index=0,
                                text=completion_text,
                                token_ids=(),
                                finish_reason="stop",
                            ),
                        )
                        for index, delta_text, is_first, chunk_logprobs in fresh_chunks(
                            completions
                        ):
                            yield _chat_content_sse_chunk(
                                content_encoder,
                                response_id,
                                created,
                                request.model,
                                index,
                                delta_text,
                                is_first=is_first,
                                include_usage=include_usage,
                                logprobs=chunk_logprobs,
                            )
                        if final_result.prompt_tokens or final_result.completion_tokens:
                            reported_usage = GenerationUsage(
                                prompt_tokens=final_result.prompt_tokens,
                                completion_tokens=final_result.completion_tokens,
                                cached_tokens=final_result.cached_tokens,
                            )
                        owner.observe(reported_usage, completions)
                    if event.kind == "error":
                        terminal_error_type = event.error_type or "RuntimeError"
                        break
        except Exception as error:  # surface as an SSE error event, then close
            logger.exception("orchestrator stream error")
            payload = {"error": {"message": type(error).__name__}}
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
            return
        if terminal_error_type is not None and final_result is not None:
            usage = _orchestration_wire_usage(
                prompt,
                completions,
                prompt_tokens=final_result.prompt_tokens,
                completion_tokens=final_result.completion_tokens,
                public_prompt_tokens=final_result.public_prompt_tokens,
                public_completion_tokens=final_result.public_completion_tokens,
            )
            metadata = _orchestrator_metadata_chunk(
                response_id,
                created,
                request.model,
                usage=usage,
                result=final_result,
                include_usage=include_usage,
                want_trace=want_trace,
            )
            if metadata is not None:
                yield metadata
            if want_trace:
                yield f": trace {' | '.join(final_result.trace)}\n\n"
            safe_type = (
                terminal_error_type if terminal_error_type.isidentifier() else "RuntimeError"
            )
            payload = {
                "error": {
                    "message": f"upstream backend error ({safe_type})",
                    "type": "upstream_error",
                    "code": "backend_error",
                }
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
            return
        final_completions = completions or (
            CompletionOutput(index=0, text="", token_ids=(), finish_reason="stop"),
        )
        for completion in sorted(final_completions, key=lambda item: item.index):
            yield _sse_chunk(
                response_id,
                created,
                request.model,
                completion.index,
                ChunkDelta(),
                finish_reason=completion.finish_reason or "stop",
                include_usage=include_usage,
            )
        if final_result is not None:
            usage = _orchestration_wire_usage(
                prompt,
                completions,
                prompt_tokens=final_result.prompt_tokens,
                completion_tokens=final_result.completion_tokens,
                public_prompt_tokens=final_result.public_prompt_tokens,
                public_completion_tokens=final_result.public_completion_tokens,
            )
            metadata = _orchestrator_metadata_chunk(
                response_id,
                created,
                request.model,
                usage=usage,
                result=final_result,
                include_usage=include_usage,
                want_trace=want_trace,
            )
            if metadata is not None:
                yield metadata
            if want_trace:
                yield f": trace {' | '.join(final_result.trace)}\n\n"
        if terminal_error_type is None:
            owner.mark_completed()
        yield "data: [DONE]\n\n"
    finally:
        owner.finalize()


async def _stream_choices(
    choices: list[Choice],
    model: str,
    usage: Usage | None = None,
    *,
    orchestration_result=None,
    stage_metrics: Sequence[GenerationStageMetric] = (),
    trace_started_at: str | None = None,
    want_trace: bool = False,
    slo_lease: AdmissionLease | None = None,
) -> AsyncIterator[str]:
    """Stream already-final choices (orchestrated or tool-call responses)."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    include_usage = usage is not None
    direct_trace = want_trace and orchestration_result is None and trace_started_at is not None
    sse_write_ns = 0
    sse_write_count = 0
    first_token_observed = False
    for choice in choices:
        tool_calls = None
        if choice.message.tool_calls:
            # attach the required per-item index so SDK stream accumulators merge
            # the tool-call fragments correctly (S6)
            tool_calls = [
                ChunkToolCall(index=i, id=tc.id, type=tc.type, function=tc.function)
                for i, tc in enumerate(choice.message.tool_calls)
            ]
        write_started_ns = time.perf_counter_ns() if direct_trace else None
        chunk = _sse_chunk(
            response_id,
            created,
            model,
            choice.index,
            ChunkDelta(
                role="assistant",
                content=choice.message.content,
                reasoning_content=choice.message.reasoning_content,
                tool_calls=tool_calls,
            ),
            include_usage=include_usage,
            logprobs=choice.logprobs,
        )
        yield chunk
        if (
            slo_lease is not None
            and not first_token_observed
            and (choice.message.content or tool_calls)
        ):
            slo_lease.finished_first_token()
            first_token_observed = True
        if write_started_ns is not None:
            sse_write_ns += time.perf_counter_ns() - write_started_ns
            sse_write_count += 1
        write_started_ns = time.perf_counter_ns() if direct_trace else None
        yield _sse_chunk(
            response_id,
            created,
            model,
            choice.index,
            ChunkDelta(),
            finish_reason=choice.finish_reason,
            include_usage=include_usage,
        )
        if write_started_ns is not None:
            sse_write_ns += time.perf_counter_ns() - write_started_ns
            sse_write_count += 1
    if usage is not None:
        write_started_ns = time.perf_counter_ns() if direct_trace else None
        yield _usage_chunk(response_id, created, model, usage)
        if write_started_ns is not None:
            sse_write_ns += time.perf_counter_ns() - write_started_ns
            sse_write_count += 1
    if orchestration_result is not None and want_trace:
        metadata = _orchestrator_metadata_chunk(
            response_id,
            created,
            model,
            usage=None,
            result=orchestration_result,
            include_usage=False,
            want_trace=True,
        )
        if metadata is not None:
            yield metadata
        yield f": trace {' | '.join(orchestration_result.trace)}\n\n"
    elif direct_trace:
        direct_metrics = tuple(stage_metrics)
        if sse_write_count:
            direct_metrics = _merge_stage_metrics(
                direct_metrics,
                (
                    GenerationStageMetric(
                        stage="sse_write",
                        duration_ns=sse_write_ns,
                        occurrences=sse_write_count,
                    ),
                ),
            )
        assert trace_started_at is not None
        yield _direct_metadata_chunk(
            response_id,
            created,
            model,
            started_at=trace_started_at,
            stage_metrics=direct_metrics,
        )
    yield "data: [DONE]\n\n"


def _completion_logprobs(completion: CompletionOutput) -> CompletionLogprobs | None:
    """Legacy four-array shape; offsets follow decoded token contributions."""
    if completion.logprob_content is None:
        return None
    tokens: list[str] = []
    token_logprobs: list[float] = []
    top_logprobs: list[dict[str, float]] = []
    text_offset: list[int] = []
    offset = 0
    has_top = False
    for entry in completion.logprob_content:
        tokens.append(entry.token)
        token_logprobs.append(entry.logprob)
        top_logprobs.append({top.token: top.logprob for top in entry.top})
        has_top = has_top or bool(entry.top)
        text_offset.append(offset)
        # ``entry.token`` is the raw vocabulary piece and can contain decoder
        # notation (``##``, ``<0xNN>``) or a skipped special-token literal.
        # Retain the pre-#362 offset basis using the decoded UTF-8 contribution
        # when it is available; malformed third-party bytes fall back safely.
        try:
            contribution = (
                bytes(entry.bytes_).decode("utf-8") if entry.bytes_ is not None else entry.token
            )
        except (TypeError, UnicodeDecodeError, ValueError):
            contribution = entry.token
        offset += len(contribution)
    return CompletionLogprobs(
        tokens=tokens,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs if has_top else None,
        text_offset=text_offset,
    )


def _completion_choice(index: int, completion: CompletionOutput) -> CompletionChoice:
    return CompletionChoice(
        index=index,
        text=completion.text,
        logprobs=_completion_logprobs(completion),
        finish_reason=completion.finish_reason or "stop",
    )


def _loglikelihood_request_error(request: CompletionRequest) -> str | None:
    """Validate the deliberately narrow continuation-scoring wire contract."""

    if type(request.prompt) is not str or not request.prompt:
        return "kairyu_continuation requires one non-empty string prompt"
    if not request.kairyu_continuation:
        return "kairyu_continuation must be a non-empty string"
    if request.stream:
        return "kairyu_continuation does not support streaming"
    if request.n != 1:
        return "kairyu_continuation requires n=1"
    if "max_tokens" not in request.model_fields_set or request.max_tokens is not None:
        return "kairyu_continuation requires max_tokens=null"
    if request.logprobs != 0:
        return "kairyu_continuation requires logprobs=0"
    if request.temperature != 0.0:
        return "kairyu_continuation requires temperature=0"
    if request.seed is not None:
        return "kairyu_continuation does not support seed"
    if "ignore_eos" in request.model_fields_set:
        return "kairyu_continuation controls ignore_eos internally"
    if "skip_special_tokens" in request.model_fields_set:
        return "kairyu_continuation controls skip_special_tokens internally"

    neutral_fields = (
        ("top_p", request.top_p, 1.0),
        ("top_k", request.top_k, -1),
        ("min_p", request.min_p, 0.0),
        ("min_tokens", request.min_tokens, 0),
        ("presence_penalty", request.presence_penalty, 0.0),
        ("frequency_penalty", request.frequency_penalty, 0.0),
        ("repetition_penalty", request.repetition_penalty, 1.0),
    )
    for name, actual, expected in neutral_fields:
        if actual != expected:
            return f"kairyu_continuation requires {name}={expected}"
    if request.stop is not None:
        return "kairyu_continuation does not support stop"
    if request.stop_token_ids is not None:
        return "kairyu_continuation does not support stop_token_ids"
    return None


def _loglikelihood_token_ids(
    engine: EngineBackend,
    context: str,
    continuation: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Call a tokenizer-owned boundary check and validate its typed evidence."""

    tokenize = getattr(engine, "tokenize_loglikelihood", None)
    if not callable(tokenize):
        raise AttributeError("tokenize_loglikelihood")
    return _validated_loglikelihood_token_ids(tokenize(context, continuation))


def _validated_loglikelihood_token_ids(
    tokenized: object,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Copy and validate tokenizer-owned boundary evidence."""

    if type(tokenized) is not tuple or len(tokenized) != 2:
        raise RuntimeError("tokenize_loglikelihood must return a pair of token-ID sequences")

    validated: list[tuple[int, ...]] = []
    for label, values in zip(("context", "continuation"), tokenized, strict=True):
        if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
            raise RuntimeError(f"tokenize_loglikelihood returned invalid {label} token IDs")
        copied = tuple(values)
        if not copied:
            raise RuntimeError(f"tokenize_loglikelihood returned empty {label} token IDs")
        if any(
            type(token_id) is not int or not 0 <= token_id <= (1 << 64) - 1 for token_id in copied
        ):
            raise RuntimeError(f"tokenize_loglikelihood returned invalid {label} token IDs")
        validated.append(copied)
    return validated[0], validated[1]


async def _loglikelihood_token_ids_async(
    engine: EngineBackend,
    context: str,
    continuation: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Use a backend tokenizer gate when one is available."""

    tokenize_async = getattr(engine, "tokenize_loglikelihood_async", None)
    if callable(tokenize_async):
        tokenized = await tokenize_async(context, continuation)
        return _validated_loglikelihood_token_ids(tokenized)
    if getattr(engine, "concurrent_tokenize_loglikelihood_safe", False) is True:
        return await run_prompt_work(
            _loglikelihood_token_ids,
            engine,
            context,
            continuation,
        )
    # The compatibility Protocol never promised a thread-safe tokenizer. Queue
    # before entering the bounded executor so waiters cannot consume its lanes.
    return await run_serialized_prompt_work(
        _loglikelihood_gate_for_running_loop(engine),
        None,
        _loglikelihood_token_ids,
        engine,
        context,
        continuation,
    )


def _loglikelihood_choice(
    result: GenerationResult,
    *,
    context_ids: tuple[int, ...],
    continuation: str,
    continuation_ids: tuple[int, ...],
) -> LogLikelihoodCompletionChoice:
    """Build a scoring choice only from exact, finite forced-token evidence."""

    returned_prompt_ids = result.prompt_token_ids
    if isinstance(returned_prompt_ids, (str, bytes, bytearray)) or not isinstance(
        returned_prompt_ids, Sequence
    ):
        raise RuntimeError("loglikelihood backend returned invalid prompt token IDs")
    returned_prompt_ids = tuple(returned_prompt_ids)
    if any(type(token_id) is not int for token_id in returned_prompt_ids):
        raise RuntimeError("loglikelihood backend returned invalid prompt token IDs")
    if returned_prompt_ids != context_ids:
        raise RuntimeError("loglikelihood backend did not process the prompt token IDs")

    result_completions = result.completions
    if len(result_completions) != 1:
        raise RuntimeError("loglikelihood backend returned the wrong choice count")
    completion = result_completions[0]
    if type(completion.index) is not int or completion.index != 0:
        raise RuntimeError("loglikelihood backend returned a non-zero choice index")
    returned_token_ids = completion.token_ids
    if isinstance(returned_token_ids, (str, bytes, bytearray)) or not isinstance(
        returned_token_ids, Sequence
    ):
        raise RuntimeError("loglikelihood backend returned invalid forced token IDs")
    returned_token_ids = tuple(returned_token_ids)
    if any(type(token_id) is not int for token_id in returned_token_ids):
        raise RuntimeError("loglikelihood backend returned invalid forced token IDs")
    if returned_token_ids != continuation_ids:
        raise RuntimeError("loglikelihood backend did not return the forced token IDs")
    if completion.finish_reason != "length":
        raise RuntimeError("loglikelihood backend did not finish the complete forced continuation")
    content = completion.logprob_content
    if content is None or len(content) != len(continuation_ids):
        raise RuntimeError("loglikelihood backend returned missing token logprobs")
    for expected_id, entry in zip(continuation_ids, content, strict=True):
        if type(entry.token_id) is not int or entry.token_id != expected_id:
            raise RuntimeError("loglikelihood backend returned misaligned token logprobs")
        value = entry.logprob
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError("loglikelihood backend returned an invalid token logprob")
        try:
            parsed = float(value)
        except (OverflowError, ValueError) as error:
            raise RuntimeError("loglikelihood backend returned an invalid token logprob") from error
        if not math.isfinite(parsed) or parsed > 0.0:
            raise RuntimeError("loglikelihood backend returned an invalid token logprob")

    logprobs = _completion_logprobs(completion)
    if logprobs is None:
        raise RuntimeError("loglikelihood backend returned missing token logprobs")
    return LogLikelihoodCompletionChoice(
        index=0,
        # This is the caller-supplied continuation, not a claim that separately
        # detokenizing its IDs is context independent (e.g. WordPiece is not).
        text=continuation,
        logprobs=logprobs,
        finish_reason="length",
        prompt_token_ids=list(context_ids),
        continuation_token_ids=list(continuation_ids),
    )


async def _stream_completions(
    engine: EngineBackend,
    generation_request: GenerationRequest,
    request: CompletionRequest,
    http_request: Request,
) -> AsyncIterator[str | bytes]:
    """Legacy text_completion stream: cumulative text deltas, not delta objects."""
    response_id = f"cmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    include_usage = bool(request.stream_options and request.stream_options.include_usage)
    text_encoder = CompletionTextSSEEncoder(
        response_id,
        created,
        request.model,
        include_usage=include_usage,
    )
    sent: dict[int, int] = {}
    last = None
    owner = _stream_usage_owner(http_request, request.model, generation_request.prompt)

    def _chunk(choices: list[CompletionChoice], usage: Usage | None = None) -> str:
        payload = CompletionChunk(
            id=response_id,
            created=created,
            model=request.model,
            choices=choices,
            usage=usage,
        )
        exclude = None if include_usage else _COMPLETION_CHUNK_EXCLUDE_USAGE_FIELDS
        if usage is not None:
            exclude = _with_usage_exclude(exclude, usage)
        serialized = escape_json_line_separators(payload.model_dump_json(exclude=exclude))
        return f"data: {serialized}\n\n"

    try:
        try:
            owner.mark_dispatched()
            async for partial in engine.stream(generation_request):
                last = partial
                owner.observe(partial.usage, partial.completions)
                for completion in partial.completions:
                    previous_offset = sent.get(completion.index, 0)
                    delta, next_offset = completion.delta_after(previous_offset)
                    if not delta and not partial.finished:
                        continue
                    sent[completion.index] = next_offset
                    finish = (completion.finish_reason or "stop") if partial.finished else None
                    if finish is None and type(completion.index) is int and type(delta) is str:
                        yield text_encoder.encode(completion.index, delta)
                    else:
                        yield _chunk(
                            [
                                CompletionChoice(
                                    index=completion.index,
                                    text=delta,
                                    finish_reason=finish,
                                )
                            ]
                        )
        except Exception as error:
            logger.exception("upstream backend error")
            payload = {  # M3: only the class name, no raw backend message
                "error": {
                    "message": f"upstream backend error ({type(error).__name__})",
                    "type": "upstream_error",
                }
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
            return
        owner.mark_completed()
        if include_usage and last is not None:
            yield _chunk(
                [],
                usage=_wire_usage(
                    generation_request.prompt,
                    last.completions,
                    owner.latest_usage,
                ),
            )
        yield "data: [DONE]\n\n"
    finally:
        owner.finalize()


def _record_usage(
    http_request: Request,
    model: str,
    usage: GenerationUsage | Usage | None,
    *,
    prompt: PromptInput,
    completions: Sequence[CompletionOutput],
    usage_exact: bool | None = None,
) -> None:
    """m11 D3/A7: metering happens in handlers (middleware can't see tokens)."""
    prompt_tokens, completion_tokens = resolve_usage_counts(
        usage, prompt=prompt, completions=completions
    )
    tenant = getattr(http_request.state, "tenant", None) or "default"
    record_state_usage(
        http_request.app.state,
        tenant=tenant,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=resolve_cached_tokens(usage),
        reservation=_tenant_reservation(http_request),
        usage_exact=(usage is not None if usage_exact is None else usage_exact),
    )


def _record_profile_judge_preflight_usage(
    http_request: Request,
    model: str,
    prompt: PromptInput,
    request: OrchestrationRequest,
) -> None:
    """Settle exact judge work when later orchestration preflight rejects."""

    event = request.role_profile_judge_event
    if event is None or event.usage is None:
        return
    usage = event.usage
    admission = _tenant_reservation(http_request)
    if admission is not None:
        admission.settle_tokens(
            usage.prompt_tokens + usage.completion_tokens,
            exact=True,
            force_refund_surplus=True,
        )

    _record_usage(
        http_request,
        model,
        GenerationUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_tokens=usage.cached_tokens,
        ),
        prompt=prompt,
        completions=(),
        usage_exact=True,
    )


def _session_id(request: ChatCompletionRequest, http_request: Request) -> str | None:
    """Session for ReplicaPool affinity: X-Session-ID header, else the OpenAI user field."""
    return http_request.headers.get("x-session-id") or request.user


def _scheduling_class(http_request: Request) -> str:
    """Resolve trusted gateway state, then the Kairyu replica transport hint."""

    request_class = getattr(http_request.state, "scheduling_class", None)
    if request_class in {"interactive", "batch"}:
        return request_class
    transported = http_request.headers.get("x-kairyu-scheduling-class")
    return transported if transported in {"interactive", "batch"} else "interactive"


def _record_preplacement_phase(
    http_request: Request,
    endpoint: str,
    phase: str,
    started_ns: int,
) -> None:
    """Publish a bounded phase without changing the placement total clock."""

    metrics = getattr(http_request.app.state, "metrics", None)
    if metrics is not None:
        metrics.record_preplacement_phase(
            endpoint,
            phase,
            max(0, time.perf_counter_ns() - started_ns),
        )


def _record_ingress_to_handler(http_request: Request, endpoint: str) -> None:
    ingress_ns = getattr(http_request.state, "placement_started_ns", None)
    if type(ingress_ns) is int:
        _record_preplacement_phase(
            http_request,
            endpoint,
            "ingress_to_handler",
            ingress_ns,
        )


def _server_sequence_budget(engines: Mapping[str, EngineBackend]) -> int | None:
    if len(engines) != 1:
        return None
    return backend_sequence_budget(next(iter(engines.values())))


def create_app(
    engines: Mapping[str, EngineBackend],
    orchestrator: Orchestrator | None = None,
    settings: ServerSettings | None = None,
    lifespan=None,
    chat_templates: Mapping[str, ChatTemplate] | None = None,
    orchestrators: Mapping[str, Orchestrator] | None = None,
    tenant_config=None,
    embedding_backends: Mapping[str, EmbeddingBackend] | None = None,
    resolved_api_keys: frozenset[str] | None = None,
    resolved_admin_keys: frozenset[str] | None = None,
    resolved_responses_compaction_key: bytes | None = None,
    price_sheet: PriceSheet | None = None,
    legacy_chat_models: AbstractSet[str] | None = None,
    orchestration_chat_models: AbstractSet[str] | None = None,
    runtime_engines: Mapping[str, EngineBackend] | None = None,
    runtime_embedding_backends: Mapping[str, EmbeddingBackend] | None = None,
    runtime_orchestrators: Mapping[str, Orchestrator] | None = None,
    async_request_body_limit: int | None = None,
) -> FastAPI:
    settings = settings or ServerSettings()
    legacy_chat_models = frozenset(legacy_chat_models or ())
    orchestration_chat_models = frozenset(orchestration_chat_models or ())
    validate_chat_policy(chat_templates, legacy_chat_models)
    if price_sheet is not None and settings.usage_ledger_path is None:
        raise ValueError("price_sheet requires usage_ledger_path")
    if price_sheet is not None:
        known_tenants = {"default"}
        if tenant_config is not None:
            known_tenants = {
                tenant_config.default_tenant,
                *tenant_config.key_tenants.values(),
            }
        unknown_discounts = price_sheet.tenant_discounts.keys() - known_tenants
        if unknown_discounts:
            raise ValueError(
                f"price_sheet discounts reference unknown tenants {sorted(unknown_discounts)}"
            )
    app = FastAPI(
        title="kairyu",
        version="0.1.0",
        lifespan=_with_usage_ledger_cleanup(lifespan),
    )
    app.router.route_class = _OffloadedRequestBodyRoute
    served_engines = dict(engines)
    # m11 D2: tiered auto models; the single-orchestrator kwarg is a shim
    auto_models: dict[str, Orchestrator] = dict(orchestrators or {})
    if orchestrator is not None:
        auto_models.setdefault(AUTO_MODEL, orchestrator)
    unknown_orchestration_chat = orchestration_chat_models - set(auto_models)
    if unknown_orchestration_chat:
        raise ValueError(
            "orchestration_chat_models contains non-orchestrated models: "
            f"{sorted(unknown_orchestration_chat)}"
        )
    templated_auto_models = set(auto_models) & set(chat_templates or {})
    if templated_auto_models:
        raise ValueError(
            "orchestrated models cannot safely consume pre-rendered chat prompts "
            "while deriving planner/worker prompts; explicitly opt in through "
            "legacy_chat_models instead: "
            f"{sorted(templated_auto_models)}"
        )
    served_embedding_backends = dict(embedding_backends or {})
    health_engines = dict(served_engines if runtime_engines is None else runtime_engines)
    health_embedding_backends = dict(
        served_embedding_backends
        if runtime_embedding_backends is None
        else runtime_embedding_backends
    )
    health_orchestrators = dict(
        auto_models if runtime_orchestrators is None else runtime_orchestrators
    )
    collisions = (
        (set(auto_models) & set(served_engines))
        | (set(served_embedding_backends) & set(served_engines))
        | (set(served_embedding_backends) & set(auto_models))
    )
    if collisions:
        raise ValueError(
            "served model names collide across engines, orchestrators, and embeddings: "
            f"{sorted(collisions)}"
        )
    missing_chat_policy = set(served_engines) - set(chat_templates or {}) - set(legacy_chat_models)
    if missing_chat_policy:
        logger.warning(
            "served models have no configured chat rendering policy; chat "
            "direct chat requests will be rejected before dispatch until each model has a "
            "ChatTemplate or legacy_chat_models membership: %s",
            sorted(missing_chat_policy),
        )

    metrics = ServerMetrics() if settings.metrics else None
    app.state.metrics = metrics
    slo_admission = (
        AdmissionController(settings.ttft_slo_s) if settings.ttft_slo_s is not None else None
    )
    app.state.slo_admission = slo_admission
    if metrics is not None:
        if slo_admission is not None:
            metrics.track_slo_admission(slo_admission)
        for name, engine in health_engines.items():
            if isinstance(engine, ReplicaPool):
                metrics.track_pool(name, engine)
            metrics.track_scheduler(name, engine)
            metrics.track_cuda_graph(name, engine)
        for auto_name, auto_orchestrator in auto_models.items():
            stage_observer_setter = getattr(auto_orchestrator, "set_stage_observer", None)
            if stage_observer_setter is not None:
                stage_observer_setter(_conductor_stage_metrics_observer(metrics, auto_name))
    api_keys = settings.resolve_api_keys() if resolved_api_keys is None else resolved_api_keys
    admin_keys = (
        settings.resolve_admin_keys() if resolved_admin_keys is None else resolved_admin_keys
    )
    responses_compaction_key = (
        settings.resolve_responses_compaction_key()
        if resolved_responses_compaction_key is None
        else resolved_responses_compaction_key
    )
    add_health_routes(
        app,
        health_engines,
        metrics,
        admin_keys=admin_keys,
        embedding_backends=health_embedding_backends,
        orchestrators=health_orchestrators,
    )
    from kairyu.entrypoints.server.extra_routes import add_extra_routes

    async def _responses_chat_dispatch(
        chat_request: ChatCompletionRequest, http_request: Request
    ):
        # Late-bound: chat_completions is defined later in this scope and the
        # forwarder only runs at request time, so AUTO models served on
        # /v1/responses reuse the full orchestration chat contract (issue #530).
        return await chat_completions(chat_request, http_request)

    add_extra_routes(
        app,
        served_engines,
        embedding_backends=served_embedding_backends,
        chat_templates=chat_templates,
        legacy_chat_models=legacy_chat_models,
        orchestrated_models=set(auto_models),
        chat_dispatch=_responses_chat_dispatch,
        responses_compaction_key=responses_compaction_key,
    )

    # add_middleware prepends, so add innermost first: metrics -> concurrency
    # guard -> auth -> access log (outermost).
    if async_request_body_limit is not None:
        app.add_middleware(
            ChatBodyLimitMiddleware,
            limit=async_request_body_limit,
            paths=("/v1/async/chat/completions",),
        )
    if settings.max_chat_body_bytes is not None:
        app.add_middleware(
            ChatBodyLimitMiddleware,
            limit=settings.max_chat_body_bytes,
        )
    if metrics is not None:
        app.add_middleware(MetricsMiddleware, metrics=metrics)
    if settings.max_concurrency is not None:
        sequence_budget = (
            _server_sequence_budget(served_engines)
            if settings.admission_wait_timeout_s is not None
            and not auto_models
            and not served_embedding_backends
            else None
        )
        if settings.admission_wait_timeout_s is not None and sequence_budget is None:
            logger.warning(
                "backend-aware admission queue disabled: serve exactly one "
                "generation backend with an advertised sequence budget and "
                "no orchestrated or embedding models"
            )
        active_limit = (
            min(settings.max_concurrency, sequence_budget)
            if sequence_budget is not None
            else settings.max_concurrency
        )
        app.add_middleware(
            ConcurrencyLimitMiddleware,
            limit=active_limit,
            total_limit=settings.max_concurrency,
            wait_timeout_s=settings.admission_wait_timeout_s,
            metrics=metrics,
        )
    if tenant_config is not None:
        from kairyu.entrypoints.server.tenancy import (
            TenantLimiter,
            TenantLimitMiddleware,
        )

        # added BEFORE auth => auth wraps it: 401 wins over 429 and
        # unauthenticated requests never drain buckets (m11 A6)
        limiter = TenantLimiter(tenant_config)
        app.state.tenant_limiter = limiter
        if metrics is not None:
            metrics.track_tenant_limiter(limiter)
        app.add_middleware(
            TenantLimitMiddleware,
            config=tenant_config,
            limiter=limiter,
            metrics=metrics,
        )
    if api_keys or admin_keys:
        app.add_middleware(
            AuthMiddleware,
            api_keys=api_keys,
            admin_keys=admin_keys,
            protect_metrics=settings.protect_metrics,
        )
    ledger = None
    if settings.usage_ledger_path:
        from kairyu.entrypoints.server.tenancy import UsageLedger

        ledger = UsageLedger(settings.usage_ledger_path)
        app.state.usage_ledger = ledger
        if metrics is not None:
            metrics.restore_usage_totals(ledger.totals())

        @app.get("/admin/usage")
        async def admin_usage(http_request: Request, tenant: str | None = None):
            """Scoped to the CALLER's tenant when tenancy is configured
            (security review: no cross-tenant disclosure); single-tenant
            deployments (no tenant_config) see everything behind auth."""
            state = http_request.scope.get("state", {})
            if state.get("is_admin"):
                return {
                    "usage": await asyncio.to_thread(
                        ledger.totals,
                        tenant,
                    )
                }
            if tenant_config is not None:
                caller = getattr(http_request.state, "tenant", None)
                if caller is None:
                    caller = tenant_config.default_tenant
                if tenant is not None and tenant != caller:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": {
                                "message": "cannot query another tenant's usage",
                                "type": "invalid_request_error",
                                "code": "tenant_forbidden",
                            }
                        },
                    )
                return {
                    "usage": await asyncio.to_thread(
                        ledger.totals,
                        caller,
                    )
                }
            return {
                "usage": await asyncio.to_thread(
                    ledger.totals,
                    tenant,
                )
            }

        if price_sheet is not None:
            app.state.price_sheet = price_sheet

            @app.get("/admin/usage.csv")
            async def admin_usage_csv(
                http_request: Request,
                tenant: str | None = None,
                start_ts: float | None = None,
                end_ts: float | None = None,
            ):
                """Export a deterministic, caller-scoped invoice snapshot."""
                state = http_request.scope.get("state", {})
                scoped_tenant = tenant
                if not state.get("is_admin") and tenant_config is not None:
                    caller = getattr(http_request.state, "tenant", None)
                    if caller is None:
                        caller = tenant_config.default_tenant
                    if tenant is not None and tenant != caller:
                        return JSONResponse(
                            status_code=403,
                            content={
                                "error": {
                                    "message": ("cannot export another tenant's invoice"),
                                    "type": "invalid_request_error",
                                    "code": "tenant_forbidden",
                                }
                            },
                        )
                    scoped_tenant = caller
                try:
                    snapshot = await asyncio.to_thread(ledger.snapshot_bytes)
                    payload = await asyncio.to_thread(
                        export_invoice_csv,
                        {"gateway-local": snapshot},
                        price_sheet,
                        tenant=scoped_tenant,
                        start_ts=start_ts,
                        end_ts=end_ts,
                    )
                except InvoiceExportError as error:
                    return JSONResponse(
                        status_code=409,
                        content={
                            "error": {
                                "message": str(error),
                                "type": "invoice_export_error",
                                "code": "invoice_ledger_invalid",
                            }
                        },
                    )
                return Response(
                    content=payload,
                    media_type="text/csv",
                    headers={
                        "content-disposition": ('attachment; filename="kairyu-usage-invoice.csv"')
                    },
                )

    if settings.tracing:
        from kairyu.entrypoints.server.middleware import TracingMiddleware
        from kairyu.telemetry import configure_tracing

        configure_tracing(True)
        app.add_middleware(TracingMiddleware)
    if settings.access_log:
        app.add_middleware(AccessLogMiddleware)
    # Always outermost: placement p99 is defined from process request receipt,
    # independent of access-log/metrics feature flags (G5 F1a).
    app.add_middleware(RequestIngressMiddleware)

    def _served_max_model_len(name: str) -> int | None:
        # Context length for the Model card (issue #495): engines advertise
        # their own limit; an orchestration serves up to its largest engine.
        engine = served_engines.get(name)
        if engine is not None:
            return backend_max_model_len(engine)
        selected = auto_models.get(name)
        if selected is not None:
            return getattr(selected, "max_model_len", None)
        return None

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        names = list(served_engines) + list(auto_models) + list(served_embedding_backends)
        return ModelList(
            data=[ModelCard(id=name, max_model_len=_served_max_model_len(name)) for name in names]
        )

    @app.get("/v1/models/{model_id:path}")
    async def retrieve_model(model_id: str):
        # OpenAI Models API retrieve (issue #496): the same served surface as
        # the list endpoint, one Model object or a standard 404 error body.
        if (
            model_id in served_engines
            or model_id in auto_models
            or model_id in served_embedding_backends
        ):
            return ModelCard(id=model_id, max_model_len=_served_max_model_len(model_id))
        return model_not_found(model_id)

    @app.post("/v1/route", response_model=RoutePreviewResponse)
    async def preview_route(
        request: RoutePreviewRequest,
        http_request: Request,
    ):
        http_request.state.model = request.model
        _record_ingress_to_handler(http_request, "route")
        selected = auto_models.get(request.model)
        if selected is not None:
            chat_request = ChatCompletionRequest(
                model=request.model,
                messages=request.messages,
            )
            validation_started_ns = time.perf_counter_ns()
            try:
                prompt = (
                    await validate_orchestration_chat_input_async(chat_request)
                    if request.model in orchestration_chat_models
                    else await validate_chat_input_async(
                        chat_request,
                        chat_templates,
                        legacy_chat_models=legacy_chat_models,
                    )
                ).prompt
            except ChatRequestError as error:
                return JSONResponse(
                    status_code=error.status_code,
                    content={"error": error.payload()},
                )
            finally:
                _record_preplacement_phase(
                    http_request,
                    "route",
                    "request_validation",
                    validation_started_ns,
                )
            routing_started_ns = time.perf_counter_ns()
            try:
                decision = await selected.preview_route_async(prompt)
            except PreviewNotSupportedError as error:
                return JSONResponse(
                    status_code=409,
                    content={
                        "error": {
                            "message": str(error),
                            "type": "invalid_request_error",
                            "code": "preview_not_supported",
                        }
                    },
                )
            finally:
                _record_preplacement_phase(
                    http_request,
                    "route",
                    "routing_preflight",
                    routing_started_ns,
                )
            payload = _route_payload(decision)
            descriptor = selected.describe_routing()
            return RoutePreviewResponse(
                model=request.model,
                orchestrated=True,
                router_type=descriptor["router"]["router_type"],
                **payload.model_dump(),
            )
        if request.model in served_engines:
            return RoutePreviewResponse(model=request.model, orchestrated=False)
        return model_not_found(request.model)

    @app.get("/routing", response_model=RoutingResponse)
    async def routing_config() -> RoutingResponse:
        return RoutingResponse(
            models={name: selected.describe_routing() for name, selected in auto_models.items()}
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        request: ChatCompletionRequest,
        http_request: Request,
    ):
        http_request.state.model = request.model  # label for the metrics middleware
        _record_ingress_to_handler(http_request, "chat")
        want_trace = http_request.headers.get("x-kairyu-trace") == "1"
        trace_started_at = utc_now_iso() if want_trace else None
        if request.model in auto_models:
            validation_started_ns = time.perf_counter_ns()
            try:
                validated_input = (
                    await validate_orchestration_chat_input_async(request)
                    if request.model in orchestration_chat_models
                    else await validate_chat_input_async(
                        request,
                        chat_templates,
                        legacy_chat_models=legacy_chat_models,
                    )
                )
            except ChatRequestError as error:
                return JSONResponse(
                    status_code=error.status_code, content={"error": error.payload()}
                )
            finally:
                _record_preplacement_phase(
                    http_request,
                    "chat",
                    "request_validation",
                    validation_started_ns,
                )
            prompt = validated_input.prompt
            normalized_tool_choice = validated_input.normalized_tool_choice
            include_usage = validated_input.include_usage
            routing_started_ns = time.perf_counter_ns()
            try:
                try:
                    sampling = sampling_params_from(request)
                except ValueError as error:
                    return invalid_request(str(error))
                orchestration_request = OrchestrationRequest(
                    prompt=prompt,
                    sampling_params=sampling,
                    tools=tuple(request.tools or ()),
                    tool_choice=request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                    tools_in_prompt=validated_input.tools_in_prompt,
                    structured_format_in_prompt=validated_input.structured_format_in_prompt,
                    conversation_affinity_key=(validated_input.conversation_affinity_key),
                    response_format=request.response_format,
                    tool_call_protocol=validated_input.tool_call_protocol.value,
                    reasoning_effort=request.reasoning_effort,
                    multimodal_prompt=validated_input.orchestration_multimodal_prompt,
                    chat_template_kwargs=(
                        request.chat_template_kwargs
                        if validated_input.orchestration_multimodal_prompt is not None
                        else None
                    ),
                )
                selected = auto_models[request.model]
                judge_will_run = selected.will_judge_role_profile(orchestration_request)
                try:
                    if judge_will_run:
                        # The judge is shared GPU work too. Reserve its bound
                        # plus the larger profile before the first dispatch.
                        bound = await selected.admission_upper_bound_async(orchestration_request)
                        admission_started_ns = time.perf_counter_ns()
                        reservation_error = _reserve_tenant_work(http_request, bound)
                        _record_preplacement_phase(
                            http_request,
                            "chat",
                            "admission",
                            admission_started_ns,
                        )
                        if reservation_error is not None:
                            return reservation_error
                        _mark_tenant_dispatched(http_request)
                        judged, disconnect_response = await _run_until_client_disconnect(
                            http_request,
                            selected.judge_role_profile(orchestration_request),
                        )
                        if disconnect_response is not None:
                            return disconnect_response
                        orchestration_request = judged
                    prepared_orchestration = await selected.prepare_request(orchestration_request)
                    if not judge_will_run:
                        bound = await selected.admission_upper_bound_async(orchestration_request)
                except UpstreamClientError as error:
                    _record_profile_judge_preflight_usage(
                        http_request,
                        request.model,
                        prompt,
                        orchestration_request,
                    )
                    chat_error = chat_error_from_upstream_client_error(error)
                    return JSONResponse(
                        status_code=chat_error.status_code,
                        content={"error": chat_error.payload()},
                    )
                except ValueError as error:
                    _record_profile_judge_preflight_usage(
                        http_request,
                        request.model,
                        prompt,
                        orchestration_request,
                    )
                    return invalid_request(str(error))
                except RuntimeError as error:
                    _record_profile_judge_preflight_usage(
                        http_request,
                        request.model,
                        prompt,
                        orchestration_request,
                    )
                    return upstream_error(error)
            finally:
                _record_preplacement_phase(
                    http_request,
                    "chat",
                    "routing_preflight",
                    routing_started_ns,
                )
            if not judge_will_run:
                admission_started_ns = time.perf_counter_ns()
                reservation_error = _reserve_tenant_work(http_request, bound)
                _record_preplacement_phase(
                    http_request,
                    "chat",
                    "admission",
                    admission_started_ns,
                )
                if reservation_error is not None:
                    return reservation_error
            # Tool choices must be validated across every final choice before
            # any bytes become irrevocable SSE output. Indexed alternatives
            # and logprobs otherwise stay on the low-latency pull-through path.
            # The one exception is the Anthropic Messages adapter's internal
            # dispatch (#573): the sentinel is per-request process state that
            # no wire request can set, and the Messages relay enforces the
            # tool gates itself on the raw stream.
            buffered_stream = bool(request.tools) and not getattr(
                http_request.state, _ANTHROPIC_INTERNAL_TOOL_STREAM_STATE_KEY, False
            )
            if request.stream and not buffered_stream:
                return sse_response(
                    _stream_orchestrator(
                        selected,
                        orchestration_request,
                        prepared_orchestration,
                        request,
                        include_usage,
                        want_trace,
                        http_request,
                    )
                )
            try:
                _mark_tenant_dispatched(http_request)
                result, disconnect_response = await _run_until_client_disconnect(
                    http_request,
                    selected.run(
                        orchestration_request,
                        prepared=prepared_orchestration,
                    ),
                )
                if disconnect_response is not None:
                    return disconnect_response
            except OrchestratorExecutionError as error:
                logger.exception("orchestrator backend error")
                result = error.result
                completions = result.completions or (
                    CompletionOutput(
                        index=0,
                        text=result.text,
                        token_ids=(),
                        finish_reason=None,
                    ),
                )
                usage = _orchestration_wire_usage(
                    prompt,
                    completions,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    public_prompt_tokens=result.public_prompt_tokens,
                    public_completion_tokens=result.public_completion_tokens,
                )
                _record_usage(
                    http_request,
                    request.model,
                    # Bill the cumulative internal-call totals, not the
                    # public-facing standard fields (m11 #196).
                    GenerationUsage(
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        cached_tokens=result.cached_tokens,
                    )
                    if result.prompt_tokens or result.completion_tokens
                    else None,
                    prompt=prompt,
                    completions=completions,
                    usage_exact=False,
                )
                payload = {
                    "error": sanitize_backend_error(error.cause),
                    "usage": usage.model_dump(
                        mode="json",
                        exclude=_unset_orchestration_usage_fields(usage),
                    ),
                }
                if want_trace:
                    payload["kairyu_trace"] = list(result.trace)
                    if result.structured_trace is not None:
                        payload["kairyu_trace_v2"] = result.structured_trace.as_dict()
                    payload["kairyu_route"] = _route_payload(result.route).model_dump(mode="json")
                return JSONResponse(status_code=502, content=payload)
            except Exception as error:
                return upstream_error(error)
            completions = result.completions or (
                CompletionOutput(index=0, text=result.text, token_ids=(), finish_reason="stop"),
            )
            # Standard usage keeps the OpenAI public meaning; cumulative AUTO
            # totals ride the orchestration_* extensions and metering below
            # bills the aggregate explicitly (issue #496, m11 #196).
            response = completion_response(
                request,
                prompt,
                completions,
                normalized_tool_choice=normalized_tool_choice,
                tool_call_protocol=validated_input.tool_call_protocol,
            )
            response.usage = _orchestration_wire_usage(
                prompt,
                completions,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                public_prompt_tokens=result.public_prompt_tokens,
                public_completion_tokens=result.public_completion_tokens,
            )
            # OrchestratorResult uses 0/0 when its backend did not report
            # usage; keep that state missing so metering derives the wire
            # approximation instead of billing zero.
            metered_usage = (
                GenerationUsage(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    cached_tokens=result.cached_tokens,
                )
                if result.prompt_tokens or result.completion_tokens
                else None
            )
            if not tool_choice_is_satisfied(
                response.choices,
                normalized_tool_choice,
            ):
                _record_usage(
                    http_request,
                    request.model,
                    metered_usage,
                    prompt=prompt,
                    completions=completions,
                    usage_exact=False,
                )
                error = ChatRequestError(
                    "upstream model did not satisfy tool_choice",
                    status_code=502,
                    code="tool_choice_not_satisfied",
                    error_type="upstream_error",
                )
                return JSONResponse(
                    status_code=error.status_code,
                    content={"error": error.payload()},
                )
            if not parallel_tool_calls_is_satisfied(
                response.choices,
                validated_input.parallel_tool_calls,
            ):
                _record_usage(
                    http_request,
                    request.model,
                    metered_usage,
                    prompt=prompt,
                    completions=completions,
                    usage_exact=False,
                )
                error = ChatRequestError(
                    "upstream model emitted multiple calls while parallel_tool_calls=false",
                    status_code=502,
                    code="parallel_tool_calls_not_satisfied",
                    error_type="upstream_error",
                )
                return JSONResponse(
                    status_code=error.status_code,
                    content={"error": error.payload()},
                )
            _record_usage(
                http_request,
                request.model,
                metered_usage,
                prompt=prompt,
                completions=completions,
                usage_exact=False,
            )
            if request.stream:
                return sse_response(
                    _stream_choices(
                        response.choices,
                        request.model,
                        usage=response.usage if include_usage else None,
                        orchestration_result=result,
                        want_trace=want_trace,
                    )
                )
            if want_trace:
                payload = _chat_response_payload(response)
                payload["kairyu_trace"] = list(result.trace)
                if result.structured_trace is not None:
                    payload["kairyu_trace_v2"] = result.structured_trace.as_dict(
                        request_id=response.id
                    )
                payload["kairyu_route"] = _route_payload(result.route).model_dump(mode="json")
                return JSONResponse(content=payload)
            return JSONResponse(content=_chat_response_payload(response))

        session_id = _session_id(request, http_request)
        validation_started_ns = time.perf_counter_ns()
        try:
            validated = await validate_chat_request_async(
                request,
                served_engines,
                chat_templates,
                request_id=(
                    getattr(http_request.state, "request_id", None)
                    or f"http-{uuid.uuid4().hex[:12]}"
                ),
                # Affinity glue (m7 D6): keeps a session's turns on the replica
                # holding its warm radix-KV prefix.
                cache_hint=CacheHint(session_id=session_id) if session_id else None,
                priority=getattr(http_request.state, "priority", None),
                scheduling_class=_scheduling_class(http_request),
                placement_started_ns=getattr(http_request.state, "placement_started_ns", None),
                trace_requested=want_trace,
                legacy_chat_models=legacy_chat_models,
            )
        except ChatRequestError as error:
            return JSONResponse(status_code=error.status_code, content={"error": error.payload()})
        finally:
            _record_preplacement_phase(
                http_request,
                "chat",
                "request_validation",
                validation_started_ns,
            )
        slo_lease = None
        if (
            slo_admission is not None
            and validated.generation_request.scheduling_class == "interactive"
        ):
            ingress_ns = getattr(http_request.state, "placement_started_ns", None)
            elapsed_s = (
                max(0, time.perf_counter_ns() - ingress_ns) / 1_000_000_000
                if type(ingress_ns) is int
                else 0.0
            )
            lease = slo_admission.begin(elapsed_s=elapsed_s)
            if lease.decision.action == "shed":
                return _slo_shed_response()
            slo_lease = lease
            http_request.scope.setdefault("state", {})[_SLO_ADMISSION_LEASE_STATE_KEY] = lease
            generation_request = validated.generation_request
            if lease.decision.action == "defer":
                admission_request = replace(
                    generation_request,
                    priority=_LOWEST_SCHEDULER_PRIORITY,
                    scheduling_class="batch",
                )
                if not backend_supports_slo_defer(
                    validated.engine,
                    admission_request,
                ):
                    return _slo_defer_to_shed_response(http_request, lease)
            elif generation_request.priority > _SLO_INTERACTIVE_PRIORITY_CEILING:
                admission_request = replace(
                    generation_request,
                    priority=_SLO_INTERACTIVE_PRIORITY_CEILING,
                )
            else:
                admission_request = generation_request
            if admission_request is not generation_request:
                try:
                    validate_backend_request_before_prepare(
                        validated.engine,
                        admission_request,
                    )
                except ValueError as error:
                    return invalid_request(str(error))
                validated = replace(
                    validated,
                    generation_request=admission_request,
                )
        prepare_started_ns = time.perf_counter_ns()
        try:
            await prepare_backend_request(
                validated.engine,
                validated.generation_request,
            )
        except ValueError as error:
            chat_error = ChatRequestError(
                str(error),
                code=getattr(error, "code", "invalid_request"),
            )
            return JSONResponse(
                status_code=chat_error.status_code,
                content={"error": chat_error.payload()},
            )
        except UpstreamClientError as error:
            chat_error = chat_error_from_upstream_client_error(error)
            return JSONResponse(
                status_code=chat_error.status_code,
                content={"error": chat_error.payload()},
            )
        except RuntimeError as error:
            return upstream_error(error)
        finally:
            _record_preplacement_phase(
                http_request,
                "chat",
                "backend_prepare",
                prepare_started_ns,
            )
        if (
            slo_lease is not None
            and slo_lease.decision.action == "defer"
            and not backend_supports_slo_defer(
                validated.engine,
                validated.generation_request,
            )
        ):
            return _slo_defer_to_shed_response(http_request, slo_lease)
        admission_started_ns = time.perf_counter_ns()
        try:
            bound = await backend_admission_upper_bound_async(
                validated.engine,
                validated.generation_request,
            )
        except ValueError as error:
            return invalid_request(str(error))
        except RuntimeError as error:
            return upstream_error(error)
        admission_ns = max(0, time.perf_counter_ns() - admission_started_ns)
        reserve_started_ns = time.perf_counter_ns()
        reservation_error = _reserve_tenant_work(http_request, bound)
        admission_ns += max(0, time.perf_counter_ns() - reserve_started_ns)
        if metrics is not None:
            metrics.record_preplacement_phase(
                "chat",
                "admission",
                admission_ns,
            )
        if reservation_error is not None:
            return reservation_error
        if metrics is not None:
            metrics.record_priority(
                validated.generation_request.scheduling_class,
                source="http",
            )
        if request.stream and not request.tools:
            return sse_response(
                _stream_engine(
                    validated.engine,
                    validated.generation_request,
                    request.model,
                    request,
                    http_request,
                    trace_started_at=trace_started_at,
                )
            )
        try:
            _mark_tenant_dispatched(http_request)
            executed = await execute_chat(validated)
        except ChatRequestError as error:
            if error.execution is not None:
                _record_usage(
                    http_request,
                    request.model,
                    error.execution.result.usage,
                    prompt=error.execution.result.prompt,
                    completions=error.execution.result.completions,
                )
            return JSONResponse(status_code=error.status_code, content={"error": error.payload()})
        except Exception as error:
            return upstream_error(error)
        response = executed.response
        _record_usage(
            http_request,
            request.model,
            executed.result.usage,
            prompt=executed.result.prompt,
            completions=executed.result.completions,
        )
        if request.stream:
            # Tool calling + streaming: generate fully, then emit structured chunks so
            # tool_calls and finish_reason stay correct.
            return sse_response(
                _stream_choices(
                    response.choices,
                    request.model,
                    usage=(response.usage if validated.input.include_usage else None),
                    stage_metrics=executed.result.stage_metrics,
                    trace_started_at=trace_started_at,
                    want_trace=want_trace,
                    slo_lease=slo_lease,
                )
            )
        payload = _chat_response_payload(response)
        if want_trace:
            assert trace_started_at is not None
            payload["kairyu_trace"] = _direct_trace_lines(executed.result.stage_metrics)
            payload["kairyu_trace_v2"] = _direct_stage_trace(
                response.id,
                started_at=trace_started_at,
                stage_metrics=executed.result.stage_metrics,
            ).as_dict()
        return JSONResponse(content=payload)

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest, http_request: Request):
        http_request.state.model = request.model
        _record_ingress_to_handler(http_request, "completions")
        validation_started_ns = time.perf_counter_ns()
        is_loglikelihood = request.kairyu_continuation is not None
        extra = request.model_extra or {}
        for unsupported in ("echo", "suffix", "best_of"):
            if extra.get(unsupported) is not None:
                return invalid_request(f"{unsupported} is not supported")
        unknown_extra = set(extra) - {"echo", "suffix", "best_of"}
        if unknown_extra:
            return invalid_request(
                "unsupported request fields: " + ", ".join(sorted(unknown_extra))
            )
        if request.logprobs is not None and not 0 <= request.logprobs <= 5:
            return invalid_request("logprobs must be between 0 and 5")
        if request.stream_options is not None and not request.stream:
            return invalid_request("stream_options is only allowed when stream is true")
        if is_loglikelihood:
            validation_error = _loglikelihood_request_error(request)
            if validation_error is not None:
                return invalid_request(validation_error)
        if isinstance(request.prompt, list) and not request.prompt:
            return invalid_request("prompt array must not be empty")
        text_batch = (
            isinstance(request.prompt, list)
            and bool(request.prompt)
            and type(request.prompt[0]) is str
        )
        token_batch = (
            isinstance(request.prompt, list)
            and bool(request.prompt)
            and type(request.prompt[0]) is list
        )
        if request.stream and (text_batch or token_batch):
            return invalid_request("streaming with a prompt array is not supported")
        engine = served_engines.get(request.model)
        if engine is None:
            if is_loglikelihood and request.model in auto_models:
                return invalid_request(
                    f"kairyu_continuation is not supported by model {request.model!r}"
                )
            return model_not_found(request.model)
        if request.n > 1 and getattr(engine, "supports_n", True) is False:
            return invalid_request(f"model {request.model!r} does not support n > 1")
        loglikelihood_context_ids: tuple[int, ...] | None = None
        loglikelihood_continuation_ids: tuple[int, ...] | None = None
        prompts: list[PromptInput]
        if is_loglikelihood:
            tokenize = getattr(engine, "tokenize_loglikelihood", None)
            tokenize_async = getattr(engine, "tokenize_loglikelihood_async", None)
            if not callable(tokenize) and not callable(tokenize_async):
                return invalid_request(
                    f"kairyu_continuation is not supported by model {request.model!r}"
                )
            assert type(request.prompt) is str
            assert request.kairyu_continuation is not None
            try:
                (
                    loglikelihood_context_ids,
                    loglikelihood_continuation_ids,
                ) = await _loglikelihood_token_ids_async(
                    engine,
                    request.prompt,
                    request.kairyu_continuation,
                )
            except ValueError:
                # Boundary validation is tokenizer owned. Do not expose
                # tokenizer internals, vocabulary pieces, or model paths.
                return invalid_request(
                    "kairyu_continuation does not align with a model token boundary"
                )
            except Exception as error:
                return upstream_error(error)
            prompts = [TokensPrompt(loglikelihood_context_ids, prompt=request.prompt)]
        elif type(request.prompt) is str:
            prompts = [request.prompt]
        elif text_batch:
            prompts = list(request.prompt)
        elif token_batch:
            prompts = [TokensPrompt(tuple(token_ids)) for token_ids in request.prompt]
        else:
            prompts = [TokensPrompt(tuple(request.prompt))]
        try:
            if is_loglikelihood:
                assert loglikelihood_continuation_ids is not None
                sampling = SamplingParams(
                    max_tokens=len(loglikelihood_continuation_ids),
                    temperature=0.0,
                    logprobs=0,
                    ignore_eos=True,
                    forced_token_ids=loglikelihood_continuation_ids,
                    skip_special_tokens=False,
                )
            else:
                sampling = SamplingParams(  # invalid params are a client error, not a 502
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    min_p=request.min_p,
                    n=request.n,
                    max_tokens=(request.max_tokens if request.max_tokens is not None else 16),
                    stop=request.stop,
                    stop_token_ids=request.stop_token_ids,
                    min_tokens=request.min_tokens,
                    ignore_eos=request.ignore_eos,
                    seed=request.seed,
                    presence_penalty=request.presence_penalty,
                    frequency_penalty=request.frequency_penalty,
                    repetition_penalty=request.repetition_penalty,
                    logprobs=request.logprobs,
                    skip_special_tokens=request.skip_special_tokens,
                ).with_generation_config_omitted(
                    {
                        name
                        for name in GENERATION_CONFIG_SAMPLING_FIELDS
                        if name not in request.model_fields_set
                    }
                )
        except ValueError as error:
            return invalid_request(str(error))

        base_request_id = (
            getattr(http_request.state, "request_id", None) or f"http-{uuid.uuid4().hex[:12]}"
        )

        def _generation_request(
            prompt: PromptInput,
            prompt_index: int,
        ) -> GenerationRequest:
            return GenerationRequest(
                request_id=(
                    base_request_id
                    if len(prompts) == 1
                    else f"{base_request_id}-prompt-{prompt_index}"
                ),
                prompt=prompt,
                sampling_params=sampling,
                placement_started_ns=getattr(http_request.state, "placement_started_ns", None),
                priority=getattr(http_request.state, "priority", request.priority),
                scheduling_class=_scheduling_class(http_request),
            )

        generation_requests = [
            _generation_request(prompt, prompt_index) for prompt_index, prompt in enumerate(prompts)
        ]
        for generation_request in generation_requests:
            if is_loglikelihood:
                # tokenize_loglikelihood is itself an explicit token-prompt
                # capability. Preserve any stronger backend-specific validator
                # without requiring the legacy fallback declaration as well.
                validate = getattr(engine, "validate_request", None)
                if callable(validate):
                    try:
                        validate(generation_request)
                    except ValueError as error:
                        return invalid_request(str(error))
            else:
                validation_error = _validate_generation_request(engine, generation_request)
                if validation_error is not None:
                    return validation_error
        _record_preplacement_phase(
            http_request,
            "completions",
            "request_validation",
            validation_started_ns,
        )
        prepare_started_ns = time.perf_counter_ns()
        preparation_tasks = [
            asyncio.create_task(prepare_backend_request(engine, generation_request))
            for generation_request in generation_requests
        ]
        try:
            try:
                preparation_results = await asyncio.gather(
                    *preparation_tasks,
                    return_exceptions=True,
                )
            except BaseException:
                for task in preparation_tasks:
                    task.cancel()
                await asyncio.gather(*preparation_tasks, return_exceptions=True)
                raise
            for result in preparation_results:
                if isinstance(result, BaseException):
                    raise result
        except ValueError as error:
            return invalid_request(str(error))
        except UpstreamClientError as error:
            chat_error = chat_error_from_upstream_client_error(error)
            return JSONResponse(
                status_code=chat_error.status_code,
                content={"error": chat_error.payload()},
            )
        except RuntimeError as error:
            return upstream_error(error)
        finally:
            _record_preplacement_phase(
                http_request,
                "completions",
                "backend_prepare",
                prepare_started_ns,
            )
        admission_started_ns = time.perf_counter_ns()
        bound_tasks = [
            asyncio.create_task(
                backend_admission_upper_bound_async(
                    engine,
                    generation_request,
                )
            )
            for generation_request in generation_requests
        ]
        try:
            try:
                bound_results = await asyncio.gather(
                    *bound_tasks,
                    return_exceptions=True,
                )
            except BaseException:
                for task in bound_tasks:
                    task.cancel()
                await asyncio.gather(*bound_tasks, return_exceptions=True)
                raise
            for result in bound_results:
                if isinstance(result, BaseException):
                    raise result
            bounds = bound_results
        except ValueError as error:
            return invalid_request(str(error))
        except UpstreamClientError as error:
            chat_error = chat_error_from_upstream_client_error(error)
            return JSONResponse(
                status_code=chat_error.status_code,
                content={"error": chat_error.payload()},
            )
        except RuntimeError as error:
            return upstream_error(error)
        admission_ns = max(0, time.perf_counter_ns() - admission_started_ns)
        reserve_started_ns = time.perf_counter_ns()
        reservation_error = _reserve_tenant_work(
            http_request,
            AdmissionUpperBound(
                tokens=sum(bound.tokens for bound in bounds),
                refundable_on_exact_usage=all(bound.refundable_on_exact_usage for bound in bounds),
            ),
        )
        admission_ns += max(0, time.perf_counter_ns() - reserve_started_ns)
        if metrics is not None:
            metrics.record_preplacement_phase(
                "completions",
                "admission",
                admission_ns,
            )
        if reservation_error is not None:
            return reservation_error
        if metrics is not None:
            for generation_request in generation_requests:
                metrics.record_priority(
                    generation_request.scheduling_class,
                    source="http",
                )
        if request.stream:
            return sse_response(
                _stream_completions(
                    engine,
                    generation_requests[0],
                    request,
                    http_request,
                )
            )
        choices: list[CompletionChoice] = []
        usage_totals = [0, 0, 0]  # prompt, completion, cached
        try:
            # run the prompt array concurrently (latency = max, not sum); order is
            # restored by prompt_index below so the response is unchanged (P-perf)
            _mark_tenant_dispatched(http_request)
            tasks = [asyncio.create_task(engine.generate(item)) for item in generation_requests]
            try:
                results = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        except Exception as error:
            return upstream_error(error)
        for prompt_index, (prompt, result) in enumerate(zip(prompts, results, strict=True)):
            if is_loglikelihood:
                assert loglikelihood_context_ids is not None
                assert loglikelihood_continuation_ids is not None
                assert request.kairyu_continuation is not None
                try:
                    choices.append(
                        _loglikelihood_choice(
                            result,
                            context_ids=loglikelihood_context_ids,
                            continuation=request.kairyu_continuation,
                            continuation_ids=loglikelihood_continuation_ids,
                        )
                    )
                except Exception as error:
                    return upstream_error(error)
            else:
                for completion in result.completions:
                    choices.append(
                        _completion_choice(
                            prompt_index * request.n + completion.index,
                            completion,
                        )
                    )
            prompt_tokens, completion_tokens = resolve_usage_counts(
                result.usage,
                prompt=prompt,
                completions=result.completions,
            )
            usage_totals[0] += prompt_tokens
            usage_totals[1] += completion_tokens
            if result.usage is not None:
                usage_totals[2] += result.usage.cached_tokens
        details = PromptTokensDetails(cached_tokens=usage_totals[2]) if usage_totals[2] else None
        _record_usage(  # S3: /v1/completions was never metered
            http_request,
            request.model,
            GenerationUsage(
                prompt_tokens=usage_totals[0],
                completion_tokens=usage_totals[1],
                cached_tokens=usage_totals[2],
            ),
            prompt="",
            completions=(),
            usage_exact=all(result.usage is not None for result in results),
        )
        response_type = LogLikelihoodCompletionResponse if is_loglikelihood else CompletionResponse
        response = response_type(
            id=f"cmpl-{uuid.uuid4().hex[:16]}",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage=Usage(
                prompt_tokens=usage_totals[0],
                completion_tokens=usage_totals[1],
                total_tokens=usage_totals[0] + usage_totals[1],
                prompt_tokens_details=details,
            ),
        )
        return JSONResponse(
            content=response.model_dump(
                mode="json",
                exclude={"usage": _USAGE_ORCHESTRATION_FIELDS},
            )
        )

    return app
