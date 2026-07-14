"""OpenAI-compatible FastAPI app: /v1/models, /v1/chat/completions (+SSE, tools).

Model name ``kairyu-auto`` routes the request through the Orchestrator behind
the same endpoint (design doc D6).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from kairyu.engine.backend import (
    CacheHint,
    EngineBackend,
    GenerationRequest,
    GenerationUsage,
)
from kairyu.entrypoints.chat_template import ChatTemplate, render_chat
from kairyu.entrypoints.server.health import add_health_routes
from kairyu.entrypoints.server.metrics import ServerMetrics
from kairyu.entrypoints.server.middleware import (
    AccessLogMiddleware,
    AuthMiddleware,
    ConcurrencyLimitMiddleware,
    MetricsMiddleware,
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
    FunctionCall,
    LogprobEntry,
    ModelCard,
    ModelList,
    PromptTokensDetails,
    ResponseMessage,
    ToolCall,
    TopLogprobEntry,
    Usage,
)
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.orchestration.replica import ReplicaPool
from kairyu.outputs import CompletionOutput, TokenLogprob
from kairyu.sampling_params import SamplingParams

AUTO_MODEL = "kairyu-auto"
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
logger = logging.getLogger(__name__)


def sampling_params_from(request: ChatCompletionRequest) -> SamplingParams:
    extra_args = (
        {"response_format": request.response_format} if request.response_format else {}
    )
    logprobs = None
    if request.logprobs:
        logprobs = request.top_logprobs or 0  # 0 = sampled token only
    return SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        n=request.n,
        max_tokens=request.max_tokens or request.max_completion_tokens,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        stop=request.stop,
        seed=request.seed,
        logprobs=logprobs,
        extra_args=extra_args,
    )


def _approx_tokens(text: str) -> int:
    return len(text.split())


def _parse_tool_calls(text: str) -> list[ToolCall]:
    calls = []
    for match in _TOOL_CALL_PATTERN.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        arguments = payload.get("arguments", {})
        calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                function=FunctionCall(
                    name=payload.get("name", ""),
                    arguments=(
                        arguments if isinstance(arguments, str) else json.dumps(arguments)
                    ),
                ),
            )
        )
    return calls


def _validate_response_format(response_format: dict | None) -> str | None:
    """400 (not an engine crash) for malformed response_format (m9 D4)."""
    if response_format is None:
        return None
    kind = response_format.get("type")
    if kind not in ("text", "json_object", "json_schema"):
        return f"response_format.type must be text, json_object or json_schema, got {kind!r}"
    if kind == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        if not isinstance(schema, dict):
            return "response_format.json_schema.schema must be a JSON schema object"
    return None


def _invalid_request(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "invalid_request",
            }
        },
    )


def render_prompt(
    request: ChatCompletionRequest,
    chat_templates: Mapping[str, ChatTemplate] | None,
) -> str:
    """Per-model HF template when configured, legacy concatenator otherwise
    (m9 D2). Shared by the HTTP path and the batch worker — the two must
    render identical prompts for the same model."""
    template = (chat_templates or {}).get(request.model)
    messages = [message.model_dump() for message in request.messages]
    if template is None:
        return render_chat(messages)
    return template.render(messages, tools=request.tools)


def _model_not_found(model: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": f"model {model!r} not found",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        },
    )


def _upstream_error(error: Exception) -> JSONResponse:
    # only the exception CLASS name leaves the gateway (M3): str(error) can carry
    # connection strings, internal hostnames/ports, and file paths from arbitrary
    # backend failures — those must not reach the client
    logger.exception("upstream backend error")
    return JSONResponse(
        status_code=502,
        content={
            "error": {
                "message": f"upstream backend error ({type(error).__name__})",
                "type": "upstream_error",
                "code": "backend_error",
            }
        },
    )


def _logprob_entries(content: tuple[TokenLogprob, ...]) -> list[LogprobEntry]:
    return [
        LogprobEntry(
            token=entry.token,
            logprob=entry.logprob,
            bytes=list(entry.bytes_) if entry.bytes_ is not None else None,
            top_logprobs=[
                TopLogprobEntry(
                    token=top.token,
                    logprob=top.logprob,
                    bytes=list(top.bytes_) if top.bytes_ is not None else None,
                )
                for top in entry.top
            ],
        )
        for entry in content
    ]


def _choice_logprobs(completion: CompletionOutput) -> ChoiceLogprobs | None:
    if completion.logprob_content is None:
        return None
    return ChoiceLogprobs(content=_logprob_entries(completion.logprob_content))


def _build_choice(
    index: int,
    text: str,
    request: ChatCompletionRequest,
    finish_reason: str | None,
    logprobs: ChoiceLogprobs | None = None,
) -> Choice:
    tool_calls = _parse_tool_calls(text) if request.tools else []
    if tool_calls:
        message = ResponseMessage(content=None, tool_calls=tool_calls)
        return Choice(
            index=index, message=message, finish_reason="tool_calls", logprobs=logprobs
        )
    return Choice(
        index=index,
        message=ResponseMessage(content=text),
        finish_reason=finish_reason or "stop",
        logprobs=logprobs,
    )


def _wire_usage(
    prompt: str, completions: Sequence[CompletionOutput], usage: GenerationUsage | None
) -> Usage:
    """Backend-reported counts when present; the word-split approximation
    survives only for usage=None producers (orchestrator until M11,
    third-party backends) — m9 D1."""
    if usage is not None:
        details = (
            PromptTokensDetails(cached_tokens=usage.cached_tokens)
            if usage.cached_tokens
            else None
        )
        return Usage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.prompt_tokens + usage.completion_tokens,
            prompt_tokens_details=details,
        )
    prompt_tokens = _approx_tokens(prompt)
    completion_tokens = sum(
        len(c.token_ids) if c.token_ids else _approx_tokens(c.text) for c in completions
    )
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def completion_response(
    request: ChatCompletionRequest,
    prompt: str,
    completions: Sequence[CompletionOutput],
    usage: GenerationUsage | None = None,
) -> ChatCompletionResponse:
    choices = [
        _build_choice(c.index, c.text, request, c.finish_reason, _choice_logprobs(c))
        for c in completions
    ]
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:16]}",
        created=int(time.time()),
        model=request.model,
        choices=choices,
        usage=_wire_usage(prompt, completions, usage),
    )


def _sse_chunk(
    response_id: str, created: int, model: str, index: int, delta: ChunkDelta,
    finish_reason: str | None = None, include_usage: bool = False,
    usage: Usage | None = None, logprobs: ChoiceLogprobs | None = None,
) -> str:
    payload = ChatCompletionChunk(
        id=response_id,
        created=created,
        model=model,
        choices=[
            ChunkChoice(
                index=index, delta=delta, finish_reason=finish_reason, logprobs=logprobs
            )
        ],
        usage=usage,
    )
    # OpenAI contract: usage key omitted unless include_usage; then explicit
    # null on non-final chunks, populated on the final choices-less chunk
    exclude = None if include_usage else {"usage"}
    return f"data: {payload.model_dump_json(exclude=exclude)}\n\n"


def _usage_chunk(response_id: str, created: int, model: str, usage: Usage) -> str:
    payload = ChatCompletionChunk(
        id=response_id, created=created, model=model, choices=[], usage=usage
    )
    return f"data: {payload.model_dump_json()}\n\n"


async def _stream_engine(
    engine: EngineBackend,
    generation_request: GenerationRequest,
    model: str,
    request: ChatCompletionRequest,
    http_request: Request | None = None,
) -> AsyncIterator[str]:
    response_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    include_usage = bool(request.stream_options and request.stream_options.include_usage)
    sent: dict[int, int] = {}
    logprobs_sent: dict[int, int] = {}
    last = None
    try:
        async for partial in engine.stream(generation_request):
            last = partial
            for completion in partial.completions:
                delta_text = completion.text[sent.get(completion.index, 0):]
                if not delta_text and not partial.finished:
                    continue
                is_first = completion.index not in sent
                sent[completion.index] = len(completion.text)
                chunk_logprobs = None
                if request.logprobs and completion.logprob_content is not None:
                    seen = logprobs_sent.get(completion.index, 0)
                    fresh = completion.logprob_content[seen:]
                    logprobs_sent[completion.index] = len(completion.logprob_content)
                    if fresh:
                        chunk_logprobs = ChoiceLogprobs(content=_logprob_entries(fresh))
                yield _sse_chunk(
                    response_id, created, model, completion.index,
                    ChunkDelta(role="assistant" if is_first else None, content=delta_text),
                    include_usage=include_usage,
                    logprobs=chunk_logprobs,
                )
    except Exception as error:  # surface backend failures inside the SSE stream
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
    for completion in last.completions if last else ():
        yield _sse_chunk(
            response_id, created, model, completion.index, ChunkDelta(),
            finish_reason=completion.finish_reason or "stop",
            include_usage=include_usage,
        )
    if include_usage and last is not None:
        yield _usage_chunk(
            response_id, created, model,
            _wire_usage(generation_request.prompt, last.completions, last.usage),
        )
    if http_request is not None and last is not None:  # S3: streaming was never metered
        _record_usage(http_request, model, last.usage)
    yield "data: [DONE]\n\n"


async def _stream_orchestrator(
    orchestrator, prompt: str, request: ChatCompletionRequest,
    include_usage: bool, want_trace: bool,
) -> AsyncIterator[str]:
    """AUTO-model SSE (m11 D1/A2): status keep-alives ride SSE COMMENT lines
    (the OpenAI SDK parses every data: payload as a chunk), deltas and the
    final chunk are standard chat chunks."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    first = True
    final_result = None
    try:
        stream = await orchestrator.run_chat(prompt, stream=True)
        async for event in stream:
            if event.kind == "status":
                yield f": status {event.text}\n\n"  # SSE comment (A2)
            elif event.kind == "delta":
                delta = (
                    ChunkDelta(role="assistant", content=event.text)
                    if first
                    else ChunkDelta(content=event.text)
                )
                first = False
                yield _sse_chunk(
                    response_id, created, request.model, 0, delta,
                    include_usage=include_usage,
                )
            else:
                final_result = event.result
    except Exception as error:  # surface as an SSE error event, then close
        logger.exception("orchestrator stream error")
        yield f"data: {{\"error\": {{\"message\": \"{type(error).__name__}\"}}}}\n\n"
        yield "data: [DONE]\n\n"
        return
    yield _sse_chunk(
        response_id, created, request.model, 0, ChunkDelta(),
        finish_reason="stop", include_usage=include_usage,
    )
    if include_usage and final_result is not None:
        usage = Usage(
            prompt_tokens=final_result.prompt_tokens,
            completion_tokens=final_result.completion_tokens,
            total_tokens=final_result.prompt_tokens + final_result.completion_tokens,
        )
        yield _usage_chunk(response_id, created, request.model, usage)
    if want_trace and final_result is not None:
        yield f": trace {' | '.join(final_result.trace)}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_choices(
    choices: list[Choice], model: str, usage: Usage | None = None
) -> AsyncIterator[str]:
    """Stream already-final choices (orchestrated or tool-call responses)."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    include_usage = usage is not None
    for choice in choices:
        tool_calls = None
        if choice.message.tool_calls:
            # attach the required per-item index so SDK stream accumulators merge
            # the tool-call fragments correctly (S6)
            tool_calls = [
                ChunkToolCall(index=i, id=tc.id, type=tc.type, function=tc.function)
                for i, tc in enumerate(choice.message.tool_calls)
            ]
        yield _sse_chunk(
            response_id, created, model, choice.index,
            ChunkDelta(
                role="assistant",
                content=choice.message.content,
                tool_calls=tool_calls,
            ),
            include_usage=include_usage,
        )
        yield _sse_chunk(
            response_id, created, model, choice.index, ChunkDelta(),
            finish_reason=choice.finish_reason,
            include_usage=include_usage,
        )
    if usage is not None:
        yield _usage_chunk(response_id, created, model, usage)
    yield "data: [DONE]\n\n"


def _completion_logprobs(completion: CompletionOutput) -> CompletionLogprobs | None:
    """Legacy four-parallel-array shape (m9 D3); offsets from 0 within text."""
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
        offset += len(entry.token)
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


async def _stream_completions(
    engine: EngineBackend, generation_request: GenerationRequest, request: CompletionRequest
) -> AsyncIterator[str]:
    """Legacy text_completion stream: cumulative text deltas, not delta objects."""
    response_id = f"cmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    include_usage = bool(request.stream_options and request.stream_options.include_usage)
    sent: dict[int, int] = {}
    last = None

    def _chunk(choices: list[CompletionChoice], usage: Usage | None = None) -> str:
        payload = CompletionChunk(
            id=response_id, created=created, model=request.model,
            choices=choices, usage=usage,
        )
        exclude = None if include_usage else {"usage"}
        return f"data: {payload.model_dump_json(exclude=exclude)}\n\n"

    try:
        async for partial in engine.stream(generation_request):
            last = partial
            for completion in partial.completions:
                delta = completion.text[sent.get(completion.index, 0):]
                if not delta and not partial.finished:
                    continue
                sent[completion.index] = len(completion.text)
                finish = (completion.finish_reason or "stop") if partial.finished else None
                yield _chunk(
                    [CompletionChoice(index=completion.index, text=delta, finish_reason=finish)]
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
    if include_usage and last is not None:
        yield _chunk([], usage=_wire_usage(generation_request.prompt, last.completions, last.usage))
    yield "data: [DONE]\n\n"


def _ledger_of(http_request: Request):
    return getattr(http_request.app.state, "usage_ledger", None)


def _record_usage(http_request: Request, model: str, usage) -> None:
    """m11 D3/A7: metering happens in handlers (middleware can't see tokens)."""
    if usage is None:
        return
    tenant = getattr(http_request.state, "tenant", None) or "default"
    ledger = _ledger_of(http_request)
    if ledger is not None:
        ledger.record(tenant, model, usage.prompt_tokens, usage.completion_tokens)
    # charge the tenant's per-minute token budget so it is actually enforced (S4)
    limiter = getattr(http_request.app.state, "tenant_limiter", None)
    if limiter is not None:
        limiter.charge_tokens(tenant, usage.prompt_tokens + usage.completion_tokens)


def _session_id(request: ChatCompletionRequest, http_request: Request) -> str | None:
    """Session for ReplicaPool affinity: X-Session-ID header, else the OpenAI user field."""
    return http_request.headers.get("x-session-id") or request.user


def create_app(
    engines: Mapping[str, EngineBackend],
    orchestrator: Orchestrator | None = None,
    settings: ServerSettings | None = None,
    lifespan=None,
    chat_templates: Mapping[str, ChatTemplate] | None = None,
    orchestrators: Mapping[str, Orchestrator] | None = None,
    tenant_config=None,
    embedding_backend=None,
) -> FastAPI:
    settings = settings or ServerSettings()
    app = FastAPI(title="kairyu", version="0.1.0", lifespan=lifespan)
    served_engines = dict(engines)
    # m11 D2: tiered auto models; the single-orchestrator kwarg is a shim
    auto_models: dict[str, Orchestrator] = dict(orchestrators or {})
    if orchestrator is not None:
        auto_models.setdefault(AUTO_MODEL, orchestrator)
    collisions = set(auto_models) & set(served_engines)
    if collisions:
        raise ValueError(f"orchestrator names collide with engines: {sorted(collisions)}")

    metrics = ServerMetrics() if settings.metrics else None
    app.state.metrics = metrics
    if metrics is not None:
        for name, engine in served_engines.items():
            if isinstance(engine, ReplicaPool):
                metrics.track_pool(name, engine)
    add_health_routes(app, served_engines, metrics, admin_keys=settings.resolve_admin_keys())
    from kairyu.entrypoints.server.extra_routes import add_extra_routes

    add_extra_routes(
        app, served_engines, embedding_backend=embedding_backend,
        chat_templates=chat_templates,
    )

    # add_middleware prepends, so add innermost first: metrics -> concurrency
    # guard -> auth -> access log (outermost).
    if metrics is not None:
        app.add_middleware(MetricsMiddleware, metrics=metrics)
    if settings.max_concurrency is not None:
        app.add_middleware(ConcurrencyLimitMiddleware, limit=settings.max_concurrency)
    if tenant_config is not None:
        from kairyu.entrypoints.server.tenancy import (
            TenantLimiter,
            TenantLimitMiddleware,
        )

        # added BEFORE auth => auth wraps it: 401 wins over 429 and
        # unauthenticated requests never drain buckets (m11 A6)
        limiter = TenantLimiter(tenant_config)
        app.state.tenant_limiter = limiter  # handlers charge tokens post-response (S4)
        app.add_middleware(
            TenantLimitMiddleware,
            config=tenant_config,
            limiter=limiter,
        )
    api_keys = settings.resolve_api_keys()
    if api_keys:
        app.add_middleware(
            AuthMiddleware, api_keys=api_keys, protect_metrics=settings.protect_metrics
        )
    ledger = None
    if settings.usage_ledger_path:
        from kairyu.entrypoints.server.tenancy import UsageLedger

        ledger = UsageLedger(settings.usage_ledger_path)
        app.state.usage_ledger = ledger

        @app.get("/admin/usage")
        async def admin_usage(http_request: Request, tenant: str | None = None):
            """Scoped to the CALLER's tenant when tenancy is configured
            (security review: no cross-tenant disclosure); single-tenant
            deployments (no tenant_config) see everything behind auth."""
            if tenant_config is not None:
                caller = getattr(http_request.state, "tenant", None)
                if caller is None:
                    caller = tenant_config.default_tenant
                if tenant is not None and tenant != caller:
                    return JSONResponse(
                        status_code=403,
                        content={"error": {
                            "message": "cannot query another tenant's usage",
                            "type": "invalid_request_error",
                            "code": "tenant_forbidden",
                        }},
                    )
                return {"usage": ledger.totals(caller)}
            return {"usage": ledger.totals(tenant)}

    if settings.tracing:
        from kairyu.entrypoints.server.middleware import TracingMiddleware
        from kairyu.telemetry import configure_tracing

        configure_tracing(True)
        app.add_middleware(TracingMiddleware)
    if settings.access_log:
        app.add_middleware(AccessLogMiddleware)

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        names = list(served_engines) + list(auto_models)
        return ModelList(data=[ModelCard(id=name) for name in names])

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, http_request: Request):
        http_request.state.model = request.model  # label for the metrics middleware
        if request.stream_options is not None and not request.stream:
            return _invalid_request("stream_options is only allowed when stream is true")
        if request.top_logprobs is not None and not request.logprobs:
            return _invalid_request("top_logprobs requires logprobs to be true")
        if request.top_logprobs is not None and not 0 <= request.top_logprobs <= 20:
            return _invalid_request("top_logprobs must be between 0 and 20")
        format_error = _validate_response_format(request.response_format)
        if format_error is not None:
            return _invalid_request(format_error)
        include_usage = bool(request.stream_options and request.stream_options.include_usage)
        # m11 D5: image parts need a vision engine; wire format only for now
        from kairyu.entrypoints.chat_template import flatten_content

        for message in request.messages:
            _, has_images = flatten_content(message.content)
            if has_images:
                return _invalid_request(
                    f"model {request.model!r} does not support image inputs"
                )
        # rendered after model resolution: templates are per served model (m9 D2)
        prompt = render_prompt(request, chat_templates)
        if request.model in auto_models:
            # the orchestrator path takes only the prompt; params it cannot honor
            # must be a 400, not silently dropped (M4 — OpenAI-compat by refusal)
            unsupported = [
                name
                for name, active in (
                    ("n>1", request.n > 1),
                    ("logprobs", bool(request.logprobs)),
                    ("tools", bool(request.tools)),
                    ("response_format", request.response_format is not None),
                )
                if active
            ]
            if unsupported:
                return _invalid_request(
                    f"model {request.model!r} (orchestrated) does not support: "
                    + ", ".join(unsupported)
                )
            selected = auto_models[request.model]
            want_trace = http_request.headers.get("x-kairyu-trace") == "1"
            if request.stream:
                return StreamingResponse(
                    _stream_orchestrator(
                        selected, prompt, request, include_usage, want_trace
                    ),
                    media_type="text/event-stream",
                )
            try:
                result = await selected.run(prompt)
            except Exception as error:
                return _upstream_error(error)
            completions = (
                CompletionOutput(index=0, text=result.text, token_ids=(), finish_reason="stop"),
            )
            # m11 A1/A3: REAL summed usage replaces the m9 usage=None fallback
            usage = GenerationUsage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )
            response = completion_response(request, prompt, completions, usage=usage)
            if want_trace:
                response = response.model_copy(update={"kairyu_trace": list(result.trace)})
            _record_usage(http_request, request.model, response.usage)
            return response

        engine = served_engines.get(request.model)
        if engine is None:
            return _model_not_found(request.model)
        if request.n > 1 and getattr(engine, "supports_n", True) is False:
            return _invalid_request(f"model {request.model!r} does not support n > 1")
        session_id = _session_id(request, http_request)
        try:
            sampling = sampling_params_from(request)  # bad temperature/top_p/n -> 400
        except ValueError as error:
            return _invalid_request(str(error))
        generation_request = GenerationRequest(
            request_id=f"http-{uuid.uuid4().hex[:12]}",
            prompt=prompt,
            sampling_params=sampling,
            # Affinity glue (m7 D6): keeps a session's turns on the replica
            # holding its warm radix-KV prefix.
            cache_hint=CacheHint(session_id=session_id) if session_id else None,
        )
        if request.stream and not request.tools:
            return StreamingResponse(
                _stream_engine(
                    engine, generation_request, request.model, request, http_request
                ),
                media_type="text/event-stream",
            )
        try:
            result = await engine.generate(generation_request)
        except Exception as error:
            return _upstream_error(error)
        response = completion_response(request, prompt, result.completions, result.usage)
        _record_usage(http_request, request.model, response.usage)
        if request.stream:
            # Tool calling + streaming: generate fully, then emit structured chunks so
            # tool_calls and finish_reason stay correct.
            return StreamingResponse(
                _stream_choices(
                    response.choices,
                    request.model,
                    usage=response.usage if include_usage else None,
                ),
                media_type="text/event-stream",
            )
        return response

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest, http_request: Request):
        http_request.state.model = request.model
        extra = request.model_extra or {}
        for unsupported in ("echo", "suffix", "best_of"):
            if extra.get(unsupported) is not None:
                return _invalid_request(f"{unsupported} is not supported")
        if request.logprobs is not None and not 0 <= request.logprobs <= 5:
            return _invalid_request("logprobs must be between 0 and 5")
        if request.stream_options is not None and not request.stream:
            return _invalid_request("stream_options is only allowed when stream is true")
        if isinstance(request.prompt, list) and request.stream:
            return _invalid_request("streaming with a prompt array is not supported")
        engine = served_engines.get(request.model)
        if engine is None:
            return _model_not_found(request.model)
        if request.n > 1 and getattr(engine, "supports_n", True) is False:
            return _invalid_request(f"model {request.model!r} does not support n > 1")
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        try:
            sampling = SamplingParams(  # invalid params are a client error, not a 502
                temperature=request.temperature,
                top_p=request.top_p,
                n=request.n,
                max_tokens=request.max_tokens,
                stop=request.stop,
                seed=request.seed,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                logprobs=request.logprobs,
            )
        except ValueError as error:
            return _invalid_request(str(error))

        def _generation_request(prompt: str) -> GenerationRequest:
            return GenerationRequest(
                request_id=f"http-{uuid.uuid4().hex[:12]}",
                prompt=prompt,
                sampling_params=sampling,
            )

        if request.stream:
            return StreamingResponse(
                _stream_completions(engine, _generation_request(prompts[0]), request),
                media_type="text/event-stream",
            )
        choices: list[CompletionChoice] = []
        usage_totals = [0, 0, 0]  # prompt, completion, cached
        try:
            # run the prompt array concurrently (latency = max, not sum); order is
            # restored by prompt_index below so the response is unchanged (P-perf)
            results = await asyncio.gather(
                *(engine.generate(_generation_request(prompt)) for prompt in prompts)
            )
        except Exception as error:
            return _upstream_error(error)
        for prompt_index, result in enumerate(results):
            for completion in result.completions:
                choices.append(
                    _completion_choice(
                        prompt_index * request.n + completion.index, completion
                    )
                )
            if result.usage is not None:
                usage_totals[0] += result.usage.prompt_tokens
                usage_totals[1] += result.usage.completion_tokens
                usage_totals[2] += result.usage.cached_tokens
        details = (
            PromptTokensDetails(cached_tokens=usage_totals[2]) if usage_totals[2] else None
        )
        _record_usage(  # S3: /v1/completions was never metered
            http_request, request.model,
            GenerationUsage(
                prompt_tokens=usage_totals[0], completion_tokens=usage_totals[1],
                cached_tokens=usage_totals[2],
            ),
        )
        return CompletionResponse(
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

    return app
