"""EngineBackend protocol and its request/result types.

Every layer above (Router, Conductor, MoA, LLM entrypoint, server) depends only
on this module, never on a concrete engine. The M2 custom engine plugs in here.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from kairyu.engine.prompt import (
    PromptInput,
    TemplatedPrompt,
    TextPrompt,
    prompt_kind,
    prompt_text,
    supplied_prompt_token_ids,
)
from kairyu.outputs import CompletionOutput
from kairyu.sampling_params import (
    STRUCTURED_OUTPUT_EXTRA_ARGS,
    SamplingParams,
    resolve_parallel_tool_calls,
    validate_prompt_owned_extra_args,
)

_GENERATION_STAGE_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_TOOL_FUNCTION_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_MAX_STAGE_DURATION_NS = (1 << 63) - 1
_MAX_STAGE_OCCURRENCES = (1 << 31) - 1

# Reservation-only decode ceiling for OpenAI-style limitless requests
# (max_tokens omitted) against a backend whose max_model_len is unknown.
# Generation itself stays bounded by the serving model's context upstream;
# admission merely needs a finite reservation, which exact usage settles.
UNLIMITED_OUTPUT_ADMISSION_TOKENS = 8192


class Shutdownable(Protocol):
    async def shutdown(self) -> None: ...


async def shutdown_all(resources: Iterable[Shutdownable], label: str) -> None:
    """Shutdown each unique resource and report every failed cleanup."""
    unique = list({id(resource): resource for resource in resources}.values())
    results = await asyncio.gather(
        *(resource.shutdown() for resource in unique), return_exceptions=True
    )
    errors: list[Exception] = []
    for resource, result in zip(unique, results, strict=True):
        if isinstance(result, Exception):
            errors.append(result)
        elif isinstance(result, BaseException):
            # ``gather(return_exceptions=True)`` returns a child coroutine's
            # self-cancellation as a value. Do not silently treat an unfinished
            # resource cleanup as success, but keep the public aggregate an
            # ExceptionGroup so ordinary lifecycle handlers can report it.
            error = RuntimeError(
                f"{type(resource).__name__}.shutdown raised "
                f"{type(result).__name__}"
            )
            error.__cause__ = result
            errors.append(error)
    if errors:
        raise ExceptionGroup(f"{label} shutdown failed", errors)


async def shutdown_all_cancellation_safe(
    resources: Iterable[Shutdownable],
    label: str,
) -> None:
    """Finish shutdown of detached resources before propagating cancellation.

    Once an owner removes a resource from its live registry, no later lifecycle
    pass can find it. Shield that final shutdown and preserve cancellation as
    the outward result, chaining any cleanup failure for diagnostics.
    """

    cleanup = asyncio.create_task(shutdown_all(resources, label))
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError as cancelled:
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        try:
            cleanup.result()
        except BaseException as cleanup_error:
            raise cancelled from cleanup_error
        raise


class UpstreamClientError(Exception):
    """A backend rejected the request itself (HTTP 4xx): the client's request
    was bad, NOT a sign the replica is unhealthy. The ReplicaPool must not count
    it as a replica failure, or one malformed client could eject the fleet (O1).
    """

    def __init__(
        self,
        message: str,
        status_code: int,
        *,
        code: str = "invalid_request",
        public_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        # Upstream response bodies and replica URLs may contain credentials or
        # internal topology. Only locally constructed validation errors opt in
        # to exposing a separately reviewed message at the tenant boundary.
        self.public_message = public_message


@dataclass(frozen=True)
class CacheHint:
    """KV-affinity hint plumbed through now, consumed by the M2 Radix KV manager.

    ``session_id`` groups requests of one orchestration. ``prefix_fingerprint``
    is an optional trusted ``xxh3-64-v1`` key for the first complete
    256-character shared-text chunk. It is rejected for token and multimodal
    prompts rather than mixing cache-key domains. An empty value declares
    session-only affinity; prefix-enabled pools do not speculate about
    cross-session reuse.
    """

    session_id: str
    prefix_fingerprint: str = ""


@dataclass(frozen=True)
class GenerationStageMetric:
    """One cumulative, request-scoped native generation stage observation."""

    stage: str
    duration_ns: int
    occurrences: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.stage) is not str
            or _GENERATION_STAGE_NAME_RE.fullmatch(self.stage) is None
        ):
            raise ValueError("generation stage must be a bounded safe identifier")
        if (
            type(self.duration_ns) is not int
            or not 0 <= self.duration_ns <= _MAX_STAGE_DURATION_NS
        ):
            raise ValueError(
                "generation stage duration_ns must be a bounded non-negative integer"
            )
        if (
            type(self.occurrences) is not int
            or not 1 <= self.occurrences <= _MAX_STAGE_OCCURRENCES
        ):
            raise ValueError(
                "generation stage occurrences must be a bounded positive integer"
            )


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    prompt: PromptInput
    sampling_params: SamplingParams
    # vLLM-compatible scheduling priority: smaller integers run first.
    priority: int = 0
    # Bounded semantic class for metrics and trusted Kairyu-to-Kairyu transport.
    # Scheduling itself uses ``priority``; arbitrary priority integers never
    # become metric labels.
    scheduling_class: str = "interactive"
    cache_hint: CacheHint | None = None
    tools: tuple[Mapping[str, object], ...] = ()
    tool_choice: str | Mapping[str, object] | None = None
    tools_in_prompt: bool = False
    # Process-local gateway timestamp, never serialized to an upstream. When
    # present, ReplicaPool records request-ingress-to-selection latency rather
    # than only the pure hash/least-load function cost (G5 F1a).
    placement_started_ns: int | None = None
    # Explicit diagnostic opt-in. Native engines leave their timing hot path
    # inert unless the public request asks for the structured stage trace.
    trace_requested: bool = False
    # OpenAI Chat Completions execution hint. Appended after the pre-existing
    # fields so positional callers retain their historical argument binding.
    # The public boundary still enforces cardinality after generation.
    parallel_tool_calls: bool | None = None
    # Parser-attested chat-template family used only by native strict-tool
    # grammar construction. Engine layers intentionally keep this as a small
    # string instead of importing the L3 enum.
    tool_call_protocol: str = "generic"
    reasoning_effort: str | None = None
    # Per-request variables for an upstream-owned HF chat template (text or
    # multimodal chat prompts; never pre-rendered or token prompts).
    chat_template_kwargs: Mapping[str, object] | None = None
    # Assistant-turn continuation for an upstream-templated chat prompt: the
    # upstream renders this as the final assistant message and generation
    # continues it (vLLM ``continue_final_message``). Used by the
    # public-output floor on chat-template workers whose reasoning span the
    # template itself opens (DTO-D15).
    assistant_prefill: str | None = None

    def __post_init__(self) -> None:
        # Defense in depth for callers holding a SamplingParams created by an
        # older process or deliberately altered through low-level reflection.
        validate_prompt_owned_extra_args(self.sampling_params.extra_args)
        resolve_parallel_tool_calls(
            self.parallel_tool_calls,
            self.sampling_params.extra_args,
        )
        if type(self.trace_requested) is not bool:
            raise ValueError("trace_requested must be a boolean")
        if self.tool_call_protocol not in {"generic", "llama", "qwen", "deepseek_v4"}:
            raise ValueError(
                "tool_call_protocol must be generic, llama, qwen or deepseek_v4"
            )
        if self.reasoning_effort not in {None, "low", "high", "max"}:
            raise ValueError("reasoning_effort must be low, high, max, or null")
        if self.chat_template_kwargs is not None:
            if not isinstance(self.chat_template_kwargs, Mapping) or any(
                not isinstance(key, str) or not key
                for key in self.chat_template_kwargs
            ):
                raise TypeError(
                    "chat_template_kwargs must map non-empty string keys or be null"
                )
            object.__setattr__(
                self,
                "chat_template_kwargs",
                dict(self.chat_template_kwargs),
            )
        kind = prompt_kind(self.prompt)
        if self.assistant_prefill is not None:
            if not isinstance(self.assistant_prefill, str) or not self.assistant_prefill:
                raise ValueError("assistant_prefill must be a non-empty string or null")
            if kind == "tokens" or isinstance(self.prompt, TemplatedPrompt):
                raise ValueError(
                    "assistant_prefill requires an upstream-templated chat prompt "
                    "(text or multimodal), not a pre-rendered or token prompt"
                )
        if self.chat_template_kwargs is not None and (
            kind == "tokens" or isinstance(self.prompt, TemplatedPrompt)
        ):
            # Pre-rendered and token prompts bypass the upstream chat template,
            # so template variables would be silently ignored.
            raise ValueError(
                "chat_template_kwargs require an upstream-templated chat prompt "
                "(text or multimodal), not a pre-rendered or token prompt"
            )
        if (
            kind != "text"
            and self.cache_hint is not None
            and self.cache_hint.prefix_fingerprint
        ):
            raise ValueError(
                "CacheHint.prefix_fingerprint is an xxh3 fingerprint of a "
                "text prefix and cannot be used with token or multimodal prompts; "
                "use a session-only CacheHint instead"
            )


def validate_backend_request(backend: object, request: GenerationRequest) -> None:
    """Run an optional backend validator with a fail-closed typed default.

    Backends predating the typed prompt contract remain compatible with legacy
    text. They must explicitly implement ``validate_request`` before token or
    multimodal values can cross their boundary.
    """

    validate = getattr(backend, "validate_request", None)
    if validate is not None:
        validate(request)
        return
    if type(request.prompt) is not str:
        kind = prompt_kind(request.prompt)
        variant = (
            "typed text"
            if isinstance(request.prompt, (TextPrompt, TemplatedPrompt))
            else kind
        )
        raise ValueError(
            f"{type(backend).__name__} does not declare support for {variant} "
            "prompts; backends without validate_request are legacy-string text-only"
        )


def validate_backend_request_before_prepare(
    backend: object,
    request: GenerationRequest,
) -> None:
    """Run cheap synchronous checks before bounded async preparation.

    Most backends use their ordinary validator. A backend whose complete
    validation contains request-sized pure CPU work may expose
    ``validate_request_before_prepare`` for structural/state checks and finish
    the expensive validation in ``prepare_request`` off the event loop.
    """

    validate = getattr(backend, "validate_request_before_prepare", None)
    if callable(validate):
        backend_type = type(backend)

        def owner(name: str) -> type | None:
            return next(
                (
                    candidate
                    for candidate in backend_type.__mro__
                    if name in candidate.__dict__
                ),
                None,
            )

        fast_owner = owner("validate_request_before_prepare")
        full_owner = owner("validate_request")
        # An inherited builtin fast hook cannot bypass a more-derived custom
        # validator. Subclasses opt into the split only by overriding the fast
        # hook at least as specifically as their full validator.
        if (
            fast_owner is None
            or full_owner is None
            or backend_type.__mro__.index(fast_owner)
            <= backend_type.__mro__.index(full_owner)
        ):
            validate(request)
            return
    validate_backend_request(backend, request)


@dataclass(frozen=True)
class AdmissionUpperBound:
    """Worst-case shared-engine work reserved before dispatch."""

    tokens: int
    # Standard usage counts prompt tokens once even when n/best_of duplicates
    # prefill work.  Only single-candidate requests may refund to wire usage.
    refundable_on_exact_usage: bool


def render_tool_intent(
    prompt: PromptInput,
    *,
    tools,
    tool_choice,
    tools_in_prompt: bool,
) -> PromptInput:
    """Render native-engine tool intent exactly once when no HF template did.

    A pre-tokenized prompt is caller-owned: adding a text suffix would silently
    mix two tokenizer owners. Multimodal prompts likewise cannot be flattened
    into text without dropping modality data. Pure function of its arguments:
    billing, admission, and /v1/messages/count_tokens all hash the same bytes.
    """

    if not tools or tools_in_prompt or tool_choice == "none":
        return prompt
    if isinstance(prompt, TemplatedPrompt):
        raise ValueError(
            "templated prompts cannot receive an implicit tool-instruction suffix; "
            "render tools inside the chat template and set tools_in_prompt=true"
        )
    kind = prompt_kind(prompt)
    if kind != "text":
        raise ValueError(
            f"{kind} prompts cannot receive an implicit tool-instruction suffix; "
            "render tools before tokenization and set tools_in_prompt=true"
        )
    text = prompt_text(prompt)
    assert text is not None
    choice = tool_choice
    if isinstance(choice, Mapping):
        named = (choice.get("function") or {}).get("name")
        policy = f"You must call the function {named!r}."
    elif choice == "required":
        policy = "You must call one of the available functions."
    else:
        policy = "Call a function when it is useful."
    schemas = json.dumps(
        list(tools),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    rendered = (
        f"{text}\n\nAvailable functions:\n{schemas}\n"
        f"{policy} Emit each call exactly as "
        '<tool_call>{"name":"function_name","arguments":{}}</tool_call>.'
    )
    return TextPrompt(rendered) if isinstance(prompt, TextPrompt) else rendered


def prompt_with_tool_intent(request: GenerationRequest) -> PromptInput:
    """Render the request's tool intent (see ``render_tool_intent``)."""

    return render_tool_intent(
        request.prompt,
        tools=request.tools,
        tool_choice=request.tool_choice,
        tools_in_prompt=request.tools_in_prompt,
    )


async def backend_count_prompt_tokens_async(
    backend: object, prompt: str
) -> int | None:
    """Probe-count prompt tokens for ``/v1/messages/count_tokens``.

    ``None`` is a first-class "declined" answer for backends that cannot
    provide an authoritative count.
    """

    counter = getattr(backend, "count_prompt_tokens_async", None)
    if not callable(counter):
        return None
    count = await counter(prompt)
    if count is None:
        return None
    if type(count) is not int or count < 0:
        raise TypeError(
            "backend count_prompt_tokens_async must return a non-negative int or None"
        )
    return count


def _strict_tool_response_format(
    request: GenerationRequest,
) -> dict[str, object] | None:
    """Build the parser-matched xgrammar format for selected strict tools."""

    if not request.tools:
        return None
    choice = request.tool_choice
    candidates: list[tuple[int, Mapping[str, object], Mapping[str, object]]] = []
    for index, tool in enumerate(request.tools):
        function = tool.get("function")
        if not isinstance(function, Mapping):
            continue
        strict = function.get("strict")
        if strict is not None and type(strict) is not bool:
            raise ValueError(f"tools[{index}].function.strict must be a boolean")
        candidates.append((index, tool, function))

    if isinstance(choice, Mapping):
        function_choice = choice.get("function")
        selected_name = (
            function_choice.get("name")
            if isinstance(function_choice, Mapping)
            else None
        )
        candidates = [
            candidate
            for candidate in candidates
            if candidate[2].get("name") == selected_name
        ]

    if choice == "none":
        return None
    if not any(
        function.get("strict") is True
        for _index, _tool, function in candidates
    ):
        return None

    functions: list[Mapping[str, object]] = []
    for index, tool, function in candidates:
        if tool.get("type") != "function":
            raise ValueError(f"tools[{index}].type must be 'function'")
        name = function.get("name")
        if not isinstance(name, str) or _TOOL_FUNCTION_NAME_RE.fullmatch(name) is None:
            raise ValueError(
                f"tools[{index}].function.name must match [A-Za-z0-9_-]{{1,64}}"
            )
        functions.append(function)

    protocol = request.tool_call_protocol
    tags: list[dict[str, object]] = []
    for function in functions:
        name = function["name"]
        parameters = function.get("parameters")
        if function.get("strict") is True:
            if not isinstance(parameters, Mapping):
                raise ValueError(
                    f"strict tool {name!r} requires a JSON-schema parameters object"
                )
            schema: object = dict(parameters)
        else:
            schema = True
        style = "json"
        if protocol == "generic":
            begin = f'<tool_call>{{"name":{json.dumps(name)},"arguments":'
            end = "}</tool_call>"
        elif protocol == "llama":
            begin = f'<|python_tag|>{{"name": {json.dumps(name)}, "parameters": '
            end = "}"
        elif protocol == "qwen":
            begin = f"<tool_call>\n<function={name}>\n"
            end = "\n</function>\n</tool_call>"
            style = "qwen_xml"
        else:
            begin = f'<｜DSML｜invoke name="{name}">\n'
            end = "</｜DSML｜invoke>\n"
            style = "deepseek_xml"
        tags.append(
            {
                "type": "tag",
                "begin": begin,
                "content": {
                    "type": "json_schema",
                    "json_schema": schema,
                    "style": style,
                    "any_order": False,
                },
                "end": end,
            }
        )

    parallel = resolve_parallel_tool_calls(
        request.parallel_tool_calls,
        request.sampling_params.extra_args,
    )
    at_least_one = choice == "required" or isinstance(choice, Mapping)
    if protocol == "deepseek_v4":
        return {
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": ["<｜DSML｜tool_calls>"],
                "tags": [
                    {
                        "type": "tag",
                        "begin": "<｜DSML｜tool_calls>\n",
                        "content": {
                            "type": "tags_with_separator",
                            "tags": tags,
                            "separator": "",
                            "at_least_one": True,
                            "stop_after_first": parallel is False,
                        },
                        "end": "</｜DSML｜tool_calls>",
                    }
                ],
                "at_least_one": at_least_one,
                "stop_after_first": True,
                "excludes": [],
            },
        }

    trigger = {
        "generic": "<tool_call>",
        "llama": "<|python_tag|>",
        "qwen": "<tool_call>\n<function=",
    }[protocol]
    return {
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": [trigger],
            "tags": tags,
            "at_least_one": at_least_one,
            "stop_after_first": protocol == "llama" or parallel is False,
            "excludes": [],
        },
    }


def native_sampling_params(request: GenerationRequest) -> SamplingParams:
    """Return native sampling with strict tool semantics made executable."""

    response_format = _strict_tool_response_format(request)
    if response_format is None:
        return request.sampling_params
    extra_args = dict(request.sampling_params.extra_args)
    explicit = extra_args.get("response_format")
    if isinstance(explicit, Mapping) and explicit.get("type") not in {None, "text"}:
        raise ValueError(
            "strict tools cannot be combined with a structured response_format"
        )
    extra_args["response_format"] = response_format
    return request.sampling_params.clone(extra_args=extra_args)


def validate_native_request_surface_before_prepare(
    request: GenerationRequest,
) -> None:
    """Reject fixed-size native fields without walking request collections.

    Tool schemas and vendor ``extra_args`` can be arbitrarily large.  Their
    complete validation belongs in the bounded prompt worker alongside prompt
    rendering/tokenization, while these scalar checks remain safe to run on
    the serving event loop.
    """

    params = request.sampling_params
    unsupported: list[str] = []
    if params.best_of is not None:
        unsupported.append("best_of")
    if params.prompt_logprobs is not None:
        unsupported.append("prompt_logprobs")
    if request.chat_template_kwargs is not None:
        unsupported.append("chat_template_kwargs")
    if request.assistant_prefill is not None:
        unsupported.append("assistant_prefill")
    if not isinstance(params.extra_args, Mapping):
        unsupported.append("extra_args")
    if unsupported:
        raise ValueError(
            "Kairyu backend does not support request fields: "
            + ", ".join(sorted(unsupported))
        )


def validate_native_request_surface(request: GenerationRequest) -> None:
    """Reject fields the native engine cannot honor instead of dropping them.

    Both native process layouts consume the same ``EngineLoop`` sampling
    surface. Keeping this check transport-neutral prevents the in-process and
    ZMQ entry points from silently diverging when the public API gains a field.
    """

    validate_native_request_surface_before_prepare(request)
    params = request.sampling_params
    unsupported = [
        f"extra_args.{key}"
        for key in params.extra_args
        if key not in STRUCTURED_OUTPUT_EXTRA_ARGS
    ]
    native_sampling_params(request)
    if unsupported:
        raise ValueError(
            "Kairyu backend does not support request fields: "
            + ", ".join(sorted(unsupported))
        )


def admission_upper_bound(
    request: GenerationRequest,
    *,
    fallback_output_tokens: int | None = None,
) -> AdmissionUpperBound:
    """Transport-neutral, no-I/O upper bound for one generation.

    The gateway intentionally does not own model tokenizers.  We count the
    complete native prompt/tool intent, response-format metadata, and a fixed
    chat-template envelope in UTF-8 work units, then multiply both prefill and
    decode by the actual candidate fan-out.  This stays O(request bytes) and
    avoids placing tenant bookkeeping in the scheduler token hot path.

    ``max_tokens=None`` follows the OpenAI chat contract (generation bounded
    by the model's remaining context), so the decode reservation falls back
    to ``fallback_output_tokens`` — the backend's ``max_model_len`` when
    known — or ``UNLIMITED_OUTPUT_ADMISSION_TOKENS``. The fallback sizes the
    reservation only; exact usage settles the difference afterwards.
    """

    params = request.sampling_params
    max_tokens = params.max_tokens
    if max_tokens is None:
        max_tokens = fallback_output_tokens or UNLIMITED_OUTPUT_ADMISSION_TOKENS
    candidates = max(params.n, params.best_of or params.n)
    prompt = prompt_with_tool_intent(request)
    metadata = json.dumps(
        params.extra_args,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    kind = prompt_kind(prompt)
    if kind == "multimodal":
        # A media processor can expand one item into a model-specific number of
        # placeholder tokens. Guessing from bytes would under-reserve some
        # models and overstate usage for others.
        raise ValueError(
            "multimodal prompt admission requires a backend-reported processed "
            "token count"
        )
    token_ids = supplied_prompt_token_ids(prompt)
    if token_ids is not None:
        # Pretokenized inputs are complete: the caller owns BOS, templates, and
        # tool rendering. Their exact sequence length is therefore the truthful
        # prefill reservation; optional display text is never counted.
        prompt_upper = len(token_ids)
    else:
        text = prompt_text(prompt)
        assert text is not None
        # Covers the fixed role/control tokens inserted by common HF/OpenAI chat
        # templates. Supplied prompt, tool schemas, and response schema are
        # counted explicitly above.
        fixed_template_envelope = 256
        prompt_upper = max(
            1,
            len(text.encode("utf-8"))
            + len(metadata.encode("utf-8"))
            + fixed_template_envelope,
        )
    return AdmissionUpperBound(
        tokens=candidates * (prompt_upper + max_tokens),
        refundable_on_exact_usage=candidates == 1,
    )


def backend_admission_upper_bound(
    backend: object,
    request: GenerationRequest,
) -> AdmissionUpperBound:
    """Use an explicit backend processor ceiling when the generic bound cannot.

    Text/token behavior remains byte-for-byte on the historical helper. A
    multimodal backend must opt in with a synchronous, I/O-free bound derived
    from its configured processor limits; media byte length is never treated as
    a token estimate.
    """

    resolve = getattr(backend, "admission_upper_bound", None)
    if callable(resolve):
        bound = resolve(request)
    else:
        bound = admission_upper_bound(
            request,
            fallback_output_tokens=getattr(backend, "max_model_len", None),
        )
    return _validated_admission_upper_bound(backend, bound)


def _validated_admission_upper_bound(
    backend: object,
    bound: object,
) -> AdmissionUpperBound:
    """Validate one optional backend admission result at every call boundary."""

    if not isinstance(bound, AdmissionUpperBound):
        raise TypeError(
            f"{type(backend).__name__}.admission_upper_bound must return "
            "AdmissionUpperBound"
        )
    if type(bound.tokens) is not int or bound.tokens < 1:
        raise ValueError("backend admission bound must contain a positive token count")
    return bound


async def backend_admission_upper_bound_async(
    backend: object,
    request: GenerationRequest,
) -> AdmissionUpperBound:
    """Resolve admission while keeping generic prompt serialization off-loop.

    Stateful composite backends can expose ``admission_upper_bound_async`` and
    snapshot their routing state on the event loop.  Synchronous overrides are
    already required to be I/O-free configured-policy calculations, so their
    request-sized serialization shares the bounded prompt lane with the pure
    transport-neutral fallback.
    """

    resolve_async = getattr(backend, "admission_upper_bound_async", None)
    if callable(resolve_async):
        bound = await resolve_async(request)
        return _validated_admission_upper_bound(backend, bound)
    resolve = getattr(backend, "admission_upper_bound", None)
    from kairyu.async_thread import run_prompt_work

    if callable(resolve):
        calculate = resolve
    else:

        def calculate(request: GenerationRequest) -> AdmissionUpperBound:
            return admission_upper_bound(
                request,
                fallback_output_tokens=getattr(backend, "max_model_len", None),
            )

    bound = await run_prompt_work(calculate, request)
    return _validated_admission_upper_bound(backend, bound)


_GENERIC_ADMISSION_CONTRACT = object()


def backend_admission_upper_bound_key(backend: object) -> object | None:
    """Return an explicit immutable key for equivalent admission semantics.

    The generic fallback depends only on ``GenerationRequest`` plus the
    backend's ``max_model_len`` (the limitless-request reservation ceiling),
    so backends sharing that ceiling share one contract. A backend override
    must opt in with its own immutable key; unknown or unhashable
    declarations are never deduped.
    """

    resolve_async = getattr(backend, "admission_upper_bound_async", None)
    resolve = getattr(backend, "admission_upper_bound", None)
    if not callable(resolve_async) and not callable(resolve):
        return (_GENERIC_ADMISSION_CONTRACT, getattr(backend, "max_model_len", None))
    key = getattr(backend, "admission_upper_bound_key", None)
    if key is None:
        return None
    typed_key = (type(backend), key)
    try:
        hash(typed_key)
    except TypeError:
        return None
    return typed_key


async def prepare_backend_request(
    backend: object,
    request: GenerationRequest,
) -> None:
    """Run optional bounded async preparation before admission/HTTP streaming."""

    prepare = getattr(backend, "prepare_request", None)
    if callable(prepare):
        await prepare(request)


@dataclass(frozen=True)
class GenerationUsage:
    """Backend-reported token accounting (m9 D1): the source of usage truth.

    ``prompt_tokens`` is counted once per request (not per completion);
    ``completion_tokens`` sums across completions; ``cached_tokens`` is the
    prompt prefix served from the radix cache.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    prompt: PromptInput
    completions: tuple[CompletionOutput, ...]
    finished: bool = True
    usage: GenerationUsage | None = None
    # Exact only when the caller supplied token IDs or the backend explicitly
    # reports the processed prompt. Text-only adapters may leave this empty.
    # Appended after the historical fields to preserve positional compatibility.
    prompt_token_ids: tuple[int, ...] = ()
    # Cumulative, stage-unique native observations. Empty means tracing was not
    # requested or the selected backend cannot provide this optional surface.
    stage_metrics: tuple[GenerationStageMetric, ...] = ()

    @property
    def text(self) -> str:
        """Convenience accessor for the first completion's text."""
        return self.completions[0].text if self.completions else ""

    @property
    def text_delta(self) -> str | None:
        """Newly visible text for the first completion, when delta-native."""
        return self.completions[0].text_delta if self.completions else None

    @property
    def text_offset(self) -> int | None:
        """Cumulative start offset of ``text_delta`` when delta-native."""
        return self.completions[0].text_offset if self.completions else None


@runtime_checkable
class EngineBackend(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]: ...

    async def shutdown(self) -> None: ...


def backend_sequence_budget(backend: object) -> int | None:
    """Return an optional backend-advertised active sequence budget."""

    budget = getattr(backend, "sequence_budget", None)
    if budget is None:
        return None
    if type(budget) is not int or budget < 1:
        raise TypeError("backend sequence_budget must be a positive integer")
    return budget


def backend_supports_slo_defer(
    backend: object,
    request: GenerationRequest | None = None,
) -> bool:
    """Resolve the running-batch isolation contract required by SLO defer."""

    request_capability = getattr(backend, "supports_slo_defer_for_request", None)
    supported = (
        request_capability(request)
        if request is not None and callable(request_capability)
        else getattr(backend, "supports_slo_defer", False)
    )
    if type(supported) is not bool:
        raise TypeError("backend supports_slo_defer must be a boolean")
    return supported


def backend_max_model_len(backend: object) -> int | None:
    """Resolve a backend's serving context length without probing a request.

    A direct engine advertises ``max_model_len``; a ``ReplicaPool`` carries it
    only as unanimous declared deployment metadata. Model-card publication
    (issue #495) needs one resolution rule for both shapes.
    """

    length = getattr(backend, "max_model_len", None)
    if isinstance(length, int) and length > 0:
        return length
    snapshot = getattr(backend, "declared_metadata_snapshot", None)
    if callable(snapshot):
        declared = snapshot().get("max_model_len")
        if isinstance(declared, int) and declared > 0:
            return declared
    return None


def backend_supports_prompt_kind(backend: object, kind: str) -> bool:
    """Resolve an optional prompt-kind capability without probing a request.

    Composite callers need this before deriving a request: in particular, an
    orchestrator must decide which workers receive structured media and which
    receive only the textual role context. Backends that do not publish the
    capability retain the typed-prompt contract's fail-closed default.
    """

    supports = getattr(backend, "supports_prompt_kind", None)
    supported = supports(kind) if callable(supports) else kind == "text"
    if type(supported) is not bool:
        raise TypeError("backend supports_prompt_kind must return a boolean")
    return supported


def backend_supports_chat_template_kwargs(
    backend: object,
    keys: frozenset[str],
) -> bool:
    """Resolve whether a backend can apply upstream HF template variables."""

    supports = getattr(backend, "supports_chat_template_kwargs", None)
    supported = supports(keys) if callable(supports) else False
    if type(supported) is not bool:
        raise TypeError("backend supports_chat_template_kwargs must return a boolean")
    return supported


@dataclass(frozen=True)
class EngineReadiness:
    """An engine's own answer to "could I serve a request right now?".

    Optional: `/readyz` only consults engines that implement ``readiness()``, so
    remote and mock backends are unaffected. It must stay CHEAP — the endpoint is
    polled by load balancers — which means reporting known-fatal state, never
    running a probe generation.

    ``fatal`` separates "stop sending work" from "replace this process". Marking a
    node unready for something it could recover from is a trap: the load balancer
    stops sending work, so the traffic that would prove recovery never arrives.
    An engine should therefore only report unready for a condition nothing
    in-process can undo — and then say so, so `/health` can ask for a restart.

    ``detail`` reaches an UNAUTHENTICATED endpoint. Exception classes and fixed
    strings only; a message can carry an upstream URL, a path, or a credential.
    """

    ready: bool
    detail: str = ""
    fatal: bool = False
