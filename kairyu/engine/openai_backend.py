"""External OpenAI-compatible API worker (OpenAI/Anthropic/Gemini compat endpoints).

Used by the Conductor for frontier-tier roles (design m1 D1) and as the remote
DP-replica member behind ReplicaPool (design m6 D2). Errors surface explicitly:
missing API key and non-2xx statuses raise RuntimeError with context.

m6 D2 fixes: one persistent pooled AsyncClient per backend (no per-request
handshake), with optional externally-owned client injection or an owned client
factory for origin-local dynamic replicas; real SSE streaming (cumulative
partials, MockBackend semantics); optional auth (``api_key_env=None`` for
keyless node-to-node replicas); and token-count passthrough (synthetic ids
carrying ``usage.completion_tokens`` / the streamed-delta count, mirroring
MockBackend's count-only ids).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import weakref
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from kairyu.async_thread import run_prompt_work
from kairyu.engine.backend import (
    UNLIMITED_OUTPUT_ADMISSION_TOKENS,
    AdmissionUpperBound,
    GenerationRequest,
    GenerationResult,
    GenerationStageMetric,
    GenerationUsage,
    UpstreamClientError,
    validate_backend_request_before_prepare,
)
from kairyu.engine.backend import (
    admission_upper_bound as default_admission_upper_bound,
)
from kairyu.engine.openai_capabilities import (
    OpenAIRequestCapabilities,
    resolve_openai_capabilities,
)
from kairyu.engine.prompt import (
    MultimodalPrompt,
    TemplatedPrompt,
    TextPrompt,
    prompt_kind,
    prompt_text,
)
from kairyu.engine.registry import register_backend
from kairyu.engine.vision import ImageInputPolicy, InvalidImageInput
from kairyu.outputs import CompletionOutput, TokenLogprob, _lazy_text
from kairyu.sampling_params import (
    GENERATION_CONFIG_SAMPLING_FIELDS,
    SamplingParams,
    resolve_parallel_tool_calls,
)
from kairyu.sse import iter_sse_data


def _raise_for_status(base_url: str, status_code: int, body: str) -> None:
    """4xx is a client-request error (not a replica health signal, O1); 5xx and
    everything else is a transport/server failure the pool should count."""
    message = f"backend {base_url} returned HTTP {status_code}: {body[:500]}"
    if 400 <= status_code < 500:
        raise UpstreamClientError(message, status_code)
    raise RuntimeError(message)


_DEFAULT_TIMEOUT_S = 60.0
_SSE_DONE = "[DONE]"
# OpenAI exposes token text and bytes in logprobs, but not tokenizer token IDs.
_UNKNOWN_TOKEN_ID = -1
_MAX_TRACE_EVENTS = 256
_MAX_TRACE_DURATION_NS = (1 << 63) - 1
_MAX_TRACE_OCCURRENCES = (1 << 31) - 1
_TRACE_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z\Z"
)


@dataclass
class _ImagePreparationFlight:
    """One exact-request image validation shared by concurrent waiters."""

    request_ref: weakref.ReferenceType[GenerationRequest]
    task: asyncio.Task[tuple[str, ...]]
    waiters: int = 0


@dataclass
class _PayloadPreparationFlight:
    """One exact-request payload validation shared by concurrent waiters."""

    request_ref: weakref.ReferenceType[GenerationRequest]
    task: asyncio.Task[bytes]
    waiters: int = 0


@dataclass
class _CompletionReasoningParser:
    """Split a completion stream that starts inside private reasoning."""

    end_tag: str
    pending: str = ""
    finished_reasoning: bool = False
    public_content_seen: bool = False

    def feed(self, delta: str) -> tuple[str, str]:
        """Return ``(private_reasoning, public_content)`` for one wire delta."""

        if self.finished_reasoning:
            if delta.strip():
                self.public_content_seen = True
            return "", delta
        self.pending += delta
        marker_at = self.pending.find(self.end_tag)
        if marker_at >= 0:
            reasoning = self.pending[:marker_at]
            content = self.pending[marker_at + len(self.end_tag) :]
            self.pending = ""
            self.finished_reasoning = True
            if content.strip():
                self.public_content_seen = True
            return reasoning, content
        keep = len(self.end_tag) - 1
        if len(self.pending) <= keep:
            return "", ""
        reasoning = self.pending[:-keep] if keep else self.pending
        self.pending = self.pending[-keep:] if keep else ""
        return reasoning, ""

    def require_finished(self) -> None:
        if not self.finished_reasoning:
            raise RuntimeError(
                "OpenAI-compatible completion ended before its configured "
                "private-reasoning terminator"
            )
        if not self.public_content_seen:
            raise RuntimeError(
                "OpenAI-compatible completion ended without public content "
                "after its private-reasoning terminator"
            )


def _split_completion_reasoning(text: str, end_tag: str) -> tuple[str, str]:
    marker_at = text.find(end_tag)
    if marker_at < 0:
        raise RuntimeError(
            "OpenAI-compatible completion omitted its configured "
            "private-reasoning terminator"
        )
    reasoning = text[:marker_at]
    content = text[marker_at + len(end_tag) :]
    if not content.strip():
        raise RuntimeError(
            "OpenAI-compatible completion ended without public content after "
            "its private-reasoning terminator"
        )
    return reasoning, content


_SHARED_PREPARED_PAYLOADS: dict[
    int,
    tuple[
        weakref.ReferenceType[GenerationRequest],
        dict[tuple[object, ...], bytes],
    ],
] = {}
_SHARED_PREPARED_PAYLOADS_LOCK = threading.Lock()


async def _await_task_cancellation_safe(task: asyncio.Task[None]) -> None:
    """Finish an owned-client close before propagating caller cancellation."""

    try:
        await asyncio.shield(task)
    except asyncio.CancelledError as cancelled:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        try:
            task.result()
        except BaseException as cleanup_error:
            raise cancelled from cleanup_error
        raise


def _usage_from(data: dict | None) -> GenerationUsage | None:
    """Parse an upstream usage object incl. prompt_tokens_details (m9 D1)."""
    if not data:
        return None
    details = data.get("prompt_tokens_details") or {}
    return GenerationUsage(
        prompt_tokens=data.get("prompt_tokens", 0),
        completion_tokens=data.get("completion_tokens", 0),
        cached_tokens=details.get("cached_tokens", 0),
    )


def _trace_timestamp(raw: object, label: str) -> datetime:
    if type(raw) is not str or _TRACE_TIMESTAMP_RE.fullmatch(raw) is None:
        raise RuntimeError(f"Kairyu upstream stage trace has an invalid {label}")
    try:
        value = datetime.fromisoformat(raw[:-1] + "+00:00")
    except ValueError as error:
        raise RuntimeError(
            f"Kairyu upstream stage trace has an invalid {label}"
        ) from error
    if value.tzinfo is None or value.astimezone(UTC).utcoffset() is None:
        raise RuntimeError(f"Kairyu upstream stage trace has an invalid {label}")
    return value


def _stage_metrics_from_trace(
    raw: object,
    *,
    response_id: object,
) -> tuple[GenerationStageMetric, ...]:
    """Validate and extract the bounded native-stage subset of a v2 trace."""

    if not isinstance(raw, Mapping):
        raise RuntimeError("Kairyu upstream returned a malformed stage trace envelope")
    if raw.get("trace_version") != "2.0":
        raise RuntimeError("Kairyu upstream returned an unsupported stage trace version")
    if (
        type(response_id) is not str
        or not response_id
        or raw.get("request_id") != response_id
    ):
        raise RuntimeError("Kairyu upstream stage trace request id does not match")
    started_at = _trace_timestamp(raw.get("started_at"), "started_at")
    completed_at = _trace_timestamp(raw.get("completed_at"), "completed_at")
    if completed_at < started_at:
        raise RuntimeError("Kairyu upstream stage trace timestamps are reversed")
    events = raw.get("events")
    if not isinstance(events, list) or len(events) > _MAX_TRACE_EVENTS:
        raise RuntimeError("Kairyu upstream returned a malformed stage trace event list")

    metrics: list[GenerationStageMetric] = []
    seen: set[str] = set()
    for expected_seq, event in enumerate(events, start=1):
        if (
            not isinstance(event, Mapping)
            or type(event.get("seq")) is not int
            or event.get("seq") != expected_seq
        ):
            raise RuntimeError("Kairyu upstream returned an invalid stage trace sequence")
        if event.get("kind") != "stage":
            continue
        if event.get("status") != "success":
            # Trace v2 is additive: unknown statuses are not measurements and
            # must not invalidate or cross this propagation boundary.
            continue
        detail = event.get("detail")
        if not isinstance(detail, Mapping):
            raise RuntimeError("Kairyu upstream stage trace event omitted detail")
        stage = detail.get("stage")
        duration_ns = detail.get("duration_ns")
        occurrences = detail.get("occurrences")
        if (
            type(stage) is not str
            or event.get("node") != stage
            or event.get("role") != "engine"
            or type(event.get("attempt")) is not int
            or event.get("attempt") != 0
            or event.get("timing") is not None
        ):
            raise RuntimeError("Kairyu upstream returned an invalid stage trace name")
        if (
            type(duration_ns) is not int
            or duration_ns < 0
            or duration_ns > _MAX_TRACE_DURATION_NS
        ):
            raise RuntimeError("Kairyu upstream returned an invalid stage duration")
        if (
            type(occurrences) is not int
            or occurrences < 1
            or occurrences > _MAX_TRACE_OCCURRENCES
        ):
            raise RuntimeError("Kairyu upstream returned an invalid stage occurrence count")
        if (
            detail.get("aggregation") != "sum"
            or detail.get("scope") != "request-observed"
        ):
            raise RuntimeError("Kairyu upstream returned an invalid stage trace scope")
        try:
            metric = GenerationStageMetric(
                stage=stage,
                duration_ns=duration_ns,
                occurrences=occurrences,
            )
        except ValueError as error:
            raise RuntimeError(
                "Kairyu upstream returned an invalid stage trace name"
            ) from error
        if metric.stage in seen:
            raise RuntimeError("Kairyu upstream returned an invalid stage trace name")
        seen.add(metric.stage)
        metrics.append(metric)
    return tuple(metrics)


def _token_logprob(raw: dict) -> TokenLogprob:
    raw_bytes = raw.get("bytes")
    return TokenLogprob(
        token=raw["token"],
        token_id=_UNKNOWN_TOKEN_ID,
        logprob=float(raw["logprob"]),
        bytes_=tuple(raw_bytes) if raw_bytes is not None else None,
        top=tuple(_token_logprob(item) for item in raw.get("top_logprobs") or ()),
    )


def _message_text(message: Mapping[str, object]) -> str:
    """Preserve native OpenAI tool calls in Kairyu's backend-neutral text form."""

    text = message.get("content")
    parts = [text] if isinstance(text, str) and text else []
    for call in message.get("tool_calls") or ():
        if not isinstance(call, Mapping):
            continue
        function = call.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        parts.append(
            "<tool_call>"
            + json.dumps(
                {"name": name, "arguments": arguments},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "</tool_call>"
        )
    return "".join(parts)


def _stream_tool_calls_text(
    calls: Mapping[int, Mapping[str, list[str]]],
) -> str:
    """Reassemble OpenAI chat tool-call deltas for backend-neutral consumers."""

    return _message_text(
        {
            "tool_calls": [
                {
                    "function": {
                        "name": "".join(call.get("name", ())),
                        "arguments": "".join(call.get("arguments", ())),
                    }
                }
                for _, call in sorted(calls.items())
            ]
        }
    )


class OpenAIRequestValidationError(ValueError):
    """A request cannot be represented by the configured upstream contract."""


def _client_error(upstream: str, message: str) -> OpenAIRequestValidationError:
    return OpenAIRequestValidationError(f"OpenAI-compatible upstream {upstream!r} {message}")


def _active_sampling_fields(params: SamplingParams) -> set[str]:
    """Return non-neutral request intent that an upstream must execute."""

    active: set[str] = set()
    values_and_neutral = {
        "best_of": (params.best_of, None),
        "frequency_penalty": (params.frequency_penalty, 0.0),
        "forced_token_ids": (params.forced_token_ids, None),
        "ignore_eos": (params.ignore_eos, False),
        "logprobs": (params.logprobs, None),
        "max_tokens": (params.max_tokens, None),
        "min_p": (params.min_p, 0.0),
        "min_tokens": (params.min_tokens, 0),
        "n": (params.n, 1),
        "presence_penalty": (params.presence_penalty, 0.0),
        "prompt_logprobs": (params.prompt_logprobs, None),
        "repetition_penalty": (params.repetition_penalty, 1.0),
        "response_format": (params.extra_args.get("response_format"), None)
        if isinstance(params.extra_args, Mapping)
        else (None, None),
        "seed": (params.seed, None),
        "skip_special_tokens": (params.skip_special_tokens, True),
        "stop": (params.stop, ()),
        "stop_token_ids": (params.stop_token_ids, ()),
        "temperature": (params.temperature, 1.0),
        "top_k": (params.top_k, -1),
        "top_p": (params.top_p, 1.0),
    }
    for field, (value, neutral) in values_and_neutral.items():
        if field in GENERATION_CONFIG_SAMPLING_FIELDS:
            if field not in params.generation_config_omitted:
                active.add(field)
        elif value != neutral:
            active.add(field)
    return active


def _validate_sampling_values(
    params: SamplingParams,
    capabilities: OpenAIRequestCapabilities,
) -> None:
    invalid: list[str] = []
    if params.logprobs is not None and params.logprobs < 0:
        invalid.append("logprobs must be non-negative")
    if params.prompt_logprobs is not None and params.prompt_logprobs < 0:
        invalid.append("prompt_logprobs must be non-negative")
    if params.best_of is not None and params.best_of < params.n:
        invalid.append("best_of must be greater than or equal to n")
    if any(token_id < 0 for token_id in params.stop_token_ids):
        invalid.append("stop_token_ids must be non-negative")
    if capabilities.max_n is not None and params.n > capabilities.max_n:
        invalid.append(f"n must be <= {capabilities.max_n}")
    if (
        capabilities.max_temperature is not None
        and params.temperature > capabilities.max_temperature
    ):
        invalid.append(f"temperature must be <= {capabilities.max_temperature}")
    if invalid:
        raise _client_error(capabilities.upstream, "rejects request: " + "; ".join(invalid))


def _sampling_payload(
    params: SamplingParams,
    capabilities: OpenAIRequestCapabilities,
    *,
    validate_json: bool = True,
) -> dict[str, object]:
    extra_args = params.extra_args
    if not isinstance(extra_args, Mapping):
        raise _client_error(capabilities.upstream, "requires extra_args to be a mapping")
    if any(not isinstance(key, str) for key in extra_args):
        raise _client_error(capabilities.upstream, "requires string extra_args keys")

    _validate_sampling_values(params, capabilities)
    active = _active_sampling_fields(params)
    unsupported = active - capabilities.sampling_fields
    if unsupported:
        raise _client_error(
            capabilities.upstream,
            "does not support request fields: " + ", ".join(sorted(unsupported)),
        )

    payload: dict[str, object] = {}
    omitted = params.generation_config_omitted
    always = {
        "temperature": params.temperature,
        "top_p": params.top_p,
        "n": params.n,
        "presence_penalty": params.presence_penalty,
        "frequency_penalty": params.frequency_penalty,
    }
    payload.update(
        (field, value)
        for field, value in always.items()
        if field not in omitted
        and field in capabilities.sampling_fields
        and (
            field in active
            or field in capabilities.forward_neutral_fields
            or field in GENERATION_CONFIG_SAMPLING_FIELDS
        )
    )

    optional = {
        "best_of": params.best_of,
        "ignore_eos": params.ignore_eos if params.ignore_eos else None,
        "min_p": params.min_p,
        "min_tokens": params.min_tokens if params.min_tokens != 0 else None,
        "prompt_logprobs": params.prompt_logprobs,
        "repetition_penalty": params.repetition_penalty,
        "seed": params.seed,
        "skip_special_tokens": False if not params.skip_special_tokens else None,
        "stop": list(params.stop) if params.stop else None,
        "stop_token_ids": list(params.stop_token_ids) if params.stop_token_ids else None,
        "top_k": params.top_k,
    }
    payload.update(
        (field, value)
        for field, value in optional.items()
        if field not in omitted
        and value is not None
        and field in capabilities.sampling_fields
    )
    if params.max_tokens is not None and "max_tokens" in capabilities.sampling_fields:
        payload[capabilities.max_tokens_wire_name] = params.max_tokens
    if params.logprobs is not None and "logprobs" in capabilities.sampling_fields:
        payload["logprobs"] = True
        payload["top_logprobs"] = params.logprobs
    response_format = extra_args.get("response_format")
    if response_format is not None and "response_format" in capabilities.sampling_fields:
        payload["response_format"] = response_format

    vendor_args = {key: value for key, value in extra_args.items() if key != "response_format"}
    unapproved = set(vendor_args) - capabilities.extra_args
    if unapproved:
        fields = ", ".join(f"extra_args.{key}" for key in sorted(unapproved))
        raise _client_error(
            capabilities.upstream,
            f"does not allow vendor extension fields: {fields}",
        )
    payload.update(vendor_args)
    if validate_json:
        try:
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise _client_error(
                capabilities.upstream,
                f"requires JSON-serializable request fields: {error}",
            ) from error
    return payload


def _validate_tools(
    request: GenerationRequest,
    capabilities: OpenAIRequestCapabilities,
) -> None:
    if capabilities.strict_tools:
        return
    for index, tool in enumerate(request.tools):
        function = tool.get("function")
        if isinstance(function, Mapping) and function.get("strict") is True:
            raise _client_error(
                capabilities.upstream,
                f"does not support request field tools[{index}].function.strict",
            )


def _validated_request_payload(
    request: GenerationRequest,
    capabilities: OpenAIRequestCapabilities,
    *,
    validate_json: bool = True,
) -> dict[str, object]:
    kind = prompt_kind(request.prompt)
    if kind not in capabilities.prompt_kinds:
        raise _client_error(
            capabilities.upstream,
            f"does not support prompt kind: {kind}",
        )
    _validate_tools(request, capabilities)
    if request.priority != 0 and not capabilities.priority:
        raise _client_error(
            capabilities.upstream,
            "does not support request field: priority",
        )
    payload = _sampling_payload(
        request.sampling_params,
        capabilities,
        validate_json=validate_json,
    )
    parallel_tool_calls = resolve_parallel_tool_calls(
        request.parallel_tool_calls,
        request.sampling_params.extra_args,
    )
    if (
        request.parallel_tool_calls is not None
        and parallel_tool_calls is not None
        and capabilities.parallel_tool_calls
    ):
        payload["parallel_tool_calls"] = parallel_tool_calls
    if capabilities.priority:
        payload["priority"] = request.priority
    if request.reasoning_effort is not None:
        payload["reasoning_effort"] = request.reasoning_effort
    if request.chat_template_kwargs is not None:
        unsupported = (
            set(request.chat_template_kwargs) - capabilities.chat_template_kwargs
        )
        if unsupported:
            raise _client_error(
                capabilities.upstream,
                "does not support chat_template_kwargs: "
                + ", ".join(sorted(unsupported)),
            )
        payload["chat_template_kwargs"] = dict(request.chat_template_kwargs)
    return payload


def _upstream_reasoning(message: object) -> str | None:
    """The private-reasoning span of an upstream chat message or delta.

    vLLM emits it as ``reasoning_content`` (DeepSeek-style) or, since the
    OpenAI-compatible rename shipped in the pinned Qwen image, ``reasoning``.
    Dropping the alias silently loses the span the public-output floor
    continues and the ``reasoning_closed`` reclaim publishes (DTO-D15).
    """

    if not isinstance(message, Mapping):
        return None
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str):
            return value
    return None


class OpenAICompatBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None = "OPENAI_API_KEY",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
        request_stream_usage: bool = True,
        upstream: str = "generic",
        capabilities: Mapping[str, object] | OpenAIRequestCapabilities | None = None,
        image_input_policy: Mapping[str, object] | ImageInputPolicy | None = None,
        allow_templated_chat_passthrough: bool = False,
        completion_reasoning_end_tag: str | None = None,
        chat_tokenizer: str | None = None,
        model_revision: str | None = None,
        max_model_len: int | None = None,
        quantization_format: str | None = None,
        cache_descriptor: Mapping[str, object] | None = None,
        tensor_parallel_size: int = 1,
        expert_parallel_size: int = 1,
        attention_data_parallel_size: int = 1,
        mtp_enabled: bool = False,
        dspark_enabled: bool = False,
        container_image_digest: str | None = None,
        client: httpx.AsyncClient | None = None,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        if client is not None and client_factory is not None:
            raise ValueError("client and client_factory are mutually exclusive")
        if client is not None and transport is not None:
            raise ValueError("client and transport are mutually exclusive")
        if client_factory is not None and transport is not None:
            raise ValueError("client_factory and transport are mutually exclusive")
        self._base_url = base_url.rstrip("/")
        self._model = model
        if type(allow_templated_chat_passthrough) is not bool:
            raise TypeError("allow_templated_chat_passthrough must be a boolean")
        self._allow_templated_chat_passthrough = allow_templated_chat_passthrough
        if completion_reasoning_end_tag is not None and (
            not isinstance(completion_reasoning_end_tag, str)
            or not completion_reasoning_end_tag
            or len(completion_reasoning_end_tag) > 128
        ):
            raise ValueError(
                "completion_reasoning_end_tag must be a non-empty string of at "
                "most 128 characters or null"
            )
        self._completion_reasoning_end_tag = completion_reasoning_end_tag
        self.chat_tokenizer = chat_tokenizer
        self.model_revision = model_revision
        self.max_model_len = max_model_len
        self.quantization_format = quantization_format
        self.cache_descriptor = dict(cache_descriptor) if cache_descriptor is not None else None
        self.tensor_parallel_size = tensor_parallel_size
        self.expert_parallel_size = expert_parallel_size
        self.attention_data_parallel_size = attention_data_parallel_size
        self.mtp_enabled = mtp_enabled
        self.dspark_enabled = dspark_enabled
        self.container_image_digest = container_image_digest
        self._api_key_env = api_key_env
        self._timeout_s = timeout_s
        self._transport = transport
        # m9 D1: upstreams only emit the usage chunk when asked; disable for
        # upstreams that reject unknown stream_options
        if type(request_stream_usage) is not bool:
            raise ValueError("request_stream_usage must be a boolean")
        self._request_stream_usage = request_stream_usage
        self._capabilities = resolve_openai_capabilities(upstream, capabilities)
        if completion_reasoning_end_tag is not None and (
            self._capabilities.upstream != "vllm"
            or not allow_templated_chat_passthrough
        ):
            raise ValueError(
                "completion_reasoning_end_tag requires upstream='vllm' and "
                "allow_templated_chat_passthrough=true"
            )
        unsupported_prompt_kinds = self._capabilities.prompt_kinds - {
            "text",
            "multimodal",
        }
        if unsupported_prompt_kinds:
            raise ValueError(
                "OpenAICompatBackend currently implements text and multimodal "
                "chat prompts; "
                "unsupported configured prompt kinds: "
                f"{sorted(unsupported_prompt_kinds)}"
            )
        if isinstance(image_input_policy, ImageInputPolicy):
            self._image_input_policy = image_input_policy
        elif isinstance(image_input_policy, Mapping):
            self._image_input_policy = ImageInputPolicy(**dict(image_input_policy))
        elif image_input_policy is None:
            self._image_input_policy = None
        else:
            raise ValueError(
                "OpenAI-compatible image_input_policy must be a mapping or "
                "ImageInputPolicy"
            )
        supports_multimodal = "multimodal" in self._capabilities.prompt_kinds
        if supports_multimodal and self._image_input_policy is None:
            raise ValueError(
                "OpenAI-compatible multimodal capability requires image_input_policy"
            )
        if not supports_multimodal and self._image_input_policy is not None:
            raise ValueError(
                "OpenAI-compatible image_input_policy requires the multimodal "
                "prompt capability"
            )
        if supports_multimodal and not self._request_stream_usage:
            raise ValueError(
                "OpenAI-compatible multimodal streaming requires "
                "request_stream_usage=true for exact processed prompt usage"
            )
        if self._image_input_policy is not None:
            self._image_input_policy.require_runtime()
        self._client = client
        self._client_factory = client_factory
        # An injected instance has an external owner. A factory creates this
        # backend's origin-local client, so normal replica shutdown owns it.
        self._owns_client = client is None
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._prepared_payloads: dict[
            int,
            tuple[weakref.ReferenceType[GenerationRequest], bytes],
        ] = {}
        self._prepared_payloads_lock = threading.Lock()
        self._preparing_payloads: dict[int, _PayloadPreparationFlight] = {}
        self._prepared_image_urls: dict[
            int,
            tuple[weakref.ReferenceType[GenerationRequest], tuple[str, ...]],
        ] = {}
        self._prepared_image_urls_lock = threading.Lock()
        self._preparing_images: dict[int, _ImagePreparationFlight] = {}

    @property
    def base_url(self) -> str:
        """The upstream OpenAI-compatible base (``.../v1``), trailing slash stripped."""
        return self._base_url

    @property
    def capabilities(self) -> OpenAIRequestCapabilities:
        """Resolved immutable request contract for this upstream."""

        return self._capabilities

    def supports_prompt_kind(self, kind: str) -> bool:
        """Publish the resolved prompt contract to composite backends."""

        return kind in self._capabilities.prompt_kinds

    def supports_chat_template_kwargs(self, keys: frozenset[str]) -> bool:
        """Publish whether this upstream owns configurable HF templating."""

        return keys <= self._capabilities.chat_template_kwargs

    @property
    def request_validation_key(self) -> tuple[
        OpenAIRequestCapabilities,
        ImageInputPolicy | None,
        bool,
    ] | None:
        """Identity for replicas with exactly equivalent request validation.

        ``ReplicaPool`` may validate one representative for backends that
        explicitly publish the same immutable key.  The OpenAI compatibility
        validator depends only on this resolved capability contract, never on
        the replica address, model instance, or HTTP client. A subclass must
        explicitly opt back in because an overridden validator may depend on
        additional per-instance state.
        """

        if type(self) is not OpenAICompatBackend:
            return None
        return (
            self._capabilities,
            self._image_input_policy,
            self._allow_templated_chat_passthrough,
        )

    @property
    def admission_upper_bound_key(self) -> tuple[
        OpenAIRequestCapabilities,
        ImageInputPolicy | None,
        bool,
        int | None,
    ] | None:
        """Identity for replicas with exactly equivalent admission ceilings."""

        if type(self) is not OpenAICompatBackend:
            return None
        return (
            self._capabilities,
            self._image_input_policy,
            self._allow_templated_chat_passthrough,
            self.max_model_len,
        )

    def _validate_request_structure(self, request: GenerationRequest) -> None:
        """Validate prompt structure without request-sized JSON serialization."""

        kind = prompt_kind(request.prompt)
        if kind not in self._capabilities.prompt_kinds:
            raise _client_error(
                self._capabilities.upstream,
                f"does not support prompt kind: {kind}",
            )
        if (
            isinstance(request.prompt, TemplatedPrompt)
            and not self._allow_templated_chat_passthrough
        ):
            raise _client_error(
                self._capabilities.upstream,
                "cannot preserve a tokenizer-owned pre-rendered chat prompt through "
                "an upstream /chat/completions template boundary",
            )
        if (
            isinstance(request.prompt, TemplatedPrompt)
            and self._completion_reasoning_end_tag is not None
            and request.sampling_params.logprobs is not None
        ):
            raise _client_error(
                self._capabilities.upstream,
                "cannot return logprobs after filtering private completion reasoning",
            )
        if request.chat_template_kwargs is not None:
            unsupported = (
                set(request.chat_template_kwargs)
                - self._capabilities.chat_template_kwargs
            )
            if unsupported:
                raise _client_error(
                    self._capabilities.upstream,
                    "does not support chat_template_kwargs: "
                    + ", ".join(sorted(unsupported)),
                )
        if request.assistant_prefill is not None and (
            self._capabilities.upstream != "vllm"
            or isinstance(request.prompt, TemplatedPrompt)
        ):
            raise _client_error(
                self._capabilities.upstream,
                "does not support assistant_prefill (a vLLM chat-template "
                "continuation of the final assistant message)",
            )
        if isinstance(request.prompt, MultimodalPrompt):
            if type(request.prompt.base) is not str and not isinstance(
                request.prompt.base,
                TextPrompt,
            ):
                raise _client_error(
                    self._capabilities.upstream,
                    "does not support token-based multimodal prompt bases",
                )
            assert self._image_input_policy is not None
            self._image_input_policy.validate_prompt_headers(request.prompt)

    def validate_request(self, request: GenerationRequest) -> None:
        """Fail unsupported intent before dispatch, metering, or client creation."""

        self._validate_request_structure(request)
        _validated_request_payload(request, self._capabilities)

    def validate_request_before_prepare(self, request: GenerationRequest) -> None:
        """Keep only bounded structural checks on the serving event loop."""

        self._validate_request_structure(request)

    def _dispatch_preflight(
        self,
        request: GenerationRequest,
    ) -> dict[str, object]:
        try:
            self._validate_request_structure(request)
            return _validated_request_payload(
                request,
                self._capabilities,
                validate_json=False,
            )
        except InvalidImageInput as error:
            raise UpstreamClientError(
                str(error),
                status_code=400,
                code=error.code,
                public_message=str(error),
            ) from error
        except OpenAIRequestValidationError as error:
            raise UpstreamClientError(
                str(error),
                status_code=400,
                public_message=str(error),
            ) from error

    def admission_upper_bound(self, request: GenerationRequest) -> AdmissionUpperBound:
        """Reserve a configured processor ceiling, never media-byte heuristics."""

        if not isinstance(request.prompt, MultimodalPrompt):
            return default_admission_upper_bound(
                request,
                fallback_output_tokens=self.max_model_len,
            )
        policy = self._image_input_policy
        if policy is None:  # Defensive: validation owns the normal error.
            raise ValueError("multimodal admission requires image_input_policy")
        max_tokens = request.sampling_params.max_tokens
        if max_tokens is None:
            # OpenAI-style limitless request: reserve the context-length
            # ceiling (or the documented constant) instead of rejecting.
            max_tokens = self.max_model_len or UNLIMITED_OUTPUT_ADMISSION_TOKENS
        candidates = max(
            request.sampling_params.n,
            request.sampling_params.best_of or request.sampling_params.n,
        )
        return AdmissionUpperBound(
            tokens=candidates * (policy.max_processed_prompt_tokens + max_tokens),
            refundable_on_exact_usage=candidates == 1,
        )

    async def fetch_backends(self, *, timeout_s: float = 2.0) -> dict | None:
        """Best-effort GET of a kairyu replica's ``/backends`` (m13 introspection).

        When this backend points at a kairyu node (the DP-replica case), the node
        is a kairyu server whose OpenAI base is ``<root>/v1``, so its introspection
        endpoint lives at ``<root>/backends``. Reuses this backend's pooled client
        (and any test transport) and keyless/auth headers. Returns ``None`` for a
        non-kairyu upstream (no ``/backends`` → non-200) or any transport error —
        introspection must never raise into the caller's ``/backends`` handler."""
        root = self._base_url[: -len("/v1")] if self._base_url.endswith("/v1") else self._base_url
        try:
            response = await self._get_client().get(
                f"{root}/backends", headers=self._headers(), timeout=timeout_s
            )
        except (httpx.HTTPError, RuntimeError):  # RuntimeError: missing api key
            return None
        if response.status_code != 200:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        engines = payload.get("engines")
        if isinstance(engines, list):
            # One replica process may serve several models. This backend routes
            # only to ``self._model``; forwarding topology from another row
            # would make the gateway report an unrelated TP/EP configuration.
            payload = dict(payload)
            payload["engines"] = [
                item
                for item in engines
                if isinstance(item, dict) and item.get("model") == self._model
            ]
        return payload

    async def count_prompt_tokens_async(
        self, prompt: str, *, timeout_s: float = 2.0
    ) -> int | None:
        """Best-effort exact count via a vLLM upstream's ``POST /tokenize``.

        Mirrors ``fetch_backends``: pooled client, keyless/auth headers, and a
        fail-soft ``None`` on any transport or shape problem. Only a vLLM
        upstream is known to expose the endpoint.
        """

        if self._capabilities.upstream != "vllm":
            return None
        root = (
            self._base_url[: -len("/v1")]
            if self._base_url.endswith("/v1")
            else self._base_url
        )
        try:
            response = await self._get_client().post(
                f"{root}/tokenize",
                json={"model": self._model, "prompt": prompt},
                headers=self._headers(),
                timeout=timeout_s,
            )
        except (httpx.HTTPError, RuntimeError):  # RuntimeError: missing api key
            return None
        if response.status_code != 200:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        count = payload.get("count") if isinstance(payload, dict) else None
        return count if type(count) is int and count >= 0 else None

    def _api_key(self) -> str:
        assert self._api_key_env is not None
        key = os.environ.get(self._api_key_env)
        if not key:
            raise RuntimeError(
                f"API key environment variable {self._api_key_env!r} is not set "
                f"(required for backend at {self._base_url})"
            )
        return key

    def _headers(self, request: GenerationRequest | None = None) -> dict[str, str]:
        headers = (
            {}
            if self._api_key_env is None
            else {"Authorization": f"Bearer {self._api_key()}"}
        )
        if request is not None and self._capabilities.upstream == "kairyu":
            if request.scheduling_class not in {"interactive", "batch"}:
                raise ValueError(
                    f"invalid scheduling class {request.scheduling_class!r}"
                )
            headers["X-Kairyu-Scheduling-Class"] = request.scheduling_class
            if request.trace_requested:
                headers["X-Kairyu-Trace"] = "1"
        # Propagate only W3C traceparent/tracestate. The helper is a deferred
        # no-op when tracing or OpenTelemetry is unavailable and intentionally
        # does not forward baggage, prompts, outputs, or credentials.
        from kairyu.telemetry import inject_trace_context

        inject_trace_context(headers)
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        if self._closed:
            raise RuntimeError(f"backend {self._base_url} is shut down")
        if self._client is not None and self._client.is_closed:
            if not self._owns_client:
                raise RuntimeError("externally owned HTTP client is closed")
            self._client = None
        if self._client is None:
            self._client = (
                self._client_factory()
                if self._client_factory is not None
                else httpx.AsyncClient(
                    timeout=self._timeout_s,
                    transport=self._transport,
                )
            )
            if self._client.is_closed:
                self._client = None
                raise RuntimeError("client_factory returned a closed HTTP client")
        return self._client

    def _peek_prepared_payload(
        self,
        request: GenerationRequest,
    ) -> bytes | None:
        key = id(request)
        with self._prepared_payloads_lock:
            cached = self._prepared_payloads.get(key)
            if cached is None:
                return None
            if cached[0]() is request:
                return cached[1]
            self._prepared_payloads.pop(key, None)
        return None

    def _shared_payload_contract(self) -> tuple[object, ...] | None:
        """Exact builtin contract whose immutable wire bytes are shareable."""

        if type(self) is not OpenAICompatBackend:
            return None
        return (
            self._capabilities,
            self._image_input_policy,
            self._model,
            self._request_stream_usage,
        )

    def _peek_shared_prepared_payload(
        self,
        request: GenerationRequest,
    ) -> bytes | None:
        contract = self._shared_payload_contract()
        if contract is None:
            return None
        key = id(request)
        with _SHARED_PREPARED_PAYLOADS_LOCK:
            cached = _SHARED_PREPARED_PAYLOADS.get(key)
            if cached is None:
                return None
            if cached[0]() is request:
                return cached[1].get(contract)
            _SHARED_PREPARED_PAYLOADS.pop(key, None)
        return None

    def _retain_shared_prepared_payload(
        self,
        request: GenerationRequest,
        payload: bytes,
    ) -> bytes:
        contract = self._shared_payload_contract()
        if contract is None:
            return payload
        key = id(request)

        def discard(reference: weakref.ReferenceType[GenerationRequest]) -> None:
            with _SHARED_PREPARED_PAYLOADS_LOCK:
                current = _SHARED_PREPARED_PAYLOADS.get(key)
                if current is not None and current[0] is reference:
                    _SHARED_PREPARED_PAYLOADS.pop(key, None)

        with _SHARED_PREPARED_PAYLOADS_LOCK:
            cached = _SHARED_PREPARED_PAYLOADS.get(key)
            if cached is None or cached[0]() is not request:
                cached = (weakref.ref(request, discard), {})
                _SHARED_PREPARED_PAYLOADS[key] = cached
            return cached[1].setdefault(contract, payload)

    def _retain_prepared_payload(
        self,
        request: GenerationRequest,
        payload: bytes,
    ) -> bytes:
        key = id(request)
        backend_ref = weakref.ref(self)

        def discard(reference: weakref.ReferenceType[GenerationRequest]) -> None:
            backend = backend_ref()
            if backend is None:
                return
            with backend._prepared_payloads_lock:
                current = backend._prepared_payloads.get(key)
                if current is not None and current[0] is reference:
                    backend._prepared_payloads.pop(key, None)

        request_ref = weakref.ref(request, discard)
        with self._prepared_payloads_lock:
            cached = self._prepared_payloads.get(key)
            if cached is not None and cached[0]() is request:
                return cached[1]
            self._prepared_payloads[key] = (request_ref, payload)
        return payload

    def _take_prepared_payload(
        self,
        request: GenerationRequest,
    ) -> bytes | None:
        key = id(request)
        with self._prepared_payloads_lock:
            cached = self._prepared_payloads.get(key)
            if cached is None:
                return None
            self._prepared_payloads.pop(key, None)
            if cached[0]() is request:
                return cached[1]
        return None

    def _discard_finished_payload_preparation(
        self,
        key: int,
        flight: _PayloadPreparationFlight,
        task: asyncio.Task[bytes],
    ) -> None:
        if not task.cancelled():
            task.exception()
        if flight.waiters == 0 and self._preparing_payloads.get(key) is flight:
            self._preparing_payloads.pop(key, None)

    async def _prepare_payload(self, request: GenerationRequest) -> None:
        if self._closed:
            raise RuntimeError(f"backend {self._base_url} is shut down")
        if self._peek_prepared_payload(request) is not None:
            return
        shared = self._peek_shared_prepared_payload(request)
        if shared is not None:
            if isinstance(request.prompt, MultimodalPrompt):
                await self._prepare_image_urls(request)
            self._retain_prepared_payload(request, shared)
            return
        try:
            validate_backend_request_before_prepare(self, request)
        except InvalidImageInput as error:
            raise UpstreamClientError(
                str(error),
                status_code=400,
                code=error.code,
                public_message=str(error),
            ) from error
        except OpenAIRequestValidationError as error:
            raise UpstreamClientError(
                str(error),
                status_code=400,
                public_message=str(error),
            ) from error
        key = id(request)
        flight = self._preparing_payloads.get(key)
        if flight is not None and flight.request_ref() is not request:
            self._preparing_payloads.pop(key, None)
            flight = None
        if flight is None:
            task = asyncio.create_task(
                self._build_wire_payload(request)
            )
            flight = _PayloadPreparationFlight(weakref.ref(request), task)
            self._preparing_payloads[key] = flight
            task.add_done_callback(
                lambda done, key=key, flight=flight: (
                    self._discard_finished_payload_preparation(key, flight, done)
                )
            )

        flight.waiters += 1
        try:
            payload = await asyncio.shield(flight.task)
            if self._closed:
                raise RuntimeError(f"backend {self._base_url} is shut down")
            payload = self._retain_shared_prepared_payload(request, payload)
            self._retain_prepared_payload(request, payload)
        finally:
            flight.waiters -= 1
            if (
                flight.task.done()
                and flight.waiters == 0
                and self._preparing_payloads.get(key) is flight
            ):
                self._preparing_payloads.pop(key, None)

    async def _payload_for_dispatch(
        self,
        request: GenerationRequest,
        *,
        stream: bool,
    ) -> bytes:
        await self._prepare_payload(request)
        payload = self._take_prepared_payload(request)
        if payload is None:
            raise RuntimeError("prepared OpenAI request payload was lost before dispatch")
        if isinstance(request.prompt, MultimodalPrompt):
            self._take_prepared_image_urls(request)
        if not stream:
            return payload
        return await run_prompt_work(
            self._stream_wire_payload,
            payload,
            self._request_stream_usage,
        )

    async def _validated_image_urls(
        self,
        request: GenerationRequest,
    ) -> tuple[str, ...] | None:
        prompt = request.prompt
        if not isinstance(prompt, MultimodalPrompt):
            return None
        assert self._image_input_policy is not None
        try:
            cached = self._image_input_policy.cached_validated_prompt(prompt)
            if cached is not None:
                return cached
            # Pillow must decompress the complete raster to reject truncated
            # or malicious compressed inputs. Keep that bounded CPU work off
            # the server event loop and do it once, before admission/headers.
            return await run_prompt_work(
                self._image_input_policy.validate_prompt,
                prompt,
            )
        except InvalidImageInput as error:
            raise UpstreamClientError(
                str(error),
                status_code=400,
                code=error.code,
                public_message=str(error),
            ) from error

    def _peek_prepared_image_urls(
        self,
        request: GenerationRequest,
    ) -> tuple[str, ...] | None:
        key = id(request)
        with self._prepared_image_urls_lock:
            cached = self._prepared_image_urls.get(key)
            if cached is None:
                return None
            if cached[0]() is request:
                return cached[1]
            self._prepared_image_urls.pop(key, None)
        return None

    def _retain_prepared_image_urls(
        self,
        request: GenerationRequest,
        image_urls: tuple[str, ...],
    ) -> tuple[str, ...]:
        key = id(request)
        backend_ref = weakref.ref(self)

        def discard(reference: weakref.ReferenceType[GenerationRequest]) -> None:
            backend = backend_ref()
            if backend is None:
                return
            with backend._prepared_image_urls_lock:
                current = backend._prepared_image_urls.get(key)
                if current is not None and current[0] is reference:
                    backend._prepared_image_urls.pop(key, None)

        request_ref = weakref.ref(request, discard)
        with self._prepared_image_urls_lock:
            cached = self._prepared_image_urls.get(key)
            if cached is not None and cached[0]() is request:
                return cached[1]
            self._prepared_image_urls[key] = (request_ref, image_urls)
        return image_urls

    def _take_prepared_image_urls(
        self,
        request: GenerationRequest,
    ) -> tuple[str, ...] | None:
        key = id(request)
        with self._prepared_image_urls_lock:
            cached = self._prepared_image_urls.get(key)
            if cached is None:
                return None
            self._prepared_image_urls.pop(key, None)
            if cached[0]() is request:
                return cached[1]
        return None

    def _discard_finished_image_preparation(
        self,
        key: int,
        flight: _ImagePreparationFlight,
        task: asyncio.Task[tuple[str, ...]],
    ) -> None:
        if not task.cancelled():
            task.exception()
        if (
            flight.waiters == 0
            and self._preparing_images.get(key) is flight
        ):
            self._preparing_images.pop(key, None)

    async def _prepare_image_urls(self, request: GenerationRequest) -> None:
        if self._closed:
            raise RuntimeError(f"backend {self._base_url} is shut down")
        if self._peek_prepared_image_urls(request) is not None:
            return
        key = id(request)
        flight = self._preparing_images.get(key)
        if flight is not None and flight.request_ref() is not request:
            self._preparing_images.pop(key, None)
            flight = None
        if flight is None:
            task = asyncio.create_task(self._validated_image_urls(request))
            flight = _ImagePreparationFlight(weakref.ref(request), task)
            self._preparing_images[key] = flight
            task.add_done_callback(
                lambda done, key=key, flight=flight: (
                    self._discard_finished_image_preparation(key, flight, done)
                )
            )

        flight.waiters += 1
        try:
            image_urls = await asyncio.shield(flight.task)
            if self._closed:
                raise RuntimeError(f"backend {self._base_url} is shut down")
            self._retain_prepared_image_urls(request, image_urls)
        finally:
            flight.waiters -= 1
            if (
                flight.task.done()
                and flight.waiters == 0
                and self._preparing_images.get(key) is flight
            ):
                self._preparing_images.pop(key, None)

    async def _image_urls_for_dispatch(
        self,
        request: GenerationRequest,
    ) -> tuple[str, ...] | None:
        if not isinstance(request.prompt, MultimodalPrompt):
            return None
        await self._prepare_image_urls(request)
        image_urls = self._take_prepared_image_urls(request)
        if image_urls is None:
            raise RuntimeError("prepared image input was lost before dispatch")
        return image_urls

    async def prepare_request(self, request: GenerationRequest) -> None:
        """Validate payload/media once before quota debit or streaming headers."""

        await self._prepare_payload(request)

    def _payload(
        self,
        request: GenerationRequest,
        *,
        validated: dict[str, object],
        image_urls: tuple[str, ...] | None,
    ) -> dict:
        prompt = request.prompt
        use_completions = self._uses_vllm_completions(request)
        if isinstance(prompt, MultimodalPrompt):
            if image_urls is None:
                raise RuntimeError(
                    "multimodal payload requires fully validated image data"
                )
            messages: list[dict[str, object]] = []
            for message in prompt.messages:
                content: list[dict[str, object]] = []
                for part in message.content:
                    if part.type == "text":
                        content.append({"type": "text", "text": part.text})
                        continue
                    assert part.item_index is not None
                    image_url: dict[str, object] = {
                        "url": image_urls[part.item_index],
                    }
                    if part.detail is not None:
                        image_url["detail"] = part.detail
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": image_url,
                        }
                    )
                messages.append({"role": message.role, "content": content})
        else:
            text = prompt_text(prompt)
            assert text is not None
            if use_completions:
                completion_args = dict(validated)
                # These chat-only fields are already represented in Kairyu's
                # rendered DeepSeek prompt and are invalid on /completions.
                completion_args.pop("parallel_tool_calls", None)
                completion_args.pop("reasoning_effort", None)
                if completion_args.get("logprobs") is True:
                    completion_args["logprobs"] = completion_args.pop(
                        "top_logprobs", 0
                    )
                else:
                    completion_args.pop("top_logprobs", None)
                return {
                    "model": self._model,
                    "prompt": text,
                    **completion_args,
                }
            messages = [{"role": "user", "content": text}]
        if request.assistant_prefill is not None:
            # The upstream template renders this final assistant turn and
            # vLLM continues it instead of opening a new generation prompt.
            messages.append({"role": "assistant", "content": request.assistant_prefill})
        payload: dict = {
            "model": self._model,
            "messages": messages,
            **validated,
        }
        if request.assistant_prefill is not None:
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False
        # A passthrough vLLM replica uses an identity chat template. Kairyu has
        # already rendered tools and reasoning controls into the model-owned
        # template, so duplicating the structured tool fields would create two
        # competing prompt owners.
        if request.tools and not isinstance(prompt, TemplatedPrompt):
            payload["tools"] = list(request.tools)
            if request.tool_choice is not None:
                payload["tool_choice"] = request.tool_choice
        return payload

    def _uses_vllm_completions(self, request: GenerationRequest) -> bool:
        """Send Kairyu-rendered text without a second vLLM chat template."""

        return (
            self._capabilities.upstream == "vllm"
            and isinstance(request.prompt, TemplatedPrompt)
        )

    def _encode_payload(
        self,
        request: GenerationRequest,
        validated: dict[str, object],
        image_urls: tuple[str, ...] | None,
    ) -> bytes:
        """Build and encode the complete upstream body on the prompt lane."""

        try:
            return json.dumps(
                self._payload(
                    request,
                    validated=validated,
                    image_urls=image_urls,
                ),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        except (TypeError, ValueError) as serialization_error:
            error = _client_error(
                self._capabilities.upstream,
                "requires JSON-serializable request fields: "
                f"{serialization_error}",
            )
            raise UpstreamClientError(
                str(error),
                status_code=400,
                public_message=str(error),
            ) from serialization_error

    async def _build_wire_payload(self, request: GenerationRequest) -> bytes:
        validated = await run_prompt_work(self._dispatch_preflight, request)
        image_urls = None
        if isinstance(request.prompt, MultimodalPrompt):
            await self._prepare_image_urls(request)
            image_urls = self._peek_prepared_image_urls(request)
            if image_urls is None:
                raise RuntimeError("prepared image input was lost before payload encoding")
        return await run_prompt_work(
            self._encode_payload,
            request,
            validated,
            image_urls,
        )

    @staticmethod
    def _stream_wire_payload(payload: bytes, include_usage: bool) -> bytes:
        """Append stream-only primitive fields without reserializing the body."""

        if not payload.endswith(b"}"):
            raise RuntimeError("prepared OpenAI request payload is not a JSON object")
        suffix = b',"stream":true'
        if include_usage:
            suffix += b',"stream_options":{"include_usage":true}'
        return payload[:-1] + suffix + b"}"

    def _json_headers(self, request: GenerationRequest) -> dict[str, str]:
        return {
            **self._headers(request),
            "Content-Type": "application/json",
        }

    @staticmethod
    def _require_exact_multimodal_usage(
        request: GenerationRequest,
        usage: GenerationUsage | None,
    ) -> None:
        if isinstance(request.prompt, MultimodalPrompt) and (
            usage is None or usage.prompt_tokens < 1
        ):
            raise RuntimeError(
                "multimodal OpenAI-compatible upstream omitted exact processed "
                "prompt usage"
            )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = await self._payload_for_dispatch(
            request,
            stream=False,
        )
        use_completions = self._uses_vllm_completions(request)
        endpoint = "completions" if use_completions else "chat/completions"
        response = await self._get_client().post(
            f"{self._base_url}/{endpoint}",
            content=payload,
            headers=self._json_headers(request),
            timeout=self._timeout_s,
        )
        if response.status_code != 200:
            _raise_for_status(self._base_url, response.status_code, response.text)
        data = response.json()
        choices = data.get("choices", [])
        completion_tokens = (data.get("usage") or {}).get("completion_tokens")
        token_ids: tuple[int, ...] = (
            tuple(range(completion_tokens))
            if completion_tokens is not None and len(choices) == 1
            else ()
        )
        completions_list = []
        for i, choice in enumerate(choices):
            raw_content = (choice.get("logprobs") or {}).get("content")
            logprob_content = (
                None if raw_content is None else tuple(_token_logprob(item) for item in raw_content)
            )
            text = (
                choice.get("text", "")
                if use_completions
                else _message_text(choice["message"])
            )
            reasoning_content = None
            if use_completions and self._completion_reasoning_end_tag is not None:
                reasoning_content, text = _split_completion_reasoning(
                    text,
                    self._completion_reasoning_end_tag,
                )
            elif not use_completions:
                reasoning_content = _upstream_reasoning(choice.get("message"))
            completions_list.append(
                CompletionOutput(
                    index=choice.get("index", i),
                    text=text,
                    reasoning_content=reasoning_content,
                    token_ids=token_ids,
                    cumulative_logprob=(
                        None
                        if logprob_content is None
                        else sum((item.logprob for item in logprob_content), 0.0)
                    ),
                    finish_reason=choice.get("finish_reason"),
                    logprob_content=logprob_content,
                )
            )
        completions = tuple(completions_list)
        if not completions:
            raise RuntimeError(f"backend {self._base_url} returned no choices: {data}")
        usage = _usage_from(data.get("usage"))
        self._require_exact_multimodal_usage(request, usage)
        stage_metrics = (
            _stage_metrics_from_trace(
                data["kairyu_trace_v2"],
                response_id=data.get("id"),
            )
            if request.trace_requested and "kairyu_trace_v2" in data
            else ()
        )
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=completions,
            usage=usage,
            stage_metrics=stage_metrics,
        )

    def _partial(
        self,
        request: GenerationRequest,
        text_parts: dict[int, list[str]],
        text_lengths: dict[int, int],
        text_deltas: dict[int, str],
        finish: dict[int, str],
        deltas_seen: dict[int, int],
        logprobs: dict[int, list[TokenLogprob]],
        finished: bool,
        usage: GenerationUsage | None = None,
        stage_metrics: tuple[GenerationStageMetric, ...] = (),
        reasoning_parts: dict[int, list[str]] | None = None,
        reasoning_lengths: dict[int, int] | None = None,
        reasoning_deltas: dict[int, str] | None = None,
    ) -> GenerationResult:
        reasoning_parts = reasoning_parts or {}
        reasoning_lengths = reasoning_lengths or {}
        reasoning_deltas = reasoning_deltas or {}
        completions = tuple(
            CompletionOutput(
                index=index,
                text=(
                    "".join(text_parts.get(index, ()))
                    if finished
                    else _lazy_text(text_parts.get(index, []))
                ),
                token_ids=(tuple(range(deltas_seen.get(index, 0))) if finished else ()),
                finish_reason=finish.get(index, "stop") if finished else None,
                cumulative_logprob=(
                    sum(item.logprob for item in logprobs[index]) if index in logprobs else None
                ),
                logprob_content=(tuple(logprobs[index]) if index in logprobs else None),
                text_delta=text_deltas.get(index, ""),
                text_offset=text_lengths.get(index, 0) - len(text_deltas.get(index, "")),
                reasoning_content=(
                    "".join(reasoning_parts[index]) if index in reasoning_parts else None
                ),
                reasoning_delta=reasoning_deltas.get(index, ""),
                reasoning_offset=(
                    reasoning_lengths.get(index, 0)
                    - len(reasoning_deltas.get(index, ""))
                ),
            )
            for index in sorted(
                text_parts.keys() | reasoning_parts.keys() | finish.keys() | logprobs.keys()
            )
        )
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=completions,
            finished=finished,
            usage=usage,
            stage_metrics=stage_metrics,
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        """Real SSE streaming with delta-native partial results."""
        payload = await self._payload_for_dispatch(
            request,
            stream=True,
        )
        use_completions = self._uses_vllm_completions(request)
        endpoint = "completions" if use_completions else "chat/completions"
        async with self._get_client().stream(
            "POST",
            f"{self._base_url}/{endpoint}",
            content=payload,
            headers=self._json_headers(request),
            timeout=self._timeout_s,
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")
                _raise_for_status(self._base_url, response.status_code, body)
            text_parts: dict[int, list[str]] = {}
            text_lengths: dict[int, int] = {}
            reasoning_parts: dict[int, list[str]] = {}
            reasoning_lengths: dict[int, int] = {}
            finish: dict[int, str] = {}
            deltas_seen: dict[int, int] = {}
            logprobs: dict[int, list[TokenLogprob]] = {}
            usage: GenerationUsage | None = None
            stage_metrics: tuple[GenerationStageMetric, ...] = ()
            trace_seen = False
            trace_response_id: str | None = None
            done_seen = False
            completion_reasoning: dict[int, _CompletionReasoningParser] = {}
            streamed_tool_calls: dict[int, dict[int, dict[str, list[str]]]] = {}
            async for data_str in iter_sse_data(response):
                data_str = data_str.strip()
                if data_str == _SSE_DONE:
                    done_seen = True
                    break
                chunk = json.loads(data_str)
                if trace_seen:
                    if "error" in chunk:
                        # Kairyu's failure contract deliberately places its
                        # known partial trace before the sanitized SSE error.
                        # Surface the validated metrics now so a gateway can
                        # retain them, then fail when the consumer resumes.
                        # Never retain or re-surface the upstream error message.
                        if stage_metrics:
                            yield self._partial(
                                request,
                                text_parts,
                                text_lengths,
                                {},
                                finish,
                                deltas_seen,
                                logprobs,
                                finished=False,
                                usage=usage,
                                stage_metrics=stage_metrics,
                                reasoning_parts=reasoning_parts,
                                reasoning_lengths=reasoning_lengths,
                            )
                        raise RuntimeError(
                            "Kairyu upstream reported an error after its stage trace"
                        )
                    raise RuntimeError(
                        "Kairyu upstream stage trace was not terminal"
                    )
                chunk_id = chunk.get("id") if request.trace_requested else None
                if request.trace_requested and isinstance(chunk_id, str) and chunk_id:
                    if trace_response_id is None:
                        trace_response_id = chunk_id
                    elif chunk_id != trace_response_id:
                        raise RuntimeError(
                            "Kairyu upstream changed response id while streaming"
                        )
                if request.trace_requested and "kairyu_trace_v2" in chunk:
                    if trace_seen:
                        raise RuntimeError(
                            "Kairyu upstream returned more than one stage trace"
                        )
                    if chunk.get("choices") != []:
                        raise RuntimeError(
                            "Kairyu upstream stage trace metadata was not terminal"
                        )
                    stage_metrics = _stage_metrics_from_trace(
                        chunk["kairyu_trace_v2"],
                        response_id=chunk_id or trace_response_id,
                    )
                    trace_seen = True
                if chunk.get("usage"):  # final usage chunk has empty choices
                    usage = _usage_from(chunk["usage"])
                changed = False
                text_deltas: dict[int, str] = {}
                reasoning_deltas: dict[int, str] = {}
                for choice in chunk.get("choices", []):
                    index = choice.get("index", 0)
                    text_parts.setdefault(index, [])
                    text_lengths.setdefault(index, 0)
                    delta = choice.get("delta") or {}
                    content = (
                        choice.get("text")
                        if use_completions
                        else delta.get("content")
                    )
                    reasoning = (
                        None
                        if use_completions
                        else _upstream_reasoning(delta)
                    )
                    if not use_completions and isinstance(delta, Mapping):
                        for position, raw_call in enumerate(delta.get("tool_calls") or ()):
                            if not isinstance(raw_call, Mapping):
                                continue
                            call_index = raw_call.get("index", position)
                            if type(call_index) is not int or call_index < 0:
                                continue
                            function = raw_call.get("function")
                            if not isinstance(function, Mapping):
                                continue
                            call = streamed_tool_calls.setdefault(index, {}).setdefault(
                                call_index,
                                {"name": [], "arguments": []},
                            )
                            name = function.get("name")
                            if isinstance(name, str):
                                call["name"].append(name)
                            arguments = function.get("arguments")
                            if isinstance(arguments, str):
                                call["arguments"].append(arguments)
                    if use_completions and self._completion_reasoning_end_tag is not None:
                        parser = completion_reasoning.setdefault(
                            index,
                            _CompletionReasoningParser(
                                self._completion_reasoning_end_tag
                            ),
                        )
                        reasoning, content = parser.feed(
                            content if isinstance(content, str) else ""
                        )
                    if isinstance(reasoning, str) and reasoning:
                        reasoning_parts.setdefault(index, []).append(reasoning)
                        reasoning_lengths[index] = reasoning_lengths.get(index, 0) + len(
                            reasoning
                        )
                        reasoning_deltas[index] = (
                            reasoning_deltas.get(index, "") + reasoning
                        )
                        deltas_seen[index] = deltas_seen.get(index, 0) + 1
                        changed = True
                    if content:
                        text_parts[index].append(content)
                        text_lengths[index] += len(content)
                        text_deltas[index] = text_deltas.get(index, "") + content
                        deltas_seen[index] = deltas_seen.get(index, 0) + 1
                        changed = True
                    if choice.get("finish_reason"):
                        finish[index] = choice["finish_reason"]
                        calls = streamed_tool_calls.pop(index, None)
                        if calls:
                            tool_text = _stream_tool_calls_text(calls)
                            if tool_text:
                                text_parts[index].append(tool_text)
                                text_lengths[index] += len(tool_text)
                                text_deltas[index] = (
                                    text_deltas.get(index, "") + tool_text
                                )
                                deltas_seen[index] = deltas_seen.get(index, 0) + 1
                                changed = True
                    raw_logprobs = (choice.get("logprobs") or {}).get("content")
                    if raw_logprobs:
                        logprobs.setdefault(index, []).extend(
                            _token_logprob(item) for item in raw_logprobs
                        )
                        changed = True
                if changed:
                    yield self._partial(
                        request,
                        text_parts,
                        text_lengths,
                        text_deltas,
                        finish,
                        deltas_seen,
                        logprobs,
                        finished=False,
                        reasoning_parts=reasoning_parts,
                        reasoning_lengths=reasoning_lengths,
                        reasoning_deltas=reasoning_deltas,
                    )
            if streamed_tool_calls:
                text_deltas = {}
                for index, calls in streamed_tool_calls.items():
                    text_parts.setdefault(index, [])
                    text_lengths.setdefault(index, 0)
                    tool_text = _stream_tool_calls_text(calls)
                    if not tool_text:
                        continue
                    text_parts[index].append(tool_text)
                    text_lengths[index] += len(tool_text)
                    text_deltas[index] = tool_text
                    deltas_seen[index] = deltas_seen.get(index, 0) + 1
                if text_deltas:
                    yield self._partial(
                        request,
                        text_parts,
                        text_lengths,
                        text_deltas,
                        finish,
                        deltas_seen,
                        logprobs,
                        finished=False,
                        reasoning_parts=reasoning_parts,
                        reasoning_lengths=reasoning_lengths,
                    )
            if trace_seen and not done_seen:
                raise RuntimeError("Kairyu upstream stage trace omitted SSE [DONE]")
        for parser in completion_reasoning.values():
            parser.require_finished()
        if not text_parts and not finish:
            raise RuntimeError(f"backend {self._base_url} streamed no choices")
        self._require_exact_multimodal_usage(request, usage)
        yield self._partial(
            request,
            text_parts,
            text_lengths,
            {},
            finish,
            deltas_seen,
            logprobs,
            finished=True,
            usage=usage,
            stage_metrics=stage_metrics,
            reasoning_parts=reasoning_parts,
            reasoning_lengths=reasoning_lengths,
        )

    async def _shutdown_impl(self) -> None:
        preparation_tasks = tuple(
            flight.task for flight in self._preparing_images.values()
        ) + tuple(
            flight.task for flight in self._preparing_payloads.values()
        )
        if preparation_tasks:
            await asyncio.gather(
                *(asyncio.shield(task) for task in preparation_tasks),
                return_exceptions=True,
            )
        self._preparing_images.clear()
        self._preparing_payloads.clear()
        with self._prepared_payloads_lock:
            self._prepared_payloads.clear()
        with self._prepared_image_urls_lock:
            self._prepared_image_urls.clear()
        if self._owns_client:
            client = self._client
            if client is not None and not client.is_closed:
                await client.aclose()
            self._client = None

    async def shutdown(self) -> None:
        task = self._close_task
        if task is None:
            # Terminal before the first await: no concurrent request can
            # recreate a just-detached replica's origin-local client.
            self._closed = True
            task = asyncio.create_task(self._shutdown_impl())
            self._close_task = task
        await _await_task_cancellation_safe(task)


register_backend("openai", OpenAICompatBackend)
