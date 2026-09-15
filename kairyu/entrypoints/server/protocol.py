"""OpenAI chat-completions wire schema (request/response/chunk models)."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from kairyu.entrypoints.server.request_body import prevalidated_model_for


class ImageURL(BaseModel):
    """One strict OpenAI image reference."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1)
    detail: Literal["auto", "low", "high"] | None = None


class ContentPart(BaseModel):
    """OpenAI vision content part (m11 D5): text or image_url."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageURL | None = None

    @model_validator(mode="after")
    def _validate_tagged_payload(self) -> ContentPart:
        """Require the payload selected by ``type`` and forbid ambiguity."""

        if self.type == "text":
            if self.text is None:
                raise ValueError("text content part requires a string 'text' field")
            if self.image_url is not None:
                raise ValueError("text content part must not include 'image_url'")
            return self
        if self.image_url is None:
            raise ValueError("image_url content part requires an 'image_url' object")
        if self.text is not None:
            raise ValueError("image_url content part must not include 'text'")
        return self


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, object]] | None = None
    reasoning_content: str | None = None


class RouteFeatures(BaseModel):
    char_len: int
    word_count: int
    has_code_fence: bool
    math_symbol_count: int
    reasoning_keyword_count: int
    multi_step_marker_count: int
    question_count: int


class RouteDecisionPayload(BaseModel):
    target: str
    confidence: float
    reason: str
    features: RouteFeatures


class RoutePreviewRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)

    @model_validator(mode="wrap")
    @classmethod
    def _reuse_prevalidated(cls, value, handler):
        prepared = prevalidated_model_for(cls, value)
        if prepared is not None:
            return prepared
        return handler(value)


class RoutePreviewResponse(BaseModel):
    model: str
    orchestrated: bool
    binding: bool = False
    router_type: str | None = None
    target: str | None = None
    confidence: float | None = None
    reason: str | None = None
    features: RouteFeatures | None = None


class RouterDescriptorPayload(BaseModel):
    router_type: str
    thresholds: dict[str, int] | None = None
    min_confidence: float | None = None
    fallback_type: str | None = None
    epsilon: float | None = None
    is_warm: bool | None = None
    min_updates_per_arm: int | None = None


class EngineDescriptorPayload(BaseModel):
    backend_type: str
    model: str | None = None


class EngineResolutionPayload(BaseModel):
    configured: bool
    engine: str
    fallback: bool


class TargetResolutionPayload(BaseModel):
    configured: bool | None = None
    engine: str | None = None
    fallback: bool | None = None
    mode: str | None = None
    engines: list[EngineResolutionPayload] = Field(default_factory=list)


class RoleSamplingPayload(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    max_tokens: int | None = None
    max_tokens_by_effort: dict[str, int] | None = None
    seed_offset: int | None = None
    stop: list[str] = Field(default_factory=list)


class RoleExecutorPayload(BaseModel):
    code_from: list[str]
    tests_from: list[str]
    mode: str


class RoleDescriptorPayload(BaseModel):
    name: str
    worker: str
    role_type: str
    depends_on: list[str]
    verifies: str | None = None
    sampling: RoleSamplingPayload | None = None
    executor: RoleExecutorPayload | None = None


class ExecutorDescriptorPayload(BaseModel):
    backend_type: str
    base_url: str | None = None


class BudgetDescriptorPayload(BaseModel):
    max_steps: int
    max_refine_depth: int
    max_cost_usd: float | None = None


class ProfileChoicePayload(BaseModel):
    profile: str
    label: str
    criteria: str


class ProfileJudgePayload(BaseModel):
    worker: str
    timeout_seconds: float
    max_tokens: int
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    fallback: str = "primary"
    choices: list[ProfileChoicePayload] = Field(default_factory=list)


class RoutingModelDescriptorPayload(BaseModel):
    router: RouterDescriptorPayload
    targets: list[str]
    configured_engines: dict[str, EngineDescriptorPayload]
    configured_executors: dict[str, ExecutorDescriptorPayload] = Field(
        default_factory=dict
    )
    target_resolution: dict[str, TargetResolutionPayload]
    roles: list[RoleDescriptorPayload]
    # Named alternative role DAGs selected per request (issue #509, DTO-D13);
    # empty when the orchestrator serves the primary profile only.
    profiles: dict[str, list[RoleDescriptorPayload]] = Field(default_factory=dict)
    profile_selector: str | None = None
    # LLM verdict configuration selecting among the profiles; null when the
    # primary profile always serves.
    profile_judge: ProfileJudgePayload | None = None
    stream_head: str | None = None
    budget: BudgetDescriptorPayload
    moa_samples: int
    internal_max_tokens: int | None
    expose_intermediate_outputs: bool = False


class RoutingResponse(BaseModel):
    models: dict[str, RoutingModelDescriptorPayload]


class StreamOptions(BaseModel):
    include_usage: bool = False


class TopLogprobEntry(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None


class LogprobEntry(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprobEntry] = Field(default_factory=list)


class ChoiceLogprobs(BaseModel):
    content: list[LogprobEntry] | None = None


_ReasoningEffort = Literal["low", "high", "max"]
_ReasoningEffortInput = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
_REASONING_EFFORT_LEVELS: dict[str, _ReasoningEffort] = {
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "max",
    "max": "max",
}


def normalize_reasoning_effort(effort: object) -> _ReasoningEffort | None:
    """Normalize OpenAI-style effort aliases onto Kairyu L3 levels."""

    if effort is None:
        return None
    if not isinstance(effort, str) or effort not in _REASONING_EFFORT_LEVELS:
        raise ValueError(f"reasoning effort {effort!r} is not supported")
    return _REASONING_EFFORT_LEVELS[effort]


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    n: int = 1
    best_of: int | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    prompt_logprobs: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None  # modern-SDK alias of max_tokens
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    min_tokens: int = 0
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    seed: int | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None
    # vLLM-compatible per-request variables for the configured HF Jinja
    # template (for example Qwen3's ``enable_thinking``). Trusted prompt
    # carriers remain reserved by ChatTemplate.
    chat_template_kwargs: dict[str, object] | None = None
    response_format: dict | None = None
    extra_args: dict[str, object] | None = None
    user: str | None = None
    reasoning_effort: _ReasoningEffortInput | None = None
    # vLLM-compatible scheduling priority. A configured gateway replaces this
    # untrusted client value with the authenticated tenant's class.
    priority: int = Field(default=0, ge=-(2**63), le=2**63 - 1)

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def _normalize_reasoning_effort(cls, value: object) -> object:
        return normalize_reasoning_effort(value)

    @model_validator(mode="wrap")
    @classmethod
    def _reuse_prevalidated(cls, value, handler):
        prepared = prevalidated_model_for(cls, value)
        if prepared is not None:
            return prepared
        return handler(value)


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: str | list[ContentPart] | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: str | None = None
    logprobs: ChoiceLogprobs | None = None


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: PromptTokensDetails | None = None
    # Additive AUTO-model accounting.  These are absent on ordinary engine
    # responses and are cumulative across every internal generation call.
    orchestration_input_tokens: int | None = Field(default=None, ge=0)
    orchestration_output_tokens: int | None = Field(default=None, ge=0)


class KairyuTraceTiming(BaseModel):
    queued_at: str | None = None
    started_at: str | None = None
    first_token_at: str | None = None
    completed_at: str | None = None


class KairyuTraceUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


class KairyuTraceBudget(BaseModel):
    max_steps: int
    steps_before: int
    steps_consumed: int
    steps_remaining: int
    max_cost_usd: float | None = None
    cost_before_usd: float
    cost_consumed_usd: float
    cost_remaining_usd: float | None = None


class KairyuTraceError(BaseModel):
    type: str
    retryable: bool = False


class KairyuTraceEvent(BaseModel):
    seq: int
    node: str
    role: str | None = None
    kind: str
    status: str
    attempt: int = 0
    worker: str | None = None
    engine: str | None = None
    model: str | None = None
    timing: KairyuTraceTiming | None = None
    usage: KairyuTraceUsage | None = None
    budget: KairyuTraceBudget | None = None
    # Scalars plus flat scalar lists: the DTO-D13 profile-judge event carries
    # the offered label list; rejecting it crashed live SSE streams mid-body.
    detail: dict[
        str,
        str | int | float | bool | None | list[str | int | float | bool | None],
    ] = Field(default_factory=dict)
    error: KairyuTraceError | None = None


class KairyuTraceV2(BaseModel):
    trace_version: str
    request_id: str
    started_at: str
    completed_at: str
    events: list[KairyuTraceEvent]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage = Usage()
    # m11 D1: explicit opt-in trace (X-Kairyu-Trace: 1); excluded when None
    kairyu_trace: list[str] | None = None
    # Additive structured trace for evaluation tooling. Like the legacy field,
    # it is populated only when X-Kairyu-Trace: 1 is requested.
    kairyu_trace_v2: KairyuTraceV2 | None = None
    # Actual route uses the same schema as route preview. It is populated only
    # for traced orchestrated responses.
    kairyu_route: RouteDecisionPayload | None = None


class ChunkToolCall(BaseModel):
    # streamed tool-call deltas require an `index` so SDK accumulators can merge
    # fragments across chunks (S6); the non-streaming ToolCall has no index
    index: int
    id: str
    type: str = "function"
    function: FunctionCall


class ChunkDelta(BaseModel):
    role: str | None = None
    content: str | list[ContentPart] | None = None
    reasoning_content: str | None = None
    tool_calls: list[ChunkToolCall] | None = None


class ChunkChoice(BaseModel):
    index: int
    delta: ChunkDelta
    finish_reason: str | None = None
    # OpenAI: logprobs sits on the chunk CHOICE (sibling of delta), never inside it
    logprobs: ChoiceLogprobs | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    # OpenAI contract (m9 D1): key OMITTED unless stream_options.include_usage,
    # then null on every chunk except the final usage chunk (choices: [])
    usage: Usage | None = None
    # Opt-in terminal metadata for AUTO and direct native-engine responses. The
    # server omits these fields from ordinary chunks and emits them only after
    # an explicit X-Kairyu-Trace opt-in. kairyu_route remains AUTO-only.
    kairyu_trace: list[str] | None = None
    kairyu_trace_v2: KairyuTraceV2 | None = None
    kairyu_route: RouteDecisionPayload | None = None


class CompletionRequest(BaseModel):
    """Legacy /v1/completions (m9 D3). echo/suffix/best_of are rejected."""

    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str | list[str] | list[int] | list[list[int]]
    max_tokens: int | None = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    n: int = 1
    logprobs: int | None = None  # legacy top-k int, capped at 5
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    min_tokens: int = 0
    ignore_eos: bool = False
    repetition_penalty: float = 1.0
    skip_special_tokens: bool = True
    seed: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: str | None = None
    priority: int = Field(default=0, ge=-(2**63), le=2**63 - 1)
    # Kairyu extension: score one caller-supplied continuation without
    # presenting it as ordinary generation or OpenAI ``echo`` support.
    kairyu_continuation: str | None = None

    @field_validator("prompt", mode="before")
    @classmethod
    def _validate_prompt_shape(cls, value: object) -> object:
        """Reject lossy or ambiguous prompt arrays before union coercion."""

        if type(value) is str:
            return value
        if type(value) is not list:
            raise ValueError(
                "prompt must be a string, an array of strings, an array of "
                "token IDs, or an array of token-ID arrays"
            )
        if not value:
            # Keep the historical endpoint-level 400 and error envelope.
            return value
        if all(type(item) is str for item in value):
            return value
        if all(type(item) is int for item in value):
            cls._validate_token_ids(value, "prompt")
            return value
        if all(type(item) is list for item in value):
            for index, token_ids in enumerate(value):
                if not token_ids:
                    raise ValueError(f"prompt[{index}] token-ID array must not be empty")
                cls._validate_token_ids(token_ids, f"prompt[{index}]")
            return value
        raise ValueError(
            "prompt array must contain only strings, only token IDs, or only "
            "non-empty token-ID arrays"
        )

    @staticmethod
    def _validate_token_ids(token_ids: list[object], location: str) -> None:
        maximum = (1 << 64) - 1
        for index, token_id in enumerate(token_ids):
            if type(token_id) is not int:
                raise ValueError(
                    f"{location}[{index}] must be an integer token ID (booleans are not token IDs)"
                )
            if not 0 <= token_id <= maximum:
                raise ValueError(
                    f"{location}[{index}] token ID must be in the unsigned "
                    f"64-bit range 0..{maximum}"
                )


class CompletionLogprobs(BaseModel):
    """Legacy four-parallel-array shape; offsets from 0 within `text` (echo
    is rejected, so there is no prompt segment to offset past)."""

    tokens: list[str] = Field(default_factory=list)
    token_logprobs: list[float] = Field(default_factory=list)
    top_logprobs: list[dict[str, float]] | None = None
    text_offset: list[int] = Field(default_factory=list)


class CompletionChoice(BaseModel):
    index: int
    text: str
    logprobs: CompletionLogprobs | None = None
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage = Usage()


class LogLikelihoodCompletionChoice(CompletionChoice):
    """Exact token evidence exposed only by the Kairyu scoring extension."""

    prompt_token_ids: list[int]
    continuation_token_ids: list[int]


class LogLikelihoodCompletionResponse(CompletionResponse):
    """Separate response type keeps the ordinary completion wire unchanged."""

    choices: list[LogLikelihoodCompletionChoice]
    mode: Literal["loglikelihood"] = "loglikelihood"


class CompletionChunk(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage | None = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "kairyu"
    # vLLM-compatible extension (issue #495): the serving context length, so
    # OpenAI-compatible clients (LiteLLM model-info lookup) can bound context
    # management instead of falling back to a guessed default.
    max_model_len: int | None = None


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]
