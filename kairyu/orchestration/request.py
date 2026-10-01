"""Transport-neutral, per-call orchestration intent."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from kairyu.engine.prompt import MultimodalPrompt
from kairyu.orchestration.trace import TraceEvent
from kairyu.sampling_params import (
    PARALLEL_TOOL_CALLS_EXTRA_ARG,
    SamplingParams,
    resolve_parallel_tool_calls,
)

# Delimiters of the conversation JSON inside the L2 ``{query}`` that the chat
# service renders (validate_orchestration_chat_input). Shared so a consumer
# can recover the message list without depending on the wrapper prose.
CONVERSATION_JSON_OPEN = "--- CONVERSATION CONTEXT JSON ---\n"
CONVERSATION_JSON_CLOSE = "\n--- END CONVERSATION CONTEXT JSON ---"


def conversation_messages(query: str) -> list[object] | None:
    """The role-tagged messages of an L2 query, or None for a plain prompt."""

    start = query.find(CONVERSATION_JSON_OPEN)
    if start < 0:
        return None
    start += len(CONVERSATION_JSON_OPEN)
    end = query.find(CONVERSATION_JSON_CLOSE, start)
    if end < 0:
        return None
    try:
        messages = json.loads(query[start:end])
    except ValueError:
        return None
    return messages if isinstance(messages, list) else None


def conversation_text(query: str) -> str:
    """The ``{conversation}`` role placeholder: the request's role-tagged
    messages without the answer-contract wrapper, or the query itself."""

    messages = conversation_messages(query)
    return query if messages is None else json.dumps(messages, ensure_ascii=False, indent=1)


@dataclass(frozen=True)
class OrchestrationRequest:
    """One immutable orchestration call.

    The orchestrator instance is shared by concurrent HTTP requests, so public
    sampling and output intent must travel with the call instead of mutating
    constructor defaults.  Tool schemas remain structured until the final
    backend boundary; native backends may also receive a prompt rendered from
    the same schemas by the chat-template layer.
    """

    prompt: str
    sampling_params: SamplingParams
    tools: tuple[Mapping[str, object], ...] = ()
    tool_choice: str | Mapping[str, object] | None = None
    tools_in_prompt: bool = False
    # The latest user turn demands a machine-parsed output format in plain
    # text (e.g. "format your response as JSON"). Like tools/response_format,
    # this intent is incompatible with a prose head opening (issue #495).
    structured_format_in_prompt: bool = False
    response_format: Mapping[str, object] | None = None
    parallel_tool_calls: bool | None = None
    tool_call_protocol: str = "generic"
    reasoning_effort: str | None = None
    multimodal_prompt: MultimodalPrompt | None = None
    chat_template_kwargs: Mapping[str, object] | None = None
    # Appended to preserve the positional constructor contract above.
    conversation_affinity_key: str | None = None
    # LLM-judged role-profile verdict (issue #509 amendment, generalized by
    # DTO-D13): the selected profile name. Attached at most once, before
    # preflight/admission, so every consumer of the pure profile function
    # reads the same decision. None means no judgment was made and the
    # orchestrator's fallback profile applies.
    role_profile_judgment: str | None = None
    # One request-local observation of the optional judge call. Carrying the
    # exact event with the immutable call lets pre-admission judgment remain
    # visible to later result accounting and structured trace construction.
    # None means the judge was deterministically skipped, not that a dispatched
    # call necessarily failed to return a verdict.
    role_profile_judge_event: TraceEvent | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", tuple(self.tools))
        sampling_format = self.sampling_params.extra_args.get("response_format")
        if self.response_format != sampling_format:
            raise ValueError("response_format intent must match sampling_params.extra_args")
        resolve_parallel_tool_calls(
            self.parallel_tool_calls,
            self.sampling_params.extra_args,
        )
        if self.tool_call_protocol not in {"generic", "llama", "qwen", "deepseek_v4"}:
            raise ValueError("tool_call_protocol must be generic, llama, qwen or deepseek_v4")
        if self.reasoning_effort not in {None, "low", "high", "max"}:
            raise ValueError("reasoning_effort must be low, high, max, or null")
        if self.role_profile_judgment is not None and (
            not isinstance(self.role_profile_judgment, str) or not self.role_profile_judgment
        ):
            raise ValueError("role_profile_judgment must be a non-empty profile name or null")
        if self.role_profile_judge_event is not None and not isinstance(
            self.role_profile_judge_event,
            TraceEvent,
        ):
            raise TypeError("role_profile_judge_event must be a TraceEvent or null")
        if self.conversation_affinity_key is not None and (
            not isinstance(self.conversation_affinity_key, str)
            or not self.conversation_affinity_key
        ):
            raise ValueError("conversation_affinity_key must be a non-empty string or null")
        if self.multimodal_prompt is not None and not isinstance(
            self.multimodal_prompt,
            MultimodalPrompt,
        ):
            raise TypeError("multimodal_prompt must be a MultimodalPrompt or null")
        if self.chat_template_kwargs is not None:
            if self.multimodal_prompt is None:
                raise ValueError("chat_template_kwargs require a multimodal orchestration prompt")
            if not isinstance(self.chat_template_kwargs, Mapping) or any(
                not isinstance(key, str) or not key for key in self.chat_template_kwargs
            ):
                raise TypeError("chat_template_kwargs must map non-empty string keys or be null")
            object.__setattr__(
                self,
                "chat_template_kwargs",
                dict(self.chat_template_kwargs),
            )

    def internal_sampling_params(
        self,
        *,
        max_tokens_cap: int | None = None,
    ) -> SamplingParams:
        """Sampling for non-final planning/proposal/verification stages.

        Intermediate alternatives are consumed as one control-flow value, so
        generating ``n`` copies or retaining token logprobs only wastes work.
        A final-output grammar must likewise apply only at the answer boundary;
        constraining planner/verifier text can corrupt the orchestration DAG.
        Their token ceiling is an orchestration policy: a large public final
        allowance must not let private control text run past backend timeouts.
        """

        extra_args = dict(self.sampling_params.extra_args)
        extra_args.pop("response_format", None)
        extra_args.pop(PARALLEL_TOOL_CALLS_EXTRA_ARG, None)
        max_tokens = self.sampling_params.max_tokens
        if max_tokens_cap is not None:
            max_tokens = max_tokens_cap if max_tokens is None else min(max_tokens, max_tokens_cap)
        return self.sampling_params.clone(
            n=1,
            best_of=None,
            logprobs=None,
            max_tokens=max_tokens,
            extra_args=extra_args,
        )


def default_orchestration_request(
    prompt: str,
    sampling_params: SamplingParams,
) -> OrchestrationRequest:
    """Build the backwards-compatible request used by the Python facade."""

    return OrchestrationRequest(
        prompt=prompt,
        sampling_params=sampling_params,
        response_format=sampling_params.extra_args.get("response_format"),
    )
