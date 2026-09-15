"""Fast-head streaming, continuation dedup, and per-role sampling (EO-D7/D9)."""

import asyncio

import pytest

from kairyu.engine.backend import (
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
)
from kairyu.orchestration.budget import Budget
from kairyu.orchestration.conductor import (
    Conductor,
    ConductorStreamError,
    RoleSamplingOverrides,
    RoleSpec,
)
from kairyu.outputs import CompletionOutput
from kairyu.sampling_params import SamplingParams


class StreamScriptedBackend:
    """Streams cumulative chunks; generate() returns the full text."""

    def __init__(self, chunks: list[str], *, fail_after: int | None = None) -> None:
        self._chunks = list(chunks)
        self._fail_after = fail_after
        self.requests_seen: list[GenerationRequest] = []

    def _result(self, request, text, *, finished):
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0,
                    text=text,
                    token_ids=(),
                    finish_reason="stop" if finished else None,
                ),
            ),
            usage=GenerationUsage(prompt_tokens=3, completion_tokens=len(text)),
            finished=finished,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests_seen.append(request)
        return self._result(request, self._chunks[-1], finished=True)

    async def stream(self, request: GenerationRequest):
        self.requests_seen.append(request)
        for index, cumulative in enumerate(self._chunks):
            if self._fail_after is not None and index >= self._fail_after:
                raise RuntimeError("scripted stream failure")
            yield self._result(
                request,
                cumulative,
                finished=index == len(self._chunks) - 1,
            )

    async def shutdown(self) -> None:
        return None


class UsageFreeBackend(StreamScriptedBackend):
    """Head backend with optional exact token IDs but no usage object."""

    def __init__(self, text: str, *, token_ids: tuple[int, ...] = ()) -> None:
        super().__init__([text])
        self._token_ids = token_ids

    def _result(self, request, text, *, finished):
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0,
                    text=text,
                    token_ids=self._token_ids,
                    finish_reason="stop" if finished else None,
                ),
            ),
            usage=None,
            finished=finished,
        )


class GatedScriptedBackend:
    """Holds generate() until released so head concurrency is observable."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.started.set()
        await self.release.wait()
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(CompletionOutput(index=0, text=self._text, token_ids=()),),
        )

    async def stream(self, request):  # pragma: no cover - unused
        yield await self.generate(request)

    async def shutdown(self) -> None:
        return None


def _head_roles() -> tuple[RoleSpec, ...]:
    return (
        RoleSpec(name="head", worker="hw", role_type="head", prompt="[head] {query}"),
        RoleSpec(
            name="draft",
            worker="dw",
            prompt="[draft] {query} :: {head}",
            depends_on=("head",),
        ),
        RoleSpec(
            name="continuation",
            worker="cw",
            role_type="publisher",
            prompt="[cont] {head} :: {draft}",
            depends_on=("head", "draft"),
        ),
    )


def _conductor(head_backend, draft_backend, continuation_backend, **kwargs):
    return Conductor(
        _head_roles(),
        {"hw": head_backend, "dw": draft_backend, "cw": continuation_backend},
        **kwargs,
    )


async def _collect(stream):
    events = []
    async for event in stream:
        events.append(event)
    return events


async def test_head_streams_public_deltas_before_dag_completes():
    head = StreamScriptedBackend(["Intro ", "Intro done. "])
    draft = GatedScriptedBackend("draft body")
    continuation = StreamScriptedBackend(["Final answer."])
    conductor = _conductor(head, draft, continuation)

    events = []
    saw_delta_while_draft_pending = False
    async for event in conductor.stream("task"):
        if event.kind == "delta" and not draft.release.is_set():
            saw_delta_while_draft_pending = True
            draft.release.set()
        events.append(event)
    assert saw_delta_while_draft_pending, "head must stream before the DAG finishes"
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Intro done. Final answer."
    assert "".join(e.text for e in events if e.kind == "delta") == result.final_text
    assert result.completions[0].text == result.final_text


async def test_reasoning_events_emit_incrementally_while_units_pending():
    head = StreamScriptedBackend(["Intro. "])
    fast_unit = StreamScriptedBackend(["fast unit output"])
    slow_unit = GatedScriptedBackend("slow unit output")
    continuation = StreamScriptedBackend(["Final."])
    roles = (
        RoleSpec(name="head", worker="hw", role_type="head", prompt="[head] {query}"),
        RoleSpec(name="fast", worker="fw", prompt="[fast] {query} {head}",
                 depends_on=("head",)),
        RoleSpec(name="slow", worker="sw", prompt="[slow] {query} {head}",
                 depends_on=("head",)),
        RoleSpec(
            name="continuation",
            worker="cw",
            role_type="publisher",
            prompt="[cont] {head} {fast} {slow}",
            depends_on=("head", "fast", "slow"),
        ),
    )
    conductor = Conductor(
        roles,
        {"hw": head, "fw": fast_unit, "sw": slow_unit, "cw": continuation},
        expose_intermediate_outputs=True,
    )

    saw_reasoning_while_slow_pending = False
    events = []
    async for event in conductor.stream("task"):
        if event.kind == "reasoning" and not slow_unit.release.is_set():
            saw_reasoning_while_slow_pending = True
        if event.kind in {"delta", "reasoning"} and not slow_unit.release.is_set():
            # Only release once the fast unit's section proved incremental.
            if saw_reasoning_while_slow_pending:
                slow_unit.release.set()
        events.append(event)
    assert saw_reasoning_while_slow_pending, (
        "a completed unit's reasoning section must stream while siblings run"
    )
    result = events[-1].result
    assert result is not None
    assert "fast unit output" in (result.reasoning_content or "")
    # The committed head is public content and must not be duplicated into
    # the reasoning sections.
    assert "Intro." not in (result.reasoning_content or "")


async def test_continuation_dedupes_exact_head_repetition():
    head = StreamScriptedBackend(["Intro. "])
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["Intro. ", "Intro. The rest."])
    conductor = _conductor(head, draft, continuation)

    events = await _collect(conductor.stream("task"))
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Intro. The rest."
    assert "".join(e.text for e in events if e.kind == "delta") == result.final_text


async def test_continuation_divergent_text_flushes_verbatim():
    head = StreamScriptedBackend(["Intro. "])
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["Int", "Interesting tail."])
    conductor = _conductor(head, draft, continuation)

    events = await _collect(conductor.stream("task"))
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Intro. Interesting tail."


async def test_continuation_sentinel_declines_and_head_is_whole_answer():
    # EO-D7 amendment (issue #495): a publisher that finds the committed
    # opening complete answers exactly NO_CONTINUATION; the conductor must
    # suppress it instead of publishing deliberation-shaped text.
    head = StreamScriptedBackend(["Intro done."])
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["NO_", "NO_CONTINUATION"])
    conductor = _conductor(head, draft, continuation)

    events = await _collect(conductor.stream("task"))
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Intro done."
    assert "".join(e.text for e in events if e.kind == "delta") == "Intro done."

    # The non-streaming merge applies the same suppression.
    run_conductor = _conductor(
        StreamScriptedBackend(["Intro done."]),
        StreamScriptedBackend(["draft"]),
        StreamScriptedBackend(["NO_CONTINUATION"]),
    )
    run_result = await run_conductor.run("task")
    assert run_result.final_text == "Intro done."


@pytest.mark.parametrize("stream", [False, True])
async def test_verified_draft_equal_to_head_skips_publisher(stream):
    roles = (
        RoleSpec(name="head", worker="hw", role_type="head", prompt="[head] {query}"),
        RoleSpec(
            name="draft",
            worker="dw",
            prompt="[draft] {query} :: {head}",
            depends_on=("head",),
        ),
        RoleSpec(
            name="check",
            worker="vw",
            prompt="[check] {draft}",
            role_type="verifier",
            verifies="draft",
            depends_on=("draft",),
        ),
        RoleSpec(
            name="continuation",
            worker="cw",
            role_type="publisher",
            prompt="[cont] {head} :: {draft} :: {check}",
            depends_on=("head", "draft", "check"),
        ),
    )
    continuation = StreamScriptedBackend(
        ['The committed opening is "OK" and already answers the request.']
    )
    conductor = Conductor(
        roles,
        {
            "hw": StreamScriptedBackend(["OK"]),
            "dw": StreamScriptedBackend(["OK"]),
            "vw": StreamScriptedBackend(["PASS"]),
            "cw": continuation,
        },
    )

    if stream:
        events = await _collect(conductor.stream("task"))
        result = events[-1].result
        assert result is not None
    else:
        result = await conductor.run("task")

    assert result.final_text == "OK"
    assert continuation.requests_seen == []
    skipped = [
        event for event in result.trace if event.kind == "skipped:complete_head"
    ]
    assert skipped and skipped[0].metadata["reason"] == "verified_complete_head"


async def test_continuation_sentinel_divergence_flushes_verbatim():
    # Only an exact sentinel-only output is suppressed; anything longer is
    # published verbatim so content is never lost.
    head = StreamScriptedBackend(["Intro done."])
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["NO_CONTINUATION", "NO_CONTINUATION needed: more."])
    conductor = _conductor(head, draft, continuation)

    events = await _collect(conductor.stream("task"))
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Intro done.NO_CONTINUATION needed: more."


async def test_head_zero_byte_failure_degrades_to_continuation_only():
    head = StreamScriptedBackend(["never"], fail_after=0)
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["Standalone answer."])
    conductor = _conductor(head, draft, continuation)

    events = await _collect(conductor.stream("task"))
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Standalone answer."
    failed = [e for e in result.trace if e.node == "head" and e.kind == "failed"]
    assert failed and failed[0].metadata.get("partial") is False


async def test_head_failure_after_bytes_commits_partial_prefix():
    head = StreamScriptedBackend(["Intro ", "never"], fail_after=1)
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["continues."])
    conductor = _conductor(head, draft, continuation)

    events = await _collect(conductor.stream("task"))
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Intro continues."
    assert result.outputs["head"] == "Intro "
    # Downstream prompts saw the committed partial prefix.
    assert ":: Intro " in draft.requests_seen[0].prompt
    failed = [e for e in result.trace if e.node == "head" and e.kind == "failed"]
    assert failed and failed[0].metadata.get("partial") is True


async def test_continuation_failure_carries_committed_text_in_stream_error():
    head = StreamScriptedBackend(["Intro. "])
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["Part ", "never"], fail_after=1)
    conductor = _conductor(head, draft, continuation)

    emitted = ""
    with pytest.raises(ConductorStreamError) as excinfo:
        async for event in conductor.stream("task"):
            if event.kind == "delta":
                emitted += event.text
    assert excinfo.value.result.final_text == emitted == "Intro. Part "
    # The failed stream released its reservation.
    state = excinfo.value.result.budget_state
    assert state.steps_reserved == 0


async def test_head_skipped_for_incompatible_public_intent():
    head = StreamScriptedBackend(["Intro. "])
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["Full answer."])
    conductor = _conductor(
        head,
        draft,
        continuation,
        final_sampling_params=SamplingParams(max_tokens=64, n=2),
    )

    events = await _collect(conductor.stream("task"))
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Full answer."
    assert "head" not in result.outputs
    skipped = [e for e in result.trace if e.kind == "skipped:intent"]
    assert skipped and skipped[0].node == "head"
    # Non-streaming parity: the head stays disabled in run() too.
    run_result = await conductor.run("task")
    assert run_result.final_text == "Full answer."
    assert "head" not in run_result.outputs


async def test_head_skipped_for_structured_format_demand():
    # Issue #495: a plain-text demand for a machine-parsed format (agent
    # harnesses ask "format your response as JSON" without tools or
    # response_format) disables the head like any other incompatible intent.
    head = StreamScriptedBackend(["Intro. "])
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(['{"analysis": "..."}'])
    conductor = _conductor(
        head,
        draft,
        continuation,
        final_structured_format_in_prompt=True,
    )

    result = await conductor.run("task")
    assert result.final_text == '{"analysis": "..."}'
    assert "head" not in result.outputs
    skipped = [e for e in result.trace if e.kind == "skipped:intent"]
    assert skipped and skipped[0].node == "head"


async def test_run_and_stream_concatenation_parity():
    def build():
        return _conductor(
            StreamScriptedBackend(["Intro. "]),
            StreamScriptedBackend(["draft"]),
            StreamScriptedBackend(["Body."]),
        )

    run_result = await build().run("task")
    events = await _collect(build().stream("task"))
    stream_result = events[-1].result
    assert stream_result is not None
    assert run_result.final_text == stream_result.final_text == "Intro. Body."
    assert run_result.completions[0].text == "Intro. Body."


async def test_head_reserves_step_within_max_steps():
    head = StreamScriptedBackend(["Intro. "])
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["Body."])
    conductor = _conductor(head, draft, continuation)

    events = await _collect(
        conductor.stream("task", budget=Budget(max_steps=3))
    )
    result = events[-1].result
    assert result is not None
    assert result.budget_state.steps_used == 3
    # One step fewer and the continuation is refused, best-so-far = head text.
    events = await _collect(
        _conductor(
            StreamScriptedBackend(["Intro. "]),
            StreamScriptedBackend(["draft"]),
            StreamScriptedBackend(["Body."]),
        ).stream("task", budget=Budget(max_steps=2))
    )
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Intro. "


async def test_per_role_sampling_overrides_reach_backend_requests():
    head = StreamScriptedBackend(["Intro. "])
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["Body."])
    roles = (
        RoleSpec(
            name="head",
            worker="hw",
            role_type="head",
            prompt="[head] {query}",
            sampling=RoleSamplingOverrides(
                temperature=0.3,
                max_tokens=256,
                stop=("\n## ",),
            ),
        ),
        RoleSpec(
            name="draft",
            worker="dw",
            prompt="[draft] {query} {head}",
            depends_on=("head",),
            sampling=RoleSamplingOverrides(
                temperature=0.9,
                top_k=20,
                min_p=0.05,
                presence_penalty=1.5,
                repetition_penalty=1.05,
                max_tokens=9000,
                seed_offset=2,
            ),
        ),
        RoleSpec(
            name="continuation",
            worker="cw",
            role_type="publisher",
            prompt="[cont] {head} {draft}",
            depends_on=("head", "draft"),
        ),
    )
    conductor = Conductor(
        roles,
        {"hw": head, "dw": draft, "cw": continuation},
        sampling_params=SamplingParams(max_tokens=1024, seed=7),
        final_sampling_params=SamplingParams(max_tokens=4096, temperature=0.6),
    )
    await _collect(conductor.stream("task"))

    head_params = head.requests_seen[0].sampling_params
    assert head_params.temperature == 0.3
    assert head_params.max_tokens == 256
    assert head_params.stop == ("\n## ",)
    assert head_params.n == 1
    draft_params = draft.requests_seen[0].sampling_params
    assert draft_params.temperature == 0.9
    assert draft_params.top_k == 20
    assert draft_params.min_p == 0.05
    assert draft_params.presence_penalty == 1.5
    assert draft_params.repetition_penalty == 1.05
    # Internal ceiling caps the override (m11 #208).
    assert draft_params.max_tokens == 1024
    assert draft_params.seed == 9
    continuation_params = continuation.requests_seen[0].sampling_params
    # The caller's public limit is one budget across the head/continuation
    # seam (issue #496): the head reported 7 completion tokens, so the
    # continuation may generate at most 4096 - 7.
    assert continuation_params.max_tokens == 4089
    assert continuation_params.temperature == 0.6


@pytest.mark.parametrize(
    ("role_cap", "expected_max"),
    [
        (131072, 4096),  # the role cap only shrinks the caller's allowance
        (2000, 2000),
    ],
)
async def test_final_role_sampling_override_is_policy_capped_by_caller(role_cap, expected_max):
    """DTO-D13: a final-unit sampling block sets the style fields and caps
    max_tokens by the caller's public limit; the caller's n stays."""

    backend = StreamScriptedBackend(["x"])
    roles = (
        RoleSpec(
            name="solo",
            worker="w",
            role_type="publisher",
            prompt="{query}",
            sampling=RoleSamplingOverrides(
                temperature=0.7, top_p=0.8, presence_penalty=1.5, max_tokens=role_cap
            ),
        ),
    )
    conductor = Conductor(
        roles,
        {"w": backend},
        final_sampling_params=SamplingParams(max_tokens=4096, temperature=0.2, top_p=1.0),
    )
    await _collect(conductor.stream("task"))

    params = backend.requests_seen[0].sampling_params
    assert params.temperature == 0.7
    assert params.top_p == 0.8
    assert params.presence_penalty == 1.5
    assert params.max_tokens == expected_max
    assert params.n == 1


def test_head_dag_shape_validation():
    backend = StreamScriptedBackend(["x"])
    with pytest.raises(ValueError, match="dependencies"):
        Conductor(
            (
                RoleSpec(name="head", worker="w", role_type="head",
                         prompt="{query}", depends_on=("other",)),
                RoleSpec(name="other", worker="w", prompt="{query}"),
            ),
            {"w": backend},
        )
    with pytest.raises(ValueError, match="at most one"):
        Conductor(
            (
                RoleSpec(name="h1", worker="w", role_type="head", prompt="{query}"),
                RoleSpec(name="h2", worker="w", role_type="head", prompt="{query}"),
                RoleSpec(name="final", worker="w", prompt="{h1}{h2}",
                         depends_on=("h1", "h2")),
            ),
            {"w": backend},
        )
    with pytest.raises(ValueError, match="depend on"):
        # The selected final unit (the synthesizer) does not transitively
        # depend on the head, so it could not continue the committed prefix.
        Conductor(
            (
                RoleSpec(name="head", worker="w", role_type="head", prompt="{query}"),
                RoleSpec(name="uses", worker="w", prompt="{head}",
                         depends_on=("head",)),
                RoleSpec(name="final", worker="w", role_type="synthesizer",
                         prompt="{query}"),
            ),
            {"w": backend},
        )


def test_initial_requests_include_enabled_head_and_exclude_disabled():
    def build(**kwargs):
        return _conductor(
            StreamScriptedBackend(["Intro. "]),
            StreamScriptedBackend(["draft"]),
            StreamScriptedBackend(["Body."]),
            **kwargs,
        )

    enabled = build()
    names = [spec.name for spec, _ in enabled.initial_requests("q", request_id_suffix="s")]
    assert names == ["head"]
    disabled = build(final_sampling_params=SamplingParams(max_tokens=64, n=2))
    assert [
        spec.name for spec, _ in disabled.initial_requests("q", request_id_suffix="s")
    ] == []


async def test_prompt_suffix_survives_refinement_attempts():
    # A worker-native scaffold (e.g. a chat assistant marker) must terminate
    # EVERY attempt's prompt; refinement feedback goes before it, or the
    # refined generation runs against a malformed prompt.
    class Recorder:
        def __init__(self, responses):
            self._responses = list(responses)
            self.prompts = []

        async def generate(self, request):
            from kairyu.engine.backend import GenerationResult

            self.prompts.append(request.prompt)
            return GenerationResult(
                request_id=request.request_id,
                prompt=request.prompt,
                completions=(
                    CompletionOutput(
                        index=0, text=self._responses.pop(0), token_ids=()
                    ),
                ),
            )

        async def stream(self, request):  # pragma: no cover - unused
            yield await self.generate(request)

        async def shutdown(self):
            return None

    backend = Recorder(["draft-1", "FAIL: broken", "draft-2", "PASS", "final"])
    roles = (
        RoleSpec(
            name="draft",
            worker="w",
            prompt="[draft] {query}",
            prompt_suffix="<ASSISTANT>",
        ),
        RoleSpec(
            name="check",
            worker="w",
            role_type="verifier",
            verifies="draft",
            prompt="[check] {draft}",
            prompt_suffix="<THINK>",
            depends_on=("draft",),
        ),
        RoleSpec(
            name="final",
            worker="w",
            role_type="synthesizer",
            prompt="[final] {draft} {check}",
            depends_on=("draft", "check"),
        ),
    )
    conductor = Conductor(roles, {"w": backend})
    result = await conductor.run("task")

    assert result.final_text == "final"
    draft_prompts = [p for p in backend.prompts if p.startswith("[draft]")]
    assert len(draft_prompts) == 2
    assert all(p.endswith("<ASSISTANT>") for p in draft_prompts)
    assert "Verifier feedback:" in draft_prompts[1]
    assert draft_prompts[1].index("Verifier feedback:") < draft_prompts[1].index(
        "<ASSISTANT>"
    )
    check_prompts = [p for p in backend.prompts if p.startswith("[check]")]
    assert all(p.endswith("<THINK>") for p in check_prompts)


class ReasoningOnlyBackend:
    """Simulates an upstream reasoning parser eating a direct answer.

    The generated text carries no emitted </think>, so the parser classifies
    the whole answer as reasoning and public text arrives empty (issue #496).
    """

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.requests_seen: list[GenerationRequest] = []

    def _result(self, request):
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0,
                    text="",
                    token_ids=(),
                    finish_reason="stop",
                    reasoning_content=self._answer,
                ),
            ),
            usage=GenerationUsage(prompt_tokens=3, completion_tokens=4),
            finished=True,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests_seen.append(request)
        return self._result(request)

    async def stream(self, request):
        self.requests_seen.append(request)
        yield self._result(request)

    async def shutdown(self) -> None:
        return None


class MixedChoiceBackend:
    """Returns an empty first choice and a usable second choice."""

    def __init__(self) -> None:
        self.requests_seen: list[GenerationRequest] = []

    def _result(self, request):
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0,
                    text="",
                    token_ids=(),
                    finish_reason="stop",
                ),
                CompletionOutput(
                    index=1,
                    text="valid second choice",
                    token_ids=(),
                    finish_reason="stop",
                ),
            ),
            usage=GenerationUsage(prompt_tokens=3, completion_tokens=3),
            finished=True,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests_seen.append(request)
        return self._result(request)

    async def stream(self, request):
        self.requests_seen.append(request)
        yield self._result(request)

    async def shutdown(self) -> None:
        return None


@pytest.mark.parametrize("stream", [False, True])
async def test_empty_first_choice_preserves_nonempty_later_choice(stream):
    backend = MixedChoiceBackend()
    conductor = Conductor(
        (
            RoleSpec(
                name="final",
                worker="w",
                role_type="publisher",
                prompt="{query}",
            ),
        ),
        {"w": backend},
        final_sampling_params=SamplingParams(n=2, max_tokens=32),
    )

    if stream:
        events = await _collect(conductor.stream("task"))
        result = events[-1].result
        assert result is not None
    else:
        result = await conductor.run("task")

    assert result.final_unit_ok
    assert [(choice.index, choice.text) for choice in result.completions] == [
        (0, ""),
        (1, "valid second choice"),
    ]
    assert len(backend.requests_seen) == 1
    assert not any(
        event.kind in {"retry:empty_output", "failed"} for event in result.trace
    )


async def test_caller_limit_is_one_budget_across_head_and_continuation():
    """Issue #496 item 1: max_tokens bounds head + continuation together."""

    def build():
        head = StreamScriptedBackend(["Intro. "])  # reports 7 completion tokens
        draft = StreamScriptedBackend(["draft"])
        continuation = StreamScriptedBackend(["Body."])
        return head, continuation, _conductor(
            head,
            draft,
            continuation,
            sampling_params=SamplingParams(max_tokens=1024),
            final_sampling_params=SamplingParams(max_tokens=100),
        )

    head, continuation, conductor = build()
    events = await _collect(conductor.stream("task"))
    result = events[-1].result
    assert result is not None
    # The head role declares no override: its cap is min(internal, caller).
    assert head.requests_seen[0].sampling_params.max_tokens == 100
    # The continuation gets the remaining public budget: 100 - 7.
    assert continuation.requests_seen[0].sampling_params.max_tokens == 93
    assert result.completions[0].finish_reason == "stop"
    # Backend-reported public tokens: head 7 + continuation 5.
    assert result.public_completion_tokens == 12

    head, continuation, conductor = build()
    run_result = await conductor.run("task")
    assert continuation.requests_seen[0].sampling_params.max_tokens == 93
    assert run_result.public_completion_tokens == 12


@pytest.mark.parametrize("stream", [False, True])
async def test_missing_head_usage_without_token_ids_consumes_head_cap(stream):
    head_text = "漢字漢字漢字漢字漢字"
    head = UsageFreeBackend(head_text)
    continuation = StreamScriptedBackend(["Body."])
    conductor = _conductor(
        head,
        StreamScriptedBackend(["draft"]),
        continuation,
        sampling_params=SamplingParams(max_tokens=1024),
        final_sampling_params=SamplingParams(max_tokens=10),
    )

    if stream:
        events = await _collect(conductor.stream("task"))
        result = events[-1].result
        assert result is not None
    else:
        result = await conductor.run("task")

    assert head.requests_seen[0].sampling_params.max_tokens == 10
    assert not continuation.requests_seen
    assert result.final_text == head_text
    assert result.completions[0].finish_reason == "length"


@pytest.mark.parametrize("stream", [False, True])
async def test_head_token_ids_size_remaining_budget_without_usage(stream):
    head = UsageFreeBackend("漢字漢", token_ids=(11, 12, 13))
    continuation = StreamScriptedBackend(["Body."])
    conductor = _conductor(
        head,
        StreamScriptedBackend(["draft"]),
        continuation,
        sampling_params=SamplingParams(max_tokens=1024),
        final_sampling_params=SamplingParams(max_tokens=10),
    )

    if stream:
        events = await _collect(conductor.stream("task"))
        result = events[-1].result
        assert result is not None
    else:
        result = await conductor.run("task")

    assert continuation.requests_seen[0].sampling_params.max_tokens == 7
    assert result.completions[0].finish_reason == "stop"


async def test_head_role_cap_is_clamped_to_caller_limit():
    head = StreamScriptedBackend(["Intro. "])
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["Body."])
    roles = (
        RoleSpec(
            name="head",
            worker="hw",
            role_type="head",
            prompt="[head] {query}",
            sampling=RoleSamplingOverrides(max_tokens=256),
        ),
    ) + _head_roles()[1:]
    conductor = Conductor(
        roles,
        {"hw": head, "dw": draft, "cw": continuation},
        sampling_params=SamplingParams(max_tokens=1024),
        final_sampling_params=SamplingParams(max_tokens=40),
    )
    await _collect(conductor.stream("task"))
    assert head.requests_seen[0].sampling_params.max_tokens == 40


async def test_exhausted_public_budget_skips_continuation_and_reports_length():
    """Issue #496 item 2 inverse: a truncating limit reports length honestly."""

    def build():
        head = StreamScriptedBackend(["Intro. "])  # consumes all 7 tokens
        draft = StreamScriptedBackend(["draft"])
        continuation = StreamScriptedBackend(["Body."])
        return continuation, _conductor(
            head,
            draft,
            continuation,
            sampling_params=SamplingParams(max_tokens=1024),
            final_sampling_params=SamplingParams(max_tokens=7),
        )

    continuation, conductor = build()
    events = await _collect(conductor.stream("task"))
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Intro. "
    assert result.completions[0].finish_reason == "length"
    assert not continuation.requests_seen
    assert any(e.kind == "skipped:public_budget" for e in result.trace)

    continuation, conductor = build()
    run_result = await conductor.run("task")
    assert run_result.final_text == "Intro. "
    assert run_result.completions[0].finish_reason == "length"
    assert not continuation.requests_seen


async def test_reasoning_closed_reclaims_parser_classified_answer():
    """Issue #496 item 4: a closed-reasoning publisher's answer is public."""

    roles = (
        RoleSpec(
            name="continuation",
            worker="cw",
            role_type="publisher",
            prompt="[cont] {query}",
            prompt_suffix="</think>",
            reasoning_closed=True,
        ),
    )

    run_result = await Conductor(
        roles, {"cw": ReasoningOnlyBackend("The answer is 5.")}
    ).run("task")
    assert run_result.final_text == "The answer is 5."
    assert run_result.completions[0].text == "The answer is 5."
    assert run_result.completions[0].reasoning_content is None
    assert run_result.final_unit_ok

    events = await _collect(
        Conductor(roles, {"cw": ReasoningOnlyBackend("The answer is 5.")}).stream(
            "task"
        )
    )
    stream_result = events[-1].result
    assert stream_result is not None
    assert stream_result.final_text == "The answer is 5."
    # The reclaimed answer still reaches the stream as public deltas.
    assert (
        "".join(e.text for e in events if e.kind == "delta") == "The answer is 5."
    )


async def test_headless_prompt_renders_when_tools_disable_the_head():
    head = StreamScriptedBackend(["Intro. "])
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["Tool answer."])
    roles = _head_roles()[:2] + (
        RoleSpec(
            name="continuation",
            worker="cw",
            role_type="publisher",
            prompt="[cont] {head} :: {draft}",
            prompt_headless="[cont-headless] {query} :: {draft}",
            depends_on=("head", "draft"),
        ),
    )
    conductor = Conductor(
        roles,
        {"hw": head, "dw": draft, "cw": continuation},
        final_tools=({"type": "function", "function": {"name": "add"}},),
    )
    result = await conductor.run("task")
    assert result.final_text == "Tool answer."
    assert continuation.requests_seen[0].prompt.startswith("[cont-headless] task")
    # With the head enabled the ordinary continuation body is used.
    head2 = StreamScriptedBackend(["Intro. "])
    continuation2 = StreamScriptedBackend(["Body."])
    conductor2 = Conductor(
        roles,
        {"hw": head2, "dw": StreamScriptedBackend(["draft"]), "cw": continuation2},
    )
    await conductor2.run("task")
    assert continuation2.requests_seen[0].prompt.startswith("[cont] Intro. ")


async def test_empty_final_output_retries_once_then_marks_run_unusable():
    """Issue #496 items 3/4: an empty answer is never a silent stop."""

    empty = StreamScriptedBackend(["", ""])
    draft = StreamScriptedBackend(["draft"])
    roles = (
        RoleSpec(name="draft", worker="dw", prompt="[draft] {query}"),
        RoleSpec(
            name="continuation",
            worker="cw",
            role_type="publisher",
            prompt="[cont] {draft}",
            depends_on=("draft",),
        ),
    )
    conductor = Conductor(roles, {"dw": draft, "cw": empty})
    result = await conductor.run("task")
    assert result.final_text == ""
    assert not result.final_unit_ok
    assert len(empty.requests_seen) == 2  # one bounded retry
    assert any(e.kind == "retry:empty_output" for e in result.trace)
    assert any(
        e.kind == "failed" and e.detail == "empty_output" for e in result.trace
    )
    # An internal stage's text never substitutes for the public answer.
    assert "draft" not in result.final_text


@pytest.mark.parametrize("refusal", ["max_steps", "max_cost"])
async def test_empty_final_retry_budget_refusal_marks_run_unusable(refusal):
    empty = StreamScriptedBackend([""])
    draft = StreamScriptedBackend(["draft"])
    roles = (
        RoleSpec(name="draft", worker="dw", prompt="[draft] {query}"),
        RoleSpec(
            name="continuation",
            worker="cw",
            role_type="publisher",
            prompt="[cont] {draft}",
            depends_on=("draft",),
        ),
    )
    cost_model = (
        (lambda request, result: 1.0 if request.prompt.startswith("[cont]") else 0.0)
        if refusal == "max_cost"
        else (lambda request, result: 0.0)
    )
    budget = (
        Budget(max_cost_usd=0.5)
        if refusal == "max_cost"
        else Budget(max_steps=2)
    )
    conductor = Conductor(
        roles,
        {"dw": draft, "cw": empty},
        cost_model=cost_model,
    )

    result = await conductor.run("task", budget=budget)

    assert result.final_text == ""
    assert not result.final_unit_ok
    assert result.outputs == {"draft": "draft"}
    assert len(empty.requests_seen) == 1
    assert any(e.kind == "retry:empty_output" for e in result.trace)
    assert any(e.kind == "skipped:budget" and e.attempt == 1 for e in result.trace)
    assert any(
        e.kind == "failed" and e.detail == "empty_output" and e.attempt == 1
        for e in result.trace
    )


async def test_committed_head_exempts_empty_continuation_from_failure():
    head = StreamScriptedBackend(["Intro. "])
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend([""])
    conductor = _conductor(head, draft, continuation)
    result = await conductor.run("task")
    # EO-D7 best-so-far: the committed head stands alone, no error.
    assert result.final_text == "Intro. "
    assert result.final_unit_ok
    assert len(continuation.requests_seen) == 1


class ThinkExhaustedBackend:
    """Reasoning-parser view of a thinking role: text empty, thinking kept.

    Returns the queued reasoning strings one call at a time, so attempt 0
    models deliberation that ate the whole budget and attempt 1 models the
    forced-close continuation whose output the parser still classifies as
    reasoning (issue #542).
    """

    def __init__(self, reasonings: list[str]) -> None:
        self._reasonings = list(reasonings)
        self.requests_seen: list[GenerationRequest] = []

    def _result(self, request):
        reasoning = self._reasonings.pop(0)
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0,
                    text="",
                    token_ids=(),
                    finish_reason="length",
                    # Delta-native shape of a reasoning-only OpenAI stream:
                    # no public text was ever offset.
                    text_delta="",
                    text_offset=0,
                    reasoning_content=reasoning or None,
                    reasoning_delta=reasoning,
                    reasoning_offset=0,
                ),
            ),
            usage=GenerationUsage(prompt_tokens=3, completion_tokens=4),
            finished=True,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests_seen.append(request)
        return self._result(request)

    async def stream(self, request):
        self.requests_seen.append(request)
        yield self._result(request)

    async def shutdown(self) -> None:
        return None

class MultiChoiceThinkBackend:
    """Returns two independent reasoning branches, then two public answers."""

    def __init__(self) -> None:
        self.requests_seen: list[GenerationRequest] = []
        self.generate_calls = 0
        self.stream_calls = 0

    def _result(self, request):
        if len(self.requests_seen) == 1:
            choices = (("", "choice-0 thought"), ("", "choice-1 thought"))
        else:
            choices = (("answer-0", None), ("answer-1", None))
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=tuple(
                CompletionOutput(
                    index=index,
                    text=text,
                    token_ids=(),
                    finish_reason="length" if reasoning else "stop",
                    reasoning_content=reasoning,
                )
                for index, (text, reasoning) in enumerate(choices)
            ),
            usage=GenerationUsage(prompt_tokens=3, completion_tokens=4),
            finished=True,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.generate_calls += 1
        self.requests_seen.append(request)
        return self._result(request)

    async def stream(self, request):
        self.stream_calls += 1
        self.requests_seen.append(request)
        yield self._result(request)

    async def shutdown(self) -> None:
        return None


def _floor_roles() -> tuple[RoleSpec, ...]:
    return (
        RoleSpec(
            name="continuation",
            worker="cw",
            role_type="publisher",
            prompt="[cont] {query}",
            prompt_suffix="<THINK>",
            reasoning_close_tag="</THINK>",
        ),
    )


@pytest.mark.parametrize(
    ("caller_cap", "expected_attempt0_max", "expected_retry_max"),
    [
        (512, 448, 64),  # floor reserved out of the thinking budget
        (32, 32, 32),  # at or below the floor: unclamped, retry gets it all
    ],
)
async def test_public_output_floor_reserves_answer_budget_and_closes_think(
    caller_cap, expected_attempt0_max, expected_retry_max
):
    """Issue #542: a small public cap must not 502 on think exhaustion."""

    backend = ThinkExhaustedBackend(["deliberating...", "The answer is 5."])
    conductor = Conductor(
        _floor_roles(),
        {"cw": backend},
        final_sampling_params=SamplingParams(max_tokens=caller_cap),
        public_output_floor=64,
    )
    result = await conductor.run("task")

    assert result.final_text == "The answer is 5."
    assert result.final_unit_ok
    first, retry = backend.requests_seen
    assert first.prompt == "[cont] task<THINK>"
    assert first.sampling_params.max_tokens == expected_attempt0_max
    assert retry.prompt == "[cont] task<THINK>deliberating...</THINK>\n\n"
    assert retry.sampling_params.max_tokens == expected_retry_max
    retry_events = [e for e in result.trace if e.kind == "retry:empty_output"]
    assert [e.metadata.get("continuation") for e in retry_events] == ["think_close"]

@pytest.mark.parametrize("stream", [False, True])
async def test_public_output_floor_uses_legacy_retry_for_multiple_choices(stream):
    backend = MultiChoiceThinkBackend()
    conductor = Conductor(
        _floor_roles(),
        {"cw": backend},
        final_sampling_params=SamplingParams(n=2, max_tokens=64),
        public_output_floor=16,
    )

    if stream:
        events = await _collect(conductor.stream("task"))
        result = events[-1].result
        assert result is not None
    else:
        result = await conductor.run("task")

    assert result.final_unit_ok
    assert [(choice.index, choice.text) for choice in result.completions] == [
        (0, "answer-0"),
        (1, "answer-1"),
    ]
    first, retry = backend.requests_seen
    assert first.prompt == retry.prompt == "[cont] task<THINK>"
    assert [request.sampling_params.max_tokens for request in (first, retry)] == [64, 64]
    retry_event = next(event for event in result.trace if event.kind == "retry:empty_output")
    assert retry_event.metadata.get("continuation") is None


async def test_public_output_floor_streaming_retry_streams_reclaimed_answer():
    backend = ThinkExhaustedBackend(["deliberating...", "The answer is 5."])
    conductor = Conductor(
        _floor_roles(),
        {"cw": backend},
        final_sampling_params=SamplingParams(max_tokens=512),
        public_output_floor=64,
    )
    events = await _collect(conductor.stream("task"))

    stream_result = events[-1].result
    assert stream_result is not None
    assert stream_result.final_text == "The answer is 5."
    assert stream_result.final_unit_ok
    first, retry = backend.requests_seen
    assert first.sampling_params.max_tokens == 448
    assert retry.prompt == "[cont] task<THINK>deliberating...</THINK>\n\n"
    assert retry.sampling_params.max_tokens == 64
    assert (
        "".join(e.text for e in events if e.kind == "delta") == "The answer is 5."
    )


async def test_public_output_floor_second_empty_still_marks_run_unusable():
    backend = ThinkExhaustedBackend(["deliberating...", ""])
    conductor = Conductor(
        _floor_roles(),
        {"cw": backend},
        final_sampling_params=SamplingParams(max_tokens=512),
        public_output_floor=64,
    )
    result = await conductor.run("task")

    assert result.final_text == ""
    assert not result.final_unit_ok
    assert len(backend.requests_seen) == 2
    assert any(
        e.kind == "failed" and e.detail == "empty_output" for e in result.trace
    )


def _chat_floor_roles() -> tuple[RoleSpec, ...]:
    return (
        RoleSpec(
            name="continuation",
            worker="cw",
            role_type="publisher",
            prompt="[cont] {query}",
            reasoning_close_tag="</THINK>",
            reasoning_continuation="chat",
            reasoning_open_tag="<THINK>",
        ),
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_public_output_floor_chat_continuation_prefills_closed_thinking_turn(
    stream,
):
    """DTO-D15: a chat-template worker whose upstream template opened the
    reasoning span continues it as a verbatim assistant prefill, not by
    appending to the user prompt."""

    backend = ThinkExhaustedBackend(["deliberating...", "The answer is 5."])
    conductor = Conductor(
        _chat_floor_roles(),
        {"cw": backend},
        final_sampling_params=SamplingParams(max_tokens=512),
        public_output_floor=64,
    )
    if stream:
        events = await _collect(conductor.stream("task"))
        result = events[-1].result
        assert result is not None
        assert "".join(e.text for e in events if e.kind == "delta") == "The answer is 5."
    else:
        result = await conductor.run("task")

    assert result.final_text == "The answer is 5."
    assert result.final_unit_ok
    first, retry = backend.requests_seen
    assert first.prompt == retry.prompt == "[cont] task"
    assert first.assistant_prefill is None
    assert retry.assistant_prefill == "<THINK>\ndeliberating...\n</THINK>\n\n"
    assert first.sampling_params.max_tokens == 448
    assert retry.sampling_params.max_tokens == 64
    retry_event = next(e for e in result.trace if e.kind == "retry:empty_output")
    assert retry_event.metadata.get("continuation") == "think_close"
    assert retry_event.metadata.get("mode") == "chat"


def test_public_output_floor_requires_final_reasoning_close_tag():
    roles = (
        RoleSpec(
            name="continuation",
            worker="cw",
            role_type="publisher",
            prompt="[cont] {query}",
            prompt_suffix="<THINK>",
        ),
    )
    with pytest.raises(ValueError, match="reasoning_close_tag"):
        Conductor(
            roles,
            {"cw": ThinkExhaustedBackend([""])},
            public_output_floor=64,
        )


@pytest.mark.parametrize("value", [True, 0, -1, 131_073, 1.5, "64"])
def test_public_output_floor_rejects_invalid_direct_values(value):
    with pytest.raises(ValueError, match="integer between 1 and 131072"):
        Conductor(
            _floor_roles(),
            {"cw": ThinkExhaustedBackend([""])},
            public_output_floor=value,
        )


class ScriptedFinalBackend:
    """Per-call script: ("text", s) answers, ("think", r) exhausts thinking."""

    def __init__(self, items: list[tuple[str, str]]) -> None:
        self._items = list(items)
        self.requests_seen: list[GenerationRequest] = []

    def _result(self, request):
        kind, value = self._items.pop(0)
        if kind == "think":
            completion = CompletionOutput(
                index=0,
                text="",
                token_ids=(),
                finish_reason="length",
                reasoning_content=value,
            )
        else:
            completion = CompletionOutput(
                index=0, text=value, token_ids=(), finish_reason="stop"
            )
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(completion,),
            usage=GenerationUsage(prompt_tokens=3, completion_tokens=4),
            finished=True,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests_seen.append(request)
        return self._result(request)

    async def stream(self, request):
        self.requests_seen.append(request)
        yield self._result(request)

    async def shutdown(self) -> None:
        return None


class FailingRefinementBackend(ScriptedFinalBackend):
    """Produce one candidate, then fail its requested refinement."""

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if self.requests_seen:
            self.requests_seen.append(request)
            raise RuntimeError("refinement failed")
        return await super().generate(request)


def _verified_final_roles(
    *, prompt_suffix: str = "", reasoning_close_tag: str = ""
) -> tuple[RoleSpec, ...]:
    return (
        RoleSpec(name="head", worker="hw", role_type="head", prompt="[head] {query}"),
        RoleSpec(
            name="synthesis",
            worker="cw",
            role_type="publisher",
            prompt="[syn] {query} :: {head}",
            depends_on=("head",),
            prompt_suffix=prompt_suffix,
            reasoning_close_tag=reasoning_close_tag,
        ),
        RoleSpec(
            name="audit",
            worker="vw",
            role_type="verifier",
            verifies="synthesis",
            prompt="[audit] {head} :: {synthesis}",
            depends_on=("synthesis", "head"),
        ),
    )


def _verified_floor_roles() -> tuple[RoleSpec, ...]:
    return _floor_roles() + (
        RoleSpec(
            name="check",
            worker="vw",
            role_type="verifier",
            verifies="continuation",
            prompt="[check] {continuation}",
            depends_on=("continuation",),
        ),
    )


async def test_stream_defers_verified_final_until_pass():
    """A verified final unit is published once, after its verdict (DTO-D10)."""

    head = StreamScriptedBackend(["Intro. "])
    final = ScriptedFinalBackend([("text", "continuation text")])
    audit = ScriptedFinalBackend([("text", "PASS")])
    conductor = Conductor(
        _verified_final_roles(),
        {"hw": head, "cw": final, "vw": audit},
    )

    events = await _collect(conductor.stream("task"))

    assert [e.text for e in events if e.kind == "delta"] == ["Intro. ", "continuation text"]
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Intro. continuation text"
    assert result.final_unit_ok
    assert "continuation text" in audit.requests_seen[0].prompt
    generated = next(
        e for e in result.trace if e.node == "synthesis" and e.kind == "generated"
    )
    assert "streamed" not in generated.metadata
    verified = next(e for e in result.trace if e.kind == "verified")
    assert verified.metadata["pass"] is True


async def test_stream_verified_final_fail_refines_and_emits_only_passed_text():
    head = StreamScriptedBackend(["Intro. "])
    final = ScriptedFinalBackend([("text", "v1"), ("text", "v2")])
    audit = ScriptedFinalBackend([("text", "FAIL: missing x"), ("text", "PASS")])
    conductor = Conductor(
        _verified_final_roles(),
        {"hw": head, "cw": final, "vw": audit},
    )

    events = await _collect(conductor.stream("task"))

    assert [e.text for e in events if e.kind == "delta"] == ["Intro. ", "v2"]
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Intro. v2"
    refine_prompt = final.requests_seen[1].prompt
    assert "Previous attempt:\nv1" in refine_prompt
    assert "Verifier feedback:\nFAIL: missing x" in refine_prompt
    assert [r.prompt.split(" :: ")[-1] for r in audit.requests_seen] == ["v1", "v2"]
    assert result.budget_state.steps_used == 5


async def test_stream_verified_final_exhaustion_publishes_last_attempt():
    head = StreamScriptedBackend(["Intro. "])
    final = ScriptedFinalBackend([("text", "v1"), ("text", "v2")])
    audit = ScriptedFinalBackend([("text", "FAIL: a"), ("text", "FAIL: b")])
    conductor = Conductor(
        _verified_final_roles(),
        {"hw": head, "cw": final, "vw": audit},
    )

    events = await _collect(conductor.stream("task", Budget(max_refine_depth=1)))

    assert [e.text for e in events if e.kind == "delta"] == ["Intro. ", "v2"]
    result = events[-1].result
    assert result is not None
    assert result.final_unit_ok
    verified = [e for e in result.trace if e.kind == "verified"]
    assert verified[-1].metadata["pass"] is False
    assert verified[-1].metadata["refinement_exhausted"] is True


@pytest.mark.parametrize("stream", [False, True])
async def test_verified_final_refinement_failure_drops_rejected_attempt(stream):
    final = FailingRefinementBackend([("text", "known-bad")])
    audit = ScriptedFinalBackend([("text", "FAIL: wrong answer")])
    conductor = Conductor(
        _verified_floor_roles(),
        {"cw": final, "vw": audit},
    )

    if stream:
        events = await _collect(conductor.stream("task", Budget(max_refine_depth=1)))
        result = events[-1].result
        assert result is not None
        assert [event.text for event in events if event.kind == "delta"] == []
    else:
        result = await conductor.run("task", Budget(max_refine_depth=1))

    assert result.final_text == ""
    assert not result.final_unit_ok
    assert "continuation" not in result.outputs
    assert any(event.kind == "failed" for event in result.trace)


@pytest.mark.parametrize("stream", [False, True])
async def test_public_output_floor_retries_before_verifying_final(stream):
    """Think exhaustion is continued with the reserve before any verdict."""

    final = ScriptedFinalBackend(
        [("think", "deliberating..."), ("text", "v1"), ("text", "v2")]
    )
    audit = ScriptedFinalBackend([("text", "FAIL: wrong"), ("text", "PASS")])
    conductor = Conductor(
        _verified_floor_roles(),
        {"cw": final, "vw": audit},
        final_sampling_params=SamplingParams(max_tokens=512),
        public_output_floor=64,
    )

    if stream:
        events = await _collect(conductor.stream("task"))
        result = events[-1].result
        assert result is not None
        assert [e.text for e in events if e.kind == "delta"] == ["v2"]
    else:
        result = await conductor.run("task")

    assert result.final_text == "v2"
    assert result.final_unit_ok
    first, retry, refine = final.requests_seen
    assert first.prompt == "[cont] task<THINK>"
    assert first.sampling_params.max_tokens == 448
    assert retry.prompt == "[cont] task<THINK>deliberating...</THINK>\n\n"
    assert retry.sampling_params.max_tokens == 64
    assert refine.prompt.endswith("<THINK>")
    assert "Verifier feedback:\nFAIL: wrong" in refine.prompt
    assert refine.sampling_params.max_tokens == 448
    assert [r.prompt for r in audit.requests_seen] == ["[check] v1", "[check] v2"]
    kinds = [e.kind for e in result.trace if e.node in {"continuation", "check"}]
    assert kinds.index("retry:empty_output") < kinds.index("verified")
    retry_event = next(e for e in result.trace if e.kind == "retry:empty_output")
    assert retry_event.metadata["continuation"] == "think_close"


async def test_verified_final_floor_retry_runs_with_committed_head():
    head = StreamScriptedBackend(["Intro. "])
    final = ScriptedFinalBackend(
        [("think", "deliberating..."), ("text", "The answer is 5.")]
    )
    audit = ScriptedFinalBackend([("text", "PASS")])
    conductor = Conductor(
        _verified_final_roles(prompt_suffix="<THINK>", reasoning_close_tag="</THINK>"),
        {"hw": head, "cw": final, "vw": audit},
        final_sampling_params=SamplingParams(max_tokens=512),
        public_output_floor=64,
    )

    result = await conductor.run("task")

    assert result.final_text == "Intro. The answer is 5."
    assert result.final_unit_ok
    assert final.requests_seen[1].prompt.endswith("deliberating...</THINK>\n\n")
    assert "The answer is 5." in audit.requests_seen[0].prompt
    retries = [e for e in result.trace if e.kind == "retry:empty_output"]
    assert [e.metadata.get("continuation") for e in retries] == ["think_close"]


@pytest.mark.parametrize("stream", [False, True])
async def test_verified_final_skips_verifier_for_multiple_choices(stream):
    final = MultiChoiceThinkBackend()
    audit = ScriptedFinalBackend([("text", "PASS")])
    conductor = Conductor(
        _verified_floor_roles(),
        {"cw": final, "vw": audit},
        final_sampling_params=SamplingParams(n=2, max_tokens=64),
        public_output_floor=16,
    )

    if stream:
        events = await _collect(conductor.stream("task"))
        result = events[-1].result
        assert result is not None
        assert final.generate_calls == 0
        assert final.stream_calls == 2
    else:
        result = await conductor.run("task")
        assert final.generate_calls == 2
        assert final.stream_calls == 0

    assert result.final_unit_ok
    assert [choice.text for choice in result.completions] == ["answer-0", "answer-1"]
    assert audit.requests_seen == []
    skipped = next(e for e in result.trace if e.kind == "skipped:intent")
    assert skipped.node == "check"
    assert skipped.metadata == {"reason": "intent", "n": 2}
