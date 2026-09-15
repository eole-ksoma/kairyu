import asyncio
import copy
import json
import threading
from dataclasses import replace

import httpx
import pytest

from kairyu.engine.backend import (
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    UpstreamClientError,
)
from kairyu.engine.mock import MockBackend
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.engine.prompt import TemplatedPrompt
from kairyu.orchestration.budget import Budget, BudgetState
from kairyu.orchestration.conductor import RoleSamplingOverrides, RoleSpec
from kairyu.orchestration.moa import MoAEvent, MoAResult
from kairyu.orchestration.orchestrator import (
    EngineDescriptor,
    Orchestrator,
    PreviewNotSupportedError,
    _bounded_profile_judge_view,
    _parse_profile_verdict,
)
from kairyu.orchestration.replica import ReplicaPool
from kairyu.orchestration.request import OrchestrationRequest
from kairyu.orchestration.router import RuleRouter
from kairyu.outputs import CompletionOutput
from kairyu.sampling_params import SamplingParams

SIMPLE = "What is 2?"
COMPLEX = (
    "First, research the options and summarize trade-offs. Then design a plan. "
    "After that, implement it. Finally, verify everything works end to end."
)


class _ShutdownBackend(MockBackend):
    def __init__(self) -> None:
        super().__init__()
        self.shutdown_count = 0

    async def shutdown(self) -> None:
        self.shutdown_count += 1


class _StartupBackend(MockBackend):
    def __init__(self) -> None:
        super().__init__()
        self.startup_count = 0

    async def startup(self) -> None:
        self.startup_count += 1


class _FailingBackend(MockBackend):
    def __init__(self, message: str = "backend failed") -> None:
        super().__init__()
        self.message = message
        self.generate_calls = 0
        self.stream_calls = 0

    async def generate(self, request: GenerationRequest):
        self.generate_calls += 1
        raise RuntimeError(self.message)

    async def stream(self, request: GenerationRequest):
        self.stream_calls += 1
        raise RuntimeError(self.message)
        yield  # pragma: no cover - makes this an async generator


class _PreparingBackend(MockBackend):
    def __init__(
        self,
        name: str,
        events: list[tuple[str, str]],
        *,
        prepare_failures: dict[str, BaseException] | None = None,
        validation_error: str | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.events = events
        self.prepare_failures = dict(prepare_failures or {})
        self.validation_error = validation_error
        self.validated: list[GenerationRequest] = []
        self.prepared: list[GenerationRequest] = []
        self.generated: list[GenerationRequest] = []
        self.streamed: list[GenerationRequest] = []

    def validate_request(self, request: GenerationRequest) -> None:
        self.validated.append(request)
        if self.validation_error is not None:
            raise ValueError(self.validation_error)
        super().validate_request(request)

    async def prepare_request(self, request: GenerationRequest) -> None:
        self.events.append((self.name, request.request_id))
        self.prepared.append(request)
        failure = self.prepare_failures.get(
            request.request_id,
            next(
                (
                    candidate
                    for pattern, candidate in self.prepare_failures.items()
                    if pattern != "*"
                    and pattern.endswith("*")
                    and request.request_id.startswith(pattern[:-1])
                ),
                self.prepare_failures.get("*"),
            ),
        )
        if failure is not None:
            raise failure

    async def generate(self, request: GenerationRequest):
        self.generated.append(request)
        return await super().generate(request)

    async def stream(self, request: GenerationRequest):
        self.streamed.append(request)
        async for result in super().stream(request):
            yield result


class _CountingRouter:
    def __init__(self) -> None:
        self._delegate = RuleRouter()
        self.preview_calls = 0
        self.route_calls = 0

    def preview(self, query: str, context: dict | None = None):
        self.preview_calls += 1
        return self._delegate.preview(query, context)

    def route(self, query: str, context: dict | None = None):
        self.route_calls += 1
        return self._delegate.route(query, context)


def _orchestrator(**kwargs) -> Orchestrator:
    engines = kwargs.pop(
        "engines",
        {"tier1": MockBackend(), "tier2": MockBackend()},
    )
    return Orchestrator(engines=engines, **kwargs)


def test_max_model_len_is_safe_for_every_orchestration_engine():
    small = MockBackend()
    small.max_model_len = 262_144
    large = MockBackend()
    large.max_model_len = 1_048_576

    orchestrator = _orchestrator(
        engines={"tier1": small, "tier2": large},
    )

    assert orchestrator.max_model_len == 262_144


def test_max_model_len_is_unknown_when_any_engine_has_no_metadata():
    known = MockBackend()
    known.max_model_len = 32_768

    orchestrator = _orchestrator(
        engines={"tier1": known, "tier2": MockBackend()},
    )

    assert orchestrator.max_model_len is None


def test_orchestrator_rejects_templated_shared_prefix_at_construction():
    with pytest.raises(ValueError, match="shared_prefix.*pre-rendered"):
        _orchestrator(shared_prefix=TemplatedPrompt("<BOS>already rendered"))


def test_orchestrator_rejects_public_output_floor_in_moa_mode():
    with pytest.raises(
        ValueError,
        match="public_output_floor cannot be combined with moa_samples > 0",
    ):
        _orchestrator(moa_samples=2, public_output_floor=64)


@pytest.mark.parametrize("value", [True, 0, -1, 131_073, 1.5, "64"])
def test_orchestrator_rejects_invalid_public_output_floor(value):
    with pytest.raises(ValueError, match="integer between 1 and 131072"):
        _orchestrator(public_output_floor=value)


def test_preview_route_is_non_dispatching_and_describes_effective_fallback():
    only = MockBackend()
    orchestrator = Orchestrator(
        engines={"worker": only},
        engine_descriptors={"worker": EngineDescriptor(backend_type="openai", model="small")},
    )

    decision = orchestrator.preview_route(SIMPLE)
    descriptor = orchestrator.describe_routing()

    assert decision.target == "tier1"
    assert only.prompts_seen == ()
    assert descriptor["configured_engines"] == {
        "worker": {"backend_type": "openai", "model": "small"}
    }
    assert descriptor["target_resolution"]["tier1"] == {
        "configured": False,
        "engine": "worker",
        "fallback": True,
    }
    assert descriptor["router"]["thresholds"]["multi_agent_min_chars"] == 2000
    assert all("prompt" not in role for role in descriptor["roles"])


def test_preview_route_rejects_router_without_preview():
    class RouteOnly:
        def route(self, query, context=None):
            return _orchestrator().preview_route(query)

    orchestrator = _orchestrator(router=RouteOnly())
    with pytest.raises(PreviewNotSupportedError, match="does not support preview"):
        orchestrator.preview_route(SIMPLE)


async def test_tier1_direct_failure_retries_tier2_once() -> None:
    tier1 = _FailingBackend()
    tier2 = MockBackend({SIMPLE: "tier2 answer"})
    orchestrator = _orchestrator(engines={"tier1": tier1, "tier2": tier2})

    result = await orchestrator.run(SIMPLE)

    assert result.text == "tier2 answer"
    assert tier1.generate_calls == 1
    assert len(tier2.prompts_seen) == 1
    assert any("retrying once on tier2" in note for note in result.trace)
    assert [event.node for event in result.structured_trace.events] == [
        "router",
        "tier1",
        "tier2",
    ]


async def test_tier1_stream_failure_before_output_retries_tier2_once() -> None:
    tier1 = _FailingBackend()
    tier2 = MockBackend({SIMPLE: "tier2 answer"})
    orchestrator = _orchestrator(engines={"tier1": tier1, "tier2": tier2})

    events = [event async for event in await orchestrator.run_chat(SIMPLE, stream=True)]

    assert tier1.stream_calls == 1
    assert len(tier2.prompts_seen) == 1
    assert events[-1].kind == "result"
    assert events[-1].result.text == "tier2 answer"
    assert any("retrying once on tier2" in note for note in events[-1].result.trace)


def test_auto_admission_bound_accounts_for_internal_fanout() -> None:
    orchestrator = _orchestrator(budget=Budget(max_steps=4))
    call = OrchestrationRequest(
        prompt=COMPLEX,
        sampling_params=SamplingParams(max_tokens=32),
    )

    bound = orchestrator.admission_upper_bound(call)

    assert bound.tokens > len(COMPLEX.encode()) + 32
    assert not bound.refundable_on_exact_usage

    direct_preview_call = OrchestrationRequest(
        prompt=SIMPLE,
        sampling_params=SamplingParams(max_tokens=32),
    )
    assert orchestrator.preview_route(SIMPLE).target == "tier1"
    assert (
        orchestrator.admission_upper_bound(direct_preview_call).tokens > len(SIMPLE.encode()) + 32
    )


def test_auto_admission_bound_accounts_for_final_tool_and_response_schemas() -> None:
    orchestrator = _orchestrator(budget=Budget(max_steps=4))

    def bound_for(schema_size: int) -> int:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "description": "r" * schema_size,
                },
            },
        }
        return orchestrator.admission_upper_bound(
            OrchestrationRequest(
                prompt=COMPLEX,
                sampling_params=SamplingParams(
                    n=3,
                    max_tokens=32,
                    extra_args={"response_format": response_format},
                ),
                tools=(
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "t" * schema_size,
                        },
                    },
                ),
                tool_choice="required",
                response_format=response_format,
            )
        ).tokens

    small = bound_for(1)
    large = bound_for(4_097)

    # Both final schemas are serialized into each of the three candidate
    # prefills, so their 8,192-byte growth must be charged three times.
    assert large - small >= 3 * 8_192


async def test_spec_default_reasoning_effort_backs_inherit_roles() -> None:
    backend = _PreparingBackend("tier1", [])
    roles = (
        RoleSpec(
            name="planner",
            worker="tier1",
            prompt="[planner] {query}",
            reasoning_effort="inherit",
        ),
        RoleSpec(
            name="worker",
            worker="tier1",
            prompt="[worker] {planner}",
            depends_on=("planner",),
        ),
    )
    orchestrator = _orchestrator(
        engines={"tier1": backend},
        roles=roles,
        default_reasoning_effort="high",
    )

    await orchestrator.run(COMPLEX)
    efforts = {
        request.request_id.rsplit("-", 2)[-2]: request.reasoning_effort
        for request in backend.generated
    }
    assert efforts == {"planner": "high", "worker": None}

    backend.generated.clear()
    await orchestrator.run(
        OrchestrationRequest(
            prompt=COMPLEX,
            sampling_params=SamplingParams(max_tokens=32),
            reasoning_effort="low",
        )
    )
    efforts = {
        request.request_id.rsplit("-", 2)[-2]: request.reasoning_effort
        for request in backend.generated
    }
    assert efforts == {"planner": "low", "worker": None}


async def test_async_prepare_binds_one_route_and_preserves_final_intent() -> None:
    events: list[tuple[str, str]] = []
    tier1 = _PreparingBackend("tier1", events)
    tier2 = _PreparingBackend("tier2", events)
    router = _CountingRouter()
    orchestrator = _orchestrator(
        engines={"tier1": tier1, "tier2": tier2},
        router=router,
        shared_prefix="[shared] ",
    )
    tool = {
        "type": "function",
        "function": {"name": "lookup", "parameters": {}},
    }
    call = OrchestrationRequest(
        prompt=SIMPLE,
        sampling_params=SamplingParams(n=2, max_tokens=7),
        tools=(tool,),
        tool_choice="required",
        tools_in_prompt=True,
        parallel_tool_calls=False,
        tool_call_protocol="qwen",
    )

    prepared_call = await orchestrator.prepare_request(call)

    assert (router.preview_calls, router.route_calls) == (0, 1)
    assert len(events) == 1
    assert events[0][0] == "tier1"
    assert events[0][1].startswith("direct-")
    assert tier2.prepared == []
    prepared = tier1.prepared[0]
    assert prepared.prompt == f"[shared] {SIMPLE}"
    assert prepared.sampling_params is call.sampling_params
    assert prepared.tools == (tool,)
    assert prepared.tool_choice == "required"
    assert prepared.tools_in_prompt is True
    assert prepared.parallel_tool_calls is False
    assert prepared.tool_call_protocol == "qwen"

    result = await orchestrator.run(call, prepared=prepared_call)

    assert result.completions
    assert tier1.generated == [prepared]
    assert (router.preview_calls, router.route_calls) == (0, 1)


async def test_async_prepare_builds_request_sized_plan_off_event_loop(
    monkeypatch,
) -> None:
    events: list[tuple[str, str]] = []
    orchestrator = _orchestrator(
        engines={
            "tier1": _PreparingBackend("tier1", events),
            "tier2": _PreparingBackend("tier2", events),
        },
    )
    event_loop_thread = threading.get_ident()
    plan_threads = []
    plan_started = threading.Event()
    release_plan = threading.Event()
    original = orchestrator._build_preparation_plan

    def gated_plan(*args):
        plan_threads.append(threading.get_ident())
        plan_started.set()
        assert release_plan.wait(timeout=5)
        return original(*args)

    monkeypatch.setattr(orchestrator, "_build_preparation_plan", gated_plan)
    preparation = asyncio.create_task(orchestrator.prepare_request(COMPLEX * 1000))
    try:
        assert await asyncio.wait_for(
            asyncio.to_thread(plan_started.wait, 1),
            timeout=2,
        )
        assert not preparation.done()
        assert plan_threads != [event_loop_thread]
    finally:
        release_plan.set()
    await preparation


@pytest.mark.parametrize("stream", [False, True])
async def test_prepared_handle_is_bound_to_its_owning_orchestrator(stream) -> None:
    events: list[tuple[str, str]] = []
    first_backend = _PreparingBackend("first", events)
    second_backend = _PreparingBackend("second", events)
    first = _orchestrator(
        engines={"tier1": first_backend, "tier2": MockBackend()},
    )
    second = _orchestrator(
        engines={"tier1": second_backend, "tier2": MockBackend()},
    )
    call = OrchestrationRequest(
        prompt=SIMPLE,
        sampling_params=SamplingParams(max_tokens=2),
    )

    prepared = await first.prepare_request(call)
    exact_request = first_backend.prepared[0]
    if stream:
        foreign_stream = await second.run_chat(
            call,
            stream=True,
            prepared=prepared,
        )
        assert [event async for event in foreign_stream][-1].kind == "result"
        assert second_backend.streamed[0] is not exact_request
    else:
        await second.run(call, prepared=prepared)
        assert second_backend.generated[0] is not exact_request
    assert prepared.consumed is False

    await first.run(call, prepared=prepared)
    assert first_backend.generated == [exact_request]


async def test_prepared_handle_copies_share_one_dispatch_capability() -> None:
    events: list[tuple[str, str]] = []
    backend = _PreparingBackend("tier1", events)
    orchestrator = _orchestrator(
        engines={"tier1": backend, "tier2": MockBackend()},
    )
    call = OrchestrationRequest(
        prompt=SIMPLE,
        sampling_params=SamplingParams(max_tokens=2),
    )
    prepared = await orchestrator.prepare_request(call)
    exact_request = backend.prepared[0]
    aliases = (
        prepared,
        copy.copy(prepared),
        copy.deepcopy(prepared),
        replace(prepared),
    )

    await asyncio.gather(*(orchestrator.run(call, prepared=alias) for alias in aliases))

    assert sum(request is exact_request for request in backend.generated) == 1
    assert len({request.request_id for request in backend.generated}) == 4
    assert prepared.consumed is True


async def test_async_prepare_direct_stream_reuses_exact_request_and_unique_ids() -> None:
    events: list[tuple[str, str]] = []
    tier1 = _PreparingBackend("tier1", events)
    orchestrator = _orchestrator(
        engines={"tier1": tier1, "tier2": MockBackend()},
    )
    first = OrchestrationRequest(
        prompt=SIMPLE,
        sampling_params=SamplingParams(max_tokens=2),
    )
    second = OrchestrationRequest(
        prompt=SIMPLE,
        sampling_params=SamplingParams(max_tokens=2),
    )

    first_prepared, second_prepared = await asyncio.gather(
        orchestrator.prepare_request(first),
        orchestrator.prepare_request(second),
    )
    first_request, second_request = tier1.prepared

    assert first_request.request_id.startswith("direct-")
    assert second_request.request_id.startswith("direct-")
    assert first_request.request_id != second_request.request_id

    stream = await orchestrator.run_chat(
        first,
        stream=True,
        prepared=first_prepared,
    )
    events_seen = [event async for event in stream]

    assert events_seen[-1].kind == "result"
    assert tier1.streamed == [first_request]
    assert second_prepared.call is second


@pytest.mark.parametrize("stream", [False, True])
async def test_string_prepare_handle_reuses_bound_route_and_exact_request(stream) -> None:
    events: list[tuple[str, str]] = []
    tier1 = _PreparingBackend("tier1", events)
    router = _CountingRouter()
    orchestrator = _orchestrator(
        engines={"tier1": tier1, "tier2": MockBackend()},
        router=router,
    )

    prepared = await orchestrator.prepare_request(SIMPLE)
    exact_request = tier1.prepared[0]
    if stream:
        result_stream = await orchestrator.run_chat(
            SIMPLE,
            stream=True,
            prepared=prepared,
        )
        result_events = [event async for event in result_stream]
        assert result_events[-1].kind == "result"
        assert tier1.streamed == [exact_request]
    else:
        result = await orchestrator.run(SIMPLE, prepared=prepared)
        assert result.completions
        assert tier1.generated == [exact_request]
    assert (router.preview_calls, router.route_calls) == (0, 1)


async def test_async_prepare_binds_actual_route_instead_of_nonbinding_preview() -> None:
    class PreviewTier1RouteTier2:
        def preview(self, query, context=None):
            return RuleRouter().preview(SIMPLE)

        def route(self, query, context=None):
            return RuleRouter().preview("x" * 600)

    events: list[tuple[str, str]] = []
    tier1 = _PreparingBackend("tier1", events)
    tier2 = _PreparingBackend("tier2", events)
    orchestrator = _orchestrator(
        engines={"tier1": tier1, "tier2": tier2},
        router=PreviewTier1RouteTier2(),
    )
    call = OrchestrationRequest(
        prompt=SIMPLE,
        sampling_params=SamplingParams(max_tokens=2),
    )

    prepared = await orchestrator.prepare_request(call)
    tier2_request = tier2.prepared[0]
    await orchestrator.run(call, prepared=prepared)

    assert tier1.prepared == []
    assert tier1.generated == []
    assert tier2.generated == [tier2_request]


async def test_async_prepare_covers_every_multi_agent_final_and_internal_intent() -> None:
    events: list[tuple[str, str]] = []
    tier1 = _PreparingBackend("tier1", events)
    tier2 = _PreparingBackend("tier2", events)
    orchestrator = _orchestrator(
        engines={"tier1": tier1, "tier2": tier2},
        sampling_params=SamplingParams(max_tokens=5),
    )
    call = OrchestrationRequest(
        prompt=COMPLEX,
        sampling_params=SamplingParams(n=2, max_tokens=20),
    )

    await orchestrator.prepare_request(call)

    assert [name for name, _request_id in events] == [
        "tier2",
        "tier1",
        "tier2",
        "tier2",
    ]
    assert events[0][1].startswith("preflight-tier2-")
    assert events[1][1].startswith("preflight-internal-tier1-")
    assert events[2][1].startswith("preflight-internal-tier2-")
    assert events[3][1].startswith("preflight-role-planner-")
    assert len({request_id.rsplit("-", 1)[-1] for _, request_id in events}) == 1
    final = tier2.prepared[0]
    internal_tier1 = tier1.prepared[0]
    internal_tier2 = tier2.prepared[1]
    initial_planner = tier2.prepared[2]
    assert final.sampling_params is call.sampling_params
    assert final.sampling_params.n == 2
    for internal in (internal_tier1, internal_tier2):
        assert internal.prompt == COMPLEX
        assert internal.sampling_params.n == 1
        assert internal.sampling_params.max_tokens == 5
        assert internal.tools == ()
        assert internal.tool_choice is None
    assert initial_planner.prompt == (
        f"[planner] Break the task into a short actionable plan.\nTask: {COMPLEX}"
    )
    assert initial_planner.sampling_params.n == 1
    assert initial_planner.sampling_params.max_tokens == 5


async def test_async_prepare_rejects_noninitial_final_sampling_policy() -> None:
    events: list[tuple[str, str]] = []

    class RejectTopKBackend(_PreparingBackend):
        def validate_request(self, request: GenerationRequest) -> None:
            super().validate_request(request)
            if request.sampling_params.top_k == 20:
                raise ValueError("top_k policy rejected")

    tier1 = _PreparingBackend("tier1", events)
    tier2 = RejectTopKBackend("tier2", events)
    roles = (
        RoleSpec(
            name="draft",
            worker="tier1",
            role_type="proposer",
            prompt="[draft] {query}",
        ),
        RoleSpec(
            name="final",
            worker="tier2",
            role_type="publisher",
            depends_on=("draft",),
            prompt="[final] {query} {draft}",
            sampling=RoleSamplingOverrides(top_k=20),
        ),
    )
    orchestrator = _orchestrator(
        engines={"tier1": tier1, "tier2": tier2},
        roles=roles,
    )
    call = OrchestrationRequest(
        prompt=COMPLEX,
        sampling_params=SamplingParams(max_tokens=64),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"^final orchestration intent is unsupported: "
            r"tier2: top_k policy rejected$"
        ),
    ):
        await orchestrator.prepare_request(call)

    assert tier2.validated
    assert tier2.validated[0].sampling_params.top_k == 20
    assert tier1.generated == []
    assert tier2.generated == []


def test_validate_request_rejects_unsupported_chat_continuation_retry() -> None:
    class RejectAssistantPrefillBackend(_PreparingBackend):
        def validate_request(self, request: GenerationRequest) -> None:
            super().validate_request(request)
            if request.assistant_prefill is not None:
                raise ValueError("assistant prefill unsupported")

    backend = RejectAssistantPrefillBackend("tier2", [])
    orchestrator = _orchestrator(
        engines={"tier2": backend},
        roles=(
            RoleSpec(
                name="final",
                worker="tier2",
                role_type="publisher",
                prompt="[final] {query}",
                reasoning_close_tag="</THINK>",
                reasoning_continuation="chat",
                reasoning_open_tag="<THINK>",
            ),
        ),
        public_output_floor=64,
    )
    call = OrchestrationRequest(
        prompt=COMPLEX,
        sampling_params=SamplingParams(max_tokens=512),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"^final orchestration intent is unsupported: "
            r"tier2: assistant prefill unsupported$"
        ),
    ):
        orchestrator.validate_request(call)

    assert [request.assistant_prefill for request in backend.validated] == [
        None,
        "<THINK>\n\n</THINK>\n\n",
    ]
    assert backend.generated == []


async def test_async_prepare_keeps_chat_retry_intent_validation_only() -> None:
    events: list[tuple[str, str]] = []
    backend = _PreparingBackend("tier2", events)
    orchestrator = _orchestrator(
        engines={"tier2": backend},
        roles=(
            RoleSpec(
                name="final",
                worker="tier2",
                role_type="publisher",
                prompt="[final] {query}",
                reasoning_close_tag="</THINK>",
                reasoning_continuation="chat",
                reasoning_open_tag="<THINK>",
            ),
        ),
        public_output_floor=64,
    )
    call = OrchestrationRequest(
        prompt=COMPLEX,
        sampling_params=SamplingParams(max_tokens=512),
    )

    prepared = await orchestrator.prepare_request(call)

    assert [request.assistant_prefill for request in backend.validated] == [
        None,
        "<THINK>\n\n</THINK>\n\n",
        None,
    ]
    assert [request.assistant_prefill for request in backend.prepared] == [
        None,
        None,
    ]
    assert backend.prepared[0].request_id.startswith("preflight-tier2-")
    assert backend.prepared[1].request_id.startswith("preflight-role-final-")

    await orchestrator.run(call, prepared=prepared)
    assert backend.generated == [backend.prepared[1]]


async def test_async_prepare_rejects_exact_initial_role_prompt_before_dispatch() -> None:
    events: list[tuple[str, str]] = []

    class PromptLimitBackend(_PreparingBackend):
        async def prepare_request(self, request: GenerationRequest) -> None:
            await super().prepare_request(request)
            if len(request.prompt) > 32:
                raise ValueError("rendered prompt too long")

    class MultiAgentRoute:
        def route(self, query, context=None):
            return RuleRouter().route(COMPLEX)

    tier1 = _PreparingBackend("tier1", events)
    tier2 = PromptLimitBackend("tier2", events)
    orchestrator = _orchestrator(
        engines={"tier1": tier1, "tier2": tier2},
        router=MultiAgentRoute(),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"^initial orchestration intent is unsupported: "
            r"tier2: rendered prompt too long$"
        ),
    ):
        await orchestrator.prepare_request("x")

    assert [name for name, _request_id in events] == [
        "tier2",
        "tier1",
        "tier2",
        "tier2",
    ]
    assert tier2.prepared[-1].prompt == (
        "[planner] Break the task into a short actionable plan.\nTask: x"
    )
    assert tier1.generated == []
    assert tier2.generated == []


@pytest.mark.parametrize("stream", [False, True])
async def test_async_prepare_reuses_exact_initial_role_request(stream) -> None:
    events: list[tuple[str, str]] = []
    tier1 = _PreparingBackend("tier1", events)
    tier2 = _PreparingBackend("tier2", events)
    orchestrator = _orchestrator(
        engines={"tier1": tier1, "tier2": tier2},
    )

    prepared = await orchestrator.prepare_request(COMPLEX)
    initial_planner = next(
        request
        for request in tier2.prepared
        if request.request_id.startswith("preflight-role-planner-")
    )

    if stream:
        result_stream = await orchestrator.run_chat(
            COMPLEX,
            stream=True,
            prepared=prepared,
        )
        result_events = [event async for event in result_stream]
        assert result_events[-1].kind == "result"
    else:
        result = await orchestrator.run(COMPLEX, prepared=prepared)
        assert result.completions

    assert tier2.generated[0] is initial_planner
    assert initial_planner.cache_hint is not None
    assert all(
        request.cache_hint is None
        or request.cache_hint.session_id == initial_planner.cache_hint.session_id
        for request in tier1.generated + tier1.streamed + tier2.generated + tier2.streamed
    )


async def test_conversation_affinity_key_keeps_turns_on_one_session() -> None:
    # Issue #495: consecutive agent turns resend the same conversation with
    # new turns appended; their KV-affinity session must stay stable so the
    # replica holding the warm prefix keeps the traffic, while a different
    # conversation maps to a different session.
    events: list[tuple[str, str]] = []
    tier1 = _PreparingBackend("tier1", events)
    tier2 = _PreparingBackend("tier2", events)
    orchestrator = _orchestrator(engines={"tier1": tier1, "tier2": tier2})

    def observed_sessions() -> set[str]:
        sessions = {
            request.cache_hint.session_id
            for backend in (tier1, tier2)
            for request in backend.generated + backend.streamed
            if request.cache_hint is not None
        }
        for backend in (tier1, tier2):
            backend.generated.clear()
            backend.streamed.clear()
        return sessions

    conversation = COMPLEX * 20

    def call(prompt: str, affinity_key: str) -> OrchestrationRequest:
        sampling = SamplingParams(max_tokens=1024)
        return OrchestrationRequest(
            prompt=prompt,
            sampling_params=sampling,
            conversation_affinity_key=affinity_key,
        )

    await orchestrator.run(call(conversation, "conv-task-a"))
    first_turn = observed_sessions()
    await orchestrator.run(call(conversation + " appended tool-result turn", "conv-task-a"))
    second_turn = observed_sessions()
    await orchestrator.run(call(conversation, "conv-task-b"))
    other_conversation = observed_sessions()

    assert len(first_turn) == 1
    assert first_turn == second_turn
    assert other_conversation != first_turn


@pytest.mark.parametrize("stream", [False, True])
async def test_async_prepare_reuses_exact_moa_proposal_requests(stream) -> None:
    events: list[tuple[str, str]] = []
    tier1 = _PreparingBackend("tier1", events)
    tier2 = _PreparingBackend("tier2", events)
    orchestrator = _orchestrator(
        engines={"tier1": tier1, "tier2": tier2},
        moa_samples=2,
        default_reasoning_effort="high",
    )

    prepared = await orchestrator.prepare_request(COMPLEX)
    proposals = tuple(tier1.prepared)
    assert len(proposals) == 2
    assert [request.sampling_params.seed for request in proposals] == [0, 1]
    assert [request.reasoning_effort for request in proposals] == ["high", "high"]

    if stream:
        result_stream = await orchestrator.run_chat(
            COMPLEX,
            stream=True,
            prepared=prepared,
        )
        assert [event async for event in result_stream][-1].kind == "result"
    else:
        result = await orchestrator.run(COMPLEX, prepared=prepared)
        assert result.completions

    assert tuple(tier1.generated) == proposals
    synthesis_requests = tier2.streamed if stream else tier2.generated
    assert [request.reasoning_effort for request in synthesis_requests] == ["high"]


async def test_async_prepare_rejects_exact_moa_seed_before_dispatch() -> None:
    events: list[tuple[str, str]] = []

    class RejectSeedBackend(_PreparingBackend):
        async def prepare_request(self, request: GenerationRequest) -> None:
            await super().prepare_request(request)
            if request.sampling_params.seed is not None:
                raise ValueError("seed unsupported")

    tier1 = RejectSeedBackend("tier1", events)
    tier2 = _PreparingBackend("tier2", events)
    orchestrator = _orchestrator(
        engines={"tier1": tier1, "tier2": tier2},
        moa_samples=2,
    )

    with pytest.raises(ValueError, match="seed unsupported"):
        await orchestrator.prepare_request(COMPLEX)

    assert [request.sampling_params.seed for request in tier1.prepared] == [0, 1]
    assert tier1.generated == []
    assert tier2.generated == []


async def test_async_prepare_aggregates_value_errors_in_stable_phase_key_order() -> None:
    events: list[tuple[str, str]] = []
    tier1 = _PreparingBackend(
        "tier1",
        events,
        prepare_failures={
            "preflight-tier1-*": ValueError("final one"),
            "preflight-internal-tier1-*": ValueError("internal one"),
        },
    )
    tier2 = _PreparingBackend(
        "tier2",
        events,
        prepare_failures={
            "preflight-tier2-*": ValueError("final two"),
            "preflight-internal-tier2-*": ValueError("internal two"),
        },
    )

    class RouteOnly:
        def route(self, query, context=None):
            return RuleRouter().route(COMPLEX)

    orchestrator = _orchestrator(
        engines={"tier2": tier2, "tier1": tier1},
        router=RouteOnly(),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"^final orchestration intent is unsupported: "
            r"tier2: final two; "
            r"internal orchestration intent is unsupported: "
            r"tier1: internal one; tier2: internal two$"
        ),
    ):
        await orchestrator.prepare_request(SIMPLE)

    assert [name for name, _request_id in events] == [
        "tier2",
        "tier1",
        "tier2",
        "tier2",
    ]
    assert events[0][1].startswith("preflight-tier2-")
    assert events[1][1].startswith("preflight-internal-tier1-")
    assert events[2][1].startswith("preflight-internal-tier2-")
    assert events[3][1].startswith("preflight-role-planner-")


async def test_concurrent_multi_agent_preflight_ids_are_unique() -> None:
    class RejectDuplicateIds(_PreparingBackend):
        def __init__(self, name, events):
            super().__init__(name, events)
            self.seen_ids: set[str] = set()

        async def prepare_request(self, request: GenerationRequest) -> None:
            if request.request_id in self.seen_ids:
                raise RuntimeError(f"duplicate request ID: {request.request_id}")
            self.seen_ids.add(request.request_id)
            await super().prepare_request(request)

    events: list[tuple[str, str]] = []
    tier1 = RejectDuplicateIds("tier1", events)
    tier2 = RejectDuplicateIds("tier2", events)
    orchestrator = _orchestrator(
        engines={"tier1": tier1, "tier2": tier2},
    )

    await asyncio.gather(
        orchestrator.prepare_request(COMPLEX),
        orchestrator.prepare_request(COMPLEX),
    )

    request_ids = [request_id for _name, request_id in events]
    assert len(request_ids) == 8
    assert len(request_ids) == len(set(request_ids))


async def test_async_prepare_keeps_sync_validation_error_and_runs_no_prepare() -> None:
    events: list[tuple[str, str]] = []
    tier1 = _PreparingBackend(
        "tier1",
        events,
        validation_error="semantic rejection",
    )
    orchestrator = _orchestrator(
        engines={"tier1": tier1, "tier2": MockBackend()},
    )
    expected = "final orchestration intent is unsupported: tier1: semantic rejection"

    with pytest.raises(ValueError, match=f"^{expected}$"):
        orchestrator.validate_request(SIMPLE)
    with pytest.raises(ValueError, match=f"^{expected}$"):
        await orchestrator.prepare_request(SIMPLE)

    assert events == []


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("backend unavailable"),
        UpstreamClientError("upstream rejected", status_code=409),
    ],
    ids=("runtime", "upstream-client"),
)
async def test_async_prepare_does_not_relabel_non_validation_failures(failure) -> None:
    events: list[tuple[str, str]] = []
    tier1 = _PreparingBackend(
        "tier1",
        events,
        prepare_failures={"*": failure},
    )
    orchestrator = _orchestrator(
        engines={"tier1": tier1, "tier2": MockBackend()},
    )

    with pytest.raises(type(failure)) as raised:
        await orchestrator.prepare_request(SIMPLE)

    assert raised.value is failure


async def test_async_prepare_reaches_every_distinct_replica_contract() -> None:
    events: list[tuple[str, str]] = []
    first = _PreparingBackend("replica-a", events)
    second = _PreparingBackend("replica-b", events)
    pool = ReplicaPool({"a": first, "b": second})
    orchestrator = _orchestrator(
        engines={"tier1": pool, "tier2": MockBackend()},
    )

    await orchestrator.prepare_request(SIMPLE)

    assert [name for name, _request_id in events] == [
        "replica-a",
        "replica-b",
    ]
    assert all(request_id.startswith("direct-") for _name, request_id in events)
    assert first.prepared[0] is second.prepared[0]


def _track_budget_releases(monkeypatch):
    releases = []
    original_release = BudgetState.release

    def tracked_release(self, steps=1, *, unknown_cost=False):
        released = original_release(
            self,
            steps=steps,
            unknown_cost=unknown_cost,
        )
        releases.append((self, steps, unknown_cost, released))
        return released

    monkeypatch.setattr(BudgetState, "release", tracked_release)
    return releases


async def test_simple_query_goes_to_tier1_engine():
    tier1 = MockBackend(responses={SIMPLE: "two"})
    orchestrator = _orchestrator(engines={"tier1": tier1, "tier2": MockBackend()})
    result = await orchestrator.run(SIMPLE)
    assert result.route.target == "tier1"
    assert result.text == "two"
    assert len(tier1.prompts_seen) == 1
    assert result.structured_trace is not None
    payload = result.structured_trace.as_dict()
    assert payload["trace_version"] == "2.0"
    assert [event["kind"] for event in payload["events"]] == [
        "routing",
        "generation",
    ]
    assert payload["events"][0]["detail"]["target"] == "tier1"
    assert payload["events"][1]["engine"] == "tier1"
    assert payload["events"][1]["status"] == "success"
    assert SIMPLE not in str(payload)


async def test_complex_query_uses_default_conductor_dag():
    tier1 = MockBackend()
    tier2 = MockBackend(responses={"[verifier]": "PASS"})
    orchestrator = _orchestrator(engines={"tier1": tier1, "tier2": tier2})
    result = await orchestrator.run(COMPLEX)
    assert result.route.target == "multi_agent"
    assert result.text
    assert len(tier1.prompts_seen) + len(tier2.prompts_seen) >= 3  # planner/worker/verifier/synth
    assert result.structured_trace is not None
    payload = result.structured_trace.as_dict()
    events = payload["events"]
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert events[0]["kind"] == "routing"
    role_events = [event for event in events if event["role"] != "router"]
    assert {event["node"] for event in role_events} >= {
        "planner",
        "worker",
        "verifier",
        "synthesizer",
    }
    assert all(event["timing"]["completed_at"] for event in role_events)
    assert COMPLEX not in str(payload)


async def test_moa_tier_charges_cost_model_and_reports_budget(tmp_path):
    # M3: the deep MoA tier must invoke the cost model and surface a budget
    # overrun in the trace, instead of being invisible to max_cost_usd.
    from kairyu.orchestration.conductor import chars_cost_model

    orchestrator = _orchestrator(
        moa_samples=2,
        cost_model=chars_cost_model(usd_per_1k_chars=1000.0),  # huge -> exceeds
        budget=Budget(max_cost_usd=0.001),
    )
    result = await orchestrator.run(COMPLEX)
    assert any("moa:" in line and "cost=" in line for line in result.trace)
    assert any("budget exceeded" in line for line in result.trace)


@pytest.mark.parametrize(
    "budget",
    (
        Budget(max_steps=2),
        Budget(max_steps=3, max_cost_usd=0.0),
    ),
    ids=("insufficient-steps", "cost-slot-unavailable"),
)
async def test_moa_budget_refusal_skips_without_dispatch(monkeypatch, budget):
    import kairyu.orchestration.moa as moa_module

    async def unexpected_run_moa(*args, **kwargs):
        pytest.fail("run_moa must not be called after budget refusal")

    monkeypatch.setattr(moa_module, "run_moa", unexpected_run_moa)
    orchestrator = _orchestrator(moa_samples=2, budget=budget)

    result = await orchestrator.run(COMPLEX)

    assert result.text == ""
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.trace[-1] == "moa: skipped:budget"


async def test_moa_success_reconciles_full_reservation_once(monkeypatch):
    import kairyu.orchestration.moa as moa_module

    events = []
    reconciled_states = []
    original_try_reserve = BudgetState.try_reserve
    original_reconcile_success = BudgetState.reconcile_success

    def tracked_try_reserve(self, steps=1, *, unknown_cost=False):
        reserved = original_try_reserve(
            self,
            steps=steps,
            unknown_cost=unknown_cost,
        )
        events.append(("reserve", steps, unknown_cost, reserved))
        return reserved

    def tracked_reconcile_success(
        self,
        steps=1,
        cost=0.0,
        *,
        unknown_cost=False,
    ):
        events.append(("reconcile", steps, unknown_cost, self))
        reconciled = original_reconcile_success(
            self,
            steps=steps,
            cost=cost,
            unknown_cost=unknown_cost,
        )
        reconciled_states.append(reconciled)
        return reconciled

    async def fake_run_moa(*args, **kwargs):
        events.append(("dispatch",))
        return MoAResult(
            final_text="synthesized",
            proposals=("proposal one", "proposal two"),
            usage=(7, 3),
            cached_tokens=5,
        )

    monkeypatch.setattr(BudgetState, "try_reserve", tracked_try_reserve)
    monkeypatch.setattr(BudgetState, "reconcile_success", tracked_reconcile_success)
    monkeypatch.setattr(moa_module, "run_moa", fake_run_moa)
    orchestrator = _orchestrator(
        moa_samples=2,
        budget=Budget(max_steps=4, max_cost_usd=1.0),
        cost_model=lambda request, result: 0.25,
    )

    result = await orchestrator.run(COMPLEX)

    assert [event[0] for event in events] == ["reserve", "dispatch", "reconcile"]
    assert events[0][1:3] == (3, True)
    reserved = events[0][3]
    assert reserved is not None
    assert reserved.steps_reserved == 3
    assert reserved.unknown_cost_reserved is True
    assert events[2][1:3] == (3, True)
    assert reconciled_states == [
        BudgetState(
            budget=Budget(max_steps=4, max_cost_usd=1.0),
            steps_used=3,
            cost_used=0.25,
        )
    ]
    assert result.text == "synthesized"
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 3
    assert result.cached_tokens == 5
    assert "moa: 2 proposals synthesized (cost=0.2500)" in result.trace
    assert "moa: budget exceeded" not in result.trace
    assert result.structured_trace is not None
    moa_event = result.structured_trace.as_dict()["events"][-1]
    assert moa_event["engine"] == "tier1,tier2"
    assert moa_event["detail"]["proposal_engine"] == "tier1"
    assert moa_event["detail"]["synthesizer_engine"] == "tier2"


async def test_moa_trace_records_resolved_fallback_engine_and_model(monkeypatch):
    import kairyu.orchestration.moa as moa_module

    shared = MockBackend()

    async def fake_run_moa(proposal_engine, *args, **kwargs):
        assert proposal_engine is shared
        assert kwargs["synthesizer"] is shared
        return MoAResult(
            final_text="shared result",
            proposals=("one", "two"),
            usage=(5, 2),
        )

    monkeypatch.setattr(moa_module, "run_moa", fake_run_moa)
    orchestrator = _orchestrator(
        engines={"shared": shared},
        engine_descriptors={
            "shared": EngineDescriptor(
                backend_type="mock",
                model="shared-model",
            )
        },
        moa_samples=2,
    )

    result = await orchestrator.run(COMPLEX)

    assert result.structured_trace is not None
    moa_event = result.structured_trace.as_dict()["events"][-1]
    assert moa_event["engine"] == "shared"
    assert moa_event["model"] == "shared-model"
    detail = moa_event["detail"]
    assert (
        detail["proposal_engine"],
        detail["proposal_model"],
        detail["synthesizer_engine"],
        detail["synthesizer_model"],
    ) == ("shared", "shared-model", "shared", "shared-model")


async def test_moa_failure_releases_full_reservation_and_reraises(monkeypatch):
    import kairyu.orchestration.moa as moa_module

    async def failing_run_moa(*args, **kwargs):
        raise RuntimeError("moa failed")

    releases = _track_budget_releases(monkeypatch)
    monkeypatch.setattr(moa_module, "run_moa", failing_run_moa)
    orchestrator = _orchestrator(
        moa_samples=2,
        budget=Budget(max_steps=3, max_cost_usd=1.0),
    )

    with pytest.raises(RuntimeError, match="moa failed"):
        await orchestrator.run(COMPLEX)

    assert len(releases) == 1
    reserved, steps, unknown_cost, released = releases[0]
    assert (steps, unknown_cost) == (3, True)
    assert (reserved.steps_reserved, reserved.unknown_cost_reserved) == (3, True)
    assert (released.steps_reserved, released.unknown_cost_reserved) == (0, False)


async def test_moa_cost_model_failure_releases_full_reservation_and_reraises(
    monkeypatch,
):
    import kairyu.orchestration.moa as moa_module

    async def fake_run_moa(*args, **kwargs):
        return MoAResult(
            final_text="synthesized",
            proposals=("proposal one", "proposal two"),
        )

    def failing_cost_model(request, result):
        raise ValueError("cost unavailable")

    releases = _track_budget_releases(monkeypatch)
    monkeypatch.setattr(moa_module, "run_moa", fake_run_moa)
    orchestrator = _orchestrator(
        moa_samples=2,
        budget=Budget(max_steps=3, max_cost_usd=1.0),
        cost_model=failing_cost_model,
    )

    with pytest.raises(ValueError, match="cost unavailable"):
        await orchestrator.run(COMPLEX)

    assert len(releases) == 1
    reserved, steps, unknown_cost, released = releases[0]
    assert (steps, unknown_cost) == (3, True)
    assert (reserved.steps_reserved, reserved.unknown_cost_reserved) == (3, True)
    assert (released.steps_reserved, released.unknown_cost_reserved) == (0, False)


async def test_moa_cancellation_releases_full_reservation_and_propagates(monkeypatch):
    import kairyu.orchestration.moa as moa_module

    started = asyncio.Event()
    blocked = asyncio.Event()

    async def blocked_run_moa(*args, **kwargs):
        started.set()
        await blocked.wait()
        raise AssertionError("unreachable")

    releases = _track_budget_releases(monkeypatch)
    monkeypatch.setattr(moa_module, "run_moa", blocked_run_moa)
    orchestrator = _orchestrator(
        moa_samples=2,
        budget=Budget(max_steps=3, max_cost_usd=1.0),
    )

    task = asyncio.create_task(orchestrator.run(COMPLEX))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(releases) == 1
    reserved, steps, unknown_cost, released = releases[0]
    assert (steps, unknown_cost) == (3, True)
    assert (reserved.steps_reserved, reserved.unknown_cost_reserved) == (3, True)
    assert (released.steps_reserved, released.unknown_cost_reserved) == (0, False)


async def test_multistage_stream_emits_periodic_keepalives(monkeypatch):
    # M8: a long multi-stage run must emit PERIODIC status keep-alives, not one
    # status then silence (which a proxy idle timeout would sever).
    import kairyu.orchestration.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "_KEEPALIVE_INTERVAL_S", 0.01)
    # backends with latency so the multi-agent run outlasts several keepalives
    slow = {"tier1": MockBackend(latency_s=0.03), "tier2": MockBackend(latency_s=0.03)}
    orchestrator = _orchestrator(engines=slow)
    events = [event async for event in await orchestrator.run_chat(COMPLEX, stream=True)]
    kinds = [e.kind for e in events]
    assert kinds.count("status") >= 2  # routing + at least one "working" keepalive
    assert kinds[-1] == "result"
    assert any(e.kind == "delta" for e in events)


async def test_direct_stream_timestamp_work_is_constant_and_skips_prefix_scan(
    monkeypatch,
) -> None:
    import kairyu.orchestration.orchestrator as orchestrator_module

    class NoPrefixScanText(str):
        def startswith(self, *args, **kwargs):
            raise AssertionError("AUTO streaming must trust the cumulative backend contract")

    class CumulativeBackend(MockBackend):
        def __init__(self, partials: int) -> None:
            super().__init__()
            self._partials = partials

        async def stream(self, request):
            for index in range(1, self._partials + 1):
                yield GenerationResult(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    completions=(
                        CompletionOutput(
                            index=0,
                            text=NoPrefixScanText("x" * index),
                            token_ids=tuple(range(index)),
                            finish_reason="length" if index == self._partials else None,
                        ),
                    ),
                    finished=index == self._partials,
                    usage=(
                        GenerationUsage(prompt_tokens=1, completion_tokens=index)
                        if index == self._partials
                        else None
                    ),
                )

    async def timestamp_calls(partials: int) -> int:
        calls: list[int] = []

        def counted_now() -> str:
            calls.append(len(calls))
            return "2026-08-07T00:00:00+00:00"

        monkeypatch.setattr(orchestrator_module, "utc_now_iso", counted_now)
        orchestrator = _orchestrator(
            engines={"tier1": CumulativeBackend(partials), "tier2": MockBackend()}
        )
        stream = await orchestrator.run_chat(SIMPLE, stream=True)
        events = [event async for event in stream]
        assert "".join(event.text for event in events if event.kind == "delta") == "x" * partials
        return len(calls)

    assert await timestamp_calls(1) == await timestamp_calls(8)


async def test_conductor_verifier_failure_precedes_streamed_final_boundary():
    backend = MockBackend(
        responses={
            "[verifier]": "FAIL: revise the draft",
            "[synthesizer]": "final answer streamed in several chunks",
        }
    )
    orchestrator = _orchestrator(
        engines={"tier1": backend, "tier2": backend},
        budget=Budget(max_refine_depth=0),
    )

    events = [event async for event in await orchestrator.run_chat(COMPLEX, stream=True)]

    result = events[-1].result
    assert result is not None
    assert "".join(event.text for event in events if event.kind == "delta") == result.text
    assert result.structured_trace is not None
    trace = result.structured_trace.as_dict()["events"]
    verifier = next(event for event in trace if event["role"] == "verifier")
    final = next(event for event in trace if event["role"] == "synthesizer")
    assert verifier["detail"]["pass"] is False
    assert final["detail"]["streamed"] is True


async def test_moa_final_synthesis_streams_and_reports_trace():
    backend = MockBackend(
        responses={"Synthesize": "MoA final answer streamed across several chunks"}
    )
    orchestrator = _orchestrator(
        engines={"tier1": backend, "tier2": backend},
        moa_samples=2,
    )

    events = [event async for event in await orchestrator.run_chat(COMPLEX, stream=True)]

    deltas = [event.text for event in events if event.kind == "delta"]
    result = events[-1].result
    assert len(deltas) > 1
    assert result is not None
    assert "".join(deltas) == result.text
    assert result.structured_trace is not None
    moa = result.structured_trace.as_dict()["events"][-1]
    assert moa["role"] == "moa"
    assert moa["detail"]["streamed"] is True


@pytest.mark.parametrize("stream", [False, True])
async def test_multistage_response_withholds_synthesizer_private_reasoning(stream):
    class ReasoningBackend(MockBackend):
        @staticmethod
        def _with_reasoning(result):
            return replace(
                result,
                completions=tuple(
                    replace(
                        completion,
                        reasoning_content="private chain",
                        reasoning_delta="private chain",
                        reasoning_offset=0,
                    )
                    for completion in result.completions
                ),
            )

        async def generate(self, request):
            return self._with_reasoning(await super().generate(request))

        async def stream(self, request):
            async for result in super().stream(request):
                yield self._with_reasoning(result)

    proposal = MockBackend()
    synthesis = ReasoningBackend(responses={"Synthesize": "public final answer"})
    orchestrator = _orchestrator(
        engines={"tier1": proposal, "tier2": synthesis},
        moa_samples=2,
    )

    if stream:
        events = [event async for event in await orchestrator.run_chat(COMPLEX, stream=True)]
        public_completions = [
            completion
            for event in events
            for completion in (
                event.completions
                if event.kind == "delta"
                else (event.result.completions if event.result is not None else ())
            )
        ]
        result = events[-1].result
        assert result is not None
    else:
        result = await orchestrator.run(COMPLEX)
        public_completions = list(result.completions)

    assert result.text == "public final answer"
    assert public_completions
    assert all(completion.reasoning_content is None for completion in public_completions)
    assert all(completion.reasoning_delta is None for completion in public_completions)


async def test_moa_stream_cancellation_closes_stream_and_releases_reservation(
    monkeypatch,
):
    import kairyu.orchestration.moa as moa_module

    closed = asyncio.Event()

    async def blocked_stream_moa(*args, **kwargs):
        try:
            yield MoAEvent(kind="delta", text="first")
            await asyncio.Event().wait()
        finally:
            closed.set()

    releases = _track_budget_releases(monkeypatch)
    monkeypatch.setattr(moa_module, "stream_moa", blocked_stream_moa)
    orchestrator = _orchestrator(
        moa_samples=2,
        budget=Budget(max_steps=3, max_cost_usd=1.0),
    )
    stream = await orchestrator.run_chat(COMPLEX, stream=True)

    assert (await anext(stream)).kind == "status"
    assert (await anext(stream)).text == "first"
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert closed.is_set()
    assert len(releases) == 1
    reserved, steps, unknown_cost, released = releases[0]
    assert (steps, unknown_cost) == (3, True)
    assert (reserved.steps_reserved, reserved.unknown_cost_reserved) == (3, True)
    assert (released.steps_reserved, released.unknown_cost_reserved) == (0, False)


async def test_missing_tier_falls_back_with_trace_note():
    only_engine = MockBackend()
    orchestrator = _orchestrator(engines={"tier1": only_engine})
    result = await orchestrator.run(
        "Prove the theorem and explain your reasoning step by step, derive it."
    )
    assert result.route.target == "tier2"
    assert result.text
    assert any("fallback" in note for note in result.trace)


async def test_shutdown_closes_each_owned_engine_once():
    shared = _ShutdownBackend()
    orchestrator = Orchestrator(engines={"tier1": shared, "tier2": shared})
    await orchestrator.shutdown()
    assert shared.shutdown_count == 1


async def test_startup_eagerly_starts_each_owned_engine_once():
    shared = _StartupBackend()
    orchestrator = Orchestrator(engines={"tier1": shared, "tier2": shared})

    await orchestrator.startup()

    assert shared.startup_count == 1


def test_run_sync_wrapper():
    orchestrator = _orchestrator()
    result = orchestrator.run_sync(SIMPLE)
    assert result.text


def test_requires_at_least_one_engine():
    with pytest.raises(ValueError, match="engine"):
        Orchestrator(engines={})


def test_direct_profiles_reject_moa_mode():
    from kairyu.orchestration.conductor import RoleSpec

    with pytest.raises(
        ValueError,
        match="profiles cannot be combined with moa_samples > 0",
    ):
        _orchestrator(
            profiles={"fast": (RoleSpec(name="fast", worker="tier1", prompt="{query}"),)},
            moa_samples=2,
        )


class _VisionMockBackend(MockBackend):
    def supports_prompt_kind(self, kind: str) -> bool:
        return kind in {"text", "multimodal"}


def _profiled_orchestrator(
    tier1,
    tier2,
    judge=None,
    *,
    judge_prompt_prefix="",
    judge_prompt_suffix="",
    fallback="primary",
):
    # Issue #509 / DTO-D13: one served model, several role profiles selected
    # per request by an optional LLM judge. The primary profile is a single
    # synthesizer; `ensemble` fans out before its final; `fast` is a direct
    # single-role route on tier1.
    from kairyu.orchestration.conductor import RoleSpec
    from kairyu.orchestration.orchestrator import ProfileChoice, ProfileJudge

    class MultiAgentRouter:
        def preview(self, query, context=None):
            return RuleRouter().route(COMPLEX)

        def route(self, query, context=None):
            return RuleRouter().route(COMPLEX)

    primary_roles = (
        RoleSpec(
            name="c_final",
            worker="tier1",
            role_type="synthesizer",
            prompt="[c_final] {query}",
            prompt_headless="[c_final_headless] {query}",
        ),
    )
    ensemble_roles = (
        RoleSpec(name="g_prop", worker="tier1", prompt="[g_prop] {query}"),
        RoleSpec(
            name="g_final",
            worker="tier2",
            role_type="synthesizer",
            depends_on=("g_prop",),
            prompt="[g_final] {query} {g_prop}",
            prompt_headless="[g_final_headless] {query} {g_prop}",
        ),
    )
    fast_roles = (
        RoleSpec(
            name="f_final",
            worker="tier2",
            role_type="publisher",
            prompt="[f_final] {query}",
        ),
    )
    engines = {"tier1": tier1, "tier2": tier2}
    if judge is not None:
        engines["judge"] = judge
    return Orchestrator(
        engines=engines,
        router=MultiAgentRouter(),
        roles=primary_roles,
        profiles={"ensemble": ensemble_roles, "fast": fast_roles},
        profile_judge=(
            ProfileJudge(
                worker="judge",
                timeout_seconds=0.05,
                prompt_prefix=judge_prompt_prefix,
                prompt_suffix=judge_prompt_suffix,
                choices=(
                    ProfileChoice("fast", "FAST", "trivial requests"),
                    ProfileChoice("primary", "CODE", "code authoring"),
                    ProfileChoice("ensemble", "GENERAL", "everything else"),
                ),
                fallback=fallback,
            )
            if judge is not None
            else None
        ),
    )


_JUDGE_PROMPT_MARKER = "Reply with exactly one word: FAST, CODE, GENERAL."


async def test_without_judge_every_turn_runs_primary_profile():
    tier1 = MockBackend()
    tier2 = MockBackend()
    orchestrator = _profiled_orchestrator(tier1, tier2)
    response_format = {"type": "json_object"}
    call = OrchestrationRequest(
        prompt="Implement a Python function that reverses a list.",
        sampling_params=SamplingParams(
            max_tokens=64,
            extra_args={"response_format": response_format},
        ),
        response_format=response_format,
    )
    result = await orchestrator.run(call)
    assert "role profile: primary" in result.trace
    prompts = tier1.prompts_seen + tier2.prompts_seen
    assert any("[c_final_headless]" in prompt for prompt in prompts)
    assert not any("[g_" in prompt or "[f_final" in prompt for prompt in prompts)


def test_bounded_profile_judge_view_preserves_request_intent_and_tail():
    intent = "Summarize this source without modifying it.\n"
    trailing_context = "\nFinal context: focus on the public API."
    prompt = intent + ("x" * 5000) + trailing_context

    view = _bounded_profile_judge_view(prompt)

    assert len(view) == 4000
    assert view.startswith(intent)
    assert view.endswith(trailing_context)
    assert "[... middle of request omitted ...]" in view
    assert _bounded_profile_judge_view("short request") == "short request"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("CODE", "primary"),
        (" code\n", "primary"),
        ("GENERAL", "ensemble"),
        ("\tgeneral ", "ensemble"),
        ("FAST.", "fast"),
        ("not CODE", None),
        ("The answer is CODE.", None),
        ("GENERAL explanation", None),
        ("CODE or GENERAL", None),
        ("ENSEMBLE", None),
        ("", None),
    ],
)
def test_parse_profile_verdict_requires_exact_offered_label(text, expected):
    from kairyu.orchestration.orchestrator import ProfileChoice

    choices = (
        ProfileChoice("fast", "FAST", "trivial"),
        ProfileChoice("primary", "CODE", "code"),
        ProfileChoice("ensemble", "GENERAL", "rest"),
    )
    assert _parse_profile_verdict(text, choices) == expected


def test_profile_judge_admission_reserves_judge_and_worst_profile():
    tier1 = MockBackend()
    tier2 = MockBackend()
    judge = MockBackend(responses={_JUDGE_PROMPT_MARKER: "GENERAL"})
    orchestrator = _profiled_orchestrator(tier1, tier2, judge)
    call = OrchestrationRequest(
        prompt="What does a program manager do?",
        sampling_params=SamplingParams(max_tokens=64),
    )

    combined = orchestrator.admission_upper_bound(call)
    profile_bounds = [
        orchestrator.admission_upper_bound(replace(call, role_profile_judgment=profile)).tokens
        for profile in ("primary", "ensemble", "fast")
    ]

    assert combined.tokens > max(profile_bounds)
    assert not combined.refundable_on_exact_usage


async def test_judge_applies_configured_worker_prompt_scaffold():
    tier1 = MockBackend()
    tier2 = MockBackend()
    judge = MockBackend(responses={_JUDGE_PROMPT_MARKER: "GENERAL"})
    orchestrator = _profiled_orchestrator(
        tier1,
        tier2,
        judge,
        judge_prompt_prefix="<bos><user>",
        judge_prompt_suffix="<assistant></think>",
    )

    await orchestrator.judge_role_profile(
        OrchestrationRequest(
            prompt="Explain Python decorators.",
            sampling_params=SamplingParams(max_tokens=64),
        )
    )

    assert len(judge.prompts_seen) == 1
    prompt = judge.prompts_seen[0]
    assert isinstance(prompt, TemplatedPrompt)
    assert prompt.startswith("<bos><user>")
    assert prompt.endswith("<assistant></think>")
    assert _JUDGE_PROMPT_MARKER in prompt
    # The spec's choices and criteria are the routing policy the judge reads,
    # fastest first, plus the deterministic request flags.
    assert prompt.index("- FAST: trivial requests") < prompt.index("- CODE: code authoring")
    assert "- GENERAL: everything else" in prompt
    assert "tool calling no; image attached no" in prompt


async def test_judge_scaffold_dispatches_over_vllm_completions_wire():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "text": "GENERAL",
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 32,
                    "completion_tokens": 1,
                    "total_tokens": 33,
                },
            },
        )

    tier1 = MockBackend()
    tier2 = MockBackend()
    judge = OpenAICompatBackend(
        base_url="http://vllm:8000/v1",
        model="judge",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
        upstream="vllm",
        allow_templated_chat_passthrough=True,
    )
    orchestrator = _profiled_orchestrator(
        tier1,
        tier2,
        judge,
        judge_prompt_prefix="<bos><user>",
        judge_prompt_suffix="<assistant></think>",
    )

    call = await orchestrator.judge_role_profile(
        OrchestrationRequest(
            prompt="Explain Python decorators.",
            sampling_params=SamplingParams(max_tokens=64),
        )
    )
    await judge.shutdown()

    assert call.role_profile_judgment == "ensemble"
    assert captured["path"] == "/v1/completions"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["prompt"].startswith("<bos><user>")
    assert body["prompt"].endswith("<assistant></think>")
    assert "messages" not in body


@pytest.mark.parametrize(
    ("verdict", "profile", "expected_marker", "absent_markers"),
    [
        ("GENERAL", "ensemble", "[g_prop]", ("[c_final", "[f_final")),
        ("CODE", "primary", "[c_final", ("[g_", "[f_final")),
        ("FAST", "fast", "[f_final", ("[g_", "[c_final")),
    ],
)
async def test_judge_verdict_selects_the_profile(
    verdict,
    profile,
    expected_marker,
    absent_markers,
):
    tier1 = MockBackend()
    tier2 = MockBackend()
    judge = MockBackend(responses={_JUDGE_PROMPT_MARKER: verdict})
    orchestrator = _profiled_orchestrator(tier1, tier2, judge)
    call = await orchestrator.judge_role_profile(
        OrchestrationRequest(
            prompt="What does a program manager do?",
            sampling_params=SamplingParams(max_tokens=64),
        )
    )
    assert call.role_profile_judgment == profile
    result = await orchestrator.run(call)
    assert f"role profile: {profile}" in result.trace
    assert f"profile judge: {profile}" in result.trace
    prompts = tier1.prompts_seen + tier2.prompts_seen
    assert any(expected_marker in prompt for prompt in prompts)
    assert not any(marker in prompt for prompt in prompts for marker in absent_markers)


async def test_tool_and_format_turns_are_judged_too():
    # DTO-D13: head-disable signals no longer pin a profile; the judge sees
    # the turn (with its tool flag) and a head-less direct route answers.
    tier1 = MockBackend()
    tier2 = MockBackend()
    judge = MockBackend(responses={_JUDGE_PROMPT_MARKER: "FAST"})
    orchestrator = _profiled_orchestrator(tier1, tier2, judge)
    response_format = {"type": "json_object"}
    call = await orchestrator.judge_role_profile(
        OrchestrationRequest(
            prompt="Implement a Python function that reverses a list.",
            sampling_params=SamplingParams(
                max_tokens=64,
                extra_args={"response_format": response_format},
            ),
            response_format=response_format,
            tools_in_prompt=True,
        )
    )
    assert call.role_profile_judgment == "fast"
    assert len(judge.prompts_seen) == 1
    assert "tool calling yes" in judge.prompts_seen[0]
    result = await orchestrator.run(call)
    assert "role profile: fast" in result.trace
    assert any("[f_final" in prompt for prompt in tier2.prompts_seen)


async def test_image_request_is_offered_only_image_capable_profiles():
    # The DeepSeek-style `fast` route (tier2, text-only) must never be
    # selected for an image request: it is withheld from the judge and an
    # out-of-set verdict falls back.
    from kairyu.engine.prompt import MultimodalItem, MultimodalPrompt

    tier1 = _VisionMockBackend()
    tier2 = MockBackend()
    judge = MockBackend(responses={"Reply with exactly one word: CODE, GENERAL.": "CODE"})
    orchestrator = _profiled_orchestrator(tier1, tier2, judge)
    query = "What is on this slide?"
    call = OrchestrationRequest(
        prompt=query,
        sampling_params=SamplingParams(max_tokens=64),
        multimodal_prompt=MultimodalPrompt(
            query, (MultimodalItem("image", "uri", "data:image/png;base64,AAAA"),)
        ),
    )
    judged = await orchestrator.judge_role_profile(call)
    assert judged.role_profile_judgment == "primary"
    prompt = judge.prompts_seen[0]
    assert "- FAST:" not in prompt
    assert "image attached yes" in prompt

    # A verdict naming the withheld route is unparseable, not a selection.
    judge_fast = MockBackend(responses={"Reply with exactly one word: CODE, GENERAL.": "FAST"})
    orchestrator = _profiled_orchestrator(tier1, tier2, judge_fast)
    judged = await orchestrator.judge_role_profile(call)
    assert judged.role_profile_judgment is None
    assert judged.role_profile_judge_event is not None
    assert judged.role_profile_judge_event.status == "failed"


async def test_judge_failure_falls_back_to_configured_profile():
    tier1 = MockBackend()
    tier2 = MockBackend()
    slow_judge = MockBackend(
        responses={_JUDGE_PROMPT_MARKER: "GENERAL"},
        latency_s=5.0,
    )
    orchestrator = _profiled_orchestrator(tier1, tier2, slow_judge, fallback="ensemble")
    call = await orchestrator.judge_role_profile(
        OrchestrationRequest(
            prompt="Implement a python function that reverses a list.",
            sampling_params=SamplingParams(max_tokens=64),
        )
    )
    assert call.role_profile_judgment is None
    result = await orchestrator.run(call)
    assert "role profile: ensemble" in result.trace
    assert "profile judge:" not in result.trace
    prompts = tier1.prompts_seen + tier2.prompts_seen
    assert any("[g_prop]" in prompt for prompt in prompts)


async def test_unparseable_judge_verdict_falls_back():
    tier1 = MockBackend()
    tier2 = MockBackend()
    judge = MockBackend(responses={_JUDGE_PROMPT_MARKER: "not CODE"})
    orchestrator = _profiled_orchestrator(tier1, tier2, judge)
    call = await orchestrator.judge_role_profile(
        OrchestrationRequest(
            prompt="Implement a python function that reverses a list.",
            sampling_params=SamplingParams(max_tokens=64),
        )
    )
    assert call.role_profile_judgment is None
    assert call.role_profile_judge_event is not None
    assert call.role_profile_judge_event.usage is not None
    assert call.role_profile_judge_event.status == "failed"
    repeated = await orchestrator.judge_role_profile(call)
    assert repeated is call
    assert len(judge.prompts_seen) == 1
    result = await orchestrator.run(call)
    assert "role profile: primary" in result.trace


async def test_judged_call_preflight_matches_execution():
    # The judged verdict travels with the call, so preflight, admission, and
    # execution agree on the selected profile.
    tier1 = MockBackend()
    tier2 = MockBackend()
    judge = MockBackend(responses={_JUDGE_PROMPT_MARKER: "GENERAL"})
    orchestrator = _profiled_orchestrator(tier1, tier2, judge)
    call = await orchestrator.judge_role_profile(
        OrchestrationRequest(
            prompt="Explain how Python became popular.",
            sampling_params=SamplingParams(max_tokens=64),
        )
    )
    prepared = await orchestrator.prepare_request(call)
    await orchestrator.admission_upper_bound_async(call)
    result = await orchestrator.run(call, prepared=prepared)
    assert "role profile: ensemble" in result.trace
    assert result.completions
    prompts = tier1.prompts_seen + tier2.prompts_seen
    assert any("[g_prop]" in prompt for prompt in prompts)
    assert not any("[c_final" in prompt for prompt in prompts)


async def test_judge_and_non_thinking_final_disable_thinking_explicitly():
    """DTO-D13 amendment: the judge (non-streamed, non-thinking) and an
    effort-less final unit tell a capable worker enable_thinking=False; the
    preflight final intent carries the same kwargs as the dispatch."""
    from kairyu.orchestration.conductor import RoleSpec

    class ThinkingSwitchBackend(MockBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.validated: list[GenerationRequest] = []
            self.generated: list[GenerationRequest] = []

        def supports_chat_template_kwargs(self, keys: frozenset[str]) -> bool:
            return keys <= {"enable_thinking"}

        def validate_request(self, request: GenerationRequest) -> None:
            self.validated.append(request)
            super().validate_request(request)

        async def generate(self, request: GenerationRequest) -> GenerationResult:
            self.generated.append(request)
            return await super().generate(request)

    class MultiAgentRouter:
        def preview(self, query, context=None):
            return RuleRouter().route(COMPLEX)

        def route(self, query, context=None):
            return RuleRouter().route(COMPLEX)

    tier1 = ThinkingSwitchBackend()
    tier2 = MockBackend()
    judge = ThinkingSwitchBackend(responses={"Reply with exactly one word": "FAST"})
    from kairyu.orchestration.orchestrator import ProfileChoice, ProfileJudge

    orchestrator = Orchestrator(
        engines={"tier1": tier1, "tier2": tier2, "judge": judge},
        router=MultiAgentRouter(),
        roles=(
            RoleSpec(name="prop", worker="tier2", prompt="[prop] {query}"),
            RoleSpec(
                name="final",
                worker="tier1",
                role_type="publisher",
                depends_on=("prop",),
                prompt="[final] {query} {prop}",
            ),
        ),
        profiles={
            "fast": (
                RoleSpec(
                    name="fast", worker="tier1", role_type="publisher", prompt="[fast] {query}"
                ),
            ),
        },
        profile_judge=ProfileJudge(
            worker="judge",
            choices=(
                ProfileChoice("fast", "FAST", "trivial"),
                ProfileChoice("primary", "SLOW", "rest"),
            ),
        ),
        default_reasoning_effort="high",
    )
    call = OrchestrationRequest(prompt="hello there", sampling_params=SamplingParams(max_tokens=32))

    judged = await orchestrator.judge_role_profile(call)
    assert judged.role_profile_judgment == "fast"
    assert judge.generated[0].chat_template_kwargs == {"enable_thinking": False}

    # Primary (non-initial final on the capable worker): preflight and
    # dispatch agree on the explicit non-thinking switch.
    primary_call = replace(judged, role_profile_judgment="primary")
    orchestrator.validate_request(primary_call)
    assert tier1.validated[-1].chat_template_kwargs == {"enable_thinking": False}
    assert tier1.validated[-1].reasoning_effort is None
    await orchestrator.run(primary_call)
    assert tier1.generated[-1].chat_template_kwargs == {"enable_thinking": False}
    assert tier1.generated[-1].reasoning_effort is None
def test_trace_envelope_starts_at_judge_queued_at() -> None:
    """A judge event queued one clock-millisecond before its started_at must
    stay inside the trace envelope (serving_bench rejects the whole trace
    otherwise)."""
    from kairyu.orchestration.orchestrator import _trace_started_at_from
    from kairyu.orchestration.trace import TraceEvent, TraceTiming

    event = TraceEvent(
        node="profile_judge",
        kind="judged",
        operation="classification",
        role="profile_judge",
        timing=TraceTiming(
            queued_at="2026-08-25T13:28:04.528Z",
            started_at="2026-08-25T13:28:04.529Z",
            completed_at="2026-08-25T13:28:04.813Z",
        ),
    )
    assert _trace_started_at_from(event) == "2026-08-25T13:28:04.528Z"
    assert _trace_started_at_from(None)  # falls back to a fresh timestamp
