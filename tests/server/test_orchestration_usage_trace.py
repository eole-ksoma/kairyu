import asyncio
import json

import pytest

from kairyu.engine.backend import GenerationResult, GenerationUsage
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.orchestration.budget import Budget
from kairyu.orchestration.orchestrator import EngineDescriptor, Orchestrator
from kairyu.outputs import CompletionOutput
from tests.server._legacy_chat import create_legacy_app

COMPLEX = (
    "First research the options and summarize trade-offs. Then design a plan. "
    "After that implement it. Finally verify everything end to end."
)


class AccountingBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    @staticmethod
    def _text(prompt: str) -> str:
        if "[verifier]" in prompt:
            return "PASS"
        if "Synthesize the best" in prompt or "[synthesizer]" in prompt:
            return "final synthesized answer"
        return "useful intermediate answer"

    @staticmethod
    def _result(request, text: str, *, completion_tokens: int = 2):
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0,
                    text=text,
                    token_ids=tuple(range(completion_tokens)),
                    finish_reason="stop",
                ),
            ),
            usage=GenerationUsage(
                prompt_tokens=10,
                completion_tokens=completion_tokens,
                cached_tokens=3,
            ),
        )

    async def generate(self, request):
        self.calls.append(request.request_id)
        return self._result(request, self._text(request.prompt))

    async def stream(self, request):
        self.calls.append(request.request_id)
        text = self._text(request.prompt)
        split = max(1, len(text) // 2)
        yield self._result(request, text[:split], completion_tokens=1)
        yield self._result(request, text)

    async def shutdown(self) -> None:
        return None


def _app(
    tmp_path,
    backend,
    *,
    moa_samples: int = 0,
    expose_intermediate_outputs: bool = False,
):
    return create_legacy_app(
        {"plain": backend},
        orchestrators={
            "auto": Orchestrator(
                {"tier1": backend, "tier2": backend},
                moa_samples=moa_samples,
                expose_intermediate_outputs=expose_intermediate_outputs,
                engine_descriptors={
                    "tier1": EngineDescriptor("mock", "Qwen3.6-27B"),
                    "tier2": EngineDescriptor("mock", "DeepSeek-V4"),
                },
            )
        },
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
    )


def _sse_payloads(body: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: {")
    ]


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize(
    ("prompt", "moa_samples", "expected_calls"),
    [
        pytest.param("hello", 0, 1, id="direct"),
        pytest.param(COMPLEX, 0, 4, id="conductor"),
        pytest.param(COMPLEX, 2, 3, id="moa"),
    ],
)
def test_auto_usage_reconciles_every_success_route(
    tmp_path,
    stream,
    prompt,
    moa_samples,
    expected_calls,
):
    backend = AccountingBackend()
    app = _app(tmp_path, backend, moa_samples=moa_samples)
    body = {
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    if stream:
        usage = next(
            payload["usage"]
            for payload in reversed(_sse_payloads(response.text))
            if payload.get("usage") is not None
        )
    else:
        usage = response.json()["usage"]
    assert len(backend.calls) == expected_calls
    assert usage["orchestration_input_tokens"] == expected_calls * 10
    assert usage["orchestration_output_tokens"] == expected_calls * 2
    # Standard usage keeps the documented OpenAI meaning (issue #496): the
    # client-visible completion, never the sum over internal calls. The
    # cumulative totals stay on the orchestration_* extension fields above.
    assert usage["completion_tokens"] == 2
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    if expected_calls > 1:
        assert usage["completion_tokens"] < usage["orchestration_output_tokens"]


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
def test_visible_intermediates_use_reasoning_content_with_model_attribution(
    tmp_path,
    stream,
):
    backend = AccountingBackend()
    app = _app(tmp_path, backend, expose_intermediate_outputs=True)

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": COMPLEX}],
                "stream": stream,
            },
        )

    assert response.status_code == 200
    if stream:
        payloads = _sse_payloads(response.text)
        reasoning = "".join(
            choice["delta"].get("reasoning_content") or ""
            for payload in payloads
            for choice in payload.get("choices", ())
        )
        content = "".join(
            choice["delta"].get("content") or ""
            for payload in payloads
            for choice in payload.get("choices", ())
        )
    else:
        message = response.json()["choices"][0]["message"]
        reasoning = message["reasoning_content"]
        content = message["content"]

    assert content == "final synthesized answer"
    if not stream:
        assert reasoning.startswith("### Final answer attribution")
        assert "L2 role: `synthesizer`" in reasoning
        assert "L1 worker: `tier2`" in reasoning
        assert "Model: `DeepSeek-V4`" in reasoning
    assert "useful intermediate answer" in reasoning
    assert "PASS" in reasoning
    assert "L2 role: `planner`" in reasoning
    assert "L1 worker: `tier2`" in reasoning
    assert "Model: `Qwen3.6-27B`" in reasoning
    assert "Model: `DeepSeek-V4`" in reasoning
    assert "final synthesized answer" not in reasoning


def test_visible_reasoning_response_round_trips_through_litellm_history(tmp_path):
    backend = AccountingBackend()
    app = _app(tmp_path, backend, expose_intermediate_outputs=True)

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        first = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": COMPLEX}],
            },
        )
        assistant = first.json()["choices"][0]["message"]
        assistant["function_call"] = None
        assistant["provider_specific_fields"] = {"refusal": None}
        second = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [
                    {"role": "user", "content": COMPLEX},
                    assistant,
                    {"role": "user", "content": "Continue."},
                ],
            },
        )

    assert first.status_code == 200
    assert assistant["reasoning_content"]
    assert second.status_code == 200


def test_non_auto_usage_shape_is_unchanged_and_auto_fields_are_discoverable(
    tmp_path,
):
    backend = AccountingBackend()
    app = _app(tmp_path, backend)

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "plain",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        usage_schema = client.get("/openapi.json").json()["components"]["schemas"]["Usage"][
            "properties"
        ]

    usage = response.json()["usage"]
    assert "orchestration_input_tokens" not in usage
    assert "orchestration_output_tokens" not in usage
    assert "orchestration_input_tokens" in usage_schema
    assert "orchestration_output_tokens" in usage_schema


def test_stream_trace_opt_in_returns_route_dag_verifier_and_timing(tmp_path):
    backend = AccountingBackend()
    app = _app(tmp_path, backend)

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Kairyu-Trace": "1"},
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": COMPLEX}],
                "stream": True,
            },
        )

    metadata = next(
        payload for payload in _sse_payloads(response.text) if "kairyu_trace_v2" in payload
    )
    assert "usage" not in metadata
    assert metadata["kairyu_route"]["target"] == "multi_agent"
    trace = metadata["kairyu_trace_v2"]
    assert trace["trace_version"] == "2.0"
    assert trace["request_id"].startswith("chatcmpl-")
    assert trace["events"][0]["kind"] == "routing"
    verifier = next(event for event in trace["events"] if event["role"] == "verifier")
    final = next(event for event in trace["events"] if event["role"] == "synthesizer")
    assert verifier["kind"] == "verification"
    assert verifier["detail"]["pass"] is True
    assert final["timing"]["first_token_at"]
    assert final["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "cached_tokens": 3,
    }
    assert COMPLEX not in response.text


def test_stream_without_trace_opt_in_omits_trace_extensions(tmp_path):
    backend = AccountingBackend()
    app = _app(tmp_path, backend)

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

    assert ": trace" not in response.text
    for payload in _sse_payloads(response.text):
        assert "kairyu_trace" not in payload
        assert "kairyu_trace_v2" not in payload
        assert "kairyu_route" not in payload


def test_openai_sdk_accepts_usage_and_trace_terminal_metadata(tmp_path):
    import openai
    from fastapi.testclient import TestClient

    backend = AccountingBackend()
    app = _app(tmp_path, backend)
    with TestClient(app) as http:
        client = openai.OpenAI(
            base_url=str(http.base_url) + "/v1",
            api_key="sk-local",
            http_client=http,
        )
        chunks = list(
            client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": COMPLEX}],
                stream=True,
                stream_options={"include_usage": True},
                extra_headers={"X-Kairyu-Trace": "1"},
            )
        )

    assert (
        "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)
        == "final synthesized answer"
    )
    metadata = next(chunk for chunk in chunks if getattr(chunk, "kairyu_trace_v2", None))
    assert metadata.usage.orchestration_input_tokens == 40
    assert metadata.usage.orchestration_output_tokens == 8
    assert metadata.kairyu_trace_v2["events"][0]["kind"] == "routing"


def test_conductor_retry_usage_equals_trace_call_sum(tmp_path):
    class RetryBackend(AccountingBackend):
        def __init__(self) -> None:
            super().__init__()
            self.verifications = 0

        async def generate(self, request):
            self.calls.append(request.request_id)
            if "[verifier]" in request.prompt:
                self.verifications += 1
                verdict = "FAIL" if self.verifications == 1 else "PASS"
                return self._result(request, verdict)
            return self._result(request, self._text(request.prompt))

    backend = RetryBackend()
    app = create_legacy_app(
        {},
        orchestrators={
            "auto": Orchestrator(
                {"tier1": backend, "tier2": backend},
                budget=Budget(max_steps=8, max_refine_depth=1),
            )
        },
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
    )

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Kairyu-Trace": "1"},
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": COMPLEX}],
            },
        )

    payload = response.json()
    assert len(backend.calls) == 6
    assert payload["usage"]["orchestration_input_tokens"] == 60
    assert payload["usage"]["orchestration_output_tokens"] == 12
    traced = [
        event["usage"]
        for event in payload["kairyu_trace_v2"]["events"]
        if event["usage"] is not None
    ]
    assert sum(item["prompt_tokens"] for item in traced) == 60
    assert sum(item["completion_tokens"] for item in traced) == 12


def test_fallback_engine_usage_and_trace_use_resolved_engine(tmp_path):
    backend = AccountingBackend()
    app = create_legacy_app(
        {},
        orchestrators={"auto": Orchestrator({"fallback": backend})},
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
    )

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Kairyu-Trace": "1"},
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    payload = response.json()
    assert payload["usage"]["orchestration_input_tokens"] == 10
    generation = payload["kairyu_trace_v2"]["events"][-1]
    assert generation["worker"] == "tier1"
    assert generation["engine"] == "fallback"


def test_partial_stream_failure_returns_known_usage_and_sanitized_trace(tmp_path):
    class PartialFailureBackend(AccountingBackend):
        async def stream(self, request):
            self.calls.append(request.request_id)
            yield self._result(request, "partial answer", completion_tokens=1)
            raise RuntimeError("credential=DO_NOT_EXPOSE")

    backend = PartialFailureBackend()
    app = _app(tmp_path, backend)

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Kairyu-Trace": "1"},
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        totals = app.state.usage_ledger.totals()["default"]

    payloads = _sse_payloads(response.text)
    metadata = next(payload for payload in payloads if "kairyu_trace_v2" in payload)
    assert metadata["usage"]["orchestration_input_tokens"] == 10
    assert metadata["usage"]["orchestration_output_tokens"] == 1
    failure = metadata["kairyu_trace_v2"]["events"][-1]
    assert failure["status"] == "failed"
    assert failure["error"] == {"type": "RuntimeError", "retryable": False}
    assert "DO_NOT_EXPOSE" not in response.text
    assert totals["requests"] == 1
    assert totals["prompt_tokens"] == 10
    assert totals["completion_tokens"] == 1


def test_unary_direct_failure_returns_zero_known_usage_and_sanitized_trace(tmp_path):
    class FailingBackend(AccountingBackend):
        async def generate(self, request):
            self.calls.append(request.request_id)
            raise RuntimeError("prompt and credential must stay server-side")

    backend = FailingBackend()
    app = _app(tmp_path, backend)

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Kairyu-Trace": "1"},
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 502
    payload = response.json()
    assert payload["usage"]["orchestration_input_tokens"] == 0
    assert payload["usage"]["orchestration_output_tokens"] == 0
    failure = payload["kairyu_trace_v2"]["events"][-1]
    assert failure["status"] == "failed"
    assert failure["error"]["type"] == "RuntimeError"
    assert "credential must stay" not in response.text


def test_moa_proposal_failure_returns_completed_call_usage_and_safe_error(tmp_path):
    class ProposalFailureBackend(AccountingBackend):
        async def generate(self, request):
            self.calls.append(request.request_id)
            if request.request_id.endswith("propose-1"):
                raise RuntimeError("secret backend URL")
            return self._result(request, self._text(request.prompt))

    backend = ProposalFailureBackend()
    app = _app(tmp_path, backend, moa_samples=2)

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Kairyu-Trace": "1"},
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": COMPLEX}],
            },
        )
        totals = app.state.usage_ledger.totals()["default"]

    assert response.status_code == 502
    payload = response.json()
    assert len(backend.calls) == 2
    assert payload["usage"]["orchestration_input_tokens"] == 10
    assert payload["usage"]["orchestration_output_tokens"] == 2
    failure = payload["kairyu_trace_v2"]["events"][-1]
    assert failure["status"] == "failed"
    assert failure["detail"]["stage"] == "proposal"
    assert failure["error"]["type"] == "RuntimeError"
    assert "secret backend URL" not in response.text
    assert totals["prompt_tokens"] == 10
    assert totals["completion_tokens"] == 2


async def test_conductor_disconnect_finalizes_prefinal_and_partial_usage(tmp_path):
    class BlockingBackend(AccountingBackend):
        def __init__(self) -> None:
            super().__init__()
            self.cancelled = False

        async def stream(self, request):
            self.calls.append(request.request_id)
            try:
                yield self._result(request, "partial answer", completion_tokens=1)
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    backend = BlockingBackend()
    app = _app(tmp_path, backend)
    body = {
        "model": "auto",
        "messages": [{"role": "user", "content": COMPLEX}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    encoded = json.dumps(body).encode()
    request_sent = False
    disconnect = asyncio.Event()

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": encoded, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and b"partial answer" in message.get(
            "body", b""
        ):
            disconnect.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"test"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(encoded)).encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }

    await asyncio.wait_for(app(scope, receive, send), timeout=2)
    totals = await asyncio.to_thread(app.state.usage_ledger.totals)
    assert backend.cancelled is True
    assert len(backend.calls) == 4
    assert totals["default"]["requests"] == 1
    assert totals["default"]["prompt_tokens"] == 40
    assert totals["default"]["completion_tokens"] == 7


async def test_conductor_disconnect_mid_prefinal_keeps_completed_call_usage(tmp_path):
    class MidPrefinalBackend(AccountingBackend):
        def __init__(self) -> None:
            super().__init__()
            self.second_started = asyncio.Event()
            self.cancelled = False

        async def generate(self, request):
            self.calls.append(request.request_id)
            if len(self.calls) == 2:
                self.second_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
            return self._result(request, self._text(request.prompt))

    backend = MidPrefinalBackend()
    app = _app(tmp_path, backend)
    body = {
        "model": "auto",
        "messages": [{"role": "user", "content": COMPLEX}],
        "stream": True,
    }
    encoded = json.dumps(body).encode()
    request_sent = False
    disconnect = asyncio.Event()

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": encoded, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        return None

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"test"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(encoded)).encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }

    request_task = asyncio.create_task(app(scope, receive, send))
    await asyncio.wait_for(backend.second_started.wait(), timeout=1)
    disconnect.set()
    await asyncio.wait_for(request_task, timeout=2)
    totals = await asyncio.to_thread(app.state.usage_ledger.totals)
    assert backend.cancelled is True
    assert len(backend.calls) == 2
    assert totals["default"]["requests"] == 1
    assert totals["default"]["prompt_tokens"] == 10
    assert totals["default"]["completion_tokens"] == 2


async def test_moa_cancellation_observer_keeps_completed_proposal_usage():
    class MidProposalBackend(AccountingBackend):
        def __init__(self) -> None:
            super().__init__()
            self.blocked = asyncio.Event()
            self.cancelled = False

        async def generate(self, request):
            self.calls.append(request.request_id)
            if request.request_id.endswith("propose-1"):
                self.blocked.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
            return self._result(request, self._text(request.prompt))

    backend = MidProposalBackend()
    observed: list[GenerationUsage] = []
    orchestrator = Orchestrator(
        {"tier1": backend, "tier2": backend},
        moa_samples=2,
    )
    stream = await orchestrator.run_chat(
        COMPLEX,
        stream=True,
        usage_observer=observed.append,
    )

    assert (await anext(stream)).kind == "status"
    pending = asyncio.create_task(anext(stream))
    await asyncio.wait_for(backend.blocked.wait(), timeout=1)
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert backend.cancelled is True
    assert observed[-1] == GenerationUsage(
        prompt_tokens=10,
        completion_tokens=2,
        cached_tokens=3,
    )


def _judge_policy():
    from kairyu.orchestration.orchestrator import ProfileChoice, ProfileJudge

    return ProfileJudge(
        worker="judge",
        choices=(
            ProfileChoice("primary", "CODE", "code authoring"),
            ProfileChoice("general", "GENERAL", "everything else"),
        ),
    )


def test_chat_endpoint_attaches_profile_judge_verdict(tmp_path):
    # Issue #509 amendment / DTO-D13: the serving boundary judges the
    # profile once before preflight/admission, and the verdict selects the
    # executed DAG.
    from kairyu.engine.mock import MockBackend
    from kairyu.orchestration.conductor import RoleSpec

    engine = MockBackend()
    judge = MockBackend(responses={"Reply with exactly one word: CODE, GENERAL.": "GENERAL"})
    orchestrator = Orchestrator(
        {"tier1": engine, "tier2": engine, "judge": judge},
        roles=(
            RoleSpec(
                name="c_final",
                worker="tier1",
                role_type="synthesizer",
                prompt="[c_final] {query}",
            ),
        ),
        profiles={
            "general": (
                RoleSpec(
                    name="g_final",
                    worker="tier1",
                    role_type="synthesizer",
                    prompt="[g_final] {query}",
                ),
            )
        },
        profile_judge=_judge_policy(),
    )
    app = create_legacy_app(
        {"plain": engine},
        orchestrators={"auto": orchestrator},
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
    )
    prompt = (
        "First research what a program manager does and summarize the "
        "trade-offs. Then design a plan. After that implement it. Finally "
        "verify everything end to end."
    )

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Kairyu-Trace": "1"},
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": prompt}],
            },
        )

    assert response.status_code == 200
    assert len(judge.prompts_seen) == 1
    payload = response.json()
    judgment = payload["kairyu_trace_v2"]["events"][0]
    assert judgment["node"] == "profile_judge"
    assert judgment["kind"] == "classification"
    assert judgment["detail"]["verdict"] == "general"
    assert judgment["usage"] is not None
    traced = [
        event["usage"]
        for event in payload["kairyu_trace_v2"]["events"]
        if event["usage"] is not None
    ]
    usage = payload["usage"]
    assert usage["orchestration_input_tokens"] == sum(item["prompt_tokens"] for item in traced)
    assert usage["orchestration_output_tokens"] == sum(item["completion_tokens"] for item in traced)
    prompts = [p for p in engine.prompts_seen if isinstance(p, str)]
    assert any("[g_final]" in p for p in prompts)
    assert not any("[c_final]" in p for p in prompts)


def test_profile_judge_quota_is_reserved_before_dispatch():
    from fastapi.testclient import TestClient

    from kairyu.engine.mock import MockBackend
    from kairyu.entrypoints.server.tenancy import TenantConfig, TenantLimits
    from kairyu.orchestration.conductor import RoleSpec

    engine = MockBackend()
    judge = MockBackend(responses={"Reply with exactly one word: CODE, GENERAL.": "GENERAL"})
    final_role = RoleSpec(
        name="final",
        worker="tier1",
        role_type="synthesizer",
        prompt="[final] {query}",
    )
    orchestrator = Orchestrator(
        {"tier1": engine, "judge": judge},
        roles=(final_role,),
        profiles={"general": (final_role,)},
        profile_judge=_judge_policy(),
    )
    app = create_legacy_app(
        {"plain": engine},
        orchestrators={"auto": orchestrator},
        tenant_config=TenantConfig(
            limits={
                "default": TenantLimits(
                    requests_per_minute=60,
                    tokens_per_minute=1,
                    token_burst=1,
                )
            }
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Write a Rust CLI."}],
            },
        )

    assert response.status_code == 429
    assert not judge.prompts_seen
    assert not engine.prompts_seen


def test_profile_judge_usage_settles_when_later_preflight_rejects(tmp_path):
    from fastapi.testclient import TestClient

    from kairyu.engine.mock import MockBackend
    from kairyu.entrypoints.server.tenancy import TenantConfig, TenantLimits
    from kairyu.orchestration.conductor import RoleSpec

    class UsageJudge(MockBackend):
        def __init__(self):
            super().__init__(
                responses={"Reply with exactly one word: CODE, GENERAL.": "GENERAL"}
            )
            self.reported_usage = None

        async def generate(self, request):
            result = await super().generate(request)
            self.reported_usage = result.usage
            return result

    class FailingPrepareBackend(MockBackend):
        async def prepare_request(self, request):
            raise ValueError("final worker rejected prepared request")

    engine = FailingPrepareBackend()
    judge = UsageJudge()
    final_role = RoleSpec(
        name="final",
        worker="tier1",
        role_type="synthesizer",
        prompt="[final] {query}",
    )
    orchestrator = Orchestrator(
        {"tier1": engine, "judge": judge},
        roles=(final_role,),
        profiles={"general": (final_role,)},
        profile_judge=_judge_policy(),
    )
    token_budget = 100_000
    app = create_legacy_app(
        {"plain": engine},
        orchestrators={"auto": orchestrator},
        tenant_config=TenantConfig(
            limits={
                "default": TenantLimits(
                    requests_per_minute=60,
                    tokens_per_minute=60,
                    token_burst=token_budget,
                )
            }
        ),
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Write a Rust CLI."}],
                "max_tokens": 8,
            },
        )

    assert response.status_code == 400
    assert len(judge.prompts_seen) == 1
    assert not engine.prompts_seen
    assert judge.reported_usage is not None
    actual_tokens = (
        judge.reported_usage.prompt_tokens
        + judge.reported_usage.completion_tokens
    )
    assert app.state.tenant_limiter.token_balance("default") == pytest.approx(
        token_budget - actual_tokens
    )
    assert app.state.tenant_limiter.reservation_snapshot()["default"] == 0
    totals = app.state.usage_ledger.totals()["default"]
    assert totals["requests"] == 1
    assert totals["prompt_tokens"] == judge.reported_usage.prompt_tokens
    assert totals["completion_tokens"] == judge.reported_usage.completion_tokens
def test_trace_v2_event_accepts_list_detail_values() -> None:
    """DTO-D13 judge events carry the offered label list; the chunk schema
    rejecting list detail values crashed live SSE streams mid-body."""
    from kairyu.entrypoints.server.protocol import (
        ChatCompletionChunk,
        KairyuTraceEvent,
    )

    event = KairyuTraceEvent(
        seq=0,
        node="profile_judge",
        role="profile_judge",
        kind="classification",
        status="success",
        detail={"verdict": "QWEN", "offered": ["QWEN", "ENSEMBLE"], "fallback": None},
    )
    assert event.detail["offered"] == ["QWEN", "ENSEMBLE"]
    chunk = ChatCompletionChunk(
        id="chatcmpl-offered",
        created=0,
        model="kairyu-auto-max",
        choices=[],
        kairyu_trace_v2={
            "trace_version": "2.0",
            "request_id": "chatcmpl-offered",
            "started_at": "2026-08-25T00:00:00Z",
            "completed_at": "2026-08-25T00:00:01Z",
            "events": [event.model_dump()],
        },
    )
    assert '"offered":["QWEN","ENSEMBLE"]' in chunk.model_dump_json()
