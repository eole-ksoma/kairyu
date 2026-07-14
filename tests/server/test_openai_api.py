import json

import httpx
import pytest

from kairyu.engine.backend import GenerationRequest
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app
from kairyu.orchestration.orchestrator import Orchestrator

TOOL_CALL_TEXT = '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture()
def app():
    engines = {
        "kairyu-mock": MockBackend(responses={"weather": TOOL_CALL_TEXT}),
    }
    orchestrator = Orchestrator(engines={"tier1": MockBackend(), "tier2": MockBackend()})
    return create_app(engines=engines, orchestrator=orchestrator)


def _chat_body(content: str, **extra) -> dict:
    return {
        "model": "kairyu-mock",
        "messages": [{"role": "user", "content": content}],
        **extra,
    }


class _FailingBackend:
    async def generate(self, request: GenerationRequest):
        raise RuntimeError("secret backend detail")

    async def shutdown(self) -> None:
        pass


async def test_models_endpoint_lists_engines_and_auto(app):
    async with _client(app) as client:
        response = await client.get("/v1/models")
    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert "kairyu-mock" in ids
    assert "kairyu-auto" in ids


async def test_chat_completion_happy_path(app):
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat_body("hello"))
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "kairyu-mock"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"]
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["total_tokens"] >= 0


async def test_backend_error_logs_traceback_without_leaking_message(caplog):
    app = create_app(engines={"boom": _FailingBackend()})
    body = _chat_body("hello", model="boom")
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 502
    message = response.json()["error"]["message"]
    assert message == "upstream backend error (RuntimeError)"
    assert "secret backend detail" not in message
    assert any(
        record.exc_info and record.getMessage() == "upstream backend error"
        for record in caplog.records
    )


async def test_streaming_reassembles_to_full_answer(app):
    async with _client(app) as client:
        full = (await client.post("/v1/chat/completions", json=_chat_body("hello"))).json()
        full_text = full["choices"][0]["message"]["content"]
        response = await client.post(
            "/v1/chat/completions", json=_chat_body("hello", stream=True)
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    deltas = []
    for line in lines[:-1]:
        chunk = json.loads(line[len("data: "):])
        assert chunk["object"] == "chat.completion.chunk"
        delta = chunk["choices"][0]["delta"]
        deltas.append(delta.get("content") or "")
    assert "".join(deltas) == full_text


async def test_tool_call_is_parsed_into_openai_schema(app):
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions", json=_chat_body("what is the weather?", tools=tools)
        )
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Tokyo"}


async def test_streamed_tool_calls_carry_required_index(app):
    # S6: each streamed delta.tool_calls[] item must include `index` so OpenAI
    # SDK stream accumulators can merge the fragments.
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]
    async with _client(app) as client:
        async with client.stream(
            "POST", "/v1/chat/completions",
            json=_chat_body("what is the weather?", tools=tools, stream=True),
        ) as response:
            saw_tool_call = False
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[len("data: "):])
                for tc in chunk["choices"][0]["delta"].get("tool_calls") or []:
                    assert "index" in tc
                    saw_tool_call = True
    assert saw_tool_call  # a tool-call delta was actually emitted


async def test_kairyu_auto_routes_through_orchestrator(app):
    body = _chat_body("What is the capital of France?")
    body["model"] = "kairyu-auto"
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"]


async def test_unknown_model_returns_404(app):
    body = _chat_body("hi")
    body["model"] = "ghost"
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 404
    assert "ghost" in response.json()["error"]["message"]


async def test_invalid_body_returns_422(app):
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json={"model": "kairyu-mock"})
    assert response.status_code == 422


async def test_invalid_sampling_params_return_400_not_500(app):
    # S2: out-of-range sampling values must be a client error (400), not an
    # unhandled ValueError surfacing as 500 (chat) or a mislabeled 502.
    async with _client(app) as client:
        for bad in ({"top_p": 0}, {"n": 0}, {"temperature": -1}):
            response = await client.post("/v1/chat/completions", json=_chat_body("hi", **bad))
            assert response.status_code == 400, (bad, response.status_code)
            assert response.json()["error"]["type"] == "invalid_request_error"


async def test_completions_invalid_sampling_returns_400(app):
    async with _client(app) as client:
        response = await client.post(
            "/v1/completions", json={"model": "kairyu-mock", "prompt": "hi", "top_p": 0}
        )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


class StubBackend:
    """Fixed completions / optional failure, for finish_reason and error tests."""

    def __init__(self, text="stub answer", finish_reason="length", error=None):
        self._text = text
        self._finish_reason = finish_reason
        self._error = error

    async def generate(self, request):
        from kairyu.engine.backend import GenerationResult
        from kairyu.outputs import CompletionOutput

        if self._error is not None:
            raise self._error
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0, text=self._text, token_ids=(), finish_reason=self._finish_reason
                ),
            ),
        )

    async def stream(self, request):
        yield await self.generate(request)

    async def shutdown(self):
        return None


async def test_streaming_with_n_gt_1_emits_all_choice_indexes(app):
    async with _client(app) as client:
        full = (
            await client.post("/v1/chat/completions", json=_chat_body("hello", n=2))
        ).json()
        response = await client.post(
            "/v1/chat/completions", json=_chat_body("hello", n=2, stream=True)
        )
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    texts: dict[int, str] = {}
    for line in lines[:-1]:
        chunk = json.loads(line[len("data: "):])
        for choice in chunk["choices"]:
            texts[choice["index"]] = texts.get(choice["index"], "") + (
                choice["delta"].get("content") or ""
            )
    expected = {c["index"]: c["message"]["content"] for c in full["choices"]}
    assert texts == expected
    assert set(texts) == {0, 1}


async def test_streaming_with_tools_signals_tool_calls(app):
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_body("what is the weather?", tools=tools, stream=True),
        )
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    chunks = [json.loads(line[len("data: "):]) for line in lines[:-1]]
    finish_reasons = [
        choice["finish_reason"] for chunk in chunks for choice in chunk["choices"]
    ]
    assert "tool_calls" in finish_reasons
    tool_deltas = [
        choice["delta"].get("tool_calls")
        for chunk in chunks
        for choice in chunk["choices"]
        if choice["delta"].get("tool_calls")
    ]
    assert tool_deltas[0][0]["function"]["name"] == "get_weather"


async def test_backend_finish_reason_passes_through():
    app = create_app(engines={"stub": StubBackend(finish_reason="length")})
    async with _client(app) as client:
        body = _chat_body("hi")
        body["model"] = "stub"
        response = await client.post("/v1/chat/completions", json=body)
    assert response.json()["choices"][0]["finish_reason"] == "length"


async def test_backend_error_returns_openai_error_envelope():
    app = create_app(
        engines={"stub": StubBackend(error=RuntimeError("secret://host:5432 key missing"))}
    )
    async with _client(app) as client:
        body = _chat_body("hi")
        body["model"] = "stub"
        response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 502
    error = response.json()["error"]
    assert error["type"] == "upstream_error"
    # M3: the backend's raw message (which may carry secrets/hosts) must NOT leak
    assert "secret://host:5432" not in error["message"]
    assert "RuntimeError" in error["message"]  # only the class name is disclosed


async def test_response_format_passes_through_to_engine():
    class CapturingBackend(StubBackend):
        def __init__(self):
            super().__init__()
            self.seen = None

        async def generate(self, request):
            self.seen = request
            return await super().generate(request)

    engine = CapturingBackend()
    app = create_app(engines={"stub": engine})
    body = _chat_body("give me json")
    body["model"] = "stub"
    body["response_format"] = {"type": "json_object"}
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert engine.seen.sampling_params.extra_args["response_format"] == {
        "type": "json_object"
    }
