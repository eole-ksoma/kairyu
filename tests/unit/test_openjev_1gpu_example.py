"""Contracts of the one-GPU OpenJev DiffusionGemma example (think = 512)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import httpx
import pytest

from kairyu import SamplingParams
from kairyu.deploy.spec import load_deployment_spec
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.config_validation import validate_backend_options
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.sampling_params import GENERATION_CONFIG_SAMPLING_FIELDS

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/openjev-diffusiongemma-26b-1gpu"
CLOSE_ID = 212  # stands in for the checkpoint's single <channel|> token id
THOUGHT = "17 times 19 is 323."
QUESTION = [{"role": "user", "content": "What is 17 * 19? Reply with only the integer."}]


@pytest.fixture
def example(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    for name in ("control", "verification", "benchmark", "think_core", "patch_openjev"):
        sys.modules.pop(name, None)
    return importlib.import_module


def _usage(prompt: int, completion: int) -> dict:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": None,
    }


class FakeVllm:
    """vLLM's chat endpoint behind OpenJev, answering the thought and answer passes.

    The thought pass is the request whose final assistant prefill is still open;
    the answer pass carries the closed thought. Responses mirror vLLM's shapes.
    """

    def __init__(
        self,
        *,
        thought: str = THOUGHT,
        thought_field: str = "content",
        thought_finish: str = "stop",
        thought_tokens: int = 9,
        answer_message: dict | None = None,
        answer_finish: str = "stop",
        answer_status: int = 200,
        thought_status: int = 200,
    ) -> None:
        self.thought = thought
        self.thought_field = thought_field
        self.thought_finish = thought_finish
        self.thought_tokens = thought_tokens
        self.answer_message = answer_message or {"content": "323"}
        self.answer_finish = answer_finish
        self.answer_status = answer_status
        self.thought_status = thought_status
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        is_answer = "<channel|>" in body["messages"][-1]["content"]
        status = self.answer_status if is_answer else self.thought_status
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "upstream refused"}})
        if is_answer:
            message, finish, usage = self.answer_message, self.answer_finish, _usage(31, 2)
        else:
            message = {self.thought_field: "\n" + self.thought}
            finish, usage = self.thought_finish, _usage(20, self.thought_tokens)
        if body.get("stream"):
            return self._stream(message, finish, usage)
        full = {"role": "assistant", "content": None, "reasoning": None, "tool_calls": []}
        full.update(message)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-upstream",
                "object": "chat.completion",
                "created": 1,
                "model": "dgemma",
                "choices": [
                    {"index": 0, "message": full, "logprobs": None, "finish_reason": finish}
                ],
                "usage": usage,
            },
        )

    @staticmethod
    def _stream(message: dict, finish: str, usage: dict) -> httpx.Response:
        def chunk(delta: dict, finish_reason: str | None = None) -> dict:
            return {
                "id": "chatcmpl-upstream",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "dgemma",
                "choices": [
                    {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}
                ],
            }

        halves = [
            chunk({field: part})
            for field, text in message.items()
            for part in (text[: len(text) // 2], text[len(text) // 2 :])
        ]
        events = [
            chunk({"role": "assistant", "content": ""}),
            *halves,
            chunk({}, finish),
            {
                "id": "chatcmpl-upstream",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "dgemma",
                "choices": [],
                "usage": usage,
            },
        ]
        # Raw UTF-8 like vLLM's model_dump_json: U+2028 and friends stay unescaped.
        payload = "".join(f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events)
        return httpx.Response(
            200,
            content=(payload + "data: [DONE]\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        )


def _client(fake: FakeVllm) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(fake), base_url="http://vllm")


def _upstream(**overrides) -> dict:
    """An OpenJev-normalized chat request whose caller turned thinking off."""

    request = {
        "model": "dgemma",
        "messages": QUESTION,
        "max_tokens": 64,
        "stop": ["\n\n"],
        "logprobs": True,
        "top_logprobs": 2,
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
        "tool_choice": "auto",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request.update(overrides)
    return request


def _events(lines: list[str]) -> list[dict]:
    return [json.loads(line[6:]) for line in lines if line.startswith("data: {")]


@pytest.mark.parametrize("thought_field", ["content", "reasoning"])
async def test_every_answer_follows_a_capped_thought_the_caller_cannot_disable(
    example, thought_field
):
    think_core = example("think_core")
    # Kairyu reads reasoning_content before reasoning, so a thought the model
    # reopens in the answer pass would be shown in place of the capped one.
    fake = FakeVllm(
        thought_field=thought_field,
        answer_message={"content": "323", "reasoning_content": "a second, unbudgeted thought"},
    )
    async with _client(fake) as client:
        body = await think_core.complete(
            client, _upstream(), [CLOSE_ID], model="diffusiongemma-26b"
        )

    thought, answer = fake.requests
    assert thought["messages"] == [
        *QUESTION,
        {"role": "assistant", "content": "<|channel>thought\n"},
    ]
    assert thought["continue_final_message"] is True
    assert thought["add_generation_prompt"] is False
    assert thought["chat_template_kwargs"] == {"enable_thinking": True}
    assert thought["max_tokens"] == 512
    assert thought["stop_token_ids"] == [CLOSE_ID]
    assert {"stop", "logprobs", "top_logprobs"}.isdisjoint(thought)
    # Tools stay in the prompt (same prefix as the answer pass), but vLLM would
    # default to tool_choice "auto" and parse calls out of the thought text.
    assert thought["tools"] == _upstream()["tools"]
    assert thought["tool_choice"] == "none"
    assert answer["messages"][-1] == {
        "role": "assistant",
        "content": "<|channel>thought\n17 times 19 is 323.\n<channel|>",
    }
    assert answer["continue_final_message"] is True
    assert answer["add_generation_prompt"] is False
    assert answer["chat_template_kwargs"] == {"enable_thinking": True}
    assert (answer["max_tokens"], answer["stop"], answer["logprobs"]) == (64, ["\n\n"], True)
    assert answer["tool_choice"] == "auto"

    (choice,) = body["choices"]
    assert body["model"] == "diffusiongemma-26b"
    assert choice["message"]["reasoning"] == THOUGHT
    assert "reasoning_content" not in choice["message"]
    assert choice["message"]["content"] == "323"
    assert choice["finish_reason"] == "stop"
    assert body["usage"] == {
        "prompt_tokens": 20,
        "completion_tokens": 11,
        "total_tokens": 31,
        "completion_tokens_details": {"reasoning_tokens": 9},
    }


async def test_a_thought_cut_at_the_budget_still_gets_an_answer(example):
    think_core = example("think_core")
    fake = FakeVllm(thought_finish="length", thought_tokens=512)
    async with _client(fake) as client:
        body = await think_core.complete(
            client, _upstream(), [CLOSE_ID], model="diffusiongemma-26b"
        )

    assert len(fake.requests) == 2
    assert body["choices"][0]["message"]["content"] == "323"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens_details"] == {"reasoning_tokens": 512}


async def test_tool_calls_from_the_answer_pass_reach_the_caller_with_the_thought(example):
    think_core = example("think_core")
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
    }
    fake = FakeVllm(answer_message={"tool_calls": [call]}, answer_finish="tool_calls")
    async with _client(fake) as client:
        body = await think_core.complete(
            client, _upstream(), [CLOSE_ID], model="diffusiongemma-26b"
        )

    (choice,) = body["choices"]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == [call]
    assert choice["message"]["reasoning"] == THOUGHT


async def test_stream_sends_the_thought_then_the_answer_under_one_completion(example):
    think_core = example("think_core")
    fake = FakeVllm(
        answer_message={"reasoning_content": "a second, unbudgeted thought", "content": "323"}
    )
    async with _client(fake) as client:
        events = await think_core.stream(
            client, _upstream(stream=True), [CLOSE_ID], model="diffusiongemma-26b"
        )
        lines = [line async for line in events]

    thought, answer = fake.requests
    assert thought["stream"] is True and thought["stream_options"] == {"include_usage": True}
    assert answer["messages"][-1]["content"] == "<|channel>thought\n17 times 19 is 323.\n<channel|>"
    assert lines[-1] == "data: [DONE]\n\n"
    chunks = _events(lines)
    assert {chunk["id"] for chunk in chunks} == {chunks[0]["id"]}
    assert {chunk["model"] for chunk in chunks} == {"diffusiongemma-26b"}
    deltas = [choice["delta"] for chunk in chunks for choice in chunk["choices"]]
    assert deltas[0]["role"] == "assistant"
    # What Kairyu shows as reasoning: it reads either field of a delta.
    reasoning = "".join(
        (delta.get("reasoning") or "") + (delta.get("reasoning_content") or "") for delta in deltas
    )
    content = "".join(delta.get("content") or "" for delta in deltas)
    assert (reasoning, content) == (THOUGHT, "323")
    last_reasoning = max(i for i, delta in enumerate(deltas) if delta.get("reasoning"))
    first_content = min(i for i, delta in enumerate(deltas) if delta.get("content"))
    assert last_reasoning < first_content
    finishes = [
        c["finish_reason"] for chunk in chunks for c in chunk["choices"] if c["finish_reason"]
    ]
    assert finishes == ["stop"]
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["completion_tokens"] == 11
    assert chunks[-1]["usage"]["completion_tokens_details"] == {"reasoning_tokens": 9}


async def test_stream_survives_unicode_line_separators_in_deltas(example):
    """vLLM leaves U+2028, U+2029 and U+0085 unescaped; a str.splitlines-based reader
    cuts the JSON there, aborts the stream and gets the only replica ejected."""

    think_core = example("think_core")
    thought, answer = "first\u2028second\u0085third", "3\u20292\u20283"
    fake = FakeVllm(thought=thought, answer_message={"content": answer})
    async with _client(fake) as client:
        events = await think_core.stream(
            client, _upstream(stream=True), [CLOSE_ID], model="diffusiongemma-26b"
        )
        lines = [line async for line in events]

    deltas = [c["delta"] for e in _events(lines) for c in e["choices"]]
    assert "".join(d.get("reasoning") or "" for d in deltas) == thought
    assert "".join(d.get("content") or "" for d in deltas) == answer
    assert lines[-1] == "data: [DONE]\n\n"


async def test_a_refused_thought_is_an_error_before_the_stream_starts(example):
    think_core = example("think_core")
    fake = FakeVllm(thought_status=400)
    async with _client(fake) as client:
        with pytest.raises(think_core.UpstreamError) as refused:
            await think_core.stream(
                client, _upstream(stream=True), [CLOSE_ID], model="diffusiongemma-26b"
            )

    assert refused.value.status == 400
    assert len(fake.requests) == 1


@pytest.mark.parametrize(
    "tool_choice", ["required", {"type": "function", "function": {"name": "bash"}}]
)
async def test_a_forced_tool_choice_is_refused_before_any_thought(example, tool_choice):
    """Diffusion models cannot run the structured outputs a forced tool call needs.
    Refused only in the answer pass, a streamed request could just be aborted,
    which Kairyu counts as a failure of the only replica."""

    think_core = example("think_core")
    fake = FakeVllm()
    async with _client(fake) as client:
        with pytest.raises(think_core.UpstreamError) as streamed:
            await think_core.stream(
                client,
                _upstream(stream=True, tool_choice=tool_choice),
                [CLOSE_ID],
                model="diffusiongemma-26b",
            )
        with pytest.raises(think_core.UpstreamError) as unstreamed:
            await think_core.complete(
                client, _upstream(tool_choice=tool_choice), [CLOSE_ID], model="diffusiongemma-26b"
            )

    assert (streamed.value.status, unstreamed.value.status) == (400, 400)
    assert fake.requests == []


async def test_an_answer_failure_after_the_thought_aborts_without_done(example):
    """Kairyu ignores mid-stream error chunks; only an aborted stream surfaces the failure."""

    think_core = example("think_core")
    fake = FakeVllm(answer_status=500)
    lines: list[str] = []
    async with _client(fake) as client:
        events = await think_core.stream(
            client, _upstream(stream=True), [CLOSE_ID], model="diffusiongemma-26b"
        )
        with pytest.raises(think_core.UpstreamError):
            async for line in events:
                lines.append(line)

    assert "".join(c["delta"].get("reasoning") or "" for e in _events(lines) for c in e["choices"])
    assert "data: [DONE]\n\n" not in lines


class RecordingStream(httpx.AsyncByteStream):
    """A response body that records whether the client closed it."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.closed = False

    async def __aiter__(self):
        yield self.payload

    async def aclose(self) -> None:
        self.closed = True


async def test_an_unread_stream_still_closes_the_thought_request(example):
    """A client gone before the body starts must not leave vLLM writing the thought."""

    think_core = example("think_core")
    sse = FakeVllm._stream({"reasoning": THOUGHT}, "stop", _usage(20, 9)).content
    recorder = RecordingStream(sse)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, stream=recorder, headers={"content-type": "text/event-stream"}
        )
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://vllm") as client:
        events = await think_core.stream(
            client, _upstream(stream=True), [CLOSE_ID], model="diffusiongemma-26b"
        )
        await events.aclose()

    assert recorder.closed


def _fake_openjev(request: httpx.Request) -> httpx.Response:
    """OpenJev's route table: /health, and chat for its generation model only."""

    if request.method == "GET" and request.url.path == "/health":
        return httpx.Response(200, json={"status": "ok"})
    if request.method == "POST" and request.url.path == "/v1/chat/completions":
        body = json.loads(request.content)
        if body.get("model") not in {"diffusiongemma-26b", "diffusiongemma"}:
            return httpx.Response(
                404,
                json={
                    "error": {
                        "message": "not found",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
            )
        message = {"role": "assistant", "content": "323", "reasoning": THOUGHT}
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-openjev",
                "object": "chat.completion",
                "created": 1,
                "model": "diffusiongemma-26b",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": _usage(20, 11),
            },
        )
    return httpx.Response(404, json={"detail": "Not Found"})


async def test_served_configuration_reaches_openjev_and_returns_its_thought(example):
    """A wrong model or path 404s on every request while /readyz stays 200."""

    spec = json.loads((EXAMPLE / "example.json").read_text())
    deployment = load_deployment_spec((EXAMPLE / "kairyu.yaml").read_text())
    (replica,) = deployment.pools["diffusiongemma-26b"].replicas
    validate_backend_options(replica.backend, replica.options)
    example("control")  # import-time check: OpenJev in-flight + queue == pool capacity
    # Above OpenJev's capacity a 529 ejects the only replica; Kairyu must answer 429 first.
    assert deployment.server.max_concurrency == spec["pool"]["max_concurrency"]
    assert replica.options.get("container_image_digest") == spec["openjev"]["image_id"]

    transport = httpx.MockTransport(_fake_openjev)
    async with httpx.AsyncClient(transport=transport) as probe:
        assert (await probe.get(replica.resolved_health_url())).status_code == 200
    backend = OpenAICompatBackend(**replica.options, transport=transport)
    sampling = SamplingParams(max_tokens=64).with_generation_config_omitted(
        GENERATION_CONFIG_SAMPLING_FIELDS
    )
    result = await backend.generate(
        GenerationRequest(request_id="r1", prompt="user: hi\nassistant:", sampling_params=sampling)
    )

    (completion,) = result.completions
    assert (completion.text, completion.reasoning_content) == ("323", THOUGHT)


@pytest.mark.parametrize(
    "reasoning,content,finish,error",
    [
        (THOUGHT, "323", "stop", None),
        (None, "323", "stop", "reasoning"),
        (THOUGHT, "  ", "stop", "empty answer"),
        (THOUGHT, "324", "stop", "expected '323'"),
        (THOUGHT, "323", "length", "finish_reason"),
    ],
)
def test_answer_validator_rejects_false_passes(example, reasoning, content, finish, error):
    control = example("control")
    body = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                },
                "finish_reason": finish,
            }
        ]
    }

    problem = control.think_answer_error(body, expected="323")

    if error is None:
        assert problem is None
    else:
        assert problem is not None and error in problem
