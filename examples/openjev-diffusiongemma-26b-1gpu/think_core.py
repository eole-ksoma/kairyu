"""Think-first chat for OpenJev's vLLM backend: a 512-token thought, then the answer.

DiffusionGemma's vLLM sampler does not apply ``thinking_token_budget``, so the
budget is enforced the way OpenJev's System One ``think`` enforces it. One pass
writes the thought, stopping at the channel close and cut at the budget. A
second pass continues after the closed thought. Both passes use the chat
endpoint with an assistant prefill, so images, tools and vLLM's reasoning and
tool parsers keep working.

This module imports nothing from OpenJev, so it can be tested without it.
``think_first`` wires it into OpenJev's chat route.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import AsyncIterator, Sequence

import httpx

THOUGHT_TOKENS = 512
THOUGHT_OPEN = "<|channel>thought\n"
THOUGHT_CLOSE = "<channel|>"
CHAT_PATH = "/v1/chat/completions"
# The caller's controls over the visible answer. The thought pass leaves them out.
ANSWER_ONLY = frozenset({"stop", "logprobs", "top_logprobs", "tool_choice"})
# Fields of an answer-pass delta that never reach the caller. The published
# thought is the capped first pass only; a thought the model opens again in the
# answer pass would exceed the budget.
NOT_ANSWER = frozenset({"role", "reasoning", "reasoning_content"})
ERROR_TEXT_LIMIT = 500


class UpstreamError(Exception):
    """vLLM refused or failed a pass. ``status`` is the HTTP status it returned."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _continued(upstream: dict, prefill: str) -> dict:
    """The request continuing an assistant prefill, with thinking on."""

    return {
        **upstream,
        "messages": [*upstream["messages"], {"role": "assistant", "content": prefill}],
        "continue_final_message": True,
        "add_generation_prompt": False,
        "chat_template_kwargs": {
            **(upstream.get("chat_template_kwargs") or {}),
            "enable_thinking": True,
        },
    }


def thought_request(upstream: dict, close_ids: Sequence[int]) -> dict:
    """The thought pass: at most THOUGHT_TOKENS tokens, stopped at the channel close."""

    base = {key: value for key, value in upstream.items() if key not in ANSWER_ONLY}
    if base.get("tools"):
        # The tools stay in the prompt so both passes share its prefix. Without
        # an explicit "none", vLLM defaults to "auto" and parses calls out of
        # the thought text.
        base["tool_choice"] = "none"
    return {
        **_continued(base, THOUGHT_OPEN),
        "max_tokens": THOUGHT_TOKENS,
        "stop_token_ids": list(close_ids),
    }


def answer_request(upstream: dict, thought: str) -> dict:
    """The answer pass: the caller's request, continued after the closed thought."""

    return _continued(upstream, f"{THOUGHT_OPEN}{thought}\n{THOUGHT_CLOSE}")


def merge_usage(thought: dict, answer: dict) -> dict:
    """The prompt counted once, the thought as reasoning tokens, both outputs as completion."""

    prompt = int(thought.get("prompt_tokens") or 0)
    thought_tokens = int(thought.get("completion_tokens") or 0)
    completion = thought_tokens + int(answer.get("completion_tokens") or 0)
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "completion_tokens_details": {"reasoning_tokens": thought_tokens},
    }
    if thought.get("prompt_tokens_details"):
        usage["prompt_tokens_details"] = thought["prompt_tokens_details"]
    return usage


def _text(part: dict) -> str:
    """A thought pass's text, whichever field vLLM's reasoning parser put it in."""

    return (part.get("reasoning") or part.get("reasoning_content") or "") + (
        part.get("content") or ""
    )


def _message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:ERROR_TEXT_LIMIT]
    error = data.get("error") if isinstance(data, dict) else None
    text = error.get("message") if isinstance(error, dict) else None
    return str(text or data)[:ERROR_TEXT_LIMIT]


def _first_choice(body: dict, name: str) -> dict:
    choices = body.get("choices") or []
    if not choices:
        raise UpstreamError(502, f"the {name} pass returned no choices")
    return choices[0]


async def _post(client: httpx.AsyncClient, body: dict) -> dict:
    response = await client.post(CHAT_PATH, json=body)
    if response.status_code >= 400:
        raise UpstreamError(response.status_code, _message(response))
    return response.json()


def _unstreamed(upstream: dict) -> dict:
    return {
        key: value for key, value in upstream.items() if key not in ("stream", "stream_options")
    }


async def complete(
    client: httpx.AsyncClient, upstream: dict, close_ids: Sequence[int], *, model: str
) -> dict:
    """One non-streamed chat completion: the thought pass, then the answer pass."""

    request = _unstreamed(upstream)
    first = await _post(client, thought_request(request, close_ids))
    thought = _text(_first_choice(first, "thought").get("message") or {}).strip("\n")
    second = await _post(client, answer_request(request, thought))
    choice = dict(_first_choice(second, "answer"))
    message = {
        key: value
        for key, value in (choice.get("message") or {}).items()
        if key not in ("reasoning", "reasoning_content")
    }
    choice["message"] = {**message, "reasoning": thought}
    return {
        **second,
        "model": model,
        "choices": [choice],
        "usage": merge_usage(first.get("usage") or {}, second.get("usage") or {}),
    }


def _streaming(body: dict) -> dict:
    return {
        **body,
        "stream": True,
        "stream_options": {**(body.get("stream_options") or {}), "include_usage": True},
    }


async def _open(client: httpx.AsyncClient, body: dict) -> httpx.Response:
    response = await client.send(client.build_request("POST", CHAT_PATH, json=body), stream=True)
    if response.status_code < 400:
        return response
    try:
        await response.aread()
        raise UpstreamError(response.status_code, _message(response))
    finally:
        await response.aclose()


async def _payloads(response: httpx.Response) -> AsyncIterator[dict]:
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        payload = json.loads(data)
        if "error" in payload:
            error = payload["error"]
            text = error.get("message") if isinstance(error, dict) else error
            raise UpstreamError(500, str(text)[:ERROR_TEXT_LIMIT])
        yield payload


def _answer_choice(choice: dict) -> dict | None:
    delta = {
        key: value
        for key, value in (choice.get("delta") or {}).items()
        if key not in NOT_ANSWER and value not in (None, "")
    }
    if not delta and choice.get("finish_reason") is None and choice.get("logprobs") is None:
        return None
    return {**choice, "delta": delta}


async def stream(
    client: httpx.AsyncClient, upstream: dict, close_ids: Sequence[int], *, model: str
) -> ThinkStream:
    """Open the thought pass and return the SSE events of the whole completion.

    A refused thought raises UpstreamError here, before any byte is sent, so the
    caller can still answer with an HTTP error. Once events flow, a failure is
    raised out of the iterator rather than ending the stream. Kairyu reads an
    aborted stream as a failed request, but would read a final chunk as a
    complete answer.
    """

    first = await _open(client, _streaming(thought_request(upstream, close_ids)))
    return ThinkStream(client, upstream, first, model)


class ThinkStream:
    """The SSE events of one think-first completion, and the vLLM responses it holds.

    ``aclose`` closes those responses even when iteration never started, e.g.
    when the client left before the response body was sent. Otherwise vLLM
    would keep writing the thought for nobody.
    """

    def __init__(
        self, client: httpx.AsyncClient, upstream: dict, first: httpx.Response, model: str
    ) -> None:
        self._client = client
        self._upstream = upstream
        self._model = model
        self._responses = [first]
        self._events = self._generate()

    def __aiter__(self) -> AsyncIterator[str]:
        return self._events

    async def aclose(self) -> None:
        await self._events.aclose()
        for response in self._responses:
            await response.aclose()

    async def _generate(self) -> AsyncIterator[str]:
        completion_id, created = f"chatcmpl-{secrets.token_hex(12)}", int(time.time())

        def event(choices: list[dict], **extra: object) -> str:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self._model,
                "choices": choices,
                **extra,
            }
            return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        def delta(fields: dict) -> list[dict]:
            return [{"index": 0, "delta": fields, "logprobs": None, "finish_reason": None}]

        try:
            yield event(delta({"role": "assistant", "content": ""}))
            parts: list[str] = []
            thought_usage: dict = {}
            async for payload in _payloads(self._responses[0]):
                thought_usage = payload.get("usage") or thought_usage
                for choice in payload.get("choices") or ():
                    text = _text(choice.get("delta") or {})
                    text = text if parts else text.lstrip("\n")
                    if text:
                        parts.append(text)
                        yield event(delta({"reasoning": text}))
            thought = "".join(parts).rstrip("\n")
            second = await _open(self._client, _streaming(answer_request(self._upstream, thought)))
            self._responses.append(second)
            answer_usage: dict = {}
            async for payload in _payloads(second):
                answer_usage = payload.get("usage") or answer_usage
                choices = [_answer_choice(choice) for choice in payload.get("choices") or ()]
                kept = [choice for choice in choices if choice is not None]
                if kept:
                    yield event(kept)
            yield event([], usage=merge_usage(thought_usage, answer_usage))
            yield "data: [DONE]\n\n"
        finally:
            for response in self._responses:
                await response.aclose()
