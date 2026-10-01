"""Think-first chat on OpenJev's vLLM backend: this example's overlay.

patch_openjev.py copies this module and think_core.py into the installed
``openjev`` package, and makes api.py build ThinkFirstGenerator instead of
Generator. Every chat completion then writes a thought of at most
``think_core.THOUGHT_TOKENS`` tokens before its answer, whatever the caller asks.
Normalization, capacity and the response shapes stay OpenJev's.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from . import think_core
from .chat import Generator, extract_json, oai_error
from .config import GEN_MODEL

TEMPLATE_ENV = "KAIRYU_THINK_CHAT_TEMPLATE"
PROBE_TURN = {"role": "user", "content": "probe"}


def _upstream_error(error: think_core.UpstreamError):
    if error.status < 500:
        return oai_error(400, error.message, "invalid_request_error")
    return oai_error(503, error.message, "api_error")


def _unavailable(error: httpx.HTTPError):
    return oai_error(
        503,
        f"inference backend unavailable: {type(error).__name__}",
        "api_error",
        headers={"retry-after": "2"},
    )


def check_template(
    tokenizer, close_ids: Sequence[int], environ: Mapping[str, str] = os.environ
) -> None:
    """Refuse to start unless vLLM renders the thought prefills this overlay sends."""

    if len(close_ids) != 1:
        raise RuntimeError(
            f"{think_core.THOUGHT_CLOSE!r} must be one token for stop_token_ids, got {close_ids}"
        )
    path = environ.get(TEMPLATE_ENV, "")
    args = shlex.split(environ.get("OPENJEV_VLLM_ARGS", ""))
    flag = args.index("--chat-template") if "--chat-template" in args else -1
    if not path or flag < 0 or flag + 1 >= len(args) or args[flag + 1] != path:
        raise RuntimeError(
            f"vLLM must start with --chat-template ${TEMPLATE_ENV} ({path!r}); "
            f"OPENJEV_VLLM_ARGS is {environ.get('OPENJEV_VLLM_ARGS', '')!r}"
        )
    template = Path(path).read_text(encoding="utf-8")
    prefills = (
        think_core.THOUGHT_OPEN,
        f"{think_core.THOUGHT_OPEN}probe\n{think_core.THOUGHT_CLOSE}",
    )
    for prefill in prefills:
        rendered = tokenizer.apply_chat_template(
            [PROBE_TURN, {"role": "assistant", "content": prefill}],
            chat_template=template,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            enable_thinking=True,
        )
        if not rendered.rstrip().endswith(prefill.strip()):
            raise RuntimeError(
                f"the chat template does not continue the prefill {prefill!r}: "
                f"...{rendered[-120:]!r}"
            )


def _json_answer(body: dict) -> dict:
    choice = body["choices"][0]
    message = choice["message"]
    content = message.get("content")
    if not isinstance(content, str):
        return body
    answer = {**message, "content": extract_json(content)}
    return {**body, "choices": [{**choice, "message": answer}]}


class ThinkFirstGenerator(Generator):
    """OpenJev's vLLM chat generator, writing a capped thought before every answer."""

    def __init__(self, settings, engine) -> None:
        super().__init__(settings)
        self.close_ids = list(engine.thought_close)
        check_template(engine.tok, self.close_ids)

    async def complete(self, upstream, json_mode):
        self.running += 1
        try:
            async with self.slots:  # one slot for both passes
                body = await think_core.complete(
                    self.client, upstream, self.close_ids, model=GEN_MODEL
                )
        except think_core.UpstreamError as error:
            return _upstream_error(error)
        except httpx.HTTPError as error:
            return _unavailable(error)
        finally:
            self.running -= 1
        return _json_answer(body) if json_mode else body

    async def stream(self, upstream, request):
        self.running += 1
        try:
            await self.slots.acquire()
        except BaseException:
            # a client that leaves while waiting must not keep capacity it never got
            self.running -= 1
            raise
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                self.slots.release()
                self.running -= 1

        try:
            events = await think_core.stream(self.client, upstream, self.close_ids, model=GEN_MODEL)
        except think_core.UpstreamError as error:
            release()
            return _upstream_error(error)
        except httpx.HTTPError as error:
            release()
            return _unavailable(error)
        except BaseException:
            release()
            raise

        async def cleanup() -> None:
            await events.aclose()
            release()

        async def body():
            try:
                async for event in events:
                    yield event
            finally:
                await cleanup()

        # Both paths run the same idempotent cleanup. Starlette skips the
        # background task when the body raises (an aborted answer pass), and
        # the body's finally never runs when the client left before it started.
        return StreamingResponse(
            body(), media_type="text/event-stream", background=BackgroundTask(cleanup)
        )
