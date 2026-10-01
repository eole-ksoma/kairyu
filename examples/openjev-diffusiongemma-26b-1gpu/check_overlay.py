#!/usr/bin/env python3
"""CPU evidence for this example's think-512 overlay (no GPU, no Docker).

1. Template: with the checkpoint's own tokenizer, the edited template renders
   every conversation that does not end in an assistant message byte for byte
   like the stock template, and continues both thought prefills.
2. Overlay: the pinned OpenJev source, patched by patch_openjev.py, serves
   think-first chat against a local fake vLLM. The checks cover forced
   thinking, stream order, slot release on success, refusal and abort, and the
   startup refusal without the template.

Downloads the checkpoint's tokenizer files and the OpenJev source at the
revisions pinned in example.json. Run from the repository root:

  uv run --no-project --with fastapi --with "uvicorn[standard]" --with httpx \
    --with "transformers>=5.8,<5.13" --with jinja2 \
    python examples/openjev-diffusiongemma-26b-1gpu/check_overlay.py
"""

from __future__ import annotations

import io
import itertools
import json
import os
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "example.json").read_text(encoding="utf-8"))
OPEN, CLOSE = "<|channel>thought\n", "<channel|>"
THOUGHT = "17 times 19 is 323."
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
BASH = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}
CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "bash", "arguments": {"command": "ls"}},
}
UNCHANGED = {
    "user": ([{"role": "user", "content": "What is 17 * 19?"}], None),
    "system+user": (
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        None,
    ),
    "tools": ([{"role": "user", "content": "List files."}], [BASH]),
    "image": (
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Color?"}]}],
        None,
    ),
    "multi-turn with a stripped thought": (
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": f"{OPEN}greet{CLOSE}Hello!"},
            {"role": "user", "content": "Again"},
        ],
        None,
    ),
    "tool result": (
        [
            {"role": "user", "content": "List files."},
            {"role": "assistant", "content": "", "tool_calls": [CALL], "reasoning": "use ls"},
            {"role": "tool", "tool_call_id": "call_1", "content": "a.txt"},
        ],
        [BASH],
    ),
    "kairyu legacy transcript": (
        [{"role": "user", "content": "user: hi\nassistant: hello\nuser: 2+2?\nassistant:"}],
        None,
    ),
}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def download_tokenizer(target: Path) -> Path:
    model = SPEC["model"]
    base = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}"
    target.mkdir(parents=True)
    for name in TOKENIZER_FILES:
        (target / name).write_bytes(fetch(f"{base}/{name}"))
    return target


def check_template(tokenizer_dir: Path) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    stock = (tokenizer_dir / "chat_template.jinja").read_text(encoding="utf-8")
    edited = (HERE / "chat_template.jinja").read_text(encoding="utf-8")

    def render(template: str, messages: list, tools: list | None, **kwargs) -> str:
        return tokenizer.apply_chat_template(
            messages, tools=tools, chat_template=template, tokenize=False, **kwargs
        )

    unchanged = {}
    for name, (messages, tools) in UNCHANGED.items():
        for thinking in (True, False):
            kwargs = {"add_generation_prompt": True, "enable_thinking": thinking}
            same = render(stock, messages, tools, **kwargs) == render(
                edited, messages, tools, **kwargs
            )
            unchanged[f"{name} / thinking={thinking}"] = same
    prefills = {}
    question = [{"role": "user", "content": "What is 17 * 19?"}]
    kwargs = {
        "add_generation_prompt": False,
        "continue_final_message": True,
        "enable_thinking": True,
    }
    for name, prefill in (("thought", OPEN), ("answer", f"{OPEN}{THOUGHT}\n{CLOSE}")):
        # vLLM's 'openai' content format sends the prefill as a list of text parts.
        for tools, shape in itertools.product((None, [BASH]), ("string", "parts")):
            content = prefill if shape == "string" else [{"type": "text", "text": prefill}]
            messages = [*question, {"role": "assistant", "content": content}]
            try:
                render(stock, messages, tools, **kwargs)
                stock_outcome = "continued"
            except ValueError as error:
                stock_outcome = f"refused: {str(error)[:60]}"
            rendered = render(edited, messages, tools, **kwargs)
            prefills[f"{name} pass / tools={bool(tools)} / {shape}"] = {
                "stock": stock_outcome,
                "edited_ends_at_prefill": rendered.rstrip().endswith(prefill.strip()),
            }
    passed = all(unchanged.values()) and all(p["edited_ends_at_prefill"] for p in prefills.values())
    return {"unchanged": unchanged, "prefills": prefills, "passed": passed}


def patched_openjev(target: Path) -> Path:
    source = SPEC["openjev"]
    repository = source["source_repository"].removeprefix("https://github.com/")
    archive = fetch(f"https://codeload.github.com/{repository}/tar.gz/{source['source_revision']}")
    target.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(target, filter="data")
    (root,) = target.iterdir()
    subprocess.run(
        [
            sys.executable,
            str(HERE / "patch_openjev.py"),
            "--version",
            source["source_version"],
            "--package",
            str(root / "openjev"),
        ],
        check=True,
        stdout=sys.stderr,  # stdout carries only the JSON report
    )
    return root


class FakeVllm:
    """vLLM's chat endpoint: the thought pass answers in ``reasoning``, the answer in content."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.thought_status = self.answer_status = 200

    def app(self):
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse, Response
        from starlette.routing import Route

        async def chat(request):
            body = await request.json()
            self.requests.append(body)
            answer = CLOSE in body["messages"][-1]["content"]
            status = self.answer_status if answer else self.thought_status
            if status != 200:
                return JSONResponse({"error": {"message": "fake refusal"}}, status_code=status)
            field, text = ("content", "323") if answer else ("reasoning", "\n" + THOUGHT)
            usage = {"prompt_tokens": 31 if answer else 20, "completion_tokens": 2 if answer else 9}
            if not body.get("stream"):
                message = {"role": "assistant", "content": None, "reasoning": None, field: text}
                choice = {"index": 0, "message": message, "finish_reason": "stop"}
                return JSONResponse({"model": "dgemma", "choices": [choice], "usage": usage})
            chunks = [
                {"choices": [{"index": 0, "delta": {field: text}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": usage},
            ]
            sse = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            return Response(sse + "data: [DONE]\n\n", media_type="text/event-stream")

        return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


def serve(app) -> str:
    import uvicorn

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}"


def check_overlay(source: Path, tokenizer_dir: Path) -> dict:
    fake = FakeVllm()
    template = str(HERE / "chat_template.jinja")
    os.environ.update(
        OPENJEV_UPSTREAM=serve(fake.app()),
        OPENJEV_TOKENIZER=str(tokenizer_dir),
        OPENJEV_WARMUP="0",
        KAIRYU_THINK_CHAT_TEMPLATE=template,
        OPENJEV_VLLM_ARGS=f"--chat-template {template}",
    )
    sys.path.insert(0, str(source))
    from fastapi.testclient import TestClient
    from openjev.api import create_app

    request = {
        "model": "diffusiongemma-26b",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "What is 17 * 19?"}],
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
    }
    report: dict = {}
    app = create_app()
    with TestClient(app) as client:
        generator = app.state.generator
        capacity = generator.slots._value

        def released() -> bool:
            return (generator.running, generator.slots._value) == (0, capacity)

        body = client.post("/v1/chat/completions", json=request).json()
        thought, answer = fake.requests[-2:]
        report["non_stream"] = {
            "generator": type(generator).__name__,
            "thought_pass": [
                thought["max_tokens"],
                thought["chat_template_kwargs"],
                thought["stop_token_ids"],
            ],
            "answer_prefill": answer["messages"][-1]["content"],
            "reasoning": body["choices"][0]["message"]["reasoning"],
            "content": body["choices"][0]["message"]["content"],
            "passed": body["choices"][0]["message"]["reasoning"] == THOUGHT
            and thought["max_tokens"] == 512
            and thought["chat_template_kwargs"]["enable_thinking"] is True,
        }
        with client.stream(
            "POST", "/v1/chat/completions", json={**request, "stream": True}
        ) as reply:
            lines = [line for line in reply.iter_lines() if line]
        deltas = [
            choice["delta"]
            for line in lines
            if line.startswith("data: {")
            for choice in json.loads(line[6:])["choices"]
        ]
        reasoning = "".join(d.get("reasoning") or "" for d in deltas)
        content = "".join(d.get("content") or "" for d in deltas)
        report["stream"] = {
            "passed": (reasoning, content) == (THOUGHT, "323")
            and lines[-1] == "data: [DONE]"
            and released(),
        }
        fake.thought_status = 400
        refused = client.post("/v1/chat/completions", json={**request, "stream": True})
        report["refused_thought"] = {"passed": refused.status_code == 400 and released()}
        fake.thought_status, fake.answer_status = 200, 500
        try:
            with client.stream(
                "POST", "/v1/chat/completions", json={**request, "stream": True}
            ) as reply:
                done = "data: [DONE]" in [line for line in reply.iter_lines() if line]
        except Exception:  # noqa: BLE001 - the abort surfaces as the server's error
            done = False
        report["answer_failure_aborts"] = {"passed": not done and released()}
    os.environ["OPENJEV_VLLM_ARGS"] = ""
    try:
        with TestClient(create_app()):
            refused_startup = False
    except RuntimeError:
        refused_startup = True
    report["startup_without_template"] = {"passed": refused_startup}
    return report


def main() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        tokenizer_dir = download_tokenizer(Path(scratch) / "tokenizer")
        report = {"template": check_template(tokenizer_dir)}
        report["overlay"] = check_overlay(patched_openjev(Path(scratch) / "openjev"), tokenizer_dir)
    print(json.dumps(report, indent=2))
    checks = [report["template"], *report["overlay"].values()]
    raise SystemExit(0 if all(check["passed"] for check in checks) else 1)


if __name__ == "__main__":
    main()
