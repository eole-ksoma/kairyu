#!/usr/bin/env python3
"""GPU verification for one OpenJev DiffusionGemma replica on one GPU (think = 512).

``l1`` talks to OpenJev inside its container and proves the thought budget
from OpenJev's own usage, which Kairyu does not pass on. The public gates run
through Kairyu (L2 ReplicaPool, L3 API).
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import control
import think_core

HERE = control.HERE
SPEC = control.SPEC
SERVED = control.SERVED
THOUGHT_TOKENS = think_core.THOUGHT_TOKENS
SERVED_CONFIG_FILES = tuple(
    HERE / name
    for name in (
        "example.json",
        "compose.yaml",
        "kairyu.yaml",
        "openjev-think.Dockerfile",
        "patch_openjev.py",
        "think_core.py",
        "think_first.py",
        "chat_template.jinja",
    )
)
ENVIRONMENT_STORAGE = control.environment_storage()
RESULTS_ROOT = Path(
    os.environ.get("VERIFICATION_RESULTS_ROOT", ENVIRONMENT_STORAGE / "verification-results")
)
PLACEMENT_LOG = ENVIRONMENT_STORAGE / "placement-log" / Path(SPEC["pool"]["placement_log"]).name
REQUEST_LOG: Path | None = None
LEVELS: list[int] | None = None  # --concurrency overrides a gate's levels
ARITHMETIC = "What is 17 * 19? Reply with only the integer."
# Asks for far more than 512 tokens of reasoning, so the thought must be cut.
LONG_THOUGHT = (
    "Before you answer, write out in your reasoning every integer from 1 to 400 "
    "together with its square, one per line. Then answer: what is 400 squared? "
    "Reply with only the integer."
)
OPENJEV_PORT, VLLM_PORT = 8080, 8000


def _api_url() -> str:
    return f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}"


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _log(payload: dict, status: int, body: object, request_id: str | None) -> None:
    if REQUEST_LOG is not None:
        row = {"request": payload, "status": status, "response": body, "request_id": request_id}
        with REQUEST_LOG.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def post_chat(payload: dict, *, timeout_s: float = 900.0) -> tuple[int, object]:
    request = urllib.request.Request(
        f"{_api_url()}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
            status, request_id = response.status, response.headers.get("x-request-id")
            body: object = raw if payload.get("stream") else json.loads(raw)
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read().decode("utf-8", "replace")
        request_id = error.headers.get("x-request-id") if error.headers else None
    _log(payload, status, body, request_id)
    return status, body


# --- inside the OpenJev container (no host port) -------------------------------------

_L1_PROGRAM = r"""
import json, sys, urllib.error, urllib.request
port, method, path = sys.argv[1:4]
data = sys.stdin.buffer.read() if method == "POST" else None
request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"}, method=method)
try:
    with urllib.request.urlopen(request, timeout=1800) as response:
        sys.stdout.write(json.dumps({"status": response.status, "body": response.read().decode()}))
except urllib.error.HTTPError as error:
    body = error.read().decode("utf-8", "replace")
    sys.stdout.write(json.dumps({"status": error.code, "body": body}))
"""


def l1_call(port: int, method: str, path: str, payload: dict | None = None) -> tuple[int, str]:
    completed = subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            control.L1_CONTAINER,
            "python3",
            "-c",
            _L1_PROGRAM,
            str(port),
            method,
            path,
        ],
        input=json.dumps(payload).encode() if payload is not None else b"",
        capture_output=True,
        timeout=1900,
        check=True,
    )
    result = json.loads(completed.stdout)
    return int(result["status"]), result["body"]


def l1_chat(payload: dict) -> str:
    status, body = l1_call(OPENJEV_PORT, "POST", "/v1/chat/completions", payload)
    _log(payload, status, body, None)
    if status != 200:
        raise ValueError(f"OpenJev returned HTTP {status}: {body[:300]}")
    return body


def vllm_metrics() -> str:
    status, body = l1_call(VLLM_PORT, "GET", "/metrics")
    if status != 200:
        raise ValueError(f"vLLM /metrics returned HTTP {status}")
    return body


def _metric_total(metrics: str, name: str) -> float:
    values = re.findall(rf"^{re.escape(name)}(?:\{{[^\n]*\}})? ([0-9.eE+-]+)$", metrics, re.M)
    if not values:
        raise ValueError(f"missing vLLM metric {name}")
    return sum(float(value) for value in values)


def vllm_active() -> float:
    metrics = vllm_metrics()
    return sum(_metric_total(metrics, f"vllm:num_requests_{g}") for g in ("running", "waiting"))


def reasoning_tokens(body: dict) -> int | None:
    return ((body.get("usage") or {}).get("completion_tokens_details") or {}).get(
        "reasoning_tokens"
    )


def stream_events(sse: str) -> tuple[list[dict], bool]:
    lines = [line for line in sse.splitlines() if line.startswith("data: ")]
    events = [json.loads(line[6:]) for line in lines if line != "data: [DONE]"]
    return events, bool(lines) and lines[-1] == "data: [DONE]"


def stream_order_error(sse: str) -> str | None:
    """Why an L1 stream is not reasoning, then content, then usage within the budget."""

    events, done = stream_events(sse)
    deltas = [c.get("delta") or {} for e in events for c in e.get("choices") or ()]
    thought = [i for i, d in enumerate(deltas) if d.get("reasoning") or d.get("reasoning_content")]
    answer = [i for i, d in enumerate(deltas) if d.get("content")]
    usage = next((e["usage"] for e in reversed(events) if e.get("usage")), {})
    tokens = reasoning_tokens({"usage": usage})
    if not (done and thought and answer and thought[-1] < answer[0]):
        return f"expected reasoning, then content, then [DONE] (done={done})"
    if not (isinstance(tokens, int) and 1 <= tokens <= THOUGHT_TOKENS):
        return f"stream reasoning_tokens {tokens!r} outside 1..{THOUGHT_TOKENS}"
    return None


def l1(run_dir: Path) -> int:
    """OpenJev directly: thinking cannot be disabled, the thought is cut at 512, two passes."""

    base = {
        "model": SERVED,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": ARITHMETIC}],
    }
    cases: dict[str, str | None] = {}
    before = _metric_total(vllm_metrics(), "vllm:request_success_total")
    body = json.loads(
        l1_chat(
            {**base, "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}
        )
    )
    passes = _metric_total(vllm_metrics(), "vllm:request_success_total") - before
    tokens = reasoning_tokens(body)
    cases["thinking_cannot_be_disabled"] = control.think_answer_error(body, expected="323") or (
        None
        if isinstance(tokens, int) and 1 <= tokens <= THOUGHT_TOKENS
        else f"reasoning_tokens {tokens!r} outside 1..{THOUGHT_TOKENS}"
    )
    cases["two_vllm_requests_per_chat"] = None if passes == 2 else f"vLLM completed {passes}"
    long = json.loads(l1_chat({**base, "messages": [{"role": "user", "content": LONG_THOUGHT}]}))
    content = (long["choices"][0]["message"].get("content") or "").strip()
    cases["thought_cut_at_budget"] = (
        None
        if reasoning_tokens(long) == THOUGHT_TOKENS and content
        else f"reasoning_tokens {reasoning_tokens(long)!r}, content {content[:60]!r}"
    )
    cases["stream_thought_then_answer"] = stream_order_error(l1_chat({**base, "stream": True}))
    return _report(run_dir, "l1", cases)


def _report(run_dir: Path, gate: str, cases: dict[str, str | None]) -> int:
    report = {"cases": cases, "passed": not any(cases.values())}
    (run_dir / f"{gate}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{gate}: {'PASS' if report['passed'] else 'FAIL'} {cases}")
    return int(not report["passed"])


# --- public gates (through Kairyu) ----------------------------------------------------

_GENERIC_TASKS = (
    "Explain how a write-ahead log lets a database recover after a crash, "
    "and name one cost it adds.",
    "Compare TCP and QUIC for a mobile client on a lossy network in three short paragraphs.",
    "A train leaves at 09:40 and travels 212 km at 80 km/h. When does it arrive? Show the steps.",
    "Summarize the trade-offs between optimistic and pessimistic locking with one example each.",
    "Give a short, correct proof that the square root of 2 is irrational.",
    "Describe how a Bloom filter works and when a false positive can occur.",
    "Explain why floating-point addition is not associative, with a concrete numeric example.",
    "Outline a rollback plan for a failed database schema migration in a web service.",
)
_CODING_TASKS = (
    "Implement an LRU cache class with get and put in O(1) time, with a small usage example.",
    "Write a function that merges overlapping intervals and returns them sorted.",
    "Implement a thread-safe token bucket rate limiter with a monotonic clock.",
    "Write a function that parses an ISO 8601 duration such as 'P1DT2H30M' into seconds.",
    "Implement Dijkstra's shortest path over an adjacency-list graph with a heap.",
    "Write a function that returns the longest palindromic substring of a string.",
    "Implement a trie with insert, search, and starts_with methods.",
    "Write a function that validates a Sudoku board given as a list of nine strings.",
)


def _task(workload: str, request: int) -> str:
    if workload == "mixed":  # generic and coding rows alternate
        workload = "generic" if request % 2 == 0 else "coding"
        request //= 2
    if workload == "generic":
        return _GENERIC_TASKS[request % len(_GENERIC_TASKS)]
    return _CODING_TASKS[request % len(_CODING_TASKS)] + " Return one self-contained Python module."


def completed_dataset(path: Path, requests: int, *, workload: str, namespace: str) -> None:
    # A unique label first defeats prefix caching across rows.
    rows = [
        {
            "prompt": f"Row label (an identifier, not an instruction): {namespace}-{request}.\n\n"
            + _task(workload, request)
        }
        for request in range(requests)
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def bench(dataset: Path, *, requests: int, concurrency: int, max_tokens: int, out: Path) -> int:
    command = [
        sys.executable, str(HERE / "benchmark.py"),
        "--base-url", f"{_api_url()}/v1",
        "--model", SERVED,
        "--dataset", str(dataset),
        "--num-requests", str(requests),
        "--concurrency", str(concurrency),
        "--max-tokens", str(max_tokens),
        "--results-dir", str(out),
    ]  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    with (out / "bench.log").open("w") as log:
        return subprocess.run(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT).returncode


def _placement_counts(offset: int, expected: int) -> int:
    deadline = time.monotonic() + 10
    while True:
        count = 0
        if PLACEMENT_LOG.exists():
            with PLACEMENT_LOG.open() as stream:
                stream.seek(offset)
                count = sum(
                    1
                    for line in stream
                    if '"kind": "replica"' in line or '"kind":"replica"' in line
                )
        if count >= expected or time.monotonic() > deadline:
            return count
        time.sleep(0.5)


def serving(run_dir: Path) -> int:
    """Completed think-first answers through Kairyu at c1/8/16/32, all on the one replica."""

    config = SPEC["verification"]["serving"]
    requests, max_tokens = int(config["requests_per_concurrency"]), int(config["answer_max_tokens"])
    warmup = run_dir / "warmup.json"
    completed_dataset(warmup, 2, workload="generic", namespace=f"{run_dir.name}-warmup")
    if bench(warmup, requests=2, concurrency=2, max_tokens=max_tokens, out=run_dir / "warmup"):
        print("warm-up row failed", file=sys.stderr)
        return 1
    for workload in config["workloads"]:
        for level in LEVELS or config["concurrency"]:
            name = f"{workload}-c{level}"
            dataset = run_dir / f"{name}.json"
            completed_dataset(
                dataset, requests, workload=workload, namespace=f"{run_dir.name}-{name}"
            )
            offset = PLACEMENT_LOG.stat().st_size if PLACEMENT_LOG.exists() else 0
            code = bench(
                dataset,
                requests=requests,
                concurrency=level,
                max_tokens=max_tokens,
                out=run_dir / name,
            )
            placed = _placement_counts(offset, requests)
            (run_dir / name / "placements.json").write_text(json.dumps({"placements": placed}))
            if code or placed != requests:
                print(f"{name}: exit {code}, placements {placed}/{requests}", file=sys.stderr)
                return 1
    return 0


def think(run_dir: Path) -> int:
    """A caller's effort does not change the thought; chat_template_kwargs stays rejected."""

    config = SPEC["verification"]["think"]
    fan = int(config["requests_per_variant"])
    variants: dict[str, dict] = {
        "default": {},
        **{e: {"reasoning_effort": e} for e in ("low", "high", "max")},
    }
    requests = [(name, overrides) for name, overrides in variants.items() for _ in range(fan)]

    def ask(item):
        name, overrides = item
        payload = {
            "model": SERVED,
            "max_tokens": int(config["max_tokens"]),
            "messages": [{"role": "user", "content": ARITHMETIC}],
            **overrides,
        }
        return name, *post_chat(payload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as pool:
        results = list(pool.map(ask, requests))
    cases: dict[str, str | None] = {}
    for name, status, body in results:
        error = (
            f"HTTP {status}"
            if status != 200 or not isinstance(body, dict)
            else (control.think_answer_error(body, expected="323"))
        )
        cases[name] = cases.get(name) or error
    status, _ = post_chat(
        {
            "model": SERVED,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": ARITHMETIC}],
        }
    )
    cases["chat_template_kwargs_rejected"] = None if status == 400 else f"HTTP {status}"
    return _report(run_dir, "think", cases)


def _stream_tool_body(sse: str) -> dict:
    events, _ = stream_events(sse)
    calls: dict[int, dict[str, str]] = {}
    finish: object = None
    for event in events:
        for choice in event.get("choices") or ():
            finish = choice.get("finish_reason") or finish
            for delta in (choice.get("delta") or {}).get("tool_calls") or ():
                slot = calls.setdefault(delta.get("index", 0), {"name": "", "arguments": ""})
                function = delta.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
    message = {"tool_calls": [{"function": slot} for _, slot in sorted(calls.items())]}
    return {"choices": [{"message": message, "finish_reason": finish}]}


def _thinking_tool_call_error(status: int, body: object) -> str | None:
    if status != 200 or not isinstance(body, dict):
        return f"HTTP {status}: {str(body)[:200]}"
    message = body["choices"][0]["message"]
    return control.tool_call_error(body) or (
        None if message.get("reasoning_content") else "no reasoning before the tool call"
    )


def tool_calling(run_dir: Path) -> int:
    """Auto tool calls (plain and streamed) with a thought, and a tool-result turn."""

    fan = 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=fan) as pool:
        results = list(pool.map(lambda _: post_chat(control.tool_request()), range(fan)))
    cases = {f"auto_tool_call_{i}": _thinking_tool_call_error(*r) for i, r in enumerate(results)}
    status, body = results[0]
    if not cases["auto_tool_call_0"]:
        first = body["choices"][0]["message"]
        call = first["tool_calls"][0]
        call_id = call.get("id") or "call_0"
        history = [
            *control.tool_request()["messages"],
            {
                "role": "assistant",
                "content": first.get("content"),
                "reasoning_content": first.get("reasoning_content"),
                "tool_calls": [{"id": call_id, "type": "function", "function": call["function"]}],
            },
            {"role": "tool", "tool_call_id": call_id, "content": "README.md\nsrc\ntests\n"},
        ]
        cases["tool_result_turn"] = _thinking_tool_call_error(
            *post_chat(control.tool_request(messages=history))
        )
    status, sse = post_chat(control.tool_request(stream=True))
    cases["streamed_tool_call"] = (
        control.tool_call_error(_stream_tool_body(sse)) if status == 200 else f"HTTP {status}"
    )
    return _report(run_dir, "tool-calling", cases)


def vision(run_dir: Path) -> int:
    """Image requests think first and name the image's color."""

    config = SPEC["verification"]["vision"]

    def ask(case: int):
        return post_chat(
            control.image_request(
                f"Vision case {case}: what single color fills this image? Answer with one word.",
                max_tokens=int(config["max_tokens"]),
            )
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=int(config["requests"])) as pool:
        results = list(pool.map(ask, range(int(config["requests"]))))
    cases = {
        f"image_{i}": (
            control.vision_answer_error(body)
            if status == 200 and isinstance(body, dict)
            else f"HTTP {status}"
        )
        for i, (status, body) in enumerate(results)
    }
    return _report(run_dir, "vision", cases)


def _outstanding_zero() -> bool:
    with urllib.request.urlopen(f"{_api_url()}/metrics", timeout=5) as response:
        metrics = response.read().decode()
    values = re.findall(r"^kairyu_replica_outstanding\{[^\n]*\} ([0-9.]+)$", metrics, re.M)
    return bool(values) and all(float(v) == 0 for v in values) and vllm_active() == 0


def _cancel_during(field: str) -> dict:
    payload = {
        "model": SERVED,
        "stream": True,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "Count upwards from one, one number per line."}],
    }
    request = urllib.request.Request(
        f"{_api_url()}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    seen = observed = False
    with urllib.request.urlopen(request, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            if any(
                (c.get("delta") or {}).get(field) for c in json.loads(line[6:]).get("choices", [])
            ):
                seen = True
                break
        observed = seen and vllm_active() > 0
    released_at, start = None, time.monotonic()
    while time.monotonic() - start < 30:
        if _outstanding_zero():
            released_at = time.monotonic() - start
            break
        time.sleep(0.25)
    return {"saw_delta": seen, "l1_active_when_cut": observed, "released_after_s": released_at}


def cancellation(run_dir: Path) -> int:
    """A disconnect during the thought and during the answer frees L1; a follow-up completes."""

    cases: dict[str, str | None] = {}
    rows = {}
    for phase, field in (("thought", "reasoning_content"), ("answer", "content")):
        rows[phase] = row = _cancel_during(field)
        ok = row["saw_delta"] and row["l1_active_when_cut"] and row["released_after_s"] is not None
        cases[f"disconnect_during_{phase}"] = None if ok else f"not released: {row}"
    status, body = post_chat(
        {"model": SERVED, "max_tokens": 256, "messages": [{"role": "user", "content": ARITHMETIC}]}
    )
    cases["follow_up_completes"] = (
        control.think_answer_error(body, expected="323")
        if status == 200 and isinstance(body, dict)
        else f"HTTP {status}"
    )
    (run_dir / "cancellation-rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    return _report(run_dir, "cancellation", cases)


def restart(run_dir: Path) -> int:
    """A normal restart of OpenJev comes back healthy and passes the readiness probes."""

    env = control._compose_env()
    control._preflight(env)
    start = time.monotonic()
    subprocess.run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(HERE),
            "--file",
            str(HERE / "compose.yaml"),
            "restart",
            "openjev",
        ],
        env=env,
        check=True,
    )
    while time.monotonic() < start + 3600:
        state = json.loads(subprocess.check_output(["docker", "inspect", control.L1_CONTAINER]))[0]
        if state["State"].get("Health", {}).get("Status") == "healthy":
            break
        time.sleep(2)
    try:
        control.validate_serving(_api_url())
        error = None
    except SystemExit as failure:
        error = str(failure)
    print(f"restart: healthy and answering after {time.monotonic() - start:.0f} s")
    return _report(run_dir, "restart", {"healthy_and_answering": error})


def _openjev_source(run_dir: Path) -> Path:
    source = SPEC["openjev"]
    repository = source["source_repository"].removeprefix("https://github.com/")
    url = f"https://codeload.github.com/{repository}/tar.gz/{source['source_revision']}"
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = response.read()
    checkout = run_dir / "openjev-source"
    checkout.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(checkout, filter="data")
    (root,) = checkout.iterdir()
    return root


def systemone_live(run_dir: Path) -> int:
    """OpenJev's own live suite (pinned revision) against Kairyu's System One and chat."""

    root = _openjev_source(run_dir)
    env = {**os.environ, "OPENJEV_LIVE_URL": _api_url()}
    with (run_dir / "pytest.log").open("w") as log:
        code = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=",
             "-rA", f"--junitxml={run_dir / 'junit.xml'}", "tests/test_live.py"],
            cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
        ).returncode  # fmt: skip
    print((run_dir / "pytest.log").read_text().strip().splitlines()[-1])
    return _report(run_dir, "systemone-live", {"test_live": None if code == 0 else f"exit {code}"})


def _openjev_url() -> str:
    address = subprocess.check_output(
        ["docker", "inspect", "--format",
         "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", control.L1_CONTAINER],
        text=True,
    ).strip()  # fmt: skip
    return f"http://{address}:{OPENJEV_PORT}"


async def _systemone_burst(base_url: str, requests: int, concurrency: int, namespace: str):
    """Cache-busted System One requests; per request (status, seconds, answer error)."""

    import httpx

    gate = asyncio.Semaphore(concurrency)

    async def one(client, index: int):
        payload = control.systemone_request(f"[{namespace}-{index}] {control.SYSTEMONE_STATE}")
        async with gate:
            start = time.perf_counter()
            response = await client.post(f"{base_url}/v1/systemone", json=payload)
            elapsed = time.perf_counter() - start
        if response.status_code != 200:
            return response.status_code, elapsed, None
        return 200, elapsed, control.systemone_answer_error(response.json())

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=300, limits=limits) as client:
        start = time.perf_counter()
        rows = await asyncio.gather(*(one(client, i) for i in range(requests)))
        return rows, time.perf_counter() - start


def _nearest_rank(values: list[float], fraction: float) -> float | None:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else None


def systemone_serving(run_dir: Path) -> int:
    """OpenJev's README method through Kairyu and direct: req/s, p50/p95 at c1/16/32/64."""

    config = SPEC["verification"]["systemone"]
    requests = int(config["requests_per_concurrency"])
    targets = {"kairyu": _api_url(), "direct": _openjev_url()}
    asyncio.run(_systemone_burst(targets["kairyu"], 8, 8, f"{run_dir.name}-warmup"))
    rows, cases = [], {}
    for level in LEVELS or config["concurrency"]:
        for target, url in targets.items():
            name = f"{target}-c{level}"
            namespace = f"{run_dir.name}-{name}"
            results, wall = asyncio.run(_systemone_burst(url, requests, level, namespace))
            latencies = [elapsed * 1000 for status, elapsed, _ in results if status == 200]
            bad = [(status, error) for status, _, error in results if status != 200 or error]
            row = {
                "row": name, "requests": requests, "ok": requests - len(bad), "wall_s": wall,
                "requests_per_s": len(latencies) / wall,
                "p50_ms": _nearest_rank(latencies, 0.5), "p95_ms": _nearest_rank(latencies, 0.95),
            }  # fmt: skip
            rows.append(row)
            print(json.dumps(row), flush=True)
            cases[name] = f"{len(bad)} failed: {bad[:3]}" if bad else None
    (run_dir / "systemone-serving.json").write_text(json.dumps(rows, indent=2) + "\n")
    return _report(run_dir, "systemone-serving", cases)


def systemone_isolation(run_dir: Path) -> int:
    """A System One burst past every limit gets Kairyu's 429, never OpenJev's 529, while
    chat keeps answering and the chat replica stays healthy."""

    config = SPEC["verification"]["systemone"]
    reads, chats = int(config["isolation_reads"]), int(config["isolation_chats"])
    chat_payload = {
        "model": SERVED,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": ARITHMETIC}],
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=chats) as pool:
        chat_results = [pool.submit(post_chat, chat_payload) for _ in range(chats)]
        burst, wall = asyncio.run(
            _systemone_burst(_api_url(), reads, reads, f"{run_dir.name}-burst")
        )
        chat_bodies = [future.result() for future in chat_results]
    statuses: dict[int, int] = {}
    for status, _, _ in burst:
        statuses[status] = statuses.get(status, 0) + 1
    metrics = urllib.request.urlopen(f"{_api_url()}/metrics", timeout=5).read().decode()
    healthy = control._healthy_replicas(metrics, SERVED)
    cases = {
        "burst_only_200_or_429": None if set(statuses) <= {200, 429} else f"statuses {statuses}",
        "burst_reaches_kairyu_limit": None if statuses.get(429) else f"no 429: {statuses}",
        "answers_valid": next((e for s, _, e in burst if s == 200 and e), None),
        "chat_answers_during_burst": next(
            (f"HTTP {s}" if s != 200 else control.think_answer_error(b, expected="323")
             for s, b in chat_bodies
             if s != 200 or control.think_answer_error(b, expected="323")),
            None,
        ),
        "chat_replica_healthy": None if healthy == 1 else f"healthy replicas {healthy}",
    }  # fmt: skip
    (run_dir / "isolation.json").write_text(
        json.dumps({"statuses": statuses, "wall_s": wall, "healthy": healthy}, indent=2) + "\n"
    )
    print(f"burst statuses {statuses} in {wall:.1f} s; healthy chat replicas {healthy}")
    return _report(run_dir, "systemone-isolation", cases)


# --- evidence ---------------------------------------------------------------------------


def served_config_sha256() -> str:
    digest = hashlib.sha256()
    for path in SERVED_CONFIG_FILES:
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def runtime_evidence() -> dict:
    """Refuse to measure a runtime that differs from the committed configuration."""

    container = json.loads(subprocess.check_output(["docker", "inspect", control.L1_CONTAINER]))[0]
    pinned = SPEC["openjev"]["image_id"]
    if os.environ.get("OPENJEV_ALLOW_UNPINNED_IMAGE") != "1" and container["Image"] != pinned:
        raise ValueError(
            f"running OpenJev image {container['Image']} differs from the pin {pinned}"
        )
    running = dict(item.split("=", 1) for item in container["Config"]["Env"] if "=" in item)
    expected = control.openjev_settings()
    drift = {key: running.get(key) for key, value in expected.items() if running.get(key) != value}
    if drift:
        raise ValueError(f"running OpenJev settings differ from this run's: {drift}")
    gateway = f"{control.PROJECT}-kairyu-1"
    mounted = subprocess.check_output(["docker", "exec", gateway, "cat", "/etc/kairyu/kairyu.yaml"])
    if mounted != (HERE / "kairyu.yaml").read_bytes():
        raise ValueError("gateway serves a different kairyu.yaml")
    return {
        "image_id": container["Image"],
        "openjev_settings": expected,
        "started_at": container["State"]["StartedAt"],
        "cpuset": container["HostConfig"]["CpusetCpus"],
        "gateway_config_sha256": hashlib.sha256(mounted).hexdigest(),
    }


GATES = {
    "l1": l1,
    "serving": serving,
    "think": think,
    "tool-calling": tool_calling,
    "vision": vision,
    "cancellation": cancellation,
    "restart": restart,
    "systemone-live": systemone_live,
    "systemone-serving": systemone_serving,
    "systemone-isolation": systemone_isolation,
}


def main() -> None:
    global REQUEST_LOG, LEVELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", choices=[*GATES, "list"])
    parser.add_argument("--run-id")
    parser.add_argument("--no-start", action="store_true")
    parser.add_argument("--concurrency", help="comma-separated levels, e.g. 1,32")
    args = parser.parse_args()
    LEVELS = [int(level) for level in args.concurrency.split(",")] if args.concurrency else None
    if args.gate == "list":
        for name, function in GATES.items():
            print(f"{name:13} {(function.__doc__ or '-').strip().splitlines()[0]}")
        return
    if not args.no_start:
        control.up()
    run_dir = RESULTS_ROOT / f"{args.run_id or _run_id()}-{args.gate}"
    run_dir.mkdir(parents=True, exist_ok=True)
    REQUEST_LOG = run_dir / "requests.jsonl"
    manifest = {
        "schema_version": 1,
        "gate": args.gate,
        "started_at": datetime.now(UTC).isoformat(),
        "served_config_sha256": served_config_sha256(),
        "runtime": runtime_evidence(),
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        code = GATES[args.gate](run_dir)
    except Exception as error:  # noqa: BLE001 - a crashed gate is a failed gate
        print(f"{args.gate} failed: {error!r}", file=sys.stderr)
        code = 1
    manifest.update(completed_at=datetime.now(UTC).isoformat(), exit_code=code)
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"artifacts: {run_dir}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
