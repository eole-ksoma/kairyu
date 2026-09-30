#!/usr/bin/env python3
"""GPU verification for DeepSeek-V4.1-Flash on six GPUs (TP2 x DP3 / EP6).

``l1`` is the correctness gate every L1 candidate must pass before it is
measured: rendered efforts, exact finite answers from all three DP ranks,
the default sampling, tools, images, and memory. The public gates run
through Kairyu (L2 ReplicaPool, L3 API).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import control

HERE = control.HERE
ROOT = control.ROOT
SPEC = control.SPEC
SERVED_MODEL = SPEC["model"]["served_name"]
DP_RANKS = int(SPEC["allocation"]["data_parallel_size"])
SERVED_CONFIG_FILES = (
    HERE / "example.json",
    HERE / "compose.yaml",
    HERE / "kairyu.yaml",
    HERE / "vllm-sm120.Dockerfile",
    HERE / "patch_sm120.py",
    HERE / "webui-reasoning-effort-filter.py",
)
ENVIRONMENT_STORAGE = control.environment_storage()
RESULTS_ROOT = Path(
    os.environ.get("VERIFICATION_RESULTS_ROOT", ENVIRONMENT_STORAGE / "verification-results")
)
PLACEMENT_LOG = ENVIRONMENT_STORAGE / "placement-log" / Path(SPEC["pool"]["placement_log"]).name
REQUEST_LOG: Path | None = None


def _api_url() -> str:
    return f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}"


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# --- HTTP helpers ------------------------------------------------------------------


def post_chat(payload: dict, *, timeout_s: float = 900.0) -> tuple[int, object, str | None]:
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
    if REQUEST_LOG is not None:
        with REQUEST_LOG.open("a", encoding="utf-8") as output:
            row = {"request": payload, "status": status, "response": body, "request_id": request_id}
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return status, body, request_id


_L1_PROGRAM = r"""
import json, sys, urllib.error, urllib.request
method, path = sys.argv[1:3]
data = sys.stdin.buffer.read() if method == "POST" else None
request = urllib.request.Request("http://127.0.0.1:8000" + path, data=data,
                                 headers={"Content-Type": "application/json"}, method=method)
try:
    with urllib.request.urlopen(request, timeout=1800) as response:
        sys.stdout.write(json.dumps({"status": response.status, "body": response.read().decode()}))
except urllib.error.HTTPError as error:
    body = error.read().decode("utf-8", "replace")
    sys.stdout.write(json.dumps({"status": error.code, "body": body}))
"""


def l1_call(method: str, path: str, payload: dict | None = None) -> tuple[int, str]:
    """Reach the vLLM server inside its container (it publishes no host port)."""
    completed = subprocess.run(
        ["docker", "exec", "-i", control.L1_CONTAINER, "python3", "-c", _L1_PROGRAM, method, path],
        input=json.dumps(payload).encode() if payload is not None else b"",
        capture_output=True,
        timeout=1900,
        check=True,
    )
    result = json.loads(completed.stdout)
    return int(result["status"]), result["body"]


def l1_json(path: str, payload: dict) -> dict:
    status, body = l1_call("POST", path, payload)
    if status != 200:
        raise ValueError(f"L1 {path} returned HTTP {status}: {body[:300]}")
    return json.loads(body)


# --- L1 correctness gate -------------------------------------------------------------

EFFORT_BUDGETS = {"default": 75, **SPEC["model"]["reasoning_effort_budgets"]}


def rendered_prompt(variant: str) -> str:
    kwargs: dict = {"thinking": variant != "chat"}
    if variant not in {"default", "chat"}:
        kwargs["reasoning_effort"] = variant
    tokens = l1_json(
        "/tokenize",
        {
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": "What is 17 * 19?"}],
            "chat_template_kwargs": kwargs,
            "add_generation_prompt": True,
        },
    )["tokens"]
    return l1_json("/detokenize", {"model": SERVED_MODEL, "tokens": tokens})["prompt"]


def rendering_error(variant: str, prompt: str) -> str | None:
    budgets = re.findall(r"Reasoning Effort: (\d+)", prompt)
    if variant == "chat":
        if budgets or not prompt.rstrip().endswith("</think>"):
            return f"chat mode must close thinking without a budget: {prompt[-120:]!r}"
        return None
    if budgets != [str(EFFORT_BUDGETS[variant])]:
        return f"{variant} rendered budgets {budgets}, expected {EFFORT_BUDGETS[variant]}"
    if not prompt.rstrip().endswith("<think>"):
        return f"{variant} does not open thinking: {prompt[-80:]!r}"
    return None


def engine_success_counts(metrics: str) -> dict[str, float]:
    """Finished-request counters per DP engine from vLLM's Prometheus text."""
    counts: dict[str, float] = {}
    pattern = re.compile(r"^vllm:request_success_total\{([^}]*)\} ([0-9.eE+-]+)$", re.M)
    for labels, value in pattern.findall(metrics):
        engine = re.search(r'engine="([^"]+)"', labels)
        if engine:
            counts[engine.group(1)] = counts.get(engine.group(1), 0.0) + float(value)
    return counts


def served_engines(before: dict[str, float], after: dict[str, float]) -> set[str]:
    return {engine for engine, value in after.items() if value > before.get(engine, 0.0)}


def _l1_probe(variant: str) -> dict:
    payload = control.arithmetic_probe()
    if variant == "chat":
        payload["chat_template_kwargs"] = {"thinking": False}
    elif variant != "default":
        payload["chat_template_kwargs"] = {"thinking": True, "reasoning_effort": variant}
    payload["model"] = SERVED_MODEL
    status, body = l1_call("POST", "/v1/chat/completions", payload)
    parsed = json.loads(body) if status == 200 else body
    error = control.arithmetic_answer_error(parsed) if status == 200 else f"HTTP {status}"
    return {"variant": variant, "status": status, "error": error, "response": parsed}


def _startup_evidence() -> dict:
    logs = subprocess.run(
        ["docker", "logs", control.L1_CONTAINER], capture_output=True, text=True, check=True
    )
    text = logs.stdout + logs.stderr
    patterns = {
        "weights": r"Model loading took ([0-9.]+) GiB",
        "kv_cache_memory": r"Available KV cache memory: ([0-9.-]+) GiB",
        "kv_cache_tokens": r"GPU KV cache size: ([0-9,]+) tokens",
        "max_concurrency": r"Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x",
        "sampling_defaults": (
            r"(Using default chat sampling params[^\n]*|Default sampling parameters[^\n]*)"
        ),
    }
    return {name: re.findall(pattern, text) for name, pattern in patterns.items()}


def gpu_memory() -> dict[str, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = dict(line.split(", ") for line in output.strip().splitlines())
    return {index: int(rows[str(index)]) for index in map(str, SPEC["allocation"]["gpu_ids"])}


def l1(run_dir: Path) -> int:
    report: dict = {"schema_version": 1, "cases": {}}
    failures: list[str] = []

    def record(name: str, error: str | None, detail: object = None) -> None:
        report["cases"][name] = {"passed": error is None, "error": error, "detail": detail}
        if error is not None:
            failures.append(f"{name}: {error}")

    report["startup"] = _startup_evidence()
    report["idle_gpu_memory_mib"] = gpu_memory()
    defaults = " ".join(report["startup"]["sampling_defaults"])
    record(
        "default_sampling_top_p_0.95",
        None if "'top_p': 0.95" in defaults and "'temperature': 1.0" in defaults else defaults,
    )
    for variant in SPEC["verification"]["l1"]["variants"]:
        try:
            prompt = rendered_prompt(variant)
            record(f"render_{variant}", rendering_error(variant, prompt), prompt[-160:])
        except Exception as error:  # noqa: BLE001 - recorded
            record(f"render_{variant}", str(error))

    probes = int(SPEC["verification"]["l1"]["probes_per_variant"])
    for variant in SPEC["verification"]["l1"]["variants"]:
        before = engine_success_counts(l1_call("GET", "/metrics")[1])
        with concurrent.futures.ThreadPoolExecutor(max_workers=probes) as pool:
            results = list(pool.map(_l1_probe, [variant] * probes))
        after = engine_success_counts(l1_call("GET", "/metrics")[1])
        errors = [row["error"] for row in results if row["error"]]
        engines = served_engines(before, after)
        detail = {"engines": sorted(engines), "errors": errors[:3]}
        (run_dir / f"probes-{variant}.json").write_text(json.dumps(results, indent=2) + "\n")
        error = f"{len(errors)}/{probes} probes failed" if errors else None
        if error is None and len(engines) != DP_RANKS:
            error = f"only engines {sorted(engines)} served; expected {DP_RANKS} DP ranks"
        record(f"probes_{variant}", error, detail)

    for name, check in (
        ("tool_call", control.validate_tool_calling),
        ("image", control.validate_vision),
    ):
        try:
            check(_api_url())
            record(name, None)
        except SystemExit as error:
            record(name, str(error))
    report["loaded_gpu_memory_mib"] = gpu_memory()
    report["passed"] = not failures
    (run_dir / "l1.json").write_text(json.dumps(report, indent=2) + "\n")
    for line in failures:
        print(f"l1: {line}", file=sys.stderr)
    print(f"l1: {'PASS' if not failures else 'FAIL'} ({len(report['cases'])} cases)")
    return 1 if failures else 0


# --- serving rows -----------------------------------------------------------------------

_WORDS = (
    "code", "review", "function", "module", "request", "result", "verify", "runtime",
    "system", "design", "state", "input", "output", "stream", "cache", "token",
)  # fmt: skip


def fixed_dataset(path: Path, requests: int, approximate_tokens: int, *, namespace: str) -> None:
    # A unique label first defeats prefix caching, so each row is a cold prefill.
    rows = [
        {
            "prompt": f"Row {namespace}-{request}: "
            + " ".join(
                _WORDS[(request * 7 + position * 11) % len(_WORDS)]
                for position in range(approximate_tokens)
            )
        }
        for request in range(requests)
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


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


def completed_dataset(path: Path, requests: int, *, workload: str, namespace: str) -> None:
    tasks = _GENERIC_TASKS if workload == "generic" else _CODING_TASKS
    suffix = (
        ""
        if workload == "generic"
        else " Return one self-contained Python module in a fenced code block."
    )
    rows = [
        {
            "prompt": f"Row label (an identifier, not an instruction): {namespace}-{request}.\n\n"
            + tasks[request % len(tasks)]
            + suffix
        }
        for request in range(requests)
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def bench(
    dataset: Path, *, mode: str, requests: int, concurrency: int, max_tokens: int, out: Path
) -> int:
    command = [
        sys.executable,
        str(HERE / "benchmark.py"),
        "--base-url",
        f"{_api_url()}/v1",
        "--model",
        SERVED_MODEL,
        "--dataset",
        str(dataset),
        "--mode",
        mode,
        "--num-requests",
        str(requests),
        "--concurrency",
        str(concurrency),
        "--max-tokens",
        str(max_tokens),
        "--results-dir",
        str(out),
    ]
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


def serving(
    run_dir: Path, concurrency: list[int] | None = None, requests: int | None = None
) -> int:
    config = SPEC["verification"]["serving"]
    requests = requests or int(config["requests_per_concurrency"])
    output_tokens = int(config["output_tokens"])
    prompt_tokens = int(config["prompt_tokens_approx"])
    warmup = run_dir / "warmup.json"
    fixed_dataset(warmup, DP_RANKS, prompt_tokens, namespace=f"{run_dir.name}-warmup")
    if bench(
        warmup,
        mode="fixed",
        requests=DP_RANKS,
        concurrency=DP_RANKS,
        max_tokens=32,
        out=run_dir / "warmup",
    ):
        print("warm-up row failed", file=sys.stderr)
        return 1
    for level in concurrency or config["concurrency"]:
        dataset = run_dir / f"fixed-c{level}.json"
        fixed_dataset(dataset, requests, prompt_tokens, namespace=f"{run_dir.name}-c{level}")
        offset = PLACEMENT_LOG.stat().st_size if PLACEMENT_LOG.exists() else 0
        peak = GpuPeak()
        with peak:
            code = bench(
                dataset,
                mode="fixed",
                requests=requests,
                concurrency=level,
                max_tokens=output_tokens,
                out=run_dir / f"fixed-c{level}",
            )
        placed = _placement_counts(offset, requests)
        (run_dir / f"fixed-c{level}" / "resources.json").write_text(
            json.dumps({"gpu_peak_mib": peak.peak, "placements": placed}, indent=2) + "\n"
        )
        if code or placed != requests:
            print(f"fixed c{level}: exit {code}, placements {placed}/{requests}", file=sys.stderr)
            return 1
    return 0


def completed(run_dir: Path) -> int:
    failures = 0
    for workload in ("generic", "coding"):
        for level in (1, 8, 32):
            dataset = run_dir / f"{workload}-c{level}.json"
            completed_dataset(
                dataset, 32, workload=workload, namespace=f"{run_dir.name}-{workload}-c{level}"
            )
            failures += (
                bench(
                    dataset,
                    mode="completed",
                    requests=32,
                    concurrency=level,
                    max_tokens=65536,
                    out=run_dir / f"{workload}-c{level}",
                )
                != 0
            )
    return int(failures > 0)


class GpuPeak:
    """Sample GPU 0-5 memory once per second while a row runs."""

    def __init__(self) -> None:
        self.peak: dict[str, int] = {}
        self._stop = False
        self._thread: object = None

    def __enter__(self):
        import threading

        def loop() -> None:
            while not self._stop:
                for index, used in gpu_memory().items():
                    self.peak[index] = max(self.peak.get(index, 0), used)
                time.sleep(1)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop = True
        self._thread.join(timeout=5)


# --- public gates --------------------------------------------------------------------------

_TOOL_SYSTEM = (
    "You are an agent operating a computer shell. Every response MUST include "
    "at least one bash tool call; never answer in plain text."
)


def _tool_request(**overrides) -> dict:
    payload = {
        "model": SERVED_MODEL,
        "messages": [
            {"role": "system", "content": _TOOL_SYSTEM},
            {"role": "user", "content": "List the files in the current directory."},
        ],
        "tools": [control._BASH_TOOL],
        "max_tokens": 8192,
    }
    payload.update(overrides)
    return payload


def tool_call_error(message: dict, finish_reason: object) -> str | None:
    if finish_reason != "tool_calls":
        return f"finish_reason is {finish_reason!r}"
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return f"tool_calls is {calls!r}"
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or function.get("name") != "bash":
            return f"unexpected tool call {call!r}"
        try:
            arguments = json.loads(function.get("arguments") or "")
        except ValueError:
            return f"arguments are not JSON: {function.get('arguments')!r}"
        if not isinstance(arguments, dict) or not isinstance(arguments.get("command"), str):
            return f"arguments lack a command: {arguments!r}"
    return None


def _stream_tool_message(sse: str) -> tuple[dict, object]:
    calls: dict[int, dict[str, str]] = {}
    finish: object = None
    for line in sse.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        for choice in json.loads(line[6:]).get("choices", ()):
            finish = choice.get("finish_reason") or finish
            for delta in (choice.get("delta") or {}).get("tool_calls") or ():
                slot = calls.setdefault(delta.get("index", 0), {"name": "", "arguments": ""})
                function = delta.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
    return {"tool_calls": [{"function": slot} for _, slot in sorted(calls.items())] or None}, finish


def tool_calling(run_dir: Path) -> int:
    cases: dict[str, str | None] = {}
    fan = 2 * DP_RANKS
    with concurrent.futures.ThreadPoolExecutor(max_workers=fan) as pool:
        results = list(pool.map(lambda _: post_chat(_tool_request()), range(fan)))
    first: dict | None = None
    error = None
    for status, body, _ in results:
        if status != 200 or not isinstance(body, dict):
            error = f"HTTP {status}: {str(body)[:200]}"
            break
        choice = body["choices"][0]
        error = tool_call_error(choice["message"], choice["finish_reason"])
        if error:
            break
        first = choice["message"]
    cases[f"auto_tool_call_x{fan}"] = error
    if first is not None:
        call = first["tool_calls"][0]
        call_id = call.get("id") or "call_0"
        status, body, _ = post_chat(
            _tool_request(
                messages=[
                    {"role": "system", "content": _TOOL_SYSTEM},
                    {"role": "user", "content": "List the files in the current directory."},
                    {
                        "role": "assistant",
                        "content": first.get("content"),
                        "reasoning_content": first.get("reasoning_content"),
                        "tool_calls": [
                            {"id": call_id, "type": "function", "function": call["function"]}
                        ],
                    },
                    {"role": "tool", "tool_call_id": call_id, "content": "README.md\nsrc\ntests\n"},
                ]
            )
        )
        cases["tool_result_turn"] = (
            tool_call_error(body["choices"][0]["message"], body["choices"][0]["finish_reason"])
            if status == 200 and isinstance(body, dict)
            else f"HTTP {status}"
        )
    status, body, _ = post_chat(_tool_request(stream=True))
    cases["streamed_tool_call"] = (
        tool_call_error(*_stream_tool_message(body)) if status == 200 else f"HTTP {status}"
    )
    for effort in ("low", "max"):
        status, body, _ = post_chat(_tool_request(reasoning_effort=effort), timeout_s=1800)
        if status != 200 or not isinstance(body, dict):
            cases[f"tool_call_effort_{effort}"] = f"HTTP {status}"
            continue
        message = body["choices"][0]["message"]
        cases[f"tool_call_effort_{effort}"] = tool_call_error(
            message, body["choices"][0]["finish_reason"]
        ) or (None if message.get("reasoning_content") else "no reasoning under thinking")
    report = {"cases": cases, "passed": not any(cases.values())}
    (run_dir / "tool-calling.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"tool-calling: {'PASS' if report['passed'] else 'FAIL'} {cases}")
    return int(not report["passed"])


def vision(run_dir: Path) -> int:
    config = SPEC["verification"]["vision"]
    fan = int(config["requests"])

    def ask(case: int):
        return post_chat(
            control.image_request(
                f"Vision case {case}: what single color fills this image? Answer with one word.",
                max_tokens=int(config["max_tokens"]),
            ),
            timeout_s=1800,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=fan) as pool:
        results = list(pool.map(ask, range(fan)))
    answers, errors = [], []
    for status, body, _ in results:
        if status != 200 or not isinstance(body, dict):
            errors.append(f"HTTP {status}")
            continue
        choice = body["choices"][0]
        content = choice["message"].get("content") or ""
        answers.append(content.strip()[:80])
        if choice.get("finish_reason") != "stop" or "red" not in content.lower():
            errors.append(f"not a completed red answer: {content[:80]!r}")
    report = {"answers": answers, "errors": errors, "passed": not errors}
    (run_dir / "vision.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"vision: {'PASS' if not errors else 'FAIL'} {answers}")
    return int(bool(errors))


def _stream(payload: dict, *, timeout_s: float = 1800) -> dict:
    request = urllib.request.Request(
        f"{_api_url()}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    first_content = None
    content = trace = ""
    finish = usage = None
    done = False
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or ""
                if text and first_content is None:
                    first_content = time.monotonic() - start
                content += text
                trace += delta.get("reasoning_content") or ""
                finish = choice.get("finish_reason") or finish
    return {
        "done": done,
        "finish_reason": finish,
        "content": content,
        "reasoning_chars": len(trace),
        "time_to_content_s": first_content,
        "total_s": time.monotonic() - start,
        "usage": usage,
    }


def reasoning(run_dir: Path) -> int:
    rows = []
    for effort in (None, "low", "high", "max"):
        payload = {
            "model": SERVED_MODEL,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 32768,
            "messages": [
                {"role": "user", "content": "What is 17 * 19? Reply with only the integer."}
            ],
        }
        if effort:
            payload["reasoning_effort"] = effort
        row = {"effort": effort or "default", **_stream(payload)}
        row["passed"] = (
            row["done"]
            and row["finish_reason"] == "stop"
            and row["content"].strip() == "323"
            and row["reasoning_chars"] > 0
        )
        rows.append(row)
    (run_dir / "reasoning.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"reasoning: {[(row['effort'], row['passed']) for row in rows]}")
    return int(not all(row["passed"] for row in rows))


def _l1_active_requests() -> float:
    status, metrics = l1_call("GET", "/metrics")
    total = 0.0
    for gauge in ("running", "waiting"):
        values = re.findall(
            rf"^vllm:num_requests_{gauge}\{{[^\n]*\}} ([0-9.eE+-]+)$", metrics, re.M
        )
        if not values:
            raise ValueError(f"missing L1 {gauge} gauge")
        total += sum(float(value) for value in values)
    return total


def cancellation(run_dir: Path) -> int:
    payload = {
        "model": SERVED_MODEL,
        "stream": True,
        "max_tokens": 8192,
        "ignore_eos": True,
        "messages": [{"role": "user", "content": "Count upwards from one, one number per line."}],
    }
    request = urllib.request.Request(
        f"{_api_url()}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    saw_delta = observed = False
    with urllib.request.urlopen(request, timeout=1800) as response:
        for line in response:
            if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
                choices = json.loads(line[6:]).get("choices", [])
                if any(
                    c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content")
                    for c in choices
                ):
                    saw_delta = True
                    break
        deadline = time.monotonic() + 15
        while saw_delta and time.monotonic() < deadline:
            if _l1_active_requests() > 0:
                observed = True
                break
            time.sleep(0.5)
    released_at = None
    start = time.monotonic()
    while time.monotonic() - start < 30:
        with urllib.request.urlopen(f"{_api_url()}/metrics", timeout=5) as response:
            metrics = response.read().decode()
        values = re.findall(r"^kairyu_replica_outstanding\{[^\n]*\} ([0-9.]+)$", metrics, re.M)
        if values and all(float(v) == 0 for v in values) and _l1_active_requests() == 0:
            released_at = time.monotonic() - start
            break
        time.sleep(0.25)
    status, body, _ = post_chat(
        {
            "model": SERVED_MODEL,
            "max_tokens": 2048,
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "Reply OK."}],
        }
    )
    usable = (
        status == 200
        and isinstance(body, dict)
        and body["choices"][0].get("finish_reason") == "stop"
        and bool(body["choices"][0]["message"].get("content"))
    )
    report = {
        "received_delta": saw_delta,
        "l1_observed_active": observed,
        "released_after_s": released_at,
        "follow_up_completed": usable,
        "passed": bool(saw_delta and observed and released_at is not None and usable),
    }
    (run_dir / "cancellation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"cancellation: {report}")
    return int(not report["passed"])


def long_context(run_dir: Path) -> int:
    import secrets

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(
        str(ENVIRONMENT_STORAGE / "models" / SPEC["model"]["slug"] / "tokenizer.json")
    )
    filler = " filler"
    if len(tokenizer.encode(filler * 100, add_special_tokens=False).ids) != 100:
        raise ValueError("long-context filler is not one token per repetition")
    maximum = int(SPEC["model"]["max_context_tokens"])
    rows = []
    for target in (32768, 131072, 262144, maximum - 8704):
        key = "K" + secrets.token_hex(12).upper()
        prefix = "Read this log and recover its archive key.\n"
        needle = f"\nThe archive key is {key}.\n"
        suffix = "\nReturn only the archive key."
        overhead = len(tokenizer.encode(prefix + needle + suffix, add_special_tokens=False).ids)
        repeats = target - overhead
        prompt = (
            prefix + filler * (repeats // 2) + needle + filler * (repeats - repeats // 2) + suffix
        )
        start = time.monotonic()
        status, body, _ = post_chat(
            {
                "model": SERVED_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8192,
            },
            timeout_s=3600,
        )
        choice = body.get("choices", [{}])[0] if isinstance(body, dict) else {}
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        content = (choice.get("message") or {}).get("content")
        passed = (
            status == 200
            and choice.get("finish_reason") == "stop"
            and isinstance(content, str)
            and content.strip() == key
            and usage.get("prompt_tokens", 0) + 8192 <= maximum
        )
        rows.append(
            {
                "target_input_tokens": target,
                "usage": usage,
                "content": content,
                "expected": key,
                "total_s": time.monotonic() - start,
                "passed": passed,
            }
        )
        (run_dir / "long-context.json").write_text(json.dumps(rows, indent=2) + "\n")
        if not passed:
            return 1
    return 0


def restart(run_dir: Path) -> int:
    """A normal service restart comes back healthy and answers again."""
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
            "deepseek",
        ],
        env=env,
        check=True,
    )
    deadline = start + 3600
    while time.monotonic() < deadline:
        state = json.loads(subprocess.check_output(["docker", "inspect", control.L1_CONTAINER]))[0]
        if state["State"].get("Health", {}).get("Status") == "healthy":
            break
        time.sleep(2)
    try:
        control.validate_serving(_api_url())
        passed = True
        error = None
    except SystemExit as failure:
        passed, error = False, str(failure)
    report = {
        "healthy_and_answering_after_s": time.monotonic() - start,
        "passed": passed,
        "error": error,
    }
    (run_dir / "restart.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"restart: {report}")
    return int(not passed)


# --- evidence ----------------------------------------------------------------------------------


def served_config_sha256() -> str:
    digest = hashlib.sha256()
    for path in SERVED_CONFIG_FILES:
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def runtime_evidence(*, allow_override: bool = False) -> dict:
    """Refuse to measure a runtime that differs from the committed configuration."""
    import yaml

    container = json.loads(subprocess.check_output(["docker", "inspect", control.L1_CONTAINER]))[0]
    service = yaml.safe_load((HERE / "compose.yaml").read_text())["services"]["deepseek"]
    if not allow_override:
        if container["Config"]["Cmd"] != service["command"]:
            raise ValueError("running L1 command differs from compose.yaml")
        if (
            os.environ.get("DEEPSEEK_ALLOW_UNPINNED_IMAGE") != "1"
            and container["Image"] != SPEC["vllm"]["image_id"]
        ):
            raise ValueError("running L1 image differs from the example.json pin")
    gateway = f"{control.PROJECT}-kairyu-1"
    mounted = subprocess.check_output(["docker", "exec", gateway, "cat", "/etc/kairyu/kairyu.yaml"])
    if mounted != (HERE / "kairyu.yaml").read_bytes():
        raise ValueError("gateway serves a different kairyu.yaml")
    return {
        "image_id": container["Image"],
        "command": container["Config"]["Cmd"],
        "started_at": container["State"]["StartedAt"],
        "cpuset": container["HostConfig"]["CpusetCpus"],
        "gateway_config_sha256": hashlib.sha256(mounted).hexdigest(),
    }


GATES = {
    "l1": l1,
    "serving": serving,
    "completed": completed,
    "tool-calling": tool_calling,
    "vision": vision,
    "reasoning": reasoning,
    "cancellation": cancellation,
    "long-context": long_context,
    "restart": restart,
}


def main() -> None:
    global REQUEST_LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", choices=[*GATES, "list"])
    parser.add_argument("--run-id")
    parser.add_argument("--no-start", action="store_true")
    args = parser.parse_args()
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
