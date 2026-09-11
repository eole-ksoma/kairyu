#!/usr/bin/env python3
"""Measured serving verification: fixed-token matrix, replica placement gate, tools, images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = json.loads((HERE / "example.json").read_text(encoding="utf-8"))
REPLICAS = int(SPEC["allocation"]["replicas"])
TENSOR_PARALLEL = int(SPEC["allocation"]["tensor_parallel_size"])
SERVED_MODEL = SPEC["model"]["served_name"]
# Files whose bytes define the served configuration (vLLM renders chat with
# the checkpoint's own encoder, so no example-owned template is involved).
SERVED_CONFIG_FILES = (
    HERE / "example.json",
    HERE / "compose.yaml",
    HERE / "kairyu.yaml",
    HERE / "vllm-sm120.Dockerfile",
    HERE / "patch_runtime.py",
    HERE / "webui-reasoning-effort-filter.py",
)


def _nvme_root() -> Path:
    configured = Path(os.environ.get("NVME_STORAGE_ROOT", SPEC["storage"]["root"]))
    if not configured.is_absolute():
        raise SystemExit("NVME_STORAGE_ROOT must be an absolute path below /mnt/nvme")
    root = configured.resolve()
    nvme = Path("/mnt/nvme")
    if root != nvme and nvme not in root.parents:
        raise SystemExit("NVME_STORAGE_ROOT must be /mnt/nvme or one of its descendants")
    return root


STORAGE_ROOT = _nvme_root()
ENVIRONMENT_STORAGE = STORAGE_ROOT / "model-volumes" / SPEC["environment"]
RESULTS_ROOT = Path(
    os.environ.get("VERIFICATION_RESULTS_ROOT", ENVIRONMENT_STORAGE / "verification-results")
)
# Host-side view of the pool's placement_log_path bind mount (compose.yaml).
PLACEMENT_LOG = ENVIRONMENT_STORAGE / "placement-log" / Path(SPEC["pool"]["placement_log"]).name
REQUEST_LOG: Path | None = None


def _run(command: list[str], *, log: Path | None = None, check: bool = True) -> int:
    print("+ " + " ".join(command), flush=True)
    if log is None:
        return subprocess.run(command, cwd=ROOT, check=check).returncode
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if check and process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return process.returncode


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _ensure_environment(no_start: bool) -> None:
    if not no_start:
        _run([sys.executable, str(HERE / "control.py"), "up"])


def _serving_dataset(path: Path, requests: int, approximate_tokens: int, *, namespace: str) -> None:
    vocabulary = (
        "code",
        "review",
        "function",
        "module",
        "request",
        "result",
        "verify",
        "runtime",
        "system",
        "design",
        "state",
        "input",
        "output",
        "stream",
        "cache",
        "token",
    )
    rows = []
    for request in range(requests):
        words = [
            vocabulary[(request * 7 + position * 11) % len(vocabulary)]
            for position in range(approximate_tokens)
        ]
        # The run/row identity and the unique case come first, so neither
        # vLLM prefix caching nor Kairyu's prefix-aware placement can turn a
        # row into a shared-prefix microbenchmark. Server-reported usage
        # remains the source of truth for the actual token count.
        prompt = f"Run {namespace}, case {request}: " + " ".join(words)
        rows.append({"conversations": [{"from": "human", "value": prompt}]})
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def _bench(
    dataset: Path,
    *,
    requests: int,
    concurrency: int,
    max_tokens: int,
    results_dir: Path,
    log: Path,
) -> int:
    return _run(
        [
            str(ROOT / ".venv/bin/python"),
            str(HERE / "benchmark.py"),
            "--base-url",
            f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}/v1",
            "--model",
            SERVED_MODEL,
            "--dataset",
            str(dataset),
            "--num-requests",
            str(requests),
            "--concurrency",
            str(concurrency),
            "--max-tokens",
            str(max_tokens),
            "--min-tokens",
            str(max_tokens),
            "--ignore-eos",
            "--temperature",
            "1.0",
            "--seed",
            "0",
            "--timeout",
            "1800",
            "--results-dir",
            str(results_dir),
            "--tensor-parallel",
            str(TENSOR_PARALLEL),
            "--dp-replicas",
            str(REPLICAS),
        ],
        log=log,
        check=False,
    )


def _placement_offset(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _placement_counts(
    path: Path,
    offset: int,
    *,
    request_ids: set[str] | None = None,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8") as stream:
        stream.seek(offset)
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("kind") != "replica":
                continue
            if request_ids is not None and row.get("request_id") not in request_ids:
                continue
            replica = row.get("replica_id", row.get("replica"))
            counts[str(replica)] += 1
    return counts


def _wait_for_placement_counts(
    path: Path,
    offset: int,
    *,
    expected_requests: int,
    request_ids: set[str] | None = None,
    settle_s: float = 10.0,
) -> Counter[str]:
    deadline = time.monotonic() + settle_s
    counts = _placement_counts(path, offset, request_ids=request_ids)
    while sum(counts.values()) < expected_requests and time.monotonic() < deadline:
        time.sleep(0.5)
        counts = _placement_counts(path, offset, request_ids=request_ids)
    return counts


def _placement_report(
    path: Path,
    offset: int,
    *,
    expected_requests: int,
    replicas: int,
    gated: bool,
    max_share_of_mean: float,
    settle_s: float = 10.0,
) -> dict:
    """Per-replica placement counts for one row, from the pool's JSONL log.

    The log is written asynchronously, so poll until the row's placements
    have landed (or the settle window expires). The gate requires exactly
    ``expected_requests`` placements, every replica to receive traffic, and
    no replica to exceed ``max_share_of_mean``
    times the ideal even share; a serial row (c1) is reported only, because
    least-outstanding ties resolve to the lowest replica id by design.
    """

    counts = _wait_for_placement_counts(
        path,
        offset,
        expected_requests=expected_requests,
        settle_s=settle_s,
    )
    total = sum(counts.values())
    mean = expected_requests / replicas if replicas else 0.0
    largest = max(counts.values(), default=0)
    even = (
        total == expected_requests
        and len(counts) == replicas
        and largest <= max_share_of_mean * mean
    )
    return {
        "schema_version": 1,
        "placements": total,
        "expected_requests": expected_requests,
        "replicas": replicas,
        "per_replica": dict(sorted(counts.items())),
        "largest_share_of_mean": round(largest / mean, 3) if mean else None,
        "max_share_of_mean": max_share_of_mean,
        "gated": gated,
        "passed": even if gated else None,
    }


def _validate_serving_row(row_dir: Path, requests: int, output_tokens: int) -> int:
    """Reject partial streams and nominally successful zero-token runs."""

    artifacts = list(row_dir.glob("*-serving.json"))
    if len(artifacts) != 1:
        print(
            f"serving row produced {len(artifacts)} result files; expected exactly one",
            file=sys.stderr,
        )
        return 1
    try:
        result = json.loads(artifacts[0].read_text(encoding="utf-8"))
        summary = result["summary"]
        samples = result["samples"]
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"invalid serving result: {error}", file=sys.stderr)
        return 1
    expected_total = requests * output_tokens
    complete = (
        summary.get("requests") == requests
        and summary.get("completion_tokens_total") == expected_total
        and isinstance(summary.get("output_tokens_per_s"), (int, float))
        and summary["output_tokens_per_s"] > 0
        and len(samples) == requests
        and all(sample.get("completion_tokens") == output_tokens for sample in samples)
    )
    if not complete:
        print(
            "serving row did not produce complete evidence: "
            f"requests={summary.get('requests')!r}, "
            f"completion_tokens_total={summary.get('completion_tokens_total')!r}, "
            f"output_tokens_per_s={summary.get('output_tokens_per_s')!r}, "
            f"samples={len(samples)!r}",
            file=sys.stderr,
        )
        return 1
    return 0


def serving(run_dir: Path) -> int:
    config = SPEC["verification"]["serving"]
    gate = config["placement_gate"]
    requests = int(config["requests_per_concurrency"])
    output_tokens = int(config["output_tokens"])
    prompt_tokens = int(config["prompt_tokens_approx"])
    run_dir.mkdir(parents=True, exist_ok=True)

    # One short request per replica so every replica's first-request work
    # (autotune, graph capture) is done before the measured rows.
    warmup_dataset = run_dir / "warmup-8k.json"
    _serving_dataset(warmup_dataset, REPLICAS, prompt_tokens, namespace=f"{run_dir.name}-warmup")
    if _bench(
        warmup_dataset,
        requests=REPLICAS,
        concurrency=REPLICAS,
        max_tokens=32,
        results_dir=run_dir / "warmup",
        log=run_dir / "warmup.log",
    ):
        print("warm-up row failed", file=sys.stderr)
        return 1

    failures = 0
    for concurrency in config["concurrency"]:
        row_dir = run_dir / f"serving-c{concurrency}"
        dataset = run_dir / f"serving-8k-c{concurrency}.json"
        _serving_dataset(
            dataset, requests, prompt_tokens, namespace=f"{run_dir.name}-c{concurrency}"
        )
        offset = _placement_offset(PLACEMENT_LOG)
        code = _bench(
            dataset,
            requests=requests,
            concurrency=concurrency,
            max_tokens=output_tokens,
            results_dir=row_dir,
            log=run_dir / f"serving-c{concurrency}.log",
        )
        if code == 0:
            code = _validate_serving_row(row_dir, requests, output_tokens)
        if code == 0:
            report = _placement_report(
                PLACEMENT_LOG,
                offset,
                expected_requests=requests,
                replicas=REPLICAS,
                gated=concurrency >= int(gate["min_concurrency"]),
                max_share_of_mean=float(gate["max_share_of_mean"]),
            )
            (row_dir / "placement.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"{row_dir.name}: placements {json.dumps(report['per_replica'])}")
            if report["passed"] is False:
                print(
                    f"{row_dir.name}: replica placement is not even "
                    f"(largest share {report['largest_share_of_mean']}x mean)",
                    file=sys.stderr,
                )
                code = 1
        failures += code != 0
        if code:
            break
    return 1 if failures else 0


# --- Tool-calling gate (PR #584 review: a served example must emit OpenAI
# tool calls, or tool-driven agents such as SWE-bench Pro's mini-swe-agent
# fail every turn). The request shape mirrors that agent: an auto-choice
# `bash` function tool on unary /chat/completions.
_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run."}},
            "required": ["command"],
        },
    },
}
_TOOL_SYSTEM = (
    "You are an agent operating a computer shell. Every response MUST include "
    "at least one bash tool call; never answer in plain text."
)
_TOOL_USER = "List the files in the current directory."


def _validate_tool_call_message(message: dict, finish_reason: object) -> str | None:
    """Reject responses a bash-tool agent loop cannot execute (None = valid)."""

    if finish_reason != "tool_calls":
        return f"finish_reason is {finish_reason!r}, expected 'tool_calls'"
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return f"message.tool_calls is {calls!r}"
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or function.get("name") != "bash":
            return f"unexpected tool call {call!r}"
        try:
            arguments = json.loads(function.get("arguments") or "")
        except ValueError:
            return f"tool call arguments are not JSON: {function.get('arguments')!r}"
        if (
            not isinstance(arguments, dict)
            or not isinstance(arguments.get("command"), str)
            or not arguments["command"]
        ):
            return f"tool call arguments lack a command string: {arguments!r}"
    return None


def _tool_placement_error(
    counts: Counter[str],
    *,
    expected_requests: int,
    replicas: int,
) -> str | None:
    total = sum(counts.values())
    if total != expected_requests:
        return (
            f"recorded {total} correlated placements, expected {expected_requests}: {dict(counts)}"
        )
    if len(counts) != replicas:
        return f"only {len(counts)} of {replicas} replicas served tool calls: {dict(counts)}"
    return None


def _reassemble_stream_tool_calls(sse_body: str) -> tuple[dict, object]:
    """Fold SSE chat chunks into (message-like dict, final finish_reason)."""

    calls: dict[int, dict[str, list[str]]] = {}
    finish_reason: object = None
    for line in sse_body.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[len("data: ") :])
        for choice in chunk.get("choices", ()):
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            for delta_call in (choice.get("delta") or {}).get("tool_calls") or ():
                index = delta_call.get("index")
                if not isinstance(index, int):
                    return {"tool_calls": None}, "missing tool_call delta index"
                slot = calls.setdefault(index, {"name": [], "arguments": []})
                function = delta_call.get("function") or {}
                if function.get("name"):
                    slot["name"].append(function["name"])
                if function.get("arguments"):
                    slot["arguments"].append(function["arguments"])
    message = {
        "tool_calls": [
            {
                "function": {
                    "name": "".join(slot["name"]),
                    "arguments": "".join(slot["arguments"]),
                }
            }
            for _, slot in sorted(calls.items())
        ]
        or None
    }
    return message, finish_reason


def _post_chat(
    payload: dict,
    *,
    timeout_s: float = 600.0,
    include_request_id: bool = False,
) -> tuple[int, object] | tuple[int, object, str | None]:
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}/v1/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            status = response.status
            request_id = response.headers.get("x-request-id")
    except urllib.error.HTTPError as error:
        result: tuple[int, object] = (error.code, error.read().decode("utf-8", "replace"))
        request_id = error.headers.get("x-request-id") if error.headers else None
    else:
        result = (status, body if payload.get("stream") else json.loads(body))
    if REQUEST_LOG is not None:
        # Each call writes one complete JSONL row; the built-in gates use
        # synthetic inputs and never include authentication credentials.
        row = (
            json.dumps(
                {
                    "request": payload,
                    "status": result[0],
                    "response": result[1],
                    "request_id": request_id,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        with REQUEST_LOG.open("a", encoding="utf-8") as output:
            output.write(row)
    return (*result, request_id) if include_request_id else result


def _tool_call_request(**overrides) -> dict:
    payload = {
        "model": SERVED_MODEL,
        "messages": [
            {"role": "system", "content": _TOOL_SYSTEM},
            {"role": "user", "content": _TOOL_USER},
        ],
        "tools": [_BASH_TOOL],
        "parallel_tool_calls": True,
        "max_tokens": 8192,
    }
    payload.update(overrides)
    return payload


def tool_calling(run_dir: Path) -> int:
    import concurrent.futures

    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"schema_version": 1, "cases": {}}
    failures: list[str] = []

    def record(name: str, error: str | None, detail: object = None) -> None:
        report["cases"][name] = {"passed": error is None, "error": error, "detail": detail}
        if error is not None:
            failures.append(f"{name}: {error}")

    # Case 1 (mini-swe-agent shape) fanned to 2x replicas concurrently, so the
    # placement log proves every replica serves tool calls, not just one.
    offset = _placement_offset(PLACEMENT_LOG)
    fan = 2 * REPLICAS
    with concurrent.futures.ThreadPoolExecutor(max_workers=fan) as pool:
        results = list(
            pool.map(
                lambda _: _post_chat(_tool_call_request(), include_request_id=True),
                range(fan),
            )
        )
    first_message: dict | None = None
    request_ids: set[str] = set()
    error = None
    for status, body, request_id in results:
        if isinstance(request_id, str) and request_id:
            request_ids.add(request_id)
        if status != 200 or not isinstance(body, dict):
            error = f"HTTP {status}: {str(body)[:300]}"
            break
        choice = body["choices"][0]
        case_error = _validate_tool_call_message(choice["message"], choice["finish_reason"])
        if case_error is not None:
            error = case_error
            break
        first_message = choice["message"]
    if error is None and len(request_ids) != fan:
        error = f"received {len(request_ids)} unique x-request-id headers, expected {fan}"
    record(f"auto_tool_call_x{fan}", error)
    if len(request_ids) == fan:
        counts = _wait_for_placement_counts(
            PLACEMENT_LOG,
            offset,
            expected_requests=fan,
            request_ids=request_ids,
        )
        placement_error = _tool_placement_error(
            counts,
            expected_requests=fan,
            replicas=REPLICAS,
        )
    else:
        counts = Counter()
        placement_error = (
            f"cannot correlate placements: received {len(request_ids)} unique "
            f"x-request-id headers, expected {fan}"
        )
    record(
        "all_replicas_served_tool_calls",
        placement_error,
        dict(sorted(counts.items())),
    )

    # Case 2: the agent loop's second turn (assistant tool_calls + tool result).
    if first_message is not None:
        call = first_message["tool_calls"][0]
        call_id = call.get("id") or "call_0"
        status, body = _post_chat(
            _tool_call_request(
                messages=[
                    {"role": "system", "content": _TOOL_SYSTEM},
                    {"role": "user", "content": _TOOL_USER},
                    {
                        "role": "assistant",
                        "content": first_message.get("content"),
                        "reasoning_content": first_message.get("reasoning_content"),
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": dict(call["function"]),
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": "README.md\nsrc\ntests\n",
                    },
                ]
            )
        )
        if status != 200 or not isinstance(body, dict):
            record("tool_result_turn", f"HTTP {status}: {str(body)[:300]}")
        else:
            choice = body["choices"][0]
            record(
                "tool_result_turn",
                _validate_tool_call_message(choice["message"], choice["finish_reason"]),
            )
    else:
        record("tool_result_turn", "skipped: no tool call from case 1")

    # Case 3: streaming deltas carry the same call.
    status, body = _post_chat(_tool_call_request(stream=True))
    if status != 200 or not isinstance(body, str):
        record("streamed_tool_call", f"HTTP {status}: {str(body)[:300]}")
    else:
        message, finish_reason = _reassemble_stream_tool_calls(body)
        record("streamed_tool_call", _validate_tool_call_message(message, finish_reason))

    # Case 4: explicit reasoning_effort must think AND still emit the call.
    status, body = _post_chat(_tool_call_request(reasoning_effort="high"), timeout_s=1200.0)
    if status != 200 or not isinstance(body, dict):
        record("thinking_tool_call", f"HTTP {status}: {str(body)[:300]}")
    else:
        choice = body["choices"][0]
        error = _validate_tool_call_message(choice["message"], choice["finish_reason"])
        if error is None and not choice["message"].get("reasoning_content"):
            error = "reasoning_content is empty under reasoning_effort=high"
        record("thinking_tool_call", error)

    # Case 5: omitted effort uses the official thinking-high default.
    status, body = _post_chat(
        {
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "max_tokens": 8192,
        }
    )
    if status != 200 or not isinstance(body, dict):
        record("thinking_high_default", f"HTTP {status}: {str(body)[:300]}")
    else:
        message = body["choices"][0]["message"]
        error = None
        if not message.get("content"):
            error = "empty content"
        elif not message.get("reasoning_content"):
            error = "reasoning_content absent with thinking-high default"
        elif body["choices"][0].get("finish_reason") != "stop":
            error = "default thinking did not reach a final answer"
        record("thinking_high_default", error)

    # An explicit low effort overrides the high default and completes.
    status, body = _post_chat(
        {
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "reasoning_effort": "low",
            "max_tokens": 512,
        }
    )
    error = None
    if status != 200 or not isinstance(body, dict):
        error = f"HTTP {status}: {str(body)[:300]}"
    else:
        choice = body["choices"][0]
        message = choice["message"]
        if not message.get("content") or choice.get("finish_reason") != "stop":
            error = "low-effort request did not complete a visible answer"
        elif not message.get("reasoning_content"):
            error = "low-effort request produced no reasoning"
    record("explicit_low_effort", error)

    report["passed"] = not failures
    (run_dir / "tool-calling.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for line in failures:
        print(f"tool-calling: {line}", file=sys.stderr)
    print(f"tool-calling: {'PASS' if not failures else 'FAIL'} ({len(report['cases'])} cases)")
    return 1 if failures else 0


# --- Vision gate. All replicas must answer OpenAI image parts through
# Kairyu (image_input_policy admission -> vLLM deepseek_v41 encoder ->
# SM120 sparse-MLA prefill), not just the replica that happened to serve
# `run.sh up`'s single probe.
_PROBE_IMAGE_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAT0lEQVR42u3PQQkAAAgEsItz/fMY"
    "xgi+hcEKLNO+FgEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQGB"
    "ywLzk8EPlvGqjQAAAABJRU5ErkJggg=="
)


def _image_request(case: int, *, max_tokens: int) -> dict:
    return {
        "model": SERVED_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_PROBE_IMAGE_PNG_BASE64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Vision case {case}: what single color fills this image? "
                            "Answer with one word."
                        ),
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
    }


def _validate_image_message(message: object) -> str | None:
    """Reject image responses that do not name the probe colour (None = valid).

    The probe image is solid red, so a correct answer contains "red". Checking
    for non-empty content alone let batched MTP + prefix-caching corruption
    ("ductduct...", vllm-project/vllm#53912) pass this gate once.
    """

    if not isinstance(message, dict):
        return f"message is {message!r}"
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return f"empty content for an image request ({message!r})"
    if "red" not in content.lower():
        return f"image answer does not name the probe colour: {content.strip()[:80]!r}"
    return None


def vision(run_dir: Path) -> int:
    import concurrent.futures

    config = SPEC["verification"]["vision"]
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"schema_version": 1, "cases": {}}
    failures: list[str] = []

    def record(name: str, error: str | None, detail: object = None) -> None:
        report["cases"][name] = {"passed": error is None, "error": error, "detail": detail}
        if error is not None:
            failures.append(f"{name}: {error}")

    # Row-unique prompts fanned to requests_per_replica x replicas at once, so
    # least-outstanding placement must reach every replica.
    offset = _placement_offset(PLACEMENT_LOG)
    fan = int(config["requests_per_replica"]) * REPLICAS
    max_tokens = int(config["max_tokens"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=fan) as pool:
        results = list(
            pool.map(
                lambda case: _post_chat(
                    _image_request(case, max_tokens=max_tokens),
                    timeout_s=1200.0,
                    include_request_id=True,
                ),
                range(fan),
            )
        )
    request_ids: set[str] = set()
    answers: list[str] = []
    error = None
    for status, body, request_id in results:
        if isinstance(request_id, str) and request_id:
            request_ids.add(request_id)
        if status != 200 or not isinstance(body, dict):
            error = f"HTTP {status}: {str(body)[:300]}"
            break
        message = body["choices"][0]["message"]
        case_error = _validate_image_message(message)
        if body["choices"][0].get("finish_reason") != "stop":
            case_error = "image response did not reach a completed answer"
        if case_error is not None:
            error = case_error
            break
        answers.append(message["content"].strip()[:80])
    if error is None and len(request_ids) != fan:
        error = f"received {len(request_ids)} unique x-request-id headers, expected {fan}"
    record(f"image_answer_x{fan}", error, answers)
    if len(request_ids) == fan:
        counts = _wait_for_placement_counts(
            PLACEMENT_LOG,
            offset,
            expected_requests=fan,
            request_ids=request_ids,
        )
        placement_error = _tool_placement_error(
            counts,
            expected_requests=fan,
            replicas=REPLICAS,
        )
    else:
        counts = Counter()
        placement_error = (
            f"cannot correlate placements: received {len(request_ids)} unique "
            f"x-request-id headers, expected {fan}"
        )
    record("all_replicas_served_images", placement_error, dict(sorted(counts.items())))

    report["passed"] = not failures
    (run_dir / "vision.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for line in failures:
        print(f"vision: {line}", file=sys.stderr)
    print(f"vision: {'PASS' if not failures else 'FAIL'} ({len(report['cases'])} cases)")
    return 1 if failures else 0


def _served_config_sha256() -> str:
    digest = hashlib.sha256()
    for path in SERVED_CONFIG_FILES:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_evidence() -> dict:
    """Reject measuring a stale runtime or a different Compose command."""
    import yaml

    project = SPEC["environment"].replace(".", "-")
    names = [f"{project}-deepseek-{i}-1" for i in range(REPLICAS)]
    inspected = json.loads(subprocess.check_output(["docker", "inspect", *names], text=True))
    compose = yaml.safe_load((HERE / "compose.yaml").read_text())
    rows = []
    for i, container in enumerate(inspected):
        command = compose["services"][f"deepseek-{i}"]["command"]
        if container["Image"] != SPEC["vllm"]["image_id"]:
            raise ValueError("Running vLLM image differs from the evidence pin")
        if container["Config"]["Cmd"] != command:
            raise ValueError("Running L1 command differs from compose.yaml")
        environment = dict(item.split("=", 1) for item in container["Config"]["Env"])
        expected_env = compose["services"][f"deepseek-{i}"]["environment"]
        for key, value in expected_env.items():
            if environment.get(key) != str(value):
                raise ValueError(f"Running L1 environment differs at {key}")
        rows.append(
            {
                "name": container["Name"],
                "image_id": container["Image"],
                "command": command,
                "started_at": container["State"]["StartedAt"],
                "devices": container["HostConfig"]["DeviceRequests"],
            }
        )
    gateway = f"{project}-kairyu-1"
    mounted = subprocess.check_output(["docker", "exec", gateway, "cat", "/etc/kairyu/kairyu.yaml"])
    if mounted != (HERE / "kairyu.yaml").read_bytes():
        raise ValueError("Gateway is serving a different mounted deployment")
    return {"replicas": rows, "gateway_config_sha256": hashlib.sha256(mounted).hexdigest()}


def reasoning(run_dir: Path) -> int:
    """Measure time to visible content and completion with real thinking budgets."""
    import urllib.request

    reports = []
    for effort in (None, "low", "high", "max"):
        payload = {
            "model": SERVED_MODEL,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 8192,
            "messages": [
                {"role": "user", "content": "What is 17 * 19? Reply with only the integer."}
            ],
        }
        if effort is not None:
            payload["reasoning_effort"] = effort
        request = urllib.request.Request(
            f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.monotonic()
        first_content = None
        content = trace = ""
        finish = usage = None
        done = False
        with urllib.request.urlopen(request, timeout=1200) as response:
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
        passed = (
            done
            and finish == "stop"
            and content.strip() == "323"
            and bool(trace)
        )
        reports.append(
            {
                "effort": effort or "default",
                "passed": passed,
                "time_to_content_s": first_content,
                "total_s": time.monotonic() - start,
                "content": content,
                "reasoning_content": trace,
                "finish_reason": finish,
                "usage": usage,
                "done": done,
            }
        )
        (run_dir / "reasoning.json").write_text(json.dumps(reports, indent=2) + "\n")
    return int(not all(row["passed"] for row in reports))


def _upstream_active_requests() -> float:
    """Read the L1 scheduler gauges without publishing an extra host port."""
    import re

    container = SPEC["environment"].replace(".", "-") + "-deepseek-0-1"
    metrics = subprocess.check_output(
        [
            "docker",
            "exec",
            container,
            "python3",
            "-c",
            "import urllib.request; print(urllib.request.urlopen("
            '"http://127.0.0.1:8000/metrics", timeout=5).read().decode())',
        ],
        text=True,
        timeout=10,
    )
    total = 0.0
    for gauge in ("running", "waiting"):
        values = re.findall(
            rf"^vllm:num_requests_{gauge}\{{[^\n]*\}} ([0-9.eE+-]+)$", metrics, re.M
        )
        if not values:
            raise ValueError(f"Missing L1 {gauge} request gauge")
        total += sum(float(value) for value in values)
    return total


def cancellation(run_dir: Path) -> int:
    """A client disconnect releases its slot and permits another request."""
    import re
    import urllib.request

    api = f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}"
    payload = {
        "model": SERVED_MODEL,
        "stream": True,
        "max_tokens": 8192,
        "ignore_eos": True,
        "messages": [{"role": "user", "content": "Count upwards from one, one number per line."}],
    }
    request = urllib.request.Request(
        api + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    saw_delta = False
    upstream_observed = False
    with urllib.request.urlopen(request, timeout=1200) as response:
        for line in response:
            if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
                chunk = json.loads(line[6:])
                if any(
                    (
                        c.get("delta", {}).get("content")
                        or c.get("delta", {}).get("reasoning_content")
                    )
                    for c in chunk.get("choices", [])
                ):
                    saw_delta = True
                    break
        # Allow the periodic L1 metrics publisher to observe the long request
        # before disconnecting; otherwise a stale zero could hide leaked work.
        deadline = time.monotonic() + 15
        while saw_delta and time.monotonic() < deadline:
            if _upstream_active_requests() > 0:
                upstream_observed = True
                break
            time.sleep(0.5)
    deadline = time.monotonic() + 30
    released = False
    upstream_released = False
    while time.monotonic() < deadline:
        with urllib.request.urlopen(api + "/metrics", timeout=5) as response:
            metrics = response.read().decode()
        values = re.findall(r"^kairyu_replica_outstanding\{[^\n]*\} ([0-9.]+)$", metrics, re.M)
        released = bool(values) and all(float(value) == 0 for value in values)
        upstream_released = _upstream_active_requests() == 0
        if released and upstream_released:
            break
        time.sleep(0.25)
    status, answer = _post_chat(
        {
            "model": SERVED_MODEL,
            "max_tokens": 512,
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "Reply OK."}],
        }
    )
    usable = (
        status == 200
        and isinstance(answer, dict)
        and answer["choices"][0].get("finish_reason") == "stop"
        and bool(answer["choices"][0]["message"].get("content"))
    )
    report = {
        "received_model_delta": saw_delta,
        "slot_released": released,
        "upstream_request_observed": upstream_observed,
        "upstream_request_released": upstream_released,
        "follow_up_completed": usable,
        "passed": bool(
            saw_delta and released and upstream_observed and upstream_released and usable
        ),
    }
    (run_dir / "cancellation.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(not report["passed"])


def long_context(run_dir: Path) -> int:
    """Long-input retrieval smokes with actual server-reported token counts."""
    import secrets

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(
        str(ENVIRONMENT_STORAGE / "models" / SERVED_MODEL / "tokenizer.json")
    )
    filler = " filler"
    if len(tokenizer.encode(filler * 100, add_special_tokens=False).ids) != 100:
        raise ValueError("Pinned long-context filler is not one token per repetition")
    reports = []
    for target in (32768, 131072, 262144, SPEC["model"]["max_context_tokens"] - 8704):
        key = "K" + secrets.token_hex(12).upper()
        prefix = "Read this log and recover its archive key.\n"
        needle = f"\nThe archive key is {key}.\n"
        suffix = "\nReturn only the archive key."
        overhead = len(tokenizer.encode(prefix + needle + suffix, add_special_tokens=False).ids)
        repeats = target - overhead
        prompt = (
            prefix + filler * (repeats // 2) + needle + filler * (repeats - repeats // 2) + suffix
        )
        measured_input = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
        start = time.monotonic()
        status, body = _post_chat(
            {
                "model": SERVED_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8192,
            },
            timeout_s=1800,
        )
        choice = body.get("choices", [{}])[0] if isinstance(body, dict) else {}
        content = choice.get("message", {}).get("content")
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        actual = usage.get("prompt_tokens", 0)
        passed = (
            status == 200
            and choice.get("finish_reason") == "stop"
            and isinstance(content, str)
            and content.strip() == key
            and actual >= measured_input
            and actual + 8192 <= SPEC["model"]["max_context_tokens"]
        )
        reports.append(
            {
                "expected_key": key,
                "target_input_tokens": target,
                "raw_input_tokens": measured_input,
                "usage": usage,
                "content": content,
                "finish_reason": choice.get("finish_reason"),
                "total_s": time.monotonic() - start,
                "passed": passed,
            }
        )
        (run_dir / "long-context.json").write_text(json.dumps(reports, indent=2) + "\n")
        if not passed:
            return 1
    return 0


def main() -> None:
    global REQUEST_LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "verification",
        choices=(
            "serving",
            "tool-calling",
            "vision",
            "reasoning",
            "cancellation",
            "long-context",
            "list",
        ),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--no-start", action="store_true")
    args = parser.parse_args()
    if args.verification == "list":
        rows = ",".join(str(c) for c in SPEC["verification"]["serving"]["concurrency"])
        print(
            f"serving       fixed 8K-input/256-output TTFT and throughput at c={rows} "
            "plus the per-row replica placement gate"
        )
        print(
            "tool-calling  OpenAI bash-tool agent contract (auto call, tool-result turn, "
            "streaming, thinking high default, explicit non-thinking) on every replica"
        )
        print(
            "vision        OpenAI image-part requests answered with visible content on "
            "every replica"
        )
        print(
            "reasoning     default/high/low/max, completed answers and time to visible content"
        )
        print("cancellation  disconnect releases the replica slot; a follow-up completes")
        print("long-context  32K/128K/256K/near-1M retrieval smokes")
        return

    _ensure_environment(args.no_start)
    run_dir = RESULTS_ROOT / (args.run_id or _run_id())
    run_dir.mkdir(parents=True, exist_ok=True)
    REQUEST_LOG = run_dir / "requests.jsonl"
    runtime = runtime_evidence()
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "started_at": datetime.now(UTC).isoformat(),
        "requested": args.verification,
        "served_config_sha256": _served_config_sha256(),
        "spec": SPEC,
        "runtime": runtime,
    }
    (run_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    target = {
        "serving": serving,
        "tool-calling": tool_calling,
        "vision": vision,
        "reasoning": reasoning,
        "cancellation": cancellation,
        "long-context": long_context,
    }[args.verification]
    try:
        code = target(run_dir)
    except Exception as error:
        print(f"{args.verification} failed: {error}", file=sys.stderr)
        code = 1
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["exit_codes"] = {args.verification: code}
    (run_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"artifacts: {run_dir}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
