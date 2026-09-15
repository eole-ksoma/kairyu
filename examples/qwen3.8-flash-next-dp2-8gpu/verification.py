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
# Files whose bytes define the served configuration, including the
# example-local chat template (official template + L3 effort alias).
SERVED_CONFIG_FILES = (
    HERE / "example.json",
    HERE / "compose.yaml",
    HERE / "kairyu.yaml",
    HERE / SPEC["model"]["chat_template"],
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


def _serving_dataset(
    path: Path, requests: int, approximate_tokens: int, *, namespace: str
) -> None:
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
            str(ROOT / "verification/l1/performance/serving_bench.py"),
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
            "86400",
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
            "properties": {
                "command": {"type": "string", "description": "The command to run."}
            },
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
        if not isinstance(arguments.get("command"), str) or not arguments["command"]:
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
            f"recorded {total} correlated placements, expected {expected_requests}: "
            f"{dict(counts)}"
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
        chunk = json.loads(line[len("data: "):])
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

    url = (
        f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}"
        "/v1/chat/completions"
    )
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
                        "content": None,
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

    # Case 5: ordinary chat stays non-thinking by default.
    status, body = _post_chat(
        {
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "max_tokens": 512,
        }
    )
    if status != 200 or not isinstance(body, dict):
        record("default_direct_chat", f"HTTP {status}: {str(body)[:300]}")
    else:
        message = body["choices"][0]["message"]
        error = None
        if not message.get("content"):
            error = "empty content"
        elif message.get("reasoning_content"):
            error = "reasoning_content present without reasoning_effort"
        record("default_direct_chat", error)

    report["passed"] = not failures
    (run_dir / "tool-calling.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for line in failures:
        print(f"tool-calling: {line}", file=sys.stderr)
    print(f"tool-calling: {'PASS' if not failures else 'FAIL'} ({len(report['cases'])} cases)")
    return 1 if failures else 0

# --- Vision gate. Both replicas must answer OpenAI image parts through
# Kairyu (image_input_policy admission -> vLLM Qwen vision encoder -> TP4
# prefill), not just the replica that happened to serve `run.sh up`'s single
# probe.
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
                        "image_url": {
                            "url": f"data:image/png;base64,{_PROBE_IMAGE_PNG_BASE64}"
                        },
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verification", choices=("serving", "tool-calling", "vision", "list"))
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
            "streaming, thinking, non-thinking default) on every replica"
        )
        print(
            "vision        OpenAI image-part requests answered with visible content on "
            "every replica"
        )
        return

    _ensure_environment(args.no_start)
    run_dir = RESULTS_ROOT / (args.run_id or _run_id())
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "started_at": datetime.now(UTC).isoformat(),
        "requested": args.verification,
        "served_config_sha256": _served_config_sha256(),
        "spec": SPEC,
    }
    (run_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    target = {"serving": serving, "tool-calling": tool_calling, "vision": vision}[
        args.verification
    ]
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
