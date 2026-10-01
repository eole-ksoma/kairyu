#!/usr/bin/env python3
"""GPU gates for checklist-verified answers.

Every gate has a time budget and writes its per-request evidence (latency,
tokens, tok/s, guarantee flag, reason, attempts, requirement count) to
model-volumes/<environment>/results/<gate>-<UTC>.json.

  l1            DeepSeek grammar-constrained JSON on every DP rank (thinking and
                chat) and System One on each OpenJev replica
  calibrate     tau_hi on InFoBench expert labels (calibrate.py)
  calibrate-g1  can per-claim G1 be calibrated (RAGTruth / PRM800K / FEVER)? today:
                no, so G1 is advisory (calibrate_g1.py, VCO-D11)
  requirements  extracted checklists cover InFoBench's gold decomposed questions
  repair        constraint-heavy requests: repairs happen and every guaranteed
                answer meets the stated constraint (independent check)
  structured    a caller json_schema survives drafting and repair
  fallback      one OpenJev down: still guaranteed; both down: 200, unverified,
                reason judge_unavailable
  serving       end-to-end latency, tokens and guarantee rate at c1/c4/c8/c16
                (all of the above use kairyu-verified-always)
  routing       Jev routes accuracy-critical conversations to the verified DAG
  think-route   everyday requests stream from deepseek_think at the default effort
  effort        the caller's effort reaches every DeepSeek step on both routes
  implicit      situational requirements are extracted and kept only when expected
  serving-routed  kairyu-verified under load: route mix, latency, tokens per route
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import control  # noqa: E402

SPEC = control.SPEC
INFOBENCH_URL = (
    "https://huggingface.co/datasets/kqsong/InFoBench/resolve/"
    "cef03a2830944bfb0d201107895ddd0e0e90bf0e/InfoBench.json"
)
INFOBENCH_SHA256 = "66a2ee8d70208a7a879a17f471ba0ffd097b698265ea4796be99a273fdcc6559"


def _env() -> dict[str, str]:
    return control._compose_env()


def _api(env: dict[str, str]) -> str:
    return f"http://127.0.0.1:{env['API_PORT']}"


def _results_dir() -> Path:
    path = control.environment_storage() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(gate: str, payload: dict) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = _results_dir() / f"{gate}-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"evidence: {path}", flush=True)
    return path


def _route(trace: dict | None, model: str) -> tuple[str, float | None]:
    """The profile that served the request and Jev's P(VERIFIED), from the trace."""

    if model == control.ALWAYS_MODEL:
        return "verified (always)", None
    events = (trace or {}).get("events") or []
    judge = next((event for event in events if event.get("node") == "profile_judge"), None)
    if judge is None:
        return "unknown", None
    detail = judge.get("detail") or {}
    verdict = detail.get("verdict")
    p_verified = detail.get("p_VERIFIED")
    if verdict is None:
        return f"deepseek_think (fallback: {detail.get('fallback')})", p_verified
    return ("verified" if verdict == "primary" else verdict), p_verified


def _efforts(trace: dict | None) -> list[str | None]:
    """The reasoning effort of every DeepSeek generation in the trace."""

    return [
        (event.get("detail") or {}).get("reasoning_effort")
        for event in (trace or {}).get("events") or []
        if event.get("kind") == "generation"
        and event.get("worker") == "deepseek"
        and event.get("status") == "success"
    ]


def _post_stream(
    url: str, payload: dict, headers: dict, timeout_s: float
) -> tuple[dict, float | None]:
    """Stream one chat request; returns an assembled body and the time to first token."""

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
    )
    started = time.monotonic()
    first: float | None = None
    content, reasoning = [], []
    body: dict = {"choices": [{"message": {}}]}
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: {"):
                continue
            chunk = json.loads(line[len("data: ") :])
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if (delta.get("content") or delta.get("reasoning_content")) and first is None:
                    first = time.monotonic() - started
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
            for key in ("usage", "kairyu_verification", "kairyu_trace_v2", "kairyu_route"):
                if chunk.get(key) is not None:
                    body[key] = chunk[key]
    body["choices"][0]["message"] = {
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
    }
    return body, first


def chat(
    env: dict[str, str],
    content: str | None = None,
    *,
    messages: list[dict] | None = None,
    model: str = control.ALWAYS_MODEL,
    trace: bool = False,
    stream: bool = False,
    timeout_s: float = 1800,
    **extra,
) -> dict:
    """One chat request; returns the per-request evidence row."""

    payload = control.verified_request(content or "", model=model, **extra)
    if messages is not None:
        payload["messages"] = messages
    headers = {"X-Kairyu-Trace": "1"} if trace else {}
    url = f"{_api(env)}/v1/chat/completions"
    started = time.monotonic()
    ttft = None
    try:
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            body, ttft = _post_stream(url, payload, headers, timeout_s)
        else:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json", **headers},
            )
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                body = json.loads(response.read())
        status = 200
    except urllib.error.HTTPError as error:
        body = {"error": error.read().decode("utf-8", "replace")[:500]}
        status = error.code
    except (OSError, urllib.error.URLError) as error:
        body = {"error": repr(error)}
        status = 0
    elapsed = time.monotonic() - started
    usage = body.get("usage") or {}
    report = body.get("kairyu_verification") or {}
    message = (body.get("choices") or [{}])[0].get("message") or {}
    output_tokens = usage.get("orchestration_output_tokens") or usage.get("completion_tokens") or 0
    trace_body = body.get("kairyu_trace_v2")
    route, p_verified = _route(trace_body, model)
    verified_route = route.startswith("verified")
    return {
        "status": status,
        "seconds": round(elapsed, 2),
        "ttft_s": round(ttft, 2) if ttft is not None else None,
        "model": model,
        "route": route,
        "p_verified": p_verified,
        "efforts": _efforts(trace_body),
        "public_completion_tokens": usage.get("completion_tokens"),
        "orchestration_input_tokens": usage.get("orchestration_input_tokens")
        or usage.get("prompt_tokens"),
        "orchestration_output_tokens": output_tokens,
        "orchestration_output_tok_per_s": round(output_tokens / elapsed, 1) if elapsed else None,
        "guaranteed": report.get("guaranteed"),
        "reason": report.get("reason"),
        "attempts": report.get("attempts"),
        "requirements": report.get("requirements") or [],
        "has_verification": "kairyu_verification" in body,
        "content": message.get("content") or "",
        "reasoning_chars": len(message.get("reasoning_content") or ""),
        "error": body.get("error"),
        "verification_error": (
            control.verification_error(body) if status == 200 and verified_route else None
        ),
    }


def _summary(rows: list[dict]) -> dict:
    ok = [row for row in rows if row["status"] == 200]
    seconds = sorted(row["seconds"] for row in ok)
    out_tokens = sum(row["orchestration_output_tokens"] or 0 for row in ok)
    in_tokens = sum(row["orchestration_input_tokens"] or 0 for row in ok)
    return {
        "requests": len(rows),
        "ok": len(ok),
        "guaranteed": sum(1 for row in ok if row["guaranteed"]),
        "reasons": {
            reason: sum(1 for row in ok if row["reason"] == reason)
            for reason in sorted({str(row["reason"]) for row in ok if not row["guaranteed"]})
        },
        "repaired": sum(1 for row in ok if (row["attempts"] or 0) > 1),
        "latency_p50_s": statistics.median(seconds) if seconds else None,
        "latency_p95_s": seconds[max(0, int(len(seconds) * 0.95) - 1)] if seconds else None,
        "orchestration_input_tokens": in_tokens,
        "orchestration_output_tokens": out_tokens,
    }


def _print_rows(rows: list[dict]) -> None:
    for index, row in enumerate(rows):
        print(
            f"  #{index:02d} status={row['status']} {row['seconds']:7.1f}s "
            f"ttft={row.get('ttft_s')} route={row['route']} p_verified={row.get('p_verified')} "
            f"in={row['orchestration_input_tokens']} "
            f"out={row['orchestration_output_tokens']} "
            f"({row['orchestration_output_tok_per_s']} tok/s) "
            f"guaranteed={row['guaranteed']} reason={row['reason']} "
            f"attempts={row['attempts']} requirements={len(row['requirements'])}",
            flush=True,
        )


def _run_concurrently(env: dict[str, str], prompts: list[dict], concurrency: int) -> list[dict]:
    def one(prompt: dict) -> dict:
        extra = {key: value for key, value in prompt.items() if key != "content"}
        return chat(env, prompt.get("content"), **extra)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, prompts))


def infobench(limit: int, seed: int) -> list[dict]:
    path = control.environment_storage() / "calibration" / "InfoBench.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        with urllib.request.urlopen(INFOBENCH_URL, timeout=120) as response:
            path.write_bytes(response.read())
    import hashlib

    if hashlib.sha256(path.read_bytes()).hexdigest() != INFOBENCH_SHA256:
        raise SystemExit(f"{path} does not match the pinned InFoBench revision")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def _infobench_content(row: dict) -> str:
    return "\n\n".join(part for part in (row["instruction"], row.get("input")) if part)


class Deadline:
    def __init__(self, gate: str, budget_s: float) -> None:
        self.gate = gate
        self.end = time.monotonic() + budget_s
        self.budget_s = budget_s

    def check(self) -> None:
        if time.monotonic() > self.end:
            raise SystemExit(f"{self.gate}: time budget of {self.budget_s:.0f} s exceeded")


def gate_l1(env: dict[str, str]) -> None:
    started = time.monotonic()
    control.validate_ready(_api(env))
    control._validate_deepseek(f"http://127.0.0.1:{env['DEEPSEEK_L1_PORT']}")
    control._validate_openjev_replicas()
    print(f"l1: PASS in {time.monotonic() - started:.0f} s", flush=True)
    _write("l1", {"passed": True, "seconds": time.monotonic() - started})


def gate_calibrate(_env: dict[str, str]) -> None:
    subprocess.run([sys.executable, str(HERE / "calibrate.py")], check=True)


def gate_calibrate_g1(_env: dict[str, str]) -> None:
    subprocess.run([sys.executable, str(HERE / "calibrate_g1.py")], check=True)


_COVERAGE_PROMPT = """For each GOLD question below, decide whether the CHECKLIST contains a \
condition that requires what the question checks (alone or together with other conditions). \
Answer as JSON {{"covered": [true/false per gold question, in order]}}.
GOLD QUESTIONS:
{gold}
CHECKLIST:
{checklist}"""


def _build_key() -> str:
    """The served code and configuration: commit, uncommitted diff, example configs."""

    root = HERE.parents[1]
    digest = hashlib.sha256()
    for command in (["git", "rev-parse", "HEAD"], ["git", "diff", "HEAD"]):
        digest.update(subprocess.run(command, cwd=root, capture_output=True, check=True).stdout)
    for name in ("kairyu.yaml", "verified.yaml", "verified-always.yaml"):
        digest.update((HERE / name).read_bytes())
    return digest.hexdigest()[:12]


def gate_requirements(env: dict[str, str], *, count: int = 40, budget_s: float = 7200) -> None:
    """Sufficiency: the judged checklist covers InFoBench's gold questions."""

    deadline = Deadline("requirements", budget_s)
    rows = infobench(count, seed=1)
    # Answers are kept so a failed coverage pass does not repeat generation,
    # but only for the code and configuration that produced them.
    answers = _results_dir() / f"requirements-answers-{count}-{_build_key()}.json"
    if answers.is_file():
        results = json.loads(answers.read_text(encoding="utf-8"))
    else:
        prompts = [{"content": _infobench_content(row)} for row in rows]
        results = _run_concurrently(env, prompts, concurrency=8)
        answers.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    deadline.check()
    l1 = f"http://127.0.0.1:{env['DEEPSEEK_L1_PORT']}"
    coverage = []
    for row, result in zip(rows, results, strict=True):
        extracted = [item for item in result["requirements"] if not item["id"].startswith("G")]
        if result["status"] != 200 or not extracted:
            coverage.append(
                {"id": row["id"], "covered": None, "gold": len(row["decomposed_questions"])}
            )
            continue
        checklist = "\n".join(f"- {item['proposition']}" for item in extracted)
        gold = "\n".join(f"{n}. {q}" for n, q in enumerate(row["decomposed_questions"], 1))
        body = control.post_json(
            f"{l1}/v1/chat/completions",
            {
                "model": SPEC["deepseek"]["served_name"],
                "messages": [
                    {
                        "role": "user",
                        "content": _COVERAGE_PROMPT.format(gold=gold, checklist=checklist),
                    }
                ],
                "max_tokens": 4096,
                "temperature": 0.0,
                # V4.1 thinks unless told not to; the verdict must be content.
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "coverage",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["covered"],
                            "properties": {
                                "covered": {"type": "array", "items": {"type": "boolean"}}
                            },
                        },
                    },
                },
            },
            timeout_s=600,
        )
        covered = json.loads(body["choices"][0]["message"]["content"]).get("covered") or []
        coverage.append(
            {
                "id": row["id"],
                "gold": len(row["decomposed_questions"]),
                "covered": sum(1 for value in covered if value is True),
                "requirements": len(extracted),
                "padded": sum(1 for item in extracted if item["id"].startswith("P_")),
            }
        )
    judged = [entry for entry in coverage if entry["covered"] is not None]
    recall = sum(entry["covered"] for entry in judged) / max(1, sum(e["gold"] for e in judged))
    summary = {**_summary(results), "gold_recall": round(recall, 4), "judged": len(judged)}
    _print_rows(results)
    print(json.dumps(summary, indent=2), flush=True)
    passed = recall >= 0.9 and len(judged) == len(rows)
    _write(
        "requirements",
        {"passed": passed, "summary": summary, "coverage": coverage, "rows": results},
    )
    print(f"requirements: {'PASS' if passed else 'FAIL'} (gold recall {recall:.3f}, gate 0.90)")
    if not passed:
        raise SystemExit(1)


# (request, independent deterministic check of the stated constraint)
_CONSTRAINED = [
    (
        "Describe the water cycle in exactly three sentences, each ending with a period.",
        lambda text: len(re.findall(r"[^.!?]+\.", text.strip())) == 3 and "?" not in text,
    ),
    (
        "List five fruits as a comma-separated line in lowercase, with no other text.",
        lambda text: bool(re.fullmatch(r"[a-z ]+(, ?[a-z ]+){4}", text.strip())),
    ),
    (
        "Explain what a hash table is in at most 200 characters.",
        lambda text: len(text.strip()) <= 200,
    ),
    (
        "Write a haiku about winter. Do not use the letter 'e' anywhere.",
        lambda text: "e" not in text.lower(),
    ),
    (
        "Give the boiling point of water at sea level in Celsius. Answer with only the number.",
        lambda text: text.strip() == "100",
    ),
    (
        "Name three planets. Answer in Japanese, one per line, with no other text.",
        lambda text: (
            len([line for line in text.strip().splitlines() if line.strip()]) == 3
            and not re.search(r"[A-Za-z]", text)
        ),
    ),
    (
        "Write one sentence about the moon that ends with the exact word END.",
        lambda text: text.strip().endswith("END"),
    ),
    (
        "Reply with a JSON object with keys name and age for a fictional person, and nothing else.",
        lambda text: _json_keys(text) == {"name", "age"},
    ),
]


def _json_keys(text: str) -> set[str] | None:
    try:
        value = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
    except ValueError:
        return None
    return set(value) if isinstance(value, dict) else None


def gate_repair(env: dict[str, str], *, rounds: int = 2, budget_s: float = 5400) -> None:
    deadline = Deadline("repair", budget_s)
    cases = _CONSTRAINED * rounds
    results = _run_concurrently(env, [{"content": text} for text, _check in cases], concurrency=8)
    deadline.check()
    findings = []
    for (text, check), result in zip(cases, results, strict=True):
        met = check(result["content"]) if result["status"] == 200 else False
        result["constraint_met"] = met
        if result["status"] != 200 or result["verification_error"]:
            findings.append(
                f"{text[:40]!r}: status={result['status']} {result['verification_error']}"
            )
        elif result["guaranteed"] and not met:
            findings.append(
                f"{text[:40]!r}: guaranteed but violates its constraint: {result['content'][:80]!r}"
            )
    summary = _summary(results)
    summary["guaranteed_constraint_violations"] = sum(
        1 for row in results if row["guaranteed"] and not row["constraint_met"]
    )
    _print_rows(results)
    print(json.dumps(summary, indent=2), flush=True)
    passed = not findings and summary["repaired"] >= 1
    if summary["repaired"] < 1:
        findings.append("no request went through a repair")
    _write("repair", {"passed": passed, "findings": findings, "summary": summary, "rows": results})
    for finding in findings:
        print(f"  finding: {finding}")
    print(f"repair: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


def gate_structured(env: dict[str, str], *, budget_s: float = 1800) -> None:
    deadline = Deadline("structured", budget_s)
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "city",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["city", "country", "population_estimate"],
                "properties": {
                    "city": {"type": "string"},
                    "country": {"type": "string"},
                    "population_estimate": {"type": "integer"},
                },
            },
        },
    }
    prompts = [
        {
            "content": "Give the largest city of Japan, its country and a population estimate.",
            "response_format": schema,
        },
        {
            "content": "Pick a European capital and describe it in exactly the requested JSON.",
            "response_format": schema,
        },
    ]
    results = _run_concurrently(env, prompts, concurrency=2)
    deadline.check()
    findings = []
    for result in results:
        keys = _json_keys(result["content"]) if result["status"] == 200 else None
        if keys != {"city", "country", "population_estimate"} or result["verification_error"]:
            findings.append(f"status={result['status']} content={result['content'][:120]!r}")
        elif not result["guaranteed"]:
            # A schema-valid answer must be judgeable, not rejected by a check
            # that misreads JSON.
            failing = [item["id"] for item in result["requirements"] if not item["passed"]]
            findings.append(f"schema-valid answer not guaranteed: failing {failing}")
    _print_rows(results)
    _write("structured", {"passed": not findings, "findings": findings, "rows": results})
    print(f"structured: {'PASS' if not findings else 'FAIL'} {findings}")
    if findings:
        raise SystemExit(1)


def _compose(env: dict[str, str], *arguments: str) -> None:
    control._compose(list(arguments), env=env)


def _wait_healthy(service: str, timeout_s: float = 1800) -> None:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        state = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Health.Status}}",
                f"{control.PROJECT}-{service}-1",
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if state == "healthy":
            return
        time.sleep(5)
    raise SystemExit(f"{service} did not become healthy again")


def gate_fallback(env: dict[str, str], *, budget_s: float = 5400) -> None:
    deadline = Deadline("fallback", budget_s)
    prompt = "In two sentences, explain why the sky looks blue."
    findings = []
    phases = {}
    try:
        _compose(env, "stop", "openjev-0")
        one_down = [chat(env, prompt) for _ in range(2)]
        phases["one_replica_down"] = one_down
        for row in one_down:
            if row["status"] != 200 or row["reason"] == "judge_unavailable":
                findings.append(f"one replica down: status={row['status']} reason={row['reason']}")
        _compose(env, "stop", "openjev-1")
        both_down = [chat(env, prompt) for _ in range(2)]
        phases["both_replicas_down"] = both_down
        # The routed model cannot judge either: it answers on the think route.
        routed = chat(env, prompt, model=control.ROUTED_MODEL, trace=True)
        phases["both_down_routed"] = [routed]
        if routed["status"] != 200 or not routed["route"].startswith("deepseek_think (fallback"):
            findings.append(
                f"routed with both down: status={routed['status']} route={routed['route']}"
            )
        for row in both_down:
            if (
                row["status"] != 200
                or row["guaranteed"] is not False
                or row["reason"] != "judge_unavailable"
            ):
                findings.append(
                    f"both down: status={row['status']} guaranteed={row['guaranteed']} "
                    f"reason={row['reason']}"
                )
            if not row["content"].strip():
                findings.append("both down: empty answer")
    finally:
        _compose(env, "start", "openjev-0", "openjev-1")
        _wait_healthy("openjev-0")
        _wait_healthy("openjev-1")
    recovered = chat(env, prompt)
    phases["recovered"] = [recovered]
    if recovered["status"] != 200 or recovered["reason"] == "judge_unavailable":
        findings.append(f"after restart: status={recovered['status']} reason={recovered['reason']}")
    deadline.check()
    for name, rows in phases.items():
        print(f"{name}:")
        _print_rows(rows)
    _write("fallback", {"passed": not findings, "findings": findings, "phases": phases})
    print(f"fallback: {'PASS' if not findings else 'FAIL'} {findings}")
    if findings:
        raise SystemExit(1)


def gate_serving(env: dict[str, str], *, budget_s: float = 14400) -> None:
    deadline = Deadline("serving", budget_s)
    plan = {1: 8, 4: 16, 8: 16, 16: 32}
    pool = infobench(sum(plan.values()), seed=2)
    offset = 0
    report = {}
    for concurrency, count in plan.items():
        prompts = [{"content": _infobench_content(row)} for row in pool[offset : offset + count]]
        offset += count
        started = time.monotonic()
        rows = _run_concurrently(env, prompts, concurrency)
        wall = time.monotonic() - started
        summary = _summary(rows)
        summary["wall_s"] = round(wall, 1)
        summary["requests_per_min"] = round(60 * len(rows) / wall, 2)
        summary["orchestration_output_tok_per_s"] = round(
            summary["orchestration_output_tokens"] / wall, 1
        )
        print(f"c{concurrency}:")
        _print_rows(rows)
        print(json.dumps(summary, indent=2), flush=True)
        report[f"c{concurrency}"] = {"summary": summary, "rows": rows}
        deadline.check()
    failures = [
        name
        for name, entry in report.items()
        if entry["summary"]["ok"] != entry["summary"]["requests"]
    ]
    _write("serving", {"passed": not failures, "failed_rows": failures, "report": report})
    print(f"serving: {'PASS' if not failures else 'FAIL'} (every request answered with a flag)")
    if failures:
        raise SystemExit(1)


DATASETS = HERE / "datasets"
ROUTED = control.ROUTED_MODEL
ALWAYS = control.ALWAYS_MODEL


def _dataset(name: str) -> list[dict]:
    return json.loads((DATASETS / name).read_text(encoding="utf-8"))


def _routing_probabilities() -> list[float | None]:
    """P(VERIFIED) for every routing-set conversation, through the served judge.

    The orchestrator is built from verified.yaml with the real System One
    backend pointed at both OpenJev replicas, so the request Jev reads is the
    one Kairyu sends in serving; no generation runs.
    """

    import asyncio

    from kairyu.dsl.loader import build_orchestrator, load_spec
    from kairyu.engine.mock import MockBackend
    from kairyu.engine.systemone import HTTPSystemOneBackend
    from kairyu.entrypoints.server.chat_service import validate_orchestration_chat_input
    from kairyu.entrypoints.server.protocol import ChatCompletionRequest
    from kairyu.orchestration.request import OrchestrationRequest
    from kairyu.sampling_params import SamplingParams

    async def judge_all() -> list[float | None]:
        backend = HTTPSystemOneBackend(
            base_urls=("http://127.0.0.1:8015", "http://127.0.0.1:8016"),
            upstream_model=SPEC["systemone"]["model"],
            max_concurrency=32,
            max_queue=1024,
            queue_wait_s=600,
        )
        orchestrator = build_orchestrator(
            load_spec(HERE / "verified.yaml"),
            engine_refs={"deepseek-v4.1-flash": MockBackend()},
            systemone_refs={SPEC["systemone"]["model"]: backend},
        )

        async def one(item: dict) -> float | None:
            chat = ChatCompletionRequest(model=ROUTED, messages=item["messages"])
            call = OrchestrationRequest(
                prompt=validate_orchestration_chat_input(chat).prompt,
                sampling_params=SamplingParams(max_tokens=1024),
            )
            judged = await orchestrator.judge_role_profile(call)
            return judged.role_profile_judge_event.metadata.get("p_VERIFIED")

        try:
            return await asyncio.gather(*(one(item) for item in _dataset("routing-set.json")))
        finally:
            await backend.shutdown()

    return asyncio.run(judge_all())


def _routes_verified(p: float, tau: float) -> bool:
    # VERIFIED when preferred (p >= tau) or simply more probable.
    return p >= tau or p > 0.5


def gate_routing(env: dict[str, str], *, budget_s: float = 1800) -> None:
    """Jev routes accuracy-critical conversations to the verified DAG.

    tau_route (prefer.min_probability) is the largest value whose miss rate
    on the calibration half (even items) stays below 10 %; the gate is the
    held-out half (odd items) at the configured tau.
    """

    deadline = Deadline("routing", budget_s)
    items = _dataset("routing-set.json")
    started = time.monotonic()
    probabilities = _routing_probabilities()
    judge_seconds = time.monotonic() - started
    rows = [
        {**item, "p_verified": p, "messages": len(item["messages"])}
        for item, p in zip(items, probabilities, strict=True)
    ]
    if any(row["p_verified"] is None for row in rows):
        raise SystemExit("routing: the judge did not answer every conversation")
    import yaml

    configured = yaml.safe_load((HERE / "verified.yaml").read_text())["profile_judge"]["prefer"][
        "min_probability"
    ]

    def miss_rate(subset: list[dict], tau: float) -> float:
        needed = [row for row in subset if row["label"] == "VERIFIED"]
        missed = [row for row in needed if not _routes_verified(row["p_verified"], tau)]
        return len(missed) / len(needed)

    def easy_to_think(subset: list[dict], tau: float) -> float:
        easy = [row for row in subset if row["label"] == "THINK"]
        return sum(1 for row in easy if not _routes_verified(row["p_verified"], tau)) / len(easy)

    calibration = rows[0::2]
    holdout = rows[1::2]
    candidates = sorted(
        {round(row["p_verified"], 4) for row in calibration} | {0.0, 0.5}, reverse=True
    )
    recommended = next(
        (tau for tau in candidates if tau <= 0.5 and miss_rate(calibration, tau) < 0.10), 0.0
    )
    by_category: dict[str, list[float]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row["p_verified"])
    summary = {
        "conversations": len(rows),
        "judge_wall_s": round(judge_seconds, 2),
        "configured_tau": configured,
        "recommended_tau": recommended,
        "calibration_miss_rate": round(miss_rate(calibration, configured), 4),
        "holdout_miss_rate": round(miss_rate(holdout, configured), 4),
        "everyday_to_think": round(easy_to_think(rows, configured), 4),
        "p_verified_by_category": {
            category: {
                "min": round(min(values), 4),
                "median": round(statistics.median(values), 4),
                "max": round(max(values), 4),
            }
            for category, values in by_category.items()
        },
    }
    deadline.check()
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    passed = summary["holdout_miss_rate"] < 0.10 and summary["calibration_miss_rate"] < 0.10
    _write("routing", {"passed": passed, "summary": summary, "rows": rows})
    print(
        f"routing: {'PASS' if passed else 'FAIL'} "
        f"(held-out miss rate {summary['holdout_miss_rate']})"
    )
    if not passed:
        raise SystemExit(1)


def gate_think_route(env: dict[str, str], *, budget_s: float = 1800) -> None:
    """Everyday requests stream from the think route at the default effort.

    The publisher's private reasoning is withheld by Kairyu's multi-stage
    contract (as on the sibling ensemble's deepseek_think route), so the gate
    checks the route, a streamed answer with its time to first token, no
    guarantee field, and that DeepSeek ran at the default effort (high).
    """

    deadline = Deadline("think-route", budget_s)
    items = [item for item in _dataset("routing-set.json") if item["label"] == "THINK"][:6]
    rows = [
        chat(env, messages=item["messages"], model=ROUTED, trace=True, stream=True)
        for item in items
    ]
    deadline.check()
    findings = []
    for row in rows:
        if row["status"] != 200 or not row["content"].strip() or row["ttft_s"] is None:
            findings.append(f"status={row['status']} empty={not row['content'].strip()}")
        elif row["route"] != "deepseek_think":
            findings.append(f"everyday request routed to {row['route']}")
        elif row["has_verification"] or row["efforts"] != ["high"]:
            findings.append(
                f"think route: verification={row['has_verification']} efforts={row['efforts']}"
            )
    _print_rows(rows)
    ttfts = [row["ttft_s"] for row in rows if row["ttft_s"] is not None]
    summary = {
        **_summary(rows),
        "think_routed": sum(1 for row in rows if row["route"] == "deepseek_think"),
        "ttft_p50_s": statistics.median(ttfts) if ttfts else None,
    }
    print(json.dumps(summary, indent=2), flush=True)
    _write(
        "think-route",
        {"passed": not findings, "findings": findings, "summary": summary, "rows": rows},
    )
    print(f"think-route: {'PASS' if not findings else 'FAIL'} {findings}")
    if findings:
        raise SystemExit(1)


def gate_effort(env: dict[str, str], *, budget_s: float = 5400) -> None:
    """The caller's effort reaches every DeepSeek step on both routes (default high)."""

    deadline = Deadline("effort", budget_s)
    easy = next(item for item in _dataset("routing-set.json") if item["label"] == "THINK")
    verified_prompt = "List three primary colors as a comma-separated line, nothing else."
    findings = []
    report = {}
    for effort in (None, "low", "high", "max"):
        extra = {} if effort is None else {"reasoning_effort": effort}
        rows = [
            chat(env, messages=easy["messages"], model=ROUTED, trace=True, **extra),
            chat(env, verified_prompt, model=ALWAYS, trace=True, **extra),
        ]
        expected = effort or "high"
        for row in rows:
            if row["status"] != 200 or not row["efforts"]:
                findings.append(f"{effort}: status={row['status']} efforts={row['efforts']}")
            elif set(row["efforts"]) != {expected}:
                findings.append(
                    f"{effort}: {row['route']} saw efforts {sorted(set(map(str, row['efforts'])))}"
                )
        print(f"effort={expected}:")
        _print_rows(rows)
        report[expected if effort else "default"] = {
            "rows": rows,
            "deepseek_steps": [len(row["efforts"]) for row in rows],
        }
        deadline.check()
    _write("effort", {"passed": not findings, "findings": findings, "report": report})
    print(f"effort: {'PASS' if not findings else 'FAIL'} {findings}")
    if findings:
        raise SystemExit(1)


_IMPLICIT_PROMPT = """For each EXPECTED condition, decide whether the CHECKLIST contains a \
condition that requires the same thing. Answer as JSON {{"covered": [true/false per expected \
condition, in order]}}.
EXPECTED:
{expected}
CHECKLIST:
{checklist}"""


def gate_implicit(env: dict[str, str], *, budget_s: float = 5400) -> None:
    """Situational requirements are extracted and kept only when expected."""

    deadline = Deadline("implicit", budget_s)
    items = _dataset("implicit-set.json")
    rows = _run_concurrently(
        env, [{"messages": item["messages"], "model": ALWAYS} for item in items], concurrency=8
    )
    deadline.check()
    l1 = f"http://127.0.0.1:{env['DEEPSEEK_L1_PORT']}"
    covered = expected_total = 0
    spurious = []
    details = []
    for item, row in zip(items, rows, strict=True):
        checklist = [
            r for r in row["requirements"] if (r.get("tags") or {}).get("origin") != "common"
        ]
        implicit = [r for r in checklist if (r.get("tags") or {}).get("origin") == "implicit"]
        entry = {"request": item["messages"][-1]["content"][:80], "implicit_kept": len(implicit)}
        if item["expected_implicit"]:
            body = control.post_json(
                f"{l1}/v1/chat/completions",
                {
                    "model": SPEC["deepseek"]["served_name"],
                    "messages": [
                        {
                            "role": "user",
                            "content": _IMPLICIT_PROMPT.format(
                                expected="\n".join(f"- {e}" for e in item["expected_implicit"]),
                                checklist="\n".join(f"- {r['proposition']}" for r in checklist),
                            ),
                        }
                    ],
                    # A thinking judge: the chat-mode verdict missed coverage
                    # spread over several conditions (VCO-D8 amendment).
                    "max_tokens": 16384,
                    "temperature": 0.0,
                    "reasoning_effort": "high",
                    "response_format": {"type": "json_object"},
                },
                timeout_s=900,
            )
            flags = json.loads(body["choices"][0]["message"]["content"]).get("covered") or []
            hit = sum(1 for flag in flags if flag is True)
            covered += hit
            expected_total += len(item["expected_implicit"])
            entry["expected_covered"] = f"{hit}/{len(item['expected_implicit'])}"
        else:
            spurious.append(len(implicit))
        details.append(entry)
    recall = covered / expected_total if expected_total else 0.0
    summary = {
        **_summary(rows),
        "implicit_recall": round(recall, 4),
        "controls": len(spurious),
        "control_implicit_kept_mean": round(statistics.mean(spurious), 3) if spurious else None,
    }
    _print_rows(rows)
    print(
        json.dumps({"summary": summary, "details": details}, indent=2, ensure_ascii=False),
        flush=True,
    )
    passed = recall >= 0.80 and (summary["control_implicit_kept_mean"] or 0) <= 0.5
    _write("implicit", {"passed": passed, "summary": summary, "details": details, "rows": rows})
    print(
        f"implicit: {'PASS' if passed else 'FAIL'} "
        f"(recall {recall:.3f} >= 0.80, control mean <= 0.5)"
    )
    if not passed:
        raise SystemExit(1)


def gate_serving_routed(env: dict[str, str], *, budget_s: float = 14400) -> None:
    """The routed product model under load: route mix, latency and tokens per route."""

    deadline = Deadline("serving-routed", budget_s)
    items = _dataset("routing-set.json")
    random.Random(3).shuffle(items)
    plan = {1: 8, 4: 16, 8: 16, 16: 32}
    report = {}
    offset = 0
    for concurrency, count in plan.items():
        batch = [items[(offset + i) % len(items)] for i in range(count)]
        offset += count
        started = time.monotonic()
        rows = _run_concurrently(
            env,
            [{"messages": item["messages"], "model": ROUTED, "trace": True} for item in batch],
            concurrency,
        )
        wall = time.monotonic() - started
        per_route = {}
        for route in sorted({row["route"] for row in rows}):
            subset = [row for row in rows if row["route"] == route]
            per_route[route] = _summary(subset)
        summary = {
            **_summary(rows),
            "wall_s": round(wall, 1),
            "requests_per_min": round(60 * len(rows) / wall, 2),
            "per_route": per_route,
        }
        summary["orchestration_output_tok_per_s"] = round(
            summary["orchestration_output_tokens"] / wall, 1
        )
        print(f"c{concurrency}:")
        _print_rows(rows)
        print(json.dumps(summary, indent=2), flush=True)
        report[f"c{concurrency}"] = {"summary": summary, "rows": rows}
        deadline.check()
    failures = [
        name
        for name, entry in report.items()
        if entry["summary"]["ok"] != entry["summary"]["requests"]
    ]
    _write("serving-routed", {"passed": not failures, "failed": failures, "report": report})
    print(f"serving-routed: {'PASS' if not failures else 'FAIL'}")
    if failures:
        raise SystemExit(1)


GATES = {
    "l1": gate_l1,
    "calibrate": gate_calibrate,
    "calibrate-g1": gate_calibrate_g1,
    "requirements": gate_requirements,
    "repair": gate_repair,
    "structured": gate_structured,
    "fallback": gate_fallback,
    "serving": gate_serving,
    "routing": gate_routing,
    "think-route": gate_think_route,
    "effort": gate_effort,
    "implicit": gate_implicit,
    "serving-routed": gate_serving_routed,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("gate", choices=(*GATES, "list"))
    args = parser.parse_args()
    if args.gate == "list":
        print("\n".join(GATES))
        return
    GATES[args.gate](_env())


if __name__ == "__main__":
    main()
