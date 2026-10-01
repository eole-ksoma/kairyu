#!/usr/bin/env python3
"""GPU gates for checklist-verified answers.

Every gate has a time budget and writes its per-request evidence (latency,
tokens, tok/s, guarantee flag, reason, attempts, requirement count) to
model-volumes/<environment>/results/<gate>-<UTC>.json.

  l1            DeepSeek grammar-constrained JSON on every DP rank (thinking and
                chat) and System One on each OpenJev replica
  calibrate     tau_hi on InFoBench expert labels (calibrate.py)
  requirements  extracted checklists cover InFoBench's gold decomposed questions
  repair        constraint-heavy requests: repairs happen and every guaranteed
                answer meets the stated constraint (independent check)
  structured    a caller json_schema survives drafting and repair
  fallback      one OpenJev down: still guaranteed; both down: 200, unverified,
                reason judge_unavailable
  serving       end-to-end latency, tokens and guarantee rate at c1/c4/c8/c16
"""

from __future__ import annotations

import argparse
import concurrent.futures
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


def chat(env: dict[str, str], content: str, *, timeout_s: float = 1800, **extra) -> dict:
    """One verified request; returns the per-request evidence row."""

    payload = control.verified_request(content, **extra)
    started = time.monotonic()
    try:
        body = control.post_json(f"{_api(env)}/v1/chat/completions", payload, timeout_s=timeout_s)
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
    content_out = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    output_tokens = usage.get("orchestration_output_tokens") or 0
    return {
        "status": status,
        "seconds": round(elapsed, 2),
        "route": "multi_agent (verified DAG)",
        "public_completion_tokens": usage.get("completion_tokens"),
        "orchestration_input_tokens": usage.get("orchestration_input_tokens"),
        "orchestration_output_tokens": output_tokens,
        "orchestration_output_tok_per_s": round(output_tokens / elapsed, 1) if elapsed else None,
        "guaranteed": report.get("guaranteed"),
        "reason": report.get("reason"),
        "attempts": report.get("attempts"),
        "requirements": report.get("requirements") or [],
        "content": content_out,
        "error": body.get("error"),
        "verification_error": control.verification_error(body) if status == 200 else None,
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
            f"route={row['route']} in={row['orchestration_input_tokens']} "
            f"out={row['orchestration_output_tokens']} "
            f"({row['orchestration_output_tok_per_s']} tok/s) "
            f"guaranteed={row['guaranteed']} reason={row['reason']} "
            f"attempts={row['attempts']} requirements={len(row['requirements'])}",
            flush=True,
        )


def _run_concurrently(env: dict[str, str], prompts: list[dict], concurrency: int) -> list[dict]:
    def one(prompt: dict) -> dict:
        extra = {key: value for key, value in prompt.items() if key != "content"}
        return chat(env, prompt["content"], **extra)

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


_COVERAGE_PROMPT = """For each GOLD question below, decide whether the CHECKLIST contains a \
condition that requires what the question checks (alone or together with other conditions). \
Answer as JSON {{"covered": [true/false per gold question, in order]}}.
GOLD QUESTIONS:
{gold}
CHECKLIST:
{checklist}"""


def gate_requirements(env: dict[str, str], *, count: int = 40, budget_s: float = 7200) -> None:
    """Sufficiency: the judged checklist covers InFoBench's gold questions."""

    deadline = Deadline("requirements", budget_s)
    rows = infobench(count, seed=1)
    # Answers are kept so a failed coverage pass does not repeat generation.
    answers = _results_dir() / f"requirements-answers-{count}.json"
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


GATES = {
    "l1": gate_l1,
    "calibrate": gate_calibrate,
    "requirements": gate_requirements,
    "repair": gate_repair,
    "structured": gate_structured,
    "fallback": gate_fallback,
    "serving": gate_serving,
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
