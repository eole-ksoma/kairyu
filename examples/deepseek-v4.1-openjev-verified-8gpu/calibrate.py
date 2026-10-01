#!/usr/bin/env python3
"""Calibrate the Conductor threshold tau_hi on InFoBench expert labels.

InFoBench's expert annotation pairs model answers with decomposed yes/no
requirements and a human pass/fail label for each. Every requirement is
judged through this example's own production path: DeepSeek turns the
question into a condition statement (like the extractor writes), the
example's state builder lists the answer's claims, and OpenJev reads the
example's checklist state and question wording through Kairyu's System One
API. tau_hi is the smallest threshold whose accepted requirements have a
one-sided 95 % Clopper-Pearson upper bound on the violation rate <= alpha on
the calibration half; the held-out half is reported unchanged.

Usage: ./verify.sh calibrate   (after ./run.sh up)
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import csv
import dataclasses
import hashlib
import json
import math
import random
import re
import sys
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import control  # noqa: E402

from kairyu.dsl.loader import load_spec, role_spec  # noqa: E402
from kairyu.engine.systemone import HTTPSystemOneBackend  # noqa: E402
from kairyu.entrypoints.server.chat_service import (  # noqa: E402
    validate_orchestration_chat_input,
)
from kairyu.entrypoints.server.protocol import ChatCompletionRequest  # noqa: E402
from kairyu.orchestration.checklist import ChecklistConfig, ChecklistRun  # noqa: E402
from kairyu.orchestration.request import conversation_text  # noqa: E402

SPEC = control.SPEC
# InFoBench expert annotation (Easy and Hard subsets), Google Drive file ids
# from the official repository's "Generation and Annotation" folder.
SOURCES = {
    "161wLlIQzuHofbgkVvvSIn8cH5y6f4jlk": (
        "ac2b5b342188e24b6e781ad7c7a595945f8dbd5ef15d021c2056f11be5bb3da6"
    ),
    "1IKIRSLR3aPnBLhTd99nO09QQ72qiyKZc": (
        "d3e4c9f2220443647118e9a80ffb9519df7c43979ac5c3ca3afacc7424c01d54"
    ),
}
OPENJEV_URLS = ("http://127.0.0.1:8015", "http://127.0.0.1:8016")
ANNOTATED_MODELS = ("gpt-3.5-turbo", "gpt-4", "claude-v1", "alpaca-7b", "vicuna-13b")


def _post(url: str, payload: dict, timeout_s: float = 1800) -> dict:
    return control.post_json(url, payload, timeout_s=timeout_s)


def download(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for file_id, digest in SOURCES.items():
        path = directory / f"{file_id}.csv"
        if not path.is_file():
            url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
            with urllib.request.urlopen(url, timeout=120) as response:
                path.write_bytes(response.read())
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"{path.name} has sha256 {actual}, expected {digest}")
        paths.append(path)
    return paths


_NUMBERED = re.compile(r"^\s*\d+\.\s*")
_CATEGORY = re.compile(r"\s*\([^()]*\)\s*$")


def samples(paths: list[Path]) -> list[dict]:
    """One sample per (instruction, model) with a complete expert label row."""

    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if not row.get("id"):
                    continue
                questions = [
                    _CATEGORY.sub("", _NUMBERED.sub("", line)).strip()
                    for line in row["decomposed_questions"].splitlines()
                    if line.strip()
                ]
                for model in ANNOTATED_MODELS:
                    output = (row.get(model) or "").strip()
                    labels = [
                        part
                        for part in (row.get(f"{model}-annotation") or "").strip().split(".")
                        if part.strip()
                    ]
                    if not output or len(labels) != len(questions) or not questions:
                        continue
                    if set(labels) - {"0", "1"}:
                        continue
                    rows.append(
                        {
                            "id": row["id"],
                            "model": model,
                            "request": "\n\n".join(
                                part for part in (row["instruction"], row.get("input", "")) if part
                            ),
                            "answer": output,
                            "questions": questions,
                            "labels": [int(label) for label in labels],
                        }
                    )
    return rows


def _roles() -> dict[str, dict]:
    spec = yaml.safe_load((HERE / "verified.yaml").read_text(encoding="utf-8"))
    return {role["name"]: role for role in spec["roles"]}


def _query(request: str) -> str:
    """The exact L2 {query} Kairyu renders for a one-turn chat request."""

    chat = ChatCompletionRequest(
        model=SPEC["public_models"][1], messages=[{"role": "user", "content": request}]
    )
    return validate_orchestration_chat_input(chat).prompt


def _deepseek(l1_url: str, prompt: str, *, response_format: dict | None = None) -> str:
    payload: dict = {
        "model": SPEC["deepseek"]["served_name"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16384,
        "temperature": 0.0,
        # chat mode, like the serving state builder (V4.1 thinks otherwise)
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if response_format is not None:
        payload["response_format"] = response_format
    body = _post(f"{l1_url}/v1/chat/completions", payload)
    return body["choices"][0]["message"]["content"]


_STATEMENT_PROMPT = (
    "Rewrite this yes/no question about a generated answer as one declarative "
    "condition the ANSWER must satisfy, in the same language. Output only the "
    "condition.\nQuestion: {question}"
)


def prepare(sample: dict, l1_url: str, roles: dict[str, dict]) -> dict:
    """Condition statements and the state builder's claims for one sample."""

    query = _query(sample["request"])
    statements = [
        _deepseek(l1_url, _STATEMENT_PROMPT.format(question=question)).strip()
        for question in sample["questions"]
    ]
    builder = roles["state_builder"]
    claims = _deepseek(
        l1_url,
        builder["prompt"].format_map(
            {"query": query, "conversation": conversation_text(query), "answer": sample["answer"]}
        ),
        response_format=builder["sampling"]["response_format"],
    )
    return {**sample, "statements": statements, "claims": claims}


def _requirement_checklist() -> ChecklistConfig:
    """The production checklist reduced to its requirement questions.

    The same Kairyu code (ChecklistRun) builds the Jev state and questions
    as in serving; only the deterministic checks and the per-claim G1
    questions are left out, because InFoBench labels requirements only.
    """

    spec = load_spec(HERE / "verified-always.yaml")
    node = next(role for role in spec.roles if role.name == "checklist")
    config = role_spec(node).checklist
    assert config is not None
    requirement = next(question for question in config.questions if question.id == "{item[id]}")
    return dataclasses.replace(config, checks=(), questions=(requirement,))


def judge(sample: dict, api_url: str, roles: dict[str, dict]) -> list[float]:
    """P(requirement satisfied) per requirement, via the production checklist."""

    del roles
    config = _requirement_checklist()
    requirements = [
        {"id": f"Q{index}", "proposition": text, "kind": "semantic", "sources": ["U1"]}
        for index, text in enumerate(sample["statements"], start=1)
    ]
    outputs = {
        "extract": json.dumps(
            {"units": [{"id": "U1", "text": sample["request"]}], "requirements": requirements},
            ensure_ascii=False,
        ),
        "answer": sample["answer"],
        "state_builder": sample["claims"],
    }
    query = _query(sample["request"])

    async def read() -> list[float]:
        # The serving path: the L2 reads both OpenJev replicas directly.
        backend = HTTPSystemOneBackend(
            base_urls=OPENJEV_URLS, upstream_model=SPEC["systemone"]["model"], timeout_s=600
        )
        try:
            run = ChecklistRun(config, target_text=sample["answer"], sources=query)
            verdict = await run.decide(backend, outputs, query)
        finally:
            await backend.shutdown()
        by_id = {item.id: item.p for item in verdict.items}
        return [by_id[item["id"]] for item in requirements]

    return asyncio.run(read())


def _binomial_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson_upper(k: int, n: int, confidence: float) -> float:
    """One-sided upper confidence bound of a binomial proportion."""

    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    low, high = k / n, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        if _binomial_cdf(k, n, middle) > 1 - confidence:
            low = middle
        else:
            high = middle
    return high


def accepted_stats(pairs: list[tuple[float, int]], tau: float, confidence: float) -> dict:
    accepted = [label for p, label in pairs if p >= tau]
    violations = accepted.count(0)
    return {
        "tau": tau,
        "accepted": len(accepted),
        "violations": violations,
        "violation_rate": violations / len(accepted) if accepted else None,
        "upper_bound": clopper_pearson_upper(violations, len(accepted), confidence),
        "acceptance": len(accepted) / len(pairs) if pairs else 0.0,
    }


def choose_tau(pairs: list[tuple[float, int]], alpha: float, confidence: float) -> dict:
    for tau in sorted({p for p, _label in pairs}):
        stats = accepted_stats(pairs, tau, confidence)
        if stats["accepted"] and stats["upper_bound"] <= alpha:
            return stats
    return {"tau": None, "accepted": 0, "violations": 0, "upper_bound": None}


def response_level(rows: list[dict], tau: float) -> dict:
    """A response passes when all its requirements pass; violated if any label is 0."""

    passed = [row for row in rows if all(p >= tau for p in row["p"])]
    violated = [row for row in passed if 0 in row["labels"]]
    return {"responses": len(rows), "passed": len(passed), "violated": len(violated)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    calibration = SPEC["calibration"]
    directory = control.environment_storage() / "calibration"
    rows = samples(download(directory))
    env = control._compose_env()
    api_url = f"http://127.0.0.1:{env['API_PORT']}"
    l1_url = f"http://127.0.0.1:{env['DEEPSEEK_L1_PORT']}"
    roles = _roles()
    cache = directory / "judged.jsonl"
    done = {}
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[(row["id"], row["model"])] = row

    def work(sample: dict) -> dict:
        key = (sample["id"], sample["model"])
        if key in done:
            return done[key]
        prepared = prepare(sample, l1_url, roles)
        return {**prepared, "p": judge(prepared, api_url, roles)}

    pending = [row for row in rows if (row["id"], row["model"]) not in done]
    print(f"{len(rows)} labelled responses; {len(pending)} to judge", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(work, pending):
            done[(row["id"], row["model"])] = row
            with cache.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    judged = [done[(row["id"], row["model"])] for row in rows]
    instructions = sorted({row["id"] for row in judged})
    random.Random(calibration["split_seed"]).shuffle(instructions)
    calibration_ids = set(instructions[: len(instructions) // 2])
    halves = {
        "calibration": [row for row in judged if row["id"] in calibration_ids],
        "holdout": [row for row in judged if row["id"] not in calibration_ids],
    }
    pairs = {
        name: [(p, label) for row in part for p, label in zip(row["p"], row["labels"], strict=True)]
        for name, part in halves.items()
    }
    alpha, confidence = float(calibration["alpha"]), float(calibration["confidence"])
    chosen = choose_tau(pairs["calibration"], alpha, confidence)
    report = {
        "alpha": alpha,
        "confidence": confidence,
        "instructions": {name: len({row["id"] for row in part}) for name, part in halves.items()},
        "labels": {name: len(values) for name, values in pairs.items()},
        "violations": {
            name: [label for _p, label in values].count(0) for name, values in pairs.items()
        },
        "calibration": chosen,
    }
    if chosen["tau"] is not None:
        report["holdout"] = accepted_stats(pairs["holdout"], chosen["tau"], confidence)
        report["holdout_responses"] = response_level(halves["holdout"], chosen["tau"])
    (directory / "tau.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
