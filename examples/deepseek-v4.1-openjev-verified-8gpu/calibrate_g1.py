#!/usr/bin/env python3
"""Calibrate the per-kind G1 thresholds on human-labelled answers.

G1 asks, per claim the state builder lists, whether the claim is supported;
each claim kind is its own requirement with its own threshold:

- G1-source (source and action claims): RAGTruth, RAG answers with human
  hallucination spans; an answer is violated when it has any span.
- G1-computation: PRM800K phase 2, MATH solutions with human step ratings;
  an answer (the solution up to the first labelled error) is violated when a
  step is rated -1. Solutions with a neutral (0) step are left out.
- G1-general: FEVER, crowd-written Wikipedia claims; REFUTES is violated,
  SUPPORTS is not, NOT ENOUGH INFO is left out.

Every answer runs this example's production path: the state builder (same
prompt, grammar and high effort) lists the claims, and OpenJev reads the
production G1 questions through Kairyu's ChecklistRun. For each kind, tau is
the smallest threshold whose accepted answers (p of that requirement >= tau)
have a one-sided 95 % Clopper-Pearson upper bound on the violation rate <=
alpha on the calibration half; the held-out half (split by problem, source
document or Wikipedia page) is reported unchanged and is the gate.

Result (2026-10-02, VCO-D11): no kind reaches alpha = 0.10 on held-out
labels, so the G1 questions are advisory (threshold 0) in verified.yaml; the
gate passes only if a future judge or wording makes every kind calibratable.

Usage: ./verify.sh calibrate-g1   (after ./run.sh up)
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import concurrent.futures
import dataclasses
import hashlib
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import calibrate  # noqa: E402
import control  # noqa: E402

from kairyu.dsl.loader import load_spec, role_spec  # noqa: E402
from kairyu.engine.systemone import HTTPSystemOneBackend  # noqa: E402
from kairyu.orchestration.checklist import ChecklistConfig, ChecklistRun  # noqa: E402
from kairyu.orchestration.request import conversation_text  # noqa: E402

SPEC = control.SPEC
_RAGTRUTH = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/c103204b9ce28d6bbad859304bf30de72b8ed8fe/dataset"
SOURCES = {
    "ragtruth-response.jsonl": (
        f"{_RAGTRUTH}/response.jsonl",
        "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
    ),
    "ragtruth-source_info.jsonl": (
        f"{_RAGTRUTH}/source_info.jsonl",
        "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
    ),
    "prm800k-phase2_test.jsonl": (
        "https://media.githubusercontent.com/media/openai/prm800k/"
        "7ecc794703b2877f63226f2477a49b34f9b25163/prm800k/data/phase2_test.jsonl",
        "6b172efa884ac8341a946dd82e06947c135b7254109fb3f7aa907c715d98aaad",
    ),
    "fever-shared_task_dev.jsonl": (
        "https://fever.ai/download/fever/shared_task_dev.jsonl",
        "e89865bfe1b4dd054e03dd57d7241a6fde24862905f31117cf0cd719f7c78df7",
    ),
}
# requirement id -> the dataset that calibrates it
KINDS = {"G1-source": "ragtruth", "G1-computation": "prm800k", "G1-general": "fever"}
PER_DATASET = 600


def download(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in SOURCES.items():
        path = directory / name
        if not path.is_file():
            with urllib.request.urlopen(url, timeout=600) as response:
                path.write_bytes(response.read())
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"{name} has sha256 {actual}, expected {digest}")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def ragtruth(directory: Path) -> list[dict]:
    prompts = {
        row["source_id"]: row["prompt"] for row in _jsonl(directory / "ragtruth-source_info.jsonl")
    }
    rows = []
    for row in _jsonl(directory / "ragtruth-response.jsonl"):
        if row["quality"] != "good":
            continue
        labels = row["labels"]
        spans = json.loads(labels) if isinstance(labels, str) else labels
        rows.append(
            {
                "id": f"ragtruth-{row['id']}",
                "group": row["source_id"],
                "request": prompts[row["source_id"]],
                "answer": row["response"],
                "violated": bool(spans),
            }
        )
    return rows


def prm800k(directory: Path) -> list[dict]:
    rows = []
    for index, row in enumerate(_jsonl(directory / "prm800k-phase2_test.jsonl")):
        label = row["label"]
        if row["is_quality_control_question"] or row["is_initial_screening_question"]:
            continue
        if label["finish_reason"] not in ("solution", "found_error"):
            continue
        # The solution as generated: the chosen completion of each step; the
        # step where the labeller found the error has none, and its first
        # completion is the generated step.
        steps = [step["completions"][step["chosen_completion"] or 0] for step in label["steps"]]
        ratings = [step["rating"] for step in steps]
        if 0 in ratings or None in ratings:
            continue
        rows.append(
            {
                "id": f"prm800k-{index}",
                "group": row["question"]["problem"],
                "request": row["question"]["problem"],
                "answer": "\n\n".join(step["text"] for step in steps),
                "violated": -1 in ratings,
            }
        )
    return rows


_WIKI = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-COLON-": ":", "_": " "}


def _title(page: str) -> str:
    for token, text in _WIKI.items():
        page = page.replace(token, text)
    return page


def fever(directory: Path) -> list[dict]:
    rows = []
    for row in _jsonl(directory / "fever-shared_task_dev.jsonl"):
        if row["label"] not in ("SUPPORTS", "REFUTES"):
            continue
        pages = [entry[2] for group in row["evidence"] for entry in group if entry[2]]
        if not pages:
            continue
        title = _title(pages[0])
        rows.append(
            {
                "id": f"fever-{row['id']}",
                "group": title,
                "request": f"Tell me one fact about {title}.",
                "answer": row["claim"],
                "violated": row["label"] == "REFUTES",
            }
        )
    return rows


def select(rows: list[dict], seed: int, *, balance: bool) -> list[dict]:
    """PER_DATASET answers, at most one per group where the data allows."""

    rng = random.Random(seed)
    rng.shuffle(rows)
    pools = [[row for row in rows if row["violated"] is flag] for flag in (True, False)]
    if not balance:
        pools = [rows]
    quota = PER_DATASET // len(pools)
    chosen = []
    for pool in pools:
        seen: set[str] = set()
        first = [row for row in pool if not (row["group"] in seen or seen.add(row["group"]))]
        taken = {row["id"] for row in first}
        rest = [row for row in pool if row["id"] not in taken]
        chosen.extend((first + rest)[:quota])
    return chosen


def split(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    groups = sorted({row["group"] for row in rows})
    random.Random(seed).shuffle(groups)
    calibration = set(groups[: len(groups) // 2])
    return {
        "calibration": [row for row in rows if row["group"] in calibration],
        "holdout": [row for row in rows if row["group"] not in calibration],
    }


def claims(sample: dict, l1_url: str, builder: dict) -> tuple[str, dict]:
    """The production state builder at the default (high) effort."""

    query = calibrate._query(sample["request"])
    sampling = builder["sampling"]
    payload = {
        "model": SPEC["deepseek"]["served_name"],
        "messages": [
            {
                "role": "user",
                "content": builder["prompt"].format_map(
                    {"conversation": conversation_text(query), "answer": sample["answer"]}
                ),
            }
        ],
        "max_tokens": sampling["max_tokens_by_effort"]["high"],
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "reasoning_effort": "high",
        "response_format": sampling["response_format"],
    }
    body = calibrate._post(f"{l1_url}/v1/chat/completions", payload)
    return body["choices"][0]["message"]["content"], body.get("usage", {})


def _g1_checklist() -> ChecklistConfig:
    """The production checklist reduced to its G1 questions (same wording,
    thresholds and state sections; checks and requirements are left out)."""

    spec = load_spec(HERE / "verified-always.yaml")
    node = next(role for role in spec.roles if role.name == "checklist")
    config = role_spec(node).checklist
    assert config is not None
    questions = tuple(question for question in config.questions if question.id in KINDS)
    assert {question.id for question in questions} == set(KINDS)
    return dataclasses.replace(config, checks=(), questions=questions)


def judge(sample: dict, listed: str) -> dict[str, float]:
    """p of every G1 requirement the answer's claims produce."""

    query = calibrate._query(sample["request"])
    outputs = {
        "extract": json.dumps({"units": [], "requirements": []}),
        "answer": sample["answer"],
        "state_builder": listed,
    }

    async def read() -> dict[str, float]:
        backend = HTTPSystemOneBackend(
            base_urls=calibrate.OPENJEV_URLS,
            upstream_model=SPEC["systemone"]["model"],
            timeout_s=600,
        )
        try:
            run = ChecklistRun(_g1_checklist(), target_text=sample["answer"], sources=query)
            verdict = await run.decide(backend, outputs, query)
        finally:
            await backend.shutdown()
        return {item.id: item.p for item in verdict.items}

    return asyncio.run(read())


def analyse(rows: list[dict], alpha: float, confidence: float, seed: int) -> dict:
    report: dict = {"alpha": alpha, "confidence": confidence, "kinds": {}}
    taus: dict[str, float | None] = {}
    halves_by_dataset = {}
    for kind, dataset in KINDS.items():
        own = [row for row in rows if row["dataset"] == dataset and "error" not in row]
        halves = split(own, seed)
        halves_by_dataset[dataset] = halves
        pairs = {
            name: [
                (row["p"][kind], 0 if row["violated"] else 1) for row in part if kind in row["p"]
            ]
            for name, part in halves.items()
        }
        chosen = calibrate.choose_tau(pairs["calibration"], alpha, confidence)
        taus[kind] = chosen["tau"]
        bases = collections.Counter(basis for row in own for basis in row["bases"])
        entry = {
            "dataset": dataset,
            "answers": {name: len(part) for name, part in halves.items()},
            "violated": {
                name: sum(row["violated"] for row in part) for name, part in halves.items()
            },
            "with_kind": {name: len(values) for name, values in pairs.items()},
            "claims_by_basis": dict(bases),
            "errors": sum(1 for row in rows if row["dataset"] == dataset and "error" in row),
            "calibration": chosen,
        }
        if chosen["tau"] is not None:
            entry["holdout"] = calibrate.accepted_stats(pairs["holdout"], chosen["tau"], confidence)
        report["kinds"][kind] = entry
    if all(tau is not None for tau in taus.values()):
        # Every G1 requirement an answer produces, each at its own tau.
        combined = {}
        for dataset, halves in halves_by_dataset.items():
            passed = [
                row
                for row in halves["holdout"]
                if all(p >= taus[kind] for kind, p in row["p"].items())
            ]
            violated = sum(row["violated"] for row in passed)
            combined[dataset] = {
                "answers": len(halves["holdout"]),
                "passed": len(passed),
                "violated": violated,
                "upper_bound": calibrate.clopper_pearson_upper(violated, len(passed), confidence),
            }
        report["holdout_all_g1"] = combined
    report["taus"] = taus
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    settings = SPEC["calibration"]
    seed = int(settings["split_seed"])
    alpha, confidence = float(settings["alpha"]), float(settings["confidence"])
    directory = control.environment_storage() / "calibration" / "g1"
    download(directory)
    samples = []
    for dataset, loader, balance in (
        ("ragtruth", ragtruth, False),
        ("prm800k", prm800k, True),
        ("fever", fever, False),
    ):
        for row in select(loader(directory), seed, balance=balance):
            samples.append({**row, "dataset": dataset})
    env = control._compose_env()
    l1_url = f"http://127.0.0.1:{env['DEEPSEEK_L1_PORT']}"
    builder = calibrate._roles()["state_builder"]
    cache = directory / "judged.jsonl"
    done = {}
    if cache.is_file():
        for row in _jsonl(cache):
            done[row["id"]] = row

    def work(sample: dict) -> dict:
        started = time.monotonic()
        try:
            listed, usage = claims(sample, l1_url, builder)
            bases = [claim["basis"] for claim in json.loads(listed)["claims"]]
            p = judge(sample, listed)
        except Exception as error:  # recorded and reported, never silently dropped
            return {**sample, "error": f"{type(error).__name__}: {error}"[:500]}
        return {
            **sample,
            "claims": listed,
            "bases": bases,
            "p": p,
            "usage": usage,
            "seconds": round(time.monotonic() - started, 1),
        }

    pending = [row for row in samples if row["id"] not in done or "error" in done[row["id"]]]
    print(f"{len(samples)} labelled answers; {len(pending)} to judge", flush=True)
    finished = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(work, pending):
            done[row["id"]] = row
            finished += 1
            with cache.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if finished % 25 == 0:
                print(f"judged {finished}/{len(pending)}", flush=True)
    rows = [done[row["id"]] for row in samples]
    report = analyse(rows, alpha, confidence, seed)
    (directory / "tau.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    results = control.environment_storage() / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / f"g1-calibration-{stamp}.json").write_text(
        json.dumps({"report": report, "rows": rows}, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    holdout_ok = all(
        entry.get("holdout", {}).get("upper_bound") is not None
        and entry["holdout"]["upper_bound"] <= alpha
        for entry in report["kinds"].values()
    )
    print(f"calibrate-g1: {'PASS' if holdout_ok else 'FAIL'}")
    if not holdout_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
