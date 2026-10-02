#!/usr/bin/env python3
"""Calibrate the G1-source threshold on human-labelled answers.

G1-source asks, per source or action claim the state builder lists, whether
the claim is supported. RAGTruth (RAG answers with human hallucination
spans) labels an answer violated when it has any span.

Every answer runs this example's production path: the state builder (same
prompt, grammar and high effort) lists the claims, and OpenJev reads the
production G1 questions through Kairyu's ChecklistRun. tau is the smallest
threshold whose accepted answers (p >= tau) have a one-sided 95 %
Clopper-Pearson upper bound on the violation rate <= alpha on the
calibration half; the held-out half (split by source document) is reported
unchanged and is the gate.

Result (2026-10-02, VCO-D11): no threshold reaches alpha = 0.10, so G1-source
is advisory (threshold 0) in verified.yaml. The computation and general
kinds (PRM800K, FEVER) failed too and left the configuration with VCO-D12;
their measurement is in MEASUREMENTS.md.

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

import verification  # noqa: E402
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
}
# requirement id -> the dataset that calibrates it
KINDS = {"G1-source": "ragtruth"}
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


def select(rows: list[dict], seed: int) -> list[dict]:
    """PER_DATASET answers, at most one per group where the data allows."""

    random.Random(seed).shuffle(rows)
    seen: set[str] = set()
    first = [row for row in rows if not (row["group"] in seen or seen.add(row["group"]))]
    taken = {row["id"] for row in first}
    rest = [row for row in rows if row["id"] not in taken]
    return (first + rest)[:PER_DATASET]


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
    samples = [{**row, "dataset": "ragtruth"} for row in select(ragtruth(directory), seed)]
    env = control._compose_env()
    l1_url = f"http://127.0.0.1:{env['DEEPSEEK_L1_PORT']}"
    builder = calibrate._roles()["state_builder"]
    # Claims depend on the state builder prompt: one cache per configuration.
    cache = directory / f"judged-{verification._build_key()}.jsonl"
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
