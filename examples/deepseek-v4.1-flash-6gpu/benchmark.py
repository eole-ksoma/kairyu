"""Streaming load generator for this example's serving rows.

Two measurements:

* ``fixed``: every request generates exactly ``--max-tokens`` tokens
  (``ignore_eos``). Throughput includes reasoning tokens; the first model
  delta (reasoning or content) is the model TTFT, and first visible content
  is recorded separately (null when a row ends inside reasoning).
* ``completed``: natural completion at the default effort; a request passes
  only when it stops with visible content. This is what a user waits for.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import httpx


async def collect(lines, start: float) -> dict:
    first_model = first_content = None
    content = reasoning = ""
    finish = usage = None
    done = 0
    async for line in lines:
        if not line.startswith("data: "):
            continue
        raw = line[6:].strip()
        if raw == "[DONE]":
            done += 1
            continue
        if done:
            raise ValueError("JSON after the terminal SSE marker")
        chunk = json.loads(raw)
        if "error" in chunk:
            raise ValueError(f"stream error: {chunk['error']}")
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            text = delta.get("content") or ""
            trace = delta.get("reasoning_content") or ""
            now = time.perf_counter() - start
            if (text or trace) and first_model is None:
                first_model = now
            if text and first_content is None:
                first_content = now
            content += text
            reasoning += trace
            finish = choice.get("finish_reason") or finish
    if done != 1 or finish not in {"stop", "length"} or not usage or first_model is None:
        raise ValueError("incomplete stream, missing model output, or absent usage")
    tokens = usage.get("completion_tokens")
    if not isinstance(tokens, int) or tokens < 1:
        raise ValueError("missing positive completion-token count")
    elapsed = time.perf_counter() - start
    return {
        "model_ttft_ms": first_model * 1000,
        "content_ttft_ms": first_content * 1000 if first_content is not None else None,
        "total_ms": elapsed * 1000,
        "tpot_ms": (elapsed - first_model) * 1000 / (tokens - 1) if tokens > 1 else None,
        "completion_tokens": tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "finish_reason": finish,
        "content": content,
        "reasoning_chars": len(reasoning),
    }


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile."""
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def sample_error(result: dict, *, mode: str, max_tokens: int) -> str | None:
    if mode == "fixed":
        if result["completion_tokens"] != max_tokens:
            return f"generated {result['completion_tokens']} tokens, expected {max_tokens}"
        return None
    if result["finish_reason"] != "stop":
        return f"finish_reason {result['finish_reason']!r} (no completed answer)"
    if not result["content"].strip():
        return "completed without visible content"
    return None


async def measure(args) -> int:
    rows = json.loads(args.dataset.read_text())[: args.num_requests]
    if len(rows) != args.num_requests:
        raise ValueError("dataset holds fewer requests than requested")
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/") + "/",
        timeout=args.timeout,
        limits=httpx.Limits(max_connections=args.concurrency),
    ) as client:

        async def run(index: int, row: dict) -> dict:
            async with semaphore:
                start = time.perf_counter()
                body = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": row["prompt"]}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "max_tokens": args.max_tokens,
                    "seed": args.seed + index,
                }
                if args.mode == "fixed":
                    body.update(min_tokens=args.max_tokens, ignore_eos=True)
                try:
                    async with client.stream("POST", "chat/completions", json=body) as response:
                        response.raise_for_status()
                        result = await collect(response.aiter_lines(), start)
                        result["request_id"] = response.headers.get("x-request-id")
                except Exception as error:  # noqa: BLE001 - recorded per sample
                    return {"index": index, "passed": False, "error": str(error)}
                error = sample_error(result, mode=args.mode, max_tokens=args.max_tokens)
                return {"index": index, "passed": error is None, "error": error, **result}

        start = time.perf_counter()
        samples = await asyncio.gather(*(run(i, row) for i, row in enumerate(rows)))
        elapsed = time.perf_counter() - start
    good = [row for row in samples if row["passed"]]
    tokens = sum(row["completion_tokens"] for row in good)
    summary = {
        "mode": args.mode,
        "requests": len(samples),
        "successful_requests": len(good),
        "wall_s": elapsed,
        "completion_tokens_total": tokens,
        "output_tokens_per_s": tokens / elapsed,
        "concurrency": args.concurrency,
    }
    for field in ("model_ttft_ms", "content_ttft_ms", "total_ms", "tpot_ms"):
        values = [row[field] for row in good if row.get(field) is not None]
        summary[field] = {
            "p50": percentile(values, 0.5),
            "p99": percentile(values, 0.99),
            "samples": len(values),
        }
    report = {
        "schema_version": 1,
        "percentile_method": "nearest_rank",
        "summary": summary,
        "samples": samples,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "row.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return int(len(good) != len(samples))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixed", "completed"), required=True)
    parser.add_argument("--num-requests", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.num_requests, args.concurrency, args.max_tokens) < 1:
        parser.error("request, concurrency and token limits must be positive")
    raise SystemExit(asyncio.run(measure(args)))


if __name__ == "__main__":
    main()
