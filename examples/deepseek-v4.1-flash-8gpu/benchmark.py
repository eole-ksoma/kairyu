"""Fixed-token L1 measurements that distinguish reasoning from final content.

The shared content-only benchmark treats an all-reasoning response's end as
TTFT. This example measures the first model delta (reasoning or content),
and records first visible content separately, leaving it null when absent.
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
            raise ValueError("JSON after terminal SSE marker")
        chunk = json.loads(raw)
        if "error" in chunk:
            raise ValueError(f"Stream error: {chunk['error']}")
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            text, trace = delta.get("content") or "", delta.get("reasoning_content") or ""
            now = time.perf_counter() - start
            if (text or trace) and first_model is None:
                first_model = now
            if text and first_content is None:
                first_content = now
            content += text
            reasoning += trace
            finish = choice.get("finish_reason") or finish
    if done != 1 or finish not in {"stop", "length"} or not usage or first_model is None:
        raise ValueError("Incomplete stream, missing model output, or absent usage")
    tokens = usage.get("completion_tokens")
    if not isinstance(tokens, int) or tokens < 1:
        raise ValueError("Missing positive completion-token count")
    elapsed = time.perf_counter() - start
    return {
        "model_ttft_ms": first_model * 1000,
        "content_ttft_ms": first_content * 1000 if first_content is not None else None,
        "total_ms": elapsed * 1000,
        "tpot_ms": (elapsed - first_model) * 1000 / (tokens - 1) if tokens > 1 else None,
        "completion_tokens": tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "usage": usage,
        "finish_reason": finish,
        "content": content,
        "reasoning_content": reasoning,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


async def measure(args) -> int:
    rows = json.loads(args.dataset.read_text())[: args.num_requests]
    if len(rows) != args.num_requests:
        raise ValueError("Dataset contains fewer requests than requested")
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/") + "/",
        timeout=args.timeout,
        limits=httpx.Limits(max_connections=args.concurrency),
    ) as client:

        async def run(index, row):
            async with semaphore:
                start = time.perf_counter()
                try:
                    body = {
                        "model": args.model,
                        "messages": [{"role": "user", "content": row["conversations"][0]["value"]}],
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "max_tokens": args.max_tokens,
                        "min_tokens": args.min_tokens,
                        "ignore_eos": args.ignore_eos,
                        "temperature": args.temperature,
                        "seed": args.seed,
                    }
                    async with client.stream("POST", "chat/completions", json=body) as response:
                        response.raise_for_status()
                        result = await collect(response.aiter_lines(), start)
                        result["request_id"] = response.headers.get("x-request-id")
                    if result["completion_tokens"] != args.max_tokens:
                        raise ValueError("Response did not reach the fixed output length")
                    return {"index": index, "passed": True, **result}
                except Exception as error:
                    return {"index": index, "passed": False, "error": str(error)}

        start = time.perf_counter()
        samples = await asyncio.gather(*(run(i, row) for i, row in enumerate(rows)))
        elapsed = time.perf_counter() - start
    good = [row for row in samples if row["passed"]]
    tokens = sum(row["completion_tokens"] for row in good)
    summary = {
        "requests": len(samples),
        "successful_requests": len(good),
        "success_rate": len(good) / len(samples),
        "wall_s": elapsed,
        "completion_tokens_total": tokens,
        "output_tokens_per_s": tokens / elapsed,
        "requests_per_s": len(good) / elapsed,
        "concurrency": args.concurrency,
        "tensor_parallel": args.tensor_parallel,
        "dp_replicas": args.dp_replicas,
    }
    for field in ("model_ttft_ms", "content_ttft_ms", "total_ms", "tpot_ms"):
        values = [row[field] for row in good if row[field] is not None]
        summary[field] = {
            "p50": percentile(values, 0.5),
            "p99": percentile(values, 0.99),
            "samples": len(values),
        }
    report = {
        "schema_version": 1,
        "measurement": "fixed_length_including_reasoning",
        "percentile_method": "nearest_rank",
        "summary": summary,
        "samples": samples,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "row-serving.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return int(len(good) != len(samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--min-tokens", type=int, required=True)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--tensor-parallel", type=int, default=8)
    parser.add_argument("--dp-replicas", type=int, default=1)
    args = parser.parse_args()
    if min(args.num_requests, args.concurrency, args.max_tokens) < 1:
        parser.error("request, concurrency and token limits must be positive")
    raise SystemExit(asyncio.run(measure(args)))


if __name__ == "__main__":
    main()
