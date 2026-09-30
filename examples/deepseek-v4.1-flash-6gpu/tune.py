"""Compare named L1 candidates one parameter at a time and restore the baseline.

Run after ``run.sh up``. Each candidate restarts only this example's vLLM
service with a Compose override, must pass the L1 correctness probes (exact
finite answers from every DP rank, a tool call, an image), and is then
measured with fixed 8K-in / 256-out rows. Raw evidence, the exact command,
and failures are kept; the committed configuration is restored in
``finally``. Final numbers come from ``verify.sh`` on the committed
configuration, not from these exploratory rows.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import control
import yaml

import verification

ENGRAM_OFFLOAD = '{"cpu_offload":true}'
SPARSE_ATTENTION = '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4"}'

# Flag -> value (None removes the flag, True adds a bare flag).
CANDIDATES: dict[str, dict[str, object]] = {
    "baseline": {},
    # The committed command after a fresh restart: run-to-run variation of a
    # single trial, measured the same way as every other candidate.
    "baseline-repeat": {},
    # Official memory-bound (8 x H100) levers.
    "batch-8k": {"--max-num-batched-tokens": "8192"},
    "batch-4k": {"--max-num-batched-tokens": "4096"},
    "memory-0.92": {"--gpu-memory-utilization": "0.92"},
    "v2-runner": {},
    # Official Blackwell scheduler value is 128 sequences; 64 is the 8-GPU example's.
    "seqs-64": {"--max-num-seqs": "64"},
    # Official Blackwell indexer option (nightly runtime only).
    "sparse-logits": {
        "--attention-config": (
            '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4",'
            '"indexer_sparse_logits":true}'
        )
    },
    # Official multi-image TTFT option.
    "encoder-dp": {"--mm-encoder-tp-mode": "data"},
    # Official Blackwell DEP recipe at six ranks.
    "dep6-official": {
        "--tensor-parallel-size": "1",
        "--data-parallel-size": "6",
        "--attention-config": (
            '{"backend":"FLASHMLA_MEGA_ATTN_DSV41","indexer_kv_dtype":"mxfp4",'
            '"indexer_sparse_logits":true}'
        ),
        "--kernel-config": '{"moe_backend":"deep_gemm_mega_moe"}',
        "--moe-backend": None,
        "--engram-config": '{"embedding_across_dp":true}',
    },
    # The official Blackwell TP degree: TP2 pairs on NUMA-local GPUs, DP3.
    "tp2-dp3": {"--tensor-parallel-size": "2", "--data-parallel-size": "3"},
    # Official NVIDIA DSpark (5-token block, adaptive verification on), and the
    # same block verified in full. The drafter's 128 experts do not divide EP6;
    # the fused-MoE path distributes the remainder (only mega MoE asserts it).
    "dspark": {
        "--speculative-config": (
            '{"method":"dspark","num_speculative_tokens":5,'
            '"draft_sample_method":"probabilistic","rejection_sample_method":"block",'
            '"enable_adaptive_verification":true}'
        )
    },
    "dspark-full-verify": {
        "--speculative-config": (
            '{"method":"dspark","num_speculative_tokens":5,'
            '"draft_sample_method":"probabilistic","rejection_sample_method":"block",'
            '"enable_adaptive_verification":false}'
        )
    },
}
# DSpark with the full-block verification needs KV room on DP6: combine it
# with the official memory levers, or with the TP2 shape (larger KV pool).
_DSPARK_FULL = CANDIDATES["dspark-full-verify"]["--speculative-config"]
CANDIDATES.update(
    {
        "dspark-batch-4k": {
            "--speculative-config": _DSPARK_FULL,
            "--max-num-batched-tokens": "4096",
        },
        "dspark-batch-4k-memory-0.92": {
            "--speculative-config": _DSPARK_FULL,
            "--max-num-batched-tokens": "4096",
            "--gpu-memory-utilization": "0.92",
        },
        "dspark-tp2-dp3": {
            "--speculative-config": _DSPARK_FULL,
            "--tensor-parallel-size": "2",
            "--data-parallel-size": "3",
        },
    }
)
CANDIDATE_ENVIRONMENT = {"v2-runner": {"VLLM_USE_V2_MODEL_RUNNER": "1"}}
BARE_FLAGS = {"--enable-expert-parallel", "--enable-prefix-caching", "--trust-remote-code"}


def candidate_command(command: list[str], name: str) -> list[str]:
    result = list(command)
    for flag, value in CANDIDATES[name].items():
        if flag in result:
            index = result.index(flag)
            width = 1 if flag in BARE_FLAGS else 2
            replacement = [] if value is None else [flag] if value is True else [flag, str(value)]
            result[index : index + width] = replacement
        elif value is True:
            result.append(flag)
        elif value is not None:
            result.extend([flag, str(value)])
    return result


def _compose_base() -> list[str]:
    return [
        "docker",
        "compose",
        "--project-directory",
        str(control.HERE),
        "--file",
        str(control.HERE / "compose.yaml"),
    ]


def restart(env: dict[str, str], override: Path | None, log: Path, timeout_s: int) -> None:
    command = _compose_base() + (["--file", str(override)] if override else [])
    command += ["up", "--detach", "--no-deps", "--force-recreate", "deepseek"]
    with log.open("w") as output:
        subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        item = json.loads(subprocess.check_output(["docker", "inspect", control.L1_CONTAINER]))[0]
        if item["RestartCount"] or not item["State"]["Running"]:
            raise RuntimeError("L1 exited during startup; see worker.log")
        if item["State"].get("Health", {}).get("Status") == "healthy":
            return
        time.sleep(5)
    raise TimeoutError(f"L1 not healthy after {timeout_s} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", nargs="+", choices=list(CANDIDATES))
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--startup-timeout", type=int, default=5400)
    parser.add_argument("--image-ttft", action="store_true", help="also time 8-image prompts")
    args = parser.parse_args()
    if args.requests < max(args.concurrency):
        parser.error("requests must cover every concurrency")
    env = control._compose_env()
    control._preflight(env)
    run_dir = verification.RESULTS_ROOT / f"{verification._run_id()}-tuning"
    run_dir.mkdir(parents=True)
    original = yaml.safe_load((control.HERE / "compose.yaml").read_text())["services"]["deepseek"][
        "command"
    ]
    api = f"http://127.0.0.1:{env['API_PORT']}"
    reports: list[dict] = []
    try:
        for index, name in enumerate(args.candidates):
            row_dir = run_dir / name
            row_dir.mkdir()
            command = candidate_command(original, name)
            environment = CANDIDATE_ENVIRONMENT.get(name, {})
            service: dict = {"command": command}
            if environment:
                service["environment"] = environment
            override = row_dir / "override.json"
            override.write_text(json.dumps({"services": {"deepseek": service}}))
            report: dict = {
                "candidate": name,
                "command": command,
                "environment_override": environment,
                "served_config_sha256": verification.served_config_sha256(),
                "rows": [],
            }
            reports.append(report)
            try:
                if index == 0 and name == "baseline":
                    verification.runtime_evidence()
                    report["reused_running_baseline"] = True
                else:
                    restart(env, override, row_dir / "startup.log", args.startup_timeout)
                control.validate_ready(api)
                control._validate_arithmetic(api)
                control.validate_tool_calling(api)
                control.validate_vision(api)
                report["startup"] = verification._startup_evidence()
                if args.image_ttft:
                    report["image_ttft"] = verification.image_ttft(row_dir)
                for level in args.concurrency:
                    dataset = row_dir / f"c{level}.json"
                    verification.fixed_dataset(
                        dataset, args.requests, 8192, namespace=f"{run_dir.name}-{name}-{level}"
                    )
                    peak = verification.GpuPeak()
                    with peak:
                        code = verification.bench(
                            dataset,
                            mode="fixed",
                            requests=args.requests,
                            concurrency=level,
                            max_tokens=256,
                            out=row_dir / f"c{level}",
                        )
                    summary = json.loads((row_dir / f"c{level}" / "row.json").read_text())[
                        "summary"
                    ]
                    report["rows"].append(
                        {
                            "concurrency": level,
                            "exit_code": code,
                            "summary": summary,
                            "gpu_peak_mib": peak.peak,
                        }
                    )
                    if code:
                        raise RuntimeError(f"c{level} row failed")
                report["passed"] = True
            except (Exception, SystemExit) as error:  # noqa: BLE001 - recorded per candidate
                report.update(passed=False, error=str(error))
            finally:
                with (row_dir / "worker.log").open("w") as log:
                    subprocess.run(
                        ["docker", "logs", "--tail", "4000", control.L1_CONTAINER],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                (run_dir / "selection.json").write_text(json.dumps(reports, indent=2) + "\n")
    finally:
        if any(name != "baseline" for name in args.candidates):
            restart(env, None, run_dir / "restore.log", args.startup_timeout)
    print(f"candidate evidence: {run_dir}")
    raise SystemExit(int(any(not report["passed"] for report in reports)))


if __name__ == "__main__":
    main()
