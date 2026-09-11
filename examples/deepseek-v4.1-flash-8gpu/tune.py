"""Compare named L1 candidates, retaining raw evidence and restoring the baseline.

Run after `run.sh up`. This restarts only this example's L1 container.
Candidate rows are exploratory; final gates must run against the committed
configuration with `verify.sh` after the winning settings have been selected.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import benchmark
import control
import yaml

import verification

CANDIDATES = {
    "baseline": {},
    "pcie-ipc": {},
    "no-spec": {"--speculative-config": None},
    "no-ep": {"--enable-expert-parallel": None},
    "batch-8k": {"--max-num-batched-tokens": "8192"},
    "batch-32k": {"--max-num-batched-tokens": "32768"},
    "seq-32": {"--max-num-seqs": "32"},
    "memory-95": {"--gpu-memory-utilization": "0.95"},
    "eager": {"--enforce-eager": True},
    "engram-cpu": {"--engram-config": '{"cpu_offload":true}'},
}


CANDIDATE_ENVIRONMENT = {"pcie-ipc": {"VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": "1"}}


def candidate_command(command: list[str], name: str) -> list[str]:
    result = list(command)
    for flag, value in CANDIDATES[name].items():
        if flag in result:
            index = result.index(flag)
            size = 1 if flag in {"--enable-expert-parallel", "--enforce-eager"} else 2
            replacement = [] if value is None else ([flag] if value is True else [flag, str(value)])
            result[index : index + size] = replacement
        elif value is True:
            result.append(flag)
        elif value is not None:
            result.extend([flag, str(value)])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", nargs="+", choices=list(CANDIDATES))
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--startup-timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.requests < max(args.concurrency) or min(args.concurrency) < 1:
        parser.error("requests must cover every positive concurrency")
    env = control._compose_env()
    control._preflight(env)
    control._ensure_vllm_image(env)
    verification.runtime_evidence()
    run_dir = verification.RESULTS_ROOT / (verification._run_id() + "-tuning")
    run_dir.mkdir(parents=True)
    baseline = yaml.safe_load((control.HERE / "compose.yaml").read_text())
    original = baseline["services"]["deepseek-0"]["command"]
    base = [
        "docker",
        "compose",
        "--project-directory",
        str(control.HERE),
        "--file",
        str(control.HERE / "compose.yaml"),
    ]

    def restart(override: Path | None, log: Path):
        command = base + (["--file", str(override)] if override else [])
        command += [
            "up",
            "--detach",
            "--no-deps",
            "--force-recreate",
            "deepseek-0",
        ]
        with log.open("w") as output:
            subprocess.run(
                command,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=args.startup_timeout,
                check=True,
            )
        deadline = time.monotonic() + args.startup_timeout
        container = env["COMPOSE_PROJECT_NAME"] + "-deepseek-0-1"
        while time.monotonic() < deadline:
            item = json.loads(subprocess.check_output(["docker", "inspect", container]))[0]
            state = item["State"]
            if item["RestartCount"] or not state["Running"]:
                raise RuntimeError("L1 exited during startup; see worker.log")
            if state.get("Health", {}).get("Status") == "healthy":
                return
            time.sleep(2)
        raise TimeoutError(f"L1 did not become healthy in {args.startup_timeout} seconds")

    reports = []
    try:
        for candidate_index, name in enumerate(args.candidates):
            row_dir = run_dir / name
            row_dir.mkdir()
            command = candidate_command(original, name)
            override = row_dir / "override.json"
            environment = CANDIDATE_ENVIRONMENT.get(name, {})
            service = {"command": command}
            if environment:
                service["environment"] = environment
            override.write_text(json.dumps({"services": {"deepseek-0": service}}))
            report = {
                "candidate": name,
                "command": command,
                "environment_override": environment,
                "base_config_sha256": verification._served_config_sha256(),
                "rows": [],
            }
            reports.append(report)
            try:
                # The initial runtime has already passed the exact config/image
                # check. Reuse it for the first baseline measurement.
                if candidate_index == 0 and name == "baseline":
                    report["reused_verified_baseline"] = True
                    control._validate_ready(f"http://127.0.0.1:{env['API_PORT']}")
                else:
                    restart(override, row_dir / "startup.log")
                if name == "pcie-ipc":
                    logs = subprocess.check_output(
                        ["docker", "logs", env["COMPOSE_PROJECT_NAME"] + "-deepseek-0-1"],
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    if "Initialized FlashInfer PCIe IPC all-reduce" not in logs:
                        raise RuntimeError("PCIe IPC was not initialized; fallback is not a trial")
                control._validate_tool_calling(f"http://127.0.0.1:{env['API_PORT']}")
                control._validate_vision(f"http://127.0.0.1:{env['API_PORT']}")
                for concurrency in args.concurrency:
                    dataset = row_dir / f"c{concurrency}.json"
                    verification._serving_dataset(
                        dataset,
                        args.requests,
                        8192,
                        namespace=f"{run_dir.name}-{name}-{concurrency}",
                    )
                    options = SimpleNamespace(
                        dataset=dataset,
                        num_requests=args.requests,
                        concurrency=concurrency,
                        base_url=f"http://127.0.0.1:{env['API_PORT']}/v1",
                        model=control.SPEC["model"]["served_name"],
                        timeout=1800,
                        max_tokens=256,
                        min_tokens=256,
                        ignore_eos=True,
                        temperature=1.0,
                        seed=0,
                        results_dir=row_dir / f"c{concurrency}",
                        tensor_parallel=8,
                        dp_replicas=1,
                    )
                    code = asyncio.run(benchmark.measure(options))
                    report["rows"].append({"concurrency": concurrency, "exit_code": code})
                    if code:
                        raise RuntimeError(f"failed c{concurrency} row")
                report["passed"] = True
            except (Exception, SystemExit) as error:
                report.update(passed=False, error=str(error))
            finally:
                container = env["COMPOSE_PROJECT_NAME"] + "-deepseek-0-1"
                inspected = subprocess.run(
                    ["docker", "inspect", container], capture_output=True, text=True, check=False
                )
                if inspected.returncode == 0:
                    item = json.loads(inspected.stdout)[0]
                    report["runtime"] = {
                        "image_id": item["Image"],
                        "command": item["Config"]["Cmd"],
                        "started_at": item["State"]["StartedAt"],
                        "environment_override": {
                            key: dict(value.split("=", 1) for value in item["Config"]["Env"]).get(
                                key
                            )
                            for key in environment
                        },
                    }
                with (row_dir / "worker.log").open("w") as log:
                    subprocess.run(
                        ["docker", "logs", "--tail", "2000", container],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                (run_dir / "selection.json").write_text(json.dumps(reports, indent=2) + "\n")
    finally:
        restart(None, run_dir / "restore.log")
    print(f"Candidate evidence: {run_dir}; baseline restored")
    raise SystemExit(int(any(not row["passed"] for row in reports)))


if __name__ == "__main__":
    main()
