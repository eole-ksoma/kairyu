"""Regression tests for truthful, locally runnable GitHub Actions gates."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from scripts import cpu_microbench_gate

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _ROOT / ".github" / "workflows"


def _load_workflow(name: str) -> tuple[dict[str, Any], str]:
    text = (_WORKFLOWS / name).read_text(encoding="utf-8")
    parsed = yaml.load(text, Loader=yaml.BaseLoader)
    assert isinstance(parsed, dict)
    return parsed, text


def _named_step(job: dict[str, Any], name: str) -> dict[str, Any]:
    steps = job.get("steps")
    assert isinstance(steps, list)
    matches = [
        step
        for step in steps
        if isinstance(step, dict) and step.get("name") == name
    ]
    assert len(matches) == 1
    return matches[0]


def _run_commands(workflow: dict[str, Any]) -> tuple[str, ...]:
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict)
    commands: list[str] = []
    for job in jobs.values():
        assert isinstance(job, dict)
        for step in job.get("steps", []):
            if isinstance(step, dict) and isinstance(step.get("run"), str):
                commands.append(step["run"])
    return tuple(commands)


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def test_f1b_pull_requests_always_run_the_local_smoke_gate() -> None:
    workflow, text = _load_workflow("f1b-rollout.yml")
    pull_request = workflow["on"]["pull_request"]
    rollout = workflow["jobs"]["rollout"]

    assert "types" not in pull_request
    assert "if" not in rollout
    assert "labeled" not in text
    assert "f1b-formal" not in text
    assert "github.event.pull_request.labels" not in text
    assert "github.event_name == 'pull_request' && 'smoke'" in rollout["env"]["PROFILE"]
    step = _named_step(rollout, "Run F1b zero-failure rollout gate")
    assert step["run"] == 'bash scripts/kind_rollout_gate.sh "--${PROFILE}"'


def test_f1b_rollout_freezes_each_image_before_kind_cluster_creation() -> None:
    script = (_ROOT / "scripts" / "kind_rollout_gate.sh").read_text(
        encoding="utf-8"
    )
    gateway_build = script.index('-t "$GATEWAY_IMAGE" "$SOURCE_ARCHIVE_DIR"')
    gateway_save = script.index(
        '"$DOCKER" image save --output "$GATEWAY_IMAGE_ARCHIVE" "$GATEWAY_IMAGE"'
    )
    mock_build = script.index('-t "$MOCK_IMAGE" "$ARCHIVE_MOCK_DIR"')
    mock_save = script.index(
        '"$DOCKER" image save --output "$MOCK_IMAGE_ARCHIVE" "$MOCK_IMAGE"'
    )
    cluster_create = script.index('"$KIND" create cluster')
    archive_load = script.index(
        '"$KIND" load image-archive "$GATEWAY_IMAGE_ARCHIVE" "$MOCK_IMAGE_ARCHIVE"'
    )

    assert gateway_build < gateway_save < mock_build < mock_save < cluster_create
    assert cluster_create < archive_load
    assert '"$KIND" load docker-image' not in script


def test_kind_ci_uses_one_verified_retried_tool_installer() -> None:
    expected = {
        ("ci.yml", "kind-smoke"): ("v0.31.0", "v1.35.0"),
        ("f1a-churn.yml", "churn"): ("v0.32.0", "v1.36.3"),
        ("f1b-rollout.yml", "rollout"): ("v0.32.0", "v1.36.3"),
        ("f1c-gateway.yml", "gateway"): ("v0.32.0", "v1.36.3"),
    }
    for (workflow_name, job_name), versions in expected.items():
        workflow, _ = _load_workflow(workflow_name)
        step = _named_step(workflow["jobs"][job_name], "Install kind and kubectl")
        assert step["run"] == "bash scripts/install_kind_tools.sh"
        assert (
            step["env"]["KIND_VERSION"],
            step["env"]["KUBECTL_VERSION"],
        ) == versions
        pull_request = workflow["on"].get("pull_request")
        if isinstance(pull_request, dict) and "paths" in pull_request:
            assert "scripts/install_kind_tools.sh" in pull_request["paths"]

    for workflow_path in _WORKFLOWS.glob("*.yml"):
        assert "helm/kind-action" not in workflow_path.read_text(encoding="utf-8")

    installer = (_ROOT / "scripts" / "install_kind_tools.sh").read_text(
        encoding="utf-8"
    )
    assert "--retry 4" in installer
    assert "for attempt in 1 2 3" in installer
    assert "sha256sum" in installer
    assert '"${TOOLS_DIR}/kind" version' in installer
    assert '"${TOOLS_DIR}/kubectl" version --client=true' in installer


def test_kind_tool_installer_recovers_from_transient_downloads(
    tmp_path: Path,
) -> None:
    fake_curl = tmp_path / "curl"
    state = tmp_path / "curl-count"
    github_path = tmp_path / "github-path"
    kind_payload = tmp_path / "kind-payload"
    kubectl_payload = tmp_path / "kubectl-payload"
    _write_executable(
        kind_payload,
        "#!/usr/bin/env bash\necho 'kind v0.test go-test linux/amd64'\n",
    )
    _write_executable(
        kubectl_payload,
        "#!/usr/bin/env bash\necho 'Client Version: v1.test'\n",
    )
    _write_executable(
        fake_curl,
        """#!/usr/bin/env bash
set -euo pipefail
count=0
if [[ -f "$FAKE_CURL_STATE" ]]; then
  count=$(<"$FAKE_CURL_STATE")
fi
count=$((count + 1))
printf '%s\n' "$count" >"$FAKE_CURL_STATE"
if ((count <= 2)); then
  exit 56
fi
url=
output=
while (($#)); do
  case "$1" in
    --output)
      output=$2
      shift 2
      ;;
    https://*)
      url=$1
      shift
      ;;
    *)
      shift
      ;;
  esac
done
case "$url" in
  */kind-linux-amd64.sha256sum)
    printf '%s  kind-linux-amd64\n' "$(sha256sum "$FAKE_KIND" | awk '{print $1}')" >"$output"
    ;;
  */kind-linux-amd64)
    cp "$FAKE_KIND" "$output"
    ;;
  */kubectl.sha256)
    sha256sum "$FAKE_KUBECTL" | awk '{print $1}' >"$output"
    ;;
  */kubectl)
    cp "$FAKE_KUBECTL" "$output"
    ;;
  *)
    exit 2
    ;;
esac
""",
    )
    env = os.environ | {
        "CURL": str(fake_curl),
        "FAKE_CURL_STATE": str(state),
        "FAKE_KIND": str(kind_payload),
        "FAKE_KUBECTL": str(kubectl_payload),
        "GITHUB_PATH": str(github_path),
        "KIND_VERSION": "v0.test",
        "KUBECTL_VERSION": "v1.test",
        "RUNNER_TEMP": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", str(_ROOT / "scripts" / "install_kind_tools.sh")],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "download attempt 1/3 failed" in result.stderr
    assert "download attempt 2/3 failed" in result.stderr
    assert state.read_text(encoding="utf-8").strip() == "6"
    assert github_path.read_text(encoding="utf-8").strip() == str(
        tmp_path / "kairyu-tools"
    )


def test_f1c_pull_requests_always_run_the_real_gateway_and_async_request_gates() -> None:
    workflow, text = _load_workflow("f1c-gateway.yml")
    pull_request = workflow["on"]["pull_request"]
    jobs = workflow["jobs"]

    assert "types" not in pull_request
    assert set(jobs) == {"gateway"}
    assert "if" not in jobs["gateway"]
    assert "needs" not in jobs["gateway"]
    assert "run-f1c" not in text
    assert "require-label" not in text
    assert "github.event.pull_request.labels" not in text
    step = _named_step(
        jobs["gateway"],
        "Run the F1c and AsyncRequest Kubernetes integration tests",
    )
    assert step["run"] == "bash scripts/kind_async_request_gate.sh"
    assert "kind_gateway_gate.sh" in Path("scripts/kind_async_request_gate.sh").read_text(
        encoding="utf-8"
    )


def test_f2a_pull_requests_measure_and_replay_fresh_formal_evidence() -> None:
    workflow, text = _load_workflow("f2a-prefix-routing.yml")
    benchmark = workflow["jobs"]["benchmark"]
    profile = benchmark["env"]["PROFILE"]

    assert "github.event_name == 'pull_request' && 'formal'" in profile
    assert "HEAD^ HEAD" not in text
    assert "F2A_RETAINED_DIR" not in text
    assert "steps.mode.outputs" not in text
    assert "F2A_EXPECTED_COMMIT" not in benchmark["env"]

    measurement = _named_step(
        benchmark,
        "Run a fresh 500-replica F2a measurement",
    )
    replay = _named_step(benchmark, "Replay the freshly emitted evidence")
    assert "if" not in measurement
    assert "if" not in replay
    assert "--output-dir \"${F2A_RESULTS_DIR}\"" in measurement["run"]
    assert "--verify-artifact \"${F2A_RESULTS_DIR}\"" in replay["run"]


def test_f2b_pull_requests_measure_and_replay_while_manual_replay_stays_local() -> None:
    workflow, text = _load_workflow("f2b-kv-event-chaos.yml")
    benchmark = workflow["jobs"]["benchmark"]
    profile = benchmark["env"]["PROFILE"]

    assert "github.event_name == 'pull_request' && 'formal'" in profile
    assert "steps.mode.outputs" not in text
    assert "gh api" not in text
    assert "gh run download" not in text
    assert "--actions-run-provenance" not in text
    assert "actions" not in workflow["permissions"]
    assert "F2B_EXPECTED_COMMIT" not in benchmark["env"]
    assert "F2B_GITHUB_HEAD_SHA" not in benchmark["env"]

    measurement = _named_step(benchmark, "Run a fresh F2b measurement")
    fresh_replay = _named_step(
        benchmark,
        "Replay freshly emitted evidence without remeasurement",
    )
    manual_replay = _named_step(
        benchmark,
        "Replay a manually selected local artifact",
    )
    for step in (measurement, fresh_replay):
        assert "github.event_name == 'pull_request'" in step["if"]
        assert "inputs.mode == 'measure'" in step["if"]
    assert manual_replay["if"] == (
        "github.event_name == 'workflow_dispatch' && inputs.mode == 'replay'"
    )
    assert "--verify-artifact \"${F2B_RESULTS_DIR}\"" in fresh_replay["run"]
    assert "--require-current-runner-context" in fresh_replay["run"]
    assert "--verify-artifact \"${REPLAY_DIR}\"" in manual_replay["run"]
    assert "--require-current-source" in manual_replay["run"]


def test_ci_jobs_are_bounded_and_uv_uses_the_committed_lockfile() -> None:
    workflow, text = _load_workflow("ci.yml")

    assert "continue-on-error" not in text
    assert workflow["jobs"]["test"]["name"] == (
        "Portable CPU tests (Python ${{ matrix.python-version }})"
    )
    assert workflow["jobs"]["test"]["strategy"]["fail-fast"] == "false"
    for name, job in workflow["jobs"].items():
        timeout = job.get("timeout-minutes")
        assert isinstance(timeout, str), name
        assert timeout.isdecimal() and int(timeout) > 0, name
    for command in _run_commands(workflow):
        if "uv sync" in command or "uv run" in command:
            assert "--frozen" in command, command
    dependency_check = _named_step(
        workflow["jobs"]["test"],
        "Verify required test dependencies",
    )["run"]
    for module in ("jsonschema", "torch", "transformers", "xgrammar", "zmq", "yaml"):
        assert f"--require-module {module}" in dependency_check
    test_command = _named_step(
        workflow["jobs"]["test"],
        "Test with coverage gate",
    )["run"]
    lint_command = _named_step(
        workflow["jobs"]["test"],
        "Lint",
    )["run"]
    assert lint_command == "uv run --frozen ruff check ."
    assert test_command.count("--fail-on-skip") == 2
    assert "coverage erase" in test_command
    assert "-n 2 --dist loadfile tests/evals" in test_command
    assert "tests --ignore=tests/evals" in test_command
    assert "--cov-report= --cov-fail-under=0" in test_command
    assert "--cov-append --cov-report=term-missing" in test_command
    assert "--cov-fail-under=80" in test_command
    docker_command = _named_step(
        workflow["jobs"]["verification-exec-container"],
        "Real Docker runner conformance",
    )["run"]
    assert "--fail-on-skip" in docker_command


def test_ci_runs_one_dedicated_cpu_microbenchmark_gate() -> None:
    workflow, text = _load_workflow("ci.yml")
    job = workflow["jobs"]["cpu-microbench"]

    assert job["name"] == "CPU microbenchmark regression gate"
    assert job["timeout-minutes"] == "10"
    assert "continue-on-error" not in job
    assert "if" not in job
    assert "needs" not in job
    assert "pull_request_target" not in text
    assert "env" not in job
    report_path = "${{ runner.temp }}/kairyu-cpu-microbench/report.json"
    run_step = _named_step(
        job,
        "Run variance-tolerant CPU microbenchmarks",
    )
    assert run_step["env"] == {
        "KAIRYU_CPU_MICROBENCH_REPORT": (
            report_path
        )
    }
    assert (
        cpu_microbench_gate.BENCHMARK_TIMEOUT_SECONDS
        * len(cpu_microbench_gate.BENCHMARKS)
        < int(job["timeout-minutes"]) * 60
    )
    setup_steps = [
        step
        for step in job["steps"]
        if isinstance(step, dict)
        and step.get("uses")
        == "astral-sh/setup-uv@d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86"
    ]
    assert len(setup_steps) == 1
    assert setup_steps[0]["with"]["python-version"] == "3.12"
    assert _named_step(job, "Sync dependencies")["run"] == (
        "uv sync --frozen --dev"
    )
    command = run_step["run"]
    assert "uv run --frozen python scripts/cpu_microbench_gate.py" in command
    assert '--output "$KAIRYU_CPU_MICROBENCH_REPORT"' in command
    assert job["permissions"] == {"contents": "read"}
    checkout = [
        step
        for step in job["steps"]
        if isinstance(step, dict)
        and step.get("uses")
        == "actions/checkout@11d5960a326750d5838078e36cf38b85af677262"
    ]
    assert len(checkout) == 1
    assert checkout[0]["with"]["persist-credentials"] == "false"
    upload = _named_step(job, "Retain CPU microbenchmark diagnostics")
    assert upload["if"] == "always()"
    assert upload["uses"] == (
        "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
    )
    assert upload["with"]["path"] == report_path
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["retention-days"] == "7"
    assert text.count("scripts/cpu_microbench_gate.py") == 1


def test_kind_job_runs_helm_tests_as_an_applicable_non_skipping_suite() -> None:
    workflow, _ = _load_workflow("ci.yml")
    assert workflow["concurrency"]["cancel-in-progress"] == "true"
    kind_job = workflow["jobs"]["kind-smoke"]
    assert "needs" not in kind_job
    steps = kind_job["steps"]
    step = _named_step(
        kind_job,
        "Run Helm integration tests",
    )
    script = (_ROOT / "scripts" / "helm_integration.sh").read_text(
        encoding="utf-8"
    )

    assert step["run"] == "bash scripts/helm_integration.sh"
    assert any(
        isinstance(candidate, dict)
        and candidate.get("uses") == "astral-sh/setup-uv@v5"
        for candidate in steps
    )
    assert "--require-executable helm" in script
    assert "-m helm tests/unit/test_fleet_elastic.py" in script
    assert "--fail-on-skip" in script


def test_postgres_integration_uses_one_local_reproducible_script() -> None:
    workflow, _ = _load_workflow("ci.yml")
    step = _named_step(
        workflow["jobs"]["compose-smoke"],
        "PostgreSQL integration tests",
    )
    script = (_ROOT / "scripts" / "postgres_integration.sh").read_text(
        encoding="utf-8"
    )

    assert step["run"] == "bash scripts/postgres_integration.sh"
    assert "postgres:17.6-bookworm@sha256:" in script
    assert "docker run --detach" in script
    assert "docker run --detach --rm" not in script
    assert 'psycopg.connect(os.environ["KAIRYU_TEST_POSTGRES_DSN"],' in script
    assert "connect_timeout=2" in script
    assert 'connection.execute("SELECT 1")' in script
    assert 'docker port "$CONTAINER" 5432/tcp' in script
    assert "KAIRYU_TEST_POSTGRES_READY_ATTEMPTS" in script
    assert 'for _ in $(seq 1 "$READY_ATTEMPTS")' in script
    assert 'if postgres_ready; then' in script
    assert 'docker exec "$CONTAINER" pg_isready' not in script
    assert 'docker logs "$CONTAINER" >&2 || true' in script
    assert "--fail-on-skip" in script
    assert "-m postgres" in script
    assert "tests/unit/test_postgres_batch_store.py" in script
    assert "tests/unit/test_postgres_request_store.py" in script


def _postgres_integration_fake_env(
    tmp_path: Path,
    *,
    attempts: int,
    succeed_at: int,
) -> tuple[dict[str, str], Path, Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    uv_log = tmp_path / "uv.log"
    ready_count = tmp_path / "ready-count"

    _write_executable(
        fake_bin / "docker",
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$DOCKER_LOG"\n'
        'case "$1" in\n'
        "  run) printf 'fake-container-id\\n' ;;\n"
        "  port) printf '127.0.0.1:45678\\n' ;;\n"
        "  logs) printf 'postgres readiness diagnostic\\n' >&2 ;;\n"
        "  rm) : ;;\n"
        "  *) exit 91 ;;\n"
        "esac\n",
    )
    _write_executable(
        fake_bin / "uv",
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$UV_LOG"\n'
        'case "$*" in\n'
        '  *"python -c"*)\n'
        "    count=0\n"
        '    if [ -f "$READY_COUNT" ]; then count=$(cat "$READY_COUNT"); fi\n'
        "    count=$((count + 1))\n"
        '    printf "%s\\n" "$count" > "$READY_COUNT"\n'
        '    if [ "$READY_SUCCEED_AT" -gt 0 ] && '
        '[ "$count" -ge "$READY_SUCCEED_AT" ]; then exit 0; fi\n'
        "    exit 1\n"
        "    ;;\n"
        "esac\n"
        "exit 0\n",
    )
    _write_executable(fake_bin / "sleep", "#!/bin/sh\nexit 0\n")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "DOCKER_LOG": str(docker_log),
        "UV_LOG": str(uv_log),
        "READY_COUNT": str(ready_count),
        "READY_SUCCEED_AT": str(succeed_at),
        "KAIRYU_TEST_POSTGRES_READY_ATTEMPTS": str(attempts),
    }
    return env, docker_log, uv_log, ready_count


def _run_postgres_integration(
    tmp_path: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(_ROOT / "scripts" / "postgres_integration.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def test_postgres_integration_retries_the_external_query_not_pg_isready(
    tmp_path: Path,
) -> None:
    env, docker_log, uv_log, ready_count = _postgres_integration_fake_env(
        tmp_path,
        attempts=3,
        succeed_at=2,
    )
    result = _run_postgres_integration(tmp_path, env)

    assert result.returncode == 0, result.stderr
    assert ready_count.read_text(encoding="utf-8").strip() == "2"
    uv_commands = uv_log.read_text(encoding="utf-8").splitlines()
    assert sum("python -c" in command for command in uv_commands) == 2
    assert any("connect_timeout=2" in command for command in uv_commands)
    docker_commands = docker_log.read_text(encoding="utf-8").splitlines()
    assert any(command.startswith("port ") for command in docker_commands)
    assert not any(command.startswith("exec ") for command in docker_commands)
    assert docker_commands[-1].startswith("rm -f ")


def test_postgres_integration_fails_closed_with_logs_after_exact_bound(
    tmp_path: Path,
) -> None:
    env, docker_log, uv_log, ready_count = _postgres_integration_fake_env(
        tmp_path,
        attempts=2,
        succeed_at=0,
    )
    result = _run_postgres_integration(tmp_path, env)

    assert result.returncode == 1
    assert ready_count.read_text(encoding="utf-8").strip() == "2"
    assert "after 2 attempts" in result.stderr
    assert "postgres readiness diagnostic" in result.stderr
    uv_commands = uv_log.read_text(encoding="utf-8").splitlines()
    assert sum("python -c" in command for command in uv_commands) == 2
    docker_commands = docker_log.read_text(encoding="utf-8").splitlines()
    assert any(command.startswith("logs ") for command in docker_commands)
    assert docker_commands[-1].startswith("rm -f ")


def test_f2_workflows_use_frozen_local_uv_commands() -> None:
    for name in ("f2a-prefix-routing.yml", "f2b-kv-event-chaos.yml"):
        workflow, text = _load_workflow(name)
        assert "continue-on-error" not in text
        for command in _run_commands(workflow):
            if "uv sync" in command or "uv run" in command:
                assert "--frozen" in command, (name, command)
        for command in _run_commands(workflow):
            if "pytest" in command:
                assert "--fail-on-skip" in command, (name, command)
