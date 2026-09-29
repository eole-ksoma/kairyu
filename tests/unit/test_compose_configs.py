"""Checked-in Compose files must resolve their deployment contracts locally."""

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import yaml

from kairyu.deploy import builder as builder_module
from kairyu.deploy.builder import build_app_from_config
from kairyu.deploy.spec import load_deployment_spec
from kairyu.engine.embedding import MockEmbeddingBackend

COMPOSE_DIR = Path("deploy/compose")
COMPOSE_FILES = sorted(COMPOSE_DIR.glob("docker-compose*.yaml"))
WEBUI_COMPOSE = COMPOSE_DIR / "docker-compose.webui.yaml"
WEBUI_CONFIG = COMPOSE_DIR / "config.yaml"
WEBUI_ORCHESTRATOR = COMPOSE_DIR / "webui-orchestrator.yaml"
WEBUI_BROWSER_DOCKERFILE = COMPOSE_DIR / "Dockerfile.webui-browser"
WEBUI_BROWSER_PACKAGE = COMPOSE_DIR / "webui-browser" / "package.json"
WEBUI_BROWSER_LOCK = COMPOSE_DIR / "webui-browser" / "package-lock.json"
WEBUI_BROWSER_SMOKE = Path("scripts/webui_browser_smoke.mjs")
WEBUI_SMOKE_PROJECT = "kairyu-webui-smoke-test"
CONTAINER_CONFIG = "/etc/kairyu/config.yaml"
CONTAINER_ORCHESTRATOR = "/etc/kairyu/webui-orchestrator.yaml"
VALIDATOR = Path("scripts/validate_compose_binds.py").resolve()
COMPOSE_SMOKE = Path("scripts/compose_smoke.sh")
OTEL_TRACE_VERIFIER = Path("scripts/verify_otel_trace.py")
WEBUI_SMOKE = Path("scripts/webui_smoke.sh")
CI_WORKFLOW = Path(".github/workflows/ci.yml")
DOCKERFILE = Path("Dockerfile")
DOCKERIGNORE = Path(".dockerignore")
EMBEDDING_MODEL_REPOSITORY = "Qdrant/all-MiniLM-L6-v2-onnx"
EMBEDDING_MODEL_REVISION = "5f1b8cd78bc4fb444dd171e59b18f3a3af89a079"
EMBEDDING_MODEL_SHA256 = (
    "bbd7b466f6d58e646fdc2bd5fd67b2f5e93c0b687011bd4548c420f7bd46f0c5"
)
EMBEDDING_PROVENANCE_SHA256 = (
    "57246a4990eb0f08755df06ba57c1fec161032bd588332435e89c7ece244639c"
)


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path} must contain a YAML mapping"
    return data


def _run_validator(*compose_files: Path, cwd: Path):
    return subprocess.run(
        [sys.executable, str(VALIDATOR), *(str(path) for path in compose_files)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _webui_smoke_env(
    tmp_path: Path,
    *,
    uv_exit: int = 0,
    rendered_base_url: str = "http://kairyu:8000/v1",
    docker_fail_phase: str | None = None,
    curl_exit: int = 0,
    curl_fail_path: str | None = None,
) -> tuple[dict[str, str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "uv", f"#!/bin/sh\nexit {uv_exit}\n")
    docker_log = tmp_path / "docker.log"
    fail_case = (
        f'  *"WEBUI_SMOKE_PHASE={docker_fail_phase}"*) exit 41 ;;\n'
        if docker_fail_phase is not None
        else ""
    )
    _write_executable(
        fake_bin / "docker",
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$DOCKER_LOG"\n'
        'case "$*" in\n'
        '  *" config --format json")\n'
        "    printf "
        "'{\"services\":{\"webui\":{\"environment\":{\"OPENAI_API_BASE_URL\":\"%s\"}}}}' "
        '"$WEBUI_RENDERED_BASE_URL"\n'
        "    ;;\n"
        f"{fail_case}"
        "esac\n",
    )
    curl_log = tmp_path / "curl.log"
    _write_executable(
        fake_bin / "curl",
        """#!/usr/bin/env python3
import json
import os
import sys

arguments = sys.argv[1:]
with open(os.environ["CURL_LOG"], "a", encoding="utf-8") as log:
    log.write(" ".join(arguments) + "\\n")

url = next(
    (
        argument
        for argument in reversed(arguments)
        if argument.startswith(("http://", "https://"))
    ),
    "",
)
exit_code = int(os.environ["CURL_EXIT"])
fail_path = os.environ["CURL_FAIL_PATH"]
if exit_code and (not fail_path or url.endswith(fail_path)):
    raise SystemExit(exit_code)

if url.endswith("/readyz"):
    json.dump({"ready": True}, sys.stdout)
elif url.endswith("/v1/embeddings"):
    vectors = []
    for index in range(2):
        vector = [0.0] * 384
        vector[index] = 1.0
        vectors.append(
            {"object": "embedding", "index": index, "embedding": vector}
        )
    json.dump(
        {
            "object": "list",
            "model": "embed-small",
            "data": vectors,
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        },
        sys.stdout,
    )
elif url.endswith("/metrics"):
    sys.stdout.write(
        'kairyu_requests_total{code="200",method="POST",'
        'model="embed-small",path="/v1/embeddings",tenant="default"} 10.0\\n'
    )
else:
    print(f"unexpected fake curl URL: {url!r}", file=sys.stderr)
    raise SystemExit(64)
""",
    )
    _write_executable(fake_bin / "sleep", "#!/bin/sh\nexit 0\n")
    return (
        {
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "DOCKER_LOG": str(docker_log),
            "CURL_LOG": str(curl_log),
            "CURL_EXIT": str(curl_exit),
            "CURL_FAIL_PATH": curl_fail_path or "",
            "WEBUI_RENDERED_BASE_URL": rendered_base_url,
            "WEBUI_SMOKE_PROJECT_NAME": WEBUI_SMOKE_PROJECT,
        },
        docker_log,
        curl_log,
    )


def _literal_relative_binds(compose_file: Path):
    services = _load_yaml(compose_file).get("services")
    assert isinstance(services, dict), f"{compose_file} must declare services"
    for service_name, service in services.items():
        assert isinstance(service, dict)
        for volume in service.get("volumes", []):
            if isinstance(volume, str):
                source, separator, _target = volume.partition(":")
                if not separator or not source.startswith("."):
                    continue
            elif isinstance(volume, dict) and volume.get("type") == "bind":
                source = volume.get("source")
                if not isinstance(source, str) or Path(source).is_absolute():
                    continue
            else:
                continue
            if "$" not in source:
                yield service_name, source


@pytest.mark.parametrize("compose_file", COMPOSE_FILES, ids=lambda path: path.name)
def test_checked_in_literal_relative_bind_sources_exist(compose_file):
    binds = list(_literal_relative_binds(compose_file))
    assert binds, f"{compose_file} has no literal relative bind sources"
    for service_name, source in binds:
        resolved = (compose_file.parent / source).resolve()
        assert resolved.exists(), (
            f"{compose_file}: service {service_name!r} has missing bind source "
            f"{source!r} ({resolved})"
        )


def _assert_immutable_image(image: str, tagged_image: str) -> None:
    prefix = f"{tagged_image}@sha256:"
    assert image.startswith(prefix)
    digest = image.removeprefix(prefix)
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)


def test_webui_compose_pins_full_authenticated_browser_topology():
    services = _load_yaml(WEBUI_COMPOSE)["services"]
    kairyu = services["kairyu"]
    webui = services["webui"]
    browser = services["browser"]

    config_mounts = [
        volume
        for volume in kairyu["volumes"]
        if isinstance(volume, str)
        and (
            f":{CONTAINER_CONFIG}" in volume
            or f":{CONTAINER_ORCHESTRATOR}" in volume
        )
    ]
    assert config_mounts == [
        f"./config.yaml:{CONTAINER_CONFIG}:ro",
        f"./webui-orchestrator.yaml:{CONTAINER_ORCHESTRATOR}:ro",
    ]
    for mount, source_file in zip(
        config_mounts, (WEBUI_CONFIG, WEBUI_ORCHESTRATOR), strict=True
    ):
        source = mount.partition(":")[0]
        assert (WEBUI_COMPOSE.parent / source).resolve() == source_file.resolve()

    assert kairyu["build"] == {
        "context": "../..",
        "args": {
            "KAIRYU_EMBEDDINGS": "1",
            "KAIRYU_EMBEDDING_MODEL_REPOSITORY": EMBEDDING_MODEL_REPOSITORY,
            "KAIRYU_EMBEDDING_MODEL_REVISION": EMBEDDING_MODEL_REVISION,
            "KAIRYU_EMBEDDING_MODEL_SHA256": EMBEDDING_MODEL_SHA256,
            "KAIRYU_EMBEDDING_PROVENANCE_SHA256": EMBEDDING_PROVENANCE_SHA256,
        },
    }

    environment = webui["environment"]
    assert environment["OPENAI_API_BASE_URL"] == "http://kairyu:8000/v1"
    assert environment["OPENAI_API_KEY"] == "sk-local"
    assert {
        key: environment[key]
        for key in (
            "RAG_EMBEDDING_ENGINE",
            "RAG_EMBEDDING_MODEL",
            "RAG_OPENAI_API_BASE_URL",
            "RAG_OPENAI_API_KEY",
            "RAG_EMBEDDING_BATCH_SIZE",
            "ENABLE_ASYNC_EMBEDDING",
            "RAG_EMBEDDING_CONCURRENT_REQUESTS",
            "ENABLE_RETRIEVAL_QUERY_GENERATION",
            "RAG_TOP_K",
            "RAG_RELEVANCE_THRESHOLD",
            "RAG_FULL_CONTEXT",
            "BYPASS_EMBEDDING_AND_RETRIEVAL",
            "RAG_RERANKING_ENGINE",
        )
    } == {
        "RAG_EMBEDDING_ENGINE": "openai",
        "RAG_EMBEDDING_MODEL": "embed-small",
        "RAG_OPENAI_API_BASE_URL": "http://kairyu:8000/v1",
        "RAG_OPENAI_API_KEY": "sk-local",
        "RAG_EMBEDDING_BATCH_SIZE": "32",
        "ENABLE_ASYNC_EMBEDDING": "true",
        "RAG_EMBEDDING_CONCURRENT_REQUESTS": "2",
        "ENABLE_RETRIEVAL_QUERY_GENERATION": "false",
        "RAG_TOP_K": "1",
        "RAG_RELEVANCE_THRESHOLD": "0",
        "RAG_FULL_CONTEXT": "false",
        "BYPASS_EMBEDDING_AND_RETRIEVAL": "false",
        "RAG_RERANKING_ENGINE": "",
    }
    assert environment["WEBUI_AUTH"] == "true"
    assert environment["ENABLE_SIGNUP"] == "true"
    # The admin update-available toast intercepted model-selector clicks and
    # made the browser gate flaky; the check must stay disabled in CI.
    assert environment["ENABLE_VERSION_UPDATE_CHECK"] == "false"
    assert "WEBUI_SECRET_KEY" not in environment
    assert (
        environment["WEBUI_SECRET_KEY_FILE"]
        == "/app/backend/data/.webui_secret_key"
    )
    assert environment["DEFAULT_MODELS"] == "default"
    assert webui["depends_on"] == {
        "kairyu": {"condition": "service_healthy"}
    }
    assert "healthcheck" in kairyu
    assert "healthcheck" in webui

    _assert_immutable_image(
        webui["image"], "ghcr.io/open-webui/open-webui:v0.11.0-slim"
    )
    assert browser["image"] == "kairyu-webui-browser:playwright-1.60.0"
    assert browser["build"] == {
        "context": "../..",
        "dockerfile": "deploy/compose/Dockerfile.webui-browser",
    }
    dockerfile_lines = WEBUI_BROWSER_DOCKERFILE.read_text(
        encoding="utf-8"
    ).splitlines()
    _assert_immutable_image(
        dockerfile_lines[0].removeprefix("FROM "),
        "mcr.microsoft.com/playwright:v1.60.0-noble",
    )
    package = json.loads(WEBUI_BROWSER_PACKAGE.read_text(encoding="utf-8"))
    lock = json.loads(WEBUI_BROWSER_LOCK.read_text(encoding="utf-8"))
    assert package["dependencies"] == {"playwright": "1.60.0"}
    assert lock["packages"]["node_modules/playwright"]["version"] == "1.60.0"
    assert lock["packages"]["node_modules/playwright-core"]["version"] == "1.60.0"
    assert "npm ci --omit=dev --ignore-scripts" in "\n".join(dockerfile_lines)
    assert browser["profiles"] == ["browser-smoke"]
    assert browser["depends_on"] == {
        "webui": {"condition": "service_healthy"}
    }
    assert browser["environment"]["WEBUI_BASE_URL"] == "http://webui:8080"
    assert browser["environment"]["EXPECTED_DIRECT_MODEL"] == "default"
    assert browser["environment"]["EXPECTED_AUTO_MODEL"] == "kairyu-auto"
    assert browser["environment"]["WEBUI_SMOKE_EMAIL"]
    assert browser["environment"]["WEBUI_SMOKE_PASSWORD"]
    assert browser["volumes"] == [
        "../../scripts/webui_browser_smoke.mjs:/work/webui_browser_smoke.mjs:ro"
    ]
    assert browser["command"] == "node /work/webui_browser_smoke.mjs"


def test_webui_browser_gate_binds_results_to_new_assistant_messages():
    smoke = WEBUI_BROWSER_SMOKE.read_text(encoding="utf-8")

    assert smoke.count("before + 2") == 2
    assert smoke.count(".nth(before + 1)") == 2
    assert ".copy-response-button" in smoke
    assert "requestPath === '/api/chat/completions'" in smoke
    assert "taskReceipt.task_ids.length === 1" in smoke
    assert "UI chat completed with a visible error" in smoke
    assert "await sendUiMessage(autoModel, reconnectMarker)" in smoke
    assert "page.locator('body').filter" not in smoke
    assert "/api/v1/files/?process=true&process_in_background=false" in smoke
    assert "/api/v1/retrieval/query/doc" in smoke
    assert "!ragQuery.includes(ragNeedle)" in smoke
    assert "!ragQuery.includes(ragAnswer)" in smoke
    assert "retrieval.text.includes(ragNeedle)" in smoke
    assert "answer.includes(ragAnswer)" in smoke
    assert "answer.includes('[1]')" in smoke
    assert (
        "document ingest, vector retrieval, and RAG answer"
        in smoke
    )
    assert "RAG retrieval and answer after gateway restart" in smoke


async def test_webui_mounted_config_discovers_chat_and_embedding_models(
    monkeypatch,
):
    assert WEBUI_CONFIG.is_file(), "WebUI DeploymentSpec is missing"
    assert WEBUI_ORCHESTRATOR.is_file(), "WebUI orchestrator spec is missing"
    spec = load_deployment_spec(WEBUI_CONFIG)
    assert list(spec.engines) == ["default"]
    assert spec.engines["default"].backend == "mock"
    assert spec.pools == {}
    assert spec.orchestrator is not None
    assert spec.orchestrator.spec == WEBUI_ORCHESTRATOR.name
    assert spec.orchestrators == {}
    assert spec.server.host == "0.0.0.0"
    assert spec.server.port == 8000
    assert spec.server.api_keys_env is None
    assert list(spec.embeddings) == ["embed-small"]
    embedding = spec.embeddings["embed-small"]
    assert embedding.backend == "fastembed"
    assert embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert embedding.model_path == "/opt/kairyu/models/all-MiniLM-L6-v2"
    assert embedding.revision == EMBEDDING_MODEL_REVISION
    assert embedding.model_sha256 == EMBEDDING_MODEL_SHA256
    assert embedding.provenance_sha256 == EMBEDDING_PROVENANCE_SHA256
    assert embedding.dimensions == 384
    assert embedding.batch_size == 64
    assert embedding.threads == 2
    assert embedding.max_concurrency == 2

    constructed: dict[str, object] = {}

    def fake_fastembed(**options):
        constructed.update(options)
        return MockEmbeddingBackend(dimensions=options["dimensions"])

    monkeypatch.setitem(
        builder_module._EMBEDDING_BACKEND_FACTORIES,
        "fastembed",
        fake_fastembed,
    )
    app = build_app_from_config(WEBUI_CONFIG)
    async with app.router.lifespan_context(app):
        async with _client(app) as client:
            assert (await client.get("/readyz")).status_code == 200
            models = await client.get(
                "/v1/models", headers={"Authorization": "Bearer sk-local"}
            )

    assert models.status_code == 200
    assert [entry["id"] for entry in models.json()["data"]] == [
        "default",
        "kairyu-auto",
        "embed-small",
    ]
    assert constructed == {
        "dimensions": 384,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "model_path": Path("/opt/kairyu/models/all-MiniLM-L6-v2"),
        "revision": EMBEDDING_MODEL_REVISION,
        "model_sha256": EMBEDDING_MODEL_SHA256,
        "provenance_sha256": EMBEDDING_PROVENANCE_SHA256,
        "batch_size": 64,
        "threads": 2,
        "max_concurrency": 2,
    }


def test_validator_accepts_checked_in_inventory_from_any_cwd(tmp_path):
    result = _run_validator(
        *(path.resolve() for path in COMPOSE_FILES), cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr


def test_validator_accepts_valid_short_and_long_bind_syntax(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("engines:\n  default:\n    backend: mock\n", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text(
        yaml.safe_dump(
            {
                "services": {
                    "short": {
                        "volumes": [
                            "./config.yaml:/etc/kairyu/config.yaml:ro"
                        ]
                    },
                    "long": {
                        "volumes": [
                            {
                                "type": "bind",
                                "source": "./data",
                                "target": "/data",
                            }
                        ]
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    result = _run_validator(compose_file, cwd=tmp_path.parent)
    assert result.returncode == 0, result.stderr


def test_validator_reports_every_missing_short_and_long_bind(tmp_path):
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text(
        yaml.safe_dump(
            {
                "services": {
                    "short-service": {"volumes": ["./missing-short:/short"]},
                    "long-service": {
                        "volumes": [
                            {
                                "type": "bind",
                                "source": "./missing-long",
                                "target": "/long",
                            }
                        ]
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    result = _run_validator(compose_file, cwd=tmp_path.parent)
    assert result.returncode == 1
    for marker in (
        str(compose_file),
        "short-service",
        "./missing-short",
        "long-service",
        "./missing-long",
    ):
        assert marker in result.stderr


def test_validator_ignores_named_substituted_and_absolute_operator_sources(tmp_path):
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text(
        yaml.safe_dump(
            {
                "services": {
                    "operator": {
                        "volumes": [
                            "cache:/cache",
                            "${HOST_DIR:-/operator/data}:/substituted",
                            "/operator/data:/absolute",
                            {
                                "type": "bind",
                                "source": "${LONG_DIR:-/operator/data}",
                                "target": "/long-substituted",
                            },
                            {
                                "type": "bind",
                                "source": "/operator/data",
                                "target": "/long-absolute",
                            },
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = _run_validator(compose_file, cwd=tmp_path.parent)
    assert result.returncode == 0, result.stderr


def test_validator_rejects_invalid_mounted_deployment_spec(tmp_path):
    (tmp_path / "invalid.yaml").write_text("engines: {}\n", encoding="utf-8")
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text(
        yaml.safe_dump(
            {
                "services": {
                    "kairyu": {
                        "volumes": [
                            "./invalid.yaml:/etc/kairyu/config.yaml:ro"
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = _run_validator(compose_file, cwd=tmp_path.parent)
    assert result.returncode == 1
    assert str(compose_file) in result.stderr
    assert "kairyu" in result.stderr
    assert "./invalid.yaml" in result.stderr
    assert "invalid DeploymentSpec" in result.stderr


def test_validator_rejects_untracked_source_for_tracked_compose(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    compose_file = repo / "compose.yaml"
    compose_file.write_text(
        "services:\n"
        "  kairyu:\n"
        "    volumes:\n"
        "      - ./config.yaml:/etc/kairyu/config.yaml:ro\n",
        encoding="utf-8",
    )
    config = repo / "config.yaml"
    config.write_text("engines:\n  default:\n    backend: mock\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "compose.yaml"], check=True)

    untracked = _run_validator(compose_file, cwd=tmp_path)
    assert untracked.returncode == 1
    assert "./config.yaml" in untracked.stderr
    assert "not tracked" in untracked.stderr

    subprocess.run(["git", "-C", str(repo), "add", "config.yaml"], check=True)
    tracked = _run_validator(compose_file, cwd=tmp_path)
    assert tracked.returncode == 0, tracked.stderr


def test_validator_accepts_relative_bind_that_resolves_to_tracked_repo_root(tmp_path):
    repo = tmp_path / "repo"
    compose_dir = repo / "deploy" / "compose"
    compose_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    compose_file = compose_dir / "docker-compose.yaml"
    compose_file.write_text(
        "services:\n"
        "  workspace:\n"
        "    volumes:\n"
        "      - ../..:/workspace:ro\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "-C", str(repo), "add", "deploy/compose/docker-compose.yaml"],
        check=True,
    )

    result = _run_validator(compose_file, cwd=tmp_path)

    assert result.returncode == 0, result.stderr


def test_smoke_validation_failure_never_invokes_docker_cleanup(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text("#!/bin/sh\nexit 23\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    docker_marker = tmp_path / "docker-called"
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        '#!/bin/sh\ntouch "$DOCKER_MARKER"\n', encoding="utf-8"
    )
    fake_docker.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "DOCKER_MARKER": str(docker_marker),
    }

    result = subprocess.run(
        ["/bin/bash", str(COMPOSE_SMOKE.resolve())],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 23
    assert not docker_marker.exists()


def test_ci_and_default_smoke_fail_fast_through_validator():
    smoke = COMPOSE_SMOKE.read_text(encoding="utf-8")
    assert smoke.index("validate_compose_binds.py") < smoke.index('echo "== up =="')
    assert 'uv run --frozen --project "$REPO_ROOT" --no-dev' in smoke

    compose_steps = _load_yaml(CI_WORKFLOW)["jobs"]["compose-smoke"]["steps"]
    validation_index = next(
        index
        for index, step in enumerate(compose_steps)
        if step.get("name") == "Validate Compose bind sources"
    )
    smoke_index = next(
        index
        for index, step in enumerate(compose_steps)
        if step.get("name") == "Compose integration test"
    )
    assert validation_index < smoke_index
    assert any(
        step.get("uses") == "astral-sh/setup-uv@v5"
        for step in compose_steps[:validation_index]
    )
    validation_command = compose_steps[validation_index]["run"]
    assert validation_command.startswith(
        "uv run --frozen --no-dev python scripts/validate_compose_binds.py"
    )
    assert "deploy/compose/docker-compose*.yaml" in validation_command


def test_compose_recovery_allows_detection_race_but_asserts_the_result():
    smoke = COMPOSE_SMOKE.read_text(encoding="utf-8")

    assert "ulimit -n 65536" in smoke
    assert "|| true # eat the eject trigger" not in smoke
    assert 'case "$convergence_status" in' in smoke
    assert "200|502) ;;" in smoke
    assert (
        '*) fail "request during ejection convergence returned '
        '$convergence_status" ;;'
        in smoke
    )
    assert (
        "wait_for_metric "
        '\'kairyu_replica_healthy{pool="llama",replica="0"}\' "0.0" 10'
        in smoke
    )
    assert (
        "wait_for_metric "
        '\'kairyu_replica_healthy{pool="llama",replica="0"}\' "1.0" 30'
        in smoke
    )
    assert '[[ "$(chat "user-$i")" == 200 ]]' in smoke


def test_cpu_image_bounds_uv_install_concurrency_for_high_core_builders():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "UV_COMPILE_BYTECODE=1" in dockerfile
    assert "UV_CONCURRENT_INSTALLS=8" in dockerfile


def test_root_docker_context_is_an_explicit_source_allowlist():
    rules = DOCKERIGNORE.read_text(encoding="utf-8").splitlines()

    assert rules[0] == "**"
    assert {
        "!Dockerfile",
        "!Dockerfile.cuda",
        "!README.md",
        "!pyproject.toml",
        "!uv.lock",
        "!kairyu/**",
        "!scripts/prefetch_embedding_model.py",
        "!scripts/async_request_retention.py",
        "!deploy/compose/Dockerfile.webui-browser",
        "!deploy/compose/webui-browser/package.json",
        "!deploy/compose/webui-browser/package-lock.json",
    }.issubset(rules)
    assert not any(rule in rules for rule in ("!.git", "!.venv", "!.claude", "!bench"))


def test_compose_smoke_runs_real_cross_process_otel_trace_gate():
    compose = _load_yaml(COMPOSE_DIR / "docker-compose.yaml")
    services = compose["services"]
    for name in ("gateway", "replica-1", "replica-2", "replica-3"):
        environment = services[name]["environment"]
        assert environment["OTEL_TRACES_EXPORTER"] == "console"
        assert environment["OTEL_SERVICE_NAME"] == f"kairyu-{name}"

    assert load_deployment_spec(COMPOSE_DIR / "gateway.yaml").server.tracing
    assert load_deployment_spec(COMPOSE_DIR / "replica.yaml").server.tracing
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert dockerfile.count("--extra fleet --extra otel ;;") == 2
    assert (
        dockerfile.count(
            "--extra fleet --extra otel --extra embeddings ;;"
        )
        == 2
    )

    smoke = COMPOSE_SMOKE.read_text(encoding="utf-8")
    assert OTEL_TRACE_VERIFIER.is_file()
    assert str(OTEL_TRACE_VERIFIER) in smoke
    assert "deployed OTel gateway-to-replica trace (gate F1d)" in smoke
    assert "--request-id" in smoke
    assert "--response-body" in smoke
    assert "--forbidden" in smoke
    assert "compose logs --no-color --no-log-prefix" in smoke


def test_compose_validator_imports_without_engine_extra():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.modules['torch'] = None; "
                "import scripts.validate_compose_binds"
            ),
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_webui_smoke_validation_failure_never_invokes_docker(tmp_path):
    env, docker_log, _curl_log = _webui_smoke_env(tmp_path, uv_exit=23)

    result = subprocess.run(
        ["/bin/bash", str(WEBUI_SMOKE.resolve())],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 23
    assert not docker_log.exists()


def test_webui_smoke_rejects_wrong_rendered_internal_url_before_startup(tmp_path):
    env, docker_log, _curl_log = _webui_smoke_env(
        tmp_path, rendered_base_url="http://wrong.example/v1"
    )

    result = subprocess.run(
        ["/bin/bash", str(WEBUI_SMOKE.resolve())],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "http://wrong.example/v1" in result.stderr
    assert docker_log.read_text(encoding="utf-8").splitlines() == [
        (
            f"compose --project-name {WEBUI_SMOKE_PROJECT} "
            f"-f {WEBUI_COMPOSE.resolve()} config --format json"
        )
    ]


def test_webui_smoke_runs_full_browser_outage_recovery_contract(tmp_path):
    env, docker_log, curl_log = _webui_smoke_env(tmp_path)

    result = subprocess.run(
        ["/bin/bash", str(WEBUI_SMOKE.resolve())],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    commands = docker_log.read_text(encoding="utf-8").splitlines()
    compose = (
        f"compose --project-name {WEBUI_SMOKE_PROJECT} "
        f"-f {WEBUI_COMPOSE.resolve()}"
    )
    assert commands == [
        f"{compose} config --format json",
        (
            f"{compose} up -d --build --wait "
            "--wait-timeout 240 kairyu webui"
        ),
        f"{compose} --profile browser-smoke build browser",
        (
            f"{compose} --profile browser-smoke run --rm --no-deps "
            "-e WEBUI_SMOKE_PHASE=initial browser"
        ),
        f"{compose} stop kairyu",
        (
            f"{compose} --profile browser-smoke run --rm --no-deps "
            "-e WEBUI_SMOKE_PHASE=outage browser"
        ),
        f"{compose} start kairyu",
        (
            f"{compose} --profile browser-smoke run --rm --no-deps "
            "-e WEBUI_SMOKE_PHASE=recovery browser"
        ),
        f"{compose} --profile browser-smoke down --volumes --remove-orphans",
    ]
    curl_commands = curl_log.read_text(encoding="utf-8").splitlines()
    assert len(curl_commands) == 4
    assert all("--connect-timeout 2 --max-time 10" in line for line in curl_commands)
    assert "/v1/embeddings" in curl_commands[0]
    assert '"model":"embed-small"' in curl_commands[0]
    assert curl_commands[1].endswith("http://localhost:8000/metrics")
    assert curl_commands[2].endswith("http://localhost:8000/readyz")
    assert curl_commands[3].endswith("http://localhost:8000/metrics")


def test_webui_smoke_tears_down_if_browser_phase_fails(tmp_path):
    env, docker_log, _curl_log = _webui_smoke_env(
        tmp_path, docker_fail_phase="outage"
    )

    result = subprocess.run(
        ["/bin/bash", str(WEBUI_SMOKE.resolve())],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert docker_log.read_text(encoding="utf-8").splitlines()[-1] == (
        f"compose --project-name {WEBUI_SMOKE_PROJECT} "
        f"-f {WEBUI_COMPOSE.resolve()} --profile browser-smoke "
        "down --volumes --remove-orphans"
    )


def test_webui_smoke_bounds_recovery_readiness_and_always_tears_down(tmp_path):
    env, docker_log, curl_log = _webui_smoke_env(
        tmp_path,
        curl_exit=22,
        curl_fail_path="/readyz",
    )

    result = subprocess.run(
        ["/bin/bash", str(WEBUI_SMOKE.resolve())],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode != 0
    commands = docker_log.read_text(encoding="utf-8").splitlines()
    assert not any("WEBUI_SMOKE_PHASE=recovery" in command for command in commands)
    assert commands[-1] == (
        f"compose --project-name {WEBUI_SMOKE_PROJECT} "
        f"-f {WEBUI_COMPOSE.resolve()} --profile browser-smoke "
        "down --volumes --remove-orphans"
    )
    curl_commands = curl_log.read_text(encoding="utf-8").splitlines()
    assert 4 <= len(curl_commands) <= 120
    assert all("--connect-timeout" in command for command in curl_commands)
    assert all("--max-time" in command for command in curl_commands)
    assert "/v1/embeddings" in curl_commands[0]
    assert curl_commands[1].endswith("/metrics")
    readiness_commands = [
        command for command in curl_commands if command.endswith("/readyz")
    ]
    assert 1 <= len(readiness_commands) <= 90
    assert len(readiness_commands) == len(curl_commands) - 2


def test_ci_runs_webui_smoke_in_dedicated_parallel_job():
    assert WEBUI_SMOKE.is_file(), "WebUI smoke script is missing"
    assert WEBUI_SMOKE.stat().st_mode & 0o111
    smoke = WEBUI_SMOKE.read_text(encoding="utf-8")
    assert smoke.index("validate_compose_binds.py") < smoke.index(
        "config --format json"
    )
    assert 'WEBUI_SMOKE_PROJECT_NAME:-kairyu-webui-smoke-' in smoke
    assert 'docker compose --project-name "$SMOKE_PROJECT_NAME"' in smoke
    assert "http://kairyu:8000/v1" in smoke
    assert "compose up -d --build --wait --wait-timeout" in smoke
    assert '"$COMPOSE_WAIT_SECONDS" kairyu webui' in smoke
    assert "compose --profile browser-smoke build browser" in smoke
    assert 'WEBUI_SMOKE_PHASE="$phase"' in smoke
    assert "browser_phase initial" in smoke
    assert "browser_phase outage" in smoke
    assert "browser_phase recovery" in smoke
    assert "compose stop kairyu" in smoke
    assert "compose start kairyu" in smoke
    assert '"$BASE_URL/readyz"' in smoke
    assert "assert_embedding_contract" in smoke
    assert "assert_embedding_metrics 4" in smoke
    assert "assert_embedding_metrics 2" in smoke
    assert '"$BASE_URL/v1/embeddings"' in smoke
    assert '"$BASE_URL/metrics"' in smoke

    jobs = _load_yaml(CI_WORKFLOW)["jobs"]
    compose_steps = jobs["compose-smoke"]["steps"]
    assert not any(
        step.get("name") == "WebUI Kairyu smoke"
        or "webui_smoke.sh" in step.get("run", "")
        for step in compose_steps
    )
    webui_job = jobs["webui-e2e"]
    assert webui_job["runs-on"] == "ubuntu-latest"
    assert webui_job["timeout-minutes"] <= 45
    assert "needs" not in webui_job
    webui_steps = webui_job["steps"]
    assert any(step.get("uses") == "actions/checkout@v4" for step in webui_steps)
    assert any(step.get("uses") == "astral-sh/setup-uv@v5" for step in webui_steps)
    assert [
        step.get("run")
        for step in webui_steps
        if "webui_smoke.sh" in step.get("run", "")
    ] == ["./scripts/webui_smoke.sh"]


def test_webui_smoke_uses_only_bounded_http_requests():
    smoke = WEBUI_SMOKE.read_text(encoding="utf-8")

    assert 'curl_bounded() { curl --connect-timeout 2 --max-time 10 "$@"; }' in smoke
    raw_curl_lines = [
        line.strip()
        for line in smoke.splitlines()
        if "curl " in line and not line.lstrip().startswith("#")
    ]
    assert raw_curl_lines == [
        'curl_bounded() { curl --connect-timeout 2 --max-time 10 "$@"; }'
    ]
    assert "curl_bounded" in smoke
