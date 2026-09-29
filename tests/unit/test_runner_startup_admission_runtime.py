"""Executable cache-placement admission runtime coverage."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

import kairyu.runners.startup_admission_runtime as runtime_module
from kairyu.entrypoints.cli import _build_parser
from kairyu.runners import (
    RUNNER_CACHE_STARTUP_SCHEDULING_GATE,
    RUNNER_CACHE_STARTUP_TARGET_ANNOTATION,
    InMemoryRunnerCachePlacementAdmissionStore,
    RunnerCachePlacementAdmissionPlan,
    RunnerCachePlacementAdmissionRuntimeConfig,
    RunnerCachePlacementBindingAuthorizationError,
    RunnerCachePlacementBindingAuthorizer,
    RunnerCacheStartupBinding,
    RunnerCacheStartupPlacement,
    build_runner_cache_placement_admission_runtime,
    load_runner_cache_placement_admission_runtime_config,
)

NOW = datetime.now(UTC)
TARGET = "statefulset/model-serving/qwen-runners"
USERNAME = "system:serviceaccount:kairyu:statefulset-controller"
TOKEN = "a" * 32


def _binding(*, suffix: str = "a") -> RunnerCacheStartupBinding:
    placement = RunnerCacheStartupPlacement(
        placement_id=f"placement-{suffix}",
        node_name=f"gpu-{suffix}",
        resource_flavor="h100-sxm",
        profile_id="h100-sxm-tp1",
        compatibility_approval_id="compat-qwen-h100",
        manifest_digest="a" * 64,
        pin_owner=f"prestage/model-serving/qwen/placement-{suffix}",
        prestage_command_id=hashlib.sha256(f"command-{suffix}".encode()).hexdigest(),
        prestage_command_generation=1,
        hint_index_revision=10,
        resident_record_generation=20,
        hint_observed_at=NOW - timedelta(seconds=1),
        hint_valid_until=NOW + timedelta(minutes=5),
    )
    payload = {
        "schema_version": "runner-cache-startup-binding-v1",
        "decision_id": f"decision-{suffix}",
        "decision_fingerprint": hashlib.sha256(f"decision-{suffix}".encode()).hexdigest(),
        "target_id": TARGET,
        "target_revision": 7,
        "deployment_id": "model-serving/qwen",
        "model_class": "qwen-14b",
        "model_id": "org/qwen",
        "model_revision": "revision-a",
        "manifest_digest": "a" * 64,
        "placement_binding_id": f"placement-binding-{suffix}",
        "prewarm_snapshot_id": f"snapshot-{suffix}",
        "prewarm_cache_revision": 9,
        "bound_at": NOW,
        "valid_until": NOW + timedelta(minutes=5),
        "placements": (placement,),
    }
    unsigned = RunnerCacheStartupBinding.model_construct(binding_id="0" * 64, **payload)
    encoded = json.dumps(
        unsigned.model_dump(mode="json", exclude={"binding_id"}),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return RunnerCacheStartupBinding(
        binding_id=hashlib.sha256(encoded).hexdigest(),
        **payload,
    )


def _files(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "postgres_dsn_file": tmp_path / "postgres-dsn",
        "authorization_bearer_token_file": tmp_path / "authority-token",
        "tls_cert_file": tmp_path / "tls.crt",
        "tls_key_file": tmp_path / "tls.key",
    }
    paths["postgres_dsn_file"].write_text(
        "postgresql://admission@postgres/admission\n", encoding="utf-8"
    )
    paths["postgres_dsn_file"].chmod(0o640)
    paths["authorization_bearer_token_file"].write_text(f"{TOKEN}\n", encoding="utf-8")
    paths["authorization_bearer_token_file"].chmod(0o640)
    paths["tls_cert_file"].write_text("test certificate\n", encoding="utf-8")
    paths["tls_key_file"].write_text("test private key\n", encoding="utf-8")
    paths["tls_key_file"].chmod(0o600)
    return paths


def _config(tmp_path: Path, **updates) -> RunnerCachePlacementAdmissionRuntimeConfig:
    value: dict[str, Any] = {
        **_files(tmp_path),
        "authorization_url": "https://authority.test/v1/reauthorize",
        "authorization_ready_url": "https://authority.test/readyz",
    }
    value.update(updates)
    return RunnerCachePlacementAdmissionRuntimeConfig.model_validate(value)


def _authorization_response(request: httpx.Request, binding: RunnerCacheStartupBinding):
    payload = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "schema_version": ("kairyu-runner-cache-placement-binding-authorization-response-v1"),
            "nonce": payload["nonce"],
            "binding": binding.model_dump(mode="json"),
        },
    )


def _authorizer(
    handler,
    *,
    request_limit: int = 1024 * 1024,
    response_limit: int = 1024 * 1024,
):
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": f"Bearer {TOKEN}"},
        trust_env=False,
    )
    return (
        RunnerCachePlacementBindingAuthorizer(
            client,
            authorization_url="https://authority.test/v1/reauthorize",
            readiness_url="https://authority.test/readyz",
            request_limit_bytes=request_limit,
            response_limit_bytes=response_limit,
        ),
        client,
    )


def test_authorizer_binds_nonce_and_exact_request() -> None:
    binding = _binding()
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return _authorization_response(request, binding)

    authorizer, client = _authorizer(handler)
    try:
        assert authorizer.reauthorize(binding) == binding
    finally:
        client.close()

    request = observed[0]
    payload = json.loads(request.content)
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    assert request.headers["cache-control"] == "no-store"
    assert payload["binding"] == binding.model_dump(mode="json")
    assert len(payload["nonce"]) == 64


@pytest.mark.parametrize("failure", ["nonce", "duplicate", "overflow", "oversized"])
def test_authorizer_rejects_untrusted_response_shapes(failure: str) -> None:
    binding = _binding()

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if failure == "nonce":
            payload["nonce"] = "0" * 64
            return httpx.Response(
                200,
                json={
                    "schema_version": (
                        "kairyu-runner-cache-placement-binding-authorization-response-v1"
                    ),
                    "nonce": payload["nonce"],
                    "binding": binding.model_dump(mode="json"),
                },
            )
        if failure == "duplicate":
            return httpx.Response(
                200,
                headers={"Content-Type": "application/json"},
                content=b'{"nonce":"a","nonce":"b"}',
            )
        if failure == "overflow":
            return httpx.Response(
                200,
                headers={"Content-Type": "application/json"},
                content=b'{"value":1e999}',
            )
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            content=b"x" * 128,
        )

    authorizer, client = _authorizer(handler, response_limit=64)
    try:
        with pytest.raises(RunnerCachePlacementBindingAuthorizationError):
            authorizer.reauthorize(binding)
    finally:
        client.close()


def test_authorizer_rejects_redirect_and_nonempty_readiness() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(307, headers={"Location": "https://evil.invalid"})
        return httpx.Response(204, content=b"x")

    authorizer, client = _authorizer(handler)
    try:
        with pytest.raises(RunnerCachePlacementBindingAuthorizationError, match="rejected"):
            authorizer.reauthorize(_binding())
        with pytest.raises(RunnerCachePlacementBindingAuthorizationError, match="must be empty"):
            authorizer.check_ready()
    finally:
        client.close()


def test_authorizer_rejects_oversized_request_before_transport() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("oversized requests must not reach the transport")

    authorizer, client = _authorizer(handler, request_limit=64)
    try:
        with pytest.raises(
            RunnerCachePlacementBindingAuthorizationError,
            match="request exceeds",
        ):
            authorizer.reauthorize(_binding())
    finally:
        client.close()
    assert calls == 0


def test_runtime_config_loader_is_strict_and_https_by_default(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = tmp_path / "runtime.json"
    path.write_text(config.model_dump_json(), encoding="utf-8")

    assert load_runner_cache_placement_admission_runtime_config(path) == config

    path.write_text(
        '{"schema_version":"kairyu-runner-cache-placement-admission-runtime-v1",'
        '"schema_version":"kairyu-runner-cache-placement-admission-runtime-v1"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config file is invalid"):
        load_runner_cache_placement_admission_runtime_config(path)
    with pytest.raises(ValueError, match="absolute HTTPS"):
        _config(
            tmp_path,
            authorization_url="http://authority.test/v1/reauthorize",
            authorization_ready_url="http://authority.test/readyz",
        )
    with pytest.raises(ValueError, match="share one origin"):
        _config(
            tmp_path,
            authorization_ready_url="https://other.test/readyz",
        )


class FakePostgresStore(InMemoryRunnerCachePlacementAdmissionStore):
    def __init__(self, _dsn: str, **_kwargs) -> None:
        super().__init__()
        self.closed = False
        self.ready_checks = 0

    def check_ready(self) -> None:
        self.ready_checks += 1

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_runtime_assembles_app_checks_dependencies_and_closes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    binding = _binding()
    stores: list[FakePostgresStore] = []
    clients: list[httpx.Client] = []

    def create_store(*args, **kwargs):
        store = FakePostgresStore(*args, **kwargs)
        stores.append(store)
        return store

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(204)
        return _authorization_response(request, binding)

    def client_factory(**kwargs):
        assert kwargs["follow_redirects"] is False
        assert kwargs["trust_env"] is False
        client = httpx.Client(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(
        runtime_module,
        "PostgresRunnerCachePlacementAdmissionStore",
        create_store,
    )
    config = _config(tmp_path)
    runtime = build_runner_cache_placement_admission_runtime(
        config,
        http_client_factory=client_factory,
    )
    stores[0].register(
        RunnerCachePlacementAdmissionPlan(
            binding=binding,
            release_id="release-a",
            namespace="model-serving",
            owner_api_version="apps/v1",
            owner_kind="StatefulSet",
            owner_name="qwen-runners",
            owner_uid="workload-uid-a",
            creator_username=USERNAME,
            registered_at=NOW + timedelta(seconds=1),
        )
    )
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "namespace": "model-serving",
            "name": "qwen-7",
            "ownerReferences": [
                {
                    "apiVersion": "apps/v1",
                    "kind": "StatefulSet",
                    "name": "qwen-runners",
                    "uid": "workload-uid-a",
                    "controller": True,
                }
            ],
            "annotations": {RUNNER_CACHE_STARTUP_TARGET_ANNOTATION: TARGET},
        },
        "spec": {
            "schedulingGates": [{"name": RUNNER_CACHE_STARTUP_SCHEDULING_GATE}],
            "containers": [{"name": "runner", "image": "runner:test"}],
        },
    }
    review = {
        "apiVersion": "admission.k8s.io/v1",
        "kind": "AdmissionReview",
        "request": {
            "uid": "admission-a",
            "kind": {"group": "", "version": "v1", "kind": "Pod"},
            "resource": {"group": "", "version": "v1", "resource": "pods"},
            "name": "qwen-7",
            "namespace": "model-serving",
            "operation": "CREATE",
            "userInfo": {"username": USERNAME},
            "object": pod,
        },
    }

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=runtime.app),
        base_url="https://admission.test",
    ) as client:
        health = await client.get("/health")
        ready = await client.get("/readyz")
        admitted = await client.post("/v1/admit", json=review)

    assert health.json() == {"status": "ok"}
    assert ready.status_code == 200
    assert admitted.json()["response"]["allowed"] is True
    assert stores[0].ready_checks == 1
    runtime.close()
    runtime.close()
    assert stores[0].closed is True
    assert clients[0].is_closed is True


def test_runtime_rejects_broad_tls_key_permissions_before_dependencies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    config.tls_key_file.chmod(0o644)
    opened = False

    def create_store(*_args, **_kwargs):
        nonlocal opened
        opened = True

    monkeypatch.setattr(
        runtime_module,
        "PostgresRunnerCachePlacementAdmissionStore",
        create_store,
    )
    with pytest.raises(ValueError, match="permissions are too broad"):
        build_runner_cache_placement_admission_runtime(config)
    assert opened is False


@pytest.mark.parametrize(
    "secret_name",
    ["postgres_dsn_file", "authorization_bearer_token_file"],
)
def test_runtime_rejects_broad_credential_permissions(
    tmp_path: Path,
    monkeypatch,
    secret_name: str,
) -> None:
    config = _config(tmp_path)
    getattr(config, secret_name).chmod(0o644)
    opened = False

    def create_store(*_args, **_kwargs):
        nonlocal opened
        opened = True

    monkeypatch.setattr(
        runtime_module,
        "PostgresRunnerCachePlacementAdmissionStore",
        create_store,
    )
    with pytest.raises(ValueError, match="permissions are too broad"):
        build_runner_cache_placement_admission_runtime(config)
    assert opened is False


@pytest.mark.parametrize("failure", ["readiness", "app"])
def test_runtime_build_failure_closes_store_and_client(
    tmp_path: Path,
    monkeypatch,
    failure: str,
) -> None:
    stores: list[FakePostgresStore] = []
    clients: list[httpx.Client] = []

    def create_store(*args, **kwargs):
        store = FakePostgresStore(*args, **kwargs)
        stores.append(store)
        return store

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(503 if failure == "readiness" else 204)
        raise AssertionError("reauthorization is not expected during build")

    def client_factory(**kwargs):
        client = httpx.Client(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(
        runtime_module,
        "PostgresRunnerCachePlacementAdmissionStore",
        create_store,
    )
    if failure == "app":

        def fail_app(**_kwargs):
            raise RuntimeError("app failed")

        monkeypatch.setattr(
            runtime_module,
            "create_runner_cache_placement_admission_app",
            fail_app,
        )

    with pytest.raises((RunnerCachePlacementBindingAuthorizationError, RuntimeError)):
        build_runner_cache_placement_admission_runtime(
            _config(tmp_path),
            http_client_factory=client_factory,
        )

    assert stores[0].closed is True
    assert clients[0].is_closed is True


def test_cli_exposes_placement_admission_serve() -> None:
    args = _build_parser().parse_args(["placement-admission", "serve", "/config.json"])
    assert args.command == "placement-admission"
    assert args.placement_admission_command == "serve"
    assert args.config == Path("/config.json")
