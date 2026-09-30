"""D3.20 production placement-binding authority assembly coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI

import kairyu.runners.startup_binding_authority_production as production
from kairyu.runners import (
    KubernetesPlacementBindingLiveTarget,
    NodeModelCacheAgentEndpoint,
    RunnerCachePlacementBindingAuthorityRuntimeConfig,
    RunnerLeaderElectionRuntimeConfig,
)


def _write(path: Path, value: str, *, mode: int = 0o600) -> Path:
    path.write_text(value, encoding="utf-8")
    path.chmod(mode)
    return path


def _config(tmp_path: Path) -> production.RunnerCachePlacementBindingProductionRuntimeConfig:
    authority_token = _write(tmp_path / "authority-token", "a" * 32)
    node_token = _write(tmp_path / "node-token", "n" * 32)
    dsn = _write(tmp_path / "postgres-dsn", "postgresql://authority@postgres/kairyu")
    kube_token = _write(tmp_path / "kube-token", "service-account-token")
    kube_ca = _write(tmp_path / "kube-ca.pem", "kube-ca", mode=0o644)
    node_ca = _write(tmp_path / "node-ca.pem", "node-ca", mode=0o644)
    tls_cert = _write(tmp_path / "tls.crt", "certificate", mode=0o644)
    tls_key = _write(tmp_path / "tls.key", "private-key")
    return production.RunnerCachePlacementBindingProductionRuntimeConfig(
        authority=RunnerCachePlacementBindingAuthorityRuntimeConfig(
            bearer_token_file=authority_token,
            tls_cert_file=tls_cert,
            tls_key_file=tls_key,
            queue_wait_timeout_s=0.1,
            request_timeout_s=1.0,
            backend_timeout_s=0.5,
            authorization_client_timeout_s=2.0,
            transport_margin_s=0.25,
        ),
        leadership=RunnerLeaderElectionRuntimeConfig(
            election_id="runner-control-plane",
            holder_id="authority-0",
        ),
        postgres=production.RunnerAuthorityPostgresConfig(
            dsn_file=dsn,
            store_id="runner-authority",
        ),
        kubernetes=production.RunnerAuthorityKubernetesConfig(
            targets=(
                KubernetesPlacementBindingLiveTarget(
                    model_class="large",
                    target_kind="Deployment",
                    namespace="models",
                    name="large-runner",
                    authority_namespace="kairyu-system",
                    kueue_namespace="models",
                    quota_snapshot_name="large-quota",
                    inventory_name="large-cache",
                    pod_set_name="runner",
                ),
            ),
            api_server="https://kubernetes.default.svc",
            token_file=kube_token,
            ca_file=kube_ca,
        ),
        node_evidence=production.RunnerAuthorityNodeEvidenceConfig(
            endpoints=(
                NodeModelCacheAgentEndpoint(
                    node_id="gpu-0",
                    base_url="https://cache-agent-gpu-0.kairyu-system.svc:8443",
                ),
            ),
            bearer_token_file=node_token,
            ca_file=node_ca,
        ),
    )


def test_config_requires_canonical_targets_endpoints_and_absolute_secrets(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="dsn_file"):
        production.RunnerAuthorityPostgresConfig(
            dsn_file=Path("relative"),
            store_id="runner-authority",
        )
    with pytest.raises(ValueError, match="canonical node IDs"):
        production.RunnerAuthorityNodeEvidenceConfig(
            endpoints=(
                NodeModelCacheAgentEndpoint(node_id="z", base_url="https://z.example"),
                NodeModelCacheAgentEndpoint(node_id="a", base_url="https://a.example"),
            ),
            bearer_token_file=config.node_evidence.bearer_token_file,
            ca_file=config.node_evidence.ca_file,
        )


def test_config_loader_rejects_duplicate_keys_and_broad_secret_permissions(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    path = _write(
        tmp_path / "runtime.json",
        json.dumps(config.model_dump(mode="json")),
        mode=0o644,
    )
    loaded = production.load_runner_cache_placement_binding_production_runtime_config(path)
    assert loaded == config

    duplicate = _write(tmp_path / "duplicate.json", '{"authority": {}, "authority": {}}')
    with pytest.raises(ValueError, match="config file is invalid"):
        production.load_runner_cache_placement_binding_production_runtime_config(duplicate)

    config.postgres.dsn_file.chmod(0o644)
    with pytest.raises(ValueError, match="permissions are too broad"):
        production.build_runner_cache_placement_binding_production_runtime(config)


def _patch_successful_assembly(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    class Resource:
        label = "resource"

        def __init__(self, *_args, **_kwargs) -> None:
            self.closed = False
            events.append(f"open:{self.label}")

        def close(self) -> None:
            if not self.closed:
                self.closed = True
                events.append(f"close:{self.label}")

    class LeaderStore(Resource):
        label = "leader-store"

    class AdmissionStore(Resource):
        label = "admission-store"

    class DecisionLog(Resource):
        label = "decision-log"

    class KubeReader(Resource):
        label = "kube-reader"

    class HttpClient(Resource):
        label = "node-http"

        async def aclose(self) -> None:
            self.close()

    class Evidence(Resource):
        label = "evidence"

        def __init__(self, client, *_args, **_kwargs) -> None:
            super().__init__()
            self.client = client

        def close(self) -> None:
            if not self.closed:
                super().close()
                self.client.close()

    class DurableReader:
        def __init__(self, *_args, **_kwargs) -> None:
            events.append("build:durable-reader")

    class CacheReader:
        def __init__(self, *_args, **_kwargs) -> None:
            events.append("build:cache-reader")

    class Source:
        def __init__(self, *_args, **_kwargs) -> None:
            events.append("build:source")

    class Leadership(Resource):
        label = "leadership"

        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()
            self.controller = object()

        def start(self) -> None:
            events.append("start:leadership")

        def check_ready(self) -> None:
            events.append("ready:leadership")

    class Authority:
        def __init__(self, *_args, **_kwargs) -> None:
            events.append("build:authority")

        def reauthorize(self, binding, **_kwargs):
            return binding

        def readiness(self, **_kwargs) -> None:
            events.append("ready:authority")

    class AuthorityRuntime:
        def __init__(self, close_resources) -> None:
            self.app = FastAPI()
            self._close_resources = close_resources
            self._closed = False

        def close(self) -> None:
            if not self._closed:
                self._closed = True
                events.append("close:authority-runtime")
                self._close_resources()

    def build_authority(_config, *, reauthorize, readiness_check, close_resources):
        assert callable(reauthorize)
        readiness_check(deadline_monotonic=1.0, backend_timeout_s=0.5)
        events.append("build:authority-runtime")
        return AuthorityRuntime(close_resources)

    monkeypatch.setattr(production, "PostgresRunnerLeaderLeaseStore", LeaderStore)
    monkeypatch.setattr(production, "PostgresRunnerCachePlacementAdmissionStore", AdmissionStore)
    monkeypatch.setattr(production, "PostgresScalingDecisionLog", DecisionLog)
    monkeypatch.setattr(
        production,
        "KubernetesKueueRunnerCachePlacementBindingReader",
        KubeReader,
    )
    monkeypatch.setattr(production.httpx, "AsyncClient", HttpClient)
    monkeypatch.setattr(production, "AuthenticatedNodeModelCacheLiveEvidenceClient", Evidence)
    monkeypatch.setattr(production, "PostgresRunnerCachePlacementBindingReader", DurableReader)
    monkeypatch.setattr(
        production,
        "AggregatingRunnerCachePlacementBindingCacheReader",
        CacheReader,
    )
    monkeypatch.setattr(production, "ComposedRunnerCachePlacementBindingLiveStateSource", Source)
    monkeypatch.setattr(production, "RunnerLeaderElectionRuntime", Leadership)
    monkeypatch.setattr(production, "ScalingControllerPlacementBindingAuthority", Authority)
    monkeypatch.setattr(
        production,
        "build_runner_cache_placement_binding_authority_runtime",
        build_authority,
    )


def test_builder_starts_after_complete_assembly_and_closes_in_reverse_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _patch_successful_assembly(monkeypatch, events)
    runtime = production.build_runner_cache_placement_binding_production_runtime(_config(tmp_path))

    assert isinstance(runtime.app, FastAPI)
    assert events.index("build:authority-runtime") < events.index("start:leadership")
    assert events[-4:] == [
        "ready:leadership",
        "ready:authority",
        "build:authority-runtime",
        "start:leadership",
    ]
    runtime.close()
    runtime.close()
    assert events[-8:] == [
        "close:authority-runtime",
        "close:leadership",
        "close:evidence",
        "close:node-http",
        "close:kube-reader",
        "close:decision-log",
        "close:admission-store",
        "close:leader-store",
    ]


def test_partial_construction_failure_closes_created_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Resource:
        def __init__(self, *_args, **_kwargs) -> None:
            self.label = _kwargs.pop("label", "resource")

        def close(self) -> None:
            events.append(f"close:{self.label}")
            if self.label == "decision-log":
                raise RuntimeError("decision log close failed")

    monkeypatch.setattr(
        production,
        "PostgresRunnerLeaderLeaseStore",
        lambda *_args, **_kwargs: Resource(label="leader-store"),
    )
    monkeypatch.setattr(
        production,
        "PostgresRunnerCachePlacementAdmissionStore",
        lambda *_args, **_kwargs: Resource(label="admission-store"),
    )
    monkeypatch.setattr(
        production,
        "PostgresScalingDecisionLog",
        lambda *_args, **_kwargs: Resource(label="decision-log"),
    )

    def fail_kube(*_args, **_kwargs):
        raise RuntimeError("kubernetes construction failed")

    monkeypatch.setattr(
        production,
        "KubernetesKueueRunnerCachePlacementBindingReader",
        fail_kube,
    )
    with pytest.raises(RuntimeError, match="kubernetes construction failed") as exc_info:
        production.build_runner_cache_placement_binding_production_runtime(_config(tmp_path))
    assert events == [
        "close:decision-log",
        "close:admission-store",
        "close:leader-store",
    ]
    assert exc_info.value.__notes__ == ["production authority cleanup also failed: RuntimeError"]


def test_resource_cleanup_failure_keeps_dependencies_open_for_retry() -> None:
    events: list[str] = []
    release_leadership = False
    resources = production._ResourceStack()

    def close_dependency() -> None:
        events.append("close:dependency")

    def close_leadership() -> None:
        events.append("close:leadership")
        if not release_leadership:
            raise RuntimeError("leader shutdown timed out")

    resources.push(close_dependency)
    resources.push(close_leadership)
    with pytest.raises(RuntimeError, match="leader shutdown timed out"):
        resources.close()
    assert events == ["close:leadership"]
    with pytest.raises(RuntimeError, match="cleanup already started"):
        resources.push(lambda: None)

    release_leadership = True
    resources.close()
    resources.close()
    assert events == [
        "close:leadership",
        "close:leadership",
        "close:dependency",
    ]
