"""Executable node model cache-agent runtime assembly coverage."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

import kairyu.runners.cache_agent_runtime as runtime_module
from kairyu.artifacts import InvalidNodeModelCacheEntryError
from kairyu.runners import (
    InMemoryNodeModelPrestageStore,
    NodeModelCacheAgentRuntime,
    NodeModelCacheAgentRuntimeConfig,
    build_node_model_cache_agent_runtime,
    load_node_model_cache_agent_runtime_config,
)

_TOKEN = "runtime-test-key-00000000000000000000000"


class FakePostgresStore(InMemoryNodeModelPrestageStore):
    def __init__(self, _dsn: str, *, node_id: str, **_kwargs) -> None:
        super().__init__(node_id=node_id)
        self.closed = False
        self.ready_checks = 0

    def check_ready(self) -> None:
        self.ready_checks += 1

    def close(self) -> None:
        self.closed = True


def _files(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "node_id_file": tmp_path / "node-id",
        "trust_store_path": tmp_path / "trust-store.json",
        "postgres_dsn_file": tmp_path / "postgres-dsn",
        "api_keys_file": tmp_path / "api-keys.json",
    }
    paths["node_id_file"].write_text("gpu-node-00\n", encoding="utf-8")
    paths["postgres_dsn_file"].write_text(
        "postgresql://cache-agent@postgres/cache\n", encoding="utf-8"
    )
    paths["api_keys_file"].write_text(
        json.dumps(
            {
                "schema_version": "kairyu-node-model-cache-agent-api-keys-v1",
                "api_keys": [_TOKEN],
            }
        ),
        encoding="utf-8",
    )
    paths["trust_store_path"].write_text(
        json.dumps(
            {
                "schema_version": "kairyu-model-trust-store-v1",
                "signers": [
                    {
                        "key_id": "release-key",
                        "public_key_base64": base64.b64encode(bytes(32)).decode(),
                        "approved_environments": ["production"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return paths


def _config(tmp_path: Path, **updates) -> NodeModelCacheAgentRuntimeConfig:
    paths = _files(tmp_path)
    cache_root = tmp_path / "cache"
    value = {
        **paths,
        "cache_root": cache_root,
        "cache_index_path": cache_root / "state/cache-index.sqlite3",
        "artifact_base_url": "https://artifacts.example.invalid/models",
    }
    value.update(updates)
    return NodeModelCacheAgentRuntimeConfig.model_validate(value)


def test_runtime_config_loader_rejects_duplicate_keys_and_insecure_defaults(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    path = tmp_path / "runtime.json"
    path.write_text(config.model_dump_json(), encoding="utf-8")

    assert load_node_model_cache_agent_runtime_config(path) == config

    path.write_text(
        '{"schema_version":"kairyu-node-model-cache-agent-runtime-v1",'
        '"schema_version":"kairyu-node-model-cache-agent-runtime-v1"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="runtime config file is invalid"):
        load_node_model_cache_agent_runtime_config(path)
    with pytest.raises(ValueError, match="explicit insecure opt-in"):
        _config(tmp_path, artifact_base_url="http://artifacts.internal/models")


def test_runtime_config_binds_index_to_cache_state_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cache_root/state"):
        _config(tmp_path, cache_index_path=tmp_path / "cache/index.sqlite3")


@pytest.mark.parametrize(
    "url",
    (
        "https://example.invalid:99999/models",
        "https://:443/models",
        "https://exa mple/models",
        "https://exa\nmple.com/models",
        "https://example.com/pa\tth",
    ),
)
def test_runtime_config_rejects_invalid_artifact_url(tmp_path: Path, url: str) -> None:
    with pytest.raises(ValueError, match="invalid (host|port)|whitespace or controls"):
        _config(tmp_path, artifact_base_url=url)


def test_bounded_reader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "secret-fifo"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="regular file"):
        runtime_module._read_bounded(fifo, max_bytes=1024, kind="test secret")


def test_bounded_reader_accepts_projected_secret_symlink(tmp_path: Path) -> None:
    version = tmp_path / "..2026_09_29"
    version.mkdir()
    target = version / "token"
    target.write_bytes(b"secret")
    projected = tmp_path / "token"
    projected.symlink_to(Path("..2026_09_29/token"))

    assert runtime_module._read_bounded(
        projected, max_bytes=1024, kind="test secret"
    ) == b"secret"


@pytest.mark.asyncio
async def test_runtime_assembles_authenticated_app_and_closes_resources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created: list[FakePostgresStore] = []

    def create_store(*args, **kwargs):
        store = FakePostgresStore(*args, **kwargs)
        created.append(store)
        return store

    monkeypatch.setattr(runtime_module, "PostgresNodeModelPrestageStore", create_store)
    config = _config(tmp_path)
    runtime = build_node_model_cache_agent_runtime(config)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=runtime.app),
        base_url="http://cache-agent.test",
    ) as client:
        health = await client.get("/health")
        unauthorized = await client.get("/v1/prestage/records")
        ready = await client.get("/readyz")
        authorized = await client.get(
            "/v1/prestage/records", headers={"Authorization": f"Bearer {_TOKEN}"}
        )
        config.cache_root.chmod(0o777)
        await asyncio.sleep(0.51)
        unsafe = await client.get("/readyz")
        config.cache_root.chmod(0o700)

    assert health.json() == {"status": "ok", "node_id": "gpu-node-00"}
    assert unauthorized.status_code == 401
    assert ready.status_code == 200
    assert unsafe.status_code == 503
    assert authorized.status_code == 200
    assert runtime.config.cache_index_path.exists()
    assert (runtime.config.cache_root / "artifacts").is_dir()
    assert (runtime.config.cache_root / ".staging").is_dir()
    assert (runtime.config.cache_root / ".locks").is_dir()
    assert created[0].ready_checks == 1
    runtime.close()
    runtime.close()
    assert created[0].closed is True


def test_runtime_rejects_multiline_secret_before_opening_postgres(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    config.postgres_dsn_file.write_text("first\n\n", encoding="utf-8")
    opened = False

    def create_store(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("invalid secrets must fail before store construction")

    monkeypatch.setattr(runtime_module, "PostgresNodeModelPrestageStore", create_store)
    with pytest.raises(ValueError, match="one non-empty line"):
        build_node_model_cache_agent_runtime(config)
    assert opened is False


def test_runtime_rejects_unsafe_cache_root_before_serving(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    config.cache_root.mkdir(mode=0o777)
    config.cache_root.chmod(0o777)
    created: list[FakePostgresStore] = []

    def create_store(*args, **kwargs):
        store = FakePostgresStore(*args, **kwargs)
        created.append(store)
        return store

    monkeypatch.setattr(runtime_module, "PostgresNodeModelPrestageStore", create_store)
    with pytest.raises(InvalidNodeModelCacheEntryError, match="cache root"):
        build_node_model_cache_agent_runtime(config)

    assert created == []
    assert not config.cache_index_path.exists()
    config.cache_root.chmod(0o700)


def test_runtime_close_can_retry_after_cleanup_failure(tmp_path: Path) -> None:
    calls = 0

    def close_resources() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient close failure")

    runtime = NodeModelCacheAgentRuntime(
        app=FastAPI(),
        config=_config(tmp_path),
        close_resources=close_resources,
    )

    with pytest.raises(RuntimeError, match="transient"):
        runtime.close()
    runtime.close()
    runtime.close()
    assert calls == 2
