"""Fail-closed process assembly for the node model cache agent."""

from __future__ import annotations

import json
import math
import os
import re
import stat
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from kairyu.artifacts import (
    HttpRangeModelArtifactBlobSource,
    NodeModelCacheAgent,
    NodeModelCacheIndex,
    load_model_artifact_trust_store,
)
from kairyu.runners.cache_agent_api import create_node_model_cache_agent_app
from kairyu.runners.cache_agent_live_evidence import (
    LocalNodeModelCacheLiveEvidenceSource,
)
from kairyu.runners.postgres_prestage import PostgresNodeModelPrestageStore
from kairyu.runners.prestage import NodeModelPrestageExecutor, _text

_MAX_CONFIG_BYTES = 64 * 1024
_DNS_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


class NodeModelCacheAgentAPIKeys(BaseModel):
    """Versioned API-key file mounted separately from non-secret config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["kairyu-node-model-cache-agent-api-keys-v1"] = (
        "kairyu-node-model-cache-agent-api-keys-v1"
    )
    api_keys: tuple[str, ...] = Field(min_length=1, max_length=16)

    @field_validator("api_keys")
    @classmethod
    def validate_api_keys(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            if (
                not isinstance(value, str)
                or not value.isascii()
                or not 32 <= len(value) <= 4096
                or not value.strip()
            ):
                raise ValueError("API keys must be 32-4096 non-empty ASCII characters")
        if len(set(values)) != len(values):
            raise ValueError("API keys must be unique")
        return values


class NodeModelCacheAgentRuntimeConfig(BaseModel):
    """Non-secret, versioned process configuration for one node agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["kairyu-node-model-cache-agent-runtime-v1"] = (
        "kairyu-node-model-cache-agent-runtime-v1"
    )
    node_id_file: Path
    cache_root: Path
    cache_index_path: Path
    artifact_base_url: str
    artifact_bearer_token_file: Path | None = None
    artifact_ca_bundle: Path | None = None
    allow_insecure_artifact_http: bool = Field(default=False, strict=True)
    trust_store_path: Path
    postgres_dsn_file: Path
    api_keys_file: Path
    postgres_store_prefix: str = "kairyu-model-cache-agent"
    initialize_postgres_schema: bool = Field(default=False, strict=True)
    max_placements: int = Field(default=100_000, ge=1, le=100_000, strict=True)
    postgres_connect_timeout_s: float = Field(default=10.0, gt=0, le=300, strict=True)
    artifact_timeout_s: float = Field(default=300.0, gt=0, le=3600, strict=True)
    fill_lock_timeout_s: float = Field(default=900.0, ge=0, le=3600, strict=True)
    request_body_limit_bytes: int = Field(
        default=64 * 1024 * 1024, ge=1, le=128 * 1024 * 1024, strict=True
    )
    active_request_limit: int = Field(default=2, ge=1, le=128, strict=True)
    total_request_limit: int = Field(default=8, ge=1, le=1024, strict=True)
    queue_wait_timeout_s: float = Field(default=1.0, gt=0, le=300, strict=True)
    live_evidence_hint_ttl_seconds: int = Field(default=30, ge=1, le=300, strict=True)
    listen_host: str = "0.0.0.0"
    listen_port: int = Field(default=8081, ge=1, le=65535, strict=True)

    @field_validator(
        "node_id_file",
        "cache_root",
        "cache_index_path",
        "artifact_bearer_token_file",
        "artifact_ca_bundle",
        "trust_store_path",
        "postgres_dsn_file",
        "api_keys_file",
    )
    @classmethod
    def validate_absolute_path(cls, value: Path | None, info) -> Path | None:
        if value is not None and (not value.is_absolute() or "\x00" in str(value)):
            raise ValueError(f"{info.field_name} must be an absolute path without NUL")
        return value

    @field_validator("postgres_store_prefix", "listen_host")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _text(value, name=info.field_name, max_length=200)

    @field_validator(
        "postgres_connect_timeout_s",
        "artifact_timeout_s",
        "fill_lock_timeout_s",
        "queue_wait_timeout_s",
    )
    @classmethod
    def validate_finite_number(cls, value: float, info) -> float:
        if not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be finite")
        return value

    @model_validator(mode="after")
    def validate_layout_and_transport(self) -> NodeModelCacheAgentRuntimeConfig:
        expected_state = self.cache_root / "state"
        if self.cache_index_path.parent != expected_state:
            raise ValueError("cache_index_path must be directly below cache_root/state")
        if any(
            character.isspace() or ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F
            for character in self.artifact_base_url
        ):
            raise ValueError("artifact_base_url must not contain whitespace or controls")
        parsed = urlsplit(self.artifact_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("artifact_base_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("artifact_base_url must not contain credentials, query, or fragment")
        if parsed.scheme == "http" and not self.allow_insecure_artifact_http:
            raise ValueError("HTTP artifact sources require explicit insecure opt-in")
        try:
            host = parsed.hostname
            port = parsed.port
        except ValueError as exc:
            raise ValueError("artifact_base_url contains an invalid port") from exc
        if host is None or not host.isascii() or any(character.isspace() for character in host):
            raise ValueError("artifact_base_url contains an invalid host")
        try:
            from ipaddress import ip_address

            ip_address(host)
        except ValueError:
            is_ip_address = False
        else:
            is_ip_address = True
        if not is_ip_address:
            labels = host.removesuffix(".").split(".")
            if (
                len(host) > 253
                or not labels
                or any(not _DNS_LABEL.fullmatch(label) for label in labels)
            ):
                raise ValueError("artifact_base_url contains an invalid host")
        if port is not None and not 1 <= port <= 65535:
            raise ValueError("artifact_base_url contains an invalid port")
        if self.total_request_limit < self.active_request_limit:
            raise ValueError("total_request_limit must be at least active_request_limit")
        return self


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("JSON object keys must be unique")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _read_bounded(path: Path, *, max_bytes: int, kind: str) -> bytes:
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{kind} must be a regular file")
        chunks: list[bytes] = []
        total = 0
        while total <= max_bytes:
            chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        value = b"".join(chunks)
    except OSError as exc:
        raise ValueError(f"cannot read {kind} file") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not value or len(value) > max_bytes:
        raise ValueError(f"{kind} file size is invalid")
    return value


def _load_json_model(path: Path, model: type[BaseModel], *, kind: str) -> BaseModel:
    raw = _read_bounded(path, max_bytes=_MAX_CONFIG_BYTES, kind=kind)
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        return model.model_validate(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(f"{kind} file is invalid") from exc


def load_node_model_cache_agent_runtime_config(
    path: Path,
) -> NodeModelCacheAgentRuntimeConfig:
    value = _load_json_model(path, NodeModelCacheAgentRuntimeConfig, kind="runtime config")
    assert isinstance(value, NodeModelCacheAgentRuntimeConfig)
    return value


def _load_text_secret(path: Path, *, kind: str, max_bytes: int = 8192) -> str:
    raw = _read_bounded(path, max_bytes=max_bytes, kind=kind)
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{kind} must be UTF-8") from exc
    if value.endswith("\n"):
        value = value[:-1]
    if not value or value != value.strip() or "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{kind} must contain one non-empty line without NUL")
    return value


class NodeModelCacheAgentRuntime:
    """Own the assembled app and its closeable transport/store resources."""

    def __init__(
        self,
        *,
        app: FastAPI,
        config: NodeModelCacheAgentRuntimeConfig,
        close_resources: Callable[[], None],
    ) -> None:
        self.app = app
        self.config = config
        self._close_resources = close_resources
        self._closed = False
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._close_resources()
            self._closed = True


def build_node_model_cache_agent_runtime(
    config: NodeModelCacheAgentRuntimeConfig,
) -> NodeModelCacheAgentRuntime:
    """Assemble all production adapters and fail before serving on bad config."""

    node_id = _text(
        _load_text_secret(config.node_id_file, kind="node identity", max_bytes=1024),
        name="node_id",
        max_length=253,
    )
    dsn = _load_text_secret(config.postgres_dsn_file, kind="PostgreSQL DSN")
    key_value = _load_json_model(config.api_keys_file, NodeModelCacheAgentAPIKeys, kind="API key")
    assert isinstance(key_value, NodeModelCacheAgentAPIKeys)
    trust_store = load_model_artifact_trust_store(config.trust_store_path)
    headers: dict[str, str] = {}
    if config.artifact_bearer_token_file is not None:
        token = _load_text_secret(
            config.artifact_bearer_token_file,
            kind="artifact bearer token",
            max_bytes=4096,
        )
        if not token.isascii():
            raise ValueError("artifact bearer token must be ASCII")
        headers["Authorization"] = f"Bearer {token}"
    verify: bool | str = (
        str(config.artifact_ca_bundle) if config.artifact_ca_bundle is not None else True
    )
    http_client = httpx.Client(
        headers=headers,
        timeout=config.artifact_timeout_s,
        verify=verify,
    )
    source = HttpRangeModelArtifactBlobSource(
        config.artifact_base_url,
        client=http_client,
        timeout_seconds=config.artifact_timeout_s,
    )
    store: PostgresNodeModelPrestageStore | None = None
    try:
        NodeModelCacheAgent(
            config.cache_root,
            source,
            lock_timeout_seconds=config.fill_lock_timeout_s,
        ).check_ready()
        index = NodeModelCacheIndex(
            config.cache_index_path,
            node_id=node_id,
            cache_root=config.cache_root,
        )
        store = PostgresNodeModelPrestageStore(
            dsn,
            store_id=f"{config.postgres_store_prefix}/{node_id}",
            node_id=node_id,
            max_placements=config.max_placements,
            connect_timeout_s=config.postgres_connect_timeout_s,
            initialize_schema=config.initialize_postgres_schema,
        )
        agent = NodeModelCacheAgent(
            config.cache_root,
            source,
            index=index,
            lock_timeout_seconds=config.fill_lock_timeout_s,
        )
        agent.check_ready()
        executor = NodeModelPrestageExecutor(
            node_id=node_id,
            agent=agent,
            index=index,
            store=store,
        )
        live_evidence_source = LocalNodeModelCacheLiveEvidenceSource(
            node_id=node_id,
            store=store,
            index=index,
            hint_ttl_seconds=config.live_evidence_hint_ttl_seconds,
        )

        def readiness_check() -> None:
            agent.check_ready()
            store.check_ready()
            index.snapshot()

        app = create_node_model_cache_agent_app(
            node_id=node_id,
            executor=executor,
            store=store,
            trust_store=trust_store,
            api_keys=key_value.api_keys,
            readiness_check=readiness_check,
            live_evidence_source=live_evidence_source,
            request_body_limit_bytes=config.request_body_limit_bytes,
            active_request_limit=config.active_request_limit,
            total_request_limit=config.total_request_limit,
            queue_wait_timeout_s=config.queue_wait_timeout_s,
        )
    except Exception:
        try:
            if store is not None:
                store.close()
        finally:
            http_client.close()
        raise

    def close_resources() -> None:
        try:
            store.close()
        finally:
            http_client.close()

    runtime = NodeModelCacheAgentRuntime(
        app=app,
        config=config,
        close_resources=close_resources,
    )
    app.router.add_event_handler("shutdown", runtime.close)
    return runtime
