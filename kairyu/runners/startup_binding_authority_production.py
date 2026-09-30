"""Production assembly for the placement-binding authority control plane."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import threading
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from kairyu.runners.leadership_runtime import (
    RunnerLeaderElectionRuntime,
    RunnerLeaderElectionRuntimeConfig,
)
from kairyu.runners.postgres_leadership import PostgresRunnerLeaderLeaseStore
from kairyu.runners.postgres_scaling_log import PostgresScalingDecisionLog
from kairyu.runners.postgres_startup_admission import (
    PostgresRunnerCachePlacementAdmissionStore,
)
from kairyu.runners.startup_binding_authority_runtime import (
    RunnerCachePlacementBindingAuthorityRuntime,
    RunnerCachePlacementBindingAuthorityRuntimeConfig,
    build_runner_cache_placement_binding_authority_runtime,
)
from kairyu.runners.startup_binding_live_authority import (
    ScalingControllerPlacementBindingAuthority,
)
from kairyu.runners.startup_binding_live_cache import (
    AggregatingRunnerCachePlacementBindingCacheReader,
    AuthenticatedNodeModelCacheLiveEvidenceClient,
    NodeModelCacheAgentEndpoint,
)
from kairyu.runners.startup_binding_live_kubernetes import (
    KubernetesKueueRunnerCachePlacementBindingReader,
    KubernetesPlacementBindingLiveTarget,
)
from kairyu.runners.startup_binding_live_postgres import (
    PostgresRunnerCachePlacementBindingReader,
)
from kairyu.runners.startup_binding_live_source import (
    ComposedRunnerCachePlacementBindingLiveStateSource,
)

_MAX_CONFIG_BYTES = 1024 * 1024


def _identity(value: str, *, name: str) -> str:
    if not value or value != value.strip() or "\x00" in value or len(value) > 255:
        raise ValueError(f"{name} must be a bounded trimmed string without NUL")
    return value


class RunnerAuthorityPostgresConfig(BaseModel):
    """Shared PostgreSQL configuration for the three authority stores."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dsn_file: Path
    store_id: str = Field(min_length=1, max_length=255)
    connect_timeout_s: float = Field(default=5.0, gt=0, le=300, strict=True)
    initialize_schema: bool = Field(default=False, strict=True)
    admission_max_targets: int = Field(default=10_000, ge=1, le=100_000, strict=True)
    admission_replay_safety_window_s: float = Field(
        default=300.0,
        gt=0,
        le=86_400,
        strict=True,
    )
    decision_max_records: int = Field(
        default=1_000_000,
        ge=1,
        le=2**63 - 1,
        strict=True,
    )

    @field_validator("dsn_file")
    @classmethod
    def validate_path(cls, value: Path) -> Path:
        if not value.is_absolute() or "\x00" in str(value):
            raise ValueError("dsn_file must be an absolute path without NUL")
        return value

    @field_validator("store_id")
    @classmethod
    def validate_store_id(cls, value: str) -> str:
        return _identity(value, name="store_id")


class RunnerAuthorityKubernetesConfig(BaseModel):
    """Trusted Kubernetes routing and service-account material."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    targets: tuple[KubernetesPlacementBindingLiveTarget, ...] = Field(
        min_length=1,
        max_length=1024,
    )
    api_server: str | None = None
    token_file: Path
    ca_file: Path
    response_limit_bytes: int = Field(
        default=2 * 1024 * 1024,
        ge=1,
        le=16 * 1024 * 1024,
        strict=True,
    )

    @field_validator("token_file", "ca_file")
    @classmethod
    def validate_path(cls, value: Path, info) -> Path:
        if not value.is_absolute() or "\x00" in str(value):
            raise ValueError(f"{info.field_name} must be an absolute path without NUL")
        return value

    @field_validator("targets")
    @classmethod
    def validate_targets(
        cls,
        value: tuple[KubernetesPlacementBindingLiveTarget, ...],
    ) -> tuple[KubernetesPlacementBindingLiveTarget, ...]:
        model_classes = tuple(target.model_class for target in value)
        if model_classes != tuple(sorted(model_classes)) or len(set(model_classes)) != len(
            model_classes
        ):
            raise ValueError("targets must use unique canonical model classes")
        return value


class RunnerAuthorityNodeEvidenceConfig(BaseModel):
    """Trusted node-agent origins, TLS root, and bounded fan-out policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoints: tuple[NodeModelCacheAgentEndpoint, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    bearer_token_file: Path
    ca_file: Path
    request_limit_bytes: int = Field(
        default=1024 * 1024,
        ge=1,
        le=16 * 1024 * 1024,
        strict=True,
    )
    response_limit_bytes: int = Field(
        default=1024 * 1024,
        ge=1,
        le=16 * 1024 * 1024,
        strict=True,
    )
    max_parallel_requests: int = Field(default=16, ge=1, le=256, strict=True)

    @field_validator("bearer_token_file", "ca_file")
    @classmethod
    def validate_path(cls, value: Path, info) -> Path:
        if not value.is_absolute() or "\x00" in str(value):
            raise ValueError(f"{info.field_name} must be an absolute path without NUL")
        return value

    @field_validator("endpoints")
    @classmethod
    def validate_endpoints(
        cls,
        value: tuple[NodeModelCacheAgentEndpoint, ...],
    ) -> tuple[NodeModelCacheAgentEndpoint, ...]:
        node_ids = tuple(endpoint.node_id for endpoint in value)
        if node_ids != tuple(sorted(node_ids)) or len(set(node_ids)) != len(node_ids):
            raise ValueError("endpoints must use unique canonical node IDs")
        return value


class RunnerCachePlacementBindingProductionRuntimeConfig(BaseModel):
    """Complete non-secret production authority configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["kairyu-runner-cache-placement-binding-production-runtime-v1"] = (
        "kairyu-runner-cache-placement-binding-production-runtime-v1"
    )
    authority: RunnerCachePlacementBindingAuthorityRuntimeConfig
    leadership: RunnerLeaderElectionRuntimeConfig
    postgres: RunnerAuthorityPostgresConfig
    kubernetes: RunnerAuthorityKubernetesConfig
    node_evidence: RunnerAuthorityNodeEvidenceConfig


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON object keys must be unique")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON constant {value!r} is not permitted")


def _read_file(
    path: Path,
    *,
    max_bytes: int,
    kind: str,
    secret: bool,
) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{kind} must be a regular file")
        if secret and stat.S_IMODE(metadata.st_mode) & 0o027:
            raise ValueError(f"{kind} permissions are too broad")
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
        raise ValueError(f"cannot read {kind}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not value or len(value) > max_bytes:
        raise ValueError(f"{kind} file size is invalid")
    return value


def _read_text_secret(path: Path, *, kind: str, max_bytes: int = 64 * 1024) -> str:
    raw = _read_file(path, max_bytes=max_bytes, kind=kind, secret=True)
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{kind} must be UTF-8") from exc
    if value.endswith("\n"):
        value = value[:-1]
    if not value or value != value.strip() or "\n" in value or "\r" in value or "\x00" in value:
        raise ValueError(f"{kind} must contain one trimmed non-empty line")
    return value


def load_runner_cache_placement_binding_production_runtime_config(
    path: Path,
) -> RunnerCachePlacementBindingProductionRuntimeConfig:
    """Load one duplicate-key-free, size-bounded JSON configuration."""

    raw = _read_file(
        path,
        max_bytes=_MAX_CONFIG_BYTES,
        kind="production authority runtime config",
        secret=False,
    )
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        return RunnerCachePlacementBindingProductionRuntimeConfig.model_validate(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError("production authority runtime config file is invalid") from exc


class _ResourceStack:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[], None]] = []
        self._closing_started = False
        self._closed = False
        self._lock = threading.Lock()

    def push(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if self._closing_started:
                raise RuntimeError("production authority resource cleanup already started")
            self._callbacks.append(callback)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closing_started = True
            # A failed close may be a bounded, retryable shutdown (notably the
            # leader runtime waiting for an in-flight operation). Keep that
            # callback and every lower dependency open for the next attempt.
            while self._callbacks:
                self._callbacks[-1]()
                self._callbacks.pop()
            self._closed = True

    def abort(self) -> tuple[BaseException, ...]:
        """Drain every partial-construction resource without retry ownership.

        Production runtime shutdown uses :meth:`close` so a bounded leadership
        timeout keeps its dependencies alive for a caller retry. A builder
        failure cannot return that stack to a caller, and leadership is started
        only after the complete graph exists, so construction abort instead
        attempts every remaining callback exactly once.
        """

        failures: list[BaseException] = []
        with self._lock:
            if self._closed:
                return ()
            self._closing_started = True
            while self._callbacks:
                callback = self._callbacks.pop()
                try:
                    callback()
                except BaseException as exc:
                    failures.append(exc)
            self._closed = True
        return tuple(failures)


class RunnerCachePlacementBindingProductionRuntime:
    """Own the fully assembled authority service and all backend clients."""

    def __init__(
        self,
        *,
        config: RunnerCachePlacementBindingProductionRuntimeConfig,
        authority_runtime: RunnerCachePlacementBindingAuthorityRuntime,
        leadership: RunnerLeaderElectionRuntime,
    ) -> None:
        self.config = config
        self.authority_runtime = authority_runtime
        self.leadership = leadership

    @property
    def app(self) -> FastAPI:
        return self.authority_runtime.app

    def close(self) -> None:
        self.authority_runtime.close()

    def __enter__(self) -> RunnerCachePlacementBindingProductionRuntime:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def _close_async_client(client: httpx.AsyncClient) -> None:
    asyncio.run(client.aclose())


def build_runner_cache_placement_binding_production_runtime(
    config: RunnerCachePlacementBindingProductionRuntimeConfig,
) -> RunnerCachePlacementBindingProductionRuntime:
    """Build and start one fail-closed production authority process."""

    if not isinstance(config, RunnerCachePlacementBindingProductionRuntimeConfig):
        raise TypeError("config must be a RunnerCachePlacementBindingProductionRuntimeConfig")
    resources = _ResourceStack()
    try:
        dsn = _read_text_secret(config.postgres.dsn_file, kind="PostgreSQL DSN")
        _read_file(
            config.kubernetes.ca_file,
            max_bytes=1024 * 1024,
            kind="Kubernetes CA bundle",
            secret=False,
        )
        _read_file(
            config.kubernetes.token_file,
            max_bytes=64 * 1024,
            kind="Kubernetes service-account token",
            secret=True,
        )
        _read_file(
            config.node_evidence.ca_file,
            max_bytes=1024 * 1024,
            kind="node-agent CA bundle",
            secret=False,
        )
        node_token = _read_text_secret(
            config.node_evidence.bearer_token_file,
            kind="node-agent bearer token",
            max_bytes=4097,
        )

        postgres = config.postgres
        leader_store = PostgresRunnerLeaderLeaseStore(
            dsn,
            store_id=postgres.store_id,
            connect_timeout_s=postgres.connect_timeout_s,
            initialize_schema=postgres.initialize_schema,
        )
        resources.push(leader_store.close)
        admission_store = PostgresRunnerCachePlacementAdmissionStore(
            dsn,
            store_id=postgres.store_id,
            max_targets=postgres.admission_max_targets,
            replay_safety_window=timedelta(seconds=postgres.admission_replay_safety_window_s),
            connect_timeout_s=postgres.connect_timeout_s,
            initialize_schema=postgres.initialize_schema,
        )
        resources.push(admission_store.close)
        decision_log = PostgresScalingDecisionLog(
            dsn,
            store_id=postgres.store_id,
            max_records=postgres.decision_max_records,
            connect_timeout_s=postgres.connect_timeout_s,
            initialize_schema=postgres.initialize_schema,
        )
        resources.push(decision_log.close)

        kubernetes = config.kubernetes
        kube_reader = KubernetesKueueRunnerCachePlacementBindingReader(
            targets=kubernetes.targets,
            api_server=kubernetes.api_server,
            token_path=kubernetes.token_file,
            ca_path=kubernetes.ca_file,
            response_limit_bytes=kubernetes.response_limit_bytes,
        )
        resources.push(kube_reader.close)

        node = config.node_evidence
        node_http = httpx.AsyncClient(
            verify=str(node.ca_file),
            follow_redirects=False,
            trust_env=False,
        )
        try:
            evidence = AuthenticatedNodeModelCacheLiveEvidenceClient(
                node_http,
                endpoints=node.endpoints,
                bearer_token=node_token,
                request_limit_bytes=node.request_limit_bytes,
                response_limit_bytes=node.response_limit_bytes,
                max_parallel_requests=node.max_parallel_requests,
            )
        except BaseException:
            _close_async_client(node_http)
            raise
        resources.push(evidence.close)

        durable_reader = PostgresRunnerCachePlacementBindingReader(
            admission_store,
            decision_log,
        )
        cache_reader = AggregatingRunnerCachePlacementBindingCacheReader(
            evidence=evidence,
            inventory=kube_reader,
        )
        source = ComposedRunnerCachePlacementBindingLiveStateSource(
            current=durable_reader,
            decisions=durable_reader,
            targets=kube_reader,
            quotas=kube_reader,
            cache=cache_reader,
        )
        leadership = RunnerLeaderElectionRuntime(
            store=leader_store,
            config=config.leadership,
        )
        resources.push(leadership.close)
        authority = ScalingControllerPlacementBindingAuthority(
            leadership.controller,
            source,
        )

        def readiness_check(
            *,
            deadline_monotonic: float,
            backend_timeout_s: float,
        ) -> None:
            leadership.check_ready()
            authority.readiness(
                deadline_monotonic=deadline_monotonic,
                backend_timeout_s=backend_timeout_s,
            )

        authority_runtime = build_runner_cache_placement_binding_authority_runtime(
            config.authority,
            reauthorize=authority.reauthorize,
            readiness_check=readiness_check,
            close_resources=resources.close,
        )
        leadership.start()
        return RunnerCachePlacementBindingProductionRuntime(
            config=config,
            authority_runtime=authority_runtime,
            leadership=leadership,
        )
    except BaseException as exc:
        for cleanup_exc in resources.abort():
            exc.add_note(f"production authority cleanup also failed: {type(cleanup_exc).__name__}")
        raise


__all__ = [
    "RunnerAuthorityKubernetesConfig",
    "RunnerAuthorityNodeEvidenceConfig",
    "RunnerAuthorityPostgresConfig",
    "RunnerCachePlacementBindingProductionRuntime",
    "RunnerCachePlacementBindingProductionRuntimeConfig",
    "build_runner_cache_placement_binding_production_runtime",
    "load_runner_cache_placement_binding_production_runtime_config",
]
