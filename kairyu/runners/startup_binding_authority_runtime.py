"""Process-boundary assembly for the live placement-binding authority API."""

from __future__ import annotations

import json
import math
import os
import stat
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from kairyu.runners.startup_binding_authority import (
    RunnerCachePlacementBindingAuthorityReadiness,
    RunnerCachePlacementBindingLiveAuthority,
    validate_runner_cache_placement_bearer_token,
)
from kairyu.runners.startup_binding_authority_api import (
    create_runner_cache_placement_binding_authority_app,
)

_MAX_CONFIG_BYTES = 64 * 1024


class RunnerCachePlacementBindingAuthorityRuntimeConfig(BaseModel):
    """Non-secret configuration for an embedded scaling-authority endpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[
        "kairyu-runner-cache-placement-binding-authority-runtime-v1"
    ] = "kairyu-runner-cache-placement-binding-authority-runtime-v1"
    bearer_token_file: Path
    request_body_limit_bytes: int = Field(
        default=1024 * 1024,
        ge=1,
        le=16 * 1024 * 1024,
        strict=True,
    )
    response_body_limit_bytes: int = Field(
        default=1024 * 1024 + 1,
        ge=2,
        le=16 * 1024 * 1024,
        strict=True,
    )
    active_request_limit: int = Field(default=16, ge=1, le=256, strict=True)
    total_request_limit: int = Field(default=64, ge=1, le=2048, strict=True)
    queue_wait_timeout_s: float = Field(default=0.25, gt=0, le=30, strict=True)
    request_timeout_s: float = Field(default=1.25, gt=0, le=30, strict=True)
    backend_timeout_s: float = Field(default=1.0, gt=0, le=30, strict=True)
    authorization_client_timeout_s: float = Field(default=2.0, gt=0, le=30, strict=True)
    transport_margin_s: float = Field(default=0.25, gt=0, le=30, strict=True)
    listen_host: str = "0.0.0.0"
    listen_port: int = Field(default=8444, ge=1, le=65535, strict=True)
    tls_cert_file: Path
    tls_key_file: Path

    @field_validator("bearer_token_file", "tls_cert_file", "tls_key_file")
    @classmethod
    def validate_absolute_path(cls, value: Path, info) -> Path:
        if not value.is_absolute() or "\x00" in str(value):
            raise ValueError(f"{info.field_name} must be an absolute path without NUL")
        return value

    @field_validator(
        "queue_wait_timeout_s",
        "request_timeout_s",
        "backend_timeout_s",
        "authorization_client_timeout_s",
        "transport_margin_s",
    )
    @classmethod
    def validate_finite_number(cls, value: float, info) -> float:
        if not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be finite")
        return value

    @field_validator("listen_host")
    @classmethod
    def validate_listen_host(cls, value: str) -> str:
        if not value.strip() or "\x00" in value or len(value) > 255:
            raise ValueError("listen_host must be non-empty and contain no NUL")
        return value

    @model_validator(mode="after")
    def validate_budgets(self) -> RunnerCachePlacementBindingAuthorityRuntimeConfig:
        if self.response_body_limit_bytes <= self.request_body_limit_bytes:
            raise ValueError(
                "response_body_limit_bytes must exceed request_body_limit_bytes"
            )
        if self.total_request_limit < self.active_request_limit:
            raise ValueError("total_request_limit must be at least active_request_limit")
        if self.backend_timeout_s >= self.request_timeout_s:
            raise ValueError("backend_timeout_s must be less than request_timeout_s")
        if (
            self.queue_wait_timeout_s
            + self.request_timeout_s
            + self.transport_margin_s
            >= self.authorization_client_timeout_s
        ):
            raise ValueError(
                "queue_wait_timeout_s + request_timeout_s + transport_margin_s "
                "must be less than authorization_client_timeout_s"
            )
        return self


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("JSON object keys must be unique")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON constant {value!r} is not permitted")


def _read_bounded_file(
    path: Path,
    *,
    max_bytes: int,
    kind: str,
    secret: bool = False,
) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"{kind} must be a regular file")
        if secret and stat.S_IMODE(file_stat.st_mode) & 0o027:
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
        raise ValueError(f"cannot read {kind} file") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not value or len(value) > max_bytes:
        raise ValueError(f"{kind} file size is invalid")
    return value


def _load_text_secret(path: Path, *, kind: str) -> str:
    raw = _read_bounded_file(path, max_bytes=4097, kind=kind, secret=True)
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{kind} must be UTF-8") from exc
    if value.endswith("\n"):
        value = value[:-1]
    if not value or value != value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{kind} must contain one non-empty line")
    return value


def load_runner_cache_placement_binding_authority_runtime_config(
    path: Path,
) -> RunnerCachePlacementBindingAuthorityRuntimeConfig:
    raw = _read_bounded_file(path, max_bytes=_MAX_CONFIG_BYTES, kind="authority runtime config")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        return RunnerCachePlacementBindingAuthorityRuntimeConfig.model_validate(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError("authority runtime config file is invalid") from exc


class RunnerCachePlacementBindingAuthorityRuntime:
    """Own the assembled authority app and caller-supplied close hook."""

    def __init__(
        self,
        *,
        app: FastAPI,
        config: RunnerCachePlacementBindingAuthorityRuntimeConfig,
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


def build_runner_cache_placement_binding_authority_runtime(
    config: RunnerCachePlacementBindingAuthorityRuntimeConfig,
    *,
    reauthorize: RunnerCachePlacementBindingLiveAuthority,
    readiness_check: RunnerCachePlacementBindingAuthorityReadiness,
    close_resources: Callable[[], None] = lambda: None,
) -> RunnerCachePlacementBindingAuthorityRuntime:
    """Load secrets and assemble the service, adopting resource ownership."""

    if not isinstance(config, RunnerCachePlacementBindingAuthorityRuntimeConfig):
        raise TypeError(
            "config must be a RunnerCachePlacementBindingAuthorityRuntimeConfig"
        )
    if not callable(close_resources):
        raise TypeError("close_resources must be callable")
    try:
        _read_bounded_file(
            config.tls_cert_file,
            max_bytes=1024 * 1024,
            kind="TLS certificate",
        )
        _read_bounded_file(
            config.tls_key_file,
            max_bytes=1024 * 1024,
            kind="TLS private key",
            secret=True,
        )
        token = validate_runner_cache_placement_bearer_token(
            _load_text_secret(
                config.bearer_token_file,
                kind="binding authority bearer token",
            )
        )
        app = create_runner_cache_placement_binding_authority_app(
            reauthorize=reauthorize,
            readiness_check=readiness_check,
            bearer_token=token,
            request_body_limit_bytes=config.request_body_limit_bytes,
            response_body_limit_bytes=config.response_body_limit_bytes,
            active_request_limit=config.active_request_limit,
            total_request_limit=config.total_request_limit,
            queue_wait_timeout_s=config.queue_wait_timeout_s,
            request_timeout_s=config.request_timeout_s,
            backend_timeout_s=config.backend_timeout_s,
        )
        runtime = RunnerCachePlacementBindingAuthorityRuntime(
            app=app,
            config=config,
            close_resources=close_resources,
        )
        app.router.add_event_handler("shutdown", runtime.close)
    except Exception as exc:
        try:
            close_resources()
        except Exception as cleanup_exc:
            exc.add_note(
                "placement-binding authority resource cleanup also failed: "
                f"{type(cleanup_exc).__name__}"
            )
        raise
    return runtime


__all__ = [
    "RunnerCachePlacementBindingAuthorityRuntime",
    "RunnerCachePlacementBindingAuthorityRuntimeConfig",
    "build_runner_cache_placement_binding_authority_runtime",
    "load_runner_cache_placement_binding_authority_runtime_config",
]
