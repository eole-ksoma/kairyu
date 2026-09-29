"""Executable runtime for the Runner cache-placement admission webhook."""

from __future__ import annotations

import json
import math
import os
import secrets
import stat
import threading
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from kairyu.runners.postgres_startup_admission import (
    PostgresRunnerCachePlacementAdmissionStore,
)
from kairyu.runners.startup_admission import RunnerCachePlacementAdmissionController
from kairyu.runners.startup_admission_api import (
    create_runner_cache_placement_admission_app,
)
from kairyu.runners.startup_binding import RunnerCacheStartupBinding

_MAX_CONFIG_BYTES = 64 * 1024


class RunnerCachePlacementBindingAuthorizationError(RuntimeError):
    """The live binding authority could not authorize an exact binding."""


class RunnerCachePlacementBindingAuthorizationRequest(BaseModel):
    """Nonce-bound request sent to the live scaling authority."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["kairyu-runner-cache-placement-binding-authorization-request-v1"] = (
        "kairyu-runner-cache-placement-binding-authorization-request-v1"
    )
    nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding: RunnerCacheStartupBinding


class RunnerCachePlacementBindingAuthorizationResponse(BaseModel):
    """Exact live binding returned by the trusted scaling authority."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["kairyu-runner-cache-placement-binding-authorization-response-v1"] = (
        "kairyu-runner-cache-placement-binding-authorization-response-v1"
    )
    nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding: RunnerCacheStartupBinding


class RunnerCachePlacementAdmissionRuntimeConfig(BaseModel):
    """Non-secret process configuration for the admission webhook."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["kairyu-runner-cache-placement-admission-runtime-v1"] = (
        "kairyu-runner-cache-placement-admission-runtime-v1"
    )
    postgres_dsn_file: Path
    postgres_store_id: str = "kairyu-runner-cache-placement-admission"
    initialize_postgres_schema: bool = Field(default=False, strict=True)
    max_targets: int = Field(default=10_000, ge=1, le=100_000, strict=True)
    replay_safety_window_s: int = Field(default=300, ge=1, le=3600, strict=True)
    postgres_connect_timeout_s: float = Field(default=1.0, gt=0, le=300, strict=True)

    authorization_url: str
    authorization_ready_url: str
    authorization_bearer_token_file: Path
    authorization_ca_bundle: Path | None = None
    authorization_timeout_s: float = Field(default=2.0, gt=0, le=30, strict=True)
    authorization_request_limit_bytes: int = Field(
        default=1024 * 1024,
        ge=1,
        le=16 * 1024 * 1024,
        strict=True,
    )
    authorization_response_limit_bytes: int = Field(
        default=1024 * 1024,
        ge=1,
        le=16 * 1024 * 1024,
        strict=True,
    )

    request_body_limit_bytes: int = Field(
        default=1024 * 1024,
        ge=1,
        le=16 * 1024 * 1024,
        strict=True,
    )
    active_request_limit: int = Field(default=16, ge=1, le=256, strict=True)
    total_request_limit: int = Field(default=64, ge=1, le=2048, strict=True)
    queue_wait_timeout_s: float = Field(default=0.5, gt=0, le=30, strict=True)
    admission_request_timeout_s: float = Field(default=4.0, gt=0, le=30, strict=True)

    listen_host: str = "0.0.0.0"
    listen_port: int = Field(default=8443, ge=1, le=65535, strict=True)
    tls_cert_file: Path
    tls_key_file: Path

    @field_validator(
        "postgres_dsn_file",
        "authorization_bearer_token_file",
        "authorization_ca_bundle",
        "tls_cert_file",
        "tls_key_file",
    )
    @classmethod
    def validate_absolute_path(cls, value: Path | None, info) -> Path | None:
        if value is not None and (not value.is_absolute() or "\x00" in str(value)):
            raise ValueError(f"{info.field_name} must be an absolute path without NUL")
        return value

    @field_validator("postgres_store_id", "listen_host")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value.strip() or "\x00" in value or len(value) > 255:
            raise ValueError(f"{info.field_name} must be non-empty and contain no NUL")
        return value

    @field_validator(
        "postgres_connect_timeout_s",
        "authorization_timeout_s",
        "queue_wait_timeout_s",
        "admission_request_timeout_s",
    )
    @classmethod
    def validate_finite_number(cls, value: float, info) -> float:
        if not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be finite")
        return value

    @model_validator(mode="after")
    def validate_transport(self) -> RunnerCachePlacementAdmissionRuntimeConfig:
        authorization = _validated_url(
            self.authorization_url,
            name="authorization_url",
        )
        readiness = _validated_url(
            self.authorization_ready_url,
            name="authorization_ready_url",
        )
        if (authorization.scheme, authorization.netloc) != (
            readiness.scheme,
            readiness.netloc,
        ):
            raise ValueError("authorization and readiness URLs must share one origin")
        if authorization.path == readiness.path:
            raise ValueError("authorization and readiness URLs must use distinct paths")
        if self.total_request_limit < self.active_request_limit:
            raise ValueError("total_request_limit must be at least active_request_limit")
        if self.authorization_timeout_s >= self.admission_request_timeout_s:
            raise ValueError(
                "authorization_timeout_s must be less than "
                "admission_request_timeout_s"
            )
        if math.ceil(self.postgres_connect_timeout_s) >= self.admission_request_timeout_s:
            raise ValueError(
                "effective PostgreSQL timeout must be less than "
                "admission_request_timeout_s"
            )
        if self.replay_safety_window_s <= self.admission_request_timeout_s:
            raise ValueError(
                "replay_safety_window_s must exceed admission_request_timeout_s"
            )
        return self


def _validated_url(value: str, *, name: str):
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or any(character.isspace() or ord(character) < 0x20 for character in value)
    ):
        raise ValueError(f"{name} is invalid")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{name} must be an absolute HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{name} must not contain credentials, query, or fragment")
    if parsed.hostname is None or not parsed.hostname.isascii():
        raise ValueError(f"{name} contains an invalid host")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{name} contains an invalid port") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f"{name} contains an invalid port")
    return parsed


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("JSON object keys must be unique")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON constant {value!r} is not permitted")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON numbers must be finite")
    return parsed


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


def _load_text_secret(path: Path, *, kind: str, max_bytes: int = 8192) -> str:
    raw = _read_bounded_file(path, max_bytes=max_bytes, kind=kind, secret=True)
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{kind} must be UTF-8") from exc
    if value.endswith("\n"):
        value = value[:-1]
    if not value or value != value.strip() or "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{kind} must contain one non-empty line without NUL")
    return value


def _load_json_model(path: Path, model: type[BaseModel], *, kind: str) -> BaseModel:
    raw = _read_bounded_file(path, max_bytes=_MAX_CONFIG_BYTES, kind=kind)
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
        return model.model_validate(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(f"{kind} file is invalid") from exc


def load_runner_cache_placement_admission_runtime_config(
    path: Path,
) -> RunnerCachePlacementAdmissionRuntimeConfig:
    value = _load_json_model(
        path,
        RunnerCachePlacementAdmissionRuntimeConfig,
        kind="admission runtime config",
    )
    assert isinstance(value, RunnerCachePlacementAdmissionRuntimeConfig)
    return value


def _read_response(response: httpx.Response, *, max_bytes: int) -> bytes:
    raw_length = response.headers.get("content-length")
    if raw_length is not None:
        try:
            content_length = int(raw_length)
        except ValueError as exc:
            raise RunnerCachePlacementBindingAuthorizationError(
                "binding authority returned an invalid response"
            ) from exc
        if content_length < 0 or content_length > max_bytes:
            raise RunnerCachePlacementBindingAuthorizationError(
                "binding authority response exceeds the configured limit"
            )
    chunks: list[bytes] = []
    received = 0
    for chunk in response.iter_bytes():
        received += len(chunk)
        if received > max_bytes:
            raise RunnerCachePlacementBindingAuthorizationError(
                "binding authority response exceeds the configured limit"
            )
        chunks.append(chunk)
    return b"".join(chunks)


class RunnerCachePlacementBindingAuthorizer:
    """Authenticated nonce-bound client for final live binding authorization."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        authorization_url: str,
        readiness_url: str,
        request_limit_bytes: int,
        response_limit_bytes: int,
    ) -> None:
        if not isinstance(client, httpx.Client):
            raise TypeError("client must be an httpx.Client")
        if type(request_limit_bytes) is not int or request_limit_bytes < 1:
            raise ValueError("request_limit_bytes must be positive")
        if type(response_limit_bytes) is not int or response_limit_bytes < 1:
            raise ValueError("response_limit_bytes must be positive")
        self._client = client
        self._authorization_url = authorization_url
        self._readiness_url = readiness_url
        self._request_limit_bytes = request_limit_bytes
        self._response_limit_bytes = response_limit_bytes

    def reauthorize(self, binding: RunnerCacheStartupBinding) -> RunnerCacheStartupBinding:
        if not isinstance(binding, RunnerCacheStartupBinding):
            raise TypeError("binding must be a RunnerCacheStartupBinding")
        nonce = secrets.token_hex(32)
        request = RunnerCachePlacementBindingAuthorizationRequest(
            nonce=nonce,
            binding=binding,
        )
        request_body = request.model_dump_json().encode("utf-8")
        if len(request_body) > self._request_limit_bytes:
            raise RunnerCachePlacementBindingAuthorizationError(
                "binding authorization request exceeds the configured limit"
            )
        try:
            with self._client.stream(
                "POST",
                self._authorization_url,
                content=request_body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Cache-Control": "no-store",
                },
            ) as response:
                if response.status_code != 200:
                    raise RunnerCachePlacementBindingAuthorizationError(
                        "binding authority rejected reauthorization"
                    )
                if response.headers.get("content-type", "").partition(";")[0].lower() != (
                    "application/json"
                ):
                    raise RunnerCachePlacementBindingAuthorizationError(
                        "binding authority returned an invalid response"
                    )
                raw = _read_response(response, max_bytes=self._response_limit_bytes)
            value = json.loads(
                raw,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
                parse_float=_finite_float,
            )
            result = RunnerCachePlacementBindingAuthorizationResponse.model_validate(value)
        except RunnerCachePlacementBindingAuthorizationError:
            raise
        except (
            httpx.HTTPError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
        ) as exc:
            raise RunnerCachePlacementBindingAuthorizationError(
                "binding authority is unavailable or returned an invalid response"
            ) from exc
        if result.nonce != nonce:
            raise RunnerCachePlacementBindingAuthorizationError(
                "binding authority response nonce does not match"
            )
        return result.binding

    def check_ready(self) -> None:
        try:
            with self._client.stream(
                "GET",
                self._readiness_url,
                headers={"Accept": "application/json", "Cache-Control": "no-store"},
            ) as response:
                if response.status_code != 204:
                    raise RunnerCachePlacementBindingAuthorizationError(
                        "binding authority is not ready"
                    )
                if _read_response(response, max_bytes=1):
                    raise RunnerCachePlacementBindingAuthorizationError(
                        "binding authority readiness response must be empty"
                    )
        except RunnerCachePlacementBindingAuthorizationError:
            raise
        except httpx.HTTPError as exc:
            raise RunnerCachePlacementBindingAuthorizationError(
                "binding authority is unavailable"
            ) from exc


class RunnerCachePlacementAdmissionRuntime:
    """Own the admission app and closeable PostgreSQL/HTTP resources."""

    def __init__(
        self,
        *,
        app: FastAPI,
        config: RunnerCachePlacementAdmissionRuntimeConfig,
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


def _validate_tls_files(config: RunnerCachePlacementAdmissionRuntimeConfig) -> None:
    _read_bounded_file(config.tls_cert_file, max_bytes=1024 * 1024, kind="TLS certificate")
    _read_bounded_file(
        config.tls_key_file,
        max_bytes=1024 * 1024,
        kind="TLS private key",
        secret=True,
    )


def build_runner_cache_placement_admission_runtime(
    config: RunnerCachePlacementAdmissionRuntimeConfig,
    *,
    http_client_factory: Callable[..., httpx.Client] = httpx.Client,
) -> RunnerCachePlacementAdmissionRuntime:
    """Assemble the production webhook adapters and verify dependencies."""

    if not isinstance(config, RunnerCachePlacementAdmissionRuntimeConfig):
        raise TypeError("config must be a RunnerCachePlacementAdmissionRuntimeConfig")
    if not callable(http_client_factory):
        raise TypeError("http_client_factory must be callable")
    _validate_tls_files(config)
    dsn = _load_text_secret(config.postgres_dsn_file, kind="PostgreSQL DSN")
    token = _load_text_secret(
        config.authorization_bearer_token_file,
        kind="binding authority bearer token",
        max_bytes=4096,
    )
    if not token.isascii() or not 32 <= len(token) <= 4096:
        raise ValueError("binding authority bearer token must be 32-4096 ASCII characters")
    verify: bool | str = (
        str(config.authorization_ca_bundle) if config.authorization_ca_bundle is not None else True
    )
    client = http_client_factory(
        headers={"Authorization": f"Bearer {token}"},
        timeout=config.authorization_timeout_s,
        verify=verify,
        follow_redirects=False,
        trust_env=False,
    )
    store: PostgresRunnerCachePlacementAdmissionStore | None = None
    try:
        store = PostgresRunnerCachePlacementAdmissionStore(
            dsn,
            store_id=config.postgres_store_id,
            max_targets=config.max_targets,
            replay_safety_window=timedelta(seconds=config.replay_safety_window_s),
            connect_timeout_s=config.postgres_connect_timeout_s,
            initialize_schema=config.initialize_postgres_schema,
        )
        authorizer = RunnerCachePlacementBindingAuthorizer(
            client,
            authorization_url=config.authorization_url,
            readiness_url=config.authorization_ready_url,
            request_limit_bytes=config.authorization_request_limit_bytes,
            response_limit_bytes=config.authorization_response_limit_bytes,
        )
        authorizer.check_ready()
        controller = RunnerCachePlacementAdmissionController(
            store,
            reauthorize=authorizer.reauthorize,
        )

        def readiness_check() -> None:
            store.check_ready()
            authorizer.check_ready()

        app = create_runner_cache_placement_admission_app(
            controller=controller,
            readiness_check=readiness_check,
            request_body_limit_bytes=config.request_body_limit_bytes,
            active_request_limit=config.active_request_limit,
            total_request_limit=config.total_request_limit,
            queue_wait_timeout_s=config.queue_wait_timeout_s,
            request_timeout_s=config.admission_request_timeout_s,
        )
    except Exception:
        try:
            if store is not None:
                store.close()
        finally:
            client.close()
        raise

    def close_resources() -> None:
        try:
            store.close()
        finally:
            client.close()

    runtime = RunnerCachePlacementAdmissionRuntime(
        app=app,
        config=config,
        close_resources=close_resources,
    )
    app.router.add_event_handler("shutdown", runtime.close)
    return runtime


__all__ = [
    "RunnerCachePlacementAdmissionRuntime",
    "RunnerCachePlacementAdmissionRuntimeConfig",
    "RunnerCachePlacementBindingAuthorizationError",
    "RunnerCachePlacementBindingAuthorizationRequest",
    "RunnerCachePlacementBindingAuthorizationResponse",
    "RunnerCachePlacementBindingAuthorizer",
    "build_runner_cache_placement_admission_runtime",
    "load_runner_cache_placement_admission_runtime_config",
]
