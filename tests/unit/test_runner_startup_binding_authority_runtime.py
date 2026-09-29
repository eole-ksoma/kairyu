"""Executable assembly coverage for the placement-binding authority boundary."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from kairyu.runners import (
    RunnerCachePlacementBindingAuthorityRuntimeConfig,
    RunnerCacheStartupBinding,
    RunnerCacheStartupPlacement,
    build_runner_cache_placement_binding_authority_runtime,
    load_runner_cache_placement_binding_authority_runtime_config,
)

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
TOKEN = "a" * 32
NONCE = "b" * 64


def _binding() -> RunnerCacheStartupBinding:
    placement = RunnerCacheStartupPlacement(
        placement_id="placement-a",
        node_name="gpu-a",
        resource_flavor="h100-sxm",
        profile_id="h100-sxm-tp1",
        compatibility_approval_id="compat-qwen-h100",
        manifest_digest="a" * 64,
        pin_owner="prestage/model-serving/qwen/placement-a",
        prestage_command_id=hashlib.sha256(b"command-a").hexdigest(),
        prestage_command_generation=1,
        hint_index_revision=10,
        resident_record_generation=20,
        hint_observed_at=NOW - timedelta(seconds=1),
        hint_valid_until=NOW + timedelta(minutes=5),
    )
    payload = {
        "schema_version": "runner-cache-startup-binding-v1",
        "decision_id": "decision-a",
        "decision_fingerprint": hashlib.sha256(b"decision-a").hexdigest(),
        "target_id": "statefulset/model-serving/qwen-runners",
        "target_revision": 7,
        "deployment_id": "model-serving/qwen",
        "model_class": "qwen-14b",
        "model_id": "org/qwen",
        "model_revision": "revision-a",
        "manifest_digest": "a" * 64,
        "placement_binding_id": "placement-binding-a",
        "prewarm_snapshot_id": "snapshot-a",
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
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return RunnerCacheStartupBinding(
        binding_id=hashlib.sha256(encoded).hexdigest(),
        **payload,
    )


def _files(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "bearer_token_file": tmp_path / "authority-token",
        "tls_cert_file": tmp_path / "tls.crt",
        "tls_key_file": tmp_path / "tls.key",
    }
    paths["bearer_token_file"].write_text(f"{TOKEN}\n", encoding="utf-8")
    paths["bearer_token_file"].chmod(0o640)
    paths["tls_cert_file"].write_text("test certificate\n", encoding="utf-8")
    paths["tls_key_file"].write_text("test private key\n", encoding="utf-8")
    paths["tls_key_file"].chmod(0o600)
    return paths


def _config(
    tmp_path: Path,
    **updates: Any,
) -> RunnerCachePlacementBindingAuthorityRuntimeConfig:
    value: dict[str, Any] = _files(tmp_path)
    value.update(updates)
    return RunnerCachePlacementBindingAuthorityRuntimeConfig.model_validate(value)


def test_authority_runtime_config_loader_is_strict_and_cross_validated(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    path = tmp_path / "runtime.json"
    path.write_text(config.model_dump_json(), encoding="utf-8")
    assert load_runner_cache_placement_binding_authority_runtime_config(path) == config

    path.write_text(
        '{"schema_version":"kairyu-runner-cache-placement-binding-authority-runtime-v1",'
        '"schema_version":"kairyu-runner-cache-placement-binding-authority-runtime-v1"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config file is invalid"):
        load_runner_cache_placement_binding_authority_runtime_config(path)
    with pytest.raises(ValueError, match="response_body_limit_bytes"):
        _config(
            tmp_path,
            request_body_limit_bytes=1024,
            response_body_limit_bytes=1024,
        )
    with pytest.raises(ValueError, match="backend_timeout_s"):
        _config(tmp_path, request_timeout_s=1.0, backend_timeout_s=1.0)
    with pytest.raises(ValueError, match="total_request_limit"):
        _config(tmp_path, active_request_limit=2, total_request_limit=1)
    with pytest.raises(ValueError, match="transport_margin_s"):
        _config(
            tmp_path,
            queue_wait_timeout_s=0.5,
            request_timeout_s=1.25,
            transport_margin_s=0.25,
            authorization_client_timeout_s=2.0,
        )


@pytest.mark.asyncio
async def test_authority_runtime_assembles_authenticated_app_and_closes_once(
    tmp_path: Path,
) -> None:
    binding = _binding()
    observed: list[tuple[float, float]] = []
    ready_checks: list[tuple[float, float]] = []
    closes = 0

    def reauthorize(
        candidate: RunnerCacheStartupBinding,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> RunnerCacheStartupBinding:
        assert candidate == binding
        observed.append((deadline_monotonic, backend_timeout_s))
        return candidate

    def ready(*, deadline_monotonic: float, backend_timeout_s: float) -> None:
        ready_checks.append((deadline_monotonic, backend_timeout_s))

    def close() -> None:
        nonlocal closes
        closes += 1

    config = _config(tmp_path)
    runtime = build_runner_cache_placement_binding_authority_runtime(
        config,
        reauthorize=reauthorize,
        readiness_check=ready,
        close_resources=close,
    )
    request = {
        "schema_version": (
            "kairyu-runner-cache-placement-binding-authorization-request-v1"
        ),
        "nonce": NONCE,
        "binding": binding.model_dump(mode="json"),
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=runtime.app),
        base_url="https://authority.test",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        health = await client.get("/health")
        readiness = await client.get("/readyz")
        authorized = await client.post("/v1/reauthorize", json=request)

    assert health.status_code == 204
    assert readiness.status_code == 204
    assert authorized.status_code == 200
    assert authorized.json()["binding"] == binding.model_dump(mode="json")
    assert observed[0][1] == config.backend_timeout_s
    assert ready_checks[0][1] == config.backend_timeout_s
    runtime.close()
    runtime.close()
    assert closes == 1


@pytest.mark.parametrize("secret", ["bearer_token_file", "tls_key_file"])
def test_authority_runtime_rejects_broad_secret_permissions(
    tmp_path: Path,
    secret: str,
) -> None:
    config = _config(tmp_path)
    getattr(config, secret).chmod(0o644)
    with pytest.raises(ValueError, match="permissions are too broad"):
        build_runner_cache_placement_binding_authority_runtime(
            config,
            reauthorize=lambda candidate, **_deadline: candidate,
            readiness_check=lambda **_deadline: None,
        )


@pytest.mark.parametrize("invalid_byte", [b"\x00", b"\x7f"])
def test_authority_runtime_rejects_header_unsafe_token(
    tmp_path: Path,
    invalid_byte: bytes,
) -> None:
    config = _config(tmp_path)
    config.bearer_token_file.write_bytes(b"a" * 31 + invalid_byte + b"\n")
    config.bearer_token_file.chmod(0o640)
    with pytest.raises(ValueError):
        build_runner_cache_placement_binding_authority_runtime(
            config,
            reauthorize=lambda candidate, **_deadline: candidate,
            readiness_check=lambda **_deadline: None,
        )


def test_authority_runtime_rejects_non_regular_material_and_invalid_close_hook(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.tls_cert_file.unlink()
    config.tls_cert_file.mkdir()
    with pytest.raises(ValueError, match="must be a regular file"):
        build_runner_cache_placement_binding_authority_runtime(
            config,
            reauthorize=lambda candidate, **_deadline: candidate,
            readiness_check=lambda **_deadline: None,
        )

    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    valid_config = _config(valid_dir)
    with pytest.raises(TypeError, match="close_resources must be callable"):
        build_runner_cache_placement_binding_authority_runtime(
            valid_config,
            reauthorize=lambda candidate, **_deadline: candidate,
            readiness_check=lambda **_deadline: None,
            close_resources=None,  # type: ignore[arg-type]
        )


def test_authority_runtime_closes_resources_once_when_assembly_fails(
    tmp_path: Path,
) -> None:
    closes = 0

    def close() -> None:
        nonlocal closes
        closes += 1

    with pytest.raises(TypeError, match="reauthorize must be callable"):
        build_runner_cache_placement_binding_authority_runtime(
            _config(tmp_path),
            reauthorize=None,  # type: ignore[arg-type]
            readiness_check=lambda **_deadline: None,
            close_resources=close,
        )
    assert closes == 1
