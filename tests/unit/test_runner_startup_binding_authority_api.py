"""Authenticated live binding-authority HTTP boundary coverage."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from kairyu.runners import (
    RunnerCachePlacementBindingAuthorizationDeniedError,
    RunnerCachePlacementBindingAuthorizationResponse,
    RunnerCacheStartupBinding,
    RunnerCacheStartupPlacement,
    create_runner_cache_placement_binding_authority_app,
)

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
TOKEN = "a" * 32
NONCE = "b" * 64


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
        "target_id": "statefulset/model-serving/qwen-runners",
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
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return RunnerCacheStartupBinding(
        binding_id=hashlib.sha256(encoded).hexdigest(),
        **payload,
    )


def _payload(binding: RunnerCacheStartupBinding | None = None) -> dict[str, Any]:
    return {
        "schema_version": ("kairyu-runner-cache-placement-binding-authorization-request-v1"),
        "nonce": NONCE,
        "binding": (binding or _binding()).model_dump(mode="json"),
    }


def _app(
    *,
    reauthorize=None,
    ready: bool = True,
    body_limit: int = 1024 * 1024,
    response_limit: int = 1024 * 1024 + 1,
    active_limit: int = 16,
    total_limit: int = 64,
    request_timeout_s: float = 1.5,
    backend_timeout_s: float = 1.0,
):
    if reauthorize is None:

        def reauthorize(candidate, **_deadline):
            return candidate

    def readiness_check(**_deadline) -> None:
        if not ready:
            raise RuntimeError("private database readiness detail")

    return create_runner_cache_placement_binding_authority_app(
        reauthorize=reauthorize,
        readiness_check=readiness_check,
        bearer_token=TOKEN,
        request_body_limit_bytes=body_limit,
        response_body_limit_bytes=response_limit,
        active_request_limit=active_limit,
        total_request_limit=total_limit,
        request_timeout_s=request_timeout_s,
        backend_timeout_s=backend_timeout_s,
    )


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="https://authority.test",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )


@pytest.mark.asyncio
async def test_authority_echoes_nonce_only_for_the_exact_live_binding() -> None:
    observed: list[RunnerCacheStartupBinding] = []
    binding = _binding()

    deadlines: list[tuple[float, float]] = []

    def reauthorize(
        candidate: RunnerCacheStartupBinding,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> RunnerCacheStartupBinding:
        observed.append(candidate)
        deadlines.append((deadline_monotonic, backend_timeout_s))
        return RunnerCacheStartupBinding.model_validate(binding.model_dump())

    async with _client(_app(reauthorize=reauthorize)) as client:
        response = await client.post("/v1/reauthorize", json=_payload(binding))

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert observed == [binding]
    assert deadlines[0][0] > 0
    assert deadlines[0][1] == 1.0
    assert response.json() == {
        "schema_version": ("kairyu-runner-cache-placement-binding-authorization-response-v1"),
        "nonce": NONCE,
        "binding": binding.model_dump(mode="json"),
    }


@pytest.mark.asyncio
async def test_authentication_precedes_request_parsing_and_protects_readiness() -> None:
    app = _app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="https://authority.test",
    ) as client:
        response = await client.post(
            "/v1/reauthorize",
            content=b"not json",
            headers={"Content-Type": "text/plain", "Authorization": "Bearer wrong"},
        )
        ready = await client.get("/readyz")

    assert response.status_code == 401
    assert response.content == b""
    assert response.headers["www-authenticate"] == "Bearer"
    assert ready.status_code == 401


@pytest.mark.asyncio
async def test_stale_or_explicitly_denied_binding_is_sanitized_conflict() -> None:
    changed = _binding(suffix="b")
    callbacks = (
        lambda _candidate, **_deadline: changed,
        lambda _candidate, **_deadline: (_ for _ in ()).throw(
            RunnerCachePlacementBindingAuthorizationDeniedError("private stale detail")
        ),
    )
    for reauthorize in callbacks:
        async with _client(_app(reauthorize=reauthorize)) as client:
            response = await client.post("/v1/reauthorize", json=_payload())
        assert response.status_code == 409
        assert response.json() == {"detail": "binding is not currently authorized"}
        assert "private" not in response.text


@pytest.mark.asyncio
async def test_backend_failure_and_readiness_are_sanitized() -> None:
    def explode(_candidate, **_deadline):
        raise RuntimeError("postgresql password is private")

    async with _client(_app(reauthorize=explode, ready=False)) as client:
        response = await client.post("/v1/reauthorize", json=_payload())
        ready = await client.get("/readyz")
        health = await client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"detail": "binding authority is temporarily unavailable"}
    assert "password" not in response.text
    assert ready.status_code == 503
    assert ready.content == b""
    assert health.status_code == 204
    assert health.content == b""


@pytest.mark.asyncio
async def test_request_deadline_releases_http_work_when_callback_is_late() -> None:
    entered = threading.Event()
    release = threading.Event()

    def late(
        candidate: RunnerCacheStartupBinding,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> RunnerCacheStartupBinding:
        assert deadline_monotonic > 0
        assert backend_timeout_s == 0.01
        entered.set()
        release.wait(timeout=1)
        return candidate

    try:
        async with _client(
            _app(
                reauthorize=late,
                request_timeout_s=0.05,
                backend_timeout_s=0.01,
            )
        ) as client:
            response = await client.post("/v1/reauthorize", json=_payload())
        assert entered.is_set()
        assert response.status_code == 503
        assert response.json() == {"detail": "binding authority exceeded its internal deadline"}
    finally:
        release.set()


@pytest.mark.asyncio
async def test_request_deadline_covers_response_validation_and_serialization(
    monkeypatch,
) -> None:
    original = RunnerCachePlacementBindingAuthorizationResponse.model_dump_json

    def slow_serialize(self, *args, **kwargs) -> str:
        time.sleep(0.1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        RunnerCachePlacementBindingAuthorizationResponse,
        "model_dump_json",
        slow_serialize,
    )
    async with _client(
        _app(request_timeout_s=0.05, backend_timeout_s=0.01)
    ) as client:
        response = await client.post("/v1/reauthorize", json=_payload())

    assert response.status_code == 503
    assert response.json() == {
        "detail": "binding authority exceeded its internal deadline"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "content_type", "status"),
    [
        (b"{}", "text/plain", 415),
        (b'{"nonce":"a","nonce":"b"}', "application/json", 400),
        (b'{"value":1e999}', "application/json", 400),
        (b"{}", "application/json", 422),
    ],
)
async def test_authority_rejects_untrusted_request_shapes(
    content: bytes,
    content_type: str,
    status: int,
) -> None:
    async with _client(_app()) as client:
        response = await client.post(
            "/v1/reauthorize",
            content=content,
            headers={"Content-Type": content_type},
        )
    assert response.status_code == status
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
async def test_declared_and_streamed_oversized_requests_are_rejected() -> None:
    app = _app(body_limit=64)
    async with _client(app) as client:
        declared = await client.post(
            "/v1/reauthorize",
            content=b"x" * 65,
            headers={"Content-Type": "application/json"},
        )

        async def chunks():
            yield b"x" * 32
            yield b"x" * 33

        streamed = await client.post(
            "/v1/reauthorize",
            content=chunks(),
            headers={"Content-Type": "application/json"},
        )

    assert declared.status_code == 413
    assert streamed.status_code == 413


class _BlockingAuthority:
    entered = threading.Event()
    release = threading.Event()

    def __call__(
        self,
        candidate: RunnerCacheStartupBinding,
        **_deadline,
    ) -> RunnerCacheStartupBinding:
        self.entered.set()
        assert self.release.wait(timeout=2)
        return candidate


@pytest.mark.asyncio
async def test_authority_bounds_total_concurrency() -> None:
    authority = _BlockingAuthority()
    authority.entered.clear()
    authority.release.clear()
    app = _app(reauthorize=authority, active_limit=1, total_limit=1)
    async with _client(app) as client:
        active = asyncio.create_task(client.post("/v1/reauthorize", json=_payload()))
        assert await asyncio.to_thread(authority.entered.wait, 2)
        overflow = await client.post("/v1/reauthorize", json=_payload())
        authority.release.set()
        completed = await active

    assert overflow.status_code == 429
    assert overflow.headers["cache-control"] == "no-store"
    assert completed.status_code == 200


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bearer_token": "short"},
        {"bearer_token": "a" * 31 + " "},
        {"bearer_token": "a" * 31 + "\x00"},
        {"bearer_token": "a" * 31 + "\x7f"},
        {"request_body_limit_bytes": 0},
        {
            "request_body_limit_bytes": 1024,
            "response_body_limit_bytes": 1024,
        },
        {"active_request_limit": 0},
        {"active_request_limit": 2, "total_request_limit": 1},
        {"queue_wait_timeout_s": float("nan")},
        {"request_timeout_s": 1.0, "backend_timeout_s": 1.0},
    ],
)
def test_authority_configuration_is_validated(kwargs: dict[str, Any]) -> None:
    values = {
        "reauthorize": lambda candidate, **_deadline: candidate,
        "readiness_check": lambda **_deadline: None,
        "bearer_token": TOKEN,
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        create_runner_cache_placement_binding_authority_app(**values)
