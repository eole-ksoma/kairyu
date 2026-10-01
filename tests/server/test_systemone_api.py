"""System One (Jev wire API) served by Kairyu in front of an upstream System One server."""

import asyncio
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
import yaml
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from kairyu.deploy.builder import build_app_from_spec
from kairyu.deploy.spec import DeploymentSpec
from kairyu.engine.mock import MockBackend
from kairyu.engine.systemone import HTTPSystemOneBackend
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.systemone_service import SystemOneModel
from kairyu.entrypoints.server.tenancy import TenantConfig, TenantLimits

QUESTIONS = {"urgent": {"type": "noul", "instructions": "Does the customer need a reply?"}}
ANSWER = {
    "model": "openjev-0.1",
    "answers": {"urgent": {"noul": 0.97}},
    "usage": {"input_tokens": 41, "output_tokens": 0},
}


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://kairyu")


def _serve(app) -> tuple[str, uvicorn.Server]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "fake upstream did not start"
        time.sleep(0.02)
    return f"http://127.0.0.1:{port}", server


async def test_spec_served_systemone_forwards_aliases_and_meters_usage(tmp_path):
    received: list[dict] = []

    async def systemone(request):
        received.append(await request.json())
        return JSONResponse(ANSWER, headers={"server-timing": "model;dur=12.0, total;dur=13.0"})

    url, upstream = _serve(Starlette(routes=[Route("/v1/systemone", systemone, methods=["POST"])]))
    spec = DeploymentSpec.model_validate(
        yaml.safe_load(
            f"""
server:
  usage_ledger_path: {tmp_path / "usage.jsonl"}
engines:
  chat: {{backend: mock}}
legacy_chat_models: [chat]
systemone:
  openjev-0.1:
    base_url: {url}
    upstream_model: openjev-0.1
    aliases: [jev-latest]
    description: DiffusionGemma read as a diffusion canvas
"""
        )
    )
    try:
        app = build_app_from_spec(spec)
        async with app.router.lifespan_context(app), _client(app) as client:
            body = {"model": "jev-latest", "state": "Everything is down.", "questions": QUESTIONS}
            answered = await client.post("/v1/systemone", json=body)
            models = (await client.get("/v1/models")).json()
            usage = (await client.get("/admin/usage")).json()["usage"]
    finally:
        upstream.should_exit = True

    assert answered.status_code == 200
    assert answered.json() == ANSWER
    assert answered.headers["server-timing"] == "model;dur=12.0, total;dur=13.0"
    assert received == [
        {"model": "openjev-0.1", "state": "Everything is down.", "questions": QUESTIONS}
    ]
    assert [card["id"] for card in models["data"]] == ["chat"]
    assert [card["name"] for card in models["models"]] == ["openjev-0.1", "jev-latest", "chat"]
    assert usage["default"]["prompt_tokens"] == 41
    assert usage["default"]["completion_tokens"] == 0


async def test_systemone_admission_is_bounded_and_separate_from_chat():
    release = asyncio.Event()
    started = asyncio.Event()

    async def upstream(request: httpx.Request) -> httpx.Response:
        started.set()
        await release.wait()
        return httpx.Response(200, json=ANSWER)

    backend = HTTPSystemOneBackend(
        base_url="http://openjev",
        upstream_model="openjev-0.1",
        max_concurrency=1,
        transport=httpx.MockTransport(upstream),
    )
    app = create_app(
        {"chat": MockBackend()},
        legacy_chat_models={"chat"},
        systemone_models={"openjev-0.1": SystemOneModel(name="openjev-0.1", backend=backend)},
        settings=ServerSettings(max_concurrency=1),
    )
    body = {"model": "openjev-0.1", "state": "s", "questions": QUESTIONS}
    chat = {"model": "chat", "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]}
    async with _client(app) as client:
        first = asyncio.create_task(client.post("/v1/systemone", json=body))
        await started.wait()
        refused = await client.post("/v1/systemone", json=body)
        # the read in flight does not occupy the server's only chat slot
        chatted = await client.post("/v1/chat/completions", json=chat)
        release.set()
        answered = await first

    assert answered.status_code == 200
    assert refused.status_code == 429
    assert refused.headers["retry-after"] == "1"
    assert refused.json()["detail"]["error_type"] == "rate_limit_error"
    assert chatted.status_code == 200


@pytest.mark.parametrize(
    ("request_body", "headers", "status", "detail"),
    [
        ({"model": "gpt-4o", "state": "s", "questions": QUESTIONS}, {"authorization": "Bearer k"},
         400, {"error_type": "api_usage_error", "message": "Unknown model: gpt-4o"}),
        ({"state": "s", "questions": QUESTIONS}, {"authorization": "Bearer k"},
         422, [{"type": "string_type", "loc": ["body", "model"],
                "msg": "Input should be a valid string"}]),
        ({"model": "openjev-0.1", "state": "s", "questions": QUESTIONS, "samples": "many"},
         {"authorization": "Bearer k"},
         422, [{"type": "int_parsing", "loc": ["body", "samples"],
                "msg": "Input should be a valid integer"}]),
        # past float's range and Python's int digit limit: still a 422, not a 500
        ({"model": "openjev-0.1", "state": "s", "questions": QUESTIONS, "samples": "9" * 5000},
         {"authorization": "Bearer k"},
         422, [{"type": "int_parsing", "loc": ["body", "samples"],
                "msg": "Input should be a valid integer"}]),
        ({"model": "openjev-0.1", "state": "s", "questions": QUESTIONS}, {},
         401, {"error_type": "authentication_error", "message": "missing or invalid API key"}),
    ],
)  # fmt: skip
async def test_systemone_errors_use_jev_shapes(monkeypatch, request_body, headers, status, detail):
    calls: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=ANSWER)

    monkeypatch.setenv("KAIRYU_TEST_KEYS", "k")
    backend = HTTPSystemOneBackend(
        base_url="http://openjev",
        upstream_model="openjev-0.1",
        transport=httpx.MockTransport(upstream),
    )
    app = create_app(
        {"chat": MockBackend()},
        legacy_chat_models={"chat"},
        systemone_models={"openjev-0.1": SystemOneModel(name="openjev-0.1", backend=backend)},
        settings=ServerSettings(api_keys_env="KAIRYU_TEST_KEYS"),
    )
    async with _client(app) as client:
        response = await client.post("/v1/systemone", json=request_body, headers=headers)

    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert calls == []


@pytest.mark.parametrize(
    ("upstream_status", "upstream_body", "status", "error_type"),
    [
        (529, {"detail": {"error_type": "overloaded_error", "message": "busy"}},
         529, "overloaded_error"),
        (200, {**ANSWER, "usage": {"input_tokens": 41, "output_tokens": "0"}}, 502, "api_error"),
        (200, {**ANSWER, "usage": {"input_tokens": 41}}, 502, "api_error"),
    ],
)  # fmt: skip
async def test_unmeterable_upstream_replies_are_not_billed(
    tmp_path, upstream_status, upstream_body, status, error_type
):
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(upstream_status, json=upstream_body, headers={"retry-after": "1"})

    backend = HTTPSystemOneBackend(
        base_url="http://openjev",
        upstream_model="openjev-0.1",
        transport=httpx.MockTransport(upstream),
    )
    app = create_app(
        {"chat": MockBackend()},
        legacy_chat_models={"chat"},
        systemone_models={"openjev-0.1": SystemOneModel(name="openjev-0.1", backend=backend)},
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
    )
    async with _client(app) as client:
        response = await client.post(
            "/v1/systemone", json={"model": "openjev-0.1", "state": "s", "questions": QUESTIONS}
        )
        usage = (await client.get("/admin/usage")).json()["usage"]

    assert response.status_code == status
    assert json.loads(response.content)["detail"]["error_type"] == error_type
    assert usage == {}


@pytest.mark.parametrize(
    ("limits", "body", "requests"),
    [
        # 32 samples of a 4,096-token thought bill far more than 10,000 tokens
        (TenantLimits(tokens_per_minute=10_000, token_burst=10_000),
         {"state": "s", "questions": {"q": {"type": "noul"}}, "samples": 32, "think": 4096}, 1),
        # the upstream reads "32" and 4096.0 as numbers too
        (TenantLimits(tokens_per_minute=10_000, token_burst=10_000),
         {"state": "s", "questions": {"q": {"type": "noul"}}, "samples": "32", "think": 4096.0},
         1),
        # a sequential read repeats every question in each group's prompt
        (TenantLimits(tokens_per_minute=150_000, token_burst=150_000),
         {"state": "s", "sequential": True, "questions": {
             f"q{i}": {"type": "noul", "instructions": "x" * 512} for i in range(128)}}, 1),
        (TenantLimits(requests_per_minute=1, request_burst=1),
         {"state": "s", "questions": QUESTIONS}, 2),
    ],
)  # fmt: skip
async def test_tenant_limits_refuse_before_dispatch_in_jev_shape(limits, body, requests):
    calls: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=ANSWER)

    backend = HTTPSystemOneBackend(
        base_url="http://openjev",
        upstream_model="openjev-0.1",
        transport=httpx.MockTransport(upstream),
    )
    app = create_app(
        {"chat": MockBackend()},
        legacy_chat_models={"chat"},
        systemone_models={"openjev-0.1": SystemOneModel(name="openjev-0.1", backend=backend)},
        tenant_config=TenantConfig(limits={"default": limits}),
    )
    async with _client(app) as client:
        responses = [
            await client.post("/v1/systemone", json={"model": "openjev-0.1", **body})
            for _ in range(requests)
        ]

    refused = responses[-1]
    assert refused.status_code == 429
    assert refused.json()["detail"]["error_type"] == "rate_limit_error"
    assert len(calls) == requests - 1


async def test_each_model_keeps_its_own_body_limit():
    calls: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=ANSWER)

    def model(name: str, limit: int) -> SystemOneModel:
        backend = HTTPSystemOneBackend(
            base_url="http://openjev", upstream_model=name, transport=httpx.MockTransport(upstream)
        )
        return SystemOneModel(name=name, backend=backend, max_body_bytes=limit)

    app = create_app(
        {"chat": MockBackend()},
        legacy_chat_models={"chat"},
        systemone_models={"small": model("small", 200), "large": model("large", 2_000)},
    )
    body = {"model": "small", "state": "x" * 500, "questions": QUESTIONS}
    async with _client(app) as client:
        response = await client.post("/v1/systemone", json=body)

    assert response.status_code == 413
    assert response.json()["detail"]["error_type"] == "api_usage_error"
    assert calls == []
