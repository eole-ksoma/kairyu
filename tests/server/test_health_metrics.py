"""/health, /readyz, /metrics (goal G3 gate C6)."""

import asyncio
import threading

import httpx
import pytest

from kairyu.async_requests import AsyncRequestState
from kairyu.deploy.prober import HealthProber
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.metrics import ServerMetrics
from kairyu.orchestration.replica import ReplicaPool
from kairyu.sampling_params import SamplingParams
from tests.server._legacy_chat import create_legacy_app


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


class _FailingBackend:
    async def generate(self, request):
        raise RuntimeError("replica down")

    async def stream(self, request):
        raise RuntimeError("replica down")
        yield  # pragma: no cover

    async def shutdown(self) -> None:
        return None


def _chat_body(content: str, model: str = "m", **extra) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        **extra,
    }


async def test_health_and_readyz_ok():
    app = create_legacy_app(engines={"m": MockBackend()})
    async with _client(app) as client:
        health = await client.get("/health")
        ready = await client.get("/readyz")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}


async def test_blocked_metrics_render_does_not_block_health() -> None:
    app = create_legacy_app(engines={"m": MockBackend()})
    render_started = threading.Event()
    release_render = threading.Event()
    original_render = app.state.metrics.render

    def blocking_render():
        render_started.set()
        assert release_render.wait(2)
        return original_render()

    app.state.metrics.render = blocking_render
    watchdog = threading.Timer(1, release_render.set)
    watchdog.start()
    try:
        async with _client(app) as client:
            metrics_task = asyncio.create_task(client.get("/metrics"))
            assert await asyncio.to_thread(render_started.wait, 0.5)
            assert not release_render.is_set()
            health = await asyncio.wait_for(client.get("/health"), timeout=0.5)
            assert health.status_code == 200
            release_render.set()
            assert (await metrics_task).status_code == 200
    finally:
        release_render.set()
        watchdog.cancel()


async def test_readyz_503_when_pool_has_no_healthy_replica():
    pool = ReplicaPool([_FailingBackend()], unhealthy_after=1)
    app = create_legacy_app(engines={"m": pool})
    request = GenerationRequest(
        request_id="r1", prompt="p", sampling_params=SamplingParams()
    )
    try:
        await pool.generate(request)
    except RuntimeError:
        pass
    assert not any(pool.healthy)
    async with _client(app) as client:
        ready = await client.get("/readyz")
    assert ready.status_code == 503
    assert ready.json()["status"] == "unready"
    assert "m" in ready.json()["pools"]


async def test_unvalidated_remote_pool_is_unready_unplaceable_and_reports_zero():
    pool = ReplicaPool([MockBackend()])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(503, json={"status": "starting"})
        )
    ) as probe_client:
        # Construction is the serve-boundary contract: a declared readiness URL
        # makes the initial remote entry unknown before any route can serve it.
        HealthProber(
            "remote",
            pool,
            ["http://replica/readyz"],
            interval_s=60,
            client=probe_client,
        )
        app = create_legacy_app(engines={"remote": pool})
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            ready = await client.get("/readyz")
            completion = await client.post(
                "/v1/chat/completions", json=_chat_body("hi", model="remote")
            )
            metrics = (await client.get("/metrics")).text

    assert pool.healthy == (False,)
    assert ready.status_code == 503
    assert ready.json()["pools"] == {
        "remote": {
            "healthy": [False],
            "draining": [False],
            "eligible_ids": [],
        }
    }
    assert completion.status_code == 502
    assert pool.decision_counts["least_outstanding"] == 0
    assert 'kairyu_replica_healthy{pool="remote",replica="0"} 0.0' in metrics


async def test_readyz_explains_healthy_but_draining_pool() -> None:
    pool = ReplicaPool({"replica": MockBackend()})
    pool.drain("replica")
    app = create_legacy_app(engines={"remote": pool})

    async with _client(app) as client:
        ready = await client.get("/readyz")

    assert ready.status_code == 503
    assert ready.json()["pools"] == {
        "remote": {
            "healthy": [True],
            "draining": [True],
            "eligible_ids": [],
        }
    }


async def test_metrics_exposes_conductor_stage_series():
    # Issue #495: orchestration DAG stage latency must be attributable per
    # stage from /metrics, not only from per-request opt-in traces.
    from kairyu.orchestration.orchestrator import Orchestrator

    orchestrator = Orchestrator(engines={"tier1": MockBackend(), "tier2": MockBackend()})
    app = create_legacy_app(
        engines={"m": MockBackend()},
        orchestrator=orchestrator,
    )
    async with _client(app) as client:
        ok = await client.post(
            "/v1/chat/completions",
            # Long enough to route multi_agent, i.e. through the Conductor DAG.
            json=_chat_body("plan this step by step " * 200, model="kairyu-auto"),
        )
        assert ok.status_code == 200
        metrics = await client.get("/metrics")
    text = metrics.text
    assert 'kairyu_conductor_stage_seconds_count{model="kairyu-auto"' in text
    assert 'operation="generation"' in text


async def test_metrics_exposes_request_and_pool_series():
    pool = ReplicaPool([MockBackend(), MockBackend()])
    app = create_legacy_app(engines={"pooled": pool})
    async with _client(app) as client:
        ok = await client.post("/v1/chat/completions", json=_chat_body("hi", model="pooled"))
        assert ok.status_code == 200
        missing = await client.post("/v1/chat/completions", json=_chat_body("hi", model="nope"))
        assert missing.status_code == 404
        metrics = await client.get("/metrics")
    assert metrics.status_code == 200
    text = metrics.text
    assert 'kairyu_requests_total{code="200",model="pooled"} 1.0' in text
    # M1: an unknown model collapses to "unknown" (bounded cardinality)
    assert 'kairyu_requests_total{code="404",model="unknown"} 1.0' in text
    assert "kairyu_request_duration_seconds_bucket" in text
    for phase in (
        "ingress_to_handler",
        "request_validation",
        "backend_prepare",
        "admission",
    ):
        assert (
            'kairyu_preplacement_phase_seconds_count{endpoint="chat",'
            f'phase="{phase}"}}'
        ) in text
    assert 'kairyu_replica_healthy{pool="pooled",replica="0"} 1.0' in text
    assert 'kairyu_replica_outstanding{pool="pooled",replica="0"} 0.0' in text
    assert 'kairyu_pool_decisions_total{pool="pooled",reason="least_outstanding"} 1.0' in text


def test_preplacement_phase_metrics_are_bounded_and_rendered():
    metrics = ServerMetrics()
    metrics.record_preplacement_phase("chat", "request_validation", 2_000_000)

    rendered = metrics.render()[0].decode()
    assert (
        'kairyu_preplacement_phase_seconds_count{endpoint="chat",'
        'phase="request_validation"} 1.0'
    ) in rendered
    assert (
        'kairyu_preplacement_phase_seconds_sum{endpoint="chat",'
        'phase="request_validation"} 0.002'
    ) in rendered
    with pytest.raises(ValueError, match="invalid preplacement endpoint"):
        metrics.record_preplacement_phase("attacker-model", "admission", 1)
    with pytest.raises(ValueError, match="invalid preplacement phase"):
        metrics.record_preplacement_phase("chat", "attacker-phase", 1)


def test_async_store_metrics_fail_open_and_retain_last_good_snapshot() -> None:
    from kairyu.async_requests import RequestQueueMetricsSnapshot

    class _Store:
        store_id = "durable"
        metrics_snapshot_nonblocking = True

        def __init__(self) -> None:
            self.fail = False

        def metrics_snapshot(self):
            if self.fail:
                raise RuntimeError("database unavailable")
            return RequestQueueMetricsSnapshot(
                state_counts={AsyncRequestState.QUEUED: 3},
                oldest_queued_age_seconds=12.5,
                transition_counts={"claim": 7, "reclaim": 2},
            )

    store = _Store()
    metrics = ServerMetrics()
    metrics.track_async_request_store(store)
    healthy = metrics.render()[0].decode()
    store.fail = True
    degraded = metrics.render()[0].decode()

    assert 'kairyu_async_request_queue_depth{store="durable"} 3.0' in healthy
    assert 'kairyu_async_request_oldest_queued_age_seconds{store="durable"} 12.5' in healthy
    assert 'kairyu_async_request_attempts_total{store="durable"} 9.0' in healthy
    assert 'kairyu_async_request_metrics_snapshot_success{store="durable"} 1.0' in healthy
    assert 'kairyu_async_request_queue_depth{store="durable"} 3.0' in degraded
    assert 'kairyu_async_request_metrics_snapshot_success{store="durable"} 0.0' in degraded


def test_blocking_async_store_warms_only_on_first_render() -> None:
    from kairyu.async_requests import RequestQueueMetricsSnapshot

    class _Store:
        store_id = "durable"

        def __init__(self) -> None:
            self.calls = 0

        def metrics_snapshot(self):
            self.calls += 1
            return RequestQueueMetricsSnapshot(
                state_counts={AsyncRequestState.QUEUED: 2},
                oldest_queued_age_seconds=4.0,
                transition_counts={},
            )

    store = _Store()
    metrics = ServerMetrics()
    metrics.track_async_request_store(store)
    assert store.calls == 0

    rendered = metrics.render()[0].decode()

    assert store.calls == 1
    assert 'kairyu_async_request_queue_depth{store="durable"} 2.0' in rendered
    assert 'kairyu_async_request_metrics_snapshot_success{store="durable"} 1.0' in rendered


def test_metrics_exposes_live_cuda_graph_eager_fallback_counter() -> None:
    class _GraphEngine:
        @staticmethod
        def decode_graph_metadata_snapshot():
            return {"cuda_graph_eager_fallbacks": 7}

    metrics = ServerMetrics()
    metrics.track_cuda_graph("qwen", _GraphEngine())

    rendered = metrics.render()[0].decode()
    assert (
        'kairyu_cuda_graph_eager_fallbacks_total{model="qwen"} 7.0'
        in rendered
    )


async def test_pooled_cuda_graph_counter_survives_resets_and_membership() -> None:
    class _GraphEngine:
        def __init__(self, count: int) -> None:
            self.count = count
            self.generation = 1
            self.fail_snapshot = False

        def decode_graph_metadata_snapshot(self):
            if self.fail_snapshot:
                raise RuntimeError("temporary child-process metric failure")
            return {"cuda_graph_eager_fallbacks": self.count}

        def decode_graph_metric_generation_snapshot(self):
            return self.generation

        async def shutdown(self) -> None:
            return None

    first = _GraphEngine(2)
    second = _GraphEngine(3)
    pool = ReplicaPool({"first": first, "second": second})
    metrics = ServerMetrics()
    metrics.track_cuda_graph("pooled", pool)

    assert (
        'kairyu_cuda_graph_eager_fallbacks_total{model="pooled"} 5.0'
        in metrics.render()[0].decode()
    )

    # Service generation, rather than a raw-value heuristic, detects child
    # restarts even when the replacement reaches the same or a greater value
    # before the next scrape.
    first.generation = 2
    assert (
        'kairyu_cuda_graph_eager_fallbacks_total{model="pooled"} 7.0'
        in metrics.render()[0].decode()
    )
    first.generation = 3
    first.count = 4
    assert (
        'kairyu_cuda_graph_eager_fallbacks_total{model="pooled"} 11.0'
        in metrics.render()[0].decode()
    )

    # A decrease inside one declared generation is still treated as a reset,
    # preserving compatibility with metric sources that lack generation IDs.
    first.count = 1
    assert (
        'kairyu_cuda_graph_eager_fallbacks_total{model="pooled"} 12.0'
        in metrics.render()[0].decode()
    )

    # Removing a replica retires its last value instead of decreasing the pool
    # total. Re-adding the same public id has a fresh generation and accumulates.
    await pool.remove_replica("second")
    assert (
        'kairyu_cuda_graph_eager_fallbacks_total{model="pooled"} 12.0'
        in metrics.render()[0].decode()
    )
    replacement = _GraphEngine(4)
    pool.add_replica("second", replacement)
    assert (
        'kairyu_cuda_graph_eager_fallbacks_total{model="pooled"} 16.0'
        in metrics.render()[0].decode()
    )

    first.fail_snapshot = True
    assert (
        'kairyu_cuda_graph_eager_fallbacks_total{model="pooled"} 16.0'
        in metrics.render()[0].decode()
    )


async def test_metrics_disabled_by_settings():
    from kairyu.entrypoints.server.settings import ServerSettings

    app = create_legacy_app(engines={"m": MockBackend()}, settings=ServerSettings(metrics=False))
    async with _client(app) as client:
        response = await client.get("/metrics")
    assert response.status_code == 404


async def test_access_log_adds_request_id_header():
    app = create_legacy_app(engines={"m": MockBackend()})
    async with _client(app) as client:
        response = await client.get("/health")
    assert response.headers.get("x-request-id")


def test_admin_drain_requires_auth_and_flips_readyz(monkeypatch):
    """m10a A5 + security review: /admin/drain is auth-protected when keys are
    configured, and a drained node reports unready."""
    from fastapi.testclient import TestClient

    from kairyu.engine.registry import create_backend
    from kairyu.entrypoints.server.settings import ServerSettings

    monkeypatch.setenv("KAIRYU_TEST_KEYS", "sk-secret")
    app = create_legacy_app(
        {"m": create_backend("mock")},
        settings=ServerSettings(api_keys_env="KAIRYU_TEST_KEYS"),
    )
    with TestClient(app) as client:
        assert client.post("/admin/drain").status_code == 401  # no key
        assert client.get("/readyz").status_code == 200
        ok = client.post(
            "/admin/drain", headers={"Authorization": "Bearer sk-secret"}
        )
        assert ok.status_code == 200
        assert client.get("/readyz").status_code == 503  # draining


def test_admin_drain_requires_admin_key_and_undrain_recovers(monkeypatch):
    # S5: with admin keys configured, an ordinary data-plane key cannot drain
    # the node, and /admin/undrain restores readiness (no restart needed).
    from fastapi.testclient import TestClient

    from kairyu.engine.registry import create_backend
    from kairyu.entrypoints.server.settings import ServerSettings

    monkeypatch.setenv("KAIRYU_TEST_KEYS", "sk-user,sk-admin")
    monkeypatch.setenv("KAIRYU_TEST_ADMIN", "sk-admin")
    app = create_legacy_app(
        {"m": create_backend("mock")},
        settings=ServerSettings(
            api_keys_env="KAIRYU_TEST_KEYS", admin_keys_env="KAIRYU_TEST_ADMIN"
        ),
    )
    with TestClient(app) as client:
        user = {"Authorization": "Bearer sk-user"}
        admin = {"Authorization": "Bearer sk-admin"}
        assert client.post("/admin/drain", headers=user).status_code == 403  # data-plane key
        assert client.get("/readyz").status_code == 200  # still ready
        assert client.post("/admin/drain", headers=admin).status_code == 200
        assert client.get("/readyz").status_code == 503  # draining
        assert client.post("/admin/undrain", headers=user).status_code == 403
        assert client.post("/admin/undrain", headers=admin).status_code == 200
        assert client.get("/readyz").status_code == 200  # recovered


class _ReadinessBackend(MockBackend):
    """A local engine that can report itself unable to serve."""

    def __init__(self, status) -> None:
        super().__init__()
        self._status = status

    def readiness(self):
        return self._status


async def test_readyz_503_when_a_local_engine_reports_itself_unable_to_serve():
    # "constructed" is not "able to serve". A KairyuBackend whose step loop or
    # spawned TP ranks have died stays constructed and used to keep /readyz at
    # 200, so a benchmark ran 14 minutes against a dead 8-GPU deployment.
    from kairyu.engine.backend import EngineReadiness

    app = create_legacy_app(
        engines={"m": _ReadinessBackend(EngineReadiness(False, "ranks not running: [1, 2]"))}
    )
    async with _client(app) as client:
        ready = await client.get("/readyz")
    assert ready.status_code == 503
    assert ready.json() == {
        "status": "unready",
        "engines": {"m": "ranks not running: [1, 2]"},
    }


async def test_readyz_200_when_the_local_engine_reports_ready():
    from kairyu.engine.backend import EngineReadiness

    app = create_legacy_app(engines={"m": _ReadinessBackend(EngineReadiness(True))})
    async with _client(app) as client:
        ready = await client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}


async def test_readyz_reports_unready_when_the_readiness_check_itself_raises():
    # introspection must never 500, but it must not report ready either: a probe
    # that cannot answer is not evidence the engine is fine
    class _Exploding(MockBackend):
        def readiness(self):
            raise RuntimeError("connect https://internal:8443 failed")

    app = create_legacy_app(engines={"m": _Exploding()})
    async with _client(app) as client:
        ready = await client.get("/readyz")
    assert ready.status_code == 503
    detail = ready.json()["engines"]["m"]
    assert "RuntimeError" in detail
    # unauthenticated endpoint: the exception MESSAGE must not travel (review [P2])
    assert "internal" not in detail


async def test_health_stays_ok_for_a_recoverable_engine_fault():
    from kairyu.engine.backend import EngineReadiness

    app = create_legacy_app(
        engines={"m": _ReadinessBackend(EngineReadiness(False, "degraded"))}
    )
    async with _client(app) as client:
        health = await client.get("/health")
        ready = await client.get("/readyz")
    # stop sending work, but do not ask to be replaced
    assert health.status_code == 200
    assert ready.status_code == 503


async def test_health_503s_on_a_fatal_engine_fault():
    # nothing in-process can undo dead TP ranks; without this the container stays
    # up forever serving nothing, which is exactly what happened on hardware
    from kairyu.engine.backend import EngineReadiness

    app = create_legacy_app(
        engines={
            "m": _ReadinessBackend(
                EngineReadiness(False, "ranks not running: [1, 2]", fatal=True)
            )
        }
    )
    async with _client(app) as client:
        health = await client.get("/health")
    assert health.status_code == 503
    assert health.json() == {
        "status": "fatal",
        "engines": {"m": "ranks not running: [1, 2]"},
    }


async def test_embedding_readiness_participates_in_readyz_and_health():
    from kairyu.engine.backend import EngineReadiness

    class _Embedding:
        dimensions = 4

        def readiness(self):
            return EngineReadiness(
                False,
                "embedding startup failed (ValueError)",
                fatal=True,
            )

    app = create_legacy_app(
        engines={"m": MockBackend()},
        embedding_backends={"embed": _Embedding()},
    )
    async with _client(app) as client:
        ready = await client.get("/readyz")
        health = await client.get("/health")

    assert ready.status_code == 503
    assert ready.json() == {
        "status": "unready",
        "embeddings": {"embed": "embedding startup failed (ValueError)"},
    }
    assert health.status_code == 503
    assert health.json() == {
        "status": "fatal",
        "embeddings": {"embed": "embedding startup failed (ValueError)"},
    }
