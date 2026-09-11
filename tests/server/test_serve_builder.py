"""DeploymentSpec -> app builder: pool wiring, affinity over HTTP, lifespan (gate C1)."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families

import kairyu.deploy.builder as builder_module
from kairyu.deploy.builder import build_app_from_config, build_app_from_spec
from kairyu.deploy.spec import ServerSection, load_deployment_spec
from kairyu.dsl.loader import load_spec
from kairyu.engine.backend import EngineReadiness
from kairyu.engine.embedding import EmbeddingResult
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import TenantLimits, UsageLedger
from kairyu.orchestration.orchestrator import Orchestrator

POOLED_YAML = """
engines:
  small: { backend: mock }
pools:
  pooled:
    replicas:
      - { backend: mock }
      - { backend: mock }
      - { backend: mock }
"""

GATEWAY_GPU_YAML = Path(__file__).parents[2] / "deploy/compose/gateway-gpu.yaml"
GATEWAY_YAML = Path(__file__).parents[2] / "deploy/compose/gateway.yaml"
ROUTING_YAML = Path(__file__).parents[2] / "deploy/compose/routing.yaml"
QWEN_AUTO_GATEWAY_YAML = (
    Path(__file__).parents[2] / "deploy/verification/qwen3-32b-multi-gpu/auto-gateway.yaml"
)


class _ShutdownBackend(MockBackend):
    def __init__(self) -> None:
        super().__init__()
        self.shutdown_count = 0

    async def shutdown(self) -> None:
        self.shutdown_count += 1


class _StartupBackend(_ShutdownBackend):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.startup_count = 0

    async def startup(self) -> None:
        self.startup_count += 1
        if self.fail:
            raise RuntimeError("backend startup failed")


class _LifecycleEmbedding:
    dimensions = 4
    reports_exact_usage = True

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.started = False
        self.startup_count = 0
        self.shutdown_count = 0

    async def startup(self) -> None:
        self.startup_count += 1
        if self.fail:
            raise RuntimeError("embedding startup failed")
        self.started = True

    def readiness(self) -> EngineReadiness:
        return EngineReadiness(self.started, "embedding is not started")

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=tuple((1.0, 0.0, 0.0, 0.0) for _ in texts),
            prompt_tokens=len(texts),
            usage_exact=True,
        )

    async def shutdown(self) -> None:
        self.shutdown_count += 1
        self.started = False


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _write_tiny_chat_tokenizer(
    path: Path,
    *,
    chat_template: str | None,
) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    path.mkdir()
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "<BOS>": 1,
                "user": 2,
                "assistant": 3,
                "hello": 4,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.save(str(path / "tokenizer.json"))
    config: dict[str, object] = {
        "unk_token": "[UNK]",
        "bos_token": {"content": "<BOS>"},
    }
    if chat_template is not None:
        config["chat_template"] = chat_template
    (path / "tokenizer_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )


def _chat_body(content: str, model: str = "pooled", **extra) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        **extra,
    }


def test_chat_policy_fails_before_backend_construction(tmp_path, monkeypatch):
    model = tmp_path / "model"
    _write_tiny_chat_tokenizer(model, chat_template=None)
    spec = load_deployment_spec(
        f"""
engines:
  unsafe:
    backend: kairyu
    options:
      model_path: {model}
"""
    )
    created: list[str] = []

    def record_create(name, **_options):
        created.append(name)
        return MockBackend()

    monkeypatch.setattr(builder_module, "create_backend", record_create)

    with pytest.raises(ValueError, match="unsafe.*has no real chat template"):
        build_app_from_spec(spec)

    assert created == []


def test_invalid_auto_loaded_template_fails_before_backend_construction(
    tmp_path,
    monkeypatch,
):
    model = tmp_path / "model"
    _write_tiny_chat_tokenizer(model, chat_template="{% if")
    spec = load_deployment_spec(
        f"""
engines:
  invalid:
    backend: kairyu
    options:
      model_path: {model}
"""
    )
    created: list[str] = []

    def record_create(name, **_options):
        created.append(name)
        return MockBackend()

    monkeypatch.setattr(builder_module, "create_backend", record_create)

    with pytest.raises(ValueError, match="compile tokenizer chat template.*invalid"):
        build_app_from_spec(spec)

    assert created == []


def test_empty_explicit_template_fails_before_backend_construction(
    tmp_path,
    monkeypatch,
):
    created: list[str] = []

    def record_create(name, **_options):
        created.append(name)
        return MockBackend()

    monkeypatch.setattr(builder_module, "create_backend", record_create)
    spec = load_deployment_spec(
        """
engines:
  invalid:
    backend: openai
    options:
      base_url: http://replica/v1
      model: upstream
chat_templates:
  invalid: "   "
"""
    )

    with pytest.raises(ValueError, match="non-empty"):
        build_app_from_spec(spec, base_dir=tmp_path)

    assert created == []


def test_remote_chat_template_fails_before_backend_construction(
    tmp_path,
    monkeypatch,
):
    created: list[str] = []

    def record_create(name, **_options):
        created.append(name)
        return MockBackend()

    monkeypatch.setattr(builder_module, "create_backend", record_create)
    spec = load_deployment_spec(
        """
engines:
  remote:
    backend: openai
    options:
      base_url: http://replica/v1
      model: upstream
      api_key_env: null
chat_templates:
  remote: "{{ messages[0].content }}"
"""
    )

    with pytest.raises(ValueError, match="pre-rendered.*cannot preserve.*legacy_chat_models"):
        build_app_from_spec(spec, base_dir=tmp_path)

    assert created == []


def test_remote_missing_policy_diagnoses_legacy_only_before_construction(monkeypatch):
    created: list[str] = []

    def record_create(name, **_options):
        created.append(name)
        return MockBackend()

    monkeypatch.setattr(builder_module, "create_backend", record_create)
    spec = load_deployment_spec(
        """
engines:
  remote:
    backend: openai
    options:
      base_url: http://replica/v1
      model: upstream
"""
    )

    with pytest.raises(
        ValueError,
        match="cannot preserve.*pre-rendered chat prompt.*legacy_chat_models",
    ):
        build_app_from_spec(spec)

    assert created == []


def test_nonlocal_explicit_template_cannot_silently_drop_bos(monkeypatch):
    created: list[str] = []

    def record_create(name, **_options):
        created.append(name)
        return MockBackend()

    monkeypatch.setattr(builder_module, "create_backend", record_create)
    spec = load_deployment_spec(
        """
engines:
  local-vllm:
    backend: vllm
    options:
      model: meta-llama/Llama-3.1-8B-Instruct
chat_templates:
  local-vllm: "{{ bos_token }}USER={{ messages[0].content }}"
"""
    )

    with pytest.raises(
        ValueError,
        match="special-token variables.*bos_token.*metadata is not available",
    ):
        build_app_from_spec(spec)

    assert created == []


def test_explicit_template_rejects_missing_local_special_token_value(
    tmp_path,
    monkeypatch,
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    created: list[str] = []

    def record_create(name, **_options):
        created.append(name)
        return MockBackend()

    monkeypatch.setattr(builder_module, "create_backend", record_create)
    spec = load_deployment_spec(
        f"""
engines:
  local:
    backend: vllm
    options: {{model: local, tokenizer: {model}}}
chat_templates:
  local: "{{{{ bos_token }}}}USER={{{{ messages[0].content }}}}"
"""
    )

    with pytest.raises(
        ValueError,
        match="special-token variables.*bos_token.*does not define",
    ):
        build_app_from_spec(spec)

    assert created == []


def test_static_pool_special_token_template_requires_metadata_for_every_replica(
    tmp_path,
    monkeypatch,
):
    model = tmp_path / "model"
    _write_tiny_chat_tokenizer(model, chat_template=None)
    created: list[str] = []

    def record_create(name, **_options):
        created.append(name)
        return MockBackend()

    monkeypatch.setattr(builder_module, "create_backend", record_create)
    spec = load_deployment_spec(
        f"""
pools:
  mixed:
    replicas:
      - backend: vllm
        options: {{model: local, tokenizer: {model}}}
      - backend: vllm
        options: {{model: org/not-materialized}}
chat_templates:
  mixed: "{{{{ bos_token }}}}USER={{{{ messages[0].content }}}}"
"""
    )

    with pytest.raises(
        ValueError,
        match="special-token variables.*metadata is not available for every",
    ):
        build_app_from_spec(spec)

    assert created == []


@pytest.mark.parametrize(
    ("served_section", "template_model", "error"),
    [
        (
            """
pools:
  dynamic:
    discovery:
      type: kubernetes_endpoints
      service: replicas
      port: http
""",
            "dynamic",
            "discovered pool.*cannot prove.*legacy_chat_models",
        ),
        (
            """
engines:
  local: {backend: mock}
orchestrators:
  auto: {spec: unused.yaml}
""",
            "auto",
            "orchestrated model.*cannot safely consume.*legacy_chat_models",
        ),
    ],
)
def test_non_preserving_chat_template_routes_fail_before_backend_construction(
    served_section,
    template_model,
    error,
    monkeypatch,
):
    created: list[str] = []

    def record_create(name, **_options):
        created.append(name)
        return MockBackend()

    monkeypatch.setattr(builder_module, "create_backend", record_create)
    spec = load_deployment_spec(
        served_section
        + f'\nchat_templates:\n  {template_model}: "{{{{ messages[0].content }}}}"\n'
    )

    with pytest.raises(ValueError, match=error):
        build_app_from_spec(spec)

    assert created == []


async def test_builder_auto_loads_template_and_special_tokens(tmp_path, monkeypatch):
    model = tmp_path / "model"
    _write_tiny_chat_tokenizer(
        model,
        chat_template=(
            "{{ bos_token }}{% for message in messages %}"
            "{{ message.role }} {{ message.content }} {% endfor %}"
            "{% if add_generation_prompt %}assistant{% endif %}"
        ),
    )
    backend = MockBackend()
    monkeypatch.setattr(
        builder_module,
        "create_backend",
        lambda _name, **_options: backend,
    )
    spec = load_deployment_spec(
        f"""
engines:
  safe:
    backend: kairyu
    options:
      model_path: {model}
"""
    )

    app = build_app_from_spec(spec)
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_body("hello", model="safe", max_tokens=1),
        )

    assert response.status_code == 200
    assert backend.prompts_seen == ("<BOS>user hello assistant",)


async def test_explicit_template_overrides_invalid_checkpoint_template(
    tmp_path,
    monkeypatch,
):
    model = tmp_path / "model"
    _write_tiny_chat_tokenizer(model, chat_template="checkpoint")
    config_path = model / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["chat_template"] = {"default": 7}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    backend = MockBackend()
    monkeypatch.setattr(
        builder_module,
        "create_backend",
        lambda _name, **_options: backend,
    )
    spec = load_deployment_spec(
        f"""
engines:
  safe:
    backend: kairyu
    options:
      model_path: {model}
chat_templates:
  safe: "{{{{ bos_token }}}}override {{{{ messages[0].content }}}}"
"""
    )

    app = build_app_from_spec(spec)
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_body("hello", model="safe", max_tokens=1),
        )

    assert response.status_code == 200
    assert backend.prompts_seen == ("<BOS>override hello",)


async def test_builder_legacy_chat_requires_model_opt_in(
    tmp_path,
    monkeypatch,
    caplog,
):
    backend = MockBackend()
    monkeypatch.setattr(
        builder_module,
        "create_backend",
        lambda _name, **_options: backend,
    )
    spec = load_deployment_spec(
        """
engines:
  remote:
    backend: openai
    options:
      base_url: http://replica/v1
      model: upstream-model
      api_key_env: null
legacy_chat_models: [remote]
"""
    )

    with caplog.at_level("WARNING", logger="kairyu.deploy"):
        app = build_app_from_spec(spec, base_dir=tmp_path)
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_body("hello", model="remote", max_tokens=1),
        )

    assert response.status_code == 200
    assert backend.prompts_seen == ("user: hello\nassistant:",)
    deploy_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "kairyu.deploy"
    ]
    assert deploy_messages == [
        "model 'remote' explicitly enables the legacy role-concatenation chat "
        "renderer; use a tokenizer-owned HF template for real models"
    ]


def test_explicit_legacy_policy_ignores_unused_checkpoint_template_metadata(
    tmp_path,
    monkeypatch,
):
    model = tmp_path / "model"
    _write_tiny_chat_tokenizer(model, chat_template="unused")
    config_path = model / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["chat_template"] = {"default": 7}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        builder_module,
        "create_backend",
        lambda _name, **_options: MockBackend(),
    )
    spec = load_deployment_spec(
        f"""
engines:
  old:
    backend: kairyu
    options:
      model_path: {model}
legacy_chat_models: [old]
"""
    )

    app = build_app_from_spec(spec)

    assert app.state.deployment_spec.legacy_chat_models == frozenset({"old"})


def _metric_value(
    text: str,
    name: str,
    labels: dict[str, str],
) -> float:
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name == name and sample.labels == labels:
                return float(sample.value)
    raise AssertionError(f"missing metric {name}{labels}")


async def test_pool_is_served_and_affinity_sticks():
    app = build_app_from_spec(load_deployment_spec(POOLED_YAML))
    async with _client(app) as client:
        models = await client.get("/v1/models")
        assert {"small", "pooled"} <= {m["id"] for m in models.json()["data"]}

        for _ in range(4):
            response = await client.post(
                "/v1/chat/completions", json=_chat_body("hi", user="alice")
            )
            assert response.status_code == 200
        anonymous = await client.post("/v1/chat/completions", json=_chat_body("hi"))
        assert anonymous.status_code == 200

        metrics = (await client.get("/metrics")).text
    assert 'kairyu_pool_decisions_total{pool="pooled",reason="session_affinity"} 4.0' in metrics
    assert 'kairyu_pool_decisions_total{pool="pooled",reason="least_outstanding"} 1.0' in metrics


async def test_pool_prefix_index_option_enables_cross_session_reuse():
    app = build_app_from_spec(
        load_deployment_spec(
            """
pools:
  pooled:
    replicas:
      - {backend: mock}
      - {backend: mock}
    prefix_index: true
"""
        )
    )
    content = "shared-prefix " * 32

    async with _client(app) as client:
        first = await client.post(
            "/v1/chat/completions",
            json=_chat_body(content),
        )
        second = await client.post(
            "/v1/chat/completions",
            json=_chat_body(content),
        )
        metrics = (await client.get("/metrics")).text

    assert first.status_code == second.status_code == 200
    assert (
        'kairyu_pool_decisions_total{pool="pooled",reason="prefix_match"} 1.0'
        in metrics
    )


async def test_gpu_gateway_exposes_canonical_default_model():
    app = build_app_from_config(GATEWAY_GPU_YAML)
    async with _client(app) as client:
        models = await client.get("/v1/models")
        ids = {model["id"] for model in models.json()["data"]}

    assert "default" in ids
    assert "llama" not in ids


def test_shipped_kairyu_gateway_hops_select_kairyu_capabilities():
    for path in (GATEWAY_YAML, GATEWAY_GPU_YAML):
        spec = load_deployment_spec(path)
        replicas = next(iter(spec.pools.values())).replicas
        assert replicas
        assert all(entry.options["upstream"] == "kairyu" for entry in replicas)

    routing = load_spec(ROUTING_YAML)
    assert routing.workers
    assert all(worker.options["upstream"] == "kairyu" for worker in routing.workers)


async def test_qwen_auto_gate_resolves_relative_orchestrator_spec():
    app = build_app_from_config(QWEN_AUTO_GATEWAY_YAML)
    async with _client(app) as client:
        models = await client.get("/v1/models")
        ids = {model["id"] for model in models.json()["data"]}

    assert ids == {"qwen3-32b", "kairyu-auto", "kairyu-auto-max"}


async def test_builder_wires_named_embedding_models():
    app = build_app_from_spec(
        load_deployment_spec(
            """
engines:
  chat: { backend: mock }
embeddings:
  embed-small: { backend: mock, dimensions: 4 }
  embed-large: { backend: mock, dimensions: 9 }
"""
        )
    )

    async with _client(app) as client:
        models = await client.get("/v1/models")
        small = await client.post(
            "/v1/embeddings",
            json={
                "model": "embed-small",
                "input": "hello",
                "encoding_format": "float",
            },
        )
        large = await client.post(
            "/v1/embeddings",
            json={
                "model": "embed-large",
                "input": "hello",
                "encoding_format": "float",
            },
        )

    assert {model["id"] for model in models.json()["data"]} == {
        "chat",
        "embed-small",
        "embed-large",
    }
    assert small.status_code == 200
    assert small.json()["model"] == "embed-small"
    assert len(small.json()["data"][0]["embedding"]) == 4
    assert large.status_code == 200
    assert large.json()["model"] == "embed-large"
    assert len(large.json()["data"][0]["embedding"]) == 9


def test_schema_rejects_unknown_embedding_backend():
    with pytest.raises(ValueError):
        load_deployment_spec(
            """
engines:
  chat: { backend: mock }
embeddings:
  embed: { backend: does-not-exist, dimensions: 4 }
"""
        )


async def test_header_session_takes_precedence_over_user():
    app = build_app_from_spec(load_deployment_spec(POOLED_YAML))
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_body("hi", user="alice"),
            headers={"X-Session-ID": "sess-1"},
        )
        assert response.status_code == 200
        metrics = (await client.get("/metrics")).text
    assert 'reason="session_affinity"} 1.0' in metrics


async def test_lifespan_starts_prober_and_shuts_down_engines():
    yaml_text = """
pools:
  remote:
    replicas:
      - backend: openai
        options: { base_url: "http://gpu-0:8000/v1", model: "m", api_key_env: null }
    probe_interval_s: 0.05
legacy_chat_models: [remote]
"""
    app = build_app_from_spec(load_deployment_spec(yaml_text))
    assert len(app.state.probers) == 1
    # httpx's ASGITransport never runs the lifespan; drive it directly. The
    # prober task must start and cancel cleanly, and shutdown must close engines.
    async with app.router.lifespan_context(app):
        async with _client(app) as client:
            assert (await client.get("/health")).status_code == 200


async def test_lifespan_starts_direct_and_static_pool_backends_before_serving(
    monkeypatch,
):
    direct = _StartupBackend()
    replica = _StartupBackend()
    created = iter((direct, replica))
    monkeypatch.setattr(
        builder_module,
        "create_backend",
        lambda *_args, **_kwargs: next(created),
    )
    app = build_app_from_spec(
        load_deployment_spec(
            """
engines:
  direct: { backend: mock }
pools:
  static:
    replicas:
      - { backend: mock }
"""
        )
    )

    assert direct.startup_count == replica.startup_count == 0
    async with app.router.lifespan_context(app):
        # Entering the context is the serving boundary: both builder-owned
        # concrete engines, including the replica hidden by the pool, are ready.
        assert direct.startup_count == replica.startup_count == 1
        assert direct.shutdown_count == replica.shutdown_count == 0

    assert direct.shutdown_count == replica.shutdown_count == 1


async def test_lifespan_eagerly_starts_linked_orchestrator_workers(
    monkeypatch,
    tmp_path,
):
    worker = _StartupBackend()
    (tmp_path / "orchestrator.yaml").write_text(
        "workers:\n  - {name: native, backend: mock}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kairyu.dsl.loader.create_backend",
        lambda *_args, **_kwargs: worker,
    )
    app = build_app_from_spec(
        load_deployment_spec(
            """
engines:
  m: {backend: mock}
orchestrator: {spec: orchestrator.yaml}
legacy_chat_models: [kairyu-auto]
"""
        ),
        base_dir=tmp_path,
    )

    assert worker.startup_count == 0
    async with app.router.lifespan_context(app):
        assert worker.startup_count == 1
        assert worker.shutdown_count == 0

    assert worker.shutdown_count == 1


async def test_lifespan_borrowed_orchestrator_engine_has_one_owner(
    monkeypatch,
    tmp_path,
):
    borrowed = _StartupBackend()
    (tmp_path / "orchestrator.yaml").write_text(
        "workers:\n  - {name: tier1, engine_ref: direct}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        builder_module,
        "create_backend",
        lambda *_args, **_kwargs: borrowed,
    )

    def unexpected_factory(*_args, **_kwargs):
        raise AssertionError("borrowed workers must not construct a backend")

    monkeypatch.setattr("kairyu.dsl.loader.create_backend", unexpected_factory)
    app = build_app_from_spec(
        load_deployment_spec(
            """
engines:
  direct: {backend: mock}
orchestrator: {spec: orchestrator.yaml}
legacy_chat_models: [kairyu-auto]
"""
        ),
        base_dir=tmp_path,
    )

    async with app.router.lifespan_context(app):
        assert borrowed.startup_count == 1
        assert borrowed.shutdown_count == 0

    assert borrowed.shutdown_count == 1


async def test_lifespan_startup_failure_prevents_serving_and_shuts_down_all_owned_resources(
    monkeypatch,
):
    direct = _StartupBackend()
    failing_replica = _StartupBackend(fail=True)
    created = iter((direct, failing_replica))
    monkeypatch.setattr(
        builder_module,
        "create_backend",
        lambda *_args, **_kwargs: next(created),
    )
    app = build_app_from_spec(
        load_deployment_spec(
            """
engines:
  direct: { backend: mock }
pools:
  static:
    replicas:
      - { backend: mock }
"""
        )
    )
    serving_started = False

    with pytest.raises(RuntimeError, match="backend startup failed"):
        async with app.router.lifespan_context(app):
            serving_started = True

    assert serving_started is False
    assert direct.startup_count == failing_replica.startup_count == 1
    assert direct.shutdown_count == failing_replica.shutdown_count == 1


async def test_lifespan_owns_embedding_startup_readiness_and_shutdown(
    tmp_path,
    monkeypatch,
):
    embedding = _LifecycleEmbedding()
    captured = {}

    def factory(**options):
        captured.update(options)
        return embedding

    monkeypatch.setitem(
        builder_module._EMBEDDING_BACKEND_FACTORIES,
        "fastembed",
        factory,
    )
    app = build_app_from_spec(
        load_deployment_spec(
            f"""
engines:
  direct: {{ backend: mock }}
embeddings:
  embed:
    backend: fastembed
    model: test/model
    model_path: models/embed
    revision: {"a" * 40}
    model_sha256: {"b" * 64}
    provenance_sha256: {"c" * 64}
    dimensions: 4
"""
        ),
        base_dir=tmp_path,
    )

    assert captured["model_path"] == tmp_path / "models/embed"
    assert captured["provenance_sha256"] == "c" * 64
    assert embedding.startup_count == embedding.shutdown_count == 0
    async with _client(app) as client:
        not_ready = await client.get("/readyz")
    assert not_ready.status_code == 503
    assert not_ready.json()["embeddings"] == {
        "embed": "embedding is not started"
    }

    async with app.router.lifespan_context(app):
        assert embedding.startup_count == 1
        async with _client(app) as client:
            assert (await client.get("/readyz")).status_code == 200
    assert embedding.shutdown_count == 1


async def test_embedding_startup_failure_prevents_serving_and_rolls_back(
    monkeypatch,
):
    embedding = _LifecycleEmbedding(fail=True)
    monkeypatch.setitem(
        builder_module._EMBEDDING_BACKEND_FACTORIES,
        "fastembed",
        lambda **_options: embedding,
    )
    app = build_app_from_spec(
        load_deployment_spec(
            f"""
engines:
  direct: {{ backend: mock }}
embeddings:
  embed:
    backend: fastembed
    model: test/model
    model_path: /models/embed
    revision: {"a" * 40}
    model_sha256: {"b" * 64}
    provenance_sha256: {"c" * 64}
    dimensions: 4
"""
        )
    )
    serving_started = False

    with pytest.raises(RuntimeError, match="embedding startup failed"):
        async with app.router.lifespan_context(app):
            serving_started = True

    assert serving_started is False
    assert embedding.startup_count == 1
    assert embedding.shutdown_count == 1


async def test_lifespan_immediate_probe_validates_only_ready_remote_replicas():
    yaml_text = """
pools:
  remote:
    replicas:
      - backend: openai
        options: { base_url: "http://gpu-0:8000/v1", model: "m", api_key_env: null }
      - backend: openai
        options: { base_url: "http://gpu-1:8000/v1", model: "m", api_key_env: null }
    probe_interval_s: 60
legacy_chat_models: [remote]
"""
    app = build_app_from_spec(load_deployment_spec(yaml_text))
    prober = app.state.probers[0]
    pool = prober._pool
    both_probed = asyncio.Event()
    requests = 0

    def probe_response(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 2:
            both_probed.set()
        if request.url.host == "gpu-0":
            return httpx.Response(200, json={"status": "ready"})
        return httpx.Response(503, json={"status": "starting"})

    prober._client = httpx.AsyncClient(transport=httpx.MockTransport(probe_response))

    # Initial URL mappings are applied during construction, before create_app
    # exposes the routes or any backend request can be used as a health signal.
    assert pool.healthy == (False, False)
    async with _client(app) as client:
        assert (await client.get("/readyz")).status_code == 503

    async with app.router.lifespan_context(app):
        await asyncio.wait_for(both_probed.wait(), timeout=1)
        for _ in range(10):
            if pool.healthy == (True, False):
                break
            await asyncio.sleep(0)

        async with _client(app) as client:
            ready = await client.get("/readyz")
            metrics = (await client.get("/metrics")).text

        assert pool.healthy == (True, False)
        assert ready.status_code == 200
        assert 'kairyu_replica_healthy{pool="remote",replica="0"} 1.0' in metrics
        assert 'kairyu_replica_healthy{pool="remote",replica="1"} 0.0' in metrics


async def test_build_from_config_resolves_orchestrator_relative_to_file(tmp_path):
    (tmp_path / "orchestrator.yaml").write_text(
        """
workers:
  - { name: tier1, backend: mock }
  - { name: tier2, backend: mock }
""",
        encoding="utf-8",
    )
    (tmp_path / "deploy.yaml").write_text(
        """
engines:
  m: { backend: mock }
orchestrator:
  spec: orchestrator.yaml
legacy_chat_models: [kairyu-auto]
""",
        encoding="utf-8",
    )
    app = build_app_from_config(tmp_path / "deploy.yaml")
    async with _client(app) as client:
        models = await client.get("/v1/models")
        ids = {m["id"] for m in models.json()["data"]}
    assert "kairyu-auto" in ids


ORCHESTRATOR_SPEC = """
workers:
  - { name: tier1, backend: mock }
  - { name: tier2, backend: mock }
"""


async def test_named_orchestrators_served_and_answer(tmp_path):
    (tmp_path / "auto.yaml").write_text(ORCHESTRATOR_SPEC, encoding="utf-8")
    (tmp_path / "auto_max.yaml").write_text(ORCHESTRATOR_SPEC, encoding="utf-8")
    (tmp_path / "deploy.yaml").write_text(
        """
engines:
  m: { backend: mock }
orchestrators:
  kairyu-auto: { spec: auto.yaml }
  kairyu-auto-max: { spec: auto_max.yaml }
legacy_chat_models: [kairyu-auto, kairyu-auto-max]
""",
        encoding="utf-8",
    )
    app = build_app_from_config(tmp_path / "deploy.yaml")
    async with _client(app) as client:
        models = await client.get("/v1/models")
        ids = {m["id"] for m in models.json()["data"]}
        assert {"m", "kairyu-auto", "kairyu-auto-max"} <= ids

        for model in ("kairyu-auto", "kairyu-auto-max"):
            response = await client.post("/v1/chat/completions", json=_chat_body("hi", model=model))
            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"]


async def test_public_models_hide_internal_engines_but_keep_them_in_readiness(
    tmp_path,
    monkeypatch,
):
    class _InternalNotReady(MockBackend):
        def readiness(self) -> EngineReadiness:
            return EngineReadiness(False, "internal L1 is not ready")

    internal = _InternalNotReady()
    monkeypatch.setattr(
        builder_module,
        "create_backend",
        lambda *_args, **_kwargs: internal,
    )
    (tmp_path / "product.yaml").write_text(
        "workers:\n  - {name: tier1, engine_ref: internal}\n",
        encoding="utf-8",
    )
    app = build_app_from_spec(
        load_deployment_spec(
            """
engines:
  internal: {backend: mock}
orchestrators:
  product: {spec: product.yaml}
public_models: [product]
"""
        ),
        base_dir=tmp_path,
    )

    async with _client(app) as client:
        models = await client.get("/v1/models")
        routing = await client.get("/routing")
        ready = await client.get("/readyz")
        chat = await client.post(
            "/v1/chat/completions",
            json=_chat_body("hidden", model="internal"),
        )
        completions = await client.post(
            "/v1/completions",
            json={"model": "internal", "prompt": "hidden"},
        )
        responses = await client.post(
            "/v1/responses",
            json={"model": "internal", "input": "hidden"},
        )

    assert {item["id"] for item in models.json()["data"]} == {"product"}
    assert set(routing.json()["models"]) == {"product"}
    assert ready.status_code == 503
    assert chat.status_code == completions.status_code == responses.status_code == 404


async def test_public_orchestrator_serves_responses_with_all_engines_hidden(
    tmp_path,
    monkeypatch,
):
    # Issue #530: /v1/models advertised the orchestrator while /v1/responses
    # resolved models against the (empty) public engine registry and 404ed.
    monkeypatch.setattr(
        builder_module,
        "create_backend",
        lambda *_args, **_kwargs: MockBackend(),
    )
    (tmp_path / "product.yaml").write_text(
        "workers:\n  - {name: tier1, engine_ref: internal}\n",
        encoding="utf-8",
    )
    app = build_app_from_spec(
        load_deployment_spec(
            """
engines:
  internal: {backend: mock}
orchestrators:
  product: {spec: product.yaml}
public_models: [product]
"""
        ),
        base_dir=tmp_path,
    )

    async with _client(app) as client:
        models = await client.get("/v1/models")
        response = await client.post(
            "/v1/responses",
            json={"model": "product", "input": "hello"},
        )

    assert {item["id"] for item in models.json()["data"]} == {"product"}
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["output"][0]["content"][0]["text"]


async def test_legacy_orchestrator_composes_with_named(tmp_path):
    (tmp_path / "auto.yaml").write_text(ORCHESTRATOR_SPEC, encoding="utf-8")
    (tmp_path / "auto_max.yaml").write_text(ORCHESTRATOR_SPEC, encoding="utf-8")
    (tmp_path / "deploy.yaml").write_text(
        """
engines:
  m: { backend: mock }
orchestrator: { spec: auto.yaml }
orchestrators:
  kairyu-auto-max: { spec: auto_max.yaml }
legacy_chat_models: [kairyu-auto, kairyu-auto-max]
""",
        encoding="utf-8",
    )
    app = build_app_from_config(tmp_path / "deploy.yaml")
    async with _client(app) as client:
        models = await client.get("/v1/models")
        ids = {m["id"] for m in models.json()["data"]}
    assert {"kairyu-auto", "kairyu-auto-max"} <= ids


def test_builder_wires_distinct_tenant_identities_and_limits(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_KEYS", "key-a,key-b")
    spec = load_deployment_spec(
        f"""
server:
  api_keys_env: KAIRYU_DEPLOYMENT_KEYS
  usage_ledger_path: {tmp_path / "usage.jsonl"}
engines:
  m: {{ backend: mock }}
tenants:
  default_tenant: fallback
  key_tenants:
    key-a: tenant-a
    key-b: tenant-b
  limits:
    tenant-a:
      requests_per_minute: 1
      tokens_per_minute: 1000
      request_burst: 1
      token_burst: 128
      max_in_flight: 2
    tenant-b: {{ requests_per_minute: 3, tokens_per_minute: 2000 }}
"""
    )
    settings_calls = 0
    api_key_resolutions = 0
    admin_key_resolutions = 0
    compaction_key_resolutions = 0
    real_to_server_settings = ServerSection.to_server_settings
    real_resolve_api_keys = ServerSettings.resolve_api_keys
    real_resolve_admin_keys = ServerSettings.resolve_admin_keys
    real_resolve_compaction_key = ServerSettings.resolve_responses_compaction_key

    def recording_server_settings(section):
        nonlocal settings_calls
        settings_calls += 1
        return real_to_server_settings(section)

    def recording_resolve_api_keys(settings):
        nonlocal api_key_resolutions
        api_key_resolutions += 1
        return real_resolve_api_keys(settings)

    def recording_resolve_admin_keys(settings):
        nonlocal admin_key_resolutions
        admin_key_resolutions += 1
        return real_resolve_admin_keys(settings)

    def recording_resolve_compaction_key(settings):
        nonlocal compaction_key_resolutions
        compaction_key_resolutions += 1
        return real_resolve_compaction_key(settings)

    monkeypatch.setattr(
        ServerSection,
        "to_server_settings",
        recording_server_settings,
    )
    monkeypatch.setattr(
        ServerSettings,
        "resolve_api_keys",
        recording_resolve_api_keys,
    )
    monkeypatch.setattr(
        ServerSettings,
        "resolve_admin_keys",
        recording_resolve_admin_keys,
    )
    monkeypatch.setattr(
        ServerSettings,
        "resolve_responses_compaction_key",
        recording_resolve_compaction_key,
    )

    app = build_app_from_spec(spec)

    config = app.state.tenant_limiter._config
    assert settings_calls == 1
    assert api_key_resolutions == 1
    assert admin_key_resolutions == 1
    assert compaction_key_resolutions == 1
    assert config.tenant_for_key("key-a") == "tenant-a"
    assert config.tenant_for_key("key-b") == "tenant-b"
    assert config.tenant_for_key("unmapped-key") == "fallback"
    assert config.limits_for("tenant-a") == TenantLimits(
        requests_per_minute=1,
        tokens_per_minute=1000,
        request_burst=1,
        token_burst=128,
        max_in_flight=2,
    )
    assert config.limits_for("tenant-b") == TenantLimits(
        requests_per_minute=3,
        tokens_per_minute=2000,
    )


def test_builder_injects_usage_sinks_into_batch_worker(tmp_path):
    spec = load_deployment_spec(
        f"""
server:
  usage_ledger_path: {tmp_path / "usage.jsonl"}
  ttft_slo_s: 1.0
engines:
  m: {{ backend: mock }}
tenants:
  default_tenant: tenant-a
batch:
  data_dir: {tmp_path / "batch-data"}
  max_concurrency: 1
"""
    )

    app = build_app_from_spec(spec)

    assert app.state.batch_worker._metrics is app.state.metrics
    assert app.state.batch_worker._usage_ledger is app.state.usage_ledger
    assert app.state.batch_worker._tenant_limiter is app.state.tenant_limiter
    assert app.state.batch_worker._admission_controller is app.state.slo_admission


def test_batch_store_config_keeps_filesystem_compatibility_and_shared_controls():
    with pytest.raises(
        ValueError,
        match="batch.data_dir is required for the filesystem store",
    ):
        load_deployment_spec(
            """
engines:
  m: { backend: mock }
batch: {}
"""
        )

    spec = load_deployment_spec(
        """
engines:
  m: { backend: mock }
batch:
  store: postgres
  dsn_env: TEST_BATCH_DSN
  store_id: fleet-a
  poll_interval_s: 0.1
  lease_seconds: 3
"""
    )

    assert spec.batch is not None
    assert spec.batch.store == "postgres"
    assert spec.batch.data_dir is None
    assert spec.batch.dsn_env == "TEST_BATCH_DSN"
    assert spec.batch.store_id == "fleet-a"
    assert spec.batch.poll_interval_s == 0.1
    assert spec.batch.lease_seconds == 3

    for invalid_batch, message in (
        (
            "store: postgres\n  data_dir: /ignored",
            "data_dir cannot be set for the postgres store",
        ),
        (
            "store: postgres\n  dsn_env: invalid-name",
            "dsn_env",
        ),
        (
            "store: postgres\n  store_id: '   '",
            "store_id must be a non-empty PostgreSQL identity",
        ),
        (
            "data_dir: /tmp/batch\n  spool_dir: /tmp/spool",
            "spool_dir is only valid for the postgres store",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            load_deployment_spec(
                f"""
engines:
  m: {{ backend: mock }}
batch:
  {invalid_batch}
"""
            )


def test_postgres_batch_dsn_fails_before_owned_backends_are_built(monkeypatch):
    monkeypatch.delenv("TEST_BATCH_DSN", raising=False)
    spec = load_deployment_spec(
        """
engines:
  m: { backend: mock }
batch:
  store: postgres
  dsn_env: TEST_BATCH_DSN
"""
    )

    def unexpected_backend(*args, **kwargs):
        raise AssertionError("backend construction must not run before batch preflight")

    monkeypatch.setattr(builder_module, "create_backend", unexpected_backend)
    with pytest.raises(
        ValueError,
        match="TEST_BATCH_DSN.*is not set",
    ):
        build_app_from_spec(spec)


def test_async_request_config_and_validation():
    spec = load_deployment_spec(
        """
engines:
  m: { backend: mock }
async_requests:
  dsn_env: TEST_ASYNC_DSN
  store_id: fleet-requests
  max_concurrency: 3
  max_body_bytes: 123456
  max_records_per_tenant: 4321
  poll_interval_s: 0.1
  lease_seconds: 5
  request_retention_s: 86400
  audit_retention_s: 604800
  retention_batch_size: 750
""",
        resolve_credentials=False,
    )

    assert spec.async_requests is not None
    assert spec.async_requests.dsn_env == "TEST_ASYNC_DSN"
    assert spec.async_requests.store_id == "fleet-requests"
    assert spec.async_requests.max_concurrency == 3
    assert spec.async_requests.max_body_bytes == 123456
    assert spec.async_requests.max_records_per_tenant == 4321
    assert spec.async_requests.poll_interval_s == 0.1
    assert spec.async_requests.lease_seconds == 5
    assert spec.async_requests.request_retention_s == 86400
    assert spec.async_requests.audit_retention_s == 604800
    assert spec.async_requests.retention_batch_size == 750

    for invalid, message in (
        ("dsn_env: invalid-name", "dsn_env"),
        ("store_id: '   '", "store_id must be a non-empty PostgreSQL identity"),
        ("unknown: value", "extra_forbidden"),
        ("request_retention_s: 0", "request_retention_s"),
        ("audit_retention_s: .nan", "audit_retention_s"),
        ("retention_batch_size: 10001", "retention_batch_size"),
    ):
        with pytest.raises(ValueError, match=message):
            load_deployment_spec(
                f"""
engines:
  m: {{ backend: mock }}
async_requests:
  {invalid}
""",
                resolve_credentials=False,
            )


def test_async_request_dsn_fails_before_owned_backends_are_built(monkeypatch):
    monkeypatch.delenv("TEST_ASYNC_DSN", raising=False)
    spec = load_deployment_spec(
        """
engines:
  m: { backend: mock }
async_requests:
  dsn_env: TEST_ASYNC_DSN
"""
    )

    def unexpected_backend(*args, **kwargs):
        raise AssertionError("backend construction must not run before request preflight")

    monkeypatch.setattr(builder_module, "create_backend", unexpected_backend)
    with pytest.raises(ValueError, match="TEST_ASYNC_DSN.*is not set"):
        build_app_from_spec(spec)


async def test_builder_wires_async_request_routes_worker_and_store_lifespan(
    monkeypatch,
):
    from kairyu.async_requests import postgres_store as request_store_module
    from kairyu.async_requests.store import InMemoryRequestStore

    stores = []

    class FakePostgresRequestStore(InMemoryRequestStore):
        def __init__(
            self,
            dsn: str,
            *,
            store_id: str,
            eager_connect: bool = True,
            max_records_per_owner: int = 64,
        ) -> None:
            super().__init__(
                store_id=store_id,
                max_records_per_owner=max_records_per_owner,
            )
            self.dsn = dsn
            self.eager_connect = eager_connect
            self.started = False
            self.closed = False
            stores.append(self)

        async def startup(self) -> None:
            self.started = True

        def close(self) -> None:
            self.closed = True

    monkeypatch.setenv("TEST_ASYNC_DSN", "postgresql://request-store")
    monkeypatch.setattr(
        request_store_module,
        "PostgresRequestStore",
        FakePostgresRequestStore,
    )
    spec = load_deployment_spec(
        """
engines:
  m: { backend: mock }
legacy_chat_models: [m]
async_requests:
  dsn_env: TEST_ASYNC_DSN
  store_id: fleet-requests
  max_concurrency: 1
  max_body_bytes: 256
  poll_interval_s: 0.05
  lease_seconds: 1
"""
    )
    app = build_app_from_spec(spec)

    assert app.state.async_request_store is stores[0]
    assert app.state.async_request_worker._store is stores[0]
    assert stores[0].eager_connect is False
    assert stores[0].started is False
    async with app.router.lifespan_context(app):
        assert stores[0].started is True
        async with _client(app) as client:
            submitted = await client.post(
                "/v1/async/chat/completions",
                json=_chat_body("durable", model="m"),
            )
            oversized = await client.post(
                "/v1/async/chat/completions",
                json=_chat_body("x" * 512, model="m"),
            )
            assert oversized.status_code == 413
            request_id = submitted.json()["request"]["id"]
            async with asyncio.timeout(1):
                while True:
                    status = await client.get(f"/v1/requests/{request_id}")
                    if status.json()["state"] == "succeeded":
                        break
                    await asyncio.sleep(0.01)

    assert stores[0].closed is True


async def test_tenant_auth_uses_the_preflight_key_snapshots(monkeypatch):
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_KEYS", "key-a,key-b")
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_ADMIN_KEYS", "admin-a,admin-b")
    spec = load_deployment_spec(
        """
server:
  api_keys_env: KAIRYU_DEPLOYMENT_KEYS
  admin_keys_env: KAIRYU_DEPLOYMENT_ADMIN_KEYS
engines:
  m: { backend: mock }
tenants:
  key_tenants:
    key-a: tenant-a
    key-b: tenant-b
"""
    )
    api_snapshots = iter(
        (
            frozenset({"key-a", "key-b"}),
            frozenset({"key-a", "key-c"}),
        )
    )
    admin_snapshots = iter(
        (
            frozenset({"admin-a", "admin-b"}),
            frozenset({"admin-a", "admin-c"}),
        )
    )
    api_key_resolutions = 0
    admin_key_resolutions = 0

    def resolve_api_keys(_settings):
        nonlocal api_key_resolutions
        api_key_resolutions += 1
        return next(api_snapshots)

    def resolve_admin_keys(_settings):
        nonlocal admin_key_resolutions
        admin_key_resolutions += 1
        return next(admin_snapshots)

    monkeypatch.setattr(ServerSettings, "resolve_api_keys", resolve_api_keys)
    monkeypatch.setattr(ServerSettings, "resolve_admin_keys", resolve_admin_keys)

    app = build_app_from_spec(spec)
    payload = _chat_body("hi", model="m")
    async with _client(app) as client:
        mapped_data = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer key-b"},
        )
        late_data = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer key-c"},
        )
        mapped_admin = await client.post(
            "/admin/undrain",
            headers={"Authorization": "Bearer admin-b"},
        )
        late_admin = await client.post(
            "/admin/undrain",
            headers={"Authorization": "Bearer admin-c"},
        )

    assert mapped_data.status_code == 200
    assert late_data.status_code == 401
    assert mapped_admin.status_code == 200
    assert late_admin.status_code == 401
    assert api_key_resolutions == 1
    assert admin_key_resolutions == 1


@pytest.mark.parametrize("late_failure", ["data", "admin"])
def test_builder_does_not_attempt_late_key_resolution(monkeypatch, late_failure):
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_KEYS", "key-a")
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_ADMIN_KEYS", "admin-a")
    spec = load_deployment_spec(
        """
server:
  api_keys_env: KAIRYU_DEPLOYMENT_KEYS
  admin_keys_env: KAIRYU_DEPLOYMENT_ADMIN_KEYS
engines:
  m: { backend: mock }
tenants:
  key_tenants:
    key-a: tenant-a
"""
    )
    resolution_counts = {"data": 0, "admin": 0}

    def resolve_api_keys(_settings):
        resolution_counts["data"] += 1
        if late_failure == "data" and resolution_counts["data"] > 1:
            raise ValueError("synthetic late data-key resolution failure")
        return frozenset({"key-a"})

    def resolve_admin_keys(_settings):
        resolution_counts["admin"] += 1
        if late_failure == "admin" and resolution_counts["admin"] > 1:
            raise ValueError("synthetic late admin-key resolution failure")
        return frozenset({"admin-a"})

    monkeypatch.setattr(ServerSettings, "resolve_api_keys", resolve_api_keys)
    monkeypatch.setattr(ServerSettings, "resolve_admin_keys", resolve_admin_keys)

    app = build_app_from_spec(spec)

    assert app.state.tenant_limiter is not None
    assert resolution_counts == {"data": 1, "admin": 1}


def test_builder_compaction_secret_failure_precedes_owned_backends(monkeypatch):
    secret_env = "KAIRYU_SHORT_RESPONSES_COMPACTION_SECRET"
    monkeypatch.setenv(secret_env, "too-short")
    spec = load_deployment_spec(
        f"""
server:
  responses_compaction_secret_env: {secret_env}
engines:
  m: {{ backend: mock }}
"""
    )
    created_backends = []
    real_create_backend = builder_module.create_backend

    def recording_create_backend(name, **options):
        backend = real_create_backend(name, **options)
        created_backends.append(backend)
        return backend

    monkeypatch.setattr(builder_module, "create_backend", recording_create_backend)

    with pytest.raises(ValueError, match="at least 32 UTF-8 bytes"):
        build_app_from_spec(spec)

    assert created_backends == []


@pytest.mark.parametrize("failing_key_set", ["data", "admin"])
def test_builder_key_resolution_failure_precedes_owned_backends(
    monkeypatch,
    failing_key_set,
):
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_KEYS", "key-a")
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_ADMIN_KEYS", "admin-a")
    spec = load_deployment_spec(
        """
server:
  api_keys_env: KAIRYU_DEPLOYMENT_KEYS
  admin_keys_env: KAIRYU_DEPLOYMENT_ADMIN_KEYS
engines:
  m: { backend: mock }
"""
    )
    created_backends = []
    real_create_backend = builder_module.create_backend

    def resolve_api_keys(_settings):
        if failing_key_set == "data":
            raise ValueError("synthetic data-key resolution failure")
        return frozenset({"key-a"})

    def resolve_admin_keys(_settings):
        if failing_key_set == "admin":
            raise ValueError("synthetic admin-key resolution failure")
        return frozenset({"admin-a"})

    def recording_create_backend(name, **options):
        backend = real_create_backend(name, **options)
        created_backends.append(backend)
        return backend

    monkeypatch.setattr(ServerSettings, "resolve_api_keys", resolve_api_keys)
    monkeypatch.setattr(ServerSettings, "resolve_admin_keys", resolve_admin_keys)
    monkeypatch.setattr(builder_module, "create_backend", recording_create_backend)

    with pytest.raises(ValueError, match=f"synthetic {failing_key_set}-key"):
        build_app_from_spec(spec)

    assert created_backends == []


async def test_deployment_tenants_isolate_rate_limits_and_usage_ledger(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_KEYS", "key-a,key-b")
    ledger_path = tmp_path / "usage.jsonl"
    app = build_app_from_spec(
        load_deployment_spec(
            f"""
server:
  api_keys_env: KAIRYU_DEPLOYMENT_KEYS
  usage_ledger_path: {ledger_path}
engines:
  m: {{ backend: mock }}
tenants:
  key_tenants:
    key-a: tenant-a
    key-b: tenant-b
  limits:
    tenant-a: {{ requests_per_minute: 1, tokens_per_minute: 200000 }}
    tenant-b: {{ requests_per_minute: 3, tokens_per_minute: 200000 }}
"""
        )
    )
    payload = _chat_body("hi", model="m")
    tenant_a = {"Authorization": "Bearer key-a"}
    tenant_b = {"Authorization": "Bearer key-b"}

    async with _client(app) as client:
        first_a = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers=tenant_a,
        )
        limited_a = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers=tenant_a,
        )
        first_b = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers=tenant_b,
        )
        usage_a = await client.get("/admin/usage", headers=tenant_a)
        usage_b = await client.get("/admin/usage", headers=tenant_b)
        metrics = (await client.get("/metrics")).text

    assert first_a.status_code == 200
    assert limited_a.status_code == 429
    assert limited_a.json()["error"]["code"] == "tenant_rate_limited"
    assert first_b.status_code == 200
    assert usage_a.status_code == 200
    assert set(usage_a.json()["usage"]) == {"tenant-a"}
    assert usage_a.json()["usage"]["tenant-a"]["requests"] == 1
    assert usage_b.status_code == 200
    assert set(usage_b.json()["usage"]) == {"tenant-b"}
    assert usage_b.json()["usage"]["tenant-b"]["requests"] == 1

    totals = UsageLedger(ledger_path).totals()
    assert set(totals) == {"tenant-a", "tenant-b"}
    assert totals["tenant-a"]["requests"] == 1
    assert totals["tenant-b"]["requests"] == 1
    assert "default" not in totals
    for tenant, usage in totals.items():
        assert (
            _metric_value(
                metrics,
                "kairyu_usage_requests_total",
                {"tenant": tenant},
            )
            == usage["requests"]
        )
        assert (
            _metric_value(
                metrics,
                "kairyu_usage_tokens_total",
                {"tenant": tenant, "type": "prompt"},
            )
            == usage["prompt_tokens"]
        )
        assert (
            _metric_value(
                metrics,
                "kairyu_usage_tokens_total",
                {"tenant": tenant, "type": "completion"},
            )
            == usage["completion_tokens"]
        )
        assert (
            _metric_value(
                metrics,
                "kairyu_usage_tokens_total",
                {"tenant": tenant, "type": "cached"},
            )
            == usage["cached_tokens"]
        )
        assert (
            _metric_value(
                metrics,
                "kairyu_usage_tokens_total",
                {"tenant": tenant, "type": "uncached"},
            )
            == usage["uncached_tokens"]
        )


async def test_deployment_usage_metrics_restore_after_truncated_tail_restart(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_KEYS", "key-a,key-b")
    ledger_path = tmp_path / "usage.jsonl"
    spec = load_deployment_spec(
        f"""
server:
  api_keys_env: KAIRYU_DEPLOYMENT_KEYS
  usage_ledger_path: {ledger_path}
engines:
  m: {{ backend: mock }}
tenants:
  key_tenants:
    key-a: tenant-a
    key-b: tenant-b
"""
    )
    payload = _chat_body("before restart", model="m")

    first_app = build_app_from_spec(spec)
    async with first_app.router.lifespan_context(first_app):
        async with _client(first_app) as client:
            response = await client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer key-a"},
            )
            assert response.status_code == 200
    first_handle = first_app.state.usage_ledger._handle
    assert first_handle is not None and first_handle.closed

    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write('{"tenant":"truncated"')

    restarted_app = build_app_from_spec(spec)
    async with restarted_app.router.lifespan_context(restarted_app):
        async with _client(restarted_app) as client:
            response = await client.post(
                "/v1/chat/completions",
                json=_chat_body("after restart", model="m"),
                headers={"Authorization": "Bearer key-b"},
            )
            metrics = (await client.get("/metrics")).text
        totals = restarted_app.state.usage_ledger.totals()

    assert response.status_code == 200
    restarted_handle = restarted_app.state.usage_ledger._handle
    assert restarted_handle is not None and restarted_handle.closed
    assert restarted_app.state.usage_ledger.malformed_lines == 1
    assert set(totals) == {"tenant-a", "tenant-b"}
    for tenant, usage in totals.items():
        assert (
            _metric_value(
                metrics,
                "kairyu_usage_requests_total",
                {"tenant": tenant},
            )
            == usage["requests"]
        )
        assert (
            _metric_value(
                metrics,
                "kairyu_usage_tokens_total",
                {"tenant": tenant, "type": "prompt"},
            )
            == usage["prompt_tokens"]
        )
        assert (
            _metric_value(
                metrics,
                "kairyu_usage_tokens_total",
                {"tenant": tenant, "type": "completion"},
            )
            == usage["completion_tokens"]
        )
        assert (
            _metric_value(
                metrics,
                "kairyu_usage_tokens_total",
                {"tenant": tenant, "type": "cached"},
            )
            == usage["cached_tokens"]
        )
        assert (
            _metric_value(
                metrics,
                "kairyu_usage_tokens_total",
                {"tenant": tenant, "type": "uncached"},
            )
            == usage["uncached_tokens"]
        )


def test_tenant_preflight_revalidates_before_constructing_owned_backends(monkeypatch):
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_KEYS", "key-a,key-b")
    spec = load_deployment_spec(
        """
server:
  api_keys_env: KAIRYU_DEPLOYMENT_KEYS
engines:
  m: { backend: mock }
tenants:
  key_tenants:
    key-a: tenant-a
    key-b: tenant-b
"""
    )
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_KEYS", "key-a")
    created_backends = []
    real_create_backend = builder_module.create_backend

    def recording_create_backend(name, **options):
        backend = real_create_backend(name, **options)
        created_backends.append(backend)
        return backend

    monkeypatch.setattr(builder_module, "create_backend", recording_create_backend)

    with pytest.raises(ValueError, match="unknown API key 'key-b'"):
        build_app_from_spec(spec)

    assert created_backends == []


def test_builder_without_tenant_section_preserves_legacy_app_state(monkeypatch):
    spec = load_deployment_spec(POOLED_YAML)
    resolution_counts = {"data": 0, "admin": 0}
    real_resolve_api_keys = ServerSettings.resolve_api_keys
    real_resolve_admin_keys = ServerSettings.resolve_admin_keys

    def recording_resolve_api_keys(settings):
        resolution_counts["data"] += 1
        return real_resolve_api_keys(settings)

    def recording_resolve_admin_keys(settings):
        resolution_counts["admin"] += 1
        return real_resolve_admin_keys(settings)

    monkeypatch.setattr(
        ServerSettings,
        "resolve_api_keys",
        recording_resolve_api_keys,
    )
    monkeypatch.setattr(
        ServerSettings,
        "resolve_admin_keys",
        recording_resolve_admin_keys,
    )

    app = build_app_from_spec(spec)

    assert not hasattr(app.state, "tenant_limiter")
    assert resolution_counts == {"data": 1, "admin": 1}


async def test_lifespan_attempts_orchestrator_shutdown_after_engine_failure(tmp_path, monkeypatch):
    class _Resource:
        def __init__(self, fail: bool = False) -> None:
            self.fail = fail
            self.shutdown_count = 0

        async def shutdown(self) -> None:
            self.shutdown_count += 1
            if self.fail:
                raise RuntimeError("shutdown failed")

    failing_engine = _Resource(fail=True)
    owned_backend = _ShutdownBackend()
    owned_orchestrator = Orchestrator(engines={"tier1": owned_backend, "tier2": owned_backend})
    (tmp_path / "auto.yaml").write_text(ORCHESTRATOR_SPEC, encoding="utf-8")
    monkeypatch.setattr(
        "kairyu.deploy.builder.create_backend", lambda *_args, **_kwargs: failing_engine
    )
    monkeypatch.setattr(
        "kairyu.deploy.builder.build_orchestrator",
        lambda _spec, **_kwargs: owned_orchestrator,
    )
    spec = load_deployment_spec(
        f"""
server:
  usage_ledger_path: {tmp_path / "usage.jsonl"}
engines:
  bad: {{ backend: mock }}
orchestrator: {{ spec: auto.yaml }}
legacy_chat_models: [kairyu-auto]
"""
    )
    app = build_app_from_spec(spec, base_dir=tmp_path)

    with pytest.raises(ExceptionGroup, match="application shutdown"):
        async with app.router.lifespan_context(app):
            app.state.usage_ledger.record("tenant-a", "bad", prompt_tokens=1, completion_tokens=2)
            app.state.usage_ledger.flush()
            ledger_handle = app.state.usage_ledger._handle

    assert failing_engine.shutdown_count == 1
    assert owned_backend.shutdown_count == 1
    assert ledger_handle is not None
    assert ledger_handle.closed
