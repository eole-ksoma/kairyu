"""`kairyu serve` CLI (design m7 D3)."""

import logging
from types import SimpleNamespace

import pytest

from kairyu.entrypoints import cli

DEPLOY_YAML = """
server:
  host: 127.0.0.1
  port: 8123
engines:
  m: { backend: mock }
"""


@pytest.fixture()
def config(tmp_path):
    path = tmp_path / "deploy.yaml"
    path.write_text(DEPLOY_YAML, encoding="utf-8")
    return path


def test_serve_runs_uvicorn_with_spec_address(monkeypatch, config):
    captured = {}

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    cli.main(["serve", str(config)])
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["app"].title == "kairyu"
    assert captured["access_log"] is False
    assert captured["loop"] == ("uvloop" if cli.sys.platform == "linux" else "auto")
    assert captured["http"] == ("httptools" if cli.sys.platform == "linux" else "auto")
    assert logging.getLogger("httpx").level == logging.WARNING


def test_serve_flags_override_spec(monkeypatch, config):
    captured = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: captured.update(kw))
    cli.main(["serve", str(config), "--host", "0.0.0.0", "--port", "9000"])
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9000


def test_generation_config_flag_overrides_every_static_native_backend(
    monkeypatch,
    tmp_path,
):
    from kairyu.engine.mock import MockBackend

    config = tmp_path / "native.yaml"
    config.write_text(
        """
engines:
  local:
    backend: kairyu
    options: {generation_config: none}
  unchanged: {backend: mock}
pools:
  native-pool:
    replicas:
      - backend: kairyu-proc
        options: {generation_config: auto}
      - backend: mock
legacy_chat_models: [local, native-pool]
""",
        encoding="utf-8",
    )
    captured = []

    def fake_create_backend(backend, **options):
        captured.append((backend, options))
        return MockBackend()

    monkeypatch.setattr(
        "kairyu.deploy.builder.create_backend",
        fake_create_backend,
    )
    monkeypatch.setattr("uvicorn.run", lambda app, **_kwargs: None)

    cli.main(["serve", str(config), "--generation-config", "vllm"])

    assert captured == [
        ("kairyu", {"generation_config": "vllm"}),
        ("mock", {}),
        ("kairyu-proc", {"generation_config": "vllm"}),
        ("mock", {}),
    ]


def test_serve_without_generation_config_flag_preserves_yaml_mode(
    monkeypatch,
    tmp_path,
):
    config = tmp_path / "native.yaml"
    config.write_text(
        """
engines:
  local:
    backend: kairyu
    options: {generation_config: none}
""",
        encoding="utf-8",
    )
    captured = {}

    def fake_build(spec, **kwargs):
        captured["spec"] = spec
        captured.update(kwargs)
        return spec

    monkeypatch.setattr(
        "kairyu.deploy.builder.build_app_from_spec",
        fake_build,
    )
    monkeypatch.setattr("uvicorn.run", lambda app, **_kwargs: None)

    cli.main(["serve", str(config)])

    assert captured["spec"].engines["local"].options["generation_config"] == "none"
    assert captured["generation_config_override"] is None


def test_generation_config_flag_rejects_a_remote_only_deployment(
    monkeypatch,
    config,
):
    def fail_if_constructed(*_args, **_kwargs):
        raise AssertionError("remote-only override must fail before backend construction")

    monkeypatch.setattr(
        "kairyu.deploy.builder.create_backend",
        fail_if_constructed,
    )
    monkeypatch.setattr("uvicorn.run", lambda app, **_kwargs: None)

    with pytest.raises(ValueError, match="local kairyu"):
        cli.main(["serve", str(config), "--generation-config", "auto"])


def test_generation_config_flag_reaches_local_orchestrator_workers(
    monkeypatch,
    tmp_path,
):
    from kairyu.engine.mock import MockBackend

    orchestrator = tmp_path / "orchestrator.yaml"
    orchestrator.write_text(
        """
workers:
  - name: native
    backend: kairyu-proc
    options: {generation_config: none}
""",
        encoding="utf-8",
    )
    config = tmp_path / "deploy.yaml"
    config.write_text(
        """
engines:
  remote: {backend: mock}
orchestrator: {spec: orchestrator.yaml}
legacy_chat_models: [kairyu-auto]
""",
        encoding="utf-8",
    )
    captured = {}

    def fake_create_backend(backend, **options):
        captured["backend"] = backend
        captured["options"] = options
        return MockBackend()

    monkeypatch.setattr(
        "kairyu.dsl.loader.create_backend",
        fake_create_backend,
    )
    monkeypatch.setattr("uvicorn.run", lambda app, **_kwargs: None)

    cli.main(["serve", str(config), "--generation-config", "vllm"])

    assert captured == {
        "backend": "kairyu-proc",
        "options": {"generation_config": "vllm"},
    }


def test_command_is_required():
    with pytest.raises(SystemExit):
        cli.main([])


def test_cache_agent_serve_uses_runtime_address_and_closes(monkeypatch, tmp_path):
    config_path = tmp_path / "cache-agent.json"
    config_path.write_text("{}", encoding="utf-8")
    config = SimpleNamespace(listen_host="127.0.0.1", listen_port=8181)
    runtime = SimpleNamespace(app=object(), close_calls=0)

    def close():
        runtime.close_calls += 1

    runtime.close = close
    captured = {}
    monkeypatch.setattr(
        "kairyu.runners.cache_agent_runtime.load_node_model_cache_agent_runtime_config",
        lambda path: config,
    )
    monkeypatch.setattr(
        "kairyu.runners.cache_agent_runtime.build_node_model_cache_agent_runtime",
        lambda value: runtime,
    )
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: captured.update(app=app, **kwargs))

    cli.main(["cache-agent", "serve", str(config_path)])

    assert captured["app"] is runtime.app
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8181
    assert captured["access_log"] is False
    assert runtime.close_calls == 1


def test_placement_admission_serve_uses_tls_runtime_and_closes(monkeypatch, tmp_path):
    config_path = tmp_path / "placement-admission.json"
    config_path.write_text("{}", encoding="utf-8")
    config = SimpleNamespace(
        listen_host="127.0.0.1",
        listen_port=8443,
        tls_cert_file=tmp_path / "tls.crt",
        tls_key_file=tmp_path / "tls.key",
    )
    runtime = SimpleNamespace(app=object(), close_calls=0)

    def close():
        runtime.close_calls += 1

    runtime.close = close
    captured = {}
    monkeypatch.setattr(
        "kairyu.runners.startup_admission_runtime."
        "load_runner_cache_placement_admission_runtime_config",
        lambda path: config,
    )
    monkeypatch.setattr(
        "kairyu.runners.startup_admission_runtime.build_runner_cache_placement_admission_runtime",
        lambda value: runtime,
    )
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: captured.update(app=app, **kwargs))

    cli.main(["placement-admission", "serve", str(config_path)])

    assert captured["app"] is runtime.app
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8443
    assert captured["ssl_certfile"] == str(config.tls_cert_file)
    assert captured["ssl_keyfile"] == str(config.tls_key_file)
    assert captured["access_log"] is False
    assert captured["workers"] == 1
    assert runtime.close_calls == 1
