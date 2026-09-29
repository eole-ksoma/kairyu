from __future__ import annotations

import json
from datetime import UTC, datetime

from kairyu.async_requests import RequestRetentionBatchResult
from scripts import async_request_retention


def _config(tmp_path) -> str:
    path = tmp_path / "deployment.yaml"
    path.write_text(
        """
engines:
  m: {backend: mock}
async_requests:
  dsn_env: TEST_RETENTION_DSN
  store_id: retention-test
  request_retention_s: 3600
  audit_retention_s: 7200
  retention_batch_size: 3
""",
        encoding="utf-8",
    )
    return str(path)


class _FakeStore:
    results: list[RequestRetentionBatchResult] = []
    calls: list[dict] = []
    prepared = False
    failure: Exception | None = None
    initialize_schema_values: list[bool] = []

    def __init__(
        self,
        dsn: str,
        *,
        store_id: str,
        allow_store_creation: bool,
        initialize_schema: bool,
    ) -> None:
        assert dsn == "postgresql://user:secret@db/name"
        assert store_id == "retention-test"
        assert allow_store_creation is False
        type(self).initialize_schema_values.append(initialize_schema)

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def prepare_retention(self) -> None:
        type(self).prepared = True

    def purge_retained_data(self, **kwargs) -> RequestRetentionBatchResult:
        type(self).calls.append(kwargs)
        if type(self).failure is not None:
            raise type(self).failure
        return type(self).results.pop(0)


def _result(*, has_more: bool, applied: bool) -> RequestRetentionBatchResult:
    now = datetime(2026, 9, 11, tzinfo=UTC)
    return RequestRetentionBatchResult(
        request_cutoff=now,
        audit_cutoff=now,
        terminal_requests_deleted=1,
        audit_events_archived=2,
        audit_events_deleted=3,
        owner_deferrals_deleted=1,
        has_more=has_more,
        applied=applied,
    )


def _reset_fake() -> None:
    _FakeStore.results = []
    _FakeStore.calls = []
    _FakeStore.prepared = False
    _FakeStore.failure = None
    _FakeStore.initialize_schema_values = []


def test_retention_cli_is_read_only_without_apply(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    _reset_fake()
    _FakeStore.results = [_result(has_more=True, applied=False)]
    monkeypatch.setenv(
        "TEST_RETENTION_DSN",
        "postgresql://user:secret@db/name",
    )
    monkeypatch.setattr(async_request_retention, "PostgresRequestStore", _FakeStore)

    assert async_request_retention.main([_config(tmp_path), "--mode", "purge"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["applied"] is False
    assert payload["batches"] == 1
    assert payload["has_more"] is True
    assert _FakeStore.calls == [
        {
            "request_retention_seconds": 3600.0,
            "audit_retention_seconds": 7200.0,
            "batch_size": 3,
            "dry_run": True,
        }
    ]
    assert _FakeStore.initialize_schema_values == [False]


def test_retention_cli_applies_bounded_batches_and_prepare(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    _reset_fake()
    monkeypatch.setenv(
        "TEST_RETENTION_DSN",
        "postgresql://user:secret@db/name",
    )
    monkeypatch.setattr(async_request_retention, "PostgresRequestStore", _FakeStore)
    config = _config(tmp_path)

    assert async_request_retention.main(
        [config, "--mode", "prepare", "--apply"]
    ) == 0
    assert _FakeStore.prepared is True
    assert _FakeStore.initialize_schema_values == [False]
    capsys.readouterr()

    _FakeStore.results = [
        _result(has_more=True, applied=True),
        _result(has_more=False, applied=True),
    ]
    assert async_request_retention.main(
        [config, "--mode", "purge", "--apply", "--max-batches", "2"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["applied"] is True
    assert payload["batches"] == 2
    assert payload["terminal_requests_deleted"] == 2
    assert len(_FakeStore.calls) == 2
    assert _FakeStore.initialize_schema_values == [False, False]


def test_retention_cli_prepare_preview_and_disabled_policy_never_connect(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    _reset_fake()
    monkeypatch.delenv("TEST_RETENTION_DSN", raising=False)
    monkeypatch.setattr(async_request_retention, "PostgresRequestStore", _FakeStore)
    config = _config(tmp_path)

    assert async_request_retention.main([config, "--mode", "prepare"]) == 0
    assert _FakeStore.initialize_schema_values == []
    capsys.readouterr()

    disabled = tmp_path / "disabled.yaml"
    disabled.write_text(
        """
engines:
  m: {backend: mock}
async_requests:
  dsn_env: TEST_RETENTION_DSN
  store_id: retention-test
""",
        encoding="utf-8",
    )
    assert async_request_retention.main(
        [str(disabled), "--mode", "purge", "--apply"]
    ) == 0
    assert json.loads(capsys.readouterr().out)["batches"] == 0
    assert _FakeStore.initialize_schema_values == []


def test_retention_cli_redacts_runtime_errors(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    _reset_fake()
    _FakeStore.failure = RuntimeError(
        "postgresql://user:secret@db/name owner=tenant-a request=req-secret"
    )
    monkeypatch.setenv(
        "TEST_RETENTION_DSN",
        "postgresql://user:secret@db/name",
    )
    monkeypatch.setattr(async_request_retention, "PostgresRequestStore", _FakeStore)

    assert async_request_retention.main(
        [_config(tmp_path), "--mode", "purge", "--apply"]
    ) == 1

    output = capsys.readouterr().out
    assert "secret" not in output
    assert "tenant-a" not in output
    assert "req-secret" not in output
    assert json.loads(output)["error_type"] == "RuntimeError"
