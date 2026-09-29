"""Authenticated node model cache-agent HTTP transport coverage."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import psycopg
import pytest

from kairyu.artifacts import (
    InvalidModelArtifactError,
    ModelArtifactAdmissionRequest,
    ModelArtifactBlob,
    ModelArtifactManifest,
    ModelArtifactQuantization,
    ModelArtifactResourceEstimate,
    ModelArtifactTokenizer,
    ModelArtifactTrustStore,
    NodeModelCacheError,
    NodeModelCacheFillResult,
    NodeModelCacheIndex,
    SignedModelArtifactManifest,
    TrustedModelSigner,
    model_file_tree_digest,
    model_manifest_digest,
)
from kairyu.runners import (
    InMemoryNodeModelPrestageStore,
    LocalNodeModelCacheLiveEvidenceSource,
    ModelCachePlacement,
    ModelCachePlacementState,
    NodeModelCacheLiveEvidenceRequest,
    NodeModelCacheLiveEvidenceResponse,
    NodeModelPrestageCapacityError,
    NodeModelPrestageConflictError,
    NodeModelPrestageRecord,
    RunnerCacheStartupPlacement,
    RunnerWriterAuthority,
    ScalingPrewarmSnapshot,
    build_node_model_prestage_commands,
    build_node_model_prestage_release_command,
    create_node_model_cache_agent_app,
    plan_cache_aware_scale_up,
)

_NOW = datetime(2026, 9, 29, 6, 0, tzinfo=UTC)
_TOKEN = "transport-test-key-0000000000000000000000"


def _artifact() -> tuple[
    SignedModelArtifactManifest,
    ModelArtifactTrustStore,
    ModelArtifactAdmissionRequest,
]:
    content = b"Apache-2.0\n"
    blob = ModelArtifactBlob(
        path="LICENSE",
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    manifest = ModelArtifactManifest(
        model_id="org/cache-api-test",
        model_revision="release-1",
        upstream_repository="https://example.invalid/org/cache-api-test",
        upstream_revision="1" * 40,
        architecture="CacheApiTestForCausalLM",
        quantization=ModelArtifactQuantization(method="none", format="safetensors"),
        tokenizer=ModelArtifactTokenizer(
            repository="https://example.invalid/org/cache-api-test",
            revision="2" * 40,
            sha256="3" * 64,
        ),
        license_id="Apache-2.0",
        license_files=("LICENSE",),
        required_gpu_profiles=("h100-sxm",),
        approved_environments=("production",),
        signer_key_id="release-key",
        resources=ModelArtifactResourceEstimate(
            disk_bytes=len(content), ram_bytes=1024, vram_bytes=2048
        ),
        files=(blob,),
        file_tree_sha256=model_file_tree_digest((blob,)),
    )
    digest = model_manifest_digest(manifest)
    signed = SignedModelArtifactManifest(
        manifest=manifest,
        manifest_digest=digest,
        signature_base64=base64.b64encode(bytes(64)).decode("ascii"),
    )
    trust = ModelArtifactTrustStore(
        signers=(
            TrustedModelSigner(
                key_id="release-key",
                public_key_base64=base64.b64encode(bytes(32)).decode("ascii"),
                approved_environments=("production",),
            ),
        )
    )
    request = ModelArtifactAdmissionRequest(
        deployment_id="production/cache-api-test",
        manifest_digest=digest,
        model_id=manifest.model_id,
        model_revision=manifest.model_revision,
        environment="production",
        gpu_profile="h100-sxm",
    )
    return signed, trust, request


def _authority(*, token: int = 4) -> RunnerWriterAuthority:
    return RunnerWriterAuthority(
        election_id="runner-controller/production",
        holder_id="controller-a",
        fencing_token=token,
        validated_at=_NOW - timedelta(seconds=1),
        lease_until=_NOW + timedelta(minutes=2),
    )


def _command(
    digest: str,
    *,
    node_id: str = "gpu-node-00",
    placement_id: str = "placement-00",
    generation: int = 20,
):
    snapshot = ScalingPrewarmSnapshot(
        snapshot_id="cache-snapshot-api",
        cache_revision=10,
        observed_at=_NOW,
        model_class="interactive-h100",
        model_revision="release-1",
        artifact_digest=digest,
        placement_binding_id="binding-prestage-h100",
        placements=(
            ModelCachePlacement(
                placement_id=placement_id,
                node_name=node_id,
                resource_flavor="h100-sxm",
                profile_id="h100-sxm",
                compatibility_approval_id="compat-h100-v1",
                state=ModelCachePlacementState.ABSENT,
            ),
        ),
    )
    plan = plan_cache_aware_scale_up(
        snapshot,
        current_replicas=0,
        quota_target_replicas=1,
        resource_flavor="h100-sxm",
    )
    return build_node_model_prestage_commands(
        plan,
        authority=_authority(),
        decision_id="scale-decision-api",
        decision_fingerprint="d" * 64,
        target_id="statefulset/production/cache-api-test",
        target_revision=8,
        deployment_id="production/cache-api-test",
        model_id="org/cache-api-test",
        command_generations={placement_id: generation},
        issued_at=_NOW,
        ttl_seconds=60,
    )[0]


class FakeExecutor:
    def __init__(self, store: InMemoryNodeModelPrestageStore) -> None:
        self.node_id = store.node_id
        self.store = store
        self.trust_stores: list[ModelArtifactTrustStore] = []

    def execute(
        self,
        command,
        *,
        claim_id,
        envelope,
        trust_store,
        request,
    ) -> NodeModelPrestageRecord:
        del envelope, request
        self.trust_stores.append(trust_store)
        self.store.claim(command, claim_id=claim_id, now=_NOW + timedelta(seconds=1))
        return self.store.complete(
            command,
            claim_id=claim_id,
            fill_result=NodeModelCacheFillResult(
                deployment_id=command.deployment_id,
                manifest_digest=command.manifest_digest,
                artifact_path=Path("/cache/artifacts") / command.manifest_digest,
                cache_hit=False,
                resumed_bytes=0,
                downloaded_bytes=11,
                file_count=1,
                total_bytes=11,
            ),
            pin_record_generation=1,
            now=_NOW + timedelta(seconds=2),
        )

    def release(self, command) -> NodeModelPrestageRecord:
        return self.store.release(command, now=_NOW + timedelta(seconds=4))


def _app(
    *,
    ready: bool = True,
    body_limit: int = 1024 * 1024,
    active_limit: int = 2,
    total_limit: int = 8,
    queue_wait_timeout_s: float = 1.0,
):
    signed, trust, admission = _artifact()
    command = _command(signed.manifest_digest)
    store = InMemoryNodeModelPrestageStore(node_id="gpu-node-00")
    executor = FakeExecutor(store)

    def readiness_check() -> None:
        if not ready:
            raise RuntimeError("database unavailable and intentionally hidden")

    app = create_node_model_cache_agent_app(
        node_id="gpu-node-00",
        executor=executor,  # type: ignore[arg-type]
        store=store,
        trust_store=trust,
        api_keys=(_TOKEN,),
        readiness_check=readiness_check,
        request_body_limit_bytes=body_limit,
        active_request_limit=active_limit,
        total_request_limit=total_limit,
        queue_wait_timeout_s=queue_wait_timeout_s,
    )
    return app, executor, store, command, signed, trust, admission


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://cache-agent.test",
    )


def _live_evidence_setup(tmp_path: Path):
    signed, trust, _admission = _artifact()
    command = _command(signed.manifest_digest)
    store = InMemoryNodeModelPrestageStore(node_id="gpu-node-00")
    store.claim(command, claim_id="b" * 64, now=_NOW + timedelta(seconds=1))
    cache_root = tmp_path / "cache"
    artifact_path = cache_root / "artifacts" / signed.manifest_digest / "tree"
    artifact_path.mkdir(parents=True)
    index = NodeModelCacheIndex(
        cache_root / "state" / "index.sqlite3",
        node_id="gpu-node-00",
        cache_root=cache_root,
    )
    index.record_verified(
        manifest_digest=signed.manifest_digest,
        model_id="org/cache-api-test",
        model_revision="release-1",
        artifact_path=artifact_path,
        total_bytes=11,
        file_count=1,
        verification_source="filled",
    )
    pinned = index.pin(
        signed.manifest_digest,
        owner=command.pin_owner,
        reason=f"prestage:{command.command_id}",
    )
    store.complete(
        command,
        claim_id="b" * 64,
        fill_result=NodeModelCacheFillResult(
            deployment_id=command.deployment_id,
            manifest_digest=signed.manifest_digest,
            artifact_path=artifact_path,
            cache_hit=False,
            resumed_bytes=0,
            downloaded_bytes=11,
            file_count=1,
            total_bytes=11,
        ),
        pin_record_generation=pinned.generation,
        now=_NOW + timedelta(seconds=2),
    )
    placement = RunnerCacheStartupPlacement(
        placement_id=command.placement_id,
        node_name=command.node_id,
        resource_flavor=command.resource_flavor,
        profile_id=command.profile_id,
        compatibility_approval_id=command.compatibility_approval_id,
        manifest_digest=command.manifest_digest,
        pin_owner=command.pin_owner,
        prestage_command_id=command.command_id,
        prestage_command_generation=command.command_generation,
        hint_index_revision=1,
        resident_record_generation=pinned.generation,
        hint_observed_at=_NOW,
        hint_valid_until=_NOW + timedelta(minutes=1),
    )
    request = NodeModelCacheLiveEvidenceRequest(
        placement=placement,
        model_id=command.model_id,
        model_revision=command.model_revision,
    )
    source = LocalNodeModelCacheLiveEvidenceSource(
        node_id="gpu-node-00",
        store=store,
        index=index,
        hint_ttl_seconds=30,
        clock=lambda: _NOW + timedelta(seconds=3),
    )
    return source, request, store, index, trust


@pytest.mark.asyncio
async def test_health_is_open_but_state_requires_bearer_authentication() -> None:
    app, _executor, _store, _command_value, _signed, _trust, _admission = _app()
    async with _client(app) as client:
        health = await client.get("/health")
        unauthenticated = await client.get("/v1/prestage/records")
        wrong = await client.get("/v1/prestage/records", headers={"Authorization": "Bearer wrong"})
        authorized = await client.get(
            "/v1/prestage/records",
            headers={"Authorization": f"Bearer {_TOKEN}"},
        )

    assert health.json() == {"status": "ok", "node_id": "gpu-node-00"}
    assert unauthenticated.status_code == 401
    assert wrong.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json()["records"] == []


def test_local_live_evidence_is_owner_scoped_and_path_free(tmp_path: Path) -> None:
    source, request, _store, _index, _trust = _live_evidence_setup(tmp_path)

    response = source.read(request)
    payload = response.model_dump(mode="json")

    assert response.prestage_record.command.command_id == (request.placement.prestage_command_id)
    assert response.pin_evidence.pin_owners == (request.placement.pin_owner,)
    assert response.pin_evidence.index_revision == response.placement_hint.index_revision
    assert response.pin_evidence.observed_at == response.placement_hint.observed_at
    assert response.placement_hint.residents[0].record_generation == (
        request.placement.resident_record_generation
    )
    assert "artifact_path" not in str(payload)
    assert "fill_result" not in payload["prestage_record"]
    assert "failure" not in payload["prestage_record"]


def test_local_live_evidence_rejects_missing_requested_owner(tmp_path: Path) -> None:
    source, request, _store, index, _trust = _live_evidence_setup(tmp_path)
    index.unpin(request.placement.manifest_digest, owner=request.placement.pin_owner)
    index.pin(
        request.placement.manifest_digest,
        owner="prestage/production/cache-api-test/other-placement",
        reason="other-placement",
    )

    with pytest.raises(NodeModelPrestageConflictError, match="does not match"):
        source.read(request)


def test_local_live_evidence_rejects_prestage_change_during_index_snapshot(
    tmp_path: Path,
) -> None:
    _source, request, store, index, _trust = _live_evidence_setup(tmp_path)
    current = store.get_record(request.placement.placement_id)
    assert current is not None

    class ChangingStore(InMemoryNodeModelPrestageStore):
        def __init__(self) -> None:
            super().__init__(node_id="gpu-node-00")
            self.reads = 0

        def get_record(self, placement_id: str):
            assert placement_id == request.placement.placement_id
            self.reads += 1
            if self.reads == 1:
                return current
            return current.model_copy(update={"attempt": current.attempt + 1})

    source = LocalNodeModelCacheLiveEvidenceSource(
        node_id="gpu-node-00",
        store=ChangingStore(),
        index=index,
        clock=lambda: _NOW + timedelta(seconds=3),
    )

    with pytest.raises(NodeModelPrestageConflictError, match="changed during"):
        source.read(request)


@pytest.mark.asyncio
async def test_live_evidence_endpoint_is_authenticated_and_strict(tmp_path: Path) -> None:
    source, request, store, _index, trust = _live_evidence_setup(tmp_path)
    executor = FakeExecutor(store)
    app = create_node_model_cache_agent_app(
        node_id="gpu-node-00",
        executor=executor,  # type: ignore[arg-type]
        store=store,
        trust_store=trust,
        api_keys=(_TOKEN,),
        readiness_check=lambda: None,
        live_evidence_source=source,
        request_body_limit_bytes=1024 * 1024,
    )
    payload = request.model_dump(mode="json")
    headers = {"Authorization": f"Bearer {_TOKEN}"}
    async with _client(app) as client:
        unauthorized = await client.post("/v1/cache/live-evidence", json=payload)
        wrong_type = await client.post(
            "/v1/cache/live-evidence",
            content=b"{}",
            headers={**headers, "Content-Type": "text/plain"},
        )
        duplicate = await client.post(
            "/v1/cache/live-evidence",
            content=b'{"schema_version":"kairyu-node-model-cache-live-evidence-request-v1",'
            b'"schema_version":"kairyu-node-model-cache-live-evidence-request-v1"}',
            headers={**headers, "Content-Type": "application/json"},
        )
        authorized = await client.post(
            "/v1/cache/live-evidence",
            json=payload,
            headers=headers,
        )

    assert unauthorized.status_code == 401
    assert wrong_type.status_code == 415
    assert duplicate.status_code == 400
    assert authorized.status_code == 200
    assert authorized.headers["cache-control"] == "no-store"
    assert authorized.json()["pin_evidence"]["pin_owners"] == [request.placement.pin_owner]
    assert "/cache/" not in authorized.text


@pytest.mark.asyncio
async def test_live_evidence_endpoint_fails_closed_on_invalid_source_output(
    tmp_path: Path,
) -> None:
    source, request, store, _index, trust = _live_evidence_setup(tmp_path)
    valid = source.read(request)
    invalid = NodeModelCacheLiveEvidenceResponse.model_construct(
        schema_version=valid.schema_version,
        node_id="",
        prestage_record=valid.prestage_record,
        placement_hint=valid.placement_hint,
        pin_evidence=valid.pin_evidence,
    )

    class InvalidSource:
        def read(self, _request):
            return invalid

    app = create_node_model_cache_agent_app(
        node_id="gpu-node-00",
        executor=FakeExecutor(store),  # type: ignore[arg-type]
        store=store,
        trust_store=trust,
        api_keys=(_TOKEN,),
        readiness_check=lambda: None,
        live_evidence_source=InvalidSource(),  # type: ignore[arg-type]
    )
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://cache-agent.test",
    ) as client:
        response = await client.post(
            "/v1/cache/live-evidence",
            json=request.model_dump(mode="json"),
            headers={"Authorization": f"Bearer {_TOKEN}"},
        )

    assert response.status_code == 500
    assert "node_id" not in response.text
    assert valid.node_id not in response.text


@pytest.mark.asyncio
async def test_ensure_uses_server_trust_store_and_release_is_idempotent() -> None:
    app, executor, _store, command, signed, trust, admission = _app()
    headers = {"Authorization": f"Bearer {_TOKEN}"}
    ensure_payload = {
        "command": command.model_dump(mode="json"),
        "claim_id": "b" * 64,
        "manifest": signed.model_dump(mode="json"),
        "admission_request": admission.model_dump(mode="json"),
    }
    release = build_node_model_prestage_release_command(
        command,
        authority=_authority(token=5),
        decision_id="scale-decision-release",
        decision_fingerprint="e" * 64,
        target_revision=9,
        command_generation=21,
        issued_at=_NOW + timedelta(seconds=3),
        ttl_seconds=60,
    )
    async with _client(app) as client:
        ensured = await client.post("/v1/prestage/ensure", json=ensure_payload, headers=headers)
        records = await client.get("/v1/prestage/records", headers=headers)
        released = await client.post(
            "/v1/prestage/release",
            json={"command": release.model_dump(mode="json")},
            headers=headers,
        )
        replay = await client.post(
            "/v1/prestage/release",
            json={"command": release.model_dump(mode="json")},
            headers=headers,
        )

    assert ensured.status_code == 200
    assert ensured.json()["state"] == "ready"
    assert "fill_result" not in ensured.json()
    assert str(Path("/cache/artifacts")) not in ensured.text
    assert records.json()["records"][0]["state"] == "ready"
    assert "fill_result" not in records.json()["records"][0]
    assert "failure" not in records.json()["records"][0]
    assert released.json()["state"] == "absent"
    assert replay.json() == released.json()
    assert executor.trust_stores == [trust]


@pytest.mark.asyncio
async def test_other_node_command_is_rejected_before_executor() -> None:
    app, executor, _store, _command_value, signed, _trust, admission = _app()
    command = _command(signed.manifest_digest, node_id="gpu-ノード-01")
    async with _client(app) as client:
        response = await client.post(
            "/v1/prestage/ensure",
            json={
                "command": command.model_dump(mode="json"),
                "claim_id": "b" * 64,
                "manifest": signed.model_dump(mode="json"),
                "admission_request": admission.model_dump(mode="json"),
            },
            headers={"Authorization": f"Bearer {_TOKEN}"},
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "prestage_conflict"
    assert executor.trust_stores == []


@pytest.mark.asyncio
async def test_invalid_signature_error_is_sanitized_and_trust_root_is_not_wire_input() -> None:
    app, executor, _store, command, signed, trust, admission = _app()
    headers = {"Authorization": f"Bearer {_TOKEN}"}
    payload = {
        "command": command.model_dump(mode="json"),
        "claim_id": "b" * 64,
        "manifest": signed.model_dump(mode="json"),
        "admission_request": admission.model_dump(mode="json"),
    }

    def reject(*_args, **_kwargs):
        raise InvalidModelArtifactError("private signature verification detail")

    executor.execute = reject  # type: ignore[method-assign]
    async with _client(app) as client:
        rejected = await client.post("/v1/prestage/ensure", json=payload, headers=headers)
        injected = await client.post(
            "/v1/prestage/ensure",
            json={**payload, "trust_store": trust.model_dump(mode="json")},
            headers=headers,
        )

    assert rejected.status_code == 422
    assert rejected.json()["error"] == {
        "code": "artifact_not_admitted",
        "message": "model artifact admission failed",
    }
    assert "private" not in rejected.text
    assert injected.status_code == 422


@pytest.mark.asyncio
async def test_readiness_failure_and_body_limit_hide_internal_details() -> None:
    app, _executor, _store, _command_value, _signed, _trust, _admission = _app(
        ready=False,
        body_limit=128,
    )
    async with _client(app) as client:
        readiness = await client.get("/readyz")
        oversized = await client.post(
            "/v1/prestage/ensure",
            content=b"x" * 129,
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                "Content-Type": "application/json",
            },
        )

    assert readiness.status_code == 503
    assert readiness.json() == {"status": "not_ready", "node_id": "gpu-node-00"}
    assert "database" not in readiness.text
    assert oversized.status_code == 413


@pytest.mark.asyncio
async def test_chunked_body_limit_is_enforced_without_content_length() -> None:
    app, _executor, _store, _command_value, _signed, _trust, _admission = _app(body_limit=128)

    async def chunks():
        yield b"x" * 65
        yield b"y" * 65

    async with _client(app) as client:
        response = await client.post(
            "/v1/prestage/ensure",
            content=chunks(),
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                "Content-Type": "application/json",
            },
        )

    assert response.status_code == 413


@pytest.mark.asyncio
async def test_records_use_bounded_keyset_pagination_and_validate_cursor() -> None:
    app, _executor, store, command, _signed, _trust, _admission = _app()
    second = _command(
        command.manifest_digest,
        placement_id="placement-01",
        generation=21,
    )
    for value, claim in ((command, "a" * 64), (second, "b" * 64)):
        store.claim(value, claim_id=claim, now=_NOW + timedelta(seconds=1))
        store.fail(
            value,
            claim_id=claim,
            failure="private database and cache path detail",
            now=_NOW + timedelta(seconds=2),
        )
    headers = {"x-api-key": _TOKEN}

    async with _client(app) as client:
        first = await client.get("/v1/prestage/records?limit=1", headers=headers)
        cursor = first.json()["next_cursor"]
        second_page = await client.get(
            "/v1/prestage/records", params={"after": cursor, "limit": 1}, headers=headers
        )
        invalid = await client.get(
            "/v1/prestage/records", params={"after": "bad\x00cursor"}, headers=headers
        )

    assert first.status_code == 200
    assert [item["command"]["placement_id"] for item in first.json()["records"]] == ["placement-00"]
    assert cursor == "placement-00"
    assert "private" not in first.text
    assert [item["command"]["placement_id"] for item in second_page.json()["records"]] == [
        "placement-01"
    ]
    assert second_page.json()["next_cursor"] is None
    assert invalid.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (NodeModelPrestageConflictError("private"), 409, "prestage_conflict"),
        (NodeModelPrestageCapacityError("private"), 507, "prestage_capacity"),
        (NodeModelCacheError("private cache path"), 502, "cache_fill_failed"),
        (psycopg.OperationalError("private DSN"), 503, "backend_unavailable"),
    ],
)
async def test_backend_error_mapping_is_sanitized(error, status, code) -> None:
    app, executor, _store, command, signed, _trust, admission = _app()

    def reject(*_args, **_kwargs):
        raise error

    executor.execute = reject  # type: ignore[method-assign]
    async with _client(app) as client:
        response = await client.post(
            "/v1/prestage/ensure",
            json={
                "command": command.model_dump(mode="json"),
                "claim_id": "b" * 64,
                "manifest": signed.model_dump(mode="json"),
                "admission_request": admission.model_dump(mode="json"),
            },
            headers={"Authorization": f"Bearer {_TOKEN}"},
        )

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert "private" not in response.text
    if status == 503:
        assert response.headers["retry-after"] == "1"


@pytest.mark.asyncio
async def test_concurrency_gate_rejects_work_beyond_total_limit() -> None:
    app, executor, _store, command, signed, _trust, admission = _app(
        active_limit=1,
        total_limit=1,
        queue_wait_timeout_s=0.01,
    )
    entered = threading.Event()
    unblock = threading.Event()
    original_execute = executor.execute

    def blocked(*args, **kwargs):
        entered.set()
        assert unblock.wait(timeout=2)
        return original_execute(*args, **kwargs)

    executor.execute = blocked  # type: ignore[method-assign]
    payload = {
        "command": command.model_dump(mode="json"),
        "claim_id": "b" * 64,
        "manifest": signed.model_dump(mode="json"),
        "admission_request": admission.model_dump(mode="json"),
    }
    headers = {"Authorization": f"Bearer {_TOKEN}"}
    async with _client(app) as client:
        first = asyncio.create_task(
            client.post("/v1/prestage/ensure", json=payload, headers=headers)
        )
        assert await asyncio.to_thread(entered.wait, 1)
        overloaded = await client.post("/v1/prestage/ensure", json=payload, headers=headers)
        unblock.set()
        completed = await first

    assert overloaded.status_code == 429
    assert completed.status_code == 200


def test_app_configuration_is_closed_by_default() -> None:
    _signed, trust, _admission = _artifact()
    store = InMemoryNodeModelPrestageStore(node_id="gpu-node-00")
    executor = FakeExecutor(store)

    with pytest.raises(ValueError, match="at least one"):
        create_node_model_cache_agent_app(
            node_id="gpu-node-00",
            executor=executor,  # type: ignore[arg-type]
            store=store,
            trust_store=trust,
            api_keys=(),
            readiness_check=lambda: None,
        )

    with pytest.raises(ValueError, match="finite and positive"):
        create_node_model_cache_agent_app(
            node_id="gpu-node-00",
            executor=executor,  # type: ignore[arg-type]
            store=store,
            trust_store=trust,
            api_keys=(_TOKEN,),
            readiness_check=lambda: None,
            queue_wait_timeout_s=float("nan"),
        )

    executor.node_id = "gpu-node-99"
    with pytest.raises(ValueError, match="executor belongs to another node"):
        create_node_model_cache_agent_app(
            node_id="gpu-node-00",
            executor=executor,  # type: ignore[arg-type]
            store=store,
            trust_store=trust,
            api_keys=(_TOKEN,),
            readiness_check=lambda: None,
        )
