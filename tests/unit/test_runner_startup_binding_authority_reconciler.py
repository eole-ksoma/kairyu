"""D3.18 leader-fenced authority CRD publisher coverage."""

from __future__ import annotations

import asyncio
import copy
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from kairyu.runners import (
    AUTHORITY_HOLDER_ID_ANNOTATION,
    AUTHORITY_SOURCE_DIGEST_ANNOTATION,
    AUTHORITY_SOURCE_REVISION_ANNOTATION,
    KubernetesAuthorityReconcileConflictError,
    KubernetesPlacementBindingAuthorityReconciler,
    KubernetesPlacementBindingLiveTarget,
    RunnerCachePlacementBindingInventory,
    RunnerCachePlacementInventoryPublication,
    RunnerScalingQuotaSnapshotPublication,
    RunnerWriterAuthority,
)
from kairyu.runners.prewarm import ModelCachePlacementCandidate
from kairyu.runners.scale_actuator import (
    SCALE_DECISION_GENERATION_ANNOTATION,
    SCALE_ELECTION_ID_ANNOTATION,
    SCALE_FENCING_TOKEN_ANNOTATION,
)
from kairyu.runners.scaling_quota import (
    KueueScalingAdmission,
    ScalingQuotaLimit,
    ScalingQuotaScope,
    ScalingQuotaSnapshot,
    kueue_scaling_workload_name,
)

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)


def _target() -> KubernetesPlacementBindingLiveTarget:
    return KubernetesPlacementBindingLiveTarget(
        model_class="qwen-14b",
        target_kind="Deployment",
        namespace="model-serving",
        name="qwen-runners",
        authority_namespace="scaling-system",
        kueue_namespace="model-serving",
        quota_snapshot_name="qwen-quota",
        inventory_name="qwen-inventory",
        pod_set_name="runners",
    )


def _authority(*, token: int = 1, holder: str = "controller-a") -> RunnerWriterAuthority:
    return RunnerWriterAuthority(
        election_id="runner-control-plane",
        holder_id=holder,
        fencing_token=token,
        validated_at=NOW,
        lease_until=NOW + timedelta(minutes=1),
    )


def _quota_publication(
    *,
    revision: int = 1,
    snapshot_id: str = "quota-a",
    observed_at: datetime = NOW,
) -> RunnerScalingQuotaSnapshotPublication:
    target_uid = "workload-uid"
    kueue = KueueScalingAdmission(
        api_version="kueue.x-k8s.io/v1beta2",
        namespace="model-serving",
        workload_name=kueue_scaling_workload_name(
            target_kind="Deployment",
            target_namespace="model-serving",
            target_name="qwen-runners",
            target_uid=target_uid,
        ),
        workload_uid="kueue-uid",
        workload_generation=3,
        resource_version=f"kueue-{revision}",
        target_kind="Deployment",
        target_namespace="model-serving",
        target_name="qwen-runners",
        target_uid=target_uid,
        local_queue="serving",
        cluster_queue="tenant-a-gpu",
        pod_set_name="runners",
        resource_flavor="h100-sxm",
        priority=100,
        admitted=True,
        admitted_pods=1,
        admitted_gpus=1,
    )
    return RunnerScalingQuotaSnapshotPublication(
        snapshot=ScalingQuotaSnapshot(
            snapshot_id=snapshot_id,
            quota_revision=revision,
            observed_at=observed_at,
            tenant_id="tenant-a",
            model_class="qwen-14b",
            model_family="qwen",
            gpus_per_replica=1,
            target_reserved_gpus=1,
            limits=tuple(
                ScalingQuotaLimit(
                    scope=scope,
                    quota_name=f"{scope.value}-quota",
                    hard_limit_gpus=10,
                    used_gpus_excluding_target=0,
                    reserved_gpus_for_higher_priority=0,
                )
                for scope in ScalingQuotaScope
            ),
            kueue=kueue,
        )
    )


def _inventory_publication(
    *,
    revision: int = 1,
    decision_generation: int = 1,
    decision_id: str = "decision-a",
    snapshot_id: str = "inventory-a",
) -> RunnerCachePlacementInventoryPublication:
    return RunnerCachePlacementInventoryPublication(
        model_class="qwen-14b",
        target_uid="workload-uid",
        binding_id="b" * 64,
        decision_id=decision_id,
        decision_generation=decision_generation,
        decision_fingerprint="d" * 64,
        model_id="qwen-model",
        model_revision="revision-a",
        manifest_digest="a" * 64,
        placement_binding_id="placement-binding-a",
        inventory=RunnerCachePlacementBindingInventory(
            snapshot_id=snapshot_id,
            cache_revision=revision,
            observed_at=NOW,
            candidates=(
                ModelCachePlacementCandidate(
                    placement_id="placement-a",
                    node_name="gpu-a",
                    resource_flavor="h100-sxm",
                    profile_id="h100-tp1",
                    compatibility_approval_id="approval-a",
                ),
            ),
        ),
    )


class _KubernetesAuthorityAPI:
    def __init__(self) -> None:
        self.resources: dict[str, dict[str, Any]] = {}
        self.requests: list[tuple[str, str, Any | None]] = []
        self.resource_version = 0
        self.conflict_method: str | None = None

    def _response(self, status: int, payload: object) -> httpx.Response:
        return httpx.Response(
            status,
            json=payload,
            headers={"content-type": "application/json"},
        )

    def _next_resource_version(self) -> str:
        self.resource_version += 1
        return str(self.resource_version)

    @staticmethod
    def _key(path: str) -> str:
        return path.removesuffix("/status")

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer service-account-token"
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        self.requests.append((request.method, path, copy.deepcopy(body)))
        if request.method == self.conflict_method:
            return self._response(409, {"kind": "Status", "reason": "Conflict"})

        key = self._key(path)
        if request.method == "GET":
            resource = self.resources.get(key)
            if resource is None:
                return self._response(404, {"kind": "Status", "reason": "NotFound"})
            return self._response(200, copy.deepcopy(resource))

        if request.method == "POST":
            assert isinstance(body, dict)
            name = body["metadata"]["name"]
            key = f"{path}/{name}"
            if key in self.resources:
                return self._response(409, {"kind": "Status", "reason": "AlreadyExists"})
            resource = copy.deepcopy(body)
            resource["metadata"].update(
                {
                    "uid": f"uid-{name}",
                    "generation": 1,
                    "resourceVersion": self._next_resource_version(),
                }
            )
            self.resources[key] = resource
            return self._response(201, copy.deepcopy(resource))

        resource = self.resources[key]
        if request.method == "PATCH":
            assert isinstance(body, list)
            previous_spec = copy.deepcopy(resource["spec"])
            for operation in body:
                if operation["op"] == "test":
                    if resource["metadata"]["resourceVersion"] != operation["value"]:
                        return self._response(409, {"kind": "Status", "reason": "Conflict"})
                elif operation["path"] == "/metadata/annotations":
                    resource["metadata"]["annotations"] = operation["value"]
                elif operation["path"].startswith("/metadata/annotations/"):
                    escaped = operation["path"].rsplit("/", 1)[1]
                    annotation = escaped.replace("~1", "/").replace("~0", "~")
                    resource["metadata"]["annotations"][annotation] = operation["value"]
                elif operation["path"] == "/spec":
                    resource["spec"] = operation["value"]
                else:
                    raise AssertionError(operation)
            if resource["spec"] != previous_spec:
                resource["metadata"]["generation"] += 1
            resource["metadata"]["resourceVersion"] = self._next_resource_version()
            return self._response(200, copy.deepcopy(resource))

        if request.method == "PUT" and path.endswith("/status"):
            assert isinstance(body, dict)
            if resource["metadata"]["resourceVersion"] != body["metadata"]["resourceVersion"]:
                return self._response(409, {"kind": "Status", "reason": "Conflict"})
            resource["status"] = body["status"]
            resource["metadata"]["resourceVersion"] = self._next_resource_version()
            return self._response(200, copy.deepcopy(resource))

        raise AssertionError((request.method, path))


def _reconciler(
    tmp_path: Path,
    api: _KubernetesAuthorityAPI,
    *,
    response_limit_bytes: int = 2 * 1024 * 1024,
) -> KubernetesPlacementBindingAuthorityReconciler:
    token = tmp_path / "token"
    token.write_text("service-account-token\n", encoding="ascii")

    async def handler(request: httpx.Request) -> httpx.Response:
        return api(request)

    return KubernetesPlacementBindingAuthorityReconciler(
        targets=(_target(),),
        api_server="https://kubernetes.test",
        token_path=token,
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            trust_env=False,
            follow_redirects=False,
        ),
        response_limit_bytes=response_limit_bytes,
    )


def _deadline() -> float:
    return time.monotonic() + 5


def _quota_key() -> str:
    return (
        "/apis/autoscaling.kairyu.ai/v1alpha1/namespaces/scaling-system/"
        "runnerscalingquotasnapshots/qwen-quota"
    )


def _inventory_key() -> str:
    return (
        "/apis/autoscaling.kairyu.ai/v1alpha1/namespaces/scaling-system/"
        "runnercacheplacementinventories/qwen-inventory"
    )


def test_quota_create_publishes_exact_reader_contract_and_fence(tmp_path: Path) -> None:
    api = _KubernetesAuthorityAPI()
    reconciler = _reconciler(tmp_path, api)
    authority = _authority()
    calls = 0

    def reauthorize() -> RunnerWriterAuthority:
        nonlocal calls
        calls += 1
        return authority

    try:
        result = reconciler.reconcile_quota_snapshot(
            _quota_publication(),
            authority,
            reauthorize=reauthorize,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
    finally:
        reconciler.close()

    assert result.spec_applied and result.status_applied
    assert calls == 2
    resource = api.resources[_quota_key()]
    assert resource["spec"] == {
        "targetRef": {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "namespace": "model-serving",
            "name": "qwen-runners",
            "uid": "workload-uid",
        },
        "tenantId": "tenant-a",
        "modelClass": "qwen-14b",
        "modelFamily": "qwen",
        "gpusPerReplica": 1,
        "kueueWorkloadRef": {
            "apiVersion": "kueue.x-k8s.io/v1beta2",
            "namespace": "model-serving",
            "name": kueue_scaling_workload_name(
                target_kind="Deployment",
                target_namespace="model-serving",
                target_name="qwen-runners",
                target_uid="workload-uid",
            ),
            "uid": "kueue-uid",
        },
    }
    assert resource["status"]["observedGeneration"] == resource["metadata"]["generation"]
    assert resource["status"]["conditions"] == [
        {"type": "Ready", "status": "True", "observedGeneration": 1}
    ]
    assert tuple(limit["scope"] for limit in resource["status"]["limits"]) == tuple(
        scope.value for scope in ScalingQuotaScope
    )
    annotations = resource["metadata"]["annotations"]
    assert annotations[SCALE_ELECTION_ID_ANNOTATION] == "runner-control-plane"
    assert annotations[AUTHORITY_HOLDER_ID_ANNOTATION] == "controller-a"
    assert annotations[SCALE_FENCING_TOKEN_ANNOTATION] == "1"
    assert annotations[AUTHORITY_SOURCE_REVISION_ANNOTATION] == "1"
    assert len(annotations[AUTHORITY_SOURCE_DIGEST_ANNOTATION]) == 64


def test_exact_quota_replay_is_read_only(tmp_path: Path) -> None:
    api = _KubernetesAuthorityAPI()
    reconciler = _reconciler(tmp_path, api)
    authority = _authority()
    try:
        reconciler.reconcile_quota_snapshot(
            _quota_publication(),
            authority,
            reauthorize=lambda: authority,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
        request_count = len(api.requests)
        replay_reauthorizations = 0

        def reauthorize_replay() -> RunnerWriterAuthority:
            nonlocal replay_reauthorizations
            replay_reauthorizations += 1
            return authority

        result = reconciler.reconcile_quota_snapshot(
            _quota_publication(),
            authority,
            reauthorize=reauthorize_replay,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
    finally:
        reconciler.close()

    assert not result.spec_applied and not result.status_applied
    assert replay_reauthorizations == 1
    assert api.requests[request_count:][0][0] == "GET"
    assert len(api.requests) == request_count + 1


def test_quota_revision_advances_status_without_rewriting_spec(tmp_path: Path) -> None:
    api = _KubernetesAuthorityAPI()
    reconciler = _reconciler(tmp_path, api)
    authority = _authority()
    try:
        reconciler.reconcile_quota_snapshot(
            _quota_publication(),
            authority,
            reauthorize=lambda: authority,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
        request_count = len(api.requests)
        result = reconciler.reconcile_quota_snapshot(
            _quota_publication(
                revision=2,
                snapshot_id="quota-b",
                observed_at=NOW + timedelta(seconds=1),
            ),
            authority,
            reauthorize=lambda: authority,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
    finally:
        reconciler.close()

    assert result.spec_applied and result.status_applied
    assert [item[0] for item in api.requests[request_count:]] == ["GET", "PATCH", "PUT"]
    assert api.resources[_quota_key()]["status"]["quotaRevision"] == 2


def test_empty_status_after_create_is_repaired_on_retry(tmp_path: Path) -> None:
    api = _KubernetesAuthorityAPI()
    reconciler = _reconciler(tmp_path, api)
    authority = _authority()
    try:
        reconciler.reconcile_quota_snapshot(
            _quota_publication(),
            authority,
            reauthorize=lambda: authority,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
        api.resources[_quota_key()]["status"] = {}
        result = reconciler.reconcile_quota_snapshot(
            _quota_publication(),
            authority,
            reauthorize=lambda: authority,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
    finally:
        reconciler.close()

    assert not result.spec_applied and result.status_applied
    assert api.resources[_quota_key()]["status"]["quotaRevision"] == 1


def test_empty_status_cannot_be_recovered_with_an_older_source_revision(tmp_path: Path) -> None:
    api = _KubernetesAuthorityAPI()
    reconciler = _reconciler(tmp_path, api)
    authority = _authority()
    try:
        reconciler.reconcile_quota_snapshot(
            _quota_publication(revision=10, snapshot_id="quota-10"),
            authority,
            reauthorize=lambda: authority,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
        api.resources[_quota_key()]["status"] = {}
        with pytest.raises(KubernetesAuthorityReconcileConflictError, match="regress"):
            reconciler.reconcile_quota_snapshot(
                _quota_publication(revision=1),
                authority,
                reauthorize=lambda: authority,
                deadline_monotonic=_deadline(),
                backend_timeout_s=2,
            )
    finally:
        reconciler.close()


@pytest.mark.parametrize(
    ("publication", "message"),
    [
        (_quota_publication(revision=1), "regress"),
        (_quota_publication(revision=2, snapshot_id="equivocation"), "reused"),
    ],
)
def test_quota_rejects_regression_and_same_revision_equivocation(
    tmp_path: Path,
    publication: RunnerScalingQuotaSnapshotPublication,
    message: str,
) -> None:
    api = _KubernetesAuthorityAPI()
    reconciler = _reconciler(tmp_path, api)
    authority = _authority()
    try:
        reconciler.reconcile_quota_snapshot(
            _quota_publication(revision=2, snapshot_id="quota-b"),
            authority,
            reauthorize=lambda: authority,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
        with pytest.raises(KubernetesAuthorityReconcileConflictError, match=message):
            reconciler.reconcile_quota_snapshot(
                publication,
                authority,
                reauthorize=lambda: authority,
                deadline_monotonic=_deadline(),
                backend_timeout_s=2,
            )
    finally:
        reconciler.close()


def test_status_revision_conflict_is_rejected_before_any_patch(tmp_path: Path) -> None:
    api = _KubernetesAuthorityAPI()
    reconciler = _reconciler(tmp_path, api)
    authority = _authority()
    try:
        reconciler.reconcile_quota_snapshot(
            _quota_publication(),
            authority,
            reauthorize=lambda: authority,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
        api.resources[_quota_key()]["status"]["quotaRevision"] = 10
        request_count = len(api.requests)
        with pytest.raises(KubernetesAuthorityReconcileConflictError, match="regress"):
            reconciler.reconcile_quota_snapshot(
                _quota_publication(revision=2, snapshot_id="quota-b"),
                authority,
                reauthorize=lambda: authority,
                deadline_monotonic=_deadline(),
                backend_timeout_s=2,
            )
    finally:
        reconciler.close()

    assert [item[0] for item in api.requests[request_count:]] == ["GET"]


def test_newer_leader_claims_exact_resource_and_stale_leader_is_denied(tmp_path: Path) -> None:
    api = _KubernetesAuthorityAPI()
    reconciler = _reconciler(tmp_path, api)
    first = _authority()
    successor = _authority(token=2, holder="controller-b")
    try:
        reconciler.reconcile_quota_snapshot(
            _quota_publication(),
            first,
            reauthorize=lambda: first,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
        successor_result = reconciler.reconcile_quota_snapshot(
            _quota_publication(),
            successor,
            reauthorize=lambda: successor,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
        with pytest.raises(KubernetesAuthorityReconcileConflictError, match="newer"):
            reconciler.reconcile_quota_snapshot(
                _quota_publication(),
                first,
                reauthorize=lambda: first,
                deadline_monotonic=_deadline(),
                backend_timeout_s=2,
            )
    finally:
        reconciler.close()

    assert successor_result.spec_applied and not successor_result.status_applied
    annotations = api.resources[_quota_key()]["metadata"]["annotations"]
    assert annotations[SCALE_FENCING_TOKEN_ANNOTATION] == "2"
    assert annotations[AUTHORITY_HOLDER_ID_ANNOTATION] == "controller-b"


def test_inventory_create_binds_decision_and_canonical_candidates(tmp_path: Path) -> None:
    api = _KubernetesAuthorityAPI()
    reconciler = _reconciler(tmp_path, api)
    authority = _authority()
    try:
        result = reconciler.reconcile_placement_inventory(
            _inventory_publication(),
            authority,
            reauthorize=lambda: authority,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
    finally:
        reconciler.close()

    assert result.spec_applied and result.status_applied
    resource = api.resources[_inventory_key()]
    assert resource["spec"] == {
        "targetRef": {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "namespace": "model-serving",
            "name": "qwen-runners",
            "uid": "workload-uid",
        },
        "bindingId": "b" * 64,
        "decisionId": "decision-a",
        "decisionFingerprint": "d" * 64,
        "modelClass": "qwen-14b",
        "modelId": "qwen-model",
        "modelRevision": "revision-a",
        "manifestDigest": "a" * 64,
        "placementBindingId": "placement-binding-a",
    }
    assert resource["status"]["candidates"] == [
        {
            "placementId": "placement-a",
            "nodeName": "gpu-a",
            "resourceFlavor": "h100-sxm",
            "profileId": "h100-tp1",
            "compatibilityApprovalId": "approval-a",
            "assigned": False,
            "healthy": True,
            "schedulable": True,
        }
    ]
    assert resource["metadata"]["annotations"][SCALE_DECISION_GENERATION_ANNOTATION] == "1"


def test_inventory_allows_new_decision_on_same_cache_revision(tmp_path: Path) -> None:
    api = _KubernetesAuthorityAPI()
    reconciler = _reconciler(tmp_path, api)
    authority = _authority()
    try:
        reconciler.reconcile_placement_inventory(
            _inventory_publication(),
            authority,
            reauthorize=lambda: authority,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
        result = reconciler.reconcile_placement_inventory(
            _inventory_publication(decision_generation=2, decision_id="decision-b"),
            authority,
            reauthorize=lambda: authority,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
    finally:
        reconciler.close()

    resource = api.resources[_inventory_key()]
    assert result.spec_applied and result.status_applied
    assert result.generation == 2
    assert resource["spec"]["decisionId"] == "decision-b"
    assert resource["status"]["observedGeneration"] == 2
    assert resource["metadata"]["annotations"][SCALE_DECISION_GENERATION_ANNOTATION] == "2"


def test_inventory_rejects_spec_change_without_new_decision_generation(tmp_path: Path) -> None:
    api = _KubernetesAuthorityAPI()
    reconciler = _reconciler(tmp_path, api)
    authority = _authority()
    try:
        reconciler.reconcile_placement_inventory(
            _inventory_publication(),
            authority,
            reauthorize=lambda: authority,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
        with pytest.raises(KubernetesAuthorityReconcileConflictError, match="did not advance"):
            reconciler.reconcile_placement_inventory(
                _inventory_publication(decision_id="decision-b"),
                authority,
                reauthorize=lambda: authority,
                deadline_monotonic=_deadline(),
                backend_timeout_s=2,
            )
    finally:
        reconciler.close()


def test_inventory_rejects_annotation_rollback_for_an_exact_spec(tmp_path: Path) -> None:
    api = _KubernetesAuthorityAPI()
    reconciler = _reconciler(tmp_path, api)
    authority = _authority()
    try:
        reconciler.reconcile_placement_inventory(
            _inventory_publication(decision_generation=2, decision_id="decision-b"),
            authority,
            reauthorize=lambda: authority,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
        with pytest.raises(KubernetesAuthorityReconcileConflictError, match="did not advance"):
            reconciler.reconcile_placement_inventory(
                _inventory_publication(decision_generation=1, decision_id="decision-b"),
                authority,
                reauthorize=lambda: authority,
                deadline_monotonic=_deadline(),
                backend_timeout_s=2,
            )
    finally:
        reconciler.close()


def test_inventory_rejects_missing_durable_decision_generation(tmp_path: Path) -> None:
    api = _KubernetesAuthorityAPI()
    reconciler = _reconciler(tmp_path, api)
    authority = _authority()
    try:
        reconciler.reconcile_placement_inventory(
            _inventory_publication(),
            authority,
            reauthorize=lambda: authority,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
        del api.resources[_inventory_key()]["metadata"]["annotations"][
            SCALE_DECISION_GENERATION_ANNOTATION
        ]
        with pytest.raises(KubernetesAuthorityReconcileConflictError, match="lacks"):
            reconciler.reconcile_placement_inventory(
                _inventory_publication(),
                authority,
                reauthorize=lambda: authority,
                deadline_monotonic=_deadline(),
                backend_timeout_s=2,
            )
    finally:
        reconciler.close()


def test_reauthorization_change_aborts_before_create(tmp_path: Path) -> None:
    api = _KubernetesAuthorityAPI()
    reconciler = _reconciler(tmp_path, api)
    authority = _authority()
    try:
        with pytest.raises(KubernetesAuthorityReconcileConflictError, match="changed"):
            reconciler.reconcile_quota_snapshot(
                _quota_publication(),
                authority,
                reauthorize=lambda: _authority(token=2, holder="controller-b"),
                deadline_monotonic=_deadline(),
                backend_timeout_s=2,
            )
    finally:
        reconciler.close()

    assert [request[0] for request in api.requests] == ["GET"]


def test_exact_replay_denies_authority_lost_after_read(tmp_path: Path) -> None:
    api = _KubernetesAuthorityAPI()
    reconciler = _reconciler(tmp_path, api)
    authority = _authority()
    try:
        reconciler.reconcile_quota_snapshot(
            _quota_publication(),
            authority,
            reauthorize=lambda: authority,
            deadline_monotonic=_deadline(),
            backend_timeout_s=2,
        )
        with pytest.raises(KubernetesAuthorityReconcileConflictError, match="changed"):
            reconciler.reconcile_quota_snapshot(
                _quota_publication(),
                authority,
                reauthorize=lambda: _authority(token=2, holder="controller-b"),
                deadline_monotonic=_deadline(),
                backend_timeout_s=2,
            )
    finally:
        reconciler.close()


@pytest.mark.parametrize("method", ["POST", "PATCH", "PUT"])
def test_kubernetes_conflicts_are_retryable_and_never_hidden(
    tmp_path: Path,
    method: str,
) -> None:
    api = _KubernetesAuthorityAPI()
    reconciler = _reconciler(tmp_path, api)
    authority = _authority()
    try:
        if method in {"PATCH", "PUT"}:
            reconciler.reconcile_quota_snapshot(
                _quota_publication(),
                authority,
                reauthorize=lambda: authority,
                deadline_monotonic=_deadline(),
                backend_timeout_s=2,
            )
        api.conflict_method = method
        publication = (
            _quota_publication()
            if method == "POST"
            else _quota_publication(
                revision=2,
                snapshot_id="quota-b",
                observed_at=NOW + timedelta(seconds=1),
            )
        )
        if method == "PATCH":
            authority = _authority(token=2, holder="controller-b")
        with pytest.raises(KubernetesAuthorityReconcileConflictError, match="concurrently"):
            reconciler.reconcile_quota_snapshot(
                publication,
                authority,
                reauthorize=lambda: authority,
                deadline_monotonic=_deadline(),
                backend_timeout_s=2,
            )
    finally:
        reconciler.close()


def test_reconciler_rejects_untrusted_target_mapping_and_expired_budget(tmp_path: Path) -> None:
    api = _KubernetesAuthorityAPI()
    reconciler = _reconciler(tmp_path, api)
    authority = _authority()
    unknown = _quota_publication().model_copy(
        update={
            "snapshot": _quota_publication().snapshot.model_copy(
                update={"model_class": "unknown-model"}
            )
        }
    )
    try:
        with pytest.raises(ValueError, match="trusted"):
            reconciler.reconcile_quota_snapshot(
                unknown,
                authority,
                reauthorize=lambda: authority,
                deadline_monotonic=_deadline(),
                backend_timeout_s=2,
            )
        with pytest.raises(TimeoutError, match="deadline"):
            reconciler.reconcile_quota_snapshot(
                _quota_publication(),
                authority,
                reauthorize=lambda: authority,
                deadline_monotonic=time.monotonic() - 1,
                backend_timeout_s=2,
            )
    finally:
        reconciler.close()

    assert api.requests == []


def test_whole_request_deadline_cancels_slow_kubernetes_response(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text("service-account-token\n", encoding="ascii")
    cancelled = False

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal cancelled
        try:
            await asyncio.sleep(10)
        finally:
            cancelled = True
        raise AssertionError("cancelled Kubernetes request resumed")

    reconciler = KubernetesPlacementBindingAuthorityReconciler(
        targets=(_target(),),
        api_server="https://kubernetes.test",
        token_path=token,
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            trust_env=False,
            follow_redirects=False,
        ),
    )
    authority = _authority()
    started_at = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            reconciler.reconcile_quota_snapshot(
                _quota_publication(),
                authority,
                reauthorize=lambda: authority,
                deadline_monotonic=time.monotonic() + 0.04,
                backend_timeout_s=1,
            )
    finally:
        reconciler.close()

    assert time.monotonic() - started_at < 0.5
    assert cancelled


def test_constructor_closes_adopted_client_when_loop_thread_start_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = tmp_path / "token"
    token.write_text("service-account-token\n", encoding="ascii")
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
        trust_env=False,
        follow_redirects=False,
    )

    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="thread unavailable"):
        KubernetesPlacementBindingAuthorityReconciler(
            targets=(_target(),),
            api_server="https://kubernetes.test",
            token_path=token,
            client=client,
        )

    assert client.is_closed
