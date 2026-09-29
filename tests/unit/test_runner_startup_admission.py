"""D3.4 incremental per-Pod cache placement admission."""

from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from kairyu.runners import (
    MODEL_ID_ANNOTATION,
    MODEL_REVISION_ANNOTATION,
    RELEASE_ID_ANNOTATION,
    RUNNER_CACHE_STARTUP_BINDING_ANNOTATION,
    RUNNER_CACHE_STARTUP_BINDING_ID_ANNOTATION,
    RUNNER_CACHE_STARTUP_BINDING_LABEL,
    RUNNER_CACHE_STARTUP_PLACEMENT_ANNOTATION,
    RUNNER_CACHE_STARTUP_SCHEDULING_GATE,
    RUNNER_CACHE_STARTUP_TARGET_ANNOTATION,
    InMemoryRunnerCachePlacementAdmissionStore,
    RunnerCachePlacementAdmissionConflictError,
    RunnerCachePlacementAdmissionController,
    RunnerCachePlacementAdmissionError,
    RunnerCachePlacementAdmissionPlan,
    RunnerCacheSchedulingError,
    RunnerCacheStartupBinding,
    RunnerCacheStartupPlacement,
    admit_runner_cache_placement_for_gated_pod,
    bind_runner_cache_to_pod_template,
)

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
DIGEST = "a" * 64
TARGET = "deployment/model-serving/qwen-runners"


def _binding(
    *,
    suffix: str = "a",
    nodes: tuple[str, ...] = ("gpu-a", "gpu-b"),
    bound_at: datetime = NOW,
) -> RunnerCacheStartupBinding:
    placements = tuple(
        RunnerCacheStartupPlacement(
            placement_id=f"placement-{index}-{suffix}",
            node_name=node,
            resource_flavor="h100-sxm",
            profile_id="h100-sxm-tp1",
            compatibility_approval_id="compat-qwen-h100",
            manifest_digest=DIGEST,
            pin_owner=f"prestage/model-serving/qwen/placement-{index}-{suffix}",
            prestage_command_id=hashlib.sha256(
                f"command-{index}-{suffix}".encode()
            ).hexdigest(),
            prestage_command_generation=index + 1,
            hint_index_revision=10 + index,
            resident_record_generation=20 + index,
            hint_observed_at=bound_at - timedelta(seconds=1),
            hint_valid_until=bound_at + timedelta(minutes=5),
        )
        for index, node in enumerate(nodes)
    )
    payload = {
        "schema_version": "runner-cache-startup-binding-v1",
        "decision_id": f"decision-{suffix}",
        "decision_fingerprint": hashlib.sha256(
            f"decision-{suffix}".encode()
        ).hexdigest(),
        "target_id": TARGET,
        "target_revision": 7,
        "deployment_id": "model-serving/qwen",
        "model_class": "qwen-14b",
        "model_id": "org/qwen",
        "model_revision": "revision-a",
        "manifest_digest": DIGEST,
        "placement_binding_id": f"placement-binding-{suffix}",
        "prewarm_snapshot_id": f"snapshot-{suffix}",
        "prewarm_cache_revision": 9,
        "bound_at": bound_at,
        "valid_until": bound_at + timedelta(minutes=5),
        "placements": placements,
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


def _pod(name: str = "qwen-7") -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "namespace": "model-serving",
            "name": name,
            "ownerReferences": [
                {
                    "apiVersion": "apps/v1",
                    "kind": "StatefulSet",
                    "name": "qwen-runners",
                    "uid": "workload-uid-a",
                    "controller": True,
                }
            ],
            "annotations": {
                RUNNER_CACHE_STARTUP_TARGET_ANNOTATION: TARGET,
                "example.com/owner": "platform",
            },
            "labels": {"app": "qwen"},
        },
        "spec": {
            "schedulingGates": [
                {"name": RUNNER_CACHE_STARTUP_SCHEDULING_GATE},
                {"name": "example.com/another-gate"},
            ],
            "affinity": {
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [
                            {
                                "matchExpressions": [
                                    {
                                        "key": "gpu.vendor",
                                        "operator": "In",
                                        "values": ["nvidia"],
                                    }
                                ]
                            },
                            {
                                "matchExpressions": [
                                    {"key": "zone", "operator": "In", "values": ["a"]}
                                ]
                            },
                        ]
                    }
                }
            },
            "containers": [{"name": "runner", "image": "runner@sha256:deadbeef"}],
        },
    }


def _plan(binding: RunnerCacheStartupBinding) -> RunnerCachePlacementAdmissionPlan:
    return RunnerCachePlacementAdmissionPlan(
        binding=binding,
        release_id="release-a",
        namespace="model-serving",
        owner_api_version="apps/v1",
        owner_kind="StatefulSet",
        owner_name="qwen-runners",
        owner_uid="workload-uid-a",
        creator_username="system:serviceaccount:kairyu:statefulset-controller",
        registered_at=binding.bound_at + timedelta(seconds=1),
    )


def test_create_admission_binds_exact_node_and_preserves_other_constraints() -> None:
    pod = _pod()
    original = copy.deepcopy(pod)
    binding = _binding()

    admitted = admit_runner_cache_placement_for_gated_pod(
        pod,
        binding,
        placement_id="placement-0-a",
        release_id="release-a",
    )

    assert pod == original
    assert admitted["spec"]["schedulingGates"] == [
        {"name": "example.com/another-gate"}
    ]
    terms = admitted["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"]
    assert all(
        term["matchFields"][-1]
        == {"key": "metadata.name", "operator": "In", "values": ["gpu-a"]}
        for term in terms
    )
    annotations = admitted["metadata"]["annotations"]
    assert annotations[RUNNER_CACHE_STARTUP_BINDING_ID_ANNOTATION] == binding.binding_id
    assert annotations[RUNNER_CACHE_STARTUP_PLACEMENT_ANNOTATION] == "placement-0-a"
    assert annotations[MODEL_ID_ANNOTATION] == "org/qwen"
    assert annotations[MODEL_REVISION_ANNOTATION] == "revision-a"
    assert annotations[RELEASE_ID_ANNOTATION] == "release-a"
    assert (
        RunnerCacheStartupBinding.model_validate_json(
            annotations[RUNNER_CACHE_STARTUP_BINDING_ANNOTATION]
        )
        == binding
    )
    assert (
        admit_runner_cache_placement_for_gated_pod(
            admitted,
            binding,
            placement_id="placement-0-a",
            release_id="release-a",
        )
        == admitted
    )


@pytest.mark.parametrize("successor", [False, True])
def test_admission_replaces_template_derived_multi_node_binding(
    successor: bool,
) -> None:
    initial = _binding()
    template = _pod()
    del template["metadata"]["annotations"][RUNNER_CACHE_STARTUP_TARGET_ANNOTATION]
    template["spec"]["schedulingGates"] = [
        {"name": "example.com/another-gate"}
    ]
    inherited = bind_runner_cache_to_pod_template(
        template,
        initial,
        release_id="release-a",
    )
    selected_binding = (
        _binding(suffix="b", nodes=("gpu-c", "gpu-d")) if successor else initial
    )

    admitted = admit_runner_cache_placement_for_gated_pod(
        inherited,
        selected_binding,
        placement_id=("placement-0-b" if successor else "placement-0-a"),
        release_id="release-a",
    )

    assert admitted["spec"]["schedulingGates"] == [
        {"name": "example.com/another-gate"}
    ]
    terms = admitted["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"]
    selected_node = "gpu-c" if successor else "gpu-a"
    assert all(
        term["matchFields"]
        == [{"key": "metadata.name", "operator": "In", "values": [selected_node]}]
        for term in terms
    )
    pod_anti_affinity = admitted["spec"]["affinity"]["podAntiAffinity"]
    assert "requiredDuringSchedulingIgnoredDuringExecution" not in pod_anti_affinity
    annotations = admitted["metadata"]["annotations"]
    assert annotations[RUNNER_CACHE_STARTUP_BINDING_ID_ANNOTATION] == (
        selected_binding.binding_id
    )


@pytest.mark.parametrize("tamper", ["label", "anti-affinity"])
def test_admission_rejects_tampered_template_derived_constraints(tamper: str) -> None:
    binding = _binding()
    template = _pod()
    del template["metadata"]["annotations"][RUNNER_CACHE_STARTUP_TARGET_ANNOTATION]
    template["spec"]["schedulingGates"] = []
    inherited = bind_runner_cache_to_pod_template(
        template,
        binding,
        release_id="release-a",
    )
    if tamper == "label":
        inherited["metadata"]["labels"].pop(RUNNER_CACHE_STARTUP_BINDING_LABEL)
    else:
        inherited["spec"]["affinity"]["podAntiAffinity"].pop(
            "requiredDuringSchedulingIgnoredDuringExecution"
        )

    with pytest.raises(RunnerCacheSchedulingError, match="template-derived"):
        admit_runner_cache_placement_for_gated_pod(
            inherited,
            binding,
            placement_id="placement-0-a",
            release_id="release-a",
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda pod: pod["spec"].pop("schedulingGates"), "exactly one"),
        (
            lambda pod: pod["spec"]["schedulingGates"].append(
                {"name": RUNNER_CACHE_STARTUP_SCHEDULING_GATE}
            ),
            "exactly one",
        ),
        (
            lambda pod: pod["metadata"]["annotations"].__setitem__(
                RUNNER_CACHE_STARTUP_TARGET_ANNOTATION, "deployment/other"
            ),
            "target annotation",
        ),
        (lambda pod: pod["spec"].__setitem__("nodeName", "gpu-z"), "different"),
    ],
)
def test_malformed_or_conflicting_gated_pod_fails_closed(mutate, message: str) -> None:
    pod = _pod()
    mutate(pod)

    with pytest.raises(RunnerCacheSchedulingError, match=message):
        admit_runner_cache_placement_for_gated_pod(
            pod,
            _binding(),
            placement_id="placement-0-a",
            release_id="release-a",
        )


def test_replay_rejects_ambiguous_binding_json() -> None:
    binding = _binding()
    admitted = admit_runner_cache_placement_for_gated_pod(
        _pod(),
        binding,
        placement_id="placement-0-a",
        release_id="release-a",
    )
    raw = admitted["metadata"]["annotations"][RUNNER_CACHE_STARTUP_BINDING_ANNOTATION]
    admitted["metadata"]["annotations"][RUNNER_CACHE_STARTUP_BINDING_ANNOTATION] = (
        '{"schema_version":"runner-cache-startup-binding-v1",' + raw[1:]
    )

    with pytest.raises(RunnerCacheSchedulingError, match="invalid"):
        admit_runner_cache_placement_for_gated_pod(
            admitted,
            binding,
            placement_id="placement-0-a",
            release_id="release-a",
        )


def test_store_claims_unique_placements_and_replays_by_pod_name() -> None:
    binding = _binding()
    store = InMemoryRunnerCachePlacementAdmissionStore()
    store.register(_plan(binding))

    first = store.claim(
        target_id=TARGET,
        binding_id=binding.binding_id,
        pod_key="model-serving/qwen-7",
        admission_uid="admission-a",
        claimed_at=NOW + timedelta(seconds=2),
    )
    replay = store.claim(
        target_id=TARGET,
        binding_id=binding.binding_id,
        pod_key="model-serving/qwen-7",
        admission_uid="admission-retry",
        claimed_at=NOW + timedelta(seconds=3),
    )
    second = store.claim(
        target_id=TARGET,
        binding_id=binding.binding_id,
        pod_key="model-serving/qwen-8",
        admission_uid="admission-b",
        claimed_at=NOW + timedelta(seconds=2),
    )

    assert replay.claim == first.claim
    assert first.created is True
    assert replay.created is False
    assert first.claim.placement_id == "placement-0-a"
    assert second.claim.placement_id == "placement-1-a"
    with pytest.raises(RunnerCachePlacementAdmissionConflictError, match="exhausted"):
        store.claim(
            target_id=TARGET,
            binding_id=binding.binding_id,
            pod_key="model-serving/qwen-9",
            admission_uid="admission-c",
            claimed_at=NOW + timedelta(seconds=2),
        )


def test_creator_rollback_cannot_release_a_claim_observed_by_a_retry() -> None:
    binding = _binding()
    store = InMemoryRunnerCachePlacementAdmissionStore()
    store.register(_plan(binding))
    created = store.claim(
        target_id=TARGET,
        binding_id=binding.binding_id,
        pod_key="model-serving/qwen-7",
        admission_uid="admission-a",
        claimed_at=NOW + timedelta(seconds=2),
    )
    replay = store.claim(
        target_id=TARGET,
        binding_id=binding.binding_id,
        pod_key="model-serving/qwen-7",
        admission_uid="admission-retry",
        claimed_at=NOW + timedelta(seconds=3),
    )

    assert created.created is True
    assert replay.created is False
    store.release(created.claim)
    second = store.claim(
        target_id=TARGET,
        binding_id=binding.binding_id,
        pod_key="model-serving/qwen-8",
        admission_uid="admission-b",
        claimed_at=NOW + timedelta(seconds=3),
    )
    assert second.claim.placement_id == "placement-1-a"


def test_concurrent_claims_are_linearizable_and_unique() -> None:
    binding = _binding(nodes=("gpu-a", "gpu-b", "gpu-c", "gpu-d"))
    store = InMemoryRunnerCachePlacementAdmissionStore()
    store.register(_plan(binding))

    def claim(index: int) -> str:
        return store.claim(
            target_id=TARGET,
            binding_id=binding.binding_id,
            pod_key=f"model-serving/qwen-{index}",
            admission_uid=f"admission-{index}",
            claimed_at=NOW + timedelta(seconds=2),
        ).claim.placement_id

    with ThreadPoolExecutor(max_workers=4) as pool:
        placements = tuple(pool.map(claim, range(4)))

    assert len(set(placements)) == 4


def test_unclaimed_plan_can_be_replaced_after_binding_expiry() -> None:
    first = _binding()
    store = InMemoryRunnerCachePlacementAdmissionStore()
    store.register(_plan(first))
    overlapping = _binding(suffix="b", bound_at=NOW + timedelta(minutes=1))

    with pytest.raises(RunnerCachePlacementAdmissionConflictError, match="safety window"):
        store.register(_plan(overlapping))

    successor = _binding(suffix="c", bound_at=NOW + timedelta(minutes=6))
    store.register(_plan(successor))
    assert store.resolve(TARGET).binding == successor


def test_controller_reauthorizes_and_assigns_each_create() -> None:
    binding = _binding()
    store = InMemoryRunnerCachePlacementAdmissionStore()
    store.register(_plan(binding))
    calls: list[str] = []

    def reauthorize(candidate: RunnerCacheStartupBinding) -> RunnerCacheStartupBinding:
        calls.append(candidate.binding_id)
        return candidate

    controller = RunnerCachePlacementAdmissionController(
        store,
        reauthorize=reauthorize,
    )

    first, first_claim = controller.admit(
        _pod("qwen-7"),
        admission_uid="admission-a",
        request_username="system:serviceaccount:kairyu:statefulset-controller",
        observed_at=NOW + timedelta(seconds=2),
    )
    second, second_claim = controller.admit(
        _pod("qwen-8"),
        admission_uid="admission-b",
        request_username="system:serviceaccount:kairyu:statefulset-controller",
        observed_at=NOW + timedelta(seconds=2),
    )

    assert calls == [binding.binding_id, binding.binding_id]
    assert first_claim.placement_id != second_claim.placement_id
    assert first["metadata"]["annotations"][RUNNER_CACHE_STARTUP_PLACEMENT_ANNOTATION]
    assert second["metadata"]["annotations"][RUNNER_CACHE_STARTUP_PLACEMENT_ANNOTATION]


def test_controller_releases_claim_when_mutation_fails() -> None:
    binding = _binding()
    store = InMemoryRunnerCachePlacementAdmissionStore()
    store.register(_plan(binding))
    controller = RunnerCachePlacementAdmissionController(
        store,
        reauthorize=lambda candidate: candidate,
    )
    invalid = _pod("qwen-invalid")
    invalid["spec"]["affinity"] = []

    with pytest.raises(RunnerCacheSchedulingError, match="affinity"):
        controller.admit(
            invalid,
            admission_uid="admission-invalid",
            request_username="system:serviceaccount:kairyu:statefulset-controller",
            observed_at=NOW + timedelta(seconds=2),
        )

    _, claim = controller.admit(
        _pod("qwen-7"),
        admission_uid="admission-a",
        request_username="system:serviceaccount:kairyu:statefulset-controller",
        observed_at=NOW + timedelta(seconds=2),
    )
    assert claim.placement_id == "placement-0-a"


def test_failed_replay_cannot_release_an_existing_successful_claim() -> None:
    binding = _binding()
    store = InMemoryRunnerCachePlacementAdmissionStore()
    store.register(_plan(binding))
    controller = RunnerCachePlacementAdmissionController(
        store,
        reauthorize=lambda candidate: candidate,
    )
    username = "system:serviceaccount:kairyu:statefulset-controller"
    _, first = controller.admit(
        _pod("qwen-7"),
        admission_uid="admission-a",
        request_username=username,
        observed_at=NOW + timedelta(seconds=2),
    )
    invalid_replay = _pod("qwen-7")
    invalid_replay["spec"]["affinity"] = []

    with pytest.raises(RunnerCacheSchedulingError, match="affinity"):
        controller.admit(
            invalid_replay,
            admission_uid="admission-retry",
            request_username=username,
            observed_at=NOW + timedelta(seconds=3),
        )

    _, second = controller.admit(
        _pod("qwen-8"),
        admission_uid="admission-b",
        request_username=username,
        observed_at=NOW + timedelta(seconds=3),
    )
    assert first.placement_id == "placement-0-a"
    assert second.placement_id == "placement-1-a"
    with pytest.raises(RunnerCachePlacementAdmissionConflictError, match="exhausted"):
        controller.admit(
            _pod("qwen-9"),
            admission_uid="admission-c",
            request_username=username,
            observed_at=NOW + timedelta(seconds=3),
        )


def test_claimed_plan_reserves_nodes_through_replay_safety_window() -> None:
    first = _binding()
    store = InMemoryRunnerCachePlacementAdmissionStore(
        replay_safety_window=timedelta(minutes=5)
    )
    store.register(_plan(first))
    for index in range(2):
        store.claim(
            target_id=TARGET,
            binding_id=first.binding_id,
            pod_key=f"model-serving/qwen-{7 + index}",
            admission_uid=f"admission-{index}",
            claimed_at=NOW + timedelta(seconds=2),
        )
    with pytest.raises(RunnerCachePlacementAdmissionConflictError, match="safety window"):
        store.register(_plan(_binding(suffix="b", bound_at=NOW + timedelta(minutes=1))))

    overlapping = _binding(
        suffix="c",
        nodes=("gpu-a", "gpu-c"),
        bound_at=NOW + timedelta(minutes=6),
    )
    with pytest.raises(RunnerCachePlacementAdmissionConflictError, match="safety window"):
        store.register(_plan(overlapping))

    successor = _binding(
        suffix="d",
        nodes=("gpu-a", "gpu-c"),
        bound_at=NOW + timedelta(minutes=11),
    )
    store.register(_plan(successor))
    claim = store.claim(
        target_id=TARGET,
        binding_id=successor.binding_id,
        pod_key="model-serving/qwen-9",
        admission_uid="admission-successor",
        claimed_at=NOW + timedelta(minutes=11, seconds=2),
    )
    assert claim.claim.placement_id == "placement-0-d"


@pytest.mark.parametrize(
    ("mutate", "username", "message"),
    [
        (
            lambda pod: pod["metadata"].__setitem__("namespace", "other"),
            "system:serviceaccount:kairyu:statefulset-controller",
            "namespace",
        ),
        (
            lambda _pod: None,
            "system:serviceaccount:other:controller",
            "creator",
        ),
        (
            lambda pod: pod["metadata"]["ownerReferences"][0].__setitem__(
                "uid", "workload-uid-other"
            ),
            "system:serviceaccount:kairyu:statefulset-controller",
            "owner",
        ),
    ],
)
def test_controller_binds_claim_authority_to_creator_and_workload(
    mutate,
    username: str,
    message: str,
) -> None:
    binding = _binding()
    store = InMemoryRunnerCachePlacementAdmissionStore()
    store.register(_plan(binding))
    controller = RunnerCachePlacementAdmissionController(
        store,
        reauthorize=lambda candidate: candidate,
    )
    pod = _pod()
    mutate(pod)

    with pytest.raises(RunnerCachePlacementAdmissionError, match=message):
        controller.admit(
            pod,
            admission_uid="admission-a",
            request_username=username,
            observed_at=NOW + timedelta(seconds=2),
        )


def test_controller_fails_closed_on_expired_or_changed_authority() -> None:
    binding = _binding()
    store = InMemoryRunnerCachePlacementAdmissionStore()
    store.register(_plan(binding))
    changed = _binding(suffix="b")
    controller = RunnerCachePlacementAdmissionController(
        store,
        reauthorize=lambda _candidate: changed,
    )

    with pytest.raises(RunnerCachePlacementAdmissionConflictError, match="changed"):
        controller.admit(
            _pod(),
            admission_uid="admission-a",
            request_username="system:serviceaccount:kairyu:statefulset-controller",
            observed_at=NOW + timedelta(seconds=2),
        )

    expired = RunnerCachePlacementAdmissionController(
        store,
        reauthorize=lambda candidate: candidate,
    )
    with pytest.raises(RunnerCachePlacementAdmissionConflictError, match="not live"):
        expired.admit(
            _pod(),
            admission_uid="admission-b",
            request_username="system:serviceaccount:kairyu:statefulset-controller",
            observed_at=binding.valid_until,
        )
