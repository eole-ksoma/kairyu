"""D3.2 Pod-template scheduling constraints for cache-bound Runner startup."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from kairyu.runners import (
    RUNNER_CACHE_STARTUP_BINDING_ANNOTATION,
    RUNNER_CACHE_STARTUP_BINDING_ID_ANNOTATION,
    RUNNER_CACHE_STARTUP_BINDING_LABEL,
    RunnerCacheSchedulingError,
    RunnerCacheStartupBinding,
    RunnerCacheStartupPlacement,
    bind_runner_cache_to_pod_template,
)

NOW = datetime(2026, 9, 29, 10, 0, tzinfo=UTC)
DIGEST = "a" * 64


def _binding(*, suffix: str = "a", nodes: tuple[str, ...] = ("gpu-a", "gpu-b")):
    placements = tuple(
        RunnerCacheStartupPlacement(
            placement_id=f"placement-{index}-{suffix}",
            node_name=node,
            resource_flavor="h100-sxm",
            profile_id="h100-sxm-tp1",
            compatibility_approval_id="compat-qwen-h100",
            manifest_digest=DIGEST,
            pin_owner=f"prestage/model-serving/qwen/placement-{index}-{suffix}",
            prestage_command_id=hashlib.sha256(f"command-{index}-{suffix}".encode()).hexdigest(),
            prestage_command_generation=index + 1,
            hint_index_revision=10 + index,
            resident_record_generation=20 + index,
            hint_observed_at=NOW,
            hint_valid_until=NOW + timedelta(minutes=5),
        )
        for index, node in enumerate(nodes)
    )
    payload = {
        "schema_version": "runner-cache-startup-binding-v1",
        "decision_id": f"decision-{suffix}",
        "decision_fingerprint": hashlib.sha256(f"decision-{suffix}".encode()).hexdigest(),
        "target_id": "deployment/model-serving/qwen-runners",
        "target_revision": 7,
        "deployment_id": "model-serving/qwen",
        "model_class": "qwen-14b",
        "model_id": "org/qwen",
        "model_revision": "revision-a",
        "manifest_digest": DIGEST,
        "placement_binding_id": "placement-binding-a",
        "prewarm_snapshot_id": "snapshot-a",
        "prewarm_cache_revision": 9,
        "bound_at": NOW + timedelta(seconds=1),
        "valid_until": NOW + timedelta(minutes=5),
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


def _template() -> dict:
    return {
        "metadata": {
            "labels": {"app": "qwen-runners"},
            "annotations": {"example.com/owner": "platform"},
        },
        "spec": {
            "affinity": {
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [
                            {
                                "matchExpressions": [
                                    {"key": "gpu.vendor", "operator": "In", "values": ["nvidia"]}
                                ]
                            },
                            {
                                "matchExpressions": [
                                    {"key": "zone", "operator": "In", "values": ["a"]}
                                ]
                            },
                        ]
                    }
                },
                "podAntiAffinity": {
                    "preferredDuringSchedulingIgnoredDuringExecution": [
                        {"weight": 10, "podAffinityTerm": {"topologyKey": "zone"}}
                    ]
                },
            },
            "containers": [{"name": "runner", "image": "runner@sha256:deadbeef"}],
        },
    }


def test_binding_constrains_every_node_term_and_persists_full_evidence() -> None:
    original = _template()
    binding = _binding()

    result = bind_runner_cache_to_pod_template(original, binding, release_id="release-a")

    assert original == _template()
    annotations = result["metadata"]["annotations"]
    assert annotations["example.com/owner"] == "platform"
    assert annotations[RUNNER_CACHE_STARTUP_BINDING_ID_ANNOTATION] == binding.binding_id
    assert (
        RunnerCacheStartupBinding.model_validate_json(
            annotations[RUNNER_CACHE_STARTUP_BINDING_ANNOTATION]
        )
        == binding
    )
    label_value = result["metadata"]["labels"][RUNNER_CACHE_STARTUP_BINDING_LABEL]
    assert len(label_value) == 63
    affinity = result["spec"]["affinity"]
    terms = affinity["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][
        "nodeSelectorTerms"
    ]
    assert all(
        term["matchFields"][-1]
        == {"key": "metadata.name", "operator": "In", "values": ["gpu-a", "gpu-b"]}
        for term in terms
    )
    assert affinity["podAntiAffinity"]["preferredDuringSchedulingIgnoredDuringExecution"]
    assert affinity["podAntiAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"] == [
        {
            "labelSelector": {
                "matchLabels": {RUNNER_CACHE_STARTUP_BINDING_LABEL: label_value}
            },
            "topologyKey": "kubernetes.io/hostname",
        }
    ]


def test_exact_replay_is_idempotent() -> None:
    binding = _binding()
    once = bind_runner_cache_to_pod_template(_template(), binding, release_id="release-a")

    assert (
        bind_runner_cache_to_pod_template(once, binding, release_id="release-a")
        == once
    )


def test_successor_binding_replaces_only_managed_constraints() -> None:
    first = _binding()
    successor = _binding(suffix="b", nodes=("gpu-c", "gpu-d"))
    prior = bind_runner_cache_to_pod_template(_template(), first, release_id="release-a")

    result = bind_runner_cache_to_pod_template(prior, successor, release_id="release-a")

    terms = result["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"]
    assert all(len(term["matchFields"]) == 1 for term in terms)
    assert all(term["matchFields"][0]["values"] == ["gpu-c", "gpu-d"] for term in terms)
    required = result["spec"]["affinity"]["podAntiAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]
    assert len(required) == 1
    assert (
        result["metadata"]["annotations"][RUNNER_CACHE_STARTUP_BINDING_ID_ANNOTATION]
        == successor.binding_id
    )


def test_incomplete_previous_binding_metadata_fails_closed() -> None:
    template = _template()
    template["metadata"]["annotations"][RUNNER_CACHE_STARTUP_BINDING_ID_ANNOTATION] = (
        "0" * 64
    )

    with pytest.raises(RunnerCacheSchedulingError, match="incomplete"):
        bind_runner_cache_to_pod_template(template, _binding(), release_id="release-a")


def test_tampered_managed_affinity_fails_closed_on_rebind() -> None:
    prior = bind_runner_cache_to_pod_template(
        _template(),
        _binding(),
        release_id="release-a",
    )
    terms = prior["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"]
    terms[0]["matchFields"].clear()

    with pytest.raises(RunnerCacheSchedulingError, match="missing or ambiguous"):
        bind_runner_cache_to_pod_template(
            prior,
            _binding(suffix="b"),
            release_id="release-a",
        )


def test_invalid_existing_affinity_shape_fails_closed() -> None:
    template = _template()
    template["spec"]["affinity"]["nodeAffinity"] = []

    with pytest.raises(RunnerCacheSchedulingError, match="nodeAffinity"):
        bind_runner_cache_to_pod_template(template, _binding(), release_id="release-a")


def test_conflicting_pod_identity_is_not_overwritten() -> None:
    template = _template()
    template["metadata"]["annotations"]["kairyu.ai/model-id"] = "org/other"

    with pytest.raises(RunnerCacheSchedulingError, match="conflicts"):
        bind_runner_cache_to_pod_template(template, _binding(), release_id="release-a")


def test_total_annotation_size_is_bounded() -> None:
    template = _template()
    template["metadata"]["annotations"]["example.com/large"] = "x" * 240_000

    with pytest.raises(RunnerCacheSchedulingError, match="safe size"):
        bind_runner_cache_to_pod_template(template, _binding(), release_id="release-a")
