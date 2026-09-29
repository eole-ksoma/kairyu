"""Kubernetes Pod-template constraints derived from a Runner cache binding."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

from kairyu.runners.kubernetes import (
    MODEL_ID_ANNOTATION,
    MODEL_REVISION_ANNOTATION,
    RELEASE_ID_ANNOTATION,
)
from kairyu.runners.startup_binding import RunnerCacheStartupBinding
from kairyu.runners.startup_metadata import (
    RUNNER_CACHE_STARTUP_BINDING_ANNOTATION,
    RUNNER_CACHE_STARTUP_BINDING_ID_ANNOTATION,
    RUNNER_CACHE_STARTUP_BINDING_LABEL,
    RUNNER_CACHE_STARTUP_PLACEMENT_ANNOTATION,
    RUNNER_CACHE_STARTUP_SCHEDULING_GATE,
    RUNNER_CACHE_STARTUP_TARGET_ANNOTATION,
    parse_runner_cache_startup_binding_annotations,
)

_NODE_NAME_FIELD = "metadata.name"
_NODE_TOPOLOGY_KEY = "kubernetes.io/hostname"
_MAX_BINDING_ANNOTATION_BYTES = 200_000
_MAX_TOTAL_ANNOTATION_BYTES = 240_000


class RunnerCacheSchedulingError(RuntimeError):
    """A Pod template cannot safely enforce a cache startup binding."""


def _string_map(value: object, *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise RunnerCacheSchedulingError(f"{name} must contain string pairs")
    return dict(value)


def _binding_payload(binding: RunnerCacheStartupBinding) -> str:
    payload = json.dumps(
        binding.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if len(payload.encode("utf-8")) > _MAX_BINDING_ANNOTATION_BYTES:
        raise RunnerCacheSchedulingError("startup binding exceeds the Pod annotation limit")
    return payload


def _binding_label(binding_id: str) -> str:
    # Kubernetes label values are limited to 63 characters. The full digest is
    # retained in the annotation and this value is used only for Pod anti-affinity.
    return f"b-{binding_id[:61]}"


def _node_requirement(node_names: tuple[str, ...]) -> dict[str, object]:
    return {"key": _NODE_NAME_FIELD, "operator": "In", "values": list(node_names)}


def _anti_affinity_term(label_value: str) -> dict[str, object]:
    return {
        "labelSelector": {"matchLabels": {RUNNER_CACHE_STARTUP_BINDING_LABEL: label_value}},
        "topologyKey": _NODE_TOPOLOGY_KEY,
    }


def _old_binding(annotations: Mapping[str, str]) -> RunnerCacheStartupBinding | None:
    payload = annotations.get(RUNNER_CACHE_STARTUP_BINDING_ANNOTATION)
    binding_id = annotations.get(RUNNER_CACHE_STARTUP_BINDING_ID_ANNOTATION)
    if (payload is None) != (binding_id is None):
        raise RunnerCacheSchedulingError(
            "Pod template has incomplete startup binding metadata"
        )
    try:
        return parse_runner_cache_startup_binding_annotations(annotations)
    except (ValueError, TypeError) as error:
        raise RunnerCacheSchedulingError(
            "Pod template startup binding metadata is invalid"
        ) from error


def _required_node_affinity(
    affinity: dict[str, Any],
    *,
    node_names: tuple[str, ...],
    previous_node_names: tuple[str, ...] | None,
) -> None:
    node_affinity = affinity.setdefault("nodeAffinity", {})
    if not isinstance(node_affinity, dict):
        raise RunnerCacheSchedulingError("Pod template nodeAffinity must be an object")
    required = node_affinity.get("requiredDuringSchedulingIgnoredDuringExecution")
    if required is None:
        required = {"nodeSelectorTerms": [{}]}
        node_affinity["requiredDuringSchedulingIgnoredDuringExecution"] = required
    if not isinstance(required, dict):
        raise RunnerCacheSchedulingError("required node affinity must be an object")
    terms = required.get("nodeSelectorTerms")
    if not isinstance(terms, list) or not terms:
        raise RunnerCacheSchedulingError("required node affinity needs selector terms")

    previous = _node_requirement(previous_node_names) if previous_node_names else None
    current = _node_requirement(node_names)
    for term in terms:
        if not isinstance(term, dict):
            raise RunnerCacheSchedulingError("node selector terms must be objects")
        fields = term.setdefault("matchFields", [])
        if not isinstance(fields, list) or not all(isinstance(item, dict) for item in fields):
            raise RunnerCacheSchedulingError("node selector matchFields must be objects")
        if previous is not None:
            matches = [index for index, item in enumerate(fields) if item == previous]
            if len(matches) != 1:
                raise RunnerCacheSchedulingError(
                    "previous managed node affinity is missing or ambiguous"
                )
            fields.pop(matches[0])
        fields.append(copy.deepcopy(current))


def _required_pod_anti_affinity(
    affinity: dict[str, Any],
    *,
    label_value: str,
    previous_label_value: str | None,
) -> None:
    pod_anti_affinity = affinity.setdefault("podAntiAffinity", {})
    if not isinstance(pod_anti_affinity, dict):
        raise RunnerCacheSchedulingError("Pod template podAntiAffinity must be an object")
    required = pod_anti_affinity.setdefault("requiredDuringSchedulingIgnoredDuringExecution", [])
    if not isinstance(required, list) or not all(isinstance(item, dict) for item in required):
        raise RunnerCacheSchedulingError("required Pod anti-affinity terms must be objects")
    if previous_label_value is not None:
        previous = _anti_affinity_term(previous_label_value)
        matches = [index for index, item in enumerate(required) if item == previous]
        if len(matches) != 1:
            raise RunnerCacheSchedulingError(
                "previous managed Pod anti-affinity is missing or ambiguous"
            )
        required.pop(matches[0])
    required.append(_anti_affinity_term(label_value))


def _remove_required_pod_anti_affinity(
    affinity: dict[str, Any],
    *,
    label_value: str,
) -> None:
    pod_anti_affinity = affinity.get("podAntiAffinity")
    if not isinstance(pod_anti_affinity, dict):
        raise RunnerCacheSchedulingError(
            "template-derived Pod requires managed Pod anti-affinity"
        )
    required = pod_anti_affinity.get("requiredDuringSchedulingIgnoredDuringExecution")
    if not isinstance(required, list) or not all(isinstance(item, dict) for item in required):
        raise RunnerCacheSchedulingError(
            "template-derived Pod required anti-affinity must contain objects"
        )
    expected = _anti_affinity_term(label_value)
    matches = [index for index, item in enumerate(required) if item == expected]
    if len(matches) != 1:
        raise RunnerCacheSchedulingError(
            "template-derived managed Pod anti-affinity is missing or ambiguous"
        )
    required.pop(matches[0])
    if not required:
        pod_anti_affinity.pop("requiredDuringSchedulingIgnoredDuringExecution")


def bind_runner_cache_to_pod_template(
    template: Mapping[str, Any],
    binding: RunnerCacheStartupBinding,
    *,
    release_id: str,
) -> dict[str, Any]:
    """Return an idempotently constrained copy of one Kubernetes Pod template.

    Existing affinity is preserved. The managed node restriction is ANDed into
    every required node-selector term, while binding-specific hard Pod
    anti-affinity prevents two newly created replicas from consuming the same
    cache placement node.
    """

    if not isinstance(binding, RunnerCacheStartupBinding):
        raise TypeError("binding must be a RunnerCacheStartupBinding")
    binding = RunnerCacheStartupBinding.model_validate(binding.model_dump())
    if (
        not isinstance(release_id, str) or not release_id.strip() or "\x00" in release_id
    ):
        raise ValueError("release_id must be a non-empty string without NUL")
    if not isinstance(template, Mapping):
        raise TypeError("template must be a mapping")
    result = copy.deepcopy(dict(template))
    metadata = result.get("metadata")
    spec = result.get("spec")
    if not isinstance(metadata, dict) or not isinstance(spec, dict):
        raise RunnerCacheSchedulingError("Pod template requires metadata and spec objects")

    annotations_present = "annotations" in metadata
    labels_present = "labels" in metadata
    annotations = _string_map(metadata.get("annotations", {}), name="Pod annotations")
    labels = _string_map(metadata.get("labels", {}), name="Pod labels")
    previous = _old_binding(annotations)
    previous_nodes = (
        tuple(placement.node_name for placement in previous.placements)
        if previous is not None
        else None
    )
    previous_label = _binding_label(previous.binding_id) if previous is not None else None

    affinity = spec.setdefault("affinity", {})
    if not isinstance(affinity, dict):
        raise RunnerCacheSchedulingError("Pod template affinity must be an object")
    nodes = tuple(placement.node_name for placement in binding.placements)
    _required_node_affinity(
        affinity,
        node_names=nodes,
        previous_node_names=previous_nodes,
    )
    label_value = _binding_label(binding.binding_id)
    _required_pod_anti_affinity(
        affinity,
        label_value=label_value,
        previous_label_value=previous_label,
    )

    annotations[RUNNER_CACHE_STARTUP_BINDING_ANNOTATION] = _binding_payload(binding)
    annotations[RUNNER_CACHE_STARTUP_BINDING_ID_ANNOTATION] = binding.binding_id
    identities = {
        RUNNER_CACHE_STARTUP_TARGET_ANNOTATION: binding.target_id,
        MODEL_ID_ANNOTATION: binding.model_id,
        MODEL_REVISION_ANNOTATION: binding.model_revision,
        RELEASE_ID_ANNOTATION: release_id,
    }
    for name, value in identities.items():
        existing = annotations.get(name)
        if existing is not None and existing != value:
            raise RunnerCacheSchedulingError(
                f"Pod template {name!r} conflicts with the startup binding"
            )
        annotations[name] = value
    annotation_bytes = sum(
        len(name.encode("utf-8")) + len(value.encode("utf-8"))
        for name, value in annotations.items()
    )
    if annotation_bytes > _MAX_TOTAL_ANNOTATION_BYTES:
        raise RunnerCacheSchedulingError("Pod template annotations exceed the safe size limit")
    labels[RUNNER_CACHE_STARTUP_BINDING_LABEL] = label_value
    gates = spec.setdefault("schedulingGates", [])
    if not isinstance(gates, list) or not all(isinstance(gate, dict) for gate in gates):
        raise RunnerCacheSchedulingError("Pod template schedulingGates must contain objects")
    managed_gate = {"name": RUNNER_CACHE_STARTUP_SCHEDULING_GATE}
    matches = [gate for gate in gates if gate == managed_gate]
    if previous is None:
        if matches:
            raise RunnerCacheSchedulingError(
                "unbound Pod template has an ambiguous cache startup scheduling gate"
            )
        gates.append(managed_gate)
    elif len(matches) != 1:
        raise RunnerCacheSchedulingError(
            "bound Pod template requires exactly one cache startup scheduling gate"
        )
    if annotations or annotations_present:
        metadata["annotations"] = annotations
    if labels or labels_present:
        metadata["labels"] = labels
    return result


def admit_runner_cache_placement_for_gated_pod(
    pod: Mapping[str, Any],
    binding: RunnerCacheStartupBinding,
    *,
    placement_id: str,
    release_id: str,
) -> dict[str, Any]:
    """Mutate one gated Pod CREATE object to one exact cache placement."""

    if not isinstance(binding, RunnerCacheStartupBinding):
        raise TypeError("binding must be a RunnerCacheStartupBinding")
    binding = RunnerCacheStartupBinding.model_validate(binding.model_dump())
    if not isinstance(placement_id, str) or not placement_id.strip() or "\x00" in placement_id:
        raise ValueError("placement_id must be a non-empty string without NUL")
    if not isinstance(release_id, str) or not release_id.strip() or "\x00" in release_id:
        raise ValueError("release_id must be a non-empty string without NUL")
    placements = {
        placement.placement_id: placement for placement in binding.placements
    }
    placement = placements.get(placement_id)
    if placement is None:
        raise RunnerCacheSchedulingError("placement_id is absent from startup binding")
    if not isinstance(pod, Mapping):
        raise TypeError("pod must be a mapping")
    result = copy.deepcopy(dict(pod))
    metadata = result.get("metadata")
    spec = result.get("spec")
    if not isinstance(metadata, dict) or not isinstance(spec, dict):
        raise RunnerCacheSchedulingError("Pod requires metadata and spec objects")
    node_name = spec.get("nodeName")
    if node_name is not None and node_name != placement.node_name:
        raise RunnerCacheSchedulingError("scheduled Pod is on a different placement node")

    annotations_present = "annotations" in metadata
    labels_present = "labels" in metadata
    annotations = _string_map(metadata.get("annotations", {}), name="Pod annotations")
    labels = _string_map(metadata.get("labels", {}), name="Pod labels")
    if annotations.get(RUNNER_CACHE_STARTUP_TARGET_ANNOTATION) != binding.target_id:
        raise RunnerCacheSchedulingError(
            "gated Pod target annotation does not match the startup binding"
        )
    previous = _old_binding(annotations)
    previous_placement_id = annotations.get(
        RUNNER_CACHE_STARTUP_PLACEMENT_ANNOTATION
    )
    if previous is None and previous_placement_id is not None:
        raise RunnerCacheSchedulingError("Pod has incomplete startup placement metadata")
    admitted_replay = previous is not None and previous_placement_id is not None
    template_derived = previous is not None and previous_placement_id is None
    if admitted_replay and (
        previous.binding_id != binding.binding_id
        or previous_placement_id != placement_id
    ):
        raise RunnerCacheSchedulingError("Pod is already bound to another placement")

    gates = spec.get("schedulingGates", [])
    if not isinstance(gates, list) or not all(isinstance(gate, dict) for gate in gates):
        raise RunnerCacheSchedulingError("Pod schedulingGates must contain objects")
    managed_gate = {"name": RUNNER_CACHE_STARTUP_SCHEDULING_GATE}
    matches = [index for index, gate in enumerate(gates) if gate == managed_gate]
    if previous is None or template_derived:
        if len(matches) != 1:
            raise RunnerCacheSchedulingError(
                "unbound Pod requires exactly one cache startup scheduling gate"
            )
        if node_name is not None:
            raise RunnerCacheSchedulingError("unbound gated Pod must not be scheduled")
        gates.pop(matches[0])
    elif matches:
        raise RunnerCacheSchedulingError("bound Pod cannot regain its scheduling gate")

    affinity = spec.setdefault("affinity", {})
    if not isinstance(affinity, dict):
        raise RunnerCacheSchedulingError("Pod affinity must be an object")
    previous_nodes = None
    if admitted_replay:
        previous_nodes = (placement.node_name,)
    elif template_derived:
        previous_nodes = tuple(
            prior_placement.node_name for prior_placement in previous.placements
        )
    _required_node_affinity(
        affinity,
        node_names=(placement.node_name,),
        previous_node_names=previous_nodes,
    )
    if template_derived:
        previous_label = _binding_label(previous.binding_id)
        if labels.get(RUNNER_CACHE_STARTUP_BINDING_LABEL) != previous_label:
            raise RunnerCacheSchedulingError(
                "template-derived Pod binding label is missing or inconsistent"
            )
        _remove_required_pod_anti_affinity(
            affinity,
            label_value=previous_label,
        )

    annotations[RUNNER_CACHE_STARTUP_BINDING_ANNOTATION] = _binding_payload(binding)
    annotations[RUNNER_CACHE_STARTUP_BINDING_ID_ANNOTATION] = binding.binding_id
    annotations[RUNNER_CACHE_STARTUP_PLACEMENT_ANNOTATION] = placement_id
    identities = {
        MODEL_ID_ANNOTATION: binding.model_id,
        MODEL_REVISION_ANNOTATION: binding.model_revision,
        RELEASE_ID_ANNOTATION: release_id,
    }
    for name, value in identities.items():
        existing = annotations.get(name)
        if existing is not None and existing != value:
            raise RunnerCacheSchedulingError(
                f"Pod {name!r} conflicts with the startup binding"
            )
        annotations[name] = value
    annotation_bytes = sum(
        len(name.encode("utf-8")) + len(value.encode("utf-8"))
        for name, value in annotations.items()
    )
    if annotation_bytes > _MAX_TOTAL_ANNOTATION_BYTES:
        raise RunnerCacheSchedulingError("Pod annotations exceed the safe size limit")
    labels[RUNNER_CACHE_STARTUP_BINDING_LABEL] = _binding_label(binding.binding_id)
    if annotations or annotations_present:
        metadata["annotations"] = annotations
    if labels or labels_present:
        metadata["labels"] = labels
    if gates:
        spec["schedulingGates"] = gates
    else:
        spec.pop("schedulingGates", None)
    return result
