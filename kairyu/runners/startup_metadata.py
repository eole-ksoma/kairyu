"""Shared Kubernetes metadata keys and parsing for cache-bound startup."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from kairyu.runners.startup_binding import RunnerCacheStartupBinding

RUNNER_CACHE_STARTUP_BINDING_ANNOTATION = "kairyu.ai/cache-startup-binding"
RUNNER_CACHE_STARTUP_BINDING_ID_ANNOTATION = "kairyu.ai/cache-startup-binding-id"
RUNNER_CACHE_STARTUP_PLACEMENT_ANNOTATION = "kairyu.ai/cache-startup-placement"
RUNNER_CACHE_STARTUP_TARGET_ANNOTATION = "kairyu.ai/cache-startup-target"
RUNNER_CACHE_STARTUP_BINDING_LABEL = "kairyu.ai/cache-startup-binding-slot"
RUNNER_CACHE_STARTUP_SCHEDULING_GATE = "kairyu.ai/cache-startup-placement"


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_number(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def parse_runner_cache_startup_binding_annotations(
    annotations: Mapping[str, Any],
) -> RunnerCacheStartupBinding | None:
    """Parse exact binding annotations using strict, ambiguity-free JSON."""

    raw = annotations.get(RUNNER_CACHE_STARTUP_BINDING_ANNOTATION)
    binding_id = annotations.get(RUNNER_CACHE_STARTUP_BINDING_ID_ANNOTATION)
    if raw is None and binding_id is None:
        return None
    if not isinstance(raw, str) or not isinstance(binding_id, str):
        raise ValueError("cache startup binding annotations must be complete strings")
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_number,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"{RUNNER_CACHE_STARTUP_BINDING_ANNOTATION!r} must contain strict JSON"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("cache startup binding annotation must contain a JSON object")
    binding = RunnerCacheStartupBinding.model_validate(payload)
    if binding.binding_id != binding_id:
        raise ValueError("cache startup binding annotations do not identify the same binding")
    return binding
