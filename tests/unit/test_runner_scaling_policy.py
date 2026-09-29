"""Executable contract for model-class Runner scaling policies."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from kairyu.runners import ScalingPolicy, ScalingPolicyCatalog


def _policy(**updates) -> ScalingPolicy:
    values = {
        "model_class": "interactive-14b",
        "policy_revision": 7,
        "min_replicas": 2,
        "max_replicas": 50,
        "warm_buffer_replicas": 3,
        "warm_buffer_ratio": 0.25,
        "scale_up_delay_seconds": 5,
        "keep_alive_seconds": 300,
        "cooldown_seconds": 60,
        "scale_to_zero": False,
        "max_multiplexing": 4,
        "max_scale_up_step": 8,
        "max_scale_down_step": 2,
    }
    values.update(updates)
    return ScalingPolicy(**values)


def test_policy_is_versioned_immutable_and_json_round_trippable() -> None:
    policy = _policy()

    assert policy.schema_version == "runner-scaling-policy-v1"
    assert policy.identity == ("interactive-14b", 7)
    assert ScalingPolicy.model_validate_json(policy.model_dump_json()) == policy
    with pytest.raises(ValidationError, match="frozen"):
        policy.max_replicas = 51  # type: ignore[misc]


def test_safe_defaults_keep_one_warm_non_multiplexed_replica() -> None:
    policy = ScalingPolicy(model_class="default", policy_revision=1)

    assert policy.min_replicas == 1
    assert policy.max_replicas == 1
    assert policy.warm_buffer_replicas == 0
    assert policy.warm_buffer_ratio == 0
    assert policy.scale_up_delay_seconds == 0
    assert policy.keep_alive_seconds == 60
    assert policy.cooldown_seconds == 60
    assert policy.max_observation_age_seconds == 30
    assert policy.scale_to_zero is False
    assert policy.scale_to_zero_approval_id is None
    assert policy.max_multiplexing == 1


def test_scale_to_zero_requires_zero_minimum_and_explicit_approval() -> None:
    policy = _policy(
        min_replicas=0,
        warm_buffer_replicas=0,
        scale_to_zero=True,
        scale_to_zero_approval_id="capacity-review-2026-09",
    )
    assert policy.scale_to_zero is True

    with pytest.raises(ValidationError, match="min_replicas=0"):
        _policy(scale_to_zero=True, scale_to_zero_approval_id="approved")
    with pytest.raises(ValidationError, match="approval ID"):
        _policy(min_replicas=0, scale_to_zero=True)
    with pytest.raises(ValidationError, match="requires scale-to-zero"):
        _policy(min_replicas=0)
    with pytest.raises(ValidationError, match="approval requires"):
        _policy(scale_to_zero_approval_id="stale-approval")
    with pytest.raises(ValidationError, match="warm_buffer_replicas=0"):
        _policy(
            min_replicas=0,
            scale_to_zero=True,
            scale_to_zero_approval_id="approved",
        )


def test_warm_buffer_uses_demand_replicas_and_rounds_ratio_up() -> None:
    policy = _policy(warm_buffer_replicas=3, warm_buffer_ratio=0.25)

    assert policy.warm_buffer_for(0) == 3
    assert policy.warm_buffer_for(8) == 3
    assert policy.warm_buffer_for(13) == 4
    assert policy.warm_buffer_for(20) == 5
    assert policy.warm_buffer_for(10**1000) == 13
    with pytest.raises(ValueError, match="non-negative integer"):
        policy.warm_buffer_for(True)
    with pytest.raises(ValueError, match="non-negative integer"):
        policy.warm_buffer_for(-1)


def test_scale_to_zero_has_no_buffer_at_zero_demand() -> None:
    policy = _policy(
        min_replicas=0,
        warm_buffer_replicas=0,
        warm_buffer_ratio=0.25,
        scale_to_zero=True,
        scale_to_zero_approval_id="approved",
    )

    assert policy.warm_buffer_for(0) == 0
    assert policy.warm_buffer_for(1) == 1


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"min_replicas": 51}, "min_replicas"),
        ({"warm_buffer_replicas": 51}, "warm_buffer_replicas"),
        ({"max_scale_up_step": 51}, "max_scale_up_step"),
        ({"max_scale_down_step": 51}, "max_scale_down_step"),
    ],
)
def test_policy_rejects_cross_field_capacity_violations(updates, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _policy(**updates)


@pytest.mark.parametrize(
    "field",
    [
        "policy_revision",
        "min_replicas",
        "max_replicas",
        "warm_buffer_replicas",
        "max_multiplexing",
        "max_scale_up_step",
        "max_scale_down_step",
    ],
)
@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_integer_fields_reject_coercion(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match="integer"):
        _policy(**{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "warm_buffer_ratio",
        "scale_up_delay_seconds",
        "keep_alive_seconds",
        "cooldown_seconds",
        "max_observation_age_seconds",
    ],
)
@pytest.mark.parametrize("value", [True, "1", math.inf, -math.inf, math.nan])
def test_duration_and_ratio_fields_require_finite_numbers(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError, match="number|finite"):
        _policy(**{field: value})


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_scale_to_zero_requires_a_real_boolean(value: object) -> None:
    with pytest.raises(ValidationError, match="boolean"):
        _policy(scale_to_zero=value)


@pytest.mark.parametrize(
    "updates",
    [
        {"policy_revision": 0},
        {"max_replicas": 0},
        {"warm_buffer_ratio": -0.1},
        {"warm_buffer_ratio": 10.1},
        {"scale_up_delay_seconds": -1},
        {"keep_alive_seconds": 604_801},
        {"cooldown_seconds": 604_801},
        {"max_observation_age_seconds": 0},
        {"max_observation_age_seconds": 86_401},
        {"max_multiplexing": 0},
        {"max_multiplexing": 1025},
    ],
)
def test_policy_rejects_out_of_range_values(updates) -> None:
    with pytest.raises(ValidationError):
        _policy(**updates)


@pytest.mark.parametrize(
    "updates",
    [
        {"model_class": ""},
        {"model_class": "   "},
        {"model_class": "bad\x00class"},
        {
            "min_replicas": 0,
            "scale_to_zero": True,
            "scale_to_zero_approval_id": " ",
        },
    ],
)
def test_policy_rejects_invalid_identities(updates) -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        _policy(**updates)


def test_policy_revalidation_rejects_model_copy_bypass() -> None:
    bypassed = _policy().model_copy(update={"max_replicas": 0})

    with pytest.raises(ValidationError):
        ScalingPolicy.model_validate(bypassed)
    with pytest.raises(ValidationError):
        _ = bypassed.identity
    with pytest.raises(ValidationError):
        bypassed.warm_buffer_for(1)
    with pytest.raises(ValidationError):
        ScalingPolicyCatalog(catalog_revision=1, policies=(bypassed,))


def test_catalog_lookup_revalidates_model_copy_bypass() -> None:
    valid = _policy()
    catalog = ScalingPolicyCatalog(catalog_revision=1, policies=(valid,))
    bypassed_policy = valid.model_copy(update={"max_replicas": 0})
    bypassed_catalog = catalog.model_copy(update={"policies": (bypassed_policy,)})

    with pytest.raises(ValidationError):
        bypassed_catalog.policy_for(valid.model_class)


def test_catalog_resolves_unique_model_classes_and_round_trips() -> None:
    interactive = _policy()
    batch = _policy(
        model_class="batch-14b",
        policy_revision=3,
        min_replicas=0,
        warm_buffer_replicas=0,
        scale_to_zero=True,
        scale_to_zero_approval_id="batch-cold-start-gate-17",
    )
    catalog = ScalingPolicyCatalog(
        catalog_revision=9,
        policies=(interactive, batch),
    )

    assert catalog.policy_for("interactive-14b") == interactive
    assert catalog.policy_for("batch-14b") == batch
    assert ScalingPolicyCatalog.model_validate_json(catalog.model_dump_json()) == catalog
    with pytest.raises(KeyError, match="unknown scaling model class"):
        catalog.policy_for("embedding")


def test_catalog_rejects_empty_duplicate_and_coerced_revision() -> None:
    with pytest.raises(ValidationError):
        ScalingPolicyCatalog(catalog_revision=1, policies=())
    with pytest.raises(ValidationError, match="unique model classes"):
        ScalingPolicyCatalog(
            catalog_revision=1,
            policies=(_policy(), _policy(policy_revision=8)),
        )
    with pytest.raises(ValidationError, match="integer"):
        ScalingPolicyCatalog(catalog_revision=True, policies=(_policy(),))
    with pytest.raises(ValidationError, match="less than or equal"):
        ScalingPolicyCatalog(catalog_revision=2**63, policies=(_policy(),))


def test_catalog_policy_lookup_rejects_invalid_identity() -> None:
    catalog = ScalingPolicyCatalog(catalog_revision=1, policies=(_policy(),))

    with pytest.raises(ValueError, match="non-empty"):
        catalog.policy_for("\x00")
