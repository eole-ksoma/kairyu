from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from kairyu.runners.scaling_quota import (
    KueueScalingAdmission,
    ScalingQuotaAdmission,
    ScalingQuotaConstraint,
    ScalingQuotaLimit,
    ScalingQuotaScope,
    ScalingQuotaSnapshot,
    admit_scaling_quota,
    kueue_scaling_workload_name,
    parse_kueue_scaling_admission,
)

NOW = datetime(2026, 9, 17, 3, 0, tzinfo=UTC)


def _kueue(*, admitted: bool = True, admitted_gpus: int = 16) -> KueueScalingAdmission:
    return KueueScalingAdmission(
        api_version="kueue.x-k8s.io/v1beta2",
        namespace="tenant-a",
        workload_name=kueue_scaling_workload_name(
            target_kind="Deployment",
            target_namespace="tenant-a",
            target_name="model-a-runners",
            target_uid="target-uid-a",
        ),
        workload_uid="workload-uid-a",
        workload_generation=3,
        resource_version="1402",
        target_kind="Deployment",
        target_namespace="tenant-a",
        target_name="model-a-runners",
        target_uid="target-uid-a",
        local_queue="serving",
        cluster_queue="tenant-a-gpu" if admitted else None,
        pod_set_name="runners",
        resource_flavor="h100-sxm" if admitted else None,
        priority_class_name="interactive-serving",
        priority_class_group="kueue.x-k8s.io",
        priority_class_kind="WorkloadPriorityClass",
        priority=1000,
        admitted=admitted,
        admitted_pods=admitted_gpus // 2 if admitted else 0,
        admitted_gpus=admitted_gpus,
    )


def _snapshot(
    *,
    limits: tuple[ScalingQuotaLimit, ...] | None = None,
    kueue: KueueScalingAdmission | None = None,
) -> ScalingQuotaSnapshot:
    resolved_kueue = kueue or _kueue()
    return ScalingQuotaSnapshot(
        snapshot_id="quota-snapshot-a",
        quota_revision=17,
        observed_at=NOW,
        tenant_id="tenant-a",
        model_class="model-a",
        model_family="family-a",
        gpus_per_replica=2,
        target_reserved_gpus=resolved_kueue.admitted_gpus,
        limits=limits
        or (
            ScalingQuotaLimit(
                scope=ScalingQuotaScope.CLUSTER,
                quota_name="primary-gpu-pool",
                hard_limit_gpus=64,
                used_gpus_excluding_target=20,
                reserved_gpus_for_higher_priority=8,
            ),
            ScalingQuotaLimit(
                scope=ScalingQuotaScope.MODEL_FAMILY,
                quota_name="family-a",
                hard_limit_gpus=40,
                used_gpus_excluding_target=12,
                reserved_gpus_for_higher_priority=4,
            ),
            ScalingQuotaLimit(
                scope=ScalingQuotaScope.TENANT_MODEL,
                quota_name="tenant-a/model-a",
                hard_limit_gpus=24,
                used_gpus_excluding_target=4,
                reserved_gpus_for_higher_priority=4,
            ),
        ),
        kueue=resolved_kueue,
    )


def _workload_payload(*, admitted: bool = True) -> dict:
    status = {
        "conditions": [
            {
                "type": "Admitted",
                "status": "True" if admitted else "False",
                "observedGeneration": 3,
            }
        ]
    }
    if admitted:
        status["admission"] = {
            "clusterQueue": "tenant-a-gpu",
            "podSetAssignments": [
                {
                    "name": "runners",
                    "flavors": {"nvidia.com/gpu": "h100-sxm"},
                    "resourceUsage": {"nvidia.com/gpu": "20", "cpu": "40"},
                    "count": 10,
                }
            ],
        }
    return {
        "apiVersion": "kueue.x-k8s.io/v1beta2",
        "kind": "Workload",
        "metadata": {
            "namespace": "tenant-a",
            "name": kueue_scaling_workload_name(
                target_kind="Deployment",
                target_namespace="tenant-a",
                target_name="model-a-runners",
                target_uid="target-uid-a",
            ),
            "uid": "workload-uid-a",
            "generation": 3,
            "resourceVersion": "1402",
            "annotations": {
                "kairyu.ai/scale-target-kind": "Deployment",
                "kairyu.ai/scale-target-namespace": "tenant-a",
                "kairyu.ai/scale-target-name": "model-a-runners",
                "kairyu.ai/scale-target-uid": "target-uid-a",
            },
        },
        "spec": {
            "queueName": "serving",
            "priorityClassRef": {
                "group": "kueue.x-k8s.io",
                "kind": "WorkloadPriorityClass",
                "name": "interactive-serving",
            },
            "priority": 1000,
        },
        "status": status,
    }


def test_parse_admitted_kueue_workload_captures_exact_capacity_identity() -> None:
    admission = parse_kueue_scaling_admission(
        _workload_payload(),
        pod_set_name="runners",
    )

    assert admission.admitted is True
    assert admission.admitted_gpus == 20
    assert admission.cluster_queue == "tenant-a-gpu"
    assert admission.resource_flavor == "h100-sxm"
    assert admission.priority == 1000
    assert admission.priority_class_name == "interactive-serving"
    assert admission.workload_generation == 3
    assert admission.target_kind == "Deployment"
    assert admission.target_name == "model-a-runners"


def test_parse_non_admitted_kueue_workload_fails_closed_without_assignment() -> None:
    admission = parse_kueue_scaling_admission(
        _workload_payload(admitted=False),
        pod_set_name="runners",
    )

    assert admission.admitted is False
    assert admission.admitted_gpus == 0
    assert admission.cluster_queue is None
    assert admission.resource_flavor is None

    inactive_payload = _workload_payload(admitted=True)
    inactive_payload["spec"]["active"] = False
    inactive = parse_kueue_scaling_admission(
        inactive_payload,
        pod_set_name="runners",
    )
    assert inactive.admitted is False
    assert inactive.admitted_gpus == 0


def test_kueue_workload_name_prevents_sequential_target_rebinding() -> None:
    payload = _workload_payload()
    payload["metadata"]["annotations"]["kairyu.ai/scale-target-name"] = (
        "other-model-runners"
    )

    with pytest.raises(ValueError, match="immutable scale target identity"):
        parse_kueue_scaling_admission(payload, pod_set_name="runners")


def test_parse_v1beta1_priority_class_name() -> None:
    payload = _workload_payload()
    payload["apiVersion"] = "kueue.x-k8s.io/v1beta1"
    payload["spec"]["priorityClassName"] = "interactive-serving"
    payload["spec"]["priorityClassSource"] = (
        "kueue.x-k8s.io/workloadpriorityclass"
    )
    del payload["spec"]["priorityClassRef"]

    admission = parse_kueue_scaling_admission(payload, pod_set_name="runners")

    assert admission.api_version == "kueue.x-k8s.io/v1beta1"
    assert admission.priority_class_name == "interactive-serving"
    assert admission.priority_class_source == "kueue.x-k8s.io/workloadpriorityclass"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["status"]["admission"]["podSetAssignments"].append(
                payload["status"]["admission"]["podSetAssignments"][0].copy()
            ),
            "exactly once",
        ),
        (
            lambda payload: payload["status"]["admission"]["podSetAssignments"][
                0
            ]["resourceUsage"].__setitem__("nvidia.com/gpu", "1500m"),
            "whole GPU",
        ),
        (
            lambda payload: payload.__setitem__("apiVersion", "kueue.x-k8s.io/v1"),
            "unsupported apiVersion",
        ),
        (
            lambda payload: payload["spec"].__delitem__("priority"),
            "signed 32-bit",
        ),
        (
            lambda payload: payload["status"]["conditions"].append(
                {"type": "Admitted", "status": "False"}
            ),
            "duplicate Admitted",
        ),
        (
            lambda payload: payload["metadata"]["annotations"].pop(
                "kairyu.ai/scale-target-uid"
            ),
            "scale-target-uid",
        ),
    ],
)
def test_parse_kueue_workload_rejects_ambiguous_or_unsupported_capacity(
    mutation,
    message: str,
) -> None:
    payload = _workload_payload()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        parse_kueue_scaling_admission(payload, pod_set_name="runners")


def test_quota_admission_applies_all_nested_and_kueue_ceilings() -> None:
    admission = admit_scaling_quota(
        _snapshot(),
        current_replicas=4,
        requested_replicas=10,
    )

    assert admission.quota_ceiling_replicas == 8
    assert admission.admitted_replicas == 8
    assert admission.constrained_by == (
        ScalingQuotaConstraint.TENANT_MODEL,
        ScalingQuotaConstraint.KUEUE_ADMISSION,
    )
    assert admission.constrained is True


def test_unconstrained_admission_preserves_requested_replicas() -> None:
    admission = admit_scaling_quota(
        _snapshot(),
        current_replicas=4,
        requested_replicas=7,
    )

    assert admission.admitted_replicas == 7
    assert admission.constrained_by == ()
    assert admission.constrained is False


def test_higher_priority_reservation_reduces_available_capacity() -> None:
    limits = list(_snapshot().limits)
    limits[0] = limits[0].model_copy(
        update={"reserved_gpus_for_higher_priority": 34}
    )

    admission = admit_scaling_quota(
        _snapshot(limits=tuple(limits), kueue=_kueue(admitted_gpus=10)),
        current_replicas=4,
        requested_replicas=8,
    )

    assert admission.quota_ceiling_replicas == 5
    assert admission.admitted_replicas == 5
    assert admission.constrained_by == (
        ScalingQuotaConstraint.CLUSTER,
        ScalingQuotaConstraint.KUEUE_ADMISSION,
    )


def test_quota_shortfall_never_turns_into_scale_down() -> None:
    limits = tuple(
        limit.model_copy(update={"used_gpus_excluding_target": limit.hard_limit_gpus})
        for limit in _snapshot().limits
    )
    admission = admit_scaling_quota(
        _snapshot(limits=limits, kueue=_kueue(admitted=False, admitted_gpus=0)),
        current_replicas=4,
        requested_replicas=6,
    )

    assert admission.quota_ceiling_replicas == 0
    assert admission.admitted_replicas == 4
    assert admission.constrained_by == (
        ScalingQuotaConstraint.CLUSTER,
        ScalingQuotaConstraint.MODEL_FAMILY,
        ScalingQuotaConstraint.TENANT_MODEL,
        ScalingQuotaConstraint.KUEUE_ADMISSION,
    )


def test_missing_kueue_admission_holds_at_current_replicas() -> None:
    admission = admit_scaling_quota(
        _snapshot(kueue=_kueue(admitted=False, admitted_gpus=0)),
        current_replicas=4,
        requested_replicas=7,
    )

    assert admission.quota_ceiling_replicas == 0
    assert admission.admitted_replicas == 4
    assert admission.constrained_by == (ScalingQuotaConstraint.KUEUE_ADMISSION,)


def test_snapshot_requires_all_nested_scopes_in_canonical_order() -> None:
    limits = _snapshot().limits
    with pytest.raises(ValidationError, match="canonical order"):
        _snapshot(limits=(limits[1], limits[0], limits[2]))


def test_target_reservation_must_fit_every_hard_limit() -> None:
    payload = _snapshot().model_dump()
    payload["target_reserved_gpus"] = 25
    with pytest.raises(ValidationError, match="authoritative Kueue admission"):
        ScalingQuotaSnapshot.model_validate(payload)


def test_kueue_reservation_must_fit_every_nested_hard_limit() -> None:
    payload = _snapshot().model_dump()
    payload["limits"][2]["hard_limit_gpus"] = 23

    with pytest.raises(ValidationError, match="fit every nested hard limit"):
        ScalingQuotaSnapshot.model_validate(payload)


@pytest.mark.parametrize("observed_generation", [None, 2, 4])
def test_parse_admitted_kueue_workload_rejects_stale_condition_generation(
    observed_generation: int | None,
) -> None:
    payload = _workload_payload()
    condition = payload["status"]["conditions"][0]
    if observed_generation is None:
        del condition["observedGeneration"]
    else:
        condition["observedGeneration"] = observed_generation

    with pytest.raises(ValueError, match="observe metadata.generation"):
        parse_kueue_scaling_admission(payload, pod_set_name="runners")


def test_kueue_pod_and_gpu_capacity_must_match_replica_shape() -> None:
    payload = _snapshot().model_dump()
    payload["gpus_per_replica"] = 3
    with pytest.raises(ValidationError, match="match gpus_per_replica"):
        ScalingQuotaSnapshot.model_validate(payload)


@pytest.mark.parametrize(
    ("admitted", "admitted_gpus"),
    [(False, 1), (True, 0)],
)
def test_kueue_admission_state_and_quantity_are_consistent(
    admitted: bool,
    admitted_gpus: int,
) -> None:
    with pytest.raises(ValidationError, match="positive|zero"):
        _kueue(admitted=admitted, admitted_gpus=admitted_gpus)


def test_admission_result_cannot_forge_ceiling_or_constraints() -> None:
    admission = admit_scaling_quota(
        _snapshot(),
        current_replicas=4,
        requested_replicas=10,
    )
    payload = admission.model_dump()
    payload["admitted_replicas"] = 9
    with pytest.raises(ValidationError, match="fail-safe quota clamp"):
        ScalingQuotaAdmission.model_validate(payload)

    payload = admission.model_dump()
    payload["constrained_by"] = []
    with pytest.raises(ValidationError, match="every limiting quota"):
        ScalingQuotaAdmission.model_validate(payload)


@pytest.mark.parametrize("current", [True, 1.5, -1, 100_001])
def test_admission_rejects_invalid_current_replica_values(current: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        admit_scaling_quota(
            _snapshot(),
            current_replicas=current,  # type: ignore[arg-type]
            requested_replicas=10,
        )


def test_admission_only_accepts_scale_up_intents() -> None:
    with pytest.raises(ValueError, match="only accepts scale-up"):
        admit_scaling_quota(
            _snapshot(),
            current_replicas=4,
            requested_replicas=4,
        )
