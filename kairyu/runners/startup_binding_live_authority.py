"""Live scaling-controller authorization for cache-startup bindings."""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.artifacts.placement_hint import NodeModelCachePlacementHintSnapshot
from kairyu.runners.leadership import (
    LeaderFencedRunnerController,
    RunnerWriterAuthority,
)
from kairyu.runners.prestage import NodeModelPrestageRecord
from kairyu.runners.prewarm import ModelCachePlacementState, ScalingPrewarmPlan
from kairyu.runners.scaling_log import (
    ScalingDecisionAction,
    ScalingDecisionRecord,
)
from kairyu.runners.scaling_quota import ScalingQuotaAdmission
from kairyu.runners.startup_binding import RunnerCacheStartupBinding
from kairyu.runners.startup_binding_authority import (
    RunnerCachePlacementBindingAuthorizationDeniedError,
)


def _aware(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


class RunnerCachePlacementBindingTargetState(BaseModel):
    """Live workload state after the binding's fenced scale write."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    observed_at: datetime
    target_kind: Literal["Deployment", "StatefulSet"]
    namespace: str = Field(max_length=253)
    name: str = Field(max_length=253)
    workload_uid: str = Field(max_length=255)
    workload_generation: int = Field(ge=1, le=2**63 - 1)
    release_id: str = Field(max_length=255)
    model_id: str = Field(max_length=255)
    model_revision: str = Field(max_length=255)
    placement_binding_id: str = Field(max_length=255)
    decision_generation: int = Field(ge=1, le=2**63 - 1)
    decision_id: str = Field(max_length=255)
    decision_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    replicas: int = Field(ge=0, le=100_000)

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="target observed_at")

    @field_validator(
        "namespace",
        "name",
        "workload_uid",
        "release_id",
        "model_id",
        "model_revision",
        "placement_binding_id",
        "decision_id",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError(f"{info.field_name} must be a non-empty string without NUL")
        return value

    @field_validator(
        "workload_generation",
        "decision_generation",
        "replicas",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        return value


class RunnerCachePlacementBindingPinEvidence(BaseModel):
    """Owner-scoped cache-index evidence from the same snapshot as a node hint."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["runner-cache-placement-binding-pin-evidence-v1"] = (
        "runner-cache-placement-binding-pin-evidence-v1"
    )
    node_id: str = Field(max_length=253)
    index_revision: int = Field(ge=1, le=2**63 - 1)
    observed_at: datetime
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str = Field(max_length=255)
    model_revision: str = Field(max_length=255)
    record_generation: int = Field(ge=1, le=2**63 - 1)
    pin_owners: tuple[str, ...] = Field(min_length=1, max_length=10_000)

    @field_validator("node_id", "model_id", "model_revision")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError(f"{info.field_name} must be a non-empty string without NUL")
        return value

    @field_validator("index_revision", "record_generation", mode="before")
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        return value

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="pin evidence observed_at")

    @field_validator("pin_owners")
    @classmethod
    def validate_pin_owners(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not owner.strip() or "\x00" in owner or len(owner) > 255 for owner in value):
            raise ValueError("pin owners must be non-empty bounded strings without NUL")
        if value != tuple(sorted(set(value))):
            raise ValueError("pin owners must be sorted and unique")
        return value


class RunnerCachePlacementBindingLiveState(BaseModel):
    """One independently refreshed scaling-controller authorization snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    observed_at: datetime
    binding: RunnerCacheStartupBinding
    decision: ScalingDecisionRecord
    target: RunnerCachePlacementBindingTargetState
    quota_admission: ScalingQuotaAdmission
    prewarm_plan: ScalingPrewarmPlan
    prestage_records: tuple[NodeModelPrestageRecord, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    placement_hints: tuple[NodeModelCachePlacementHintSnapshot, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    pin_evidence: tuple[RunnerCachePlacementBindingPinEvidence, ...] = Field(
        min_length=1,
        max_length=100_000,
    )

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="observed_at")

    @model_validator(mode="after")
    def validate_live_evidence_order(self) -> RunnerCachePlacementBindingLiveState:
        placement_ids = tuple(record.command.placement_id for record in self.prestage_records)
        if placement_ids != tuple(sorted(placement_ids)) or len(set(placement_ids)) != len(
            placement_ids
        ):
            raise ValueError("prestage records must use unique canonical placement IDs")
        node_ids = tuple(hint.node_id for hint in self.placement_hints)
        if node_ids != tuple(sorted(node_ids)) or len(set(node_ids)) != len(node_ids):
            raise ValueError("placement hints must use unique canonical node IDs")
        pin_nodes = tuple(evidence.node_id for evidence in self.pin_evidence)
        if pin_nodes != tuple(sorted(pin_nodes)) or len(set(pin_nodes)) != len(pin_nodes):
            raise ValueError("pin evidence must use unique canonical node IDs")
        if (
            self.target.observed_at > self.observed_at
            or self.quota_admission.snapshot.observed_at > self.observed_at
            or self.prewarm_plan.snapshot.observed_at > self.observed_at
            or any(record.updated_at > self.observed_at for record in self.prestage_records)
            or any(hint.observed_at > self.observed_at for hint in self.placement_hints)
            or any(evidence.observed_at > self.observed_at for evidence in self.pin_evidence)
        ):
            raise ValueError("live evidence cannot be newer than its aggregate snapshot")
        return self


@runtime_checkable
class RunnerCachePlacementBindingLiveStateSource(Protocol):
    """Deadline-aware reads owned by the process with live scaling state."""

    def read(
        self,
        binding: RunnerCacheStartupBinding,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> RunnerCachePlacementBindingLiveState: ...

    def readiness(
        self,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> None: ...


class ScalingControllerPlacementBindingAuthority:
    """Authorize exact bindings under current leader and scaling-state fences.

    The state source belongs to the hosting scaling controller.  It must perform
    fresh, deadline-bounded reads of the durable decision, live target, quota,
    prewarm, binding, and cache state before returning a snapshot.  This class
    validates that snapshot under the controller's current leader lease and
    rechecks the lease after the reads complete.
    """

    def __init__(
        self,
        controller: LeaderFencedRunnerController,
        source: RunnerCachePlacementBindingLiveStateSource,
        *,
        monotonic_clock=time.monotonic,
    ) -> None:
        if not isinstance(controller, LeaderFencedRunnerController):
            raise TypeError("controller must be a LeaderFencedRunnerController")
        if not isinstance(source, RunnerCachePlacementBindingLiveStateSource):
            raise TypeError("source must implement RunnerCachePlacementBindingLiveStateSource")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        self._controller = controller
        self._source = source
        self._monotonic_clock = monotonic_clock

    def _check_deadline(self, deadline_monotonic: float) -> None:
        if self._monotonic_clock() >= deadline_monotonic:
            raise TimeoutError("placement-binding authority deadline expired")

    @staticmethod
    def _validate_budget(deadline_monotonic: float, backend_timeout_s: float) -> None:
        for name, value in (
            ("deadline_monotonic", deadline_monotonic),
            ("backend_timeout_s", backend_timeout_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TypeError(f"{name} must be a finite number")
        if backend_timeout_s <= 0:
            raise ValueError("backend_timeout_s must be positive")

    @staticmethod
    def _deny(message: str) -> None:
        raise RunnerCachePlacementBindingAuthorizationDeniedError(message)

    @classmethod
    def _validate_decision(
        cls,
        binding: RunnerCacheStartupBinding,
        decision: ScalingDecisionRecord,
        authority: RunnerWriterAuthority,
    ) -> None:
        target = decision.target_revision
        quota = decision.quota_admission
        prewarm = decision.prewarm_plan
        if (
            decision.action is not ScalingDecisionAction.SCALE_UP
            or decision.decision_generation is None
            or target is None
            or quota is None
            or prewarm is None
        ):
            cls._deny("binding decision is not a durable prewarmed scale-up")
        assert target is not None
        assert prewarm is not None
        expected_target_id = f"{target.target_kind.lower()}/{target.namespace}/{target.name}"
        if (
            target.election_id != authority.election_id
            or target.fencing_token != authority.fencing_token
            or binding.decision_id != decision.decision_id
            or binding.decision_fingerprint != decision.fingerprint
            or binding.target_id != expected_target_id
            or binding.target_revision != target.workload_generation
            or binding.model_class != decision.window.model_class
            or binding.model_revision != target.model_revision
            or binding.manifest_digest != prewarm.snapshot.artifact_digest
            or binding.placement_binding_id != prewarm.snapshot.placement_binding_id
            or binding.prewarm_snapshot_id != prewarm.snapshot.snapshot_id
            or binding.prewarm_cache_revision != prewarm.snapshot.cache_revision
            or binding.bound_at < decision.decided_at
            or len(binding.placements) != decision.target_delta
        ):
            cls._deny("binding does not match the live durable scaling decision")
        planned = {
            placement.placement_id: placement
            for placement in prewarm.snapshot.placements
            if placement.placement_id in prewarm.runner_start_placement_ids
        }
        if tuple(placement.placement_id for placement in binding.placements) != tuple(
            sorted(prewarm.runner_start_placement_ids)
        ) or any(
            placement.placement_id not in planned
            or (
                placement.node_name,
                placement.resource_flavor,
                placement.profile_id,
                placement.compatibility_approval_id,
            )
            != (
                planned[placement.placement_id].node_name,
                planned[placement.placement_id].resource_flavor,
                planned[placement.placement_id].profile_id,
                planned[placement.placement_id].compatibility_approval_id,
            )
            or (
                placement.hint_observed_at,
                placement.hint_valid_until,
                placement.hint_index_revision,
            )
            != (
                planned[placement.placement_id].cache_hint_observed_at,
                planned[placement.placement_id].cache_hint_valid_until,
                planned[placement.placement_id].cache_hint_index_revision,
            )
            for placement in binding.placements
        ):
            cls._deny("binding placements do not match the live prewarm decision")

    @classmethod
    def _validate_quota(
        cls,
        decision: ScalingDecisionRecord,
        refreshed: ScalingQuotaAdmission,
        *,
        validated_at: datetime,
    ) -> None:
        original = decision.quota_admission
        assert original is not None
        age = (validated_at - refreshed.snapshot.observed_at).total_seconds()
        original_kueue = original.snapshot.kueue
        refreshed_kueue = refreshed.snapshot.kueue
        identity_fields = (
            "api_version",
            "namespace",
            "workload_name",
            "workload_uid",
            "workload_generation",
            "target_kind",
            "target_namespace",
            "target_name",
            "target_uid",
            "local_queue",
            "cluster_queue",
            "pod_set_name",
            "resource_flavor",
            "resource_name",
            "priority_class_name",
            "priority_class_group",
            "priority_class_kind",
            "priority_class_source",
            "priority",
        )
        if (
            not 0 <= age <= decision.policy.max_observation_age_seconds
            or refreshed.snapshot.tenant_id != original.snapshot.tenant_id
            or refreshed.snapshot.model_class != original.snapshot.model_class
            or refreshed.snapshot.model_family != original.snapshot.model_family
            or refreshed.snapshot.gpus_per_replica != original.snapshot.gpus_per_replica
            or refreshed.current_replicas != original.current_replicas
            or refreshed.requested_replicas != original.requested_replicas
            or refreshed.snapshot.observed_at < original.snapshot.observed_at
            or refreshed.snapshot.quota_revision < original.snapshot.quota_revision
            or any(
                getattr(refreshed_kueue, field) != getattr(original_kueue, field)
                for field in identity_fields
            )
            or not refreshed_kueue.admitted
            or refreshed.admitted_replicas < decision.desired_replicas
        ):
            cls._deny("quota no longer authorizes the cache-bound scale-up")

    @classmethod
    def _validate_target(
        cls,
        binding: RunnerCacheStartupBinding,
        decision: ScalingDecisionRecord,
        target_state: RunnerCachePlacementBindingTargetState,
        *,
        validated_at: datetime,
    ) -> None:
        target = decision.target_revision
        assert target is not None
        assert decision.decision_generation is not None
        age = (validated_at - target_state.observed_at).total_seconds()
        if (
            not 0 <= age <= decision.policy.max_observation_age_seconds
            or target_state.target_kind != target.target_kind
            or target_state.namespace != target.namespace
            or target_state.name != target.name
            or target_state.workload_uid != target.workload_uid
            or target_state.workload_generation != target.workload_generation + 1
            or target_state.release_id != target.release_id
            or target_state.model_id != binding.model_id
            or target_state.model_revision != target.model_revision
            or target_state.placement_binding_id != binding.placement_binding_id
            or target_state.decision_generation != decision.decision_generation
            or target_state.decision_id != decision.decision_id
            or target_state.decision_fingerprint != decision.fingerprint
            or target_state.replicas != decision.desired_replicas
        ):
            cls._deny("live target no longer carries the binding's fenced scale decision")

    @classmethod
    def _validate_prewarm(
        cls,
        binding: RunnerCacheStartupBinding,
        decision: ScalingDecisionRecord,
        refreshed: ScalingPrewarmPlan,
        *,
        validated_at: datetime,
    ) -> None:
        original = decision.prewarm_plan
        assert original is not None
        age = (validated_at - refreshed.snapshot.observed_at).total_seconds()
        if (
            not 0 <= age <= decision.policy.max_observation_age_seconds
            or refreshed.snapshot.model_class != original.snapshot.model_class
            or refreshed.snapshot.model_revision != original.snapshot.model_revision
            or refreshed.snapshot.artifact_digest != original.snapshot.artifact_digest
            or refreshed.snapshot.placement_binding_id != original.snapshot.placement_binding_id
            or refreshed.resource_flavor != original.resource_flavor
            or refreshed.current_replicas != original.current_replicas
            or refreshed.quota_target_replicas != original.quota_target_replicas
            or refreshed.snapshot.observed_at < original.snapshot.observed_at
            or refreshed.snapshot.cache_revision < original.snapshot.cache_revision
            or refreshed.runner_start_placement_ids != original.runner_start_placement_ids
        ):
            cls._deny("prewarm authority changed for the cache-bound scale-up")
        original_placements = {
            placement.placement_id: placement
            for placement in original.snapshot.placements
            if placement.placement_id in original.runner_start_placement_ids
        }
        refreshed_placements = {
            placement.placement_id: placement for placement in refreshed.snapshot.placements
        }
        binding_placements = {placement.placement_id: placement for placement in binding.placements}
        if refreshed.runner_target_replicas < decision.desired_replicas or not all(
            placement_id in refreshed_placements
            and (
                refreshed_placements[placement_id].node_name,
                refreshed_placements[placement_id].resource_flavor,
                refreshed_placements[placement_id].profile_id,
                refreshed_placements[placement_id].compatibility_approval_id,
            )
            == (
                original_placement.node_name,
                original_placement.resource_flavor,
                original_placement.profile_id,
                original_placement.compatibility_approval_id,
            )
            and not refreshed_placements[placement_id].assigned
            and refreshed_placements[placement_id].healthy
            and refreshed_placements[placement_id].schedulable
            and refreshed_placements[placement_id].state is ModelCachePlacementState.READY
            and refreshed_placements[placement_id].cache_hint_observed_at is not None
            and refreshed_placements[placement_id].cache_hint_valid_until is not None
            and refreshed_placements[placement_id].cache_hint_index_revision is not None
            and refreshed_placements[placement_id].cache_hint_observed_at
            >= binding_placements[placement_id].hint_observed_at
            and refreshed_placements[placement_id].cache_hint_index_revision
            >= binding_placements[placement_id].hint_index_revision
            and validated_at < refreshed_placements[placement_id].cache_hint_valid_until
            for placement_id, original_placement in original_placements.items()
        ):
            cls._deny("prewarm authority no longer provides live cache capacity")

    @classmethod
    def _validate_live_cache_evidence(
        cls,
        binding: RunnerCacheStartupBinding,
        decision: ScalingDecisionRecord,
        state: RunnerCachePlacementBindingLiveState,
        *,
        validated_at: datetime,
    ) -> None:
        records = {record.command.placement_id: record for record in state.prestage_records}
        hints = {hint.node_id: hint for hint in state.placement_hints}
        pin_evidence = {evidence.node_id: evidence for evidence in state.pin_evidence}
        refreshed = {
            placement.placement_id: placement
            for placement in state.prewarm_plan.snapshot.placements
        }
        target = decision.target_revision
        assert target is not None
        for placement in binding.placements:
            record = records.get(placement.placement_id)
            hint = hints.get(placement.node_name)
            pin = pin_evidence.get(placement.node_name)
            live_placement = refreshed.get(placement.placement_id)
            if record is None or hint is None or pin is None or live_placement is None:
                cls._deny("live cache evidence is incomplete for a bound placement")
            assert record is not None
            assert hint is not None
            assert pin is not None
            assert live_placement is not None
            command = record.command
            resident = hint.resident_for(
                manifest_digest=binding.manifest_digest,
                model_id=binding.model_id,
                model_revision=binding.model_revision,
            )
            if (
                record.state is not ModelCachePlacementState.READY
                or record.pin_record_generation != placement.resident_record_generation
                or command.action != "ensure"
                or command.command_id != placement.prestage_command_id
                or command.command_generation != placement.prestage_command_generation
                or command.decision_id != binding.decision_id
                or command.decision_fingerprint != binding.decision_fingerprint
                or command.target_id != binding.target_id
                or command.target_revision != binding.target_revision
                or command.deployment_id != binding.deployment_id
                or command.placement_binding_id != binding.placement_binding_id
                or command.snapshot_id != binding.prewarm_snapshot_id
                or command.cache_revision != binding.prewarm_cache_revision
                or command.placement_id != placement.placement_id
                or command.node_id != placement.node_name
                or command.resource_flavor != placement.resource_flavor
                or command.profile_id != placement.profile_id
                or command.compatibility_approval_id != placement.compatibility_approval_id
                or command.model_id != binding.model_id
                or command.model_revision != binding.model_revision
                or command.manifest_digest != binding.manifest_digest
                or command.pin_owner != placement.pin_owner
                or command.authority.election_id != target.election_id
                or command.authority.fencing_token != target.fencing_token
                or record.updated_at > hint.observed_at
                or hint.index_revision < placement.hint_index_revision
                or hint.observed_at < placement.hint_observed_at
                or not hint.observed_at <= state.observed_at
                or not validated_at < hint.valid_until
                or resident is None
                or not resident.pinned
                or resident.record_generation != placement.resident_record_generation
                or pin.index_revision != hint.index_revision
                or pin.observed_at != hint.observed_at
                or pin.manifest_digest != binding.manifest_digest
                or pin.model_id != binding.model_id
                or pin.model_revision != binding.model_revision
                or pin.record_generation != placement.resident_record_generation
                or placement.pin_owner not in pin.pin_owners
                or live_placement.cache_hint_observed_at != hint.observed_at
                or live_placement.cache_hint_valid_until != hint.valid_until
                or live_placement.cache_hint_index_revision != hint.index_revision
            ):
                cls._deny("live pin, prestage, or cache-hint lineage changed")

    @classmethod
    def _validate_state(
        cls,
        candidate: RunnerCacheStartupBinding,
        state: RunnerCachePlacementBindingLiveState,
        *,
        authority: RunnerWriterAuthority,
    ) -> None:
        if state.binding != candidate:
            cls._deny("candidate is not the scaling controller's current binding")
        if not candidate.bound_at <= authority.validated_at < candidate.valid_until:
            cls._deny("candidate binding is outside its live authorization window")
        age = (authority.validated_at - state.observed_at).total_seconds()
        if (
            state.observed_at < candidate.bound_at
            or not 0 <= age <= state.decision.policy.max_observation_age_seconds
        ):
            cls._deny("live scaling state is stale or temporally inconsistent")
        cls._validate_decision(candidate, state.decision, authority)
        cls._validate_target(
            candidate,
            state.decision,
            state.target,
            validated_at=authority.validated_at,
        )
        cls._validate_quota(
            state.decision,
            state.quota_admission,
            validated_at=authority.validated_at,
        )
        cls._validate_prewarm(
            candidate,
            state.decision,
            state.prewarm_plan,
            validated_at=authority.validated_at,
        )
        cls._validate_live_cache_evidence(
            candidate,
            state.decision,
            state,
            validated_at=authority.validated_at,
        )

    def reauthorize(
        self,
        binding: RunnerCacheStartupBinding,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> RunnerCacheStartupBinding:
        """Return only an exact binding still authorized by all live fences."""

        if not isinstance(binding, RunnerCacheStartupBinding):
            raise TypeError("binding must be a RunnerCacheStartupBinding")
        binding = RunnerCacheStartupBinding.model_validate(binding.model_dump())
        self._validate_budget(deadline_monotonic, backend_timeout_s)
        self._check_deadline(deadline_monotonic)

        def authorize(initial: RunnerWriterAuthority) -> RunnerCacheStartupBinding:
            self._check_deadline(deadline_monotonic)
            state = self._source.read(
                binding,
                deadline_monotonic=deadline_monotonic,
                backend_timeout_s=backend_timeout_s,
            )
            self._check_deadline(deadline_monotonic)
            if not isinstance(state, RunnerCachePlacementBindingLiveState):
                raise TypeError("live-state source returned an invalid state")
            state = RunnerCachePlacementBindingLiveState.model_validate(state.model_dump())

            def finish(refreshed: RunnerWriterAuthority) -> RunnerCacheStartupBinding:
                self._check_deadline(deadline_monotonic)
                if (
                    refreshed.tenure != initial.tenure
                    or refreshed.validated_at < initial.validated_at
                ):
                    self._deny("leader authority changed during binding authorization")
                self._validate_state(binding, state, authority=refreshed)
                self._check_deadline(deadline_monotonic)
                return RunnerCacheStartupBinding.model_validate(binding.model_dump())

            return self._controller.mutate_autoscaler(finish)

        return self._controller.mutate_autoscaler(authorize)

    def readiness(
        self,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> None:
        """Require both current leadership and live-state dependencies."""

        self._validate_budget(deadline_monotonic, backend_timeout_s)
        self._check_deadline(deadline_monotonic)

        def ready(initial: RunnerWriterAuthority) -> None:
            self._source.readiness(
                deadline_monotonic=deadline_monotonic,
                backend_timeout_s=backend_timeout_s,
            )
            self._check_deadline(deadline_monotonic)

            def finish(refreshed: RunnerWriterAuthority) -> None:
                if refreshed.tenure != initial.tenure:
                    raise RuntimeError("leader authority changed during readiness check")
                self._check_deadline(deadline_monotonic)

            self._controller.mutate_autoscaler(finish)

        self._controller.mutate_autoscaler(ready)


__all__ = [
    "RunnerCachePlacementBindingLiveState",
    "RunnerCachePlacementBindingLiveStateSource",
    "RunnerCachePlacementBindingPinEvidence",
    "RunnerCachePlacementBindingTargetState",
    "ScalingControllerPlacementBindingAuthority",
]
