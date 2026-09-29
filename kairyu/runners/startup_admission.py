"""Per-Pod CREATE admission for incremental cache-bound Runner scale-up."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.runners.startup_binding import RunnerCacheStartupBinding
from kairyu.runners.startup_metadata import RUNNER_CACHE_STARTUP_TARGET_ANNOTATION
from kairyu.runners.startup_scheduling import (
    RunnerCacheSchedulingError,
    admit_runner_cache_placement_for_gated_pod,
)


class RunnerCachePlacementAdmissionError(RuntimeError):
    """A gated Pod cannot consume an incremental startup placement."""


class RunnerCachePlacementAdmissionConflictError(RunnerCachePlacementAdmissionError):
    """Admission state changed or no unique placement remains."""


class RunnerCachePlacementAdmissionTimeoutError(RuntimeError):
    """The internal admission deadline expired before a safe response."""


def _text(value: str, *, name: str, max_length: int = 255) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL")
    if len(value) > max_length:
        raise ValueError(f"{name} exceeds maximum length")
    return value


def _aware(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


class RunnerCachePlacementAdmissionPlan(BaseModel):
    """One active binding published before an incremental replica increase."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    binding: RunnerCacheStartupBinding
    release_id: str = Field(max_length=512)
    namespace: str = Field(max_length=253)
    owner_api_version: str = Field(max_length=255)
    owner_kind: str = Field(max_length=63)
    owner_name: str = Field(max_length=253)
    owner_uid: str = Field(max_length=255)
    creator_username: str = Field(max_length=255)
    registered_at: datetime

    @field_validator(
        "release_id",
        "namespace",
        "owner_api_version",
        "owner_kind",
        "owner_name",
        "owner_uid",
        "creator_username",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        maximum = 512 if info.field_name == "release_id" else 253
        if info.field_name in {"owner_api_version", "owner_uid", "creator_username"}:
            maximum = 255
        if info.field_name == "owner_kind":
            maximum = 63
        return _text(value, name=info.field_name, max_length=maximum)

    @field_validator("registered_at")
    @classmethod
    def validate_registered_at(cls, value: datetime) -> datetime:
        return _aware(value, name="registered_at")

    @model_validator(mode="after")
    def validate_window(self) -> RunnerCachePlacementAdmissionPlan:
        if not self.binding.bound_at <= self.registered_at < self.binding.valid_until:
            raise ValueError("admission plan must be registered while its binding is live")
        return self


class RunnerCachePlacementAdmissionClaim(BaseModel):
    """Idempotent assignment of one API-server Pod name to one placement."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    binding_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_id: str = Field(max_length=255)
    pod_key: str = Field(max_length=512)
    admission_uid: str = Field(max_length=255)
    placement_id: str = Field(max_length=255)
    claimed_at: datetime

    @field_validator("target_id", "pod_key", "admission_uid", "placement_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        maximum = 512 if info.field_name == "pod_key" else 255
        return _text(value, name=info.field_name, max_length=maximum)

    @field_validator("claimed_at")
    @classmethod
    def validate_claimed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="claimed_at")


class RunnerCachePlacementAdmissionAllocation(BaseModel):
    """Claim result that distinguishes a new CAS write from an exact replay."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    claim: RunnerCachePlacementAdmissionClaim
    created: bool


@runtime_checkable
class RunnerCachePlacementAdmissionStore(Protocol):
    """Linearizable plan/claim boundary required by an admission replica set."""

    def register(self, plan: RunnerCachePlacementAdmissionPlan) -> None: ...

    def resolve(self, target_id: str) -> RunnerCachePlacementAdmissionPlan: ...

    def claim(
        self,
        *,
        target_id: str,
        binding_id: str,
        pod_key: str,
        admission_uid: str,
        claimed_at: datetime,
    ) -> RunnerCachePlacementAdmissionAllocation: ...

    def release(self, claim: RunnerCachePlacementAdmissionClaim) -> None: ...


class InMemoryRunnerCachePlacementAdmissionStore:
    """Thread-safe executable specification; production requires a shared store."""

    def __init__(
        self,
        *,
        max_targets: int = 10_000,
        replay_safety_window: timedelta = timedelta(minutes=5),
    ) -> None:
        if type(max_targets) is not int or max_targets < 1:
            raise ValueError("max_targets must be a positive integer")
        if (
            not isinstance(replay_safety_window, timedelta)
            or replay_safety_window <= timedelta(0)
        ):
            raise ValueError("replay_safety_window must be positive")
        self._max_targets = max_targets
        self._replay_safety_window = replay_safety_window
        self._plans: dict[str, RunnerCachePlacementAdmissionPlan] = {}
        self._claims: dict[str, dict[str, RunnerCachePlacementAdmissionClaim]] = {}
        self._protected_claims: set[tuple[str, str]] = set()
        self._lock = threading.RLock()

    def register(self, plan: RunnerCachePlacementAdmissionPlan) -> None:
        if not isinstance(plan, RunnerCachePlacementAdmissionPlan):
            raise TypeError("plan must be a RunnerCachePlacementAdmissionPlan")
        plan = RunnerCachePlacementAdmissionPlan.model_validate(plan.model_dump())
        target_id = plan.binding.target_id
        with self._lock:
            previous = self._plans.get(target_id)
            if previous is not None and previous.binding.binding_id == plan.binding.binding_id:
                if previous != plan:
                    raise RunnerCachePlacementAdmissionConflictError(
                        "binding ID is already registered with different plan evidence"
                    )
                return
            prior_claims = self._claims.get(target_id, {})
            replace_after = (
                previous.binding.valid_until
                + (self._replay_safety_window if prior_claims else timedelta(0))
                if previous is not None
                else None
            )
            if replace_after is not None and plan.registered_at < replace_after:
                raise RunnerCachePlacementAdmissionConflictError(
                    "admission plan cannot be replaced before its replay safety window"
                )
            if previous is None and len(self._plans) >= self._max_targets:
                raise RunnerCachePlacementAdmissionConflictError(
                    "admission plan capacity is exhausted"
                )
            self._protected_claims.difference_update(
                (target_id, pod_key) for pod_key in prior_claims
            )
            self._plans[target_id] = plan
            self._claims[target_id] = {}

    def resolve(self, target_id: str) -> RunnerCachePlacementAdmissionPlan:
        target_id = _text(target_id, name="target_id")
        with self._lock:
            try:
                plan = self._plans[target_id]
            except KeyError as error:
                raise RunnerCachePlacementAdmissionConflictError(
                    "no active cache placement admission plan"
                ) from error
            return RunnerCachePlacementAdmissionPlan.model_validate(plan.model_dump())

    def claim(
        self,
        *,
        target_id: str,
        binding_id: str,
        pod_key: str,
        admission_uid: str,
        claimed_at: datetime,
    ) -> RunnerCachePlacementAdmissionAllocation:
        target_id = _text(target_id, name="target_id")
        binding_id = _text(binding_id, name="binding_id")
        pod_key = _text(pod_key, name="pod_key", max_length=512)
        admission_uid = _text(admission_uid, name="admission_uid")
        claimed_at = _aware(claimed_at, name="claimed_at")
        with self._lock:
            plan = self._plans.get(target_id)
            if plan is None or plan.binding.binding_id != binding_id:
                raise RunnerCachePlacementAdmissionConflictError(
                    "admission plan changed before placement claim"
                )
            if not plan.binding.bound_at <= claimed_at < plan.binding.valid_until:
                raise RunnerCachePlacementAdmissionConflictError(
                    "cache placement admission binding is not live"
                )
            claims = self._claims[target_id]
            existing = claims.get(pod_key)
            if existing is not None:
                self._protected_claims.add((target_id, pod_key))
                return RunnerCachePlacementAdmissionAllocation(
                    claim=RunnerCachePlacementAdmissionClaim.model_validate(
                        existing.model_dump()
                    ),
                    created=False,
                )
            used = {claim.placement_id for claim in claims.values()}
            placement = next(
                (
                    placement
                    for placement in plan.binding.placements
                    if placement.placement_id not in used
                ),
                None,
            )
            if placement is None:
                raise RunnerCachePlacementAdmissionConflictError(
                    "cache placement admission plan is exhausted"
                )
            claim = RunnerCachePlacementAdmissionClaim(
                binding_id=binding_id,
                target_id=target_id,
                pod_key=pod_key,
                admission_uid=admission_uid,
                placement_id=placement.placement_id,
                claimed_at=claimed_at,
            )
            claims[pod_key] = claim
            return RunnerCachePlacementAdmissionAllocation(claim=claim, created=True)

    def release(self, claim: RunnerCachePlacementAdmissionClaim) -> None:
        if not isinstance(claim, RunnerCachePlacementAdmissionClaim):
            raise TypeError("claim must be a RunnerCachePlacementAdmissionClaim")
        claim = RunnerCachePlacementAdmissionClaim.model_validate(claim.model_dump())
        with self._lock:
            current = self._claims.get(claim.target_id, {}).get(claim.pod_key)
            key = (claim.target_id, claim.pod_key)
            if current == claim and key not in self._protected_claims:
                del self._claims[claim.target_id][claim.pod_key]
                self._protected_claims.discard(key)


class RunnerCachePlacementAdmissionController:
    """Resolve, reauthorize, claim, and mutate one Kubernetes Pod CREATE."""

    def __init__(
        self,
        store: RunnerCachePlacementAdmissionStore,
        *,
        reauthorize: Callable[[RunnerCacheStartupBinding], RunnerCacheStartupBinding],
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(store, RunnerCachePlacementAdmissionStore):
            raise TypeError("store must implement RunnerCachePlacementAdmissionStore")
        if not callable(reauthorize):
            raise TypeError("reauthorize must be callable")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        self._store = store
        self._reauthorize = reauthorize
        self._monotonic_clock = monotonic_clock

    def _check_deadline(self, deadline_monotonic: float | None) -> None:
        if deadline_monotonic is not None and self._monotonic_clock() >= deadline_monotonic:
            raise RunnerCachePlacementAdmissionTimeoutError(
                "cache placement admission deadline expired"
            )

    def admit(
        self,
        pod: Mapping[str, Any],
        *,
        admission_uid: str,
        request_username: str,
        observed_at: datetime,
        deadline_monotonic: float | None = None,
    ) -> tuple[dict[str, Any], RunnerCachePlacementAdmissionClaim]:
        if not isinstance(pod, Mapping):
            raise TypeError("pod must be a mapping")
        admission_uid = _text(admission_uid, name="admission_uid")
        request_username = _text(
            request_username,
            name="request_username",
            max_length=255,
        )
        observed_at = _aware(observed_at, name="observed_at")
        if deadline_monotonic is not None and (
            isinstance(deadline_monotonic, bool)
            or not isinstance(deadline_monotonic, (int, float))
            or not math.isfinite(float(deadline_monotonic))
        ):
            raise TypeError("deadline_monotonic must be a number or None")
        self._check_deadline(deadline_monotonic)
        metadata = pod.get("metadata")
        if not isinstance(metadata, Mapping):
            raise RunnerCachePlacementAdmissionError("Pod metadata must be an object")
        namespace = _text(metadata.get("namespace"), name="namespace", max_length=253)
        name = _text(metadata.get("name"), name="name", max_length=253)
        annotations = metadata.get("annotations", {})
        if not isinstance(annotations, Mapping):
            raise RunnerCachePlacementAdmissionError("Pod annotations must be an object")
        target_id = annotations.get(RUNNER_CACHE_STARTUP_TARGET_ANNOTATION)
        if not isinstance(target_id, str):
            raise RunnerCachePlacementAdmissionError(
                "gated Pod requires a cache startup target annotation"
            )
        plan = self._store.resolve(target_id)
        self._check_deadline(deadline_monotonic)
        if namespace != plan.namespace:
            raise RunnerCachePlacementAdmissionError(
                "Pod namespace does not match the admission plan"
            )
        if request_username != plan.creator_username:
            raise RunnerCachePlacementAdmissionError(
                "Pod creator does not match the admission plan"
            )
        owner_references = metadata.get("ownerReferences")
        if not isinstance(owner_references, list):
            raise RunnerCachePlacementAdmissionError(
                "gated Pod requires controller ownerReferences"
            )
        controllers = [
            owner
            for owner in owner_references
            if isinstance(owner, Mapping) and owner.get("controller") is True
        ]
        expected_owner = {
            "apiVersion": plan.owner_api_version,
            "kind": plan.owner_kind,
            "name": plan.owner_name,
            "uid": plan.owner_uid,
        }
        if len(controllers) != 1 or any(
            controllers[0].get(key) != value for key, value in expected_owner.items()
        ):
            raise RunnerCachePlacementAdmissionError(
                "Pod controller owner does not match the admission plan"
            )
        refreshed = self._reauthorize(plan.binding)
        self._check_deadline(deadline_monotonic)
        if not isinstance(refreshed, RunnerCacheStartupBinding):
            raise TypeError("reauthorize must return RunnerCacheStartupBinding")
        refreshed = RunnerCacheStartupBinding.model_validate(refreshed.model_dump())
        if refreshed != plan.binding:
            raise RunnerCachePlacementAdmissionConflictError(
                "cache startup binding changed during admission"
            )
        pod_key = f"{namespace}/{name}"
        self._check_deadline(deadline_monotonic)
        allocation = self._store.claim(
            target_id=target_id,
            binding_id=plan.binding.binding_id,
            pod_key=pod_key,
            admission_uid=admission_uid,
            claimed_at=observed_at,
        )
        try:
            self._check_deadline(deadline_monotonic)
            admitted = admit_runner_cache_placement_for_gated_pod(
                pod,
                plan.binding,
                placement_id=allocation.claim.placement_id,
                release_id=plan.release_id,
            )
            self._check_deadline(deadline_monotonic)
        except (
            RunnerCachePlacementAdmissionTimeoutError,
            RunnerCacheSchedulingError,
            TypeError,
            ValueError,
        ):
            if allocation.created:
                self._store.release(allocation.claim)
            raise
        return admitted, allocation.claim
