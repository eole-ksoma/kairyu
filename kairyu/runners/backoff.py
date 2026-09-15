"""Bounded crash backoff and failure-domain quarantine for Runners."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.runners.models import (
    RunnerFailureDomainKind,
    RunnerState,
    RunnerStatus,
)


class InvalidRunnerFailureEvidenceError(RuntimeError):
    """Failure evidence is stale, contradictory, or cannot be domain-keyed."""


class RunnerFailureCapacityError(RuntimeError):
    """The bounded failure-domain ledger cannot safely admit another domain."""


def _non_empty(value: str, *, name: str) -> str:
    if not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL")
    return value


def _aware(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


class RunnerFailureDomain(BaseModel):
    """Canonical revision, node, or physical-GPU failure-domain identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["runner-failure-domain-v1"] = "runner-failure-domain-v1"
    kind: RunnerFailureDomainKind
    release_id: str | None = Field(default=None, max_length=512)
    model_id: str | None = Field(default=None, max_length=512)
    model_revision: str | None = Field(default=None, max_length=512)
    node_name: str | None = Field(default=None, max_length=253)
    gpu_uuid: str | None = Field(default=None, max_length=255)

    @field_validator(
        "release_id",
        "model_id",
        "model_revision",
        "node_name",
        "gpu_uuid",
    )
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _non_empty(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_shape(self) -> RunnerFailureDomain:
        revision = (self.release_id, self.model_id, self.model_revision)
        if self.kind is RunnerFailureDomainKind.REVISION:
            if any(value is None for value in revision):
                raise ValueError("revision failure domains require release, model, and revision")
            if self.node_name is not None or self.gpu_uuid is not None:
                raise ValueError("revision failure domains cannot carry node or GPU")
        elif self.kind is RunnerFailureDomainKind.NODE:
            if self.node_name is None:
                raise ValueError("node failure domains require node_name")
            if any(value is not None for value in revision) or self.gpu_uuid is not None:
                raise ValueError("node failure domains can carry only node_name")
        else:
            if self.gpu_uuid is None:
                raise ValueError("GPU failure domains require gpu_uuid")
            if any(value is not None for value in revision) or self.node_name is not None:
                raise ValueError("GPU failure domains can carry only gpu_uuid")
        return self

    @property
    def identity(self) -> tuple[str, ...]:
        if self.kind is RunnerFailureDomainKind.REVISION:
            assert self.release_id is not None
            assert self.model_id is not None
            assert self.model_revision is not None
            return (
                self.kind.value,
                self.release_id,
                self.model_id,
                self.model_revision,
            )
        if self.kind is RunnerFailureDomainKind.NODE:
            assert self.node_name is not None
            return (self.kind.value, self.node_name)
        assert self.gpu_uuid is not None
        return (self.kind.value, self.gpu_uuid)


class RunnerBackoffPolicy(BaseModel):
    """Bounded exponential-backoff and quarantine policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["runner-backoff-policy-v1"] = "runner-backoff-policy-v1"
    base_delay_seconds: float = Field(default=5.0, gt=0, le=3600)
    backoff_factor: int = Field(default=2, ge=1, le=16)
    max_delay_seconds: float = Field(default=300.0, gt=0, le=86400)
    failure_window_seconds: float = Field(default=600.0, gt=0, le=604800)
    quarantine_threshold: int = Field(default=3, ge=2, le=100)
    quarantine_seconds: float = Field(default=900.0, gt=0, le=604800)
    max_events_per_domain: int = Field(default=32, ge=2, le=10000)
    max_domains: int = Field(default=4096, ge=1, le=100000)
    max_observations: int = Field(default=131072, ge=1, le=1000000)

    @field_validator(
        "base_delay_seconds",
        "max_delay_seconds",
        "failure_window_seconds",
        "quarantine_seconds",
        mode="before",
    )
    @classmethod
    def validate_duration(cls, value: object, info) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{info.field_name} must be a number")
        return value

    @field_validator(
        "backoff_factor",
        "quarantine_threshold",
        "max_events_per_domain",
        "max_domains",
        "max_observations",
        mode="before",
    )
    @classmethod
    def validate_integer(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer")
        return value

    @model_validator(mode="after")
    def validate_bounds(self) -> RunnerBackoffPolicy:
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds cannot be below base_delay_seconds")
        if self.max_events_per_domain < self.quarantine_threshold:
            raise ValueError("max_events_per_domain cannot be below quarantine_threshold")
        return self


class RunnerFailureEvent(BaseModel):
    """One immutable unhealthy transition attributed to one failure domain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["runner-failure-event-v1"] = "runner-failure-event-v1"
    domain: RunnerFailureDomain
    runner_id: str = Field(max_length=255)
    state_version: int = Field(ge=0)
    failure_code: str = Field(max_length=128)
    failed_at: datetime

    @field_validator("runner_id", "failure_code")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator("state_version", mode="before")
    @classmethod
    def validate_state_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("state_version must be an integer")
        return value

    @field_validator("failed_at")
    @classmethod
    def validate_failed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="failed_at")

    @property
    def observation_identity(self) -> tuple[str, int]:
        return (self.runner_id, self.state_version)


class RunnerFailureObservation(BaseModel):
    """Window-bounded tombstone for exact-once failure observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["runner-failure-observation-v1"] = "runner-failure-observation-v1"
    runner_id: str = Field(max_length=255)
    state_version: int = Field(ge=0)
    failure_code: str = Field(max_length=128)
    failed_at: datetime
    domains: tuple[RunnerFailureDomain, ...] = ()

    @field_validator("runner_id", "failure_code")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _non_empty(value, name=info.field_name)

    @field_validator("state_version", mode="before")
    @classmethod
    def validate_state_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("state_version must be an integer")
        return value

    @field_validator("failed_at")
    @classmethod
    def validate_failed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="failed_at")

    @model_validator(mode="after")
    def validate_domains(self) -> RunnerFailureObservation:
        identities = tuple(domain.identity for domain in self.domains)
        if identities != tuple(sorted(identities)):
            raise ValueError("observation domains must be ordered deterministically")
        if len(set(identities)) != len(identities):
            raise ValueError("observation domains must be unique")
        return self

    @property
    def identity(self) -> tuple[str, int]:
        return (self.runner_id, self.state_version)


class RunnerFailureDomainState(BaseModel):
    """Replayable bounded ledger for one failure domain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["runner-failure-domain-state-v1"] = "runner-failure-domain-state-v1"
    domain: RunnerFailureDomain
    events: tuple[RunnerFailureEvent, ...] = ()
    quarantine_until: datetime | None = None

    @field_validator("quarantine_until")
    @classmethod
    def validate_quarantine_until(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _aware(value, name="quarantine_until")

    @model_validator(mode="after")
    def validate_events(self) -> RunnerFailureDomainState:
        expected = sorted(
            self.events,
            key=lambda event: (
                event.failed_at,
                event.runner_id,
                event.state_version,
            ),
        )
        if list(self.events) != expected:
            raise ValueError("failure events must be ordered deterministically")
        if any(event.domain != self.domain for event in self.events):
            raise ValueError("failure event domain must match its ledger")
        identities = tuple(event.observation_identity for event in self.events)
        if len(set(identities)) != len(identities):
            raise ValueError("failure observations must be unique within a domain")
        return self


class RunnerFailureGuardSnapshot(BaseModel):
    """Serializable controller state for deterministic restart recovery."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["runner-failure-guard-v1"] = "runner-failure-guard-v1"
    observed_at: datetime
    policy: RunnerBackoffPolicy
    observations: tuple[RunnerFailureObservation, ...] = ()
    domains: tuple[RunnerFailureDomainState, ...] = ()

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _aware(value, name="observed_at")

    @model_validator(mode="after")
    def validate_domains(self) -> RunnerFailureGuardSnapshot:
        observation_identities = tuple(observation.identity for observation in self.observations)
        if observation_identities != tuple(sorted(observation_identities)):
            raise ValueError("failure observations must be ordered deterministically")
        if len(set(observation_identities)) != len(observation_identities):
            raise ValueError("failure observations must be unique")
        if len(self.observations) > self.policy.max_observations:
            raise ValueError("snapshot exceeds max_observations")
        identities = tuple(state.domain.identity for state in self.domains)
        if identities != tuple(sorted(identities)):
            raise ValueError("failure domains must be ordered deterministically")
        if len(set(identities)) != len(identities):
            raise ValueError("failure domains must be unique")
        if len(self.domains) > self.policy.max_domains:
            raise ValueError("snapshot exceeds max_domains")
        if any(len(state.events) > self.policy.max_events_per_domain for state in self.domains):
            raise ValueError("snapshot exceeds max_events_per_domain")
        if any(
            event.failed_at > self.observed_at for state in self.domains for event in state.events
        ):
            raise ValueError("snapshot cannot contain future failure events")
        cutoff = self.observed_at - timedelta(seconds=self.policy.failure_window_seconds)
        quarantine_evidence = {
            event.observation_identity
            for state in self.domains
            if state.quarantine_until is not None
            for event in state.events
        }
        if any(
            (observation.failed_at < cutoff and observation.identity not in quarantine_evidence)
            or observation.failed_at > self.observed_at
            for observation in self.observations
        ):
            raise ValueError("snapshot contains out-of-window failure observations")
        observation_by_identity = {
            observation.identity: observation for observation in self.observations
        }
        state_by_identity = {state.domain.identity: state for state in self.domains}
        for state in self.domains:
            for event in state.events:
                observation = observation_by_identity.get(event.observation_identity)
                if observation is None:
                    raise ValueError("failure event lacks an observation tombstone")
                if (
                    event.failure_code != observation.failure_code
                    or event.failed_at != observation.failed_at
                    or event.domain not in observation.domains
                ):
                    raise ValueError("failure event conflicts with its observation")
        for state in self.domains:
            expected_until: datetime | None = None
            if len(state.events) >= self.policy.quarantine_threshold:
                candidate = state.events[-1].failed_at + timedelta(
                    seconds=self.policy.quarantine_seconds
                )
                if candidate > self.observed_at:
                    expected_until = candidate
            if state.quarantine_until != expected_until:
                raise ValueError("quarantine deadline does not match its evidence")

        expected_events: dict[tuple[str, ...], list[RunnerFailureEvent]] = {}
        for observation in self.observations:
            for domain in observation.domains:
                ledger_state = state_by_identity.get(domain.identity)
                if observation.failed_at < cutoff and (
                    ledger_state is None or ledger_state.quarantine_until is None
                ):
                    continue
                expected_events.setdefault(domain.identity, []).append(
                    RunnerFailureEvent(
                        domain=domain,
                        runner_id=observation.runner_id,
                        state_version=observation.state_version,
                        failure_code=observation.failure_code,
                        failed_at=observation.failed_at,
                    )
                )
        for identity in set(state_by_identity) | set(expected_events):
            ledger_state = state_by_identity.get(identity)
            actual = () if ledger_state is None else ledger_state.events
            expected = tuple(
                sorted(
                    expected_events.get(identity, ()),
                    key=lambda event: (
                        event.failed_at,
                        event.runner_id,
                        event.state_version,
                    ),
                )[-self.policy.max_events_per_domain :]
            )
            if actual != expected:
                raise ValueError("failure domain ledger conflicts with its observations")
        return self


class RunnerBackoffDecision(BaseModel):
    """Point-in-time placement gate for one failure domain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["runner-backoff-decision-v1"] = "runner-backoff-decision-v1"
    domain: RunnerFailureDomain
    evaluated_at: datetime
    failure_count: int = Field(ge=0)
    last_failure_at: datetime | None = None
    backoff_until: datetime | None = None
    quarantine_until: datetime | None = None
    eligible_at: datetime
    blocked: bool
    quarantined: bool

    @field_validator("failure_count", mode="before")
    @classmethod
    def validate_failure_count(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("failure_count must be an integer")
        return value

    @field_validator(
        "evaluated_at",
        "last_failure_at",
        "backoff_until",
        "quarantine_until",
        "eligible_at",
    )
    @classmethod
    def validate_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return _aware(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_decision(self) -> RunnerBackoffDecision:
        if self.failure_count == 0:
            if self.last_failure_at is not None or self.backoff_until is not None:
                raise ValueError("empty decisions cannot carry failure timestamps")
        elif self.last_failure_at is None or self.backoff_until is None:
            raise ValueError("non-empty decisions require failure timestamps")
        if self.quarantined != (
            self.quarantine_until is not None and self.quarantine_until > self.evaluated_at
        ):
            raise ValueError("quarantined must reflect quarantine_until")
        if self.blocked != (self.eligible_at > self.evaluated_at):
            raise ValueError("blocked must reflect eligible_at")
        return self


def revision_failure_domain(
    *,
    release_id: str,
    model_id: str,
    model_revision: str,
) -> RunnerFailureDomain:
    return RunnerFailureDomain(
        kind=RunnerFailureDomainKind.REVISION,
        release_id=release_id,
        model_id=model_id,
        model_revision=model_revision,
    )


def node_failure_domain(node_name: str) -> RunnerFailureDomain:
    return RunnerFailureDomain(
        kind=RunnerFailureDomainKind.NODE,
        node_name=node_name,
    )


def gpu_failure_domain(gpu_uuid: str) -> RunnerFailureDomain:
    return RunnerFailureDomain(
        kind=RunnerFailureDomainKind.GPU,
        gpu_uuid=gpu_uuid,
    )


def failure_domains_for_status(status: RunnerStatus) -> tuple[RunnerFailureDomain, ...]:
    """Derive exact domains only from an unhealthy lifecycle snapshot."""

    if not isinstance(status, RunnerStatus):
        raise TypeError("status must be a RunnerStatus")
    if status.state is not RunnerState.UNHEALTHY or status.failure is None:
        raise InvalidRunnerFailureEvidenceError(
            "failure domains require an unhealthy Runner status"
        )
    kind = status.failure.domain
    if kind is None and status.startup is not None:
        if status.startup.failure == status.failure:
            kind = RunnerFailureDomainKind.REVISION
    if kind is None:
        return ()
    if kind is RunnerFailureDomainKind.REVISION:
        return (
            revision_failure_domain(
                release_id=status.release_id,
                model_id=status.model_id,
                model_revision=status.model_revision,
            ),
        )
    if kind is RunnerFailureDomainKind.NODE:
        if status.node_name is None:
            raise InvalidRunnerFailureEvidenceError("node-scoped failure lacks node identity")
        return (node_failure_domain(status.node_name),)
    if not status.gpu_uuids:
        raise InvalidRunnerFailureEvidenceError("GPU-scoped failure lacks GPU identity")
    return tuple(gpu_failure_domain(uuid) for uuid in sorted(status.gpu_uuids))


def candidate_failure_domains(
    *,
    release_id: str,
    model_id: str,
    model_revision: str,
    node_name: str | None = None,
    gpu_uuids: Iterable[str] = (),
) -> tuple[RunnerFailureDomain, ...]:
    """Build the complete known placement domains for admission checks."""

    if isinstance(gpu_uuids, (str, bytes)):
        raise TypeError("gpu_uuids must be an iterable of UUID strings")
    domains = [
        revision_failure_domain(
            release_id=release_id,
            model_id=model_id,
            model_revision=model_revision,
        )
    ]
    if node_name is not None:
        domains.append(node_failure_domain(node_name))
    domains.extend(gpu_failure_domain(uuid) for uuid in sorted(set(gpu_uuids)))
    return tuple(domains)


class RunnerFailureGuard:
    """In-memory policy engine with a serializable, bounded source of truth."""

    def __init__(
        self,
        policy: RunnerBackoffPolicy | None = None,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        supplied_policy = RunnerBackoffPolicy() if policy is None else policy
        if not isinstance(supplied_policy, RunnerBackoffPolicy):
            raise TypeError("policy must be a RunnerBackoffPolicy")
        self._policy = RunnerBackoffPolicy.model_validate(supplied_policy.model_dump())
        self._now = now
        self._last_observed_at: datetime | None = None
        self._states: dict[tuple[str, ...], RunnerFailureDomainState] = {}
        self._observations: dict[tuple[str, int], RunnerFailureObservation] = {}

    @classmethod
    def from_snapshot(
        cls,
        snapshot: RunnerFailureGuardSnapshot,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> RunnerFailureGuard:
        if not isinstance(snapshot, RunnerFailureGuardSnapshot):
            raise TypeError("snapshot must be a RunnerFailureGuardSnapshot")
        validated = RunnerFailureGuardSnapshot.model_validate(snapshot.model_dump())
        guard = cls(validated.policy, now=now)
        guard._states = {state.domain.identity: state for state in validated.domains}
        guard._observations = {
            observation.identity: observation for observation in validated.observations
        }
        guard._last_observed_at = validated.observed_at
        return guard

    @property
    def policy(self) -> RunnerBackoffPolicy:
        return self._policy

    def _at(self, at: datetime | None) -> datetime:
        observed_at = _aware(self._now() if at is None else at, name="at")
        if self._last_observed_at is not None and observed_at < self._last_observed_at:
            raise InvalidRunnerFailureEvidenceError("failure guard time cannot move backwards")
        return observed_at

    def _compact(
        self,
        states: dict[tuple[str, ...], RunnerFailureDomainState],
        *,
        at: datetime,
    ) -> dict[tuple[str, ...], RunnerFailureDomainState]:
        cutoff = at - timedelta(seconds=self._policy.failure_window_seconds)
        compacted: dict[tuple[str, ...], RunnerFailureDomainState] = {}
        for key, state in states.items():
            quarantine_until = state.quarantine_until
            if quarantine_until is not None and quarantine_until <= at:
                quarantine_until = None
            events = (
                state.events
                if quarantine_until is not None
                else tuple(event for event in state.events if event.failed_at >= cutoff)
            )
            if not events and quarantine_until is None:
                continue
            compacted[key] = RunnerFailureDomainState(
                domain=state.domain,
                events=events,
                quarantine_until=quarantine_until,
            )
        return compacted

    def _compact_observations(
        self,
        observations: dict[tuple[str, int], RunnerFailureObservation],
        *,
        at: datetime,
        states: dict[tuple[str, ...], RunnerFailureDomainState],
    ) -> dict[tuple[str, int], RunnerFailureObservation]:
        cutoff = at - timedelta(seconds=self._policy.failure_window_seconds)
        retained_by_quarantine = {
            event.observation_identity
            for state in states.values()
            if state.quarantine_until is not None
            for event in state.events
        }
        return {
            identity: observation
            for identity, observation in observations.items()
            if observation.failed_at >= cutoff or identity in retained_by_quarantine
        }

    def record(
        self,
        status: RunnerStatus,
        *,
        at: datetime | None = None,
    ) -> tuple[RunnerBackoffDecision, ...]:
        """Record one unhealthy transition exactly once across all its domains."""

        evaluated_at = self._at(at)
        domains = failure_domains_for_status(status)
        failure = status.failure
        assert failure is not None
        if status.state_changed_at > evaluated_at:
            raise InvalidRunnerFailureEvidenceError("failure transition cannot be in the future")
        states = self._compact(dict(self._states), at=evaluated_at)
        observations = self._compact_observations(
            dict(self._observations),
            at=evaluated_at,
            states=states,
        )
        expected_observation = RunnerFailureObservation(
            runner_id=status.runner_id,
            state_version=status.state_version,
            failure_code=failure.code,
            failed_at=status.state_changed_at,
            domains=domains,
        )
        existing_observation = observations.get(expected_observation.identity)
        if existing_observation is not None:
            if existing_observation != expected_observation:
                raise InvalidRunnerFailureEvidenceError(
                    "failure observation replay changed its evidence"
                )
            decisions = tuple(
                self._decision_for_domain(states, domain, at=evaluated_at) for domain in domains
            )
            self._states = states
            self._observations = observations
            self._last_observed_at = evaluated_at
            return decisions
        cutoff = evaluated_at - timedelta(seconds=self._policy.failure_window_seconds)
        if status.state_changed_at < cutoff:
            self._states = states
            self._observations = observations
            self._last_observed_at = evaluated_at
            return ()
        if len(observations) >= self._policy.max_observations:
            raise RunnerFailureCapacityError(
                "active failure observations exceed the configured capacity"
            )
        if not domains:
            observations[expected_observation.identity] = expected_observation
            self._states = states
            self._observations = observations
            self._last_observed_at = evaluated_at
            return ()
        expected_events = tuple(
            RunnerFailureEvent(
                domain=domain,
                runner_id=status.runner_id,
                state_version=status.state_version,
                failure_code=failure.code,
                failed_at=status.state_changed_at,
            )
            for domain in domains
        )
        for event in expected_events:
            key = event.domain.identity
            state = states.get(key)
            events = () if state is None else state.events
            quarantine_until = None if state is None else state.quarantine_until
            ordered = tuple(
                sorted(
                    (*events, event),
                    key=lambda item: (
                        item.failed_at,
                        item.runner_id,
                        item.state_version,
                    ),
                )
            )[-self._policy.max_events_per_domain :]
            if len(ordered) >= self._policy.quarantine_threshold:
                candidate = ordered[-1].failed_at + timedelta(
                    seconds=self._policy.quarantine_seconds
                )
                if quarantine_until is None or candidate > quarantine_until:
                    quarantine_until = candidate
            states[key] = RunnerFailureDomainState(
                domain=event.domain,
                events=ordered,
                quarantine_until=quarantine_until,
            )

        if len(states) > self._policy.max_domains:
            raise RunnerFailureCapacityError(
                "active failure domains exceed the configured capacity"
            )
        observations[expected_observation.identity] = expected_observation
        self._states = states
        self._observations = observations
        decisions = tuple(
            self._decision(states[domain.identity], at=evaluated_at) for domain in domains
        )
        self._last_observed_at = evaluated_at
        return decisions

    def record_many(
        self,
        statuses: Iterable[RunnerStatus],
        *,
        at: datetime | None = None,
    ) -> dict[str, tuple[RunnerBackoffDecision, ...]]:
        """Atomically record a complete reconciler view of unhealthy Runners."""

        evaluated_at = self._at(at)
        selected: dict[str, RunnerStatus] = {}
        for status in statuses:
            if not isinstance(status, RunnerStatus):
                raise TypeError("statuses must contain RunnerStatus values")
            if status.state is not RunnerState.UNHEALTHY:
                continue
            if status.runner_id in selected:
                raise InvalidRunnerFailureEvidenceError(
                    "statuses contain duplicate unhealthy Runner IDs"
                )
            selected[status.runner_id] = status

        staged = RunnerFailureGuard(self._policy, now=self._now)
        staged._last_observed_at = self._last_observed_at
        staged._states = dict(self._states)
        staged._observations = dict(self._observations)
        decisions = {
            runner_id: staged.record(status, at=evaluated_at)
            for runner_id, status in sorted(selected.items())
        }
        if not selected:
            staged._last_observed_at = evaluated_at
        self._states = staged._states
        self._observations = staged._observations
        self._last_observed_at = staged._last_observed_at
        return decisions

    def _decision(
        self,
        state: RunnerFailureDomainState,
        *,
        at: datetime,
    ) -> RunnerBackoffDecision:
        cutoff = at - timedelta(seconds=self._policy.failure_window_seconds)
        events = tuple(event for event in state.events if event.failed_at >= cutoff)
        count = len(events)
        last_failure_at = None if not events else events[-1].failed_at
        backoff_until: datetime | None = None
        if last_failure_at is not None:
            delay = min(
                self._policy.max_delay_seconds,
                self._policy.base_delay_seconds * self._policy.backoff_factor ** min(count - 1, 63),
            )
            backoff_until = last_failure_at + timedelta(seconds=delay)
        quarantine_until = state.quarantine_until
        if quarantine_until is not None and quarantine_until <= at:
            quarantine_until = None
        eligible_at = at
        for boundary in (backoff_until, quarantine_until):
            if boundary is not None and boundary > eligible_at:
                eligible_at = boundary
        return RunnerBackoffDecision(
            domain=state.domain,
            evaluated_at=at,
            failure_count=count,
            last_failure_at=last_failure_at,
            backoff_until=backoff_until,
            quarantine_until=quarantine_until,
            eligible_at=eligible_at,
            blocked=eligible_at > at,
            quarantined=quarantine_until is not None and quarantine_until > at,
        )

    def _decision_for_domain(
        self,
        states: dict[tuple[str, ...], RunnerFailureDomainState],
        domain: RunnerFailureDomain,
        *,
        at: datetime,
    ) -> RunnerBackoffDecision:
        state = states.get(domain.identity)
        if state is not None:
            return self._decision(state, at=at)
        return RunnerBackoffDecision(
            domain=domain,
            evaluated_at=at,
            failure_count=0,
            eligible_at=at,
            blocked=False,
            quarantined=False,
        )

    def decision(
        self,
        domain: RunnerFailureDomain,
        *,
        at: datetime | None = None,
    ) -> RunnerBackoffDecision:
        if not isinstance(domain, RunnerFailureDomain):
            raise TypeError("domain must be a RunnerFailureDomain")
        evaluated_at = self._at(at)
        decision = self._decision_for_domain(self._states, domain, at=evaluated_at)
        self._last_observed_at = evaluated_at
        return decision

    def blocked_domains(
        self,
        *,
        release_id: str,
        model_id: str,
        model_revision: str,
        node_name: str | None = None,
        gpu_uuids: Iterable[str] = (),
        at: datetime | None = None,
    ) -> tuple[RunnerBackoffDecision, ...]:
        evaluated_at = self._at(at)
        return tuple(
            decision
            for domain in candidate_failure_domains(
                release_id=release_id,
                model_id=model_id,
                model_revision=model_revision,
                node_name=node_name,
                gpu_uuids=gpu_uuids,
            )
            if (decision := self.decision(domain, at=evaluated_at)).blocked
        )

    def snapshot(
        self,
        *,
        at: datetime | None = None,
    ) -> RunnerFailureGuardSnapshot:
        observed_at = self._at(at)
        states = self._compact(dict(self._states), at=observed_at)
        observations = self._compact_observations(
            dict(self._observations),
            at=observed_at,
            states=states,
        )
        snapshot = RunnerFailureGuardSnapshot(
            observed_at=observed_at,
            policy=self._policy,
            observations=tuple(observations[key] for key in sorted(observations)),
            domains=tuple(states[key] for key in sorted(states)),
        )
        self._last_observed_at = observed_at
        return snapshot
