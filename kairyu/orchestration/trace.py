"""Stable, privacy-safe orchestration trace contract.

The legacy ``kairyu_trace`` string list remains the human-readable compatibility
surface.  These frozen dataclasses carry the same events with enough structure
for evaluation tooling to render DAG execution and timing without retaining
prompts or generated text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TypeAlias

from kairyu.orchestration.budget import BudgetState

TRACE_VERSION = "2.0"

JsonScalar: TypeAlias = str | int | float | bool | None
# Trace metadata values: scalars plus flat scalar lists (e.g. the profile
# judge's offered label list). Nested containers stay disallowed.
JsonValue: TypeAlias = JsonScalar | list[JsonScalar]


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class WorkerTraceIdentity:
    engine: str
    model: str | None = None


@dataclass(frozen=True)
class TraceTiming:
    queued_at: str | None = None
    started_at: str | None = None
    first_token_at: str | None = None
    completed_at: str | None = None


@dataclass(frozen=True)
class TraceUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


@dataclass(frozen=True)
class TraceBudget:
    max_steps: int
    steps_before: int
    steps_consumed: int
    steps_remaining: int
    max_cost_usd: float | None
    cost_before_usd: float
    cost_consumed_usd: float
    cost_remaining_usd: float | None

    @classmethod
    def between(
        cls,
        before: BudgetState,
        after: BudgetState,
        *,
        steps_consumed: int | None = None,
        cost_consumed_usd: float | None = None,
    ) -> TraceBudget:
        max_cost = after.budget.max_cost_usd
        return cls(
            max_steps=after.budget.max_steps,
            steps_before=before.steps_used + before.steps_reserved,
            steps_consumed=(
                max(0, after.steps_used - before.steps_used)
                if steps_consumed is None
                else steps_consumed
            ),
            steps_remaining=max(
                0,
                after.budget.max_steps - after.steps_used - after.steps_reserved,
            ),
            max_cost_usd=max_cost,
            cost_before_usd=before.cost_used,
            cost_consumed_usd=(
                max(0.0, after.cost_used - before.cost_used)
                if cost_consumed_usd is None
                else cost_consumed_usd
            ),
            cost_remaining_usd=(
                max(0.0, max_cost - after.cost_used)
                if max_cost is not None
                else None
            ),
        )


@dataclass(frozen=True)
class TraceError:
    type: str
    retryable: bool = False


@dataclass(frozen=True)
class TraceEvent:
    """One legacy-compatible event with optional structured observations.

    The first three fields intentionally preserve the original constructor
    shape used by Conductor callers.
    """

    node: str
    kind: str
    detail: str = ""
    operation: str | None = None
    status: str = "success"
    attempt: int = 0
    role: str | None = None
    worker: str | None = None
    engine: str | None = None
    model: str | None = None
    timing: TraceTiming | None = None
    usage: TraceUsage | None = None
    budget: TraceBudget | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    error: TraceError | None = None

    def as_v2(self, seq: int) -> dict[str, object]:
        structured_detail = dict(self.metadata)
        if self.detail:
            structured_detail.setdefault("message", self.detail)
        return {
            "seq": seq,
            "node": self.node,
            "role": self.role,
            "kind": self.operation or self.kind,
            "status": self.status,
            "attempt": self.attempt,
            "worker": self.worker,
            "engine": self.engine,
            "model": self.model,
            "timing": asdict(self.timing) if self.timing is not None else None,
            "usage": asdict(self.usage) if self.usage is not None else None,
            "budget": asdict(self.budget) if self.budget is not None else None,
            "detail": structured_detail,
            "error": asdict(self.error) if self.error is not None else None,
        }


@dataclass(frozen=True)
class StructuredTrace:
    request_id: str
    started_at: str
    completed_at: str
    events: tuple[TraceEvent, ...]
    trace_version: str = TRACE_VERSION

    def as_dict(self, *, request_id: str | None = None) -> dict[str, object]:
        return {
            "trace_version": self.trace_version,
            "request_id": request_id or self.request_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "events": [
                event.as_v2(seq)
                for seq, event in enumerate(self.events, start=1)
            ],
        }
