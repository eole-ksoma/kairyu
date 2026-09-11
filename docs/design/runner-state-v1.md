# Runner State v1

`RunnerState` is Kairyu's logical view of one inference-serving Runner. It is
not a projection of Kubernetes Pod phase: a running container remains
`image_pull`, `model_loading`, or `warming` until the corresponding serving
evidence is complete.

This slice fixes the controller-neutral contract only. Kubernetes watches,
persistence, leader election, routing changes, and scale writes are later
work; no component should claim those behaviors merely by constructing these
models.

## Identity and snapshot rules

Every `RunnerStatus` carries opaque but non-empty `runner_id`, `release_id`,
`model_id`, and `model_revision` values. Node, Pod UID, and GPU UUIDs are
optional while scheduling is incomplete. GPU UUIDs must be unique. The status
schema is explicitly tagged `runner-status-v1`.

Snapshots are immutable, reject unknown fields, and use timezone-aware,
monotonic observation times. `state_version` advances only when the logical
state changes; an idempotent observation of the same state updates
`observed_at` without manufacturing a transition.

`ready` means startup evidence is complete and `active_requests` is zero.
`busy` requires complete startup evidence and at least one active request.
`draining` and `unhealthy` may retain a non-zero count so a controller does not
erase in-flight work while removing the Runner from new dispatch. An
`unhealthy` snapshot requires a bounded, sanitized failure and other states
forbid one. State updates inherit the prior active count unless the caller
supplies a newly observed value; entering `ready`, `terminating`, or
`terminated` from a non-empty snapshot therefore requires an explicit zero.
Any snapshot with active work requires complete startup evidence, including
while draining or unhealthy.

## State transitions

```text
requested → scheduling → image_pull → model_loading → warming → ready ↔ busy
     │            │           │              │           │        │
     └────────────┴───────────┴──────────────┴───────────┴──→ draining
          active states ────────────────────────────────→ unhealthy

ready/busy → draining → terminating → terminated
unhealthy  → draining or terminating
```

Self-transitions are accepted as idempotent observations. Skipping readiness
stages, returning a draining Runner to service, or resurrecting a terminated
Runner is rejected. Recovery creates a new Runner generation/status rather
than rewriting terminated history.

## Startup phase report

The versioned `runner-startup-v1` report decomposes cold start into this
canonical, gap-free order:

1. `image_pull`
2. `model_fetch`
3. `model_load`
4. `graph_compile`
5. `warmup`

Each phase has a start time and either remains in progress or ends as
`succeeded`, `failed`, or `skipped`. A failure is terminal and requires a
bounded sanitized error. `image_pull` and `model_load` cannot be skipped;
cache-hit model fetch, eager-mode graph compilation, and an explicitly disabled
warmup must still be represented as `skipped`, so missing telemetry cannot be
mistaken for success.

The report rejects duplicates, gaps, overlap, future timestamps, more than one
in-progress phase, and any phase after failure. Helper functions return new
reports rather than mutating prior evidence. Once attached to a Runner status,
updates must stay in the same startup attempt and monotonically extend its phase
history. Completed phases are immutable; a new startup attempt requires a new
Runner generation instead of replacing prior evidence. A phase completion or
new phase start also cannot be backdated before the preceding observation.

## Next integration boundary

The serving controller will combine Kubernetes Pod/EndpointSlice conditions,
Kairyu readiness, model revision, startup phase evidence, and active request
counts into `RunnerStatus`. Only a validated `ready` or `busy` snapshot may be
routing-eligible. The next slice should add the read-only watcher/reconciler;
scale actuation and leader election remain separate changes.
