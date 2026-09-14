# Runner State v1

`RunnerState` is Kairyu's logical view of one inference-serving Runner. It is
not a projection of Kubernetes Pod phase: a running container remains
`image_pull`, `model_loading`, or `warming` until the corresponding serving
evidence is complete.

This slice fixes the controller-neutral contract and a read-only Kubernetes
observation/reconciliation boundary. Persistence, leader election, routing
mutation, drain authorization, and scale writes remain later work.

## Identity and snapshot rules

Every `RunnerStatus` carries opaque but non-empty `runner_id`, `release_id`,
`model_id`, and `model_revision` values. Node, Pod UID, and GPU UUIDs are
optional while scheduling is incomplete. GPU UUIDs must be unique. The status
schema is explicitly tagged `runner-status-v1`.

Snapshots are immutable, reject unknown fields, and use timezone-aware,
monotonic observation times. `state_version` advances only when the logical
state changes; an idempotent observation of the same state updates
`observed_at` without manufacturing a transition. Runtime evidence carries
its independent source timestamp and a persisted payload fingerprint. An
equal-timestamp replay is accepted only when its complete payload is identical,
including after controller restart.

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

## Read-only Kubernetes observation and reconciliation

`KubernetesRunnerWatcher` issues only authenticated Pod and EndpointSlice LIST
requests. It rotates the projected service-account token on every poll and
joins resources by the stable Pod UID. Polls are single-flight; when either
LIST fails, its sibling is cancelled and joined before the poll lock is
released. Every successful full-list epoch records a watcher identity,
monotonic epoch number, both Kubernetes resource versions, source-start time,
and completion time. This remains explicit when the selected fleet is empty,
allowing the reconciler to distinguish a confirmed empty list from a failed or
missing poll. The source-start time prevents slow I/O from refreshing old
Kubernetes evidence.

Managed Pods publish immutable deployment identity through these annotations:

- `kairyu.ai/release-id`
- `kairyu.ai/model-id`
- `kairyu.ai/model-revision`
- `kairyu.ai/gpu-uuids`, encoded as a JSON string array once assigned
- `kairyu.ai/runner-container`, required when the Pod has multiple containers

Missing release/model identity, malformed GPU data, duplicate Pod UIDs, an
ambiguous serving container, and unexpected runtime rows reject the entire
observation. The runtime side is deliberately an injected, timeout-bounded,
read-only source: it owns Kairyu readiness, active-request counts, and the
startup report, while Kubernetes owns Pod scheduling/container evidence and
EndpointSlice membership. The watcher may also request runtime/request-store
evidence for tracked Runners whose Pods have disappeared, without treating
those rows as live Pods.

`RunnerStatusReconciler` atomically aggregates each full observation epoch.
Pod phase never decides serving eligibility by itself. A Runner becomes
`ready` or `busy` only when all of the following agree:

1. the Pod is `Running` and Pod-ready;
2. at least one GPU UUID assignment is observed;
3. the Pod UID is a ready, non-terminating EndpointSlice target;
4. Kairyu runtime readiness is true;
5. the complete ordered startup report is present.

Routing eligibility fails closed immediately when any latest serving gate is
false, its evidence is older than the caller's freshness lease, or timestamps
appear to come from the future. The gate age is the conservative minimum of the
Kubernetes source-start time and that Runner's accepted runtime timestamp, so
runtime staleness and routing freshness are not added together. A brief
cross-resource mismatch keeps the lifecycle state stable for a configurable
elapsed-time grace period, but never keeps it routable. Continuous loss past
that grace changes the state to sticky `unhealthy`; it cannot self-resurrect
without a new Runner generation.

Runtime timestamps are accepted only within a bounded age and clock-skew
tolerance. Older or excessively future evidence is unavailable rather than
authoritative. Node and GPU assignment cannot drift for the same Pod UID.
Image pull, crash loop, OOM, GPU Xid, non-zero exit, Pod, startup, and readiness
failures retain distinct bounded reason codes; `lastState.terminated` preserves
the underlying OOM/GPU/exit cause during a CrashLoopBackOff.

A Pod deletion or confirmed full-list disappearance enters and remains
`draining`, preserving identity, startup evidence, and the latest trustworthy
active count. This slice deliberately does not infer `terminating` from a zero
count: only the next drain-handshake slice may authorize termination after new
dispatch is fenced and the request store confirms post-fence zero.
`RunnerStatusReconciler.routing_eligible()` exposes the stateful, freshness-
checked `ready | busy` decision for later routing integration.

This component does not patch Kubernetes objects, alter `ReplicaPool`, choose a
replica count, persist status, elect a writer, mutate routing, or authorize Pod
termination.

## Next integration boundary

The next slice should implement the drain/termination handshake: stop queue
dispatch, remove routing eligibility, continue reading the request store until
`active_requests == 0`, and only then authorize Kubernetes termination. Scale
actuation, persistence, crash backoff, and leader election remain separate
changes.
