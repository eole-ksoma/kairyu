# Runner State v1

`RunnerState` is Kairyu's logical view of one inference-serving Runner. It is
not a projection of Kubernetes Pod phase: a running container remains
`image_pull`, `model_loading`, or `warming` until the corresponding serving
evidence is complete.

This slice fixes the controller-neutral contract, a read-only Kubernetes
observation/reconciliation boundary, a fence-bound drain/termination handshake,
bounded failure-domain backoff/quarantine, and a lease-fenced single-writer
boundary. It also defines the immutable model-class scaling policy, observation
window, and durable decision-log boundary. Durable Runner-state persistence,
Kubernetes termination writes, and scale actuation remain later work.

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
forbid one. Fatal failures may additionally declare a `revision`, `node`, or
`gpu` domain for restart protection. State updates inherit the prior active
count unless the caller supplies a newly observed value; entering `ready`,
`terminating`, or `terminated` from a non-empty snapshot therefore requires an
explicit zero.
Any snapshot with active work requires complete startup evidence, including
while draining or unhealthy.

## State transitions

```text
requested → scheduling → image_pull → model_loading → warming → ready ↔ busy
     │            │           │              │           │        │
     └────────────┴───────────┴──────────────┴───────────┴──→ draining
          active states ────────────────────────────────→ unhealthy

ready/busy → draining → terminating → terminated
unhealthy  → draining
```

Self-transitions are accepted as idempotent observations. Skipping readiness
stages, changing a fenced draining Runner back to unhealthy, returning it to
service, or resurrecting a terminated Runner is rejected. Fatal evidence found
during drain is recorded operationally while the logical drain remains sticky;
recovery creates a new Runner generation/status rather than rewriting fenced or
terminated history.

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
active count. Reconciliation alone never infers `terminating` from a zero count.
`RunnerStatusReconciler.routing_eligible()` exposes the stateful, freshness-
checked `ready | busy` decision for later routing integration.

## Drain and termination handshake

Termination uses three immutable, versioned pieces of evidence:

1. `RunnerDispatchFence` binds a unique fence ID and monotonic sequence to the
   Runner Pod UID, drain-state version, and concrete replica generation. It
   records when new dispatch stopped and routing exclusion completed.
2. `RunnerDrainActivityObservation` binds an authoritative active-request count
   to that exact fence and generation. An observation from before the routing
   exclusion, from another fence, or from a re-added replica is rejected.
3. `RunnerTerminationAuthorization` persists the matching fence and post-fence
   zero evidence in `RunnerStatus`, so authorization remains auditable and
   idempotent across controller restart.

`ReplicaPoolDrainController` is the executable single-pool adapter. Acquiring
its drain lease immediately removes the Runner from new placement and also
invalidates an already-prepared placement lease before backend dispatch. It
retains the lease while observing outstanding work. Its `commit_termination`
operation revalidates the retained fence and replica generation, reads zero,
and creates authorization without an asynchronous yield between those steps.
The backend-neutral `RunnerDrainController` protocol requires the same atomic
commit boundary from a production shared dispatcher/request store; that
implementation must use one transaction or compare-and-swap operation across
the durable fence, generation, active count, and authorization record.

`RunnerStatusReconciler.authorize_termination()` delegates to that atomic commit
and accepts only a `draining` status with matching post-fence
`active_requests == 0`. Stale zero observed
before the fence, a non-zero count, a different Pod/generation/state version,
or evidence older than the latest runtime observation fails closed. The
resulting `terminating` status must carry the persisted authorization; only a
subsequent full Kubernetes epoch that confirms Pod disappearance advances it
to `terminated`.

This slice mutates local `ReplicaPool` eligibility but does not patch or delete
Kubernetes objects, choose a replica count, persist controller status, or elect
a writer. A multi-gateway deployment must implement the protocol with a shared,
durable dispatch fence before using its authorization for Pod deletion.

## Crash backoff and failure-domain quarantine

`RunnerFailureGuard` consumes immutable `unhealthy` transitions. A transition
is counted exactly once by `runner_id + state_version`; an exact replay is
idempotent and a conflicting replay fails closed. A separate window-bounded
observation tombstone retains this identity even when a domain has no action or
its bounded event list evicts the original row. Tombstone capacity overflow is
rejected atomically rather than weakening deduplication. The stateful reconciler
submits its complete updated view through `record_many()` before committing its
own state, so a guard capacity/error rejection leaves both components unchanged.
Failure scopes are explicit in `RunnerFailure.domain` rather than inferred from
free-form messages:

- revision failures use the complete `release_id + model_id + model_revision`
  identity, preventing one poisoned release from suppressing another;
- node failures use the stable Kubernetes node name and apply across releases;
- GPU failures use physical GPU UUIDs and apply across nodes/releases. When a
  Pod reports more than one GPU and the source cannot identify the exact device,
  every assigned UUID is conservatively fenced.

The reconciler marks image-pull failures, OOM, crash loops, non-zero container
exit, Pod failure, and fatal runtime readiness as revision-scoped; GPU Xid as
GPU-scoped; and unknown Pod state as node-scoped. Startup-owned failures without
an explicit scope are revision-scoped because their immutable startup report
binds them to the release/model revision. Transient serving-gate loss without a
fatal classification does not poison a domain.

`RunnerBackoffPolicy` bounds the base delay, exponential factor, maximum delay,
failure window, quarantine threshold/duration, retained events per domain, and
total domain count. Before the threshold, `RunnerBackoffDecision` exposes an
exponential `backoff_until`; at the threshold it also exposes a time-bounded
`quarantine_until`. `blocked_domains()` evaluates all currently known revision,
node, and GPU identities for a candidate placement. A capacity overflow raises
instead of silently dropping an active quarantine.

`RunnerFailureGuardSnapshot` serializes the policy, bounded event ledgers, and
quarantine deadlines in deterministic order. Active quarantine retains its
threshold evidence beyond the ordinary failure window, and snapshot validation
reconstructs each bounded domain ledger from the tombstones and recomputes the
exact deadline from that evidence and policy. Restored guards reject clock
rollback, future evidence, changed replays, missing node/GPU identity,
one-sided/corrupt ledgers, and out-of-capacity state. The snapshot is the
persistence seam for the later single-writer controller; this slice itself
performs no database or Kubernetes writes and assumes one event-loop/controller
owner.

## Leader election and single-writer fencing

`RunnerLeaderLeaseStore` is the shared coordination contract for every Runner
controller contender. `acquire()` elects at most one unexpired holder for an
election ID, `renew()` extends only the same tenure, and `authorize()` uses the
store-owned clock to issue one immediate `RunnerWriterAuthority`. Lease expiry
is exclusive: at `lease_until` the old holder has no authority and takeover may
start. Every takeover or release/reacquire increments a durable, monotonic
`fencing_token`; renewal never changes it.

Controller `holder_id` values must be globally unique per live process (for
example Pod UID plus process boot UUID). The production store must be shared,
linearizable, durable across controller restarts, retain token tombstones, and
derive all expiry decisions from its own authoritative clock. The included
`InMemoryRunnerLeaderLeaseStore` is a bounded, thread-safe executable
specification for tests and single-process development; it is not an HA backend.
`PostgresRunnerLeaderLeaseStore` is the production implementation: each
environment uses an explicit store ID, PostgreSQL advisory locks and row locks
serialize contenders, `clock_timestamp()` owns expiry, and the retained row
preserves the fencing-token tombstone across release and process restart. Store
DDL, catalog checks, and lease operations all use the explicitly qualified
`public` control schema; connection `search_path` cannot select an independent
election. Startup fails closed unless both tables resolve to that namespace and
their relation kind, ordered column types/nullability, primary key, foreign key
target, and validated check constraints exactly match schema version 1. Both
must be permanent rather than unlogged tables so crash recovery cannot reset
the fencing-token tombstone; a version marker alone is not accepted as
readiness evidence. The namespace OID is pinned for the store lifetime and
revalidated after reconnect, preventing silent attachment to recreated state.

`LeaderFencedRunnerController` obtains a freshly store-validated authority
immediately before each synchronous, bounded callback. Validation requires the
exact held tenure and lease deadline and cannot predate its latest renewal. Its
`reconcile()` gate covers both the reconciler and its attached failure guard;
`authorize_termination()` covers the final drain transition; and
`mutate_autoscaler()` is the required entry point for the later scale actuator.
Followers, expired holders, cached/conflicting authority, stale tokens, clock
rollback, capacity exhaustion, and fencing-token regression fail closed before
the callback runs.

The authority check never holds the coordination store lock while application
code runs, so an expired leader cannot prevent takeover by hanging. Therefore
the authority token must be persisted with any external decision and checked
atomically by its mutation target; output from a callback that outlives its
lease is stale even if the local function returns normally. WP3.4 remains
responsible for propagating this token into the scale decision generation/CAS
boundary. The current slice performs no Kubernetes mutation and does not add
cluster RBAC.

## Model-class scaling policy

`ScalingPolicy` (`runner-scaling-policy-v1`) is the immutable input contract for
one bounded model class. Its durable identity is `(model_class,
policy_revision)`; `ScalingPolicyCatalog` adds a separately versioned, bounded
collection and rejects duplicate class keys. WP3.2 decision records must copy
both the catalog revision and resolved policy identity instead of referring to
mutable configuration by name alone.

The policy separates minimum and maximum replicas, absolute and ratio-based
warm buffer, scale-up delay, idle keep-alive, decision cooldown, maximum
request multiplexing per Runner, and maximum per-decision scale-up/down steps.
`max_observation_age_seconds` fixes the freshness lease in the copied policy,
so replaying a record cannot reinterpret whether its inputs were stale.
The buffer is exactly the larger of `warm_buffer_replicas` and
`ceil(demand_replicas * warm_buffer_ratio)`; the ratio is based on unbuffered
demand replica count capped at `max_replicas`, not current or desired capacity.
The later decision engine will add that buffer, then clamp the target and each
step to the hard bounds. No field in this schema changes the existing
`autoscale_decision()` behavior yet.

The safe default is one fixed warm, non-multiplexed replica. A zero minimum is
valid only when `scale_to_zero=true` and a non-empty measurement/review approval
ID is present. Its absolute buffer must be zero; the ratio buffer still permits
headroom under load but evaluates to zero at zero demand. An approval ID is
rejected when scale-to-zero is disabled. This makes the plan's model-class
approval rule machine-checkable rather than an operator convention. Public
identity, buffer, and catalog lookup paths revalidate serialized content so
unchecked Pydantic copies cannot cross the control-plane boundary.

## Autoscaler observation window and decision log

`ScalingObservation` captures one coherent, source-timestamped input snapshot
for exactly one model class. Queue inputs include total and interactive/batch
depth, oldest age, arrival rate, deadline percentiles, predicted TTFT, and
goodput ratio. Runner inputs retain current, busy, ready, loading, unhealthy,
and draining counts. Optional resource inputs cover GPU, HBM, KV-cache,
multiplexing, and batch occupancy; optional startup inputs retain model-cache
residency and bounded EMA/p95 metrics for each canonical startup phase. Source
timestamps cannot postdate the assembled observation, class counts cannot
exceed their totals, and every numeric field is finite and bounded.

`ScalingObservationWindow` (`runner-scaling-window-v1`) is a non-empty,
time-bounded sequence of these snapshots. Observation times must be strictly
increasing and inside the window, IDs must be unique, and all rows must match
the window model class. A canonical SHA-256 fingerprint binds the complete
window. These rules prevent partial, reordered, cross-class, or unchecked
Pydantic copies from entering a decision.

`ScalingDecisionRecord` (`runner-scaling-decision-v1`) copies the exact window,
catalog revision, and immutable resolved policy rather than pointing at mutable
configuration. It records action, enumerated primary reason and bounded detail,
unbuffered demand, the policy-derived buffered target, desired replicas, and
delta from the latest observed replica count. Validation recomputes the policy
buffer, target delta, min/max bounds, and per-decision step limit. Staleness is
derived from the oldest source timestamp in the latest observation (including
optional resource/startup evidence when present) and the copied policy's
freshness lease; the persisted boolean and reason must agree with that result.
Stale input may hold or perform a policy-step-bounded scale-up, but can never
authorize scale-down. A canonical record fingerprint makes an exact retry
idempotent and a changed retry under the same decision ID a conflict.

If a newly resolved policy puts the currently observed replica count outside
its bounds, one decision may remain outside the new range only while moving by
at most the configured step toward the nearest bound. Holding is still valid
when a freshness or cooldown safety rule prevents mutation. Moving farther
away or crossing past the bound is rejected. The buffered target is explicitly
the pre-clamp demand-plus-buffer value; its field bound includes the maximum
legal demand and ratio buffer, while `desired_replicas` remains the actuated
step target.

`ScalingDecisionLog` is the backend-neutral append/get/list contract. The
in-memory implementation is a bounded, thread-safe reference backend only.
`PostgresScalingDecisionLog` is the shared production backend: an environment
uses an explicit store ID, and that store durably fixes its capacity so two
controllers cannot silently apply different bounds. A row lock on the registry
serializes capacity checks and inserts. The primary key provides exactly-once
decision IDs; reads revalidate both the JSON payload and its duplicated indexed
metadata/fingerprint before returning it.

PostgreSQL objects live in the explicitly qualified `public` schema. Startup
fails closed unless ordered columns, nullability, primary/foreign/check
constraints, the model/time lookup index (including sort direction), permanent
table persistence, and schema version all match. The namespace OID is pinned
and rechecked after reconnect. The schema is append-only through the public API;
retention/export policy is deliberately deferred until evidence establishes an
operational horizon.

## Next integration boundary

WP3.3 should consume these validated records to calculate and apply a bounded
desired replica count, without yet weakening the single-writer fence. WP3.4
then propagates the leader fencing token through the decision/CAS mutation
boundary. Durable Runner-status persistence and Kubernetes deletion remain
separate changes.
