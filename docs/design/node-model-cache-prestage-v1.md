# Node model cache pre-stage v1

Status: WP4.7 implemented as fenced controller commands, placement-level
claim/CAS, verified node execution, owner-scoped cache pins, and status
projection. Command transport, a shared durable command-store backend, and
Kubernetes node binding remain deployment wiring.

## Purpose

WP3.6 records which absent placements are needed before a quota-admitted
scale-out, but intentionally does not make that durable intent executable.
WP4.7 converts each selected placement into one bounded command. The command
binds the scaling decision, workload target, placement binding, cache snapshot,
node, approved hardware profile, exact signed artifact, leader tenure, and a
caller-allocated monotonic generation.

The controller flow is:

```text
durable ScalingPrewarmPlan
  -> one fenced ensure command per absent placement
  -> node claim/CAS -> verified fill -> pre-stage owner pin -> ready completion
  -> later monotonic ScalingPrewarmSnapshot
  -> new scaling decision -> ready-only Runner start
```

The earlier HOLD or partial scale decision is never edited or replayed as a
cache mutation.

## Command fence

`build_node_model_prestage_commands()` accepts only the plan's canonical
`cache_fill_placement_ids`. Its generation map must cover those IDs exactly.
Every `NodeModelPrestageCommand` includes:

- a SHA-256 command ID over the complete canonical payload;
- the durable scaling decision ID and fingerprint;
- target ID/revision, deployment, placement binding, and source cache revision;
- exact placement/node/flavor/profile/compatibility identities;
- model ID, model revision, and manifest digest;
- a deterministic owner-scoped pin name;
- controller election, holder, fencing token, and validation window; and
- issued/expiry times wholly inside that leader lease.

Payload substitution with a retained command ID fails model validation. A
command may be claimed only on its named node and while its time window is
active. The command generation is allocated outside this pure builder by the
production durable controller store; it is not inferred from process memory.

## Claim and state transitions

`NodeModelPrestageStore` is the atomic backend contract. Records use the four
WP3.6 states:

```text
absent --ensure/claim--> filling --complete--> ready
                              \--failure----> failed --retry--> filling
ready/failed --fresh release command--------------------------> absent
```

One placement accepts only one active claim ID. Exact replay is idempotent;
another claim conflicts. A successor must advance command generation, keep the
election identity, never change holder under the same token, never regress the
fencing token, and never roll back the same target's revision. Active fills and
failed work cannot be replaced by a different ensure command: failed work must
retry the exact command or receive a release first. Ready work also requires a
separately hashed `release` command with fresh leader authority, a higher
generation, and a later target revision, preventing an expired leader from
dropping another controller's pin. Every mutating CAS rejects an `updated_at`
clock rollback.

Claim validity is checked before the external download begins. Completion or
failure may be recorded after command expiry only for that exact outstanding
claim; this permits large artifacts to finish without turning lease expiry
into an unfenced second mutation. A new claim or release always requires a
currently valid command.

`InMemoryNodeModelPrestageStore` is a bounded, thread-safe executable
specification for tests and development. Production must provide the same
linearizable operations over a shared durable store. Its CAS must persist the
full JSON record, not only state strings or generations.

## Node execution and pins

`NodeModelPrestageExecutor.execute()` validates the command against the GitOps
admission request before claiming. It then calls the WP4.2
`NodeModelCacheAgent.ensure_cached()`, so signature, environment, GPU profile,
blob digest, tree digest, resumable staging, and atomic publication checks are
unchanged. Only after a verified result exists does it add the deterministic
WP4.3 owner pin and CAS the command to `ready`.

Fill errors produce a bounded `failed` record and can retry the exact command
under a new claim while it remains valid. A completion conflict never converts
another claim. Pin and completion use one stable owner, so retry is idempotent
even if a process stops between them.

Release first commits the fresh release command as no longer desired and then
removes only its named pin. Repeating release converges; active and rollback
owners are untouched. A stop after the state change can leave a conservative
extra pin, which the same release command removes on retry. It cannot expose an
artifact to eviction before desired work is withdrawn.

## Controller feedback

`apply_node_model_prestage_records()` joins unique, exact-identity records into
a later monotonic `ScalingPrewarmSnapshot`. Cross-binding, cross-node,
cross-profile, model/revision/digest, or duplicate placement evidence fails
closed. `filling`, `ready`, and `failed` are projected directly. A released
`absent` record does not erase a still-current WP4.4 verified residency hint;
release removes retention intent, not necessarily the reusable bytes. A ready
command record cannot upgrade an absent physical hint: while its completion is
newer than the placement's retained source-hint time it remains `filling`, and
a later fresh absent hint makes it `failed`. The controller aggregation time is
not used as a substitute for node evidence. Only a fresh WP4.4 verified
residency hint carrying its source observation, expiry, and node index revision
can publish `ready`; the original expiry is checked during overlay and again at
final scale authorization. This prevents an old completion or an overlay from
resurrecting a corrupt, missing, evicted, or expired tree.

The controller persists the later snapshot and computes a new WP3.6 decision.
Only placements still ready and eligible at the actuator's final
reauthorization can justify Runner creation.

## Deployment boundary

`private-ai-cloud-iac` must still provide:

- authenticated controller-to-node command transport and per-node identity;
- a shared durable `NodeModelPrestageStore` implementation with atomic
  generation/fencing CAS and retention;
- a node DaemonSet/service hosting the executor and WP4.2 cache agent;
- reconciliation that replays desired ensure/release commands after restart;
- scheduler/affinity enforcement for the placement binding;
- metrics and audit export for command latency, bytes, attempts, failures, and
  pin reconciliation; and
- live node-pool acceptance with real S3, NVMe, Kueue, and Runner startup.

Until that wiring exists, the in-memory store is not production authority and
the feature must remain disabled for production autoscaling.

## CPU verification

Tests cover deterministic command hashing, exact generation allocation,
admission and node binding, claim concurrency, expiry, verified fill and pin,
idempotent replay, failure/retry, fresh fenced release, preservation of other
pin owners, and exact monotonic status projection.
