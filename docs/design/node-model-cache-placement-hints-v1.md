# Node model cache placement hints v1

Status: WP4.4 implemented as a path-free node publication contract and a
controller adapter to the WP3.6 staged scale-out snapshot. Kubernetes transport,
RBAC, and scheduler binding remain deployment wiring.

## Purpose

WP4.4 lets a scheduler or controller prefer a node that already holds the exact
signed model artifact. It does not turn the cache index into artifact authority
and does not allow locality to bypass health, schedulability, hardware
compatibility, Kueue quota, or the Runner's WP4.1/WP4.2 admission checks.

The data flow is:

```text
WP4.3 node cache index
  -> short-lived node placement-hint snapshot
  -> controller-owned placement inventory join
  -> WP3.6 ScalingPrewarmSnapshot
  -> ready-only staged Runner scale-out
```

The final Runner still requests a manifest digest and validates the published
tree through `NodeModelCacheAgent`. A hint can improve placement but cannot
authorize a path mount or an implicit model-revision fallback.

## Node publication contract

`NodeModelCacheIndex.snapshot()` reads the global index revision and all rows in
one SQLite read transaction. The durable global revision starts at one and
advances exactly once for each observable index mutation. It does not advance
for an idempotent touch, pin, unpin, or repeated failure mark.

The global revision is a user-version-2 schema feature. Opening a version-1
index performs a bounded, transactional migration, rebuilds the entry and pin
tables with signed-64-bit generation bounds, and changes the schema/user
version. Schema-level mutation triggers advance the revision even when a
version-1 connection passed its version check immediately before migration; a
newly opened version-1 writer fails closed on version 2. Migration rejects any
generation sum that cannot fit the signed 64-bit revision instead of committing
a REAL or an unusable counter.

`NodeModelCachePlacementHintPublisher` projects that snapshot into
`NodeModelCachePlacementHintSnapshot`:

- only `verified=True` rows are included;
- each resident entry carries the exact manifest digest, model ID, model
  revision, bytes, file count, verification/access times, pin boolean, and row
  generation;
- the local artifact path and owner-scoped pin identities are not exposed;
- entries retain canonical model/revision/digest order;
- the node ID and global index revision identify the source state; and
- `observed_at` and `valid_until` bound freshness to at most 300 seconds.

An unverified row disappears from the next publication. This is fail closed:
the controller treats the node as a cache miss until a digest-verified refill
restores the row. Refreshing an unchanged snapshot may keep the same index
revision while advancing its observation time, because the node is renewing
liveness rather than claiming a cache mutation.

## Controller aggregation

`ModelCachePlacementCandidate` contains controller-owned facts that a node
publication cannot assert:

- placement ID and Kubernetes node name;
- Kueue ResourceFlavor;
- approved hardware profile and compatibility approval;
- assignment state;
- node health; and
- schedulability.

`build_cache_placement_snapshot()` joins those candidates with one current hint
per node. A candidate becomes `READY` only when its node publication:

1. is not from the future relative to the controller observation;
2. has not reached `valid_until`; and
3. contains the exact requested manifest digest, model ID, and model revision.

A missing, expired, future-dated, wrong-digest, wrong-model, or wrong-revision
hint becomes `ABSENT`. The adapter never selects a related revision or model.
Duplicate node publications and duplicate placement IDs fail closed instead of
using input order as conflict resolution.

The adapter copies health, assignment, schedulability, flavor, profile, and
compatibility fields from the controller candidate. It cannot upgrade them.
The existing `plan_cache_aware_scale_up()` therefore excludes an unhealthy,
assigned, unschedulable, or wrong-flavor candidate even when its cache state is
`READY`. The controller must supply only approved profile/compatibility pairs;
their identities remain bound into the snapshot and the final reauthorization.

Multiple placement units may reference one node and reuse one resident model
tree only when the controller inventory declares those units independently
schedulable. The node hint does not infer GPU capacity.

## Revisions, freshness, and replay

There are two revision domains:

- `index_revision` is node-local and proves which WP4.3 state produced one
  publication;
- `ScalingPrewarmSnapshot.cache_revision` is the controller's global,
  monotonically increasing inventory revision.

The pure aggregation function requires the controller revision as an explicit
input. The production publication store must allocate it through durable CAS
and must reject a node publication whose `index_revision` rolls back from the
last accepted value for that node. That store and transport are deployment
components; an in-process counter is deliberately not presented as durable
authority.

WP3.6 already persists the global cache revision in every scale decision and
requires a fresh `reauthorize_prewarm()` immediately before the Kubernetes
mutation. Revision rollback, freshness failure, loss of a ready placement, or
binding changes abort scale-out. The short node TTL limits how long a silent
node can contribute ready evidence, while the final cache-agent validation
protects against filesystem state changing after scheduling.

## Transport and deployment boundary

The Python contracts are JSON-serializable and transport-neutral. The private
cloud deployment must still provide:

- a node DaemonSet or service that publishes the local snapshot;
- an authenticated CRD/API or equivalent durable registry;
- per-node writer identity and RBAC preventing cross-node publication;
- CAS on node index revision and controller cache revision;
- garbage collection after node deletion or publication expiry;
- a controller-owned candidate inventory from Kubernetes/Kueue state; and
- scheduler affinity or binding that constrains new Pods to the selected node.

These resources belong in `private-ai-cloud-iac`. Until that wiring exists,
WP4.4 is a verified library boundary and must not be treated as an enabled
production scheduler integration.

## Failure behavior

- SQLite read or identity failure produces no publication.
- Unverified residency is omitted.
- Missing or stale publication produces `ABSENT`, not `READY`.
- Exact identity mismatch produces `ABSENT`; there is no fallback.
- Duplicate publications or placements reject the aggregate.
- Cache locality never changes controller health or quota facts.
- A later WP4.6 unverified mark removes the hint on the next publication.
- A Runner placed from a still-fresh but stale-positive hint must fail or refill
  through WP4.2 before start; the hint alone is never mount authority.

## Deferred work

- WP4.5: capacity watermarks and generation-fenced eviction. Eviction must
  advance the global index revision so the next hint removes the resident row.
- WP4.6: corruption detection, quarantine, audit, and automatic unverified
  publication withdrawal.
- WP4.7: fenced pre-staging commands and transitions through absent, filling,
  ready, and failed.
- Deployment: transport, durable global revision CAS, scheduler binding, RBAC,
  metrics, and live startup-p95 comparison against non-local placement.

CPU tests cover verified-only/path-free projection, bounded freshness, exact
identity joins, stale/future handling, duplicate rejection, global index
revision changes, canonical ordering, and preservation of controller gates.
