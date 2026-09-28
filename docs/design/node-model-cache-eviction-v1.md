# Node model cache eviction v1

Status: WP4.5 implemented as a node-local capacity planner and a
generation-fenced filesystem/index executor. Daemon scheduling, configured
production watermarks, metrics, and live NVMe pressure evidence remain
deployment work.

## Purpose

The node cache is regenerable, but deletion is permitted only when the index
proves that the exact artifact generation is unpinned. The policy uses absolute
indexed artifact bytes and two thresholds:

- at or below `high_watermark_bytes`, eviction does not start;
- above the high watermark, oldest eligible artifacts are selected until their
  declared bytes would return the index to `low_watermark_bytes`; and
- a strict low-below-high relation supplies hysteresis and prevents continuous
  churn near one threshold.

The configured thresholds must leave separate headroom for staging files,
SQLite, filesystem metadata, and eviction tombstones. They are not a claim
about raw filesystem occupancy.

## Protection contract

Every owner-scoped WP4.3 pin and every WP4.6 recovery-required row excludes its
artifact from LRU selection. The node daemon must maintain pins for all
protected roles, including:

- a currently active Runner or deployment;
- an approved rollback target;
- operator/manual retention; and
- later WP4.7 pre-stage ownership while the target remains required.

There is no weaker name-based exception. A missing active or rollback pin is a
control-plane bug, not permission for the cache layer to infer protection.
When eligible bytes cannot reach the low watermark,
`blocked_reclaim_bytes` records the shortfall and no pinned artifact is added to
the plan.

## Deterministic planning

`plan_node_model_cache_eviction()` consumes one transactionally consistent
`NodeModelCacheIndexSnapshot`. It sums bounded `total_bytes`, starts only above
the high watermark, and sorts unpinned victims by:

```text
(last_access_at_ns ascending, manifest_digest ascending)
```

Each victim binds the digest, model identity, declared bytes, last-access time,
and row generation. The plan also binds the node ID and global index revision.
Malformed ordering, duplicate digests, inconsistent totals, integer coercion,
and signed-64-bit overflow fail closed.

## Fenced execution

`NodeModelCacheEvictor.execute()` serializes plans with one node-local eviction
lock and first requires the current node ID and global index revision to equal
the plan. It recomputes the canonical plan from that snapshot and the embedded
policy and requires complete equality, so a caller cannot substitute victims or
accounting. Before every later victim it also requires the revision to equal
the source revision plus this executor's completed deletions. The same expected
revision is checked again inside the victim's SQLite write transaction before
the filesystem detach. Any unrelated or last-moment mutation stops the
remaining work instead of racing or over-reclaiming. For each victim it then:

1. takes the same digest lock used by cache fill;
2. opens an SQLite immediate transaction;
3. verifies the exact row generation, absence of every pin, and absence of a
   recovery ID;
4. atomically renames `<root>/artifacts/<digest>` to the private
   `<root>/.evicting/<digest>.<index-revision>.<generation>` namespace;
5. deletes that exact index row and commits, advancing the global revision by a
   schema trigger; and
6. removes the detached directory and syncs the affected directories.

Concurrent fill cannot pass the digest lock. Concurrent touch, verification,
or pin mutation cannot pass the SQLite write transaction. A stale plan,
generation change, or late pin therefore leaves the published tree and index
record intact.

Filesystem rename and SQLite commit cannot form one cross-resource atomic
operation. The deterministic tombstone is the recovery journal:

- if the row and global revision still equal both pre-delete fences and the
  published directory is absent, recovery restores the tombstone;
- if the row is absent, recovery removes the already committed tombstone; and
- if a later verified tree and index revision are present, recovery removes the
  superseded tombstone without confusing a reset row generation for the old
  one. Ambiguous identities, unsafe paths, or conflicting fences stop recovery
  without deleting data.

The daemon must call recovery before publishing placement hints or applying a
new plan. A cleanup failure after database commit is loud and recoverable on the
next pass; it never recreates an index claim for missing bytes.

## Interaction with placement and Runner startup

Successful index deletion advances the WP4.4 node revision, so the next hint no
longer reports the artifact. A still-live older hint remains advisory only: the
Runner must pass WP4.2 validation or refill before use. Eviction cannot select a
different revision as fallback.

## Deployment boundary

The private-cloud deployment must still provide:

- per-node high/low values sized from usable NVMe capacity;
- owner pin reconciliation for active, rollback, and pre-stage lifecycles;
- a bounded daemon loop that recovers, snapshots, plans, executes, and repeats
  if pressure remains;
- metrics for used/planned/reclaimed/blocked bytes, conflicts, and cleanup
  failures; and
- alerts when pinned capacity prevents reaching the low watermark.

## Deferred work

- WP4.6 is implemented in
  `docs/design/node-model-cache-corruption-recovery-v1.md`; quarantined evidence
  is outside the eviction namespace and has a separate retention policy.
- WP4.7: controller-driven pre-stage intent and its pin lifecycle.
- Deployment acceptance: fill the real node cache under active/rollback pins
  and prove no protected revision is removed.

CPU tests cover hysteresis, deterministic LRU, active/rollback/manual pin
protection, insufficient eligible capacity, global-plan staleness,
generation/pin races, revision advancement, tree removal, and both interrupted
rename recovery outcomes.
