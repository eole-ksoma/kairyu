# Node model cache index v1

Status: WP4.3 implemented as a durable node-local SQLite index and integrated
with the WP4.2 cache fill agent. WP4.4 publishes its verified residency as
bounded placement hints. Deployment daemon wiring remains open.

## Purpose and authority boundary

The cache index records which signed model revisions a node has successfully
materialized, how many bytes they occupy, when their verified publication was
observed and last accessed, and which owners currently pin them. It is the
durable input for later placement and eviction policy; it is not a model
artifact authority.

SeaweedFS S3 remains the artifact source of record. The signed manifest remains
the model identity, and the WP4.2 published tree plus completion marker remains
the local evidence used before returning a path. An index row alone never
authorizes a Runner start. A stale row may cause a future placement miss, but
the cache agent must still validate or refill the tree before use.

## Storage and node binding

`NodeModelCacheIndex` owns one SQLite database inside the cache root, normally:

```text
<cache-root>/cache-index.sqlite3
```

The database is bound at creation to:

- schema identity `kairyu-node-model-cache-index-v3`;
- SQLite application ID `KAIC` and user version 3; and
- one explicit node ID supplied by the node daemon or Downward API; and
- the exact cache-root path containing the database.

Opening the database with another node ID, cache root, application ID, or schema
version fails closed. Copying an index to another node or root cannot claim
residency there.
The path must be absolute, its parent must be owned by the cache UID and not
group/world writable, and the database must be a single-link regular file owned
by that UID and not group/world writable.

SQLite runs in WAL mode with foreign keys enabled, a bounded busy timeout, and
FULL synchronous writes. Mutations use `BEGIN IMMEDIATE`, so concurrent cache
processes serialize changes without lost pin or access updates. The schema and
indexes are created transactionally.

WP4.4 upgrades the original user-version-1 schema through version 2, rebuilding
the entry/pin tables with signed-64-bit generation bounds and creating the
global revision. WP4.6 then upgrades version 2 to version 3 by adding the
dedicated recovery ID and database recovery guards. Both paths are one
transaction, so a v1 database reaches v3 or rolls back unchanged. New v1/v2
writers reject v3. A v2 connection that passed its version check before
migration is still prevented by `BEFORE UPDATE/DELETE` guards from clearing or
deleting a recovery-required row. Trigger presence and security-relevant guard
definitions are checked whenever v3 opens.

## Entry contract

Each `manifest_digest` primary key binds immutable cache metadata:

- node ID;
- model ID and model revision;
- the exact `<cache-root>/artifacts/<digest>/tree` path;
- total bytes and file count from the signed manifest;
- verified state, its evidence source, an optional failure reason, and an
  optional globally unique recovery ID;
- verification and last-access Unix timestamps in nanoseconds; and
- a monotonically increasing row generation.

The database also holds a global revision that starts at one and advances once
for each observable row or pin mutation. `snapshot()` reads that revision and
all records in one SQLite snapshot, providing WP4.4 with a monotonic node-local
publication source without mixing revisions.

New v1 entries are created only from verified resident trees.
`verification_source` is `filled` when this process downloaded and hashed every
blob, or `published_marker` when it admitted and validated an existing WP4.2
publication. A later hit cannot downgrade `filled` evidence. Re-recording an
identical digest advances last access; any model/path/size/file-count conflict
for that digest is rejected rather than overwritten.

`mark_unverified()` durably revokes the verified state with a bounded generic
reason. An unverified row cannot be touched or pinned, and a structural
`published_marker` hit cannot restore it. A generic unverified row may be
restored by a new `filled` result. WP4.6 instead uses `begin_recovery()`, whose
dedicated recovery ID cannot be cleared by `record_verified()`. Only
`complete_recovery()` may restore it, under the exact recovery ID, row
generation, immutable identity, full-digest verification, digest lock, and
successful audit. Recovery-required rows also cannot be deleted by eviction.

Wall-clock regression cannot move `last_access_at_ns` backward. A row generation
advances only when durable observable state changes, allowing later eviction or
placement code to detect stale snapshots.

## Owner-scoped pins

Pins are separate rows keyed by `(manifest_digest, owner)`. This prevents one
deployment, active Runner set, or rollback controller from releasing another
owner's protection. `pin(digest, owner, reason)` is idempotent for the same
reason and updates only that owner's reason; `unpin` releases only that owner
and is also idempotent.

`NodeModelCacheRecord.pin_owners` is sorted and unique, and `pinned` is derived
from whether any owner remains. Pinning an unknown digest fails closed. Reasons
and pin timestamps remain durable internal evidence. WP4.5 treats every owner
pin as authoritative protection; controllers use distinct owners for active,
rollback-target, pre-stage, and administrative retention lifecycles.

## Cache-agent integration

`NodeModelCacheAgent` accepts an optional index instance. It writes the index
only after:

1. signature and GitOps admission succeed;
2. a cold fill is digest-verified and atomically published, or an existing
   publication passes WP4.2 validation; and
3. the final published path is available.

Interrupted, short, oversized, digest-mismatched, or otherwise unpublished
fills never create residency rows. If index persistence fails after atomic
publication, the cache operation fails to its caller while leaving the complete
disposable tree in place. A later admitted hit can repair the missing index row.
The tree is not rolled back because SQLite is metadata, not artifact authority.

An index must live in the same cache root as its agent. `record_verified`
requires the exact digest-derived artifact path and rejects an index attached
to another root.

## Reads and consistency

`get()` returns one digest, while `list_records()` provides a stable
model-ID/model-revision/digest ordered node snapshot. Reads validate database
identity, node binding, cache-root binding, and digest-derived artifact paths on
every connection. Entry state and its owner pins are read in one explicit
SQLite snapshot so the returned generation and pin set cannot be torn across a
concurrent update. Returned records are immutable, strictly validated Pydantic
values.

The index does not scan or trust arbitrary filesystem trees. If an operator
deletes a cache entry outside the cache service, its row can remain stale until
the later eviction/reconciliation path removes it. Consumers therefore treat
residency as a placement hint and must call the WP4.2 agent before mounting the
artifact.

WP4.4's projection and controller join are defined in
`docs/design/node-model-cache-placement-hints-v1.md`.

## Deferred work

- WP4.5 is implemented in `docs/design/node-model-cache-eviction-v1.md` with
  capacity watermarks, generation-fenced LRU deletion, and pin protection.
- WP4.6 is implemented in
  `docs/design/node-model-cache-corruption-recovery-v1.md`.
- WP4.7: deployment/autoscale pre-staging and its pin lifecycle.

The node daemon, service account, cache path, node ID injection, and live load
evidence remain deployment work. CPU tests cover concurrent constructors and
writers, snapshot-consistent reads, transactional initialization,
persistence/reopen, node/root identity conflicts, clock regression, pin
composition, permission checks, post-publication index repair, and fail-closed
corruption/recovery behavior.
