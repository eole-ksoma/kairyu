# Node model cache corruption recovery v1

Status: WP4.6 implemented as a fail-closed Runner-start verification,
quarantine, audit, and verified refill library. Node-daemon/Runner wiring,
production audit transport, quarantine retention, and live NVMe fault injection
remain deployment work.

## Purpose

WP4.2 proves bytes while publishing them, but a resident 10-100 GB tree can be
damaged later without changing file size. `verify_for_runner_start()` therefore
re-runs signed-manifest admission and hashes every declared blob before a
Runner may use a cache path. Structural validation, the durable WP4.3 verified
bit, and full SHA-256 verification are all mandatory.

This is a startup authorization boundary, not a periodic cache hit. The node
daemon should call it once for the exact digest immediately before activating a
Runner and must consume the returned decision. A plain WP4.2 cache hit or WP4.4
placement hint is never sufficient startup authority.

## Verification and startup decision

Verification holds the same per-digest lock used by fill and eviction. It
requires the signed manifest identity to match the indexed model, revision,
digest-derived path, byte total, and file count. For each blob it opens the
declared regular file with `O_NOFOLLOW`, checks ownership, mode, link count, and
size, then hashes the complete file through that descriptor.

Only a `runner_start_allowed=true`, `reason=verified` decision exposes the tree
path. Any detected corruption makes the current decision false even when an
automatic refill succeeds. The caller must make a new verification call; this
prevents the failed verification attempt from becoming the Runner launch
authorization.

## Quarantine and refill transaction

On a digest mismatch, structural failure, or pre-existing unverified index row,
the agent:

1. durably starts or resumes a globally unique `recovery_id` and marks the exact
   index row unverified;
2. atomically renames the published digest directory into the private
   `.quarantine/<digest>.<recovery-id>` namespace (or records an empty marker
   when the published tree is already missing);
3. syncs both affected directories;
4. synchronously emits a `corruption_quarantined` event;
5. performs a WP4.2 refill without changing the recovery-required index row;
6. re-locks the digest and fully verifies the replacement;
7. synchronously emits either `corruption_refetched` or
   `corruption_refetch_failed`; and
8. only after successful refetch audit, completes an exact recovery-ID and row-
   generation CAS that restores verified state.

The refill uses the original signed envelope and exact GitOps admission request.
It cannot select another revision or mutable tag. Quarantined bytes are retained
for forensics; this library never makes them mountable and never deletes them.

The index stays unverified if download, digest validation, publication, or audit
fails. If a process crashes, or an audit write fails, the durable recovery ID
selects the same incident and resumes the audit/refill sequence. A valid
replacement already published for that incident is re-hashed and can resume at
the final audit/CAS step. Ordinary `ensure_cached()` calls cannot clear a
recovery-required row, and a changed token or generation makes stale completion
fail closed.

## Audit contract

`NodeModelCacheAuditSink.emit()` is synchronous and must persist or raise. Each
bounded event includes the event kind, nanosecond time, node/deployment/model
identity, manifest digest, globally unique recovery ID, quarantined record
generation, reason, and absolute quarantine path. A production sink should
deduplicate retries by at least `(node_id, manifest_digest, recovery_id, event)`
and durably export the events to the private-cloud audit backend.

No source URL, signer secret, artifact contents, or arbitrary exception text is
included. A sink failure is a cache error and denies Runner startup; silent
event loss is forbidden.

## Interaction with placement and eviction

Marking the row unverified advances the node-local index revision. The next
WP4.4 publication omits it, and WP3.6 observes `ABSENT` rather than `READY`.
WP4.5 excludes every recovery-required row in both planning and its final
SQLite deletion fence. Owner-scoped pins remain attached during recovery so a
corruption incident cannot silently drop active or rollback protection.

The private quarantine namespace is outside indexed capacity and eviction.
Production deployment must bound its retention separately and preserve any
required incident evidence before cleanup.

## Deployment boundary

The private-cloud deployment must still provide:

- a node-daemon call immediately before Runner activation;
- a durable, retry-safe audit sink and alerts for every failed refill or audit;
- quarantine retention/export and capacity limits;
- source credentials and network policy for verified refetch;
- metrics for verification duration, bytes hashed, corruptions, retries, and
  recovery latency; and
- live same-size corruption tests proving that no Runner starts from bad bytes.

WP4.7 adds controller-driven pre-staging. CPU tests cover clean startup,
same-size corruption, atomic quarantine, verified refill, current-start denial,
refetch failure, audit failure before and after refill, ordinary-fill bypass,
missing published trees, schema migration/old-writer fencing, eviction
exclusion, and recovery-ID/generation-fenced completion.
