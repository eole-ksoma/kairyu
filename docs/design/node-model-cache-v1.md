# Node model cache v1

Status: WP4.2 fill and WP4.6 Runner-start corruption recovery are implemented
and integrated with the WP4.3 durable index. A fail-closed cache-agent process
entrypoint is implemented; DaemonSet/Pod wiring remains open.

## Purpose and authority boundary

The v1 node cache materializes an admitted, signed model artifact from an
authoritative immutable object source onto node-local storage. SeaweedFS S3 is
the intended production source of record. Node NVMe is a disposable,
reconstructable cache and never becomes an artifact authority.

`NodeModelCacheAgent.ensure_cached()` accepts the same signed envelope, trust
store, and exact GitOps admission request defined by WP4.1. Admission is checked
before cache directories are touched and again after waiting for the per-digest
fill lock. A mutable model name, tag, or source path cannot select cache
content.

WP4.2 supplies the verified fill primitive. `kairyu cache-agent serve` now
assembles it into an authenticated node process, but does not mutate a
Kubernetes workload or assert that a Runner mounted the returned directory.
Those integrations must preserve this API's digest-bound admission result
rather than reconstructing it from labels or paths.

## Published and staging layout

One cache root owns the source-independent layout:

```text
<root>/
  artifacts/<manifest-sha256>/
    manifest.json
    completion.json
    tree/<declared blob paths>
  .staging/<manifest-sha256>/
    manifest.json
    completion.json              # present only after every blob verifies
    tree/<partial or complete blobs>
  .locks/<manifest-sha256>.lock
```

Only `artifacts/<digest>/tree` is a published artifact path. Callers must never
mount or inspect `.staging`. The signed manifest already enforces safe relative
POSIX blob paths, canonical ordering, unique names, bounded path components,
sizes, and lowercase SHA-256 values.

The agent serializes fills with an advisory `flock` keyed by the manifest
digest. Different digests may fill concurrently. After taking the lock, a
caller either validates an existing published entry and returns a cache hit, or
owns the sole fill for that digest. Lock waits can be bounded; timeout is a
fail-closed error and does not create a second writer.

## Download, verification, and resume

`ModelArtifactBlobSource` is the narrow transport boundary. It must return the
bytes for one signed blob beginning at the exact requested offset. Two adapters
are included:

- `HttpRangeModelArtifactBlobSource` addresses
  `<base-url>/<manifest-digest>/<blob-path>`. Initial reads require HTTP 200;
  resumed reads require HTTP 206 with an exact `Content-Range`. A caller may
  supply an authenticated/TLS-configured `httpx.Client` for an S3-compatible
  gateway or presigned endpoint.
- `LocalModelArtifactBlobSource` reads the same digest-namespaced layout from a
  trusted read-only filesystem, useful for offline staging and tests.

For each blob, the agent hashes all existing partial bytes before requesting
the remaining range. An oversized partial is discarded. A complete partial is
reused only after its SHA-256 matches. Appended bytes may not exceed the signed
size; early EOF fails. The final size and SHA-256 must both match the signed
blob declaration. A digest mismatch deletes that blob's partial so the same
invalid bytes cannot be treated as resumable progress.

Source interruptions retain valid partial bytes. A later invocation re-hashes
them and resumes from their current length. Existing staging content is reused
only when its canonical manifest and directory shape match the requested
digest. Structural validation failures discard the staging entry so the next
attempt can cold-fill; invalid or unrelated staging content is
non-authoritative.

## Atomic publication and durability

The fill order is:

1. verify admission and acquire the digest lock;
2. write or resume every blob under `.staging`;
3. verify every blob's signed size and SHA-256;
4. write the canonical completion marker;
5. re-open and sync every declared blob plus manifest/completion metadata,
   then sync staging directories;
6. atomically rename the complete staging directory into `artifacts`; and
7. sync the published parent directory and validate the published entry.

Staging and published paths share one cache root so the rename remains on one
filesystem. No incomplete tree is ever named in the published namespace.
Failure before rename leaves resumable staging only. Failure to publish or
validate is surfaced to the caller; there is no fallback to another revision.
The final file-level durability barrier also runs after a recovered complete
partial, so a process crash followed by retry cannot make only the marker and
rename durable while previously buffered blob data remains unsynchronized.

The completion marker binds the manifest digest, file-tree digest, file count,
and total bytes. Its JSON is canonical when written, bounded when read, rejects
duplicate keys/non-finite constants, and is validated with a strict schema.
Published-hit validation compares the canonical manifest and completion marker,
then requires the exact declared regular-file set and sizes and rejects
symlinks/non-directory path components.

The cache root and its control directories must be owned by the effective
cache user and must not be group/world writable. Cache metadata, locks, and
blobs must have the same owner, must not be group/world writable, and must have
exactly one hard link. Opens use `O_NOFOLLOW`; intermediate components are
checked without following symlinks, and hashing plus append use one open file
descriptor. The cache service account is the trust boundary: no unrelated
process may run with the same UID or receive write access to the root.

Published-hit validation deliberately does not hash a 10-100 GB tree on every
lookup. The initial publisher hashes all content. WP4.6 adds a separate
Runner-start guard that hashes every resident blob, marks corruption
unverified, atomically quarantines it, emits mandatory audit events, and
performs a verified refill. The current startup remains denied after recovery
and must pass a fresh guard call. The full contract is in
`docs/design/node-model-cache-corruption-recovery-v1.md`.

## Result contract

`NodeModelCacheFillResult` returns:

- the admitted deployment and manifest digest;
- the published `tree` path;
- cache-hit status;
- bytes reused from verified partial/complete staging files;
- bytes downloaded during this invocation; and
- signed file-count and total-byte totals.

These fields are process-local evidence for WP4.2. WP4.3 persists node
residency, verification timestamps, last access, and owner-scoped pins in the
index described by `docs/design/node-model-cache-index-v1.md`.

## Fail-closed behavior

The agent refuses publication or cache use when any of these conditions holds:

- signature, signer authority, environment, GPU profile, or GitOps identity
  admission fails;
- the source ignores a resume range or reports inconsistent range/length data;
- source bytes are short, oversized, non-byte chunks, or digest-mismatched;
- cache control, staging, metadata, or published paths have an unexpected type;
- published metadata, file set, or file sizes differ from the signed manifest;
  or
- the digest lock cannot be acquired within its configured deadline.

No error path silently selects a mutable tag, another digest, or another model
revision.

## Deferred work

- WP4.4 is implemented in
  `docs/design/node-model-cache-placement-hints-v1.md`.
- WP4.5 is implemented in `docs/design/node-model-cache-eviction-v1.md` with
  high/low watermarks, deterministic LRU, and generation/pin-fenced deletion.
- WP4.6 is implemented in
  `docs/design/node-model-cache-corruption-recovery-v1.md`.
- WP4.7: deployment/autoscale-driven pre-staging across the target node pool.

Production acceptance still requires deployed 10-100 GB cold/hit/other-node
and concurrent-fill measurements. Those hardware/environment tests should run
after the remaining Phase 4 wiring is available, while the deterministic
failure and concurrency properties remain covered by CPU tests in each WP.
