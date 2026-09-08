# M10 Design: Fleet Elasticity (M10a) + KV-Aware Routing (M10b) — CPU Halves

Status: **M10a + M10b Implemented** (2026-07-03; D7/A13 amended
2026-07-27; D1/D2/D5/A16–A26 amended 2026-07-28; D5/A27 amended
2026-07-28; D7/A31 amended 2026-07-28; D6/A32 amended 2026-07-29;
D8/A33 amended 2026-07-29; D4/A34 amended 2026-07-31; D5/A35 amended
2026-08-03; D6 amended 2026-08-07 for issue #344; D5/A19 amended
2026-08-07 for issue #347).
Reviewed (1-reviewer panel with repo-line evidence; §6 binding; covers
M10a+M10b).
Milestone: M10a/M10b (roadmap Track F1/F2; goal G5 base)
Date: 2026-07-03
Depends on: m7 ReplicaPool/JsonlRouterLog, m7 deploy (spec/builder/prober),
m9 server surface, M18 (KV events source shape). Consumed by: G5 fleet
gates, M11 product surface.

## 1. Goal

Thousands-of-GPUs operation needs (a) replicas that come and go without
restarts, (b) a discovery/reconciliation loop, (c) traces for debugging a
distributed request path, and (d) routing that knows where the KV prefix
lives. All logic CPU-testable; k8s manifests exercised by a kind smoke.

## 2. M10a decisions

### D1 — Dynamic ReplicaPool membership (`orchestration/replica.py` rework)

Index-keyed lists → id-keyed `_ReplicaEntry` (backend, outstanding,
consecutive_failures, manual drain owner, drain-lease set). API:
`add_replica(replica_id, backend)`,
async `remove_replica(replica_id)` (refuses while outstanding>0 unless force,
otherwise removes ownership and awaits shared `shutdown_all` exactly once),
`drain(replica_id)` / `cancel_drain(replica_id)` (acquire/release only the
manual owner), `acquire_drain(replica_id) -> DrainLease` /
`release_drain(replica_id, lease)` (opaque independent owners),
`is_manually_draining(replica_id)` (manual owner only),
`entry_generation(replica_id)` (opaque identity for one entry lifetime),
`require_probe(replica_id)`, and `probe(replica_id)`. A remote entry with a
declared health URL starts unvalidated; only a successful readiness probe makes
it healthy and eligible. Entries without a readiness URL remain locally trusted.
A replica is draining while the manual owner or any lease
is active. Removing and re-adding the same ID creates a new entry generation.
Releasing one owner does not alter other owners, health, or outstanding state.
HRW hashing keys on `replica_id` STRINGS (not indices)
— adding/removing a replica remaps only ~1/N sessions (property test).
Constructor accepts `dict[str, EngineBackend]` or the legacy sequence
(auto-ids `"0".."N-1"`, existing tests unchanged). Selection precedence:
healthy ∧ not-draining → session HRW → queue-depth fallback →
least-outstanding.

### D2 — Registry + reconciler + discovery (`deploy/registry.py`)

Discovery carries frozen `ReplicaConfig(address, model?, api_key_env)` values;
the reconciler resolves each one to a complete frozen
`ReplicaIdentity(address, model, api_key_env)` before any pool mutation.
`ReplicaRegistry` stores typed TTL-heartbeat membership
(`register(id, address, ttl_s, model?, api_key_env)`, `heartbeat(id)`,
`alive()` — monotonic-clock injected for tests). `DiscoverySource` protocol:
`poll() -> dict[id, ReplicaConfig]`; `StaticDiscovery` snapshots either string
addresses or typed configs and `RegistryDiscovery` exposes registry snapshots.
The Kubernetes source is a lifecycle-owned in-cluster EndpointSlice REST
adapter behind that protocol. It reads the projected service-account token on
every poll, verifies the API server with the mounted CA, selects one named or
numeric TCP port, and admits only `ready=true`, non-terminating IPv4/IPv6
endpoints. Pod UID is the stable identity when present, with name/address
fallback; dual-stack rows for the same UID are aggregated once using an
explicit address-family preference and deterministic address selection.

`PoolReconciler(pool, source, factory, default_model?)` passes only complete
identities to `Callable[[ReplicaIdentity], (EngineBackend, health_url)]`.
Discovery's non-empty model wins over the default and its auth value is complete,
including explicit `None`; a missing model fails before mutation. Reconciliation
tracks applied identities together with each pool entry's opaque generation and
runs deterministic phases: add missing IDs in
desired order, construct-before-drain replacement of changed identities in
desired order, then drain/remove absent IDs in pool order. Removal is awaited and
closes the removed backend through `shutdown_all`; unused candidates are also
closed through that helper, with cleanup failures propagated. In-flight refusal
keeps the applied identity for retry. The reconciler keeps the opaque lease returned
for each replacement/removal it initiates: if intent reverts to the applied identity,
or a retry factory fails, it releases only that lease. Manual/admin drains remain
active whether acquired before or after the reconciler lease, and manual undrain
cannot release a reconciler lease. Successful same-ID replacement transfers only
the manual owner to the new backend entry; the reconciler lease is not transferred,
and a new remote entry starts unvalidated with fresh failure and outstanding
state. Lease tracking
is cleared when a replica disappears or a new backend is successfully installed.
If an external caller removes and re-adds the same ID between ticks, the changed
generation invalidates every applied identity and lease owned for the old entry.
An absent desired ID then receives a fresh lease on the new entry; a present desired
ID baselines the externally installed entry at the current desired identity without
factory, replacement, or shutdown side effects. A fresh entry's manual owner is
never changed by generation reconciliation.
Each entry generation is a serializable random token. Effective membership
transitions (add/remove, drain/undrain, first successful probe, health ejection,
and restore) enter a reconciler-owned ordered outbox with a source ID,
monotonic sequence, and stable event ID. A sink failure retains the head event
for identical retry before further discovery; already-applied partial
mutations are therefore not lost. Placement rows carry the same generation,
the server-assigned request ID, pool/eligible sizes, and process-ingress to
selection latency.
Server: `POST /admin/drain` marks the pool replica draining and flips `/readyz`
to 503 (existing prober contract).

### D3 — `BatchStoreProtocol` and streaming file storage

The file-backed batch store's surface (`create/get/update/list`) is extracted
to a Protocol so M11 tenancy ledgers and tests can fake it. The surface also
includes owner-scoped lazy input-line iteration, an async streaming upload
transaction, owner-scoped fixed-chunk content iteration, and a transactional
JSONL writer. Streaming uploads enforce their byte limit incrementally and
remove partial state on cancellation or failure; both writers publish metadata
only after temporary content is closed and atomically renamed.

### D4 — OTel tracing (`entrypoints/server/tracing.py`)

Deferred `import opentelemetry`; `ServerSettings.tracing=False` default.
`traced_span(name, attrs)` context manager: no-op when disabled or OTel
missing (server runs without the dependency). Spans: gateway request →
pool placement (replica_id, reason) → backend generate; Conductor stages.
Tests use OTel's InMemorySpanExporter (dev dependency) and assert the span
tree + attributes.

### D5 — Helm chart + kind gates

`deploy/helm/kairyu/`: Deployment (readiness=/readyz, liveness=/healthz),
Service, ConfigMap (DeploymentSpec JSON), values.yaml (replicas, image,
resources; `values-gpu.yaml` arrives in M19). `scripts/kind_smoke.sh`:
kind create → build image → load → helm install → wait ready → curl
/v1/models + a completion → teardown. CI job (`kind-smoke`) runs it on
ubuntu-latest; locally optional. A CPU test pins that the chart renders
(`helm template` golden) so drift fails fast without kind.

G5 F1a adds a separate resource-bounded kind topology: one dynamic gateway,
one headless Service, and a 200-member parallel StatefulSet using a static
standard-library-only mock. Its fixed-seed driver applies ten disjoint
20-pod churn batches at absolute one-minute boundaries under retry-free
open-loop traffic. It joins the echoed server request ID to the raw gateway
placement row, joins the selected replica UID to the mock response, and records
pod/EndpointSlice/resource snapshots plus the exact kill/join schedule.
The verifier rehashes and replays every JSONL sidecar. Pull requests run the
same protocol at smoke scale; the frozen 200-replica profile is scheduled and
manually dispatchable. Images and manifests are built from a clean Git archive,
carry the source revision, and run on digest-pinned kind/Kubernetes inputs.
Gate traffic reaches a fixed NodePort through kind's localhost port mapping;
`kubectl port-forward` is excluded because its process-local file-descriptor
and tunnel lifecycle are not part of the gateway-under-test.

## 3. M10b decisions (implemented after M10a in the same doc's scope)

### D6 — Prefix index + KV-aware selection (`orchestration/prefix_index.py`)

Block-granular approximate trie: `observe(replica_id, token_blocks)`,
`overlap(replica_id, token_blocks) -> int`. `ReplicaPool` gains optional
`prefix_index=` + score `α·overlap − β·outstanding` over reverse-indexed warm
members of the immutable eligible snapshot. Cold placement has a conservative
score baseline of zero: the warm maximum wins when its score is non-negative,
a zero warm/cold tie prefers reuse, and a negative warm maximum falls through
to the existing session-HRW plus queue-depth valve (or least-outstanding for a
request without a session). Equal-score warm candidates use session HRW as the
deterministic tiebreak. `enabled=False` is the default. Only a successful unary
generation or a stream whose first backend result proves prefill landed
advertises the selected prefix. The
approximate, process-local index uses versioned XXH3-64 cumulative keys
(`xxh3-64-v1`), matching vLLM Router's non-cryptographic routing-key approach;
the 64-bit key width is unchanged and a collision can only cause a cache miss,
never alter generated output. Conductor carries a trusted
`CacheHint.prefix_fingerprint` only when its shared prefix covers the complete
default 256-character root chunk; the carried value is the exact same XXH3 key
as local hashing. A blank `CacheHint` declares session-only affinity and bypasses
native prefix hashing/publication; this covers empty/short Conductor prefixes
and normal HTTP session traffic without charging them for undeclared
cross-session reuse. Those callers retain same-session HRW locality but do not
learn or discover cross-session prefixes until they provide an exact root.
Sessionless callers retain local discovery, while
malformed non-empty hints and custom chunk sizes retain the exact local
fallback. Each tracked request owns one bounded, lazy hash chain shared by
selection and successful publication. A cold success initially advertises only
its root key through a dedicated root-publication fast path,
which is sufficient to make the next related request discoverable; a successful
warm `prefix_match` extends that same request-local chain and promotes the entry
to full usable depth. A streaming request publishes that root immediately before
yielding its first backend result, which proves prefill landed; a failure or
cancellation before that boundary cannot poison the index. Normal completion
still performs full-key promotion for a warm hit. Unary generation has no
observable first-token boundary and therefore publishes only after its
successful result. Thus a complete prompt chunk is hashed at most once per
request, one-off cold prompts do not populate unneeded deep index entries, and
no process-global prompt cache needs invalidation. Decision reason
`prefix_match` enters the router log.

### D7 — RadixKV events → gateway index (`radix_kv.py` event_sink + `kv_index.py`)

`RadixKVCache(event_sink=...)`: emits BlockStored/BlockRemoved
(vLLM-compatible schema) from allocate/commit/evict. A background ZMQ
publisher and gateway subscriber update a separate exact block-hash index.
Only a contiguous, high-water-confirmed source epoch inside its route-time
lease may enter exact scoring. The default lease is 400 ms and F2b uses
250 ms; expiry at the boundary, a sequence gap, an epoch transition, or an
inactive member makes the whole eligible request fall back to the approximate
trie. Bounded replay or an authoritative snapshot restores exact routing
without restarting the pool, router, index, publisher, or subscriber.

**Incremental hash-chain amendment (2026-07-27, issue #220).** Event-enabled
radix nodes retain the SHA-256 continuation state for the canonical
`repr(tuple(prefix))` token stream, the prefix token count, and their local
block-boundary digests. A child copies its parent's continuation once and
feeds only its own token suffix. Tuple closing punctuation is appended only to
a digest snapshot, including the singleton trailing comma, so emitted hashes
remain byte-identical to the original full-prefix protocol. Splits reuse the
existing digest suffix and derive only the new upper node; stored and removed
events read node-local hashes directly. Caches without an event sink allocate
no hashers and perform no SHA work. Reproduce the long-prefix comparison with
`uv run python verification/fleet/correctness/kv_event_hash_bench.py --tokens 32768 --repeats 5`.

### D8 — Learning placement

Production `JsonlRouterLog` placement rows (`kind=replica`) are joined
one-to-one by request ID with benchmark `placement_outcome` rows carrying TTFT,
then consumed by `learning/dataset.py`. The former chosen-action-agreement
grid is withdrawn: scoring each candidate only where it agrees with the
logging policy selects a different observed subset for each candidate and
does not reveal the counterfactual TTFT of its unchosen actions.

For the prefix score `α * prefix - β * load`, multiplying both weights by the
same positive constant preserves every score ordering, zero threshold, and
tie. Only `λ = β / α` is identifiable, so offline policy candidates normalize
`α=1`; the declared production baseline is `λ=0.25`. Tuning uses complete
stateful episode replay from the same frozen initial cache/background-load
state for every candidate, with training and held-out request families
disjoint. Every declared candidate is evaluated on training episodes, the
minimum-mean-TTFT candidate is frozen, and held-out episodes compare only that
frozen candidate with the baseline on the same trace from the same frozen
initial state. This remains deterministic offline policy selection, not
online learning or the M4 request-family bandit.

## 4. Non-goals

- Cross-cluster federation; autoscaler execution (M11 F5 logic only).
- KV-event compression/batching tuning (deploy-day).

## 5. Verification

- HRW remap property: removing 1 of N replicas remaps only sessions that
  lived on it; adding remaps ≤ ceil(S/N)+slack.
- Drain: no new placements, in-flight completes, then removable; /readyz
  503 while draining.
- Reconciler diff: add/remove/no-op paths with a fake source; TTL expiry
  drops replicas.
- EndpointSlice discovery: readiness/termination filtering, named and numeric
  ports, token rotation, deterministic dual-stack UID aggregation, lifecycle
  close, and failure-resilient polling.
- F1a: one gateway plus 200 mock replicas, ten minutes of 10%-per-minute churn,
  zero failed requests, and placement p99 below 10 ms. Raw request,
  membership, pod, EndpointSlice, resource, and kill/join sidecars must pass
  the independent artifact replay.
- F1b: one gateway plus an exact 100-replica StatefulSet undergoes one
  drain-first, partitioned rolling restart under retry-free traffic. Every old
  ordinal is replaced exactly once, every offered request succeeds, and raw
  rollout, drain, readiness, membership, placement, Pod, and EndpointSlice
  evidence must pass the independent artifact replay.
- F1c: three independently identifiable gateways sit behind an
  `X-Session-ID` rendezvous-hash L7 load balancer and share one PostgreSQL
  BatchStore. Cross-gateway affinity, file/job visibility, fenced lease
  takeover after owner-Pod loss, and identical terminal output through every
  gateway must pass one raw-evidence replay.
- Tracing: span tree with InMemorySpanExporter; disabled → zero overhead
  (no otel import).
- Helm: `helm template` golden test; kind smoke in CI.
- M10b: prefix routing beats the same-request session-HRW baseline on the
  replayable F2a 500-entry shared-prefix trace, remains non-inferior on the
  paired uniform trace, and keeps placement p99 below 10 ms; staleness fallback
  chaos test.

## 6. Review record (binding amendments)

- **A1**: auto-ids are "0".."N-1" (Prometheus labels stay stable);
  ``probe()`` accepts int (ordinal) or str id; the router log keeps
  ``replica`` as ordinal and ADDS ``replica_id``; ``outstanding``/``healthy``
  stay insertion-order tuples (+ ``*_by_id`` variants).
- **A2**: in-flight completion on a removed id is a no-op (guarded
  decrement); streams count as in-flight until generator close.
- **A3**: HRW runs over ELIGIBLE entries (healthy ∧ not draining) — draining
  remaps its sessions immediately; all-draining raises like all-unhealthy.
- **A4**: health URLs live with the pool entries (dict[id, url]); the prober
  keys by id; probe() resets failures but NEVER clears draining.
- **A5**: /admin/drain semantics split by node role — replica node: sets
  app.state.draining → /readyz 503 (the prober sees it); gateway: drains a
  pool member; only zero-ELIGIBLE replicas 503 the gateway readyz.
- **A6**: reconciler factory =
  `Callable[[ReplicaIdentity], (EngineBackend, health_url)]`; the default closes
  over `create_backend("openai")` with the identity's address, resolved model,
  and auth environment plus the readiness URL `/v1`-strip rule. Reconcile
  tolerates in-flight remove refusal, closes each unused candidate, and retries
  next tick without changing the applied identity.
- **A7**: registry takes ``now: Callable[[], float] = time.monotonic``.
- **A8**: BatchStoreProtocol is the FULL 12-method surface (save_file,
  save_file_streaming, get_file, read_file_content, iter_file_content,
  iter_file_lines, create_jsonl_writer, create_batch, get_batch, list_batches,
  update_batch, recover_orphans) + FileObject/BatchJob/JsonlFileWriter models.
- **A9**: traced_span lives in ``kairyu/telemetry.py`` (L2 must not import
  L3); the gateway request span is a middleware; ServerSection threading
  already copies model_fields — only the field addition is needed.
- **A10**: opentelemetry-sdk added to the dev group (and an ``otel`` extra).
- **A11**: chart mounts the DeploymentSpec YAML at exactly
  /etc/kairyu/config.yaml (the Dockerfile CMD); kind-smoke is a third CI
  job; the gateway image needs the ``fleet`` extra for the D7 subscriber.
- **A12 (M10b)**: the gateway has NO token ids — the approximate trie keys
  on fixed-size TEXT chunks of the prompt (documented approximation); the
  KV-event index is a SEPARATE per-replica block-hash structure with
  staleness tracking; key unification via gateway tokenization is a
  deploy-time option (install tokenizers in the gateway image).
- **A13 (M10b)**: BlockStored emits on the computed False→True TRANSITION
  (mark_computed + commit_and_release decode-extension nodes; never
  allocate; guard the _release double-fire); BlockRemoved only from
  _ensure_free eviction; _split emits nothing; release_preempted emits
  nothing (never stored); vLLM schema fields block_hashes/parent_block_hash/
  token_ids/block_size + ts. Its original replay-endpoint non-goal is
  superseded by A31's bounded recovery protocol.
  Block hashes use the node-cached incremental SHA continuation while remaining
  exactly compatible with the original `sha256(repr(prefix))[:16]` values.
- **A14**: drain cancellation is ownership-scoped. `_ReplicaEntry` separately
  stores the manual owner and opaque `DrainLease` owners; effective draining is
  their OR. `PoolReconciler` records the exact lease it acquired, and a desired
  identity reverting to the applied identity or a retry factory failure releases
  only that lease. Manual/admin drains remain authoritative even when asserted
  after a reconciler lease; manual undrain likewise cannot release that lease.
  A successful same-ID replacement carries the manual owner to the new entry but
  not the reconciler lease; health and outstanding state reset with the backend
  entry. Removal, disappearance, and successful installation clear reconciler
  tracking. Applied identities and leases are bound to the pool's opaque entry
  generation, so an external same-ID remove/add also clears old tracking: desired
  absence acquires a fresh lease, while desired presence baselines the fresh entry
  without replacing it. This never clears a fresh entry's manual owner. Releasing
  any owner never changes health or outstanding state.
- **A15**: readiness validation is generation-scoped. A declared remote health
  URL makes a new entry unknown and ineligible until `probe()` succeeds; no URL
  retains locally trusted compatibility. The serve-layer prober performs its
  first tick immediately, snapshots unknown/ejected entries by ID, entry
  generation, and URL, probes them with bounded concurrency, isolates per-entry
  failures, and applies a 200 response only to the same generation. `/readyz`
  and placement use the same validated-and-not-ejected predicate, while
  `probe()` resets failures without clearing any drain owner.
- **A16**: the deploy-day Kubernetes adapter is now production code rather than
  a non-goal. EndpointSlice `targetRef.uid` is the replica identity and the
  source rejects not-ready or terminating endpoints before reconciliation.
  Reconciliation and health changes publish post-transition snapshots through an
  ordered retryable outbox; placement and membership rows share serializable
  entry generations. F1a evidence is fail-closed: no request retry, 429/5xx,
  transport errors, unsent arrivals, missing request-to-placement joins,
  incomplete final gateway membership, missing raw sidecars, or any placement
  p99 window at or above 10 ms may pass.
- **A17 (superseded by A18)**: one dynamic `ReplicaPool` owns one outbound
  `AsyncClient`, shared by every discovered OpenAI backend and its readiness
  prober. Replica removal and replacement never close that transport;
  application teardown stops reconcilers/probers/backends first and then closes
  the sole owner exactly once, cancellation-safely. Data calls retain their
  backend timeout and probes retain their shorter timeout per request. Active
  and idle connection-count limits are open because trusted EndpointSlice
  membership has no declared fleet ceiling; gateway admission and the probe
  semaphore bound active use, while a 30-second idle expiry bounds sockets for
  churned-out Pod IPs. This reduces client and TLS setup from O(replicas) to
  O(dynamic pools), matching the shared-client design used by current vLLM
  routers.
- **A18**: one dynamic `ReplicaPool` eagerly creates one immutable TLS context,
  then each discovered OpenAI backend lazily creates and owns an origin-local
  `AsyncClient` from it. Removing or replacing a replica closes only that
  replica's client, cancellation-safely through the existing backend lifecycle.
  When configured, gateway admission bounds active data connections; the
  transport does not impose a second queue. When an origin's client is lazily
  created, a gateway with finite concurrency `C` and `R` live replicas retains
  `min(64, max(1, ceil(C / R)))` demand-created idle connections for that
  origin. Thus a 128-client, DP=2 serving wave retains 64 per origin, while
  F1a's 256-request/200-replica envelope retains two. An unbounded gateway uses
  an explicit conservative fallback of eight. The limit is a creation-time
  membership snapshot: scale-out does not resize existing transports, and a
  surviving origin after scale-in can retain its former lower limit until that
  replica is replaced. This avoids closing an active transport during a
  membership mutation. Idle connections become expiry-eligible after 30
  seconds and are reclaimed by later activity on that same origin-local client;
  removing or replacing a replica closes its client immediately. Readiness uses
  a separately owned client capped at the prober's 16 concurrent connections.
  Both paths ignore proxy environment variables because EndpointSlice addresses
  are cluster-internal. Sharing the TLS context removes synchronous per-replica
  CA loading while separate transports avoid httpcore 1.0's fleet-wide flat
  connection scan. vLLM's shared aiohttp/reqwest clients are not a
  counterexample: both index reusable connections by origin, whereas httpcore
  1.0 scans one cross-origin list and performs quadratic idle cleanup. A18
  therefore supersedes A17's shared-httpx-transport choice while retaining its
  pool-scope TLS setup objective and per-operation timeout contract.
- **A19**: request validation preserves the intersection of all eligible
  replica contracts without repeating an identical immutable contract for
  every replica. A backend may explicitly publish a hashable
  `request_validation_key`; `ReplicaPool` validates one representative only
  when both the concrete backend type and key match. Missing, `None`, or
  unhashable keys remain per-replica, and every distinct type/key remains in
  the intersection. `OpenAICompatBackend` uses its frozen resolved capability
  contract as that key because its validator depends on no address, client, or
  model-instance state; subclasses fall back to per-replica validation unless
  they explicitly publish their complete contract. Native in-process and
  process-split backends publish their model path, effective string tokenizer
  source, and `max_model_len`; custom tokenizer objects and subclasses remain
  unkeyed.
  This collapses synchronous prompt tokenization across equivalent pool members
  without deduplicating the per-member async preparation required before
  placement. At 200 equivalent replicas
  this reduces measured validation from 1.785 ms to 0.143 ms median without
  changing placement or rejection semantics. Router and membership records
  still snapshot and enter the bounded queue synchronously, preserving
  event-loop/outbox order and immediate overflow behavior, but JSON encoding
  now executes on the existing lifecycle-owned writer thread. Writer batches
  are bounded by both 128 records and 64 KiB of encoded JSON, preventing a
  burst of large membership rows from retaining the Python GIL across an
  event-loop scheduling quantum while preserving batching for small placement
  rows. Encoding and filesystem failures remain sticky and surface through
  append, flush, or close; no generic executor task or second unordered outbox
  is introduced.
- **A20**: raw EndpointSlice withdrawal evidence is collected by a dedicated
  endpoint-only observer armed concurrently with each pod-delete command. It
  polls on absolute 250 ms deadlines until the old UID set is disjoint, before
  the slower multi-pod readiness recovery begins. Every attempt records its
  epoch, contiguous observer sequence, scheduled time, fetch-start time,
  observation time, error, and raw API payload. Artifact replay requires the
  sequence and absolute schedule to be exact, requires
  `scheduled <= fetch_started <= observed`, and requires the final observer
  row to be the disjoint snapshot named by `old_withdrawn_ns`. The existing
  one-second causal bracket is not relaxed: it conservatively spans the last
  old snapshot's fetch start through the first disjoint snapshot's observation
  time. This separates withdrawal proof from pod readiness polling, whose
  multi-name Kubernetes request can take seconds.
- **A21**: image provenance follows the OCI config identity across Docker
  storage backends instead of assuming Docker's ``.Id`` always names the same
  descriptor as containerd's tag target. The gate records Docker's
  store-specific image ID and canonical config digest, reads the config digest
  referenced by the loaded containerd manifest, and reads the CRI status ID.
  The raw manifest blob is retained and its SHA-256 must equal containerd's
  target descriptor. All three config digests must match; Docker's image ID
  must name either that config or the containerd target. The existing
  source-revision label, raw CRI metadata hash, and per-pod runtime image-ID
  checks remain mandatory.
- **A22**: the F1a driver and gateway isolate the latency-bearing path from
  measurement and control-plane amplification without weakening the frozen
  gate. Raw benchmark JSON encoding and writes use the same bounded,
  lifecycle-owned deferred writer as production audit logs; close still drains
  every admitted row before hashing and replay. Recovery replaces twenty
  per-name Pod GETs with one label LIST and local name filtering, then polls at
  one second after the independent 250 ms EndpointSlice withdrawal observer has
  completed. The observer's absolute schedule, raw payloads, five-second bound,
  and conservative one-second bracket are unchanged. Uvicorn's duplicate
  access log and httpx's successful-request INFO summaries are suppressed while
  the authoritative `kairyu.access` record and warnings/errors remain.
  Finally, the formal gateway requests 500m CPU against a measured 233m maximum,
  giving it more than 2x headroom and five times its former CFS weight while the
  200 mocks together request only 200m. The 10 ms placement bound, 99% open-loop
  pacing requirement, 50 requests/s, retry prohibition, churn batches, and all
  raw evidence remain unchanged.
- **A23**: F1a periodic evidence and recovery are deadline-paced, not
  backlog-paced. A periodic EndpointSlice or resource fetch that overruns its
  interval records both the scheduled deadline and number of skipped intervals,
  then resumes at the first future deadline; it never performs catch-up fetches.
  Every periodic row retains `scheduled_ns`, `skipped_intervals_before`,
  `fetch_started_ns`, and `observed_ns`. Artifact replay requires the first skip
  count to be zero, re-derives each later deadline as exactly one interval plus
  the declared skipped intervals after the preceding deadline, and requires the
  preceding observation to finish before that future deadline and
  `scheduled_ns <= fetch_started_ns <= observed_ns`. A missing, duplicated,
  backdated, or understated skip therefore fails closed instead of fabricating
  a denser evidence cadence than the runner actually achieved. The first
  periodic slot must cover traffic start and the final slot/observation must
  cover traffic completion, so a valid suffix cannot mask a truncated prefix
  or tail. Every row also retains the expected schema/kind and a structurally
  valid EndpointSlice or node/Pod resource payload; timestamps cannot make an
  empty or wrong-kind sample count as evidence.
  Pod evidence no longer polls a 200-object LIST every second. It consists of
  the exact initial 200 identities, one label-selected collection LIST of the
  twenty replaced Pods after each epoch's EndpointSlice recovery, and the exact
  final 200 identities. Every one of those initial, per-epoch, and final Pod
  captures must prove Ready state, the expected Pod name and UID, the frozen
  container image, and the runtime image/config identity pinned by A21.
  Initial and final captures cover the exact expected 200 ordinal names; an
  epoch capture covers exactly its twenty scheduled names, whose UIDs must
  differ from their old UIDs and equal the EndpointSlice mapping.
  The single gateway capture is likewise anchored to its observed expected
  Pod name/UID and pinned gateway container image rather than accepted as an
  arbitrary well-formed Pod identity.

  EndpointSlice recovery accepts only ready, non-terminating target references
  with `kind=Pod`, the exact namespace, and non-empty name and UID. Those
  references must form a strict one-to-one expected-name-to-UID mapping: a
  duplicate pair, one name mapped to multiple UIDs, one UID mapped to multiple
  names, an unexpected/missing ordinal name, or any address/name fallback fails
  closed. Recovery first requires that exact 200-member mapping, new UIDs for
  all twenty scheduled names, unchanged current UIDs for every other name, and
  complete old-UID withdrawal. Only then may the targeted Pod LIST corroborate
  Ready, identity, and image state. Exactly one raw recovery row may claim each
  epoch's `endpoint_recovered_ns`; a second contradictory row at the same
  timestamp or an unknown capture class fails replay. All observations from
  traffic start through the first delete must equal the initial mapping, and
  all observations after the final recovery must equal the final mapping.

  Initial, epoch, and final Pod fetches retain causal brackets. Replay requires
  the initial observation before traffic/churn, and for each epoch requires
  `scheduled <= delete API start <= delete API completion`, old withdrawal
  before exact EndpointSlice recovery, and
  `endpoint_recovered_ns <= new_pods_fetch_started_ns
  <= new_pods_observed_ns == new_ready_ns <= recovered_ns`. The final exact-200
  fetch begins only after traffic and every epoch recovery. Missing or inverted
  raw timestamps, or derived durations that do not match them, fail closed.

  Gateway placement evidence is copied after traffic on a worker thread: the
  bounded writer drains and retries on backpressure rather than dropping rows,
  growing the queue, or failing after a completed measurement. The formal
  backend probe interval is one second, still inside the five-second withdrawal
  bound while reducing the maximum unknown/ejected-replica probe burst by four.
  This supersedes A22's per-second full-fleet Pod LIST. The 250 ms withdrawal
  observer, one-second raw causality bracket, 10 ms placement bound, 99% pacing
  requirement, 50 requests/s, retry prohibition, churn schedule, exact-200
  checks, and raw artifact replay remain unchanged.
- **A24**: F1a control-plane observation no longer creates a fresh `kubectl`
  process for each read. One lifecycle-owned `kubectl proxy` and one persistent
  HTTP client carry every EndpointSlice, Pod LIST, and kubelet resource read;
  only the ten mutating Pod DELETE commands remain subprocesses during traffic.
  HTTP reads have a ten-second fail-closed timeout, and normal, exceptional, and
  cancelled exits terminate and reap the proxy. Raw Kubernetes payloads,
  fetch/observation timestamps, absolute schedules, hashes, and replay checks
  are unchanged. The F1a gateway's own EndpointSlice discovery cadence is
  500 ms rather than 250 ms: even the historical 3.618-second maximum
  withdrawal leaves at least 1.13 seconds of worst-phase margin under the
  unchanged five-second limit, while halving full 180–200-endpoint parsing
  during replacement.

  ReplicaPool retains byte-identical SHA-256 rendezvous hashing but copies the
  hash state after the session prefix and caches encoded replica IDs, the
  insertion-order ordinal map, and the eligible ring at membership mutation
  boundaries. Least-outstanding relies on that ring's stable order instead of
  rebuilding an ordinal map per request. Membership audit capture traverses
  entries once for one internally consistent snapshot. Dynamic add, remove,
  drain, health ejection, and probe transitions refresh the immutable ring
  before publishing their event, so placement and artifact replay observe the
  same state. A 200-replica/10,000-request exact-allocation A/B measured
  1.541 seconds to 0.808 seconds (1.91x) without changing any winner.

  Exact-head Actions run 30365961550 motivated this amendment: all 33,000
  requests returned 2xx and overall placement p99 was 7.998 ms, but all ten
  post-delete windows reached 10.761–26.315 ms. Placement overlapping both
  synchronized EndpointSlice/resource reads had 36.025 ms p99 versus 4.188 ms
  when neither ran, despite gateway CPU peaking at only 279m and memory at
  56 MiB. The request rate, retry prohibition, ten-minute schedule, 10 ms
  placement bound, evidence cadence, and every fail-closed replay condition
  remain frozen.
- **A25**: F1a removes the remaining measurement-side process and kubelet
  statistics amplification and gives the latency-bearing gateway an explicit
  shared-node resource contract. This supersedes A24's remaining Pod DELETE
  subprocess exception: each twenty-Pod batch now uses the existing persistent
  proxy/client for at most eight concurrent Kubernetes REST DELETE requests.
  Every request explicitly selects background propagation while omitting
  `gracePeriodSeconds`, preserving the Pod's declared preStop and termination
  grace. All names are attempted, failures are aggregated deterministically,
  cancellation still propagates, and the existing delete-start,
  delete-completion, withdrawal-observer, and replay boundaries remain
  unchanged.

  Formal resource evidence retains its frozen five-second absolute cadence and
  node/Pod CPU and memory schema, but requests kubelet's
  `/stats/summary?only_cpu_and_memory=true` representation. Network remains a
  nullable field in the existing artifact schema. EndpointSlice parsing now
  validates each address once with `socket.inet_pton` and retains only the
  current minimum candidate per replica, preserving the historical
  family-rank, numeric-address, port, and original-string ordering exactly. On
  the recorded 200-endpoint payload this reduced one parse from 1.150 ms to
  0.308 ms (3.74x) with an identical member map.

  The checked-in one-container formal gateway now has matching requests and
  limits of 2 CPU and 256 MiB, giving it Guaranteed QoS and a two-core CFS
  weight on the shared four-core kind node. Measured gateway demand remained
  below 300m CPU and 56 MiB while replacement activity drove the node to
  3.459 cores, so idle requested CPU remains available to kubelet/containerd
  while contention gives the latency-bearing process the intended weight.
  A separate discovery/reconciliation execution context, as used by vLLM's
  watcher designs, remains a possible later amendment rather than part of A25:
  the exact evidence attributes the dominant tail to shared-node descheduling,
  not membership-transition or HRW work.

  Exact-head Actions run 30370270740 served 33,000/33,000 requests with zero
  429, 5xx, transport, or unsent failures and 7.554 ms overall placement p99,
  but nine of ten churn windows exceeded the unchanged 10 ms bound
  (9.920–15.147 ms). Of 88 churn-window samples at or above 10 ms, 29 of the
  30 early samples overlapped the Pod DELETE subprocess interval; all 58 later
  samples occurred during replacement container/kubelet startup. The quiet
  middle of recovery contributed almost none. Maximum old-UID withdrawal
  remained 2.266 seconds. Request rate, retry prohibition, churn schedule,
  latency and withdrawal bounds, evidence cadence, and all fail-closed replay
  checks remain frozen.
- **A26**: F1a applies its written placement SLO once to the complete frozen
  ten-minute measurement: nearest-rank placement p99 across all measurement
  requests must be strictly below 10 ms. Every epoch and every ten-second
  post-delete window remains mandatory, hashed, published, and independently
  replayed diagnostic evidence, but its local p99 is not an additional SLO.
  This supersedes only A16's later all-window rejection clause. Zero 429, 5xx,
  transport, and unsent requests; 99% open-loop pacing; exact lifecycle and
  identity joins; five-second withdrawal; raw sidecars; provenance; replica
  count; churn schedule; strict inequality; and every other fail-closed
  condition remain unchanged.

  The all-window clause was statistically unsound for the declared SLO. A
  post-delete window contains 500 samples, so nearest-rank p99 passes when at
  most five samples reach 10 ms. Even when the true exceedance probability is
  exactly 1%, one such window passes with probability only 61.60%; requiring
  all ten to pass has probability 0.79%. That repeated small-sample rule
  therefore rejected ordinary run-to-run and host scheduling variation while
  claiming to test one measurement-wide percentile.

  Actions run 30374404150 supplies the immutable raw evidence behind this
  correction. It served 33,000/33,000 requests with zero errors; the 30,000
  measurement samples have 5.221 ms overall p99 and 128 samples (0.427%) at or
  above 10 ms. Four diagnostic post-delete windows reached 10.242–21.040 ms.
  Of 57 post-delete tail samples, 47 occurred six to ten seconds after deletion
  during replacement container startup; only one was within 100 ms of a
  membership event. Node CPU peaked at 3.434/4 cores while the gateway used
  0.251 core, attributing the local bursts to shared-node lifecycle scheduling
  rather than placement, DELETE, or reconciliation logic. Maximum old-UID
  withdrawal remained 1.117 seconds. Independent A26 replay passes every
  amended gate check and every raw sidecar integrity check. The legacy manifest
  wrapper differs only in the old published check map and old `passed` value
  that this amendment intentionally replaces.
- **A27**: F1b is a drain-first, partitioned `RollingUpdate` of one exact
  100-replica StatefulSet, not a replay of F1a's `OnDelete` Pod churn. The
  StatefulSet starts with `rollingUpdate.partition=100`, so staging the new
  template and the driver's single `kubectl rollout restart` cannot terminate
  a Pod. The provenance-pinned driver then walks ordinals 99 through 0. For
  each ordinal it first calls that replica's `/admin/drain` through the
  authenticated Kubernetes Pod proxy, waits until raw EndpointSlice evidence
  has withdrawn the old ready,
  non-terminating UID and the gateway membership stream marks the same
  generation ineligible, and waits for its outstanding work to reach zero.
  Only then may it lower the partition to that ordinal. It waits for the same
  Pod name to acquire exactly one new UID at the update revision, become Ready,
  and re-enter gateway eligibility before proceeding to the next ordinal.
  Direct Pod DELETE, pre-draining the whole fleet, advancing a partition after
  a timeout, or updating more than the one released ordinal fails the gate.

  The binding boundary for "no new work reaches a terminating replica" is the
  later raw observation of gateway ineligibility and EndpointSlice withdrawal,
  not receipt of the local drain HTTP response. Distributed readiness
  propagation between those observations remains published diagnostic
  evidence. The partition cannot advance before the binding boundary and
  outstanding count reaches zero, so requests placed before the boundary may
  complete normally while no later placement may select that old UID.

  Formal traffic is absolute-deadline open loop with retry count zero. Every
  scheduled arrival has exactly one attempt; an unsent arrival, transport
  error or timeout, 429, 5xx, other non-2xx response, missing request-to-
  placement join, or mismatched response Pod identity is a failure. Initial
  and final evidence must each contain exactly 100 Ready ordinals. The 100 old
  UIDs and 100 new UIDs must be disjoint, each ordinal must have exactly one
  old-to-new transition, all final Pods must carry the update revision, and an
  extra replacement or container restart fails closed.

  Raw `requests`, gateway placement/membership, rollout/partition events,
  drain/readiness transitions, EndpointSlices, and Pods are retained with safe
  relative paths, row counts, and SHA-256 digests. Post-run Kubernetes events
  and resource descriptions remain auxiliary diagnostics in the same CI
  artifact rather than F1b acceptance inputs. The independent verifier replays
  the traffic schedule, drain and partition ordering, UID/revision transitions,
  placement eligibility, availability, image/source provenance, and every
  published result rather than trusting summary booleans. A pull-request smoke
  uses the same state machine at reduced scale, but only one clean, exact-head
  100-replica formal artifact can satisfy F1b.

  "No operator action" means that, after the CI job starts, no human decision,
  repair, or per-Pod command is admitted. The checked-in driver performs the
  predetermined drain, wait, and partition writes; it is test orchestration,
  not a cluster-installed rollout Operator. Retained F1a formal evidence may
  establish shared-runner capacity, image provenance, NodePort traffic,
  discovery, and request/placement/membership joins. It cannot establish the
  partitioned rollout, drain-first ordering, exact-100 lifecycle, or unattended
  completion because F1a deliberately performs `OnDelete` batch churn. F1b
  therefore requires its own new formal run; F1a need not be rerun when its
  frozen inputs remain unchanged.
- **A28**: F1b's formal whole-rollout deadline is a stuck-run safety cap, not
  an unstated completion-latency SLO. It is 1,500 seconds while the binding
  per-ordinal withdrawal and replacement bounds remain five and 60 seconds.
  Exact-head formal run `30385162649` completed ordinals 99 through 12 with
  every recorded drain, withdrawal, readiness, endpoint, and replacement
  condition true before the former 900-second cap cancelled ordinal 11.
  Those 88 completed cycles took 890.208 seconds: mean 10.116, p95 11.165, and
  maximum 12.134 seconds including inter-step orchestration. The 1,500-second
  cap rounds up a 100-step
  extrapolation of the observed maximum plus 20% runner-jitter margin
  (`12.133584415 * 100 * 1.20 = 1,456.03`) and still leaves ten minutes for setup,
  cooldown, replay, and artifact upload inside the 35-minute CI job. The
  failed run's 46,500 completed request records were each sent exactly once,
  returned valid HTTP 200 responses, and had no transport error. One additional
  placement was interrupted by timeout cleanup before an outcome was recorded;
  the run is diagnostic evidence only because it did not complete all 100
  steps.
- **A29**: F1c separates gateway affinity from durable batch ownership.
  The edge load balancer uses deterministic SHA-256 rendezvous hashing over the
  explicit `X-Session-ID` header and three immutable gateway identities.
  Kubernetes `ClientIP` affinity and a JSON-body `user` field are not accepted
  as proof of that L7 contract. Each converged gateway retains ReplicaPool's
  existing SHA-256 rendezvous hash over the same eligible replica UID set, so
  the selected model replica remains independent of which gateway receives the
  request.

  The shared BatchStore backend is PostgreSQL, selected explicitly by
  `batch.store: postgres`; the filesystem backend and its restart-fails-orphans
  behavior remain the backward-compatible single-gateway default. The DSN is
  read only from the configured environment-variable name. Files are stored as
  ordered database chunks rather than one 512 MiB value or an RWX filesystem.
  SQLite or the filesystem store on NFS is not a supported HA substitute
  because correctness would depend on network-filesystem locking and would
  retain unfenced read-modify-write races.

  A shared worker discovers jobs from the database; its process-local submit
  signal is only a wake-up hint. Claiming a validating job, or an in-progress
  job whose lease expired, is one `FOR UPDATE SKIP LOCKED` transaction using
  the database clock. Every takeover increments a fencing token. Lease renewal,
  cancellation, and terminal publication require the current immutable worker
  identity and token; cancellation invalidates the token, and a stale claimant
  rolls back any output publication. Startup never calls the filesystem
  `recover_orphans` policy for this backend. The resulting guarantee is
  at-least-once inference execution after a crash and exactly one fenced
  terminal publication, not exactly-once inference or usage accounting.

  One exact-head F1c kind drill is binding. It records three Ready gateway Pod
  UIDs, a common store identity and converged replica membership; sends each
  affinity request once with no retries; independently recomputes both
  rendezvous choices; and joins the raw LB decision, gateway placement and
  response replica UID. Files and batches are created and read through
  different gateways. The driver kills the immutable active claim-owner Pod,
  then requires a different gateway UID and larger fencing token to reclaim
  the job after lease expiry, one terminal commit, and byte-identical output
  through all three gateways including the restarted one. Pod absence is an
  eventually consistent Kubernetes polling observation, not the actual stop
  timestamp in the PostgreSQL clock domain: replay therefore requires the kill
  request under the old lease, no later old-fence renewal, exact prior-lease
  expiry before the higher-fence reclaim, and eventual old-UID absence, but
  does not order the absence-observation timestamp before reclaim. Likewise,
  Kubernetes may omit a source pin from the display `status.image`; for a
  digest-pinned imported image, replay proves the complete identity chain:
  the source registry pin is present in Docker `RepoDigests`, that image's
  Docker config ID equals the kind CRI config ID, and the Pod runtime digest is
  either the source pin or one of the CRI-reported import identities. A digest
  outside that chain fails. Raw traffic, LB, placement, membership, Pod, batch
  HTTP and claim-audit sidecars are hashed and independently replayed. F1a run
  `30374404150` and F1b run `30387260062` retain their capacity, discovery,
  provenance and raw-evidence precedents; neither is rerun. F1c adds no latency
  percentile SLO: elapsed times are diagnostic and only a generous stuck-run
  cap is binding. Exact-head source run `30399229234` at `be40b97` passed all
  26 replay checks; its complete artifact is retained under
  `bench/results/f1c-three-gateway/`.
- **A30**: F2a compares prefix-aware routing with session hashing without
  changing the request contract. Both arms receive the same ordered prompts,
  the same per-request unique session IDs, 500 immutable eligible replica IDs,
  and independently cloned initial simulated cache placement. The shared trace
  is deliberately cross-session: it measures reuse that session affinity cannot
  express, while keeping the complete request stream identical between arms.
  The treatment differs only by enabling `PrefixIndex`. A warm maximum-score
  candidate with a non-negative score is selected before ordinary session
  placement; a zero warm/cold tie prefers reuse and session HRW breaks equal
  warm scores. If no candidate has usable overlap, or the best warm score is
  negative, the pre-F2 session-HRW and queue-depth behavior is retained without
  scanning every cold replica. A bounded first-chunk reverse map limits warm
  scoring to eligible candidates that can have non-zero overlap. This
  supersedes D6's unimplemented power-of-two wording and the implementation's
  former session-first bypass.

  Cache truth belongs to the selected mock backend, not to `prefix_match` or
  `PrefixIndex.overlap`. The backend reports whether the family was resident
  before dispatch and updates its cache only after successful generation.
  `ReplicaPool` likewise advertises an approximate prefix only after successful
  unary completion or immediately before a stream's first backend result;
  upstream error, client error, and cancellation before that result cannot
  create a false warm entry. Because cumulative chunk
  keys are unusable after an ancestor is evicted, one over-cap prompt retains
  the root-side bounded prefix rather than unreachable tail keys. Selection and
  successful publication consume one request-local lazy key chain. A cold
  success publishes only its already-computed XXH3-64 root through the native
  cold fast path; a successful warm route extends that chain once and promotes
  it to full usable depth.

  F2a's distinct claim is selector quality and cost at 500 logical
  `ReplicaPool` entries. Retained F1a run `30374404150` already proves the kind
  deployment, NodePort ingress-to-selection clock, discovery, membership,
  provenance, and hosted-runner capacity; it is not rerun and contributes none
  of F2a's hit-rate, goodput, or 500-entry measurements. The F2a CPU bench
  drives the public `ReplicaPool.generate` path and its production placement
  logger against 500 concrete mock backends. It does not claim a 500-Pod
  Kubernetes deployment.

  Shared-prefix hit rate is cached prompt-work chunks divided by total prompt
  chunks, derived from raw backend observations, and must beat the non-zero
  session-HRW baseline by at least 2×. Uniform traffic uses 21 matched rounds
  with alternating A/B order. Before them, one full non-binding uniform
  calibration pair exercises both fresh policy pools; its raw selections,
  cache outcomes, summaries, order, and `binding=false` marker are replayed but
  enter no metric. This keeps CPython's first arena allocation out of a
  steady-state goodput claim without discarding any binding sample after seeing
  its value. Uniform calibration, run-in, and binding requests deliberately
  carry blank root hints, so the no-loss claim binds the production session-only
  opt-out/HRW path rather than general cold prefix-tracking overhead or a
  trace-only family oracle. Immediately before every
  calibration and binding arm's clock, that same pool and policy also execute a
  declared 512-request uniform run-in over
  disjoint prompts. Its deterministic trace digest, completed count, and
  positive time interval remain in the arm summary and are independently bound
  before the measured interval, but none of its requests enter a gate. Per-arm
  run-in prevents pool construction, cold code, CPU-frequency ramp, or scheduler
  migration from systematically favoring whichever arm happens to run second.
  SLO goodput is the number of successful requests whose production placement
  latency is below 10 ms divided by the summed end-to-end
  `ReplicaPool.generate` dispatch intervals for every offered request in that
  arm; a request missing the SLO remains in the denominator. Round wall time is
  retained only for ordering and audit. Each paired round is the independent
  experimental unit; the 512 request timings inside it may be arbitrarily
  correlated. No Gaussian or symmetry assumption is made across those paired
  round ratios. The binding location statistic is the exact distribution-free
  one-sided lower confidence bound for the population median:
  with 21 paired ratios, the largest order-statistic rank with at least 95%
  coverage is the seventh (`P[Binomial(21, 0.5) >= 7] = 0.960823`), which must
  be at least 0.99. This is mathematically identical to requiring at least
  15/21 individual ratios to be at least 0.99. The geometric mean of all 21
  ratios must independently be at least 0.99, so the resistant median cannot
  conceal a small number of large losses. No round is trimmed, winsorized, or
  excluded. The former Student-t log-mean lower bound remains in the manifest
  as diagnostic evidence only. The 1% non-inferiority margin, exact median
  inference, full-sample magnitude guard, and sign count jointly distinguish
  systematic regression from time-local runner interference. Nearest-rank
  treatment placement p99 is computed over each complete shared and uniform raw
  population, and the worse value must be strictly below 10 ms.

  Raw cache seeds, prompt/session hashes, selections, cache hits, receipt and
  selection timestamps, round order, and wall times are JSONL. Prompt bodies
  are excluded. An adjacent manifest binds row count and SHA-256, the clean
  source commit, workflow and relevant source hashes at both ends of the
  measurement, non-empty GitHub run identity tied to the expected head, runner
  environment, all derived metrics, and every gate check. The independent
  verifier reconstructs both policies, cache evolution, SLO-goodput numerator,
  round intervals, and statistics rather than trusting the manifest. One clean
  exact-source formal run is binding. A later evidence-only commit may retain
  those immutable bytes without rerunning when the source commit is an ancestor
  and every frozen gate-input hash is unchanged.

  Exact-source run `30411111758` at `c067cb8` closes F2a. Shared cached
  prompt-work improved 37.9259x over the non-zero HRW baseline. Across the 21
  blank-root paired rounds, the goodput-ratio median was 1.002142, the exact
  96.0823%-coverage median lower bound was 0.999512, the full-sample geometric
  mean was 1.008610, and all 21 ratios met 0.99. Worst-trace placement p99 was
  0.145979 ms. The independently replayed 24,709-row artifact is retained at
  `bench/results/f2a-prefix-routing-500-2026-07-28/`; its evidence-only
  retention commit does not repeat the measurement.
- **A31**: F2b makes exact KV-event placement a recoverable, fail-closed
  protocol instead of treating arrival recency as cache truth. Each source
  owns a cache-lifetime epoch and monotonically increasing sequence. The
  publisher mirrors the authoritative hash set and appends a serialized delta
  to a bounded replay journal before attempting non-blocking PUB delivery.
  Heartbeats carry the current high-water mark. A subscriber that observes a
  gap, an epoch change, or a stale lease may stage deltas but cannot use them
  for production exact routing until a matching high-water confirmation,
  complete replay, or authoritative snapshot proves completeness. Retired
  epochs are retained for the process lifetime, and inactive replica
  tombstones reject delayed frames across remove/re-add; replacing a cache
  rotates epoch, sequence, journal, and mirror on the same publisher object
  and socket thread. Each cache owns a
  generation-bound publisher callback, so an old cache racing the rotation
  cannot stamp its event with the replacement epoch. Two-frame legacy delivery
  remains compatible for direct legacy callers but is never admitted by the
  production atomic exact-routing surface.

  Exact engine block hashes and approximate gateway text chunks remain
  different score spaces. `KvRoutingIndex` therefore chooses one mode for the
  entire request: it uses one atomic `KvEventIndex.route_overlaps` observation
  for every eligible replica, or it discards all exact scores and follows the
  existing approximate `PrefixIndex` path.
  Provider, index, lifecycle, malformed-vector, and freshness failures degrade
  to approximate placement rather than failing generation. A failed exact
  membership mutation quarantines that replica until a complete successful
  forget/register reset, so a partially mutated old view cannot re-enter
  exact scoring. While a replica is inactive, unknown epochs are rejected
  without displacing the active epoch tombstone. Successful exact choices use
  the distinct `kv_event_match` decision reason; approximate stream roots
  publish at the same first-result boundary, while full-key promotion still
  requires successful completion.

  **Issue #348 amendment:** `KvEventIndex` captures feed identity, epoch,
  sequence, and hash-set references under its lock, performs the
  prompt-length membership scan outside the lock, then rejects the vector
  unless a second lock epoch proves every eligible feed unchanged and live.
  `KvRoutingIndex` likewise releases its lifecycle lock around the exact call
  and rejects a result if a register/forget revision changed. Approximate
  root candidates cannot prune this pass: the F2b oracle deliberately proves
  that approximate text-chunk and exact engine-block winners may differ.
  Thus event ingestion and membership churn are not serialized behind exact
  scoring without mixing the two score spaces or adding another index.

  The F2b formal profile reuses F1a run `30374404150` for the exact seed-175
  200-replica, ten-by-twenty churn identity schedule and reuses F2a run
  `30411111758` only for its production `ReplicaPool`/prefix-routing precedent.
  Neither measurement is repeated. F2b compresses only wall pacing to one
  second per churn epoch and exercises 200 logical feeds: 199 sequenced
  in-process feeds plus one representative physical ZMQ PUB/SUB/replay feed.
  Because one unavailable eligible feed makes the request globally
  approximate, killing that representative socket binds the same 200-entry
  routing decision without claiming 200 physical transports.

  The freshness limit is strictly below 500 ms and the route-time lease is
  250 ms, leaving explicit scheduling headroom. Gates use actual emission,
  receipt, apply, pause, resume, replay-completion, and route-selection times;
  absolute open-loop ticks and missing routes are replayed. Every pause/resume
  action, offered-route start, and selected-route gap must itself remain
  strictly below 500 ms, so an OS or event-loop stall cannot be hidden by a
  later catch-up iteration; observed samples are never excluded or relabeled.
  Every exact route must use authoritative logical and wire truth younger than
  the lease. After actual expiry every route must match the independently
  reconstructed approximate oracle, and after restore a complete replay must
  converge to the authoritative 200-replica digest on the same process and
  object identities in under 500 ms. Raw event, membership, kill schedule,
  replay, route, state, source, and environment rows are independently
  rehashed and replayed. One clean exact-source formal run is binding; an
  evidence-only descendant may retain it without remeasurement only when every
  frozen gate-input hash is unchanged, the recorded run resolves through the
  GitHub Actions API to the matching completed successful workflow, and the
  retained raw and manifest bytes equal the original Actions artifact.

  Exact-source run `30417507859` at `f383806` closes F2b. Its 500 routes split
  into 175 fresh exact, 140 stale approximate, and 185 restored exact
  decisions. Maximum exact truth age was 232.314498 ms; the first stale
  approximate route followed pause by 251.339950 ms; complete replay restored
  the same process and object identities 50.740933 ms after resume. Maximum
  offered-route lateness and selected-route gap were 3.608193 and 21.536138
  ms. The independently replayed 2,196-row artifact is retained byte-identically
  at `bench/results/f2b-kv-event-retained/`; its evidence-only retention does
  not repeat the binding run.
- **A32 (closed 2026-07-29)**: F2c measures the real D6 placement path, not a
  gateway mock. The candidate is a production `ReplicaPool` with
  `PrefixIndex`; the control is
  the same pool with `prefix_index=None`. Both use `OpenAICompatBackend` to
  stream from real Qwen3-32B engines and `JsonlRouterLog` to retain the
  production selection reason and replica. Four independent TP2 engines occupy
  all eight GPUs. Endpoints A0/A1 form one two-replica cohort and B0/B1 form a
  second cache-disjoint cohort. Candidate and control run concurrently, then
  exchange cohorts every round for eight rounds so GPU, NUMA, thermal, and
  time-order effects cannot remain attached to one policy.

  TP2 is the deliberate memory/capacity point for this proof. On the measured
  95.59 GiB Blackwell GPUs, each TP2 rank carries approximately 31.96 GiB of
  Qwen3-32B weights and 16 GiB of KV pages at the frozen 8,192-page capacity,
  leaving approximately 47.6 GiB for graphs, workspaces, and runtime
  allocations. TP1 would carry the complete approximately 65.5 GB checkpoint
  on one device and discard most of that validated margin. TP2 also fills all
  eight GPUs while retaining the two independently cached replicas that each
  policy needs; a single TP8 engine would not test routing.

  Each run carries a recorded namespace, and each round creates 16 unique
  2,048-word RAG families whose namespaced identities differ inside the first
  256 characters. Smoke and formal namespaces must differ, and a namespace is
  never reused against persistent caches. A non-binding cold seed uses a session whose
  production HRW target alternates between logical replicas; the measured
  session hashes to the opposite replica. Thus successful seeding makes the
  candidate's `prefix_match` return to the warm engine, while the control's
  `session_affinity` selects the deliberately cold opposite engine. Every
  candidate/control pair receives byte-identical prompt content, session and
  prefix hints, greedy sampling, and exactly eight output tokens
  (`min_tokens=max_tokens=8`, `ignore_eos=true`). Each family predeclares one
  canonical assistant continuation as trace input; the trace descriptor binds
  both its digest and the resulting turn-2 prompt digest. Turn 1 must complete
  successfully on both policies before phase 2 begins, but neither observed
  output becomes input to a later request. Both arms instead receive the same
  frozen canonical transcript in turn 2. Missing, failed, duplicate,
  non-terminal, prompt/transcript-mismatched, invalid-output-digest, or
  inexact-usage rows fail the proof. Cross-arm output digest matches are
  retained as a count, total, and rate, but remain diagnostic only.

  The first formal execution stopped at round 1 family 0 on the former exact
  output-equality assertion. A targeted fixed-endpoint reproduction left both
  endpoints fully warm (2,544/2,546 cached prompt tokens): each endpoint
  repeated its own continuation twice, but B0 and A1 differed. A separate
  longer family matched between the warm and cold arms. The result is
  consistent with a BF16/TP near-tie under different cross-endpoint and
  cache-population execution shapes; A1/B0 prefix KV may have been formed
  through different chunk/prefill histories, so it does not establish semantic
  cache corruption or isolate a physical GPU-pair effect. This is the
  numerical behavior already covered by G2's rule that free-running greedy
  sequence equality is not a correctness gate: one moved near-tie changes the
  later autoregressive prefix without proving a broken route or cache. The
  frozen transcript preserves identical post-turn-1 work and removes
  post-treatment output dependence without weakening routing, engine-usage,
  or performance evidence.

  TTFT uses nearest-rank p95 with no interpolation, trimming, or exclusion.
  Candidate/control p95 must be at most 0.70 in the pooled population, in the
  geometric mean of all eight round ratios, and at the seventh ordered round
  ratio. The last condition has 96.484375% one-sided binomial coverage for the
  median and is equivalent to at least seven of eight rounds meeting the 30%
  reduction. Goodput counts every successful request at or below the frozen
  60-second TTFT SLO over the arm's first-receipt-to-last-terminal interval;
  every planned request remains in scope. Its candidate/control ratio must be
  at least 0.99 pooled, at the second ordered round ratio, and in the geometric
  mean, likewise requiring at least seven non-regressing rounds. Cache truth is
  the token-weighted engine usage
  `sum(cached_tokens) / sum(prompt_tokens)`, not a router-derived hit label: the
  candidate must improve it strictly when pooled and be no lower in any round.

  Paired receipt skew and absolute-schedule lateness are retained with
  monotonic timestamps and recomputed as diagnostics. They have no arbitrary
  millisecond fail cutoff; simultaneous disjoint arms, cohort crossover, the
  seven-of-eight order statistics, and full-sample geometric means are the
  controls for ordinary OS jitter. The short changed-scope smoke binds only
  evidence integrity and production-path semantics; it retains but does not
  gate on the formal performance statistics. The formal profile alone binds
  the thresholds above. The artifact still binds every planned
  request, route decision, selected replica, topology and GPU identity, exact
  configuration, model revision and weight digests, clean expected source
  commit, relevant source/config hashes, prompt identities, valid per-request
  output digests, diagnostic output-match rate, and exact engine usage. Its
  offline verifier rehashes the raw JSONL and reconstructs the trace, routing
  contract, statistics, and manifest verdict without trusting derived fields.

  The exact-source formal run retained at
  `bench/results/f2c-kv-aware-ttft-qwen3-32b-2026-07-29/` closes A32 and F2c.
  All offline-replayed checks passed over 512 binding requests with zero
  failures. Control-to-candidate pooled TTFT p95 fell from 527.957623 ms to
  134.357747 ms: the pooled ratio was 0.2544858548, the seventh ordered ratio
  was 0.2550841404, and the eight-round geometric mean was 0.2530080045.
  Engine-token cache rate rose from 0.4994645560 to 0.9843917326 with every
  round noninferior. Candidate/control SLO-goodput ratios were 0.9999979014
  pooled, 0.9998437390 at the second ordered round, and 0.9999978783 by
  geometric mean. Output matches were 239/256 (0.93359375), diagnostic only;
  maximum paired receipt skew and schedule lateness were likewise diagnostic
  at 5.182959 ms and 7.470463 ms.

  The artifact binds clean source
  `80b039b5d429c656871a480c2740740951b29b97`, runtime image
  `kairyu-f2c@sha256:d2c01580964f461a3d3d2a02ced5303e69c681696d4a38179162084e1624121f`,
  raw SHA-256
  `4cfcdeba2b7473aa6c2b28409dbf21de23d775d9b08e971beed6bdab875abe64`,
  and trace SHA-256
  `51d188671432bf791c02d66d91e6a7d785eb2bd01f64e29a41a62e74f9957dad`.

  This direct L2 fixture is the narrow F2c proof: normal HTTP session-only blank
  hints intentionally bypass cross-session `PrefixIndex`, so using that
  gateway path would compare HRW with itself. It does not add
  DeploymentSpec/builder wiring for D7 exact KV events. That product path also
  requires a tokenizer-compatible block-hash provider plus subscriber and
  publisher lifecycle ownership; adding it here would mix a separate
  deployment responsibility into the D6 routing performance experiment.

- **A33 (closed, 2026-07-29)**: F2d replaces chosen-action agreement with
  complete policy replay. The production decision source is
  `JsonlRouterLog`: each `kind=replica` row must join exactly once by
  `request_id` to a `placement_outcome` row containing the request TTFT, and
  `learning/dataset.py` must reject missing, duplicate, or ambiguous joins.
  Those joined rows retain the observed production evidence, but are not
  treated as counterfactual rewards for actions the logging policy did not
  take.

  The candidate family is the declared grid over `λ = β / α` with `α=1`;
  positive common rescaling cannot change a placement, so searching redundant
  `(α, β)` pairs would claim distinctions the policy cannot identify. The
  fixed hand-tuned baseline is `λ=0.25`. Request families are partitioned
  before execution into disjoint training and held-out sets; every request
  and every state transition from one family remains in exactly one split.
  Every declared candidate runs every complete training episode from the same
  frozen initial cache/background-load state. The candidate with the
  lowest mean training TTFT is frozen before any held-out result is inspected.

  Held-out execution runs only the frozen candidate and `λ=0.25`. Each pair
  receives the same complete trace and frozen initial state, with no cache,
  load, or queue state carried from another candidate or prior episode.
  Replay advances deterministic virtual time, so arm execution order is
  neither a gate nor a retained diagnostic. The binding performance result is
  only strict improvement in held-out mean TTFT by the frozen candidate.
  There is no additional 10% improvement threshold and no p95 acceptance
  gate; p95 and action-selection differences are retained as diagnostics.

  The proof also binds complete planned-request accounting, zero failed or
  non-terminal requests, balanced family/work allocation, exact one-to-one
  decision/outcome joins, split isolation, and trace, configuration, source,
  and result integrity. Raw JSONL and a hash-bound manifest are retained. An
  independent offline verifier reconstructs both splits, replays every state
  transition and policy decision, recomputes all joins and means from raw
  rows, and reaches the manifest verdict without trusting derived fields.

  The exact-source formal artifact at
  `86dde278d0f2a093bde64f5d1d9cba9aca9e1221` passed this contract. Seven
  normalized policies replayed 768 requests each over 48 training families;
  the tuner froze `λ=1.0`. On 16 disjoint held-out families, mean TTFT for the
  same 256 requests was 4.43359375 virtual ticks under the frozen policy versus
  8.5 under `λ=0.25`. All 5,888 production placement rows joined exactly once
  to successful outcomes and independently replayed. The retained artifact is
  `bench/results/f2d-prefix-weight-replay-2026-07-29/`, with manifest SHA-256
  `3205721922fd8c013ae6336aaa4ffcb0a1938a40059e70acb500b5acba86ac3c`,
  raw SHA-256
  `1ccc5ab012e5ee6677f96709ec60cc15ea5db32cefb72360941238ca505c75eb`,
  and production-router SHA-256
  `3296fdd000aede574ea5c3a152ff1ef0f54e204545bfb1f9aa61f7b47c83546f`.

- **A34 (closed, 2026-07-31)**: F1d now binds one distributed W3C Trace
  Context tree rather than isolated local spans. The gateway SERVER request
  span is held through the final ASGI response body; route, actual pool
  selection, replica CLIENT call, remote replica SERVER request, and every
  Conductor stage remain descendants of that request. Only `traceparent` and
  `tracestate` cross the process boundary; baggage is neither extracted nor
  injected.

  Tracing remains optional and disabled by default. Kairyu never replaces the
  process-global provider: an application may inject a private provider, use
  its existing global provider, or opt into a private compact console exporter
  for the deployed smoke. Automatic OpenTelemetry exception events and status
  descriptions are disabled because they can contain prompt or output text.
  Failed and cancelled spans retain only the exception type, cancellation
  marker, and description-free ERROR status.

  The deterministic in-memory fixture proves the complete span tree,
  attributes, success, error, cancellation, streaming lifetime, and privacy
  contract. The mandatory separate-container Compose smoke additionally joins
  gateway and replica records by response request ID plus trace, span, and
  parent IDs, checks distinct service identities and the final response, and
  rejects prompt/output canaries in every exported span. Log order, timing,
  and randomly generated IDs are not acceptance inputs.

- **A35 (2026-08-03)**: F1a separates gateway membership convergence from
  benchmark EndpointSlice evidence cadence. The gateway polls EndpointSlices
  every 500 ms, so both smoke and formal replay allow two configured discovery
  intervals: one for worst-case polling phase and one for the bounded API read,
  reconciliation, and scheduling. Every old UID must leave gateway eligibility
  within one second of the independent observer's first disjoint EndpointSlice
  snapshot.
  The deadline is also capped at five seconds from the Pod DELETE start, so
  evidence-observer delay cannot extend the graceful-withdrawal contract.

  This leaves the binding formal one-second deadline unchanged. It corrects
  only smoke, whose 500 ms evidence capture cadence had accidentally become a
  one-poll gateway deadline after A24 changed discovery from 250 to 500 ms.
  A one-poll deadline provides no time for the Kubernetes read, reconciliation,
  or event-loop scheduling and therefore depends on polling phase rather than
  product behavior. Retry-free traffic, zero failure, placement-to-membership
  joins, no old-UID placement after the deadline, no eligibility reappearance,
  the five-second absolute bound, and all fail-closed replay checks remain
  mandatory.

- **A36 (2026-09-07)**: durable online work receives its own AsyncRequest state
  and `RequestStoreProtocol`; it does not reuse OpenAI Batch jobs or their file
  lifecycle. Caller-visible, top-level-frozen snapshots are deep-copy isolated
  from store state and move through `queued`,
  `claimed`, `running`, and exactly one terminal state (`succeeded`, `failed`,
  `cancelled`, or `expired`). Worker ownership is a separate, expiring claim.
  Every claim or takeover increments a fencing token, and terminal publication
  requires the current worker identity, token, and unexpired lease.

  Request bodies and results are restricted to JSON values so every backend can
  preserve the same contract. Idempotency keys are scoped by tenant. An exact replay returns the existing
  request, while the same key with different normalized intent fails as a
  conflict. Claim selection is deterministic: ascending signed-int64 priority
  (vLLM-compatible smaller values run first), then creation
  time and request ID. Deadlines and cancellation invalidate current claims.
  The process-local in-memory backend is only the executable reference contract;
  it is not an HA or durable deployment option. The production backend remains
  PostgreSQL with database-clock leases and transactional terminal publication;
  Redis may later supply wake-up hints but cannot own request truth.

- **A37 (2026-09-07)**: `PostgresRequestStore` is the production persistence
  implementation of A36. Its normalized PostgreSQL schema independently checks
  lifecycle state, claim shape, terminal payload shape, JSON object storage,
  priority, attempts, and tenant-scoped idempotency. Schema initialization is
  serialized by a transaction advisory lock and guarded by a per-store version.

  Submission uses a partial unique index over `(store, owner, idempotency_key)`;
  concurrent exact submissions converge on one row and conflicting normalized
  intent fails. Claim selection uses `FOR UPDATE SKIP LOCKED`, descending
  priority, deterministic age/ID ordering, and the database clock. Claim,
  renewal, running transition, and terminal publication check worker identity,
  fencing token, lease, and deadline in the mutating SQL statement. Takeover
  increments both attempt and fencing token. Cancellation and deadline expiry
  clear ownership and invalidate the old fence. Terminal state and its result
  or structured error commit in the same transaction as the audit event.
  Status lists use one database statement timestamp to project deadline state;
  ordinary reads never skip locked rows, while bounded cleanup persists expired
  rows opportunistically without delaying the caller.

  Separate general and lease connections prevent unrelated reads/submissions
  from starving worker heartbeats inside one process. The retained integration
  suite runs against the CI-pinned PostgreSQL image and verifies cross-instance
  races; Redis remains outside this consistency boundary.

- **A38 (2026-09-08)**: the durable request transport is an additive,
  non-streaming Chat Completions path. `POST /v1/async/chat/completions` stores
  the authenticated tenant and validated JSON intent, returns 202 with stable
  status/result/cancel URLs, and accepts `Idempotency-Key`,
  and `X-Kairyu-Deadline-At`. Queue priority comes only from authenticated
  tenant policy; untrusted callers cannot raise global queue position.
  Tenant-scoped `GET /v1/requests`, status, result, and cancel routes return 404
  across tenant boundaries. Pending result reads return 202; only a fenced
  successful terminal row returns the stored Chat Completion result. Direct
  streaming remains on `/v1/chat/completions`; `stream=true` is rejected before
  queue insertion.

  Receipt, list, and status responses are bounded projections and never return
  persisted request or result bodies. Async control routes establish identity
  without consuming inference RPM; submissions use an independent per-tenant
  RPM bucket plus a hard per-tenant stored-record allocation to bound queue and
  retained-state growth, and the worker owns the single inference
  request-quota admission. Lease takeover remains at-least-once for usage metering even
  though fenced publication is single-winner; request-ID-idempotent accounting
  is a later extension.

  Transient tenant RPM, token-balance, and in-flight rejection is queue pressure,
  not a terminal request failure: a fenced defer returns the request to queued
  state and persists a short owner-wide cooldown so other tenants remain
  claimable. Bounds larger than the tenant token burst are terminal because no
  refill can make them admissible.

  `AsyncRequestWorker` uses a fixed consumer pool over the shared store. Each
  direct served model traverses the same chat validation, backend preparation,
  tool-choice checks, tenant admission/reservation, metering, and usage boundary
  as other background chat work, with the tenant's trusted batch scheduling
  priority. A dedicated heartbeat renews the database lease during validation,
  SLO-pressure waiting, and inference. Local cancellation aborts the backend
  task immediately; remote cancellation is observed through renewal failure,
  and all publication remains fenced. Shutdown stops new dispatch, aborts local
  active work, leaves unfinished claims for lease-based takeover, then closes
  the store. Orchestrated/AUTO and Responses async inputs remain later API
  extensions; v1 accepts direct publicly served Chat models only.

  Deployment enables the surface only with an `async_requests` section and a
  preflighted PostgreSQL DSN. It owns worker concurrency, polling, lease length,
  store identity, and an independent persisted-body byte cap (8 MiB by default).
  Request IDs are collapsed in metric route labels; high-cardinality IDs remain
  in request state and logs rather than Prometheus labels.
