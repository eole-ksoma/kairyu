# Node model cache pre-stage v1

Status: WP4.7 implemented as fenced controller commands, placement-level
claim/CAS, verified node execution, owner-scoped cache pins, status projection,
and a shared durable PostgreSQL command store. The authenticated node HTTP API
and executable runtime are implemented. D3.1 additionally binds every proposed
Runner start to a decision-matching pre-stage pin and a later fresh physical
residency hint on the exact scheduler node. D3.2 atomically applies that binding
to a scale-from-zero workload Pod template and replica count. D3.3 requires an
exact Runner-side startup proof before readiness. D3.4 adds the library-side
CREATE admission and linearizable claim contract for incremental per-Pod
placement. D3.5 adds bounded-batch released-placement compaction into durable
per-placement fencing high-water marks. D3.6 adds the shared PostgreSQL plan and
claim store required by a replicated admission service. D3.7 adds the strict
Kubernetes AdmissionReview v1 HTTP boundary and deterministic JSON Patch
mutation. D3.8 adds the executable TLS webhook runtime, shared-store assembly,
and authenticated nonce-bound binding reauthorization client. D3.9 adds the
authenticated scaling-authority HTTP boundary that serves that client. D3.10
adds its embedded process-boundary assembly with strict configuration,
file-backed credentials, TLS material validation, bounded budgets, and owned
shutdown. D3.11 adds leader-fenced validation of the live binding, durable
decision, target, quota, prewarm capacity, cache freshness, and lifetime.
D3.12 composes those facts through explicit deadline-bounded current-binding,
decision, target, quota, and cache backend readers with a second binding read as
a replacement fence. D3.13 exposes path-free, owner-scoped live pre-stage and
cache-index evidence from the authenticated node agent. D3.14 adds the strict
authenticated controller client and bounded fan-out aggregator, joining those
node responses only to scheduler placement facts observed afterward.
D3.15 adds deadline-bounded PostgreSQL current-binding and exact durable-decision
readers. D3.16 adds deadline-bounded Kubernetes target, Kueue quota, and
controller-owned placement-inventory readers. D3.17 adds a lifecycle-owned,
scheduled caller for the bounded compaction contract. CRD reconciliation,
runtime assembly, deployment wiring, and live acceptance remain deployment
tasks.

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
`build_runner_start_prestage_commands()` applies the same fenced command and pin
contract to the canonical `runner_start_placement_ids`; a cache hit still passes
full artifact verification before the deployment-scoped owner pin is recorded.
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

`PostgresNodeModelPrestageStore` is that production adapter. One registry row
binds a stable store ID to exactly one node ID, schema version, and placement
capacity. Every mutation locks that row with `FOR UPDATE`, validates the full
stored record against projected placement, command, generation, fencing,
revision, state, and time columns, applies the same pure transition functions
used by the in-memory specification, and atomically upserts the full JSONB
record. The registry lock serializes capacity checks and placement CAS across
processes; exact replays remain idempotent. Tables are schema-qualified,
permanent ordinary `public` tables with strict column and constraint checks.
Reconnect rejects a replaced namespace or incompatible store configuration.

This first durable version deliberately takes one lock per node-scoped store,
favoring linearizable fencing and bounded capacity over parallel placement
mutation throughput. Deployments should allocate a distinct stable store ID per
node. A later sharded design may replace the coarse lock only if it preserves
the same transition and capacity semantics.

Released `absent` records remain durable fencing tombstones and count against
`max_placements` until explicitly compacted. The store fails closed at that
bound. They must not be deleted merely because their command expired: doing so
would discard the last accepted generation, leader fence, and target revision,
allowing an older lineage to be treated as a new placement.

`compact_absent_records()` moves only absent rows at or before an explicit
operator-selected retirement cutoff, in batches of at most 1,000. Under the
same node-scoped linearization lock it first upserts a minimal, schema-validated
`NodeModelPrestageHighWaterMark`, then deletes the full active row. The mark
retains placement and release-command identity, generation, election/holder and
fencing token, target/revision, attempt, and source/compaction times. A compacted
release replays exactly without recreating an active row; a successor ensure or
release must still advance the retained fences. Release identity is retained as
a canonical digest so a changed artifact, workload, node, profile, or pin owner
cannot inherit the lineage. High-water rows do not consume active placement
capacity and are one bounded record per placement ID.

PostgreSQL schema version 2 adds the high-water table. Initialization creates it
under the existing advisory schema lock and upgrades only the initializing
store's version-1 registry row after strict table validation. Other node/store
rows remain at version 1 until their own agent upgrades, allowing a rolling
deployment; startup without schema initialization fails closed on the old
layout. In-memory and PostgreSQL adapters expose the same optional compaction
extension without expanding the runtime-checked execution-store protocol.

## Node agent HTTP boundary

`create_node_model_cache_agent_app()` exposes only the bounded operations needed
by a trusted controller:

- `POST /v1/prestage/ensure` accepts one fenced command, claim digest, signed
  manifest, and GitOps admission request;
- `POST /v1/prestage/release` accepts one separately fenced release command;
- `POST /v1/cache/live-evidence` returns one path-free, owner-scoped placement
  lineage joined from the durable pre-stage store and one cache-index snapshot;
- `GET /v1/prestage/records` returns at most 100 node-scoped status projections
  with a placement cursor; and
- `/health` and `/readyz` provide low-disclosure liveness and an injected
  backend readiness check.

All `/v1` routes require one of the locally configured API keys as a bearer
credential or `x-api-key`, compare credentials in constant time, enforce the
app's fixed node identity before execution, bound request bytes and
concurrent/queued work, and offload blocking cache/DB operations from the event
loop. Status responses omit local artifact paths and durable failure details.
Artifact trust roots are constructor inputs and cannot be supplied or replaced
over the wire. Known command, capacity, admission, cache, and PostgreSQL errors
use sanitized responses; backend-unavailable responses are retryable. Internal
failure details stay in the durable failed record and operator logs. Readiness
checks are coalesced and briefly cached so unauthenticated probes can consume at
most one blocking worker at a time.

Bearer authentication is the application floor, not the whole deployment
security boundary. The DaemonSet must receive rotated credentials through the
approved secret mechanism and be isolated by NetworkPolicy; production may add
mTLS at the service mesh or sidecar without changing the command contract.

## Executable runtime boundary

`kairyu cache-agent serve <runtime.json>` is the deployment entrypoint. Its
versioned JSON config contains only paths and non-secret limits. Node identity,
PostgreSQL DSN, controller API keys, optional artifact-gateway bearer token,
and public manifest trust roots are loaded from separately mounted files before
the listener starts. Duplicate config keys, non-finite values, embedded URL
credentials, invalid hosts/ports, plaintext HTTP without an explicit opt-in,
multiline scalar secrets (apart from one optional terminal LF), non-regular
config/secret files, and an index outside `<cache-root>/state` fail startup.
Projected-secret symlinks remain supported because validation applies to the
opened file descriptor.

The runtime constructs one node-bound SQLite index, HTTP range source,
PostgreSQL pre-stage store, verified cache agent, executor, and HTTP app. The
PostgreSQL schema is validate-only by default; migration/initialization remains
an explicit deployment action. Startup and readiness validate the owned cache
directories, durable registry, and local index. Shutdown closes the PostgreSQL
connection and artifact HTTP client idempotently.

The index accepts an explicit artifact cache root independently of its SQLite
file location. Production therefore stores metadata at
`<cache-root>/state/cache-index.sqlite3` while all validated artifact paths
remain under `<cache-root>/artifacts/<digest>/tree`.

## Node execution and pins

`NodeModelPrestageExecutor.execute()` validates the command against the GitOps
admission request before claiming. It then calls the WP4.2
`NodeModelCacheAgent.ensure_cached()`, so signature, environment, GPU profile,
blob digest, tree digest, resumable staging, and atomic publication checks are
unchanged. Only after a verified result exists does it add the deterministic
WP4.3 owner pin and CAS the command to `ready`. That durable completion also
stores the cache record generation returned by the pin transaction. Older ready
records without this additive evidence remain readable but cannot authorize a
D3 startup binding; they must be released and ensured again.

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

## Runner startup binding

`build_runner_cache_startup_binding()` is the D3.1 handoff to scheduler
actuation. A ready placement is not sufficient by itself. For every planned
Runner start, the builder requires all of the following to agree exactly:

- scaling decision ID/fingerprint and target revision;
- deployment, placement binding, placement/node, resource flavor, hardware
  profile, and compatibility approval;
- model ID, revision, and lowercase manifest SHA-256;
- a completed `ensure` record with the deterministic deployment owner pin; and
- a node hint observed after that completion which still publishes the exact
  artifact as verified and pinned at binding time, with the same cache record
  generation captured by the completed pin.

The generation equality is load-bearing: releasing the deployment owner while
another owner remains advances the cache record, so a later `pinned=true` hint
cannot be combined with the stale completion. The resulting
`RunnerCacheStartupBinding` lists the scheduler node for every
placement, preserves the pin command/generation and node index/generation
evidence, expires at the earliest source hint, and hashes its complete canonical
payload into `binding_id`. Duplicate placements/nodes, stale or pre-completion
hints, unpinned residency, cross-decision records, and payload substitution fail
closed.

## Atomic cold-start scheduling

D3.2 passes a `RunnerCacheStartupBinding` to the fenced Kubernetes scale
actuator. For a scale-from-zero decision, the actuator verifies the binding
against the durable decision fingerprint, workload UID/generation, model and
manifest, original prewarm snapshot, exact placement IDs/nodes/profiles, and
target delta. It then builds one JSON Patch which tests the workload resource
version, leader fence, placement binding, current replica count, and complete
Pod template before replacing both the template and replica count. Kubernetes
therefore cannot create a new Pod from the scale decision before its cache
constraint is present.

The managed Pod template carries the complete canonical binding JSON and full
binding digest as annotations. Model/revision/release identity is copied or
must already match. Required node affinity ANDs the binding's node names into
every existing required selector term; existing preferred and required policy
is preserved. A binding-scoped required Pod anti-affinity term prevents two
new replicas from occupying the same hostname. The template also carries the
target identity and managed scheduling gate consumed by D3.4 before each Pod
is persisted. Exact replay removes and
rebuilds only the prior managed terms and produces byte-equivalent template
state. Malformed prior evidence, ambiguous managed terms, conflicting identity,
or annotation-size overflow fails closed.

Immediately before the patch, leader, quota, prewarm, and binding authority are
reauthorized. The binding must still be live and byte-identical. The API
response must advance the workload generation exactly once and reproduce the
expected template and decision annotations. An exact retry succeeds only while
that template remains intact.

This shared-template mechanism is deliberately limited to scale-from-zero.
Changing a Deployment or StatefulSet template while replicas already exist can
roll existing Pods and cannot express a distinct placement per new ordinal.
Such an incremental scale-up is rejected before mutation. A later D3 unit must
use the D3.4 Pod CREATE admission path described below; it must not weaken this
scale-from-zero guard by rewriting a live workload template.

## Runner startup attestation

D3.3 closes the scale-from-zero evidence loop before readiness. The Kubernetes
watcher strictly parses the full inherited binding and its independent digest
annotation, rejects duplicate JSON keys, non-finite values, incomplete metadata,
or model/revision conflicts, and retains the validated binding with the Pod
observation.

After scheduling, the Runner performs the WP4.6 full-digest verification on its
local node. An allowed verification is converted into a canonical hash-bound
`RunnerCacheStartupProof` containing the Pod UID, actual node, binding and
placement IDs, scaling decision fingerprint, model/revision/manifest digest,
pre-stage command generation, and resident record generation. The local artifact
path is deliberately omitted. A denied verification, unselected node, proof
predating the binding, future proof, altered generation, or cross-Pod replay
fails closed.

Legacy Pods without cache-binding metadata retain the existing readiness
contract. A cache-bound Pod must additionally return the exact proof through the
Runner runtime observation; until then, it remains `WARMING` even if Kubernetes,
EndpointSlice, startup phases, and the ordinary Runner readiness flag are all
ready. Only the fully matched proof makes the Runner routing eligible. The proof
is based on current physical digest verification, so it need not complete before
the short-lived placement hint expires. It must not predate binding beyond the
same bounded Runner/controller clock-skew policy used by reconciliation, and it
must precede the same-clock runtime observation that carries it. Once a Pod is
observed with a binding, its digest is retained in `RunnerStatus`; removing or
changing the Pod annotation cannot downgrade that Runner into the legacy
readiness path.

## Incremental per-Pod placement

D3.4 supplies the library-side admission boundary for scale-up while replicas
already exist. The controller registers one still-live
`RunnerCacheStartupBinding` before increasing desired replicas. The workload
template already carries `kairyu.ai/cache-startup-target` and the
`kairyu.ai/cache-startup-placement` scheduling gate. A mutating admission
handler processes each Pod CREATE before persistence; it does not attempt to
change immutable scheduling fields on a live Pod or rewrite the shared workload
template.

When the CREATE object inherits a D3.2 full binding but has no selected-placement
annotation, admission treats it as template-derived rather than already
admitted. It strictly removes the old multi-node managed affinity, binding
anti-affinity, and binding label, then writes the same or successor binding's
single selected node. Once a selected-placement annotation exists, only an
exact replay is accepted. This distinction lets the initial scale-from-zero and
later incremental scale share one gated template without rolling existing Pods.

Admission resolves the active target plan, then matches the request namespace,
controller owner API version/kind/name/UID, and AdmissionRequest creator username
before any capacity is consumed. It reauthorizes the complete binding and
atomically claims the next canonical placement by namespace/name and binding
digest. A retry for the same Pod name returns the same claim even if the API
server supplies a new admission UID. Concurrent names receive unique placement
IDs. Plan replacement, expiration, exhaustion, authority drift, and malformed
Pod metadata all fail closed.

The admitted Pod has the managed scheduling gate removed and an exact
`metadata.name In [node]` match field ANDed into every existing required node
selector term. Other gates and affinity remain intact. The full canonical
binding, digest, selected placement, model/revision/release identity, and
binding label are written into Pod metadata for the D3.3 observer and startup
proof. The watcher retains the selected placement separately, checks it against
the actual scheduler node, and D3.3 requires the Runner proof to name that same
placement. Reconciliation prevents the placement annotation from disappearing
or changing after observation. Replay accepts only the same binding and placement and strictly rejects
duplicate JSON keys, non-finite values, incomplete annotations, conflicting
identity, or changed managed affinity.

`InMemoryRunnerCachePlacementAdmissionStore` is an executable thread-safe
specification for tests and a single admission process. A production webhook
replica set requires a shared linearizable implementation of the same
register/resolve/claim/release protocol. Rollback applies only to a newly created
claim that no concurrent retry has observed; once replayed, it is conservatively
protected even if one request later fails mutation. A claim is retained after
an allowed response because the webhook cannot prove that later admission
stages persisted the Pod. A claimed plan cannot be replaced—even when
exhausted—until its binding expiry plus a configured replay safety window. The
window must exceed the outer API-server webhook timeout. Thus old placement
nodes stay reserved until no prior allowed response can plausibly persist; an
unclaimed plan may rotate immediately after binding expiry.

`PostgresRunnerCachePlacementAdmissionStore` is the D3.6 shared implementation.
One registry row fixes the store ID, schema version, target capacity, and replay
safety window. Every register, claim, replay-protection, and rollback mutation
locks that row with `FOR UPDATE`, so webhook replicas serialize placement
allocation before a unique `(store, target, placement)` database constraint is
reached. Plans and claims retain their full canonical JSON beside projected
identity/time columns. A canonical full-plan SHA-256 also covers release,
namespace, owner, and creator authorization that is not individually projected;
any disagreement fails closed on read. A claim replay
atomically makes the claim rollback-protected before returning the original
allocation, so a concurrently failing creator request cannot free a placement
already observed by another API-server retry.

Plan replacement preserves the in-memory replay boundary: an unclaimed plan can
rotate at binding expiry, while a claimed plan retains every node through
`valid_until + replay_safety_window`. Replacement deletes the old binding's
claims only after that boundary. The PostgreSQL authoritative clock must cross
that boundary and must independently see a binding as live for registration and
claim; future caller timestamps cannot rotate a plan early and backdated request
timestamps cannot claim an expired binding. Store capacity and replay configuration are
durably fixed, validate-only startup is supported for explicit schema
provisioning, and schema/table/constraint plus JSON-projection checks reject a
replaced or corrupted PostgreSQL namespace. The initial implementation uses one
coarse lock per admission store, prioritizing cross-replica correctness over
parallel target throughput.

An admission replica can fail after the claim transaction commits but before its
rollback runs. Such an unprotected orphan remains reserved: reclaiming it by age
alone is unsafe because the API server may already have persisted the admitted
Pod. Recovery is therefore fail-closed and durable rather than eager. The next
plan registration reclaims all claims only after the old binding expiry plus the
replay safety window, using the PostgreSQL clock. Tests cover both early
replacement rejection and successful reuse after this boundary. This bounded
capacity reservation is the safe crash-recovery contract until a future
Kubernetes-observed claim reconciler can prove Pod absence.

The D3.7 HTTP boundary accepts only strict `admission.k8s.io/v1` JSON for a
core/v1 Pod CREATE at `/v1/admit`. It binds the response UID to the request,
checks the request and embedded Pod name/namespace identities, rejects
subresources, `oldObject`, and side-effectful dry runs, and supplies a
server-owned UTC observation time to D3.4. A successful decision is encoded as
a deterministic RFC 6902 JSON Patch; an exact replay omits empty patch fields.
Semantic denials remain HTTP 200 AdmissionReview responses, while malformed
JSON, unsupported media, oversized bodies, and local overload fail at the HTTP
boundary. Conflict, authorization, scheduling, and backend details are reduced
to bounded public messages. The service exposes only low-disclosure `/health`
and `/readyz` probes and bounds both request size and in-process concurrency.

The AdmissionReview `userInfo` is trustworthy only when the endpoint is
reachable exclusively through the Kubernetes API-server webhook path. D3.7
does not make that wire object a client-authentication mechanism. The deployed
Service therefore requires TLS trust, ingress restriction (NetworkPolicy and,
where available, authenticated mTLS), and a fail-closed webhook policy.

D3.8 exposes the runtime as `kairyu placement-admission serve <config>`. The
versioned JSON config supplies only absolute secret/certificate paths and
bounded process settings; the PostgreSQL DSN and binding-authority bearer token
are loaded separately from regular files. Secret files reject group write or
execute and every other-user permission while allowing owner access and an
optional group read for Kubernetes `fsGroup`; deployments must mount them with
mode `0640` or stricter. The process validates these permissions from the open
file descriptor, serves one Uvicorn worker with the configured certificate/key,
opens the D3.6 store, and requires both PostgreSQL and the authority to pass
readiness before serving.

Each admission receives an internal monotonic deadline from the runtime config.
The controller checks it before and after every blocking resolve,
reauthorization, and claim boundary. It never starts a claim after expiry; if a
new claim returns after expiry, it releases that claim before the request is
denied. Deployments must keep this deadline below the API-server webhook
timeout and reserve enough outer budget for queueing, a late-claim release, and
response delivery. This cooperative deadline does not interrupt a database
statement in flight; the PostgreSQL statement/lock timeout remains the hard
bound for each database operation.

Final binding reauthorization is an authenticated HTTPS POST to the configured
scaling-authority HTTPS origin; no plaintext opt-in exists. The request includes
the complete candidate binding and a fresh 256-bit nonce. Both serialized
request and response are independently size-bounded before network send or
JSON validation, and the strict response must echo that nonce and return a
complete binding. The configured response limit must exceed the request limit
because the response schema identifier is one byte longer. D3.4 then requires
the returned binding to be
byte-equivalent to the plan. Redirect following, environment proxy inheritance,
duplicate keys, non-finite numbers, cross-origin readiness, unbounded response
bodies, and cached responses are rejected. The authority readiness endpoint
must return an empty HTTP 204 response. The bearer token is 32--4096 visible
ASCII characters so it is safe to place in an HTTP header and authenticates this
internal client; it does not replace API-server-to-webhook TLS and NetworkPolicy.

D3.9 defines the corresponding library-side authority endpoint. It requires an
exact bearer token before parsing authorization JSON, rejects duplicate keys,
non-finite numbers, oversized bodies, unknown fields, and unsupported media,
then passes the complete binding to a caller-owned live-authority callback.
Only a byte-equivalent current binding receives a nonce-echoed HTTP 200
response. A stale or explicitly denied binding returns a sanitized 409;
dependency failures return a sanitized 503. Readiness is authenticated and
empty-body HTTP 204, matching the D3.8 client contract. Request concurrency is
bounded, responses are marked `no-store`, and backend detail never crosses the
HTTP boundary. Both callbacks receive an absolute monotonic deadline and a
shorter backend timeout. They must apply that hard timeout to every database or
remote call: the outer asynchronous deadline releases the HTTP request but
cannot stop a worker thread that ignores the callback contract. The live
callback remains responsible for leader, target revision, quota, prewarm,
binding lifetime, and cache-freshness reauthorization.

D3.10 assembles that endpoint at the scaling-controller process boundary. Its
versioned, unknown-field-rejecting configuration carries only non-secret
settings and absolute paths. The builder loads a bounded bearer-token file,
validates bounded TLS certificate/private-key files, rejects overly broad
token/key permissions, and applies independently bounded queue, request,
backend, request-body, response-body, and concurrency budgets to the D3.9 app.
The configuration declares the D3.8 client authorization timeout and reserves
an explicit transport margin; validation requires authority queue wait plus
authority request plus that margin to remain strictly below the client timeout.
D3.8 separately requires that client timeout to remain below its overall
admission deadline. Deployments must set the declared value equal to D3.8's
actual `authorization_timeout_s`.

The builder adopts the supplied resource closer after validating its arguments.
It invokes the closer exactly once on any later assembly failure, or registers
it as an idempotent application-shutdown hook after success. It returns the app
together with the validated listen/TLS configuration. The hosting
scaling controller must run that app with the returned TLS/listen settings and
provide deadline-aware reauthorization/readiness callbacks backed by its own
live leader, target, quota, prewarm, and binding state. This deliberately does
not dynamically import callbacks or create a standalone placeholder authority:
the process that owns the live scaling state also owns service startup.

D3.11 supplies `ScalingControllerPlacementBindingAuthority` as the callback
adapter for that process. Every authorization executes under
`LeaderFencedRunnerController`, reads one deadline-bounded
`RunnerCachePlacementBindingLiveState` from the controller-owned source, and
then re-authorizes leadership after the read. The candidate must remain the
source's exact current binding and be inside its validity window. Its durable
decision must still be a fenced, generated scale-up whose target, fingerprint,
model revision, artifact, placement set, and original prewarm identity match.

The refreshed quota may advance its source revision but must retain the Kueue
and tenant/model identity, stay within the policy freshness window, and still
admit the decided replica count. The refreshed prewarm view may likewise
advance, but each originally selected placement must remain healthy,
schedulable, unassigned, cache-ready, identity-equivalent, and live at the
leader store's authoritative validation time. The binding's original hint
observation, expiry, and index revision must exactly match the durable decision.
The live source must also return the current pre-stage record and physical node
hint for every selected placement, plus owner-scoped pin evidence derived from
the same node-index revision and observation as that hint. Command
identity/generation, the binding owner remaining in the current pin set,
resident generation, pinned state, and artifact identity must still match;
refreshed hint time/revision may advance but cannot roll back, and its exact
values must agree with the refreshed prewarm view. This prevents unpin and
evict/refill ABA from laundering an old binding through a merely READY cache
state. State snapshots that predate the binding, exceed the decision freshness
limit, or claim a different current binding are denied.

Time freshness is evaluated only after the source read, using the leader
store's refreshed authority time. A snapshot legitimately observed between the
initial authority check and final reauthorization is therefore accepted, while
a tenure change during the read is denied. The absolute deadline is checked
both around the read and after the final full validation. Backend failures and
loss of leadership remain dependency failures; HTTP sanitization is owned by
D3.9.

The source protocol intentionally requires both `read` and `readiness` to
receive the absolute request deadline and per-backend timeout. The concrete
scaling-controller process must apply that timeout to every PostgreSQL,
Kubernetes, Kueue, and cache-state call. D3.11 does not create an echo source
from the admission plan store: the live source must be independently backed by
the process's durable decision and current controller state.

D3.12 adds `ComposedRunnerCachePlacementBindingLiveStateSource`, the concrete
assembly layer between that protocol and five explicit backend readers. It
reads the current binding, durable decision, Kubernetes target, refreshed
quota, and one coherent cache/pre-stage evidence group in order, capping every
backend timeout by the remaining absolute request budget. It then reads the
current binding again; replacement before the second read is an authorization
denial rather than a mixed snapshot. Readiness checks every distinct backend
once under the same budget. Reader outputs are deep-validated before the D3.11
authority consumes them, and the cache reader must return prewarm, path-free
pre-stage projections, physical hints, and owner-scoped pin evidence together.
Production adapters for PostgreSQL and Kubernetes/Kueue remained outside that
assembly boundary.

D3.13 adds `LocalNodeModelCacheLiveEvidenceSource` and the authenticated
`POST /v1/cache/live-evidence` node-agent endpoint. The request carries one
exact binding placement and model identity. The source reads the durable
pre-stage record before and after one transactionally consistent cache-index
snapshot, rejects lineage drift, and requires the requested command generation,
pin owner, artifact identity, and resident generation to remain current. Its
response contains only a path-free ready pre-stage projection, one filtered
physical residency hint, and owner-scoped pin evidence derived from the same
index revision and observation time; local artifact paths, fill results, and
failure details never cross the transport. The production node-agent runtime
always constructs this source and uses a bounded hint TTL. Missing or changed
lineage is a sanitized conflict, while unavailable state remains fail closed.

D3.14 adds `AuthenticatedNodeModelCacheLiveEvidenceClient` and
`AggregatingRunnerCachePlacementBindingCacheReader` on the scaling-controller
side. Node URLs come only from a canonical, credential-free HTTPS endpoint
registry; a binding cannot supply or redirect the destination. Every request
uses an explicit bounded bearer credential, disables redirect following,
rejects environment proxy/CA inheritance, applies the smaller of the backend
timeout and remaining absolute deadline, bounds request and streaming response
bytes, rejects duplicate/non-finite JSON, requires JSON plus
`Cache-Control: no-store`, and binds the returned node, command, artifact, pin
owner, generation, hint, and pin evidence to the exact request. Conflict is an
authorization denial; transport, authentication, and malformed responses
remain dependency failures.

The client adopts one `httpx.AsyncClient` on a single owned event loop and
performs cancellable fan-out through a fixed, configured number of worker tasks
for exactly the binding's placements. It never creates one task per placement.
Total async timeouts, per-chunk absolute-deadline checks, sibling cancellation,
and joined shutdown prevent slow-drip responses or partial failure from leaving
background workers and connections. The aggregator then
reads `RunnerCachePlacementBindingInventory` from an injected controller-owned
scheduler adapter. Inventory must be canonical, cover exactly the bound
placements, have a non-regressing cache revision, and be observed no earlier
than every node hint while all hints remain live. Only then does the aggregator
rebuild the prewarm snapshot and plan. This ordering prevents stale health,
schedulability, or assignment values from the original decision from being
presented as current facts. Readiness covers every configured node and the
inventory backend under the same deadline budget. The hosting runtime owns and
closes the adopted client before process shutdown.

D3.15 adds `PostgresRunnerCachePlacementBindingReader` as the current-binding
and durable-decision side of the D3.12 source. It resolves the target's current
admission plan and the exact decision ID named by that binding from their
existing shared PostgreSQL stores. Missing plans or decisions become explicit
authorization denials; connection, schema, projection, and data-corruption
errors remain dependency failures. Every live read and readiness check applies
the smaller of the per-backend timeout and remaining absolute deadline. The
store methods subtract Python lock wait from that budget and set transaction-
local PostgreSQL statement and lock timeouts from the remaining budget before
every validation or selection query. A whole-operation PostgreSQL cancel timer
also covers transaction startup and timeout-configuration round trips. Timed
authority reads never perform a synchronous libpq reconnect because its timeout
is per connection candidate and cannot be bounded by the short request deadline;
they fail fast on a broken connection. Connection establishment and repair stay
in non-request startup/health lifecycle calls. Results are deep-validated again
before crossing the reader boundary.
One reader instance implements both D3.12 protocols so composed readiness checks
the paired durable dependencies once.

D3.16 adds `KubernetesKueueRunnerCachePlacementBindingReader` for the remaining
Kubernetes/Kueue boundaries. Trusted per-model configuration fixes the API
origin, workload identity, authority namespace, Kueue namespace/API version,
pod set, resource name, and CRD names; no binding field can redirect a request.
It reads the exact `apps/v1` Deployment or StatefulSet and projects the applied
decision annotations, generation, UID, model identity, placement binding, and
replica count into `RunnerCachePlacementBindingTargetState`.

The quota read joins the exact Kueue Workload to an
`autoscaling.kairyu.ai/v1alpha1` `RunnerScalingQuotaSnapshot`. The CRD spec binds
the immutable workload and Kueue references plus tenant/model identity. Its
ready status carries the monotonic quota revision, source timestamp, exact
Kueue resourceVersion, and the three canonical cluster/model-family/tenant
limits. A projection that does not observe the returned Kueue resourceVersion
is denied instead of mixing snapshots. The live Kueue admission is parsed with
the existing strict v1beta1/v1beta2 contract and deterministically reapplied to
the refreshed limits.

The placement read consumes an `autoscaling.kairyu.ai/v1alpha1`
`RunnerCachePlacementInventory`. Its spec binds the target UID, binding,
decision/fingerprint, model artifact, and placement-binding identity. Its ready
status carries the monotonic cache revision, source timestamp, and canonical
controller-owned candidate facts: node, ResourceFlavor, hardware profile,
compatibility approval, assignment, health, and schedulability. D3.14 still
joins those facts only after collecting node evidence and enforces exact
placement coverage.

All reads re-read the projected service-account token, reject redirects and
environment proxy/CA inheritance, require strict bounded JSON, and apply one
shared operation budget across Python lock wait and every sequential HTTP
request. The reader adopts and closes one `httpx.AsyncClient` on a dedicated
event loop; a whole-request async timeout completes cancellation before the
synchronous callback returns, and shutdown joins the loop thread. A missing
target, Workload, or authority CRD is an authorization
denial; API transport/RBAC/server failures and malformed responses remain
dependency failures. Readiness verifies the configured workload, both CRDs,
and Kueue collection access under that same budget.

D3.17 adds `NodeModelPrestageCompactionRuntime` as the scheduled caller for the
D3.5/D3.6 store contract. Each cycle selects one explicit retirement cutoff
from its timezone-aware start time and configured retention age, then reuses
that cutoff and compaction time for every batch. Both the batch size (at most
1,000) and calls per cycle are bounded, and their product cannot exceed 10,000
retained results. A short batch ends the cycle; consuming the complete call
budget is reported as possible backlog rather than causing an unbounded catch-up
loop.

The worker starts with an immediate cycle, retries ordinary backend failures on
the fixed interval, and owns a stop event plus bounded join. It does not own or
close the injected store. Process-local status preserves successful batch and
record totals even when a later batch in the same cycle fails, but exposes no
backend exception text. Readiness requires a live worker, at least one recent
successful cycle, and fewer than the configured consecutive-failure threshold.
The last successful cycle contains the exact compacted high-water marks, while
`list_high_water_marks_page()` exposes at most 1,000 deep-validated durable marks
through a placement-ID cursor without materializing the unbounded history. A
synchronous `run_once()` uses the same contract for explicit jobs and
deterministic tests, rejects overlapping cycles, and is included in the bounded
shutdown wait.

## Deployment boundary

`private-ai-cloud-iac` must still provide:

- service discovery, rotated API credentials or mTLS, and NetworkPolicy for the
  authenticated per-node HTTP API;
- a node DaemonSet/service invoking the implemented `cache-agent serve`
  entrypoint;
- reconciliation that replays desired ensure/release commands after restart;
- assembly and configuration of the implemented scheduled compaction runtime,
  including an operator-approved retention age plus export of runtime status
  and per-placement high-water monitoring;
- reconcilers and CRD definitions that publish the implemented
  `RunnerScalingQuotaSnapshot` and `RunnerCachePlacementInventory` contracts,
  with least-privilege read RBAC for the implemented Kubernetes/Kueue reader;
- assembly and deployment of the scaling-authority
  reauthorization/readiness endpoint consumed by D3.8, plus a highly available
  Deployment/Service/MutatingWebhookConfiguration, TLS
  certificate issuance/rotation and trust, ingress restriction, orchestration
  that registers the plan before the D3.2 scale write, and fail-closed webhook
  policy;
- metrics and audit export for command latency, bytes, attempts, failures, and
  pin reconciliation; and
- live node-pool acceptance with real S3, NVMe, Kueue, and Runner startup.

Until that wiring exists, the PostgreSQL and Kubernetes readers form a
fail-closed library authority but no production node agent consumes its work;
the feature must remain disabled for production autoscaling.

## CPU verification

Tests cover deterministic command hashing, exact generation allocation,
admission and node binding, claim concurrency, expiry, verified fill and pin,
idempotent replay, failure/retry, fresh fenced release, preservation of other
pin owners, bounded compaction, post-compaction replay and successor fencing,
scheduled cutoff/batch limits, retry/readiness/shutdown behavior, canonical
high-water monitoring, and exact monotonic status projection. Real PostgreSQL
tests add cross-instance claim races, completion/release replay, shared capacity
reclamation, durable high-water visibility, store configuration mismatch, and
projected-metadata corruption detection.
HTTP tests cover API-key protection, local-only trust roots, node binding,
ensure/release replay, path/failure-detail redaction, readiness sanitization,
bounded cursor pagination, error mappings, concurrent admission, and declared
or chunked request-size bounds.
Scheduling tests cover affinity preservation, exact replay/rebinding, malformed
managed state, decision and model mismatch, scale-from-zero-only enforcement,
binding expiry, atomic template/replica CAS, response tampering, and retry-time
template integrity.
Real PostgreSQL admission tests cover cross-instance plan visibility, concurrent
unique claims, same-name retry protection, safe creator rollback, target
capacity, replay-window plan rotation, fixed configuration, validate-only
startup, plan/claim projection corruption, and deadline-bounded live reads.
AdmissionReview tests cover strict v1 envelope parsing, request/Pod identity,
UID-bound allow and deny responses, deterministic patch application, exact
replay, dry-run/subresource rejection, request-size bounds, readiness, and
failure-detail redaction.
Runtime tests cover strict bounded config/secrets, HTTPS/same-origin policy,
nonce and full-binding request/response integrity, redirect/proxy suppression,
response-size and non-finite-number rejection, TLS key permissions, dependency
readiness, resource cleanup, end-to-end admission, and TLS CLI arguments.
Authority API tests cover bearer protection, nonce echo, exact-binding denial,
strict/size-bounded JSON, authenticated readiness, concurrency limits, and
failure-detail redaction.
