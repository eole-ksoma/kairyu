# P0 AsyncRequest v1 Implementation Plan

**Goal:** Establish the durable online-request lifecycle contract required before
adding a PostgreSQL queue, worker, and `/v1/requests` API.

**Architecture:** Keep asynchronous online requests separate from OpenAI Batch
jobs. A frozen public snapshot represents caller-visible state; a separate fenced
claim represents temporary worker ownership. The initial in-memory backend is an
executable specification only. A production backend must implement the same
protocol with shared durable storage and database-clock leases.

## Task 1 — State and submission contracts

- [x] Add top-level-frozen, extra-forbidden Pydantic models for submission,
  request, structured error, state, and worker claim; deep-copy at store edges.
- [x] Restrict request bodies and results to recursively JSON-compatible values.
- [x] Require timezone-aware timestamps and valid terminal payloads.
- [x] Keep claim ownership and lease details out of caller-visible request state.

## Task 2 — RequestStore protocol and reference backend

- [x] Add submit/get/list/claim/renew/run/succeed/fail/cancel operations.
- [x] Scope idempotency keys by tenant and reject conflicting replays.
- [x] Prioritize claims by priority, then creation time and request ID.
- [x] Reclaim expired leases with monotonically increasing fencing tokens.
- [x] Make cancellation and deadlines invalidate active claims.

## Task 3 — Verification and review

- [x] Cover tenant isolation, idempotency, lifecycle transitions, ordering,
  renewal, reclaim, cancellation, deadlines, and structured failure.
- [x] Run the focused suite (19 tests) and repository-wide Ruff.
- [ ] Re-run the complete portable CPU suite in the locked development image;
  the available macOS environment lacks `langdetect`/`cryptography` and has an
  existing `/proc/cpuinfo` collection dependency.
- [x] Obtain an independent code review and resolve accepted findings; the final
  review has no remaining high- or medium-severity findings.

## Task 4 — PostgreSQL durable backend

- [x] Add a PostgreSQL implementation using `FOR UPDATE SKIP LOCKED`, database-clock
  leases, transactional terminal publication, and claim audit events.
- [x] Enforce lifecycle, lease shape, terminal payload, tenant idempotency, and
  JSON object constraints in the database schema.
- [x] Verify cross-instance submission, claim exclusivity, renewal, takeover,
  stale-fence rejection, deadline precedence, terminal races, and audit order
  against a real PostgreSQL server.

## Task 5 — HTTP API and shared chat worker

- [x] Add `POST /v1/async/chat/completions` plus tenant-scoped
  `/v1/requests` status/list/result/cancel routes, 202 receipts, deadline and
  idempotency controls, bounded bodies, and non-streaming enforcement.
- [x] Run claimed work through the shared direct-chat validation, preparation,
  tool gates, tenant admission/metering, priority, and usage contract.
- [x] Maintain the request lease during inference, abort local execution after
  cancellation or lost fencing, and stop claiming new work during shutdown.
- [x] Wire PostgreSQL construction, fixed worker consumers, and store closure
  through `DeploymentSpec` and the application lifespan.
- [x] Bound receipt/status/list projections independently of persisted bodies;
  use narrow PostgreSQL reads, tenant-policy queue priority, one worker-owned
  inference quota charge, owner-wide durable admission deferrals,
  deadline-aware heartbeats, and sanitized error logs.

## Task 6 — Staged deployment validation

- [x] Extend the disposable F1c topology with a deliberately slow direct mock
  model and the AsyncRequest PostgreSQL configuration on all three gateways.
- [x] Add a CPU/kind smoke driver for cross-gateway idempotency and reads,
  cancellation, deadline expiry, bounded large-body responsiveness, fenced
  owner takeover, and PostgreSQL restart/reconnection.
- [x] Compose the smoke after the existing binding F1c gate, require one clean
  source commit, refuse pre-existing clusters, and clean up by default.
- [x] Run the committed CPU/kind gate from source
  `9e74a910ae64a3c042dea2edf38292b97704204b`. The existing F1c replay passed
  all 26 checks, and the six-check AsyncRequest report passed before bounded
  evidence collection and automatic cluster deletion.
- [ ] Run multi-tenant fairness with the tenant-policy integration slice.
- [ ] Run GPU correctness, performance, and soak gates after the planned
  implementation slices are complete.

## Following production slices

- [x] Expose shared-store queue depth, oldest queued age, lifecycle transition,
  attempt, and snapshot-health metrics with bounded labels. Persist transition
  totals independently of per-request audit retention, use transaction-sharded
  durable counters, and retain the last good snapshot when PostgreSQL is
  temporarily unavailable. Existing stores use an explicit, drain-first v2
  maintenance migration; ordinary Pod startup never scans or locks history.
- [x] Extend the three-gateway staged gate to require identical shared queue
  telemetry, failover/expiry/cancellation counters, empty final depth, and no
  request-ID or prompt leakage.
- [x] Add opt-in AsyncRequest request/audit TTLs, bounded DB-clock purge,
  audit-safe archival, online index preparation, dry-run/apply CLI controls,
  and persistent transition counters. Request deletion ends the idempotency
  guarantee for that key; audit retention can remain longer than payload
  retention without keeping the request body or result.
- [ ] Add Redis wake-up hints only after measured PostgreSQL polling pressure
  justifies another operational dependency. Redis must not become the source
  of truth.
