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

## Deferred production slices

- [ ] PostgreSQL implementation using `FOR UPDATE SKIP LOCKED`, database-clock
  leases, transactional terminal publication, and claim audit events.
- [ ] `/v1/requests` submit/status/list/cancel API and shared chat dispatch worker.
- [ ] Redis wake-up hints, queue-age/admission metrics, and multi-gateway failover
  drills. Redis must not become the source of truth.
