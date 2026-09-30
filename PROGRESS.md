# Progress

Cross-session memory. Rules: `.claude/rules/progress-log.md`.
Older Change Log entries: `docs/progress/archive/change-log.md`.

## Product

**Goal** (master roadmap `docs/roadmap.md`, accepted 2026-07-03): serve a
Fugu-class orchestration product — multi-model auto-routing plus agent
ensemble/synthesis behind one OpenAI-compatible API and a chat UI — from an
on-prem DC of thousands of GPUs across **two first-class hardware profiles**:
NVLink-HBM nodes (8× H100-class, TP-first) and PCIe-GDDR nodes (RTX PRO 6000
Blackwell, DP-first / PP for capacity / EP over RDMA). TTFT/TPOT/goodput must
beat frontier APIs as measured by the committed harness (G6 gate P-C1).

- L1 stays Kairyu's own engine (kernel libraries like FlashInfer are used;
  scheduler, radix KV, spec decode, orchestration are ours). Layering:
  L3 Interface / L2 Orchestration / L1 Engines.
- Target model classes: small dense ~14B (latency tier), mid dense ~70B,
  mid MoE 100–300B, frontier MoE 500B+ (multi-node EP).
- Parallelism/quantization strategy is derived per measured hardware profile,
  never assumed. Details: `docs/goals/g2..g6-*.md`.

## Current Status

Snapshot date: 2026-09-26. Hardware context: all GPU evidence so far is on
8× RTX PRO 6000 Blackwell (SM120), PCIe-only interconnect (P2P 30–37 GB/s);
NVLink-HBM (H100-class) formal gates still need hardware. Evidence lives in
`bench/results/` (see `index.json`); decisions and rationale in `docs/design/`.

### Milestones

| Milestone | Status |
|---|---|
| M1 Orchestration+Interface | Complete: Router/Conductor/MoA, vLLM-compat API, OpenAI server, DSL |
| M2 Core engine | CPU half done; unified EngineLoop + device sampling GPU-validated; NVLink perf gates pending |
| M3 Spec/graphs/P-D | n-gram spec, CUDA-graph serving, intra-node P-D GPU-validated; EAGLE-3/MTP greedy serving integrated |
| M4 Router learning | Implemented CPU-only, design reviewed |
| M5 Intra-node multi-GPU | CPU half done; TP/DP/P-D plumbing live; GPU phase per runbook |
| M6 Inter-node multi-GPU | CPU half done; production stage-sharded PP remains a roadmap item |
| M7 Productionization | CPU half done: serve CLI, gateway, batch, compose smoke; `kairyu validate` preflight |
| M8 Engine CPU core | Complete (amended 2026-08-08) |
| M9 Truthful API | Complete |
| M10a/M10b Fleet base + KV routing | Complete |
| M11 Product surface + tenancy | Complete |
| M12 Dense model zoo | Complete; generation-default contract amended 2026-08-04 |
| M13 Attention backends | Complete; FA3/FA4 added, `auto` stays FlashInfer (measured faster on SM120) |
| M14 Quant compute | GPU-validated: FP8/INT8/AWQ/GPTQ/NVFP4 production-dispatch, fail-closed |
| M15–M18 MoE/MLA, distributed, graphs/drafts, KV transport | Complete |

### Formal gates

- G2 A1/A2/A7/A9 and A12 are closed; A8 is an accepted deviation; A6 remains
  open at 0.466× vLLM SLO-goodput.
- Quant evidence is retained: INT8 passes; AWQ/GPTQ quality and FP8-E4M3 KV
  exact-output/decode gates fail closed.
- G4 M-A2 is complete; M-A1 remains a retained failure and M-A3 an accepted
  scope deviation. G5 F1/F2/F4a/F4b and F5a/b/c are closed.
- G6 P-A, P-B1–P-B4, and P-C2/C3/C4 are green; remaining P-C gates continue.
- TP8 long-generation stability passed after the deadlock fix.

### What works today

- Native and vLLM-backed serving cover the accepted dense, vision, Qwen3.8, and
  DeepSeek V4/V4.1 profiles; exact hardware evidence is indexed under `bench/results/`.
- Engine paths include TP/DP/EP, device sampling, speculative decoding, CUDA
  graphs, structured masks, production quant dispatch, and process-split transport.
- The gateway provides auth, tenancy/metering, priority/SLO admission, batch,
  embeddings/RAG, Responses, tool calling, and trace-v2 accounting.
- Fleet support includes 3-gateway HA, PostgreSQL stores, prefix routing, DRAM KV
  tiering, immutable Helm deployments, safe rollout/drain, and monitoring.
- Tiered orchestration and replica-pool examples are GPU-verified; their exact
  policy, measurements, limitations, and run IDs remain in example documentation.
- The CPU suite, microbenchmark smoke, and nightly regression series are green.

### Open items / blockers

- G2 A6 remains the hard performance gate; #333 excluded process split and #318
  excluded admission depth beyond two as material fixes.
- Production stage-sharded PP, learned-draft evidence, frontier long-context and
  recovery gates, and remaining G6 P-C gates remain open.
- NVLink gates require H100/A100-class hardware; E4/E5 require the planned PCIe
  switch and ≥400 Gb/s RDMA environment.
- AsyncRequest retention-expanded Kind evidence awaits registry access.
- Runner autoscaling WP3.1–WP3.7 is fail-closed and CPU-tested; deployment wiring,
  durable runtime status, instrumentation, and live acceptance remain.
- Model-cache WP4.1–WP4.7 and D3.1–D3.17 provide signed identity, verified cache,
  fenced pre-stage/startup/admission, PostgreSQL/Kubernetes/Kueue authority, and
  TLS HTTP plus scheduled-compaction runtimes. CRD reconciliation, runtime
  assembly, deployment, and acceptance remain.
- Qwen3.8 MTP stays disabled pending vllm#53912; DTO-D15 verification/re-pin and
  human sign-off for M2–M4 remain pending.

## Change Log

Newest first; only the most recent entries are kept here (see the size budget
in `.claude/rules/progress-log.md`).

### 2026-09-30 — [progress] Scheduled pre-stage tombstone compaction
- What: added D3.17 cutoff-stable bounded compaction cycles, an immediate
  periodic worker, cumulative status, fail-closed readiness, bounded shutdown,
  and a validated durable high-water monitoring view.
- Why: released placement capacity must be reclaimed without discarding fences,
  running an unbounded catch-up loop, or hiding partial progress and repeated
  backend failures from operators.
- Refs: docs/design/node-model-cache-prestage-v1.md;
  kairyu/runners/prestage_compaction_runtime.py;
  tests/unit/test_runner_prestage_compaction_runtime.py

### 2026-09-30 — [progress] Kubernetes/Kueue live authority readers
- What: added D3.16 bounded target, Kueue quota, and controller-owned placement
  inventory reads with strict JSON/RBAC readiness and exact CRD projections.
- Why: D3.12 and D3.14 must refresh workload fences, quota, and scheduler facts
  without binding-controlled routing or mixing Kueue and quota revisions.
- Refs: docs/design/node-model-cache-prestage-v1.md;
  kairyu/runners/startup_binding_live_kubernetes.py;
  tests/unit/test_runner_startup_binding_live_authority.py

### 2026-09-30 — [progress] Deadline-bounded PostgreSQL live readers
- What: added the D3.15 current-binding and exact durable-decision adapter plus
  transaction-local PostgreSQL statement/lock budgets and readiness reads.
- Why: D3.12 must consume shared durable authority without treating missing
  rows as generic outages or allowing database waits to ignore its deadline.
- Refs: docs/design/node-model-cache-prestage-v1.md;
  kairyu/runners/startup_binding_live_postgres.py;
  tests/unit/test_runner_startup_binding_live_authority.py

### 2026-09-30 — [progress] Authenticated node-evidence aggregation
- What: added the D3.14 strict HTTPS node-agent client and bounded controller
  fan-out adapter that joins exact owner-scoped evidence only to placement
  inventory observed after the node reads.
- Why: live reauthorization must not trust binding-controlled destinations,
  unbounded or ambiguous wire data, or reuse decision-time node health and
  assignment facts as if they were current.
- Refs: docs/design/node-model-cache-prestage-v1.md;
  kairyu/runners/startup_binding_live_cache.py;
  tests/unit/test_runner_startup_binding_live_authority.py

### 2026-09-30 — [progress] Owner-scoped node live-evidence transport
- What: added the D3.13 authenticated node-agent endpoint that joins a stable
  ready pre-stage record to one cache-index snapshot and returns only path-free
  physical hint and exact current pin-owner/generation evidence.
- Why: D3.11 must prove that the binding's owner still pins the exact resident
  generation without exposing node-local artifact paths or trusting a generic
  `pinned=true` flag that permits different-owner ABA.
- Refs: docs/design/node-model-cache-prestage-v1.md;
  kairyu/runners/cache_agent_live_evidence.py;
  tests/unit/test_node_model_cache_agent_api.py

### 2026-09-30 — [progress] Deadline-bounded live-state source composition
- What: added the D3.12 source that composes current binding, durable decision,
  target, quota, and coherent cache/pin evidence, rereads the binding as a
  replacement fence, and budgets every backend/readiness call by one deadline.
- Why: D3.11's validator needs an executable integration seam that cannot mix
  evidence across a superseded binding or let per-backend timeouts exceed the
  authority request budget.
- Refs: docs/design/node-model-cache-prestage-v1.md;
  kairyu/runners/startup_binding_live_source.py;
  tests/unit/test_runner_startup_binding_live_authority.py

### 2026-09-30 — [progress] Leader-fenced placement-binding authorization
- What: added the D3.11 scaling-controller adapter that rechecks leadership and
  validates exact live binding, durable decision/target, quota, prewarm/cache
  freshness, current pre-stage/pin/resident lineage, lifetime, and
  deadline-bounded dependency readiness.
- Why: the D3.10 service must not authorize from its admission-store echo or a
  stale callback after controller leadership or scale-up capacity changes.
- Refs: docs/design/node-model-cache-prestage-v1.md;
  kairyu/runners/startup_binding_live_authority.py;
  tests/unit/test_runner_startup_binding_live_authority.py

### 2026-09-30 — [progress] Placement-binding authority runtime assembly
- What: added the embedded D3.10 authority runtime with a strict versioned
  config, bounded regular-file loading for bearer and TLS material, secret-file
  permission checks, cross-service timeout headroom, app assembly, failed-build
  cleanup, and idempotent shutdown of adopted live-authority resources.
- Why: the D3.9 protocol boundary needs a fail-closed process assembly contract
  without inventing a second source of scaling truth or dynamically importing
  privileged callbacks.
- Refs: docs/design/node-model-cache-prestage-v1.md;
  kairyu/runners/startup_binding_authority_runtime.py;
  tests/unit/test_runner_startup_binding_authority_runtime.py
