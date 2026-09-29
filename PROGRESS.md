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
- Model-cache WP4.1–WP4.7 and D3.1–D3.10 provide signed identity, verified cache,
  fenced pre-stage/startup/admission, PostgreSQL authority, and TLS HTTP runtime.
  Live controller callbacks, scheduled compaction, deployment, and acceptance remain.
- Qwen3.8 MTP stays disabled pending vllm#53912; DTO-D15 verification/re-pin and
  human sign-off for M2–M4 remain pending.

## Change Log

Newest first; only the most recent entries are kept here (see the size budget
in `.claude/rules/progress-log.md`).

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

### 2026-09-29 — [progress] Live placement-binding authority API
- What: added the authenticated D3.9 scaling-authority HTTP boundary with strict
  nonce-bound request/response models, exact live-binding reauthorization,
  bounded body/concurrency, protected empty-body readiness, no-store responses,
  sanitized stale/dependency failures, and deadline-aware backend contracts.
- Why: the D3.8 admission runtime must revalidate its stored binding against a
  live authority over a concrete fail-closed service boundary before consuming
  placement capacity.
- Refs: docs/design/node-model-cache-prestage-v1.md;
  kairyu/runners/startup_binding_authority{,_api}.py;
  tests/unit/test_runner_startup_binding_authority_api.py

### 2026-09-29 — [progress] Executable placement webhook runtime
- What: added `kairyu placement-admission serve` with versioned strict config,
  TLS certificate/key handling, D3.6 PostgreSQL store assembly, bounded
  concurrency/readiness, and an authenticated HTTPS binding-authority client
  using fresh nonce echo, full binding validation, bounded strict JSON, and
  no redirects or environment proxy inheritance.
- Why: the protocol-correct D3.7 app needs a production process boundary, and
  admission must re-check live scaling authority rather than treating a stored
  plan as permanent authorization.
- Refs: docs/design/node-model-cache-prestage-v1.md;
  kairyu/runners/startup_admission_runtime.py;
  tests/unit/test_runner_startup_admission_runtime.py

### 2026-09-29 — [progress] Kubernetes placement admission transport
- What: added the D3.7 AdmissionReview v1 HTTP boundary for Pod CREATE,
  including strict JSON/schema limits, request-to-Pod identity checks,
  server-clock observation, UID-bound allow/deny responses, deterministic RFC
  6902 patches, replay handling, readiness, concurrency bounds, and sanitized
  failure responses.
- Why: the D3.4 controller and D3.6 shared store need a protocol-correct,
  fail-closed API-server boundary before production webhook runtime and IaC can
  safely consume them.
- Refs: docs/design/node-model-cache-prestage-v1.md;
  kairyu/runners/startup_admission_api.py;
  tests/unit/test_runner_startup_admission_api.py

### 2026-09-29 — [progress] Shared incremental placement admission state
- What: added a strict PostgreSQL plan/claim store for D3.4 with cross-replica
  serialization, unique placement constraints, same-name replay protection,
  bounded target capacity, durable replay-window configuration, validate-only
  startup, authoritative database-clock expiry/safety gates, and canonical
  full-plan JSON corruption checks.
- Why: a highly available admission webhook must not allocate one cache
  placement to two Pods or let a failed creator request undo a claim already
  observed by an API-server retry.
- Refs: docs/design/node-model-cache-prestage-v1.md;
  kairyu/runners/postgres_startup_admission.py;
  tests/unit/test_postgres_runner_startup_admission.py
