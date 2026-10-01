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

Snapshot date: 2026-09-30. Hardware context: all GPU evidence so far is on
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

- G2 A1: complete — TP1/2 logprob-agreement closure on Llama-3.1-8B
- G2 A2: complete — TP2/4/8 closure on Llama-3.3-70B FP8-dynamic
- G2 A6 (perf vs vLLM): **open** — TP4 ShareGPT 0.466× SLO-goodput HTTP; matrix deferred while gap closes
- G2 A7: closed — >80% KV cache-hit rates on Qwen3-32B TP4/TP8, direct and gateway
- G2 A8 (DP scaling): `passed: false` (1.7993× vs 1.9× threshold); owner accepted as explicit closure deviation
- G2 A9: closed — DP=2×TP4 vs TP8 production-topology report on Qwen3-32B
- A12 (batch-invariance determinism, #360): closed — exact-match verdict passed on Qwen3-32B TP8
- #356 real-checkpoint quant parity: evidence complete — INT8 PASS; AWQ/GPTQ formal FAIL retained with SHA-bound same-GPU oracle replay isolating checkpoint quantization loss
- B7 (KV answer-equivalence, #373): operator implemented and portable-validated; additive over F2/F4
- G4 MoE: M-A1 formal FAIL retained; M-A2 complete; M-A3 scope-closed by owner deviation (perf gate stays FAIL); dense BF16 MoE uses sort-by-expert grouped GEMM and fixed-capacity EP transport, then combines returned rows in fixed FP32 order before one model-dtype cast
- G4 E-KV: unit-scale and calibrated per-layer K/V FP8-E4M3 re-bakes **FAIL** retained; calibrated cache metrics/logprobs pass but 16K/32K exact tokens and decode envelope do not; `fp8_e4m3` startup rejected
- G5: F1a–F1d, F2a–F2d, F4a, F4b all closed; F4c decided (keep per-replica RadixKV + F2 routing, thresholded revisit)
- F5a/b/c (priority, noisy-neighbor, SLO admission): closed
- G6: P-A, P-B1–P-B4, P-C2/C3/C4 green (incl. Open WebUI P-B3 browser gate); remaining P-C gates continue
- #150 TP8 long-generation stability gate: passed after deadlock fix; #364 `logits_dtype`: valid negative, withdrawn

### What works today

- `kairyu serve --tp N` on real hardware: Qwen3-32B TP8, Llama-3.1-8B, Llama-3.3-70B FP8, Qwen3-VL-32B (via vLLM replica)
- Attention backends: `auto`/torch/FlashInfer/FA3/FA4 with `/backends` reporting; capable CUDA models pre-capture decode graphs before readiness
- Quantized serving: FP8/INT8/AWQ/GPTQ/NVFP4 without full dequantization; opt-in FP8 EAGLE/MTP draft loading
- Incremental architecture-state paths for Qwen3.6 and DeepSeek V4 plus an explicit recompute diagnostic mode; DeepSeek EP2/4/8 Attention-DP and direct packed-FP4 execution are implemented, with SM120 single-kernel and two-rank NCCL smokes green
- Device-side sampling, penalties, spec verification, page-table caching; TP step headers sleep on Gloo while fixed-layout delta payloads use the bounded NCCL model group and rare controls remain Gloo objects; structured masks stay on CUDA with only selected IDs returned to the host matcher; deterministic n-gram/EAGLE-3/MTP drafts preserve T>0 and penalized sampling
- Hardened gateway: auth, tenancy metering/invoicing, priority + SLO admission, batch API, embeddings/RAG, Responses API; System One (Jev wire API) at `/v1/systemone` outside ReplicaPool (m11 D8), GPU-verified with OpenJev in `openjev-diffusiongemma-26b-1gpu` (512-token think-first chat, Jev-style playground)
- Orchestration (Conductor/MoA) with streaming, usage accounting, trace v2; assistant history round-trips typed `reasoning_content` while assistant-only LiteLLM provider objects and nullable legacy function calls are ignored before rendering and other extras remain fail-closed; MoA keeps the original response contract distinct from untrusted candidate drafts, with configured completion delimiters and the multi-stage boundary withholding private synthesis reasoning; prefix-aware replica placement obeys the configured queue-depth overload valve; Codex CLI and IDE tool-calling work end-to-end, including AUTO models over /v1/responses (#530)
- Fleet: 3-gateway HA with PostgreSQL BatchStore, KV-aware prefix routing, DRAM KV tiering; Helm supports immutable images, split-role labels, safe rollout/drain, hardened Pods, and ServiceMonitor plus the kind CI drill
- AsyncRequest v1 (opt-in `async_requests`): PostgreSQL-backed non-streaming Chat with tenant-scoped status/result/cancel, lease-fenced workers, shared queue telemetry and bounded request/audit retention; the retention-inclusive three-gateway Kind gate runs in F1c CI. Runner State v1 contracts (Kubernetes observation/reconciliation, fenced drain, failure-domain backoff, PostgreSQL leader lease, WP3.1 scaling policies, WP3.2 decision log) exist as a library
- Checkout-only eval tooling retains explicit Core, Quantization, Structured Output, and Long Context suites with hash-chained quality history, config A/B comparisons, and quantization sweeps; Kairyu correctness and performance gates are owned by `verification/`, not evals
- The tiered RTX PRO example (DTO-D13, 2026-08-22) puts a bounded Qwen non-thinking route judge in front of five profiles — four single-call direct routes (Qwen non-thinking, Qwen thinking-medium, DeepSeek non-thinking on the re-added `tier2-direct` pool, DeepSeek thinking at the L3 effort; official per-mode sampling fixed on the final unit, vendor-official caps 131072/393216) and the ensemble — selecting per request with fallback to the ensemble; the L2 DSL now has N named `profiles` + a judge with spec-defined `choices`, final-unit sampling overrides (caps min()'d with the caller), and route-aware serving gates. The ensemble (`primary`) profile is the dual-track policy-ensemble L2 DAG (DTO-D1..D12, amended by DTO-D14) over four Qwen3.8 TP1 vLLM workers (no MTP pending c16/c32 evidence) + the measured DeepSeek TP4/EP4 DSpark worker: a Qwen head streams the public opening from t=0 (semantic-TTFT gate ≤2× DeepSeek-direct, inherited); one thinking DeepSeek call writes 4 maximally different policies fanned out to 4 policy-bound Qwen answers in parallel while thinking DeepSeek critically refines a quick Qwen draft; thinking DeepSeek `synthesis` weighs the 5 candidates as peers and writes one better answer, and an inline Qwen thinking-medium (DTO-D14) `audit` (PASS/FAIL, ≤2 refinements, last attempt published on exhaustion) gates the streamed remainder (DTO-D10); a Qwen `image_description` stage runs on image requests only and feeds the text-only DeepSeek roles (DTO-D11); DeepSeek budgets halved to 8192/32768/65536 with a 65536 ceiling and Chat UI default for the Terminal-Bench 900 s turn envelope (DTO-D12). The sandbox executor stays deployed but unreferenced. Last green verify.sh runs 20260825T161729Z (coding) and 20260825T173343Z (generic) on the DTO-D8..D14 served config: coding TTFT rows all not_applicable (the judge routes every coding request to the ungated qwen_think_medium route), generic route-aware stage validation green. Composed L1 workers remain vLLM-backed until the native full-checkpoint gate closes
- Replica-pool scale-out examples (FN-D9, 2026-09-01): Qwen3.8 TP1 x 8 and DeepSeek TP4+EP4 x 2 behind one public model each; `verify.sh serving` proves the even per-replica split from the pool placement log and `verify.sh tool-calling` proves OpenAI tool calls on every replica (see their MEASUREMENTS.md); two vision replica examples (FN-D9 amendment 2026-09-04: DeepSeek-V4-Flash-Vision-Exp TP4+EP4 x 2, Qwen3.8-Flash-Next-FP8 TP4 x 2 on a shared upstream-main SM120 overlay image, Chat UI reasoning-effort dropdown, `verify.sh vision`) are GPU-verified (2026-09-04: pins locked, serving/tool-calling/vision gates PASS, MEASUREMENTS.md written); the Qwen example serves without the recipe's MTP k=3 because prefix caching + MTP corrupts batched output on this vLLM revision (vllm#53912)
- DeepSeek V4.1 Flash single-replica example (FN-D9 amendment, 2026-09-11) is GPU-verified on TP8/EP8 SM120 with the V4 ReplicaPool/API/UI structure and official thinking-high default; bounded L1 comparisons select DSpark 5, 16K batching and NCCL. The 320-request matrix, reasoning/tool/vision/cancellation, normal restart and retrieval through 1,039,909 prompt tokens pass; exact evidence and limitations are in its `MEASUREMENTS.md`.
- DeepSeek V4.1 Flash six-GPU example (FN-D9 six-GPU amendment, 2026-09-30): one DP6/EP6 replica on GPUs 0–5 with the 8-GPU example's L2/L3 structure and its own scripts; official-first L1 (pinned vLLM nightly + SM120 overlay, Engram offload, 4K batch / 0.92 from the recipe's memory-bound arm, DSpark 5 with full verification). Serving 102 / 591 / 718 tok/s at c1/c32/c64; gate evidence in its `MEASUREMENTS.md`.
- Qwen3.8 + DeepSeek-V4.1 ensemble example (DTO-D16/D17, 2026-10-01): V4.1 DP6/EP6 (GPU 0–5, the six-GPU example's L1) + Qwen TP1 × 2 (GPU 6, 7). A Qwen judge picks one of two routes: thinking DeepSeek, or the dual-track ensemble with two policies and a three-candidate synthesis. Every role takes images natively; DeepSeek uses the official V4.1 encoder with per-request effort, and the example's overlay continues the floor's assistant prefill. ENSEMBLE criteria loosened and judge fallback moved to `deepseek_think` (DTO-D17 amendments). GPU-verified 2026-10-01: all gates pass, including the ensemble TTFT gate at c1–c32 (c32 at 95.8 % of the limit); 4 of 128 coding requests exceed the 900 s turn envelope after two audit refinements.
- Process-split backend (`kairyu-proc`) with delta wire, TP group attestation, graceful lifecycle
- CPU suite green (thousands of tests, no selected skips); CPU microbenchmark smoke + nightly regression series in CI

### Open items / blockers

- G2 A6 performance gap vs vLLM is the open hard gate; full TP4/8 HTTP matrix deferred until closed
- Issue #333 verdict: process-split is not the A6 cause (`no_material_reduction`, ratio 0.92 vs ≤0.90 line)
- Issue #318 verdict: depth beyond the two-step admission horizon is not an A6 fix (`no_measured_benefit_depth_gt_2`)
- Production stage-sharded pipeline parallelism is a separate roadmap dependency (current PP report is not it)
- Learned-draft real-checkpoint acceptance/performance evidence remains open; FP8-E4M3 KV remains disabled after its calibrated re-bake failed exact-output and decode-envelope checks
- Frontier full-checkpoint 262K/1M correctness/performance evidence, DeepSeek EP4/EP8 topology lock, CUDA Graph pointer stability, MTP/DSpark selection, 30-minute soak, and failure recovery remain open
- NVLink-profile gates blocked on H100/A100-class hardware; PCIe-switch chassis and ≥400 Gb/s RDMA NICs gate E4/E5
- G6 remaining P-C gates still in progress
- Runner control plane: scale actuation (WP3.3), leader-token propagation to mutations (WP3.4), deployment wiring, durable Runner status, Kubernetes mutations and runtime instrumentation remain open; `kairyu/runners` has no serving caller yet
- Qwen3.8-Flash-Next MTP speculative decoding stays off in `qwen3.8-flash-next-dp2-8gpu` until upstream fixes vllm#53912 (prefix caching + MTP output corruption on hybrid GDN); single-stream decode 104 vs 175 tok/s
- DTO-D15 (2026-08-26) changed the served tiered-example config: verify.sh coding/generic gates and the digest re-pin are pending before the example status can be claimed green again
- Human sign-off pending on M2–M4 design reviews

## Change Log

Newest first; only the most recent entries are kept here (see the size budget
in `.claude/rules/progress-log.md`).

### 2026-10-01 — [amendment] System One review fixes (PR #614)
- What: `/v1/systemone` reserves the billed upper bound (questions as separate reads, think × samples), enforces each model's body limit, 502s unless both usage counts are valid, and tenant 429s use Jev's shape. The OpenJev overlay refuses empty `stop`/out-of-range `top_logprobs` before the thought; preflight exempts only the GPU the L1 holds; playground fixes. Gates pass on overlay `46530fa7`; `l1` thought-cut-then-answer fails intermittently (2/4, answer pass reopens a thought) — open finding.
- Why: owner review: the old bound let one request bill 13× a tenant's bucket, and answer-only refusals after the thought ejected the only replica.
- Refs: m11 D8 metering; example `MEASUREMENTS.md` "Review-fix rerun"

### 2026-10-01 — [design] System One API through Kairyu; OpenJev example GPU-verified
- What: Kairyu serves `POST /v1/systemone` (Jev wire API) via `HTTPSystemOneBackend`, not a pool member; `/v1/models` adds Jev's `models` list. The OpenJev example serves System One through Kairyu with a Jev-style playground, fixes the prefill template for vLLM's `openai` content format, pins the overlay, adopts 32 generations in flight + 8 queued (+32 % c32 tok/s), and passes every GPU gate including OpenJev's own live suite against Kairyu.
- Why: owner request (Web UI following Jev, served by Kairyu). System One is a public wire format with several servers, so auth/tenancy/metering/admission belong in Kairyu; a pool member would let a System One 529 eject the chat replica.
- Refs: m11 D8 (`docs/design/m11-product.md`); FN-D9 OpenJev amendment; example `MEASUREMENTS.md`; PR #614

### 2026-10-01 — [design] OpenJev DiffusionGemma on one GPU, think = 512
- What: new example `openjev-diffusiongemma-26b-1gpu`: one OpenJev replica (DiffusionGemma 26B-A4B NVFP4 on vLLM) behind the single-replica L2/L3. Every chat completion thinks first with a fixed 512-token thought, through an example-owned two-pass overlay on the published OpenJev image. `kairyu/` is unchanged. CPU tests and CPU evidence pass; GPU gates are pending.
- Why: owner request. DiffusionGemma's `DiffusionSampler` does not apply vLLM's `thinking_token_budget`, and OpenJev's chat route has no budget, so OpenJev's own System One `think` method is applied to chat.
- Refs: FN-D9 OpenJev one-GPU amendment in `docs/design/frontier-native-runtime.md`; plan `docs/superpowers/plans/2026-10-01-openjev-diffusiongemma-1gpu-example.md`; example `MEASUREMENTS.md`

### 2026-10-01 — [progress] V4.1 ensemble example GPU gates pass on the amended DTO-D17
- What: readiness, vision, tool-calling, generic and coding matrices, and the browser smoke pass. Ensemble TTFT gate PASS at c1/c8/c16/c32 (8.7/7.7/39.6/95.8 % of 2× direct). Judge timeouts 3/269, all served by `deepseek_think`. 4/128 coding requests exceed 900 s after two audit refinements.
- Refs: example `MEASUREMENTS.md` (runs `fb-*`); DTO-D17 in `docs/design/example-dual-track-orchestration.md`; PR #613

### 2026-10-01 — [amendment] V4.1 ensemble example: judge fallback is deepseek_think (DTO-D17)
- What: a judge timeout, backend error, or unparseable verdict now routes to `deepseek_think`, not the ensemble (`profile_judge.fallback`). `kairyu/` is unchanged. Every GPU gate is re-run.
- Why: owner decision. A slow or failed judge says nothing about difficulty; escalating to the heavier route on a 5 s timeout added load when the system was busiest.
- Refs: DTO-D17 second amendment in `docs/design/example-dual-track-orchestration.md`; PR #613

### 2026-10-01 — [amendment] V4.1 ensemble example: looser ENSEMBLE criteria (DTO-D17)
- What: the judge now sends hard problems and ties to ENSEMBLE (math/logic, algorithms, multi-file coding, proofs, design/planning, multi-step analysis, ambiguous requests). DEEPSEEK_THINK keeps everyday requests and routine agent turns. The example's GPU gates are re-run.
- Why: owner request. The first DTO-D17 criteria sent 100% of the gate traffic and 44/44 live benchmark requests to deepseek_think.
- Refs: DTO-D17 amendment in `docs/design/example-dual-track-orchestration.md`; PR #613

### 2026-10-01 — [design] V4.1 ensemble example: two routes, two policies (DTO-D17)
- What: `qwen3.8-deepseek-v4.1-8gpu` only. The judge offers DEEPSEEK_THINK and ENSEMBLE, and the other three routes are removed. `policies` writes two policies for `answer_1`/`answer_2`, one per Qwen replica. `synthesis` merges three candidates. Budget `{16, 2}`; the TTFT gate applies to the ensemble only. The V4 example and `kairyu/` are unchanged.
- Why: owner decision. On five routes and four policies the Qwen pair was the bottleneck (c32 Qwen-route TTFT p50 43.9 s). The earlier GPU results are kept as superseded in the example's MEASUREMENTS.md.
- Refs: DTO-D17 in `docs/design/example-dual-track-orchestration.md`; PR #613

### 2026-09-30 — [design] Qwen3.8 + DeepSeek-V4.1 six-GPU ensemble example (DTO-D16)
- What: new example `qwen3.8-deepseek-v4.1-8gpu`. It keeps the V4 ensemble method (judge + 5 routes, dual-track DAG, audit, budgets) on V4.1 DP6/EP6 (GPU 0–5) + Qwen TP1 × 2 (GPU 6, 7). `image_description` is removed; budget `{18, 2}`. DeepSeek roles use the official encoder via chat requests with one pool: `reasoning_effort` for thinking, `enable_thinking: false` for chat mode, no server-wide thinking default. Overlay edit 7 makes the V4.1 encoder continue a final `<think>` prefill (DTO-D9/D15 floor). `kairyu/` is unchanged.
- Why: owner request — V4.1 takes images natively, so the Qwen description proxy is unnecessary. The encoder ignored `continue_final_message` and ORs `thinking`/`enable_thinking` (probed on the live six-GPU L1).
- Refs: DTO-D16 in `docs/design/example-dual-track-orchestration.md`; plan `docs/superpowers/plans/2026-09-30-qwen38-deepseek-v41-8gpu-example.md`; example `MEASUREMENTS.md`

### 2026-09-30 — [design] DeepSeek V4.1 Flash on six GPUs
- What: new example `deepseek-v4.1-flash-6gpu` (GPUs 0–5, same L2/L3 as the 8-GPU example, all scripts example-owned). Bounded one-parameter comparisons select DP6/EP6 over the official TP2 degree (c32 +44–47 %) and DSpark with full verification plus the recipe's 4K / 0.92 memory levers (c1 +77 %, c32 +20 %).
- Why: 6 × 96 GB is below the checkpoint's 614 GB official minimum; the recipe's DEP kernels, `indexer_sparse_logits` and adaptive verification do not run on SM120 with this runtime; masked sparse-KV rows need a zero row on EP6.
- Refs: FN-D9 six-GPU amendment in `docs/design/frontier-native-runtime.md`; plan `docs/superpowers/plans/2026-09-30-deepseek-v41-flash-6gpu-example.md`; example `MEASUREMENTS.md`

### 2026-09-30 — [amendment] m10 A39: AsyncRequest v1 pre-merge review fixes
- What: an unstorable result publishes a fenced `result_persistence_failed` error after one retry; the third lease expiry fails a request with `lease_expired` instead of re-running it (defers and releases do not count); heartbeats retry transient renewal errors until the lease would expire; shutdown returns unfinished claims to the queue at once (zero-delay defer, no tenant cooldown); `/metrics` collectors render on the event loop with only the blocking store warmup off-loop; the capacity 429 omits `Retry-After` when request retention is off, and startup warns.
- Why: PR #604 review — each replaced path re-ran inference (without bound for an unstorable result), read unlocked `ReplicaPool` state from a second thread, or promised a retry that could not succeed.
- Refs: `docs/design/m10-fleet-cpu.md` A39 (supersedes A38's shutdown sentence); PR #604; `kairyu/async_requests/{worker,store,postgres_store}.py`, `kairyu/entrypoints/server/{metrics,health}.py`
