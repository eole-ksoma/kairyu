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

Snapshot date: 2026-08-17. Hardware context: all GPU evidence so far is on
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
- Hardened gateway: auth, tenancy metering/invoicing, priority + SLO admission, batch API, embeddings/RAG, Responses API
- Orchestration (Conductor/MoA) with streaming, usage accounting, trace v2; assistant history round-trips typed `reasoning_content` while assistant-only LiteLLM provider objects and nullable legacy function calls are ignored before rendering and other extras remain fail-closed; MoA keeps the original response contract distinct from untrusted candidate drafts, with configured completion delimiters and the multi-stage boundary withholding private synthesis reasoning; prefix-aware replica placement obeys the configured queue-depth overload valve; Codex CLI and IDE tool-calling work end-to-end, including AUTO models over /v1/responses (#530)
- Fleet: 3-gateway HA with PostgreSQL BatchStore, KV-aware prefix routing, DRAM KV tiering, Helm chart + kind CI drill
- Checkout-only eval tooling retains explicit Core, Quantization, Structured Output, and Long Context suites with hash-chained quality history, config A/B comparisons, and quantization sweeps; Kairyu correctness and performance gates are owned by `verification/`, not evals
- The tiered RTX PRO example (DTO-D13, 2026-08-22) puts a bounded Qwen non-thinking route judge in front of five profiles — four single-call direct routes (Qwen non-thinking, Qwen thinking-medium, DeepSeek non-thinking on the re-added `tier2-direct` pool, DeepSeek thinking at the L3 effort; official per-mode sampling fixed on the final unit, vendor-official caps 131072/393216) and the ensemble — selecting per request with fallback to the ensemble; the L2 DSL now has N named `profiles` + a judge with spec-defined `choices`, final-unit sampling overrides (caps min()'d with the caller), and route-aware serving gates. The ensemble (`primary`) profile is the dual-track policy-ensemble L2 DAG (DTO-D1..D12, amended by DTO-D14) over four Qwen3.8 TP1 vLLM workers (no MTP pending c16/c32 evidence) + the measured DeepSeek TP4/EP4 DSpark worker: a Qwen head streams the public opening from t=0 (semantic-TTFT gate ≤2× DeepSeek-direct, inherited); one thinking DeepSeek call writes 4 maximally different policies fanned out to 4 policy-bound Qwen answers in parallel while thinking DeepSeek critically refines a quick Qwen draft; thinking DeepSeek `synthesis` weighs the 5 candidates as peers and writes one better answer, and an inline Qwen thinking-medium (DTO-D14) `audit` (PASS/FAIL, ≤2 refinements, last attempt published on exhaustion) gates the streamed remainder (DTO-D10); a Qwen `image_description` stage runs on image requests only and feeds the text-only DeepSeek roles (DTO-D11); DeepSeek budgets halved to 8192/32768/65536 with a 65536 ceiling and Chat UI default for the Terminal-Bench 900 s turn envelope (DTO-D12). The sandbox executor stays deployed but unreferenced. Last green verify.sh runs 20260825T161729Z (coding) and 20260825T173343Z (generic) on the DTO-D8..D14 served config: coding TTFT rows all not_applicable (the judge routes every coding request to the ungated qwen_think_medium route), generic route-aware stage validation green. Composed L1 workers remain vLLM-backed until the native full-checkpoint gate closes
- Replica-pool scale-out examples (FN-D9, 2026-09-01): Qwen3.8 TP1 x 8 and DeepSeek TP4+EP4 x 2 behind one public model each; `verify.sh serving` proves the even per-replica split from the pool placement log and `verify.sh tool-calling` proves OpenAI tool calls on every replica (see their MEASUREMENTS.md); two vision replica examples (FN-D9 amendment 2026-09-04: DeepSeek-V4-Flash-Vision-Exp TP4+EP4 x 2, Qwen3.8-Flash-Next-FP8 TP4 x 2 on a shared upstream-main SM120 overlay image, Chat UI reasoning-effort dropdown, `verify.sh vision`) are GPU-verified (2026-09-04: pins locked, serving/tool-calling/vision gates PASS, MEASUREMENTS.md written); the Qwen example serves without the recipe's MTP k=3 because prefix caching + MTP corrupts batched output on this vLLM revision (vllm#53912)
- DeepSeek V4.1 Flash single-replica example (FN-D9 amendment, 2026-09-11) is GPU-verified on TP8/EP8 SM120 with the V4 ReplicaPool/API/UI structure and official thinking-high default; bounded L1 comparisons select DSpark 5, 16K batching and NCCL. The 320-request matrix, reasoning/tool/vision/cancellation, normal restart and retrieval through 1,039,909 prompt tokens pass; exact evidence and limitations are in its `MEASUREMENTS.md`.
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
- Qwen3.8-Flash-Next MTP speculative decoding stays off in `qwen3.8-flash-next-dp2-8gpu` until upstream fixes vllm#53912 (prefix caching + MTP output corruption on hybrid GDN); single-stream decode 104 vs 175 tok/s
- DTO-D15 (2026-08-26) changed the served tiered-example config: verify.sh coding/generic gates and the digest re-pin are pending before the example status can be claimed green again
- Human sign-off pending on M2–M4 design reviews

## Change Log

Newest first; only the most recent entries are kept here (see the size budget
in `.claude/rules/progress-log.md`).

### 2026-09-11 — [progress] V4.1 L1 selection and final GPU gates complete
- What: select TP8/EP8, DSpark 5, 16K batching and NCCL; the 320-request matrix, default/explicit reasoning, tools, images, cancellation, normal restart and four long-context retrieval smokes pass. Best measured aggregate throughput is 326.82 tok/s at c32; near-1M retrieval completes in 203.02 s.
- Why: DSpark improves c1 throughput 1.91×; EP-off exhausts KV memory at the same limits, PCIe IPC stalls during autotuning, and 8K batching shows no throughput gain. Keep unmeasured alternatives and broad quality claims outside this evidence.
- Refs: PR #597; FN-D9 V4.1 amendment; example `MEASUREMENTS.md` records exact configuration, run IDs, hashes and limitations.

### 2026-09-11 — [progress] V4.1 full-model API gates pass on TP8
- What: the SM120 overlay starts all eight GPUs, captures graphs and serves default/low/high/max reasoning, tools, images and cancellation; all initial API gates pass. UI effort selection uses the existing top-level L3 field. Performance selection and final context/restart gates remain pending.
- Why: the experimental off toggle used template kwargs rejected by the unchanged legacy L3; retaining V4's effort vocabulary keeps the requested L2/L3 structure.
- Refs: PR #597; example `MEASUREMENTS.md` initial runs `20260911T032048Z` through `20260911T032052Z`.

### 2026-09-11 — [amendment] V4.1 indexer requires 64-token blocks and MXFP4 on SM120
- What: correct the preceding 128-token manager-block candidate to 64/BLHNC, with SWA=64, C1=64, C2=32. Enable the existing MXFP4 indexer only for V4.1 on SM120. All 16 sparse-attention and four real indexer writer/prefill/decode numerical cases pass; full-model serving remains pending.
- Why: DeepGEMM rejects C1 pages of 128 and SM120 FP8 C2 pages of 32; its MXFP4 path supports both required sizes. The indexer oracle independently unpacks actual Q/K bytes (max error 2.4e-7), and CPU guards retain rejection for unverified model/device combinations.
- Refs: PR #597; FN-D9 V4.1 amendment; example `MEASUREMENTS.md`, `check_sm120_pages.py`, `check_sm120_indexer.py`. Supersedes the block-size choice in the preceding SM120 cache-compatibility entry.

### 2026-09-11 — [progress] V4.1 SM120 cache compatibility
- What: pin an example-local L1 overlay with 64-token SWA pages and C1 128-token dual-cache prefill instantiations; use manager blocks 128/BLHNC and disable unsupported adaptive verification. All 16 packed-cache GPU numerical cases pass at upstream DSV4 tolerances; full-model serving and tuning remain pending.
- Why: the official V4.1 image's SWA pages and indexer layout assumptions fail startup on SM120 before serving. Source-anchored adaptations retain the existing kernel arithmetic and keep L2/L3 unchanged.
- Refs: PR #597; `examples/deepseek-v4.1-flash-8gpu/{patch_runtime.py,check_sm120_pages.py,MEASUREMENTS.md}`; FN-D9 V4.1 amendment.

### 2026-09-11 — [amendment] FN-D9: V4.1 Flash on one eight-GPU replica
- What: add a separate V4.1 example with the existing V4 vision ReplicaPool/API/UI path; default thinking is the official high (75). Pin the checkpoint manifest and isolate runtime encoder alignment. Fixed-token measurements distinguish model output from visible content; completed-answer gates stay separate. CPU contracts pass; GPU selection is pending.
- Why: the owner revised the initial two-replica request to one TP8 replica; the initial vLLM encoder maps high differently from the checkpoint, and content-only timing mismeasures all-reasoning output.
- Refs: FN-D9 amendment in `docs/design/frontier-native-runtime.md`; `examples/deepseek-v4.1-flash-8gpu/`; implementation plan `2026-09-11-deepseek-v41-flash-example.md`.

### 2026-09-04 — [amendment] FN-D9: vision examples GPU-verified; Qwen drops MTP k=3
- What: both vision replica examples pinned (tree SHA, image ID `b47e2210`) and all gates
  PASS — DeepSeek c64 689 tok/s, Qwen c32 548 tok/s, placement 32/32 at every row ≥c8,
  tool-calling 6/6, vision 2/2. `qwen3.8-flash-next-dp2-8gpu` now serves without the
  recipe's `--speculative-config mtp k=3` and with `--kv-cache-memory` pinned (a cold
  torch.compile cache made vLLM's start-up profile shrink replica 0's KV cache to 741K
  tokens vs 3.45M); `verify.sh vision` requires the answer to name the probe colour.
- Why: with prefix caching + MTP, `vllm@27a94d1c` corrupts batched answers on the hybrid
  GDN checkpoint (`ductduct…`; 13/274 at 2-12 concurrent, 0/1,508 with either off,
  63.8% with `--no-async-scheduling`; upstream vllm#53912). Prefix caching is what
  Kairyu's prefix-aware placement and multi-turn traffic use, so MTP is the one dropped.
- Refs: FN-D9 amendment in `docs/design/frontier-native-runtime.md`; `examples/*/MEASUREMENTS.md`; supersedes the "MTP k=3" wording in the 2026-09-04 FN-D9 entry below

### 2026-09-04 — [amendment] FN-D9: two vision replica-pool examples
- What: `examples/deepseek-v4-flash-vision-exp-dp2-8gpu` (TP4+EP4 × 2, official recipe
  + SM120 marlin, 1M ctx) and `examples/qwen3.8-flash-next-dp2-8gpu` (TP4 × 2, official
  rtx_pro_6000_4x FP8 layout, MTP k=3, 256K ctx): one public text+image model each,
  no-login Chat UI with a reasoning-effort dropdown in each model's official vocabulary,
  `verify.sh vision` gate, shared upstream-main `27a94d1c` + FlashInfer `60b49158`
  SM120 overlay image with a fail-closed image-ID pin.
- Why: both checkpoints need upstream `main` (official tags predate the support PRs;
  FlashInfer 0.6.18 breaks SM120 sparse-MLA on the first image); Qwen's template
  rejects L3-normalized efforts, so an example-local alias restores them.
- Refs: FN-D9 amendment 2026-09-04 (docs/design/frontier-native-runtime.md); tests/unit/test_replica_examplectl.py; GPU evidence pending

### 2026-09-02 — [amendment] FN-D9: replica examples must serve OpenAI tool calls
- What: DP2 DeepSeek drops the Kairyu-rendered /completions passthrough (forwards
  no tools; DSML parse is whole-block only) for the Qwen-style legacy path: vLLM
  renders with the checkpoint's deepseek_v4 encoder + `--tool-call-parser
  deepseek_v4`, Kairyu forwards tools to /chat/completions and normalizes. Both
  examples gain a fail-closed readiness tool probe + `verify.sh tool-calling`
  and non-thinking default kwargs (Qwen gate caught empty `content` on plain chat).
- Why: PR #584 review — SWE-bench Pro got `tool_calls: null` every turn (22/22
  RepeatedFormatError). GPU-verified: both tool gates 6/6, both matrices
  re-pinned, SWE-bench Pro smoke 3/3 (kairyu-bench `20260902T010540Z-3bf671e8`).
- Refs: PR #584; FN-D9 amendment; examples/{qwen3.8-27b-dp8-8gpu,deepseek-v4-flash-0731-dp2-8gpu}/

### 2026-09-02 — [amendment] FN-D9: replica placement gates reject material skew
- What: both replica-pool examples now limit a replica to 1.25× the even share.
  Behavior tests reject the 8-way `16,16,16,8,2,2,2,2` and 2-way `63,1`
  skews while the retained exact-even distributions pass. The verification-only
  config change does not require a GPU rerun.
- Why: the former 2× bound admitted materially skewed distributions as passing.
- Refs: PR #585; FN-D9; examples/{qwen3.8-27b-dp8-8gpu,deepseek-v4-flash-0731-dp2-8gpu}/

### 2026-09-01 — [amendment] FN-D9: two replica-pool 8-GPU examples (no orchestration)
- What: `examples/qwen3.8-27b-dp8-8gpu` (Qwen3.8 TP1 × 8) and
  `examples/deepseek-v4-flash-0731-dp2-8gpu` (DeepSeek TP4+EP4 × 2) expose one
  public model each; L2 is only the `ReplicaPool` (`prefix_index: true`,
  `queue_depth_threshold: 0`) and `verify.sh serving` gates the per-replica
  split from `placement_log_path`. Same run/verify UX; no product code changed.
- Why: a plain scale-out serving path (one API over N identical L1 replicas)
  next to the orchestrated tiered example. GPU-verified 2026-09-01: gates green,
  exact 8x8 / 32x2 splits; Qwen 313.7 tok/s at c8 (8.0x c1), DeepSeek 471 tok/s
  at c32 (1.95x one replica) — MEASUREMENTS.md runs 20260901T133331Z / 20260901T140112Z.
- Refs: FN-D9 (docs/design/frontier-native-runtime.md); tests/unit/test_replica_examplectl.py
