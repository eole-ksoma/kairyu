# Progress

Cross-session memory of design changes and project progress.
Maintained per the rules in `.claude/rules/progress-log.md`.

## Current Status

**Deploy-ready (2026-07-03): every milestone of the local-complete plan
(M8–M19) is implemented and CPU-verified — 646 tests, 92% cov. The only
remaining work is GPU execution: performance gates, kernel tuning, fabric
bring-up, `pytest -m gpu`, and `scripts/gpu_gates/` (all pre-written and
dry-run pinned).**

_Last updated: 2026-07-03_

Master roadmap: `docs/roadmap.md` (2026-07-03) — dual hardware profiles (NVLink-HBM
A100/H100/B200 nodes AND the PCIe-only RTX PRO 6000 fleet, A100 and later all
supported), three tracks (E: L1 engine → SOTA incl. MoE, F: fleet-scale control
plane, G6/P: product surface). Next actions: **E1** (single-GPU real engine — RTX
6000 Pro units are available now) + **P-A** (truthful API core, CPU).

| Milestone | Status |
|-----------|--------|
| M1 — Orchestration (L2) + Interface (L3) | Complete and merged. Router / Conductor / MoA, vLLM-compatible `LLM` + `AsyncLLMEngine`, OpenAI-compatible server, YAML/decorator DSL. |
| M2 — Core engine (overlap scheduler + Radix-Paged KV) | CPU half done: scheduler, KV manager, EngineCore step loop, overlap pipeline, pre-GPU robustness (EOS, preemption, abort, pin TTL). Paged-KV attention validated with real tensors on CPU (greedy-equivalence). **Blocked on GPU hardware** for the GPU phase. |
| M3 — Spec decode / CUDA graphs / P-D separation | n-gram draft spec-decode policy and xgrammar structured output implemented CPU-side. CUDA graphs and the rest gated on M2 GPU phase. |
| M4 — Router learning pipeline | Implemented CPU-only (logs → distilled classifier → contextual bandit). Design reviewed. |
| M5 — Intra-node multi-GPU (TP, DP replicas, P-D intra-node) | Design reviewed; **CPU half done** (Communicator/StepInput/TPModelRunner, TP plumbing live, ReplicaPool + affinity, PDCoordinator + `resume_with_kv`). GPU phase: `docs/gpu-runbook.md` §6, prereq M2 Gates 1–3. |
| M6 — Inter-node multi-GPU (2-node DP, KV transfer plane, P-D inter-node, PP) | Design reviewed; **CPU half done** (ClusterSpec, KVTransport + loopback + `bench/kv_transfer_bench.py`, openai_backend replica fixes, async runner contract + PipelinedEngineCore). GPU phase: runbook §7, prereq all M5 gates. |
| M7 — Productionization (serve CLI, gateway wiring, batch, observability) | **CPU half done** (design m7 D1–D8, goal G3): health/readyz/metrics/auth/concurrency guard, `kairyu serve` + DeploymentSpec, ReplicaPool gateway wiring + prober, HTTP session affinity, batch API, Dockerfile + compose + CI smoke drill, `docs/deployment.md`. GPU bring-up: runbook §9. |
| M8 — Engine CPU core (real tokens/sampling/multi-token commit/spec decode/quant基盤/process split) | **Complete** (2026-07-03, `docs/design/m8-engine-cpu.md`): HF tokenizer seam + SSE-safe stop strings, full sampler + xgrammar in-path, scheduler spec reservation, n-gram SpeculativeRunner (spec ≡ greedy pinned), NVFP4/HardwareProfile/safetensors reader, ZMQ `kairyu-proc` process split. 437 tests, 95% cov. |
| M9 — Truthful API (usage/templates/logprobs/completions/n>1) | **Complete** (2026-07-03, `docs/design/m9-truthful-api.md`): G6 P-A gates CPU-green — real usage + cached_tokens + include_usage, HF Jinja templates (transformers byte-match), logprobs + /v1/completions, n>1 fan-out, response_format validation, bench token-TPOT. 471 tests. |
| M12 — Real model zoo dense (Llama/Qwen, PagedKVPool, PagedModelRunner) | **Complete** (2026-07-03, `docs/design/m12-model-zoo.md`): full-engine greedy == transformers generate (3 archs); loader + model_path wiring; pytest gpu/hf_hub/dist markers. 501 tests. |
| M13 — AttentionBackend seam (torch/MLA reference/FlashInfer adapter/selector) | **Complete** (2026-07-03, `docs/design/m13-attention-backend.md`): fake-pinned FlashInfer contract + tests/gpu mirror; MLA two-form equivalence oracle. 514 tests. |
| M14 — Quant compute (fp8/int8/awq/gptq/nvfp4 CPU references + Triton stubs) | **Complete** (2026-07-03, `docs/design/m14-quant-compute.md`): all 5 schemes load + run through the full engine on CPU; formats pinned vs live Hub checkpoints. 530 tests. |
| M15 — MoE + MLA archs (Qwen3-MoE, DeepSeek-V3 incl. yarn) | **Complete** (2026-07-03, `docs/design/m15-moe-mla.md`): full-engine greedy == hf.generate; latent MLA pool (M18-ready). 547 tests. |
| M16 — Distributed execution (gloo-tested TP/EP/PP; NCCL by constructor) | **Complete** (2026-07-03, `docs/design/m16-distributed.md`): TP=2/EP=2/PP=2 spawn parity gates green in the default suite. 553 tests. |
| M17 — StepExecutor (CUDA-graph seam) + EAGLE-3/MTP drafts | **Complete** (2026-07-03, `docs/design/m17-graphs-drafts.md`): fake-graph lifecycle suite; perfect-draft e2e ≡ greedy; corrected EAGLE-3/MTP formats. 571 tests. |
| M18 — KV transport (serde/remote handoff/NIXL adapter) + 2-process P-D | **Complete** (2026-07-03, `docs/design/m18-kv-transport.md`): TCP byte-parity E2E green. 584 tests. |
| G4 — MoE engine (fused experts, EP, MTP, NVFP4, MLA) | Goal defined (`docs/goals/g4-moe-engine.md`); lifts the G2 MoE non-goal. Design doc + review required before implementation. |
| M10a — Elastic fleet base (dynamic pool/registry/tracing/Helm) | **Complete** (2026-07-03, `docs/design/m10-fleet-cpu.md`). 594 tests. |
| M10b — KV-aware routing (prefix trie / KV events / offline tuning) | **Complete** (2026-07-03). 610 tests. |
| G5 — Fleet scale (elasticity, KV-aware routing, P/D pools, tiering, tenancy) | Goal defined (`docs/goals/g5-fleet-scale.md`); amends m7 D2 (k8s as machine layer), m5 D4/m7 D6 (prefix-aware placement), m6 D1 staticness, ClusterSpec cap, m7 D8 (OTel). F1/F2 are CPU-mock-testable now. |
| M11 — Product surface + tenancy (streaming auto/tenancy/responses/embeddings/F5) | **Complete** (2026-07-03, `docs/design/m11-product.md`). 627 tests. |
| G6 — Product surface (truthful API, Fugu-class product, frontier scoreboard) | Goal defined (`docs/goals/g6-product-surface.md`). P-A (usage truth, HF chat templates, logprobs, structured outputs) is CPU work, start now. |

What works today: full stack on CPU — `kairyu` EngineBackend wired through the
OpenAI-compatible server with the mock/CPU runner; serving/router/multiturn benchmarks
in `bench/`; `kairyu serve <deployment.yaml>` runs a hardened gateway (pool of remote
replicas, auth, metrics, batch) or a replica node, and the compose topology
(1 gateway + 3 mock replicas) passes the CI smoke drill incl. kill/recover.
`kairyu bench run` executes the 11-slot Fugu-release quality suite against any
deployed gateway (single models and named orchestrations as scoreboard columns)
with dataset downloaders, LLM-judge/vision/docker degradation, and a dated
footnoted scoreboard (G6 P-C1).

Active blockers: RTX 6000 Pro units are now partially available — M2/E1 GPU phase is
unblocked on the PCIe profile (H100 boxes still wanted for NVLink-profile gates);
execution plan is `docs/gpu-runbook.md` + `docs/roadmap.md` §4. Hardware procurement
(PCIe-switch chassis, ≥400 Gb/s RDMA NICs) gates E4/E5 and is decided during E3 from
E1's measured P2P matrix. Human sign-off pending on M2–M4 design reviews.

## Change Log

### 2026-07-14 — [progress] FlashInfer SM120 (Blackwell) enabled end-to-end: GPU device placement + AOT packaging
- What: The single-process GPU serve path had NO device placement — `build_engine_loop` loaded the
  model and `PagedKVPool` in fp32 on CPU, so the "GPU" deployment actually ran inference on CPU via the
  device-agnostic torch backend. FlashInfer (the SM120 default) surfaced this as `KeyError: torch.float32`
  (its prefill `plan()` has no fp32 kernel). Fixes:
  - probe-driven device/dtype in `build_engine_loop`: cuda → bf16 on device, cpu → fp32 on host (guarded,
    so every CPU path is byte-identical). `load_model(dtype=…)` + `model.to(device)` +
    `PagedKVPool.for_cache(dtype, device)`.
  - device-correct input/index tensors in `PagedModelRunner` and `PagedKVPool._flat_indices`/`gather`.
  - `Sampler` samples on CPU (seeded RNG + penalties/enforcer) so the m8-D2 determinism pins hold while
    logits arrive on cuda.
  - `FlashInferBackend.attend_batched` implemented; decode passes a 3D `[1,H,D]` query (0.6.14 contract).
  - observability: backend-exception handlers log `logger.exception` (client still gets only the class name).
  - `Dockerfile.cuda` → multi-stage AOT: CUDA 13.0 `-devel` build stage compiles `flashinfer-jit-cache`
    scoped to `FLASHINFER_CUDA_ARCH_LIST=12.0f` + bf16/head_dim-64/FA2 (avoids FP4/FP8 SM120 kernel risk and
    the full 3402-step build); slim `13.0.1-runtime` (no nvcc) installs the wheel `--no-deps`.
- Why: run the FlashInfer paged-attention kernels on RTX PRO 6000 Blackwell (SM120). The torch backend was
  a correct-but-slow interim and — as discovered — CPU-bound. AOT removes the ~20s first-request JIT so a
  cold replica is not ejected by the gateway readiness probe.
- Verified (stg-gpu-01, RTX PRO 6000 / CUDA 13.0): `pytest -m gpu` 7/7 (flashinfer == torch reference,
  fp16/bf16 across prefill/chunk/decode/batched-decode); direct-replica + gateway `/v1/chat/completions`
  200 with correct output; cold-start first request 0.996s (was 19.6s under runtime JIT); no `/readyz`
  eject. torch backend now also runs on GPU/bf16.
- Refs: docs/design/flashinfer-sm120-aot.md; kairyu/engine/kairyu_backend.py,
  engine/core/{model_runner,kv_pool,sampler}.py, engine/core/attention/flashinfer_gpu.py,
  entrypoints/server/app.py, Dockerfile.cuda.

### 2026-07-09 — [progress] Single-node GPU compose: dedicated gateway config + attention-backend env
- What: `docker-compose.gpu.yaml` now mounts a new `deploy/compose/gateway-gpu.yaml`
  (single `replica` upstream, forwards `model: default`) instead of the shared
  `gateway.yaml`, and passes `KAIRYU_ATTENTION_BACKEND` through to the replica
  (empty → auto-select; set `torch` to bypass FlashInfer). `gateway.yaml` is left
  in its CPU-smoke form (three `replica-1/2/3` upstreams, `model: llama`).
- Why: `gateway.yaml` was mounted by BOTH `docker-compose.yaml` (CPU smoke: three
  `replica-N` services serving engine id `llama`) and `docker-compose.gpu.yaml`
  (single `replica` service serving engine id `default`). The two topologies have
  different service names and engine ids, so one file cannot serve both — pointing
  it at the single GPU replica breaks the CPU compose and the CI `compose_smoke.sh`
  drill. Splitting into `gateway-gpu.yaml` lets each topology stand alone. The
  attention env exists because FlashInfer has no Blackwell/sm_120 kernels yet, so a
  Blackwell/RTX PRO 6000 replica can pin `torch` (selector: `KAIRYU_ATTENTION_BACKEND`,
  honored on the single-process `model_path` engine path).
- Refs: `deploy/compose/{gateway-gpu.yaml,docker-compose.gpu.yaml,gateway.yaml}`,
  `kairyu/engine/core/attention/selector.py`, `scripts/compose_smoke.sh` (m19 D2).

### 2026-07-09 — [progress] GPU image base bumped to CUDA 12.8.1 / Ubuntu 24.04
- What: `Dockerfile.cuda` base image `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
  → `nvidia/cuda:12.8.1-runtime-ubuntu24.04`. Ubuntu 24.04 ships `python3.12`
  in its default repos, so the existing `apt-get install python3.12` line now
  resolves natively; nothing else in the image changed.
- Why: The GPU execution host runs Ubuntu 24.04, so the deployment image should
  match the host OS. The CUDA 12.8 runtime also adds SM120 (Blackwell /
  RTX PRO 6000) support that the roadmap's PCIe fleet targets and that the 12.4
  runtime lacked.
- Refs: `Dockerfile.cuda`; supersedes the base image recorded in the
  2026-07-03 M19 deploy-packaging entry (m19 D1).

### 2026-07-04 — [design] Review remediation Phase 6: GPU-day seam changes (CPU design + C5 contract test)
- What: Captured the five GPU-day seam changes from the full-repo review in
  `docs/design/gpu-day-seams.md` (C5 CUDA-graph static buffers, C4 batched
  execution, E3 engine-loop unification, TP delta-broadcast + sampling ownership,
  KVTransport region ownership), and landed the **C5 contract test**: a faithful
  `SnapshotGraphBackend` that freezes page_tables/seq_lens at capture (as a real
  CUDA graph does), plus `test_graph_replay_reflects_current_page_tables`
  (`xfail(strict=True)`) that concretely proves `GraphStepExecutor` currently
  rebinds page tables as Python attributes a real graph never sees. The test
  flips to pass when the static-device-buffer fix lands.
- Why: These CPU-pinned abstractions silently break (C5) or make a perf gate
  unreachable (C4) when real kernels/NCCL/FlashInfer replace the CPU references;
  they must be designed + contract-tested on CPU before GPU time, but can only be
  fully validated on hardware. The full implementations are a scheduled GPU-day
  design milestone (land before the runbook perf gates), not a same-session edit.
- Refs: review report; `docs/design/gpu-day-seams.md`,
  `kairyu/engine/core/step_executor.py` (`SnapshotGraphBackend`),
  `tests/unit/test_step_executor.py`.
### 2026-07-05 — [progress] Real multi-process TP wired into `kairyu serve --tp N`
- What: `build_engine_loop(model_path=…, tensor_parallel_size>1)` no longer
  raises "not yet wired" — it spawns a `DistTPLauncher` group (rank 0 in the
  serve process, ranks 1.. as workers running `worker_step_loop`) and drives it
  through `DistTPModelRunner`. The loop carries a `.tp_launcher` handle that
  `KairyuBackend.shutdown()` calls to stop the workers and destroy the group.
  Added `load_generation_defaults` (public eos/stop loader for the sharded path).
- Why: M16's distributed TP was spawn-tested only in `tests/dist` and unreachable
  from the serve entrypoint — so real tensor-parallel models could not be
  deployed. Now `kairyu serve --tp 2` runs end to end.
- Refs: `kairyu/engine/kairyu_backend.py` (`_build_dist_tp_loop`),
  `kairyu/engine/core/worker.py` (`DistTPLauncher`, `_tp_worker_entry`),
  `kairyu/models/loader.py`, test
  `tests/dist/test_distributed.py::test_dist_tp_launcher_serve_path_matches_single_process`.

### 2026-07-04 — [progress] Review remediation Phase 8: packaging + doc accuracy
- What: Fixed the cross-cutting packaging/doc defects from the full-repo review.
  Added an **`[engine]` extra** (torch + xgrammar + tokenizers + safetensors) so
  real models run WITHOUT the dev group, and pointed **`Dockerfile.cuda`** at it
  (`--extra engine` replaces `--extra hf`) so the production GPU image ships
  xgrammar and can serve `response_format: json_schema` (was missing). Fixed the
  misplaced comment above the `otel` extra (it described the fleet transports).
  `build_engine_loop`'s TP>1 error/docstring now state the truth — the
  multi-process `DistTPModelRunner` exists (m16, tests/dist) but is not yet wired
  into the single-process serve path — instead of "arrives in M16". Refreshed
  `docs/gpu-runbook.md` §0/§1: corrected the stale "177 tests" count, the
  `--group gpu`/`uv sync --dev` command errors (now `--extra gpu`/`--group dev`),
  and the "replace KairyuBackend._tokenize / TorchPagedRunner" instructions that
  M8/M12/M13 already delivered, with a note that the seams exist and GPU-day is
  enabling/tuning them.
- Why: The GPU image couldn't serve structured outputs, there was no non-dev
  install path for real models, and the runbook (the artifact GPU day executes
  from) contradicted the codebase.
- Refs: review report; `pyproject.toml`, `uv.lock`, `Dockerfile.cuda`,
  `kairyu/engine/kairyu_backend.py`, `docs/gpu-runbook.md`.
  **Deferred follow-up:** `kairyu validate` cross-artifact command, typed
  `GenerationRequest.prompt` (token-ids/multimodal), `deploy/spec.py`
  ServerSection compose-not-inherit, and the `kairyu/bench/` package boundary.
### 2026-07-04 — [progress] Review remediation Phase 7: host-path performance (safe subset)
- What: Fixed the provably-safe, output-preserving host-path hot spots from the
  full-repo review. **P5**: `prompt_chunks` re-hashed the whole prompt prefix per
  256-char chunk (O(L²) sha256 on the placement path, event-loop-blocking) and
  the pool called `overlap()` twice per replica; replaced with ONE streaming
  sha256 chain (byte-identical keys, proven equivalent over random trials) and a
  single `overlap()` per replica. **P-perf (completions)**: `/v1/completions`
  ran a prompt array serially (`await` per prompt = sum of latencies); now
  `asyncio.gather` runs them concurrently with order restored by index (response
  byte-unchanged).
- Why: Both are event-loop-blocking / latency costs on the request path that
  survive the GPU swap; both are output-identical so they carry no correctness
  risk.
- Refs: review report; `kairyu/orchestration/{prefix_index,replica}.py`,
  `kairyu/entrypoints/server/app.py`; tests `tests/unit/test_kv_routing.py`.
  **Deferred (risk/complexity, need care or their own change):** P1 incremental
  detokenization (correctness-sensitive output path — a subtle detok bug corrupts
  generation, and CPU tests can't cover every tokenizer edge, so not worth a
  perf-only rewrite), P3 (process-split delta wire), P4 (async ledger/router I/O
  — file-handle lifecycle), P6 (eviction leaf heap), P7 (batched spec verify),
  and the MEDIUM-perf items (sampler penalty state, stop-string offset, queue
  coalescing, scheduler deque, KV-event hash chain, page-table cache).
### 2026-07-04 — [progress] Review remediation Phase 5: bench scoring correctness + security
- What: Fixed the scoring-integrity and security defects in the Fugu bench suite.
  **B1**: the MCQ answer-extraction regex matched "answer" + the first letter of
  the following word (so "Answer: B, because the answer depends…" extracted D)
  and the fallback picked lone lowercase articles/pronouns — tightened to a
  bounded letter after the marker and an uppercase-only fallback. **B2**: an
  un-typed `normalize()` error (schema drift KeyError, image/codec, unpickling)
  crashed the whole suite run; it now degrades THAT dataset to `unavailable`
  ("degradation is data, not control flow"). **M6**: the dataset cache is now
  invalidated when the pinned dataset/revision changes, so bumping `hf_revision`
  re-downloads instead of scoring stale rows. **M7**: private-test blobs unpickle
  through a `_RestrictedUnpickler` that blocks class/global loading (was a
  download-time arbitrary-code vector); the judge response fed into the prompt is
  length-capped. **M8**: LCB solutions that start with `from __future__ import`
  no longer become a SyntaxError when the import header is prepended (the future
  import is hoisted). **M10**: the judge verdict regex accepts markdown-emphasized
  labels (`**correct:** yes`) and the judge token budget was raised so a reasoning
  judge is not truncated before its verdict.
- Why: Each silently corrupts the scoreboard (wrong scores, crashed runs, stale
  data) or is a security hole (ACE at download time).
- Refs: review report; `kairyu/bench/{adapters/base,adapters/livecodebench,cache,judge}.py`;
  tests under `tests/bench/`. **Deferred follow-up (design/policy):** B3 (resume
  per-pair config hash), B4 + denominator policy (skipped/unjudged as 0 or n/a,
  show per-target n_scored), LCB per-line/tolerant scoring, sandbox NPROC/session
  hardening, self-judge (judge==target) scoreboard flag, judge prompt delimiters.
### 2026-07-04 — [progress] Review remediation Phase 4: model + quant parity
- What: Fixed the parity-affecting model/quant defects from the full-repo review.
  **M3 (rope)**: unsupported `rope_scaling` kinds (linear/dynamic/longrope) now
  raise instead of silently dropping to None — a silent parity break vs
  hf.generate. **M4 (fp8 load)**: `Fp8Linear` adopts the checkpoint's
  weight_scale shape, so static per-tensor `(1,)` FP8 (and modelopt FP8) load
  instead of a size-mismatch crash. **M1 (nvfp4 oracle)**: the RNE tie table
  applied LUT *values* as indices and dropped two boundaries, corrupting the
  GPU-kernel packing oracle by up to 60%; replaced with the correct even-index
  table for all seven boundaries (and the test that pinned the wrong behavior).
  MEDIUM: DeepSeek MoE config now falls back to HF defaults
  (norm_topk_prob=True, routed_scaling=2.5, first_k_dense=3, n_group/topk_group
  8/4) for trimmed configs, and a missing expert count raises clearly instead of
  `int(None)`; GPTQ/AWQ `group_size=-1` (single whole-input group) normalizes to
  in_features instead of a negative buffer count; `tp_view` fails fast on MoE
  (no dense down_proj to row-parallelize) like it already does for MLA; bare
  `quant_method: "fp8"` rejects block-wise FP8 (weight_block_size) loudly
  instead of mis-routing DeepSeek block-FP8 to the per-channel path.
- Why: Each is a silent wrong-output or load-time failure on real checkpoints;
  all are CPU-validatable and covered by new tests.
- Refs: review report; `kairyu/models/{config,parallel}.py`,
  `kairyu/quant/{linear,nvfp4}.py`, `kairyu/engine/core/quant_config.py`;
  `tests/unit/test_config_and_fp8_load.py`, `test_quant_compute.py`.
  **Deferred (needs GPU + SpecForge reference to validate):** EAGLE-3 midlayer
  RoPE (H1) and KV-cached rollout feedback (H2) — both affect draft ACCEPTANCE
  RATE only, not output correctness (verification is by the target), so no CPU
  test can validate a fix; plus the design items (linear_factory context,
  forward_fused wiring, HF-name-preserving TP/EP wrappers, draft-head quant).
### 2026-07-04 — [progress] Review remediation Phase 3: orchestration + fleet reliability
- What: Fixed the L2 fleet/orchestration HIGH defects from the full-repo review.
  **O1**: request errors were all counted as replica failures — a new
  `UpstreamClientError` (4xx) is raised by the openai backend and excluded from
  `consecutive_failures`, so one misbehaving client can no longer cascade-eject
  the pool. **O2**: the HealthProber was ordinal-keyed against a dynamic
  id-keyed pool (wrong-replica restore / IndexError / silent prober death);
  it is now id-keyed, resolves URLs per id, and `run()` swallows a bad tick.
  **O3**: the prober now probes `/readyz` (readiness) not `/health` (liveness),
  so a drained/wedged node stays ejected — O1+O3 together kill the flap loop.
  **O4**: the Conductor wraps each unit so a transient backend error records a
  trace event and returns best-so-far instead of raising and discarding every
  completed output. MEDIUM: **M2** orchestrator direct calls no longer mint a
  random per-request session_id (which defeated prefix + least-outstanding
  routing); **M4** KvEventIndex stamps freshness only after a valid apply,
  handles vLLM `AllBlocksCleared`, and the ZMQ drain drops malformed frames
  instead of aborting; **M5** `remove_replica` calls `prefix_index.forget_replica`
  so a re-added id can't inherit phantom prefixes; **M7** lifespan shutdown
  isolates a crashed background task and shuts every engine down independently.
- Why: These are DoS / flap-loop / cost-and-routing-correctness defects the
  single-node CPU tests could not exercise.
- Refs: review report; `kairyu/orchestration/{replica,conductor,orchestrator,kv_index}.py`,
  `kairyu/engine/{backend,openai_backend}.py`, `kairyu/deploy/{prober,registry,spec,builder}.py`;
  tests under `tests/unit/`. Deferred follow-up: M1 (verifier non-target deps +
  _SafeDict masking), M3 (MoA path Budget/cost wiring), M8 (run_chat periodic
  keep-alive), and the KvEventIndex↔ReplicaPool integration (design item).
### 2026-07-04 — [progress] Review remediation Phase 2: API security + tenant isolation
- What: Fixed the CRITICAL/HIGH L3-server defects from the full-repo review.
  **C3 (CRITICAL) batch/file tenant isolation**: File/Batch objects gained an
  `owner`; the store scopes every get/read/list/cancel and cross-tenant access
  reads as not-found — a tenant can no longer enumerate or read another's batch
  prompts/outputs (worker output/error files inherit the batch owner). **S1**:
  a non-object JSONL line becomes a per-line error instead of wedging the job
  in_progress forever. **S2**: invalid sampling params (top_p=0, n=0,
  temperature<0) return 400, not a 500/mislabeled-502. **S3**: streaming chat
  and /v1/completions are now metered (were a billing bypass) — usage flows to
  the ledger; orchestrator-stream/responses/embeddings metering still TODO.
  **S4**: `tokens_per_minute` is enforced via a per-tenant token bucket charged
  post-response. **S5**: `/admin/drain` requires an admin key when configured
  (was any data-plane key = one-request DoS) and gains `/admin/undrain`.
  **S6**: streamed `delta.tool_calls[]` carry the required `index` (SDK
  accumulation). **S7**: `/v1/files` upload is size-capped (413) to prevent
  gateway OOM.
- Why: These are cross-tenant disclosure, billing bypass, and DoS holes that
  the single-tenant CPU test suite could not see.
- Refs: review report; `kairyu/batch/{store,worker}.py`,
  `kairyu/entrypoints/server/{batch_routes,app,health,settings,tenancy,protocol}.py`;
  tests under `tests/server/` + `tests/unit/test_batch_store_tenancy.py`.
  Deferred to a Phase 2 follow-up: MEDIUM items (Prometheus label cardinality,
  /v1/responses store bounds + tenant scope, error-body leak scrub, AUTO-model
  param handling, non-ASCII bearer 401, embeddings validation) and full S3
  metering coverage.

### 2026-07-04 — [progress] Repo-wide review remediation Phase 1: engine-core correctness
- What: Fixed the CRITICAL/HIGH engine-core defects found in the 2026-07-04
  full-repo review (report in job scratch). **C1 radix cache poisoning**:
  `commit_and_release` folded the final sampled token's page as computed even
  though the decode loop never writes that token's KV — a page-boundary
  completion poisoned the next multi-turn prefix (silent wrong output ~1/16 of
  requests). Now caps committable length below the unwritten final token
  (`radix_kv.py`). **C2 oversized-prompt permanent death**: a prompt larger
  than the whole KV cache blocked the head of line forever, turning every empty
  schedule into a fatal engine stall that killed all concurrent requests. The
  scheduler now rejects unadmittable prompts at admission (finish_reason
  "length", drained via `drain_rejected`), and all four engine cores
  (EngineCore/OverlapEngineCore/PipelinedEngineCore/EngineLoop) replace the
  fatal stall with `reject_waiting_head`. **E1 ZMQ receiver death**: a dead
  receiver left every subsequent request hanging; `_ensure_started` now respawns
  a fresh child over a crashed one and per-frame errors no longer kill the loop.
  **E2 state leaks**: `Scheduler.forget` + runner `release` reclaim finished
  per-request state (output lists, sampler seeds, grammar enforcers) — wired
  into `EngineLoop`. MEDIUM: engine_service per-message fault isolation,
  `resume_with_kv` honors ignore_eos/min_tokens/stop_token_ids/finish_reason,
  `RemoteKVReceiver.adopt` frees the allocation on failure, `zmq generate()`
  aborts on cancel, NIXL send yields instead of busy-spinning. LOW: PagePool
  rejects duplicate free ids, torch attention builds indices on the query
  device.
- Why: The CPU test suite was single-turn/single-tenant and could not see these
  multi-turn / long-running / crash-path failures; each is output-corrupting,
  a DoS, or an unbounded leak on the deploy-day paths.
- Refs: review report; `kairyu/engine/core/{radix_kv,scheduler,engine_core,
  overlap,pipeline,pd_remote,pages,model_runner,spec_runner,engine_service}.py`,
  `kairyu/engine/{engine_loop,zmq_backend}.py`; tests under `tests/unit/`

### 2026-07-03 — [progress] Fugu benchmark suite: one-command quality scoreboard (G6 P-C1)
- What: 646 → 730+ tests. New `kairyu/bench/` package + `kairyu bench
  run/download/report/list` CLI. All 11 rows of the Fugu release table
  (sakana.ai/fugu-release) implemented as adapters: GPQA Diamond, HLE,
  LiveCodeBench(+Pro community mirror), SciCode, CharXiv Reasoning, MRCRv2,
  LongBench-v2 (annotated substitute for the unpublished "Long Context
  Reasoning"), τ³-Bench Banking / SWE-Bench Pro (mini-swe-agent scaffold) /
  Terminal-Bench 2.1 (Harbor) as official-harness wrappers. One command
  downloads missing datasets (normalized JSONL cache under
  ~/.cache/kairyu/benchmarks, $KAIRYU_BENCH_CACHE), runs every benchmark ×
  every target, and writes bench/results/fugu/<run_id>/ with per-item
  evidence, methodology (dataset revisions, judge model, truncation policy)
  and a footnoted Fugu-layout scoreboard (JSON+MD). Degradation is data:
  docker/gated-dataset/judge/vision/context-length preconditions produce
  skipped/partial cells with reasons — exit 1 only on hard failures; same
  --run-id resumes. Configurable LLM judge endpoint (HLE free-form, CharXiv,
  τ user-simulator; unjudgeable items recorded, never guessed). Execution
  scoring in an rlimit subprocess sandbox (documented as not a security
  boundary). Orchestration measured as plain model names (kairyu-auto,
  kairyu-auto-max) via the new `orchestrators:` DeploymentSpec map. New
  extras: [bench] (datasets/hub/pillow/h5py), [bench-agentic]
  (mini-swe-agent/swebench/harbor; tau3 documented as git install). Offline
  fixtures keep the default CPU suite and --offline-fixtures runs hermetic;
  networked download tests are hf_hub-marked.
- Refs: goal G6 P-C1/P-B4, roadmap §6 evidence rules; `kairyu/bench/`,
  `kairyu/entrypoints/cli.py`, `docs/benchmarks.md`,
  `examples/{deploy_multi_orchestrator,bench_fugu,agent_pool_max}.yaml`,
  `tests/bench/`

### 2026-07-03 — [design] DeploymentSpec gains named `orchestrators:` (m7 D3 / m11 D2 amendment)
- What: `DeploymentSpec.orchestrators: dict[name, OrchestratorSection]` serves any
  number of named orchestrations (e.g. `kairyu-auto` + `kairyu-auto-max`) from one
  YAML; the legacy single `orchestrator:` key stays and is still served as
  `kairyu-auto`. Validators: name collisions with engines/pools rejected at spec
  load; `orchestrator:` + `orchestrators["kairyu-auto"]` double-declaration
  rejected. Builder passes the named map to `create_app(orchestrators=)` — the
  m11 tiered-auto path was already server-side, just not YAML-expressible.
- Why: The Fugu-suite benchmark work (G6 P-C1) needs "orchestration with an
  arbitrary model composition" to be deployable, then benchmarked as just another
  model name on the same endpoint. Previously `kairyu-auto-max` was reachable
  only via the `create_app` kwarg in tests, never from `kairyu serve`.
- Refs: `kairyu/deploy/{spec,builder}.py`,
  `tests/unit/test_deployment_spec.py`, `tests/server/test_serve_builder.py`

### 2026-07-03 — [progress] M19 complete: deploy-ready — the local-complete plan is DONE
- What: 627 → 646 tests. Dockerfile.cuda (nvidia/cuda 12.4 + gpu/hf/fleet
  extras), GPU compose (device reservations, model volume), Helm
  values-gpu.yaml (per-profile nodeSelector: pcie-gddr / nvlink-hbm),
  scripts/gpu_gates/ covering runbook §0/1/2/3/6/7/9 + G4/G5 — every script
  --dry-run capable, with a CPU suite pinning that dry-runs emit command
  plans AND every referenced path exists today. [gpu] extra with
  sys_platform=='linux' markers (macOS uv sync clean — verified).
  **All 13 milestones of the local-complete plan (M8–M19 + M10a/b) are
  implemented.** Remaining work is strictly the hardware list: performance
  gates, kernel selection/tuning, fabric bring-up, and `pytest -m gpu` /
  scripts/gpu_gates execution.
- Refs: `docs/design/m19-deploy-packaging.md`, `Dockerfile.cuda`,
  `scripts/gpu_gates/`, `deploy/helm/kairyu/values-gpu.yaml`

### 2026-07-03 — [progress] M11 complete: Fugu-class product surface + tenancy
- What: 610 → 627 tests. Usage threaded through MoA/Conductor/Orchestrator
  (was dropped at three layers) — the AUTO path now returns REAL summed
  usage (the m9 usage=None fallback removed at that call site only).
  run_chat streaming: direct route streams live token deltas; multi-stage
  routes emit SSE COMMENT keep-alives (data: lines would break the OpenAI
  SDK) then a buffered final; X-Kairyu-Trace: 1 opts into a kairyu_trace
  field. Tiered auto models (orchestrators dict; kairyu-auto-max routes
  multi_agent through MoA — previously dead code). Tenancy v1: auth stores
  the matched key in scope state, TenantLimitMiddleware runs INSIDE auth
  (401 wins; unauthenticated never drains buckets), per-tenant token
  buckets, O_APPEND JSONL usage ledger written from handlers,
  /admin/usage; isolation + exact reconciliation gates. /v1/responses
  (reviewed subset: exact output-item shapes, input/output_tokens usage
  names, instructions, previous_response_id store; stream descoped) and
  /v1/embeddings (base64 = the SDK default) — both OpenAI SDK round-trip
  tested (openai>=1.66). Vision wire format (content-parts flattening
  everywhere incl. batch-worker path; image parts 400 on non-vision
  engines). F5 CPU: priority admission with aging (injectable clock,
  effective priority at sort time, head still blocks on KVCacheFull),
  AdmissionController (gateway-observable TTFT EMA; admit/defer/shed),
  autoscale_decision hysteresis. Open WebUI compose + frontier_compare
  bench harness (scoreboard schema pinned).
- Refs: `docs/design/m11-product.md` (Status: Implemented);
  `kairyu/entrypoints/server/{tenancy,extra_routes,slo}.py`,
  `kairyu/orchestration/{orchestrator,conductor,moa}.py`,
  `bench/frontier_compare.py`, `deploy/compose/docker-compose.webui.yaml`

### 2026-07-03 — [progress] M10b complete: KV-aware routing
- What: 594 → 610 tests. PrefixIndex (text-chunk approximate trie — the
  gateway has no token ids, review A12; prefix-chained keys, LRU-capped);
  ReplicaPool opt-in prefix scoring (α·overlap − β·outstanding, session
  affinity still first, prefix_match decision reason; disabled default keeps
  m5 behavior byte-identical). RadixKVCache(event_sink=): BlockStored on the
  computed False→True transition + decode-extension nodes (allocate never
  emits; _release double-fire guarded), BlockRemoved ONLY from eviction —
  removed hashes proven identical to stored hashes. KvEventIndex (precise
  per-replica block hashes, staleness > 500 ms → None = fall back to the
  trie) + ZMQ PUB/SUB transport with a chaos gate (publisher killed →
  staleness fallback). Offline (α, β) grid tuner over PlacementRecords.
  Security-review hardening: /admin/drain pinned auth-protected when keys
  are configured (keyless = trusted-mesh mode by explicit m7 D5 choice).
- Refs: `docs/design/m10-fleet-cpu.md` (M10a+M10b Implemented);
  `kairyu/orchestration/{prefix_index,kv_index}.py`,
  `kairyu/engine/core/radix_kv.py`, `kairyu/orchestration/learning/dataset.py`

### 2026-07-03 — [progress] M10a complete: elastic fleet base
- What: 584 → 594 tests. ReplicaPool reworked to id-keyed dynamic membership
  (legacy sequences auto-id "0".."N-1" so HRW mappings AND Prometheus labels
  are unchanged — zero existing-test edits beyond one error message).
  add/drain/remove lifecycle: drain stops NEW placements (HRW runs over
  eligible = healthy ∧ not-draining), remove refuses in-flight unless forced,
  late completion on removed ids is a no-op. HRW remap property gates:
  removal moves ONLY the departed replica's sessions; addition moves ~1/N.
  deploy/registry.py: TTL-heartbeat ReplicaRegistry (injected clock),
  DiscoverySource protocol (static + registry; k8s-endpoints is a deploy-day
  adapter), PoolReconciler (drain-then-remove, tolerates in-flight refusal
  across ticks). POST /admin/drain flips /readyz 503 (node role).
  kairyu/telemetry.py traced_span (L2-safe, no-op without the otel extra) +
  pure-ASGI TracingMiddleware + pool-placement span; opentelemetry-sdk in
  dev group + otel extra. BatchStoreProtocol (full 8-method surface). Helm
  chart (readiness /readyz, config at /etc/kairyu/config.yaml) +
  scripts/kind_smoke.sh + CI kind-smoke job + helm-template render test.
- Refs: `docs/design/m10-fleet-cpu.md` (M10a Implemented);
  `kairyu/orchestration/replica.py`, `kairyu/deploy/registry.py`,
  `kairyu/telemetry.py`, `deploy/helm/kairyu/`

### 2026-07-03 — [progress] M18 complete: real-byte KV transfer + two-process P-D
- What: 571 → 584 tests. kv_serde (PagedKVPool ⇄ PageFrame, layer-major
  fragments, MLA empty-v contract, loud mismatch errors, pool_fingerprint
  handshake). KVHandoff seam widened to carry source page ids (a
  byte-extracting handoff cannot recover the tail page from tokens — the
  freed tail gets reallocated). RemoteKVHandoff/RemoteKVReceiver over the m6
  transport protocol: copy-before-commit ordering, receiver-side dedup skips
  injection of radix-cached pages, sender page ids remapped to
  new_full_pages+(tail). StreamCopyKVHandoff (side-stream copy window;
  synchronize even on failure). NIXL adapter (deferred import;
  registration-once + descriptor math pinned via fake module). FLAGSHIP:
  two REAL processes over TCP — prefill extracts page bytes between
  execute() and update(), decode adopts via resume_with_kv and decodes;
  outputs == single-engine greedy AND per-page sha256 byte parity.
- Refs: `docs/design/m18-kv-transport.md` (Status: Implemented);
  `kairyu/engine/core/{kv_serde,pd_remote,handoff_stream,kv_transport_nixl_gpu}.py`,
  `tests/dist/test_pd_two_process.py`

### 2026-07-03 — [progress] M17 complete: graph-capture seam + EAGLE-3/MTP draft heads
- What: 553 → 571 tests. StepExecutor seam: decode_buckets policy
  (vLLM-style sizes), GraphStepExecutor (capture-once-per-bucket, static
  buffer copy-in, padding to scratch page with outputs dropped, invalidate(),
  oversize→eager) fully pinned against FakeGraphBackend; cuda_graph_gpu.py
  holds the only CUDA lines (side-stream warmup, shared pool). DraftSource
  protocol: n-gram default byte-identical; ModelDraftSource e2e gate — a
  perfect draft through the FULL spec pipeline == plain greedy with >0.9
  acceptance. EAGLE-3 head per corrected review pins (2H midlayer, pre-norm
  residual, fc [H,3H] once per cycle, TRAINED reduced-vocab lm_head + d2t
  offset map, target-aliased embeddings) + SpecForge loader with format-drift
  guards. DeepSeek MTP head (embedding-first eh_proj, separate physical
  head/embed tensors, MoE decoder block at layer_index=num_hidden_layers) +
  extra-layer checkpoint loader. Scope honesty: batched decode capture rides
  FlashInfer's decode wrapper on deploy day (A1); grammar-rollback spec
  stays deferred.
- Refs: `docs/design/m17-graphs-drafts.md` (Status: Implemented);
  `kairyu/engine/core/{step_executor,graph_buckets,draft,cuda_graph_gpu}.py`,
  `kairyu/models/{eagle,mtp}.py`

### 2026-07-03 — [progress] M16 complete: TP/EP/PP run over real multi-process collectives
- What: 547 → 553 tests (incl. 5 gloo spawn gates that run in the default
  suite). TorchDistCommunicator (m5 protocol + tensor extension; NCCL is a
  constructor argument on deploy day). TP: pre-sharded-config scheme
  (tp_view divides heads/kv/intermediate — modules and kv pools come out
  rank-local automatically), get_slice per-rank loading with FULL-config
  bounds, RowParallelLinear (bias once, after the all_reduce), embed/lm_head
  replicated (every rank holds full logits → every rank samples identically,
  m5 D1 kept). TP=2 spawn gate: EngineCore on rank 0 via DistTPModelRunner
  (snapshot broadcast + A11 handshake), worker_step_loop on rank 1 — greedy
  output IDENTICAL to single-process for llama AND qwen2 (bias) tinies.
  EP: EpMoeBlock over uneven all_to_all_single (counts exchange first);
  EP=2 ≡ single-block to 1e-5. PP: PpStageModel stage seam (embed/mid/final,
  rebased per-stage pools) + hidden send/recv; PP=2 greedy ≡ single-process.
  RequestSnapshot finally extended per the m12 mandate (outputs/sampling/
  num_cached_tokens + allocation aliases). Quantized × TP rejected loudly.
- Refs: `docs/design/m16-distributed.md` (Status: Implemented);
  `kairyu/engine/core/{dist_comm,worker,pp_worker,step_input}.py`,
  `kairyu/models/{parallel,moe_parallel}.py`, `tests/dist/`

### 2026-07-03 — [progress] M15 complete: Qwen3-MoE and DeepSeek-V3 with full parity
- What: 530 → 547 tests. Sparse MoE blocks (Qwen3 softmax top-k with fp32
  routing; DeepSeek sigmoid + correction-bias grouped top-k matched exactly —
  bias affects selection only, top-2 group scores, +1e-20 renorm eps,
  routed_scaling on routed only, shared experts, first_k_dense_replace).
  MlaAttention over the latent pool (post-kv_a_layernorm c_kv ‖ roped k_pe as
  ONE kv head, v width 0 — M18 serde contract), q-LoRA and plain-q paths,
  INTERLEAVED rope (DeepSeek default; half-split is wrong), decompress form
  for prefill / absorbed for decode, HF's hardcoded 1e-6 MLA norm eps. yarn
  rope (inv_freq ramp + attention factor + mscale_all_dim² softmax scale).
  Config: dual-alias expert counts, MLA head_dim pinned to qk dims (never
  hidden//heads), kv-pool props (1 head, r+d_rope wide, v=0). Flagship gates:
  logits < 1e-4 AND full-engine greedy == hf.generate for Qwen3-MoE and
  DeepSeek-V3 (q_lora int/None, yarn on/off). Fixture note: random tiny gates
  produce near-tied routing that fp32 noise flips (block itself matches to
  1e-9 on identical inputs) — gates scaled for decisive margins.
- Refs: `docs/design/m15-moe-mla.md` (Status: Implemented);
  `kairyu/models/{moe,mla}.py`, `kairyu/models/{config,layers,llama}.py`,
  `kairyu/engine/core/kv_pool.py`

### 2026-07-03 — [progress] M14 complete: quantized checkpoints load and RUN on CPU
- What: 514 → 530 tests. kairyu/quant/ reference implementations with formats
  verified against AutoAWQ/AutoGPTQ/vLLM/compressed-tensors source and LIVE
  Hub safetensors headers: FP8-E4M3 (clamp-before-cast — torch CPU cast is
  non-saturating), INT8 W8A8 (exact int32 accumulation — the GPU kernels'
  bit-exact oracle), AWQ (out-axis nibble ORDER [0,2,4,6,1,3,5,7], no +1),
  GPTQ (sequential in-axis packing, z-1 storage offset, g_idx always), NVFP4
  (low-nibble-even packing, bit-3 sign, fp8 block scales × fp32 global, RNE
  boundaries). QuantizedLinear modules hold packed buffers under checkpoint
  names; forward_fused is the Triton seam (kairyu/kernels/ stubs, gpu-marked).
  Loader: linear_factory hook live, state_dict-based iteration (non-persistent
  buffers excluded), quantized payloads verbatim + assign=True + lm_head
  re-tie. Guards: AWQ non-gemm, GPTQ v2/non-4bit, compressed-tensors FP4
  (different names + inverted scale) all rejected loudly. Flagship gate: all
  five schemes quantize the tiny llama, write HF-format checkpoints, load,
  and generate through the FULL engine on CPU (8-bit ≥50% greedy agreement;
  4-bit non-degenerate at hidden-64).
- Refs: `docs/design/m14-quant-compute.md` (Status: Implemented);
  `kairyu/quant/{fp8,int8,awq,gptq,nvfp4,linear}.py`, `kairyu/kernels/`,
  `tests/gpu/test_quant_kernels.py`

### 2026-07-03 — [progress] M13 complete: AttentionBackend seam + FlashInfer adapter + MLA reference
- What: attention extracted into a swappable seam (501 → 514 tests, all M12
  parity suites unchanged — the extraction is behavior-free). Backends are
  plain objects (never nn.Module; state_dict safety), ONE instance shared
  across layers (FlashInfer workspace/plan-cache is per-instance).
  FlashInfer adapter written locally with the reviewed API pins (head_dim_qk
  spelling, workspace buffers, explicit q/kv dtypes, int32 host/device index
  arrays, bottom-right causal assertion, per-chunk plan cache) — logic
  CPU-pinned against an injected fake module, kernels mirrored in tests/gpu/
  (7 deselected until deploy day). MLA reference math (decompress ≡ absorbed
  ≡ naive oracle at the pinned (d_nope+d_rope)^-0.5 scale; shared single-head
  k_pe; post-RoPE cache layout) — M15's trusted oracle for the highest-risk
  kernel work. Selector: env override + hw-profile kernel tier;
  build_engine_loop(model_path=) picks the backend from probe() — deploy day
  is config-free.
- Refs: `docs/design/m13-attention-backend.md` (Status: Implemented);
  `kairyu/engine/core/attention/{__init__,torch_backend,mla_torch,flashinfer_gpu,selector}.py`,
  `tests/gpu/test_flashinfer_gpu.py`

### 2026-07-03 — [progress] M12 complete: real dense models with transformers parity
- What: all five m12 phases landed (471 → 501 tests, 95% cov). ModelConfig
  parses both config.json generations; DenseDecoder (HF-exact module tree)
  covers Llama-3.x / Qwen2 / Qwen3 with verified numerics (rotate_half RoPE,
  llama3 scaling, Qwen3 per-head qk-norm, rectangular chunk masks — SDPA
  is_causal measured wrong over cached prefixes); layer-major PagedKVPool;
  PagedModelRunner behind the m8 ModelRunner protocol with the canonical
  state-access contract, KV-write skip below num_cached_tokens, and
  SpeculativeRunner-compatible decode reads. Flagship gates: fp32 logits
  < 1e-4 vs transformers AND full-engine greedy == hf.generate through
  chunked prefill / radix reuse / page-crossing decode / EOS, per arch.
  Loader (tied embeddings mandatory — safetensors omits lm_head; eos LISTS
  from generation_config; quantized checkpoints fail fast until M14);
  KairyuBackend(model_path=) + kairyu-proc model_path (port reported before
  model load). pytest markers gpu/hf_hub/dist (+strict, default-deselected);
  scripts/parity_real_model.py is the opt-in pre-deploy real-model gate.
- Refs: `docs/design/m12-model-zoo.md` (Status: Implemented);
  `kairyu/models/{config,layers,attention,llama,loader}.py`,
  `kairyu/engine/core/{kv_pool,model_runner}.py`, `scripts/parity_real_model.py`

### 2026-07-03 — [progress] M9 complete: the API is truthful (usage, templates, logprobs, n>1)
- What: all five m9 phases landed (437 → 471 tests, 94% cov; goal G6 gates
  P-A1..P-A5 CPU-green). D1 usage truth — GenerationUsage reported by every
  backend (kairyu/proc/mock/openai-passthrough incl. cached_tokens), OpenAI
  include_usage chunk contract exact, batch JSONL outputs truthful. D2 HF Jinja
  chat templates with transformers byte-match parity (trim/lstrip blocks,
  loopcontrols, HF tojson), per-model DeploymentSpec.chat_templates threaded to
  HTTP AND batch identically. D3 logprobs surfaced (TokenLogprob with bytes,
  chunk-choice placement), /v1/completions (legacy four-array logprobs), real
  n>1 via engine sub-request fan-out (seed identity at i=0, cumulative merged
  streams, sibling aborts, prompt counted once). D4 response_format validated
  (400 not crash) + server-level schema-valid-JSON gate with grammar-stop.
  D5 serving_bench: bearer auth, token-granularity TPOT via include_usage with
  labeled chunk fallback, timestamped results JSON.
- Refs: `docs/design/m9-truthful-api.md` (Status: Implemented);
  `kairyu/entrypoints/server/{app,protocol}.py`, `kairyu/entrypoints/chat_template.py`,
  `kairyu/outputs.py`, `kairyu/engine/kairyu_backend.py`, `bench/serving_bench.py`

### 2026-07-03 — [progress] M8 complete: engine CPU core is real (tokens, sampling, spec decode, process split)
- What: all six m8 phases landed (328 → 437 tests, 95% cov). D1 tokenizer seam
  (HF `tokenizers` + incremental detokenizer, SSE-safe stop-string holdback,
  `finish_early` radix-commit path, finish_reason). D2 real sampling
  (SampledToken/StepOutput protocol ripple across every runner and bench;
  grammar-mask-first with xgrammar stop-token termination; raw-logits logprobs;
  sha256+splitmix64 seeding). D3 scheduler multi-token commit (capped spec
  reservation via chunk.num_tokens, capacity degrade-to-1, budget-accurate,
  exact shortfall release via recorded reservation). D4 n-gram
  SpeculativeRunner (overlay-state scoring, spec ≡ greedy pinned with measured
  acceptance > 0, per-request bypass gating). D5 NVFP4/modelopt/INT8 detection,
  HardwareProfile capability matrix + env-record writer, safetensors
  CheckpointReader with get_slice (M16 seam). D6 process split: shared
  `EngineLoop` extracted; ZMQ ROUTER `engine_service` child process (msgpack,
  ephemeral-port pipe handshake) + `kairyu-proc` backend (lazy zmq.asyncio,
  death detection, shutdown escalation, atexit); parity/stop/abort/usage-fields
  pinned across the process boundary. New deps: tokenizers/safetensors ([hf]
  extra), pyzmq/msgpack ([fleet] extra); coverage configured for the spawned
  service.
- Refs: `docs/design/m8-engine-cpu.md` (Status: Implemented);
  `kairyu/engine/{tokenizer,engine_loop,zmq_backend}.py`,
  `kairyu/engine/core/{sampler,sampling_types,spec_runner,hw_profile,weights,engine_service}.py`

### 2026-07-03 — [design] M8 engine-CPU-core designed and reviewed (local-complete program begins)
- What: `docs/design/m8-engine-cpu.md` — real tokenizer/incremental detokenizer
  (toy stays default), real sampling (SampledToken, StepOutput protocol ripple,
  grammar-mask-first, raw-logits logprobs, sha256 seeds), scheduler multi-token
  commit (capped reservation, degrade-not-stall, scheduler-enforced spec
  precondition), n-gram SpeculativeRunner (overlay-state scoring, per-request
  gating), quant/NVFP4 detection + HardwareProfile + safetensors reader, and the
  ZMQ/msgpack API↔engine process split. 3-reviewer panel APPROVE-WITH-AMENDMENTS;
  amendments applied inline (§6): stop-string SSE holdback + `finish_early`
  radix-commit path, step-thread op discipline (fixes a pre-existing add/abort
  race), budget/watermark accounting for spec chunks, loud update() validation.
- Why: The local-complete mandate (implement everything before GPU hardware;
  only measurement/tuning waits) starts with the engine core. Implementation
  milestones M8–M19 continue the m1..m7 numbering and map to roadmap tracks:
  M8/M9→E1-E2/P-A, M10→F1-F2, M11→P-B/P-C/F5, M12–M18→E-track local halves
  (model zoo, attention backends, quant compute, MoE/MLA, gloo/NCCL distributed,
  CUDA-graph/EAGLE seams, KV transport), M19→deploy packaging.
- Refs: `docs/design/m8-engine-cpu.md`; roadmap §4 Track E/P

### 2026-07-03 — [amendment] G2 hardware contract widened to capability profiles (A100+); fleet-scale decisions amended
- What: G2 §7 gains 2026-07-03 amendments: the goal now spans capability profiles
  covering all NVIDIA GPUs from A100 (SM80) onward — original NVLink arithmetic and
  gates A1–A10 stand on NVLink-HBM profiles; the PCIe-GDDR profile (RTX PRO 6000,
  96 GB, no NVLink) uses TP=1/DP as the 70B scaling base and replaces A3–A5 with a
  placement-crossover report; B2/A10 fabric budgets restated against measured link
  rates; the §6 MoE, autoscaling, and H100-only non-goals are lifted. Related
  amendments: m7 D2 no-k8s → k8s as machine layer only (its own revisit triggers fire
  at thousands of GPUs); m5 D4/m7 D6 session-hash affinity → two-step prefix-aware
  placement then KV tiering; m6 D1 static-only topology relaxed (no-Ray stands);
  ClusterSpec coherence-domain cap 2 → 8; m7 D8 no-OTel flipped. Status notes added
  to m5/m6/m7 design docs and `docs/gpu-runbook.md` (§ header note, §6.1 NVLS scoped
  to NVLink profile).
- Why: The product target is an on-prem DC of thousands of GPUs across BOTH fleet
  shapes (8×H100-class NVLink nodes remain possible; the volume fleet is PCIe-only
  RTX PRO 6000, where 96 GB flips the 70B memory arithmetic and PCIe all-reduce
  latency makes TP-first the wrong default), serving all four model classes
  including MoE — the single-hardware, dense-only, static-fleet assumptions no
  longer hold. Original entries are preserved per progress-log rules.
- Refs: `docs/goals/g2-multi-gpu.md` §7, `docs/roadmap.md` §2/§5,
  `docs/design/m5-*.md` / `m6-*.md` / `m7-*.md` status notes, `docs/gpu-runbook.md`

### 2026-07-03 — [design] Master roadmap + goals G4/G5/G6 defined (gap analysis vs frontier serving)
- What: `docs/roadmap.md` — three-track improvement roadmap (E: own-L1 engine to
  SOTA — real runner/sampling/quant per SM, CUDA graphs + EAGLE-3/MTP via a scheduler
  multi-token commit, profile-aware multi-GPU, MoE/EP, frontier MoE over RDMA;
  F: fleet control plane — dynamic ReplicaPool + registry + k8s machine layer,
  prefix/KV-aware routing fed by RadixKV events, NIXL-candidate KV transport + P/D
  pools, DRAM/NVMe KV tiering, tenancy/SLO admission/autoscaling; P: product
  surface — tokenizer-true usage + cached_tokens, HF Jinja chat templates, streaming
  `kairyu-auto` with orchestration-usage/trace disclosure, Open WebUI integration,
  Responses API/embeddings, nightly frontier-API scoreboard). New goal docs:
  `docs/goals/g4-moe-engine.md`, `g5-fleet-scale.md`, `g6-product-surface.md`.
  Grounding research recorded in roadmap §3/§7: Sakana Fugu product facts (GA
  2026-06, orchestration-as-a-model, Responses API, orchestration token accounting,
  no latency win — Kairyu's wedge is orchestration quality at direct-call latency
  plus trace transparency), SM120 kernel-support gotcha list, and the fleet
  control-plane convergence (Dynamo/llm-d/SGLang gateway/Mooncake/AIBrix:
  prefix-cache-aware routing is the top lever; Kairyu's own RadixKV enables native
  KV-event routing; learned multi-model routing is uncovered white space).
- Why: The product goal (Fugu-class orchestration API + chat UI on an on-prem
  multi-thousand-GPU DC, beating Claude/GPT on TTFT/TPOT/goodput) needed a
  comprehensive gap analysis: the engine compute is placeholder, MoE/quant/spec-decode
  paths are absent, the control plane is static, and the API surface cannot yet
  support billing or honest benchmarks. The roadmap sequences the gaps by impact
  (E1+P-A first) while preserving every existing protocol seam.
- Refs: `docs/roadmap.md`, `docs/goals/g4-moe-engine.md`,
  `docs/goals/g5-fleet-scale.md`, `docs/goals/g6-product-surface.md`

### 2026-07-02 — [progress] M7 Phase 5: deployment guide, runbook §9, README — M7 CPU half complete
- What: `docs/deployment.md` (DC topology, security duty split with the managed cloud
  edge, systemd + compose node setup with documented k8s revisit triggers, config
  walkthrough, rolling model-update drill, observability, interconnect sizing, untested
  k8s appendix); `docs/gpu-runbook.md` §9 (production bring-up on real GPUs: real-engine
  compose smoke, affinity/radix hit-rate measurement through the gateway, rolling-update
  and batch-under-load drills on hardware); README M7 row + serving quickstart. With
  this, every CPU-verifiable G3 gate is implemented and tested (328 tests, 95% cov).
- Refs: `docs/deployment.md`, `docs/gpu-runbook.md` §9, README.md;
  m7 status line updated to Implemented — CPU half

### 2026-07-02 — [progress] M7 Phase 4: OpenAI-compatible batch API
- What: `/v1/files` (multipart upload/metadata/content) and `/v1/batches`
  (create/get/list/cancel) backed by a filesystem `BatchStore` (atomic JSON job state,
  JSONL input/output/error files) and an in-gateway `BatchWorker` lifespan task that
  drains jobs through the same served engines/pools under its own semaphore — strictly
  below the server's global concurrency guard, so interactive traffic stays admitted
  (gate C4, pinned by test). Cancel skips remaining lines; restart recovery marks
  in-flight jobs failed with an explicit resubmit message (single-gateway scope, m7 D7).
  Server helpers `sampling_params_from` / `completion_response` made public for reuse.
  New dep: python-multipart (FastAPI form uploads).
- Refs: m7 D7, G3 gate C4; `kairyu/batch/{store,worker}.py`,
  `kairyu/entrypoints/server/batch_routes.py`, `kairyu/deploy/builder.py`;
  tests `tests/server/test_batches.py`

### 2026-07-02 — [progress] M7 Phase 3: container image, compose topology, CI smoke drill
- What: Multi-stage uv `Dockerfile` (one image for every role; the mounted
  DeploymentSpec decides gateway vs replica), `deploy/compose/` (1 gateway + 3 mock
  replicas with healthchecks, gateway/replica YAML configs), `scripts/compose_smoke.sh`
  (readiness → completion → SSE → affinity-by-metrics → replica kill/eject/zero-5xx →
  prober recovery), and a `compose-smoke` CI job separate from the coverage-gated
  pytest job. The full drill was verified end-to-end with the same configs as local
  processes (kill/eject: 10/10 subsequent 200s; prober auto-restore observed in the
  gateway JSON log); the container build itself runs in CI — this dev environment's
  network policy blocks registry CDNs.
- Refs: m7 D1/D2, G3 gates C1–C3; `Dockerfile`, `deploy/compose/`,
  `scripts/compose_smoke.sh`, `.github/workflows/ci.yml`

### 2026-07-02 — [progress] M7 Phase 2: `kairyu serve` CLI, DeploymentSpec, pool wiring, prober, HTTP affinity
- What: `kairyu serve <deployment.yaml>` console entrypoint builds gateway or replica
  from one YAML: `DeploymentSpec` (new, composes with — does not extend — ClusterSpec,
  m7 D3) declares engines, pools (N remote `openai` members, keyless node-to-node),
  server settings, optional DSL orchestrator, batch section. Builder wraps pool members
  in `ReplicaPool` and passes it into `create_app` unchanged (the pool IS an
  EngineBackend); lifespan starts a `HealthProber` per pool (GETs ejected replicas'
  `/health`, restores via existing `probe()`) and shuts engines down gracefully.
  HTTP affinity gap closed: OpenAI `user` field / `X-Session-ID` header now map to
  `CacheHint(session_id=...)`, so external multi-turn traffic reaches the radix-KV
  warm replica (previously cache_hint was never set on the HTTP path).
- Refs: m7 D3/D4/D6; `kairyu/deploy/{spec,builder,prober}.py`,
  `kairyu/entrypoints/cli.py`, `kairyu/entrypoints/server/{app,protocol}.py`;
  tests `tests/unit/test_{deployment_spec,prober,cli}.py`,
  `tests/server/test_serve_builder.py`

### 2026-07-02 — [progress] M7 Phase 1: server hardening landed
- What: `/health`, `/readyz` (pool-aware: 503 unless every ReplicaPool has ≥1 healthy
  replica), `/metrics` (per-app Prometheus registry; request counts/latency histograms,
  scrape-time pool collector for outstanding/health/decision counts), optional static
  API-key auth (env-sourced, constant-time, health endpoints exempt), global concurrency
  guard (429 + Retry-After on /v1/*), JSON access log with X-Request-ID — all pure-ASGI
  middleware so SSE streams hold their concurrency slot to the last byte. `create_app`
  gains an optional `ServerSettings`; defaults preserve pre-M7 behavior. `ReplicaPool`
  gains read-only `healthy`/`replica_count`/`decision_counts` accessors (still no
  background tasks, m5 D4). New dep: prometheus-client.
- Refs: m7 D4/D5/D8; `kairyu/entrypoints/server/{health,metrics,middleware,settings}.py`,
  `kairyu/orchestration/replica.py`; tests `tests/server/test_{health_metrics,auth,limits}.py`

### 2026-07-02 — [design] M7 productionization designed (G3 goal, D1–D8); G2 2-node scope clarified
- What: Wrote `docs/goals/g3-production-deployment.md` (gates C1–C7) and
  `docs/design/m7-productionization.md`. Decisions: D1 on-prem-DC topology (managed
  cloud WAF/LB front → private interconnect → stateless CPU gateway tier running
  `create_app` + Orchestrator + ReplicaPool of remote `openai`-backend replicas → N GPU
  replica nodes running the same artifact); D2 no Kubernetes — systemd + docker compose
  with everything containerized and documented k3s revisit triggers; D3 new
  `DeploymentSpec` (ClusterSpec untouched — it binds the TP/PP coherence domain, not
  fleet size); D4 health/readyz/metrics + serve-layer background prober (pool stays
  passive); D5 edge-owned WAF/TLS, gateway static API keys + concurrency guard, keyless
  node-to-node; D6 cache layer = per-replica radix KV + pool session affinity, no Redis
  (revisit trigger recorded) — includes fixing the gap that the HTTP path never set
  `cache_hint`; D7 minimal filesystem-backed `/v1/files` + `/v1/batches`; D8
  prometheus-client + stdlib JSON logs, no OTel.
- Why: The product-infrastructure review (LB/scaling, WAF, k8s, GPU pool + API layer,
  cache layer, batch orchestrator, DC–cloud interconnect) found all deployment
  machinery absent: components exist in-process (ReplicaPool, remote-replica backend)
  but nothing wires, launches, secures, observes, or packages them.
- Refs: `docs/goals/g3-production-deployment.md`, `docs/design/m7-productionization.md`;
  amendment: g2 §6 "exactly 2 nodes" clarified as TP/PP coherence-domain cap, not a
  ReplicaPool fleet-size cap (`docs/goals/g2-multi-gpu.md` §6, G3 §5).

### 2026-07-02 — [progress] Repo renamed to `ytworks/kairyu`; README refreshed for M5/M6
- What: GitHub repository renamed from `ytworks/rLLM` to `ytworks/kairyu` (local origin
  updated). README brought up to date: M5/M6 rows in the roadmap, TP / P-D / KV-transport /
  PP components in architecture, engine-core and project-layout sections, test count 290+,
  clone URL. Remaining `rLLM` references in CLAUDE.md / AGENTS.md / gpu-runbook fixed.
- Refs: README.md, CLAUDE.md, AGENTS.md, docs/gpu-runbook.md

### 2026-07-02 — [progress] M5/M6 GPU-independent halves implemented (177 → 289 tests)
- What: All CPU-testable pieces of both designs landed with tests (95% coverage):
  M5 — `Communicator`/`FakeCommunicator`, typed immutable `StepInput`, `TPModelRunner`
  (divergence-checked driver protocol; TP=2 greedy-equivalent to TP=1 through
  KairyuBackend), `tensor_parallel_size` plumbed end-to-end (no-op resolved),
  `ReplicaPool` (rendezvous-hash affinity, queue-depth valve, health ejection,
  `record_replica` JSONL), `PDCoordinator` + `LocalKVHandoff` + `Scheduler.resume_with_kv`
  (copy-before-commit ordering, preemption shield, P-D greedy-equivalence).
  M6 — `ClusterSpec` (topology validation), `KVTransport` protocol + `LocalFabric` +
  TCP-loopback transport, `bench/kv_transfer_bench.py` (CPU-runnable, real fragment
  layout), `openai_backend` replica fixes (real SSE streaming, pooled client, optional
  auth, token counts), async submit/handle runner contract + `PipelinedModelRunner`/
  `PipelinedEngineCore` (inter-step pipelining, bubble accounting: depth-2 <0.2 vs
  depth-1 ≈0.5 pinned by test). GPU-runbook §6/§7 added for the GPU days.
- Refs: `kairyu/engine/core/{comm,step_input,tp_runner,pd,kv_transport,pipeline}.py`,
  `kairyu/orchestration/{replica,cluster}.py`, `docs/gpu-runbook.md` §6–7

### 2026-07-02 — [design] M5/M6 designs written and reviewed (APPROVE-WITH-AMENDMENTS)
- What: `docs/design/m5-intra-node-parallelism.md` (TP runner with non-rank driver +
  typed StepInput prerequisite, ReplicaPool with session affinity, P-D copy-on-handoff
  with copy-before-commit protocol and `resume_with_kv`) and
  `docs/design/m6-inter-node-parallelism.md` (static ClusterSpec — no Ray, KVTransport
  with fragment aggregation, streamed P-D with layer-group final chunk, PP=2 via
  inter-step pipelining on an async ModelRunner handle). Three-reviewer agent panel
  fixed 6 blockers: zero-copy P-D donation withdrawn (dual-tree pool accounting unsound
  + incompatible with disjoint-GPU roles); PP intra-step micro-batching withdrawn
  (bounded at ~1.33× vs B4's 1.6×); `openai_backend` "no change" claim corrected (fake
  streaming, per-request client, mandatory auth, empty token_ids all block B1).
- Why: G2 requires reviewed design docs before implementing each milestone; review
  against the real scheduler/radix code caught mechanisms that could not work as drafted.
- Refs: `docs/design/m5-intra-node-parallelism.md` §7, `m6-inter-node-parallelism.md`
  §7, `docs/goals/g2-multi-gpu.md` §7 Amendments

### 2026-07-02 — [amendment] m1 "Ray arrives with multi-node" superseded
- What: M6 D1 uses a static ClusterSpec + torchrun-style rendezvous for the 2-node
  topology; Ray is not adopted.
- Why: G2 excludes elasticity; a dynamic-placement framework for two static nodes fails
  YAGNI. m1 §3/D4's note was forward-looking, not a binding decision.
- Refs: `docs/design/m6-inter-node-parallelism.md` D1;
  `docs/design/m1-orchestration-and-interface.md` §3

### 2026-07-02 — [design] Multi-GPU goal (G2) defined — drives M5/M6
- What: Wrote `docs/goals/g2-multi-gpu.md`, the acceptance contract for intra-node
  (M5: TP, DP replicas via L2 Router, P-D intra-node) and inter-node (M6: 2-node DP,
  page-granular KV transfer plane, P-D inter-node, PP=2) multi-GPU serving.
  Targets: Llama-3.3-70B FP8 on 8×H100 + 2 nodes (IB/RoCE); vLLM-parity-or-better plus
  absolute scaling-efficiency gates (A1–A10, B1–B5); TP=2 is the scaling base (70B FP8
  cannot run TP=1). MoE/expert parallelism is an explicit non-goal.
- Why: Multi-GPU support existed only as a no-op `tensor_parallel_size` arg and the
  P-D admission-policy half; M5/M6 need a G1-style evidence-first goal to drive
  autonomous development. `docs/goals/` created since the original G1 goal was never
  filed as a document.
- Refs: `docs/goals/g2-multi-gpu.md`; seams: `kairyu/engine/core/engine_core.py`
  (ModelRunner), `scheduler.py` (`pd_separation`), `kairyu/orchestration/router.py`

### 2026-07-02 — [progress] Design-change memory harness added
- What: Added PROGRESS.md, `.claude/rules/progress-log.md`, CLAUDE.md, and AGENTS.md so
  Claude Code and Codex sessions share the same record of design changes and progress.
- Why: Design decisions were scattered across design docs, review-amendment commits, and
  session context; new sessions had no single place to recover project state.
- Refs: `.claude/rules/progress-log.md`

### 2026-07-02 — [progress] README enriched; GPU-day runbook added
- What: README expanded with architecture, roadmap, usage guides, and open-model setup
  (Kimi, Qwen). GPU-day runbook consolidates all remaining GPU-gated work into ordered,
  gated execution steps.
- Refs: commits 9d35360, cc45b08; `docs/gpu-runbook.md`

### 2026-07-02 — [progress] xgrammar structured output integrated (M3)
- What: Token-bitmask enforcer and `response_format` plumbing through the engine and
  OpenAI server.
- Refs: commits ad4e18c, a0851f9; `docs/design/m3-spec-decode-and-graphs.md`

### 2026-07-02 — [progress] Engine core validated end-to-end on CPU (M2 CPU half)
- What: `kairyu` EngineBackend exposed and wired through the OpenAI server; paged-KV
  attention proven greedy-equivalent with real torch tensors on CPU; pre-GPU robustness
  items landed (EOS under overlap, preemption, watermark, abort, pin TTL).
- Refs: commits 991832b, aa382f8, e977e1c; `docs/design/m2-engine.md`

### 2026-07-02 — [amendment] Design-review amendments applied (M2/M4)
- What: Compute-skip, computed gating, output caching, and bandit-router fixes applied
  across M2 KV/scheduler and M4 learning pipeline, per the agent design-review panel.
- Why: Review found gaps in the original D-decisions; docs updated with APPROVE-WITH-
  AMENDMENTS status and amendment sections (§5/§6).
- Refs: commits c14f035, 22b0b53; `docs/design/m2-engine.md` §6,
  `docs/design/m4-router-learning.md` §5

### 2026-07-02 — [design] M2–M4 designs written and reviewed; M4 pulled forward
- What: Design docs for M2 (overlap scheduler + Radix-Paged KV), M3 (spec decode, CUDA
  graphs, P-D separation), M4 (router learning) written and agent-review-approved with
  amendments. M4 was pulled ahead of schedule because it is GPU-independent.
- Why: M2's remaining half needs GPU; GPU-independent work (M3 CPU-side, M4) proceeds
  first to keep momentum.
- Refs: `docs/design/m2-engine.md`, `docs/design/m3-spec-decode-and-graphs.md`,
  `docs/design/m4-router-learning.md`; commits d2675a7, c976c64, 4d83229

### 2026-07-02 — [progress] M1 complete: orchestration + vLLM-compatible interface
- What: L2 orchestration (rule-based Router with JSONL decision log, Conductor role-DAG
  with verifier-gated refinement, MoA, immutable budget) and L3 interface
  (vLLM-signature `LLM`, `AsyncLLMEngine`, OpenAI-compatible server with SSE streaming
  and tool calls, YAML/decorator DSL) built on the `EngineBackend` protocol
  (mock / vLLM / external-OpenAI backends).
- Refs: commit 633fa37; `docs/design/m1-orchestration-and-interface.md`
