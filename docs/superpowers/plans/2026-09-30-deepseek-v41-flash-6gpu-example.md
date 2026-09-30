# DeepSeek V4.1 Flash single-replica example on six GPUs

Status: **Approved 2026-09-30 and implemented; see Outcome.**

Accepted scope (owner, 2026-09-30):

- Only six GPUs are used (GPUs 0–5 by default). GPUs 6–7 are not part of
  the example.
- L2 and L3 are the `deepseek-v4.1-flash-8gpu` structure, unchanged: one
  public model backed by one `ReplicaPool` replica, the OpenAI-compatible
  API, and Open WebUI. No orchestration DAG, no judge, no second model.
- Fresh branch `claude/deepseek-v41-flash-6gpu` from `main`; separate PR.
  Nothing is reused from the unmerged `claude/v41-tiered-six-gpu` branch.
  Its measured findings (and those of closed PR #598) are inputs only.
- "Optimal" means: start from the official settings, deviate only where
  this hardware forces it or a bounded measurement shows a gain, and write
  down every deviation with its reason.

Example directory: `examples/deepseek-v4.1-flash-6gpu/`. Ports: API 8008,
Chat UI 3008 (8007/3007 are already taken by two examples).

## Sources (official first)

1. Model author: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
   (revision `dba1be0a40aa45a94ad051997016db3960a90277`, still `main` on
   2026-09-30) and its `encoding/README.md`.
2. vLLM recipe: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash,
   source `vllm-project/recipes` `models/deepseek-ai/DeepSeek-V4.1-Flash.yaml`
   at `1343d0f1` (`date_updated: 2026-09-29`).
3. Kairyu evidence: `examples/deepseek-v4.1-flash-8gpu/MEASUREMENTS.md`
   (TP8/EP8 on this host), and the six-GPU candidate table on
   `claude/v41-tiered-six-gpu` (`MEASUREMENTS.md`, V41T-D1).

When sources 1 and 2 disagree, the model author wins (same rule as the
8-GPU example).

## What the official sources say, and what applies to 6 × RTX PRO 6000

| Topic | Official | Applies here? |
|---|---|---|
| Memory | Checkpoint 476 GiB: experts 259.5, Engram 183.1, scales 21.9, dense 6.9, embed/head 3.9. `vram_minimum_gb: 614`. | 6 × 96 GB = 576 GB < 614: below the official minimum, like the recipe's 8 × H100 arm (640 GB). |
| Memory-bound arm (H100) | Engram `cpu_offload: true` (tables in pinned host DRAM via UVA, "output is unchanged"), `--max-num-batched-tokens 4096` (indexer logits buffer = batched × max-model-len × 2 B), `--gpu-memory-utilization 0.92`, `VLLM_USE_V2_MODEL_RUNNER=1`, `expandable_segments:True`. | Yes. Engram offload is mandatory (host has 1 TB; ≈190 GiB pinned). The batched-token / memory-utilization / V2-runner levers are candidates, not givens: the tiered branch measured 21.4 GiB KV per GPU at 16,384 on the older image. |
| Blackwell TP | TP2, `FLASHINFER_MLA_SPARSE_DSV41`, `indexer_kv_dtype: mxfp4`, `indexer_sparse_logits: true`, `--kv-cache-dtype fp8`, `--max-num-seqs 128`, Engram offload. "For high interactivity use TP4." | TP2 yes. TP4 cannot use six GPUs; TP6 cannot divide 64 heads / 8 output groups. `indexer_sparse_logits` does not exist in the pinned `0.1.dev20904` image (checked). |
| Blackwell DEP | TP1 × DP-N + EP, `FLASHMLA_MEGA_ATTN_DSV41`, DeepGEMM mega MoE, `engram-config {"embedding_across_dp": true}`. | The mega kernels target SM100-class parts; the FlashMLA sparse backend already failed on SM120 (no 64-token block, tiered candidate 5). Tried once as a bounded candidate, expected to fail at startup. |
| Replica shape for 6 GPUs | Not covered (recipe arms are 1/2/4/8 GPUs). | Derived: **TP2 × attention-DP3 with EP6**. 384 routed experts / 6 = 64 per rank. TP2 pairs land on NUMA-local GPU pairs (0,1)(2,3)(4,5) (`nvidia-smi topo`: NODE inside a pair, SYS across). |
| DSpark | Only speculative method for V4.1; k = 5, probabilistic draft, block rejection, adaptive verification **on** for NVIDIA. | Blocked on EP6: the drafter has 128 experts per stage and `vllm/models/deepseek_v4/nvidia/model.py:915` asserts `n_physical_experts % ep_size == 0` (redundant experts need EPLB, which would also have to keep 384 + r divisible by 6). Bounded investigation only (L1-6 below). |
| CUDA graphs | `--max-cudagraph-capture-size` must cover `max-num-seqs × (1 + k)`. | Set from the selected `max-num-seqs` (k = 0 while DSpark is off). |
| Vision | Keep the ViT. `--mm-encoder-tp-mode data` is the official option for multi-image TTFT. | Keep images (L3 parity with 8-GPU). Encoder DP is a candidate. |
| Effort | Model author: `low`=50, `high`=75, `max`=100, **default `high` (75)**. The recipe's 25/50/75/100 mapping is the vLLM encoder's. | Model author wins: default thinking high = 75, 50/75/100 verified by `/tokenize`, exactly as in the 8-GPU example. |
| Sampling | Model card: temperature 1.0, top_p 0.95 or 1.0. Published results use 1.0 / 0.95. | Default 1.0 / 0.95 (the published-results setting), provided the top-p gate in L1-0 passes; otherwise 1.0 / 1.0 (the other official value) with the reason recorded. |
| Output budget | ≥ 256K tokens for evaluation; `max` effort for hard tasks. | API accepts caller limits up to the 1M context; Chat UI default stays 32,768 as in the 8-GPU example. |
| Image | `vllm/vllm-openai:nightly` from 2026-09-10 on (`#56228`); Python frontend if the Rust one misbehaves. | Candidate base images in L1-0. Python frontend kept (8-GPU example: the Rust frontend's effort aliases differ from the model author's). |

## L1 plan (vLLM on GPUs 0–5)

Baseline command (every flag traceable to the table above):

```
--tensor-parallel-size 2 --data-parallel-size 3 --enable-expert-parallel
--engram-config '{"cpu_offload":true}'
--attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4"[,"indexer_sparse_logits":true]}'
--kv-cache-dtype fp8 --block-size 64 --moe-backend marlin
--max-model-len 1048576 --max-num-seqs 128 --max-num-batched-tokens 16384
--gpu-memory-utilization 0.90 --enable-prefix-caching
--tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41
--enable-auto-tool-choice --tool-call-parser deepseek_v41
--default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"high"}'
--limit-mm-per-prompt '{"image":8}'
```

`--block-size 64`, `--moe-backend marlin` and the SM120 FlashInfer build are
the 8-GPU example's measured SM120 requirements, not recipe values. Each DP
rank's container cpuset follows its NUMA node.

Stages run in order. Each has a pass rule and a time box; a failure is
recorded verbatim in `MEASUREMENTS.md` and the stage moves on.

- **L1-0 Runtime image (time box: 1 GPU day).** Candidates, first passing
  wins:
  (a) a current `vllm/vllm-openai:nightly` pinned by digest plus this
  example's own SM120 FlashInfer overlay (it carries `indexer_sparse_logits`
  and upstream fixes since 09-09);
  (b) the 0909-era base (`0.1.dev20904`) with this example's own overlay;
  (c) (b) plus the masked-KV and seeded top-p fixes the tiered branch found
  necessary for EP6 (re-derived and SHA-pinned in this example).
  Every overlay file is this example's own; no image or file is taken from
  another example.
  Pass rule = the correctness gate: SM120 kernel checks
  (`check_sm120_pages.py`, `check_sm120_indexer.py`), rendered effort
  50/75/100 and chat mode via `/tokenize`, 12 concurrent `17 * 19` probes
  per mode reaching all three DP ranks with exact `323` and finite
  log-probabilities, a top-p 0.95 thinking probe that closes `</think>`,
  a tool call, an image answer, and cancellation that returns the running
  and waiting gauges to zero. The tiered branch showed that the older image
  without (c)'s fixes returns NaN / garbage on every EP6 shape, so
  first-request success is never accepted as evidence.
- **L1-1 Baseline numbers.** Fixed 8K-in / 256-out matrix at c1/8/16/32/64
  (32 requests per row), plus completed-answer rows for the generic and
  coding datasets at c1/8/32. Record weights, KV per GPU, KV tokens per DP
  engine, GPU peaks and pinned host memory.
- **L1-2 Scheduler (official tuning order: one parameter at a time).**
  `max-num-batched-tokens` 16384 vs 8192 vs 4096 (H100 arm);
  `gpu-memory-utilization` 0.90 vs 0.92 (H100 arm); `VLLM_USE_V2_MODEL_RUNNER=1`;
  `max-num-seqs` 128 vs 64 if 128 shows queueing without throughput gain;
  `max-cudagraph-capture-size` covering the selected `max-num-seqs`.
- **L1-3 Vision encoder.** `--mm-encoder-tp-mode data` vs default on a
  2- and 8-image TTFT probe.
- **L1-4 DEP6 (official DEP kernels).** One startup attempt of TP1 × DP6 with
  `FLASHMLA_MEGA_ATTN_DSV41`, DeepGEMM mega MoE and `embedding_across_dp`;
  then TP1 × DP6 with the baseline attention/MoE backends and Engram
  offload (tiered candidate 3, re-run on the L1-0 image). Adopt only if it
  passes the L1-0 gate and beats TP2 × DP3 under the selection rule.
- **L1-5 Memory headroom.** A 1,048,576-token prompt must fit (tiered
  measured 8.13× at 16K batching); four long-context retrieval probes up to
  ≈1M tokens.
- **L1-6 DSpark investigation (time box: 1 GPU day, owner decides beyond
  it).** Establish from source whether the drafter can run without EP
  (replicated or TP-only draft experts) under the selected image. Only a
  configuration-level path is tried. A runtime patch is proposed to the owner
  with its size first, not applied on my own. Official reference point:
  8-GPU DSpark gave 1.91× c1 throughput.

**Selection rule.** A candidate replaces the current choice only if it
passes the full L1-0 correctness gate and improves c1 output tok/s or c32
output tok/s by ≥ 5 % without regressing the other or c8 TTFT p50 by more
than 5 % (single trials; smaller differences are reported as no change, as
in the 8-GPU example). The official value is kept when there is no
measurable difference.

## L2 / L3 (copy of the 8-GPU structure)

- `kairyu.yaml`: pool `deepseek-v4.1-flash` with one replica, `upstream:
  vllm`, `queue_depth_threshold: 0`, `prefix_index: true`, placement log;
  replica metadata records `tensor_parallel_size: 2`,
  `expert_parallel_size: 6`, `attention_data_parallel_size: 3`,
  `dspark_enabled` as selected, image digest, and model revision.
- Server `max_concurrency` and `admission_wait_timeout_s` sized from the
  selected `max-num-seqs` (8-GPU: 64 / 600 s); image policy and 1M
  `max_processed_prompt_tokens` unchanged.
- Open WebUI with the effort dropdown (default/low/high/max → high default)
  and the same filter.
- Public model name: `deepseek-v4.1-flash` (the two examples never run
  together: they share GPUs 0–5).

## Files and code ownership

- New: `examples/deepseek-v4.1-flash-6gpu/{README.md, MEASUREMENTS.md,
  example.json, kairyu.yaml, compose.yaml, run.sh, verify.sh}`.
- Lifecycle, verification, benchmark and tuning scripts are this example's
  own (owner, 2026-09-30: no script sharing between examples). The example
  writes its own `control.py`, `verification.py`, `benchmark.py`, `tune.py`,
  runtime overlay (Dockerfile / patch scripts / SM120 kernel checks) and
  Chat UI filter, written for the TP2 × DP3 / EP6 topology from the start
  (per-DP-rank NUMA cpusets, three-rank probe coverage, Engram host-memory
  check). No file under another example is imported, referenced or
  modified; the 8-GPU example is read for reference only.
- Kairyu (`kairyu/`) is not changed. The native engine's `ep_size ∈
  {1,2,4,8}` limit (`kairyu/models/deepseek_v4.py:193`) is why L1 stays on
  vLLM; lifting it is out of scope.
- Tests (CLAUDE.md test policy): one example-owned test file,
  `tests/unit/test_deepseek_v41_6gpu_example.py`, covering only behaviour
  with a concrete failure mode: DP-rank → GPU pair → NUMA cpuset mapping,
  fail-closed runtime patch anchors, and readiness rejecting wrong or
  non-finite answers. Shared example tests are not extended. No tests that
  restate `example.json` contents.
- Docs: FN-D9 V4.1 six-GPU amendment in `docs/design/frontier-native-runtime.md`
  (topology derivation, official deviations, selected L1); `examples/README.md`
  row; `PROGRESS.md` Current Status + Change Log entry.

## Final gates (on the selected configuration)

`verify.sh` serving (c1/8/16/32/64 × 64, placements all on replica 0),
tool-calling, vision, reasoning (default / explicit / chat mode),
cancellation, long-context (four probes to ≈1M), and a normal restart.
Evidence, run IDs, hashes and limitations go in `MEASUREMENTS.md`.

## Checklist

- [x] Owner approves this plan.
- [x] Branch, example files and example-owned scripts, CPU tests, lint.
- [x] Release GPUs 0–5 (they were idle; nothing was stopped).
- [x] L1-0 … L1-6 in order; record every candidate.
- [x] Final gates; MEASUREMENTS.md; FN-D9 amendment; PROGRESS.md.

## Outcome and deviations from this plan

- L1-0: candidate (a), the pinned nightly, passed first and was adopted;
  (b) was not re-run (the tiered branch already showed it failing on EP6),
  (c) passed the kernel gate but was not needed.
- The replica shape changed from the planned TP2 × DP3 baseline to DP6 /
  EP6: L1-4's DP6 row (with this example's SM120 kernels, not the official
  SM100-only DEP kernels) beat TP2 × DP3 by 44–47 % at c32.
- L1-6: DSpark needed no runtime change. The drafter's 128 experts load
  under EP6 on the fused-MoE path (only the mega-MoE path asserts
  divisibility — the earlier plan text was wrong). Adaptive verification is
  rejected by the V4.1 indexer backend; full verification needs the recipe's
  4K / 0.92 memory levers on DP6 and was adopted.
- `indexer_sparse_logits` and the official DEP kernels fail on SM120; the
  encoder-DP option and V2 runner showed no measurable difference.
- One L1 container cannot pin each DP rank to its own NUMA node; its cpuset
  is the union of the nodes of GPUs 0–5.
- The unused 0909 patch profile was removed before the final gates, so the
  final image differs from the tuning image only by that script content.
