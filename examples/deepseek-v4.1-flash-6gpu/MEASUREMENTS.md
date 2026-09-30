# deepseek-v4.1-flash-6gpu evidence

Status: **all final gates PASS on the committed configuration (2026-09-30).**

Hardware: 6 × NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB,
SM120, PCIe) out of an 8-GPU host; 1 TiB host memory, 4 NUMA nodes with
GPU pairs (0,1)(2,3)(4,5)(6,7) each on one node.
Model: `deepseek-ai/DeepSeek-V4.1-Flash` revision
`dba1be0a40aa45a94ad051997016db3960a90277`, tree SHA-256
`d21211ca29ad7eba1fda84e49b1a34a73214ec9b84b23928cde63902c3318bfd`.
Artifacts live under
`/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-flash-6gpu/`.

## SM120 kernel gate (`check_sm120_kernels.py`)

| Image | Result |
|---|---|
| nightly overlay, first build (FlashInfer prebuilt JIT caches still installed) | **FAIL**: `non-finite attention output (poisoned=True, extra_page=32)` — the prebuilt `flashinfer-jit-cache-*` packages (sm120f among them) shadow the patched sources. The Dockerfile now removes every JIT cache. This run's log was overwritten by the re-run; the failure line is quoted from the session. |
| nightly overlay (vLLM `0.30.1rc1.dev396+gac68c3087`, FlashInfer `0.7.0.post1`) | PASS 28/28 |
| 0909 overlay (vLLM `0.1.dev20904+g179dd0fa9`, FlashInfer `60b49158`) | PASS 28/28 |

The first test revision drew unit-normal inputs and failed 2/4096 elements
of an unmasked 0909 case by 0.013 over the 0.05 tolerance; the gate now
draws inputs as FlashInfer's own DSV4 tests do (`randn / 10`, clamped), to
which that tolerance belongs, keeping per-tile scale variation. Logs:
`kernel-checks/`.

## L1-0 runtime image

Candidates in the plan's order; the first that passes the full correctness
gate is adopted.

| # | Image | Result |
|---|---|---|
| a | `vllm/vllm-openai:nightly@sha256:b5406fca…` (vLLM `0.30.1rc1.dev396+gac68c3087`, FlashInfer `0.7.0.post1`) + this overlay; image `sha256:0a209afd6593376cb9d6d0b670ade745ce69e6ebd558f140696f249ed612ba77` | **ADOPTED** — kernel gate 28/28; `run.sh up` readiness; `verify.sh l1` PASS 13/13 (run `candidate-a-nightly-r2-l1`) |
| b | 0909 base unpatched | not re-run: the same base without the masked-KV fix returned NaN log-probabilities / unrelated text on every EP6 shape on this host (branch `claude/v41-tiered-six-gpu`, `MEASUREMENTS.md` candidates 1–4) |
| c | 0909 base + this overlay (`local/vllm-openai:deepseek-v41-6gpu-0909`) | kernel gate 28/28; not needed after (a) passed |

The nightly already carries the model author's effort mapping upstream
(`low` 50, `high` 75, `max` 100, plus `xhigh` 75); the overlay only checks it.
It still lacks every SM120 edit in `patch_sm120.py`.

Startup (TP2 × DP3, EP6, Engram offload, memory 0.90, 16K batched, 128
sequences): weights 52.62 GiB per GPU (≈117 s load), 19.85 GiB KV per GPU,
**7,903,437 KV tokens per DP engine** (7.54 × a 1,048,576-token request),
three engines and three API servers; idle GPU 0–5 at 87.5 GiB. vLLM reports
the L1 default sampling `{'temperature': 1.0, 'top_p': 0.95}`.

`verify.sh l1` (`candidate-a-nightly-r2-l1`): rendered efforts default 75,
low 50, high 75, max 100, chat mode closes thinking with no budget; 12
concurrent `17 * 19` probes for each of default/low/high/max/chat all return
exactly `323` with finite log-probabilities, and every variant reached all
three DP engines (per-engine `vllm:request_success_total`); tool call and
image pass. The first run (`candidate-a-nightly-l1`) failed only the
sampling-default check because this vLLM words the log line differently; the
regex was corrected and the gate re-run.

## L1-2 stage 1: one parameter at a time on TP2 × DP3

Run `20260930T085255Z-tuning` (`tune.py`; an earlier start
`20260930T085242Z-tuning-aborted` was stopped before any row because its
shell had a 2-hour limit). Each candidate restarts L1 with one change,
passes the readiness probes (exact finite answers, tool call, image), then
runs 32 requests of ≈8K unique input and exactly 256 output tokens (thinking
high, reasoning tokens counted). Cells: output tok/s / model TTFT p50 ms /
TPOT p50 ms. Single trials.

| Candidate | c1 | c8 | c32 | KV tokens per DP engine | Result |
|---|---|---|---|---|---|
| baseline (TP2 × DP3, 16K batched, 0.90, 128 seqs) | 58.1 / 955 / 13.2 | 222.1 / 4,249 / 17.0 | 352.9 / 10,412 / 39.8 | 7,903,437 | PASS |
| batch 8K | 58.6 / 967 / 13.3 | 229.4 / 4,239 / 20.3 | 382.6 / 8,754 / 44.1 | 11,795,398 | PASS |
| batch 4K (official 8 × H100 arm) | 57.2 / 987 / 13.8 | 237.9 / 3,051 / 20.9 | 383.3 / 8,409 / 47.5 | 13,904,321 | PASS |
| memory 0.92 (official 8 × H100 arm) | 58.5 / 961 / 13.5 | 218.7 / 4,410 / 17.1 | 384.2 / 8,709 / 39.2 | 8,659,622 | PASS |
| V2 model runner (official 8 × H100 arm) | 58.9 / 953 / 13.2 | 218.1 / 4,549 / 17.0 | 381.1 / 8,717 / 39.7 | 7,903,437 | PASS |
| 64 sequences | 58.5 / 951 / 13.5 | 211.5 / 4,783 / 20.2 | 383.6 / 8,719 / 39.3 | 8,298,426 | PASS |
| `indexer_sparse_logits` (official Blackwell) | — | — | — | — | **FAIL** at model init: `SparseMQAIndexer requires an SM100-class GPU and DeepGEMM >= 2.8` |
| official DEP6 (`FLASHMLA_MEGA_ATTN_DSV41`, DeepGEMM mega MoE, `embedding_across_dp`) | — | — | — | — | **FAIL** at model init: `nvfp4_ds_mla needs the V4.1 KV records, which FlashMLA decodes only on SM100` |
| **DP6 / EP6 (TP1) with this example's SM120 backends** | **58.8 / 603 / 14.7** | **248.5 / 2,505 / 22.2** | **564.3 / 5,578 / 34.8** | 2,930,708 | PASS |

Reading: every restarted candidate lands at 381–384 tok/s at c32, so the
baseline's 352.9 (measured on the container that had just served the
readiness probes, not restarted) is taken as single-trial variation, not a
5 % effect of those candidates; stage 2 re-measures a restarted baseline.
DP6 is far outside that variation (c32 +47 %, c1 TTFT −37 %, c8 TTFT −41 %,
c1 throughput unchanged) and is adopted as the replica shape. It holds
57.37 GiB of weights and 7.36 GiB of KV per GPU (2.79 × a 1M-token request
per engine). The official Blackwell kernels behind `indexer_sparse_logits`
and DEP are SM100-only; DEP6 here uses the TP path's SM120 kernels.
Logs: `<run>/<candidate>/{startup.log,worker.log,c*/row.json}`.

## L1-2 stage 2: one parameter at a time on DP6 / EP6

Run `20260930T102844Z-tuning`, same rows as stage 1, plus the median of five
8-image prompts (1024 × 1024 noise PNGs, 16 output tokens) as `img8` TTFT.
`verify.sh l1` on DP6 before this stage: PASS 13/13 (`dp6-r2-l1`; 72 probes,
all six engines answered every variant). The first run (`dp6-l1`) sent one
12-probe round per variant and failed only coverage: vLLM's balancer gave
no `max` probe to engine 2 (all 60 answers correct). The gate now sends up
to four rounds until every engine answered, checking every answer.

| Candidate | c1 | c8 | c32 | img8 TTFT s | KV tokens per engine | Result |
|---|---|---|---|---|---|---|
| baseline (DP6, 16K, 0.90, 128 seqs), not restarted | 58.8 / 604 / 14.7 | 249.9 / 2,253 / 22.6 | 539.0 / 5,555 / 37.5 | 2.23 | 2,930,708 | PASS |
| baseline, restarted | 58.9 / 604 / 14.7 | 252.0 / 2,266 / 23.1 | 560.0 / 5,624 / 35.1 | 2.11 | 2,930,708 | PASS |
| batch 8K | 58.7 / 637 / 14.6 | 246.0 / 2,520 / 23.5 | 548.1 / 5,563 / 36.3 | 2.18 | 7,567,008 | PASS |
| batch 4K | 58.0 / 672 / 14.7 | 267.2 / 2,157 / 20.3 | 558.5 / 4,876 / 37.9 | 2.31 | 10,948,556 | PASS |
| memory 0.92 | 58.8 / 604 / 14.7 | 240.2 / 2,413 / 23.4 | 554.9 / 5,573 / 35.8 | 2.16 | 3,686,853 | PASS |
| official DSpark (adaptive verification) | — | — | — | — | — | **FAIL** at startup: `Adaptive verification trims verification requests on device, which the DeepseekV41IndexerBackend attention backend does not support` |
| DSpark, full-block verification | — | — | — | — | — | **FAIL** at startup: `No available memory for the cache blocks` — the drafter loads under EP6 (weights 59.96 GiB/GPU), but spec-decode CUDA graphs lower effective memory to 0.83 and KV to −0.35 GiB |
| TP2 × DP3 (re-measured) | 59.0 / 956 / 13.2 | 215.0 / 4,562 / 20.1 | 383.4 / 8,717 / 39.3 | 2.45 | 7,903,437 | PASS |
| `--mm-encoder-tp-mode data` (official) | 59.1 / 600 / 14.7 | 241.1 / 2,381 / 21.4 | 553.1 / 5,584 / 35.9 | 2.17 | 2,930,708 | PASS |

Reading: the two baseline runs differ by 3.9 % at c32 and not at c1, which
sizes single-trial noise. No single lever moves c1 or c32 throughput by
≥ 5 %, so the committed values stay. Batch 4K trades +11 % c1 TTFT for
3.7× KV capacity; encoder DP shows no 8-image TTFT change on this shape
(TP1 ranks have no encoder TP to remove). TP2 × DP3 is confirmed slower
(DP6 c32 +44 %).

## L1-2 stage 3: DSpark with the official memory levers

Run `20260930T120144Z-tuning`. DSpark uses the model's 5-token block with
probabilistic drafts and block rejection, verified in full (adaptive
verification is rejected by the V4.1 indexer backend, stage 2).

| Candidate | c1 | c8 | c32 | KV tokens per engine | Result |
|---|---|---|---|---|---|
| DP6 + DSpark + batch 4K | 102.9 / 681 / 6.9 | 274.2 / 1,501 / 21.5 | 632.9 / 5,087 / 29.1 | 7,236,246 | PASS |
| **DP6 + DSpark + batch 4K + memory 0.92** | **104.2 / 695 / 6.9** | **292.9 / 1,224 / 21.6** | **657.0 / 5,154 / 24.4** | 8,283,516 | PASS |
| TP2 × DP3 + DSpark | — | — | — | — | **FAIL** while capturing DSpark CUDA graphs: `Cuda error custom_all_reduce.cuh:164 'invalid argument'` (TP2 custom all-reduce) |

Weights with the drafter are 59.02 GiB per GPU. Against the DP6 baseline
(mean of its two runs: c1 58.85, c32 549.5 tok/s), the adopted row gives
c1 +77 % (TPOT 14.7 → 6.9 ms), c32 +20 % and c8 TTFT −46 %; c1 TTFT rises
604 → 695 ms. Batch 4K and memory 0.92 are both the official 8 × H100
memory-bound values; between the two DSpark rows the c1/c32 difference is
below 5 %, and 0.92 is kept for its 14 % larger KV pool and c8 row.
`--max-cudagraph-capture-size` stays at the runtime default: the measured
concurrency (≤ 64 over six engines) stays far below the official
`max-num-seqs × (1 + 5)` sizing; larger graphs would cost KV memory.

## Selected L1

DP6 / EP6 (TP1), Engram CPU offload, `FLASHINFER_MLA_SPARSE_DSV41`, FP8 KV,
MXFP4 indexer, 64-token pages, Marlin MoE, DSpark 5 (full verification),
`--max-num-batched-tokens 4096`, `--gpu-memory-utilization 0.92`,
`--max-num-seqs 128`, prefix caching, temperature 1.0 / top_p 0.95,
thinking high by default.

## Final gates on the committed configuration

Image `sha256:119afb09d8b71203707c50e8f9da1300c9992aea6c7aaa959cdb18df7e2483c5`
(the rebuild after dropping the unused 0909 profile; kernel gate 28/28,
`kernel-checks/kernels-final.log`). Served-config SHA-256
`611653384bea009287513ecf60254491ab3a00932025020c0939c32bc5182048`. L1
cpuset: NUMA nodes 0–2 (`0-15,64-79,16-31,80-95,32-47,96-111`). Run IDs
`final-<gate>` under `verification-results/`.

`l1`: PASS 13/13. Weights 59.02 GiB per GPU, 15.02 GiB KV per GPU,
**8,283,516 KV tokens per DP engine** (7.90 × a 1,048,576-token request),
idle GPU 0–5 at 87.6 GiB; default sampling `{'temperature': 1.0, 'top_p': 0.95}`;
rendered efforts 75/50/75/100 and chat mode; 72 probes all `323` with
finite log-probabilities, every variant answered by all six engines; tool
call and image pass.

`serving` (≈8K unique input, exactly 256 output tokens, thinking high,
reasoning tokens counted, 64 requests per row, every request placed on the
replica; nearest-rank percentiles):

| c | success | model TTFT p50 ms | p99 ms | TPOT p50 ms | output tok/s | placements | GPU peak MiB |
|---|---|---|---|---|---|---|---|
| 1 | 64/64 | 677 | 749 | 7.20 | 102.3 | 64/64 | 89,717–89,977 |
| 8 | 64/64 | 1,195 | 2,493 | 21.88 | 306.3 | 64/64 | 89,721–89,981 |
| 16 | 64/64 | 1,459 | 5,072 | 36.52 | 386.9 | 64/64 | 89,721–89,981 |
| 32 | 64/64 | 2,919 | 8,551 | 39.19 | 591.1 | 64/64 | 89,721–89,981 |
| 64 | 64/64 | 8,480 | 16,804 | 52.53 | 718.1 | 64/64 | 89,723–89,983 |

For orientation only (different GPU count, same host and row shape): the
8-GPU example's final matrix is 126.8 / 275.4 / 305.4 / 326.8 / 314.3 tok/s
at c1/8/16/32/64.

`tool-calling`: PASS (12 concurrent auto tool calls, tool-result turn,
streamed call, low and max effort each thinking and calling).
`vision`: PASS 6/6 completed red answers. `reasoning`: default/low/high/max
all `323` with reasoning. `cancellation`: PASS — L1 observed the request,
L1 and L2 released 0.29 s after the disconnect, a follow-up completed.

`completed` (natural completion at the default thinking-high effort, 32
requests per row, `max_tokens` 65,536; every request stopped with visible
content; generic = short explanatory questions, coding = self-contained
Python modules):

| workload | c | completed | first content p50 ms | completion p50 ms | completion p99 ms | output tok/s |
|---|---|---|---|---|---|---|
| generic | 1 | 32/32 | 2,806 | 5,516 | 49,515 | 157.9 |
| generic | 8 | 32/32 | 3,326 | 7,501 | 49,871 | 517.1 |
| generic | 32 | 32/32 | 4,955 | 9,472 | 54,210 | 788.3 |
| coding | 1 | 32/32 | 7,546 | 10,218 | 55,906 | 182.5 |
| coding | 8 | 32/32 | 17,965 | 19,218 | 113,580 | 770.6 |
| coding | 32 | 32/32 | 22,714 | 27,976 | 114,766 | 1,049.4 |

These rows check completion and latency, not answer quality.

`long-context` (retrieval of a random key placed mid-prompt; server-reported
prompt tokens): 32,805 in 3.2 s, 131,109 in 10.6 s, 262,181 in 21.8 s and
**1,039,909 in 119.6 s**, each returning exactly the key. Retrieval smokes,
not a long-context quality evaluation.

`restart` (`docker compose restart deepseek`): healthy and passing the
readiness probes (exact finite answers on the DP ranks, tool call, image)
again after 435.7 s.
