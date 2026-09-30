# deepseek-v4.1-flash-6gpu evidence

Status: **in progress — L1 selection running.**

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
