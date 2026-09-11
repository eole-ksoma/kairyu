# Measurement status

The selected TP8/EP8 configuration passes the complete 320-request serving
matrix, thinking/tool/vision/cancellation gates, normal restart, and four
long-context retrieval smokes through 1,039,909 actual prompt tokens.
Measurements below apply to the pinned runtime and this hardware.

Hardware target: 8 × NVIDIA RTX PRO 6000 Blackwell Server Edition,
97,887 MiB per GPU, PCIe, one TP8 replica.

Model revision: `dba1be0a40aa45a94ad051997016db3960a90277`.
Model tree SHA-256: `d21211ca29ad7eba1fda84e49b1a34a73214ec9b84b23928cde63902c3318bfd`.

## Preflight evidence

- Related portable CPU contracts: 62 passed. Repository lint and progress-size
  checks passed.
- Built runtime: `vllm 0.1.dev20904+g179dd0fa9`, `flashinfer-python 0.6.18`,
  matching `flashinfer-cubin 0.6.18`, `torch 2.13.0+cu130`.
- Runtime image ID:
  `sha256:027bf47b2bd6f0d0abe54b296e7e9e3d31ee103bb6e46fa0a9807117681c2359`.
- The actual wrapped checkpoint tokenizer rendered default/high with budget
  75, low with 50, max with 100, and explicit off without a reasoning
  prefix. The FlashInfer capability query accepts `(8 heads, 128 top-k)`
  and `(8 heads, 192 top-k)`. The public API gates are recorded below;
  direct tokenizer checks alone do not establish public API capabilities.
- `uv pip check --system` reports the same single mismatch in both the
  official base and the overlay: torch declares NCCL 2.29.7, while the
  official image supplies NCCL 2.30.7. The overlay preserves that base
  package; the initial TP8 live gates below pass with it.

## SM120 compatibility selection

The official image loads target and DSpark weights (62.07 GiB per GPU),
but the initial recipe settings fail before serving on this host:

- Adaptive verification is unsupported by the indexer backend; disable it.
- BLHNC cannot split interleaved manager blocks into smaller kernel blocks.
  LBNHC is rejected by the packed indexer backend.
- V4.1 hardcodes SWA pages to 32; FlashInfer SM120 requires 64.
- V4.1 compression is C1/C2. With 128-token manager blocks, C1 indexer pages
  exceed DeepGEMM's 32/64-page envelope. Use 64-token manager blocks for
  both the V4.1 SM120 MLA and indexer backends: C1=64 and C2=32.
- SM120 FP8 indexer decode supports only 64-token pages. Select MXFP4
  indexer Q/K for C1=64/C2=32. The official binary includes both kernels;
  enable the vLLM dtype gate only for this V4.1/SM120 combination.
- Instantiate FlashInfer's existing generic dual-cache prefill templates
  for 32-token secondary pages; retain its existing arithmetic and address
  calculations. The SM120 attention subclass uses SWA=64.

`patch_runtime.py` checks the exact pinned source anchors. GPU numerical
parity passes all 16 C1/C2, decode/prefill, 128/192 top-k and mask combinations
using independent PyTorch attention over actual packed cache bytes, padded
block strides and attention sinks. The absolute/relative tolerances
0.05/0.05 match the pinned FlashInfer DSV4 tests. Maximum observed absolute
error is 0.0642 (within combined tolerance); the 32/64-page outputs have the
same errors as the earlier 64/128-page probe using the same random inputs.
Evidence: `sm120-page32-parity.log`; reproducible with `check_sm120_pages.py`.

Rejected startups are preserved in the NVMe example directory:
`initial-worker.log`, `no-adaptive-worker.log`, `block128-worker.log`,
`lbnhc-worker.log`, and `sm120-pages-live.log`.

MXFP4 indexer GPU evidence: all four C1/C2 prefill/decode cases passed with
real V4.1 Q quantization and K norm/RoPE/quantization/store, padded block
strides, and independently unpacked PyTorch logits. Decode error was zero;
maximum prefill absolute error was 2.3842e-7. `check_sm120_indexer.py` uses
the same `clean_logits=False` as serving and compares only populated logits.
Retained evidence: `sm120-indexer-parity.log`. The FP8 full-model rejection
is retained in `block64-worker.log`. The kernel test compares the actual
quantized values, not the model's BF16 indexer quality. The completed-answer
smokes below do not establish broad MXFP4/BF16 quality equivalence.

## Initial full-model gates

Runtime image: `027bf47b2bd6` (full ID above). Served-config SHA-256:
`ce09cf2323b40a0123f2d992a0b6ecdb7538231b43d24f6b54d89d2681ce408d`.
All artifacts are under the NVMe example's `verification-results/` directory.

| Gate | Result | Run |
|---|---|---|
| Reasoning default/low/high/max | 4/4 completed `323`, with reasoning | `20260911T032048Z` |
| Tool calling and effort overrides | 7/7 | `20260911T032049Z` |
| Vision | 2/2 completed colour answers | `20260911T032051Z` |
| Cancellation | L1 observed active, then L1/L2 released; follow-up completed | `20260911T032052Z` |

The UI uses V4's default/low/high/max choices and the standard top-level
`reasoning_effort`. The unchanged legacy L3 rejects `chat_template_kwargs`
on text requests; an initial experimental off toggle was removed after
that HTTP 400 was observed (`20260911T031732Z`). The L1 tokenizer's direct
off preflight is not a public-gateway capability claim.

## L1 comparison

Stage 1: `20260911T032149Z-tuning`; batch comparison:
`20260911T034604Z-tuning`. Each row uses 32 requests, unique
approximately 8K-token inputs, exactly 256 output tokens, default thinking
high, and the same base configuration SHA above. Throughput includes
reasoning; these rows emitted no final content. Each successful candidate
also passed completed tool-call and image probes before measurement.

| Candidate | c1 output tok/s | c8 output tok/s | c32 output tok/s | Outcome |
|---|---:|---:|---:|---|
| TP8/EP8, DSpark 5 | 128.55 | 275.99 | 324.72 | 32/32 at every row |
| DSpark off | 67.34 | 225.73 | 312.95 | 32/32 at every row |
| Batch limit 8K | 126.82 | 271.49 | 320.53 | 32/32 at every row |
| EP off, DSpark 5 | — | — | — | Startup failed: no available KV memory |
| FlashInfer PCIe IPC all-reduce | — | — | — | Autotuning stalled; operator stopped trial |

DSpark improves aggregate output throughput by 1.91× / 1.22× / 1.04× at
c1 / c8 / c32. At c1 its median model TTFT is 715 ms and TPOT 4.99 ms,
versus 716 ms / 12.09 ms without DSpark. Retain DSpark 5.

EP off consumes 73.56 GiB of model weights per GPU (EP8: 62.07 GiB), leaving
−0.62 GiB for KV under the same 0.90 memory utilization and 16K batch limit.
This rejects that candidate under these limits, not every possible EP-off
configuration. Keep EP8.

PCIe IPC completed 20/53 autotuning profiles, then showed no progress for
over three minutes while all GPUs remained busy. Its previous profiles
took 1–16 seconds each. The API never became ready; the trial was stopped
and NCCL restored. `pcie-ipc/operator-abort.json` and `worker.log` retain the
observation. The underlying stall cause is not established. No fallback
measurement is counted as a PCIe IPC result.

The 8K batch candidate increases the reported KV capacity from 3,752,210 to
5,812,918 tokens, but has 1.3–1.6% lower output throughput in these single
trials. Median model TTFT is 741 / 1,429 / 11,363 ms at c1 / c8 / c32,
versus 715 / 749 / 11,189 ms for 16K. This is not evidence of a throughput
gain, so retain the 16K batch limit; the baseline already fits the advertised
1M request and the 64 × 8K verification workload. These bounded comparisons do
not establish a global optimum or statistical significance for small gaps.

Selected L1: TP8/EP8, Marlin, DSpark 5 with probabilistic draft sampling and
block rejection, adaptive verification off, batch limit 16,384, sequence
limit 64, memory utilization 0.90, prefix cache, FP8 MLA KV, MXFP4 indexer,
64-token manager/SWA pages, GPU-resident Engram, and the runtime's breakable
CUDA graphs. NCCL is the all-reduce backend. Eager execution, Engram CPU
offload, higher memory utilization and additional batch/sequence limits are
available exploratory candidates but were not measured or selected here.

## Final serving matrix

Run: `20260911T035520Z-final-serving`. Same image and served-config SHA
as above, after restoring the selected 16K configuration. Each row completes
64/64 requests with 8K input tokens (approximately), exactly 256 generated
tokens and all 64 placements correlated to replica 0. Model TTFT measures
the first reasoning or content delta. Percentiles use nearest rank.

| Concurrency | Model TTFT p50 (ms) | Model TTFT p99 (ms) | TPOT p50 (ms) | Output tok/s | Success |
|---:|---:|---:|---:|---:|---|
| 1 | 714.93 | 725.72 | 5.038 | 126.82 | 64/64 |
| 8 | 1,367.31 | 5,386.45 | 23.533 | 275.41 | 64/64 |
| 16 | 2,050.01 | 10,609.27 | 41.485 | 305.42 | 64/64 |
| 32 | 4,676.24 | 21,251.30 | 69.543 | 326.82 | 64/64 |
| 64 | 21,570.32 | 46,596.12 | 112.932 | 314.29 | 64/64 |

Of 320 responses, 318 emit only reasoning. One c16 response begins content
at 9,544.77 ms and one c64 response at 50,322.45 ms; both still end at the
fixed token limit. Content TTFT is null for all other samples. These two
observations are not a useful completed-answer latency distribution.

The highest measured aggregate rate is at c32; c64 remains functional but
does not improve throughput in this workload. This is a warm, finite batch
measurement, not a sustained-arrival SLO or a completed-answer benchmark.
Startup reports 9.79 GiB of KV per GPU and 3,752,210 cache tokens (3.58×
the 1M context). This matches the initial baseline capacity.

## Final API gates

All use the same served-config SHA and runtime image as the serving matrix.

| Gate | Result | Run |
|---|---|---|
| Reasoning default/low/high/max | 4/4 completed `323`, reasoning present | `20260911T040108Z-final-reasoning` |
| Tool calling | 7/7, including streaming and tool-result turn | `20260911T040108Z-final-tool-calling` |
| Vision | 2/2 checks, two completed image answers | `20260911T040111Z-final-vision` |
| Cancellation | L1/L2 released; follow-up completed | `20260911T040112Z-final-cancellation` |

## Long-context retrieval

Run: `20260911T040114Z-final-long-context`, default thinking high. Each
request places a newly generated archive key in the middle of repeated
filler and requires the exact key in a completed answer. These inputs approach
the actual text context limit; counts below come from API usage, including
chat framing. Each request reserves 8,192 output
tokens and stays within the 1,048,576-token limit.

| Target raw input | Actual prompt tokens | Completion tokens | Total time (s) | Result |
|---:|---:|---:|---:|---|
| 32,768 | 32,805 | 86 | 3.31 | exact key, stop |
| 131,072 | 131,109 | 111 | 12.70 | exact key, stop |
| 262,144 | 262,181 | 103 | 25.67 | exact key, stop |
| 1,039,872 | 1,039,909 | 109 | 203.02 | exact key, stop |

These four synthetic retrieval smokes establish execution and retained-key
retrieval for these inputs. They do not establish broad long-context quality,
dense-document reasoning accuracy or MXFP4/BF16 model-quality equivalence.

## Normal restart

Run: `20260911T040555Z-final-restart`. A normal `docker restart --timeout 20`
of L1 became healthy in 165.26 seconds (166.77 seconds including post-restart
checks). The container start timestamp changed while its image, command and
served-config hash stayed fixed. API readiness, a completed tool call, an
image colour answer, and all four default/low/high/max reasoning cases pass.
The full startup log and before/after runtime identities are retained.

## Raw evidence identity

Artifacts live under
`/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-flash-8gpu/`.
Gate and tuning paths below are relative to `verification-results/`;
the two kernel logs are in the parent directory. JSON reports retain raw
samples, placements, requests/responses or runtime identity as applicable.

| Artifact | SHA-256 |
|---|---|
| `20260911T035520Z-final-serving/serving-c1/row-serving.json` | `928fbec0cedc3b30103d2c9c1720a7634f11684d9c8b4f733a6c37474a4cc5ec` |
| `20260911T035520Z-final-serving/serving-c8/row-serving.json` | `e3c6d8d15945a8ef1ca2b9d3b901b09140d89bcd34ba026cc5d4c09655153dbf` |
| `20260911T035520Z-final-serving/serving-c16/row-serving.json` | `60c7bd28b52fb240a85573cfc5963fb28688c1eb5b80749457bb0d59255e03d7` |
| `20260911T035520Z-final-serving/serving-c32/row-serving.json` | `4361748006597581dd4c7c132503979b68676550759b0554e79c98c114bca0b9` |
| `20260911T035520Z-final-serving/serving-c64/row-serving.json` | `631b08943a7009c27af2604b688529f49b602b6a31e3512252a9efc2ca273ad1` |
| `20260911T040108Z-final-reasoning/reasoning.json` | `65738129496d89912665a565fa0897aa5ab6ce87f83bef57cc3ccd9cf8c291a8` |
| `20260911T040108Z-final-tool-calling/tool-calling.json` | `11964f33f25abd3d696d0550b5773d45f8c75980f1e70d99c80dea9d67ff1c06` |
| `20260911T040111Z-final-vision/vision.json` | `78348c0d532a56fcc69fe1cd6f11816b5f9db455238ba9f02d7c8e97c7bfefd7` |
| `20260911T040112Z-final-cancellation/cancellation.json` | `4eee96887076ef1481701355588ebd5eacdfa677ea28359721df15f847d28e91` |
| `20260911T040114Z-final-long-context/long-context.json` | `f36a6517993046fbd3e6e66ccce8e689f2a5b0a8c6e7f9c0d32cc978090158f9` |
| `20260911T040555Z-final-restart/restart.json` | `1f4c1e7f5a3e9bff2044fee3b6aa6f1438babaea1980eb65d2af8b4dc7bff131` |
| `20260911T032149Z-tuning/selection.json` | `4fc876d213eb128a343d0f16bb7c28e29d205fc79b2951e3f116119d208f486b` |
| `20260911T034604Z-tuning/selection.json` | `418d798fdd46d2ac8867d4d644fe7fb0ac1af5af04736d0ed790694661c2f01d` |
| `sm120-page32-parity.log` | `45493a4f8905749f435ce91f82faeefcbb4e154850c7a99584ff3bf222941931` |
| `sm120-indexer-parity.log` | `1ffe04e48d4ce35bc8c248b92091cf098c8c46fcf2540525d73799d587a28db5` |
