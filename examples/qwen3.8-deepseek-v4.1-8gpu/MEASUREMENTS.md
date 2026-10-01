# qwen3.8-deepseek-v4.1-8gpu evidence

Status: **GPU gates pass on the current configuration** (DTO-D17 with the
loosened ENSEMBLE criteria and judge fallback `deepseek_think`, 2026-10-01).

Hardware: 8 × NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120, PCIe). The
DeepSeek-V4.1-Flash L1 runs on GPUs 0–5; the two Qwen3.8-27B replicas run on
GPUs 6 and 7.

## Carried-over L1 evidence

- The DeepSeek service is the committed configuration of
  `deepseek-v4.1-flash-6gpu`: DP6/EP6, Engram offload, 4K / 0.92, and DSpark
  5 with full verification. Its selection and rejected candidates are in that
  example's `MEASUREMENTS.md`.
- The Qwen service is the V4 ensemble example's.

## Overlay edit 7 (CPU rendering check, 2026-09-30)

The check applied `patch_sm120.py` to the pinned base image
`vllm/vllm-openai:nightly@sha256:b5406fca…` (vLLM
`0.30.1rc1.dev396+gac68c3087`) and rendered prompts with the checkpoint's
tokenizer, without a GPU:

| Request | Rendered tail |
|---|---|
| user turn, default | `<｜User｜>Q?<｜Assistant｜><think>` with `Reasoning Effort: 75` |
| `continue_final_message`, prefill `<think>\nR\n</think>\n\n` | `<｜Assistant｜><think>\nR\n</think>\n\n` (no end-of-sentence) |
| the same with `reasoning_effort: max` | the same tail, with `Reasoning Effort: 100` |
| the same prefill without `continue_final_message` | unchanged: `<think></think><think>\nR\n</think>\n\n<｜end▁of▁sentence｜>` |
| `enable_thinking: false` (no server default) | `<｜User｜>Q?<｜Assistant｜></think>` with no effort prefix |

Before edit 7, the running six-GPU service rendered the continuation as
`<think></think><think>\nR\n</think>\n\n<｜end▁of▁sentence｜>`. With its
`thinking: true` server default, it also rendered `enable_thinking: false` in
thinking mode. These two findings are why edit 7 exists and why the
server-wide default is removed.

## GPU gates on the five-route, four-policy configuration (superseded by DTO-D17)

These results describe the configuration before DTO-D17 and are kept for
reference only. They are not evidence for the current two-route,
two-policy configuration.

| Gate (2026-09-30/10-01) | Result |
|---|---|
| `up` readiness probes | PASS: efforts 50/75/100, chat mode, edit-7 continuation, `323` from all 6 DP ranks in default/low/chat mode, image |
| `vision` | PASS: 4/4 ensemble; the DeepSeek L1 processed 12 images for 12 DeepSeek stages |
| `tool-calling` | PASS: `bash ls -la` via `qwen_direct` |
| `serving-auto-max` | PASS (completeness and stage traces). Numbers below. |
| `serving-auto-max-coding` | stopped by the owner during c1 (about 16/32); no TTFT verdict |

`serving-auto-max` rows: about 8K tokens in, 32 requests per row, natural
completion.

| c | routes | judge p50 | TTFT p50 / p99 | E2E p50 / p99 | public tok/s |
|---|---|---|---|---|---|
| 1 | qwen_direct 32 | 234 ms | 2.26 s / 2.30 s | 7.69 s / 11.47 s | 31.9 |
| 8 | qwen_direct 31, primary 1 | 344 ms | 6.24 s / 25.85 s | 23.43 s / 217.91 s | 29.0 |
| 16 | qwen_direct 29, primary 3 | 955 ms | 21.52 s / 33.14 s | 46.23 s / 659.23 s | 13.4 |
| 32 | qwen_direct 32 | 2,428 ms | 43.86 s / 60.82 s | 63.33 s / 69.97 s | 107.7 |

Observations:

- The ensemble's head opening and the synthesis continuation sometimes join
  without whitespace ("red.The"; 2 of 4 vision answers).
- vLLM preserves a leading newline in both streaming and non-streaming
  modes, so the likely cause is the model not emitting the requested blank
  line.

## GPU gates on the first DTO-D17 judge criteria (superseded by the criteria amendment)

These are kept for reference only. They were measured before the ENSEMBLE
criteria were loosened.

Runs are stored under
`/mnt/nvme/kairyu/model-volumes/qwen3.8-deepseek-v4.1-8gpu/verification-results/`
as `d17-vision`, `d17-tool-calling`, `d17-serving`, and `d17-coding`.

| Gate | Result |
|---|---|
| `up` readiness probes | PASS: efforts 50/75/100, chat mode, edit-7 continuation, `323` from all 6 DP ranks, image; the served L2 matches `example.json` (8 roles, 2 routes, `{16, 2}`) |
| `vision` | PASS: 4/4 primary; the DeepSeek L1 processed 12 images for 12 DeepSeek stages |
| `tool-calling` | PASS: `bash {"command": "ls -la"}` via `deepseek_think` |
| `serving-auto-max` | PASS (completeness and stage traces); every row was judged 32/32 `deepseek_think` |
| `serving-auto-max-coding` | exit 0. TTFT gate `not_applicable` at c1/8/16/32: the judge sent 32/32 of each row to `deepseek_think`, so the ensemble gate has no sample |
| browser smoke (`webui-browser-smoke.mjs`, this example's own) | PASS: one product model; a folded, attributed internal-work item; a separate final answer |

`serving-auto-max` rows use about 8K tokens in, 32 requests per row, and
natural completion. Output tok/s counts thinking; public tok/s counts the
answer only.

| c | wall | TTFT p50 / p99 | E2E p50 / p99 | output tok/s | public tok/s | TPOT mean | judge p50 / p99 |
|---|---|---|---|---|---|---|---|
| 1 | 462.4 s | 11.84 s / 27.41 s | 13.02 s / 28.52 s | 134.3 | 18.6 | 5.04 ms | 308 / 310 ms |
| 8 | 116.5 s | 22.85 s / 48.60 s | 25.79 s / 51.75 s | 460.4 | 70.5 | 10.73 ms | 305 / 733 ms |
| 16 | 103.5 s | 31.20 s / 52.21 s | 35.97 s / 53.67 s | 526.1 | 81.2 | 15.20 ms | 308 / 1,209 ms |
| 32 | 53.5 s | 37.80 s / 51.83 s | 41.26 s / 53.46 s | 988.4 | 156.2 | 10.93 ms | 1,706 / 2,236 ms |

For c1, one request breaks down as a 308 ms judge, 1,345 ms to DeepSeek's
first token, and 11,330 ms of thinking and answer.

`serving-auto-max-coding` rows use about 1.5K tokens in and 32 requests per
row. The paired direct row sends the same dataset straight to the DeepSeek
L1 with `max_tokens` 512.

| c | wall | TTFT p50 / p99 | E2E p50 / p99 | output tok/s | public tok/s | TPOT mean | direct TTFT p50 / p99 | direct output tok/s |
|---|---|---|---|---|---|---|---|---|
| 1 | 627.6 s | 13.09 s / 45.63 s | 14.26 s / 47.04 s | 167.8 | 14.1 | 4.28 ms | 3.71 s / 4.23 s | 137.5 |
| 8 | 163.4 s | 25.01 s / 114.45 s | 26.44 s / 115.54 s | 685.0 | 54.7 | 7.15 ms | 7.14 s / 8.92 s | 559.2 |
| 16 | 133.3 s | 34.60 s / 110.81 s | 38.21 s / 112.67 s | 910.6 | 66.8 | 7.88 ms | 8.64 s / 11.85 s | 875.6 |
| 32 | 94.9 s | 40.35 s / 93.75 s | 42.20 s / 94.87 s | 1,133.6 | 96.0 | 8.60 ms | 9.42 s / 10.23 s | 1,600.7 |

Open points:

- **Ensemble TTFT is unmeasured.** Under the DTO-D17 criteria the judge
  routed every synthetic generic and coding request to `deepseek_think`.
  Only the vision gate exercised the ensemble.
- **Seam formatting is unconfirmed.** The head/synthesis seam problem (see
  above) could not be re-checked, because the gates store only the first
  400 characters of each answer.
- **The 900 s turn envelope holds.** The largest E2E p99 was 115.5 s.
- **Not re-run for this example.** The SM120 kernel check was not repeated
  here: edit 7 touches only the Python encoder, and the kernel edits are
  those of `deepseek-v4.1-flash-6gpu`, which passed 28/28.

## GPU gates on the loosened ENSEMBLE criteria (superseded by the fallback amendment)

These ran with `fallback: primary` and are kept for reference only. Runs are
stored as `amend-*` under the verification-results directory above.

| Gate | Result |
|---|---|
| `up` readiness probes | PASS |
| `vision` | PASS: 4/4 primary; 12 DeepSeek images for 12 DeepSeek stages |
| `tool-calling` | PASS: `bash {"command": "ls -la"}` via `deepseek_think` |
| `serving-auto-max` | exit 0. Routes (`deepseek_think` / `primary`): c1 28/4, c8 18/14, c16 17/15, c32 22/10 |
| `serving-auto-max-coding` | stopped during c1 (first attempt) and again for the fallback change (second attempt); no verdict |
| browser smoke | not run |

Per-route TTFT p50 (`deepseek_think` / `primary`): c1 12.81 / 2.26 s, c8
16.58 / 6.34 s, c16 29.17 / 25.83 s, c32 38.55 / 35.77 s. Judge p50: 304,
365, 848, 2,101 ms.

## GPU gates with judge fallback `deepseek_think` (DTO-D17 second amendment, 2026-10-01)

Runs are stored as `fb-*` under the verification-results directory above. The
gateway was restarted at 09:31 with the new `auto-max.yaml`; the L1 services
were not restarted.

| Gate | Result |
|---|---|
| `up` readiness probes | PASS; `/routing` reports `fallback: deepseek_think` |
| `vision` | PASS: 4/4 primary; 12 DeepSeek images for 12 DeepSeek stages |
| `tool-calling` | PASS: `bash {"command": "ls -la"}` via `deepseek_think` |
| `serving-auto-max` | exit 0. 2 judge timeouts in 128 requests (c16), both served by `deepseek_think` |
| `serving-auto-max-coding` | exit 0. Ensemble TTFT gate PASS at c1/c8/c16/c32; 4 of 128 requests exceed the 900 s turn envelope |
| browser smoke (`webui-browser-smoke.mjs`) | PASS |

Routes are `deepseek_think` / `primary` / judge fallbacks (the fallbacks are
included in the `deepseek_think` count). Output tok/s counts thinking; public
tok/s counts the answer only.

`serving-auto-max` (about 8K tokens in, 32 requests per row):

| c | routes | wall | TTFT p50 / p99 | E2E p50 / p99 | output tok/s | public tok/s | judge p50 / p99 |
|---|---|---|---|---|---|---|---|
| 1 | 22 / 10 / 0 | 2,353.8 s | 11.18 / 19.99 s | 15.20 / 350.90 s | 23.9 | 4.3 | 306 / 323 ms |
| 8 | 28 / 4 / 0 | 386.2 s | 24.58 / 52.69 s | 28.47 / 315.81 s | 170.3 | 24.2 | 363 / 4,926 ms |
| 16 | 27 / 5 / 2 | 366.1 s | 30.19 / 64.48 s | 39.39 / 334.45 s | 153.9 | 26.1 | 960 / 5,002 ms |
| 32 | 26 / 6 / 0 | 427.8 s | 37.33 / 51.38 s | 43.11 / 427.82 s | 130.0 | 21.6 | 1,974 / 2,135 ms |

Ensemble TTFT p50 rises with concurrency (2.28 s at c1, 4.70 s, 14.73 s, 33.20 s
at c32) because the `head` stage waits for the two Qwen replicas.

`serving-auto-max-coding` (about 3.2K tokens in, 32 requests per row; the
paired direct row sends the same dataset to the DeepSeek L1 with `max_tokens`
512):

| c | routes | wall | TTFT p50 / p99 | E2E p50 / p99 | output tok/s | public tok/s | ensemble TTFT p50 | direct TTFT p50 | gate (≤ 2×) | > 900 s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 19 / 13 / 0 | 5,296.7 s | 6.46 / 43.30 s | 22.02 / 985.27 s | 39.9 | 2.2 | 626 ms | 3,616 ms | PASS (8.7 %) | 2 |
| 8 | 15 / 17 / 0 | 1,065.5 s | 4.85 / 54.24 s | 128.30 / 848.17 s | 143.6 | 11.6 | 1,036 ms | 6,760 ms | PASS (7.7 %) | 0 |
| 16 | 19 / 13 / 0 | 1,037.2 s | 16.63 / 61.53 s | 41.61 / 995.56 s | 192.1 | 11.3 | 6,654 ms | 8,411 ms | PASS (39.6 %) | 1 |
| 32 | 18 / 14 / 0 | 963.6 s | 18.93 / 47.52 s | 43.51 / 963.61 s | 120.4 | 12.0 | 18,772 ms | 9,797 ms | PASS (95.8 %) | 1 |

Observations:

- **Fallback.** The judge timed out 3 times in 269 judged requests (1 in the
  generic warmup, 2 at generic c16). All three were served by
  `deepseek_think`; none escalated to the ensemble.
- **Ensemble E2E depends on audit refinements.** Coding ensembles without a
  refinement take 128–580 s; one refinement takes 541–746 s; two take
  848–996 s. 12 of 57 coding ensembles were refined at least once. Every
  request above the 900 s envelope (4 measured plus 1 warmup) reached the
  two-refinement limit, and its refined synthesis ran for several minutes.
- **The c32 TTFT margin is small.** Ensemble TTFT at c32 is 95.8 % of the
  limit. `head` averages 19.2 s at 10.2 tok/s there, because the judge and the
  ensemble's Qwen stages share the two Qwen replicas.
- **The Qwen replicas bound the ensemble.** Qwen stages run at about 31–45 tok/s
  per request (`answer_1`/`answer_2` about 85–100 s, `draft` 43–60 s); the
  DeepSeek stages run at 70–210 tok/s.
- **Warmups.** Both matrices send 4 warmup requests before c1. Since the
  ENSEMBLE criteria were loosened, warmups include ensembles and take 267 s
  (generic) and 940 s (coding) instead of 12–103 s.
- **Not re-checked.** The head/synthesis seam formatting (stored answers are
  truncated to 400 characters), and the SM120 kernel check (L1 unchanged).

