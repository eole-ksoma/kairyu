# qwen3.8-deepseek-v4.1-8gpu evidence

Status: **CPU half complete; GPU gates pending.**

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

## GPU gates on the loosened ENSEMBLE criteria (DTO-D17 amendment, 2026-10-01)

Pending re-run: readiness, `vision`, `tool-calling`, `serving-auto-max`,
`serving-auto-max-coding`, and the browser smoke.

