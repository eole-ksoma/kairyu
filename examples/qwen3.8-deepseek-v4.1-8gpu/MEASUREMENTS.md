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

## GPU gates on the current configuration

Pending: `up` readiness, `vision`, `tool-calling`, `serving-auto-max`,
`serving-auto-max-coding` (TTFT and the 900 s turn envelope), and the
browser smoke.
