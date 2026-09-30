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

## GPU gates

Pending: the kernel check, `up` readiness, `vision`, `tool-calling`,
`serving-auto-max`, `serving-auto-max-coding` (TTFT and the 900 s turn
envelope), and the browser smoke.
