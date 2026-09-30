# Qwen3.8-27B × 2 + DeepSeek-V4.1-Flash (six GPUs) ensemble example

Status: CPU half in progress; GPU gates pending.

Accepted scope (owner, 2026-09-30):

- New example `examples/qwen3.8-deepseek-v4.1-8gpu/`. Its ensemble method is
  the one in main's `qwen3.8-deepseek-v4-8gpu`: the judge plus five routes,
  and the dual-track DAG with head, draft, policies, answer_1..4, critique,
  synthesis, and audit.
- DeepSeek-V4.1-Flash runs on GPUs 0–5 and Qwen3.8-27B on GPUs 6 and 7
  (TP1, two replicas).
- The Qwen `image_description` stage is removed, because V4.1 takes images
  itself.
- Parameters may change from V4 for V4.1 performance; the ensemble method
  may not.
- The branch restarts from main.
- The DeepSeek public-output floor keeps working through a patch in this
  example's own overlay (owner choice over dropping the floor or changing
  the framework).
- Ports: API 8009, Chat UI 3009, DeepSeek L1 loopback 8010.

Design: DTO-D16 in `docs/design/example-dual-track-orchestration.md`.

## L1

- **DeepSeek:** the committed `deepseek-v4.1-flash-6gpu` service (DP6/EP6,
  Engram offload, 4K / 0.92, DSpark 5 with full verification, FP8 KV,
  64-token pages, image 8), with two changes:
  - `--default-chat-template-kwargs` is removed, so that `enable_thinking:
    false` reaches chat mode.
  - The loopback port is published for the tokenizer oracle and the paired
    TTFT denominator.
- **DeepSeek overlay:** this example owns `vllm-sm120.Dockerfile`,
  `patch_sm120.py`, and `check_sm120_kernels.py`.
  `patch_sm120.py` carries the six SM120 edits plus edit 7, the
  continuation of a final assistant prefill in the V4.1 encoder.
- **Qwen:** the V4 example's Qwen service (v0.23.0, FP8, 32 seqs, 32K
  batched, PIECEWISE, no MTP, one image), running two replicas.
- **Tuning:** one parameter at a time, only when a gate fails.

## L2 / L3

`auto-max.yaml` is the V4 spec, with these changes:

- The `tier2-direct` worker is removed.
- DeepSeek roles lose their inline scaffold and suffix.
- `image_description` is removed, with every dependency on it and every
  prompt block that renders it.
- The DeepSeek final units add `reasoning_continuation: chat` and
  `reasoning_open_tag`.
- Budget becomes `{18, 2}`.
- Direct DeepSeek caps become 262144, the V4.1 model card's minimum stated
  output.

`kairyu.yaml` has one Qwen pool with two replicas and one V4.1 pool
(multimodal, `allow_chat_template_kwargs: [enable_thinking]`). The rest of
the product surface is the V4 example's: embeddings, the one public chat
model, and the Open WebUI effort filter. The sandbox executor is not deployed;
no role references it.

## Files and ownership

- **Framework:** every file lives in the example directory. `kairyu/` is
  unchanged.
- **Tests:** `tests/unit/test_frontier_examplectl.py` gains the new directory
  name. `tests/unit/test_qwen38_deepseek_v41_example.py` covers only
  behavior:
  - the production loaders accept the configs;
  - an image request passes orchestration validation, with every DeepSeek
    role receiving the image;
  - effort-less DeepSeek roles send `enable_thinking: false`;
  - the launcher's GPU split and cpusets.

## Gates

- **`l1`:**
  - SM120 kernel check.
  - Rendered efforts: 50/75/100 through top-level `reasoning_effort`.
  - `enable_thinking: false` gives chat mode.
  - The prefill continuation renders without end-of-sentence.
  - Probes on every DP rank, plus tool and image checks.
- **`vision`:** a one-image request, run both forced to the ensemble and
  through the judge. The trace must show no `image_description` and must show
  multimodal DeepSeek stages. The answer must name the image content.
- **`serving-auto-max`:** generic matrix with route-aware stage validation.
- **`serving-auto-max-coding`:** TTFT ≤ 2× the paired direct-DeepSeek row;
  coding-row E2E is reported against the 900 s turn envelope.
- **Other:** tool calling and the Open WebUI browser smoke.

## Checklist

- [x] Plan, DTO-D16
- [ ] Example files, overlay edit 7, tests, CPU suite and ruff
- [ ] GPU: kernel check, `l1`, `vision`, serving matrices, browser smoke
- [ ] MEASUREMENTS.md, image digests, PROGRESS.md status
