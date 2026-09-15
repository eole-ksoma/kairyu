# Measured performance

These are local measurements of the exact example configuration, not
manufacturer specifications or estimates. The locked run measures the
tool-calling configuration (vLLM-owned chat rendering via the checkpoint's
`deepseek_v4` encoder, `--tool-call-parser deepseek_v4`, thinking off by
default, image input enabled) against an already-running stack, with
row-unique prompt prefixes so neither vLLM prefix caching nor Kairyu's
prefix-aware placement can inflate the matrix.

## Locked system

- Hardware: 8 x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB per
  GPU, compute capability 12.0, PCIe-only interconnect (no NVLink paths).
- Model revision: `6821d6ad3681a4b137b066b76094fa82ebd0a380`
- Model tree SHA-256:
  `7bd9e830b8400fdc3a4ac407910f82121d2e975a583c830015d84219e671f0a6`
- vLLM: upstream `27a94d1ce4e3fc100c4732439ccec10f8246a804` (digest-pinned
  nightly wheel) with FlashInfer `60b49158ab4fb81718aef486c2d3c89aec4c1901`
  overlaid from source (`vllm-sm120.Dockerfile`), image ID
  `sha256:b47e22101829b471e8d3b0702a50ab884f58390cc2dfd50cb83aaf43ae30e9d4`
- Served-config SHA-256:
  `a6b34046431b668d064dbe54f97cd5f45220c4fe501651660fdecf5344cf6362`
  (identical across the three gate runs below).
- L1: 2 x TP4+EP4 replicas (GPU 0-3, GPU 4-7), each with DSpark-3
  (probabilistic draft sampling, adaptive verification off), Marlin MoE
  backend, `max_num_batched_tokens=16384`, `max_num_seqs=32`, FP8 KV,
  256-token blocks, prefix caching, full/piecewise CUDA Graphs; vLLM renders
  chat with the checkpoint's own `deepseek_v4` encoder (image parts become
  `<｜deepseek_image｜>` placeholders) and parses DSML tool calls
  (`--enable-auto-tool-choice --tool-call-parser deepseek_v4`, thinking off
  by default). KV cache per replica: 2,327,629 tokens (36.12 GiB, 2.22x the
  1M context) — identical on both replicas on the measured first boot and on
  a second boot (2026-09-04); this checkpoint runs with vLLM
  `CompilationMode.NONE`, so the cold-compile-cache profile inflation that
  shrinks the Qwen example's KV cache does not apply here.
- L2: one pool, `prefix_index: true`, `queue_depth_threshold: 0`,
  `unhealthy_after: 1`; Kairyu forwards tools to /chat/completions
  (`legacy_chat_models`) and admits images under `image_input_policy`.

## Warm serving result

Each operating point completed 64/64 requests with approximately 8K input
tokens and exactly 256 output tokens per request (success rate 1.0 in every
row). Percentiles use the harness's nearest-rank method. The placement gate
requires exactly the row's 64 placements, both replicas served, and neither
above 1.25x the even share (40 of 64) at concurrency >= 8.

| Concurrency | TTFT p50 (ms) | TTFT p99 (ms) | E2E p50 (ms) | Mean TPOT (ms) | Requests/s | Output tok/s | Placements (replica 0, 1) | Placement gate |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 674.59 | 692.08 | 1,914.47 | 4.849 | 0.52 | 133.91 | 64, 0 | reported only |
| 8 | 701.82 | 2,738.86 | 4,308.45 | 13.913 | 1.74 | 445.79 | 32, 32 | pass |
| 16 | 1,342.63 | 5,060.80 | 7,230.22 | 22.519 | 2.05 | 523.71 | 32, 32 | pass |
| 32 | 3,190.68 | 9,959.70 | 12,600.74 | 33.406 | 2.50 | 638.79 | 32, 32 | pass |
| 64 | 10,605.37 | 19,935.55 | 23,374.21 | 47.870 | 2.69 | 689.37 | 32, 32 | pass |

Run ID: `20260904T115256Z`.

## Tool-calling gate (same served configuration)

`./verify.sh tool-calling` run `20260904T115201Z`: **PASS, 6/6 cases** —
4 concurrent auto bash tool calls (the SWE-bench Pro mini-swe-agent request
shape) split 2/2 across the replicas, the `role: "tool"` follow-up turn, the
streamed variant, `reasoning_effort: high` with tools (reasoning_content
present, call parsed), and the non-thinking default. `run.sh up` additionally
fails closed on a readiness tool-call probe.

## Vision gate (same served configuration)

`./verify.sh vision` run `20260904T115245Z`: **PASS, 2/2 cases** — 4
concurrent image requests (2 per replica; the same solid red PNG with a
row-unique "Vision case N: what single color fills this image? Answer with
one word." prompt) all returned non-empty content ("Red" x4), and the
placement log correlated them 2/2 across the replicas by `x-request-id`. `run.sh up` additionally fails closed on a
readiness image probe, which is the first request that exercises the SM120
sparse-MLA prefill path for image tokens on this FlashInfer revision.

## Reading the result

- **Placement is even.** At every concurrency >= 8 the two replicas received
  exactly 32/32 of the 64 requests; c1 sends all 64 requests to replica 0
  because least-outstanding ties resolve to the lowest replica id (reported
  only, by design).
- **c1 is within 15% of the sibling `deepseek-v4-flash-0731-dp2-8gpu` row**
  (TTFT p50 674.6 vs 776.5 ms, 133.9 vs 128.8 output tok/s) despite the added
  vision encoder, a different vLLM revision (upstream `27a94d1` vs the
  `jasl/vllm` SM120 fork) and DSpark-3 instead of DSpark-5; the pool adds no
  measurable overhead on the single-request path.
- **Two replicas scale past 5x the c1 output rate at c64** (689.4 vs 133.9
  tok/s) with TTFT at c32/c64 (16/32 concurrent 8K prompts per replica)
  prefill-bound within each replica. The c64 rate is 1.4x the sibling
  example's 487.5 tok/s; the two rows differ in vLLM revision, MoE backend
  (Marlin here) and draft length, so the gap is not attributable to one knob.

## Raw evidence identity

Artifacts are below
`/mnt/nvme/kairyu/model-volumes/deepseek-v4-flash-vision-exp-dp2-8gpu/verification-results/20260904T115256Z/`
on the measurement host (`run.json`, per-row `*-serving.json`,
`placement.json`, and logs); the tool-calling gate is under
`…/20260904T115201Z/tool-calling.json` and the vision gate under
`…/20260904T115245Z/vision.json`.

| Artifact | SHA-256 |
|---|---|
| Serving concurrency 1 | `3df14066e60364187db27f54f14a20f25cc4ebe8e5ec7dac461c8aab132da8de` |
| Serving concurrency 8 | `91ec7785a43d68be23c63addd624e79dd8b7801bd7327cad93fb3599dd18a7ab` |
| Serving concurrency 16 | `e35309e5572e21e89ecfe26065e6cc972ec7a7751a4910ec519a84ce3a834bda` |
| Serving concurrency 32 | `f758d7c9efbd65d9c186565524955fcecb8991b6ba450dc99917da7d1709e0f2` |
| Serving concurrency 64 | `280c4bfeecd9e64e6fceb85224b8651b88dda1f2f9f70265e2df66fa442f9e8f` |
| Tool-calling gate | `24a031a4de8689e82bbea5d49616e43be18804983e4bf5761dc855d70f2f2a12` |
| Vision gate | `5be1a6fbf2e9431ff5c4b39b52ebbe3c2da171778a85bf590e9471d042365302` |
