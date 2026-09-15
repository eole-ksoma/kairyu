# Measured performance

These are local measurements of the exact example configuration, not
manufacturer specifications or estimates. The locked run measures the
tool-calling configuration (vLLM-owned chat rendering via the checkpoint's
`deepseek_v4` encoder, `--tool-call-parser deepseek_v4`, thinking off by
default) against an already-running stack, with row-unique prompt prefixes so
neither vLLM prefix caching nor Kairyu's prefix-aware placement can inflate
the matrix.

## Locked system

- Hardware: 8 x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB per
  GPU, compute capability 12.0, PCIe-only interconnect (no NVLink paths).
- Model revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Model tree SHA-256:
  `90bd164d6f778d798eeaecd3517d83b87d49d300756a9217ada14a2b15203754`
- vLLM revision: `aa0d51302747ea80f282e26949708b3253409fe2`, image ID
  `sha256:99756b54424a4697f69476b29aa02fb7f8112aaa74fa8203a7bf8a0bae4ca6f1`
- Served-config SHA-256: `00f34af6b0c4021b1a877d50af4a7945642083dd12d14157b135cf406193e47c`.
  The earlier passthrough-era runs (`20260901T135600Z`/`20260901T140112Z`,
  measurement digest `b6ccf1d629b42383f329011398bccc89949d89029e9a42c320182a4aa84564c0`)
  measured the same latency envelope but could not emit OpenAI tool calls
  (PR #584 review) and are superseded by this configuration.
- L1: 2 x TP4+EP4 replicas (GPU 0-3, GPU 4-7), each with DSpark-5,
  `max_num_batched_tokens=16384`, `max_num_seqs=32`, FP8 KV, 256-token blocks,
  prefix caching, full/piecewise CUDA Graphs; vLLM renders chat with the
  checkpoint's own `deepseek_v4` encoder and parses DSML tool calls
  (`--enable-auto-tool-choice --tool-call-parser deepseek_v4`, thinking off
  by default); SM100-only MegaMoE and FP4 indexer cache disabled on SM120.
- L2: one pool, `prefix_index: true`, `queue_depth_threshold: 0`,
  `unhealthy_after: 1`; Kairyu forwards tools to /chat/completions
  (`legacy_chat_models`).

## Warm serving result

Each operating point completed 64/64 requests with approximately 8K input
tokens and exactly 256 output tokens per request (success rate 1.0 in every
row). Percentiles use the harness's nearest-rank method. The placement gate
requires exactly the row's 64 placements, both replicas served, and neither
above 1.25x the even share (40 of 64) at concurrency >= 8.

| Concurrency | TTFT p50 (ms) | TTFT p99 (ms) | E2E p50 (ms) | Mean TPOT (ms) | Requests/s | Output tok/s | Placements (replica 0, 1) | Placement gate |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 776.52 | 790.99 | 2,006.72 | 4.750 | 0.50 | 128.77 | 64, 0 | reported only |
| 8 | 803.73 | 3,232.57 | 5,420.65 | 15.718 | 1.51 | 387.41 | 31, 33 | pass |
| 16 | 1,488.86 | 6,081.78 | 8,600.03 | 27.223 | 1.73 | 441.75 | 32, 32 | pass |
| 32 | 2,398.64 | 12,153.40 | 16,565.87 | 49.545 | 1.71 | 438.95 | 32, 32 | pass |
| 64 | 12,479.38 | 23,581.46 | 29,521.90 | 65.333 | 1.90 | 487.53 | 32, 32 | pass |

Run ID: `20260902T005136Z`.

## Tool-calling gate (same served configuration)

`./verify.sh tool-calling` run `20260902T005122Z`: **PASS, 6/6 cases** —
4 concurrent auto bash tool calls (the SWE-bench Pro mini-swe-agent request
shape) split 2/2 across the replicas, the `role: "tool"` follow-up turn, the
streamed variant, `reasoning_effort: high` with tools (reasoning_content
present, call parsed), and the non-thinking default. `run.sh up` additionally
fails closed on a readiness tool-call probe.

SWE-bench Pro smoke (sibling `kairyu-bench` repo, `--only swe-bench-pro
--limit 3`, 2 workers, official mini-swe-agent harness): run
`20260902T010540Z-3bf671e8`. All 3 trajectories ended `Submitted` with a
parsed `tool_calls` entry on every assistant turn (61/61, 73/73, 80/80); no
`RepeatedFormatError`, which is the failure the review reported (22/22)
against the previous templated-passthrough configuration. The official
evaluation resolved 3/3; the smoke gates only the tool-call plumbing, not the
resolve rate.

## Reading the result

- **Placement is even.** At every concurrency >= 8 the two replicas received
  32/32 of the 64 requests (31/33 at c8); c1 sends all 64 requests to replica 0 because
  least-outstanding ties resolve to the lowest replica id (reported only, by
  design).
- **c1 matches the single-replica TP4+EP4 row measured by the tiered example**
  through Kairyu L3 (TTFT p50 776.5 vs 779.2 ms, 128.8 vs 130.8 output tok/s,
  `../qwen3.8-deepseek-v4-8gpu/MEASUREMENTS.md`, "Tier2 … selection"): the
  pool adds no measurable overhead on the single-request path.
- **Two replicas roughly double saturated throughput**: 487.5 output tok/s at
  c64 versus 241.62 tok/s for one TP4+EP4 replica at c32 (1.9-2.0x). TTFT at
  c32/c64 (16/32 concurrent 8K prompts per replica) is prefill-bound within
  each replica, as in the single-replica rows.
- The TP8+EP8 single-replica example reached 972.93 tok/s at c32 on a
  shared-prefix dataset with `max_num_seqs=64`; that figure is not comparable
  to this unique-prefix matrix.

## Raw evidence identity

Artifacts are below
`/mnt/nvme/kairyu/model-volumes/deepseek-v4-flash-0731-dp2-8gpu/verification-results/20260902T005136Z/`
on the measurement host (`run.json`, per-row `*-serving.json`,
`placement.json`, and logs); the tool-calling gate is under
`…/20260902T005122Z/tool-calling.json`.

| Artifact | SHA-256 |
|---|---|
| Serving concurrency 1 | `29ee13018dd5298bc51f28f8d24664370d8482a021618096f20622e8b8916753` |
| Serving concurrency 8 | `8a13922336790b8fbf0b1dd1a16fc09ddfe66ae79413f317cd06a5cef67b0227` |
| Serving concurrency 16 | `8b16506ac9f1297dba34d66086d51901cc202cdd7e2b3a14771f365e8761d196` |
| Serving concurrency 32 | `175ef1cb786fc1ea1fbc1ffdc84d2705ba0e7ea3f9ce9b68c09528fde030b46d` |
| Serving concurrency 64 | `7b8e8a55a0a3b3fe36c7ca7096cfa75391342393cfb83600f38097fb129ce967` |
