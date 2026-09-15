# Measured performance

These are local measurements of the exact example configuration, not
manufacturer specifications or estimates. The locked run is the second matrix
against an already-running stack (warm replicas: every replica had served the
warm-up row and one full matrix before it), with row-unique prompt prefixes so
neither vLLM prefix caching nor Kairyu's prefix-aware placement can inflate the
matrix.

## Locked system

- Hardware: 8 x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB per
  GPU, compute capability 12.0, PCIe-only interconnect (no NVLink paths).
- Model revision: `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- Model tree SHA-256:
  `9825ce119c9693172e04dd2a1f2437884503ceab9bf55606141e6662c9fe301e`
- vLLM: `v0.23.0`, image ID
  `sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`
- Kairyu verification commit: `de16206f42986198ffd61ce6f2f62f2ef8d8ea7d`
  plus the uncommitted tool-calling change set of PR #584 (the served-config
  digest below covers that state).
- Served-config SHA-256 at measurement time:
  `140bc1c720d30673eaea42b9a34ddbcb3ab09aa9bd1745bc02b3e1f05a83f31d`
  (tool-calling configuration: replicas pass
  `--default-chat-template-kwargs '{"enable_thinking": false}'`; the matrix
  was rerun because that is a runtime flag. The earlier matrix
  `20260901T133331Z` under digest `3b2fd507…` produced the same placement
  split and every latency within 1%.)
- L1: 8 x TP1 replicas, each with `max_num_batched_tokens=32768`,
  `max_num_seqs=32`, FP8 KV, FP16 Gated-DeltaNet state, piecewise CUDA Graphs,
  MTP off — the single-GPU example's measured envelope — plus `qwen3` reasoning
  parser, `qwen3_coder` tool parser and the non-thinking default kwargs.
- L2: one pool, `prefix_index: true`, `queue_depth_threshold: 0`,
  `unhealthy_after: 1`.

## Warm serving result

Each operating point completed 64/64 requests with approximately 8K input
tokens and exactly 256 output tokens per request (success rate 1.0 in every
row). Percentiles use the harness's nearest-rank method. "Placements" is the
per-replica request count from the pool's placement log for that row; the
gate requires every replica to receive traffic and no replica to exceed 1.25x
the even share (10 of 64) at concurrency >= 8.

| Concurrency | TTFT p50 (ms) | TTFT p99 (ms) | E2E p50 (ms) | Mean TPOT (ms) | Requests/s | Output tok/s | Placements (replica 0..7) | Placement gate |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 940.36 | 954.05 | 6,517.14 | 21.870 | 0.15 | 39.28 | 64, 0, 0, 0, 0, 0, 0, 0 | reported only |
| 8 | 937.43 | 975.58 | 6,513.39 | 21.881 | 1.23 | 313.76 | 8, 8, 8, 8, 8, 8, 8, 8 | pass |
| 16 | 1,808.62 | 2,330.61 | 7,768.11 | 23.373 | 2.02 | 517.65 | 8, 8, 8, 8, 8, 8, 8, 8 | pass |
| 32 | 3,863.87 | 9,338.10 | 9,998.80 | 24.071 | 2.51 | 642.58 | 8, 8, 8, 8, 8, 8, 8, 8 | pass |
| 64 | 14,079.76 | 20,674.84 | 20,268.38 | 25.797 | 2.38 | 610.07 | 8, 8, 8, 8, 8, 8, 8, 8 | pass |

Run ID: `20260902T080357Z` (warm replicas: the stack had served the
tool-calling gate and the warm-up row first). The pre-tool-calling matrices
`20260901T133331Z` / `20260901T123716Z` in the same results tree produced the
same placement split and every figure within 1%.

## Tool-calling gate (same served configuration)

`./verify.sh tool-calling` run `20260902T080344Z`: **PASS, 6/6 cases** —
16 concurrent auto bash tool calls (the SWE-bench Pro mini-swe-agent request
shape) split 2 per replica across all eight, the `role: "tool"` follow-up
turn, the streamed variant, `reasoning_effort: high` with tools
(reasoning_content present, call parsed), and the non-thinking default.
`run.sh up` additionally fails closed on a readiness tool-call probe.

The first gate run against this stack (`20260902T075702Z`, before the default
kwargs flag) failed exactly one case, `default_direct_chat`: a plain question
came back with `content: ""` and the answer in `reasoning_content`, because
vLLM's `qwen3` reasoning parser assumes thinking is on unless
`enable_thinking=false` is present in the effective chat-template kwargs. The
flag fixes that; the other five cases passed in both runs.

## Reading the result

- **Placement is exactly even.** At every concurrency >= 8 each of the eight
  replicas received exactly 8 of the 64 requests; c1 sends all 64 requests to
  replica 0 because least-outstanding ties resolve to the lowest replica id
  (reported only, by design).
- **c1 and c8 are indistinguishable** (TTFT ~940 ms, TPOT ~21.9 ms): eight
  concurrent requests land one-per-replica, so each replica sees the c1 load.
  Aggregate output throughput scales 8.0x (39.28 -> 313.76 tok/s).
- **From c16 the matrix is prefill-bound within each replica.** Every replica
  holds 2/4/8 concurrent 8K prompts at c16/c32/c64; chunked prefill of
  16K/32K/64K prompt tokens per replica dominates TTFT (p50 1.8 s / 3.9 s /
  14.0 s) while TPOT stays 23-26 ms. Output throughput saturates around
  610-643 tok/s at c32-c64 because each replica runs only 4-8 sequences; the
  single-GPU example reached 867.58 tok/s at c32 with 32 sequences on one
  card. This matrix therefore characterizes the latency regime (<= 8
  sequences per replica). Reaching the replicas' full batching regime needs
  concurrency near 8 x 32 = 256, which is not part of the committed matrix.

## Raw evidence identity

Artifacts are below
`/mnt/nvme/kairyu/model-volumes/qwen3.8-27b-dp8-8gpu/verification-results/20260902T080357Z/`
on the measurement host (`run.json`, per-row `*-serving.json`,
`placement.json`, and logs); the tool-calling gate is under
`…/20260902T080344Z/tool-calling.json`.

| Artifact | SHA-256 |
|---|---|
| Serving concurrency 1 | `073aef7b6182e14da95b6361ff3226fbae1615950e3a0f9ce3883791a823dac0` |
| Serving concurrency 8 | `1b25876bebe62b110a3444e0bcb366279931fbce0a2ae24d738c3f47b4877a82` |
| Serving concurrency 16 | `c5efc7d4b637548c201d3af7e68088e3f975b59f9a493c9eb8a54af97fbadda7` |
| Serving concurrency 32 | `e61f22f82e5bc19429f15c0c476c8331b7eb9ab6a0940e31d606135cb100b152` |
| Serving concurrency 64 | `20efe752b2fff431923b57b0ff66e73800b1ac63f8c5ef753dc1d010b7335329` |
