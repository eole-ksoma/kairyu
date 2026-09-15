# Measurements

> **Scope note (2026-08-18):** the served L2 policy is now the dual-track
> policy-ensemble DAG (`docs/design/example-dual-track-orchestration.md`,
> DTO-D1..D5). The "Dual-track DAG serving matrices" section below is the
> only section measured on it. **Every other section predates that DAG** and
> must not be attributed to the current deployment: the coding-DAG serving
> matrix was measured on the superseded nine-role coding DAG (its paired
> DeepSeek-direct denominators remain the pinned `example.json` fallbacks),
> and the older sections informed the unchanged L1 topology selection
> (TP1×4 Qwen, DSpark-5/16K DeepSeek).
>
> **Scope note (2026-08-20):** the DTO-D10..D12 changes (synthesis weighs
> five peer candidates and an inline DeepSeek audit gates publication, the
> image-only `image_description` stage, DeepSeek budgets halved to
> 8192/32768/65536 with a 65536 ceiling) are **not yet measured**; the
> matrices below bind to the pre-DTO-D8 served-config digest.
>
> **Scope note (2026-08-22):** DTO-D13 puts a Qwen non-thinking route judge
> in front of five profiles (four single-call direct routes + the ensemble).
> Nothing below measures the judged policy: the judge's per-turn latency,
> the route distribution of the harness datasets, per-route TTFT/E2E, and
> the re-defined TTFT gate (gated routes only; thinking direct routes
> reported) are all **not yet measured** and land with the next GPU window.
>
> **Scope note (2026-08-25):** DTO-D14 moves the Qwen thinking roles to the
> medium tier (spec `high`, example-local graded template, doubled budgets)
> and the audit verifier to Qwen tier1 (fixed medium, one 16384-token cap).
> The DTO-D14 served config is GPU-verified — see the 2026-08-25 section
> below; every older section predates it and does not transfer.

## Judged five-route serving gates (current deployment, 2026-08-25)

Run IDs: `20260825T161729Z` (`./verify.sh serving-auto-max-coding`) and
`20260825T173343Z` (`./verify.sh serving-auto-max`), both green (exit 0) on
served-config digest
`69702ab5af0ab3c00302855db18b33e89ec91823a2cc770161dfb44f1519da85`
(DTO-D8..D14: judged five-route policy, Qwen medium tier, audit on Qwen).
Artifacts under the NVMe `verification-results/<run-id>/` directories.

Coding matrix (32 requests per row, `max_tokens 65536`, T=0, seed 0): the
judge routed **every** coding request to `qwen_think_medium`, so all four
TTFT rows record `not_applicable` (thinking direct routes are reported, not
gated); paired DeepSeek-direct denominator rows are in the artifacts.

| c | route (all 32) | route TTFT p50 |
|---|---|---|
| 1 | qwen_think_medium | 31,657 ms |
| 8 | qwen_think_medium | 22,528 ms |
| 16 | qwen_think_medium | 33,979 ms |
| 32 | qwen_think_medium | 40,765 ms |

Generic matrix (8K-token prompts, 32 requests per row): every sample passed
the route-aware stage/trace validation (judge classification + exactly one
final unit; primary samples additionally all internal stages and the audit
verdict).

| c | routes | TTFT p50 |
|---|---|---|
| 1 | qwen_direct 32 | 2,263 ms |
| 8 | qwen_direct 32 | 6,355 ms |
| 16 | qwen_direct 29 / primary 3 | 17,428 / 19,739 ms |
| 32 | qwen_direct 32 | 34,823 ms |

The verification surfaced and fixed three defects before going green:
trace v2 rejected list detail values (the judge's offered labels) and
crashed streamed SSE bodies; the trace envelope started at the judge's
`started_at` while its `queued_at` was one clock read earlier; and the
Qwen medium preamble initially dropped the DeepSeek preambles'
think-close guard sentence, so long L2-wrapped contexts could end inside
the think span and fail the route with EmptyFinalOutput.

## Dual-track DAG serving matrices (pre-DTO-D8 9-role DAG)

Run ID: `20260818T025710Z` (`./verify.sh serving-auto-max-coding` then
`./verify.sh serving-auto-max`, same run directory); artifacts under the NVMe
`verification-results/20260818T025710Z/` directory, including the coding
matrix's `ttft-gate.json`. Served policy: the 9-role dual-track DAG
(DTO-D1..D5) — head + draft + policies wave, answer_1..4 ∥ critique wave,
compose publisher; no general profile, no profile judge, no verifier/refine
loop, no sandbox stages. Datasets, concurrencies, request counts, sampling,
and the paired same-dataset DeepSeek-direct denominator rows are unchanged
from the previous coding-DAG run.

Coding matrix (32 deterministic Python tasks per row, natural completion,
`max_tokens 4096`, public tokens via the DeepSeek loopback tokenizer oracle;
gate: product semantic TTFT p50 ≤ 2.0× the paired direct row):

| c | product semantic TTFT p50/p99 (ms) | DeepSeek-direct TTFT p50/p99 (ms) | ratio | gate | E2E p50/p99 (ms) | public tok/s | success |
|---|---|---|---|---|---|---|---|
| 1 | 418.95 / 436.40 | 2,348.34 / 2,767.83 | 0.18× | PASS | 46,142 / 84,820 | 46.18 | 32/32 |
| 8 | 589.17 / 1,980.65 | 7,346.63 / 15,039.34 | 0.08× | PASS | 85,124 / 147,936 | 136.21 | 32/32 |
| 16 | 724.55 / 7,322.23 | 11,399.93 / 16,341.58 | 0.06× | PASS | 135,345 / 179,807 | 205.80 | 32/32 |
| 32 | 9,361.45 / 14,837.27 | 14,054.86 / 19,033.20 | 0.67× | PASS | 235,645 / 293,637 | 185.29 | 32/32 |

All four rows pass against their paired direct denominators
(`denominator_source: paired_direct`). The previously marginal c32 row
improves from 1.87× (nine-role coding DAG, run `20260815T213146Z`) to
0.67×: wave 1 of the dual-track DAG places only two small Qwen calls (head,
draft) per request where the coding DAG placed three larger ones plus an
LLM profile judge on the serial admission path, so head TTFT stays clear of
Qwen TP1 saturation at c32. Coding E2E is mixed versus the old DAG (c1
58.5 s → 46.1 s p50; c32 164.6 s → 235.6 s p50) — E2E is unconstrained by
design, and the c32 growth tracks the wave-2 fan-out of four 2048-token
Qwen answers per request at saturation.

Generic matrix (~8K-token prompts, natural completion, head/compose stream
traced with `require_head`; not TTFT-gated):

| c | semantic TTFT p50/p99 (ms) | E2E p50/p99 (ms) | public output tok/s | success |
|---|---|---|---|---|
| 1 | 2,040.14 / 2,109.24 | 25,662 / 37,867 | 19.33 | 32/32 |
| 8 | 5,931.48 / 19,776.62 | 87,690 / 158,306 | 35.56 | 32/32 |
| 16 | 13,577.00 / 43,337.77 | 145,755 / 229,581 | 47.66 | 32/32 |
| 32 | 54,847.07 / 71,917.04 | 293,867 / 330,481 | 47.76 | 32/32 |

Versus the previous generic run (`20260815T191509Z`, seven-role general
profile): TTFT p50 improves at every concurrency (c32 89,988 → 54,847 ms)
and E2E contracts (c1 73.2 s → 25.7 s p50; c32 p99 510 s → 330 s), keeping
the worst case far inside the 600 s admission-wait bound. Single-request
smokes on the same build: normal chat turn 14.9 s E2E with all seven
internal stages traced; tool turn (head-disabled, `prompt_headless`) 11.7 s
returning a correct `tool_calls` envelope.

## Coding DAG serving matrix (superseded nine-role coding DAG)

Run ID: `20260815T213146Z` (`./verify.sh serving-auto-max-coding`); artifacts
under the NVMe `verification-results/20260815T213146Z/serving-auto-max-coding/`
directory, including per-row `ttft-gate.json`. This run includes every PR
#488 review amendment (networkless UDS sandbox transport, capped subprocess
output draining, abnormal-pytest-exit rejection, the strict
`execution_status`-based gate, the burst-queueing executor with 8 slots, the
subreaper-based escaped-descendant sweep, and the queue-inclusive deadline
budget with runner-side admission control) plus the issue #496 output-contract
amendments: caller-limit budgeting across the head/continuation seam, the
`reasoning_closed`/`prompt_headless` continuation contract, and the amended
head/continuation prompts (exact-answer branch; "output nothing further"
clause).
Dataset: 32 deterministic self-contained Python implementation tasks per row
(8 templates × 4 namespaced variants, ~1.5K prompt tokens), temperature 0,
natural completion, `max_tokens 4096`, public tokens counted by the DeepSeek
loopback tokenizer oracle. Semantic TTFT is the first public `content` SSE
token at L3. The paired DeepSeek-direct rows run the same dataset at the
same concurrency against the loopback L1 endpoint (`:8005/v1`), so the
committed `TTFT ≤ 2.0×` gate compares like against like. "Sandbox-executed"
uses the strict definition: BOTH `exec_matrix` and `exec_draft` reported an
`execution_status` containing `ok` (degraded `unavailable` stages never
count).

| Concurrency | product semantic TTFT p50/p99 (ms) | DeepSeek-direct TTFT p50/p99 (ms) | gate (≤2.0×) | E2E p50/p99 (ms) | sandbox-executed (strict) | success |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 417.39 / 427.56 | 2,348.66 / 2,661.04 | **PASS** (0.18×) | 58,533 / 131,746 | 29/32 | 32/32 |
| 8 | 921.44 / 7,071.53 | 6,459.02 / 9,350.11 | **PASS** (0.14×) | 109,914 / 300,794 | 29/32 | 32/32 |
| 16 | 1,310.88 / 11,930.31 | 9,564.01 / 19,139.84 | **PASS** (0.14×) | 114,722 / 282,043 | 30/32 | 32/32 |
| 32 | 32,434.50 / 32,921.68 | 17,341.50 / 23,111.54 | **PASS** (1.87×) | 164,597 / 271,452 | 30/32 | 32/32 |

Every sample carried a valid trace with a successful `head` and
`continuation` stage; zero execution stages degraded to `unavailable`. The
few non-executed samples are drafts that never produced a runnable fenced
block across their attempts — honest gaps, above the ≥90% floor. E2E is
deliberately unconstrained: it covers the full proposal/test fan-out, sandbox
runs, synthesis, execution-evidence verification, and bounded refinement
behind the committed opening (the head keeps the user reading from ~0.4 s at
c1). The c32 row passes only against its paired direct denominator (1.87×),
not against the historical 8K-prompt pinned fallback — the paired
same-dataset comparison is the committed gate, and it is marginal at c32:
both sides of the ratio are saturation-dominated and swing run to run.

Superseded coding-matrix runs: `20260815T042353Z` (pre-review-amendment
deployment, pre-strict gate), `20260815T111017Z` (post-UDS, failed the
strict gate at c32 with 25/32 — the measured evidence behind the gate and
executor amendments), `20260815T121447Z` (all rows green, pre-sweep/
deadline-budget amendments), `20260815T145039Z` (all rows green, pre-#496
output-contract amendments), and `20260815T203129Z` (first post-#496 run:
c1/8/16 green, c32 2.086× — product 31,456.78 ms vs direct 15,079.38 ms.
No #496 code path fired in its traces — zero `retry:empty_output` or
`skipped:public_budget` events — and the failing margin came from a
run-variance swing: the direct denominator measured 13,698–17,342 ms and
refinement load 48–60 `draft_synthesis` attempts across the last four runs;
the immediate same-deployment re-run above passes at 1.87×).

## Generic serving matrix (current deployment, executor skip path)

Run ID: `20260815T191509Z` (`./verify.sh serving-auto-max`), measured on the
issue #496 output-contract deployment (amended head/continuation prompts and
caller-limit budgeting). ~8K-token
generic prompts, natural completion, temperature 0. Every sample skipped
both executor stages locally (the everyday non-coding degrade path), carried
a valid trace with successful `head` and `continuation` stages, and
returned a non-empty public answer.

| Concurrency | semantic TTFT p50/p99 (ms) | E2E p50/p99 (ms) | public output tok/s | success |
|---:|---:|---:|---:|---:|
| 1 | 2,056.04 / 2,105.55 | 73,245 / 120,134 | 8.55 | 32/32 |
| 8 | 8,119.83 / 38,291.13 | 186,487 / 253,495 | 29.38 | 32/32 |
| 16 | 24,013.28 / 77,483.64 | 279,399 / 378,427 | 33.35 | 32/32 |
| 32 | 89,987.95 / 150,280.31 | 510,497 / 560,453 | 40.82 | 32/32 |

Superseded generic run: `20260815T052328Z` (pre-#488-review and pre-#496
amendments; its rows are within run-to-run variance of the table above).

Generic TTFT is dominated by the Qwen head's ~8K-token prefill (compare the
historical Tier1 TP1 c1 TTFT of 2,620.97 ms at the same prompt length); the
committed TTFT gate is the coding matrix above. Against the historical
MoA-3 auto-max rows on comparable 8K prompts (semantic TTFT p50
15,559.95 ms at c1), the head-streamed DAG improves first-public-token
latency by 7.5× while additionally running the full verification pipeline.

> **Historical evidence for all sections below:** these rows predate the
> current coding DAG.

## Current deployment validation

On 2026-08-14 UTC (2026-08-15 JST), the complete eight-GPU deployment reached
healthy status with four Qwen3.8 TP1 replicas on official vLLM v0.23.0 and one
DeepSeek TP4+EP4 replica on the measured `aa0d513027` DSpark build. The Qwen
replicas all reported the ported 32K/32-sequence/FP8-KV/piecewise-graph/no-MTP
configuration.

The launcher validated the unchanged seven-role verifier-gated DAG and exposed
exactly `kairyu-auto-max` plus `embed-small`. A 384-dimensional embedding
request completed, followed by an end-to-end product chat that returned a
non-empty final answer and 3,750 characters of model-attributed
`reasoning_content`. The stack was stopped after validation to release all
eight GPUs. This is a deployment/correctness smoke, not a new layered-path
performance matrix.

> **Historical evidence:** these measurements predate both the current
> verifier-gated `kairyu-auto-max` role DAG and the Qwen3.8 replacement. They
> must not be attributed to the current deployment; a fresh layered-path run
> is required.

Runtime validation was complete for the measured policy. All performance values
in this document were measured at the Kairyu L3 OpenAI-compatible endpoint used
by ChatUI, never at a vLLM L1 endpoint.

## Selected deployment

- L1: four Qwen3.6-27B-FP8 TP1 replicas on GPUs 0-3, plus one
  DeepSeek-V4-Flash-0731 TP4+EP4 replica on GPUs 4-7.
- Tier2: DSpark-5, 16,384 batch tokens, 32 sequences, FP8 KV, prefix caching,
  chunked prefill, and `FULL_AND_PIECEWISE` CUDA Graphs.
- L2 `kairyu-auto-max`: fixed quality route, three parallel Qwen proposals,
  private-thinking DeepSeek synthesis, `internal_max_tokens=2048`, four total
  steps, and no recursive refinement. `kairyu-auto` remains the static-rule
  low-latency mode and uses direct routes for ordinary requests.
- L3: Kairyu OpenAI-compatible API on loopback port 8003. Only unauthenticated
  Open WebUI is externally exposed, and it defaults to `kairyu-auto-max`.

## L3 auto-max performance selection

These rows measure complete Qwen proposal fan-out, DeepSeek synthesis, L2, and
L3 streaming on unique approximately 8K-token prompts. `public TPS` counts only
the assistant answer visible to the user; `internal TPS` is the cumulative
proposal-plus-synthesis output reported by orchestration. Every selected row
has a non-empty public answer and a valid trace with the exact proposal count.

| L2 candidate | Concurrency | semantic TTFT p50/p99 (ms) | E2E p50/p99 (ms) | TPOT mean (ms/public token) | req/s | public TPS | internal TPS | success / valid trace |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MoA-3, ordinary DeepSeek synthesis | 1 | 14,762.34 / 19,008.44 | 15,965.11 / 19,695.69 | 4.828 | 0.06 | 14.56 | 85.86 | 32/32 / 32/32 |
| MoA-3, ordinary DeepSeek synthesis | 8 | 32,257.80 / 37,529.51 | 39,267.84 / 43,239.06 | 35.025 | 0.20 | 44.26 | 276.16 | 32/32 / 32/32 |
| MoA-3, ordinary DeepSeek synthesis | 16 | 53,530.21 / 68,127.32 | 69,949.14 / 85,593.48 | 87.895 | 0.23 | 73.68 | 336.97 | 32/32 / 32/32 |
| MoA-3, ordinary DeepSeek synthesis | 32 | 100,263.47 / 126,709.95 | 130,182.47 / 133,476.95 | 172.232 | 0.24 | 48.98 | 324.82 | 32/32 / 32/32 |
| **MoA-3, private-thinking DeepSeek, 2048 internal cap** | **1** | **15,559.95 / 19,801.79** | **16,899.93 / 20,997.39** | **4.384** | **0.06** | **17.05** | **96.61** | **32/32 / 32/32** |
| **MoA-3, private-thinking DeepSeek, 2048 internal cap** | **8** | **38,325.65 / 48,881.13** | **41,234.34 / 50,111.05** | **13.707** | **0.19** | **47.81** | **302.65** | **32/32 / 32/32** |
| **MoA-3, private-thinking DeepSeek, 2048 internal cap** | **16** | **68,184.60 / 85,067.02** | **72,678.04 / 85,897.75** | **15.862** | **0.21** | **50.62** | **357.03** | **32/32 / 32/32** |
| **MoA-3, private-thinking DeepSeek, 2048 internal cap** | **32** | **129,700.49 / 150,492.27** | **138,592.39 / 153,445.40** | **29.284** | **0.21** | **54.32** | **351.29** | **32/32 / 32/32** |
| **MoA-2, ordinary DeepSeek synthesis** | **1** | **14,968.53 / 18,565.29** | **16,285.04 / 19,684.75** | **4.897** | **0.06** | **14.88** | **65.73** | **32/32 / 32/32** |
| **MoA-2, ordinary DeepSeek synthesis** | **8** | **26,741.72 / 33,050.82** | **34,450.35 / 37,173.33** | **36.660** | **0.23** | **54.50** | **235.91** | **32/32 / 32/32** |
| **MoA-2, ordinary DeepSeek synthesis** | **16** | **42,287.00 / 55,686.19** | **58,878.42 / 64,166.79** | **83.318** | **0.26** | **65.04** | **279.93** | **32/32 / 32/32** |
| **MoA-2, ordinary DeepSeek synthesis** | **32** | **79,506.41 / 104,720.95** | **108,613.41 / 114,351.48** | **169.822** | **0.28** | **56.15** | **285.42** | **32/32 / 32/32** |

Run IDs:

- `l3-auto-max-chat-moa3-public-v1-20260812`
- `l3-auto-max-chat-moa2-public-v1-20260812`
- `l3-auto-max-thinking2048-public-v1-20260812`

Against the selected warm `kairyu-auto` Tier1-direct path, `auto-max` pays the
intended quality cost. At c1/c8/c16/c32 its median semantic TTFT is
5.94x/7.40x/6.70x/7.92x higher and median E2E is
1.96x/3.55x/4.28x/5.03x higher. The exact corresponding `auto` rows are the
TP1 x 4 rows in the Tier1 table below. The comparison is a user-visible
latency envelope, not a same-output-length TPS A/B: `auto` emits exactly 256
tokens while thinking `auto-max` ends naturally after its public answer and
also performs three private proposals plus synthesis.

MoA-2 retains MoA-3's c1 envelope while reducing cumulative internal output
by 21.5%. At c8 it improves median TTFT by 17.1%, median E2E by 12.3%, and
public aggregate TPS by 23.1%. At c16 it improves TTFT/E2E by 21.0%/15.8%; at
c32 by 20.7%/16.6%, while request throughput rises from 0.24 to 0.28 req/s.
The lower c16 aggregate public TPS reflects shorter answers (247 versus 327
tokens/request), not slower delivery: per-request visible generation rises
from 10.25 to 11.37 tok/s.

A separate 1024-versus-512 private-proposal cap A/B was stopped after c1/c8.
The 512 cap produced 32/32 valid responses but changed c1 TTFT from 14,762.34
to 14,821.93 ms and c8 from 32,257.80 to 32,645.11 ms, while internal output
did not decrease. Qwen proposals usually ended naturally below 512 tokens, so
the performance candidate restores the quality-preserving 1024-token default.
Run ID: `l3-auto-max-chat-moa3-private512-public-v1-20260812`.

The hidden-thinking MoA-3 candidate was rejected before c16: c1 completed
32/32 with TTFT 15,375.10 ms, but c8 returned one response containing private
reasoning usage and no public answer (31/32 usable). L2 now converts that case
to an explicit failure instead of a successful empty response, but a quality
path must be reliable before scoring. Run ID:
`l3-auto-max-moa3-public-v2-20260812`.

The 2048-token private-thinking MoA-3 configuration passed the full L3 matrix: all 128 requests returned
non-empty public answers with 128/128 valid traces, exactly three proposals,
and zero request errors. At c1 its median TTFT/E2E are only 1.2%/0.8% above the
old 1024 candidate. At c8 it trades 18.8% higher median TTFT than ordinary MoA-3
for a 5.0% E2E increase and 8.0% higher public TPS, while eliminating the old
hidden-thinking empty-answer failure. At c16 the TTFT increase is 27.4%, but
median E2E increases only 3.9%. This establishes its serving envelope; product evaluation is owned by `evals/`.

## Tier1 topology selection

The comparison uses the same Qwen3.6-27B FP8 checkpoint and vLLM settings on
GPUs 0-3. Every row has an explicit topology-sized warm-up followed by 32
requests with approximately 8,192 prompt tokens and exactly 256 generated
tokens. Each concurrency row has a different namespace in its prompt prefix,
so a later row cannot become a full-prefix-cache benchmark. Kairyu trace-v2 was
requested and validated for every request; every trace selected the Tier1
direct-generation route.

| Tier1 topology | Concurrency | semantic TTFT p50/p99 (ms) | E2E p50/p99 (ms) | TPOT mean (ms/token) | per-request generation TPS p50 | aggregate output TPS | success / valid trace |
|---|---:|---:|---:|---:|---:|---:|---:|
| TP1 x 4 replicas | 1 | 2,620.97 / 2,630.50 | 8,631.38 / 8,642.58 | 23.571 | 42.59 | 29.66 | 32/32 / 32/32 |
| TP1 x 4 replicas | 8 | 5,177.94 / 5,304.99 | 11,622.34 / 11,688.60 | 25.277 | 39.66 | 175.90 | 32/32 / 32/32 |
| TP1 x 4 replicas | 16 | 10,169.72 / 10,489.80 | 16,971.69 / 17,130.61 | 30.098 | 37.47 | 240.41 | 32/32 / 32/32 |
| TP1 x 4 replicas | 32 | 16,373.42 / 20,904.68 | 27,545.04 / 27,728.81 | 44.838 | 22.84 | 295.19 | 32/32 / 32/32 |
| TP2 x 2 replicas | 1 | 2,078.98 / 2,092.62 | 5,868.14 / 5,879.39 | 14.858 | 67.57 | 43.62 | 32/32 / 32/32 |
| TP2 x 2 replicas | 8 | 6,338.02 / 8,524.49 | 12,423.15 / 12,828.95 | 24.848 | 39.88 | 163.48 | 32/32 / 32/32 |
| TP2 x 2 replicas | 16 | 11,802.63 / 16,197.90 | 20,753.74 / 20,852.76 | 36.580 | 28.49 | 196.87 | 32/32 / 32/32 |
| TP2 x 2 replicas | 32 | 19,487.37 / 32,171.86 | 37,069.33 / 37,186.92 | 67.419 | 14.56 | 220.17 | 32/32 / 32/32 |
| TP4 x 1 replica | 1 | 1,644.54 / 1,662.67 | 4,387.91 / 4,406.00 | 10.758 | 93.33 | 58.33 | 32/32 / 32/32 |
| TP4 x 1 replica | 8 | 9,773.81 / 13,226.58 | 15,897.35 / 16,627.64 | 27.789 | 37.66 | 127.40 | 32/32 / 32/32 |
| TP4 x 1 replica | 16 | 15,120.87 / 24,992.45 | 28,644.03 / 28,898.91 | 52.447 | 18.73 | 142.39 | 32/32 / 32/32 |
| TP4 x 1 replica | 32 | 27,005.18 / 49,913.22 | 54,199.46 / 54,421.42 | 102.705 | 9.41 | 150.48 | 32/32 / 32/32 |

Run IDs:

- `l3-auto-tp1x4-baseline-unique-20260811`
- `l3-auto-tp2x2-unique-20260811`
- `l3-auto-tp4x1-unique-20260811`

TP4 is the best c1 topology, improving median TTFT by 37.3% and aggregate
output TPS by 96.7% over TP1. It loses decisively once requests overlap: at c8
TP1 has 47.0% lower median TTFT and 38.1% higher output TPS; at c32 it has
39.4% lower median TTFT and 96.2% higher output TPS. TP2 is the same compromise
at a smaller scale: it improves c1 but loses TTFT and throughput to TP1 at every
tested concurrency from c8 onward.

The selected Tier1 topology is therefore **TP1 x 4 replicas**. This example is
quality-first and its principal `auto-max` path fans out 2-4 Qwen proposals.
Four independent replicas let those proposals execute concurrently, while TP2
and TP4 necessarily queue part of the fan-out. The selection is based on the L3
matrix and the intended orchestration workload, not on the fastest isolated L1
request.

The PCIe server required vLLM's supported `--disable-custom-all-reduce` option
for TP2/TP4. Without it, the TP2 candidate failed during CUDA-graph memory
profiling in the custom all-reduce kernel; with the option, both NCCL candidates
started and completed the matrix. TP4's first cache build reported 201.41 s for
engine initialization, including 109.54 s compilation. These candidate-only
transport settings are not present in the selected TP1 deployment.

## Tier1 speculative decoding selection (issue #509)

Measured 2026-08-16 UTC on GPU 3 (one TP1 replica slot of the deployed
topology) with the deployed Qwen3.8-27B-FP8 worker configuration, comparing
the no-speculation baseline against `{"method":"mtp","num_speculative_tokens":3}`
on a role-shaped workload: ~5,077-token prefix-cache-friendly prompts with
unique tails, 512 generated tokens per request at temperature 0.7 (the
proposal-role regime), via the worker's OpenAI endpoint. Speculative decoding
is lossless (vLLM rejection sampling preserves the output distribution), so
this is a pure latency/throughput decision.

| Candidate | c1 per-request tok/s p50 | c4 aggregate tok/s | c8 aggregate tok/s |
|---|---:|---:|---:|
| no speculation (baseline) | 45.63 | 166.78 | 334.42 |
| MTP-3 | **65.64 (+43.9%)** | **210.66 (+26.3%)** | **421.11 (+25.9%)** |

MTP-3 remains candidate-only. The role-shaped c1/c4/c8 result does not prove
that the deployed public envelope is safe: Kairyu admits 256 requests and the
committed serving matrices exercise c16/c32, while the matching Qwen TP1
saturation rows measured MTP-3 regressing. MTP also disables
`min_p`/`logit_bias`. The four Qwen workers therefore keep the no-speculation
baseline until the same deployed configuration passes c1/c8/c16/c32 without
aggregate-throughput or tail-latency regression.

## Tier2 speculation, batch-budget, and CUDA Graph selection

The Tier2 comparison keeps DeepSeek TP4+EP4, FP8 KV, max sequences 32, prefix
caching, and full/piecewise CUDA Graphs fixed. It changes only the named
candidate variable. The dataset and L3 measurement protocol are the same as
the Tier1 matrix. Direct-model requests enter through Kairyu L3; they do not
call the DeepSeek vLLM endpoint directly.

| Tier2 candidate | Concurrency | semantic TTFT p50/p99 (ms) | E2E p50/p99 (ms) | TPOT mean (ms/token) | per-request generation TPS p50 | aggregate output TPS | success |
|---|---:|---:|---:|---:|---:|---:|---:|
| DSpark-5, 16K batch | 1 | 779.22 / 791.75 | 1,926.21 / 2,240.48 | 4.617 | 223.00 | 130.79 | 32/32 |
| DSpark-5, 16K batch | 8 | 833.16 / 6,641.27 | 9,700.31 / 15,773.82 | 32.336 | 29.39 | 187.17 | 32/32 |
| DSpark-5, 16K batch | 16 | 2,710.53 / 11,899.12 | 17,749.63 / 29,026.48 | 49.887 | 19.82 | 221.88 | 32/32 |
| DSpark-5, 16K batch | 32 | 12,399.45 / 23,535.78 | 30,373.40 / 33,890.59 | 66.807 | 14.65 | 241.62 | 32/32 |
| DSpark-5, 32K batch | 1 | 771.59 / 786.90 | 2,007.40 / 2,251.42 | 4.789 | 207.32 | 128.39 | 32/32 |
| DSpark-5, 32K batch | 8 | 809.94 / 6,040.70 | 9,718.16 / 15,700.83 | 29.201 | 31.96 | 211.44 | 32/32 |
| DSpark-5, 32K batch | 16 | 3,306.91 / 11,602.01 | 16,664.29 / 28,367.76 | 46.221 | 23.27 | 234.90 | 32/32 |
| DSpark-5, 32K batch | 32 | 12,219.31 / 23,008.60 | 31,538.33 / 35,025.73 | 70.727 | 13.40 | 233.78 | 32/32 |
| No speculation, 16K batch | 1 | 774.25 / 1,186.28 | 3,406.64 / 3,817.17 | 10.326 | 97.21 | 74.88 | 32/32 |
| No speculation, 16K batch | 8 | 3,965.50 / 6,191.15 | 9,524.89 / 10,508.00 | 21.526 | 44.22 | 213.53 | 32/32 |
| No speculation, 16K batch | 16 | 6,549.86 / 11,780.05 | 16,120.20 / 21,357.75 | 38.401 | 26.73 | 252.94 | 32/32 |
| No speculation, 16K batch | 32 | 12,414.10 / 23,381.04 | 29,184.28 / 29,296.87 | 63.230 | 15.27 | 279.41 | 32/32 |
| DSpark-5, 16K, Graph NONE | 1 | 906.50 / 910.99 | 15,710.47 / 18,312.96 | 59.105 | 17.13 | 16.06 | 32/32 |
| DSpark-5, 16K, Graph NONE | 8 | 1,099.37 / 6,302.06 | 26,115.42 / 35,678.58 | 96.659 | 10.65 | 71.66 | 32/32 |
| DSpark-5, 16K, Graph NONE | 16 | 1,834.91 / 12,025.04 | 31,821.59 / 47,515.43 | 110.630 | 9.19 | 108.23 | 32/32 |
| DSpark-5, 16K, Graph NONE | 32 | 12,495.09 / 23,732.88 | 40,037.46 / 56,320.56 | 112.944 | 8.59 | 145.41 | 32/32 |

Run IDs:

- `l3-deepseek-tp4ep4-dspark5-b16k-unique-20260811`
- `l3-deepseek-tp4ep4-dspark5-b32k-unique-20260811`
- `l3-deepseek-tp4ep4-nospec-b16k-nvmecache-unique-20260811`
- `l3-deepseek-tp4ep4-dspark5-b16k-cudagraph-none-unique-20260811`

DSpark-3 is not a supported candidate in the pinned vLLM revision. Startup
fails closed because DeepSeek's DSpark block size is five and values below five
can produce incorrect output. It was rejected before serving any request.

The current winner is **DSpark-5 with a 16K batch-token budget**. Against no
speculation it gives nearly identical c1 median TTFT but 43.5% lower median E2E
latency and 74.7% higher aggregate output TPS. No-spec improves aggregate TPS
under c8-c32 saturation, but it delays the first token at c8/c16 and is much
slower for the single DeepSeek synthesis request on the principal auto-max
path. The 32K budget improves c8/c16 throughput but loses c1 E2E/generation
speed and c32 throughput/TPOT, so 16K is the better latency-first balance.

`FULL_AND_PIECEWISE` is also retained. Disabling CUDA Graph shortened a
persistent-cache engine initialization from 73.05 s to 29.55 s, but that
startup-only saving is not worth the serving regression. At c1, Graph NONE
increased median E2E latency from 1.93 s to 15.71 s and reduced aggregate
output throughput from 130.79 to 16.06 tok/s. It remained slower in E2E and
throughput at c8, c16, and c32; even its lower c16 median TTFT was paired with
79.3% higher median E2E latency. The selected Tier2 configuration is therefore
TP4+EP4, DSpark-5, 16K batch tokens, max 32 sequences, FP8 KV, prefix caching,
and full/piecewise CUDA Graphs.

Before the cache fix, a DSpark-5/32K cold engine initialization took 560.68 s,
including 375.98 s of mHC warm-up. The no-spec/16K initial build in the
corrected layout took 519.00 s, including 404.55 s of mHC warm-up. After
restoring the selected DSpark-5/16K configuration, the first restart against
that persistent cache reduced mHC warm-up to 11.98 s and total engine
initialization to 73.05 s. The generated caches exist outside the containers at
`/mnt/nvme/kairyu/model-volumes/qwen3.6-deepseek-v4-8gpu/compile-cache/`:
94 MiB for DeepSeek and approximately 239-240 MiB for each Qwen replica after
the reuse check. The selected DeepSeek process also reported 45.74 GiB of KV
cache, or 2,947,608 tokens (2.81x the configured 1M-token context), with FP8 KV.

## Cold/warm separation and runtime identity

All request-latency tables exclude an explicit warm-up and use a fresh
row-specific prefix namespace; they are warm serving measurements and cannot be
misread as cold-start latency or a full-prefix replay. Cold startup was measured
separately. Before the corrected persistent-cache layout, DeepSeek engine
initialization took 560.68 s; restarting the selected DSpark-5/16K configuration
with the warmed NVMe cache took 73.05 s, an 87.0% reduction. The first TP4 Qwen
candidate cache build took 201.41 s, including 109.54 s compilation; those
candidate-only artifacts were not used as warm request samples.

The measurements use eight NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs.
Qwen revision is `e89b16ebf1988b3d6befa7de50abc2d76f26eb09`; DeepSeek revision
is `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`; vLLM source revision is
`aa0d51302747ea80f282e26949708b3253409fe2`; and the vLLM image ID is
`sha256:99756b54424a4697f69476b29aa02fb7f8112aaa74fa8203a7bf8a0bae4ca6f1`.
The externally reachable no-auth ChatUI validated after the run is
`http://61.206.39.14:3000`; Kairyu L3 remains loopback-only.
