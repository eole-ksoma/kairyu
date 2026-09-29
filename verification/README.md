# Verification framework

`verification/` is Kairyu's checkout-only framework for validating the
implementation and deployment. It is deliberately separate from model
`evals/`: HF/TP parity, logprob agreement, batch invariance, quantization/KV
equivalence, TTFT, TPOT/TPS, throughput, goodput, and vLLM comparisons are
Kairyu verification, not model-evaluation benchmark rows.

## Ownership boundary

| Surface | Contract |
|---|---|
| `kairyu/` | Installable product code. It must not import `verification`, `evals`, or repository-only evidence tooling. |
| `evals/` | Checkout-only model evaluation: Core, Quantization, Structured Output, and Long Context. It does not own Kairyu correctness/performance gates. |
| `verification/` | Checkout-only correctness, performance, resilience, and diagnostic gates, organized as `scope/kind`. It must not import `evals`. |
| `evidence/` | Neutral strict JSON/hash/reporting contracts shared by checkout tooling; it does not own gate policy. |
| `bench/results/` | Immutable legacy evidence location. Existing tracked bytes and historical paths remain unchanged. |

The four scopes are `l1`, `orchestration`, `fleet`, and `product`. The four
kinds are `correctness`, `performance`, `resilience`, and `diagnostic`.
`verification/registry.toml` is the strict v1 source of truth for every active
entrypoint and records a stable ID, scope, kind, prerequisites, documentation,
output schema, formal status, module, path, and invocation forms.

## Registry and execution

List or validate the checkout without executing a gate:

```bash
uv run --frozen python -m verification list
uv run --frozen python -m verification list --scope l1 --kind correctness --json
uv run --frozen python -m verification check
uv run --frozen python scripts/verify_verification_registry.py
```

Run a registered gate by ID and put its arguments after `--` (a gate with no
formal status writes no execution record):

```bash
uv run --frozen python -m verification run \
  orchestration.performance.router_latency -- --json --assert-gate
```

A formal correctness run writes a schema-v1 execution record. A formal
performance run additionally requires an accepted correctness record and binds
its exact SHA-256 into the performance execution record:

```bash
uv run --frozen python -m verification run \
  l1.performance.dp_scaling_g2_a8_bench \
  --correctness-artifact /path/to/accepted-correctness.json \
  -- --help
```

The runner also exposes the accepted correctness gate ID, artifact path, and
SHA-256 to the child process as `KAIRYU_CORRECTNESS_*` environment variables.
Gate-specific artifacts retain their existing schemas and default
`bench/results/` paths; the standardized execution envelope is additive.

The CPU-only durable-request deployment gate is registered as
`verification/fleet/resilience/async_request_gateway_smoke.py` and is normally
run through `scripts/kind_async_request_gate.sh`, which provisions its F1c kind
topology before invoking the entrypoint.


Every registered entrypoint must have a module docstring, a non-executing
`--help`, an exact registry record, and documentation that names its path.
`python -m verification check` inventories main guards recursively, verifies
doc references, rejects undeclared entrypoint-to-entrypoint imports, rejects
`verification -> evals`, and rejects `kairyu/evals/evidence -> verification`.

## Legacy evidence

`bench/results/` stays in place because retained formal artifacts bind exact
historical paths and bytes. `bench/results/index.json` remains the discovery
catalog, and `scripts/verify_bench_results_index.py` validates it against
tracked files. New runtime outputs under `verification/results/` are ignored.

### CPU microbenchmark CI smoke gate

Pull requests run six source-checkout-only CPU benchmarks in one dedicated
Python 3.12 job:

```bash
uv run --frozen python scripts/cpu_microbench_gate.py \
  --output /tmp/kairyu-cpu-microbench.json
```

The runner executes scheduler queue, radix eviction, operation queue, sampler
penalty-state, process-wire, and router-latency measurements sequentially with
CUDA hidden and native math thread pools fixed to one. Optimized/legacy ratios
are measured within the same child process. The deliberately loose checks catch
large regressions while tolerating shared-runner frequency and scheduling
variance: scheduler speedups must remain at least 2.0x/10x/0.90x, radix
eviction at least 100x, operation-queue throughput at least 0.50x with exact
coalesced container/ID counts, and both sampler legacy
speedups at least 5x. The deterministic process-wire gate retains its own byte
growth contract, and router p99 remains strictly below 10 ms.

The JSON report is a short-lived CI diagnostic, not formal benchmark evidence.
Absolute timings and the profiled runner must not be used to claim a product
speedup; reproduce any suspected regression on the owning benchmark before
changing production behavior or a formal performance gate.

### Nightly CPU performance series

The main-only `.github/workflows/nightly-cpu-perf.yml` workflow reuses the
same `scripts/cpu_microbench_gate.py` report once per night and on explicit
manual dispatch.  It adds a cross-run smoke alarm without turning shared
GitHub-hosted runner timings into formal performance evidence.  Only these
eight higher-is-better, same-process ratios are hard-alert metrics:

- scheduler FIFO-drain, indexed-removal, and priority-drain speedups;
- radix-eviction speedup;
- operation-queue add and abort elapsed-time speedups; and
- sampler legacy and append-legacy speedups.

For each metric, the current observation is compared with the median of at
most the seven most recent preceding records anywhere in the chain with its
exact compatibility fingerprint.  Intervening incompatible records are
ignored, and the current observation is never part of its own baseline.  Five
preceding compatible records are required; an unseen or younger fingerprint
is reported as warmup and cannot raise a cross-run regression alert.  Returning
to an exact prior fingerprint resumes its prior compatible history.  A hard
alert occurs when the current value is at least 15% below that trailing median.
Absolute legacy and optimized timings, router p50/p99, process-wire sizes and
growth, and all other report fields remain report-only for the cross-run
trailing-median comparison.  They can still fail the unchanged issue-#378
same-run source gate and keep the workflow red.  A nightly alert must be
reproduced on the owning benchmark before it supports a product-performance
claim or a threshold change.

Compatibility segmentation binds the report/methodology schema, benchmark
inventory and fixed arguments, CPU-only environment controls, Python
implementation and major/minor version, runner operating-system, architecture,
and hosted-image class, metric paths and directions, and the comparator policy
(15%, seven, and five).  A previously unseen methodology or runtime-class
fingerprint therefore starts a new warmup segment.
The measured source commit is provenance, not compatibility: including it in
the segment key would prevent comparison across main commits.

Every completed record uploads `report.json`, `series.jsonl`, and
`comparison.json` from a directory under `RUNNER_TEMP`, with 90-day retention;
an earlier infrastructure/input failure can retain only the files it safely
produced and is never a history candidate.  The canonical JSONL series is
append-only and SHA-256 chained.  Recovery scans this workflow's
completed main-branch artifacts newest first, including artifacts from runs
whose regression-enforcement step failed.  Expired, incomplete, malformed,
non-finite, duplicate, non-canonical, or hash-chain-invalid candidates are
skipped in full; they are never truncated into an apparently valid baseline.
Only an authoritative empty artifact listing starts first-run warmup; if one
or more eligible candidates exist but none validates, recovery fails closed
instead of silently resetting the baseline.  Artifact-listing, attempt-metadata,
and download transport failures also fail immediately.  The comparison command
validates the existing report and writes the updated series and comparison
before returning a regression status.  The workflow uploads those outputs
together with the unchanged report using `always()` before propagating that
status.
If the reused issue-#378 absolute gate returns nonzero but emits a structurally
valid report, the record is still appended with that source-gate outcome while
the workflow remains red; dropping it would create a measurement blind spot.
Only explicitly allowlisted performance failures receive that treatment.
Fixed requests, repeats, benchmark configuration, protocol, shape, and
structural-source mismatches are invalid and are not recorded.
The seven-day pull-request reports produced by issue #378 do not seed this
separate nightly chain.

### Target and credential migration

Every benchmark target now uses one package-owned grammar:
`name=base_url=model[=api_key_env]`. The optional fourth field is the **name of
an environment variable**, never a credential value. Older
`verification/product/performance/frontier_compare.py` help text treated that field as a literal key; set
the key in the environment and replace the literal field with its variable
name:

```bash
export FRONTIER_API_KEY=...
python verification/product/performance/frontier_compare.py \
  --target frontier=https://api.example/v1=model=FRONTIER_API_KEY
```

`verification/l1/performance/serving_bench.py --api-key` remains as a deprecated compatibility flag
for existing commands. New commands should use `--api-key-env` or the shared
`--target` form. An explicitly named but unset credential variable fails
closed across installed and repository-only runners. Resolved secret values
are never written to result config or validation errors.
Direct, YAML, split-flag, and combined-target construction all normalize API
roots to `/v1` and validate credential variable names before fingerprinting.
Historical run directories remain reportable, but a pre-migration run whose
stored YAML URL omitted `/v1` has a different fingerprint; use a new run ID
instead of silently resuming it.

Generic serving/frontier reports now record
`percentile_method: nearest-rank-v1`. Older reports from those two wrappers
that lack the field used frontier's floor-index calculation and serving's
median p50/floor-index p99; they must not be compared as though the percentile
definitions were identical. Formal gate wrappers retain their versioned,
source-bound artifact schemas; those embedded methods are gate contracts, not
alternative reusable reporting helpers.

`verification.l1.performance.profiling` owns the optional in-process PyTorch profiler contract
for checkout benchmarks and GPU gates. It imports torch only for an enabled
scope, maps explicit CPU/CUDA activities without fallback, and deliberately
does not add warm-up, synchronization, schedules, or profiler steps. The two
benchmark wrappers and all GPU tests that consume torch profiler events use
this helper while retaining their original measurement boundaries.

`verification/l1/performance/serving_bench.py --profile` records only a CPU trace of the local HTTP
load-generator process. It does not profile the target server or CUDA work; use
`scripts/profile_server.py` for the bounded py-spy/Nsight process-tree capture,
and use `--stage-trace` for target-reported request stages. A profiled run needs
PyTorch and a non-empty results directory, writes a paired UTC-microsecond
`*-serving.json` plus `*.client.pt.trace.json`, and records the sidecar's
relative name, size, SHA-256, format, activity, and diagnostic-only scope.
Trace export is strict, 64 MiB-bounded, private, and non-overwriting. Any path,
collision, export failure, or result-publication failure cannot leave a
successful or unbound pair. The example multi-GPU serving reporter excludes
profiled files from performance tables. The default path does not import torch,
and profiled latency/goodput is not comparable to an ordinary run. Inspect raw
traces for local operator/runtime metadata before sharing them.

The human-readable path index below mirrors the manifest. Kind, prerequisites,
and documentation metadata are authoritative in the TOML and in the
``python -m verification list --json`` output.

```text
verification/fleet/correctness/agentic_kv_tier_f4b_bench.py
verification/l1/performance/attention_backend_profile_bench.py
verification/l1/diagnostic/audit_io_bench.py
verification/orchestration/performance/auto_params_bench.py
verification/l1/correctness/batch_invariance_bench.py
verification/l1/performance/batched_prefill_qwen.py
verification/l1/correctness/batched_spec_verify_qwen.py
verification/l1/performance/decode_page_table_cache_qwen.py
verification/l1/performance/dp_scaling_g2_a8_bench.py
verification/l1/correctness/draft_quant_qwen.py
verification/fleet/performance/dram_kv_tier_qwen.py
verification/fleet/resilience/fleet_churn_bench.py
verification/fleet/resilience/fleet_gateway_bench.py
verification/fleet/resilience/fleet_rollout_bench.py
verification/fleet/diagnostic/fleet_usage_replay.py
verification/l1/correctness/fp8_kv_g4_ekv_bench.py
verification/product/performance/frontier_compare.py
verification/l1/performance/future_token_bench.py
verification/l1/performance/g2_a6_vllm_bench.py
verification/l1/performance/g2_a9_dp_tp_crossover_bench.py
verification/l1/correctness/g4_ma1_nvfp4_correctness_bench.py
verification/l1/correctness/g4_ma1_qwen3_235b_nvfp4_bench.py
verification/l1/correctness/g4_ma1_qwen3_235b_nvfp4_capture.py
verification/l1/correctness/g4_ma2_qwen3_235b_ep_kv_bench.py
verification/l1/performance/g4_ma3_kairyu_server.py
verification/l1/performance/g4_ma3_sglang_bench.py
verification/l1/correctness/gate_a1.py
verification/l1/correctness/gate_a2.py
verification/l1/correctness/gate_logits_dtype.py
verification/fleet/diagnostic/global_kv_pool_decision.py
verification/l1/performance/issue_333_proc_http_bench.py
verification/l1/correctness/kv_answer_equivalence_bench.py
verification/fleet/performance/kv_aware_ttft_f2c_bench.py
verification/fleet/correctness/kv_event_f2b_bench.py
verification/fleet/correctness/kv_event_hash_bench.py
verification/l1/performance/kv_transfer_bench.py
verification/fleet/performance/multiturn_prefix.py
verification/fleet/resilience/noisy_neighbor_bench.py
verification/fleet/resilience/noisy_neighbor_gpu_bench.py
verification/l1/performance/op_queue_bench.py
verification/orchestration/performance/orchestration_mock_bench.py
verification/orchestration/correctness/orchestration_stream_bench.py
verification/l1/correctness/parity_hf.py
verification/l1/correctness/parity_tp.py
verification/l1/performance/pd_mixed.py
verification/l1/performance/pd_overlap_qwen.py
verification/fleet/performance/prefix_routing_f2a_bench.py
verification/fleet/correctness/prefix_weight_f2d_bench.py
verification/fleet/resilience/priority_overload_bench.py
verification/fleet/resilience/priority_overload_gpu_bench.py
verification/l1/performance/proc_wire_bench.py
verification/l1/correctness/quant_checkpoint_parity_bench.py
verification/l1/performance/quant_gemm_bench.py
verification/l1/performance/radix_eviction_bench.py
verification/l1/performance/reduce_scatter_bench.py
verification/orchestration/performance/router_latency.py
verification/l1/performance/run_g2_a6_formal.py
verification/l1/performance/sampler_penalty_state_bench.py
verification/l1/performance/scheduler_queue_bench.py
verification/l1/performance/serving_bench.py
verification/fleet/resilience/slo_admission_bench.py
verification/product/performance/tiered_auto_bench.py
verification/l1/performance/tp_kv_hit_g2_a7_bench.py
verification/l1/correctness/tp_sampling_owner_bench.py
verification/l1/correctness/tp_sampling_owner_qwen.py
verification/l1/correctness/typed_prompt_qwen.py
verification/orchestration/diagnostic/usage_architecture_bench.py
verification/l1/performance/vllm_quant_kernel_bench.py
```

### G2 A8 DP scaling evidence

`verification/l1/performance/dp_scaling_g2_a8_bench.py` is the checkout-only formal operator for
G2 A8. It compares one Qwen3-32B TP4 replica with two independent TP4 replicas
behind the L2 gateway on the same eight-GPU host. The performance arm uses a
predeclared open-loop arrival-rate grid, at least three fixed-seed paired runs,
with explicit seed 0, and excludes warmup. It does not substitute A6's synchronized-concurrency
binding point for A8's saturation sweep.

The operator retains every request sample and correlates gateway responses with
the replica-placement JSONL log. Its independent verifier recomputes all three
binding verdicts:

- the median paired peak-goodput ratio is at least 1.9;
- nearest-rank ingress-to-replica-selection latency p99 is below 10 ms; and
- the engine-originated multi-turn KV hit rate through the two-replica
  session-affinity gateway is at least 90% of the single-replica value.

Gateway counters and placement reasons prove routing behavior but are never
used as cache-hit truth; cache hits come only from response
`prompt_tokens_details.cached_tokens`. A report cannot say PASS without
complete raw performance, placement, and cache-usage evidence from a real
eight-GPU run. Offline fixtures and unit tests validate the verifier only.
Before traffic, the live read-only model volume is full-hashed (all 17 weight
shards plus index, tokenizer, and model config) and compared with the pinned A7
checkpoint evidence.
The exact launch and replay procedure is in `docs/gpu-runbook.md` §6.

The retained 2026-07-31 eight-GPU artifact contains 2,992/2,992 successful,
retry-free requests and 1,496 correlated placement rows. Router p99 is
3.723 ms and DP retains 99.53% of the single-replica cache-hit rate. The three
paired peak-goodput ratios are 1.9988×, 1.7342×, and 1.7993×; the 1.7993×
median misses the original 1.9× threshold, so verify and replay intentionally
report `passed: false`. The product owner accepted this measured median as an
explicit closure deviation; neither the artifact nor the operator rewrites the
original threshold or claims a formal PASS. Evidence is retained under
`bench/results/g2-a8-dp-qwen3-32b-rtxpro6000-2026-07-31/`.

### Quantized EAGLE-3 draft-head evidence

`verification/l1/correctness/draft_quant_qwen.py` compares the pinned public
`thoughtworks/Qwen3-32B-Eagle3` dense checkpoint with four offline-packed
dynamic FP8 variants against the same real Qwen3-32B target traces. Each arm
receives identical target embeddings, auxiliary residuals, KV contents, and
verification positions. The report retains exact greedy proposals,
target-corrected committed-token counts, draft and verification timing,
acceptance, module/CUDA memory, generated-checkpoint hashes, and environment
provenance. Accepted prefixes must exactly match the independently generated
teacher trace. Target verification and sequential teacher shapes may choose
different tokens only when both selected tokens remain within the established
0.25-nat reciprocal selected-logprob bound; exact correction counts and the
two token IDs, four cross-distribution log-probabilities, individual deltas,
and maximum delta remain visible in the artifact. The bound is fixed by the
formal operator and is not a CLI-adjustable pass criterion. The operator selects the
highest-goodput quantized arm that retains
at least 95% of dense acceptance, then requires memory reduction and at least
95% of dense standalone-cycle committed-token goodput. Five prompts and three
repeats rotate all five arm orders after every measured context; every draft
and target-verify shape is warmed first. This is not production serving E2E: it
includes context construction through target correction but excludes scheduler
and serving overhead. It does not enable model-draft serving or turn a slower
quantized result into a default. The exact invocation and pinned public-weight
digest are in `docs/gpu-runbook.md` §3.1.

The retained RTX PRO 6000 Blackwell run on source `d8dbdba` passes every gate.
Dense and all four FP8 arms accepted exactly 90/270 proposals (33.33%); the
selected `fp8_dense_fc` arm retained 100% of acceptance, used 55.06% of dense
module memory (861,854,720 versus 1,565,296,640 bytes), and retained 98.74% of
dense standalone-cycle goodput (25.842 versus 26.172 committed token/s). Its
draft median was 5.218 ms versus dense 4.287 ms, so the measured result is a
memory win with a 21.71% draft-latency cost and a 1.26% cycle-goodput cost, not
a speedup claim. Every teacher prefix was exact. Corrections were exact in
87/90 repeated rows per arm; the one unique cross-shape divergence had a
0.13118-nat reciprocal delta, and the maximum over every compared correction
was 0.16395 nat, both below the fixed 0.25-nat bound. Evidence:
`bench/results/issue-234-draft-quant-qwen3-32b-rtxpro6000-2026-07-31.json`
(SHA-256 `850191a039edd6e3ff5ae4bf974eadeef3227b3700b1747d281c595daad63c59`).

No compatible trained public MTP target/checkpoint is present in this
environment. MTP therefore makes no trained acceptance or performance claim:
its evidence is canonical packed-checkpoint loading, numerical tolerance, and
the real fused CUDA path with dequantization made fatal. Native EAGLE/MTP
proposal-state integration remains the existing G4 runtime boundary.

### G2 A9 DP-versus-TP crossover evidence

`verification/l1/performance/g2_a9_dp_tp_crossover_bench.py` produces the report-only Qwen3-32B
DP=2×TP4 versus TP8 arrival-sweep artifact. It independently replays the
post-SSE-fix A8 DP evidence retained under
`bench/results/g2-a8-dp-qwen3-32b-rtxpro6000-2026-07-31-ssefix/`, then
measures only the missing TP8 arm through a one-replica production gateway.
That comparator has 2,992/2,992 retry-free successes and 1,496 exact
placements; its sole false check is the accepted 1.9× target at a measured
1.7711× median. Both arms use the same image, commit-`86d4922` materialized
read-only runtime source, checkpoint, workload,
per-engine 8,192 usable KV pages, scheduler limits, pipeline depth, and CUDA
Graph envelope. The report explicitly records that two DP replicas therefore
have twice TP8's aggregate configured KV, sequence, batch-token, and graph-batch
capacity; this is a deployment-topology comparison, not an equal-aggregate-
capacity microbenchmark. TP8 uses the exact A8 DP cache namespace so request
bytes and token IDs match the retained comparator. Every rate
retains three fixed-seed repeats, exact request/placement rows, goodput,
TTFT, and versioned terminal-stream TPOT. PASS means evidence completeness
only; no topology wins by threshold. The report explicitly records whether
any measured ordering transition exists and attaches both arms' observed
concurrency to every transition bracket. The exact launch and replay procedure is
in `docs/gpu-runbook.md`.

The retained 2026-07-31 A9 artifact passed all 14 checks with 984/984
retry-free TP8 successes and placements. Median DP/TP8 goodput at
4/8/16/32/64 offered req/s was 3.884/3.902, 7.383/7.313,
12.948/8.994, 16.042/11.707, and 19.612/12.440 req/s. The measured
ordering transition is between 4 and 8 req/s with no interpolation, and DP is
first noninferior at 8 req/s. TP8 retains lower terminal-stream TPOT at
16–64 req/s, while DP retains higher goodput and lower TTFT under load.
Evidence is retained under
`bench/results/g2-a9-dp-tp-qwen3-32b-rtxpro6000-2026-07-31-ssefix/`.

### G4 M-A2 EP radix-KV evidence

`verification/l1/correctness/g4_ma2_qwen3_235b_ep_kv_bench.py` runs the fixed 512-request
50%-shared-prefix lineage on the real Qwen3-235B NVFP4 EP4 path. One
persistent radix cache, scheduler, and engine loop serve the requests
serially. The logical hit rate comes only from terminal engine usage and must
be strictly above 80%; the four rank rows prove identical first-prefill
allocation/page views and are not additional rate samples. Raw
`BlockStored` events, workload/topology, native kernel inventory, and
source/checkpoint/container/GPU provenance are retained. Completed semantic
failures still write an artifact. GitHub CI runs deterministic tamper tests,
manifest verification, and raw-only replay rather than pretending to have the
four-GPU 235B environment. The exact hardware procedure is in
`docs/gpu-runbook.md` §9.12.

The clean-commit real EP4 run passed all 12 checks: 512/512 requests,
491,008 cached / 557,056 prompt tokens (88.143382%), identical totals and page
identities on all four ranks, 512 raw cache events, and 4,128 retained blocks.
Both retained-copy verification and raw-only replay pass. Evidence, including
the running-container inspect record, is retained under
`bench/results/g4-ma2-ep-kv-qwen3-235b-rtxpro6000-2026-08-02/`.

### G4 M-A3 SGLang comparison evidence

`verification/l1/performance/g4_ma3_sglang_bench.py` is the fail-closed evidence operator for the
fixed Qwen3-235B NVFP4 comparison against SGLang v0.5.16. It is deliberately
not a Docker supervisor: every `run` measures one already-started fresh server
and writes one immutable raw JSONL shard. The companion
`verification/l1/performance/g4_ma3_kairyu_server.py` launches the bounded production Kairyu arm;
model geometry, TP1/attention-DP4/EP4, BF16 KV, FCFS limits, 65,536 aggregate
cache tokens, packed-QKV/native-NVFP4 execution, direct NCCL, and CUDA-graph
limits are fixed, while its sole performance choice is the already-selected
pipeline depth 5. SGLang is pinned by source commit and immutable image digest
with TP4/DP4/EP4, `--enable-dp-attention`, FlashInfer CUTLASS FP4/MoE, no MoE
A2A, BF16 KV, 16,384 cache tokens per owner, HTTP logging fixed at `warning`,
decode CUDA-graph batch size capped at 32, and prefill CUDA graph disabled.
For decode batches larger than one, ordinary non-aliasing same-device scalar
`int64` sampled tokens update Kairyu's persistent input slots through one
vectorized batched D2D operation rather than per-row scalar copies.
Destination-aliasing compatibility views are staged first; public
sampled-token D2H remains unchanged.

`prepare` hashes the exact seed-0 ShareGPT dataset, Qwen tokenizer, 128-request
trace, and a disjoint 348-request graph-warmup trace. Every scenario first
runs four serial requests and global graph bursts of
`4,8,16,32,64,96,128`, retaining each request for 16 completion tokens so the
HTTP arrival wave reaches steady decode. Those bursts cover Kairyu's seven local owner buckets
`1,2,4,8,16,24,32` without charging lazy capture to the measurement. For a
Kairyu shard, `/backends` must show direct NCCL active, all seven buckets
captured, zero eager fallback, no capture/fallback change across traffic, and
a strictly increased replay count.

The model probe and all warmups use one tracked warmup client/pool. Their
traffic must complete and that pool must be fully closed before a distinct
measurement pool is created. The measurement pool starts with zero prior
requests, assigns ordinal zero to the first synchronized measurement request,
and is fully closed after its final runtime witness. Raw shards retain the
client roles, lifecycle timestamps, exact request paths and order, and request
ordinals; `assemble`, `verify`, and raw-only `replay` reject missing, tampered,
reused, overlapping, or out-of-order lifecycle evidence.

The complete matrix is exactly ten fresh, strictly sequential server
generations: one fixed-candidate preflight per arm, then eight binding formal
shards in K/S, S/K, S/K, K/S order. The preflights freeze their raw hashes
before formal traffic. Each formal shard releases the same 128 prompts at
concurrency 128 and requires exactly 128 streamed output tokens per request.
Throughput is successful completion tokens divided by the
first-start-to-last-terminal span and four GPUs; TTFT p99 is nearest-rank over
all 128 requests. The verdict is the exact median of four paired K/S ratios,
requiring throughput at least 1 and TTFT at most 1, with no retry/failure
exclusion, outlier removal, or round-before-gate. No additional measurement
generation is part of the artifact.

The operator hashes the complete checkpoint once before the ten-generation
matrix and once after it; assembly requires identical start/end descriptors
and binds every shard to the start capture. Each shard also binds the
trace/selection, clean source, image RepoDigest/platform/config identities,
fresh container generation, read-only model mount with no read-write volume
consumer, physical GPUs 4–7, runtime argv and package versions, and a live
`/backends` or `/server_info` response. After traffic it re-observes those
container, source, runtime, GPU-process, and volume-consumer facts while the
same server is still running and retains that end witness. The
`capture-provenance --checkpoint-start` command derives that record from the
running container, Docker image/mount/resource state, clean source,
host/container GPU inventories and process ownership, and the live runtime
endpoint; handwritten declarations are not accepted as formal procedure. All
operator commands use the detached, clean `SOURCE_ROOT`, never the mutable
working-checkout script. Formal assembly requires both `--checkpoint-start`
and `--checkpoint-end`. Assembly writes authoritative
`g4-ma3-sglang-raw.jsonl` plus derived
`g4-ma3-sglang-manifest.json`. `verify` checks the stored manifest against a
fresh raw replay; `replay` ignores the manifest entirely. The report always
discloses that SM120 uses FlashInfer CUTLASS instead of the SM100-only
TRTLLM-gen MoE path, disables prefill CUDA graph while retaining decode graph,
and leaves MTP/speculation to M-A4. These limitations never change the gate.

The first complete clean-commit matrix remains a formal FAIL against the
unchanged 1.0 throughput and 1.0 TTFT thresholds; it is not reclassified. The
earlier comparison of Kairyu 571.542867 with SGLang 449.965–481.865 completion
tok/s/GPU is withdrawn because the two arms used incompatible client-pool
lifecycles. A corrected, non-binding fresh-server/fresh-measurement-pool
diagnostic measured Kairyu 536.690626 versus SGLang 551.731445 completion
tok/s/GPU (K/S 0.972739), with TTFT-p99 K/S 0.868731 (1,519.31 versus 1,748.88
ms). A full-server SM120 CUTLASS override reached 530.616804 tok/s/GPU, 1.13%
below `auto` at 536.690626, so throughput priority retains FlashInfer `auto`.
The product owner accepts the remaining 2.73% diagnostic throughput gap as an
explicit closure deviation. M-A3 issue scope is therefore closed without a
formal PASS or a threshold change. The retained formal procedure, clean-server
launch order, CLI sequence, and provenance contract remain in
`docs/gpu-runbook.md` §9.13.

### G4 E-KV FP8 KV evidence

`verification/l1/correctness/fp8_kv_g4_ekv_bench.py` is the formal G4 E-KV correctness operator.
It measures the pinned Qwen3-32B checkpoint on one visible SM120 GPU, writes
raw JSONL plus a derived manifest even on failure, and supports independent
`verify` and raw-only `replay` commands. BF16 KV remains the product default;
the E4M3 candidate arm is explicit and timing is non-binding. The retained
2026-07-31 bake failed output, common-prefix logprob, and cache-NRMSE gates, so
public `fp8_e4m3` startup remains disabled. The exact mounted source-JIT
procedure, thresholds, and retained evidence are in `docs/gpu-runbook.md`
§9.9.

### G5 F4a DRAM KV crossover evidence

`verification/fleet/performance/dram_kv_tier_qwen.py` measures the production rank-local pinned-DRAM
tier on Qwen3-32B at TP4 and TP8. Each shard uses the real RadixKV and
all-rank Gloo control path, compares restore with uncached model recomputation
over the fixed 16–8,192-token grid, and retains nine alternating paired
measurements per length. The primary metric is the rank-0 controller wall from
empty destination pages through one next-token result; pure CUDA D2H/H2D
intervals remain diagnostic evidence rather than replacing the production
boundary.

Restore validates the logical page checksums, transfers KV, replays the final
prompt-token query, and samples. Cold recompute processes the complete prompt
through its natural production chunks, including the final prompt token, then
samples that hidden state without an extra one-token model invocation. Schema
v2 binds every raw shard and runtime profile to the exact versioned
fragment-major CUDA transfer backend. Schema-v1 raw cannot be mixed into,
relabelled as, or used to seed a v2 profile.

Run TP4 and TP8 in separate, non-overlapping containers from the same clean
commit, immutable image, and checkpoint, then seal and verify the artifact:

```bash
python verification/fleet/performance/dram_kv_tier_qwen.py run --tp 4 ... --output tp4-raw.jsonl
python verification/fleet/performance/dram_kv_tier_qwen.py run --tp 8 ... --output tp8-raw.jsonl
python verification/fleet/performance/dram_kv_tier_qwen.py assemble \
  --tp4-raw tp4-raw.jsonl --tp8-raw tp8-raw.jsonl \
  --output-dir bench/results/<f4a-artifact> --assert-gate
python verification/fleet/performance/dram_kv_tier_qwen.py verify \
  --artifact-dir bench/results/<f4a-artifact> --assert-gate
```

The generated TP-specific profiles are startup inputs, not editable tuning
files. Runtime binding replays every raw pair, requires a stable measured
suffix (median restore/recompute ratio below 1 with at least 8/9 restore wins),
and rejects any model, TP, KV-layout, attention implementation, source,
hardware-transport, or host-placement identity mismatch.

The retained 2026-08-01 exact-source run closes F4a. TP4's stable suffix starts
at 1,024 tokens: its 512-token cell failed at a 1.021531 median ratio and 2/9
wins, while 1,024 tokens passed at 0.975449 and 8/9 and every larger cell
passed. TP8 passed all cells from the 16-token measured lower bound, so the
manifest reports the crossover at or below 16 tokens and its deployable
profile conservatively sets `min_restore_tokens` to 16. Assembly, verification
from the retained copy, and independent raw replay all pass. The raw shards,
profiles, manifest, and full container provenance are retained at
`bench/results/g5-f4a-dram-kv-tier-qwen3-32b-rtxpro6000-2026-08-01/`.

### G5 F4b agentic DRAM-tier evidence

`verification/fleet/correctness/agentic_kv_tier_f4b_bench.py` is the checkout-only formal operator for
the Qwen3-32B TP4 agentic tier on/off gate. The predeclared transcript has 16
sessions and eight turns per session. Every prompt contains a 2,048-token
fleet-shared prefix plus 512 new session-history tokens per turn, and every
request generates exactly 32 output tokens. Cached-prompt truth comes from the
production engine's own cached-token accounting, not from an expected prefix
length or a benchmark-side cache model.

The experiment uses two disjoint four-GPU cohorts and four fresh, sequential,
non-overlapping arms. Round 0 runs tier off on GPUs 0--3 and tier on on GPUs
4--7; round 1 swaps the arm assignment, running tier on on GPUs 0--3 and tier
off on GPUs 4--7. This AB/BA cohort swap separates the tier effect from a
fixed cohort effect. Assembly requires a positive engine cached-token-rate
gain in the pooled requests and independently in each cohort, and reports both
absolute and relative gains.

TPOT is request-level `stream-terminal-token-v1`: each request retains its
stream token-boundary timestamps, including the terminal boundary, and p99 is
computed by nearest rank. Tier-on is noninferior only when both the pooled
tier-on/off p99 ratio and the geometric mean of the two cohort ratios are at
most 1.10. The 10% bound is predeclared to tolerate ordinary host and OS
timing variation; it does not permit missing requests, retries, overlapping
arms, or replacement of real stream timestamps with model-only CUDA timing.

Decode-path exclusion is measured at synchronous `EngineLoop` step
boundaries. The raw evidence records DRAM-tier counters and free-page counts
at every output boundary. Offload and restore counters must not advance after
the first output token. The expected free-page decrease from output 16 to 17
must also be observed, proving that the same boundary instrumentation sees a
real decode-page allocation rather than making the zero tier-delta check
vacuous. Raw request timing, cached-token usage, output tokens, and tier/cache
events are retained for independent replay.

Free-running greedy output equality across tier-off and tier-on is retained as
a diagnostic, not treated as a semantic invariant. A BF16 near-tie can choose
a different first token after the cache-prefill shape changes even when both
paths remain numerically equivalent; later tokens then have different
generated prefixes and cannot be compared as though they had the same input.
The performance arms still bind the prompt/order exactly and report same-arm
cross-cohort output stability.

Distribution quality is therefore measured in a separate timing-nonbinding
companion run so that requesting logprobs cannot alter the TPOT evidence. Two
fresh, sequential cohort-A containers repeat the exact tier-off and tier-on
trace with top-64 logprobs. Their checkpoint, trace, GPU cohort, calibrated
image, engine/runtime identity, output tokens, cached-token usage, and
tier-state behavior must match the corresponding performance arms. While the
generated prefix is still common, every selected-token logprob must agree
within 0.25 nat. At the first divergent token, each arm's selected token must
be present in the other arm's top 64 and both cross-arm differences must also
be at most 0.25 nat. Positions after that divergence are not compared. The
quality artifact retains and independently validates the two containers'
created/exited Docker records just as the performance artifact does.

After four raw arms have completed from the same clean source, checkpoint,
retained TP4 F4a runtime profile, and the exact immutable image in which that
profile was calibrated, assemble and replay the evidence without editing a
measured verdict. The calibrated execution image and the read-only
bind-mounted measured source are separate provenance authorities: the image
pins the compiled runtime dependencies, while the source record pins every
executed Kairyu Python file and the engine rollup. Assembly requires the image
inspect plus each arm's full container ID and created/exited inspect pair. It
validates both provenance authorities and those lifecycle records, embeds
their descriptors in the combined raw, and copies the exact input files into
the artifact's `container-metadata/` directory.

```bash
python verification/fleet/correctness/agentic_kv_tier_f4b_bench.py assemble \
  --raw round0-off-raw.jsonl \
  --raw round0-on-raw.jsonl \
  --raw round1-on-raw.jsonl \
  --raw round1-off-raw.jsonl \
  --metadata-dir metadata \
  --output-dir bench/results/<f4b-performance-artifact> --assert-gate
python verification/fleet/correctness/agentic_kv_tier_f4b_bench.py verify \
  --artifact bench/results/<f4b-performance-artifact> --assert-gate
python verification/fleet/correctness/agentic_kv_tier_f4b_bench.py replay \
  --raw \
    bench/results/<f4b-performance-artifact>/agentic-kv-tier-f4b-raw.jsonl \
  --assert-gate
python verification/fleet/correctness/agentic_kv_tier_f4b_bench.py seal-quality \
  --performance-artifact bench/results/<f4b-performance-artifact> \
  --quality-raw quality-off-raw.jsonl \
  --quality-raw quality-on-raw.jsonl \
  --quality-metadata-dir quality-metadata \
  --output-dir bench/results/<f4b-artifact> --assert-gate
python verification/fleet/correctness/agentic_kv_tier_f4b_bench.py verify-quality \
  --artifact bench/results/<f4b-artifact> --assert-gate
python verification/fleet/correctness/agentic_kv_tier_f4b_bench.py replay-quality \
  --performance-raw \
    bench/results/<f4b-artifact>/agentic-kv-tier-f4b-raw.jsonl \
  --quality-raw \
    bench/results/<f4b-artifact>/agentic-kv-tier-f4b-quality-raw.jsonl \
  --assert-gate
```

`replay` recomputes the verdict from the combined raw and validates its
embedded container lifecycle. `verify` does that same replay, checks the
retained manifest, and additionally rehashes every retained
`container-metadata/` byte against the embedded descriptors.
`replay-quality` independently recomputes the distribution verdict from the
unchanged performance raw plus the quality raw, while `verify-quality` also
rehashes both metadata trees and both retained manifests. A publishable
artifact therefore retains the performance and quality combined raw files,
both manifests, and the complete `container-metadata/` and
`quality-container-metadata/` trees together.

The complete clean-source container procedure is in
`docs/gpu-runbook.md` §9.8b. F4b is closed by the retained Qwen3-32B TP4
artifact at
`bench/results/g5-f4b-agentic-kv-tier-qwen3-32b-rtxpro6000-2026-08-01/`.
Tiering raised the pooled engine prefix-hit rate from 47.7941% to 60.2338%
(+12.4397 percentage points). The pooled TPOT p99 ratio was 1.03721 and the
cohort-ratio geometric mean was 1.04488, both within 1.10. No tier counter
advanced after first content, while the decode allocation control fired.
The separate quality arms reproduced their parent outputs, cache usage, and
per-request tier counters exactly. Across 3,968 comparable generated-prefix
positions the maximum selected-logprob difference was 0.195256 nat; the
maximum reciprocal difference at four first divergences was 0.213124 nat.
Every divergent tier-on request restored 160 pages with no fallback or
ownership failure. Sealing, retained-copy verification, and independent raw
replay all passed.

### B7 KV cache-hit/miss answer-equivalence evidence

`verification/l1/correctness/kv_answer_equivalence_bench.py` is an additive, checkout-only semantic
companion for F2c, F2d, F4a, and F4b. It leaves every parent schema, artifact,
performance threshold, and verdict unchanged. Its production matrix is five
fixed topology cells in this exact order: `f2c-tp2`, `f2d-tp2`, `f4a-tp4`,
`f4a-tp8`, and `f4b-tp4`. Before constructing a GPU runtime, every child
replays its exact parent artifact through that gate's owning verifier and
content-binds the reviewed manifest/raw pair. F4b additionally requires the
sealed sibling quality manifest/raw in the same parent directory and replays
the combined performance+quality closure through `verify_quality_artifact`;
a performance-only F4b parent cannot publish B7 evidence.

`--cell` selects all geometry; callers cannot override TP, page counts, DRAM
capacity, decode mode, batch-token bounds, or model-length bounds. All cells
use 16-token pages and a 1,024-token comparison prompt.

| Cell | TP | Device pages | DRAM pages | Decode | Max batched tokens | Max model length |
|---|---:|---:|---:|---|---:|---:|
| `f2c-tp2` | 2 | 8,192 | 0 | CUDA graph | 1,024 | 8,192 |
| `f2d-tp2` | 2 | 8,192 | 0 | CUDA graph | 1,024 | 8,192 |
| `f4a-tp4` | 4 | 1,024 | 512 | eager | 2,048 | 8,193 |
| `f4a-tp8` | 8 | 1,024 | 512 | eager | 2,048 | 8,193 |
| `f4b-tp4` | 4 | 1,024 | 2,048 | eager | 2,048 | 8,192 |

Within one persistent native runtime, the operator submits an isolated prompt
first with a proven cold cache and then with a proven warm cache. Both requests
use identical retained prompt token IDs, `temperature=0`, `seed=0`,
`min_tokens=max_tokens=32`, and `ignore_eos=true`. Each response must report
32 native token IDs and pieces, `finish_reason=length`, and exact usage; the
cold and warm token IDs, pieces, and text must be exactly equal. F2c/F2d rows
prove a native RadixKV miss-to-hit transition, including an independent
non-mutating cache-residency probe immediately before the warm request rather
than copying that request's usage claim. F4a/F4b rows place one or more
pressure requests only between the comparison requests and additionally prove
positive DRAM offload during the pressure interval and positive restore during
the following warm interval, with no restore fallback or ownership failure.
Separating those intervals prevents unrelated pressure traffic or a vacuous
cache label from satisfying the gate.

Every child full-hashes the pinned Qwen3-32B checkpoint: all 17 weight shards,
both weight rollups, architecture/config identity, the exact six metadata files
(`config.json`, `generation_config.json`, `tokenizer.json`,
`tokenizer_config.json`, `vocab.json`, and `merges.txt`), and the separate
safetensors-index hash. F2d's parent contains no model/checkpoint assertion, so
its `aggregate-common-f2c` constraint inherits the checkpoint bound by
`f2c-tp2` instead of inventing parent evidence. Assembly requires one common
checkpoint and clean-source identity across all five cells, plus five distinct
run nonces. Parent manifest/raw/profile bytes, including F2d's router sibling
and both F4b quality files, are snapshotted and checked for drift around owner
replay and semantic extraction. Reviewed structured values use canonical JSON type-exact equality
(so booleans, integers, and floats cannot coerce), and the quality raw must be
strict canonical JSONL without duplicate keys or non-finite values. Parent
row indices, fixed scalar types, and F2d router rows are independently
schema-checked. Source identity is checked
both before runtime construction and after shutdown, including rejection of
untracked/ignored executable import artifacts anywhere in the repository and
verification that loaded repo modules are tracked `HEAD` bytes. Repository
`__pycache__` bytecode is deliberately not exempt: remove every such directory
before a formal capture. The repo-local `.venv` is the sole generated-code
exclusion, so its locked interpreter, site bootstrap, and installed packages
are part of the formal environment TCB. The complete checkpoint and any DRAM profile are likewise
rehashed after shutdown, so evidence cannot straddle a source, parent, profile,
or model swap.

Formal B7 evidence uses only the trusted direct checkout path shown below, in
a fresh `python -I -B` process. Evidence commands refuse a plain interpreter
rather than attempting an in-script restart after `sitecustomize` may have
run. After isolated startup and before adding the
repository to `sys.path`, the wrapper creates an unpredictable, exclusive 0700
directory under `/tmp` with `tempfile.TemporaryDirectory`, assigns it to
`sys.pycache_prefix`, and runs the tracked-clean and import-artifact preflight.
That internal pre-import check proves `git diff --quiet HEAD --` and rejects
every untracked or ignored `.py`, `.pyc`, `.pth`, native `.so`, or symlink
import artifact outside `.venv`; an operator may run the same Git checks in
advance, but they do not replace the wrapper check. `-I` excludes ambient
`PYTHONPATH` and user-site imports, while the selected interpreter, `.venv`
bootstrap, and installed packages remain explicit TCB. The module form is
excluded from the declared inventory; only unsupported, non-evidence `--help`
remains as a convenience. `python -m`
evidence capture, assembly, verification, and replay are unsupported and
non-publishable.

The existing F2c and F4b cross-arm output comparisons remain diagnostic. Those
arms deliberately differ in replica, GPU, container, or prefill shape, and a
BF16 near tie may move one free-running greedy token without demonstrating KV
corruption. B7's same-runtime isolation removes those confounds; it does not
promote the old diagnostics or replace F4b's distribution-quality evidence.

```bash
python -I -B \
  verification/l1/correctness/kv_answer_equivalence_bench.py run-native \
  --cell f2c-tp2 --model-path <model-path> \
  --parent-manifest <f2c-parent-manifest> --parent-raw <f2c-parent-raw> \
  --output <run-root>/f2c-tp2.jsonl --assert-gate

# Repeat run-native with --cell f2d-tp2. Run f4a-tp4, f4a-tp8, and
# f4b-tp4 with their exact parent pair and the additional
#   --dram-profile <cell-bound-profile>.
python -I -B \
  verification/l1/correctness/kv_answer_equivalence_bench.py assemble \
  --f2c-tp2 <run-root>/f2c-tp2.jsonl \
  --f2d-tp2 <run-root>/f2d-tp2.jsonl \
  --f4a-tp4 <run-root>/f4a-tp4.jsonl \
  --f4a-tp8 <run-root>/f4a-tp8.jsonl \
  --f4b-tp4 <run-root>/f4b-tp4.jsonl \
  --output-dir bench/results/<b7-artifact> --assert-gate
python -I -B \
  verification/l1/correctness/kv_answer_equivalence_bench.py verify \
  --artifact bench/results/<b7-artifact> --assert-gate
python -I -B \
  verification/l1/correctness/kv_answer_equivalence_bench.py replay \
  --artifact bench/results/<b7-artifact> --assert-gate
```

`verify` checks retained hashes and the derived manifest before replaying the
raw rows. `replay` reads each raw file into one immutable byte buffer and both
parses and hashes that same buffer before independently recomputing the
five-cell verdict. Portable tests exercise fake-native plumbing, tamper
rejection, and replay only; they never claim a real model, GPU cache hit, or
GPU DRAM transfer. The complete contract and real-GPU procedure are in
`docs/design/issue-373-kv-answer-equivalence.md` and
`docs/gpu-runbook.md` §9.8c.

### A12 batch-invariance determinism evidence

`verification/l1/correctness/batch_invariance_bench.py` is the checkout-only formal gate for issue
#360. It runs one retained 129-token factual prompt under four production
shapes on the pinned Qwen3-32B checkpoint: cold in a real batch of 32, cold
alone after a proved eviction, warm alone after a proved 128-token page-aligned
radix hit plus the required one-token terminal recomputation,
and cold alone in a fresh runtime with the prefill budget forced to 32 tokens.
The last arm must schedule the exact chunk sequence `32, 32, 32, 32, 1`.

All target requests use greedy seed 0, produce exactly 32 tokens, ignore EOS,
and preserve special tokens. The gate requires exact equality of native output
token IDs, raw vocabulary pieces, and final text across all four arms. It does
not use latency, text-only equality, prefix equality, or a disagreement
tolerance. Thirty-one distinct distractors are forced to remain live for the
whole batch arm, and retained scheduler observations must prove one 32-row
prefill followed by target-bearing 32-row decode cohorts. A configured batch
size or a synthetic scenario label is not sufficient evidence.

The production geometry is fixed at TP8, FlashInfer, BF16 KV, CUDA graph,
128 configured 16-token pages (127 allocatable after graph scratch), model
length 192, graph batch 32/page width 16/three warmups, a 512-token full-runtime
budget, and a 32-token fresh-runtime budget. Exactly six fixed lowercase
129-token pressure requests are required; their roots are disjoint from the
distractors, and five do not evict the target under this geometry. All eight
ranks must report the expected topology, FlashInfer backend,
graph bucket/capture/replay activity, and zero eager fallbacks. The cache proof
uses independent non-mutating probes: zero target tokens after pressure and
before the alone-cold request, then 128 cached tokens before the warm request.
Native usage must agree and the warm schedule must recompute the final prompt
token, as required by RadixKV's page-aligned reuse contract. Its one-token
attention call is deliberately bound to the production decode-backend counters,
not misclassified as a FlashInfer prefill-backend call.

The target's exact text, 129 tokenizer IDs, and both hashes are schema-fixed.
Every prompt and output ID must be below Qwen3's 151,669-entry tokenizer
boundary; the model's padded 151,936-row logits head is not accepted as a
tokenizer domain. The operator full-hashes all 17 checkpoint shards, tokenizer
metadata, source paths, and the hardware/runtime environment before execution
and again after both native runtimes shut down.

Formal `run-native`, `verify`, and `replay` commands are path-only and must
start with an already isolated `python -I -B` interpreter. They retain one
canonical JSONL raw stream plus a derived manifest, never overwrite an
existing artifact, and independently replay every schedule, cache, rank,
provenance, and equality check rather than trusting stored `passed`. Portable
tests exercise strict replay, tamper rejection, fake-native plumbing, and the
ordinary DeepSeek/MLA batch fallback only; they do not claim a real Qwen3-32B
or GPU result. The complete contract is in
`docs/design/issue-360-batch-invariance.md`; the exact capture commands are in
`docs/gpu-runbook.md` §9.8d.

## Results and package boundary

New verification runs write to ignored paths below `verification/results/`;
retained formal evidence may instead bind an explicit legacy `bench/results/`
path. Existing tracked files below `bench/results/` are immutable migration
inputs and are not moved or rewritten. Neutral serialization, hashing, and
reporting contracts live in `evidence/`.

Evaluation fixtures and suite policy belong to checkout-only `evals/`, not to
verification or the product wheel. The packaging gate builds a real wheel,
inspects its contents, and imports it from an isolated temporary directory. It
proves that the product CLI is present while the top-level `bench/`, `evals/`,
`evidence/`, `verification/`, and `tests/` trees are absent:

```bash
uv run --frozen python scripts/verify_verification_registry.py
uv run --frozen python scripts/verify_bench_results_index.py
uv run --frozen python scripts/verify_bench_wheel.py
```

The registry check exercises each declared wrapper through its registered
`--help` forms. The results-index check preserves legacy discovery without
mutating tracked artifacts.
