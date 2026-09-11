# Frontier model native runtime and example boundary

Status: **DeepSeek native EP/Attention-DP and SM120 packed-FP4 execution are
implemented; full-checkpoint native production gates remain open; the example
surface is superseded by FN-D8** (2026-08-11)

This document amends FZ-D1 in `frontier-model-zoo.md`. It records what the
frontier example rebuild may claim before full-checkpoint GPU evidence exists.

## FN-D1 — Production selection is explicit and fail-closed

`execution_mode: native` is mandatory in the new Qwen3.6 and DeepSeek V4
Kairyu example configs. `execution_mode: reference` is retained only for CPU,
small fixtures, and diagnostics. A frontier architecture cannot silently enter
the generic paged-KV runner, use vLLM inside Kairyu, enable a draft decoder, or
reduce `max_model_len` when a capability or memory check fails.

The native single-rank runner advances only new tokens through the official
architecture implementation's `forward_cached` contract. Kairyu, rather than
that model wrapper, owns admission, scheduler lifetime, prefix identity,
sampling, cancellation, and cache rollback.

## FN-D2 — CacheDescriptor is the scheduler-facing ABI

`CacheDescriptor` and `CacheHandle` expose a model-specific composite cache
without pretending all state is KV:

- Qwen: FP32 gated-DeltaNet recurrent/conv state plus BF16 paged KV for the
  full-attention layers.
- DeepSeek: block-256 HCA and CSA state, 4/128 compression metadata, sparse
  top-k/indexer state, FP4-indexer-cache provenance, and mHC state.
- Prefix reuse stores only complete state snapshots in a byte-bounded LRU.
  A generic token-prefix hit alone never skips recurrent/compressed work.
- Transactions clone opaque state before commit and restore it on rollback.
  Nested transactions are rejected.

The current runner keeps opaque addresses model-owned and therefore remains
eager. CUDA Graph pointer stability and model-specific speculative
commit/rollback require their separate GPU gates before they can be enabled.

## FN-D3 — Checkpoint and parser trust boundary

Qwen3.6 and DeepSeek V4 are loaded through pinned Transformers architecture
classes with remote code disabled in the Kairyu process. The DeepSeek loader
validates every official checkpoint header, shards only routed experts,
preserves packed E2M1/UE8M0 experts and block-FP8 nonexperts, and disables
remote code. The pinned fine-grained Triton kernel executes FP8 activations
against the checkpoint's FP4 bytes directly on SM120; single-GPU kernel and
two-rank NCCL dispatch smokes are green. Full-checkpoint numerical and 1M
evidence remains a separate gate.

The L3 API normalizes OpenAI-style reasoning_effort aliases
(minimal/low→low, medium/high→high, xhigh/max→max), preserves
reasoning_content in complete and streamed responses, and parses the pinned
DeepSeek DSML tool-call envelope. OpenAI-compatible replica gateways render the
checkpoint chat template before sending an identity-wrapped request; Kairyu
does not use legacy role concatenation.

## FN-D4 — DeepSeek native distributed execution

The native worker supports request-owned Attention-DP with EP2/4/8. Each rank
retains its own sliding/HCA/CSA state, while every prefill/decode phase agrees
its forward count and pads missing-rank work before entering expert
collectives. Routed experts use equal-capacity NCCL all-to-all dispatch and
combine; ragged rank token counts require no host-derived split vectors and
top-k contributions are restored in deterministic slot order before one BF16
cast.

Two EP4 replicas remain the default example. A separate one-replica EP8 Compose
profile is selected only by the committed topology gate. No EP8 topology lock
is generated until real-checkpoint EP4/EP8 quality, 1M context, stability and
SLO-goodput evidence passes, with EP8 at least 2% ahead. CUDA Graph, DSpark,
30-minute soak, failure recovery and full-checkpoint 1M results remain open.

## FN-D5 — Orchestration policy

The L2 DSL can load a SHA-256-pinned calibrated router artifact. Artifacts
below a 0.99 quality-ratio confidence lower bound are rejected. `auto-max`
maps to three Tier1 proposals plus Tier2 synthesis. Tier1 direct failures retry
Tier2 once; a stream retries only before any output has been emitted, avoiding
mixed answers. Inputs are never truncated by the router or gateway.

The checked-in router is an all-Tier2 structural baseline. It is safe by
construction but does not claim Tier1 goodput. A measured train/holdout
artifact may replace it only after the benchmark calibration gate passes.

## FN-D6 — Rebuilt examples and evidence

`examples/` contains only the shared controllers and the Qwen 1-GPU,
DeepSeek 8-GPU, and combined 8-GPU environments. Model revisions, external
images, CUDA bases, contexts, GPU counts, VRAM, disk, SM120 capability, and
NUMA-local CPU sets are fail-closed. The first download hashes every model file
and subsequent starts mount the same volume read-only with offline mode.

Each environment exposes `run.sh` for lifecycle management and `verify.sh` for
serving verification. CLI enumeration, shell syntax, Compose expansion, and
report mechanics are CPU/static gates; measured performance remains a GPU gate.
Model and product evaluation is invoked separately through `python -m evals`.

## FN-D7 — Enablement gates

- Qwen MTP stays off until greedy equality, sampling-path invariants, and at
  least 5% SLO-goodput improvement pass.
- DeepSeek DSpark stays off under the checkpoint-declared 5-token gate.
- `PROGRESS.md` must not claim production frontier support until the real
  Qwen 262K and DeepSeek EP4/EP8 1M GPU runs, 30-minute soak, OOM/worker-failure
  recovery, and vLLM comparison all close.

## FN-D8 — The user-facing example is one measured vLLM deployment

This decision supersedes FN-D4's default-example topology and FN-D6's three
environment surface; it does not remove or weaken the native-engine gates.
`examples/` now contains exactly one deployment for the available 8 x RTX PRO
6000 Blackwell Server Edition host: Open WebUI calls Kairyu L3, and Kairyu calls
one vLLM L1 using all eight GPUs. The checkpoint's exact prompt encoder remains
owned by Kairyu and is preserved through an identity template at vLLM.

The committed default is selected only after same-host topology and feature
measurements. Its verification report records TTFT and output throughput. Model
and product evaluation is a separate checkout-only workflow and is not a
prerequisite embedded in the serving-performance runner. Public heterogeneous
figures are comparison context, not a substitute for local measurements.

**Layered-product amendment (2026-08-13, EO-D2..EO-D5).** The measured vLLM
services may remain transitional L1 workers while the tiered example proves
its direct L2-to-L1 object boundary, bounded verifier loop, one-model public
inventory, and separate model-attributed intermediate-output UI. That
structural pass does not satisfy the native production gate: the default may
be called native Kairyu L1 only after the full-checkpoint gates in FN-D7 pass.
The binding example contract is `example-layered-orchestration.md`.

## FN-D9 — Replica-pool scale-out examples (amendment, 2026-09-01)

Status: accepted; implemented; GPU-verified 2026-09-01/02 (placement gates
green at c8–c64; runs `20260901T133331Z` Qwen, `20260902T005136Z` DeepSeek
after the tool-calling amendment below; `verify.sh tool-calling` green on both
— see the examples' `MEASUREMENTS.md`).

This amends the FN-D6/FN-D8 example surface: `examples/` gains two
environments in which Kairyu L2 does **no orchestration** — it is only the
`ReplicaPool` spreading one public model over identical vLLM L1 replicas on
the eight-card host:

- `qwen3.8-27b-dp8-8gpu`: Qwen3.8-27B-FP8 as 8 × TP1 replicas (one per GPU),
  each carrying the single-GPU example's measured L1 envelope.
- `deepseek-v4-flash-0731-dp2-8gpu`: DeepSeek-V4-Flash-0731 as 2 × TP4+EP4
  replicas (GPU 0-3, 4-7), each carrying the tiered example's measured Tier2
  envelope (DSpark-5, 16K batch, 32 sequences).

Placement policy (both pools): `prefix_index: true`, `queue_depth_threshold: 0`,
`unhealthy_after: 1`. A warm prefix is reused only while its replica is idle;
otherwise strict least-outstanding (m5 D4 / m10 D6 semantics), so concurrent
traffic spreads one-per-replica before any replica takes a second request.
The pool's `placement_log_path` is the evidence surface: `verify.sh serving`
reads the per-row JSONL delta and fails a row at concurrency ≥ 8 unless every
replica received traffic and none took more than 1.25× the even share; c1 is
reported only (least-outstanding ties resolve to the lowest replica id).
Checkpoints, templates, images, and L1 flags are shared by reference with
the sibling examples; no product code changed. Existing FN-D7 gates are not
weakened: these are vLLM-backed L1 deployments, not native-engine claims.

**Tool-calling amendment (2026-09-02, PR #584 review).** The served DP2
example returned `tool_calls: null` (DSML markup leaked into `content`), so
the official SWE-bench Pro mini-swe-agent failed every turn — a served example
that cannot drive tool agents does not satisfy this decision. The
Kairyu-rendered `/completions` passthrough shape is abandoned for these
examples: Kairyu never forwards `tools` on that path, and Kairyu's DSML parser
accepts only a whole-completion DSML block, which the prose-plus-call agent
format never satisfies. Both replica examples now use the Qwen examples'
layering — vLLM owns the chat rendering (for DeepSeek via the checkpoint's own
`deepseek_v4` encoder, which also renders DSML tools and merges `tool` turns)
plus `--enable-auto-tool-choice --tool-call-parser {qwen3_coder|deepseek_v4}`,
and Kairyu (`legacy_chat_models`) forwards tools to `/chat/completions` and
normalizes the parsed calls. Thinking defaults off in both via
`--default-chat-template-kwargs` (`thinking`/`enable_thinking: false`);
`reasoning_effort` re-enables it. For Qwen the flag is required even though
the Kairyu-owned template already renders non-thinking prompts: vLLM's `qwen3`
reasoning parser otherwise assumes thinking and files a plain answer (no
`</think>`) as `reasoning_content`, leaving `content` empty — caught by the
gate's non-thinking case on the first GPU run. The
example contract now includes fail-closed tool-calling evidence: a readiness
probe in `run.sh up` and the `verify.sh tool-calling` gate (auto call on every
replica, tool-result turn, streaming, thinking, non-thinking default).

**Vision replica amendment (2026-09-04).** Two more replica-pool examples
apply the same layering to the newly released vision-language checkpoints,
each as 2 × TP4 replicas (GPU 0-3, 4-7) — the only eight-card split that fits
either checkpoint (neither fits one 96 GB card; Qwen's 128-wide FP8 blocks
reject TP8 and pipeline parallel is unsupported):

- `deepseek-v4-flash-vision-exp-dp2-8gpu`: DeepSeek-V4-Flash-Vision-Exp
  (revision `6821d6ad`) with the official recipe's TP4+EP, FP8 KV, 256-token
  blocks, DSpark k=3 probabilistic drafting, `deepseek_v4` parsers, 1M
  context; SM120 pins `--moe-backend marlin` and disables DSpark adaptive
  verification. Effort levels `low/high/max` are the encoder's own vocabulary.
- `qwen3.8-flash-next-dp2-8gpu`: Qwen3.8-Flash-Next-FP8 (revision `236dfdf2`)
  with the official recipe's verified `rtx_pro_6000_4x` layout (16 sequences,
  8K batch tokens, 0.95 memory, prefix caching, `qwen3_xml`/`qwen3` parsers),
  256K context, **without the recipe's MTP k=3**: on `vllm@27a94d1c` prefix
  caching + MTP corrupts batched answers on hybrid GDN models
  (vllm-project/vllm#53912; reproduced 13/274 at 2-12 concurrent, 0/1,508
  with either feature off, 63.8% with `--no-async-scheduling`). Prefix
  caching stays because Kairyu's prefix-aware placement and multi-turn
  traffic depend on it; the price is single-stream decode 104 vs 175 tok/s.
  The KV budget is pinned (`--kv-cache-memory` 43.16 GiB, the warm-start
  value for 0.95 utilization) because a cold torch.compile cache inflates
  vLLM's start-up memory profile and a first boot otherwise serves 741K KV
  tokens instead of 3.45M. Kairyu L3 normalizes the wire vocabulary `medium→high`,
  `xhigh→max`, and the official template rejects anything outside
  `low/medium/xhigh` (HTTP 400), so the example-local template aliases
  `high→medium`, `max→xhigh` at the top and is otherwise byte-identical.
  Selecting an effort in the Chat UI also drops the pinned instruct sampling
  (T 0.7 / top_p 0.8 / presence 1.5) so vLLM applies the checkpoint's
  thinking `generation_config` — the two official sampling modes, not a
  blend.

Both are vision-capable (`allow_prompt_kinds: [multimodal]` paired with an
`image_input_policy`, Kairyu built with the `vision` extra), run the Chat UI
without login and provision the tiered example's Reasoning Effort dropdown
(fail-closed enum check) from `run.sh up`, and add a `verify.sh vision` gate
plus an image readiness probe (the first image request is where the SM120
sparse-MLA path failed on FlashInfer 0.6.18). vLLM image: no release or
official tag carries the merged support, so both share one overlay image —
upstream's digest-pinned nightly of `vllm@27a94d1c` plus FlashInfer
`60b49158` (#4802) with the stale AOT module cache removed — and `run.sh up`
refuses to serve if the built image ID differs from the pinned
`container_image_digest`. The vision gate requires the answer to name the
probe colour (a non-empty check let the corrupted `ductduct…` output pass
once). Status: GPU-verified 2026-09-04 on 8 × RTX PRO 6000 (tree hash /
image ID pinned; all three gates PASS for both examples; results in each
example's `MEASUREMENTS.md`).

### V4.1 Flash single-replica amendment (2026-09-11)

Status: accepted; implementation and GPU validation in progress.

`examples/deepseek-v4.1-flash-8gpu` uses one TP8 replica across GPUs 0–7,
retaining the V4 vision example's ReplicaPool, legacy OpenAI chat/tool path,
image admission, and Chat UI. The owner's revised allocation supersedes the
initial two-replica proposal. TP/EP, DSpark, batch limits, CUDA graphs, and
Engram CPU/GPU placement are L1 measurement choices on the SM120 host.

Omitted effort means thinking `high`. The model-author encoder at the pinned
checkpoint maps `low/high/max` to `50/75/100`; the initial vLLM image instead
maps `high` to 50. An image-local encoder adjustment aligns those aliases,
with the Python frontend selected and rendered-prefix checks at build time.
Kairyu's L2/L3 effort normalization remains the existing contract.

Fixed-length performance rows record first model output (reasoning or
content) separately from first visible content, which stays null if no
content was emitted. Completed-answer/tool/image gates are independent.
Only SHA-bound measurements in the example's `MEASUREMENTS.md` establish
the final runtime and performance claims.
