# OpenJev DiffusionGemma single-replica example on one GPU (think = 512)

Status: **Approved 2026-10-01; CPU half implemented, GPU gates pending.**

Accepted scope (owner, 2026-10-01):

- L1 is OpenJev (https://github.com/razorback16/openjev): one container
  runs vLLM with DiffusionGemma 26B-A4B NVFP4 behind OpenJev's API server
  (`/health`, OpenAI-style `/v1/chat/completions` as `diffusiongemma-26b`,
  and the Jev `/v1/systemone` API).
- One GPU, one replica.
- L2 and L3 pass requests through, exactly like the other single-model
  examples: one `ReplicaPool` replica, the legacy OpenAI chat/tool path,
  image admission, and a no-login Open WebUI. No orchestration.
- Every chat completion thinks first, with a thought of at most 512
  tokens. The cap is hard: a longer thought is cut, with the same meaning
  as OpenJev's System One `think`. Then the answer is written. The budget
  is fixed: a caller's `enable_thinking: false` or `reasoning_effort` does
  not change it, and the Chat UI has no effort selector.
- The branch stacks on PR #613 (`claude/v41-tiered-six-gpu`) and becomes a
  separate PR.
- GPU verification is not run on the development machine. Every GPU gate
  is a remaining task in the PR.

Example directory: `examples/openjev-diffusiongemma-26b-1gpu/`. Ports: API
8010, Chat UI 3010 (both free on PR #613 and on `main`).

## Sources (pinned)

1. OpenJev `dcd20947b5ddad5be4a8f5aed6aa6dd245653823`, which was `main` on
   2026-09-29 (package 0.5.0). Upstream publishes no tags or releases.
2. OpenJev image `razorback16/openjev:0.5.1@sha256:65f88680e0093c9229d8cc55c0d2f279db27aba78a7ce74ea762b8f2da82d47d`.
   Its build history records vLLM `1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8`
   (vllm#57250, structured reads for DiffusionGemma) and OpenJev's two
   build edits: the 512 logprob-id cap and the vision prefix-LM patch.
   Its image ID (config digest) is
   `sha256:b219be87fdde14896b0029edabe5586b2e99c89380784a9ec73f1405a0453757`.
3. Checkpoint `nvidia/diffusiongemma-26B-A4B-it-NVFP4` at
   `ec4ff3df205028f4e81c954c2227f9312b3ec2ea`: Apache-2.0, not gated, 14
   files, 18.86 GB. It supports a 256K context, `enable_thinking`, native
   function calling, and images.

## What the sources say, and what applies here

| Topic | Source | Applies here? |
|---|---|---|
| Serving stack | OpenJev's image runs vLLM on :8000 inside the container and its API on :8080. `OPENJEV_VLLM_ARGS` appends `vllm serve` flags. | Yes. The image is used unchanged except for this example's overlay, described below. |
| Chat endpoint | `diffusiongemma-26b`; sampling fields dropped (vLLM refuses them for diffusion models); `max_tokens` default 1024, cap 8192; 8 generations in flight plus 32 queued, then 529. | Yes. Kairyu bounds concurrency at 40, so a 529 never ejects the only replica. |
| Thinking on chat | OpenJev forces `enable_thinking: false` unless the caller sends it, and drops `reasoning_effort`. | Replaced by the think-512 overlay. |
| vLLM thinking budget | `thinking_token_budget` with `--reasoning-config` exists at `1b3b88ec`. | Rejected: DiffusionGemma's `DiffusionSampler` replaces the AR sampler and never applies `ThinkingBudgetState`, so the cap would not be enforced. |
| System One `think` | Two passes: a thought of at most N tokens (stops at `<channel|>`, cut at N), then a read after the closed thought. | The same method is applied to chat. |
| Chat template | `strip_thinking` removes a thought from every assistant message. | One edit: the final assistant message is rendered verbatim, so a thought prefill can be continued. |
| Laya, Verdict, CLM, JevK5 | Optional extra containers on the same GPU. | Not used: one GPU, one model. |
| Canvas, context | `OPENJEV_CANVAS=64`, `--max-model-len 65536`, `--max-num-seqs 64`, `--gpu-memory-utilization 0.9`. | OpenJev's defaults are kept; none are tuned. |

## L1 plan (OpenJev on one GPU)

Overlay (`openjev-think.Dockerfile`, built from the pinned image):

- The build first checks the base, and fails if any check does not pass:
  vLLM must carry the 512 logprob-id cap, the vision prefix-LM patch, and
  version `1b3b88ec`.
- It installs OpenJev at the pinned revision with `--no-deps`.
- It copies in `think_core.py` and `think_first.py`. `patch_openjev.py`
  wires `ThinkFirstGenerator` into `openjev/api.py`, using exactly-once
  anchors.
- It bakes in `chat_template.jinja` and sets `OPENJEV_VLLM_ARGS` to use it.

Every chat completion makes two calls to vLLM, inside one generation slot:

1. **Thought.** The prompt is the conversation plus an assistant prefill
   `<|channel>thought\n`, with `continue_final_message`, thinking on,
   `max_tokens: 512` and `stop_token_ids: [<channel|>]`.
2. **Answer.** The conversation plus `<|channel>thought\n{thought}\n<channel|>`.
   The caller's `max_tokens`, `stop`, `logprobs`, `tools` and `tool_choice`
   apply to this call.

The response returns the thought as `reasoning`. Usage counts the prompt
once, plus thought and answer tokens. When the client streams, it receives
the reasoning deltas, then the answer deltas, then the merged usage. At
startup the overlay checks three things and refuses to start if any fails:
the baked template, the rendered prefill continuation, and that the
thought-close marker is a single token.

Stages run in order:

- **L1-0 Image and correctness.**
  - `OPENJEV_ALLOW_UNPINNED_IMAGE=1 ./run.sh` builds the overlay.
  - The image ID is pinned in `example.json` and in `kairyu.yaml`
    `container_image_digest`. Serving is then rerun with the pin.
  - Pass rule: model attestation, the startup checks, and the readiness
    probes (`17 * 19`, a tool call, a red image, each with reasoning
    present) all pass, and `verify.sh l1` passes. `verify.sh l1` must show
    that `enable_thinking: false` still thinks, that `reasoning_tokens` is at
    most 512, that a budget-exhausting prompt returns exactly 512 thought
    tokens plus an answer, and that vLLM receives two requests per chat.
- **L1-1 Generation concurrency.** Compare `OPENJEV_GEN_MAX_INFLIGHT` at 8
  (OpenJev's default), 16 and 32, with `OPENJEV_GEN_MAX_QUEUE = 40 −
  inflight`. The Kairyu configuration does not change between candidates.
  Use the completed-answer matrix at c1, 8, 16 and 32.

**Selection rule.** A candidate replaces the current choice only if it
passes L1-0 and improves c32 output tok/s by ≥ 5 % without regressing c1 by
more than 5 %. Smaller differences are reported as no change. OpenJev's
default is kept when nothing wins.

## L2 / L3 (the single-replica structure)

- `kairyu.yaml`:
  - pool `diffusiongemma-26b` with one `backend: openai` replica;
  - `health_url: http://openjev:8080/health` (Kairyu's default `/readyz`
    does not exist on OpenJev), `api_key_env: null`, `timeout_s: 1800`,
    `upstream: vllm`;
  - multimodal admission with up to 8 images;
  - `deny_sampling_fields` for the fields OpenJev drops silently, so Kairyu
    rejects them with a 400. `temperature` stays accepted and ignored, as
    OpenJev documents;
  - `legacy_chat_models`, with no Kairyu chat template;
  - `server.max_concurrency: 40`, which equals OpenJev's 8 in flight plus 32
    queued.
- `reasoning_effort` is forwarded and then ignored at L1.
  `chat_template_kwargs` is rejected by Kairyu's existing legacy-chat
  contract.
- Open WebUI runs without login and without an effort selector. Its
  default `max_tokens` is 8192.
- The public model name is `diffusiongemma-26b`. OpenJev's System One API
  stays inside the compose network.

## Files and code ownership

- New: `examples/openjev-diffusiongemma-26b-1gpu/{README.md, MEASUREMENTS.md,
  example.json, compose.yaml, kairyu.yaml, run.sh, verify.sh, control.py,
  verification.py, benchmark.py, openjev-think.Dockerfile, patch_openjev.py,
  think_core.py, think_first.py, chat_template.jinja}`.
- Every script and overlay file belongs to this example. No file of
  another example is imported or modified.
- Kairyu (`kairyu/`) is not changed. The reasoning budget and the runtime
  adaptation are example-owned (`.claude/rules/framework-boundary.md`).
- Tests (CLAUDE.md test policy):
  - one example-owned file, `tests/unit/test_openjev_1gpu_example.py`;
  - it covers the think-512 contract against a fake vLLM: non-stream,
    stream, budget exhaustion, tool calls, and a failure in the second pass;
  - it checks that the served configuration loads and drives OpenJev's chat
    path;
  - it checks that the shared answer validator rejects false passes.
  - The registry list in `tests/unit/test_frontier_examplectl.py` gains one
    entry.
- Docs: FN-D9 OpenJev one-GPU amendment in `docs/design/frontier-native-runtime.md`;
  an `examples/README.md` row; a `PROGRESS.md` Change Log entry and one
  open-items bullet.

## CPU evidence (development machine)

- The patch applies to OpenJev `dcd20947`, and the patched modules compile.
- With the checkpoint tokenizer, the edited template renders every
  conversation that does not end in an assistant message byte for byte like
  the stock template. Thought prefills continue as designed.
- The checkpoint tree hash is computed from HF metadata plus the small
  files.

## Final gates (on the selected configuration)

`verify.sh` gates:

- `l1`
- `serving` (completed answers at c1/8/16/32, with placement on replica 0)
- `think` (default and `reasoning_effort` low/high still think; reasoning
  round-trips)
- `tool-calling`
- `vision`
- `cancellation` (during the thought and during the answer)
- `restart`

Plus a manual Chat UI check on :3010. Evidence, run IDs, hashes and
limitations go in `MEASUREMENTS.md`.

## Checklist

- [x] Owner approves this plan.
- [x] Branch, example files and example-owned scripts, CPU tests, lint, CPU evidence.
- [ ] L1-0 and L1-1 on the GPU host; record every candidate.
- [ ] Final gates; MEASUREMENTS.md; FN-D9 amendment status; PROGRESS.md.
