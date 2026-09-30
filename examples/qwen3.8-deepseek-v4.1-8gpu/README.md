# Qwen3.8 × 2 + DeepSeek-V4.1 (six GPUs) judged ensemble on 8 × RTX PRO 6000

This example serves the ensemble product of
[`qwen3.8-deepseek-v4-8gpu`](../qwen3.8-deepseek-v4-8gpu/README.md) with a
different L1:

- DeepSeek-V4.1-Flash replaces DeepSeek-V4-Flash-0731 and runs on six GPUs.
- Qwen3.8-27B runs on the other two GPUs.

V4.1 accepts images, so every role sees attached images directly and there
is no Qwen image-description stage. Design: DTO-D16 in
[`example-dual-track-orchestration.md`](../../docs/design/example-dual-track-orchestration.md).

```text
Open WebUI (:3009)
    -> Kairyu L3 product API (:8009; model kairyu-auto-max)
        -> Kairyu L2 route judge (one Qwen non-thinking call, DTO-D13):
             QWEN / QWEN_THINK / DEEPSEEK / DEEPSEEK_THINK / ENSEMBLE
        -> primary profile: the dual-track ensemble DAG
           (10 roles: 9 generation + 1 audit verifier)
             Track A: DeepSeek writes 4 maximally different answer policies;
                      4 Qwen answerers follow them in parallel on 2 replicas
             Track B: a Qwen quick draft, critically refined by thinking DeepSeek
             Merge:   thinking DeepSeek synthesizes one better answer from the
                      5 peer candidates; a Qwen audit verifies it (PASS/FAIL,
                      <= 2 refinement rounds) before the remainder streams
             Images:  every role receives the request's images itself
        -> L1 pools: DeepSeek-V4.1-Flash DP6/EP6 (GPU 0-5),
                     2 x Qwen3.8-27B-FP8 TP1 (GPU 6, 7)
Embedding clients -> Kairyu L3 embeddings API (:8009; model embed-small)
```

## What is inherited and what changed

Inherited unchanged from the V4 example:

- the judge, its five routes, and its criteria;
- the DAG's roles, prompt bodies, seeds, and sampling;
- the DeepSeek effort budgets (8192/32768/65536) and the 65536 ceiling;
- the public-output floor (256);
- the audit loop and the `{…, 2}` refinement bound;
- the head stream and its TTFT gate;
- embeddings, the one public chat model, and the Chat UI effort dropdown.

| | V4 example | This example |
|---|---|---|
| GPUs | Qwen 0–3, DeepSeek 4–7 | DeepSeek 0–5, Qwen 6–7 |
| DeepSeek L1 | V4-0731 TP4/EP4 | V4.1 DP6/EP6: the measured L1 of [`deepseek-v4.1-flash-6gpu`](../deepseek-v4.1-flash-6gpu/README.md) |
| Qwen replicas | 4 (one answerer each) | 2 (two answerers each) |
| Images | Qwen `image_description` feeds text-only DeepSeek roles | every role gets the image; every route is offered on image requests |
| DeepSeek prompting | inline completion scaffold, passthrough template | chat requests rendered by the official V4.1 encoder |
| DeepSeek pools | thinking pool + non-thinking pool | one pool: `reasoning_effort` → thinking (50/75/100), none → `enable_thinking: false` |
| Budget | `{19, 2}` | `{18, 2}` (one generation unit fewer) |
| Direct DeepSeek cap | 393216 | 262144 (V4.1 model card: max_tokens ≥ 256K) |
| Sandbox executor | deployed, unreferenced | not deployed |

The ensemble accepts one image per request, because that is the Qwen
pool's image policy. A request with more images is rejected by the
ensemble's Qwen roles. DeepSeek alone accepts up to eight.

## DeepSeek L1 and its overlay

The DeepSeek service copies the committed six-GPU command and differs in two
ways:

- **No `--default-chat-template-kwargs`.** The V4.1 encoder ORs `thinking`
  and `enable_thinking`, so a server-wide `thinking: true` would make the
  non-thinking route think. With neither key set, the encoder defaults to
  thinking at high (75).
- **A loopback port (8010).** It serves the launcher's probes, the
  benchmark tokenizer, and the paired direct TTFT row.

`vllm-sm120.Dockerfile` + `patch_sm120.py` build this example's own overlay
on the pinned vLLM nightly. It carries the six SM120 edits of the six-GPU
example and adds **edit 7**.

The V4.1 encoder ignores `continue_final_message`. Kairyu's public-output
floor (DTO-D9/D15) continues a thinking span that used up its cap by sending
`<think>\n…\n</think>\n\n` as the final assistant turn. Edit 7 renders that
turn as an open continuation:

- the span becomes the turn's reasoning;
- the end-of-sentence token is omitted.

Other renderings are byte-identical. `check_sm120_kernels.py` is the
numerical kernel gate; run it inside the built image as the six-GPU README
describes.

## Start

```sh
export KAIRYU_RESPONSES_COMPACTION_SECRET="$(openssl rand -hex 32)"
# Optional: hard-link already downloaded checkpoints instead of re-fetching.
export DEEPSEEK_MODEL_SEED=/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-flash-6gpu/models/deepseek-v4.1-flash
export QWEN_MODEL_SEED=/mnt/nvme/kairyu/model-volumes/qwen3.8-27b-1gpu/models/qwen3.8-27b-fp8
./examples/qwen3.8-deepseek-v4.1-8gpu/run.sh
```

`up` performs the following steps:

- Refuses to evict other workloads: every GPU must be idle, and the host
  needs 256 GiB available for the Engram tables.
- Pins DeepSeek to the NUMA nodes of GPUs 0–5 and each Qwen replica to its
  GPU's node.
- Builds the overlay and refuses any image ID other than the pinned one.
- Attests both checkpoints by tree SHA-256; a seeded copy is re-hashed.
- Waits for the stack.
- Probes the DeepSeek L1 directly:
  - rendered efforts, chat mode, and the edit-7 continuation;
  - exact finite `323` answers from every DP rank in default, low, and chat
    mode;
  - one image.
- Validates the served L2 policy against `example.json`.
- Installs the Chat UI effort dropdown.

`run.sh down|status|logs` manage the stack.

## Verification

```sh
./examples/qwen3.8-deepseek-v4.1-8gpu/verify.sh list
./examples/qwen3.8-deepseek-v4.1-8gpu/verify.sh serving-auto-max --no-start
./examples/qwen3.8-deepseek-v4.1-8gpu/verify.sh serving-auto-max-coding --no-start
./examples/qwen3.8-deepseek-v4.1-8gpu/verify.sh vision --no-start
./examples/qwen3.8-deepseek-v4.1-8gpu/verify.sh tool-calling --no-start
./examples/qwen3.8-deepseek-v4.1-8gpu/browser-smoke.sh
```

- **`serving-auto-max` and `serving-auto-max-coding`** are the V4 example's
  route-aware matrices. The coding matrix's TTFT gate is ≤ 2 × the paired
  direct V4.1 row. No pinned fallback denominator exists yet, so a failed
  paired row fails the gate.
- **`vision`** sends one-image requests through the judge. Every answer must
  name the image colour, and at least one request must take the ensemble.
  The DeepSeek L1's `vllm:mm_cache_queries_total` must grow by at least one
  image per DeepSeek generation stage, which proves those stages received
  the image itself.
- **`tool-calling`** requires one executable bash tool call through L3.

Results and every measured deviation go to [`MEASUREMENTS.md`](MEASUREMENTS.md).
