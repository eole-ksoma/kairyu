# DeepSeek-V4-Flash-Vision-Exp as 2 x TP4+EP4 replicas on 8 x RTX PRO 6000 Blackwell

This example starts one public DeepSeek vision-language model backed by two
identical four-GPU vLLM replicas with one command:

```text
Open WebUI -> Kairyu L3 (:8005; model deepseek-v4-flash-vision-exp)
                -> Kairyu L2 replica pool (placement only, no orchestration)
                    -> 2 x vLLM L1, DeepSeek-V4-Flash-Vision-Exp TP4+EP4
                       (replica 0 on GPU 0-3, replica 1 on GPU 4-7)
```

Clients see exactly one OpenAI-compatible model, `deepseek-v4-flash-vision-exp`,
that accepts text and OpenAI image parts (`image_url`, base64 data URLs or
`http(s)` URLs). Kairyu L2 does nothing but choose a replica for each request,
so the two halves of the node behave like one server with twice the sequence
capacity of one TP4 replica — the same layout as the
[text-only DeepSeek replica example](../deepseek-v4-flash-0731-dp2-8gpu/README.md).

The L1 command follows the
[official vLLM recipe](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V4-Flash-Vision-Exp.html)
(TP4 + expert parallel, FP8 KV cache, 256-token blocks, DSpark k=3 with
probabilistic drafting, `deepseek_v4` tool/reasoning parsers) at the
checkpoint's native 1,048,576-token context, with two SM120 adjustments:
the Marlin MXFP4 MoE kernel is pinned (vLLM's higher-priority MXFP4 MoE
backends target SM100 datacenter Blackwell) and DSpark adaptive verification
is off (rejected by the SM120 indexer). The 16K batch-token budget and 32
sequences per replica are carried over from the sibling example's
measurements on the same GPUs. Image preprocessing is fixed by the checkpoint
(at most 387 prompt tokens per image, no per-request image cap); Kairyu's
`image_input_policy` bounds admission to 8 images of up to 16 Mpx / 8 MiB
each.

vLLM renders every chat request with the checkpoint's own prompt encoder
(`--tokenizer-mode deepseek_v4`), so **OpenAI function-tool calling works end
to end**: declared `tools` are rendered into the DSML prompt form, the model's
tool calls are parsed by vLLM's `deepseek_v4` tool parser, Kairyu normalizes
them into `choices[0].message.tool_calls`, and `role: "tool"` results
round-trip on the next turn. Ordinary requests default to direct chat mode; an
explicit `reasoning_effort` selects thinking mode. The checkpoint publishes no
non-thinking sampling values, so Open WebUI pins only the 32,768-token output
budget and vLLM applies the checkpoint's `generation_config` for the rest.

## Reasoning effort in the Chat UI

The Chat UI runs without login (single auto-admin session, as in the
[tiered example](../qwen3.8-deepseek-v4-8gpu/README.md)). `run.sh up`
installs a global Open WebUI filter
([`webui-reasoning-effort-filter.py`](webui-reasoning-effort-filter.py)) whose
user valve renders as a dropdown under **Chat Controls -> Valves -> Reasoning
Effort** with the levels `default / low / high / max` — the encoder's own
vocabulary (`encoding_dsv4.py`); `low`, `high`, and `max` are forwarded verbatim
as the OpenAI `reasoning_effort` body field (Kairyu forwards it unchanged and
vLLM turns it into the encoder's thinking switch), and `default` leaves the
field out so the non-thinking direct-chat default applies. Provisioning is
fail-closed: `run.sh up` verifies the installed dropdown exposes exactly those
levels.

## How requests are spread over the replicas

The pool in [`kairyu.yaml`](kairyu.yaml) uses Kairyu's built-in placement
policy with two knobs set for even, efficient spreading:

- `prefix_index: true` — a request whose prompt prefix is already warm in a
  replica's KV cache (a multi-turn chat continuing on the same replica) goes
  back to that replica **only while it is idle**, so a long cached context is
  reused instead of being prefilled again on the other replica.
- `queue_depth_threshold: 0` — the moment the preferred replica has one
  in-flight request, placement falls through to **least-outstanding**: the
  replica with fewer in-flight requests wins, ties going to replica 0.
  Concurrent traffic therefore alternates between the two replicas before
  either takes a second request; strictly serial traffic uses replica 0.

A replica is eligible only after its readiness probe succeeds and is ejected
after one failed request (`unhealthy_after: 1`) until it probes healthy
again. Every placement decision is appended to the pool's placement log
(`placement_log_path`), which `verify.sh` reads to prove the distribution.

## vLLM image (SM120)

No vLLM release serves this checkpoint, and the official
`vllm/vllm-openai:deepseekv4-flash-vision` tag predates the merged support PR
and has no SM120 validation. This example therefore runs upstream vLLM `main`:
[`vllm-sm120.Dockerfile`](vllm-sm120.Dockerfile) starts from upstream's own
digest-pinned nightly build of commit `27a94d1c` (the CI source build of
exactly that commit; it carries vLLM #54566, #53896, #43477 and #53574) and
overlays FlashInfer `main` at `60b49158` (FlashInfer #4802, the SM120
sparse-MLA prefill path that the first image request needs; the 0.6.18 AOT
module cache is removed so stale prebuilt kernels cannot shadow it — kernels
JIT-compile into the per-replica compile cache on first use). The resulting
image, `local/vllm-openai:sm120-27a94d1-flashinfer-60b4915`, is shared with
the [Qwen3.8-Flash-Next replica example](../qwen3.8-flash-next-dp2-8gpu/README.md);
whichever example runs first builds it.

## Start

```sh
./run.sh
```

The command verifies exactly eight matching GPUs (indices 0-7), pins each
replica to the union of its four GPUs' NUMA-local CPUs, reuses or builds the
pinned SM120 image (and refuses to serve if its image ID differs from the
`container_image_digest` pinned in `kairyu.yaml`), downloads and hashes the
exact model revision into
`/mnt/nvme/kairyu/model-volumes/deepseek-v4-flash-vision-exp-dp2-8gpu/models`,
builds Kairyu with the `vision` extra, waits for readiness, checks that
Kairyu reports exactly one public model with two healthy replicas, proves one
live bash tool call round-trip and one live image request (the environment is
not "ready" if `tool_calls` comes back null or an image request returns no
content), installs the Reasoning Effort dropdown, and prints:

```text
OpenAI API: http://127.0.0.1:8005/v1
Chat UI:    http://127.0.0.1:3005 (no authentication)
```

The first build/download is large (about 168 GB of weights); DeepSeek's first
engine initialization is also long (FlashInfer JIT compilation, CUDA-graph
capture, mHC warm-up), and the per-replica compilation caches below
`/mnt/nvme/kairyu/model-volumes/deepseek-v4-flash-vision-exp-dp2-8gpu/` make
later starts much faster. Set `VERIFY_MODEL=1` to rehash every checkpoint
file. Lifecycle commands are `./run.sh up`, `./run.sh status`,
`./run.sh logs`, and `./run.sh down`.

## Serving verification

```sh
./verify.sh list
./verify.sh serving
./verify.sh tool-calling
./verify.sh vision
```

`tool-calling` gates the OpenAI agent contract the serving claim depends on:
an auto-choice `bash` tool call (the SWE-bench Pro mini-swe-agent request
shape) fanned across both replicas, the follow-up turn with the `role: "tool"`
result, the streamed variant, thinking mode (`reasoning_effort`) with tools,
and the non-thinking default. `vision` fans image requests (two per replica,
concurrently) through Kairyu and requires visible content from every one of
them and, from the placement log, that both replicas served images. Artifacts
land in the same results directory as `serving`.

`serving` warms both replicas with one short request each, then records TTFT,
TPOT, requests/s, and output tokens/s for fixed approximately 8K-token inputs
and exactly 256 generated tokens at concurrency 1, 8, 16, 32, and 64 (64
requests per row). Each row's prompts carry a row-unique prefix first, so
neither vLLM prefix caching nor Kairyu's prefix-aware placement can inflate the
matrix. For every row it also reads the placement log delta and writes
`placement.json` with the per-replica request counts; at concurrency >= 8 the
row **fails** unless both replicas received traffic and neither took more than
1.25x the even share (40 of 64 requests). Artifacts go to
`/mnt/nvme/kairyu/model-volumes/deepseek-v4-flash-vision-exp-dp2-8gpu/verification-results/<UTC-run-id>/`
(override with `VERIFICATION_RESULTS_ROOT`). The locked results are in
[MEASUREMENTS.md](MEASUREMENTS.md). Model and product evaluations are invoked
explicitly through `python -m evals`.

## Reproducibility pins

- Model revision: `6821d6ad3681a4b137b066b76094fa82ebd0a380`
- Model tree SHA-256: pinned in `example.json` after the first attested
  download (`run.sh up` prints the tree hash it computed)
- vLLM: upstream `vllm-project/vllm@27a94d1ce4e3fc100c4732439ccec10f8246a804`
  (nightly image digest in `example.json`) plus FlashInfer
  `flashinfer-ai/flashinfer@60b49158ab4fb81718aef486c2d3c89aec4c1901`
- vLLM image ID: `example.json` / `kairyu.yaml` (`container_image_digest`)
- Open WebUI: `v0.11.0-slim` plus the digest in `example.json`

Override ports with `API_PORT` and `CHAT_UI_PORT`. Override images with
`DEEPSEEK_VLLM_IMAGE` or `OPEN_WEBUI_IMAGE`; a non-default vLLM override must
already exist locally and carry the image ID pinned in `example.json` /
`kairyu.yaml` (an alias tag of the same build), because the pool reports that
ID as `container_image_digest` evidence — `run.sh up` refuses any other
image. Dotenv files are intentionally ignored and credentials are not
written into evidence.

The Chat UI has no login, so bind it to loopback (the default) or to a
network you trust; a deliberately public UI needs TLS and a firewall/reverse
proxy in front of it:

```sh
CHAT_UI_BIND_ADDRESS=0.0.0.0 PUBLIC_HOST=<public-ip> \
WEBUI_URL=http://<public-ip>:3005 ./run.sh up
```
