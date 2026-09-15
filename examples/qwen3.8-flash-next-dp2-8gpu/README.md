# Qwen3.8-Flash-Next as 2 x TP4 replicas on 8 x RTX PRO 6000 Blackwell

This example starts one public Qwen vision-language model backed by two
identical four-GPU vLLM replicas with one command:

```text
Open WebUI -> Kairyu L3 (:8007; model qwen3.8-flash-next)
                -> Kairyu L2 replica pool (placement only, no orchestration)
                    -> 2 x vLLM L1, Qwen3.8-Flash-Next-FP8 TP4
                       (replica 0 on GPU 0-3, replica 1 on GPU 4-7)
```

Clients see exactly one OpenAI-compatible model, `qwen3.8-flash-next`, that
accepts text and OpenAI image parts (`image_url`, base64 data URLs or
`http(s)` URLs). Kairyu L2 does nothing but choose a replica for each
request, so the two halves of the node behave like one server with twice the
sequence capacity of one TP4 replica — the same layout as the
[DeepSeek replica examples](../deepseek-v4-flash-0731-dp2-8gpu/README.md).

The L1 command is the
[official vLLM recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.8-Flash-Next.html)'s
verified `rtx_pro_6000_4x` layout for the FP8 checkpoint (TP4, 16
sequences, 8K batch tokens, prefix caching,
FlashInfer autotune off, `qwen3_xml` tool parser and `qwen3` reasoning
parser) at the checkpoint's native 262,144-token context, **without the
recipe's MTP k=3 speculative decoding**: on the pinned vLLM revision, prefix
caching and MTP together silently corrupt about 5% of answers whenever
requests are batched (upstream
[vllm-project/vllm#53912](https://github.com/vllm-project/vllm/issues/53912);
reproduced here as `ductduct…`/`Register Register…` output in 13 of 274
greedy requests at 2-12 concurrent, 0 of 1,508 with either feature off).
Prefix caching is the one kept because Kairyu's prefix-aware placement and
multi-turn/agent traffic depend on it (8K-context follow-up TTFT 194 ms vs
1,340 ms with caching off); the cost is single-stream decode at 104 instead
of 175 tokens/s. Re-enable MTP only after upstream fixes the interaction and
the `verify.sh vision` batch check stays clean. `--kv-cache-memory`
(43.16 GiB, the value vLLM derives for the recipe's 0.95 utilization on a
warm start) replaces the utilization-based sizing: vLLM sizes the cache from
a start-up memory profile, and on a cold torch.compile cache that profile
includes about 35 GiB of compile scratch, so a first boot otherwise gets
741K KV tokens per replica instead of 3.45M. TP8 is
incompatible with the 128-wide FP8 weight blocks and pipeline parallel is
unsupported, so two TP4 replicas are the only eight-GPU split. The
checkpoint's preprocessor accepts 65,536 - 16,777,216 px per image with no
per-request cap; Kairyu's `image_input_policy` bounds admission to 4 images
of up to 16 Mpx / 8 MiB each, and video input is closed (`--limit-mm-per-prompt.video 0`)
because the policy validates images only.

vLLM renders every chat request with the example-local template
([`qwen3.8-flash-next-chat.jinja`](qwen3.8-flash-next-chat.jinja): the
official template plus a two-line effort alias, see below), so **OpenAI
function-tool calling works end to end**: declared `tools` are rendered into
Qwen's XML tool prompt, the model's tool calls are parsed by vLLM's
`qwen3_xml` parser, Kairyu normalizes them into
`choices[0].message.tool_calls`, and `role: "tool"` results round-trip on the
next turn. Ordinary requests default to direct (non-thinking) chat; an
explicit `reasoning_effort` selects thinking mode.

## Reasoning effort in the Chat UI

The Chat UI runs without login (single auto-admin session, as in the
[tiered example](../qwen3.8-deepseek-v4-8gpu/README.md)). `run.sh up`
installs a global Open WebUI filter
([`webui-reasoning-effort-filter.py`](webui-reasoning-effort-filter.py)) whose
user valve renders as a dropdown under **Chat Controls -> Valves -> Reasoning
Effort** with Qwen's official levels `default / low / medium / xhigh`:

- `default` leaves `reasoning_effort` out, so the non-thinking direct-chat
  default applies with Qwen's published instruct sampling
  (temperature 0.7, top_p 0.8, presence_penalty 1.5 from the Chat UI's
  `DEFAULT_MODEL_PARAMS`).
- `low / medium / xhigh` are sent as the OpenAI `reasoning_effort` body
  field, and the filter drops the instruct sampling fields from the request
  so vLLM applies the checkpoint's thinking `generation_config`
  (temperature 1.0, top_p 0.95, top_k 20) — the sampling Qwen publishes for
  thinking mode.

Kairyu L3 accepts the OpenAI wire vocabulary and normalizes `medium -> high`
and `xhigh -> max` before forwarding, while the official template rejects
anything outside `low / medium / xhigh` with HTTP 400. The example-local
template therefore maps `high -> medium` and `max -> xhigh` at the top and is
otherwise byte-identical to the checkpoint's template, so the model receives
exactly the effort the client selected. Provisioning is fail-closed:
`run.sh up` verifies the installed dropdown exposes exactly those levels.

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
`vllm/vllm-openai:qwen38-flash-next` tag predates the merged support PR and
has no SM120 validation. This example therefore runs upstream vLLM `main`:
[`vllm-sm120.Dockerfile`](vllm-sm120.Dockerfile) starts from upstream's own
digest-pinned nightly build of commit `27a94d1c` (the CI source build of
exactly that commit; it carries vLLM #53896, #54566, #43477 and #53574) and
overlays FlashInfer `main` at `60b49158` (the SM120 sparse-MLA fix the
sibling DeepSeek vision example needs; the 0.6.18 AOT module cache is
removed so stale prebuilt kernels cannot shadow it — kernels JIT-compile
into the per-replica compile cache on first use). The resulting image,
`local/vllm-openai:sm120-27a94d1-flashinfer-60b4915`, is shared with the
[DeepSeek-V4-Flash-Vision-Exp replica example](../deepseek-v4-flash-vision-exp-dp2-8gpu/README.md);
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
`/mnt/nvme/kairyu/model-volumes/qwen3.8-flash-next-dp2-8gpu/models`, builds
Kairyu with the `vision` extra, waits for readiness, checks that Kairyu
reports exactly one public model with two healthy replicas, proves one live
bash tool call round-trip and one live image request (the environment is
not "ready" if `tool_calls` comes back null or an image request returns no
content), installs the Reasoning Effort dropdown, and prints:

```text
OpenAI API: http://127.0.0.1:8007/v1
Chat UI:    http://127.0.0.1:3007 (no authentication)
```

The first build/download is large (about 173 GiB of weights); the first
engine initialization is also long (kernel JIT compilation, CUDA-graph
capture), and the per-replica compilation caches below
`/mnt/nvme/kairyu/model-volumes/qwen3.8-flash-next-dp2-8gpu/` make later
starts much faster. Set `VERIFY_MODEL=1` to rehash every checkpoint file.
Lifecycle commands are `./run.sh up`, `./run.sh status`, `./run.sh logs`, and
`./run.sh down`.

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
and exactly 256 generated tokens at concurrency 1, 8, 16, and 32 (64 requests
per row; 32 is the admission limit of 2 x 16 sequences). Each row's prompts
carry a row-unique prefix first, so neither vLLM prefix caching nor Kairyu's
prefix-aware placement can inflate the matrix. For every row it also reads
the placement log delta and writes `placement.json` with the per-replica
request counts; at concurrency >= 8 the row **fails** unless both replicas
received traffic and neither took more than 1.25x the even share (40 of 64
requests). Artifacts go to
`/mnt/nvme/kairyu/model-volumes/qwen3.8-flash-next-dp2-8gpu/verification-results/<UTC-run-id>/`
(override with `VERIFICATION_RESULTS_ROOT`). The locked results are in
[MEASUREMENTS.md](MEASUREMENTS.md). Model and product evaluations are invoked
explicitly through `python -m evals`.

## Reproducibility pins

- Model revision: `236dfdf285828023ca3bcd3f37366c58a3469b13`
- Model tree SHA-256: pinned in `example.json` after the first attested
  download (`run.sh up` prints the tree hash it computed)
- vLLM: upstream `vllm-project/vllm@27a94d1ce4e3fc100c4732439ccec10f8246a804`
  (nightly image digest in `example.json`) plus FlashInfer
  `flashinfer-ai/flashinfer@60b49158ab4fb81718aef486c2d3c89aec4c1901`
- vLLM image ID: `example.json` / `kairyu.yaml` (`container_image_digest`)
- Chat template: `qwen3.8-flash-next-chat.jinja` (part of the served-config
  SHA-256 recorded by `verify.sh`)
- Open WebUI: `v0.11.0-slim` plus the digest in `example.json`

Override ports with `API_PORT` and `CHAT_UI_PORT`. Override images with
`QWEN_VLLM_IMAGE` or `OPEN_WEBUI_IMAGE`; a non-default vLLM override must
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
WEBUI_URL=http://<public-ip>:3007 ./run.sh up
```
