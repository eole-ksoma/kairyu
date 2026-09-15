# DeepSeek-V4-Flash-0731 as 2 x TP4+EP4 replicas on 8 x RTX PRO 6000 Blackwell

This example starts one public DeepSeek model backed by two identical
four-GPU vLLM replicas with one command:

```text
Open WebUI -> Kairyu L3 (:8006; model deepseek-v4-flash-0731)
                -> Kairyu L2 replica pool (placement only, no orchestration)
                    -> 2 x vLLM L1, DeepSeek-V4-Flash-0731 TP4+EP4
                       (replica 0 on GPU 0-3, replica 1 on GPU 4-7)
```

Clients see exactly one OpenAI-compatible model, `deepseek-v4-flash-0731`.
Kairyu L2 does nothing but choose a replica for each request, so the two
halves of the node behave like one server with twice the sequence capacity of
one TP4 replica — the topology the
[tiered example](../qwen3.8-deepseek-v4-8gpu/MEASUREMENTS.md) measured and
selected for DeepSeek ("Tier2 speculation, batch-budget, and CUDA Graph
selection"): mixed FP4/FP8 weights, FP8 KV cache, native 1,048,576-token
context, 256-token blocks, prefix caching, full/piecewise CUDA Graphs,
five-token DSpark speculation, a 16K batch-token budget, and 32 sequences per
replica. The SM100-only MegaMoE and FP4 indexer-cache paths stay disabled on
SM120, as in the [TP8 example](../deepseek-v4-flash-0731-8gpu/README.md).

vLLM renders every chat request with the checkpoint's own prompt encoder
(`--tokenizer-mode deepseek_v4`, the same `encoding_dsv4.py` that ships inside
the checkpoint), so **OpenAI function-tool calling works end to end**: declared
`tools` are rendered into the official DSML prompt form, the model's tool calls
are parsed by vLLM's `deepseek_v4` tool parser, Kairyu normalizes them into
`choices[0].message.tool_calls`, and `role: "tool"` results round-trip on the
next turn. This is the same layering as the Qwen examples and is what
tool-driven agents (Codex CLI, SWE-bench Pro's mini-swe-agent) require.
Ordinary requests default to direct chat mode; an explicit `reasoning_effort`
selects thinking mode (vLLM maps it onto the encoder's thinking switch). Open
WebUI defaults output to 32,768 tokens.

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

## Start

```sh
./run.sh
```

The command verifies exactly eight matching GPUs (indices 0-7), pins each
replica to the union of its four GPUs' NUMA-local CPUs, reuses or builds the
pinned SM120 vLLM source revision if its image is absent, reuses the
NVMe-backed checkpoint volume shared with the TP8 and tiered examples (or
downloads and hashes the exact model revision), builds Kairyu, waits for
readiness, checks that Kairyu reports exactly one public model with two
healthy replicas, proves one live bash tool call round-trip (the environment
is not "ready" if `tool_calls` comes back null), and prints:

```text
OpenAI API: http://127.0.0.1:8006/v1
Chat UI:    http://127.0.0.1:3000
```

The first build/download is large; DeepSeek's first engine initialization is
also long (compilation and mHC warm-up), and the per-replica compilation
caches below `/mnt/nvme/kairyu/model-volumes/deepseek-v4-flash-0731-dp2-8gpu/`
make later starts much faster. Set `VERIFY_MODEL=1` to rehash every checkpoint
file. Lifecycle commands are `./run.sh up`, `./run.sh status`, `./run.sh logs`,
and `./run.sh down`.

## Serving verification

```sh
./verify.sh list
./verify.sh serving
./verify.sh tool-calling
```

`tool-calling` gates the OpenAI agent contract the serving claim depends on:
an auto-choice `bash` tool call (the SWE-bench Pro mini-swe-agent request
shape) fanned across both replicas, the follow-up turn with the `role: "tool"`
result, the streamed variant, thinking mode (`reasoning_effort`) with tools,
and the non-thinking default. Artifacts land in the same results directory as
`serving`.

`serving` warms both replicas with one short request each, then records TTFT,
TPOT, requests/s, and output tokens/s for fixed approximately 8K-token inputs
and exactly 256 generated tokens at concurrency 1, 8, 16, 32, and 64 (64
requests per row). Each row's prompts carry a row-unique prefix first, so
neither vLLM prefix caching nor Kairyu's prefix-aware placement can inflate the
matrix. For every row it also reads the placement log delta and writes
`placement.json` with the per-replica request counts; at concurrency >= 8 the
row **fails** unless both replicas received traffic and neither took more than
1.25x the even share (40 of 64 requests). Artifacts go to
`/mnt/nvme/kairyu/model-volumes/deepseek-v4-flash-0731-dp2-8gpu/verification-results/<UTC-run-id>/`
(override with `VERIFICATION_RESULTS_ROOT`). The locked results are in
[MEASUREMENTS.md](MEASUREMENTS.md). Model and product evaluations are invoked
explicitly through `python -m evals`.

## Reproducibility pins

- Model revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Model tree SHA-256: `90bd164d6f778d798eeaecd3517d83b87d49d300756a9217ada14a2b15203754`
- vLLM source: `jasl/vllm@aa0d51302747ea80f282e26949708b3253409fe2`
- vLLM image digest:
  `sha256:99756b54424a4697f69476b29aa02fb7f8112aaa74fa8203a7bf8a0bae4ca6f1`
- Open WebUI: `v0.11.0-slim` plus the digest in `example.json`

Override ports with `API_PORT` and `CHAT_UI_PORT`. Override images with
`DEEPSEEK_VLLM_IMAGE` or `OPEN_WEBUI_IMAGE`; a non-default vLLM override must
already exist locally. Dotenv files are intentionally ignored and credentials
are not written into evidence.

For a deliberately public UI, bind only the UI and keep the unauthenticated
Kairyu API on loopback:

```sh
CHAT_UI_BIND_ADDRESS=0.0.0.0 PUBLIC_HOST=<public-ip> \
WEBUI_URL=http://<public-ip>:3000 \
ENABLE_SIGNUP=false WEBUI_ADMIN_EMAIL=<email> \
WEBUI_ADMIN_PASSWORD=<strong-secret> ./run.sh up
```

Public production use should put TLS and a firewall/reverse proxy in front of
port 3000.
