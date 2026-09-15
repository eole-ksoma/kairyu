# DeepSeek V4.1 Flash on eight RTX PRO 6000 GPUs

One DeepSeek-V4.1-Flash replica uses all eight 96 GB RTX PRO 6000 Blackwell
GPUs. The existing V4 vision example's L2 ReplicaPool and L3 OpenAI-compatible
API / Open WebUI structure are retained. vLLM renders messages, images, and
tools with the V4.1 encoder and parses V4.1 DSML tool calls.

```text
Open WebUI (:3007) -> Kairyu (:8007) -> ReplicaPool -> vLLM (TP8, GPUs 0–7)
```

## Start

```sh
./run.sh up
./run.sh status
./verify.sh tool-calling --no-start
./verify.sh vision --no-start
./verify.sh serving --no-start
./verify.sh reasoning --no-start
./verify.sh cancellation --no-start
./verify.sh long-context --no-start
```

The model is approximately 510 GB. `run.sh` checks the exact GPU inventory,
uses NVMe storage under `/mnt/nvme/kairyu`, pins CPU affinity, builds the
runtime, downloads the fixed model revision, verifies its SHA-256 manifest,
and checks readiness, a tool call, and an image answer. Existing inference
services must release GPUs 0–7 before starting this example.

API: `http://127.0.0.1:8007/v1`, model `deepseek-v4.1-flash`.
Chat UI: `http://127.0.0.1:3007` (same local, auth-disabled UI as V4 Vision).
Override ports with `API_PORT` / `CHAT_UI_PORT`.
`./run.sh down` stops the stack while preserving model, UI, and compilation
cache storage. `VERIFY_MODEL=1 ./run.sh up` rehashes the cached model.

## Thinking and sampling

Omitted effort means **thinking high**, both through the API and in Chat UI.
The pinned DeepSeek encoder defines `low=50`, `high=75`, and `max=100`.
The example aligns the L1 encoder with those definitions. The UI's `default`
selection inherits high; the dropdown exposes default/low/high/max.

```json
{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"What is 17 * 19?"}],"max_tokens":8192}
```

An explicit `reasoning_effort` selects `low`, `high`, or `max`.
The existing Kairyu L3 effort aliases are preserved. Thinking needs room in
the output budget: a length-limited reasoning trace without a final answer
is incomplete. Chat UI retains V4's 32,768-token default; callers can set
their own limit within the context budget.

Sampling defaults are `temperature=1.0`, `top_p=1.0`, within the model card's
recommended range. Explicit request sampling remains supported. The model
card's evaluation output allowance (at least 256K tokens) is distinct from
the interactive UI default.

## L1 selection and evidence

The SM120 FlashInfer overlay starts from the same 0.6.18 source revision as
the GPU-verified V4 Vision example. V4.1 additionally needs 64-token SWA
pages and dual-cache prefill instantiations for its 32-token C2 compressed
pages. `patch_runtime.py` makes these L1 adaptations against exact source
anchors; changed upstream sources fail the build. Manager blocks are 64
tokens with the default BLHNC layout. Adaptive DSpark verification is off
because the pinned indexer backend does not support it. A V4.1-only SM120
indexer subclass selects the same 64-token manager blocks: DeepGEMM
accepts MXFP4 compressed pages of 32 or 64 tokens. The indexer uses
MXFP4; its real Q/K quantizers, paged store, prefill and decode are checked
against independently dequantized PyTorch logits by
`check_sm120_indexer.py`. Enabling it is scoped to V4.1 on SM120.

`check_sm120_pages.py` runs inside the L1 image with one SM120 GPU. It checks
decode and prefill against independent PyTorch attention using the actual
packed cache bytes, C1/C2 page sizes, padded block strides, masks and sinks.
Its tolerances follow the pinned FlashInfer DSV4 correctness tests.

To reproduce the two kernel gates before starting the stack, from this
directory with GPU 0 available:

```sh
for check in check_sm120_pages.py check_sm120_indexer.py; do
  docker run --rm --gpus device=0 --ipc=host \
    -v "$PWD:/checks:ro" --entrypoint python3 \
    local/vllm-openai:deepseek-v41-sm120 "/checks/$check"
done
```

Runtime, model, and configuration pins live in `example.json`,
`model-manifest.json`, and `kairyu.yaml`. See `MEASUREMENTS.md` for the tested
configuration, comparisons, and limitations. The measured selection uses
TP8/EP8, Marlin, FP8 KV, MXFP4 indexer, prefix caching, DSpark 5, a 16K
batch limit, 64 sequences, GPU-resident Engram and breakable CUDA graphs.
The default all-reduce backend is NCCL.

The local image ID is an evidence pin. A fresh source build can produce a
different ID; startup then stops until both image pins are updated and the
GPU gates are rerun for that build.

For bounded L1 comparisons after startup:

```sh
../../.venv/bin/python tune.py baseline no-spec no-ep
```

The tuner records each actual command and restores the baseline in `finally`.
The retained comparisons cover EP, DSpark, PCIe IPC and an 8K batch limit.
Additional named batch/memory/graph candidates are optional experiments;
final measurements use `verify.sh` against the committed configuration.

`serving` uses unique prompt prefixes, approximately 8K input tokens,
exactly 256 generated tokens, and concurrency 1/8/16/32/64. It verifies all
requests were placed on the single replica. These fixed-length rows include
reasoning tokens and do not measure completed-answer quality.
`tool-calling` and `vision` separately require completed, usable outputs.
`reasoning` checks default and explicit efforts and records time to final
content. `cancellation` checks slot release. `long-context` retains retrieval
smokes at 32K/128K/256K and near the 1M context boundary; these are retrieval
smokes, not a comprehensive long-context quality evaluation.

Results are written under the example's NVMe `verification-results` directory;
each run records the served-config hash. Sampling outcomes are evidence for
the retained cases, not a universal model-quality guarantee.

## Primary references

- [DeepSeek model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- [Pinned official encoder specification](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/encoding/README.md)
- [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)

The official recipe's GB200 results are not measurements of RTX PRO 6000.
