# DeepSeek V4.1 Flash on six RTX PRO 6000 GPUs

One DeepSeek-V4.1-Flash replica on GPUs 0–5 of 96 GB RTX PRO 6000
Blackwell (SM120, PCIe) cards, behind the same L2/L3 structure as
`deepseek-v4.1-flash-8gpu`: one Kairyu `ReplicaPool` replica, the
OpenAI-compatible API, and Open WebUI. GPUs 6–7 are not used.

```text
Open WebUI (:3008) -> Kairyu (:8008) -> ReplicaPool -> vLLM (DP6 / EP6 + DSpark, GPUs 0-5)
```

## Start

```sh
./run.sh up          # preflight, image build, model download + hash check, readiness
./run.sh status
./verify.sh l1 --no-start
./verify.sh serving --no-start
./verify.sh completed --no-start
./verify.sh tool-calling --no-start
./verify.sh vision --no-start
./verify.sh reasoning --no-start
./verify.sh cancellation --no-start
./verify.sh long-context --no-start
./verify.sh restart --no-start
./run.sh down
```

`run.sh up` refuses to start when GPUs 0–5 are busy or when the host has
less than 256 GiB of available memory (the Engram tables are pinned in host
memory). The checkpoint (≈476 GiB) is downloaded to
`/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-flash-6gpu/models` and hashed
against the pinned tree. To reuse a copy that is already on the same NVMe
filesystem, set `DEEPSEEK_MODEL_SEED=<path to deepseek-v4.1-flash>`: it is
hard-linked (no extra space) and then re-hashed; nothing is trusted from it.

API: `http://127.0.0.1:8008/v1`, model `deepseek-v4.1-flash`.
Chat UI: `http://127.0.0.1:3008` (local, no authentication).

## Why this configuration

The settings start from the official sources (model card, encoder
specification, vLLM recipe) and change only what this hardware forces or
what a bounded measurement in `MEASUREMENTS.md` supports (one parameter at a
time; a change is adopted only for ≥ 5 % on c1 or c32 throughput without a
> 5 % loss elsewhere).

| Setting | Value | Source / reason |
|---|---|---|
| Runtime | `vllm/vllm-openai:nightly` pinned by digest + `patch_sm120.py` | The recipe's image. It already renders the model author's efforts; the SM120 edits below are still missing upstream. |
| Replica shape | DP6 / EP6 (TP1 per rank) | Official Blackwell "DEP" shape. 384 routed experts / 6 = 64 per rank. Measured against the official TP2 degree (TP2 × DP3 on NUMA-local pairs): c32 +44–47 %, c1 TTFT −37 %. TP6 cannot divide 64 heads / 8 output groups. The official DEP kernels (FlashMLA mega attention, DeepGEMM mega MoE) are SM100-only, so DEP runs on the TP path's SM120 kernels. |
| Engram | `cpu_offload: true` | 6 × 96 GB is below the checkpoint's official 614 GB minimum; the recipe's memory-bound 8 × H100 arm moves the 183 GiB tables to pinned host memory ("output is unchanged"). |
| Batched tokens / memory | 4096 / 0.92 | The same memory-bound arm's values. Needed to fit DSpark on DP6 (at 16K / 0.90 its KV pool is negative). |
| DSpark | 5-token block, probabilistic drafts, block rejection, full verification | Official method and block. Adaptive verification (the NVIDIA default) is rejected by the V4.1 indexer backend. Measured: c1 +77 % (TPOT 14.7 → 6.9 ms), c32 +20 %. |
| Attention | `FLASHINFER_MLA_SPARSE_DSV41`, FP8 KV, MXFP4 indexer | Official Blackwell settings. `indexer_sparse_logits` fails at start on SM120 (SM100-only DeepGEMM kernel). |
| Pages | 64-token blocks | SM120 requirement (FlashInfer SWA pages, DeepGEMM C1/C2 indexer pages). |
| MoE | Marlin | SM120 MXFP4 MoE path. |
| Sequences | 128 per DP engine | Official Blackwell value; 64 measured no different. |
| Vision encoder | default (not `--mm-encoder-tp-mode data`) | Official option measured no different on TP1 ranks. |
| Effort | default thinking high (75); low 50, max 100 | The model author's encoder (`encoding/README.md`), which the runtime renders; the recipe's 25/50/75/100 table is the older vLLM encoder's. |
| Sampling | temperature 1.0, top_p 0.95 | Model card; the setting DeepSeek's published results use. Applied at L1 with `--override-generation-config`; callers can still send their own. |

Each of the six DP engines holds about 8.3M KV tokens, so a full
1,048,576-token request fits on any engine.

## SM120 runtime overlay

`vllm-sm120.Dockerfile` builds on the pinned vLLM image and runs
`patch_sm120.py`, which edits exactly one known source anchor per change and
fails the build if any anchor is missing or repeated:

- 64-token SWA / MLA / indexer pages and C2 (32-token) dual-cache prefill;
- MXFP4 indexer enabled on SM120 for V4.1 only;
- masked sparse-KV rows read a zero row instead of a recycled slot (without
  it every EP6 shape returned NaN log-probabilities or unrelated text);
- the split top-p cutoff keeps a candidate when a forced logit rounds the
  cutoff to the maximum;
- a build-time check that the encoder renders the model author's efforts.

Prebuilt FlashInfer JIT caches are removed so the patched sources are the
ones compiled. `check_sm120_kernels.py` is the numerical gate for these
kernels (including a NaN-poisoned masked slot):

```sh
docker run --rm --gpus device=0 -v "$PWD:/checks:ro" --entrypoint python3 \
  local/vllm-openai:deepseek-v41-6gpu-nightly /checks/check_sm120_kernels.py
```

## Verification gates

- `l1`: rendered efforts (default/low/high/max/chat), rounds of 12
  concurrent `17 * 19` probes per variant with exact answers and finite
  log-probabilities until all six DP engines have answered, the L1 default
  sampling, tool call, image, and memory.
- `serving`: fixed 8K-in / 256-out rows at c1/8/16/32/64 (64 requests each),
  every request placed on the replica, GPU peaks.
- `completed`: natural-completion generic and coding rows at c1/8/32 with
  the default effort; every request must stop with visible content.
- `tool-calling`, `vision`, `reasoning`, `cancellation`, `long-context`
  (32K/128K/256K/≈1M retrieval), `restart`.

L1 candidates are compared with `tune.py` (one parameter at a time, the
committed configuration restored afterwards). Results and every official
deviation are in `MEASUREMENTS.md`.

## Primary references

- [DeepSeek-V4.1-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- [Encoder specification](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/encoding/README.md)
- [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
  (`vllm-project/recipes` `models/deepseek-ai/DeepSeek-V4.1-Flash.yaml` at `1343d0f1`)
