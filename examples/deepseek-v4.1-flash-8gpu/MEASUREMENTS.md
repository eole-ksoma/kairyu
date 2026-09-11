# Measurement status

Implementation and GPU selection are in progress. No V4.1 performance or
correctness gate is claimed passed until the exact run evidence is recorded.

Hardware target: 8 × NVIDIA RTX PRO 6000 Blackwell Server Edition,
97,887 MiB per GPU, PCIe, one TP8 replica.

Model revision: `dba1be0a40aa45a94ad051997016db3960a90277`.
Model tree SHA-256: `d21211ca29ad7eba1fda84e49b1a34a73214ec9b84b23928cde63902c3318bfd`.

## Preflight evidence

- Related portable CPU contracts: 61 passed. Repository lint and progress-size
  checks passed.
- Built runtime: `vllm 0.1.dev20904+g179dd0fa9`, `flashinfer-python 0.6.18`,
  matching `flashinfer-cubin 0.6.18`, `torch 2.13.0+cu130`.
- Runtime image ID:
  `sha256:027bf47b2bd6f0d0abe54b296e7e9e3d31ee103bb6e46fa0a9807117681c2359`.
- The actual wrapped checkpoint tokenizer rendered default/high with budget
  75, low with 50, max with 100, and explicit off without a reasoning
  prefix. The FlashInfer capability query accepts `(8 heads, 128 top-k)`
  and `(8 heads, 192 top-k)`. These are preflight checks; full-model GPU
  correctness and performance are still pending.
- `uv pip check --system` reports the same single mismatch in both the
  official base and the overlay: torch declares NCCL 2.29.7, while the
  official image supplies NCCL 2.30.7. The overlay preserves that base
  package; TP8 execution still needs the live gates below.

## SM120 compatibility selection

The official image loads target and DSpark weights (62.07 GiB per GPU),
but the initial recipe settings fail before serving on this host:

- Adaptive verification is unsupported by the indexer backend; disable it.
- BLHNC cannot split interleaved manager blocks into smaller kernel blocks.
  LBNHC is rejected by the packed indexer backend.
- V4.1 hardcodes SWA pages to 32; FlashInfer SM120 requires 64.
- V4.1 compression is C1/C2. With 128-token manager blocks, C1 indexer pages
  exceed DeepGEMM's 32/64-page envelope. Use 64-token manager blocks for
  both the V4.1 SM120 MLA and indexer backends: C1=64 and C2=32.
- SM120 FP8 indexer decode supports only 64-token pages. Select MXFP4
  indexer Q/K for C1=64/C2=32. The official binary includes both kernels;
  enable the vLLM dtype gate only for this V4.1/SM120 combination.
- Instantiate FlashInfer's existing generic dual-cache prefill templates
  for 32-token secondary pages; retain its existing arithmetic and address
  calculations. The SM120 attention subclass uses SWA=64.

`patch_runtime.py` checks the exact pinned source anchors. GPU numerical
parity passes all 16 C1/C2, decode/prefill, 128/192 top-k and mask combinations
using independent PyTorch attention over actual packed cache bytes, padded
block strides and attention sinks. The absolute/relative tolerances
0.05/0.05 match the pinned FlashInfer DSV4 tests. Maximum observed absolute
error is 0.0642 (within combined tolerance); the 32/64-page outputs have the
same errors as the earlier 64/128-page probe using the same random inputs.
Evidence: `sm120-page32-parity.log`; reproducible with `check_sm120_pages.py`.

Rejected startups are preserved in the NVMe example directory:
`initial-worker.log`, `no-adaptive-worker.log`, `block128-worker.log`,
`lbnhc-worker.log`, and `sm120-pages-live.log`. Full-model gates remain pending.

MXFP4 indexer GPU evidence: all four C1/C2 prefill/decode cases passed with
real V4.1 Q quantization and K norm/RoPE/quantization/store, padded block
strides, and independently unpacked PyTorch logits. Decode error was zero;
maximum prefill absolute error was 2.3842e-7. `check_sm120_indexer.py` uses
the same `clean_logits=False` as serving and compares only populated logits.
Retained evidence: `sm120-indexer-parity.log`. The FP8 full-model rejection
is retained in `block64-worker.log`. Full-model quality with the selected
MXFP4 indexer still needs verification; the kernel test compares the actual
quantized values, not the model's BF16 indexer quality.
