# Measurement status

Implementation and GPU selection are in progress. No V4.1 performance or
correctness gate is claimed passed until the exact run evidence is recorded.

Hardware target: 8 × NVIDIA RTX PRO 6000 Blackwell Server Edition,
97,887 MiB per GPU, PCIe, one TP8 replica.

Model revision: `dba1be0a40aa45a94ad051997016db3960a90277`.
Model tree SHA-256: `d21211ca29ad7eba1fda84e49b1a34a73214ec9b84b23928cde63902c3318bfd`.

## Preflight evidence

- Related portable CPU contracts: 56 passed. Repository lint and progress-size
  checks passed.
- Built runtime: `vllm 0.1.dev20904+g179dd0fa9`, `flashinfer-python 0.6.18`,
  matching `flashinfer-cubin 0.6.18`, `torch 2.13.0+cu130`.
- Runtime image ID:
  `sha256:c4701cee4df917c8cc8c4afa34b5f7038662c2001dc2e963ab26aa5a02626483`.
- The actual wrapped checkpoint tokenizer rendered default/high with budget
  75, low with 50, max with 100, and explicit off without a reasoning
  prefix. The FlashInfer capability query accepts `(8 heads, 128 top-k)`
  and `(8 heads, 192 top-k)`. These are preflight checks; full-model GPU
  correctness and performance are still pending.
- `uv pip check --system` reports the same single mismatch in both the
  official base and the overlay: torch declares NCCL 2.29.7, while the
  official image supplies NCCL 2.30.7. The overlay preserves that base
  package; TP8 execution still needs the live gates below.
