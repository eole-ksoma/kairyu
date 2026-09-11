# Measurement status

Implementation and GPU selection are in progress. No V4.1 performance or
correctness gate is claimed passed until the exact run evidence is recorded.

Hardware target: 8 × NVIDIA RTX PRO 6000 Blackwell Server Edition,
97,887 MiB per GPU, PCIe, one TP8 replica.

Model revision: `dba1be0a40aa45a94ad051997016db3960a90277`.
Model tree SHA-256: `d21211ca29ad7eba1fda84e49b1a34a73214ec9b84b23928cde63902c3318bfd`.

## Preflight evidence

- Related portable CPU contracts: 57 passed. Repository lint and progress-size
  checks passed.
- Built runtime: `vllm 0.1.dev20904+g179dd0fa9`, `flashinfer-python 0.6.18`,
  matching `flashinfer-cubin 0.6.18`, `torch 2.13.0+cu130`.
- Runtime image ID:
  `sha256:82cdd20125d935a175c790a3ba3a4d27006f8c04cf024571ef05cfeed98aa46d`.
- The actual wrapped checkpoint tokenizer rendered default/high with budget
  75, low with 50, max with 100, and explicit off without a reasoning
  prefix. The FlashInfer capability query accepts `(8 heads, 128 top-k)`
  and `(8 heads, 192 top-k)`. These are preflight checks; full-model GPU
  correctness and performance are still pending.
- `uv pip check --system` reports the same single mismatch in both the
  official base and the overlay: torch declares NCCL 2.29.7, while the
  official image supplies NCCL 2.30.7. The overlay preserves that base
  package; TP8 execution still needs the live gates below.

## Initial startup selection

The first TP8/EP8 attempt loaded the target and DSpark weights (62.07 GiB
per GPU), but `enable_adaptive_verification=true` failed during CUDA Graph
memory profiling: the pinned `DeepseekV4IndexerBackend` does not support
on-device verification-length trimming. Set adaptive verification to false;
retain the five-token DSpark candidate. The failed attempt is preserved in
`initial-worker.log` on the NVMe example storage.

The next attempt with adaptive verification disabled reached KV allocation,
but the default BLHNC layout cannot split a 256-token manager block into the
SM120 backend's 128-token kernel blocks. Set `--block-size=128` and match
the gateway cache descriptor. The rejected 256/BLHNC attempt is preserved
in `no-adaptive-worker.log`.

The 128/BLHNC attempt passed KV view construction but exposed the V4.1
attention class's hardcoded 32-token SWA pages (SM120 requires 64). The
alternative LBNHC layout is rejected by the packed indexer backend. Keep
128-token manager blocks and the default BLHNC layout. The example overlay
sets SWA to 64 on SM120 only and instantiates FlashInfer's existing generic
dual-cache prefill templates for C1's 128-token compressed pages (C2 uses
64). No numerical operation or layout address calculation is replaced.
The failed attempts are retained in `block128-worker.log` and
`lbnhc-worker.log`. Kernel parity passed all 16 decode/prefill, C1/C2, 128/192 top-k and
mask combinations using independent PyTorch attention over the packed cache
bytes (`check_sm120_pages.py`; retained `sm120-page-parity.log`). The
absolute/relative tolerances 0.05/0.05 match the pinned FlashInfer DSV4
tests; maximum observed absolute error was 0.0642 (within combined tolerance).
Full-model gates remain pending.
