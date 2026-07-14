"""FlashInfer paged-attention adapter (m13 D4).

API pins (reviewed against docs.flashinfer.ai 0.6.x — the fake-module contract
tests enforce every one of them):
- both wrappers take a WORKSPACE buffer first (128 MB uint8), constructed once
  — the adapter is stateful and must be ONE shared instance across all layers;
- prefill ``plan()`` takes ``head_dim_qk`` (NOT ``head_dim`` — that spelling
  is decode-wrapper-only);
- ``q_data_type``/``kv_data_type`` are passed explicitly (defaults are fp16);
- indptr/indices/last_page_len are int32 tensors — indptr and last_page_len on
  HOST, indices on the device;
- ``causal=True`` is bottom-right aligned: correct iff
  ``chunk_start + T == seq_len`` (asserted; every call site satisfies it);
- the plan is cached per (page_table, seq_len, chunk_start, T) so layer 0
  plans and layers 1..N-1 just run.

This module is coverage-omitted (``*_gpu.py``); its LOGIC is still CPU-tested
via the injected fake module.
"""

from __future__ import annotations

import torch

from kairyu.engine.core.kv_pool import PagedKVPool

_WORKSPACE_BYTES = 128 * 1024 * 1024


class FlashInferBackend:
    """One instance serves every layer: shared workspace + plan cache."""

    def __init__(self, device: str = "cuda") -> None:
        import flashinfer  # deferred: not installable on macOS; [gpu] extra

        self._flashinfer = flashinfer
        self._device = device
        workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        self._prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD"
        )
        decode_workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        self._decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            decode_workspace, kv_layout="NHD", use_tensor_cores=True
        )
        self._plan_key: tuple | None = None
        self._planned_decode = False

    def _paged_arrays(
        self, page_table: list[int], seq_len: int, page_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_pages = -(-seq_len // page_size)
        indptr = torch.tensor([0, num_pages], dtype=torch.int32)  # host
        indices = torch.tensor(
            page_table[:num_pages], dtype=torch.int32, device=self._device
        )
        last_page_len = torch.tensor(
            [(seq_len - 1) % page_size + 1], dtype=torch.int32
        )  # host
        return indptr, indices, last_page_len

    def _plan(
        self,
        query: torch.Tensor,
        kv_pool: PagedKVPool,
        page_table: list[int],
        seq_len: int,
        chunk_start: int,
    ) -> bool:
        """Plan (or reuse the cached plan); returns True for the decode path."""
        chunk_len = query.shape[0]
        assert chunk_start + chunk_len == seq_len, (
            "FlashInfer causal=True is bottom-right aligned: the chunk must be "
            f"the tail of the sequence (chunk_start={chunk_start} T={chunk_len} "
            f"seq_len={seq_len})"
        )
        key = (tuple(page_table), seq_len, chunk_start, chunk_len)
        is_decode = chunk_len == 1
        if key == self._plan_key and is_decode == self._planned_decode:
            return is_decode
        indptr, indices, last_page_len = self._paged_arrays(
            page_table, seq_len, kv_pool.page_size
        )
        if is_decode:
            self._decode.plan(
                indptr,
                indices,
                last_page_len,
                query.shape[1],  # num_qo_heads
                kv_pool.num_kv_heads,
                kv_pool.head_dim,
                kv_pool.page_size,
                q_data_type=query.dtype,
                kv_data_type=kv_pool.k.dtype,
            )
        else:
            qo_indptr = torch.tensor([0, chunk_len], dtype=torch.int32)  # host
            self._prefill.plan(
                qo_indptr,
                indptr,
                indices,
                last_page_len,
                query.shape[1],  # num_qo_heads
                kv_pool.num_kv_heads,
                head_dim_qk=kv_pool.head_dim,  # NOT head_dim (prefill spelling)
                page_size=kv_pool.page_size,
                causal=True,
                q_data_type=query.dtype,
                kv_data_type=kv_pool.k.dtype,
            )
        self._plan_key = key
        self._planned_decode = is_decode
        return is_decode

    def attend(
        self,
        query: torch.Tensor,
        kv_pool: PagedKVPool,
        layer: int,
        page_table: list[int],
        seq_len: int,
        chunk_start: int,
    ) -> torch.Tensor:
        is_decode = self._plan(query, kv_pool, page_table, seq_len, chunk_start)
        paged_kv = (kv_pool.k[layer], kv_pool.v[layer])  # NHD tuple form
        wrapper = self._decode if is_decode else self._prefill
        if is_decode:
            out = wrapper.run(query, paged_kv)  # decode: [B=1, H, D] query
            return out.reshape(1, -1)
        out = wrapper.run(query, paged_kv)  # [T, H, D]
        return out.reshape(query.shape[0], -1)

    def attend_batched(
        self,
        queries: list[torch.Tensor],
        kv_pool: PagedKVPool,
        layer: int,
        page_tables: list[list[int]],
        seq_lens: list[int],
        chunk_starts: list[int],
    ) -> list[torch.Tensor]:
        """Per-sequence contexts, identical to per-sequence ``attend``.

        This correctness-first loop satisfies the backend contract used by
        batched decode. A later optimization can collapse it into one FlashInfer
        batched plan/run over multi-row indptr arrays.
        """
        return [
            self.attend(
                queries[i], kv_pool, layer, page_tables[i], seq_lens[i], chunk_starts[i]
            )
            for i in range(len(queries))
        ]
