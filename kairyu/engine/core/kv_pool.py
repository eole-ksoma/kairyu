"""PagedKVPool: multi-layer KV tensors indexed by the scheduler's page ids (m12 D3).

Layout is layer-major — ``[num_layers, num_pages, page_size, num_kv_heads,
head_dim]`` per K and V — so M18's KVTransport fragments (per-layer ×
per-shard) slice contiguously. The same page ids index every layer;
``RadixKVCache`` and the ``Scheduler`` stay page-id accounting only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.models.config import ModelConfig


class PagedKVPool:
    def __init__(
        self,
        num_layers: int,
        num_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        v_head_dim: int | None = None,
    ) -> None:
        # MLA caches only the latent in k (v width 0) — m15 A7
        v_dim = head_dim if v_head_dim is None else v_head_dim
        self.k = torch.zeros(
            (num_layers, num_pages, page_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v = torch.zeros(
            (num_layers, num_pages, page_size, num_kv_heads, v_dim),
            dtype=dtype,
            device=device,
        )
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.v_head_dim = v_dim

    @classmethod
    def for_cache(
        cls,
        cache: RadixKVCache,
        config: ModelConfig,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> PagedKVPool:
        """Single source of truth for sizing (m12 D3): pool follows the cache."""
        return cls(
            num_layers=config.num_hidden_layers,
            num_pages=cache.num_pages,
            page_size=cache.page_size,
            num_kv_heads=config.kv_cache_num_heads,
            head_dim=config.kv_cache_head_dim,
            dtype=dtype,
            device=device,
            v_head_dim=config.kv_cache_v_head_dim,
        )

    @property
    def bytes_per_token(self) -> int:
        element = self.k.element_size()
        per_head = self.head_dim + self.v_head_dim
        return self.num_layers * self.num_kv_heads * per_head * element

    def _flat_indices(self, page_table: list[int], positions: torch.Tensor) -> torch.Tensor:
        # index tensors must live on the pool's device: on GPU `positions` arrives
        # on-device, so a CPU page-table tensor would raise on the gather. CPU pools
        # keep the original behaviour (self.k.device == cpu).
        table = torch.tensor(page_table, dtype=torch.long, device=self.k.device)
        pages = table[positions // self.page_size]
        return pages * self.page_size + positions % self.page_size

    def write(
        self,
        layer: int,
        page_table: list[int],
        positions: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """keys/values: [T, num_kv_heads, head_dim] at absolute positions [T]."""
        flat = self._flat_indices(page_table, positions)
        self.k[layer].reshape(-1, self.num_kv_heads, self.head_dim)[flat] = keys
        if self.v_head_dim:  # MLA pools have no v payload (width 0)
            self.v[layer].reshape(-1, self.num_kv_heads, self.v_head_dim)[flat] = values

    def gather(
        self, layer: int, page_table: list[int], seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (K, V) of shape [seq_len, num_kv_heads, head_dim]."""
        num_pages = -(-seq_len // self.page_size)
        index = torch.tensor(page_table[:num_pages], dtype=torch.long, device=self.k.device)
        keys = self.k[layer, index].reshape(-1, self.num_kv_heads, self.head_dim)[:seq_len]
        if self.v_head_dim:
            values = self.v[layer, index].reshape(-1, self.num_kv_heads, self.v_head_dim)[
                :seq_len
            ]
        else:  # MLA: the latent lives in k; v is a width-0 placeholder
            values = keys[:, :, :0]
        return keys, values
