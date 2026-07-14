"""m13: backend contract, MLA equivalence, fake-flashinfer pins, selector."""

import sys
import types

import pytest
import torch

from kairyu.engine.core.attention import TorchAttentionBackend, select_backend
from kairyu.engine.core.attention.mla_torch import mla_absorbed, mla_decompress, mla_scale
from kairyu.engine.core.hw_profile import HardwareProfile
from kairyu.engine.core.kv_pool import PagedKVPool

PAGE = 4


def _pool(layers=1, kv_heads=2, head_dim=8) -> PagedKVPool:
    return PagedKVPool(layers, 16, PAGE, kv_heads, head_dim)


class TestTorchBackend:
    def test_matches_naive_attention(self):
        torch.manual_seed(0)
        pool = _pool()
        seq_len, heads = 10, 4
        keys = torch.randn(seq_len, 2, 8)
        values = torch.randn(seq_len, 2, 8)
        pool.write(0, [0, 1, 2], torch.arange(seq_len), keys, values)
        query = torch.randn(3, heads, 8)  # chunk positions 7..10
        out = TorchAttentionBackend().attend(query, pool, 0, [0, 1, 2], seq_len, 7)
        # naive oracle with repeated kv heads
        k_rep = keys.repeat_interleave(2, dim=1)
        v_rep = values.repeat_interleave(2, dim=1)
        scores = torch.einsum("thd,shd->hts", query, k_rep) * 8**-0.5
        mask = torch.arange(seq_len)[None, :] <= (torch.arange(3)[:, None] + 7)
        scores = scores.masked_fill(~mask[None], float("-inf"))
        naive = torch.einsum("hts,shd->thd", torch.softmax(scores, -1), v_rep)
        assert torch.allclose(out, naive.reshape(3, -1), atol=1e-6)


class TestMlaEquivalence:
    @pytest.mark.parametrize("d_v", [8, 12])  # d_nope != d_v exercises fold direction
    def test_decompress_equals_absorbed_and_oracle(self, d_v):
        torch.manual_seed(1)
        heads, d_nope, d_rope, rank, seq, chunk = 3, 8, 4, 16, 9, 9
        q_nope = torch.randn(chunk, heads, d_nope)
        q_pe = torch.randn(chunk, heads, d_rope)
        c_kv = torch.randn(seq, rank)
        k_pe = torch.randn(seq, d_rope)  # shared single head, post-RoPE
        w_uk = torch.randn(heads, rank, d_nope)
        w_uv = torch.randn(heads, rank, d_v)
        scale = mla_scale(d_nope, d_rope)
        a = mla_decompress(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, scale)
        b = mla_absorbed(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, scale)
        assert torch.allclose(a, b, atol=1e-5)
        # naive full-materialization oracle: per-head K = [k_nope || k_pe]
        k_nope = torch.einsum("sr,hrd->hsd", c_kv, w_uk)
        k_full = torch.cat([k_nope, k_pe[None].expand(heads, -1, -1)], dim=-1)
        q_full = torch.cat([q_nope, q_pe], dim=-1)
        scores = torch.einsum("thd,hsd->hts", q_full, k_full) * scale
        mask = torch.arange(seq)[None, :] <= torch.arange(chunk)[:, None]
        scores = scores.masked_fill(~mask[None], float("-inf"))
        values = torch.einsum("sr,hrd->hsd", c_kv, w_uv)
        oracle = torch.einsum("hts,hsd->thd", torch.softmax(scores, -1), values)
        assert torch.allclose(a, oracle, atol=1e-5)

    def test_wrong_scale_breaks_equivalence_gate(self):
        # guards against default-vs-default silently passing (review B1)
        torch.manual_seed(2)
        q_nope = torch.randn(4, 2, 8)
        q_pe = torch.randn(4, 2, 4)
        c_kv = torch.randn(4, 16)
        k_pe = torch.randn(4, 4)
        w_uk = torch.randn(2, 16, 8)
        w_uv = torch.randn(2, 16, 8)
        right = mla_decompress(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, mla_scale(8, 4))
        wrong = mla_decompress(
            q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, (16 + 4) ** -0.5
        )
        assert not torch.allclose(right, wrong, atol=1e-5)


class _FakeWrapper:
    def __init__(self, workspace, kv_layout=None, use_tensor_cores=None):
        assert workspace.dtype == torch.uint8 and workspace.numel() >= 1
        self.kv_layout = kv_layout
        self.use_tensor_cores = use_tensor_cores
        self.plans: list[dict] = []
        self.runs: list[tuple] = []

    def plan(self, *args, **kwargs):
        self.plans.append({"args": args, "kwargs": kwargs})

    def run(self, query, paged_kv):
        assert self.plans, "run() before plan() (contract violation)"
        self.runs.append((query, paged_kv))
        heads_dim = query.shape[-2:] if query.dim() >= 2 else (1, 1)
        if query.dim() == 2:
            raise AssertionError("FlashInfer decode.run expects [B, H, D]")
        return torch.zeros(query.shape[0], *heads_dim)


@pytest.fixture()
def fake_flashinfer(monkeypatch):
    module = types.ModuleType("flashinfer")
    module.BatchPrefillWithPagedKVCacheWrapper = _FakeWrapper
    module.BatchDecodeWithPagedKVCacheWrapper = _FakeWrapper
    monkeypatch.setitem(sys.modules, "flashinfer", module)
    return module


class TestFlashInferAdapterContract:
    def _backend(self):
        from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend

        return FlashInferBackend(device="cpu")

    def test_prefill_plan_pins_indptr_math_and_kwargs(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool()
        query = torch.randn(5, 4, 8)
        backend.attend(query, pool, 0, [3, 1, 2], seq_len=10, chunk_start=5)
        plan = backend._prefill.plans[-1]
        qo_indptr, kv_indptr, kv_indices, last_page_len = plan["args"][:4]
        assert qo_indptr.tolist() == [0, 5] and qo_indptr.dtype == torch.int32
        assert kv_indptr.tolist() == [0, 3]  # ceil(10/4) pages
        assert kv_indices.tolist() == [3, 1, 2] and kv_indices.dtype == torch.int32
        assert last_page_len.tolist() == [2]  # (10-1) % 4 + 1
        # prefill spelling: head_dim_qk, NOT head_dim (review A1)
        assert plan["kwargs"]["head_dim_qk"] == 8
        assert "head_dim" not in plan["kwargs"]
        assert plan["kwargs"]["causal"] is True
        assert plan["kwargs"]["q_data_type"] == query.dtype  # explicit (A3)
        assert plan["kwargs"]["kv_data_type"] == pool.k.dtype

    def test_decode_path_uses_decode_wrapper(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool()
        query = torch.randn(1, 4, 8)
        backend.attend(query, pool, 0, [0, 1], seq_len=6, chunk_start=5)
        assert backend._decode.plans and not backend._prefill.plans
        assert backend._decode.use_tensor_cores is True
        run_query, _ = backend._decode.runs[-1]
        assert run_query.shape == (1, 4, 8)

    def test_plan_cached_across_layers(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool(layers=3)
        query = torch.randn(2, 4, 8)
        for layer in range(3):
            backend.attend(query, pool, layer, [0, 1], seq_len=6, chunk_start=4)
        assert len(backend._prefill.plans) == 1  # layer 0 plans; 1..2 reuse (A8)
        assert len(backend._prefill.runs) == 3

    def test_non_tail_chunk_asserts(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool()
        with pytest.raises(AssertionError, match="bottom-right"):
            backend.attend(torch.randn(2, 4, 8), pool, 0, [0, 1], seq_len=8, chunk_start=2)

    def test_run_receives_nhd_tuple(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool(layers=2)
        backend.attend(torch.randn(2, 4, 8), pool, 1, [0], seq_len=2, chunk_start=0)
        _, paged_kv = backend._prefill.runs[-1]
        assert paged_kv[0].shape == (16, PAGE, 2, 8)  # pool.k[layer] NHD slice
        assert torch.equal(paged_kv[0], pool.k[1])

    def test_attend_batched_matches_per_sequence_contract(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool()
        queries = [torch.randn(1, 4, 8), torch.randn(1, 4, 8)]
        out = backend.attend_batched(
            queries,
            pool,
            0,
            page_tables=[[0, 1], [2, 3]],
            seq_lens=[5, 7],
            chunk_starts=[4, 6],
        )
        assert len(out) == 2
        assert [tuple(t.shape) for t in out] == [(1, 32), (1, 32)]
        assert len(backend._decode.runs) == 2


class TestSelector:
    def test_cpu_profile_selects_torch(self):
        assert isinstance(select_backend(None), TorchAttentionBackend)
        cpu = HardwareProfile(arch="cpu")
        assert isinstance(select_backend(cpu), TorchAttentionBackend)

    def test_env_override_torch(self, monkeypatch):
        monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "torch")
        sm120 = HardwareProfile(arch="cuda", sm=120)
        assert isinstance(select_backend(sm120), TorchAttentionBackend)

    def test_env_override_invalid_fails_loudly(self, monkeypatch):
        monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "bogus")
        with pytest.raises(ValueError, match="bogus"):
            select_backend(None)

    def test_gpu_tier_selects_flashinfer(self, fake_flashinfer, monkeypatch):
        from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend

        monkeypatch.setattr(FlashInferBackend, "__init__", lambda self, device="cuda": None)
        sm120 = HardwareProfile(arch="cuda", sm=120)
        assert isinstance(select_backend(sm120), FlashInferBackend)
