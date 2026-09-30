"""GPU numerical gate for this example's SM120 kernel adaptations.

Run inside the built L1 image with one SM120 GPU:

    docker run --rm --gpus device=0 -v "$PWD:/checks:ro" --entrypoint python3 \
        <image> /checks/check_sm120_kernels.py

1. Sparse-MLA attention with 64-token main pages and 32/64-token secondary
   (C2/C1) pages, decode and prefill, 128/192 top-k, masked and unmasked,
   against independent PyTorch attention over the actual packed cache bytes.
2. Masked rows: the cache slot an invalid (-1) index used to alias is filled
   with NaN bytes; the output must stay finite and equal the reference.
3. The MXFP4 indexer (Q/K writers, dense prefill and paged decode logits) for
   C1/C2 pages against independently unpacked PyTorch logits.
"""

import importlib
import itertools
import json
import math

import torch
from flashinfer.mla._sparse_mla_sm120 import _SparseMLAPagedAttentionRunner
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
)


def _first(*candidates):
    for module, name in candidates:
        try:
            return getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError):
            continue
    raise ImportError(f"none of {candidates} is importable")


quantize_and_insert_k_cache = _first(
    ("vllm.models.deepseek_v4.common.ops.cache_utils", "quantize_and_insert_k_cache"),
    ("vllm.models.deepseek_v41.common.ops.cache_utils", "quantize_and_insert_k_cache"),
)
fused_indexer_q_rope_quant = _first(
    ("vllm.models.deepseek_v4.common.ops.fused_indexer_q", "fused_indexer_q_rope_quant"),
    ("vllm.models.deepseek_v41.common.ops.fused_indexer_q", "fused_indexer_q_rope_quant"),
)
indexer_k_norm_rope_store = _first(
    ("vllm.models.deepseek_v4_1.common.ops.indexer_k_store", "indexer_k_norm_rope_store"),
    ("vllm.models.deepseek_v41.common.ops.indexer_k_store", "indexer_k_norm_rope_store"),
)
kv_cache_as_quant_view = _first(
    ("vllm.model_executor.layers.sparse_attn_indexer", "kv_cache_as_quant_view"),
    ("vllm.model_executor.kernels.attention.dsa.sparse_attn_indexer", "kv_cache_as_quant_view"),
)

ROWS = 512
ROW_BYTES = 584  # 448 FP8 nope + 64 BF16 rope + 8 scale bytes per token


def packed_cache(page_size: int):
    # FlashInfer's DSV4 tests draw randn / 10 clamped to [-1, 1]; the per-tile
    # factors additionally exercise distinct UE8M0 scales per 64-wide tile.
    source = (torch.randn(ROWS, 512, device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    source[:, :448] *= torch.linspace(0.1, 2, 7, device="cuda").repeat_interleave(64)
    storage = torch.zeros(
        ROWS // page_size, page_size * ROW_BYTES + 128, device="cuda", dtype=torch.uint8
    )
    packed = storage[:, : page_size * ROW_BYTES]
    quantize_and_insert_k_cache(source, packed, torch.arange(ROWS, device="cuda"), page_size)
    # Decode the stored bytes independently of the kernel under test.
    data = packed[:, : page_size * 576].reshape(-1, page_size, 576)
    exponents = packed[:, page_size * 576 :].reshape(-1, page_size, 8)[..., :7]
    scales = torch.exp2(exponents.float() - 127).repeat_interleave(64, dim=-1)
    nope = data[..., :448].view(torch.float8_e4m3fn).float() * scales
    rope = data[..., 448:].view(torch.bfloat16).float()
    reference = torch.cat((nope, rope), dim=-1).reshape(ROWS, 512)
    return packed, reference


def poison_row_zero(packed: torch.Tensor, page_size: int) -> None:
    """Fill token 0's bytes (data and scale) with NaN/garbage patterns."""
    packed[0, :576] = 0xFF  # FP8 E4M3 0xFF is NaN; BF16 0xFFFF is NaN
    packed[0, page_size * 576 : page_size * 576 + 8] = 0xFF


def attention_cases() -> list[dict]:
    runner = _SparseMLAPagedAttentionRunner(d_v=512)
    rows = []
    for extra_page, tokens, topk, masked, poisoned in itertools.product(
        (32, 64), (1, 128), (128, 192), (False, True), (False, True)
    ):
        if poisoned and not masked:
            continue
        main, main_ref = packed_cache(64)
        extra, extra_ref = packed_cache(extra_page)
        # Valid indices never select token 0 when it is poisoned.
        low = 1 if poisoned else 0
        indices = torch.randint(low, ROWS, (tokens, topk), device="cuda", dtype=torch.int32)
        extra_indices = torch.randint(low, ROWS, (tokens, 128), device="cuda", dtype=torch.int32)
        lens = extra_lens = None
        if masked:
            lens = torch.randint(1, topk, (tokens,), device="cuda", dtype=torch.int32)
            extra_lens = torch.randint(1, 128, (tokens,), device="cuda", dtype=torch.int32)
            indices.masked_fill_(torch.arange(topk, device="cuda")[None] >= lens[:, None], -1)
            extra_indices.masked_fill_(
                torch.arange(128, device="cuda")[None] >= extra_lens[:, None], -1
            )
        if poisoned:
            poison_row_zero(main, 64)
            poison_row_zero(extra, extra_page)
        q = (torch.randn(tokens, 8, 512, device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
        sink = torch.linspace(-1, 3, 8, device="cuda")
        output = torch.empty_like(q)
        runner.run(
            q,
            main.view(-1, 64, 1, ROW_BYTES),
            indices,
            output,
            1 / math.sqrt(512),
            topk_length=lens,
            attn_sink=sink,
            extra_kv_cache=extra.view(-1, extra_page, 1, ROW_BYTES),
            extra_indices=extra_indices,
            extra_topk_length=extra_lens,
        )
        values = torch.cat(
            (main_ref[indices.clamp_min(0).long()], extra_ref[extra_indices.clamp_min(0).long()]),
            dim=1,
        )
        valid = torch.cat((indices >= 0, extra_indices >= 0), dim=1)
        values = values.masked_fill(~valid[..., None], 0.0)
        logits = torch.einsum("thd,tkd->thk", q.float(), values) / math.sqrt(512)
        logits.masked_fill_(~valid[:, None, :], -torch.inf)
        weights = torch.cat((logits, sink[None, :, None].expand(tokens, -1, -1)), dim=-1).softmax(
            -1
        )[..., :-1]
        reference = torch.einsum("thk,tkd->thd", weights, values)
        if not torch.isfinite(output.float()).all():
            raise AssertionError(
                f"non-finite attention output (poisoned={poisoned}, extra_page={extra_page})"
            )
        # FlashInfer's DSV4 SM120 test tolerances (FP8 QK/XV intermediates).
        torch.testing.assert_close(output.float(), reference, atol=0.05, rtol=0.05)
        rows.append(
            {
                "check": "attention",
                "extra_page": extra_page,
                "tokens": tokens,
                "topk": topk,
                "masked": masked,
                "poisoned_row_zero": poisoned,
                "max_abs_error": (output.float() - reference).abs().max().item(),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    return rows


def _unpack_mxfp4(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    nibbles = torch.stack((values & 15, values >> 4), dim=-1).flatten(-2)
    signed = lut[(nibbles & 7).long()] * torch.where(nibbles >= 8, -1, 1)
    return signed * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)


def indexer_cases() -> list[dict]:
    rows = []
    for ratio in (1, 2):
        page, heads, dim = 64 // ratio, 32, 128
        positions = torch.arange(ROWS * ratio, device="cuda")
        angles = torch.randn(ROWS * ratio, 32, device="cuda")
        rope = torch.cat((angles.cos(), angles.sin()), dim=-1)
        storage = torch.zeros(ROWS // page, page * 68 + 128, device="cuda", dtype=torch.uint8)
        cache = storage[:, : page * 68].view(-1, page, 68)
        k = torch.randn(ROWS * ratio, dim, device="cuda", dtype=torch.bfloat16)
        indexer_k_norm_rope_store(
            k,
            positions,
            rope,
            torch.ones(dim, device="cuda"),
            1e-6,
            cache,
            positions // ratio,
            ratio,
            True,
        )
        data = storage[:, : page * 64].reshape(ROWS, 64).contiguous()
        sf = storage[:, page * 64 : page * 68].reshape(ROWS, 4).contiguous()
        k_ref = _unpack_mxfp4(data, sf)
        for tokens in (2, 128):
            q = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
            weight = torch.rand(tokens, heads, device="cuda", dtype=torch.bfloat16)
            (qv, qs), weights = fused_indexer_q_rope_quant(
                positions[:tokens],
                q,
                rope,
                weight,
                1 / math.sqrt(dim),
                1 / math.sqrt(heads),
                use_fp4=True,
            )
            q_ref = _unpack_mxfp4(qv, qs.unsqueeze(-1).view(torch.uint8))
            oracle = (torch.einsum("mhd,nd->mhn", q_ref, k_ref).relu() * weights[..., None]).sum(1)
            if tokens == 2:
                context = torch.full((tokens, 1), ROWS, device="cuda", dtype=torch.int32)
                table = torch.arange(ROWS // page, device="cuda", dtype=torch.int32)[None].repeat(
                    tokens, 1
                )
                metadata = get_paged_mqa_logits_metadata(
                    context, page, torch.cuda.get_device_properties(0).multi_processor_count
                )
                result = fp8_fp4_paged_mqa_logits(
                    (qv.view(torch.int8).unsqueeze(1), qs.unsqueeze(1)),
                    kv_cache_as_quant_view(cache, dim, True),
                    weights,
                    context,
                    table,
                    metadata,
                    max_model_len=ROWS,
                    clean_logits=False,
                )
            else:
                result = fp8_fp4_mqa_logits(
                    (qv.view(torch.int8), qs),
                    (data.view(torch.int8), sf.view(torch.int32).squeeze(-1)),
                    weights,
                    torch.zeros(tokens, device="cuda", dtype=torch.int32),
                    torch.full((tokens,), ROWS, device="cuda", dtype=torch.int32),
                    clean_logits=False,
                )
            torch.testing.assert_close(result[:, :ROWS], oracle, atol=0.01, rtol=0.01)
            rows.append(
                {
                    "check": "indexer",
                    "ratio": ratio,
                    "page": page,
                    "tokens": tokens,
                    "max_abs_error": (result[:, :ROWS] - oracle).abs().max().item(),
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    return rows


def main() -> None:
    torch.manual_seed(4175)
    rows = attention_cases() + indexer_cases()
    print(json.dumps({"passed": True, "cases": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
