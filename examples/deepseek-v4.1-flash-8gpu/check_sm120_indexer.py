"""Exercise V4.1's real MXFP4 writers and DeepGEMM indexer on SM120.

The oracle dequantizes the stored Q/K nibbles independently in PyTorch.
This verifies the kernel path before enabling it through the example overlay.
"""

import json
import math

import torch
from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import fused_indexer_q_rope_quant
from vllm.models.deepseek_v4_1.common.ops.indexer_k_store import indexer_k_norm_rope_store
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
)


def unpack(values, scales):
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    nibbles = torch.stack((values & 15, values >> 4), dim=-1).flatten(-2)
    signed = lut[(nibbles & 7).long()] * torch.where(nibbles >= 8, -1, 1)
    return signed * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)


def main():
    torch.manual_seed(4175)
    rows = []
    for ratio in (1, 2):
        page, n, heads, dim = 64 // ratio, 512, 32, 128
        positions = torch.arange(n * ratio, device="cuda")
        angles = torch.randn(n * ratio, 32, device="cuda")
        rope = torch.cat((angles.cos(), angles.sin()), dim=-1)
        storage = torch.zeros(n // page, page * 68 + 128, device="cuda", dtype=torch.uint8)
        cache = storage[:, : page * 68].view(-1, page, 68)
        k = torch.randn(n * ratio, dim, device="cuda", dtype=torch.bfloat16)
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
        data = storage[:, : page * 64].reshape(n, 64).contiguous()
        sf = storage[:, page * 64 : page * 68].reshape(n, 4).contiguous()
        k_ref = unpack(data, sf)
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
            q_ref = unpack(qv, qs.unsqueeze(-1).view(torch.uint8))
            oracle = (torch.einsum("mhd,nd->mhn", q_ref, k_ref).relu() * weights[..., None]).sum(1)
            if tokens == 2:
                context = torch.full((tokens, 1), n, device="cuda", dtype=torch.int32)
                blocks = torch.arange(n // page, device="cuda", dtype=torch.int32)
                table = blocks[None].repeat(tokens, 1)
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
                    max_model_len=n,
                    clean_logits=False,
                )
            else:
                result = fp8_fp4_mqa_logits(
                    (qv.view(torch.int8), qs),
                    (data.view(torch.int8), sf.view(torch.int32).squeeze(-1)),
                    weights,
                    torch.zeros(tokens, device="cuda", dtype=torch.int32),
                    torch.full((tokens,), n, device="cuda", dtype=torch.int32),
                    clean_logits=False,
                )
            torch.testing.assert_close(result[:, :n], oracle, atol=0.01, rtol=0.01)
            rows.append(
                {
                    "ratio": ratio,
                    "page": page,
                    "tokens": tokens,
                    "max_abs_error": (result[:, :n] - oracle).abs().max().item(),
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    print(json.dumps({"passed": True, "cases": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
