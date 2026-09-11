"""GPU numerical gate for the example's V4.1 sparse-MLA page adaptations.

Run inside the pinned L1 image with one SM120 GPU. Uses real packed FP8 caches
with padded block strides, independent PyTorch dequantization/attention, both
compression ratios, decode/prefill, masking, sinks, and DSpark-width indices.
"""

import itertools
import json
import math

import torch
from flashinfer.mla._sparse_mla_sm120 import _SparseMLAPagedAttentionRunner
from vllm.models.deepseek_v4.common.ops.cache_utils import quantize_and_insert_k_cache


def cache(page_size):
    n = 512
    source = torch.randn(n, 512, device="cuda", dtype=torch.bfloat16)
    source[:, :448] *= torch.linspace(0.1, 2, 7, device="cuda").repeat_interleave(64)
    storage = torch.zeros(n // page_size, page_size * 584 + 128, device="cuda", dtype=torch.uint8)
    packed = storage[:, : page_size * 584]
    quantize_and_insert_k_cache(source, packed, torch.arange(n, device="cuda"), page_size)
    # Decode the actual stored bytes independently of the kernel under test.
    data = packed[:, : page_size * 576].reshape(-1, page_size, 576)
    exponents = packed[:, page_size * 576 :].reshape(-1, page_size, 8)[..., :7]
    scales = torch.exp2(exponents.float() - 127).repeat_interleave(64, dim=-1)
    nope = data[..., :448].view(torch.float8_e4m3fn).float() * scales
    rope = data[..., 448:].view(torch.bfloat16).float()
    reference = torch.cat((nope, rope), dim=-1).reshape(n, 512)
    return packed.view(-1, page_size, 1, 584), reference


def main():
    torch.manual_seed(4175)
    runner = _SparseMLAPagedAttentionRunner(d_v=512)
    results = []
    for extra_page, tokens, topk, masked in itertools.product(
        (32, 64), (1, 128), (128, 192), (False, True)
    ):
        main_cache, main_ref = cache(64)
        extra_cache, extra_ref = cache(extra_page)
        q = torch.randn(tokens, 8, 512, device="cuda", dtype=torch.bfloat16)
        sink = torch.linspace(-1, 3, 8, device="cuda")
        indices = torch.randint(0, 512, (tokens, topk), device="cuda", dtype=torch.int32)
        extra_indices = torch.randint(0, 512, (tokens, 128), device="cuda", dtype=torch.int32)
        lens = extra_lens = None
        if masked:
            lens = torch.randint(1, topk, (tokens,), device="cuda", dtype=torch.int32)
            extra_lens = torch.randint(1, 128, (tokens,), device="cuda", dtype=torch.int32)
            indices.masked_fill_(torch.arange(topk, device="cuda")[None] >= lens[:, None], -1)
            extra_indices.masked_fill_(
                torch.arange(128, device="cuda")[None] >= extra_lens[:, None], -1
            )
        output = torch.empty_like(q)
        runner.run(
            q,
            main_cache,
            indices,
            output,
            1 / math.sqrt(512),
            topk_length=lens,
            attn_sink=sink,
            extra_kv_cache=extra_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_lens,
        )
        values = torch.cat(
            (main_ref[indices.clamp_min(0).long()], extra_ref[extra_indices.clamp_min(0).long()]),
            dim=1,
        )
        valid = torch.cat((indices >= 0, extra_indices >= 0), dim=1)
        logits = torch.einsum("thd,tkd->thk", q.float(), values) / math.sqrt(512)
        logits.masked_fill_(~valid[:, None, :], -torch.inf)
        weights = torch.cat((logits, sink[None, :, None].expand(tokens, -1, -1)), dim=-1).softmax(
            -1
        )[..., :-1]
        reference = torch.einsum("thk,tkd->thd", weights, values)
        # Match FlashInfer's pinned DSV4 tests (FP8 QK/XV intermediates):
        # tests/attention/test_sparse_mla_sm120.py at 60b49158, dual-cache gates.
        torch.testing.assert_close(output.float(), reference, atol=0.05, rtol=0.05)
        results.append(
            {
                "extra_page": extra_page,
                "tokens": tokens,
                "topk": topk,
                "masked": masked,
                "max_abs_error": (output.float() - reference).abs().max().item(),
            }
        )
        print(json.dumps(results[-1]), flush=True)
    print(json.dumps({"passed": True, "cases": len(results)}), flush=True)


if __name__ == "__main__":
    main()
