"""Runtime adaptations for DeepSeek-V4.1-Flash on six RTX PRO 6000 GPUs (SM120).

Every edit replaces exactly one known source anchor and fails the image build
when the anchor is missing or repeated, so an upstream change can never be
patched silently. The edits target one exact vLLM version (``VLLM_VERSION``);
any other version fails the build.

1. Effort aliases: the model author's encoder defines low=50, high=75, max=100
   with high as the default (checkpoint ``encoding/README.md``). The pinned
   runtime already renders them; the build only checks it.
2. Page sizes: FlashInfer's SM120 sparse-MLA kernels need 64-token SWA pages,
   V4.1's C1/C2 compression needs 64-token manager blocks (C1=64, C2=32), and
   DeepGEMM's SM120 indexer accepts only those page sizes.
3. MXFP4 indexer on SM120 for V4.1 only (SM120 FP8 indexer decode does not
   cover the C2 page size).
4. FlashInfer SM120 dual-cache prefill: instantiate the existing generic
   templates for 32-token secondary (C2) pages.
5. Masked sparse-KV rows: an invalid (-1) candidate index is clamped to row 0
   and read with weight 0, but 0 * NaN is NaN when that recycled slot holds
   non-finite bytes. Invalid rows are read from an aligned zero row instead;
   copy sizes and barrier transaction counts stay unchanged. Without this,
   every EP6 topology returned NaN log-probabilities or unrelated text after
   the first requests (six-GPU tiered branch, candidates 1-4).
6. Split top-p cutoff: a large forced logit (the thinking terminator) can
   round the top-p cutoff up to the maximum, and the strict ``>`` then masks
   every token. Mirror the monolithic sampler's guard.
7. Final assistant prefill: the V4.1 encoder ignores ``continue_final_message``,
   so a final assistant turn is closed with end-of-sentence and its content is
   rendered after an empty ``<think></think>``. Kairyu's public-output floor
   (DTO-D9/D15) continues a thinking span that consumed its cap by sending
   ``<think>\n{reasoning}\n</think>\n\n`` as that turn. The encoder now
   renders it as the open turn ``<｜Assistant｜><think>\n{reasoning}\n</think>\n\n``:
   the span becomes the turn's reasoning and end-of-sentence is omitted.
"""

# ruff: noqa: E501  (C++ anchors keep their pinned line boundaries)
from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

OFFICIAL_EFFORTS = {"low": 50, "high": 75, "max": 100}


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise ValueError(
            f"expected exactly one source anchor, found {source.count(before)}: {before[:120]!r}"
        )
    if after in source:
        raise ValueError(f"source already contains the replacement: {after[:120]!r}")
    return source.replace(before, after, 1)


# --- 1. effort aliases ---------------------------------------------------------


def check_efforts(encoder_path: Path) -> None:
    encoder = runpy.run_path(str(encoder_path))
    render = encoder["render_reasoning_effort"]
    for name, budget in OFFICIAL_EFFORTS.items():
        if f"Reasoning Effort: {budget} " not in render(0, "thinking", name):
            raise ValueError(f"encoder renders {name!r} without budget {budget}")
    if "Reasoning Effort: 75 " not in render(0, "thinking", None):
        raise ValueError("encoder default is not high (75)")
    if render(0, "chat", None) != "":
        raise ValueError("chat mode renders an effort prefix")


# --- 2. page sizes -------------------------------------------------------------

SM120_ATTENTION_DECLARATION = (
    "class DeepseekV4FlashInferSM120Attention(DeepseekV4Attention):\n"
    '    """DeepSeek V4 sparse MLA attention through FlashInfer\'s SM120 kernels."""\n'
    "\n"
    "    backend_cls = DeepseekV4FlashInferMLASparseBackend\n"
    "    swa_backend_cls = DeepseekSparseSWAFlashInferBackend\n"
)


def sm120_swa_pages(source: str) -> str:
    return replace_once(
        source,
        SM120_ATTENTION_DECLARATION,
        SM120_ATTENTION_DECLARATION + "    swa_block_size: ClassVar[int] = 64\n",
    )


def configurable_swa_pages(source: str) -> str:
    return replace_once(
        source,
        "            block_size=32,\n",
        "            block_size=getattr(self, 'swa_block_size', 32),\n",
    )


def sm120_mla_pages(source: str) -> str:
    return replace_once(
        source,
        "        return [128]\n",
        "        from vllm.platforms import current_platform\n"
        "        return [64 if current_platform.is_device_capability_family(120) else 128]\n",
    )


def sm120_indexer_pages(source: str) -> str:
    """The V4.1 indexer backend picks its page size per device family."""
    return replace_once(
        source,
        "        return [64 if current_platform.is_device_capability_family(90) else 128]\n",
        "        return [\n"
        "            64\n"
        "            if current_platform.is_device_capability_family(90)\n"
        "            or current_platform.is_device_capability_family(120)\n"
        "            else 128\n"
        "        ]\n",
    )


# --- 3. MXFP4 indexer gate -----------------------------------------------------


def sm120_v41_mxfp4_indexer(source: str) -> str:
    return replace_once(
        source,
        "    if use_fp4 and not current_platform.is_device_capability_family(100):\n",
        "    sm120_v41 = (\n"
        "        current_platform.is_device_capability_family(120)\n"
        "        and vllm_config.model_config.hf_config.model_type == 'deepseek_v41'\n"
        "    )\n"
        "    if use_fp4 and not (current_platform.is_device_capability_family(100) or sm120_v41):\n",
    )


# --- 4. FlashInfer C2 prefill pages ----------------------------------------------


def flashinfer_page32_prefill(source: str) -> str:
    edits = (
        (
            "(extra_page_block_size == 64 || extra_page_block_size == 2)",
            "(extra_page_block_size == 64 || extra_page_block_size == 32 || extra_page_block_size == 2)",
        ),
        (
            "      DISPATCH_FULLTILE_BY_NH_PBSX(64);\n    } else {",
            "      DISPATCH_FULLTILE_BY_NH_PBSX(64);\n"
            "    } else if (extra_page_block_size == 32) {\n"
            "      DISPATCH_FULLTILE_BY_NH_PBSX(32);\n    } else {",
        ),
        (
            "    DISPATCH_BY_NH_PBSX(64);\n  } else if (extra_page_block_size == 2)",
            "    DISPATCH_BY_NH_PBSX(64);\n"
            "  } else if (extra_page_block_size == 32) {\n"
            "    DISPATCH_BY_NH_PBSX(32);\n  } else if (extra_page_block_size == 2)",
        ),
    )
    for before, after in edits:
        source = replace_once(source, before, after)
    return source


# --- 5. masked sparse-KV rows ------------------------------------------------------

ZERO_ROW = """// Kairyu: invalid sparse candidates read this zero row instead of slot 0,
// whose recycled bytes may be non-finite (0 * NaN is NaN). Same copy size and
// barrier accounting as a valid row; never written.
static __device__ __align__(16) uint8_t {name}[2048] = {{}};

"""


def masked_kv_io(source: str) -> str:
    source = replace_once(
        source,
        "// KV cache IO: gather BI entries",
        ZERO_ROW.format(name="kKairyuInvalidKVZeroRow") + "// KV cache IO: gather BI entries",
    )
    source = replace_once(
        source,
        "  idx = (idx >= 0) ? idx : 0;\n\n  const uint8_t* src;",
        "  const bool valid_idx = idx >= 0;\n  idx = valid_idx ? idx : 0;\n\n  const uint8_t* src;",
    )
    source = replace_once(
        source,
        "  if constexpr (USE_L2_HINT)\n    cp_async_bulk_g2s_l2hint",
        "  static_assert(COPY_BYTES <= sizeof(kKairyuInvalidKVZeroRow));\n"
        "  if (!valid_idx) src = kKairyuInvalidKVZeroRow;\n"
        "  if constexpr (USE_L2_HINT)\n    cp_async_bulk_g2s_l2hint",
    )
    return replace_once(
        source,
        "  idx = (idx >= 0) ? idx : 0;\n\n  int block_idx",
        "  if (idx < 0) {\n"
        "    *reinterpret_cast<uint64_t*>(scale_dst + io_tid * SCALE_BYTES) = 0;\n"
        "    return;\n  }\n\n  int block_idx",
    )


def masked_kv_decode(source: str) -> str:
    # Decode is its own translation unit and does not include the common IO.
    source = replace_once(
        source,
        "namespace flashinfer::sparse_mla_sm120 {\n",
        "namespace flashinfer::sparse_mla_sm120 {\n\n"
        + ZERO_ROW.format(name="kKairyuDecodeInvalidKVZeroRow"),
    )
    return replace_once(
        source,
        "          section_kv + (size_t)block_idx_g * section_stride + (size_t)local_idx_g * IO_STRIDE;\n      cp_async_bulk_g2s(kv_fp8_dst",
        "          idx_raw[e] >= 0\n"
        "              ? section_kv + (size_t)block_idx_g * section_stride + (size_t)local_idx_g * IO_STRIDE\n"
        "              : kKairyuDecodeInvalidKVZeroRow;\n"
        "      static_assert(D_NOPE + DSV4_BULK_ROPE_BYTES <= sizeof(kKairyuDecodeInvalidKVZeroRow));\n"
        "      cp_async_bulk_g2s(kv_fp8_dst",
    )


def masked_kv_prefill(source: str) -> str:
    return replace_once(
        source, "  idx = (idx >= 0) ? idx : 0;", "  if (idx < 0) return kKairyuInvalidKVZeroRow;"
    )


# --- 6. split top-p cutoff ------------------------------------------------------------

TOP_P_ANCHOR = """    logZ = tl.log(Z)
    pivot_logit = tl.log(pivot) + logZ + M
    # numdup/numkeep are exact small integers held in fp32.
"""
TOP_P_REPLACEMENT = """    logZ = tl.log(Z)
    pivot_logit = tl.log(pivot) + logZ + M
    # Kairyu: like the monolithic sampler, strict > must keep a candidate when
    # a large forced logit rounds the cutoff to M (or the cutoff is NaN).
    if not (pivot_logit < M):
        pivot_logit = -float("inf")
    # numdup/numkeep are exact small integers held in fp32.
"""


def top_p_guard(source: str) -> str:
    return replace_once(source, TOP_P_ANCHOR, TOP_P_REPLACEMENT)


# --- 7. final assistant prefill -------------------------------------------------------

CONTINUATION_ANCHOR = "            conversation = _normalize_messages(messages)\n"
CONTINUATION_REPLACEMENT = """            conversation = _normalize_messages(messages)
            # Kairyu: continue a final assistant prefill instead of closing it.
            if (
                kwargs.get("continue_final_message")
                and conversation
                and conversation[-1].get("role") == "assistant"
            ):
                final = conversation[-1]
                text = final.get("content") or ""
                if text.startswith("<think>") and "</think>" in text:
                    reasoning, _, rest = text[len("<think>") :].partition("</think>")
                    final["reasoning_content"] = reasoning
                    final["content"] = rest
                final["wo_eos"] = True
"""


def final_assistant_prefill(source: str) -> str:
    return replace_once(source, CONTINUATION_ANCHOR, CONTINUATION_REPLACEMENT)


# --- profiles ----------------------------------------------------------------------

FLASHINFER_SM120 = "data/include/flashinfer/attention/sparse_mla_sm120"

# vllm/vllm-openai:nightly of 2026-09-30 (FlashInfer 0.7.0.post1 bundled).
VLLM_VERSION = "0.30.1rc1.dev396+gac68c3087"
VLLM_EDITS = [
    ("vllm", "models/deepseek_v41/attention.py", (configurable_swa_pages,)),
    (
        "vllm",
        "models/deepseek_v41/nvidia/flashinfer_sparse.py",
        (sm120_swa_pages, sm120_mla_pages),
    ),
    (
        "vllm",
        "v1/attention/backends/mla/indexer.py",
        (sm120_indexer_pages, sm120_v41_mxfp4_indexer),
    ),
    ("vllm", "v1/sample/ops/topk_topp_triton.py", (top_p_guard,)),
    ("vllm", "tokenizers/deepseek_v41.py", (final_assistant_prefill,)),
]
FLASHINFER_EDITS = [
    ("flashinfer", "data/csrc/sparse_mla_sm120_prefill.cu", (flashinfer_page32_prefill,)),
    ("flashinfer", f"{FLASHINFER_SM120}/common/kv_cache_io.cuh", (masked_kv_io,)),
    ("flashinfer", f"{FLASHINFER_SM120}/decode_dsv4_kernel.cuh", (masked_kv_decode,)),
    ("flashinfer", f"{FLASHINFER_SM120}/prefill_common.cuh", (masked_kv_prefill,)),
]


def _package_root(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"package {name} is not installed")
    return Path(spec.origin).parent


def main() -> None:
    import vllm

    if vllm.__version__ != VLLM_VERSION:
        raise SystemExit(f"patches target vLLM {VLLM_VERSION}, found {vllm.__version__}")
    check_efforts(_package_root("vllm") / "tokenizers/deepseek_v41_encoding.py")
    print("V4.1 encoder: low=50, high=75, max=100; default high")

    # Transform everything first; write only when every anchor matched.
    updates: list[tuple[Path, str]] = []
    for package, relative, edits in [*VLLM_EDITS, *FLASHINFER_EDITS]:
        path = _package_root(package) / relative
        source = path.read_text()
        for edit in edits:
            source = edit(source)
        updates.append((path, source))
    for path, source in updates:
        path.write_text(source)
        print(f"patched {path}")


if __name__ == "__main__":
    main()
