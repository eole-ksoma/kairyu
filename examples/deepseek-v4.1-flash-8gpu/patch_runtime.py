"""Align the pinned V4.1 runtime with official efforts and SM120 page sizes.

The selected runtime's Python frontend is used so a separate Rust renderer
cannot bypass this change. No Kairyu L2/L3 effort policy is modified.
"""

from __future__ import annotations

import ast
from pathlib import Path

OFFICIAL_EFFORTS = {"low": 50, "high": 75, "max": 100}
INITIAL_VLLM_EFFORTS = {"low": 25, "high": 50, "xhigh": 75, "max": 100}


def replace_once(source: str, before: str, after: str) -> str:
    """Accept exactly the pinned source or the already patched source."""
    if source.count(after) == 1 and before not in source.replace(after, "", 1):
        return source
    if source.count(before) != 1 or after in source:
        raise ValueError(f"Pinned runtime source drift: {before!r}")
    return source.replace(before, after, 1)


def align_swa_pages(source: str, *, sm120: bool = False) -> str:
    if sm120:
        declaration = (
            "class DeepseekV4FlashInferSM120Attention(DeepseekV4Attention):\n"
            '    """DeepSeek V4 sparse MLA attention through FlashInfer\'s SM120 kernels."""\n'
            "\n"
            "    backend_cls = DeepseekV4FlashInferMLASparseBackend\n"
            "    swa_backend_cls = DeepseekSparseSWAFlashInferBackend\n"
        )
        return replace_once(
            source,
            declaration,
            declaration + "    swa_block_size: ClassVar[int] = 64\n",
        )
    return replace_once(
        source,
        "            block_size=32,\n",
        "            block_size=getattr(self, 'swa_block_size', 32),\n",
    )


def enable_compressed_page32(source: str) -> str:
    """Instantiate existing generic dual-cache kernels for V4.1 C2 pages."""
    edits = (
        (
            "(extra_page_block_size == 64 || extra_page_block_size == 2)",
            "(extra_page_block_size == 64 || extra_page_block_size == 32 || "
            "extra_page_block_size == 2)",
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


def align_indexer_pages(source: str) -> str:
    declaration = """class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):"""
    backend = """class DeepseekV41SM120IndexerBackend(DeepseekV4IndexerBackend):
    @staticmethod
    def get_supported_kernel_block_sizes():
        return [64]


"""
    source = replace_once(source, declaration, backend + declaration)
    return replace_once(
        source,
        "        return DeepseekV4IndexerBackend\n",
        "        from vllm.platforms import current_platform\n"
        "        if current_platform.is_device_capability_family(120):\n"
        "            return DeepseekV41SM120IndexerBackend\n"
        "        return DeepseekV4IndexerBackend\n",
    )


def align_mla_pages(source: str) -> str:
    return replace_once(
        source,
        "        return [128]\n",
        "        from vllm.platforms import current_platform\n"
        "        return [64 if current_platform.is_device_capability_family(120) else 128]\n",
    )


def enable_sm120_v41_indexer(source: str) -> str:
    return replace_once(
        source,
        "    if use_fp4 and not current_platform.is_device_capability_family(100):\n",
        "    sm120_v41 = (\n"
        "        current_platform.is_device_capability_family(120)\n"
        "        and vllm_config.model_config.hf_config.model_type == 'deepseek_v41'\n"
        "    )\n"
        "    if use_fp4 and not ("
        "current_platform.is_device_capability_family(100) or sm120_v41):\n",
    )


def align_efforts(source: str) -> str:
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "REASONING_EFFORT_MAPPINGS"
    ]
    if len(matches) != 1:
        raise ValueError("Expected one pinned V4.1 effort mapping")
    node = matches[0]
    current = ast.literal_eval(node.value)
    if current not in (INITIAL_VLLM_EFFORTS, OFFICIAL_EFFORTS):
        raise ValueError(f"Unrecognized upstream effort mapping: {current!r}")
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1 : node.end_lineno] = [
        f"REASONING_EFFORT_MAPPINGS: Dict[str, int] = {OFFICIAL_EFFORTS!r}\n"
    ]
    return "".join(lines)


def main() -> None:
    import importlib.util
    import runpy

    module = importlib.util.find_spec("vllm.tokenizers.deepseek_v41_encoding")
    if module is None or module.origin is None:
        raise RuntimeError("The pinned image has no V4.1 Python encoder")
    path = Path(module.origin)
    path.write_text(align_efforts(path.read_text()))
    encoder = runpy.run_path(str(path))
    render = encoder["render_reasoning_effort"]
    for name, budget in OFFICIAL_EFFORTS.items():
        assert f"Reasoning Effort: {budget} " in render(0, "thinking", name)
    assert "Reasoning Effort: 75 " in render(0, "thinking", None)
    assert render(0, "chat", None) == ""
    print("V4.1 encoder verified: low=50, high=75, max=100; default=high")

    vllm = importlib.util.find_spec("vllm")
    flashinfer = importlib.util.find_spec("flashinfer")
    assert vllm and vllm.origin and flashinfer and flashinfer.origin
    model = Path(vllm.origin).parent / "models/deepseek_v4_1"
    attention = model / "attention.py"
    attention.write_text(align_indexer_pages(align_swa_pages(attention.read_text())))
    sm120 = model / "nvidia/flashinfer_sparse.py"
    sm120.write_text(align_mla_pages(align_swa_pages(sm120.read_text(), sm120=True)))
    indexer = Path(vllm.origin).parent / "v1/attention/backends/mla/indexer.py"
    indexer.write_text(enable_sm120_v41_indexer(indexer.read_text()))
    prefill = Path(flashinfer.origin).parent / "data/csrc/sparse_mla_sm120_prefill.cu"
    prefill.write_text(enable_compressed_page32(prefill.read_text()))
    print("SM120 pages: SWA=64, compressed C1=64/C2=32 (manager block=64)")
    print("V4.1 SM120 MXFP4 indexer enabled (real writer/prefill/decode numerical gate)")


if __name__ == "__main__":
    main()
