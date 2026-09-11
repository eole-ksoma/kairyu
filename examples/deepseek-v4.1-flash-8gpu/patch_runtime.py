"""Align the pinned vLLM Python encoder with DeepSeek's published effort aliases.

The selected runtime's Python frontend is used so a separate Rust renderer
cannot bypass this change. No Kairyu L2/L3 effort policy is modified.
"""

from __future__ import annotations

import ast
from pathlib import Path

OFFICIAL_EFFORTS = {"low": 50, "high": 75, "max": 100}
INITIAL_VLLM_EFFORTS = {"low": 25, "high": 50, "xhigh": 75, "max": 100}


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


if __name__ == "__main__":
    main()
