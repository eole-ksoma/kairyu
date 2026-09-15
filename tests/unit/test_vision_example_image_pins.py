"""Fail closed when a vision example overrides its pinned vLLM image tag."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = (
    (
        "deepseek-v4-flash-vision-exp-dp2-8gpu",
        "DEEPSEEK_VLLM_IMAGE",
    ),
    (
        "qwen3.8-flash-next-dp2-8gpu",
        "QWEN_VLLM_IMAGE",
    ),
)


def _load_control(path: Path, name: str):
    module_spec = importlib.util.spec_from_file_location(name, path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("environment", "env_key"), EXAMPLES)
def test_vllm_image_override_must_match_pinned_image_id(
    environment: str,
    env_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load_control(
        ROOT / "examples" / environment / "control.py",
        f"{environment}_image_pin",
    )
    expected = control.SPEC["vllm"]["image_id"]
    override = "local/operator-override:latest"

    monkeypatch.setattr(control, "_image_id", lambda _image: "sha256:unexpected")
    with pytest.raises(SystemExit, match="example.json and kairyu.yaml pin"):
        control._ensure_vllm_image({env_key: override})

    monkeypatch.setattr(control, "_image_id", lambda _image: expected)
    control._ensure_vllm_image({env_key: override})
