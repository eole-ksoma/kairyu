"""Keep locked example measurements bound to the checked-in served config."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCKED_EXAMPLES = (
    "deepseek-v4-flash-vision-exp-dp2-8gpu",
    "qwen3.8-flash-next-dp2-8gpu",
)


def _load_verification(path: Path, name: str):
    module_spec = importlib.util.spec_from_file_location(name, path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("environment", LOCKED_EXAMPLES)
def test_locked_measurement_hash_matches_served_config(environment: str) -> None:
    example = ROOT / "examples" / environment
    verification = _load_verification(
        example / "verification.py",
        f"{environment}_measurement_hash",
    )
    measurements = (example / "MEASUREMENTS.md").read_text(encoding="utf-8")
    match = re.search(
        r"^- Served-config SHA-256:\n\s+`([0-9a-f]{64})`$",
        measurements,
        re.MULTILINE,
    )

    assert match is not None, "MEASUREMENTS.md must record one locked served-config hash"
    assert match.group(1) == verification._served_config_sha256()
