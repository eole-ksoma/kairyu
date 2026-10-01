#!/usr/bin/env python3
"""Wire this example's think-first generator into the installed OpenJev, failing closed.

openjev-think.Dockerfile runs this at image build time. Every edit must match
exactly one anchor of the pinned OpenJev source, and every file is transformed
before any is written. A moved anchor therefore fails the build instead of
leaving a half-patched server.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
OVERLAY_MODULES = ("think_core.py", "think_first.py")
IMPORT = "from .chat import Generator, MlxGenerator, add_chat_routes\n"
EDITS = {
    "api.py": (
        (IMPORT, IMPORT + "from .think_first import ThinkFirstGenerator\n"),
        ("else Generator(settings))", "else ThinkFirstGenerator(settings, app.state.engine))"),
    ),
}


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise ValueError(
            f"expected exactly one source anchor, found {source.count(before)}: {before[:120]!r}"
        )
    if after in source:
        raise ValueError(f"source already contains the replacement: {after[:120]!r}")
    return source.replace(before, after, 1)


def installed_package() -> Path:
    spec = importlib.util.find_spec("openjev")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("openjev is not installed")
    return Path(next(iter(spec.submodule_search_locations)))


def package_version(package: Path) -> str:
    source = (package / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"$', source, re.MULTILINE)
    if match is None:
        raise SystemExit("cannot read openjev.__version__")
    return match.group(1)


def patched_sources(package: Path) -> dict[Path, str]:
    sources = {}
    for name, edits in EDITS.items():
        path = package / name
        text = path.read_text(encoding="utf-8")
        for before, after in edits:
            text = replace_once(text, before, after)
        sources[path] = text
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="the pinned OpenJev package version")
    parser.add_argument(
        "--package", type=Path, help="openjev package directory (default: installed)"
    )
    args = parser.parse_args()
    package = args.package or installed_package()
    version = package_version(package)
    if version != args.version:
        raise SystemExit(f"openjev {version} is installed; this overlay targets {args.version}")
    present = [name for name in OVERLAY_MODULES if (package / name).exists()]
    if present:
        raise SystemExit(f"openjev already contains {present}; refusing to patch twice")
    sources = patched_sources(package)  # every transform before any write
    for path, text in sources.items():
        path.write_text(text, encoding="utf-8")
    for name in OVERLAY_MODULES:
        shutil.copyfile(HERE / name, package / name)
    print(f"patched openjev {version} at {package}")


if __name__ == "__main__":
    main()
