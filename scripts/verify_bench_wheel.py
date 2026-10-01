#!/usr/bin/env python3
"""Build a wheel and verify checkout-only tooling is excluded."""

from __future__ import annotations

import argparse
import configparser
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import zipfile
from pathlib import Path, PurePosixPath

CONSOLE_TARGET = "kairyu.entrypoints.cli:main"
FORBIDDEN_PREFIXES = (
    "bench/",
    "evals/",
    "evidence/",
    "verification/",
    "tests/",
    "kairyu/bench/",
)


class VerificationError(RuntimeError):
    """The built wheel does not preserve the declared package boundary."""


def _build_wheel(repo: Path, output_dir: Path, uv: str) -> Path:
    subprocess.run(
        [
            uv,
            "build",
            "--wheel",
            "--out-dir",
            str(output_dir),
            "--no-create-gitignore",
            str(repo),
        ],
        check=True,
    )
    wheels = sorted(output_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise VerificationError(
            f"expected one wheel in {output_dir}, found {[path.name for path in wheels]}"
        )
    return wheels[0]


def _inspect_wheel(wheel: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(wheel) as archive:
        names = tuple(sorted(archive.namelist()))
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise VerificationError(f"unsafe wheel member: {name}")
        if "kairyu/__init__.py" not in names:
            raise VerificationError("wheel omits the kairyu package")
        forbidden = tuple(
            name for name in names if name.startswith(FORBIDDEN_PREFIXES)
        )
        if forbidden:
            raise VerificationError(
                f"wheel contains checkout-only files: {list(forbidden)}"
            )

        metadata_files = [
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        ]
        if len(metadata_files) != 1:
            raise VerificationError(
                f"expected one entry_points.txt, found {metadata_files}"
            )
        parser = configparser.ConfigParser()
        parser.read_string(archive.read(metadata_files[0]).decode("utf-8"))
        try:
            target = parser["console_scripts"]["kairyu"]
        except KeyError as error:
            raise VerificationError("wheel omits the kairyu console script") from error
        if target != CONSOLE_TARGET:
            raise VerificationError(
                f"kairyu console target is {target!r}, expected {CONSOLE_TARGET!r}"
            )
    return names


def _isolated_run(
    extracted: Path,
    dependency_site: Path,
    code: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSAFEPATH"):
        env.pop(name, None)
    bootstrap = f"import sys; sys.path.append({str(dependency_site)!r}); {code}"
    return subprocess.run(
        [sys.executable, "-S", "-c", bootstrap],
        cwd=extracted,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _verify_isolated_runtime(wheel: Path, scratch: Path) -> None:
    extracted = scratch / "site"
    extracted.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(extracted)

    dependency_site = Path(sysconfig.get_paths()["purelib"]).resolve()
    code = textwrap.dedent(
        """
        import argparse
        import importlib
        import importlib.util
        from pathlib import Path

        import kairyu
        from kairyu.entrypoints.cli import _build_parser

        def command_tree(parser):
            actions = [
                action
                for action in parser._actions
                if isinstance(action, argparse._SubParsersAction)
            ]
            assert len(actions) <= 1, actions
            if not actions:
                return {}
            return {
                name: command_tree(child)
                for name, child in actions[0].choices.items()
            }

        root = Path.cwd().resolve()
        module = Path(kairyu.__file__).resolve()
        assert module.is_relative_to(root), (module, root)
        parser = _build_parser()
        expected_tree = {
            "artifact": {"admit": {}, "validate": {}},
            "cache-agent": {"serve": {}},
            "placement-admission": {"serve": {}},
            "placement-authority": {"serve": {}},
            "serve": {},
            "validate": {},
        }
        actual_tree = command_tree(parser)
        assert actual_tree == expected_tree, actual_tree
        assert "bench" not in parser.format_help().lower()
        command_modules = (
            "kairyu.artifacts",
            "kairyu.deploy.builder",
            "kairyu.deploy.validation",
            "kairyu.runners.cache_agent_runtime",
            "kairyu.runners.startup_admission_runtime",
            "kairyu.runners.startup_binding_authority_production",
        )
        imported = [importlib.import_module(name) for name in command_modules]
        imported_paths = [Path(item.__file__).resolve() for item in imported]
        assert all(path.is_relative_to(root) for path in imported_paths), imported_paths
        assert importlib.util.find_spec("evals") is None
        assert importlib.util.find_spec("evidence") is None
        assert importlib.util.find_spec("verification") is None
        print(module)
        """
    )
    result = _isolated_run(extracted, dependency_site, code)
    module_path = Path(result.stdout.strip()).resolve()
    if not module_path.is_relative_to(extracted.resolve()):
        raise VerificationError(
            f"isolated check imported outside the wheel: {module_path}"
        )


def verify(repo: Path, *, uv: str, wheel: Path | None = None) -> Path:
    repo = repo.resolve()
    with tempfile.TemporaryDirectory(prefix="kairyu-wheel-") as temp:
        scratch = Path(temp)
        built_wheel = (
            wheel.resolve()
            if wheel is not None
            else _build_wheel(repo, scratch / "dist", uv)
        )
        _inspect_wheel(built_wheel)
        _verify_isolated_runtime(built_wheel, scratch)
        print(
            f"verified {built_wheel.name}: product CLI only; "
            "evals/evidence/verification/bench/tests excluded"
        )
        return built_wheel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        default=None,
        help="verify an existing wheel instead of building one",
    )
    parser.add_argument(
        "--uv",
        default=shutil.which("uv") or "uv",
        help="uv executable used to build the wheel",
    )
    args = parser.parse_args(argv)
    verify(args.repo, uv=args.uv, wheel=args.wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
