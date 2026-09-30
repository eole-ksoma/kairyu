"""``kairyu`` console entrypoint: serving and offline validation commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kairyu.models.generation import GENERATION_CONFIG_MODES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kairyu", description="Kairyu serving CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser(
        "serve",
        help="Run the OpenAI-compatible server from a DeploymentSpec YAML "
        "(gateway or replica role — the config decides).",
    )
    serve.add_argument("config", type=Path, help="Path to a DeploymentSpec YAML")
    serve.add_argument("--host", default=None, help="Override server.host")
    serve.add_argument("--port", type=int, default=None, help="Override server.port")
    serve.add_argument(
        "--generation-config",
        choices=sorted(GENERATION_CONFIG_MODES),
        default=None,
        help=(
            "Override model generation defaults for every local native backend "
            "(auto, vllm, or none)"
        ),
    )
    validate = subparsers.add_parser(
        "validate",
        help="Validate a DeploymentSpec and its local linked artifacts without starting a server.",
    )
    validate.add_argument("config", type=Path, help="Path to a DeploymentSpec YAML")

    artifact = subparsers.add_parser(
        "artifact",
        help="Validate or admit a signed model-artifact manifest.",
    )
    artifact_commands = artifact.add_subparsers(
        dest="artifact_command",
        required=True,
    )
    artifact_validate = artifact_commands.add_parser(
        "validate",
        help="Verify a manifest digest, signer trust, and Ed25519 signature.",
    )
    artifact_validate.add_argument("manifest", type=Path)
    artifact_validate.add_argument("--trust-store", type=Path, required=True)

    artifact_admit = artifact_commands.add_parser(
        "admit",
        help="Verify and authorize an exact GitOps deployment binding.",
    )
    artifact_admit.add_argument("manifest", type=Path)
    artifact_admit.add_argument("--trust-store", type=Path, required=True)
    artifact_admit.add_argument(
        "--request",
        type=Path,
        required=True,
        help="Version-controlled GitOps model deployment-intent JSON.",
    )

    cache_agent = subparsers.add_parser(
        "cache-agent",
        help="Run the authenticated node-local model cache agent.",
    )
    cache_agent_commands = cache_agent.add_subparsers(
        dest="cache_agent_command",
        required=True,
    )
    cache_agent_serve = cache_agent_commands.add_parser(
        "serve",
        help="Assemble and serve a cache agent from a versioned JSON config.",
    )
    cache_agent_serve.add_argument("config", type=Path)

    placement_admission = subparsers.add_parser(
        "placement-admission",
        help="Run the cache-placement mutating admission webhook.",
    )
    placement_admission_commands = placement_admission.add_subparsers(
        dest="placement_admission_command",
        required=True,
    )
    placement_admission_serve = placement_admission_commands.add_parser(
        "serve",
        help="Assemble and serve the TLS admission webhook from versioned JSON config.",
    )
    placement_admission_serve.add_argument("config", type=Path)

    placement_authority = subparsers.add_parser(
        "placement-authority",
        help="Run the live cache-placement scaling authority.",
    )
    placement_authority_commands = placement_authority.add_subparsers(
        dest="placement_authority_command",
        required=True,
    )
    placement_authority_serve = placement_authority_commands.add_parser(
        "serve",
        help="Assemble and serve the production authority from versioned JSON config.",
    )
    placement_authority_serve.add_argument("config", type=Path)
    return parser


def _safe_cli_text(value: object) -> str:
    rendered = "".join(
        character if character.isprintable() else character.encode("unicode_escape").decode("ascii")
        for character in str(value)
    )
    return rendered if len(rendered) <= 1024 else rendered[:1021] + "..."


def _run_artifact_command(args: argparse.Namespace) -> None:
    from pydantic import ValidationError

    from kairyu.artifacts import (
        InvalidModelArtifactError,
        admit_model_artifact,
        load_model_artifact_admission_request,
        load_model_artifact_trust_store,
        load_signed_model_artifact,
        verify_model_artifact_manifest,
    )

    try:
        envelope = load_signed_model_artifact(args.manifest)
        trust_store = load_model_artifact_trust_store(args.trust_store)
        if args.artifact_command == "validate":
            verified = verify_model_artifact_manifest(envelope, trust_store)
            print(
                f"VALID manifest={verified.manifest_digest} "
                f"signer={_safe_cli_text(verified.signer_key_id)}"
            )
            return
        request = load_model_artifact_admission_request(args.request)
        admission = admit_model_artifact(envelope, trust_store, request)
        print(
            f"ADMITTED manifest={admission.manifest_digest} "
            f"deployment={_safe_cli_text(admission.deployment_id)} "
            f"model={_safe_cli_text(admission.model_id)} "
            f"revision={_safe_cli_text(admission.model_revision)} "
            f"environment={_safe_cli_text(admission.environment)} "
            f"gpu_profile={_safe_cli_text(admission.gpu_profile)} "
            f"signer={_safe_cli_text(admission.signer_key_id)}"
        )
    except (InvalidModelArtifactError, ValidationError) as exc:
        if isinstance(exc, ValidationError):
            detail = exc.errors(include_input=False, include_url=False)[0]
            location = ".".join(str(part) for part in detail.get("loc", ()))
            message = f"admission schema validation failed at {location or '<root>'}"
        else:
            message = str(exc)
        print(f"INVALID artifact: {_safe_cli_text(message)}")
        raise SystemExit(1) from None


def _run_cache_agent(args: argparse.Namespace) -> None:
    import uvicorn

    from kairyu.entrypoints.server.middleware import configure_json_logging
    from kairyu.runners.cache_agent_runtime import (
        build_node_model_cache_agent_runtime,
        load_node_model_cache_agent_runtime_config,
    )

    configure_json_logging()
    config = load_node_model_cache_agent_runtime_config(args.config)
    runtime = build_node_model_cache_agent_runtime(config)
    try:
        uvicorn.run(
            runtime.app,
            host=config.listen_host,
            port=config.listen_port,
            loop="uvloop" if sys.platform == "linux" else "auto",
            http="httptools" if sys.platform == "linux" else "auto",
            log_config=None,
            access_log=False,
        )
    finally:
        runtime.close()


def _run_placement_admission(args: argparse.Namespace) -> None:
    import uvicorn

    from kairyu.entrypoints.server.middleware import configure_json_logging
    from kairyu.runners.startup_admission_runtime import (
        build_runner_cache_placement_admission_runtime,
        load_runner_cache_placement_admission_runtime_config,
    )

    configure_json_logging()
    config = load_runner_cache_placement_admission_runtime_config(args.config)
    runtime = build_runner_cache_placement_admission_runtime(config)
    try:
        uvicorn.run(
            runtime.app,
            host=config.listen_host,
            port=config.listen_port,
            loop="uvloop" if sys.platform == "linux" else "auto",
            http="httptools" if sys.platform == "linux" else "auto",
            log_config=None,
            access_log=False,
            workers=1,
            ssl_certfile=str(config.tls_cert_file),
            ssl_keyfile=str(config.tls_key_file),
        )
    finally:
        runtime.close()


def _run_placement_authority(args: argparse.Namespace) -> None:
    import uvicorn

    from kairyu.entrypoints.server.middleware import configure_json_logging
    from kairyu.runners.startup_binding_authority_production import (
        build_runner_cache_placement_binding_production_runtime,
        load_runner_cache_placement_binding_production_runtime_config,
    )

    configure_json_logging()
    config = load_runner_cache_placement_binding_production_runtime_config(args.config)
    runtime = build_runner_cache_placement_binding_production_runtime(config)
    server = config.authority
    try:
        uvicorn.run(
            runtime.app,
            host=server.listen_host,
            port=server.listen_port,
            loop="uvloop" if sys.platform == "linux" else "auto",
            http="httptools" if sys.platform == "linux" else "auto",
            log_config=None,
            access_log=False,
            workers=1,
            ssl_certfile=str(server.tls_cert_file),
            ssl_keyfile=str(server.tls_key_file),
        )
    finally:
        runtime.close()


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "serve":
        import uvicorn

        from kairyu.deploy.builder import build_app_from_spec
        from kairyu.deploy.spec import load_deployment_spec
        from kairyu.entrypoints.server.middleware import configure_json_logging

        configure_json_logging()
        spec = load_deployment_spec(args.config)
        app = build_app_from_spec(
            spec,
            base_dir=args.config.parent,
            generation_config_override=args.generation_config,
        )
        uvicorn.run(
            app,
            host=args.host or spec.server.host,
            port=args.port or spec.server.port,
            # The production Linux dependency set installs both packages.
            # Select them explicitly so a missing/broken image fails at
            # startup instead of silently benchmarking asyncio + h11.
            loop="uvloop" if sys.platform == "linux" else "auto",
            http="httptools" if sys.platform == "linux" else "auto",
            log_config=None,  # keep the JSON root logger
            # AccessLogMiddleware is the single structured access-log owner.
            # Leaving Uvicorn's logger enabled duplicates every request and
            # makes ServerSettings.access_log=False ineffective.
            access_log=False,
        )
    elif args.command == "validate":
        from kairyu.deploy.validation import validate_deployment

        report = validate_deployment(args.config)
        print(report.render_text())
        if not report.valid:
            sys.exit(1)
    elif args.command == "artifact":
        _run_artifact_command(args)
    elif args.command == "cache-agent":
        _run_cache_agent(args)
    elif args.command == "placement-admission":
        _run_placement_admission(args)
    elif args.command == "placement-authority":
        _run_placement_authority(args)


if __name__ == "__main__":
    main()
