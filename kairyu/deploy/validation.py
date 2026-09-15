"""Deterministic offline validation for DeploymentSpec artifact graphs.

The validator deliberately stops before runtime ownership boundaries: it does
not resolve credentials, construct backends, load tensors, contact endpoints,
or inspect hardware.  It validates only the schema and local input artifacts
that the current DeploymentSpec can actually declare.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import ValidationError

from kairyu.deploy import builder as deployment_builder
from kairyu.deploy.spec import (
    DeploymentSpec,
    _DuplicateKeyError,
    _UniqueKeySafeLoader,
    load_deployment_spec,
)
from kairyu.dsl.loader import (
    _role_executor as _loader_role_executor,
)
from kairyu.dsl.loader import (
    _role_sampling as _loader_role_sampling,
)
from kairyu.dsl.loader import load_spec
from kairyu.dsl.spec import OrchestratorSpec, RoleNodeSpec
from kairyu.engine.config_validation import validate_backend_options
from kairyu.engine.core.quant_config import (
    QuantMethod,
    load_checkpoint_quantization,
    validate_model_quantization,
    validate_tensor_parallel_quantization,
)
from kairyu.engine.core.weights import CheckpointReader
from kairyu.engine.registry import (
    known_backends,
    uses_builtin_backend_contract,
)
from kairyu.engine.tokenizer import resolve_tokenizer
from kairyu.entrypoints.chat_template import ChatTemplate
from kairyu.models.config import (
    parse_model_config,
    validate_tensor_parallel_config,
)
from kairyu.models.generation import parse_generation_defaults
from kairyu.orchestration.conductor import Conductor, RoleSpec

CheckClass = Literal["schema", "filesystem"]
CheckStatus = Literal["passed", "failed", "skipped", "indeterminate", "not_run"]

_NATIVE_BACKENDS = frozenset({"kairyu", "kairyu-proc"})
_SENSITIVE_FIELD_PARTS = (
    "api_key",
    "key_tenants",
    "password",
    "secret",
    "token",
    "dsn",
)


@dataclass(frozen=True, order=True)
class ValidationFinding:
    """One privacy-safe validation error."""

    artifact: str
    field: str
    check: CheckClass
    code: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Stable result returned by :func:`validate_deployment`."""

    config: str
    findings: tuple[ValidationFinding, ...]
    checks: Mapping[str, CheckStatus]

    @property
    def valid(self) -> bool:
        return not self.findings

    def render_text(self) -> str:
        """Render stable human-readable output without source values."""

        state = "VALID" if self.valid else "INVALID"
        lines = [f"{state} {_display_text(self.config)}"]
        for finding in self.findings:
            lines.append(
                f"ERROR [{finding.check}] {_display_text(finding.artifact)} "
                f"{_display_text(finding.field)} ({finding.code}): "
                f"{_display_text(finding.message)}"
            )
        lines.append(
            "checks: "
            + " ".join(
                f"{name}={self.checks[name]}"
                for name in ("schema", "filesystem", "network", "hardware")
            )
        )
        return "\n".join(lines)


def _display_text(value: object) -> str:
    return "".join(
        character
        if character.isprintable()
        else character.encode("unicode_escape").decode("ascii")
        for character in str(value)
    )


def _display_path(path: Path) -> str:
    return _display_text(path)


def _canonical(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return path.absolute()


def _field_path(parts: Iterable[object]) -> str:
    parts = tuple(parts)
    rendered = ""
    for index, part in enumerate(parts):
        if index > 0 and parts[index - 1] == "key_tenants":
            part = "<redacted>"
        if isinstance(part, int):
            rendered += f"[{part}]"
        elif rendered:
            rendered += f".{part}"
        else:
            rendered = str(part)
    return rendered or "<root>"


def _is_sensitive_field(field: str) -> bool:
    lowered = field.lower()
    return any(part in lowered for part in _SENSITIVE_FIELD_PARTS)


def _safe_pydantic_message(detail: Mapping[str, object], field: str) -> str:
    error_type = str(detail.get("type", "value_error"))
    if _is_sensitive_field(field):
        if error_type == "extra_forbidden":
            return "extra field is not permitted"
        if error_type == "missing":
            return "required field is missing"
        return "invalid sensitive configuration"
    message = str(detail.get("msg", "invalid value")).removeprefix(
        "Value error, "
    )
    if error_type == "value_error":
        # Preserve one useful, finite capability diagnostic without forwarding
        # arbitrary source values from custom validator exception strings.
        capability_prefix = (
            "vendor extra_args may not override backend-owned fields: "
        )
        if message.startswith(capability_prefix):
            fields = tuple(
                item.strip()
                for item in message.removeprefix(capability_prefix).split(",")
            )
            safe_fields = frozenset(
                {
                    "messages",
                    "model",
                    "stream",
                    "stream_options",
                    "tool_choice",
                    "tools",
                    "top_logprobs",
                    "max_completion_tokens",
                    "prompt_logprobs",
                    "priority",
                }
            )
            if fields and all(item in safe_fields for item in fields):
                return capability_prefix + ", ".join(fields)
        if "capabilit" in message.lower() or "upstream" in message.lower():
            return "OpenAI capability configuration is invalid"
        return "configuration value is invalid"
    return message


def _pydantic_findings(
    error: ValidationError,
    *,
    artifact: Path,
    prefix: str = "",
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    for detail in error.errors(include_input=False, include_url=False):
        local_field = _field_path(detail.get("loc", ()))
        field = (
            f"{prefix}.{local_field}"
            if prefix and local_field != "<root>"
            else (prefix or local_field)
        )
        findings.append(
            ValidationFinding(
                artifact=_display_path(artifact),
                field=field,
                check="schema",
                code=f"schema.{detail.get('type', 'invalid')}",
                message=_safe_pydantic_message(detail, field),
            )
        )
    return findings


def _yaml_finding(error: Exception, artifact: Path) -> ValidationFinding:
    if isinstance(error, yaml.MarkedYAMLError) and error.problem_mark is not None:
        mark = error.problem_mark
        message = f"invalid YAML at line {mark.line + 1}, column {mark.column + 1}"
        field = "<root>"
        code = "schema.invalid_yaml"
    elif isinstance(error, _DuplicateKeyError):
        field = _field_path(error.path)
        message = "duplicate mapping key"
        code = "schema.duplicate_key"
    elif "duplicate mapping key" in str(error):
        match = re.search(r" at ([A-Za-z0-9_.<>\[\]-]+)$", str(error))
        field = (
            _field_path(match.group(1).split("."))
            if match
            else "<root>"
        )
        message = "duplicate mapping key"
        code = "schema.duplicate_key"
    else:
        field = "<root>"
        message = "could not parse YAML"
        code = "schema.invalid_yaml"
    return ValidationFinding(
        artifact=_display_path(artifact),
        field=field,
        check="schema",
        code=code,
        message=message,
    )


def _mapping(value: object) -> Mapping[object, object]:
    return value if isinstance(value, Mapping) else {}


def _iter_deployment_backends(
    raw: Mapping[object, object],
) -> Iterable[tuple[str, str, Mapping[object, object]]]:
    for name, entry in _mapping(raw.get("engines")).items():
        data = _mapping(entry)
        backend = data.get("backend")
        if isinstance(backend, str):
            yield (
                f"engines.{name}",
                backend,
                _mapping(data.get("options")),
            )
    for pool_name, pool in _mapping(raw.get("pools")).items():
        replicas = _mapping(pool).get("replicas")
        if not isinstance(replicas, list):
            continue
        for index, entry in enumerate(replicas):
            data = _mapping(entry)
            backend = data.get("backend")
            if isinstance(backend, str):
                yield (
                    f"pools.{pool_name}.replicas[{index}]",
                    backend,
                    _mapping(data.get("options")),
                )


def _iter_orchestrator_references(
    raw: Mapping[object, object],
) -> Iterable[tuple[str, str]]:
    legacy = _mapping(raw.get("orchestrator")).get("spec")
    if isinstance(legacy, str):
        yield "orchestrator.spec", legacy
    for name, section in _mapping(raw.get("orchestrators")).items():
        spec = _mapping(section).get("spec")
        if isinstance(spec, str):
            yield f"orchestrators.{name}.spec", spec


def _finding(
    *,
    artifact: Path,
    field: str,
    check: CheckClass,
    code: str,
    message: str,
) -> ValidationFinding:
    return ValidationFinding(
        artifact=_display_path(artifact),
        field=field,
        check=check,
        code=code,
        message=message,
    )


def _read_json_mapping(
    path: Path,
    *,
    field: str,
    required: bool,
) -> tuple[dict | None, list[ValidationFinding]]:
    if not path.is_file():
        if not required:
            return None, []
        return None, [
            _finding(
                artifact=path,
                field=field,
                check="filesystem",
                code="filesystem.missing_file",
                message="required file does not exist",
            )
        ]
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError):
        return None, [
            _finding(
                artifact=path,
                field=field,
                check="filesystem",
                code="filesystem.unreadable",
                message="file is not readable as UTF-8",
            )
        ]
    except json.JSONDecodeError as error:
        return None, [
            _finding(
                artifact=path,
                field=field,
                check="schema",
                code="schema.invalid_json",
                message=f"invalid JSON at line {error.lineno}, column {error.colno}",
            )
        ]
    except RecursionError:
        return None, [
            _finding(
                artifact=path,
                field=field,
                check="schema",
                code="schema.invalid_json",
                message="JSON nesting is too deep",
            )
        ]
    if not isinstance(loaded, dict):
        return None, [
            _finding(
                artifact=path,
                field=field,
                check="schema",
                code="schema.expected_mapping",
                message="JSON document must be an object",
            )
        ]
    return loaded, []


def _validate_tokenizer(
    tokenizer: object,
    *,
    field: str,
    reference_artifact: Path,
) -> tuple[list[ValidationFinding], int | None]:
    if tokenizer == "toy":
        return [], 50_000
    if not isinstance(tokenizer, str):
        return (
            [
                _finding(
                    artifact=reference_artifact,
                    field=field,
                    check="schema",
                    code="schema.invalid_tokenizer_reference",
                    message="tokenizer must be 'toy' or a local path",
                )
            ],
            None,
        )
    path = Path(tokenizer)
    tokenizer_json = path / "tokenizer.json" if path.is_dir() else path
    _data, findings = _read_json_mapping(
        tokenizer_json,
        field=field,
        required=True,
    )
    config_path = tokenizer_json.parent / "tokenizer_config.json"
    _config, config_findings = _read_json_mapping(
        config_path,
        field=field,
        required=False,
    )
    findings.extend(config_findings)
    vocab_size = None
    if not findings:
        try:
            vocab_size = len(resolve_tokenizer(tokenizer).vocab())
        except Exception:
            findings.append(
                _finding(
                    artifact=tokenizer_json,
                    field=field,
                    check="schema",
                    code="schema.invalid_tokenizer",
                    message="tokenizer artifact is incompatible",
                )
            )
    return findings, vocab_size


def _validate_checkpoint_index(
    model_dir: Path,
    *,
    field: str,
    required_names: Iterable[str] | None,
) -> list[ValidationFinding]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        if any(model_dir.glob("*.safetensors")):
            return []
        return [
            _finding(
                artifact=model_dir,
                field=field,
                check="filesystem",
                code="filesystem.missing_weights",
                message="no safetensors checkpoint files exist",
            )
        ]
    index, findings = _read_json_mapping(index_path, field=field, required=True)
    if index is None:
        return findings
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        findings.append(
            _finding(
                artifact=index_path,
                field=field,
                check="schema",
                code="schema.invalid_weight_map",
                message="safetensors index must contain a non-empty weight_map",
            )
        )
        return findings
    invalid_shards = [
        shard
        for shard in weight_map.values()
        if not isinstance(shard, str) or not shard
    ]
    if required_names is None:
        checked_shards = weight_map.values()
    else:
        checked_shards = (
            weight_map[name]
            for name in required_names
            if name in weight_map
        )
    shard_names = {
        shard
        for shard in checked_shards
        if isinstance(shard, str) and shard
    }
    if invalid_shards:
        findings.append(
            _finding(
                artifact=index_path,
                field=field,
                check="schema",
                code="schema.invalid_shard_reference",
                message="weight_map shard references must be non-empty strings",
            )
        )
    for shard in sorted(shard_names):
        shard_path = model_dir / shard
        if not shard_path.is_file():
            findings.append(
                _finding(
                    artifact=shard_path,
                    field=field,
                    check="filesystem",
                    code="filesystem.missing_shard",
                    message="referenced checkpoint shard does not exist",
                )
            )
    return findings


def _expected_checkpoint_contract(
    model_config,
    quant_config,
) -> tuple[dict[str, tuple[int, ...]], dict[str, int | None]]:
    """Build meta tensors to reuse the runtime model's exact state contract."""

    import torch

    from kairyu.models.loader import build_model
    from kairyu.models.parallel import checkpoint_member_specs
    from kairyu.quant.linear import linear_factory

    with torch.device("meta"):
        model = build_model(
            model_config,
            linear_factory=linear_factory(quant_config),
        )
    shapes = {
        name: tuple(tensor.shape)
        for name, tensor in model.state_dict().items()
        if not (
            name == "lm_head.weight"
            and model_config.tie_word_embeddings
        )
    }
    specs = checkpoint_member_specs(model)
    return shapes, {
        name: specs[name].shard_dim
        for name in shapes
    }


def _checkpoint_shapes_are_compatible(
    actual: Mapping[str, tuple[int, ...]],
    expected: Mapping[str, tuple[int, ...]],
    *,
    quant_config,
    checkpoint_shard_dims: Mapping[str, int | None],
    tensor_parallel_size: object,
) -> bool:
    """Apply the same FP8 scale-shape contract as the runtime loaders."""

    sharded = (
        isinstance(tensor_parallel_size, int)
        and not isinstance(tensor_parallel_size, bool)
        and tensor_parallel_size > 1
    )
    for name, expected_shape in expected.items():
        actual_shape = actual.get(name)
        if actual_shape == expected_shape:
            continue
        if not (
            quant_config.method is QuantMethod.FP8
            and name.endswith(".weight_scale")
            and actual_shape == (1,)
        ):
            return False
        if sharded and checkpoint_shard_dims[name] is not None:
            return False
    return True


def _validate_model(
    model_path: object,
    *,
    field: str,
    tokenizer: object | None,
    tensor_parallel_size: object,
    reference_artifact: Path,
    generation_config: object = "auto",
) -> list[ValidationFinding]:
    if not isinstance(model_path, str):
        return [
            _finding(
                artifact=reference_artifact,
                field=field,
                check="schema",
                code="schema.invalid_model_reference",
                message="model_path must be a local path string",
            )
        ]
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        return [
            _finding(
                artifact=model_dir,
                field=field,
                check="filesystem",
                code="filesystem.missing_directory",
                message="model directory does not exist",
            )
        ]

    config, findings = _read_json_mapping(
        model_dir / "config.json",
        field=field,
        required=True,
    )
    model_config = None
    quant_config = None
    expected_checkpoint_shapes: dict[str, tuple[int, ...]] | None = None
    expected_checkpoint_shard_dims: dict[str, int | None] | None = None
    if config is not None:
        try:
            model_config = parse_model_config(config)
            from kairyu.models.loader import (
                declared_quantization_config,
                reference_quant_loader_kind,
            )
            declared_quant = declared_quantization_config(config)
            quant_source = (
                {**config, "quantization_config": declared_quant}
                if declared_quant is not None
                else config
            )
            reference_loader = reference_quant_loader_kind(
                config, model_config.architecture
            )
            if reference_loader is None:
                quant_config = load_checkpoint_quantization(
                    model_dir, quant_source
                ).weights
            official_frontier_decoder = (
                model_config.requires_full_recompute
                or model_config.uses_stateful_frontier_cache
            )
            if not official_frontier_decoder:
                validate_model_quantization(
                    quant_config,
                    is_mla=model_config.is_mla,
                    architecture=model_config.architecture,
                )
                (
                    expected_checkpoint_shapes,
                    expected_checkpoint_shard_dims,
                ) = _expected_checkpoint_contract(
                    model_config,
                    quant_config,
                )
            elif reference_loader is None and quant_config.method is not QuantMethod.NONE:
                raise ValueError(
                    "reference execution requires an unquantized checkpoint"
                )
        except Exception:
            findings.append(
                _finding(
                    artifact=model_dir / "config.json",
                    field=field,
                    check="schema",
                    code="schema.incompatible_model",
                    message="model metadata is incompatible",
                )
            )
    generation_findings: list[ValidationFinding] = []
    if generation_config != "none":
        _generation, generation_findings = _read_json_mapping(
            model_dir / "generation_config.json",
            field=field,
            required=False,
        )
        findings.extend(generation_findings)
    if config is not None and not generation_findings:
        try:
            parse_generation_defaults(
                model_dir,
                config,
                generation_config,
            )
        except Exception:
            generation_path = model_dir / "generation_config.json"
            findings.append(
                _finding(
                    artifact=(
                        generation_path
                        if generation_config != "none"
                        and generation_path.is_file()
                        else model_dir / "config.json"
                    ),
                    field=field,
                    check="schema",
                    code="schema.invalid_generation_config",
                    message="generation metadata is incompatible",
                )
            )

    checkpoint_findings = _validate_checkpoint_index(
        model_dir,
        field=field,
        required_names=(
            expected_checkpoint_shapes
            if expected_checkpoint_shapes is not None
            else None
        ),
    )
    findings.extend(checkpoint_findings)
    if not checkpoint_findings:
        try:
            reader = CheckpointReader(model_dir)
            if expected_checkpoint_shapes is None:
                names = reader.names()
                if not names:
                    raise ValueError("checkpoint contains no tensors")
                reader.shapes(names)
            else:
                actual_shapes = reader.shapes(expected_checkpoint_shapes)
                if not _checkpoint_shapes_are_compatible(
                    actual_shapes,
                    expected_checkpoint_shapes,
                    quant_config=quant_config,
                    checkpoint_shard_dims=expected_checkpoint_shard_dims,
                    tensor_parallel_size=tensor_parallel_size,
                ):
                    raise ValueError("checkpoint tensor shapes are incompatible")
        except Exception:
            findings.append(
                _finding(
                    artifact=model_dir,
                    field=field,
                    check="schema",
                    code="schema.invalid_checkpoint",
                    message="checkpoint metadata is incompatible",
                )
            )

    tokenizer_findings, tokenizer_vocab_size = _validate_tokenizer(
        model_path if tokenizer is None else tokenizer,
        field=f"{field.rsplit('.', 1)[0]}.tokenizer",
        reference_artifact=reference_artifact,
    )
    findings.extend(tokenizer_findings)
    if (
        model_config is not None
        and tokenizer_vocab_size is not None
        and tokenizer_vocab_size > model_config.vocab_size
    ):
        findings.append(
            _finding(
                artifact=model_dir,
                field=f"{field.rsplit('.', 1)[0]}.tokenizer",
                check="schema",
                code="schema.incompatible_tokenizer_vocab",
                message="tokenizer vocabulary exceeds the model vocabulary",
            )
        )
    if (
        model_config is not None
        and isinstance(tensor_parallel_size, int)
        and not isinstance(tensor_parallel_size, bool)
    ):
        try:
            if (
                model_config.requires_full_recompute
                and tensor_parallel_size != 1
            ):
                raise ValueError(
                    "hybrid reference execution is single-device only"
                )
            validate_tensor_parallel_config(
                model_config,
                tensor_parallel_size,
            )
            if tensor_parallel_size > 1 and quant_config is not None:
                validate_tensor_parallel_quantization(quant_config)
        except Exception:
            findings.append(
                _finding(
                    artifact=model_dir / "config.json",
                    field=f"{field.rsplit('.', 1)[0]}.tensor_parallel_size",
                    check="schema",
                    code="schema.incompatible_tensor_parallel_size",
                    message="tensor_parallel_size cannot shard this model",
                )
            )
    return findings


def _validate_backend(
    *,
    backend: str,
    options: Mapping[object, object],
    field: str,
    artifact: Path,
    checked_models: set[tuple[str, str, str, str]],
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    if backend not in known_backends():
        findings.append(
            _finding(
                artifact=artifact,
                field=f"{field}.backend",
                check="schema",
                code="schema.unknown_backend",
                message="backend is not registered",
            )
        )
        return findings

    try:
        validate_backend_options(backend, options)
    except (TypeError, ValueError):
        findings.append(
            _finding(
                artifact=artifact,
                field=f"{field}.options",
                check="schema",
                code="schema.invalid_backend_options",
                message="backend configuration is invalid",
            )
        )

    builtin_native = (
        backend in _NATIVE_BACKENDS
        and uses_builtin_backend_contract(backend)
    )
    if builtin_native:
        model_path = options.get("model_path")
        tokenizer = options.get("tokenizer")
        if model_path is not None:
            identity = (
                f"{type(model_path).__qualname__}:{model_path!r}",
                f"{type(tokenizer).__qualname__}:{tokenizer!r}",
                (
                    f"{type(options.get('tensor_parallel_size', 1)).__qualname__}:"
                    f"{options.get('tensor_parallel_size', 1)!r}"
                ),
                (
                    f"{type(options.get('generation_config', 'auto')).__qualname__}:"
                    f"{options.get('generation_config', 'auto')!r}"
                ),
            )
            if identity not in checked_models:
                checked_models.add(identity)
                findings.extend(
                    _validate_model(
                        model_path,
                        field=f"{field}.options.model_path",
                        tokenizer=tokenizer,
                        tensor_parallel_size=options.get(
                            "tensor_parallel_size",
                            1,
                        ),
                        reference_artifact=artifact,
                        generation_config=options.get(
                            "generation_config",
                            "auto",
                        ),
                    )
                )
        elif tokenizer is not None:
            tokenizer_findings, _vocab_size = _validate_tokenizer(
                tokenizer,
                field=f"{field}.options.tokenizer",
                reference_artifact=artifact,
            )
            findings.extend(tokenizer_findings)
    return findings


def _validate_orchestrator_topology(
    spec: OrchestratorSpec,
    *,
    artifact: Path,
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    worker_names = [worker.name for worker in spec.workers]
    if len(worker_names) != len(set(worker_names)):
        findings.append(
            _finding(
                artifact=artifact,
                field="workers",
                check="schema",
                code="schema.duplicate_worker",
                message="worker names must be unique",
            )
        )
        return findings
    if not spec.roles:
        return findings
    generation_workers = {
        worker.name for worker in spec.workers if worker.executor_ref is None
    }
    execution_workers = {
        worker.name for worker in spec.workers if worker.executor_ref is not None
    }
    profiles: tuple[tuple[str, tuple[RoleNodeSpec, ...]], ...] = (("roles", spec.roles),)
    for profile in spec.profiles:
        profiles += ((f"profiles.{profile.name}", profile.roles),)
    for field_name, node_specs in profiles:
        roles = tuple(
            RoleSpec(
                name=role.name,
                worker=role.worker,
                prompt=role.prompt,
                role_type=role.role_type,
                depends_on=role.depends_on,
                verifies=role.verifies,
                sampling=_loader_role_sampling(role),
                executor=_loader_role_executor(role),
                prompt_suffix=role.prompt_suffix,
                prompt_headless=role.prompt_headless,
                reasoning_closed=role.reasoning_closed,
                reasoning_effort=role.reasoning_effort,
                reasoning_close_tag=role.reasoning_close_tag,
                reasoning_continuation=role.reasoning_continuation,
                reasoning_open_tag=role.reasoning_open_tag,
                requires=role.requires,
            )
            for role in node_specs
        )
        try:
            conductor = Conductor(
                roles,
                {name: object() for name in generation_workers},
                execution_workers={name: object() for name in execution_workers},
            )
            final_unit = conductor._selected_final_unit().name
            for role in roles:
                if role.prompt_headless and role.name != final_unit:
                    findings.append(
                        _finding(
                            artifact=artifact,
                            field=f"{field_name}.{role.name}.prompt_headless",
                            check="schema",
                            code="schema.headless_prompt_unused",
                            message=(
                                "prompt_headless is only rendered for the selected "
                                f"final unit ({final_unit!r}); it is never used on "
                                f"role {role.name!r}"
                            ),
                        )
                    )
        except ValueError as error:
            message = (
                "orchestrator role graph contains a cycle"
                if "cycle" in str(error).lower()
                else "orchestrator role graph is invalid"
            )
            findings.append(
                _finding(
                    artifact=artifact,
                    field=field_name,
                    check="schema",
                    code="schema.invalid_role_graph",
                    message=message,
                )
            )
    return findings


def _validate_orchestrator(
    path: Path,
    *,
    reference_field: str,
    checked_models: set[tuple[str, str, str, str]],
    deployment_engine_refs: frozenset[str],
    deployment_executor_refs: frozenset[str] = frozenset(),
) -> list[ValidationFinding]:
    if not path.is_file():
        return [
            _finding(
                artifact=path,
                field=reference_field,
                check="filesystem",
                code="filesystem.missing_reference",
                message="referenced orchestrator spec does not exist",
            )
        ]
    try:
        spec = load_spec(path)
    except ValidationError as error:
        return _pydantic_findings(error, artifact=path)
    except (OSError, UnicodeError):
        return [
            _finding(
                artifact=path,
                field=reference_field,
                check="filesystem",
                code="filesystem.unreadable",
                message="orchestrator spec is not readable as UTF-8",
            )
        ]
    except (ValueError, yaml.YAMLError, RecursionError) as error:
        return [_yaml_finding(error, path)]

    findings = _validate_orchestrator_topology(spec, artifact=path)
    for index, worker in enumerate(spec.workers):
        worker_field = f"workers[{index}]"
        if worker.engine_ref is not None:
            if worker.engine_ref not in deployment_engine_refs:
                findings.append(
                    _finding(
                        artifact=path,
                        field=f"{worker_field}.engine_ref",
                        check="schema",
                        code="schema.unknown_engine_ref",
                        message="engine_ref does not match an engines: or pools: name",
                    )
                )
            continue
        if worker.executor_ref is not None:
            if worker.executor_ref not in deployment_executor_refs:
                findings.append(
                    _finding(
                        artifact=path,
                        field=f"{worker_field}.executor_ref",
                        check="schema",
                        code="schema.unknown_executor_ref",
                        message="executor_ref does not match an executors: name",
                    )
                )
            continue
        options: dict[object, object] = dict(worker.options)
        builtin_native = (
            worker.backend in _NATIVE_BACKENDS
            and uses_builtin_backend_contract(worker.backend)
        )
        if worker.model is not None and not builtin_native:
            options.setdefault("model", worker.model)
        if worker.base_url is not None:
            options.setdefault("base_url", worker.base_url)
        if worker.backend == "openai" or worker.api_key_env is not None:
            options.setdefault("api_key_env", worker.api_key_env)
        if (
            builtin_native
            and worker.model is not None
        ):
            findings.append(
                _finding(
                    artifact=path,
                    field=f"{worker_field}.model",
                    check="schema",
                    code="schema.incompatible_backend_option",
                    message="native Kairyu workers use options.model_path, not model",
                )
            )
        findings.extend(
            _validate_backend(
                backend=worker.backend,
                options=options,
                field=worker_field,
                artifact=path,
                checked_models=checked_models,
            )
        )
    return findings


def _validate_chat_templates(
    raw: Mapping[object, object],
    *,
    config_path: Path,
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    for model, source in _mapping(raw.get("chat_templates")).items():
        field = f"chat_templates.{model}"
        if not isinstance(source, str):
            continue
        candidate = source
        artifact = config_path
        if source.endswith(".jinja"):
            path = Path(source)
            if not path.is_absolute():
                path = config_path.parent / path
            candidate = str(path)
            artifact = path
            if not path.is_file():
                findings.append(
                    _finding(
                        artifact=path,
                        field=field,
                        check="filesystem",
                        code="filesystem.missing_reference",
                        message="referenced chat template does not exist",
                    )
                )
                continue
        try:
            ChatTemplate.load(candidate)
        except OSError:
            findings.append(
                _finding(
                    artifact=artifact,
                    field=field,
                    check="filesystem",
                    code="filesystem.unreadable",
                    message="chat template is not readable as UTF-8",
                )
            )
        except Exception as error:
            line = getattr(error, "lineno", None)
            suffix = f" at line {line}" if isinstance(line, int) else ""
            findings.append(
                _finding(
                    artifact=artifact,
                    field=field,
                    check="schema",
                    code="schema.invalid_chat_template",
                    message=f"invalid Jinja template{suffix}",
                )
            )
    return findings


def _validate_chat_policy(
    spec: DeploymentSpec,
    *,
    config_path: Path,
) -> list[ValidationFinding]:
    """Reuse runtime preflight without constructing a backend or emitting logs."""

    try:
        # ``_resolve_chat_policy`` is the runtime source of truth and is
        # metadata-only. Explicit-legacy warnings belong to ``serve`` startup,
        # not deterministic offline validation.
        deployment_builder._resolve_chat_policy(
            spec,
            config_path.parent,
            emit_warnings=False,
        )
    except ValueError as error:
        message = str(error)
        if "has no real chat template" in message:
            code = "schema.missing_chat_template"
            field = "chat_templates"
            safe_message = (
                "served text models require a tokenizer-owned or configured "
                "chat template, or an explicit legacy opt-in"
            )
        elif "has no supported chat policy" in message:
            code = "schema.missing_legacy_chat_policy"
            field = "legacy_chat_models"
            safe_message = (
                "non-template-preserving served models require an explicit "
                "model-scoped legacy chat policy"
            )
        else:
            code = "schema.invalid_chat_policy"
            field = "chat_templates"
            safe_message = (
                "chat template policy is incompatible with the local deployment "
                "artifacts"
            )
        return [
            _finding(
                artifact=config_path,
                field=field,
                check="schema",
                code=code,
                message=safe_message,
            )
        ]
    return []


def validate_deployment(config: str | Path) -> ValidationReport:
    """Validate one DeploymentSpec and all safely discoverable local inputs."""

    config_path = Path(config).absolute()
    findings: list[ValidationFinding] = []
    schema_ran = False
    filesystem_ran = True
    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        findings.append(
            _finding(
                artifact=config_path,
                field="<root>",
                check="filesystem",
                code="filesystem.missing_file",
                message="deployment config does not exist",
            )
        )
        return _report(
            config_path,
            findings,
            schema_ran=schema_ran,
            filesystem_ran=filesystem_ran,
        )
    except (OSError, UnicodeError, ValueError):
        findings.append(
            _finding(
                artifact=config_path,
                field="<root>",
                check="filesystem",
                code="filesystem.unreadable",
                message="deployment config is not readable as UTF-8",
            )
        )
        return _report(
            config_path,
            findings,
            schema_ran=schema_ran,
            filesystem_ran=filesystem_ran,
        )

    try:
        raw = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except (ValueError, yaml.YAMLError, RecursionError) as error:
        findings.append(_yaml_finding(error, config_path))
        return _report(
            config_path,
            findings,
            schema_ran=True,
            filesystem_ran=filesystem_ran,
        )
    schema_ran = True
    if not isinstance(raw, dict):
        findings.append(
            _finding(
                artifact=config_path,
                field="<root>",
                check="schema",
                code="schema.expected_mapping",
                message="deployment spec YAML must be a mapping at the top level",
            )
        )
        return _report(
            config_path,
            findings,
            schema_ran=schema_ran,
            filesystem_ran=filesystem_ran,
        )

    spec: DeploymentSpec | None = None
    try:
        spec = load_deployment_spec(config_path, resolve_credentials=False)
    except ValidationError as error:
        findings.extend(_pydantic_findings(error, artifact=config_path))
    except (ValueError, yaml.YAMLError, RecursionError) as error:
        findings.append(_yaml_finding(error, config_path))
    checked_models: set[tuple[str, str, str, str]] = set()
    for field, backend, options in _iter_deployment_backends(raw):
        findings.extend(
            _validate_backend(
                backend=backend,
                options=options,
                field=field,
                artifact=config_path,
                checked_models=checked_models,
            )
        )

    checked_orchestrators: set[Path] = set()
    deployment_engine_refs = frozenset(
        {
            *(name for name in _mapping(raw.get("engines")) if isinstance(name, str)),
            *(name for name in _mapping(raw.get("pools")) if isinstance(name, str)),
        }
    )
    deployment_executor_refs = frozenset(
        name for name in _mapping(raw.get("executors")) if isinstance(name, str)
    )
    for field, reference in _iter_orchestrator_references(raw):
        path = Path(reference)
        if not path.is_absolute():
            path = config_path.parent / path
        path = _canonical(path)
        if path in checked_orchestrators and path.is_file():
            continue
        if path.is_file():
            checked_orchestrators.add(path)
        findings.extend(
            _validate_orchestrator(
                path,
                reference_field=field,
                checked_models=checked_models,
                deployment_engine_refs=deployment_engine_refs,
                deployment_executor_refs=deployment_executor_refs,
            )
        )
    template_findings = _validate_chat_templates(raw, config_path=config_path)
    findings.extend(template_findings)
    if spec is not None and not template_findings:
        findings.extend(_validate_chat_policy(spec, config_path=config_path))
    return _report(
        config_path,
        findings,
        schema_ran=schema_ran,
        filesystem_ran=filesystem_ran,
    )


def _report(
    config_path: Path,
    findings: Iterable[ValidationFinding],
    *,
    schema_ran: bool,
    filesystem_ran: bool,
) -> ValidationReport:
    unique = tuple(sorted(set(findings)))
    schema_failed = any(finding.check == "schema" for finding in unique)
    filesystem_failed = any(finding.check == "filesystem" for finding in unique)
    checks: dict[str, CheckStatus] = {
        "schema": (
            "failed" if schema_failed else ("passed" if schema_ran else "not_run")
        ),
        "filesystem": (
            "failed"
            if filesystem_failed
            else ("passed" if filesystem_ran else "not_run")
        ),
        "network": "skipped",
        "hardware": "indeterminate",
    }
    return ValidationReport(
        config=_display_path(config_path),
        findings=unique,
        checks=checks,
    )
