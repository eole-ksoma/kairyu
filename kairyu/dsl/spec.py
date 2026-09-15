"""Pydantic schema shared by the YAML loader and the decorator front-end (D7)."""

from __future__ import annotations

from string import Formatter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kairyu.engine.openai_capabilities import resolve_openai_capabilities


class WorkerSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    engine_ref: str | None = Field(default=None, min_length=1)
    # A deployment-owned sandbox execution service (ECO-D1). Mutually exclusive
    # with engine_ref and with every generation factory field.
    executor_ref: str | None = Field(default=None, min_length=1)
    backend: str = "mock"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    options: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_openai_capabilities(self) -> WorkerSpec:
        if self.engine_ref is not None and self.executor_ref is not None:
            raise ValueError("a worker cannot declare both engine_ref and executor_ref")
        if self.engine_ref is not None or self.executor_ref is not None:
            kind = "engine_ref" if self.engine_ref is not None else "executor_ref"
            factory_fields = {
                "backend",
                "model",
                "base_url",
                "api_key_env",
                "options",
            } & self.model_fields_set
            if factory_fields:
                raise ValueError(
                    f"{kind} workers cannot also declare factory fields: "
                    f"{sorted(factory_fields)}"
                )
            return self
        if self.backend == "openai":
            resolve_openai_capabilities(
                self.options.get("upstream", "generic"),
                self.options.get("capabilities"),
            )
        return self


class EffortMaxTokensSpec(BaseModel):
    """Token budget (private thinking + answer together) per resolved
    reasoning effort (DTO-D8). All three tiers are required so the map is
    total over every resolvable level; the plain ``max_tokens`` remains the
    fallback when the resolved effort is null."""

    model_config = ConfigDict(frozen=True)

    low: int = Field(ge=1, le=393216)
    high: int = Field(ge=1, le=393216)
    max: int = Field(ge=1, le=393216)


class RoleSamplingSpec(BaseModel):
    """Per-role sampling overrides (EO-D9). Internal roles are capped by
    ``internal_max_tokens``. On the selected final unit the style fields are
    deployment policy layered over the caller's public params, and
    ``max_tokens`` is a cap that is min()'d with the caller's public allowance
    (DTO-D13); the caller's intent carriers (n, logprobs, response_format,
    tools) always apply. Omitted fields keep the caller's or engine's
    defaults."""

    model_config = ConfigDict(frozen=True)

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=393216)
    max_tokens_by_effort: EffortMaxTokensSpec | None = None
    seed_offset: int | None = Field(default=None, ge=0)
    stop: tuple[str, ...] = ()


class ExecutionLimitsSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    wall_time_s: float = Field(default=10.0, gt=0, le=120)
    cpu_time_s: float = Field(default=8.0, gt=0, le=120)
    memory_mb: int = Field(default=512, ge=32, le=4096)
    processes: int = Field(default=64, ge=1, le=1024)
    output_bytes: int = Field(default=32768, ge=1024, le=1048576)


class ExecutorRoleSpec(BaseModel):
    """Configuration for a sandbox execution role (ECO-D3)."""

    model_config = ConfigDict(frozen=True)

    code_from: tuple[str, ...] = Field(min_length=1)
    tests_from: tuple[str, ...] = Field(min_length=1)
    mode: Literal["single", "matrix"] = "single"
    limits: ExecutionLimitsSpec = ExecutionLimitsSpec()


def _check_prompt_placeholders(role: str, field_name: str, template: str) -> None:
    """Reject a prompt template the Conductor cannot render.

    Role prompts are rendered with ``str.format_map`` over ``{query}`` and the
    dependency outputs; a literal brace pair such as ``{...}`` is a positional
    field that raises at request time — after the stage's dependencies already
    ran. Fail at load so ``kairyu validate`` and startup catch it.
    """

    try:
        fields = [
            name
            for _literal, name, _format_spec, _conversion in Formatter().parse(template)
            if name is not None
        ]
    except ValueError as error:
        raise ValueError(
            f"role {role!r} {field_name} is not a valid template ({error}); "
            "escape literal braces as '{{' and '}}'"
        ) from error
    bad = [name for name in fields if not name.isidentifier()]
    if bad:
        raise ValueError(
            f"role {role!r} {field_name} contains positional or non-identifier "
            f"placeholder(s) {bad}; escape literal braces as '{{' and '}}'"
        )


class RoleNodeSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    worker: str
    prompt: str = ""
    role_type: str = "worker"
    depends_on: tuple[str, ...] = ()
    verifies: str | None = None
    sampling: RoleSamplingSpec | None = None
    executor: ExecutorRoleSpec | None = None
    # Appended after the prompt body on every attempt, including verifier
    # refinements — for worker-native scaffolding that must end the prompt.
    prompt_suffix: str = ""
    # Alternate body rendered for the selected final unit when the head is
    # absent or disabled for the call (tools, n>1, ...): the publisher then
    # writes the complete answer instead of continuing a committed opening
    # (issue #496). Uses the same placeholders and prompt_suffix.
    prompt_headless: str = ""
    # Declares that this role's rendered scaffold already closed the model's
    # private-reasoning span, so upstream reasoning-classified output with an
    # empty public text is the answer itself, not hidden deliberation.
    reasoning_closed: bool = False
    # Reasoning effort sent on every attempt of this role. A level value is
    # fixed deployment policy: it is independent of (and never overridden by)
    # the caller's request-level effort. "inherit" forwards the caller's
    # request-level effort instead (None when the caller sent none).
    reasoning_effort: Literal["low", "high", "max", "inherit"] | None = None
    # The literal that closes the private-reasoning span this role's
    # prompt_suffix opens (e.g. "</think>"). Scaffold knowledge, declared as
    # config; public_output_floor uses it to force-close deliberation in the
    # final unit's empty-output re-dispatch (issue #542).
    reasoning_close_tag: str = ""
    # How the public-output floor continues exhausted deliberation (DTO-D15).
    # "prefix": the role's own text scaffold opened the span, so the captured
    # reasoning plus reasoning_close_tag is appended to the prompt. "chat":
    # the worker's upstream chat template opened the span, so the captured
    # reasoning plus the close tag is sent as an assistant-turn prefill the
    # template re-renders as a closed thinking turn and generation continues.
    reasoning_continuation: Literal["prefix", "chat"] = "prefix"
    # The literal that opens the span for "chat" continuation (e.g.
    # "<think>"): vLLM continues the final assistant message only when it
    # appears verbatim in the rendered chat, so the prefill must reproduce the
    # template's closed thinking turn "<open>\n<reasoning>\n<close>\n\n".
    reasoning_open_tag: str = ""
    # Dispatch condition. "image": the role runs only when the request carries
    # image input; on text requests it is skipped entirely (no model call, no
    # budget step), its dependents run as if it were absent, and its template
    # slot renders as "" (DTO-D11). Head, final, verifier, and executor roles
    # cannot be conditional.
    requires: Literal["image"] | None = None

    @model_validator(mode="after")
    def _executor_shape(self) -> RoleNodeSpec:
        if (self.role_type == "executor") != (self.executor is not None):
            raise ValueError(
                f"role {self.name!r}: role_type 'executor' and an executor block "
                "must be declared together"
            )
        if self.executor is not None:
            if self.sampling is not None:
                raise ValueError(f"executor role {self.name!r} cannot declare sampling")
            if self.prompt_headless or self.reasoning_closed:
                raise ValueError(
                    f"executor role {self.name!r} cannot declare prompt_headless "
                    "or reasoning_closed"
                )
            if self.reasoning_close_tag:
                raise ValueError(
                    f"executor role {self.name!r} cannot declare reasoning_close_tag"
                )
            if self.reasoning_effort is not None:
                raise ValueError(
                    f"executor role {self.name!r} cannot declare reasoning_effort"
                )
            deps = set(self.depends_on)
            missing = (set(self.executor.code_from) | set(self.executor.tests_from)) - deps
            if missing:
                raise ValueError(
                    f"executor role {self.name!r} references roles outside its "
                    f"depends_on: {sorted(missing)}"
                )
        elif not self.prompt:
            raise ValueError(f"role {self.name!r} requires a prompt")
        for field_name in ("prompt", "prompt_headless"):
            _check_prompt_placeholders(self.name, field_name, getattr(self, field_name))
        if self.requires is not None and self.role_type in {"verifier", "executor"}:
            raise ValueError(
                f"role {self.name!r}: a {self.role_type} role cannot declare requires"
            )
        if (
            self.sampling is not None
            and self.sampling.max_tokens_by_effort is not None
            and self.reasoning_effort != "inherit"
        ):
            # A fixed-level role has a constant effort (use max_tokens) and an
            # effort-less role never consults the map — both are dead config.
            raise ValueError(
                f"role {self.name!r}: max_tokens_by_effort requires "
                "reasoning_effort: inherit"
            )
        if self.reasoning_close_tag and self.reasoning_closed:
            raise ValueError(
                f"role {self.name!r}: reasoning_close_tag declares an open "
                "reasoning span and cannot combine with reasoning_closed"
            )
        if self.reasoning_continuation == "chat" and not (
            self.reasoning_close_tag and self.reasoning_open_tag
        ):
            raise ValueError(
                f"role {self.name!r}: reasoning_continuation 'chat' requires "
                "reasoning_open_tag and reasoning_close_tag"
            )
        if self.reasoning_open_tag and self.reasoning_continuation != "chat":
            raise ValueError(
                f"role {self.name!r}: reasoning_open_tag is only used by "
                "reasoning_continuation 'chat'"
            )
        return self


class BudgetSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=16, ge=1)
    max_refine_depth: int = Field(default=2, ge=0)
    max_cost_usd: float | None = Field(default=None, gt=0)
    cost_per_1k_chars_usd: float | None = Field(default=None, gt=0)


class RouterThresholdSpec(BaseModel):
    """Validated operator overrides for the deterministic rule router."""

    model_config = ConfigDict(frozen=True)

    multi_step_markers: int = Field(default=3, ge=0)
    multi_agent_min_chars: int = Field(default=2000, ge=1)
    reasoning_keywords: int = Field(default=2, ge=0)
    math_symbols: int = Field(default=3, ge=0)
    tier2_min_chars: int = Field(default=600, ge=1)


class RouterSpec(BaseModel):
    """Immutable routing policy loaded from a calibrated artifact."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["rules", "calibrated"] = "rules"
    artifact: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    target_mode: Literal["auto", "auto-max"] = "auto"
    thresholds: RouterThresholdSpec | None = None

    @model_validator(mode="after")
    def _calibrated_artifact_is_pinned(self) -> RouterSpec:
        if self.kind == "calibrated" and (self.artifact is None or self.sha256 is None):
            raise ValueError("calibrated router requires artifact and sha256")
        if self.kind == "calibrated" and self.thresholds is not None:
            raise ValueError("calibrated router thresholds come from its pinned artifact")
        if self.kind == "rules" and (self.artifact is not None or self.sha256 is not None):
            raise ValueError("rules router cannot specify artifact or sha256")
        return self


class ProfileChoiceSpec(BaseModel):
    """One route the profile judge may pick (DTO-D13): the label the judge
    answers with, the role profile it selects, and the routing criteria shown
    to the judge."""

    model_config = ConfigDict(frozen=True)

    profile: str = Field(min_length=1)
    label: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$", max_length=32)
    criteria: str = Field(min_length=1)


class ProfileJudgeSpec(BaseModel):
    """LLM route selection among the role profiles (issue #509, generalized by
    DTO-D13).

    A bounded verdict-only call on a configured generation worker answers one
    of the choice labels over the latest user turn; the selected profile's
    DAG then serves the request. On judge timeout, backend error, or an
    unparseable verdict the ``fallback`` profile applies.
    """

    model_config = ConfigDict(frozen=True)

    worker: str
    timeout_seconds: float = Field(default=5.0, gt=0.0, le=60.0)
    max_tokens: int = Field(default=8, ge=1, le=64)
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    choices: tuple[ProfileChoiceSpec, ...] = Field(min_length=2)
    fallback: str = "primary"

    @model_validator(mode="after")
    def _choices_are_distinct(self) -> ProfileJudgeSpec:
        labels = [choice.label for choice in self.choices]
        if len(set(labels)) != len(labels):
            raise ValueError("profile_judge choice labels must be unique")
        profiles = [choice.profile for choice in self.choices]
        if len(set(profiles)) != len(profiles):
            raise ValueError("profile_judge choices must reference distinct profiles")
        return self


class RoleProfileSpec(BaseModel):
    """A named alternative role DAG served under the same model (DTO-D13).
    The primary ``roles`` DAG is the implicit profile ``primary``."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    roles: tuple[RoleNodeSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _name_is_identifier(self) -> RoleProfileSpec:
        if not self.name.isidentifier() or self.name == "primary":
            raise ValueError(
                f"profile name {self.name!r} must be an identifier other than 'primary'"
            )
        return self


class OrchestratorSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    workers: tuple[WorkerSpec, ...] = Field(min_length=1)
    roles: tuple[RoleNodeSpec, ...] = ()
    # Optional named alternative role DAGs served under the same model name
    # (issue #509, generalized by DTO-D13). The orchestrator selects one
    # profile per request; a profile may be a full ensemble or a single-role
    # direct route.
    profiles: tuple[RoleProfileSpec, ...] = ()
    # Optional LLM verdict selecting among the profiles. Without it the
    # primary DAG always serves.
    profile_judge: ProfileJudgeSpec | None = None
    budget: BudgetSpec = BudgetSpec()
    router: RouterSpec = RouterSpec()
    shared_prefix: str = ""
    # Private planning/proposal/verification output is bounded independently
    # from the caller's public final-answer allowance.  Keeping this in the
    # orchestration policy makes latency/cost tuning portable across backends.
    internal_max_tokens: int = Field(default=1024, ge=1, le=131072)
    # Public-answer tokens reserved for the always-thinking final unit
    # (issue #542): its attempt-0 thinking is capped at the caller's budget
    # minus this floor, and the bounded empty-output re-dispatch answers with
    # the reserve after a forced reasoning close. Requires the final unit to
    # declare reasoning_close_tag. None keeps today's behavior.
    public_output_floor: int | None = Field(default=None, ge=1, le=131072)
    # Zero keeps the standard Conductor route. A positive value turns the
    # multi-agent route into that many parallel MoA proposals plus synthesis.
    moa_samples: int = Field(default=0, ge=0, le=16)
    # Completed pre-final stage outputs may be surfaced separately from the
    # answer. Hidden is the safe and backward-compatible default.
    expose_intermediate_outputs: bool = False
    # Effort assumed for a request whose caller sent no reasoning_effort.
    # Consumed by "inherit" roles (and the direct/MoA routes); an explicit
    # request-level effort always overrides it.
    default_reasoning_effort: Literal["low", "high", "max"] | None = None

    @model_validator(mode="after")
    def _roles_reference_known_workers(self) -> OrchestratorSpec:
        worker_names = [worker.name for worker in self.workers]
        known = set(worker_names)
        if len(known) != len(worker_names):
            raise ValueError("worker names must be unique")
        executor_workers = {
            worker.name for worker in self.workers if worker.executor_ref is not None
        }
        profile_names = [profile.name for profile in self.profiles]
        if len(set(profile_names)) != len(profile_names):
            raise ValueError("profile names must be unique")
        if self.profiles and not self.roles:
            raise ValueError("profiles require a primary roles DAG")
        if self.profiles and self.moa_samples > 0:
            raise ValueError(
                "profiles cannot be combined with moa_samples > 0; "
                "choose role DAG profiles or MoA mode"
            )
        if self.public_output_floor is not None and self.moa_samples > 0:
            raise ValueError(
                "public_output_floor cannot be combined with moa_samples > 0; "
                "the floor applies only to the Conductor final unit"
            )
        if self.profile_judge is not None:
            if not self.profiles:
                raise ValueError("profile_judge requires at least one profile")
            known_profiles = {"primary", *profile_names}
            for choice in self.profile_judge.choices:
                if choice.profile not in known_profiles:
                    raise ValueError(
                        f"profile_judge choice {choice.label!r} references unknown "
                        f"profile {choice.profile!r}"
                    )
            if self.profile_judge.fallback not in known_profiles:
                raise ValueError(
                    f"profile_judge fallback references unknown profile "
                    f"{self.profile_judge.fallback!r}"
                )
            if self.profile_judge.worker not in known:
                raise ValueError(
                    f"profile_judge references unknown worker "
                    f"{self.profile_judge.worker!r}"
                )
            if self.profile_judge.worker in executor_workers:
                raise ValueError("profile_judge worker must be a generation worker")
        for profile, roles in (
            ("roles", self.roles),
            *((f"profiles.{spec.name}", spec.roles) for spec in self.profiles),
        ):
            for role in roles:
                if role.worker not in known:
                    raise ValueError(
                        f"{profile}: role {role.name!r} references unknown worker "
                        f"{role.worker!r}; known workers: {sorted(known)}"
                    )
                if (role.role_type == "executor") != (role.worker in executor_workers):
                    raise ValueError(
                        f"{profile}: role {role.name!r}: executor roles must use an "
                        "executor_ref worker and generation roles must not"
                    )
            heads = [role for role in roles if role.role_type == "head"]
            if len(heads) > 1:
                raise ValueError(f"{profile}: at most one role may declare role_type 'head'")
            if heads:
                head = heads[0]
                if head.depends_on:
                    raise ValueError(
                        f"{profile}: head role {head.name!r} cannot declare dependencies"
                    )
                if any(role.verifies == head.name for role in roles):
                    raise ValueError(f"{profile}: head role {head.name!r} cannot be verified")
                if head.requires is not None:
                    raise ValueError(
                        f"{profile}: head role {head.name!r} cannot declare requires"
                    )
                if not any(head.name in role.depends_on for role in roles):
                    raise ValueError(
                        f"{profile}: head role {head.name!r} must feed at least one "
                        "downstream role"
                    )
        return self
