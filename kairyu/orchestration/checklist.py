"""Checklist verifiers: deterministic checks plus System One probabilities.

A checklist verifier judges its target attempt without a generation call:

1. ``pre`` checks run deterministic primitives (:mod:`kairyu.orchestration.checks`)
   on the attempt; any failure is a FAIL without calling a model.
2. Inline-bound roles (for example a claim extractor) run on the attempt.
3. ``post`` checks run on the attempt and the inline outputs.
4. Questions go to a System One (Jev wire API) decision backend as ``noul``
   reads; each answer is a probability.
5. Each checklist item gets a probability ``p`` of being satisfied (a check is
   1 or 0; questions sharing an item id aggregate by minimum) and passes when
   ``p >= threshold``. PASS needs every item to pass.

The verdict text is the existing verifier contract (first line PASS/FAIL,
then one feedback line per failing item), so the Conductor's refine loop is
unchanged. Policy — which checks and questions exist, their wording, the
threshold, and how a list output is curated afterwards — is configuration.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from string import Formatter
from typing import Protocol

from kairyu.orchestration.checks import (
    CheckContext,
    CheckParameterError,
    json_path,
    parse_json_output,
    run_check,
    static_check_is_valid,
)
from kairyu.orchestration.request import conversation_messages


class DecisionBackend(Protocol):
    """The System One surface a checklist needs (HTTPSystemOneBackend)."""

    async def decide(self, body: dict): ...


class ChecklistUnavailable(Exception):
    """The checklist could not be judged (backend down, overloaded, malformed)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class ItemSource:
    """Items taken from a role's JSON output.

    ``path`` is a dotted key path to a list of objects; ``where`` keeps only
    objects whose keys equal the given values. ``pairs_sharing`` turns the
    selection into every unordered pair of items whose list-valued key shares
    at least one value; templates then see ``a`` and ``b`` instead of
    ``item``.
    """

    role: str
    path: str = ""
    where: Mapping[str, object] = field(default_factory=dict)
    pairs_sharing: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", dict(self.where))


@dataclass(frozen=True)
class ChecklistCheck:
    """One deterministic check, static or one per selected item."""

    id: str
    proposition: str
    primitive: str = ""
    params: Mapping[str, object] = field(default_factory=dict)
    # Per-item checks read the primitive and its parameters from the item.
    foreach: ItemSource | None = None
    primitive_key: str = ""
    params_key: str = ""
    sources_key: str = ""
    # "pre" runs before inline roles, "post" after them.
    stage: str = "pre"
    group: str = "checklist"
    # When an item's primitive or parameters are unusable, ask System One
    # whether the subject satisfies the item's proposition instead of failing.
    semantic_fallback: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", dict(self.params))
        if self.stage not in {"pre", "post"}:
            raise ValueError(f"check {self.id!r}: stage must be 'pre' or 'post'")
        if self.foreach is None:
            if not self.primitive:
                raise ValueError(f"check {self.id!r} needs a primitive")
            static_check_is_valid(self.primitive, self.params)
        elif not (self.primitive or self.primitive_key):
            raise ValueError(f"check {self.id!r} needs primitive or primitive_key")
        if self.foreach is not None and self.foreach.pairs_sharing:
            raise ValueError(f"check {self.id!r}: checks cannot iterate pairs")
        for name in ("id", "proposition"):
            _validate_template(f"check {self.id!r} {name}", getattr(self, name))


@dataclass(frozen=True)
class ChecklistQuestion:
    """One System One ``noul`` question, static or per selected item (or pair).

    By default the question is built from the requirement: Jev is asked
    whether the checklist ``subject`` satisfies ``proposition``, with yes/no
    criteria for full versus partial or missing satisfaction. ``ask`` (with
    optional ``criteria_true`` / ``criteria_false``) replaces that with an
    explicit yes/no question. ``context`` adds item-specific fields (for
    example a quoted claim and its evidence) to the question object, so the
    question stays self-contained without placing item text in the state.
    """

    id: str
    proposition: str
    foreach: ItemSource | None = None
    sources_key: str = ""
    # "yes": p is the probability of yes. "no": p is the probability of no.
    expect: str = "yes"
    threshold: float | None = None
    group: str = "checklist"
    subject: str = ""
    ask: str = ""
    criteria_true: str = ""
    criteria_false: str = ""
    context: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", dict(self.context))
        if self.expect not in {"yes", "no"}:
            raise ValueError(f"question {self.id!r}: expect must be 'yes' or 'no'")
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"question {self.id!r}: threshold must be in [0, 1]")
        if (self.criteria_true or self.criteria_false) and not self.ask:
            raise ValueError(f"question {self.id!r}: criteria need an explicit ask")
        if {"question", "requirement"} & set(self.context):
            raise ValueError(f"question {self.id!r}: context cannot redefine question/requirement")
        for name in ("id", "proposition", "subject", "ask", "criteria_true", "criteria_false"):
            _validate_template(f"question {self.id!r} {name}", getattr(self, name))
        for key, template in self.context.items():
            _validate_template(f"question {self.id!r} context {key}", template)


@dataclass(frozen=True)
class StateSection:
    """One field of the System One state object.

    ``source`` is ``query`` (the request: its role-tagged messages when the
    query is Kairyu's chat transcript) or a role name (its output, embedded
    as a JSON value when it parses as JSON). A text value longer than
    ``max_chars`` is cut with an explicit marker.
    """

    key: str
    source: str
    max_chars: int | None = None

    def __post_init__(self) -> None:
        if not self.key or not self.source:
            raise ValueError("a state section needs a key and a source")
        if self.max_chars is not None and self.max_chars < 1:
            raise ValueError(f"state section {self.key!r}: max_chars must be positive")


@dataclass(frozen=True)
class CurationConfig:
    """Edit the target's JSON list after the verdict (no model call).

    Items whose ``drop_group`` probability is below ``drop_below`` are
    removed; an item pair whose ``merge_group`` probability is below
    ``merge_below`` is merged (first item kept, sources united, propositions
    joined); units no remaining item cites get a ``pad`` item rendered from
    the unit. Probabilities are the normalized pass probabilities.
    """

    items_path: str
    id_key: str = "id"
    sources_key: str = "sources"
    proposition_key: str = "proposition"
    drop_group: str = ""
    drop_below: float = 0.5
    merge_group: str = ""
    merge_below: float = 0.5
    units_path: str = ""
    unit_id_key: str = "id"
    pad: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pad", dict(self.pad))
        if self.units_path and not self.pad:
            raise ValueError("curation units_path requires a pad item template")


@dataclass(frozen=True)
class ChecklistConfig:
    checks: tuple[ChecklistCheck, ...] = ()
    questions: tuple[ChecklistQuestion, ...] = ()
    # System One state: a JSON object with one field per section.
    state: tuple[StateSection, ...] = ()
    # What the requirement questions are about (for example "the answer").
    subject: str = "the state"
    threshold: float = 0.5
    samples: int | None = None
    think: int | None = None
    steps: int | None = None
    max_questions_per_call: int = 64
    max_questions: int = 256
    max_state_chars: int | None = None
    feedback_header: str = "The following requirements are not met:"
    feedback_item: str = "- [{id}] {proposition} (sources: {sources}; p={p}){detail}"
    max_refinements: int | None = None
    # "last": publish the last attempt; "latest_checks_passed": the newest
    # attempt whose deterministic checks all passed (else the first attempt).
    on_exhausted: str = "last"
    # "error": an unjudgeable checklist fails the unit (verifier contract).
    # "publish_unverified": publish ``unverified_from`` (or the current
    # attempt) and report the result as unverified.
    on_unavailable: str = "error"
    unverified_from: str = ""
    curate: CurationConfig | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "questions", tuple(self.questions))
        object.__setattr__(self, "state", tuple(self.state))
        if not self.checks and not self.questions:
            raise ValueError("a checklist needs at least one check or question")
        if self.questions and not self.state:
            raise ValueError("checklist questions need System One state sections")
        keys = [section.key for section in self.state]
        if len(set(keys)) != len(keys):
            raise ValueError("checklist state section keys must be unique")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("checklist threshold must be in [0, 1]")
        if self.on_exhausted not in {"last", "latest_checks_passed"}:
            raise ValueError("on_exhausted must be 'last' or 'latest_checks_passed'")
        if self.on_unavailable not in {"error", "publish_unverified"}:
            raise ValueError("on_unavailable must be 'error' or 'publish_unverified'")
        if self.unverified_from and self.on_unavailable != "publish_unverified":
            raise ValueError("unverified_from requires on_unavailable: publish_unverified")
        if self.max_refinements is not None and self.max_refinements < 0:
            raise ValueError("max_refinements must be >= 0")
        if not 1 <= self.max_questions_per_call <= self.max_questions:
            raise ValueError("max_questions_per_call must be in [1, max_questions]")
        for name in ("subject", "feedback_header"):
            _validate_template(f"checklist {name}", getattr(self, name))

    def referenced_roles(self) -> set[str]:
        """Roles whose outputs this checklist reads (for DAG validation)."""

        names = {
            source.role
            for source in (
                *(check.foreach for check in self.checks),
                *(question.foreach for question in self.questions),
            )
            if source is not None
        }
        for check in self.checks:
            role = check.params.get("role")
            if isinstance(role, str):
                names.add(role)
        names.update(section.source for section in self.state if section.source != "query")
        return names


@dataclass(frozen=True)
class ChecklistItem:
    """One judged requirement as reported to the caller."""

    id: str
    proposition: str
    sources: tuple[str, ...]
    kind: str  # "deterministic" | "semantic"
    group: str
    p: float
    passed: bool
    detail: str = ""
    members: tuple[str, ...] = ()
    # False when a deterministic failure ended the attempt before this item
    # was read; it is reported (the checklist stays complete) but not scored.
    judged: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "proposition": self.proposition,
            "sources": list(self.sources),
            "kind": self.kind,
            "group": self.group,
            "p": round(self.p, 6) if self.judged else None,
            "passed": self.passed,
            "judged": self.judged,
        }


@dataclass(frozen=True)
class ChecklistVerdict:
    passed: bool
    items: tuple[ChecklistItem, ...]
    # True when every deterministic item passed (used by on_exhausted).
    checks_passed: bool
    text: str
    usage: tuple[int, int] = (0, 0)
    reads: int = 0


@dataclass(frozen=True)
class VerificationReport:
    """The public verification outcome of the final unit."""

    guaranteed: bool
    reason: str | None
    threshold: float | None
    items: tuple[ChecklistItem, ...] = ()
    attempts: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "guaranteed": self.guaranteed,
            "reason": self.reason,
            "threshold": self.threshold,
            "attempts": self.attempts,
            "requirements": [item.as_dict() for item in self.items],
        }


class _TemplateValues(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _validate_template(label: str, template: str) -> None:
    try:
        list(Formatter().parse(template))
    except ValueError as error:
        raise ValueError(f"{label} is not a valid template ({error})") from error


def _display(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _display(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return ", ".join(str(_display(inner)) for inner in value)
    return value


def render(template: str, values: Mapping[str, object]) -> str:
    """Render a checklist template; lists read as comma-separated text."""

    prepared = _TemplateValues({key: _display(value) for key, value in values.items()})
    try:
        return template.format_map(prepared)
    except (KeyError, IndexError, AttributeError, TypeError, ValueError) as error:
        raise CheckParameterError(f"template cannot be rendered: {error}") from error


def _source_items(source: ItemSource, outputs: Mapping[str, str]) -> list[Mapping[str, object]]:
    raw = outputs.get(source.role)
    if raw is None:
        raise ChecklistUnavailable("checklist_unavailable", f"{source.role!r} has no output")
    try:
        items = json_path(parse_json_output(raw), source.path)
    except (ValueError, KeyError) as error:
        raise ChecklistUnavailable(
            "checklist_unavailable",
            f"{source.role!r} output has no JSON list at {source.path!r}",
        ) from error
    if not isinstance(items, list):
        raise ChecklistUnavailable(
            "checklist_unavailable", f"{source.role!r} output at {source.path!r} is not a list"
        )
    return [
        item
        for item in items
        if isinstance(item, Mapping)
        and all(item.get(key) == value for key, value in source.where.items())
    ]


def _bindings(source: ItemSource | None, outputs: Mapping[str, str]) -> list[dict[str, object]]:
    if source is None:
        return [{}]
    items = _source_items(source, outputs)
    if not source.pairs_sharing:
        return [{"item": item} for item in items]
    key = source.pairs_sharing
    pairs = []
    for index, first in enumerate(items):
        first_refs = first.get(key)
        if not isinstance(first_refs, list):
            continue
        for second in items[index + 1 :]:
            second_refs = second.get(key)
            if isinstance(second_refs, list) and set(map(str, first_refs)) & set(
                map(str, second_refs)
            ):
                pairs.append({"a": first, "b": second})
    return pairs


def _sources(binding: Mapping[str, object], key: str) -> tuple[str, ...]:
    if not key:
        return ()
    values: list[str] = []
    for name in ("item", "a", "b"):
        bound = binding.get(name)
        if isinstance(bound, Mapping) and isinstance(bound.get(key), list):
            values.extend(str(value) for value in bound[key] if str(value) not in values)
    return tuple(values)


def _members(binding: Mapping[str, object], id_key: str = "id") -> tuple[str, ...]:
    return tuple(
        str(binding[name].get(id_key))
        for name in ("a", "b")
        if isinstance(binding.get(name), Mapping)
    )


@dataclass(frozen=True)
class _PendingQuestion:
    item_id: str
    proposition: str
    sources: tuple[str, ...]
    group: str
    # The Jev ``noul`` question object (instructions and criteria).
    payload: Mapping[str, object]
    expect: str
    threshold: float
    members: tuple[str, ...]


def requirement_question(
    subject: str,
    proposition: str,
    context: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """A Jev ``noul`` question asking whether ``subject`` meets a requirement."""

    return {
        "type": "noul",
        "instructions": {
            "question": f"Does {subject} satisfy the requirement below?",
            "requirement": proposition,
            **dict(context or {}),
        },
        "criteria": {
            "true": f"{subject} fully satisfies the requirement",
            "false": f"{subject} does not satisfy the requirement, or satisfies it only partly",
        },
    }


def _question_payload(
    config: ChecklistConfig,
    question: ChecklistQuestion,
    binding: Mapping[str, object],
    proposition: str,
) -> dict[str, object]:
    context = {key: render(value, binding) for key, value in question.context.items()}
    if not question.ask:
        subject = render(question.subject, binding) if question.subject else config.subject
        return requirement_question(subject, proposition, context)
    payload: dict[str, object] = {
        "type": "noul",
        "instructions": (
            {"question": render(question.ask, binding), **context}
            if context
            else render(question.ask, binding)
        ),
    }
    if question.criteria_true or question.criteria_false:
        payload["criteria"] = {
            "true": render(question.criteria_true, binding),
            "false": render(question.criteria_false, binding),
        }
    return payload


def _run_checks(
    config: ChecklistConfig,
    stage: str,
    ctx: CheckContext,
    outputs: Mapping[str, str],
) -> tuple[list[ChecklistItem], list[_PendingQuestion]]:
    items: list[ChecklistItem] = []
    fallbacks: list[_PendingQuestion] = []
    for check in config.checks:
        if check.stage != stage:
            continue
        for binding in _bindings(check.foreach, outputs):
            item_id = render(check.id, binding)
            proposition = render(check.proposition, binding)
            sources = _sources(binding, check.sources_key)
            item = binding.get("item")
            try:
                primitive = check.primitive
                params: object = check.params
                if isinstance(item, Mapping):
                    if check.primitive_key:
                        primitive = json_path(item, check.primitive_key)
                    if check.params_key:
                        params = json_path(item, check.params_key)
                if not isinstance(primitive, str) or not isinstance(params, Mapping):
                    raise CheckParameterError("item has no usable primitive or parameters")
                outcome = run_check(primitive, params, ctx)
            except (CheckParameterError, KeyError) as error:
                if check.semantic_fallback:
                    fallbacks.append(
                        _PendingQuestion(
                            item_id=item_id,
                            proposition=proposition,
                            sources=sources,
                            group=check.group,
                            payload=requirement_question(config.subject, proposition),
                            expect="yes",
                            threshold=config.threshold,
                            members=(),
                        )
                    )
                    continue
                items.append(
                    ChecklistItem(
                        id=item_id,
                        proposition=proposition,
                        sources=sources,
                        kind="deterministic",
                        group=check.group,
                        p=0.0,
                        passed=False,
                        detail=f"check unusable: {error}",
                    )
                )
                continue
            items.append(
                ChecklistItem(
                    id=item_id,
                    proposition=proposition,
                    sources=sources,
                    kind="deterministic",
                    group=check.group,
                    p=1.0 if outcome.passed else 0.0,
                    passed=outcome.passed,
                    detail=outcome.detail,
                )
            )
    return items, fallbacks


def _pending_questions(
    config: ChecklistConfig,
    outputs: Mapping[str, str],
) -> list[_PendingQuestion]:
    pending = []
    for question in config.questions:
        for binding in _bindings(question.foreach, outputs):
            proposition = render(question.proposition, binding)
            pending.append(
                _PendingQuestion(
                    item_id=render(question.id, binding),
                    proposition=proposition,
                    sources=_sources(binding, question.sources_key),
                    group=question.group,
                    payload=_question_payload(config, question, binding, proposition),
                    expect=question.expect,
                    threshold=(
                        config.threshold if question.threshold is None else question.threshold
                    ),
                    members=_members(binding),
                )
            )
    return pending


def _read_probabilities(body: bytes, keys: Sequence[str]) -> dict[str, float]:
    try:
        answers = json.loads(body)["answers"]
    except (ValueError, KeyError, TypeError) as error:
        raise ChecklistUnavailable("judge_unavailable", "malformed System One reply") from error
    probabilities = {}
    for key in keys:
        answer = answers.get(key) if isinstance(answers, Mapping) else None
        value = answer.get("noul") if isinstance(answer, Mapping) else None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ChecklistUnavailable("judge_unavailable", f"no probability for {key!r}")
        probabilities[key] = float(value)
    return probabilities


async def _decide(
    backend: DecisionBackend,
    config: ChecklistConfig,
    state: dict[str, object],
    questions: Sequence[_PendingQuestion],
) -> tuple[list[float], tuple[int, int], int]:
    """Ask every question; returns P(yes) per question, usage, and reads."""

    from kairyu.engine.systemone import (
        SystemOneCapacityError,
        SystemOneUnavailableError,
    )

    chunks = [
        list(range(start, min(start + config.max_questions_per_call, len(questions))))
        for start in range(0, len(questions), config.max_questions_per_call)
    ]

    async def one(indices: list[int]) -> tuple[dict[str, float], tuple[int, int]]:
        keys = [f"q{index}" for index in indices]
        body: dict[str, object] = {
            "model": "",
            "state": state,
            "questions": {
                key: dict(questions[index].payload)
                for key, index in zip(keys, indices, strict=True)
            },
        }
        for name in ("samples", "think", "steps"):
            value = getattr(config, name)
            if value is not None:
                body[name] = value
        try:
            reply = await backend.decide(body)
        except (SystemOneCapacityError, SystemOneUnavailableError) as error:
            raise ChecklistUnavailable("judge_unavailable", str(error)) from error
        if reply.status != 200:
            raise ChecklistUnavailable("judge_unavailable", f"System One status {reply.status}")
        return _read_probabilities(reply.body, keys), (
            reply.input_tokens or 0,
            reply.output_tokens or 0,
        )

    results = await asyncio.gather(*(one(chunk) for chunk in chunks))
    merged: dict[str, float] = {}
    prompt_tokens = completion_tokens = 0
    for probabilities, (inputs, outputs) in results:
        merged.update(probabilities)
        prompt_tokens += inputs
        completion_tokens += outputs
    return (
        [merged[f"q{index}"] for index in range(len(questions))],
        (prompt_tokens, completion_tokens),
        len(chunks),
    )


def _aggregate(
    questions: Sequence[_PendingQuestion],
    yes_probabilities: Sequence[float],
) -> list[ChecklistItem]:
    grouped: dict[tuple[str, str], list[tuple[_PendingQuestion, float]]] = {}
    for question, p_yes in zip(questions, yes_probabilities, strict=True):
        p = p_yes if question.expect == "yes" else 1.0 - p_yes
        grouped.setdefault((question.group, question.item_id), []).append((question, p))
    items = []
    for (group, item_id), members in grouped.items():
        first = members[0][0]
        p = min(value for _question, value in members)
        sources: list[str] = []
        for question, _value in members:
            sources.extend(source for source in question.sources if source not in sources)
        items.append(
            ChecklistItem(
                id=item_id,
                proposition=first.proposition,
                sources=tuple(sources),
                kind="semantic",
                group=group,
                p=p,
                passed=p >= first.threshold,
                members=first.members,
            )
        )
    return items


def feedback_text(config: ChecklistConfig, items: Sequence[ChecklistItem]) -> str:
    failing = [item for item in items if not item.passed and item.judged]
    if not failing:
        return "PASS"
    lines = ["FAIL", config.feedback_header]
    for item in failing:
        lines.append(
            render(
                config.feedback_item,
                {
                    "id": item.id,
                    "proposition": item.proposition,
                    "sources": ", ".join(item.sources) or "-",
                    "p": f"{item.p:.2f}",
                    "detail": f" — {item.detail}" if item.detail else "",
                    "group": item.group,
                },
            )
        )
    return "\n".join(lines)


class ChecklistRun:
    """One verification attempt, split so inline roles run between stages."""

    def __init__(
        self,
        config: ChecklistConfig,
        *,
        target_text: str,
        sources: str,
    ) -> None:
        self._config = config
        self._target_text = target_text
        self._sources = sources
        self._items: list[ChecklistItem] = []
        self._fallbacks: list[_PendingQuestion] = []

    def _context(self, outputs: Mapping[str, str]) -> CheckContext:
        return CheckContext(text=self._target_text, sources=self._sources, outputs=outputs)

    def checks(self, stage: str, outputs: Mapping[str, str]) -> bool:
        """Run one check stage; True when every deterministic item passed."""

        items, fallbacks = _run_checks(self._config, stage, self._context(outputs), outputs)
        self._items.extend(items)
        self._fallbacks.extend(fallbacks)
        return all(item.passed for item in self._items)

    def early_verdict(self, outputs: Mapping[str, str]) -> ChecklistVerdict:
        """FAIL on deterministic failures alone, without any model read.

        The questions that were not read are still listed (unjudged), so the
        published report shows the whole checklist; questions over a list no
        role has produced yet (for example claims) are omitted.
        """

        unjudged: dict[tuple[str, str], ChecklistItem] = {}
        for question in self._config.questions:
            try:
                pending = _pending_questions(
                    replace(self._config, questions=(question,)), outputs
                )
            except ChecklistUnavailable:
                continue
            for entry in [*self._fallbacks, *pending]:
                unjudged.setdefault(
                    (entry.group, entry.item_id),
                    ChecklistItem(
                        id=entry.item_id,
                        proposition=entry.proposition,
                        sources=entry.sources,
                        kind="semantic",
                        group=entry.group,
                        p=0.0,
                        passed=False,
                        detail="not judged: a deterministic check failed",
                        members=entry.members,
                        judged=False,
                    ),
                )
        items = (*self._items, *unjudged.values())
        return ChecklistVerdict(
            passed=False,
            items=items,
            checks_passed=False,
            text=feedback_text(self._config, items),
        )

    async def decide(
        self,
        backend: DecisionBackend | None,
        outputs: Mapping[str, str],
        query: str,
    ) -> ChecklistVerdict:
        config = self._config
        pending = [*self._fallbacks, *_pending_questions(config, outputs)]
        checks_passed = all(item.passed for item in self._items)
        semantic: list[ChecklistItem] = []
        usage = (0, 0)
        reads = 0
        if pending:
            if backend is None:
                raise ChecklistUnavailable("judge_unavailable", "no decision backend")
            if len(pending) > config.max_questions:
                raise ChecklistUnavailable(
                    "checklist_unavailable",
                    f"{len(pending)} questions exceed the limit of {config.max_questions}",
                )
            state = build_state(config.state, outputs, query)
            size = len(json.dumps(state, ensure_ascii=False))
            if config.max_state_chars is not None and size > config.max_state_chars:
                raise ChecklistUnavailable(
                    "checklist_unavailable",
                    f"state of {size} characters exceeds {config.max_state_chars}",
                )
            yes, usage, reads = await _decide(backend, config, state, pending)
            semantic = _aggregate(pending, yes)
        items = (*self._items, *semantic)
        return ChecklistVerdict(
            passed=all(item.passed for item in items),
            items=items,
            checks_passed=checks_passed,
            text=feedback_text(config, items),
            usage=usage,
            reads=reads,
        )


def _cut(text: str, limit: int | None) -> str:
    if limit is None or len(text) <= limit:
        return text
    return f"{text[:limit]}\n[... {len(text) - limit} more characters cut ...]"


def build_state(
    sections: Sequence[StateSection],
    outputs: Mapping[str, str],
    query: str,
) -> dict[str, object]:
    """The System One state object: one field per section.

    The request is unwrapped to its role-tagged messages when the query is
    Kairyu's chat transcript; role outputs that are JSON are embedded as JSON
    values, so the judge reads structure instead of escaped text and the
    material it judges cannot pose as instructions.
    """

    state: dict[str, object] = {}
    for section in sections:
        if section.source == "query":
            messages = conversation_messages(query)
            if messages is not None:
                state[section.key] = [
                    {**message, "content": _cut(message["content"], section.max_chars)}
                    if isinstance(message, dict) and isinstance(message.get("content"), str)
                    else message
                    for message in messages
                ]
                continue
            raw = query
        else:
            raw = outputs.get(section.source, "")
            try:
                state[section.key] = parse_json_output(raw)
                continue
            except ValueError:
                pass
        state[section.key] = _cut(raw, section.max_chars)
    return state


def curate(config: CurationConfig, text: str, items: Sequence[ChecklistItem]) -> str:
    """Apply drop / merge / pad to the target's JSON list; returns JSON text."""

    try:
        document = parse_json_output(text)
        entries = json_path(document, config.items_path)
    except (ValueError, KeyError):
        return text
    if not isinstance(document, dict) or not isinstance(entries, list):
        return text
    kept = [dict(entry) for entry in entries if isinstance(entry, Mapping)]
    by_id = {str(entry.get(config.id_key)): entry for entry in kept}
    if config.drop_group:
        dropped = {
            item.id
            for item in items
            if item.group == config.drop_group and item.p < config.drop_below
        }
        kept = [entry for entry in kept if str(entry.get(config.id_key)) not in dropped]
        by_id = {str(entry.get(config.id_key)): entry for entry in kept}
    if config.merge_group:
        for item in items:
            if item.group != config.merge_group or item.p >= config.merge_below:
                continue
            if len(item.members) != 2:
                continue
            first, second = (by_id.get(member) for member in item.members)
            if first is None or second is None or first is second:
                continue
            merged_sources = list(first.get(config.sources_key) or [])
            for source in second.get(config.sources_key) or []:
                if source not in merged_sources:
                    merged_sources.append(source)
            first[config.sources_key] = merged_sources
            first[config.proposition_key] = (
                f"{first.get(config.proposition_key, '')}; "
                f"{second.get(config.proposition_key, '')}"
            )
            kept = [entry for entry in kept if entry is not second]
            by_id[str(item.members[1])] = first
    if config.units_path:
        try:
            units = json_path(document, config.units_path)
        except KeyError:
            units = []
        cited = {
            str(source)
            for entry in kept
            for source in (entry.get(config.sources_key) or [])
            if isinstance(entry.get(config.sources_key), list)
        }
        for unit in units if isinstance(units, list) else []:
            if not isinstance(unit, Mapping):
                continue
            unit_id = str(unit.get(config.unit_id_key))
            if unit_id in cited:
                continue
            padded = {
                key: (render(value, {"unit": unit}) if isinstance(value, str) else value)
                for key, value in config.pad.items()
            }
            padded[config.sources_key] = [unit.get(config.unit_id_key)]
            kept.append(padded)
            cited.add(unit_id)
    _set_path(document, config.items_path, kept)
    return json.dumps(document, ensure_ascii=False)


def _set_path(document: dict, path: str, value: object) -> None:
    keys = [key for key in path.split(".") if key]
    current: object = document
    for key in keys[:-1]:
        assert isinstance(current, dict)
        current = current[key]
    assert isinstance(current, dict) and keys
    current[keys[-1]] = value


def unverified_report(reason: str, attempts: int, threshold: float | None) -> VerificationReport:
    return VerificationReport(
        guaranteed=False,
        reason=reason,
        threshold=threshold,
        attempts=attempts,
    )


def verdict_report(
    verdict: ChecklistVerdict,
    *,
    threshold: float,
    attempts: int,
) -> VerificationReport:
    return VerificationReport(
        guaranteed=verdict.passed,
        reason=None if verdict.passed else "refinement_limit",
        threshold=threshold,
        items=verdict.items,
        attempts=attempts,
    )


__all__ = [
    "ChecklistCheck",
    "ChecklistConfig",
    "ChecklistItem",
    "ChecklistQuestion",
    "ChecklistRun",
    "ChecklistUnavailable",
    "ChecklistVerdict",
    "CurationConfig",
    "DecisionBackend",
    "ItemSource",
    "StateSection",
    "build_state",
    "requirement_question",
    "VerificationReport",
    "curate",
    "feedback_text",
    "unverified_report",
    "verdict_report",
]
