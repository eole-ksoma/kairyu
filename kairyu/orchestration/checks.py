"""Deterministic answer checks for checklist verifiers (L2, no model call).

A check is one named primitive with JSON-like parameters. The orchestration
policy (which checks apply, with which parameters) lives in the DSL: either as
static checks or as checks an upstream role emitted in its JSON output. This
module only owns the primitive library and how one primitive judges a text.

Every primitive returns a :class:`CheckOutcome`. ``passed`` is a strict bool:
there is no partial credit, so the deterministic half of a checklist never
needs a probability threshold.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass

# A JSON value as produced by json.loads.
JSONValue = object


@dataclass(frozen=True)
class CheckOutcome:
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class CheckContext:
    """What a primitive may read.

    ``text`` is the attempt under verification, ``sources`` the request
    material the answer may rely on (conversation, attachments, tool results),
    and ``outputs`` the completed role outputs of this run.
    """

    text: str
    sources: str
    outputs: Mapping[str, str]


class CheckParameterError(ValueError):
    """A primitive was given parameters it cannot evaluate."""


_TYPOGRAPHY = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}
)


def _whitespace_normalized(text: str) -> str:
    """Compare text the way a reader would: Unicode compatibility forms,
    straight quotes and dashes, and collapsed whitespace."""

    return " ".join(unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHY).split())


_CODE = re.compile(r"```.*?```|`[^`\n]+`", re.DOTALL)


_JSON_ESCAPE = re.compile(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})')
_ESCAPED = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


def _unescaped(match: re.Match[str]) -> str:
    code = match.group(1)
    if code.startswith("u"):
        return chr(int(code[1:], 16))
    return _ESCAPED.get(code, code)


def _source_corpus(sources: str) -> str:
    """Whitespace-normalized sources, plus their JSON-unescaped reading.

    Orchestrated requests carry the conversation as JSON, so material text
    appears with JSON string escapes; a verbatim quote must match either form.
    """

    normalized = _whitespace_normalized(sources)
    unescaped = _whitespace_normalized(_JSON_ESCAPE.sub(_unescaped, sources))
    return normalized if unescaped == normalized else f"{normalized}\n{unescaped}"


def parse_json_output(text: str) -> JSONValue:
    """Parse a role's JSON output, tolerating one surrounding code fence."""

    stripped = text.strip()
    fence = re.fullmatch(r"```[A-Za-z0-9_-]*\s*\n(.*)\n```", stripped, re.DOTALL)
    if fence is not None:
        stripped = fence.group(1).strip()
    return json.loads(stripped)


def json_path(value: JSONValue, path: str) -> JSONValue:
    """Follow a dotted key path through JSON objects (empty path = value)."""

    current = value
    for key in filter(None, path.split(".")):
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(path)
        current = current[key]
    return current


def _require(params: Mapping[str, object], key: str, kind: type | tuple[type, ...]) -> object:
    value = params.get(key)
    if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
        raise CheckParameterError(f"parameter {key!r} must be {kind}")
    return value


def _optional_int(params: Mapping[str, object], key: str) -> int | None:
    value = params.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckParameterError(f"parameter {key!r} must be a non-negative integer")
    return value


_REGEX_FLAGS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}


def _compiled(params: Mapping[str, object]) -> re.Pattern[str]:
    pattern = _require(params, "pattern", str)
    flags_text = params.get("flags", "")
    if not isinstance(flags_text, str) or set(flags_text) - _REGEX_FLAGS.keys():
        raise CheckParameterError("parameter 'flags' may only contain i, m, s")
    flags = 0
    for letter in flags_text:
        flags |= _REGEX_FLAGS[letter]
    try:
        return re.compile(str(pattern), flags)
    except re.error as error:
        raise CheckParameterError(f"invalid pattern: {error}") from error


def _regex(ctx: CheckContext, params: Mapping[str, object]) -> CheckOutcome:
    match = _compiled(params).search(ctx.text)
    return CheckOutcome(match is not None, "" if match else "pattern not found")


def _not_regex(ctx: CheckContext, params: Mapping[str, object]) -> CheckOutcome:
    match = _compiled(params).search(ctx.text)
    if match is None:
        return CheckOutcome(True)
    return CheckOutcome(False, f"forbidden text found: {match.group(0)[:80]!r}")


def _length(ctx: CheckContext, params: Mapping[str, object]) -> CheckOutcome:
    bounds = {
        key: _optional_int(params, key)
        for key in ("min_chars", "max_chars", "min_words", "max_words")
    }
    if all(value is None for value in bounds.values()):
        raise CheckParameterError("length needs at least one bound")
    chars = len(ctx.text.strip())
    words = len(ctx.text.split())
    problems = []
    if bounds["min_chars"] is not None and chars < bounds["min_chars"]:
        problems.append(f"{chars} characters < {bounds['min_chars']}")
    if bounds["max_chars"] is not None and chars > bounds["max_chars"]:
        problems.append(f"{chars} characters > {bounds['max_chars']}")
    if bounds["min_words"] is not None and words < bounds["min_words"]:
        problems.append(f"{words} words < {bounds['min_words']}")
    if bounds["max_words"] is not None and words > bounds["max_words"]:
        problems.append(f"{words} words > {bounds['max_words']}")
    return CheckOutcome(not problems, "; ".join(problems))


def _contains_args(params: Mapping[str, object]) -> tuple[str, bool]:
    needle = _require(params, "text", str)
    if not needle:
        raise CheckParameterError("parameter 'text' must be non-empty")
    ignore_case = params.get("ignore_case", False)
    if not isinstance(ignore_case, bool):
        raise CheckParameterError("parameter 'ignore_case' must be a bool")
    return str(needle), ignore_case


def _found(ctx: CheckContext, params: Mapping[str, object]) -> bool:
    needle, ignore_case = _contains_args(params)
    haystack = ctx.text
    if ignore_case:
        return needle.casefold() in haystack.casefold()
    return needle in haystack


def _contains(ctx: CheckContext, params: Mapping[str, object]) -> CheckOutcome:
    found = _found(ctx, params)
    return CheckOutcome(found, "" if found else f"missing {params['text']!r}")


def _not_contains(ctx: CheckContext, params: Mapping[str, object]) -> CheckOutcome:
    found = _found(ctx, params)
    return CheckOutcome(not found, f"forbidden {params['text']!r} present" if found else "")


def _json_valid(ctx: CheckContext, params: Mapping[str, object]) -> CheckOutcome:
    required = params.get("required_keys", [])
    if not isinstance(required, list) or not all(isinstance(key, str) for key in required):
        raise CheckParameterError("parameter 'required_keys' must be a list of strings")
    try:
        value = parse_json_output(ctx.text)
    except ValueError as error:
        return CheckOutcome(False, f"not valid JSON: {error}")
    if required:
        if not isinstance(value, Mapping):
            return CheckOutcome(False, "JSON value is not an object")
        missing = [key for key in required if key not in value]
        if missing:
            return CheckOutcome(False, f"missing keys {missing}")
    return CheckOutcome(True)


# Quoted spans: CJK corner brackets and straight/curly double quotes. A JSON
# answer's straight quotes delimit keys and strings, not quotations.
_QUOTED = re.compile(r"「([^「」]+)」|“([^“”]+)”|\"([^\"\n]+)\"")
_QUOTED_NOT_JSON = re.compile(r"「([^「」]+)」|“([^“”]+)”")


def _quotes_in_sources(ctx: CheckContext, params: Mapping[str, object]) -> CheckOutcome:
    min_chars = _optional_int(params, "min_chars")
    floor = 8 if min_chars is None else min_chars
    sources = _source_corpus(ctx.sources)
    try:
        parse_json_output(ctx.text)
        pattern = _QUOTED_NOT_JSON
    except ValueError:
        pattern = _QUOTED
    missing = []
    # String literals inside code are not quotations.
    for match in pattern.finditer(_CODE.sub(" ", ctx.text)):
        quote = next(group for group in match.groups() if group is not None)
        normalized = _whitespace_normalized(quote)
        if len(normalized) >= floor and normalized not in sources:
            missing.append(normalized[:80])
    if missing:
        return CheckOutcome(False, f"quotes not found in the sources: {missing[:5]}")
    return CheckOutcome(True)


_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    found = set()
    for match in _NUMBER.finditer(text):
        token = match.group(0).lstrip("+").replace(",", "")
        if token.endswith("."):
            token = token[:-1]
        found.add(token)
    return found


def _numbers_in_sources(ctx: CheckContext, params: Mapping[str, object]) -> CheckOutcome:
    ignore_below = params.get("ignore_below", 10)
    if isinstance(ignore_below, bool) or not isinstance(ignore_below, (int, float)):
        raise CheckParameterError("parameter 'ignore_below' must be a number")
    allowed = _numbers(_source_corpus(ctx.sources))
    unsourced = []
    for token in sorted(_numbers(ctx.text)):
        try:
            magnitude = abs(float(token))
        except ValueError:
            continue
        if magnitude < ignore_below:
            continue
        if token not in allowed:
            unsourced.append(token)
    if unsourced:
        return CheckOutcome(False, f"numbers absent from the sources: {unsourced[:10]}")
    return CheckOutcome(True)


def selected_items(
    outputs: Mapping[str, str],
    params: Mapping[str, object],
) -> list[Mapping[str, object]]:
    """The JSON objects ``params`` selects from another role's output."""

    role = _require(params, "role", str)
    path = params.get("path", "")
    where = params.get("where", {})
    if not isinstance(path, str) or not isinstance(where, Mapping):
        raise CheckParameterError("'path' must be a string and 'where' a mapping")
    raw = outputs.get(str(role))
    if raw is None:
        raise CheckParameterError(f"role {role!r} produced no output")
    try:
        items = json_path(parse_json_output(raw), path)
    except (ValueError, KeyError) as error:
        raise CheckParameterError(f"{role!r} output has no JSON list at {path!r}") from error
    if not isinstance(items, list):
        raise CheckParameterError(f"{role!r} output at {path!r} is not a list")
    return [
        item
        for item in items
        if isinstance(item, Mapping)
        and all(item.get(key) == value for key, value in where.items())
    ]


def _items_in_sources(ctx: CheckContext, params: Mapping[str, object]) -> CheckOutcome:
    key = _require(params, "key", str)
    min_chars = _optional_int(params, "min_chars") or 1
    sources = _source_corpus(ctx.sources)
    missing = []
    for item in selected_items(ctx.outputs, params):
        value = item.get(str(key))
        if not isinstance(value, str) or len(value.strip()) < min_chars:
            missing.append(str(item.get("id", "?")))
            continue
        if _whitespace_normalized(value) not in sources:
            missing.append(str(item.get("id", value[:40])))
    if missing:
        return CheckOutcome(False, f"items without verbatim support in the sources: {missing[:10]}")
    return CheckOutcome(True)


def _coverage(ctx: CheckContext, params: Mapping[str, object]) -> CheckOutcome:
    """Every unit id in the attempt's JSON is cited by at least one item."""

    units_path = _require(params, "units_path", str)
    items_path = _require(params, "items_path", str)
    key = _require(params, "key", str)
    unit_key = params.get("unit_key", "id")
    try:
        document = parse_json_output(ctx.text)
        units = json_path(document, str(units_path))
        items = json_path(document, str(items_path))
    except (ValueError, KeyError) as error:
        return CheckOutcome(False, f"output is not the expected JSON: {error}")
    if not isinstance(units, list) or not isinstance(items, list):
        return CheckOutcome(False, "units and items must be lists")
    cited: set[object] = set()
    for item in items:
        refs = item.get(str(key)) if isinstance(item, Mapping) else None
        if isinstance(refs, list):
            cited.update(ref for ref in refs if isinstance(ref, (str, int)))
    uncovered = [
        unit.get(unit_key)
        for unit in units
        if isinstance(unit, Mapping) and unit.get(unit_key) not in cited
    ]
    if uncovered:
        return CheckOutcome(False, f"uncovered units: {uncovered}")
    return CheckOutcome(True)


PRIMITIVES: dict[str, Callable[[CheckContext, Mapping[str, object]], CheckOutcome]] = {
    "regex": _regex,
    "not_regex": _not_regex,
    "length": _length,
    "contains": _contains,
    "not_contains": _not_contains,
    "json_valid": _json_valid,
    "quotes_in_sources": _quotes_in_sources,
    "numbers_in_sources": _numbers_in_sources,
    "items_in_sources": _items_in_sources,
    "coverage": _coverage,
}


def run_check(
    primitive: str,
    params: Mapping[str, object],
    ctx: CheckContext,
) -> CheckOutcome:
    """Evaluate one primitive; unknown names and bad parameters raise."""

    try:
        function = PRIMITIVES[primitive]
    except KeyError:
        raise CheckParameterError(f"unknown check primitive {primitive!r}") from None
    if not isinstance(params, Mapping):
        raise CheckParameterError("check parameters must be a mapping")
    return function(ctx, params)


def static_check_is_valid(primitive: str, params: Mapping[str, object]) -> None:
    """Load-time validation of a statically configured check.

    Evaluates the primitive on empty input so parameter errors surface at
    startup. Primitives that read another role's output are only checked for
    their own required keys here.
    """

    if primitive not in PRIMITIVES:
        raise CheckParameterError(f"unknown check primitive {primitive!r}")
    if primitive == "items_in_sources":
        _require(params, "role", str)
        _require(params, "key", str)
        return
    run_check(primitive, params, CheckContext(text="", sources="", outputs={}))
