"""POST /v1/systemone: the System One (Jev) wire API in front of HTTPSystemOneBackend.

Kairyu resolves the model (or one of its aliases), bounds the work it admits,
meters ``usage`` and forwards the body. The upstream owns the question schema,
so its 400/422 answers pass through unchanged. Errors Kairyu raises itself use
Jev's shapes: a FastAPI-style ``{"detail": [...]}`` list for a malformed body,
``{"detail": {"error_type", "message"}}`` otherwise.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from kairyu.engine.systemone import (
    HTTPSystemOneBackend,
    SystemOneCapacityError,
    SystemOneUnavailableError,
)
from kairyu.entrypoints.server.metering import record_state_usage

SYSTEMONE_PATH = "/v1/systemone"
# Tokens reserved per image before the upstream reports exact usage; a
# DiffusionGemma image costs about 280 input tokens.
_IMAGE_TOKEN_BOUND = 1024
# System prompt and answer scaffold around each read, beyond the state and
# the group's own questions (OpenJev bills about 100 for three questions).
_READ_OVERHEAD_TOKENS = 512
# One earlier answer as a sequential read repeats it ("id: label").
_ANSWER_TOKEN_BOUND = 64
# Options whose number sizes the work; normalized before reserving.
_COUNT_OPTIONS = ("samples", "think", "steps")


@dataclass(frozen=True)
class SystemOneModel:
    name: str
    backend: HTTPSystemOneBackend
    aliases: frozenset[str] = frozenset()
    description: str | None = None
    max_questions: int = 256
    max_body_bytes: int = 64 * 1024 * 1024


def wants_jev_envelope(path: str) -> bool:
    return path == SYSTEMONE_PATH


def jev_error_payload(error_type: str, message: str) -> dict:
    return {"detail": {"error_type": error_type, "message": message}}


def jev_error_type_for_status(status: int) -> str:
    return {
        401: "authentication_error",
        403: "permission_error",
        413: "api_usage_error",
        429: "rate_limit_error",
    }.get(status, "api_error")


def _error(status: int, error_type: str, message: str, retry_after: str | None = None):
    headers = {"retry-after": retry_after} if retry_after else None
    return JSONResponse(jev_error_payload(error_type, message), status_code=status, headers=headers)


def _invalid(loc: list, kind: str, message: str) -> JSONResponse:
    return JSONResponse({"detail": [{"type": kind, "loc": loc, "msg": message}]}, status_code=422)


_WHOLE_NUMBER = re.compile(r"\s*([+-]?\d+)(\.0*)?\s*")


def _whole_number(value: object) -> int | None:
    """``value`` as an int when a Pydantic int field would accept it, else None."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    match = _WHOLE_NUMBER.fullmatch(value) if isinstance(value, str) else None
    if match is not None:
        # the digits themselves, never through float (which overflows to inf);
        # int() refuses strings past Python's digit limit with ValueError
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _normalize_counts(body: dict) -> JSONResponse | None:
    """Rewrite the resource-sizing options as ints, or refuse them before dispatch.

    The upstream parses "32" or 32.0 as 32, so the reservation must see the
    same number it will run; anything else is refused rather than forwarded.
    """

    for name in _COUNT_OPTIONS:
        if body.get(name) is None:
            continue
        number = _whole_number(body[name])
        if number is None:
            return _invalid(["body", name], "int_parsing", "Input should be a valid integer")
        body[name] = number
    return None


def _work_bound(body: dict) -> int:
    """An upper bound on the tokens one request can bill, before dispatch.

    UTF-8 bytes bound tokens. A server may read every question in its own
    group: each group's prompt carries the state, and with ``think`` each
    group writes a thought, reads its prompt once to do so, and then reads
    prompt + thought once per sample. A ``sequential`` read carries every
    question and the answers so far in each group's prompt. Re-reads a server
    adds by its own policy are not billed (OpenJev), so they are not reserved.
    """

    images = body.get("images")
    state = len(json.dumps(body.get("state"), ensure_ascii=False).encode())
    state += _IMAGE_TOKEN_BOUND * (len(images) if isinstance(images, list) else 0)
    questions = body.get("questions")
    sizes = (
        [len(json.dumps(q, ensure_ascii=False).encode()) for q in questions.values()]
        if isinstance(questions, dict) and questions
        else [0]
    )
    if body.get("sequential") in (None, False):
        prompts = sum(_READ_OVERHEAD_TOKENS + state + size for size in sizes)
    else:  # any other value may mean sequential to the upstream
        schema = sum(sizes) + _ANSWER_TOKEN_BOUND * len(sizes)
        prompts = len(sizes) * (_READ_OVERHEAD_TOKENS + state + schema)
    reads = max(1, body.get("samples") or 1)
    think = max(0, body.get("think") or 0)
    input_tokens = (reads + (1 if think else 0)) * prompts + reads * len(sizes) * think
    return input_tokens + len(sizes) * think


def systemone_model_cards(models: Mapping[str, SystemOneModel]) -> list[dict]:
    """Jev's ``/v1/models`` entries: every name a request may use."""

    cards = []
    for model in models.values():
        for name in (model.name, *sorted(model.aliases)):
            card = {"name": name}
            if model.description:
                card["description"] = model.description
            cards.append(card)
    return cards


def add_systemone_route(app: FastAPI, models: Mapping[str, SystemOneModel]) -> None:
    by_name = {name: model for model in models.values() for name in (model.name, *model.aliases)}

    @app.post(SYSTEMONE_PATH)
    async def systemone(http_request: Request):
        raw = await http_request.body()
        try:
            body = json.loads(raw)
        except ValueError:
            return _invalid(["body"], "json_invalid", "JSON decode error")
        if not isinstance(body, dict):
            return _invalid(["body"], "model_attributes_type", "Input should be an object")
        name = body.get("model")
        if not isinstance(name, str):
            return _invalid(["body", "model"], "string_type", "Input should be a valid string")
        model = by_name.get(name)
        if model is None:
            return _error(400, "api_usage_error", f"Unknown model: {name}")
        http_request.state.model = model.name
        # the middleware bounds the largest model's limit; each model keeps its own
        if len(raw) > model.max_body_bytes:
            return _error(
                413, "api_usage_error", f"request body is larger than {model.max_body_bytes} bytes"
            )
        questions = body.get("questions")
        if isinstance(questions, dict) and len(questions) > model.max_questions:
            return JSONResponse(
                {"detail": f"at most {model.max_questions} questions per request"},
                status_code=400,
            )
        refused = _normalize_counts(body)  # the reserved and the forwarded numbers agree
        if refused is not None:
            return refused
        owner = getattr(http_request.state, "tenant", None) or "default"
        admission = getattr(http_request.state, "tenant_admission", None)
        if admission is not None:
            admitted = admission.reserve_tokens(_work_bound(body), refundable_on_exact_usage=True)
            metrics = getattr(http_request.app.state, "metrics", None)
            if metrics is not None:
                metrics.record_tenant_admission(
                    owner, source="http", admitted=admitted, reason=admission.reason
                )
            if not admitted:
                return _error(
                    429,
                    "rate_limit_error",
                    f"tenant {owner!r} admission limit exceeded ({admission.reason})",
                    retry_after="1",
                )
            http_request.state.tenant_metric_admitted = True
        try:
            reply = await model.backend.decide(body)
        except SystemOneCapacityError as error:
            return _error(429, "rate_limit_error", str(error), retry_after="1")
        except SystemOneUnavailableError as error:
            return _error(503, "api_error", f"inference backend unavailable: {error}", "2")
        if reply.status == 200:
            if reply.input_tokens is None or reply.output_tokens is None:
                # an answer that cannot be metered is not passed on unbilled
                return _error(502, "api_error", "upstream answer has no valid usage")
            # Only answered reads are billed; any other outcome leaves the
            # reservation undispatched, so the tenant middleware refunds it.
            if admission is not None:
                admission.mark_dispatched()
            record_state_usage(
                http_request.app.state,
                tenant=owner,
                model=model.name,
                prompt_tokens=reply.input_tokens,
                completion_tokens=reply.output_tokens,
                reservation=admission,
            )
        return Response(reply.body, status_code=reply.status, headers=reply.headers)
