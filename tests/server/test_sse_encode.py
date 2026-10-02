"""Golden-wire coverage for allocation-light outgoing SSE encoders."""

from __future__ import annotations

import json

import pytest
from pydantic_core import PydanticSerializationError

import kairyu.entrypoints.server.sse_encode as sse_encode
from kairyu.entrypoints.server.app import _chat_content_sse_chunk
from kairyu.entrypoints.server.protocol import (
    ChatCompletionChunk,
    ChoiceLogprobs,
    ChunkChoice,
    ChunkDelta,
    CompletionChoice,
    CompletionChunk,
)
from kairyu.entrypoints.server.responses_service import _sse
from kairyu.entrypoints.server.sse_encode import (
    ChatContentSSEEncoder,
    CompletionTextSSEEncoder,
    ResponsesTextDeltaSSEEncoder,
)
from kairyu.sse import escape_json_line_separators

_CHAT_EXTENSIONS = frozenset(
    {"kairyu_trace", "kairyu_trace_v2", "kairyu_route", "kairyu_verification"}
)
_CONTENT_CASES = (
    "",
    "token",
    'quote" slash\\ line\nreturn\rtab\t',
    "日本語 😀",
    "before\u0085middle\u2028after\u2029",
    "\x00\x1f",
)


def test_chat_content_encoder_has_explicit_golden_wire() -> None:
    encoded = ChatContentSSEEncoder(
        "chatcmpl-id",
        123,
        "model",
        include_usage=False,
    ).encode(0, "tok")

    assert encoded == (
        b'data: {"id":"chatcmpl-id","object":"chat.completion.chunk",'
        b'"created":123,"model":"model","choices":[{"index":0,"delta":'
        b'{"role":null,"content":"tok","reasoning_content":null,"tool_calls":null},'
        b'"finish_reason":null,"logprobs":null}]}\n\n'
    )


def _chat_model_wire(
    response_id: str,
    created: int,
    model: str,
    index: int,
    content: str,
    *,
    include_usage: bool,
) -> bytes:
    payload = ChatCompletionChunk(
        id=response_id,
        created=created,
        model=model,
        choices=[
            ChunkChoice(
                index=index,
                delta=ChunkDelta(content=content),
            )
        ],
    )
    exclude = _CHAT_EXTENSIONS if include_usage else _CHAT_EXTENSIONS | frozenset({"usage"})
    serialized = escape_json_line_separators(payload.model_dump_json(exclude=exclude))
    return f"data: {serialized}\n\n".encode()


@pytest.mark.parametrize("include_usage", [False, True])
@pytest.mark.parametrize("index", [0, 7])
@pytest.mark.parametrize("content", _CONTENT_CASES)
def test_chat_content_encoder_matches_pydantic_golden_wire(
    include_usage: bool,
    index: int,
    content: str,
) -> None:
    encoder = ChatContentSSEEncoder(
        'chatcmpl-"id-日本\u0085',
        123,
        'model-"\\-日本\u2028\u2029',
        include_usage=include_usage,
    )

    encoded = encoder.encode(index, content)

    assert type(encoded) is bytes
    assert encoded == _chat_model_wire(
        'chatcmpl-"id-日本\u0085',
        123,
        'model-"\\-日本\u2028\u2029',
        index,
        content,
        include_usage=include_usage,
    )


def test_chat_fast_path_is_limited_to_repeated_roleless_content() -> None:
    encoder = ChatContentSSEEncoder("chatcmpl-id", 123, "model", include_usage=False)

    first = _chat_content_sse_chunk(
        encoder,
        "chatcmpl-id",
        123,
        "model",
        0,
        "first",
        is_first=True,
        include_usage=False,
        logprobs=None,
    )
    repeated = _chat_content_sse_chunk(
        encoder,
        "chatcmpl-id",
        123,
        "model",
        0,
        "next",
        is_first=False,
        include_usage=False,
        logprobs=None,
    )

    assert type(first) is str
    assert type(repeated) is bytes
    assert repeated == _chat_model_wire(
        "chatcmpl-id",
        123,
        "model",
        0,
        "next",
        include_usage=False,
    )


def test_chat_logprobs_stays_on_pydantic_fallback() -> None:
    encoder = ChatContentSSEEncoder("chatcmpl-id", 123, "model", include_usage=False)

    encoded = _chat_content_sse_chunk(
        encoder,
        "chatcmpl-id",
        123,
        "model",
        0,
        "token",
        is_first=False,
        include_usage=False,
        logprobs=ChoiceLogprobs(content=[]),
    )

    assert type(encoded) is str
    payload = json.loads(encoded.removeprefix("data: "))
    assert payload["choices"][0]["logprobs"] == {"content": []}


def test_chat_non_string_content_stays_on_pydantic_fallback() -> None:
    encoder = ChatContentSSEEncoder("chatcmpl-id", 123, "model", include_usage=False)

    encoded = _chat_content_sse_chunk(
        encoder,
        "chatcmpl-id",
        123,
        "model",
        0,
        b"token",
        is_first=False,
        include_usage=False,
        logprobs=None,
    )

    assert type(encoded) is str
    payload = json.loads(encoded.removeprefix("data: "))
    assert payload["choices"][0]["delta"]["content"] == "token"


def test_chat_encoder_serializes_only_dynamic_text_per_repeated_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original = sse_encode.to_json

    def counted(value: str) -> bytes:
        calls.append(value)
        return original(value)

    monkeypatch.setattr(sse_encode, "to_json", counted)
    encoder = ChatContentSSEEncoder("chatcmpl-id", 123, "model", include_usage=False)
    calls.clear()

    for _ in range(4096):
        encoder.encode(0, "tok")

    assert calls == ["tok"] * 4096
    assert len(encoder._prefixes) == 1


def test_json_string_bytes_keeps_no_match_serializer_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serialized = b'"token"'
    monkeypatch.setattr(sse_encode, "to_json", lambda value: serialized)

    assert sse_encode._json_string_bytes("token") is serialized


def test_fast_encoder_rejects_bool_index_and_invalid_unicode() -> None:
    encoder = ChatContentSSEEncoder("chatcmpl-id", 123, "model", include_usage=False)

    with pytest.raises(TypeError):
        encoder.encode(False, "token")
    with pytest.raises(PydanticSerializationError):
        encoder.encode(0, "\ud800")


def _completion_model_wire(
    response_id: str,
    created: int,
    model: str,
    index: int,
    text: str,
    *,
    include_usage: bool,
) -> bytes:
    payload = CompletionChunk(
        id=response_id,
        created=created,
        model=model,
        choices=[CompletionChoice(index=index, text=text)],
    )
    exclude = None if include_usage else frozenset({"usage"})
    serialized = escape_json_line_separators(payload.model_dump_json(exclude=exclude))
    return f"data: {serialized}\n\n".encode()


@pytest.mark.parametrize("include_usage", [False, True])
@pytest.mark.parametrize("index", [0, 11])
@pytest.mark.parametrize("text", _CONTENT_CASES)
def test_completion_text_encoder_matches_pydantic_golden_wire(
    include_usage: bool,
    index: int,
    text: str,
) -> None:
    encoder = CompletionTextSSEEncoder(
        'cmpl-"id-日本\u0085',
        456,
        'model-"\\-日本\u2028\u2029',
        include_usage=include_usage,
    )

    encoded = encoder.encode(index, text)

    assert type(encoded) is bytes
    assert encoded == _completion_model_wire(
        'cmpl-"id-日本\u0085',
        456,
        'model-"\\-日本\u2028\u2029',
        index,
        text,
        include_usage=include_usage,
    )


@pytest.mark.parametrize("sequence_number", [0, 17])
@pytest.mark.parametrize("delta", _CONTENT_CASES)
def test_responses_text_delta_encoder_matches_generic_golden_wire(
    sequence_number: int,
    delta: str,
) -> None:
    message_id = 'msg_"\\-日本\u0085\u2028\u2029'
    encoder = ResponsesTextDeltaSSEEncoder(message_id)

    encoded = encoder.encode(sequence_number, delta)
    generic = _sse(
        "response.output_text.delta",
        sequence_number,
        item_id=message_id,
        output_index=0,
        content_index=0,
        delta=delta,
        logprobs=[],
    ).encode()

    assert type(encoded) is bytes
    assert encoded == generic
    assert json.loads(encoded.split(b"data: ", 1)[1])["delta"] == delta
