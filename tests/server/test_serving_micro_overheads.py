"""Regression coverage for issue #349 serving micro-overheads."""

from __future__ import annotations

import asyncio

from kairyu.entrypoints.server.middleware import (
    ChatBodyLimitMiddleware,
    _template_path,
)


def test_metrics_path_template_cache_is_bounded() -> None:
    _template_path.cache_clear()
    try:
        assert (
            _template_path("/v1/requests/req-0123456789abcdef01234567")
            == "/v1/requests/{id}"
        )
        assert (
            _template_path("/v1/requests/attacker-random/result")
            == "/v1/requests/{id}/result"
        )
        paths = [f"/v1/files/file-{index:04d}" for index in range(1025)]
        for path in paths:
            assert _template_path(path) == "/v1/files/{id}"

        assert _template_path.cache_info().currsize == 1024
        hits = _template_path.cache_info().hits
        assert _template_path(paths[-1]) == "/v1/files/{id}"
        assert _template_path.cache_info().hits == hits + 1
    finally:
        _template_path.cache_clear()


async def test_chat_body_limit_forwards_chunks_without_buffering() -> None:
    first_consumed = asyncio.Event()
    received: list[bytes] = []
    messages = iter(
        (
            {"type": "http.request", "body": b"first", "more_body": True},
            {"type": "http.request", "body": b"second", "more_body": False},
        )
    )

    async def receive() -> dict:
        message = next(messages)
        if message["body"] == b"second":
            await first_consumed.wait()
        return message

    async def app(_scope: dict, app_receive, send) -> None:
        first = await app_receive()
        received.append(first["body"])
        first_consumed.set()
        second = await app_receive()
        received.append(second["body"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    middleware = ChatBodyLimitMiddleware(app, limit=32)
    await asyncio.wait_for(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [],
            },
            receive,
            send,
        ),
        timeout=1,
    )

    assert received == [b"first", b"second"]
    assert sent[0]["status"] == 200
