"""System One (Jev wire API) forwarding backend.

A System One server answers ``POST /v1/systemone``: a ``state`` and typed
``questions`` in, per-question probabilities and ``usage`` out. TypeSafe's Jev,
OpenJev and Codiv serve the same wire API. Kairyu forwards a request to one
of the configured upstream replicas and owns admission in front of them, so an
overloaded upstream sees a bounded load and the caller gets a 429 before the
upstream would answer 529. The upstream owns the schema and its errors; they
pass through unchanged.

With several replicas a request goes to the one with the fewest requests in
flight. An unreachable replica or an overload/unavailable status moves the
request to another replica once; reads are side-effect free, so one retry is
safe (m11 D8 replica amendment).

This backend is never a ReplicaPool member: System One reads are not
generation requests, and an upstream 529 must not eject a chat replica that
shares the same server.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import httpx

# Upstream headers a System One client reads. Server-Timing carries the
# model/server split; retry-after belongs to 429/503/529.
_PASSED_HEADERS = ("content-type", "retry-after", "server-timing")
# Upstream statuses that say "this replica cannot serve now", not "this
# request is wrong": the read moves to another replica once.
_RETRY_ON_OTHER_REPLICA = frozenset({429, 502, 503, 504, 529})


class SystemOneCapacityError(RuntimeError):
    """Kairyu's admission queue for this model is full or the wait timed out."""


class SystemOneUnavailableError(RuntimeError):
    """The upstream could not be reached or did not answer in time."""


@dataclass(frozen=True)
class SystemOneReply:
    status: int
    body: bytes
    headers: dict[str, str]
    input_tokens: int | None = None
    output_tokens: int | None = None


class HTTPSystemOneBackend:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        upstream_model: str,
        base_urls: tuple[str, ...] = (),
        api_key_env: str | None = None,
        timeout_s: float = 300.0,
        max_concurrency: int = 64,
        max_queue: int = 0,
        queue_wait_s: float = 0.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if max_concurrency < 1 or max_queue < 0:
            raise ValueError("max_concurrency must be >= 1 and max_queue >= 0")
        if max_queue and queue_wait_s <= 0:
            raise ValueError("a System One admission queue requires queue_wait_s > 0")
        api_key = None
        if api_key_env is not None:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(f"System One API key variable {api_key_env!r} is not set")
        urls = (base_url,) if base_url is not None else tuple(base_urls)
        if (base_url is None) == (not base_urls) or not urls:
            raise ValueError("System One requires exactly one of base_url or base_urls")
        if len(set(urls)) != len(urls):
            raise ValueError("System One base_urls must be distinct")
        self._urls = tuple(url.rstrip("/") + "/v1/systemone" for url in urls)
        self._in_flight = dict.fromkeys(self._urls, 0)
        self._upstream_model = upstream_model
        self._headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self._slots = asyncio.Semaphore(max_concurrency)
        self._max_queue = max_queue
        self._queue_wait_s = queue_wait_s
        self._waiting = 0
        self._client = httpx.AsyncClient(
            timeout=timeout_s,
            transport=transport,
            trust_env=False,
            limits=httpx.Limits(
                max_connections=max_concurrency,
                max_keepalive_connections=max_concurrency,
            ),
        )

    async def _acquire(self) -> None:
        if not self._slots.locked():
            await self._slots.acquire()
            return
        if self._waiting >= self._max_queue:
            raise SystemOneCapacityError("System One admission queue is full")
        self._waiting += 1
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=self._queue_wait_s)
        except TimeoutError:
            raise SystemOneCapacityError("System One admission queue wait timed out") from None
        finally:
            self._waiting -= 1

    @property
    def replica_count(self) -> int:
        return len(self._urls)

    def _replica_order(self) -> list[str]:
        # Fewest in flight first; ties keep configuration order.
        return sorted(self._urls, key=lambda url: self._in_flight[url])

    async def _post(self, url: str, body: dict) -> httpx.Response:
        self._in_flight[url] += 1
        try:
            return await self._client.post(
                url,
                json={**body, "model": self._upstream_model},
                headers=self._headers,
            )
        finally:
            self._in_flight[url] -= 1

    async def decide(self, body: dict) -> SystemOneReply:
        """Forward one request body (any model alias) to an upstream replica.

        Raises ``SystemOneUnavailableError`` only when every tried replica was
        unreachable; an upstream status reply (including 529) is returned so
        the public route can pass it through unchanged.
        """

        await self._acquire()
        try:
            response: httpx.Response | None = None
            last_error: httpx.HTTPError | None = None
            for url in self._replica_order()[:2]:
                try:
                    response = await self._post(url, body)
                except httpx.HTTPError as error:
                    last_error = error
                    response = None
                    continue
                if response.status_code not in _RETRY_ON_OTHER_REPLICA:
                    break
            if response is None:
                assert last_error is not None
                raise SystemOneUnavailableError(type(last_error).__name__) from last_error
        finally:
            self._slots.release()
        headers = {
            name: response.headers[name] for name in _PASSED_HEADERS if name in response.headers
        }
        input_tokens = output_tokens = None
        if response.status_code == 200:
            try:
                usage = response.json().get("usage") or {}
            except ValueError as error:
                raise SystemOneUnavailableError("upstream returned invalid JSON") from error
            input_tokens = _count(usage.get("input_tokens"))
            output_tokens = _count(usage.get("output_tokens"))
        return SystemOneReply(
            status=response.status_code,
            body=response.content,
            headers=headers,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def shutdown(self) -> None:
        await self._client.aclose()


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
