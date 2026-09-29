"""Wire contract for live Runner cache-startup binding authorization."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from kairyu.runners.startup_binding import RunnerCacheStartupBinding


def validate_runner_cache_placement_bearer_token(value: str) -> str:
    """Return a header-safe shared secret containing visible ASCII only."""

    if not isinstance(value, str):
        raise TypeError("bearer token must be a string")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("bearer token must contain visible ASCII only") from exc
    if not 32 <= len(encoded) <= 4096 or any(not 0x21 <= byte <= 0x7E for byte in encoded):
        raise ValueError("bearer token must be 32-4096 visible ASCII characters")
    return value


class RunnerCachePlacementBindingLiveAuthority(Protocol):
    """Deadline-aware backend contract for exact live binding authorization."""

    def __call__(
        self,
        binding: RunnerCacheStartupBinding,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> RunnerCacheStartupBinding: ...


class RunnerCachePlacementBindingAuthorityReadiness(Protocol):
    """Deadline-aware dependency-readiness contract for the authority."""

    def __call__(
        self,
        *,
        deadline_monotonic: float,
        backend_timeout_s: float,
    ) -> None: ...


class RunnerCachePlacementBindingAuthorizationError(RuntimeError):
    """The live binding authority could not authorize an exact binding."""


class RunnerCachePlacementBindingAuthorizationDeniedError(
    RunnerCachePlacementBindingAuthorizationError
):
    """The candidate binding is not the authority's current live binding."""


class RunnerCachePlacementBindingAuthorizationRequest(BaseModel):
    """Nonce-bound request sent to the live scaling authority."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
    )

    schema_version: Literal["kairyu-runner-cache-placement-binding-authorization-request-v1"] = (
        "kairyu-runner-cache-placement-binding-authorization-request-v1"
    )
    nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding: RunnerCacheStartupBinding


class RunnerCachePlacementBindingAuthorizationResponse(BaseModel):
    """Exact live binding returned by the trusted scaling authority."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
    )

    schema_version: Literal["kairyu-runner-cache-placement-binding-authorization-response-v1"] = (
        "kairyu-runner-cache-placement-binding-authorization-response-v1"
    )
    nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding: RunnerCacheStartupBinding


__all__ = [
    "RunnerCachePlacementBindingAuthorizationDeniedError",
    "RunnerCachePlacementBindingAuthorizationError",
    "RunnerCachePlacementBindingAuthorizationRequest",
    "RunnerCachePlacementBindingAuthorizationResponse",
    "RunnerCachePlacementBindingAuthorityReadiness",
    "RunnerCachePlacementBindingLiveAuthority",
    "validate_runner_cache_placement_bearer_token",
]
