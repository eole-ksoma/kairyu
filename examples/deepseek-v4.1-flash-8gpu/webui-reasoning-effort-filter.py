"""
title: Reasoning Effort
description: Select DeepSeek reasoning effort; default is thinking high.
version: 0.1.0
"""

from typing import Literal

from pydantic import BaseModel, Field


class Filter:
    """Open WebUI global filter exposing the L3 effort knob as a dropdown.

    The pinned Open WebUI v0.11.0 renders enum-typed user valves as a
    ``<select>`` in Chat Controls, replacing the stock free-text Advanced
    Params field. The selection is forwarded verbatim as the
    OpenAI-compatible ``reasoning_effort`` body field; the levels are the
    official checkpoint encoder's vocabulary. ``default`` leaves the effort
    unspecified so L1's thinking-high default applies.
    """

    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        reasoning_effort: Literal["default", "low", "high", "max"] = Field(
            default="default",
            description=(
                "Reasoning effort for DeepSeek-V4.1-Flash. "
                "default = thinking high (75)."
            ),
        )

    def __init__(self):
        self.valves = self.Valves()

    def inlet(self, body: dict, __user__: dict | None = None) -> dict:
        valves = (__user__ or {}).get("valves")
        effort = getattr(valves, "reasoning_effort", None) if valves else None
        if effort == "default":
            body.pop("reasoning_effort", None)
        elif effort:
            body["reasoning_effort"] = effort
        # Legacy chat uses L1's renderer; only the top-level effort is public.
        body.pop("chat_template_kwargs", None)
        return body
