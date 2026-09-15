"""
title: Reasoning Effort
description: Select the DeepSeek reasoning effort (low/high/max) from a dropdown in Chat Controls.
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
    checkpoint encoder's own vocabulary (``encoding_dsv4.py``), and ``default``
    leaves the field out so the server's non-thinking direct-chat default
    applies.
    """

    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        reasoning_effort: Literal["default", "low", "high", "max"] = Field(
            default="default",
            description=(
                "Reasoning effort for DeepSeek-V4-Flash-Vision-Exp. "
                "default = direct chat without thinking."
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
        return body
