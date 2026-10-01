"""
title: Reasoning Effort
description: DeepSeek-V4.1 reasoning effort for every DeepSeek step; default high (75).
version: 0.1.0
"""

from typing import Literal

from pydantic import BaseModel, Field


class Filter:
    """Open WebUI global filter that turns the effort knob into a dropdown.

    Open WebUI v0.11.0 renders an enum-typed user valve as a ``<select>`` in
    Chat Controls. The choice is sent as the OpenAI ``reasoning_effort`` body
    field (low=50, high=75, max=100); Kairyu applies it to every DeepSeek
    step of the chosen route. ``default`` omits the field, so the
    orchestration default (high) applies.
    """

    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        reasoning_effort: Literal["default", "low", "high", "max"] = Field(
            default="default",
            description="Reasoning effort for every DeepSeek-V4.1 step. default = high (75).",
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
        # Only the top-level effort is public; Kairyu renders each step itself.
        body.pop("chat_template_kwargs", None)
        return body
