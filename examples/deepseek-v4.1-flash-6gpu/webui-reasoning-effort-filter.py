"""
title: Reasoning Effort
description: Select DeepSeek-V4.1-Flash reasoning effort; default is thinking high (75).
version: 0.1.0
"""

from typing import Literal

from pydantic import BaseModel, Field


class Filter:
    """Open WebUI global filter that turns the effort knob into a dropdown.

    Open WebUI v0.11.0 renders an enum-typed user valve as a ``<select>`` in
    Chat Controls. The choice is sent as the OpenAI ``reasoning_effort`` body
    field using the model author's vocabulary (low=50, high=75, max=100);
    ``default`` omits the field so the L1 default (thinking high) applies.
    """

    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        reasoning_effort: Literal["default", "low", "high", "max"] = Field(
            default="default",
            description="Reasoning effort for DeepSeek-V4.1-Flash. default = thinking high (75).",
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
        # Only the top-level effort is public; L1 renders the chat itself.
        body.pop("chat_template_kwargs", None)
        return body
