"""
title: Reasoning Effort
description: Select the Qwen reasoning effort (low/medium/xhigh) from a dropdown in Chat Controls.
version: 0.1.0
"""

from typing import Literal

from pydantic import BaseModel, Field

# Sampling fields the Chat UI pins to Qwen's published instruct values
# (DEFAULT_MODEL_PARAMS in compose.yaml). They are dropped when an effort is
# selected so vLLM applies the checkpoint's thinking generation_config
# (temperature 1.0, top_p 0.95, top_k 20) instead.
_INSTRUCT_SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "presence_penalty")


class Filter:
    """Open WebUI global filter exposing the L3 effort knob as a dropdown.

    The pinned Open WebUI v0.11.0 renders enum-typed user valves as a
    ``<select>`` in Chat Controls, replacing the stock free-text Advanced
    Params field. The selection is forwarded as the OpenAI-compatible
    ``reasoning_effort`` body field; the levels are Qwen's official vocabulary
    (Kairyu L3 normalizes medium/xhigh on the wire and the example-local chat
    template maps them back), and ``default`` leaves the field out so the
    server's non-thinking direct-chat default applies.
    """

    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        reasoning_effort: Literal["default", "low", "medium", "xhigh"] = Field(
            default="default",
            description=(
                "Reasoning effort for Qwen3.8-Flash-Next. "
                "default = direct chat without thinking (instruct sampling); "
                "any effort enables thinking with Qwen's thinking sampling."
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
            for field in _INSTRUCT_SAMPLING_FIELDS:
                body.pop(field, None)
        return body
