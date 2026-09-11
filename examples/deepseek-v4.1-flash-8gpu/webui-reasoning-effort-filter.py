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
    unspecified so L1's thinking-high default applies. ``off`` explicitly
    disables thinking without sending an unsupported L3 effort value.
    """

    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        reasoning_effort: Literal["default", "low", "high", "max", "off"] = Field(
            default="default",
            description=(
                "Reasoning effort for DeepSeek-V4.1-Flash. "
                "default = thinking high (75); off = direct chat."
            ),
        )

    def __init__(self):
        self.valves = self.Valves()

    def inlet(self, body: dict, __user__: dict | None = None) -> dict:
        valves = (__user__ or {}).get("valves")
        effort = getattr(valves, "reasoning_effort", None) if valves else None
        kwargs = dict(body.get("chat_template_kwargs") or {})
        if effort == "off":
            body.pop("reasoning_effort", None)
            kwargs.update(thinking=False, enable_thinking=False)
        elif effort == "default":
            body.pop("reasoning_effort", None)
            for key in ("thinking", "enable_thinking", "reasoning_effort"):
                kwargs.pop(key, None)
        elif effort:
            body["reasoning_effort"] = effort
            kwargs.update(thinking=True, enable_thinking=True)
            kwargs.pop("reasoning_effort", None)
        if kwargs:
            body["chat_template_kwargs"] = kwargs
        else:
            body.pop("chat_template_kwargs", None)
        return body
