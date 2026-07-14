"""The Sampler: penalties, temperature, min-p/top-k/top-p, grammar (design m8 D2).

Order of operations (reviewed convention, m8 §6):

1. raw logits → capture ``log_softmax`` for logprob reporting (vLLM v1
   ``raw_logprobs`` default; temperature-independent, OpenAI-style).
2. xgrammar mask FIRST (mask-last can leave zero legal tokens after top-k/top-p
   → NaN multinomial; mask-first plus keep-1 filters guarantees support).
3. Penalties — repetition over prompt + committed outputs; presence/frequency
   over committed outputs only (vLLM/HF agreement).
4. ``temperature == 0`` → argmax on the masked logits; else scale.
5. min_p, then top-k, then top-p (vLLM v1 order; HF differs — recorded).
6. softmax → seeded sample.

Determinism: base seed is ``sampling.seed`` or sha256(request_id) (never
Python ``hash()`` — process-randomized); the per-position generator seed is a
splitmix64 mix of (base, position), so TP rank runners sample identically
given identical logits, and repeated sampling of the same position (shared
runner instance across CPU TP ranks) is idempotent. Grammar ``accept`` is
likewise idempotent per position — the matcher advances exactly once per
committed token.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from kairyu.engine.core.sampling_types import (
    EngineSampling,
    SampledToken,
    mix_seed,
    stable_request_seed,
)
from kairyu.engine.core.structured import XGrammarEnforcer


class _RequestSamplerState:
    __slots__ = ("base_seed", "enforcer", "accepted_position")

    def __init__(self, base_seed: int, enforcer: XGrammarEnforcer | None) -> None:
        self.base_seed = base_seed
        self.enforcer = enforcer
        self.accepted_position = -1


class Sampler:
    """Per-request sampling state: seeded generators + grammar enforcers.

    ``vocab_provider`` supplies token strings for grammar compilation; required
    only when a request carries ``json_schema``/``json_mode``. State is dropped
    via ``release`` (owners call it on finish; unreleased state is a few ints).
    """

    def __init__(self, vocab_provider: Callable[[], list[str]] | None = None) -> None:
        self._vocab_provider = vocab_provider
        self._states: dict[str, _RequestSamplerState] = {}

    def _state_for(
        self,
        request_id: str,
        sampling: EngineSampling,
        eos_token_id: int | None = None,
    ) -> _RequestSamplerState:
        state = self._states.get(request_id)
        if state is None:
            enforcer = None
            if sampling.needs_grammar:
                if self._vocab_provider is None:
                    raise RuntimeError(
                        "structured output requires a Sampler with a vocab_provider"
                    )
                if eos_token_id is None:
                    raise RuntimeError(
                        "structured output requires an eos token id — a completed "
                        "grammar terminates by sampling it (m8 D2)"
                    )
                enforcer = XGrammarEnforcer(
                    self._vocab_provider(),
                    json_schema=sampling.json_schema,
                    stop_token_id=eos_token_id,
                )
            base = sampling.seed if sampling.seed is not None else stable_request_seed(request_id)
            state = _RequestSamplerState(base, enforcer)
            self._states[request_id] = state
        return state

    def release(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    def sample(
        self,
        request_id: str,
        sampling: EngineSampling,
        position: int,
        logits: torch.Tensor,
        *,
        prompt: tuple[int, ...] = (),
        outputs: Sequence[int] = (),
        eos_token_id: int | None = None,
    ) -> SampledToken:
        state = self._state_for(request_id, sampling, eos_token_id)
        # Sampling runs on CPU: the seeded torch.Generator + multinomial and the
        # penalty/enforcer index tensors are all CPU, and the reproducibility pins
        # (m8 D2, spec ≡ greedy) are defined by the CPU RNG stream — a CUDA
        # generator would diverge. On GPU this pulls the [vocab] logits row to host
        # (cheap); on CPU it is the pre-existing no-op + private clone.
        logits = logits.detach().to(device="cpu", dtype=torch.float32).clone()

        raw_logsoftmax: torch.Tensor | None = None
        if sampling.logprobs is not None:
            raw_logsoftmax = torch.log_softmax(logits, dim=-1)

        if state.enforcer is not None:
            state.enforcer.mask_logits(logits)

        self._apply_penalties(logits, sampling, prompt, outputs)

        if sampling.temperature == 0.0:
            token_id = int(torch.argmax(logits).item())
        else:
            token_id = self._sample_scaled(logits, sampling, state.base_seed, position)

        logprob, top_logprobs = self._report(raw_logsoftmax, sampling, token_id)
        terminated = False
        if state.enforcer is not None:
            terminated = self._accept_once(state, position, token_id)
        return SampledToken(token_id, logprob, top_logprobs, terminated)

    @staticmethod
    def _apply_penalties(
        logits: torch.Tensor,
        sampling: EngineSampling,
        prompt: tuple[int, ...],
        outputs: Sequence[int],
    ) -> None:
        if sampling.repetition_penalty != 1.0:
            seen = torch.tensor(sorted(set(prompt) | set(outputs)), dtype=torch.long)
            if seen.numel():
                values = logits[seen]
                logits[seen] = torch.where(
                    values > 0,
                    values / sampling.repetition_penalty,
                    values * sampling.repetition_penalty,
                )
        if outputs and (sampling.presence_penalty != 0.0 or sampling.frequency_penalty != 0.0):
            counts = torch.bincount(
                torch.tensor(outputs, dtype=torch.long), minlength=logits.shape[-1]
            ).to(logits.dtype)
            logits -= sampling.frequency_penalty * counts
            logits -= sampling.presence_penalty * (counts > 0).to(logits.dtype)

    @staticmethod
    def _sample_scaled(
        logits: torch.Tensor, sampling: EngineSampling, base_seed: int, position: int
    ) -> int:
        probs = torch.softmax(logits / sampling.temperature, dim=-1)
        if sampling.min_p > 0.0:
            probs = torch.where(
                probs >= sampling.min_p * probs.max(), probs, torch.zeros_like(probs)
            )
        if 0 < sampling.top_k < probs.shape[-1]:
            threshold = torch.topk(probs, sampling.top_k).values[-1]
            probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
        if sampling.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # drop tokens whose exclusive cumulative mass already reaches top_p;
            # the highest-probability token always survives (keep-1)
            cut = cumulative - sorted_probs >= sampling.top_p * cumulative[-1].clamp(min=1e-12)
            sorted_probs[cut] = 0.0
            probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
        total = probs.sum()
        if total <= 0:  # every candidate filtered (degenerate): fall back to argmax
            return int(torch.argmax(logits).item())
        generator = torch.Generator().manual_seed(mix_seed(base_seed, position))
        return int(torch.multinomial(probs / total, 1, generator=generator).item())

    @staticmethod
    def _report(
        raw_logsoftmax: torch.Tensor | None, sampling: EngineSampling, token_id: int
    ) -> tuple[float | None, tuple[tuple[int, float], ...] | None]:
        if raw_logsoftmax is None:
            return None, None
        logprob = float(raw_logsoftmax[token_id].item())
        top: tuple[tuple[int, float], ...] | None = None
        assert sampling.logprobs is not None
        if sampling.logprobs > 0:
            k = min(sampling.logprobs, raw_logsoftmax.shape[-1])
            values, indices = torch.topk(raw_logsoftmax, k)
            top = tuple(
                (int(index.item()), float(value.item()))
                for index, value in zip(indices, values, strict=True)
            )
        return logprob, top

    @staticmethod
    def _accept_once(state: _RequestSamplerState, position: int, token_id: int) -> bool:
        """Advance the grammar matcher exactly once per output position."""
        assert state.enforcer is not None
        if position <= state.accepted_position:
            return state.enforcer.is_terminated()
        state.accepted_position = position
        if not state.enforcer.accept(token_id):
            raise RuntimeError(
                f"grammar rejected token {token_id} sampled under its own mask "
                "(invariant violation)"
            )
        return state.enforcer.is_terminated()
