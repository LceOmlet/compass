from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from numbers import Integral, Real
from types import ModuleType
from typing import Any

import torch
from flashtrace import core as flashtrace_core
from flashtrace import improved as flashtrace_improved

from .b10_flashtrace_offload import (
    OffloadedCaptureLease,
    OffloadedFlashTrace,
    OffloadedLLMIFRAttribution,
    _BACKEND_LOCK,
    _OffloadedLLMIFRAttributionBoth,
)


_OFFICIAL_CORE_SINK_AGGREGATE = flashtrace_core.compute_ifr_sentence_aggregate
_OFFICIAL_IMPROVED_SINK_AGGREGATE = flashtrace_improved.compute_ifr_sentence_aggregate


def _integer_ids(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in result):
        raise TypeError(f"{name} must contain integer token IDs")
    if any(int(value) < 0 for value in result):
        raise ValueError(f"{name} must contain non-negative token IDs")
    return tuple(int(value) for value in result)


def token_surfaces(tokenizer: Any, token_ids: Sequence[int]) -> tuple[str, ...]:
    """Decode one public surface label per exact token ID, without re-tokenizing text."""

    ids = _integer_ids(token_ids, "token_ids")
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("tokenizer must expose decode")
    surfaces = tuple(
        decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in ids
    )
    if any(not isinstance(surface, str) for surface in surfaces):
        raise TypeError("tokenizer.decode must return one text surface per token ID")
    return surfaces


@contextmanager
def shared_exact_token_capture(model: Any) -> Iterator[OffloadedCaptureLease]:
    """Lease one full-sequence offloaded capture to two exact-ID views."""

    with OffloadedCaptureLease(model) as capture_lease:
        yield capture_lease


def _generation_ids_with_one_eos(tokenizer: Any, values: Sequence[int]) -> tuple[int, ...]:
    ids = _integer_ids(values, "generation_ids")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, bool) or not isinstance(eos_id, Integral):
        raise TypeError("tokenizer must expose one integer eos_token_id")
    eos = int(eos_id)
    if len(ids) < 2:
        raise ValueError("generation_ids must contain content and one terminal EOS")
    if ids[-1] != eos or ids.count(eos) != 1:
        raise ValueError("generation_ids must end in exactly one tokenizer EOS token")
    return ids


def _inclusive_span(
    value: tuple[int, int] | None,
    *,
    name: str,
    non_eos_length: int,
    required: bool,
) -> tuple[int, int] | None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be one inclusive (start, end) tuple")
    start, end = value
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
        raise TypeError(f"{name} boundaries must be integers")
    result = (int(start), int(end))
    if result[0] < 0 or result[1] < result[0] or result[1] >= non_eos_length:
        raise ValueError(f"{name} must be non-empty, in bounds, and exclude terminal EOS")
    return result


def _sink_weights(values: Sequence[Real]) -> tuple[float, ...]:
    raw = tuple(values)
    if not raw:
        raise ValueError("sink weights must be non-empty")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw):
        raise TypeError("sink weights must contain real scalars")
    weights = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("sink weights must be finite and non-negative")
    if math.fsum(weights) <= 0.0:
        raise ValueError("sink weights must have positive total mass")
    return weights


@contextmanager
def _weighted_first_sink_aggregate(
    *,
    module: ModuleType,
    official: Any,
    expected_span: tuple[int, int],
    external_weights: tuple[float, ...],
) -> Iterator[None]:
    """Inject weights only into an official algorithm's first sink aggregate.

    The official multi-hop orchestration and the b10 offload/streaming scope stay
    in control.  Positive-uniform weights still execute this injected official
    weighted path, so parity against the existing span path tests the new facade.
    """

    if len(external_weights) != expected_span[1] - expected_span[0] + 1:
        raise ValueError("sink weights do not match the exact sink token span")
    with _BACKEND_LOCK:
        previous = getattr(module, "compute_ifr_sentence_aggregate", None)
        if previous is not official:
            raise RuntimeError("another component replaced official weighted-sink IFR")
        call_count = 0

        def weighted_first(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_index = call_count
            call_count += 1
            if call_index != 0:
                return previous(*args, **kwargs)
            if args or "sink_start" not in kwargs or "sink_end" not in kwargs:
                raise RuntimeError("official first sink aggregate changed its call contract")
            actual_span = (int(kwargs["sink_start"]), int(kwargs["sink_end"]))
            if actual_span != expected_span:
                raise RuntimeError("official first sink aggregate changed its sink span")

            official_weights = kwargs.get("sink_weights")
            combined = torch.tensor(external_weights, dtype=torch.float32)
            if official_weights is not None:
                official_tensor = torch.as_tensor(
                    official_weights,
                    dtype=torch.float32,
                    device="cpu",
                ).reshape(-1)
                if official_tensor.numel() != combined.numel():
                    raise RuntimeError("official sink mask and external weights are misaligned")
                if not torch.isfinite(official_tensor).all() or (official_tensor < 0).any():
                    raise RuntimeError("official sink mask is not finite and non-negative")
                combined = combined * official_tensor
            if float(combined.sum().item()) <= 0.0:
                raise RuntimeError("official sink mask removes all externally weighted tokens")
            forwarded = dict(kwargs)
            forwarded["sink_weights"] = combined
            return previous(**forwarded)

        setattr(module, "compute_ifr_sentence_aggregate", weighted_first)
        completed = False
        try:
            yield
            completed = True
        finally:
            setattr(module, "compute_ifr_sentence_aggregate", previous)
        if completed and call_count == 0:
            raise RuntimeError("official weighted-sink IFR did not execute a sink aggregate")


class _WeightedSinkScopeMixin:
    _active_external_sink_weights: tuple[float, ...] | None = None

    @contextmanager
    def weighted_sink_scope(self, weights: Sequence[Real]) -> Iterator[None]:
        """Bind one exact external sink measure to the next public IFR call."""

        # The exact-ID facades temporarily bind caller-owned state on a shared
        # FlashTrace engine and patch official module globals downstream.  One
        # engine call therefore owns the existing backend lock for its whole
        # scope; concurrent epoch tasks queue here without being retried.
        with _BACKEND_LOCK:
            if self._active_external_sink_weights is not None:
                raise RuntimeError("an external weighted-sink call is already active")
            self._active_external_sink_weights = _sink_weights(weights)
            try:
                yield
            finally:
                self._active_external_sink_weights = None


@dataclass(frozen=True, slots=True)
class _ExactTokenState:
    prompt: str
    prompt_ids: tuple[int, ...]
    target: str
    generation_ids: tuple[int, ...]
    prompt_surfaces: tuple[str, ...]
    generation_surfaces: tuple[str, ...]


class _ExactTokenStateMixin:
    _active_exact_token_state: _ExactTokenState | None = None

    def _make_exact_token_state(
        self,
        *,
        prompt: str,
        prompt_ids: Sequence[int],
        target: str,
        generation_ids: Sequence[int],
    ) -> _ExactTokenState:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be non-empty text")
        if not isinstance(target, str) or not target:
            raise ValueError("target must be non-empty text")
        if getattr(self, "use_chat_template", None) is not False:
            raise RuntimeError("exact prompt IDs require use_chat_template=False")
        if getattr(self, "processor", None) is not None or getattr(self, "images", None) is not None:
            raise RuntimeError("the exact token-ID bridge is defined only for text-only attribution")

        exact_prompt_ids = _integer_ids(prompt_ids, "prompt_ids")
        if not exact_prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        exact_generation_ids = _generation_ids_with_one_eos(self.tokenizer, generation_ids)
        total_length = len(exact_prompt_ids) + len(exact_generation_ids)
        context_limit = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if isinstance(context_limit, Integral) and not isinstance(context_limit, bool):
            if total_length > int(context_limit):
                raise ValueError("exact prompt and generation IDs exceed model max_position_embeddings")

        return _ExactTokenState(
            prompt=prompt,
            prompt_ids=exact_prompt_ids,
            target=target,
            generation_ids=exact_generation_ids,
            prompt_surfaces=token_surfaces(self.tokenizer, exact_prompt_ids),
            generation_surfaces=token_surfaces(self.tokenizer, exact_generation_ids),
        )

    @contextmanager
    def _exact_token_scope(self, state: _ExactTokenState) -> Iterator[None]:
        # Exact token coordinates are mutable request-local state on the
        # official attribution facade.  Reuse the backend's existing global
        # re-entrant lock so independent epoch tasks wait for the single model
        # resource instead of racing or failing.  Nested official patch scopes
        # below remain valid because the lock is deliberately re-entrant.
        with _BACKEND_LOCK:
            if self._active_exact_token_state is not None:
                raise RuntimeError("an exact token-ID attribution call is already active")
            self._active_exact_token_state = state
            try:
                yield
            finally:
                self._active_exact_token_state = None

    def _ensure_generation(
        self,
        prompt: str,
        target: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        state = self._active_exact_token_state
        if state is None:
            raise RuntimeError("use the public exact token-ID attribution method")
        if prompt != state.prompt or target != state.target:
            raise RuntimeError("official attribution changed the bound prompt or target text")

        prompt_ids = torch.tensor([state.prompt_ids], dtype=torch.long, device=self.device)
        generation_ids = torch.tensor([state.generation_ids], dtype=torch.long, device=self.device)
        prompt_len = len(state.prompt_ids)
        gen_len = len(state.generation_ids)

        self.user_prompt = state.prompt
        self.prompt = state.prompt
        self.user_prompt_ids = prompt_ids
        self.prompt_ids = prompt_ids
        self.user_prompt_tokens = list(state.prompt_surfaces)
        self.prompt_tokens = list(state.prompt_surfaces)
        self.user_prompt_indices = list(range(prompt_len))
        self.chat_prompt_indices = []
        self.prompt_feature_indices = list(range(prompt_len))
        self.attribution_prompt_tokens = list(state.prompt_surfaces)
        self.visual_token_indices = []
        self._prompt_model_inputs = {
            "input_ids": prompt_ids,
            "attention_mask": torch.ones_like(prompt_ids),
        }

        self.generation = state.target
        self.generation_ids = generation_ids
        self.generation_tokens = list(state.generation_surfaces)

        input_ids_all = torch.cat((prompt_ids, generation_ids), dim=1)
        attention_mask = torch.ones_like(input_ids_all)
        return input_ids_all, attention_mask, prompt_len, gen_len


class ExactTokenOffloadedLLMIFRAttribution(
    _WeightedSinkScopeMixin,
    _ExactTokenStateMixin,
    OffloadedLLMIFRAttribution,
):
    """Existing offloaded dependency IFR with exact caller-supplied token IDs."""

    def calculate_ifr_multi_hop_weighted_sink_ids(
        self,
        prompt: str,
        *,
        sink_weights: Sequence[Real],
        **kwargs: Any,
    ) -> Any:
        """Run the existing exact-ID method with one weighted official base sink."""

        with self.weighted_sink_scope(sink_weights):
            return self.calculate_ifr_multi_hop_ids(prompt, **kwargs)

    def calculate_ifr_multi_hop_ids(
        self,
        prompt: str,
        *,
        prompt_ids: Sequence[int],
        target: str,
        generation_ids: Sequence[int],
        sink_span: tuple[int, int],
        thinking_span: tuple[int, int],
        n_hops: int = 1,
        renorm_threshold: float | None = None,
        observation_mask: torch.Tensor | Sequence[float] | None = None,
        capture_lease: OffloadedCaptureLease | None = None,
    ) -> Any:
        if isinstance(n_hops, bool) or int(n_hops) != 1:
            raise ValueError("the exact dependency bridge requires n_hops=1")
        state = self._make_exact_token_state(
            prompt=prompt,
            prompt_ids=prompt_ids,
            target=target,
            generation_ids=generation_ids,
        )
        non_eos_length = len(state.generation_ids) - 1
        sink = _inclusive_span(
            sink_span,
            name="sink_span",
            non_eos_length=non_eos_length,
            required=True,
        )
        thinking = _inclusive_span(
            thinking_span,
            name="thinking_span",
            non_eos_length=non_eos_length,
            required=True,
        )
        assert sink is not None and thinking is not None
        if thinking[1] >= sink[0]:
            raise ValueError("thinking_span must precede and not overlap sink_span")
        with (
            self._exact_token_scope(state),
            self._capture_lease_scope(capture_lease),
        ):
            # Read request-local weights only after exact_token_scope owns the
            # shared backend lock.  Otherwise an unweighted task can observe a
            # concurrently bound upstream task's weights before it blocks.
            weights = self._active_external_sink_weights
            weighted_scope = (
                _weighted_first_sink_aggregate(
                    module=flashtrace_core,
                    official=_OFFICIAL_CORE_SINK_AGGREGATE,
                    expected_span=(
                        len(state.prompt_ids) + sink[0],
                        len(state.prompt_ids) + sink[1],
                    ),
                    external_weights=weights,
                )
                if weights is not None
                else nullcontext()
            )
            with weighted_scope:
                return super().calculate_ifr_multi_hop(
                    prompt,
                    target=target,
                    sink_span=sink,
                    thinking_span=thinking,
                    n_hops=1,
                    renorm_threshold=renorm_threshold,
                    observation_mask=observation_mask,
                )


class _ExactTokenOffloadedLLMIFRAttributionBoth(
    _ExactTokenStateMixin,
    _OffloadedLLMIFRAttributionBoth,
):
    def calculate_ifr_multi_hop_both_ids(
        self,
        prompt: str,
        *,
        prompt_ids: Sequence[int],
        target: str,
        generation_ids: Sequence[int],
        sink_span: tuple[int, int],
        thinking_span: tuple[int, int] | None,
        n_hops: int,
        renorm_threshold: float | None,
        sink_weights: Sequence[Real] | None = None,
        capture_lease: OffloadedCaptureLease | None = None,
    ) -> Any:
        if isinstance(n_hops, bool) or int(n_hops) != 1:
            raise ValueError("the exact credit bridge requires n_hops=1")
        state = self._make_exact_token_state(
            prompt=prompt,
            prompt_ids=prompt_ids,
            target=target,
            generation_ids=generation_ids,
        )
        non_eos_length = len(state.generation_ids) - 1
        sink = _inclusive_span(
            sink_span,
            name="output_span",
            non_eos_length=non_eos_length,
            required=True,
        )
        thinking = _inclusive_span(
            thinking_span,
            name="reasoning_span",
            non_eos_length=non_eos_length,
            required=False,
        )
        assert sink is not None
        weights = _sink_weights(sink_weights) if sink_weights is not None else None
        weighted_scope = (
            _weighted_first_sink_aggregate(
                module=flashtrace_improved,
                official=_OFFICIAL_IMPROVED_SINK_AGGREGATE,
                expected_span=(len(state.prompt_ids) + sink[0], len(state.prompt_ids) + sink[1]),
                external_weights=weights,
            )
            if weights is not None
            else nullcontext()
        )
        with (
            self._exact_token_scope(state),
            self._capture_lease_scope(capture_lease),
            weighted_scope,
        ):
            return super().calculate_ifr_multi_hop_both(
                prompt,
                target=target,
                sink_span=sink,
                thinking_span=thinking,
                n_hops=1,
                renorm_threshold=renorm_threshold,
            )


class ExactTokenOffloadedFlashTrace(_WeightedSinkScopeMixin, OffloadedFlashTrace):
    """Existing offloaded FlashTrace credit facade with an exact-ID entry point."""

    def trace_ids(
        self,
        *,
        prompt: str,
        prompt_ids: Sequence[int],
        target: str,
        generation_ids: Sequence[int],
        output_span: tuple[int, int],
        reasoning_span: tuple[int, int] | None = None,
        hops: int = 1,
        method: str = "flashtrace",
        renorm_threshold: float | None = None,
        capture_lease: OffloadedCaptureLease | None = None,
    ) -> Any:
        if method != "flashtrace" or isinstance(hops, bool) or int(hops) != 1:
            raise ValueError("the exact credit facade supports only method='flashtrace', hops=1")
        if self.use_chat_template is not False:
            raise RuntimeError("exact prompt IDs require use_chat_template=False")
        if getattr(self, "processor", None) is not None:
            raise RuntimeError("the exact token-ID credit facade is text-only")

        with _BACKEND_LOCK:
            engine = _ExactTokenOffloadedLLMIFRAttributionBoth(
                self.model,
                self.tokenizer,
                generate_kwargs=self.generate_kwargs,
                chunk_tokens=self.chunk_tokens,
                sink_chunk_tokens=self.sink_chunk_tokens,
                recompute_attention=True,
                use_chat_template=False,
            )
            raw = engine.calculate_ifr_multi_hop_both_ids(
                prompt,
                prompt_ids=prompt_ids,
                target=target,
                generation_ids=generation_ids,
                sink_span=output_span,
                thinking_span=reasoning_span,
                n_hops=1,
                renorm_threshold=renorm_threshold,
                sink_weights=self._active_external_sink_weights,
                capture_lease=capture_lease,
            )
            return self._build_result(
                raw,
                method="flashtrace",
                output_span=output_span,
                reasoning_span=reasoning_span,
            )


__all__ = [
    "ExactTokenOffloadedFlashTrace",
    "ExactTokenOffloadedLLMIFRAttribution",
    "shared_exact_token_capture",
    "token_surfaces",
]
