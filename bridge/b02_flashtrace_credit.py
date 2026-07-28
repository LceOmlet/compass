from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from flashtrace import FlashTrace

from .b10_flashtrace_offload import OffloadedCaptureLease
from .qwen_native_tokens import QwenNativeTokenLayout
from .b14_flashtrace_token_ids import ExactTokenOffloadedFlashTrace, token_surfaces


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


@dataclass(frozen=True, slots=True)
class FlashTraceCredit:
    """Sparse credit on positions of one raw, non-EOS completion."""

    advantage: float
    rollout_token_ids: tuple[int, ...]
    reasoning_positions: tuple[int, ...]
    answer_positions: tuple[int, ...]
    token_weights: tuple[float, ...]
    history_token_count: int
    answer_target_token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _DirectSpan:
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _NativeCompletionLayout:
    """Exact token partition of ``<think>`` + T + ``</think>`` + O."""

    reasoning_chars: _DirectSpan
    output_chars: _DirectSpan
    opening_positions: tuple[int, ...]
    reasoning_positions: tuple[int, ...]
    closing_positions: tuple[int, ...]
    output_positions: tuple[int, ...]

    @property
    def history_positions(self) -> tuple[int, ...]:
        return self.opening_positions + self.reasoning_positions + self.closing_positions


def _integer_ids(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in result):
        raise TypeError(f"{name} must contain integer token IDs")
    return tuple(int(value) for value in result)


def _encode_with_offsets(tokenizer: Any, text: str, name: str) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = _integer_ids(encoded["input_ids"], f"{name} input_ids")
    offsets = tuple((int(start), int(end)) for start, end in encoded["offset_mapping"])
    if len(ids) != len(offsets):
        raise RuntimeError(f"{name} token IDs and offsets have different lengths")
    if any(start < 0 or end <= start or end > len(text) for start, end in offsets):
        raise RuntimeError(f"{name} has a non-direct token offset")
    decoded = tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded != text:
        raise RuntimeError(f"{name} failed the exact token-ID round trip")
    return ids, offsets


def _positions_for_span(
    text: str,
    offsets: tuple[tuple[int, int], ...],
    span: _DirectSpan,
    name: str,
) -> tuple[int, ...]:
    if span.start == span.end:
        return ()
    positions = tuple(
        index
        for index, (start, end) in enumerate(offsets)
        if start < span.end and span.start < end
    )
    if not positions or positions != tuple(range(positions[0], positions[-1] + 1)):
        raise RuntimeError(f"{name} is not one contiguous direct token cover")
    cover_start = offsets[positions[0]][0]
    cover_end = offsets[positions[-1]][1]
    if cover_start > span.start or cover_end < span.end:
        raise RuntimeError(f"{name} token cover does not contain its exact character span")
    if positions[0] > 0 and offsets[positions[0] - 1][1] > span.start:
        raise RuntimeError(f"{name} token cover omitted an overlapping left token")
    if positions[-1] + 1 < len(offsets) and offsets[positions[-1] + 1][0] < span.end:
        raise RuntimeError(f"{name} token cover omitted an overlapping right token")
    return positions


def _native_completion_layout(
    *,
    completion: str,
    offsets: tuple[tuple[int, int], ...],
    native_reasoning: str,
    native_output: str,
) -> _NativeCompletionLayout:
    """Validate and partition the exact Qwen native completion.

    Delimiters are FORMAT.  T and O are the only credited spans.  Requiring
    the four token covers to partition the captured token sequence also
    rejects a tokenizer token that crosses a provenance boundary.
    """

    if not isinstance(native_reasoning, str):
        raise TypeError("native reasoning_content must be text")
    if not isinstance(native_output, str) or not native_output:
        raise ValueError("native visible output must be non-empty text")
    expected = _THINK_OPEN + native_reasoning + _THINK_CLOSE + native_output
    if completion != expected:
        raise RuntimeError(
            "raw completion must equal '<think>' + native reasoning_content + "
            "'</think>' + native visible output"
        )

    opening_chars = _DirectSpan(0, len(_THINK_OPEN))
    reasoning_chars = _DirectSpan(opening_chars.end, opening_chars.end + len(native_reasoning))
    closing_chars = _DirectSpan(reasoning_chars.end, reasoning_chars.end + len(_THINK_CLOSE))
    output_chars = _DirectSpan(closing_chars.end, len(completion))
    opening_positions = _positions_for_span(
        completion, offsets, opening_chars, "native opening delimiter"
    )
    reasoning_positions = _positions_for_span(
        completion, offsets, reasoning_chars, "native reasoning content"
    )
    closing_positions = _positions_for_span(
        completion, offsets, closing_chars, "native closing delimiter"
    )
    output_positions = _positions_for_span(
        completion, offsets, output_chars, "native visible output"
    )
    partition = opening_positions + reasoning_positions + closing_positions + output_positions
    if partition != tuple(range(len(offsets))):
        raise RuntimeError(
            "native completion tokens do not form the exact FORMAT/T/FORMAT/O partition"
        )
    return _NativeCompletionLayout(
        reasoning_chars=reasoning_chars,
        output_chars=output_chars,
        opening_positions=opening_positions,
        reasoning_positions=reasoning_positions,
        closing_positions=closing_positions,
        output_positions=output_positions,
    )


def build_flashtrace_credit(
    *,
    tracer: FlashTrace,
    tokenizer: Any,
    prompt_text: str,
    prompt_token_ids: Sequence[int],
    completion_text: str,
    token_layout: QwenNativeTokenLayout,
    native_reasoning_text: str,
    native_output_text: str,
    advantage: Real,
    capture_lease: OffloadedCaptureLease | None,
) -> FlashTraceCredit:
    """Call official FlashTrace once and assign credit to native T/O tokens."""

    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError("captured canonical prompt text must be non-empty")
    if not isinstance(completion_text, str) or not completion_text:
        raise ValueError("raw completion text must be non-empty")
    if not isinstance(native_reasoning_text, str):
        raise TypeError("captured native reasoning_content must be text")
    if not isinstance(native_output_text, str) or not native_output_text:
        raise ValueError("captured native visible output must be non-empty text")
    if isinstance(advantage, bool) or not isinstance(advantage, Real):
        raise TypeError("advantage must be a real scalar")
    advantage_value = float(advantage)
    if not math.isfinite(advantage_value):
        raise ValueError("advantage must be finite")
    if not isinstance(tracer, ExactTokenOffloadedFlashTrace):
        raise TypeError("tracer must expose the adopted exact token-ID FlashTrace entry point")
    if getattr(tracer, "tokenizer", None) is not tokenizer:
        raise RuntimeError("FlashTrace and the bridge must share the exact tokenizer instance")
    if getattr(tracer, "use_chat_template", None) is not False:
        raise RuntimeError("FlashTrace must be initialized with use_chat_template=False")

    old_prompt_ids = _integer_ids(prompt_token_ids, "prompt_token_ids")
    if not isinstance(token_layout, QwenNativeTokenLayout):
        raise TypeError("token_layout must be the captured QwenNativeTokenLayout")
    rollout_ids = _integer_ids(token_layout.nonterminal_token_ids, "nonterminal completion token IDs")
    if not rollout_ids:
        raise ValueError("raw non-EOS completion token IDs must be non-empty")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos_text = getattr(tokenizer, "eos_token", None)
    if isinstance(eos_id, bool) or not isinstance(eos_id, Integral) or not isinstance(eos_text, str) or not eos_text:
        raise RuntimeError("the locked tokenizer must expose one textual EOS token and integer EOS ID")
    if (
        token_layout.full_token_ids != rollout_ids + (int(eos_id),)
        or token_layout.eos_position != len(rollout_ids)
    ):
        raise RuntimeError("captured Qwen layout does not contain exactly one terminal EOS")

    old_prompt_text = prompt_text
    encoded_old_prompt_ids, _ = _encode_with_offsets(tokenizer, old_prompt_text, "old prompt")
    if encoded_old_prompt_ids != old_prompt_ids:
        raise RuntimeError("rendered old prompt IDs differ from captured Arbor prompt IDs")
    decoded_completion = tokenizer.decode(
        list(rollout_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded_completion != completion_text:
        raise RuntimeError("captured raw completion text differs from its exact token IDs")
    parsed_reasoning_positions = token_layout.reasoning_positions
    answer_positions = token_layout.output_positions
    history_positions = token_layout.history_positions
    if history_positions != tuple(range(len(history_positions))):
        raise RuntimeError("native thinking history must be a token prefix")
    if answer_positions != tuple(range(len(history_positions), len(history_positions) + len(answer_positions))):
        raise RuntimeError("native visible output must begin at the first token after thinking history")
    if parsed_reasoning_positions and parsed_reasoning_positions[-1] >= answer_positions[0]:
        raise RuntimeError("native reasoning token span must precede the visible output token span")

    history_ids = tuple(rollout_ids[index] for index in history_positions)
    answer_ids = tuple(rollout_ids[index] for index in answer_positions)
    history_text = tokenizer.decode(
        list(history_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    output_text = tokenizer.decode(
        list(answer_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if output_text != native_output_text:
        raise RuntimeError("native visible output IDs differ from the provider content")
    reasoning_ids = tuple(rollout_ids[index] for index in parsed_reasoning_positions)
    decoded_reasoning = tokenizer.decode(
        list(reasoning_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded_reasoning != native_reasoning_text:
        raise RuntimeError("native reasoning IDs differ from the provider reasoning_content")
    combined_prompt_text = old_prompt_text + history_text
    combined_prompt_ids = old_prompt_ids + history_ids
    generation_ids = answer_ids + (int(eos_id),)
    flashtrace_answer_span = (0, len(answer_ids) - 1)

    result = tracer.trace_ids(
        prompt=combined_prompt_text,
        prompt_ids=combined_prompt_ids,
        target=output_text,
        generation_ids=generation_ids,
        output_span=flashtrace_answer_span,
        hops=1,
        method="flashtrace",
        renorm_threshold=0.0,
        capture_lease=capture_lease,
    )

    expected_prompt_tokens = token_surfaces(tokenizer, combined_prompt_ids)
    expected_generation_tokens = token_surfaces(tokenizer, generation_ids)
    if tuple(result.prompt_tokens) != expected_prompt_tokens:
        raise RuntimeError("FlashTrace prompt token positions differ from the exact bridge positions")
    if tuple(result.generation_tokens) != expected_generation_tokens:
        raise RuntimeError("FlashTrace generation token positions differ from native output plus EOS")
    if len(result.scores) != len(combined_prompt_ids):
        raise RuntimeError("FlashTrace returned the wrong number of prompt scores")
    scores = tuple(float(score) for score in result.scores)
    if any(not math.isfinite(score) or score < 0.0 for score in scores):
        raise RuntimeError("FlashTrace public scores must be finite and non-negative")

    reasoning_positions = parsed_reasoning_positions
    reasoning_scores = tuple(scores[len(old_prompt_ids) + position] for position in reasoning_positions)
    reasoning_total = math.fsum(reasoning_scores)
    if reasoning_positions and reasoning_total <= 0.0:
        raise RuntimeError("non-empty reasoning has zero total FlashTrace attribution")

    selected_count = len(reasoning_positions) + len(answer_positions)
    weights = [0.0] * len(rollout_ids)
    if reasoning_positions:
        reasoning_mass = len(reasoning_positions) / selected_count
        for position, score in zip(reasoning_positions, reasoning_scores, strict=True):
            weights[position] = reasoning_mass * score / reasoning_total
    answer_weight = 1.0 / selected_count
    for position in answer_positions:
        weights[position] = answer_weight
    if not math.isclose(math.fsum(weights), 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError("constructed token credit does not sum to one")

    return FlashTraceCredit(
        advantage=advantage_value,
        rollout_token_ids=rollout_ids,
        reasoning_positions=reasoning_positions,
        answer_positions=answer_positions,
        token_weights=tuple(weights),
        history_token_count=len(history_ids),
        answer_target_token_ids=answer_ids,
    )
