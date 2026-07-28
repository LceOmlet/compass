from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any


QWEN3_THINK_OPEN_ID = 151667
QWEN3_THINK_CLOSE_ID = 151668
QWEN3_IM_END_ID = 151645


ArborResponseTokenKey = tuple[tuple[int, ...], tuple[int, ...]]
TOKEN_ID_MODE_PROVIDER_EXACT = "provider_exact"
TOKEN_ID_MODE_LOCAL_CANONICAL = "local_canonical"


@dataclass(frozen=True, slots=True)
class QwenNativeTokenLayout:
    """Exact ``FORMAT/T/FORMAT/O/EOS`` partition of one Arbor completion."""

    full_token_ids: tuple[int, ...]
    nonterminal_token_ids: tuple[int, ...]
    opening_positions: tuple[int, ...]
    reasoning_positions: tuple[int, ...]
    closing_positions: tuple[int, ...]
    output_positions: tuple[int, ...]
    eos_position: int

    @property
    def history_positions(self) -> tuple[int, ...]:
        return self.opening_positions + self.reasoning_positions + self.closing_positions


def _decoded_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    value = tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(value, str):
        raise TypeError("the Qwen tokenizer must decode token IDs to text")
    return value


def _validate_qwen_control_ids(tokenizer: Any) -> None:
    for text, expected_id in (
        ("<think>", QWEN3_THINK_OPEN_ID),
        ("</think>", QWEN3_THINK_CLOSE_ID),
    ):
        encoded = tuple(tokenizer.encode(text, add_special_tokens=False))
        if encoded != (expected_id,):
            raise RuntimeError(f"the pinned Qwen tokenizer changed the {text!r} token ID")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, bool) or not isinstance(eos_id, Integral):
        raise TypeError("the Qwen tokenizer must expose one integer eos_token_id")
    if int(eos_id) != QWEN3_IM_END_ID:
        raise RuntimeError("the pinned Qwen tokenizer changed the <|im_end|> EOS token ID")


def build_qwen_native_token_layout(
    *,
    tokenizer: Any,
    completion_token_ids: Sequence[int],
    raw_completion_text: str,
    native_reasoning_text: str,
    native_output_text: str,
) -> QwenNativeTokenLayout:
    """Partition captured IDs directly; text is validation metadata, never re-encoded."""

    if not isinstance(raw_completion_text, str) or not raw_completion_text:
        raise ValueError("the Qwen raw completion must be non-empty text")
    if not isinstance(native_reasoning_text, str) or not native_reasoning_text:
        raise ValueError("the Qwen reasoning_content must be non-empty text")
    if not isinstance(native_output_text, str) or not native_output_text:
        raise ValueError("the Qwen visible content must be non-empty text")
    values = tuple(completion_token_ids)
    if any(isinstance(token_id, bool) or not isinstance(token_id, Integral) for token_id in values):
        raise TypeError("Arbor completion token IDs must be integers")
    full_ids = tuple(int(token_id) for token_id in values)
    if any(token_id < 0 for token_id in full_ids):
        raise ValueError("Arbor completion token IDs must be non-negative")

    _validate_qwen_control_ids(tokenizer)
    if len(full_ids) < 5:
        raise ValueError("the native completion must contain FORMAT, T, O, and terminal EOS")
    if full_ids[-1] != QWEN3_IM_END_ID or full_ids.count(QWEN3_IM_END_ID) != 1:
        raise ValueError("the native completion must end in exactly one tokenizer EOS")
    nonterminal_ids = full_ids[:-1]
    if nonterminal_ids[0] != QWEN3_THINK_OPEN_ID:
        raise ValueError("the native completion must begin with the Qwen <think> token")
    if nonterminal_ids.count(QWEN3_THINK_OPEN_ID) != 1:
        raise ValueError("the native completion must contain exactly one Qwen <think> token")
    close_positions = tuple(
        index for index, token_id in enumerate(nonterminal_ids) if token_id == QWEN3_THINK_CLOSE_ID
    )
    if len(close_positions) != 1:
        raise ValueError("the native completion must contain exactly one Qwen </think> token")
    close_position = close_positions[0]
    if close_position <= 1 or close_position >= len(nonterminal_ids) - 1:
        raise ValueError("the native reasoning and visible output token spans must be non-empty")

    reasoning_positions = tuple(range(1, close_position))
    output_positions = tuple(range(close_position + 1, len(nonterminal_ids)))
    decoded_raw = _decoded_ids(tokenizer, nonterminal_ids)
    decoded_reasoning = _decoded_ids(
        tokenizer,
        tuple(nonterminal_ids[position] for position in reasoning_positions),
    )
    decoded_output = _decoded_ids(
        tokenizer,
        tuple(nonterminal_ids[position] for position in output_positions),
    )
    expected_raw = f"<think>{native_reasoning_text}</think>{native_output_text}"
    if decoded_raw != raw_completion_text or raw_completion_text != expected_raw:
        raise RuntimeError("captured IDs, raw completion, and provider T/O fields disagree")
    if decoded_reasoning != native_reasoning_text:
        raise RuntimeError("captured reasoning token IDs differ from provider reasoning_content")
    if decoded_output != native_output_text:
        raise RuntimeError("captured output token IDs differ from provider visible content")

    return QwenNativeTokenLayout(
        full_token_ids=full_ids,
        nonterminal_token_ids=nonterminal_ids,
        opening_positions=(0,),
        reasoning_positions=reasoning_positions,
        closing_positions=(close_position,),
        output_positions=output_positions,
        eos_position=len(full_ids) - 1,
    )


class QwenNativeCompletion(str):
    """Exact native ``<think>T</think>O`` carrying provider response fields."""

    native_reasoning_text: str
    native_output_text: str
    finish_reason: str
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]

    def __new__(
        cls,
        native_output_text: str,
        *,
        native_reasoning_text: str,
        finish_reason: str,
        raw_completion_text: str,
        prompt_token_ids: Sequence[int],
        completion_token_ids: Sequence[int],
    ) -> QwenNativeCompletion:
        value = super().__new__(cls, raw_completion_text)
        value.native_reasoning_text = native_reasoning_text
        value.native_output_text = native_output_text
        value.finish_reason = finish_reason
        value.prompt_token_ids = tuple(int(token_id) for token_id in prompt_token_ids)
        value.completion_token_ids = tuple(int(token_id) for token_id in completion_token_ids)
        return value


class QwenNativeOutput(str):
    """Visible output carrying only the exact Arbor response-token identity."""

    arbor_response_token_key: ArborResponseTokenKey

    def __new__(
        cls,
        text: str,
        *,
        prompt_token_ids: Sequence[int],
        completion_token_ids: Sequence[int],
    ) -> QwenNativeOutput:
        value = super().__new__(cls, text)
        value.arbor_response_token_key = (
            tuple(int(token_id) for token_id in prompt_token_ids),
            tuple(int(token_id) for token_id in completion_token_ids),
        )
        return value


@dataclass(frozen=True, slots=True)
class DecodedQwenChoice:
    raw_completion_text: str
    native_reasoning_text: str
    native_output_text: str
    finish_reason: str
    completion_token_ids: tuple[int, ...]
    token_layout: QwenNativeTokenLayout


def decode_qwen_choice(*, tokenizer: Any, choice: Any) -> DecodedQwenChoice:
    """Recover exact raw text from Arbor token IDs and preserve provider T/O fields."""

    message = getattr(choice, "message", None)
    native_output = getattr(message, "content", None)
    native_reasoning = getattr(message, "reasoning_content", None)
    finish_reason = getattr(choice, "finish_reason", None)
    for name, value in (
        ("visible content", native_output),
        ("reasoning_content", native_reasoning),
        ("finish_reason", finish_reason),
    ):
        if value is not None and not isinstance(value, str):
            raise TypeError(f"the Qwen-native response has non-textual {name}")

    token_ids = getattr(choice, "token_ids", None)
    if token_ids is None:
        provider_fields = getattr(choice, "provider_specific_fields", None)
        if isinstance(provider_fields, Mapping):
            token_ids = provider_fields.get("token_ids")
    if token_ids is None:
        raise ValueError("the Arbor response omitted completion token IDs")
    values = tuple(token_ids)
    if any(isinstance(token_id, bool) or not isinstance(token_id, Integral) for token_id in values):
        raise TypeError("Arbor completion token IDs must be integers")
    completion_ids = tuple(int(token_id) for token_id in values)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, bool) or not isinstance(eos_id, Integral):
        raise TypeError("the Qwen tokenizer must expose one integer eos_token_id")
    raw_ids = completion_ids[:-1] if completion_ids and completion_ids[-1] == int(eos_id) else completion_ids
    raw_completion = _decoded_ids(tokenizer, raw_ids)
    native_reasoning_text = native_reasoning or ""
    native_output_text = native_output or ""
    token_layout = build_qwen_native_token_layout(
        tokenizer=tokenizer,
        completion_token_ids=completion_ids,
        raw_completion_text=raw_completion,
        native_reasoning_text=native_reasoning_text,
        native_output_text=native_output_text,
    )
    return DecodedQwenChoice(
        raw_completion_text=raw_completion,
        native_reasoning_text=native_reasoning_text,
        native_output_text=native_output_text,
        finish_reason=finish_reason or "",
        completion_token_ids=completion_ids,
        token_layout=token_layout,
    )


def canonicalize_qwen_choice(*, tokenizer: Any, choice: Any) -> DecodedQwenChoice:
    """Map remote Qwen text onto one deterministic local token sequence."""

    message = getattr(choice, "message", None)
    visible = getattr(message, "content", None)
    reasoning = getattr(message, "reasoning_content", None)
    finish_reason = getattr(choice, "finish_reason", None)
    for name, value in (
        ("visible content", visible),
        ("reasoning_content", reasoning),
        ("finish_reason", finish_reason),
    ):
        if value is not None and not isinstance(value, str):
            raise TypeError(f"the Qwen-native response has non-textual {name}")

    visible_text = visible or ""
    reasoning_text = reasoning or ""
    if visible_text.startswith("<think>"):
        if visible_text.count("<think>") != 1 or visible_text.count("</think>") != 1:
            raise ValueError("the remote Qwen content has an ambiguous native thinking block")
        close = visible_text.index("</think>")
        embedded_reasoning = visible_text[len("<think>") : close]
        embedded_output = visible_text[close + len("</think>") :]
        if reasoning_text and reasoning_text != embedded_reasoning:
            raise ValueError("remote reasoning_content disagrees with the embedded thinking block")
        reasoning_text = embedded_reasoning
        visible_text = embedded_output
    if not reasoning_text:
        raise ValueError("the Qwen reasoning_content must be non-empty text")
    if not visible_text:
        raise ValueError("the Qwen visible content must be non-empty text")

    _validate_qwen_control_ids(tokenizer)
    reasoning_ids = tuple(tokenizer.encode(reasoning_text, add_special_tokens=False))
    output_ids = tuple(tokenizer.encode(visible_text, add_special_tokens=False))
    if not reasoning_ids or not output_ids:
        raise ValueError("local canonicalization produced an empty reasoning or output span")
    canonical_reasoning = _decoded_ids(tokenizer, reasoning_ids)
    canonical_output = _decoded_ids(tokenizer, output_ids)
    completion_ids = (
        QWEN3_THINK_OPEN_ID,
        *reasoning_ids,
        QWEN3_THINK_CLOSE_ID,
        *output_ids,
        QWEN3_IM_END_ID,
    )
    raw_completion = f"<think>{canonical_reasoning}</think>{canonical_output}"
    token_layout = build_qwen_native_token_layout(
        tokenizer=tokenizer,
        completion_token_ids=completion_ids,
        raw_completion_text=raw_completion,
        native_reasoning_text=canonical_reasoning,
        native_output_text=canonical_output,
    )
    return DecodedQwenChoice(
        raw_completion_text=raw_completion,
        native_reasoning_text=canonical_reasoning,
        native_output_text=canonical_output,
        finish_reason=finish_reason or "",
        completion_token_ids=completion_ids,
        token_layout=token_layout,
    )
