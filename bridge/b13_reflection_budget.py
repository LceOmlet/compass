from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from gepa_artifact.gepa.instruction_proposal import ProposeNewInstructionModule


class _OfficialPromptCaptureLM:
    """Capture the prompt and output budget emitted by the pinned proposer."""

    def __init__(self) -> None:
        self.prompt: str | None = None
        self.max_tokens: int | None = None

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        if self.prompt is not None:
            raise RuntimeError("the official proposer unexpectedly called its LM more than once")
        if not isinstance(prompt, str) or messages is not None:
            raise TypeError("the pinned proposer must issue one text-prompt LM call")
        if set(kwargs) != {"max_tokens"}:
            raise RuntimeError("the pinned proposer LM arguments changed")
        max_tokens = kwargs["max_tokens"]
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, Integral) or max_tokens <= 0:
            raise TypeError("the official reflection max_tokens must be a positive integer")
        self.prompt = prompt
        self.max_tokens = int(max_tokens)
        return ["```prompt capture```"]


@dataclass(frozen=True, slots=True)
class ReflectionInputSelection:
    samples: tuple[Any, ...]
    excluded_positions: tuple[int, ...]
    prompt_tokens: int | None
    official_max_output_tokens: int
    combined_fits: bool


class OfficialReflectionInputBudget:
    """Use the official proposer renderer to exclude individually overlong records."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        context_window_tokens: Integral,
        enable_thinking: bool,
    ) -> None:
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise TypeError("tokenizer must expose the official apply_chat_template")
        if (
            isinstance(context_window_tokens, bool)
            or not isinstance(context_window_tokens, Integral)
            or context_window_tokens <= 0
        ):
            raise ValueError("context_window_tokens must be a positive integer")
        if enable_thinking is not True:
            raise ValueError("the Qwen reflection budget requires enable_thinking=true")
        self._tokenizer = tokenizer
        self._context_window_tokens = int(context_window_tokens)

    def _measure(
        self,
        *,
        base_program: Any,
        samples: Sequence[Any],
    ) -> tuple[int, int]:
        if not samples:
            raise ValueError("reflection prompt measurement requires at least one sample")
        capture = _OfficialPromptCaptureLM()
        proposer = ProposeNewInstructionModule(
            base_program=base_program,
            instruction_lm=capture,
            dataset_with_feedback=list(samples),
            knowledgebase_qe=None,
        )
        proposer.compile()
        if capture.prompt is None or capture.max_tokens is None:
            raise RuntimeError("the official proposer did not issue its reflection LM call")
        prompt_ids = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": capture.prompt}],
            tokenize=True,
            add_generation_prompt=True,
            continue_final_message=False,
            enable_thinking=True,
        )
        prompt_tokens = len(prompt_ids)
        if prompt_tokens <= 0:
            raise RuntimeError("the official reflection prompt tokenization is empty")
        return prompt_tokens, capture.max_tokens

    def select(
        self,
        *,
        base_program: Any,
        dataset_with_feedback: Sequence[Any],
    ) -> ReflectionInputSelection:
        samples = tuple(dataset_with_feedback)
        if not samples:
            raise ValueError("reflection input selection requires a non-empty dataset")

        full_prompt_tokens, official_max_output_tokens = self._measure(
            base_program=base_program,
            samples=samples,
        )
        if full_prompt_tokens + official_max_output_tokens <= self._context_window_tokens:
            return ReflectionInputSelection(
                samples=samples,
                excluded_positions=(),
                prompt_tokens=full_prompt_tokens,
                official_max_output_tokens=official_max_output_tokens,
                combined_fits=True,
            )

        retained: list[Any] = []
        excluded: list[int] = []
        for position, sample in enumerate(samples):
            prompt_tokens, max_output_tokens = self._measure(
                base_program=base_program,
                samples=(sample,),
            )
            if max_output_tokens != official_max_output_tokens:
                raise RuntimeError("the official reflection output budget changed across samples")
            if prompt_tokens + max_output_tokens <= self._context_window_tokens:
                retained.append(sample)
            else:
                excluded.append(position)

        if not retained:
            return ReflectionInputSelection(
                samples=(),
                excluded_positions=tuple(excluded),
                prompt_tokens=None,
                official_max_output_tokens=official_max_output_tokens,
                combined_fits=False,
            )

        retained_prompt_tokens, retained_max_output_tokens = self._measure(
            base_program=base_program,
            samples=retained,
        )
        if retained_max_output_tokens != official_max_output_tokens:
            raise RuntimeError("the official reflection output budget changed after filtering")
        combined_fits = (
            retained_prompt_tokens + retained_max_output_tokens
            <= self._context_window_tokens
        )
        return ReflectionInputSelection(
            samples=tuple(retained),
            excluded_positions=tuple(excluded),
            prompt_tokens=retained_prompt_tokens,
            official_max_output_tokens=official_max_output_tokens,
            combined_fits=combined_fits,
        )
