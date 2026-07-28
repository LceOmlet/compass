from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

from dspy.adapters.chat_adapter import ChatAdapter
from gepa_artifact.benchmarks.livebench_math.livebenchmath_utils.util import (
    last_boxed_only_string,
    remove_boxed,
)

from .b00_artifact_lm import ArtifactLMDispatcher
from .qwen_native_tokens import (
    ArborResponseTokenKey,
    DecodedQwenChoice,
    QWEN3_IM_END_ID,
    QWEN3_THINK_CLOSE_ID,
    QWEN3_THINK_OPEN_ID,
    QwenNativeCompletion,
    QwenNativeOutput,
    QwenNativeTokenLayout,
    TOKEN_ID_MODE_LOCAL_CANONICAL,
    TOKEN_ID_MODE_PROVIDER_EXACT,
    build_qwen_native_token_layout,
    canonicalize_qwen_choice,
    decode_qwen_choice,
)


# Exact recommendation in the pinned Qwen3-8B checkpoint README, section
# "Best Practices", item 3 (Math Problems).
QWEN3_MATH_BOXED_INSTRUCTION = (
    r"Please reason step by step, and put your final answer within \boxed{}."
)


class QwenNativeAIMEAdapter(ChatAdapter):
    """Map the official AIME signature onto Qwen's native thinking protocol."""

    def __init__(self, *, math_instruction: str) -> None:
        super().__init__()
        if math_instruction != QWEN3_MATH_BOXED_INSTRUCTION:
            raise ValueError("math_instruction must equal the pinned Qwen3 README sentence")
        self._math_instruction = math_instruction

    @staticmethod
    def _validate_output_fields(signature: Any) -> None:
        output_fields = set(getattr(signature, "output_fields", {}))
        if output_fields not in (
            {"reasoning", "answer"},
            {"reasoning", "native_output", "answer"},
        ):
            raise ValueError(
                "the Qwen-native AIME bridge requires reasoning and answer, "
                "with native_output either explicit or supplied by this adapter"
            )

    def format(
        self,
        signature: Any,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, str]]:
        if demos:
            raise ValueError("the Qwen-native AIME bridge does not admit demonstrations")
        if set(getattr(signature, "input_fields", {})) != {"problem"}:
            raise ValueError("the Qwen-native AIME bridge requires exactly the problem input")
        self._validate_output_fields(signature)
        if set(inputs) != {"problem"}:
            raise ValueError("the Qwen-native AIME call requires exactly the problem value")
        skill = getattr(signature, "instructions", None)
        problem = inputs["problem"]
        if not isinstance(skill, str) or not skill:
            raise ValueError("the GEPA skill must be non-empty text")
        if not isinstance(problem, str) or not problem:
            raise ValueError("the AIME problem must be non-empty text")
        content = f"{skill}\n\n{problem}\n\n{self._math_instruction}"
        return [{"role": "user", "content": content}]

    def parse(self, signature: Any, completion: str) -> dict[str, Any]:
        self._validate_output_fields(signature)
        if not isinstance(completion, QwenNativeCompletion):
            raise ValueError("the task LM omitted the Qwen-native completion envelope")
        if completion.finish_reason != "stop":
            raise ValueError("the Qwen-native rollout did not terminate with finish_reason='stop'")
        reasoning = completion.native_reasoning_text
        native_output_text = completion.native_output_text
        if not reasoning:
            raise ValueError("the Qwen-native rollout has no completed thinking content")
        if not native_output_text:
            raise ValueError("the Qwen-native rollout has no visible output")
        expected_raw = f"<think>{reasoning}</think>{native_output_text}"
        if str(completion) != expected_raw:
            raise ValueError("the Qwen-native rollout is not one complete native thinking block")
        boxed = last_boxed_only_string(native_output_text)
        if boxed is None:
            raise ValueError("the Qwen-native visible output has no complete final boxed answer")
        answer = remove_boxed(boxed)
        native_output = QwenNativeOutput(
            native_output_text,
            prompt_token_ids=completion.prompt_token_ids,
            completion_token_ids=completion.completion_token_ids,
        )
        return {
            "reasoning": reasoning,
            "native_output": native_output,
            "answer": answer,
        }


class QwenNativeAIMELMDispatcher(ArtifactLMDispatcher):
    """Apply the explicit Qwen task budget while retaining Artifact reflection."""

    def __init__(
        self,
        *,
        task_tokenizer: Any,
        task_context_window_tokens: int,
        task_competition_max_output_tokens: int,
        reflection_max_output_tokens: int,
        token_id_mode: str,
        **kwargs: Any,
    ) -> None:
        if not callable(getattr(task_tokenizer, "apply_chat_template", None)):
            raise TypeError("task_tokenizer must expose apply_chat_template")
        if (
            isinstance(task_context_window_tokens, bool)
            or not isinstance(task_context_window_tokens, Integral)
            or task_context_window_tokens <= 0
        ):
            raise ValueError("task_context_window_tokens must be a positive integer")
        if (
            isinstance(task_competition_max_output_tokens, bool)
            or not isinstance(task_competition_max_output_tokens, Integral)
            or task_competition_max_output_tokens <= 0
        ):
            raise ValueError("task_competition_max_output_tokens must be a positive integer")
        if task_competition_max_output_tokens > task_context_window_tokens:
            raise ValueError("the competition output budget cannot exceed the context window")
        if (
            isinstance(reflection_max_output_tokens, bool)
            or not isinstance(reflection_max_output_tokens, Integral)
            or reflection_max_output_tokens <= 0
        ):
            raise ValueError("reflection_max_output_tokens must be a positive integer")
        self._task_tokenizer = task_tokenizer
        self._task_context_window_tokens = int(task_context_window_tokens)
        self._task_competition_max_output_tokens = int(task_competition_max_output_tokens)
        self._reflection_max_output_tokens = int(reflection_max_output_tokens)
        if token_id_mode not in {
            TOKEN_ID_MODE_PROVIDER_EXACT,
            TOKEN_ID_MODE_LOCAL_CANONICAL,
        }:
            raise ValueError("token_id_mode must be provider_exact or local_canonical")
        self._token_id_mode = token_id_mode
        super().__init__(**kwargs)

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[str] | list[dict[str, Any]]:
        if prompt is not None:
            requested = kwargs.get("max_tokens")
            if isinstance(requested, bool) or not isinstance(requested, Integral) or requested <= 0:
                raise ValueError("the official reflection call must request positive max_tokens")
            if requested > self._reflection_max_output_tokens:
                kwargs = dict(kwargs)
                kwargs["max_tokens"] = self._reflection_max_output_tokens
        return super().__call__(prompt=prompt, messages=messages, **kwargs)

    def _process_lm_response(
        self,
        response: Any,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        **kwargs: Any,
    ) -> list[str] | list[dict[str, Any]]:
        outputs = super()._process_lm_response(response, prompt, messages, **kwargs)
        if messages is None:
            return outputs
        choices = getattr(response, "choices", None)
        if not isinstance(choices, Sequence) or len(choices) != len(outputs):
            raise RuntimeError("the task response choices do not match the processed LM outputs")
        if self._token_id_mode == TOKEN_ID_MODE_PROVIDER_EXACT:
            prompt_token_ids = self._integer_ids(
                getattr(response, "prompt_token_ids", None),
                "provider response prompt_token_ids",
            )
        else:
            prompt_token_ids = self._rendered_prompt_ids(messages)
        native: list[str] = []
        for choice in choices:
            if self._token_id_mode == TOKEN_ID_MODE_PROVIDER_EXACT:
                decoded = decode_qwen_choice(tokenizer=self._task_tokenizer, choice=choice)
            else:
                decoded = canonicalize_qwen_choice(tokenizer=self._task_tokenizer, choice=choice)
            native.append(
                QwenNativeCompletion(
                    decoded.native_output_text,
                    native_reasoning_text=decoded.native_reasoning_text,
                    finish_reason=decoded.finish_reason,
                    raw_completion_text=decoded.raw_completion_text,
                    prompt_token_ids=prompt_token_ids,
                    completion_token_ids=decoded.completion_token_ids,
                )
            )
        return native

    @staticmethod
    def _integer_ids(value: Any, name: str) -> tuple[int, ...]:
        if value is None:
            raise ValueError(f"{name} were omitted")
        ids = tuple(value)
        if any(isinstance(token_id, bool) or not isinstance(token_id, Integral) for token_id in ids):
            raise TypeError(f"{name} must contain integer token IDs")
        return tuple(int(token_id) for token_id in ids)

    def _rendered_prompt_ids(self, messages: list[dict[str, Any]]) -> tuple[int, ...]:
        return self._integer_ids(
            self._task_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                continue_final_message=False,
                enable_thinking=True,
            ),
            "Qwen-native rendered prompt",
        )

    def _task_kwargs(
        self,
        call_kwargs: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if len(messages) != 1 or messages[0].get("role") != "user" or set(messages[0]) != {"role", "content"}:
            raise ValueError("the Qwen-native task request must contain one plain user message")
        prompt_ids = self._rendered_prompt_ids(messages)
        remaining = self._task_context_window_tokens - len(prompt_ids)
        if remaining <= 0:
            raise ValueError("the rendered Qwen-native prompt exhausts the context window")
        max_tokens = min(self._task_competition_max_output_tokens, remaining)

        routed = super()._task_kwargs(call_kwargs, messages)
        routed["max_tokens"] = max_tokens
        extra_body = routed["extra_body"]
        if not isinstance(extra_body, Mapping):
            raise TypeError("the merged task extra_body must be a mapping")
        merged_extra = dict(extra_body)
        chat_template_kwargs = merged_extra.get("chat_template_kwargs")
        if chat_template_kwargs is None:
            merged_chat_template_kwargs: dict[str, Any] = {}
        elif isinstance(chat_template_kwargs, Mapping):
            merged_chat_template_kwargs = dict(chat_template_kwargs)
        else:
            raise TypeError("extra_body.chat_template_kwargs must be a mapping")
        configured_thinking = merged_chat_template_kwargs.get("enable_thinking")
        if configured_thinking is not None and configured_thinking is not True:
            raise ValueError("enable_thinking conflicts with the Qwen-native task contract")
        merged_chat_template_kwargs["enable_thinking"] = True
        merged_extra["chat_template_kwargs"] = merged_chat_template_kwargs
        if self._token_id_mode == TOKEN_ID_MODE_LOCAL_CANONICAL:
            merged_extra.pop("return_token_ids", None)
        routed["extra_body"] = merged_extra
        return routed
