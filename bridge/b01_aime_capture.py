from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Any

import dspy
from dspy.teleprompt.bootstrap_finetune import FailedPrediction
from dspy.utils.callback import BaseCallback

from .b12_qwen_native_aime import QwenNativeAIMEAdapter
from .qwen_native_tokens import (
    ArborResponseTokenKey,
    QwenNativeOutput,
    QwenNativeTokenLayout,
    TOKEN_ID_MODE_LOCAL_CANONICAL,
    TOKEN_ID_MODE_PROVIDER_EXACT,
    canonicalize_qwen_choice,
    decode_qwen_choice,
)


@dataclass(frozen=True, slots=True)
class ArborTokenRecord:
    prompt_messages: tuple[dict[str, Any], ...]
    prompt_text: str
    completion_text: str
    native_reasoning_text: str
    native_output_text: str
    finish_reason: str
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    token_layout: QwenNativeTokenLayout


@dataclass(frozen=True, slots=True)
class CapturedRollout:
    iteration: int
    instance_index: int
    reward: Real
    skill_text: str
    signature: Any
    demos: tuple[Any, ...]
    reasoning_text: str
    answer_text: str
    native_reasoning_text: str
    native_output_text: str
    module_inputs: dspy.Example
    predictor_inputs: dict[str, Any]
    token_record: ArborTokenRecord


class ArborTokenCallback(BaseCallback):
    """Match raw Arbor responses to the exact official DSPy chat messages."""

    def __init__(self, *, tokenizer: Any, token_id_mode: str) -> None:
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise TypeError("tokenizer must expose the official apply_chat_template")
        self._tokenizer = tokenizer
        if token_id_mode not in {
            TOKEN_ID_MODE_PROVIDER_EXACT,
            TOKEN_ID_MODE_LOCAL_CANONICAL,
        }:
            raise ValueError("token_id_mode must be provider_exact or local_canonical")
        self._token_id_mode = token_id_mode
        self._lock = threading.RLock()
        self._lm_inputs: dict[str, dict[str, Any]] = {}
        self._pending: dict[ArborResponseTokenKey, list[ArborTokenRecord]] = {}
        self._pending_by_prompt: dict[tuple[int, ...], list[ArborResponseTokenKey]] = {}
        self._errors_by_prompt: dict[tuple[int, ...], list[BaseException]] = {}

    def _prompt_key(self, messages: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
        normalized: list[dict[str, str]] = []
        for message in messages:
            if set(message) != {"role", "content"}:
                raise TypeError("the Arbor task messages must contain only role and content")
            role = message["role"]
            content = message["content"]
            if not isinstance(role, str) or not isinstance(content, str):
                raise TypeError("the Arbor task message role and content must be text")
            normalized.append({"role": role, "content": content})
        if not normalized:
            raise ValueError("the Arbor task message list must be non-empty")
        values = self._tokenizer.apply_chat_template(
            normalized,
            tokenize=True,
            add_generation_prompt=True,
            continue_final_message=False,
            enable_thinking=True,
        )
        return self._integer_ids(values, "rendered task prompt token IDs")

    def _prompt_text(self, messages: Sequence[Mapping[str, Any]]) -> str:
        normalized = [dict(message) for message in messages]
        value = self._tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
            enable_thinking=True,
        )
        if not isinstance(value, str) or not value:
            raise TypeError("the task chat template must render non-empty text")
        return value

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        with self._lock:
            self._lm_inputs[call_id] = copy.deepcopy(inputs)

    @staticmethod
    def _integer_ids(value: Any, field: str) -> tuple[int, ...]:
        if value is None:
            raise ValueError(f"Arbor response omitted {field}")
        result = tuple(value)
        if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
            raise TypeError(f"{field} must contain integer token IDs")
        return result

    def _record_from_response(self, call_id: str, response: Any) -> ArborTokenRecord | None:
        choices = getattr(response, "choices", None)
        if choices is None:
            return None
        if len(choices) != 1:
            raise ValueError("the first AIME bridge requires exactly one Arbor completion")

        with self._lock:
            inputs = self._lm_inputs.get(call_id)
        if inputs is None:
            raise RuntimeError("the raw Arbor response has no matching LM call inputs")
        messages = inputs.get("messages")
        if not isinstance(messages, list):
            raise TypeError("the Arbor task call must carry DSPy chat messages")

        choice = choices[0]
        if self._token_id_mode == TOKEN_ID_MODE_PROVIDER_EXACT:
            decoded = decode_qwen_choice(tokenizer=self._tokenizer, choice=choice)
            prompt_token_ids = self._integer_ids(
                getattr(response, "prompt_token_ids", None),
                "prompt_token_ids",
            )
        else:
            decoded = canonicalize_qwen_choice(tokenizer=self._tokenizer, choice=choice)
            prompt_token_ids = self._prompt_key(messages)
        visible_completion_text = decoded.native_output_text
        reasoning_content = decoded.native_reasoning_text
        completion_text = decoded.raw_completion_text
        finish_reason = decoded.finish_reason
        if finish_reason != "stop":
            raise ValueError(f"the complete-rollout bridge requires finish_reason='stop', got {finish_reason!r}")

        return ArborTokenRecord(
            prompt_messages=tuple(copy.deepcopy(messages)),
            prompt_text=self._prompt_text(messages),
            completion_text=completion_text,
            native_reasoning_text=reasoning_content,
            native_output_text=visible_completion_text,
            finish_reason=finish_reason,
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=self._integer_ids(
                decoded.completion_token_ids,
                "choices[0].token_ids",
            ),
            token_layout=decoded.token_layout,
        )

    def on_lm_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        with self._lock:
            inputs = self._lm_inputs.get(call_id)
        if exception is not None or outputs is None or inputs is None:
            with self._lock:
                self._lm_inputs.pop(call_id, None)
            return
        messages = inputs.get("messages")
        if not isinstance(messages, list):
            with self._lock:
                self._lm_inputs.pop(call_id, None)
            return
        try:
            record = self._record_from_response(call_id, outputs)
        except BaseException as error:
            prompt_key = self._prompt_key(messages)
            with self._lock:
                self._lm_inputs.pop(call_id, None)
                self._errors_by_prompt.setdefault(prompt_key, []).append(error)
            return
        with self._lock:
            self._lm_inputs.pop(call_id, None)
        if record is not None:
            key = (record.prompt_token_ids, record.completion_token_ids)
            with self._lock:
                self._pending.setdefault(key, []).append(record)
                self._pending_by_prompt.setdefault(record.prompt_token_ids, []).append(key)

    def take_for_prediction(
        self,
        messages: Sequence[Mapping[str, Any]],
        native_output: str,
    ) -> ArborTokenRecord:
        if not isinstance(native_output, QwenNativeOutput):
            raise RuntimeError("Prediction.native_output omitted the exact Arbor response-token key")
        key = native_output.arbor_response_token_key
        expected_prompt = self._prompt_key(messages)
        if key[0] != expected_prompt:
            raise RuntimeError("Prediction is bound to different official task messages")
        with self._lock:
            values = self._pending.get(key)
            if not values:
                raise RuntimeError("no Arbor token record matches the exact Prediction response")
            value = values.pop(0)
            if not values:
                del self._pending[key]
            prompt_values = self._pending_by_prompt.get(expected_prompt)
            if not prompt_values:
                raise RuntimeError("the Arbor prompt index lost an exact response record")
            try:
                prompt_values.remove(key)
            except ValueError as error:
                raise RuntimeError("the Arbor prompt and response indices disagree") from error
            if not prompt_values:
                del self._pending_by_prompt[expected_prompt]
        return replace(value, prompt_messages=tuple(copy.deepcopy(messages)))

    def take_for_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> ArborTokenRecord:
        """Bind a plain official ChatAdapter prediction to its FIFO raw response."""

        prompt_key = self._prompt_key(messages)
        with self._lock:
            response_keys = self._pending_by_prompt.get(prompt_key)
            if not response_keys:
                errors = self._errors_by_prompt.get(prompt_key)
                if errors:
                    error = errors.pop(0)
                    if not errors:
                        del self._errors_by_prompt[prompt_key]
                    raise RuntimeError(
                        "the matching official task response could not be canonicalized"
                    ) from error
                raise RuntimeError("no Arbor token record matches the official task messages")
            response_key = response_keys.pop(0)
            if not response_keys:
                del self._pending_by_prompt[prompt_key]
            values = self._pending.get(response_key)
            if not values:
                raise RuntimeError("the Arbor prompt index references a missing response record")
            value = values.pop(0)
            if not values:
                del self._pending[response_key]
        return replace(value, prompt_messages=tuple(copy.deepcopy(messages)))

    def take_for_messages_matching(
        self,
        messages: Sequence[Mapping[str, Any]],
        predicate: Callable[[ArborTokenRecord], bool],
    ) -> ArborTokenRecord:
        """Consume the unique canonical response matching one parsed Prediction."""

        if not callable(predicate):
            raise TypeError("predicate must be callable")
        prompt_key = self._prompt_key(messages)
        with self._lock:
            response_keys = self._pending_by_prompt.get(prompt_key)
            if not response_keys:
                errors = self._errors_by_prompt.get(prompt_key)
                if errors:
                    error = errors.pop(0)
                    if not errors:
                        del self._errors_by_prompt[prompt_key]
                    raise RuntimeError(
                        "the matching official task response could not be canonicalized"
                    ) from error
                raise RuntimeError("no Arbor token record matches the official task messages")

            occurrences: dict[ArborResponseTokenKey, int] = {}
            matches: list[tuple[int, ArborResponseTokenKey, int]] = []
            for prompt_position, response_key in enumerate(response_keys):
                value_position = occurrences.get(response_key, 0)
                occurrences[response_key] = value_position + 1
                values = self._pending.get(response_key)
                if values is None or value_position >= len(values):
                    raise RuntimeError("the Arbor prompt and response indices disagree")
                if predicate(values[value_position]):
                    matches.append((prompt_position, response_key, value_position))
            if len(matches) != 1:
                raise RuntimeError(
                    "the official task Prediction does not identify one canonical response"
                )

            prompt_position, response_key, value_position = matches[0]
            value = self._pending[response_key].pop(value_position)
            if not self._pending[response_key]:
                del self._pending[response_key]
            response_keys.pop(prompt_position)
            if not response_keys:
                del self._pending_by_prompt[prompt_key]
        return replace(value, prompt_messages=tuple(copy.deepcopy(messages)))

    def discard_for_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> None:
        self.take_for_messages(messages)

    def discard_pending_records(self) -> int:
        """Clear completed, unclaimed LM responses at a task-batch boundary."""

        with self._lock:
            if self._lm_inputs:
                raise RuntimeError("cannot discard token records while LM calls are still active")
            count = sum(len(values) for values in self._pending.values())
            self._pending.clear()
            self._pending_by_prompt.clear()
            self._errors_by_prompt.clear()
        return count


class CapturingFeedback:
    """Record AIME feedback without changing the official feedback result."""

    def __init__(
        self,
        *,
        delegate: Callable[..., dict[str, Any]],
        token_callback: ArborTokenCallback,
        chat_adapter: QwenNativeAIMEAdapter,
        trainset: Sequence[dspy.Example],
        failed_prediction_feedback: str | None,
    ) -> None:
        self._delegate = delegate
        self._token_callback = token_callback
        if not isinstance(chat_adapter, QwenNativeAIMEAdapter):
            raise TypeError("chat_adapter must be the configured QwenNativeAIMEAdapter")
        self._chat_adapter = chat_adapter
        self._trainset = tuple(trainset)
        if failed_prediction_feedback is not None and (
            not isinstance(failed_prediction_feedback, str) or not failed_prediction_feedback
        ):
            raise ValueError("failed_prediction_feedback must be non-empty text or None")
        self._failed_prediction_feedback = failed_prediction_feedback
        self._gepa: Any | None = None
        self._lock = threading.RLock()
        self._records: dict[int, dict[int, CapturedRollout]] = {}

    def bind_gepa(self, optimizer: Any) -> None:
        if self._gepa is not None:
            raise RuntimeError("the feedback bridge is already bound to a GEPA instance")
        self._gepa = optimizer

    def _current_state(self) -> tuple[Any, int, tuple[int, ...]]:
        if self._gepa is None or self._gepa.gepa_state is None:
            raise RuntimeError("the feedback bridge is not bound to an active GEPA state")
        state = self._gepa.gepa_state
        if not state.full_program_trace:
            raise RuntimeError("GEPA has no current full_program_trace entry")
        iteration = state.i
        subsample_ids = tuple(state.full_program_trace[-1]["subsample_ids"])
        return state, iteration, subsample_ids

    def _instance_index(self, module_inputs: dspy.Example, subsample_ids: tuple[int, ...]) -> int:
        matches = [index for index in subsample_ids if self._trainset[index] is module_inputs]
        if len(matches) != 1:
            raise RuntimeError(
                "module_inputs must map by object identity to exactly one current GEPA subsample index"
            )
        return matches[0]

    @staticmethod
    def _predictor_state(
        predictor_output: dspy.Prediction,
        captured_trace: Sequence[tuple[Any, dict[str, Any], Any]],
    ) -> tuple[str, Any, tuple[Any, ...]]:
        matches = [entry for entry in captured_trace if entry[2] is predictor_output]
        if len(matches) != 1:
            raise RuntimeError("predictor_output must occur exactly once by identity in captured_trace")
        predictor = matches[0][0]
        signature = getattr(predictor, "signature", None)
        if signature is None:
            raise TypeError("the selected predictor has no signature")
        skill = signature.instructions
        if not isinstance(skill, str):
            raise TypeError("the selected predictor instruction must be text")
        demos_value = getattr(predictor, "demos", None)
        if not isinstance(demos_value, list | tuple):
            raise TypeError("the selected predictor demos must be an ordered sequence")
        demos = tuple(copy.deepcopy(demo) for demo in demos_value)
        return skill, signature, demos

    def __call__(
        self,
        predictor_output: dspy.Prediction,
        predictor_inputs: dict[str, Any],
        module_inputs: dspy.Example,
        module_outputs: dspy.Prediction,
        captured_trace: list[tuple[Any, dict[str, Any], Any]],
    ) -> dict[str, Any]:
        official_result = self._delegate(
            predictor_output=predictor_output,
            predictor_inputs=predictor_inputs,
            module_inputs=module_inputs,
            module_outputs=module_outputs,
            captured_trace=captured_trace,
        )
        if not isinstance(official_result, dict) or "feedback_score" not in official_result:
            raise TypeError("the official feedback delegate must return feedback_score")
        reward = official_result["feedback_score"]
        if isinstance(reward, bool) or not isinstance(reward, Real):
            raise TypeError("feedback_score must be numeric")

        reasoning_text = getattr(predictor_output, "reasoning", None)
        native_output_text = getattr(predictor_output, "native_output", None)
        answer_text = getattr(predictor_output, "answer", None)
        if not isinstance(reasoning_text, str):
            raise TypeError("the official CoT prediction reasoning must be text")
        if not isinstance(answer_text, str):
            raise TypeError("the official CoT prediction answer must be text")
        if not isinstance(native_output_text, str) or not native_output_text:
            raise ValueError("the Qwen-native visible output must be non-empty text")
        skill_text, signature, demos = self._predictor_state(predictor_output, captured_trace)
        prompt_messages = self._chat_adapter.format(
            signature,
            list(demos),
            predictor_inputs,
        )

        _, iteration, subsample_ids = self._current_state()
        instance_index = self._instance_index(module_inputs, subsample_ids)
        token_record = self._token_callback.take_for_prediction(
            prompt_messages,
            native_output_text,
        )
        if token_record.native_reasoning_text != reasoning_text:
            raise RuntimeError("Prediction.reasoning differs from the captured Qwen reasoning_content")
        if token_record.native_output_text != native_output_text:
            raise RuntimeError("Prediction.native_output differs from the captured Qwen visible content")
        record = CapturedRollout(
            iteration=iteration,
            instance_index=instance_index,
            reward=reward,
            skill_text=skill_text,
            signature=signature,
            demos=demos,
            reasoning_text=reasoning_text,
            answer_text=answer_text,
            native_reasoning_text=reasoning_text,
            native_output_text=native_output_text,
            module_inputs=module_inputs,
            predictor_inputs=dict(predictor_inputs),
            token_record=token_record,
        )

        with self._lock:
            iteration_records = self._records.setdefault(iteration, {})
            if instance_index in iteration_records:
                raise RuntimeError("the AIME bridge captured the same instance twice in one GEPA iteration")
            iteration_records[instance_index] = record
        return official_result

    def current_minibatch(self) -> tuple[CapturedRollout, ...]:
        _, iteration, subsample_ids = self._current_state()
        with self._lock:
            iteration_records = dict(self._records.get(iteration, {}))
        ordered = tuple(iteration_records[index] for index in subsample_ids if index in iteration_records)
        if not ordered:
            raise RuntimeError("the current GEPA iteration has no captured parsed AIME rollout")
        if len(ordered) != len(iteration_records):
            raise RuntimeError("captured rollout indices do not match the current GEPA subsample")
        return ordered

    def discard_failed_predictions(
        self,
        *,
        predictor: Any,
        dataset_with_feedback: Sequence[Mapping[str, Any]],
    ) -> None:
        signature = getattr(predictor, "signature", None)
        demos = getattr(predictor, "demos", None)
        if signature is None or not isinstance(demos, list | tuple):
            raise TypeError("the official predictor must expose signature and demos")
        for sample in dataset_with_feedback:
            generated = sample.get("generated_output")
            if not isinstance(generated, FailedPrediction):
                continue
            inputs = sample.get("inputs")
            if not isinstance(inputs, Mapping):
                raise TypeError("failed official feedback sample omitted predictor inputs")
            messages = self._chat_adapter.format(signature, list(demos), dict(inputs))
            self._discard_for_messages(messages, allow_missing=True)

    def rewrite_failed_prediction_feedback(
        self,
        dataset_with_feedback: Sequence[Mapping[str, Any]],
    ) -> None:
        """Replace only GEPA's DSPy-format parse advice for the native task path."""

        if self._failed_prediction_feedback is None:
            return
        for sample in dataset_with_feedback:
            if not isinstance(sample.get("generated_output"), FailedPrediction):
                continue
            if not isinstance(sample, dict):
                raise TypeError("failed feedback samples must be mutable dictionaries")
            if "feedback" not in sample:
                raise RuntimeError("failed official feedback sample omitted feedback text")
            sample["feedback"] = self._failed_prediction_feedback

    def _discard_for_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        allow_missing: bool = False,
    ) -> None:
        try:
            self._token_callback.discard_for_messages(messages)
        except RuntimeError as error:
            message = str(error)
            if allow_missing and message == "no Arbor token record matches the official task messages":
                return
            raise

    def discard_program_rollouts(
        self,
        *,
        program: dspy.Module,
        examples: Sequence[dspy.Example],
        predictions: Sequence[dspy.Prediction | None],
    ) -> None:
        prediction_values = tuple(predictions)
        if len(prediction_values) != len(examples):
            raise ValueError("candidate predictions must align with candidate examples")
        predictors = tuple(program.named_predictors())
        if len(predictors) != 1:
            raise ValueError("AIME token cleanup requires exactly one predictor")
        _, predictor = predictors[0]
        signature = getattr(predictor, "signature", None)
        demos = getattr(predictor, "demos", None)
        if signature is None or not isinstance(demos, list | tuple):
            raise TypeError("the official predictor must expose signature and demos")
        for example, prediction in zip(examples, prediction_values, strict=True):
            messages = self._chat_adapter.format(
                signature,
                list(demos),
                dict(example.inputs()),
            )
            if prediction is None:
                self._discard_for_messages(messages, allow_missing=True)
                continue
            native_output = getattr(prediction, "native_output", None)
            if not isinstance(native_output, str):
                raise TypeError("candidate Prediction.native_output must be text")
            self._token_callback.take_for_prediction(messages, native_output)
