from __future__ import annotations

import copy
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.utils import format_field_value

from .b01_aime_capture import ArborTokenCallback, CapturedRollout
from .b02_flashtrace_credit import _encode_with_offsets, _integer_ids
from .b06_dependency_gate import (
    CharacterSpan,
    DependencyMeasureUnavailableError,
    RenderedPromptProvenance,
    TaskObservationField,
)
from .b09_aime_dependency import AIMEReplayDependencyInputs, _historical_target


class IFBenchCaptureCoordinator:
    """Bind one officially selected IFBench predictor to its raw Qwen response."""

    def __init__(
        self,
        *,
        token_callback: ArborTokenCallback,
        chat_adapter: ChatAdapter,
        trainset: Sequence[dspy.Example],
    ) -> None:
        if not isinstance(token_callback, ArborTokenCallback):
            raise TypeError("token_callback must be ArborTokenCallback")
        if not isinstance(chat_adapter, ChatAdapter):
            raise TypeError("chat_adapter must be the shared official ChatAdapter")
        self._token_callback = token_callback
        self._chat_adapter = chat_adapter
        self._trainset = tuple(trainset)
        self._gepa: Any | None = None
        self._lock = threading.RLock()
        self._records: dict[int, dict[int, CapturedRollout]] = {}
        self._capture_failures: list[Exception] = []

    def bind_gepa(self, optimizer: Any) -> None:
        if self._gepa is not None:
            raise RuntimeError("the IFBench capture coordinator is already bound")
        self._gepa = optimizer

    def _current_state(self) -> tuple[Any, int, tuple[int, ...], str]:
        if self._gepa is None or self._gepa.gepa_state is None:
            raise RuntimeError("the IFBench capture coordinator is not bound to active GEPA state")
        state = self._gepa.gepa_state
        if not state.full_program_trace:
            raise RuntimeError("GEPA has no current full_program_trace entry")
        trace = state.full_program_trace[-1]
        iteration = state.i
        subsample_ids = tuple(trace["subsample_ids"])
        predictor_id = trace["predictor_to_update_id"]
        predictor_name = state.list_of_named_predictors[predictor_id]
        return state, iteration, subsample_ids, predictor_name

    def _instance_index(
        self,
        module_inputs: dspy.Example,
        subsample_ids: tuple[int, ...],
    ) -> int:
        matches = [index for index in subsample_ids if self._trainset[index] is module_inputs]
        if len(matches) != 1:
            raise RuntimeError(
                "module_inputs must map by identity to one current IFBench subsample index"
            )
        return matches[0]

    @staticmethod
    def _selected_predictor_state(
        predictor_output: dspy.Prediction,
        predictor_inputs: Mapping[str, Any],
        captured_trace: Sequence[tuple[Any, dict[str, Any], Any]],
    ) -> tuple[Any, Any, tuple[Any, ...]]:
        matches = [entry for entry in captured_trace if entry[2] is predictor_output]
        if len(matches) != 1:
            raise RuntimeError("selected predictor output must occur once by identity in trace")
        predictor, traced_inputs, _ = matches[0]
        if dict(traced_inputs) != dict(predictor_inputs):
            raise RuntimeError("selected predictor inputs differ from the official trace")
        signature = getattr(predictor, "signature", None)
        demos = getattr(predictor, "demos", None)
        if signature is None or not isinstance(demos, list | tuple):
            raise TypeError("the official selected predictor must expose signature and demos")
        return predictor, signature, tuple(copy.deepcopy(demo) for demo in demos)

    def capture(
        self,
        *,
        predictor_name: str,
        official_result: Mapping[str, Any],
        predictor_output: dspy.Prediction,
        predictor_inputs: Mapping[str, Any],
        module_inputs: dspy.Example,
        captured_trace: Sequence[tuple[Any, dict[str, Any], Any]],
    ) -> None:
        if "feedback_score" not in official_result:
            raise TypeError("the official IFBench feedback result omitted feedback_score")
        reward = official_result["feedback_score"]
        if (
            isinstance(reward, bool)
            or not isinstance(reward, Real)
            or not math.isfinite(float(reward))
        ):
            raise TypeError("the official IFBench feedback_score must be finite numeric")

        _, iteration, subsample_ids, selected_name = self._current_state()
        if predictor_name != selected_name:
            raise RuntimeError("feedback wrapper differs from the officially scheduled predictor")
        _, signature, demos = self._selected_predictor_state(
            predictor_output,
            predictor_inputs,
            captured_trace,
        )
        skill = getattr(signature, "instructions", None)
        if not isinstance(skill, str):
            raise TypeError("the selected IFBench instruction must be text")
        messages = self._chat_adapter.format(
            signature,
            list(demos),
            dict(predictor_inputs),
        )
        expected = {
            field_name: predictor_output[field_name]
            for field_name in signature.output_fields
        }

        def matches_prediction(record: Any) -> bool:
            try:
                parsed = self._chat_adapter.parse(signature, record.native_output_text)
            except Exception:
                return False
            return parsed == expected

        token_record = self._token_callback.take_for_messages_matching(
            messages,
            matches_prediction,
        )

        output_fields = tuple(
            field_name for field_name in signature.output_fields if field_name != "reasoning"
        )
        if len(output_fields) != 1:
            raise RuntimeError("IFBench predictor must expose one non-reasoning output field")
        visible_reasoning = predictor_output["reasoning"]
        answer_text = predictor_output[output_fields[0]]
        if not isinstance(visible_reasoning, str) or not isinstance(answer_text, str):
            raise TypeError("official IFBench predictor outputs must be text")

        instance_index = self._instance_index(module_inputs, subsample_ids)
        record = CapturedRollout(
            iteration=iteration,
            instance_index=instance_index,
            reward=reward,
            skill_text=skill,
            signature=signature,
            demos=demos,
            reasoning_text=visible_reasoning,
            answer_text=answer_text,
            native_reasoning_text=token_record.native_reasoning_text,
            native_output_text=token_record.native_output_text,
            module_inputs=module_inputs,
            predictor_inputs=dict(predictor_inputs),
            token_record=token_record,
        )
        with self._lock:
            iteration_records = self._records.setdefault(iteration, {})
            if instance_index in iteration_records:
                raise RuntimeError("IFBench captured one selected instance more than once")
            iteration_records[instance_index] = record

    def current_minibatch(self) -> tuple[CapturedRollout, ...]:
        _, iteration, subsample_ids, _ = self._current_state()
        with self._lock:
            records = dict(self._records.get(iteration, {}))
            failures = tuple(self._capture_failures)
        if failures:
            raise RuntimeError(
                "IFBench capture instrumentation failed after official feedback completed"
            ) from failures[0]
        ordered = tuple(records[index] for index in subsample_ids if index in records)
        if not ordered:
            raise RuntimeError("the current IFBench iteration has no captured selected rollout")
        if len(ordered) != len(records):
            raise RuntimeError("captured IFBench rollout indices lie outside the current minibatch")
        return ordered

    def record_capture_failure(self, error: Exception) -> None:
        if not isinstance(error, Exception):
            raise TypeError("capture failure must be an Exception")
        with self._lock:
            self._capture_failures.append(error)

    def finish_parent_batch(self) -> None:
        self._token_callback.discard_pending_records()

    def discard_program_rollouts(
        self,
        *,
        program: dspy.Module,
        examples: Sequence[dspy.Example],
        predictions: Sequence[dspy.Prediction | None],
    ) -> None:
        del program
        if len(tuple(examples)) != len(tuple(predictions)):
            raise ValueError("candidate predictions must align with candidate examples")
        self._token_callback.discard_pending_records()


class IFBenchCapturingFeedback:
    """Delegate official IFBench feedback and capture only its scheduled predictor."""

    def __init__(
        self,
        *,
        predictor_name: str,
        delegate: Callable[..., dict[str, Any]],
        coordinator: IFBenchCaptureCoordinator,
    ) -> None:
        if not isinstance(predictor_name, str) or not predictor_name:
            raise ValueError("predictor_name must be non-empty text")
        if not callable(delegate):
            raise TypeError("delegate must be the official IFBench feedback function")
        if not isinstance(coordinator, IFBenchCaptureCoordinator):
            raise TypeError("coordinator must be IFBenchCaptureCoordinator")
        self._predictor_name = predictor_name
        self._delegate = delegate
        self._coordinator = coordinator

    def __call__(
        self,
        predictor_output: dspy.Prediction,
        predictor_inputs: dict[str, Any],
        module_inputs: dspy.Example,
        module_outputs: dspy.Prediction,
        captured_trace: list[tuple[Any, dict[str, Any], Any]],
    ) -> dict[str, Any]:
        result = self._delegate(
            predictor_output=predictor_output,
            predictor_inputs=predictor_inputs,
            module_inputs=module_inputs,
            module_outputs=module_outputs,
            captured_trace=captured_trace,
        )
        if not isinstance(result, dict):
            raise TypeError("the official IFBench feedback function must return a dictionary")
        try:
            self._coordinator.capture(
                predictor_name=self._predictor_name,
                official_result=result,
                predictor_output=predictor_output,
                predictor_inputs=predictor_inputs,
                module_inputs=module_inputs,
                captured_trace=captured_trace,
            )
        except Exception as error:
            # Instrumentation cannot rewrite the already-computed official feedback.
            # The optimizer observes this integrity failure after the parent batch.
            self._coordinator.record_capture_failure(error)
        return result

    def finish_parent_batch(self) -> None:
        self._coordinator.finish_parent_batch()

    def discard_program_rollouts(self, **kwargs: Any) -> None:
        self._coordinator.discard_program_rollouts(**kwargs)


def _chat_template(
    *,
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tokenize: bool,
) -> Any:
    return tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tokenize=tokenize,
        add_generation_prompt=True,
        continue_final_message=False,
        enable_thinking=True,
    )


def _plain_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise DependencyMeasureUnavailableError(
                f"official message {index} is not a mapping"
            )
        copied = dict(message)
        if not isinstance(copied.get("role"), str) or not isinstance(
            copied.get("content"), str
        ):
            raise DependencyMeasureUnavailableError(
                "IFBench dependency replay requires plain-text official messages"
            )
        result.append(copied)
    if not result:
        raise DependencyMeasureUnavailableError("official IFBench messages are empty")
    return tuple(result)


def _message_source_spans(
    *,
    chat_adapter: ChatAdapter,
    signature: Any,
    messages: tuple[dict[str, Any], ...],
    inputs: Mapping[str, str],
) -> tuple[tuple[int, CharacterSpan], tuple[tuple[str, int, CharacterSpan], ...]]:
    system_message = messages[0]
    if system_message["role"] != "system":
        raise DependencyMeasureUnavailableError(
            "official IFBench messages do not begin with the task description"
        )
    task_description = chat_adapter.format_task_description(signature)
    if not isinstance(task_description, str):
        raise DependencyMeasureUnavailableError(
            "official task description is not plain text"
        )
    system_content = system_message["content"]
    objective_start = task_description.find("\n")
    if not system_content.endswith(task_description) or objective_start < 0:
        raise DependencyMeasureUnavailableError(
            "official task description cannot expose an exact rendered skill block"
        )
    rendered_skill = task_description[objective_start:]
    leading_format = len(rendered_skill) - len(rendered_skill.lstrip())
    skill_start = (
        len(system_content)
        - len(task_description)
        + objective_start
        + leading_format
    )
    skill_span = CharacterSpan(skill_start, len(system_content))
    if skill_span.start == skill_span.end:
        raise DependencyMeasureUnavailableError(
            "official task description contains no rendered skill block"
        )

    user_index = len(messages) - 1
    user_message = messages[user_index]
    if user_message["role"] != "user":
        raise DependencyMeasureUnavailableError(
            "official IFBench messages do not end with the current user request"
        )
    user_content = user_message["content"]
    cursor = 0
    field_spans: list[tuple[str, int, CharacterSpan]] = []
    for field_name in signature.input_fields:
        if field_name not in inputs:
            raise DependencyMeasureUnavailableError(
                f"official input field {field_name!r} is absent from dependency replay"
            )
        value = inputs[field_name]
        formatted_value = format_field_value(
            field_info=signature.input_fields[field_name],
            value=value,
        )
        field_block = chat_adapter.format_user_message_content(
            signature,
            {field_name: value},
        )
        if not isinstance(formatted_value, str) or not isinstance(field_block, str):
            raise DependencyMeasureUnavailableError(
                f"official input field {field_name!r} is not rendered as text"
            )
        if not field_block.endswith(formatted_value):
            raise DependencyMeasureUnavailableError(
                f"official input field {field_name!r} has no exact rendered value suffix"
            )
        block_start = user_content.find(field_block, cursor)
        if block_start < 0:
            raise DependencyMeasureUnavailableError(
                f"official user message does not expose input field {field_name!r}"
            )
        value_start = block_start + len(field_block) - len(formatted_value)
        value_end = value_start + len(formatted_value)
        field_spans.append(
            (field_name, user_index, CharacterSpan(value_start, value_end))
        )
        cursor = block_start + len(field_block)
    return (0, skill_span), tuple(field_spans)


def _lift_message_span(
    *,
    tokenizer: Any,
    messages: tuple[dict[str, Any], ...],
    actual_text: str,
    message_index: int,
    span: CharacterSpan,
    label: str,
) -> CharacterSpan:
    content = messages[message_index]["content"]
    if span.start < 0 or span.end < span.start or span.end > len(content):
        raise DependencyMeasureUnavailableError(
            f"message-local {label} span lies outside the official message"
        )
    surface = content[span.start : span.end]
    marker = f"__GEPA_POST_CHAT_BOUNDARY_{label}_{message_index}_{span.start}_{span.end}__"
    while marker in actual_text or any(marker in message["content"] for message in messages):
        marker += "_"

    marked_messages = [dict(message) for message in messages]
    marked_messages[message_index]["content"] = (
        content[: span.start] + marker + content[span.end :]
    )
    marked_text = _chat_template(
        tokenizer=tokenizer,
        messages=marked_messages,
        tokenize=False,
    )
    if not isinstance(marked_text, str):
        raise DependencyMeasureUnavailableError(
            "Qwen chat template did not return text for a provenance probe"
        )
    marker_start = marked_text.find(marker)
    if marker_start < 0 or marked_text.find(marker, marker_start + len(marker)) >= 0:
        raise DependencyMeasureUnavailableError(
            f"Qwen chat template did not preserve one post-render {label} boundary"
        )
    prefix = marked_text[:marker_start]
    suffix = marked_text[marker_start + len(marker) :]
    if actual_text != prefix + surface + suffix:
        raise DependencyMeasureUnavailableError(
            f"Qwen chat template cannot lift the exact post-render {label} span"
        )
    return CharacterSpan(len(prefix), len(prefix) + len(surface))


def _render_ifbench_prompt(
    *,
    tokenizer: Any,
    chat_adapter: ChatAdapter,
    signature: Any,
    demos: tuple[Any, ...],
    inputs: Mapping[str, Any],
    actual_messages: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[RenderedPromptProvenance, tuple[dict[str, Any], ...], tuple[int, ...]]:
    if not isinstance(chat_adapter, ChatAdapter):
        raise TypeError("chat_adapter must be the shared official ChatAdapter")
    instruction = getattr(signature, "instructions", None)
    if not isinstance(instruction, str):
        raise TypeError("IFBench instruction must be text")
    input_names = tuple(signature.input_fields)
    if set(inputs) != set(input_names):
        raise RuntimeError("IFBench predictor inputs differ from the official signature")
    values: dict[str, str] = {}
    for field_name in input_names:
        value = inputs[field_name]
        if not isinstance(value, str):
            raise TypeError("IFBench task/observation fields must be text")
        values[field_name] = value

    if actual_messages is None:
        rendered_messages = chat_adapter.format(signature, list(demos), dict(values))
    else:
        rendered_messages = actual_messages
    messages = _plain_messages(rendered_messages)
    skill_source, field_sources = _message_source_spans(
        chat_adapter=chat_adapter,
        signature=signature,
        messages=messages,
        inputs=values,
    )

    actual_text = _chat_template(
        tokenizer=tokenizer,
        messages=messages,
        tokenize=False,
    )
    if not isinstance(actual_text, str):
        raise DependencyMeasureUnavailableError(
            "Qwen chat template must return text when tokenize=False"
        )
    skill_span = _lift_message_span(
        tokenizer=tokenizer,
        messages=messages,
        actual_text=actual_text,
        message_index=skill_source[0],
        span=skill_source[1],
        label="skill",
    )
    fields = tuple(
        TaskObservationField(
            field_name,
            _lift_message_span(
                tokenizer=tokenizer,
                messages=messages,
                actual_text=actual_text,
                message_index=message_index,
                span=span,
                label=f"input_{index}_{field_name}",
            ),
        )
        for index, (field_name, message_index, span) in enumerate(field_sources)
    )
    try:
        encoded_ids, _ = _encode_with_offsets(
            tokenizer,
            actual_text,
            "IFBench rendered prompt",
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise DependencyMeasureUnavailableError(
            "local tokenizer cannot represent the official rendered prompt exactly"
        ) from error
    template_ids = _integer_ids(
        _chat_template(tokenizer=tokenizer, messages=messages, tokenize=True),
        "IFBench chat-template token IDs",
    )
    if template_ids != encoded_ids:
        raise DependencyMeasureUnavailableError(
            "IFBench chat-template IDs differ from exact rendered text"
        )
    return (
        RenderedPromptProvenance(
            text=actual_text,
            token_ids=encoded_ids,
            task_observation_fields=fields,
            skill_spans=(skill_span,),
        ),
        messages,
        encoded_ids,
    )


def build_ifbench_replay_dependency_inputs(
    *,
    tokenizer: Any,
    chat_adapter: ChatAdapter,
    rollout: CapturedRollout,
    parent_skill: str,
    candidate_skill: str,
) -> AIMEReplayDependencyInputs:
    """Render one selected IFBench call into the existing b06 replay contract."""

    if not isinstance(rollout, CapturedRollout):
        raise TypeError("rollout must be CapturedRollout")
    if rollout.skill_text != parent_skill or rollout.signature.instructions != parent_skill:
        raise RuntimeError("captured IFBench rollout does not belong to the parent instruction")
    old_prompt, _, old_ids = _render_ifbench_prompt(
        tokenizer=tokenizer,
        chat_adapter=chat_adapter,
        signature=rollout.signature,
        demos=rollout.demos,
        inputs=rollout.predictor_inputs,
        actual_messages=rollout.token_record.prompt_messages,
    )
    if old_ids != tuple(rollout.token_record.prompt_token_ids):
        raise RuntimeError("official IFBench prompt IDs differ from captured canonical IDs")

    candidate_prompt, _, _ = _render_ifbench_prompt(
        tokenizer=tokenizer,
        chat_adapter=chat_adapter,
        signature=rollout.signature.with_instructions(candidate_skill),
        demos=rollout.demos,
        inputs=rollout.predictor_inputs,
    )
    return AIMEReplayDependencyInputs(
        instance_id=rollout.instance_index,
        old_prompt=old_prompt,
        candidate_prompt=candidate_prompt,
        target=_historical_target(tokenizer=tokenizer, rollout=rollout),
    )
