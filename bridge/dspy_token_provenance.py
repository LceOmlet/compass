from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from typing import Any, Literal, TypeAlias

from dspy.utils.callback import BaseCallback
from dspy.utils.callback_context import ACTIVE_CALL_ID

from .qwen_native_tokens import QwenNativeTokenLayout, canonicalize_qwen_choice


CallbackKind: TypeAlias = Literal["module", "format", "lm", "parse"]
SourceKind: TypeAlias = Literal["task_observation", "skill", "format"]


class LineageUnavailableError(RuntimeError):
    """The official TraceData entry has no unique completed callback lineage."""


class TokenProvenanceUnavailableError(RuntimeError):
    """Actual DSPy messages or output cannot yield exact token provenance."""


@dataclass(frozen=True, slots=True)
class DSPyCallbackCall:
    """One completed public DSPy callback call."""

    kind: CallbackKind
    sequence: int
    call_id: str
    parent_call_id: str | None
    instance: Any
    inputs: Mapping[str, Any]
    output: Any
    exception: BaseException | None


@dataclass(frozen=True, slots=True)
class DSPyPredictLineage:
    """The actual successful format/LM/parse branch of one predictor call."""

    predict_call: DSPyCallbackCall
    format_call: DSPyCallbackCall
    lm_call: DSPyCallbackCall
    parse_call: DSPyCallbackCall


@dataclass(frozen=True, slots=True)
class SourceCoordinate:
    """Source coordinate; every token crossing a source boundary is FORMAT."""

    kind: SourceKind
    field_name: str | None = None
    source_token_index: int | None = None


@dataclass(frozen=True, slots=True)
class TokenProvenanceTrajectory:
    """Actual plain-text DSPy exchange with deterministic local Qwen token IDs."""

    canonical_prompt_text: str
    canonical_completion_text: str
    canonical_prompt_token_ids: tuple[int, ...]
    canonical_completion_token_ids: tuple[int, ...]
    canonical_prompt_coordinates: tuple[SourceCoordinate, ...]
    canonical_skill_character_span: tuple[int, int]
    canonical_task_observation_character_spans: tuple[tuple[str, tuple[int, int]], ...]
    messages: tuple[Mapping[str, Any], ...]
    raw_response: Any
    native_reasoning_text: str
    native_output_text: str
    canonical_token_layout: QwenNativeTokenLayout


@dataclass(frozen=True, slots=True)
class _PendingCall:
    kind: CallbackKind
    sequence: int
    parent_call_id: str | None
    instance: Any
    inputs: Mapping[str, Any]


class DSPyCallbackCollector(BaseCallback):
    """Record the official DSPy callback tree without changing execution."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._next_sequence = 0
        self._pending: dict[str, _PendingCall] = {}
        self._completed: list[DSPyCallbackCall] = []

    def _start(
        self,
        *,
        kind: CallbackKind,
        call_id: str,
        instance: Any,
        inputs: Mapping[str, Any],
    ) -> None:
        parent_call_id = ACTIVE_CALL_ID.get()
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
            self._pending[call_id] = _PendingCall(
                kind=kind,
                sequence=sequence,
                parent_call_id=parent_call_id,
                instance=instance,
                inputs=dict(inputs),
            )

    def _end(
        self,
        *,
        kind: CallbackKind,
        call_id: str,
        output: Any,
        exception: BaseException | None,
    ) -> None:
        with self._lock:
            pending = self._pending.pop(call_id, None)
            if pending is None or pending.kind != kind:
                return
            self._completed.append(
                DSPyCallbackCall(
                    kind=kind,
                    sequence=pending.sequence,
                    call_id=call_id,
                    parent_call_id=pending.parent_call_id,
                    instance=pending.instance,
                    inputs=pending.inputs,
                    output=output,
                    exception=exception,
                )
            )

    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._start(kind="module", call_id=call_id, instance=instance, inputs=inputs)

    def on_module_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        self._end(kind="module", call_id=call_id, output=outputs, exception=exception)

    def on_adapter_format_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        self._start(kind="format", call_id=call_id, instance=instance, inputs=inputs)

    def on_adapter_format_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        self._end(kind="format", call_id=call_id, output=outputs, exception=exception)

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._start(kind="lm", call_id=call_id, instance=instance, inputs=inputs)

    def on_lm_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        self._end(kind="lm", call_id=call_id, output=outputs, exception=exception)

    def on_adapter_parse_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        self._start(kind="parse", call_id=call_id, instance=instance, inputs=inputs)

    def on_adapter_parse_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        self._end(kind="parse", call_id=call_id, output=outputs, exception=exception)

    def snapshot(self) -> tuple[DSPyCallbackCall, ...]:
        with self._lock:
            if self._pending:
                raise LineageUnavailableError("DSPy callback calls are still active")
            return tuple(sorted(self._completed, key=lambda call: call.sequence))

    def completed_snapshot(self) -> tuple[DSPyCallbackCall, ...]:
        """Return completed calls while official speculative replicas may still run."""

        with self._lock:
            return tuple(sorted(self._completed, key=lambda call: call.sequence))

    def clear(self) -> None:
        with self._lock:
            if self._pending:
                raise LineageUnavailableError("cannot clear active DSPy callback calls")
            self._completed.clear()


def _trace_entry(
    trace_data: Mapping[str, Any],
    trace_index: int,
) -> tuple[Any, Mapping[str, Any], Any]:
    trace = trace_data.get("trace")
    if not isinstance(trace, Sequence) or isinstance(trace, str | bytes):
        raise LineageUnavailableError("official TraceData.trace is not an ordered sequence")
    if isinstance(trace_index, bool) or not isinstance(trace_index, Integral):
        raise TypeError("trace_index must be an integer")
    index = int(trace_index)
    if index < 0 or index >= len(trace):
        raise IndexError("trace_index lies outside official TraceData.trace")
    entry = trace[index]
    if not isinstance(entry, tuple) or len(entry) != 3:
        raise LineageUnavailableError("official TraceData trace entry is malformed")
    predictor, inputs, prediction = entry
    if not isinstance(inputs, Mapping):
        raise LineageUnavailableError("official predictor inputs are not a mapping")
    return predictor, inputs, prediction


def _prediction_values(prediction: Any) -> dict[str, Any]:
    items = getattr(prediction, "items", None)
    if not callable(items):
        raise LineageUnavailableError("official predictor Prediction has no mapping items")
    try:
        return dict(items())
    except (TypeError, ValueError) as error:
        raise LineageUnavailableError("official predictor Prediction is not a text-field mapping") from error


def _module_return_contains(call: DSPyCallbackCall, prediction: Any) -> bool:
    if call.output is prediction:
        return True
    return (
        isinstance(call.output, tuple)
        and len(call.output) == 2
        and call.output[0] is prediction
        and isinstance(call.output[1], list)
    )


class ExactTraceDataLineageResolver:
    """Resolve TraceData by callback IDs plus predictor/Prediction identity."""

    def __call__(
        self,
        *,
        trace_data: Mapping[str, Any],
        trace_index: int,
        callback_calls: Sequence[DSPyCallbackCall],
    ) -> DSPyPredictLineage:
        predictor, _, prediction = _trace_entry(trace_data, trace_index)
        calls = tuple(callback_calls)
        if any(not isinstance(call, DSPyCallbackCall) for call in calls):
            raise TypeError("callback_calls must contain DSPyCallbackCall values")
        call_ids = [call.call_id for call in calls]
        if len(call_ids) != len(set(call_ids)):
            raise LineageUnavailableError("DSPy callback call IDs are not unique")

        predict_matches = [
            call
            for call in calls
            if call.kind == "module"
            and call.instance is predictor
            and _module_return_contains(call, prediction)
            and call.exception is None
        ]
        if len(predict_matches) != 1:
            raise LineageUnavailableError(
                "TraceData predictor/Prediction identity does not select one module callback"
            )
        predict_call = predict_matches[0]
        children = tuple(
            sorted(
                (call for call in calls if call.parent_call_id == predict_call.call_id),
                key=lambda call: call.sequence,
            )
        )
        expected_prediction = _prediction_values(prediction)
        matching_parses = [
            call
            for call in children
            if call.kind == "parse"
            and call.exception is None
            and isinstance(call.output, Mapping)
            and dict(call.output) == expected_prediction
        ]
        if len(matching_parses) != 1:
            raise LineageUnavailableError(
                "predictor Prediction does not select one successful parse callback"
            )
        parse_call = matching_parses[0]

        lm_candidates = [
            call
            for call in children
            if call.kind == "lm"
            and call.sequence < parse_call.sequence
            and call.exception is None
            and call.output is not None
        ]
        if not lm_candidates:
            raise LineageUnavailableError("successful parse has no preceding LM callback")
        lm_call = max(lm_candidates, key=lambda call: call.sequence)

        format_candidates = [
            call
            for call in children
            if call.kind == "format"
            and call.sequence < lm_call.sequence
            and call.exception is None
            and call.output is not None
        ]
        if not format_candidates:
            raise LineageUnavailableError("selected LM callback has no preceding format callback")
        format_call = max(format_candidates, key=lambda call: call.sequence)
        return DSPyPredictLineage(
            predict_call=predict_call,
            format_call=format_call,
            lm_call=lm_call,
            parse_call=parse_call,
        )


def _integer_ids(values: Any, name: str) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError as error:
        raise TokenProvenanceUnavailableError(f"{name} is not an ID sequence") from error
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in result):
        raise TokenProvenanceUnavailableError(f"{name} must contain integer token IDs")
    return tuple(int(value) for value in result)


def _encode_with_offsets(
    tokenizer: Any,
    text: str,
    name: str,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if not isinstance(encoded, Mapping):
        raise TokenProvenanceUnavailableError(f"{name} tokenizer result is not a mapping")
    ids = _integer_ids(encoded.get("input_ids"), f"{name} input_ids")
    try:
        offsets = tuple((int(start), int(end)) for start, end in encoded["offset_mapping"])
    except (KeyError, TypeError, ValueError) as error:
        raise TokenProvenanceUnavailableError(f"{name} has no direct token offsets") from error
    if len(ids) != len(offsets):
        raise TokenProvenanceUnavailableError(f"{name} IDs and offsets have different lengths")
    if any(start < 0 or end <= start or end > len(text) for start, end in offsets):
        raise TokenProvenanceUnavailableError(f"{name} contains a non-direct token offset")
    decoded = tokenizer.decode(
        list(ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != text:
        raise TokenProvenanceUnavailableError(f"{name} failed exact local token round-trip")
    return ids, offsets


def _plain_messages(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TokenProvenanceUnavailableError("LM callback messages are not an ordered sequence")
    messages: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise TokenProvenanceUnavailableError(f"LM message {index} is not a mapping")
        message = dict(raw)
        if not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            raise TokenProvenanceUnavailableError(
                "current token provenance requires actual plain-text role/content messages"
            )
        if not message["content"]:
            raise TokenProvenanceUnavailableError("actual DSPy message content is empty")
        messages.append(message)
    if not messages:
        raise TokenProvenanceUnavailableError("actual DSPy message list is empty")
    return tuple(messages)


def _all_occurrences(text: str, needle: str) -> tuple[int, ...]:
    if not needle:
        return ()
    result: list[int] = []
    start = 0
    while True:
        found = text.find(needle, start)
        if found < 0:
            return tuple(result)
        result.append(found)
        start = found + 1


def _unique_ordered_spans(text: str, needles: Sequence[str]) -> tuple[tuple[int, int], ...]:
    candidates = tuple(_all_occurrences(text, needle) for needle in needles)
    if any(not positions for positions in candidates):
        raise TokenProvenanceUnavailableError("chat template did not preserve actual message content")

    @lru_cache(maxsize=None)
    def solve(index: int, cursor: int) -> tuple[int, tuple[tuple[int, int], ...]]:
        if index == len(needles):
            return 1, ()
        count = 0
        selected: tuple[tuple[int, int], ...] = ()
        needle = needles[index]
        for start in candidates[index]:
            if start < cursor:
                continue
            child_count, child_path = solve(index + 1, start + len(needle))
            if child_count:
                if count == 0:
                    selected = ((start, start + len(needle)),) + child_path
                count = min(2, count + child_count)
            if count == 2:
                break
        return count, selected

    count, spans = solve(0, 0)
    if count != 1:
        raise TokenProvenanceUnavailableError(
            "actual messages do not have one ordered embedding in the rendered prompt"
        )
    return spans


def _unique_span(text: str, value: str, name: str) -> tuple[int, int]:
    positions = _all_occurrences(text, value)
    if len(positions) != 1:
        raise TokenProvenanceUnavailableError(f"{name} is not a unique exact substring")
    return positions[0], positions[0] + len(value)


def official_rendered_instruction_span(
    *,
    adapter: Any,
    signature: Any,
    messages: Sequence[Mapping[str, Any]],
) -> tuple[int, tuple[int, int]]:
    """Locate the official candidate-dependent task-description suffix.

    DSPy's ChatAdapter does not preserve a multiline ``signature.instructions``
    value as one literal substring: its official task-description renderer
    dedents and indents the instruction.  Use that renderer for both the exact
    signature and the same signature with an empty instruction.  Their checked
    prefix relation identifies the one contiguous rendered intervention span
    without re-running ``Adapter.format`` or inventing a per-line source map.

    Returns the index of the unique system message and a half-open character
    span local to that message.
    """

    plain_messages = _plain_messages(messages)
    instruction = getattr(signature, "instructions", None)
    if not isinstance(instruction, str) or not instruction:
        raise TokenProvenanceUnavailableError(
            "format callback has no non-empty textual instruction"
        )
    format_task_description = getattr(adapter, "format_task_description", None)
    with_instructions = getattr(signature, "with_instructions", None)
    if not callable(format_task_description) or not callable(with_instructions):
        raise TokenProvenanceUnavailableError(
            "official adapter/signature cannot render an instruction baseline"
        )

    rendered = format_task_description(signature)
    # DSPy treats ``with_instructions("")`` as a request for its generated
    # default instruction.  Create a fresh official Signature first, then use
    # its public instructions setter to obtain the actual empty baseline
    # without mutating the trajectory's signature.
    empty_signature = with_instructions(instruction)
    if empty_signature is signature:
        raise TokenProvenanceUnavailableError(
            "official with_instructions did not create a fresh signature"
        )
    try:
        empty_signature.instructions = ""
    except Exception as error:
        raise TokenProvenanceUnavailableError(
            "official signature setter rejected the empty-instruction baseline"
        ) from error
    if getattr(empty_signature, "instructions", None) != "":
        raise TokenProvenanceUnavailableError(
            "official signature setter did not preserve the empty-instruction baseline"
        )
    empty_rendered = format_task_description(empty_signature)
    if not isinstance(rendered, str) or not rendered:
        raise TokenProvenanceUnavailableError(
            "official task-description renderer returned no text"
        )
    if not isinstance(empty_rendered, str) or not rendered.startswith(empty_rendered):
        raise TokenProvenanceUnavailableError(
            "official full task description does not extend its empty-instruction baseline"
        )
    if len(rendered) == len(empty_rendered):
        raise TokenProvenanceUnavailableError(
            "official task-description renderer exposed no instruction-dependent suffix"
        )

    system_indices = tuple(
        index for index, message in enumerate(plain_messages) if message["role"] == "system"
    )
    if len(system_indices) != 1:
        raise TokenProvenanceUnavailableError(
            "official messages do not contain exactly one system message"
        )
    system_index = system_indices[0]
    task_start, task_end = _unique_span(
        plain_messages[system_index]["content"],
        rendered,
        "official rendered task description",
    )
    return system_index, (task_start + len(empty_rendered), task_end)


def _actual_lm_history(call: DSPyCallbackCall) -> tuple[tuple[dict[str, Any], ...], Any]:
    history = getattr(call.instance, "history", None)
    if not isinstance(history, list):
        raise TokenProvenanceUnavailableError("LM instance has no official history list")
    matches = [
        entry
        for entry in history
        if isinstance(entry, Mapping) and entry.get("outputs") is call.output
    ]
    if len(matches) != 1:
        raise TokenProvenanceUnavailableError(
            "LM callback output does not identify one official history response"
        )
    entry = matches[0]
    if "response" not in entry:
        raise TokenProvenanceUnavailableError("official LM history omitted the raw response")
    return _plain_messages(entry.get("messages")), entry["response"]


def _non_overlapping(
    spans: Sequence[tuple[str, tuple[int, int]]],
    name: str,
) -> None:
    ordered = sorted(spans, key=lambda item: item[1])
    for (_, left), (_, right) in zip(ordered, ordered[1:], strict=False):
        if left[1] > right[0]:
            raise TokenProvenanceUnavailableError(f"{name} source spans overlap")


class ActualDSPyTokenProvenance:
    """Tokenize one resolved DSPy exchange without re-running Adapter.format or its LM."""

    def __init__(self, *, tokenizer: Any, chat_template_kwargs: Mapping[str, Any]) -> None:
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise TypeError("tokenizer must expose apply_chat_template")
        if "tokenize" in chat_template_kwargs:
            raise ValueError("chat_template_kwargs must not contain tokenize")
        self._tokenizer = tokenizer
        self._chat_template_kwargs = dict(chat_template_kwargs)

    def __call__(
        self,
        *,
        trace_data: Mapping[str, Any],
        trace_index: int,
        lineage: DSPyPredictLineage,
    ) -> TokenProvenanceTrajectory:
        predictor, _, prediction = _trace_entry(trace_data, trace_index)
        if lineage.predict_call.instance is not predictor or not _module_return_contains(
            lineage.predict_call, prediction
        ):
            raise LineageUnavailableError("lineage belongs to a different TraceData entry")
        if (
            lineage.format_call.parent_call_id != lineage.predict_call.call_id
            or lineage.lm_call.parent_call_id != lineage.predict_call.call_id
            or lineage.parse_call.parent_call_id != lineage.predict_call.call_id
        ):
            raise LineageUnavailableError("format/LM/parse calls are not children of the predictor call")

        messages, raw_response = _actual_lm_history(lineage.lm_call)
        format_messages = _plain_messages(lineage.format_call.output)
        callback_messages = _plain_messages(lineage.lm_call.inputs.get("messages"))
        if messages != format_messages or messages != callback_messages:
            raise TokenProvenanceUnavailableError(
                "format output, LM callback input, and LM history messages differ"
            )
        template_messages = [dict(message) for message in messages]
        prompt_text = self._tokenizer.apply_chat_template(
            template_messages,
            tokenize=False,
            **self._chat_template_kwargs,
        )
        if not isinstance(prompt_text, str) or not prompt_text:
            raise TokenProvenanceUnavailableError("chat template did not return prompt text")
        prompt_ids, prompt_offsets = _encode_with_offsets(
            self._tokenizer,
            prompt_text,
            "canonical prompt rendered from actual DSPy messages",
        )
        message_spans = _unique_ordered_spans(
            prompt_text,
            tuple(message["content"] for message in messages),
        )

        format_inputs = lineage.format_call.inputs
        signature = format_inputs.get("signature")
        system_index, local_skill_span = official_rendered_instruction_span(
            adapter=lineage.format_call.instance,
            signature=signature,
            messages=messages,
        )
        skill_span = (
            message_spans[system_index][0] + local_skill_span[0],
            message_spans[system_index][0] + local_skill_span[1],
        )

        predictor_inputs = format_inputs.get("inputs")
        input_fields = getattr(signature, "input_fields", None)
        if not isinstance(predictor_inputs, Mapping) or not isinstance(input_fields, Mapping):
            raise TokenProvenanceUnavailableError("format callback has no official input-field mapping")
        last_message = messages[-1]
        if last_message["role"] != "user":
            raise TokenProvenanceUnavailableError("actual DSPy prompt does not end in the current user input")
        last_message_start = message_spans[-1][0]
        field_names = tuple(input_fields)
        field_values: list[str] = []
        for field_name in field_names:
            value = predictor_inputs.get(field_name)
            if not isinstance(value, str) or not value:
                raise TokenProvenanceUnavailableError(
                    f"input field {field_name!r} is not non-empty text"
                )
            field_values.append(value)
        local_field_spans = _unique_ordered_spans(
            last_message["content"],
            tuple(field_values),
        )
        task_spans = [
            (
                field_name,
                (last_message_start + local_start, last_message_start + local_end),
            )
            for field_name, (local_start, local_end) in zip(
                field_names,
                local_field_spans,
                strict=True,
            )
        ]
        explicit_prompt_spans = [("skill", skill_span), *task_spans]
        _non_overlapping(explicit_prompt_spans, "prompt")

        prompt_coordinates: list[SourceCoordinate] = []
        source_indices = {field_name: 0 for field_name, _ in task_spans}
        for start, end in prompt_offsets:
            task_matches = [
                field_name
                for field_name, (span_start, span_end) in task_spans
                if span_start <= start and end <= span_end
            ]
            skill_match = skill_span[0] <= start and end <= skill_span[1]
            if len(task_matches) + int(skill_match) > 1:
                raise TokenProvenanceUnavailableError("one prompt token crosses distinct source spans")
            if task_matches:
                field_name = task_matches[0]
                token_index = source_indices[field_name]
                source_indices[field_name] += 1
                prompt_coordinates.append(
                    SourceCoordinate("task_observation", field_name, token_index)
                )
            elif skill_match:
                prompt_coordinates.append(SourceCoordinate("skill"))
            else:
                prompt_coordinates.append(SourceCoordinate("format"))

        parsed_completion = lineage.parse_call.inputs.get("completion")
        if not isinstance(parsed_completion, str) or not parsed_completion:
            raise TokenProvenanceUnavailableError("parse callback has no completion text")
        parsed_output = lineage.parse_call.output
        if not isinstance(parsed_output, Mapping):
            raise TokenProvenanceUnavailableError("parse callback output is not a mapping")
        if dict(parsed_output) != _prediction_values(prediction):
            raise LineageUnavailableError("parse output differs from the TraceData Prediction")
        choices = getattr(raw_response, "choices", None)
        if not isinstance(choices, Sequence) or len(choices) != 1:
            raise TokenProvenanceUnavailableError(
                "current trajectory bridge requires exactly one raw LM choice"
            )
        choice = choices[0]
        message = getattr(choice, "message", None)
        native_reasoning = getattr(message, "reasoning_content", None)
        native_output = getattr(message, "content", None)
        if not isinstance(native_reasoning, str) or not native_reasoning:
            raise TokenProvenanceUnavailableError(
                "raw Qwen response has no native reasoning_content"
            )
        if not isinstance(native_output, str) or not native_output:
            raise TokenProvenanceUnavailableError("raw Qwen response has no native visible content")
        if native_output != parsed_completion:
            raise TokenProvenanceUnavailableError(
                "native visible output differs from the text passed to Adapter.parse"
            )
        decoded = canonicalize_qwen_choice(tokenizer=self._tokenizer, choice=choice)
        if decoded.finish_reason != "stop":
            raise TokenProvenanceUnavailableError(
                f"Qwen completion is not terminal: {decoded.finish_reason!r}"
            )

        return TokenProvenanceTrajectory(
            canonical_prompt_text=prompt_text,
            canonical_completion_text=decoded.raw_completion_text,
            canonical_prompt_token_ids=prompt_ids,
            canonical_completion_token_ids=decoded.completion_token_ids,
            canonical_prompt_coordinates=tuple(prompt_coordinates),
            canonical_skill_character_span=skill_span,
            canonical_task_observation_character_spans=tuple(task_spans),
            messages=messages,
            raw_response=raw_response,
            native_reasoning_text=native_reasoning,
            native_output_text=native_output,
            canonical_token_layout=decoded.token_layout,
        )
