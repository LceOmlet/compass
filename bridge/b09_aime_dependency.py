from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from dspy.adapters.chat_adapter import ChatAdapter

from .b01_aime_capture import CapturedRollout
from .b02_flashtrace_credit import (
    FlashTraceCredit,
    _encode_with_offsets,
    _integer_ids,
)
from .b06_dependency_gate import (
    CharacterSpan,
    DependencyMeasure,
    DependencyMeasureUnavailableError,
    HistoricalRolloutTarget,
    RenderedPromptProvenance,
    TaskObservationField,
    batch_dependency_distance,
    build_dependency_measure,
)
from .b12_qwen_native_aime import QwenNativeAIMEAdapter
from .b14_flashtrace_token_ids import (
    ExactTokenOffloadedLLMIFRAttribution,
    shared_exact_token_capture,
)
from .b11_terminal_likelihood import AIMEHistoricalReplayScorer


_SKILL_MARKER = "__GEPA_AIME_SKILL_SOURCE_2F674E50C186__"
_PROBLEM_MARKER = "__GEPA_AIME_PROBLEM_SOURCE_03A6183BF285__"


@dataclass(frozen=True, slots=True)
class AIMEReplayDependencyInputs:
    """One captured rollout rendered under its old and candidate skills."""

    instance_id: int
    old_prompt: RenderedPromptProvenance
    candidate_prompt: RenderedPromptProvenance
    target: HistoricalRolloutTarget


def _messages(
    *,
    adapter: QwenNativeAIMEAdapter,
    signature: Any,
    problem: str,
) -> tuple[dict[str, Any], ...]:
    formatted = adapter.format(signature, [], {"problem": problem})
    if not isinstance(formatted, list):
        raise TypeError("official ChatAdapter.format must return a message list")
    messages = tuple(formatted)
    if len(messages) != 1:
        raise RuntimeError("Qwen-native AIME must render exactly one user message")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError("official ChatAdapter messages must be dictionaries")
        if set(message) != {"role", "content"}:
            raise RuntimeError("AIME v1 messages must contain only role and content")
        if message["role"] != "user":
            raise RuntimeError(f"unexpected Qwen-native AIME message role at index {index}")
        if not isinstance(message["content"], str):
            raise TypeError("AIME v1 requires plain-text ChatAdapter message content")
    return messages


def _marked_messages_and_sources(
    *,
    adapter: QwenNativeAIMEAdapter,
    signature: Any,
    problem: str,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    str,
    str,
]:
    if _SKILL_MARKER in signature.instructions or _PROBLEM_MARKER in signature.instructions:
        raise ValueError("AIME skill collides with a fixed provenance marker")
    if _SKILL_MARKER in problem or _PROBLEM_MARKER in problem:
        raise ValueError("AIME problem collides with a fixed provenance marker")

    actual = _messages(adapter=adapter, signature=signature, problem=problem)
    marked_signature = signature.with_instructions(_SKILL_MARKER)
    marked = _messages(
        adapter=adapter,
        signature=marked_signature,
        problem=_PROBLEM_MARKER,
    )
    if marked[0]["role"] != actual[0]["role"]:
        raise RuntimeError("source rendering changed the Qwen-native message role")
    emitted_skill = signature.instructions
    emitted_problem = problem
    if not isinstance(emitted_skill, str) or not emitted_skill:
        raise ValueError("AIME skill must be non-empty text")
    _rendered_source_spans(
        actual_text=actual[0]["content"],
        marked_text=marked[0]["content"],
        emitted_skill=emitted_skill,
        emitted_problem=emitted_problem,
    )
    return actual, marked, emitted_skill, emitted_problem


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


def _rendered_source_spans(
    *,
    actual_text: str,
    marked_text: str,
    emitted_skill: str,
    emitted_problem: str,
) -> tuple[CharacterSpan, CharacterSpan]:
    replacements = {
        _SKILL_MARKER: emitted_skill,
        _PROBLEM_MARKER: emitted_problem,
    }
    events: list[tuple[int, str]] = []
    for marker in replacements:
        start = marked_text.find(marker)
        if start < 0 or marked_text.find(marker, start + len(marker)) >= 0:
            raise RuntimeError("chat template did not preserve one unique provenance marker")
        events.append((start, marker))
    events.sort()

    marked_cursor = 0
    actual_cursor = 0
    spans: dict[str, CharacterSpan] = {}
    for marker_start, marker in events:
        unchanged = marked_text[marked_cursor:marker_start]
        if actual_text[actual_cursor : actual_cursor + len(unchanged)] != unchanged:
            raise RuntimeError("chat template changed text outside explicit source markers")
        actual_cursor += len(unchanged)
        replacement = replacements[marker]
        source_start = actual_cursor
        if actual_text[source_start : source_start + len(replacement)] != replacement:
            raise RuntimeError("chat template marker replacement does not match the actual source")
        actual_cursor += len(replacement)
        spans[marker] = CharacterSpan(source_start, actual_cursor)
        marked_cursor = marker_start + len(marker)

    tail = marked_text[marked_cursor:]
    if actual_text[actual_cursor:] != tail:
        raise RuntimeError("chat template suffix differs after source reconstruction")
    return spans[_SKILL_MARKER], spans[_PROBLEM_MARKER]


def _render_prompt_provenance(
    *,
    tokenizer: Any,
    adapter: QwenNativeAIMEAdapter,
    signature: Any,
    problem: str,
) -> tuple[RenderedPromptProvenance, tuple[dict[str, Any], ...], tuple[int, ...]]:
    actual, marked, emitted_skill, emitted_problem = _marked_messages_and_sources(
        adapter=adapter,
        signature=signature,
        problem=problem,
    )
    actual_text = _chat_template(tokenizer=tokenizer, messages=actual, tokenize=False)
    marked_text = _chat_template(tokenizer=tokenizer, messages=marked, tokenize=False)
    if not isinstance(actual_text, str) or not isinstance(marked_text, str):
        raise TypeError("tokenizer chat template must return text when tokenize=False")
    skill_span, problem_span = _rendered_source_spans(
        actual_text=actual_text,
        marked_text=marked_text,
        emitted_skill=emitted_skill,
        emitted_problem=emitted_problem,
    )

    encoded_ids, _ = _encode_with_offsets(tokenizer, actual_text, "AIME rendered prompt")
    template_ids = _integer_ids(
        _chat_template(tokenizer=tokenizer, messages=actual, tokenize=True),
        "AIME chat-template token IDs",
    )
    if template_ids != encoded_ids:
        raise RuntimeError("tokenized chat template differs from its exact rendered text")
    return (
        RenderedPromptProvenance(
            text=actual_text,
            token_ids=encoded_ids,
            task_observation_fields=(TaskObservationField("problem", problem_span),),
            skill_spans=(skill_span,),
        ),
        actual,
        encoded_ids,
    )


def _historical_target(
    *,
    tokenizer: Any,
    rollout: CapturedRollout,
) -> HistoricalRolloutTarget:
    completion = rollout.token_record.completion_text
    if not isinstance(completion, str) or not completion:
        raise ValueError("captured raw completion must be non-empty text")
    layout = rollout.token_record.token_layout
    captured_ids = _integer_ids(layout.full_token_ids, "captured completion token IDs")
    if captured_ids != tuple(rollout.token_record.completion_token_ids):
        raise RuntimeError("captured completion IDs differ from the shared Qwen token layout")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, bool) or not isinstance(eos_id, Integral):
        raise TypeError("the shared tokenizer must expose one integer eos_token_id")
    eos_id = int(eos_id)
    if captured_ids[-1] != eos_id or captured_ids.count(eos_id) != 1:
        raise ValueError("captured completion must end in exactly one tokenizer EOS token")
    completion_ids = layout.nonterminal_token_ids
    decoded_completion = tokenizer.decode(
        list(completion_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded_completion != completion:
        raise RuntimeError("captured completion text differs from its exact Arbor token IDs")

    native_reasoning = getattr(rollout, "native_reasoning_text", None)
    native_output = getattr(rollout, "native_output_text", None)
    if not isinstance(native_reasoning, str):
        raise TypeError("CapturedRollout.native_reasoning_text must be exact native text")
    if not isinstance(native_output, str) or not native_output:
        raise ValueError("CapturedRollout.native_output_text must be non-empty exact native text")
    if getattr(rollout.token_record, "native_reasoning_text", None) != native_reasoning:
        raise RuntimeError("native reasoning differs between rollout and Arbor token record")
    if getattr(rollout.token_record, "native_output_text", None) != native_output:
        raise RuntimeError("native output differs between rollout and Arbor token record")
    parsed_reasoning_positions = layout.reasoning_positions
    output_positions = layout.output_positions
    history_positions = layout.history_positions
    if not parsed_reasoning_positions or not output_positions:
        raise RuntimeError("dependency replay requires non-empty native T and O token spans")
    if history_positions != tuple(range(len(history_positions))):
        raise RuntimeError("native thinking history must be a token prefix")
    if output_positions != tuple(
        range(len(history_positions), len(history_positions) + len(output_positions))
    ):
        raise RuntimeError("native visible output must begin after the thinking history")
    if parsed_reasoning_positions and parsed_reasoning_positions[-1] >= output_positions[0]:
        raise RuntimeError("native reasoning token span must precede native output")
    return HistoricalRolloutTarget(
        text=completion,
        token_ids=completion_ids,
        reasoning_span=(parsed_reasoning_positions[0], parsed_reasoning_positions[-1]),
        output_span=(output_positions[0], output_positions[-1]),
    )


def build_aime_replay_dependency_inputs(
    *,
    tokenizer: Any,
    chat_adapter: QwenNativeAIMEAdapter,
    rollout: CapturedRollout,
    parent_skill: str,
    candidate_skill: str,
) -> AIMEReplayDependencyInputs:
    """Render one captured AIME rollout into the exact b06 replay contract."""

    if not isinstance(chat_adapter, QwenNativeAIMEAdapter):
        raise TypeError("chat_adapter must be the shared QwenNativeAIMEAdapter instance")
    if not isinstance(rollout, CapturedRollout):
        raise TypeError("rollout must be a CapturedRollout")
    if not isinstance(parent_skill, str) or not isinstance(candidate_skill, str):
        raise TypeError("parent_skill and candidate_skill must be text")
    if rollout.skill_text != parent_skill or rollout.signature.instructions != parent_skill:
        raise RuntimeError("captured rollout does not belong to the supplied parent skill")
    if rollout.demos:
        raise ValueError("AIME dependency bridge v1 requires no demonstrations")
    if tuple(rollout.signature.input_fields) != ("problem",):
        raise ValueError("AIME dependency bridge v1 requires exactly the problem input field")
    if set(rollout.predictor_inputs) != {"problem"}:
        raise ValueError("captured predictor inputs must contain exactly problem")
    problem = rollout.predictor_inputs["problem"]
    if not isinstance(problem, str):
        raise TypeError("AIME problem must be plain text")

    old_prompt, old_messages, old_prompt_ids = _render_prompt_provenance(
        tokenizer=tokenizer,
        adapter=chat_adapter,
        signature=rollout.signature,
        problem=problem,
    )
    captured_messages = tuple(rollout.token_record.prompt_messages)
    if old_messages != captured_messages:
        raise RuntimeError(
            "official old ChatAdapter messages differ from the captured Arbor messages"
        )
    captured_prompt_ids = _integer_ids(
        rollout.token_record.prompt_token_ids,
        "captured prompt token IDs",
    )
    if old_prompt_ids != captured_prompt_ids:
        raise RuntimeError(
            "official old rendered prompt differs from captured Arbor prompt token IDs"
        )

    candidate_signature = rollout.signature.with_instructions(candidate_skill)
    candidate_prompt, _, _ = _render_prompt_provenance(
        tokenizer=tokenizer,
        adapter=chat_adapter,
        signature=candidate_signature,
        problem=problem,
    )
    target = _historical_target(
        tokenizer=tokenizer,
        rollout=rollout,
    )
    return AIMEReplayDependencyInputs(
        instance_id=rollout.instance_index,
        old_prompt=old_prompt,
        candidate_prompt=candidate_prompt,
        target=target,
    )


class AIMEPreparedReplayDependency:
    """Reuse the parent measure while candidates replay one historical target."""

    def __init__(
        self,
        *,
        attributor: ExactTokenOffloadedLLMIFRAttribution,
        tokenizer: Any,
        chat_adapter: ChatAdapter,
        input_builder: Callable[..., AIMEReplayDependencyInputs],
        parent_skill: str,
        rollouts: tuple[CapturedRollout, ...],
        reference_inputs: tuple[AIMEReplayDependencyInputs, ...],
        old_measures: tuple[DependencyMeasure, ...],
    ) -> None:
        if not rollouts:
            raise ValueError("dependency replay minibatch must be non-empty")
        if len(reference_inputs) != len(rollouts) or len(old_measures) != len(rollouts):
            raise RuntimeError("parent dependency inputs, measures, and rollouts are misaligned")
        for rollout, item, measure in zip(
            rollouts,
            reference_inputs,
            old_measures,
            strict=True,
        ):
            if item.instance_id != rollout.instance_index or measure.instance_id != item.instance_id:
                raise RuntimeError("parent dependency instance coordinates are misaligned")
            if item.old_prompt != item.candidate_prompt:
                raise RuntimeError("a parent dependency input changed its old prompt")
            if (
                measure.target_token_ids != item.target.token_ids
                or measure.reasoning_span != item.target.reasoning_span
                or measure.output_span != item.target.output_span
            ):
                raise RuntimeError("a compact parent dependency measure changed its target")
            if measure.attributor_identity != id(attributor):
                raise RuntimeError("a parent measure came from a different official attributor")
        self._attributor = attributor
        self._tokenizer = tokenizer
        self._chat_adapter = chat_adapter
        self._input_builder = input_builder
        self._parent_skill = parent_skill
        self._rollouts = rollouts
        self._targets = tuple(item.target for item in reference_inputs)
        self._old = old_measures

    def distance(self, candidate_skill: str) -> float:
        inputs = tuple(
            self._input_builder(
                tokenizer=self._tokenizer,
                chat_adapter=self._chat_adapter,
                rollout=rollout,
                parent_skill=self._parent_skill,
                candidate_skill=candidate_skill,
            )
            for rollout in self._rollouts
        )
        if tuple(item.target for item in inputs) != self._targets:
            raise RuntimeError("candidate dependency replay changed the captured parent target")
        candidate = tuple(
            build_dependency_measure(
                attributor=self._attributor,
                tokenizer=self._tokenizer,
                instance_id=item.instance_id,
                prompt=item.candidate_prompt,
                target=item.target,
                capture_lease=None,
            )
            for item in inputs
        )
        return batch_dependency_distance(self._old, candidate)


@dataclass(frozen=True, slots=True)
class AIMEPreparedParentAnalysis:
    """Compact parent credit and old dependency produced from shared captures."""

    rollout_credits: tuple[tuple[CapturedRollout, FlashTraceCredit], ...]
    dependency: AIMEPreparedReplayDependency


class AIMEReplayDependencyDistance:
    """Compose AIME provenance with the b06 official-IFR distance bridge."""

    def __init__(
        self,
        *,
        attributor: ExactTokenOffloadedLLMIFRAttribution,
        tokenizer: Any,
        chat_adapter: ChatAdapter,
        input_builder: Callable[..., AIMEReplayDependencyInputs] | None = None,
    ) -> None:
        if not isinstance(attributor, ExactTokenOffloadedLLMIFRAttribution):
            raise TypeError("attributor must expose the adopted exact token-ID dependency entry point")
        if not isinstance(chat_adapter, ChatAdapter):
            raise TypeError("chat_adapter must be the shared official ChatAdapter instance")
        if input_builder is None:
            if not isinstance(chat_adapter, QwenNativeAIMEAdapter):
                raise TypeError(
                    "the default dependency input builder requires QwenNativeAIMEAdapter"
                )
            input_builder = build_aime_replay_dependency_inputs
        if not callable(input_builder):
            raise TypeError("input_builder must be callable")
        self._attributor = attributor
        self._tokenizer = tokenizer
        self._chat_adapter = chat_adapter
        self._input_builder = input_builder

    def _reference_inputs(
        self,
        *,
        parent_skill: str,
        rollouts: tuple[CapturedRollout, ...],
    ) -> tuple[AIMEReplayDependencyInputs, ...]:
        if not rollouts:
            raise ValueError("dependency replay minibatch must be non-empty")
        return tuple(
            self._input_builder(
                tokenizer=self._tokenizer,
                chat_adapter=self._chat_adapter,
                rollout=rollout,
                parent_skill=parent_skill,
                candidate_skill=parent_skill,
            )
            for rollout in rollouts
        )

    def _prepared_dependency(
        self,
        *,
        parent_skill: str,
        rollouts: tuple[CapturedRollout, ...],
        reference_inputs: tuple[AIMEReplayDependencyInputs, ...],
        old_measures: tuple[DependencyMeasure, ...],
    ) -> AIMEPreparedReplayDependency:
        return AIMEPreparedReplayDependency(
            attributor=self._attributor,
            tokenizer=self._tokenizer,
            chat_adapter=self._chat_adapter,
            input_builder=self._input_builder,
            parent_skill=parent_skill,
            rollouts=rollouts,
            reference_inputs=reference_inputs,
            old_measures=old_measures,
        )

    def prepare(
        self,
        *,
        parent_skill: str,
        rollouts: tuple[CapturedRollout, ...],
    ) -> AIMEPreparedReplayDependency:
        reference_inputs = self._reference_inputs(
            parent_skill=parent_skill,
            rollouts=rollouts,
        )
        old_measures = tuple(
            build_dependency_measure(
                attributor=self._attributor,
                tokenizer=self._tokenizer,
                instance_id=item.instance_id,
                prompt=item.old_prompt,
                target=item.target,
                capture_lease=None,
            )
            for item in reference_inputs
        )
        return self._prepared_dependency(
            parent_skill=parent_skill,
            rollouts=rollouts,
            reference_inputs=reference_inputs,
            old_measures=old_measures,
        )

    def prepare_parent_analysis(
        self,
        *,
        historical_replay: AIMEHistoricalReplayScorer,
        parent_skill: str,
        rollouts: tuple[CapturedRollout, ...],
    ) -> AIMEPreparedParentAnalysis:
        if not isinstance(historical_replay, AIMEHistoricalReplayScorer):
            raise TypeError("historical_replay must be the shared AIME scorer")
        if historical_replay.model is not getattr(self._attributor, "model", None):
            raise RuntimeError("parent credit and dependency must share one frozen model")
        if not any(historical_replay.advantage(rollout.reward) != 0.0 for rollout in rollouts):
            raise DependencyMeasureUnavailableError(
                "sum(abs(advantage)) is zero; skip this mutation"
            )

        reference_inputs = self._reference_inputs(
            parent_skill=parent_skill,
            rollouts=rollouts,
        )
        rollout_credits: list[tuple[CapturedRollout, FlashTraceCredit]] = []
        old_measures: list[DependencyMeasure] = []
        for rollout, item in zip(rollouts, reference_inputs, strict=True):
            with shared_exact_token_capture(self._attributor.model) as capture_lease:
                credit = historical_replay.credit_for_rollout(
                    rollout,
                    capture_lease=capture_lease,
                )
                old_measure = build_dependency_measure(
                    attributor=self._attributor,
                    tokenizer=self._tokenizer,
                    instance_id=item.instance_id,
                    prompt=item.old_prompt,
                    target=item.target,
                    capture_lease=capture_lease,
                )
            if credit is not None:
                rollout_credits.append(credit)
            old_measures.append(old_measure)
        if not rollout_credits:
            raise DependencyMeasureUnavailableError(
                "sum(abs(advantage)) is zero; skip this mutation"
            )

        return AIMEPreparedParentAnalysis(
            rollout_credits=tuple(rollout_credits),
            dependency=self._prepared_dependency(
                parent_skill=parent_skill,
                rollouts=rollouts,
                reference_inputs=reference_inputs,
                old_measures=tuple(old_measures),
            ),
        )

    def __call__(
        self,
        *,
        parent_skill: str,
        candidate_skill: str,
        rollouts: tuple[CapturedRollout, ...],
    ) -> float:
        prepared = self.prepare(parent_skill=parent_skill, rollouts=rollouts)
        return prepared.distance(candidate_skill)
