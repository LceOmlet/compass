from __future__ import annotations

import math
import threading
from collections.abc import Callable, Hashable, Mapping, Sequence
from concurrent.futures import Executor, Future
from contextlib import nullcontext
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Any

import dspy
from dspy.adapters.chat_adapter import ChatAdapter

from .b02_flashtrace_credit import FlashTraceCredit, build_flashtrace_credit
from .b03_token_replay import TokenReplayUtility
from .b06_dependency_gate import (
    CharacterSpan,
    DependencyMeasure,
    FieldTokenMeasure,
    HistoricalRolloutTarget,
    RenderedPromptProvenance,
    TaskObservationField,
    build_dependency_measure,
)
from .b14_flashtrace_token_ids import (
    ExactTokenOffloadedFlashTrace,
    ExactTokenOffloadedLLMIFRAttribution,
    shared_exact_token_capture,
)
from .b18_dspy_straggler_bound import bounded_dspy_straggler_resubmission
from .dspy_token_provenance import (
    ActualDSPyTokenProvenance,
    DSPyCallbackCall,
    DSPyCallbackCollector,
    ExactTraceDataLineageResolver,
    LineageUnavailableError,
    SourceCoordinate,
    TokenProvenanceUnavailableError,
    TokenProvenanceTrajectory,
    _unique_ordered_spans,
    official_rendered_instruction_span,
)
from .terminal_reflection import (
    PreparedTerminalAnalysis,
    TerminalAnalysisUnavailableError,
)


CandidateKey = frozenset[tuple[str, str]]

_IFBENCH_UPSTREAM_COMPONENT = "generate_response_module.predict"
_IFBENCH_DOWNSTREAM_COMPONENT = "ensure_correct_response_module.predict"
_IFBENCH_QUERY_FIELD = "query"
_IFBENCH_EDGE_OUTPUT_FIELD = "response"
_IFBENCH_TERMINAL_OUTPUT_FIELD = "final_response"
_IFBENCH_PROGRAM_OUTPUT_FIELD = "response"


def _candidate_key(candidate: Mapping[str, str]) -> CandidateKey:
    if not isinstance(candidate, Mapping):
        raise TypeError("a GEPA candidate must be a component mapping")
    items = tuple(candidate.items())
    if not items or any(
        not isinstance(name, str)
        or not name
        or not isinstance(text, str)
        for name, text in items
    ):
        raise TypeError("every GEPA candidate component must map non-empty names to text")
    if len({name for name, _ in items}) != len(items):
        raise ValueError("a GEPA candidate contains duplicate component names")
    return frozenset(items)


@dataclass(frozen=True, slots=True)
class DSPyTokenRecord:
    """Canonical callback-derived fields consumed by b02/b03."""

    prompt_text: str
    completion_text: str
    native_reasoning_text: str
    native_output_text: str
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    token_layout: Any


@dataclass(frozen=True, slots=True)
class _FieldTokenCoordinate:
    token_position: int
    token_id: int
    relative_start: int
    relative_end: int


@dataclass(frozen=True, slots=True)
class DSPyHistoricalRollout:
    """The minimal old-rollout view consumed by the existing analysis bridges."""

    instance_index: Hashable
    reward: float
    component: str
    skill_text: str
    signature: Any
    demos: tuple[Any, ...]
    predictor_inputs: dict[str, Any]
    input_field_tokens: dict[str, tuple[_FieldTokenCoordinate, ...]]
    output_field_tokens: dict[str, tuple[_FieldTokenCoordinate, ...]]
    token_record: DSPyTokenRecord
    provenance: TokenProvenanceTrajectory


@dataclass(frozen=True, slots=True)
class DSPyReplayDependencyInputs:
    instance_id: Hashable
    old_prompt: RenderedPromptProvenance
    candidate_prompt: RenderedPromptProvenance
    target: HistoricalRolloutTarget


@dataclass(frozen=True, slots=True)
class PreparedParentReplay:
    rollout_credits: tuple[tuple[DSPyHistoricalRollout, FlashTraceCredit], ...]
    paths: tuple["_PreparedIFBenchPath", ...]
    old_cross_measures: tuple["_CrossDependencyMeasure", ...]


@dataclass(frozen=True, slots=True)
class DSPyObservationFact:
    """One official execution atomically bound to a program-instance maximum."""

    score: float
    output: Any
    trajectory: Mapping[str, Any]
    callback_calls: tuple[DSPyCallbackCall, ...]


@dataclass(frozen=True, slots=True)
class DSPyReferenceObservation:
    """A selected admission reference plus its official rendered program."""

    program_idx: int
    instance_id: Hashable
    candidate: Mapping[str, str]
    fact: DSPyObservationFact
    stage_signatures: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _CrossDependencyMeasure:
    """End-to-end IFBench measure after contracting the declared response edge."""

    instance_id: Hashable
    query_token_coordinates: tuple[tuple[int, int, int], ...]
    query_masses: tuple[float, ...]
    skill_mass: float
    format_mass: float
    unresolved_mass: float
    attributor_identity: int


@dataclass(frozen=True, slots=True)
class _PreparedIFBenchPath:
    upstream: DSPyHistoricalRollout
    downstream: DSPyHistoricalRollout
    upstream_target: HistoricalRolloutTarget
    downstream_target: HistoricalRolloutTarget
    upstream_old_measure: DependencyMeasure | None
    downstream_old_measure: DependencyMeasure


def _integer_ids(values: Any, name: str) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError as error:
        raise TerminalAnalysisUnavailableError(f"{name} is not a token-ID sequence") from error
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise TerminalAnalysisUnavailableError(f"{name} must contain integer token IDs")
    return result


def _encode_with_offsets(
    tokenizer: Any,
    text: str,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if not isinstance(encoded, Mapping):
        raise TerminalAnalysisUnavailableError("the tokenizer did not return an encoding mapping")
    ids = _integer_ids(encoded.get("input_ids"), "canonical prompt IDs")
    try:
        offsets = tuple((int(start), int(end)) for start, end in encoded["offset_mapping"])
    except (KeyError, TypeError, ValueError) as error:
        raise TerminalAnalysisUnavailableError(
            "the tokenizer did not expose direct prompt offsets"
        ) from error
    if len(ids) != len(offsets) or any(
        start < 0 or end <= start or end > len(text)
        for start, end in offsets
    ):
        raise TerminalAnalysisUnavailableError("canonical prompt offsets are invalid")
    decoded = tokenizer.decode(
        list(ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != text:
        raise TerminalAnalysisUnavailableError("canonical prompt failed exact local round-trip")
    return ids, offsets


def _single_occurrence(text: str, value: str, name: str) -> tuple[int, int]:
    start = text.find(value)
    if not value or start < 0 or text.find(value, start + 1) >= 0:
        raise TerminalAnalysisUnavailableError(f"{name} is not one exact substring")
    return start, start + len(value)


def _direct_field_token_coordinates(
    *,
    tokenizer: Any,
    text: str,
    token_ids: tuple[int, ...],
    value: str,
    name: str,
    position_map: tuple[int, ...] | None = None,
) -> tuple[_FieldTokenCoordinate, ...]:
    """Return fully-contained tokens in exact field-relative character coordinates."""

    field_start, field_end = _single_occurrence(text, value, name)
    encoded_ids, offsets = _encode_with_offsets(tokenizer, text)
    if encoded_ids != token_ids:
        raise TerminalAnalysisUnavailableError(
            f"{name} token IDs differ from the canonical visible output"
        )
    if position_map is None:
        position_map = tuple(range(len(token_ids)))
    if len(position_map) != len(token_ids) or len(set(position_map)) != len(position_map):
        raise TerminalAnalysisUnavailableError(f"{name} token-position map is invalid")
    contained = tuple(
        index
        for index, (start, end) in enumerate(offsets)
        if field_start <= start and end <= field_end
    )
    if not contained:
        raise TerminalAnalysisUnavailableError(
            f"{name} contains no directly attributable token"
        )
    result = tuple(
        _FieldTokenCoordinate(
            token_position=position_map[index],
            token_id=token_ids[index],
            relative_start=offsets[index][0] - field_start,
            relative_end=offsets[index][1] - field_start,
        )
        for index in contained
    )
    keys = tuple(
        (token.relative_start, token.relative_end, token.token_id)
        for token in result
    )
    if len(set(keys)) != len(keys) or any(
        left.relative_end > right.relative_start
        for left, right in zip(result, result[1:], strict=False)
    ):
        raise TerminalAnalysisUnavailableError(
            f"{name} has ambiguous field-relative token coordinates"
        )
    return result


def _input_field_token_coordinates(
    *,
    tokenizer: Any,
    trajectory: TokenProvenanceTrajectory,
) -> dict[str, tuple[_FieldTokenCoordinate, ...]]:
    """Project actual prompt-source tokens into their field-relative coordinates."""

    token_ids, offsets = _encode_with_offsets(
        tokenizer,
        trajectory.canonical_prompt_text,
    )
    if token_ids != trajectory.canonical_prompt_token_ids or len(
        trajectory.canonical_prompt_coordinates
    ) != len(token_ids):
        raise TerminalAnalysisUnavailableError(
            "actual prompt IDs, offsets, and provenance coordinates are misaligned"
        )
    spans = dict(trajectory.canonical_task_observation_character_spans)
    if len(spans) != len(trajectory.canonical_task_observation_character_spans):
        raise TerminalAnalysisUnavailableError(
            "actual prompt contains duplicate task/observation field coordinates"
        )
    result: dict[str, list[_FieldTokenCoordinate]] = {
        field_name: [] for field_name in spans
    }
    for position, (token_id, (start, end), source) in enumerate(
        zip(
            token_ids,
            offsets,
            trajectory.canonical_prompt_coordinates,
            strict=True,
        )
    ):
        if source.kind != "task_observation":
            continue
        field_name = source.field_name
        if field_name not in spans:
            raise TerminalAnalysisUnavailableError(
                "actual prompt provenance names an undeclared task/observation field"
            )
        field_start, field_end = spans[field_name]
        if not (field_start <= start and end <= field_end):
            raise TerminalAnalysisUnavailableError(
                f"task/observation field {field_name!r} contains a crossing token"
            )
        field_tokens = result[field_name]
        if source.source_token_index != len(field_tokens):
            raise TerminalAnalysisUnavailableError(
                f"task/observation field {field_name!r} token indices are not direct"
            )
        field_tokens.append(
            _FieldTokenCoordinate(
                token_position=position,
                token_id=token_id,
                relative_start=start - field_start,
                relative_end=end - field_start,
            )
        )
    finalized = {field_name: tuple(tokens) for field_name, tokens in result.items()}
    for field_name, tokens in finalized.items():
        keys = tuple(
            (token.relative_start, token.relative_end, token.token_id)
            for token in tokens
        )
        if len(set(keys)) != len(keys) or any(
            left.relative_end > right.relative_start
            for left, right in zip(tokens, tokens[1:], strict=False)
        ):
            raise TerminalAnalysisUnavailableError(
                f"task/observation field {field_name!r} has ambiguous token coordinates"
            )
    return finalized


def _source_coordinates(
    *,
    offsets: tuple[tuple[int, int], ...],
    skill_span: tuple[int, int],
    task_spans: tuple[tuple[str, tuple[int, int]], ...],
) -> tuple[SourceCoordinate, ...]:
    coordinates: list[SourceCoordinate] = []
    source_indices = {field_name: 0 for field_name, _ in task_spans}
    for start, end in offsets:
        task_matches = [
            field_name
            for field_name, (left, right) in task_spans
            if left <= start and end <= right
        ]
        skill_match = skill_span[0] <= start and end <= skill_span[1]
        if len(task_matches) + int(skill_match) > 1:
            raise TerminalAnalysisUnavailableError(
                "one canonical prompt token belongs to two explicit sources"
            )
        if task_matches:
            field_name = task_matches[0]
            token_index = source_indices[field_name]
            source_indices[field_name] += 1
            coordinates.append(SourceCoordinate("task_observation", field_name, token_index))
        elif skill_match:
            coordinates.append(SourceCoordinate("skill"))
        else:
            coordinates.append(SourceCoordinate("format"))
    return tuple(coordinates)


def _coordinate_run_span(
    *,
    coordinates: tuple[SourceCoordinate, ...],
    offsets: tuple[tuple[int, int], ...],
    predicate: Callable[[SourceCoordinate], bool],
    empty_position: int,
    name: str,
) -> CharacterSpan:
    positions = tuple(index for index, coordinate in enumerate(coordinates) if predicate(coordinate))
    if not positions:
        return CharacterSpan(empty_position, empty_position)
    if positions != tuple(range(positions[0], positions[-1] + 1)):
        raise TerminalAnalysisUnavailableError(f"{name} token coordinates are not contiguous")
    return CharacterSpan(offsets[positions[0]][0], offsets[positions[-1]][1])


def _rendered_prompt_from_coordinates(
    *,
    text: str,
    token_ids: tuple[int, ...],
    coordinates: tuple[SourceCoordinate, ...],
    skill_character_span: tuple[int, int],
    task_character_spans: tuple[tuple[str, tuple[int, int]], ...],
    tokenizer: Any,
) -> RenderedPromptProvenance:
    encoded_ids, offsets = _encode_with_offsets(tokenizer, text)
    if encoded_ids != token_ids or len(coordinates) != len(token_ids):
        raise TerminalAnalysisUnavailableError(
            "canonical prompt IDs, offsets, and source coordinates are misaligned"
        )
    task_fields = tuple(
        TaskObservationField(
            field_name,
            _coordinate_run_span(
                coordinates=coordinates,
                offsets=offsets,
                predicate=lambda coordinate, field_name=field_name: (
                    coordinate.kind == "task_observation"
                    and coordinate.field_name == field_name
                ),
                empty_position=character_span[0],
                name=f"task/observation field {field_name!r}",
            ),
        )
        for field_name, character_span in task_character_spans
    )
    skill_span = _coordinate_run_span(
        coordinates=coordinates,
        offsets=offsets,
        predicate=lambda coordinate: coordinate.kind == "skill",
        empty_position=skill_character_span[0],
        name="skill",
    )
    return RenderedPromptProvenance(
        text=text,
        token_ids=token_ids,
        task_observation_fields=task_fields,
        skill_spans=(skill_span,),
    )


def _candidate_prompt(
    *,
    tokenizer: Any,
    chat_adapter: ChatAdapter,
    chat_template_kwargs: Mapping[str, Any],
    rollout: DSPyHistoricalRollout,
    candidate_skill: str,
) -> RenderedPromptProvenance:
    signature = rollout.signature.with_instructions(candidate_skill)
    messages = tuple(
        chat_adapter.format(
            signature=signature,
            demos=list(rollout.demos),
            inputs=dict(rollout.predictor_inputs),
        )
    )
    if not messages or any(
        not isinstance(message, Mapping)
        or not isinstance(message.get("role"), str)
        or not isinstance(message.get("content"), str)
        or not message["content"]
        for message in messages
    ):
        raise TerminalAnalysisUnavailableError(
            "official ChatAdapter.format did not return non-empty plain-text messages"
        )
    prompt_text = tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tokenize=False,
        **dict(chat_template_kwargs),
    )
    if not isinstance(prompt_text, str) or not prompt_text:
        raise TerminalAnalysisUnavailableError("Qwen chat template did not return prompt text")
    prompt_ids, offsets = _encode_with_offsets(tokenizer, prompt_text)
    try:
        message_spans = _unique_ordered_spans(
            prompt_text,
            tuple(message["content"] for message in messages),
        )
    except TokenProvenanceUnavailableError as error:
        raise TerminalAnalysisUnavailableError(
            "candidate messages do not have one ordered embedding in the rendered prompt"
        ) from error

    try:
        system_index, local_skill_span = official_rendered_instruction_span(
            adapter=chat_adapter,
            signature=signature,
            messages=messages,
        )
    except TokenProvenanceUnavailableError as error:
        raise TerminalAnalysisUnavailableError(
            "candidate official rendered instruction span is unavailable"
        ) from error
    message_start = message_spans[system_index][0]
    skill_span = (
        message_start + local_skill_span[0],
        message_start + local_skill_span[1],
    )

    last_message = messages[-1]
    if last_message.get("role") != "user" or not isinstance(last_message.get("content"), str):
        raise TerminalAnalysisUnavailableError("official prompt does not end in a plain user message")
    user_content = last_message["content"]
    user_start = message_spans[-1][0]
    field_names = tuple(signature.input_fields)
    field_values: list[str] = []
    for field_name in field_names:
        value = rollout.predictor_inputs.get(field_name)
        if not isinstance(value, str) or not value:
            raise TerminalAnalysisUnavailableError(
                f"task/observation field {field_name!r} is not non-empty text"
            )
        field_values.append(value)
    try:
        local_field_spans = _unique_ordered_spans(
            user_content,
            tuple(field_values),
        )
    except TokenProvenanceUnavailableError as error:
        raise TerminalAnalysisUnavailableError(
            "candidate task fields do not have one ordered embedding in the current user message"
        ) from error
    task_spans = [
        (
            field_name,
            (user_start + local_start, user_start + local_end),
        )
        for field_name, (local_start, local_end) in zip(
            field_names,
            local_field_spans,
            strict=True,
        )
    ]
    coordinates = _source_coordinates(
        offsets=offsets,
        skill_span=skill_span,
        task_spans=tuple(task_spans),
    )
    return _rendered_prompt_from_coordinates(
        text=prompt_text,
        token_ids=prompt_ids,
        coordinates=coordinates,
        skill_character_span=skill_span,
        task_character_spans=tuple(task_spans),
        tokenizer=tokenizer,
    )


def _historical_target(rollout: DSPyHistoricalRollout) -> HistoricalRolloutTarget:
    layout = rollout.provenance.canonical_token_layout
    reasoning = tuple(layout.reasoning_positions)
    output = tuple(layout.output_positions)
    if not reasoning or not output:
        raise TerminalAnalysisUnavailableError(
            "old rollout must contain non-empty native reasoning and output spans"
        )
    return HistoricalRolloutTarget(
        text=rollout.provenance.canonical_completion_text,
        token_ids=tuple(layout.nonterminal_token_ids),
        reasoning_span=(reasoning[0], reasoning[-1]),
        output_span=(output[0], output[-1]),
    )


def _field_measures(measure: DependencyMeasure) -> dict[str, FieldTokenMeasure]:
    fields = {field.field_id: field for field in measure.task_observation}
    if len(fields) != len(measure.task_observation):
        raise TerminalAnalysisUnavailableError(
            "a local dependency measure contains duplicate source fields"
        )
    return fields


def _finite_nonnegative(values: Sequence[Real], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise TerminalAnalysisUnavailableError(f"{name} must be finite and non-negative")
    return result


def _response_sink_weights(
    *,
    downstream_measure: DependencyMeasure,
    downstream_rollout: DSPyHistoricalRollout,
    upstream: DSPyHistoricalRollout,
    upstream_target: HistoricalRolloutTarget,
) -> tuple[float, ...]:
    """Bind response-source mass by exact field-relative token coordinates."""

    response = _field_measures(downstream_measure).get(_IFBENCH_EDGE_OUTPUT_FIELD)
    if response is None:
        raise TerminalAnalysisUnavailableError(
            "the downstream IFBench call omitted its declared response input"
        )
    downstream_tokens = downstream_rollout.input_field_tokens.get(
        _IFBENCH_EDGE_OUTPUT_FIELD
    )
    if not downstream_tokens:
        raise TerminalAnalysisUnavailableError(
            "downstream response contains no directly attributable source token"
        )
    if response.token_ids != tuple(token.token_id for token in downstream_tokens):
        raise TerminalAnalysisUnavailableError(
            "downstream response dependency tokens differ from actual prompt provenance"
        )
    field_weights = _finite_nonnegative(
        response.masses,
        "downstream response token attribution",
    )
    if len(field_weights) != len(downstream_tokens):
        raise TerminalAnalysisUnavailableError(
            "downstream response attribution and source-token coordinates are misaligned"
        )

    start, end = upstream_target.output_span
    upstream_tokens = upstream.output_field_tokens.get(_IFBENCH_EDGE_OUTPUT_FIELD)
    if not upstream_tokens or any(
        token.token_position < start or token.token_position > end
        for token in upstream_tokens
    ):
        raise TerminalAnalysisUnavailableError(
            "upstream response field has no direct token coordinates inside its visible output"
        )
    if any(
        upstream_target.token_ids[token.token_position] != token.token_id
        for token in upstream_tokens
    ):
        raise TerminalAnalysisUnavailableError(
            "upstream response coordinates differ from the old rollout token IDs"
        )
    upstream_by_key = {
        (token.relative_start, token.relative_end, token.token_id): token
        for token in upstream_tokens
    }
    if len(upstream_by_key) != len(upstream_tokens):
        raise TerminalAnalysisUnavailableError(
            "upstream response has ambiguous field-relative token coordinates"
        )
    downstream_keys = tuple(
        (token.relative_start, token.relative_end, token.token_id)
        for token in downstream_tokens
    )
    if len(set(downstream_keys)) != len(downstream_keys):
        raise TerminalAnalysisUnavailableError(
            "downstream response has ambiguous field-relative token coordinates"
        )
    weights = [0.0] * (end - start + 1)
    used_upstream_positions: set[int] = set()
    for key, weight in zip(downstream_keys, field_weights, strict=True):
        upstream_token = upstream_by_key.get(key)
        if (
            upstream_token is None
            or upstream_token.token_position in used_upstream_positions
        ):
            raise TerminalAnalysisUnavailableError(
                "one downstream response-source token has no one-to-one exact upstream binding"
            )
        used_upstream_positions.add(upstream_token.token_position)
        weights[upstream_token.token_position - start] = weight
    return tuple(weights)


def _ground_answer_credit(
    credit: FlashTraceCredit,
    sink_weights: tuple[float, ...],
) -> FlashTraceCredit:
    """Ground b02's direct answer mass without changing its reasoning credit."""

    weights = _finite_nonnegative(sink_weights, "cross-predictor answer credit")
    if len(weights) != len(credit.answer_positions):
        raise TerminalAnalysisUnavailableError(
            "cross-predictor answer credit is not aligned to upstream output tokens"
        )
    sink_total = math.fsum(weights)
    if sink_total <= 0.0:
        raise TerminalAnalysisUnavailableError(
            "cross-predictor answer credit has zero total mass"
        )
    token_weights = list(credit.token_weights)
    answer_mass = math.fsum(token_weights[position] for position in credit.answer_positions)
    for position in credit.answer_positions:
        token_weights[position] = 0.0
    for position, weight in zip(credit.answer_positions, weights, strict=True):
        token_weights[position] = answer_mass * weight / sink_total
    if not math.isclose(
        math.fsum(token_weights),
        math.fsum(credit.token_weights),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise TerminalAnalysisUnavailableError(
            "grounding upstream answer tokens changed b02 total credit mass"
        )
    return replace(credit, token_weights=tuple(token_weights))


def _query_coordinates_and_masses(
    *,
    rollout: DSPyHistoricalRollout,
    measure: FieldTokenMeasure | None,
    mass_scale: float,
    name: str,
) -> tuple[tuple[tuple[int, int, int], ...], tuple[float, ...]]:
    if not math.isfinite(mass_scale) or mass_scale < 0.0:
        raise TerminalAnalysisUnavailableError(
            f"{name} query attribution scale must be finite and non-negative"
        )
    tokens = rollout.input_field_tokens.get(_IFBENCH_QUERY_FIELD)
    if tokens is None:
        raise TerminalAnalysisUnavailableError(
            f"{name} omitted the official query source coordinates"
        )
    coordinates = tuple(
        (token.relative_start, token.relative_end, token.token_id)
        for token in tokens
    )
    if len(set(coordinates)) != len(coordinates):
        raise TerminalAnalysisUnavailableError(
            f"{name} has ambiguous query source coordinates"
        )
    if measure is None:
        masses = (0.0,) * len(coordinates)
    else:
        if measure.token_ids != tuple(token.token_id for token in tokens):
            raise TerminalAnalysisUnavailableError(
                f"{name} query dependency tokens differ from actual prompt provenance"
            )
        raw_masses = _finite_nonnegative(measure.masses, f"{name} query attribution")
        if len(raw_masses) != len(coordinates):
            raise TerminalAnalysisUnavailableError(
                f"{name} query attribution and source coordinates are misaligned"
            )
        masses = tuple(mass_scale * mass for mass in raw_masses)
        _finite_nonnegative(masses, f"scaled {name} query attribution")
    return coordinates, masses


def _merge_query_coordinates(
    *,
    downstream_coordinates: tuple[tuple[int, int, int], ...],
    downstream_masses: tuple[float, ...],
    upstream_coordinates: tuple[tuple[int, int, int], ...],
    upstream_masses: tuple[float, ...],
) -> tuple[tuple[tuple[int, int, int], ...], tuple[float, ...]]:
    """Direct-union two query tokenizations without redistributing character mass."""

    if len(downstream_coordinates) != len(downstream_masses) or len(
        upstream_coordinates
    ) != len(upstream_masses):
        raise TerminalAnalysisUnavailableError(
            "query coordinates and attribution masses are misaligned"
        )
    merged_coordinates: list[tuple[int, int, int]] = []
    merged_masses: list[float] = []
    downstream_index = 0
    upstream_index = 0
    while (
        downstream_index < len(downstream_coordinates)
        and upstream_index < len(upstream_coordinates)
    ):
        downstream_coordinate = downstream_coordinates[downstream_index]
        upstream_coordinate = upstream_coordinates[upstream_index]
        if downstream_coordinate == upstream_coordinate:
            merged_coordinates.append(downstream_coordinate)
            merged_masses.append(
                downstream_masses[downstream_index] + upstream_masses[upstream_index]
            )
            downstream_index += 1
            upstream_index += 1
            continue
        downstream_start, downstream_end, _ = downstream_coordinate
        upstream_start, upstream_end, _ = upstream_coordinate
        if downstream_end <= upstream_start:
            merged_coordinates.append(downstream_coordinate)
            merged_masses.append(downstream_masses[downstream_index])
            downstream_index += 1
            continue
        if upstream_end <= downstream_start:
            merged_coordinates.append(upstream_coordinate)
            merged_masses.append(upstream_masses[upstream_index])
            upstream_index += 1
            continue
        raise TerminalAnalysisUnavailableError(
            "the repeated IFBench query has overlapping but non-identical token coordinates"
        )
    for index in range(downstream_index, len(downstream_coordinates)):
        merged_coordinates.append(downstream_coordinates[index])
        merged_masses.append(downstream_masses[index])
    for index in range(upstream_index, len(upstream_coordinates)):
        merged_coordinates.append(upstream_coordinates[index])
        merged_masses.append(upstream_masses[index])
    ordered = tuple(
        sorted(
            zip(merged_coordinates, merged_masses, strict=True),
            key=lambda item: item[0],
        )
    )
    coordinates = tuple(coordinate for coordinate, _ in ordered)
    masses = tuple(mass for _, mass in ordered)
    if len(set(coordinates)) != len(coordinates):
        raise TerminalAnalysisUnavailableError(
            "merged query coordinates are not a direct union"
        )
    _finite_nonnegative(masses, "merged IFBench query attribution")
    return coordinates, masses


def _compose_ifbench_dependency(
    *,
    upstream: DependencyMeasure | None,
    downstream: DependencyMeasure,
    upstream_rollout: DSPyHistoricalRollout,
    downstream_rollout: DSPyHistoricalRollout,
) -> _CrossDependencyMeasure:
    """Contract the official edge without creating attribution mass.

    Direct downstream sources remain unchanged.  The response-field mass is
    replaced by the weighted upstream visible measure after normalizing that
    measure to the exact edge mass.  If it has zero visible mass, the edge mass
    moves to the explicit UNRESOLVED coordinate instead of being guessed.
    """

    downstream_fields = _field_measures(downstream)
    if set(downstream_fields) != {
        _IFBENCH_QUERY_FIELD,
        _IFBENCH_EDGE_OUTPUT_FIELD,
    }:
        raise TerminalAnalysisUnavailableError(
            "actual IFBench source fields differ from the official two-stage program"
        )

    if downstream.instance_id != downstream_rollout.instance_index:
        raise TerminalAnalysisUnavailableError(
            "downstream dependency belongs to a different historical rollout"
        )
    downstream_query = downstream_fields[_IFBENCH_QUERY_FIELD]
    downstream_query_coordinates, downstream_query_masses = (
        _query_coordinates_and_masses(
            rollout=downstream_rollout,
            measure=downstream_query,
            mass_scale=1.0,
            name="downstream",
        )
    )

    edge = downstream_fields[_IFBENCH_EDGE_OUTPUT_FIELD]
    edge_masses = _finite_nonnegative(edge.masses, "downstream response attribution")
    edge_mass = math.fsum(edge_masses)
    downstream_total = float(downstream.total_mass)
    if not math.isfinite(downstream_total) or downstream_total < 0.0:
        raise TerminalAnalysisUnavailableError("local dependency mass is invalid")
    if edge_mass > downstream_total and not math.isclose(
        edge_mass,
        downstream_total,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise TerminalAnalysisUnavailableError(
            "downstream response attribution exceeds total visible attribution"
        )

    upstream_query: FieldTokenMeasure | None = None
    if edge_mass > 0.0:
        if upstream is None:
            raise TerminalAnalysisUnavailableError(
                "positive downstream response mass has no weighted upstream measure"
            )
        if upstream.instance_id != downstream.instance_id:
            raise TerminalAnalysisUnavailableError(
                "the two IFBench stage measures belong to different instances"
            )
        if upstream.attributor_identity != downstream.attributor_identity:
            raise TerminalAnalysisUnavailableError(
                "the two IFBench stage measures use different attribution engines"
            )
        upstream_fields = _field_measures(upstream)
        if set(upstream_fields) != {_IFBENCH_QUERY_FIELD}:
            raise TerminalAnalysisUnavailableError(
                "the upstream IFBench call differs from the official source fields"
            )
        upstream_query = upstream_fields[_IFBENCH_QUERY_FIELD]
        if upstream.instance_id != upstream_rollout.instance_index:
            raise TerminalAnalysisUnavailableError(
                "upstream dependency belongs to a different historical rollout"
            )
        upstream_total = float(upstream.total_mass)
        if not math.isfinite(upstream_total) or upstream_total < 0.0:
            raise TerminalAnalysisUnavailableError("weighted upstream dependency mass is invalid")
        if upstream_total > 0.0:
            transport = edge_mass / upstream_total
            unresolved = 0.0
        else:
            transport = 0.0
            unresolved = edge_mass
    else:
        transport = 0.0
        unresolved = 0.0
    upstream_query_value = upstream_rollout.predictor_inputs.get(_IFBENCH_QUERY_FIELD)
    downstream_query_value = downstream_rollout.predictor_inputs.get(_IFBENCH_QUERY_FIELD)
    if (
        not isinstance(upstream_query_value, str)
        or not upstream_query_value
        or upstream_query_value != downstream_query_value
    ):
        raise TerminalAnalysisUnavailableError(
            "the repeated IFBench query text differs across historical stages"
        )
    upstream_query_coordinates, upstream_query_masses = _query_coordinates_and_masses(
        rollout=upstream_rollout,
        measure=upstream_query,
        mass_scale=transport,
        name="upstream",
    )
    query_coordinates, query_masses = _merge_query_coordinates(
        downstream_coordinates=downstream_query_coordinates,
        downstream_masses=downstream_query_masses,
        upstream_coordinates=upstream_query_coordinates,
        upstream_masses=upstream_query_masses,
    )
    skill_mass = float(downstream.skill_mass) + transport * (
        float(upstream.skill_mass) if upstream is not None else 0.0
    )
    format_mass = float(downstream.format_mass) + transport * (
        float(upstream.format_mass) if upstream is not None else 0.0
    )
    values = (*query_masses, skill_mass, format_mass, unresolved)
    _finite_nonnegative(values, "composed IFBench dependency")
    composed_total = math.fsum(values)
    if not math.isclose(
        composed_total,
        downstream_total,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise TerminalAnalysisUnavailableError(
            "contracting the official IFBench edge did not conserve attribution mass"
        )
    return _CrossDependencyMeasure(
        instance_id=downstream.instance_id,
        query_token_coordinates=query_coordinates,
        query_masses=query_masses,
        skill_mass=skill_mass,
        format_mass=format_mass,
        unresolved_mass=unresolved,
        attributor_identity=downstream.attributor_identity,
    )


def _batch_cross_dependency_distance(
    old: Sequence[_CrossDependencyMeasure],
    candidate: Sequence[_CrossDependencyMeasure],
) -> float:
    old_measures = tuple(old)
    candidate_measures = tuple(candidate)
    if not old_measures or len(old_measures) != len(candidate_measures):
        raise TerminalAnalysisUnavailableError(
            "old and candidate IFBench dependency batches are misaligned"
        )
    numerators: list[float] = []
    denominators: list[float] = []
    for old_measure, candidate_measure in zip(
        old_measures,
        candidate_measures,
        strict=True,
    ):
        if (
            old_measure.instance_id != candidate_measure.instance_id
            or old_measure.query_token_coordinates
            != candidate_measure.query_token_coordinates
            or old_measure.attributor_identity != candidate_measure.attributor_identity
        ):
            raise TerminalAnalysisUnavailableError(
                "old and candidate IFBench dependency coordinates differ"
            )
        old_values = (
            *old_measure.query_masses,
            old_measure.skill_mass,
            old_measure.format_mass,
            old_measure.unresolved_mass,
        )
        candidate_values = (
            *candidate_measure.query_masses,
            candidate_measure.skill_mass,
            candidate_measure.format_mass,
            candidate_measure.unresolved_mass,
        )
        numerators.append(
            math.fsum(
                abs(old_value - candidate_value)
                for old_value, candidate_value in zip(
                    old_values,
                    candidate_values,
                    strict=True,
                )
            )
        )
        denominators.append(math.fsum(old_values) + math.fsum(candidate_values))
    denominator = math.fsum(denominators)
    if denominator == 0.0:
        return 0.0
    distance = math.fsum(numerators) / denominator
    if not math.isfinite(distance) or distance < 0.0 or distance > 1.0:
        raise TerminalAnalysisUnavailableError(
            "composed IFBench dependency distance lies outside [0, 1]"
        )
    return distance


class DSPyParentAnalysisBuilder:
    """Convert max-bound admission observations into existing analysis objects."""

    def __init__(
        self,
        *,
        model: Any,
        tracer: ExactTokenOffloadedFlashTrace,
        attributor: ExactTokenOffloadedLLMIFRAttribution,
        tokenizer: Any,
        chat_adapter: ChatAdapter,
        provenance: ActualDSPyTokenProvenance,
        lineage_resolver: ExactTraceDataLineageResolver,
        chat_template_kwargs: Mapping[str, Any],
        teacher_forcing_batch_size: Integral = 1,
    ) -> None:
        if not isinstance(tracer, ExactTokenOffloadedFlashTrace):
            raise TypeError("tracer must be the existing exact-token FlashTrace facade")
        if not isinstance(attributor, ExactTokenOffloadedLLMIFRAttribution):
            raise TypeError("attributor must be the existing exact-token dependency engine")
        if getattr(tracer, "model", None) is not model or getattr(attributor, "model", None) is not model:
            raise RuntimeError("credit, dependency, and teacher forcing must share one frozen model")
        if getattr(tracer, "tokenizer", None) is not tokenizer or getattr(attributor, "tokenizer", None) is not tokenizer:
            raise RuntimeError("credit and dependency must share the canonical tokenizer")
        if not isinstance(chat_adapter, ChatAdapter):
            raise TypeError("chat_adapter must be the official shared ChatAdapter")
        if (
            isinstance(teacher_forcing_batch_size, bool)
            or not isinstance(teacher_forcing_batch_size, Integral)
            or int(teacher_forcing_batch_size) <= 0
        ):
            raise TypeError("teacher_forcing_batch_size must be a positive integer")
        self._model = model
        self._tracer = tracer
        self._attributor = attributor
        self._tokenizer = tokenizer
        self._chat_adapter = chat_adapter
        self._provenance = provenance
        self._lineage_resolver = lineage_resolver
        self._chat_template_kwargs = dict(chat_template_kwargs)
        self._replay = TokenReplayUtility(
            model=model,
            tokenizer=tokenizer,
            adapter=chat_adapter,
            max_batch_size=teacher_forcing_batch_size,
        )
        self._credit_cache: dict[tuple[Any, ...], FlashTraceCredit] = {}
        # Derived analysis is process-local. Observation facts themselves are
        # persisted by the adapter; after resume this cache is deliberately
        # rebuilt against the newly loaded frozen model/attributor objects.
        self._prepared_cache: dict[tuple[Any, ...], PreparedParentReplay] = {}

    @property
    def replay(self) -> TokenReplayUtility:
        return self._replay

    @staticmethod
    def _official_stage_trace_index(
        trace_data: Mapping[str, Any],
        component_signature: Any,
        component: str,
    ) -> int:
        trace = trace_data.get("trace")
        if not isinstance(trace, Sequence) or isinstance(trace, str | bytes):
            raise TerminalAnalysisUnavailableError("official TraceData.trace is not a sequence")
        matches: list[int] = []
        for index, entry in enumerate(trace):
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise TerminalAnalysisUnavailableError("official TraceData entry is malformed")
            signature = getattr(entry[0], "signature", None)
            equals = getattr(signature, "equals", None)
            if callable(equals) and equals(component_signature):
                matches.append(index)
        if len(matches) != 1:
            raise TerminalAnalysisUnavailableError(
                f"the actual IFBench trajectory does not contain exactly one {component!r} "
                "invocation declared by the official two-stage program"
            )
        return matches[0]

    def _rollout(
        self,
        *,
        trace_data: Mapping[str, Any],
        trace_index: int,
        instance_index: Hashable,
        reward: Real,
        component: str,
        parent_skill: str,
        callback_calls: tuple[DSPyCallbackCall, ...],
    ) -> DSPyHistoricalRollout:
        try:
            lineage = self._lineage_resolver(
                trace_data=trace_data,
                trace_index=trace_index,
                callback_calls=callback_calls,
            )
            trajectory = self._provenance(
                trace_data=trace_data,
                trace_index=trace_index,
                lineage=lineage,
            )
        except (LineageUnavailableError, TokenProvenanceUnavailableError) as error:
            raise TerminalAnalysisUnavailableError(
                "selected official trajectory has no exact analyzable token provenance"
            ) from error
        predictor = lineage.predict_call.instance
        signature = getattr(predictor, "signature", None)
        demos = getattr(predictor, "demos", None)
        if signature is None or not isinstance(demos, list | tuple):
            raise TerminalAnalysisUnavailableError(
                "official traced predictor omitted its signature or demonstrations"
            )
        if getattr(signature, "instructions", None) != parent_skill:
            raise TerminalAnalysisUnavailableError(
                f"official trace for component {component!r} changed the parent instruction"
            )
        trace = trace_data["trace"]
        _, predictor_inputs, prediction = trace[trace_index]
        if not isinstance(predictor_inputs, Mapping):
            raise TerminalAnalysisUnavailableError("official traced predictor inputs are not a mapping")
        example = trace_data.get("example")
        score = float(reward)
        if not math.isfinite(score):
            raise TerminalAnalysisUnavailableError("official parent reward is not finite")
        layout = trajectory.canonical_token_layout
        nonterminal_ids = tuple(layout.nonterminal_token_ids)
        canonical_reasoning = self._tokenizer.decode(
            [nonterminal_ids[position] for position in layout.reasoning_positions],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        canonical_output = self._tokenizer.decode(
            [nonterminal_ids[position] for position in layout.output_positions],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not canonical_reasoning or not canonical_output:
            raise TerminalAnalysisUnavailableError(
                "canonical old rollout has an empty reasoning or output span"
            )
        input_field_tokens = _input_field_token_coordinates(
            tokenizer=self._tokenizer,
            trajectory=trajectory,
        )
        output_field_tokens: dict[str, tuple[_FieldTokenCoordinate, ...]] = {}
        if component == _IFBENCH_UPSTREAM_COMPONENT:
            items = getattr(prediction, "items", None)
            if not callable(items):
                raise TerminalAnalysisUnavailableError(
                    "official upstream Prediction is not a field mapping"
                )
            try:
                output_values = dict(items())
            except (TypeError, ValueError) as error:
                raise TerminalAnalysisUnavailableError(
                    "official upstream Prediction fields are malformed"
                ) from error
            edge_value = output_values.get(_IFBENCH_EDGE_OUTPUT_FIELD)
            if not isinstance(edge_value, str) or not edge_value:
                raise TerminalAnalysisUnavailableError(
                    "official upstream response field is not non-empty text"
                )
            output_ids = tuple(
                nonterminal_ids[position]
                for position in layout.output_positions
            )
            output_field_tokens[_IFBENCH_EDGE_OUTPUT_FIELD] = (
                _direct_field_token_coordinates(
                    tokenizer=self._tokenizer,
                    text=canonical_output,
                    token_ids=output_ids,
                    value=edge_value,
                    name="actual upstream response field",
                    position_map=tuple(layout.output_positions),
                )
            )
        token_record = DSPyTokenRecord(
            prompt_text=trajectory.canonical_prompt_text,
            completion_text=trajectory.canonical_completion_text,
            native_reasoning_text=canonical_reasoning,
            native_output_text=canonical_output,
            prompt_token_ids=trajectory.canonical_prompt_token_ids,
            completion_token_ids=trajectory.canonical_completion_token_ids,
            token_layout=trajectory.canonical_token_layout,
        )
        return DSPyHistoricalRollout(
            instance_index=instance_index,
            reward=score,
            component=component,
            skill_text=parent_skill,
            signature=signature,
            demos=tuple(demos),
            predictor_inputs=dict(predictor_inputs),
            input_field_tokens=input_field_tokens,
            output_field_tokens=output_field_tokens,
            token_record=token_record,
            provenance=trajectory,
        )

    @staticmethod
    def _raw_reward(reward: Real) -> float:
        if isinstance(reward, bool) or not isinstance(reward, Real):
            raise TypeError("the official reward must be numeric")
        value = float(reward)
        if not math.isfinite(value):
            raise TerminalAnalysisUnavailableError("the official reward must be finite")
        return value

    def _credit(
        self,
        rollout: DSPyHistoricalRollout,
        advantage: float,
        capture_lease: Any,
        sink_weights: tuple[float, ...] | None = None,
    ) -> FlashTraceCredit:
        record = rollout.token_record
        key = (
            record.prompt_text,
            record.prompt_token_ids,
            record.completion_text,
            record.completion_token_ids,
            record.native_reasoning_text,
            record.native_output_text,
            advantage,
            sink_weights,
        )
        cached = self._credit_cache.get(key)
        if cached is not None:
            return cached
        weighted_scope = (
            self._tracer.weighted_sink_scope(sink_weights)
            if sink_weights is not None
            else nullcontext()
        )
        with weighted_scope:
            credit = build_flashtrace_credit(
                tracer=self._tracer,
                tokenizer=self._tokenizer,
                prompt_text=record.prompt_text,
                prompt_token_ids=record.prompt_token_ids,
                completion_text=record.completion_text,
                token_layout=record.token_layout,
                native_reasoning_text=record.native_reasoning_text,
                native_output_text=record.native_output_text,
                advantage=advantage,
                capture_lease=capture_lease,
            )
        if sink_weights is not None:
            credit = _ground_answer_credit(credit, sink_weights)
        self._credit_cache[key] = credit
        return credit

    def _local_old_measure(
        self,
        *,
        rollout: DSPyHistoricalRollout,
        capture_lease: Any,
        sink_weights: tuple[float, ...] | None = None,
    ) -> tuple[HistoricalRolloutTarget, DependencyMeasure]:
        inputs = self.dependency_inputs(
            tokenizer=self._tokenizer,
            chat_adapter=self._chat_adapter,
            rollout=rollout,
            parent_skill=rollout.skill_text,
            candidate_skill=rollout.skill_text,
        )
        weighted_scope = (
            self._attributor.weighted_sink_scope(sink_weights)
            if sink_weights is not None
            else nullcontext()
        )
        with weighted_scope:
            measure = build_dependency_measure(
                attributor=self._attributor,
                tokenizer=self._tokenizer,
                instance_id=inputs.instance_id,
                prompt=inputs.old_prompt,
                target=inputs.target,
                capture_lease=capture_lease,
            )
        return inputs.target, measure

    def _prepare_parent_replay(
        self,
        *,
        component: str,
        paths: tuple[tuple[DSPyHistoricalRollout, DSPyHistoricalRollout], ...],
    ) -> PreparedParentReplay:
        """Prepare only the officially selected component's replay utility.

        A downstream mutation uses raw ``R`` once.  An upstream mutation uses
        per-instance ``R * rho`` before b03's confirmed minibatch
        normalization, so ``rho`` changes relative instance weighting rather
        than absolute global potential strength.  The two unit-normalized b02
        credits are never combined in one proposal task.
        """

        credits: list[tuple[DSPyHistoricalRollout, FlashTraceCredit]] = []
        prepared_paths: list[_PreparedIFBenchPath] = []
        cross_measures: list[_CrossDependencyMeasure] = []
        for upstream, downstream in paths:
            reward = self._raw_reward(downstream.reward)
            with shared_exact_token_capture(self._model) as downstream_capture:
                downstream_target, downstream_measure = self._local_old_measure(
                    rollout=downstream,
                    capture_lease=downstream_capture,
                )
                if component == _IFBENCH_DOWNSTREAM_COMPONENT and reward != 0.0:
                    credits.append(
                        (downstream, self._credit(downstream, reward, downstream_capture))
                    )

            upstream_target = _historical_target(upstream)
            response_weights = _response_sink_weights(
                downstream_measure=downstream_measure,
                downstream_rollout=downstream,
                upstream=upstream,
                upstream_target=upstream_target,
            )
            downstream_total = float(downstream_measure.total_mass)
            response_mass = math.fsum(response_weights)
            if downstream_total < 0.0 or response_mass < 0.0:
                raise TerminalAnalysisUnavailableError(
                    "downstream dependency mass is negative"
                )
            rho = response_mass / downstream_total if downstream_total > 0.0 else 0.0
            if not math.isfinite(rho) or rho < 0.0 or rho > 1.0:
                raise TerminalAnalysisUnavailableError(
                    "downstream response path share lies outside [0, 1]"
                )
            if component == _IFBENCH_UPSTREAM_COMPONENT and rho == 0.0:
                raise TerminalAnalysisUnavailableError(
                    "the selected upstream component has zero terminal response path mass"
                )
            propagated_reward = (
                reward * rho
            )

            upstream_measure: DependencyMeasure | None = None
            if response_mass > 0.0:
                with shared_exact_token_capture(self._model) as upstream_capture:
                    measured_target, upstream_measure = self._local_old_measure(
                        rollout=upstream,
                        capture_lease=upstream_capture,
                        sink_weights=response_weights,
                    )
                    if measured_target != upstream_target:
                        raise TerminalAnalysisUnavailableError(
                            "weighted upstream dependency changed the old rollout target"
                        )
                    if component == _IFBENCH_UPSTREAM_COMPONENT and propagated_reward != 0.0:
                        credits.append(
                            (
                                upstream,
                                self._credit(
                                    upstream,
                                    propagated_reward,
                                    upstream_capture,
                                    sink_weights=response_weights,
                                ),
                            )
                        )

            prepared_path = _PreparedIFBenchPath(
                upstream=upstream,
                downstream=downstream,
                upstream_target=upstream_target,
                downstream_target=downstream_target,
                upstream_old_measure=upstream_measure,
                downstream_old_measure=downstream_measure,
            )
            prepared_paths.append(prepared_path)
            cross_measures.append(
                _compose_ifbench_dependency(
                    upstream=upstream_measure,
                    downstream=downstream_measure,
                    upstream_rollout=upstream,
                    downstream_rollout=downstream,
                )
            )
        if not credits:
            raise TerminalAnalysisUnavailableError(
                "the official raw reward sends no non-zero credit to the selected component"
            )
        return PreparedParentReplay(
            rollout_credits=tuple(credits),
            paths=tuple(prepared_paths),
            old_cross_measures=tuple(cross_measures),
        )

    def _candidate_measure(
        self,
        *,
        rollout: DSPyHistoricalRollout,
        candidate_skill: str,
        expected_target: HistoricalRolloutTarget,
        sink_weights: tuple[float, ...] | None,
    ) -> DependencyMeasure:
        inputs = self.dependency_inputs(
            tokenizer=self._tokenizer,
            chat_adapter=self._chat_adapter,
            rollout=rollout,
            parent_skill=rollout.skill_text,
            candidate_skill=candidate_skill,
        )
        if inputs.target != expected_target:
            raise TerminalAnalysisUnavailableError(
                "candidate dependency replay changed the canonical old rollout"
            )
        weighted_scope = (
            self._attributor.weighted_sink_scope(sink_weights)
            if sink_weights is not None
            else nullcontext()
        )
        with weighted_scope:
            return build_dependency_measure(
                attributor=self._attributor,
                tokenizer=self._tokenizer,
                instance_id=inputs.instance_id,
                prompt=inputs.candidate_prompt,
                target=inputs.target,
                capture_lease=None,
            )

    def candidate_dependency_distance(
        self,
        *,
        candidate_program: Mapping[str, str],
        prepared: PreparedParentReplay,
    ) -> float:
        expected_components = {
            _IFBENCH_UPSTREAM_COMPONENT,
            _IFBENCH_DOWNSTREAM_COMPONENT,
        }
        if set(candidate_program) != expected_components:
            raise TerminalAnalysisUnavailableError(
                "candidate dependency requires the complete official IFBench program"
            )
        candidate_cross: list[_CrossDependencyMeasure] = []
        for path in prepared.paths:
            # The candidate is one complete child of the sampled parent. Each
            # heterogeneous admission reference remains only the fixed old
            # trajectory; never splice the mutation into that reference.
            downstream_measure = self._candidate_measure(
                rollout=path.downstream,
                candidate_skill=candidate_program[_IFBENCH_DOWNSTREAM_COMPONENT],
                expected_target=path.downstream_target,
                sink_weights=None,
            )
            response_weights = _response_sink_weights(
                downstream_measure=downstream_measure,
                downstream_rollout=path.downstream,
                upstream=path.upstream,
                upstream_target=path.upstream_target,
            )
            changed_upstream = (
                self._candidate_measure(
                    rollout=path.upstream,
                    candidate_skill=candidate_program[_IFBENCH_UPSTREAM_COMPONENT],
                    expected_target=path.upstream_target,
                    sink_weights=response_weights,
                )
                if math.fsum(response_weights) > 0.0
                else None
            )
            candidate_cross.append(
                _compose_ifbench_dependency(
                    upstream=changed_upstream,
                    downstream=downstream_measure,
                    upstream_rollout=path.upstream,
                    downstream_rollout=path.downstream,
                )
            )
        return _batch_cross_dependency_distance(
            prepared.old_cross_measures,
            candidate_cross,
        )

    def prepare(
        self,
        *,
        component: str,
        references: tuple[DSPyReferenceObservation, ...],
        known_candidates: frozenset[CandidateKey],
    ) -> "DSPyPreparedTerminalAnalysis":
        expected_components = {
            _IFBENCH_UPSTREAM_COMPONENT,
            _IFBENCH_DOWNSTREAM_COMPONENT,
        }
        if component not in expected_components:
            raise TerminalAnalysisUnavailableError(
                "terminal analysis requires the official IFBench two-stage components"
            )
        if not references:
            raise TerminalAnalysisUnavailableError(
                "admission selected no complete IFBench reference observations"
            )

        paths: list[tuple[DSPyHistoricalRollout, DSPyHistoricalRollout]] = []
        cache_key: tuple[Any, ...] = (
            component,
            tuple(
                (
                    reference.program_idx,
                    reference.instance_id,
                    id(reference.fact),
                    reference.fact.score,
                    _candidate_key(reference.candidate),
                )
                for reference in references
            ),
        )
        cached = self._prepared_cache.get(cache_key)
        if cached is not None:
            return DSPyPreparedTerminalAnalysis(
                component=component,
                known_candidates=known_candidates,
                builder=self,
                replay=self._replay,
                prepared=cached,
            )

        for reference in references:
            if set(reference.candidate) != expected_components:
                raise TerminalAnalysisUnavailableError(
                    "an admission reference omitted an official IFBench stage"
                )
            if set(reference.stage_signatures) != expected_components:
                raise TerminalAnalysisUnavailableError(
                    "an admission reference has incomplete official stage signatures"
                )
            trace_data = reference.fact.trajectory
            upstream_index = self._official_stage_trace_index(
                trace_data,
                reference.stage_signatures[_IFBENCH_UPSTREAM_COMPONENT],
                _IFBENCH_UPSTREAM_COMPONENT,
            )
            downstream_index = self._official_stage_trace_index(
                trace_data,
                reference.stage_signatures[_IFBENCH_DOWNSTREAM_COMPONENT],
                _IFBENCH_DOWNSTREAM_COMPONENT,
            )
            upstream = self._rollout(
                trace_data=trace_data,
                trace_index=upstream_index,
                instance_index=reference.instance_id,
                reward=reference.fact.score,
                component=_IFBENCH_UPSTREAM_COMPONENT,
                parent_skill=reference.candidate[_IFBENCH_UPSTREAM_COMPONENT],
                callback_calls=reference.fact.callback_calls,
            )
            downstream = self._rollout(
                trace_data=trace_data,
                trace_index=downstream_index,
                instance_index=reference.instance_id,
                reward=reference.fact.score,
                component=_IFBENCH_DOWNSTREAM_COMPONENT,
                parent_skill=reference.candidate[_IFBENCH_DOWNSTREAM_COMPONENT],
                callback_calls=reference.fact.callback_calls,
            )
            trace = trace_data["trace"]
            upstream_prediction = trace[upstream_index][2]
            downstream_inputs = trace[downstream_index][1]
            downstream_prediction = trace[downstream_index][2]
            program_prediction = trace_data.get("prediction")
            try:
                upstream_value = upstream_prediction[_IFBENCH_EDGE_OUTPUT_FIELD]
                downstream_value = downstream_inputs[_IFBENCH_EDGE_OUTPUT_FIELD]
                terminal_value = downstream_prediction[_IFBENCH_TERMINAL_OUTPUT_FIELD]
                program_value = program_prediction[_IFBENCH_PROGRAM_OUTPUT_FIELD]
            except (KeyError, TypeError) as error:
                raise TerminalAnalysisUnavailableError(
                    "actual IFBench trajectory omitted a declared program edge"
                ) from error
            if (
                not isinstance(upstream_value, str)
                or not upstream_value
                or upstream_value != downstream_value
                or not isinstance(terminal_value, str)
                or not terminal_value
                or terminal_value != program_value
            ):
                raise TerminalAnalysisUnavailableError(
                    "actual IFBench values disagree with the official two-stage program edge"
                )
            upstream_query = upstream.predictor_inputs.get(_IFBENCH_QUERY_FIELD)
            downstream_query = downstream.predictor_inputs.get(_IFBENCH_QUERY_FIELD)
            if (
                not isinstance(upstream_query, str)
                or not upstream_query
                or upstream_query != downstream_query
            ):
                raise TerminalAnalysisUnavailableError(
                    "actual IFBench stages disagree on the shared query input"
                )
            paths.append((upstream, downstream))

        if len(paths) != len(references):
            raise TerminalAnalysisUnavailableError(
                "one or more admission references lacks a complete two-stage path"
            )
        prepared = self._prepare_parent_replay(
            component=component,
            paths=tuple(paths),
        )
        self._prepared_cache[cache_key] = prepared
        return DSPyPreparedTerminalAnalysis(
            component=component,
            known_candidates=known_candidates,
            builder=self,
            replay=self._replay,
            prepared=prepared,
        )

    def dependency_inputs(
        self,
        *,
        tokenizer: Any,
        chat_adapter: ChatAdapter,
        rollout: DSPyHistoricalRollout,
        parent_skill: str,
        candidate_skill: str,
    ) -> DSPyReplayDependencyInputs:
        if tokenizer is not self._tokenizer or chat_adapter is not self._chat_adapter:
            raise RuntimeError("dependency replay changed the shared tokenizer or ChatAdapter")
        if rollout.skill_text != parent_skill:
            raise TerminalAnalysisUnavailableError("old rollout belongs to a different parent skill")
        trajectory = rollout.provenance
        old_prompt = _rendered_prompt_from_coordinates(
            text=trajectory.canonical_prompt_text,
            token_ids=trajectory.canonical_prompt_token_ids,
            coordinates=trajectory.canonical_prompt_coordinates,
            skill_character_span=trajectory.canonical_skill_character_span,
            task_character_spans=trajectory.canonical_task_observation_character_spans,
            tokenizer=tokenizer,
        )
        candidate_prompt = (
            old_prompt
            if candidate_skill == parent_skill
            else _candidate_prompt(
                tokenizer=tokenizer,
                chat_adapter=chat_adapter,
                chat_template_kwargs=self._chat_template_kwargs,
                rollout=rollout,
                candidate_skill=candidate_skill,
            )
        )
        return DSPyReplayDependencyInputs(
            instance_id=rollout.instance_index,
            old_prompt=old_prompt,
            candidate_prompt=candidate_prompt,
            target=_historical_target(rollout),
        )


class DSPyPreparedTerminalAnalysis(PreparedTerminalAnalysis):
    def __init__(
        self,
        *,
        component: str,
        known_candidates: frozenset[CandidateKey],
        builder: DSPyParentAnalysisBuilder,
        replay: TokenReplayUtility,
        prepared: PreparedParentReplay,
    ) -> None:
        self._component = component
        self._known_candidates = known_candidates
        self._builder = builder
        self._replay = replay
        self._prepared = prepared

    def is_known_candidate(self, candidate: Mapping[str, str]) -> bool:
        return _candidate_key(candidate) in self._known_candidates

    def teacher_forcing_scores(
        self,
        *,
        component: str,
        candidate_programs: tuple[Mapping[str, str], ...],
    ) -> tuple[float, ...]:
        if component != self._component:
            raise TerminalAnalysisUnavailableError("terminal analysis component changed")
        if not candidate_programs:
            raise TypeError("candidate_programs must be non-empty")
        try:
            candidate_texts = tuple(candidate[component] for candidate in candidate_programs)
        except (KeyError, TypeError) as error:
            raise TerminalAnalysisUnavailableError(
                "a terminal candidate omitted the selected official component"
            ) from error
        if any(not isinstance(text, str) for text in candidate_texts):
            raise TypeError("candidate component texts must be strings")
        # The omitted old-skill likelihood is identical for every candidate,
        # so the existing batched scorer preserves the exact U(v) ordering
        # without re-formatting the captured old prompt.
        return self._replay.scores(
            candidate_texts,
            self._prepared.rollout_credits,
        )

    def dependency_distance(
        self,
        *,
        component: str,
        candidate_program: Mapping[str, str],
    ) -> float:
        if component != self._component:
            raise TerminalAnalysisUnavailableError("terminal analysis component changed")
        return self._builder.candidate_dependency_distance(
            candidate_program=candidate_program,
            prepared=self._prepared,
        )


@dataclass(frozen=True, slots=True)
class _BoundAnalysis:
    parent_key: CandidateKey
    component: str
    reflective_dataset: object
    future: Future[DSPyPreparedTerminalAnalysis]


class _AbstainingPreparedTerminalAnalysis(PreparedTerminalAnalysis):
    """Fail-closed result for a parent minibatch that was not fully analyzable."""

    def is_known_candidate(self, candidate: Mapping[str, str]) -> bool:
        _candidate_key(candidate)
        return True

    def teacher_forcing_scores(
        self,
        *,
        component: str,
        candidate_programs: tuple[Mapping[str, str], ...],
    ) -> tuple[float, ...]:
        del component, candidate_programs
        raise TerminalAnalysisUnavailableError("the selected parent minibatch abstained")

    def dependency_distance(
        self,
        *,
        component: str,
        candidate_program: Mapping[str, str],
    ) -> float:
        del component, candidate_program
        raise TerminalAnalysisUnavailableError("the selected parent minibatch abstained")


class DSPyTerminalAnalysisBridge:
    """Provider bound to one exact official reflection/admission job.

    The prepared reference analysis is immutable and may be read by every proposal
    attempt that GEPA makes for the same reflective-dataset object.  GEPA, not
    this bridge, owns retry policy.
    """

    def __init__(self, *, builder: DSPyParentAnalysisBuilder, executor: Executor) -> None:
        if not isinstance(builder, DSPyParentAnalysisBuilder):
            raise TypeError("builder must be DSPyParentAnalysisBuilder")
        if not callable(getattr(executor, "submit", None)):
            raise TypeError("executor must expose submit")
        self._builder = builder
        self._executor = executor
        self._lock = threading.RLock()
        self._pool: frozenset[CandidateKey] | None = None
        self._bound: _BoundAnalysis | None = None

    def update_candidate_pool(self, candidates: Sequence[Mapping[str, str]]) -> None:
        pool = frozenset(_candidate_key(candidate) for candidate in candidates)
        if len(pool) != len(candidates):
            raise RuntimeError("the official GEPA candidate pool contains duplicate programs")
        with self._lock:
            stale = self._bound
            self._bound = None
            self._pool = pool
        if stale is not None:
            stale.future.cancel()

    def bind_admission_references(
        self,
        *,
        parent_candidate: Mapping[str, str],
        component: str,
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        references: tuple[DSPyReferenceObservation, ...],
    ) -> None:
        parent_key = _candidate_key(parent_candidate)
        with self._lock:
            if self._bound is not None:
                raise RuntimeError("a terminal parent analysis is already bound")
            if self._pool is None:
                raise RuntimeError("candidate pool must be observed before parent selection")
            known = self._pool
            future = self._executor.submit(
                self._builder.prepare,
                component=component,
                references=references,
                known_candidates=known,
            )
            self._bound = _BoundAnalysis(
                parent_key,
                component,
                reflective_dataset,
                future,
            )

    def take(
        self,
        *,
        parent_candidate: Mapping[str, str],
        component: str,
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> PreparedTerminalAnalysis:
        parent_key = _candidate_key(parent_candidate)
        with self._lock:
            bound = self._bound
            if bound is None:
                raise TerminalAnalysisUnavailableError("no parent analysis is bound")
            if bound.parent_key != parent_key or bound.component != component:
                raise TerminalAnalysisUnavailableError(
                    "bound terminal analysis belongs to a different parent/component"
                )
            if bound.reflective_dataset is not reflective_dataset:
                raise TerminalAnalysisUnavailableError(
                    "bound terminal analysis belongs to a different official reflection job"
                )
        try:
            return bound.future.result()
        except TerminalAnalysisUnavailableError:
            return _AbstainingPreparedTerminalAnalysis()

    def discard(
        self,
        *,
        parent_candidate: Mapping[str, str],
        component: str,
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> None:
        parent_key = _candidate_key(parent_candidate)
        with self._lock:
            bound = self._bound
            if bound is None:
                return
            if bound.parent_key != parent_key or bound.component != component:
                raise TerminalAnalysisUnavailableError(
                    "cannot discard terminal analysis for a different parent/component"
                )
            if bound.reflective_dataset is not reflective_dataset:
                raise TerminalAnalysisUnavailableError(
                    "cannot discard terminal analysis for a different official reflection job"
                )
            self._bound = None
        bound.future.cancel()

    def clear(self) -> None:
        with self._lock:
            bound = self._bound
            self._bound = None
        if bound is not None:
            bound.future.cancel()


class DSPyTerminalAnalysisCaptureMixin:
    """Delegate official adapter methods while observing the exact parent trajectory."""

    @staticmethod
    def _restore_failed_trace_slots(
        *,
        batch: Sequence[Any],
        result: Any,
        failure_score: float,
    ) -> Any:
        """Restore failures dropped by official ``bootstrap_trace_data``.

        DSPy's trace bootstrap retains the original ``example_ind`` on every
        successful item but omits an item when its prediction cannot be
        unpacked.  GEPA's adapter contract still requires one output, score and
        trajectory per input.  Preserve every retained official value and fill
        only those omitted indices with the configured official failure score.
        """

        trajectories = getattr(result, "trajectories", None)
        if trajectories is None or len(trajectories) == len(batch):
            return result

        outputs = list(getattr(result, "outputs", ()))
        scores = list(getattr(result, "scores", ()))
        trajectories = list(trajectories)
        objective_scores = getattr(result, "objective_scores", None)
        if len(outputs) != len(trajectories) or len(scores) != len(trajectories):
            raise RuntimeError("official DSPy trace result vectors are internally misaligned")
        if len(trajectories) > len(batch):
            raise RuntimeError("official DSPy returned more traces than minibatch instances")
        if objective_scores is not None and len(objective_scores) != len(trajectories):
            raise RuntimeError("official DSPy objective scores are not trace-aligned")

        aligned_outputs: list[Any | None] = [None] * len(batch)
        aligned_scores: list[float | None] = [None] * len(batch)
        aligned_trajectories: list[Any | None] = [None] * len(batch)
        aligned_objectives: list[Any | None] | None = (
            [None] * len(batch) if objective_scores is not None else None
        )

        for result_index, trace_data in enumerate(trajectories):
            if not isinstance(trace_data, Mapping):
                raise RuntimeError("official DSPy trajectory is not a mapping")
            example_index = trace_data.get("example_ind")
            if (
                isinstance(example_index, bool)
                or not isinstance(example_index, int)
                or not 0 <= example_index < len(batch)
            ):
                raise RuntimeError("official DSPy trajectory has an invalid example index")
            if aligned_trajectories[example_index] is not None:
                raise RuntimeError("official DSPy returned duplicate traces for one instance")
            if trace_data.get("example") is not batch[example_index]:
                raise RuntimeError("official DSPy trajectory no longer identifies its input instance")
            if trace_data.get("prediction") is not outputs[result_index]:
                raise RuntimeError("official DSPy output no longer identifies its trajectory")
            aligned_outputs[example_index] = outputs[result_index]
            aligned_scores[example_index] = scores[result_index]
            aligned_trajectories[example_index] = trace_data
            if aligned_objectives is not None:
                aligned_objectives[example_index] = objective_scores[result_index]

        for example_index, trajectory in enumerate(aligned_trajectories):
            if trajectory is not None:
                continue
            failed_prediction = dspy.Prediction()
            aligned_outputs[example_index] = failed_prediction
            aligned_scores[example_index] = float(failure_score)
            aligned_trajectories[example_index] = {
                "example_ind": example_index,
                "example": batch[example_index],
                "prediction": failed_prediction,
                "trace": [],
                "score": float(failure_score),
            }
            if aligned_objectives is not None:
                aligned_objectives[example_index] = {}

        return replace(
            result,
            outputs=aligned_outputs,
            scores=aligned_scores,
            trajectories=aligned_trajectories,
            objective_scores=aligned_objectives,
        )

    def install_terminal_analysis_bridge(self, bridge: DSPyTerminalAnalysisBridge) -> None:
        if not isinstance(bridge, DSPyTerminalAnalysisBridge):
            raise TypeError("bridge must be DSPyTerminalAnalysisBridge")
        if getattr(self, "_terminal_analysis_bridge", None) is not None:
            raise RuntimeError("terminal analysis bridge is already installed")
        self._terminal_analysis_bridge = bridge
        self._terminal_observation_facts: dict[
            tuple[int, Hashable], DSPyObservationFact
        ] = {}

    def evaluate(self, batch: Any, candidate: Any, capture_traces: bool = False) -> Any:
        with bounded_dspy_straggler_resubmission():
            bridge = getattr(self, "_terminal_analysis_bridge", None)
            if not capture_traces or bridge is None:
                return super().evaluate(batch, candidate, capture_traces)
            collector = DSPyCallbackCollector()
            configured = tuple(getattr(dspy.settings, "callbacks", ()) or ())
            callbacks = list(configured)
            if all(callback is not collector for callback in callbacks):
                callbacks.append(collector)
            with dspy.context(callbacks=callbacks):
                result = super().evaluate(batch, candidate, capture_traces)
            result = self._restore_failed_trace_slots(
                batch=batch,
                result=result,
                failure_score=self.failure_score,
            )
            callback_calls = collector.completed_snapshot()
            trajectories = tuple(getattr(result, "trajectories", ()) or ())
            enriched_trajectories: list[Any] = []
            for trajectory in trajectories:
                if not isinstance(trajectory, Mapping):
                    raise RuntimeError("official DSPy trajectory is not a mapping")
                enriched = dict(trajectory)
                enriched["_flashtrace_callback_calls"] = callback_calls
                enriched_trajectories.append(enriched)
            return replace(
                result,
                trajectories=enriched_trajectories,
            )

    def commit_program_observations(
        self,
        *,
        program_idx: int,
        evaluation_ids: Sequence[Hashable],
        evaluation: Any,
        committed_ids: Sequence[Hashable],
    ) -> None:
        """Persist only observations confirmed by official GEPA state."""

        ids = tuple(evaluation_ids)
        outputs = tuple(getattr(evaluation, "outputs", ()) or ())
        scores = tuple(getattr(evaluation, "scores", ()) or ())
        trajectories = tuple(getattr(evaluation, "trajectories", ()) or ())
        if not (len(ids) == len(outputs) == len(scores) == len(trajectories)):
            raise RuntimeError("official observation vectors are not instance-aligned")
        committed = set(committed_ids)
        unknown = committed.difference(ids)
        if unknown:
            raise RuntimeError("official state committed an ID outside the evaluated batch")
        facts = getattr(self, "_terminal_observation_facts", None)
        if facts is None:
            raise RuntimeError("terminal analysis bridge is not installed")
        for instance_id, output, raw_score, trajectory in zip(
            ids, outputs, scores, trajectories, strict=True
        ):
            if instance_id not in committed:
                continue
            if not isinstance(trajectory, Mapping):
                raise RuntimeError("official DSPy trajectory is not a mapping")
            callback_calls = trajectory.get("_flashtrace_callback_calls")
            if not isinstance(callback_calls, tuple):
                raise RuntimeError("official trajectory omitted its actual LM callback capture")
            score = float(raw_score)
            if not math.isfinite(score):
                raise RuntimeError("official observation reward is not finite")
            clean_trajectory = dict(trajectory)
            clean_trajectory.pop("_flashtrace_callback_calls", None)
            key = (int(program_idx), instance_id)
            previous = facts.get(key)
            if previous is not None and score <= previous.score:
                continue
            facts[key] = DSPyObservationFact(
                score=score,
                output=output,
                trajectory=clean_trajectory,
                callback_calls=callback_calls,
            )

    def get_program_observation(
        self,
        program_idx: int,
        instance_id: Hashable,
    ) -> DSPyObservationFact | None:
        facts = getattr(self, "_terminal_observation_facts", None)
        if facts is None:
            raise RuntimeError("terminal analysis bridge is not installed")
        return facts.get((int(program_idx), instance_id))

    def get_adapter_state(self) -> dict[str, Any]:
        base_getter = getattr(super(), "get_adapter_state", None)
        state = dict(base_getter()) if callable(base_getter) else {}
        facts = getattr(self, "_terminal_observation_facts", None)
        if facts is None:
            raise RuntimeError("terminal analysis bridge is not installed")
        state["flashtrace_admission"] = {
            "schema_version": 1,
            "observation_facts": dict(facts),
        }
        return state

    def set_adapter_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("adapter state must be a mapping")
        copied = dict(state)
        payload = copied.pop("flashtrace_admission", None)
        base_setter = getattr(super(), "set_adapter_state", None)
        if callable(base_setter):
            base_setter(copied)
        if payload is None:
            self._terminal_observation_facts = {}
            return
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise RuntimeError("unsupported FlashTrace admission adapter-state schema")
        raw_facts = payload.get("observation_facts")
        if not isinstance(raw_facts, Mapping):
            raise RuntimeError("FlashTrace admission facts are not a mapping")
        facts: dict[tuple[int, Hashable], DSPyObservationFact] = {}
        for key, fact in raw_facts.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or isinstance(key[0], bool)
                or not isinstance(key[0], int)
                or not isinstance(fact, DSPyObservationFact)
            ):
                raise RuntimeError("FlashTrace admission fact entry is malformed")
            facts[(key[0], key[1])] = fact
        self._terminal_observation_facts = facts

    def make_reflective_dataset(
        self,
        candidate: Mapping[str, str],
        eval_batch: Any,
        components_to_update: Sequence[str],
    ) -> Any:
        """Preserve the official DSPy reflective-dataset behavior verbatim."""

        return super().make_reflective_dataset(
            candidate,
            eval_batch,
            components_to_update,
        )


__all__ = [
    "DSPyHistoricalRollout",
    "DSPyObservationFact",
    "DSPyParentAnalysisBuilder",
    "DSPyPreparedTerminalAnalysis",
    "DSPyReferenceObservation",
    "DSPyReplayDependencyInputs",
    "DSPyTerminalAnalysisBridge",
    "DSPyTerminalAnalysisCaptureMixin",
]
