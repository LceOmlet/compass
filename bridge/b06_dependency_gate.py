from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from .b10_flashtrace_offload import OffloadedCaptureLease
from .b14_flashtrace_token_ids import (
    ExactTokenOffloadedLLMIFRAttribution,
    token_surfaces,
)


class DependencyMeasureUnavailableError(RuntimeError):
    """The bridge cannot derive an exact dependency measure from valid official data."""


@dataclass(frozen=True, slots=True)
class CharacterSpan:
    """Half-open character span in one exact rendered prompt."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class TaskObservationField:
    """One stable task/observation source field in a rendered prompt."""

    field_id: str
    span: CharacterSpan


@dataclass(frozen=True, slots=True)
class RenderedPromptProvenance:
    """Exact rendered prompt plus its direct source character spans."""

    text: str
    token_ids: tuple[int, ...]
    task_observation_fields: tuple[TaskObservationField, ...]
    skill_spans: tuple[CharacterSpan, ...]


@dataclass(frozen=True, slots=True)
class HistoricalRolloutTarget:
    """Exact raw non-EOS completion and parsed inclusive token spans."""

    text: str
    token_ids: tuple[int, ...]
    reasoning_span: tuple[int, int]
    output_span: tuple[int, int]


@dataclass(frozen=True, slots=True)
class FieldTokenMeasure:
    """Attribution mass on field-local task/observation token coordinates."""

    field_id: str
    token_ids: tuple[int, ...]
    masses: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DependencyMeasure:
    """Non-negative finite FlashTrace measure on stable source coordinates."""

    instance_id: Hashable
    task_observation: tuple[FieldTokenMeasure, ...]
    skill_mass: float
    format_mass: float
    target_token_ids: tuple[int, ...]
    reasoning_span: tuple[int, int]
    output_span: tuple[int, int]
    attributor_identity: int

    @property
    def total_mass(self) -> float:
        return math.fsum(
            (
                *(mass for field in self.task_observation for mass in field.masses),
                self.skill_mass,
                self.format_mass,
            )
        )


def _integer_ids(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in result):
        raise TypeError(f"{name} must contain integer token IDs")
    if any(int(value) < 0 for value in result):
        raise ValueError(f"{name} must contain non-negative token IDs")
    return tuple(int(value) for value in result)


def _encode_with_offsets(
    tokenizer: Any,
    text: str,
    name: str,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = _integer_ids(encoded["input_ids"], f"{name} input_ids")
    offsets = tuple((int(start), int(end)) for start, end in encoded["offset_mapping"])
    if len(ids) != len(offsets):
        raise RuntimeError(f"{name} token IDs and offsets have different lengths")
    if any(start < 0 or end <= start or end > len(text) for start, end in offsets):
        raise RuntimeError(f"{name} has a non-direct token offset")
    decoded = tokenizer.decode(
        list(ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != text:
        raise RuntimeError(f"{name} failed the exact token-ID round trip")
    return ids, offsets


def _validated_span(span: CharacterSpan, text: str, name: str) -> CharacterSpan:
    if not isinstance(span, CharacterSpan):
        raise TypeError(f"{name} must be a CharacterSpan")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (span.start, span.end)):
        raise TypeError(f"{name} boundaries must be integers")
    start, end = int(span.start), int(span.end)
    if start < 0 or end < start or end > len(text):
        raise ValueError(f"{name} lies outside the rendered prompt")
    return CharacterSpan(start, end)


def _validated_provenance(
    prompt: RenderedPromptProvenance,
) -> tuple[tuple[TaskObservationField, ...], tuple[CharacterSpan, ...]]:
    if not isinstance(prompt, RenderedPromptProvenance):
        raise TypeError("prompt must be RenderedPromptProvenance")
    if not isinstance(prompt.text, str) or not prompt.text:
        raise ValueError("the exact rendered prompt must be non-empty text")

    fields: list[TaskObservationField] = []
    field_ids: set[str] = set()
    explicit: list[tuple[str, CharacterSpan]] = []
    for index, field in enumerate(prompt.task_observation_fields):
        if not isinstance(field, TaskObservationField):
            raise TypeError("task_observation_fields must contain TaskObservationField values")
        if not isinstance(field.field_id, str) or not field.field_id:
            raise ValueError("each task/observation field_id must be non-empty text")
        if field.field_id in field_ids:
            raise ValueError(f"duplicate task/observation field_id: {field.field_id!r}")
        field_ids.add(field.field_id)
        span = _validated_span(field.span, prompt.text, f"task/observation span {index}")
        fields.append(TaskObservationField(field.field_id, span))
        if span.start != span.end:
            explicit.append((f"task/observation {field.field_id!r}", span))

    skill_spans: list[CharacterSpan] = []
    for index, raw_span in enumerate(prompt.skill_spans):
        span = _validated_span(raw_span, prompt.text, f"skill span {index}")
        skill_spans.append(span)
        if span.start != span.end:
            explicit.append((f"skill span {index}", span))

    for left_index, (left_name, left) in enumerate(explicit):
        for right_name, right in explicit[left_index + 1 :]:
            if left.start < right.end and right.start < left.end:
                raise ValueError(f"overlapping provenance spans: {left_name} and {right_name}")
    return tuple(fields), tuple(skill_spans)


def _validated_target(
    tokenizer: Any,
    target: HistoricalRolloutTarget,
) -> tuple[tuple[int, ...], tuple[int, int], tuple[int, int], tuple[str, ...]]:
    if not isinstance(target, HistoricalRolloutTarget):
        raise TypeError("target must be HistoricalRolloutTarget")
    if not isinstance(target.text, str) or not target.text:
        raise ValueError("the historical raw completion must be non-empty text")
    expected_ids = _integer_ids(target.token_ids, "historical target token_ids")
    if not expected_ids:
        raise ValueError("the historical target must contain tokens")

    decoded_target = tokenizer.decode(
        list(expected_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded_target != target.text:
        raise RuntimeError("historical target text differs from its exact captured token IDs")

    reasoning_span = tuple(target.reasoning_span)
    output_span = tuple(target.output_span)
    if len(reasoning_span) != 2 or len(output_span) != 2:
        raise ValueError("reasoning_span and output_span must each contain two indices")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in reasoning_span + output_span
    ):
        raise TypeError("reasoning_span and output_span indices must be integers")
    reasoning = (int(reasoning_span[0]), int(reasoning_span[1]))
    output = (int(output_span[0]), int(output_span[1]))
    if (
        reasoning[0] < 0
        or reasoning[1] < reasoning[0]
        or reasoning[1] >= len(expected_ids)
    ):
        raise ValueError("reasoning_span must be non-empty and lie within the raw completion")
    if (
        output[0] < 0
        or output[1] < output[0]
        or output[1] >= len(expected_ids)
    ):
        raise ValueError("output_span must be non-empty and lie within the raw completion")
    if reasoning[1] >= output[0]:
        raise ValueError("reasoning_span must precede and not overlap output_span")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, bool) or not isinstance(eos_id, Integral):
        raise TypeError("the shared tokenizer must expose one integer eos_token_id")
    with_eos_ids = expected_ids + (int(eos_id),)
    tokens = token_surfaces(tokenizer, with_eos_ids)
    return expected_ids, reasoning, output, tokens


def _float_vector(value: Any, name: str) -> tuple[float, ...]:
    try:
        raw = value.detach().cpu().reshape(-1).tolist()
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(f"{name} must be the official one-dimensional tensor") from error
    vector = tuple(float(item) for item in raw)
    if any(not math.isfinite(item) or item < 0.0 for item in vector):
        raise RuntimeError(f"{name} must be finite and non-negative")
    return vector


def _prompt_token_sources(
    *,
    prompt: RenderedPromptProvenance,
    offsets: tuple[tuple[int, int], ...],
    fields: tuple[TaskObservationField, ...],
    skill_spans: tuple[CharacterSpan, ...],
) -> tuple[tuple[str, str | None], ...]:
    sources: list[tuple[str, str | None]] = []
    explicit: list[tuple[str, str | None, CharacterSpan]] = [
        ("task_observation", field.field_id, field.span)
        for field in fields
        if field.span.start != field.span.end
    ]
    explicit.extend(
        ("skill", None, span)
        for span in skill_spans
        if span.start != span.end
    )

    for token_index, (start, end) in enumerate(offsets):
        matches = [
            (kind, field_id)
            for kind, field_id, span in explicit
            if start < span.end and span.start < end
        ]
        if len(matches) > 1:
            raise RuntimeError(
                f"rendered prompt token {token_index} crosses two explicit source spans"
            )
        sources.append(matches[0] if matches else ("format", None))
    return tuple(sources)


def build_dependency_measure(
    *,
    attributor: ExactTokenOffloadedLLMIFRAttribution,
    tokenizer: Any,
    instance_id: Hashable,
    prompt: RenderedPromptProvenance,
    target: HistoricalRolloutTarget,
    capture_lease: OffloadedCaptureLease | None,
) -> DependencyMeasure:
    """Run official one-hop IFR and project it onto fixed source coordinates."""

    if not isinstance(attributor, ExactTokenOffloadedLLMIFRAttribution):
        raise TypeError("attributor must expose the adopted exact token-ID dependency entry point")
    if getattr(attributor, "tokenizer", None) is not tokenizer:
        raise RuntimeError("the dependency bridge and attributor must share one tokenizer instance")
    if getattr(attributor, "use_chat_template", None) is not False:
        raise RuntimeError("the exact rendered prompt requires use_chat_template=False")
    if not isinstance(instance_id, Hashable):
        raise TypeError("instance_id must be hashable")

    fields, skill_spans = _validated_provenance(prompt)
    prompt_ids, prompt_offsets = _encode_with_offsets(tokenizer, prompt.text, "rendered prompt")
    if not prompt_ids:
        raise ValueError("the exact rendered prompt must contain tokens")
    if prompt_ids != _integer_ids(prompt.token_ids, "rendered prompt token_ids"):
        raise RuntimeError("rendered prompt provenance token IDs differ from the official encoding")
    target_ids, reasoning_span, output_span, generation_tokens = _validated_target(
        tokenizer,
        target,
    )
    sources = _prompt_token_sources(
        prompt=prompt,
        offsets=prompt_offsets,
        fields=fields,
        skill_spans=skill_spans,
    )

    result = attributor.calculate_ifr_multi_hop_ids(
        prompt.text,
        prompt_ids=prompt_ids,
        target=target.text,
        generation_ids=target_ids + (int(tokenizer.eos_token_id),),
        sink_span=output_span,
        thinking_span=reasoning_span,
        n_hops=1,
        renorm_threshold=0.0,
        observation_mask=None,
        capture_lease=capture_lease,
    )
    expected_prompt_tokens = token_surfaces(tokenizer, prompt_ids)
    if tuple(result.prompt_tokens) != expected_prompt_tokens:
        raise RuntimeError("official IFR prompt tokens do not match the direct prompt offsets")
    if tuple(result.generation_tokens) != generation_tokens:
        raise RuntimeError("official IFR generation tokens do not match the raw completion plus EOS")

    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise RuntimeError("official IFR result omitted metadata")
    ifr = metadata.get("ifr")
    if not isinstance(ifr, Mapping):
        raise RuntimeError("official IFR result omitted IFR metadata")
    if ifr.get("type") != "multi_hop" or ifr.get("n_hops") != 1:
        raise RuntimeError("official IFR result is not the requested one-hop dependency measure")
    if tuple(ifr.get("sink_span_generation", ())) != output_span:
        raise RuntimeError("official IFR changed the output sink span")
    if tuple(ifr.get("thinking_span_generation", ())) != reasoning_span:
        raise RuntimeError("official IFR changed the reasoning span")
    observation = ifr.get("observation_projected")
    if not isinstance(observation, Mapping) or "sum" not in observation:
        raise RuntimeError("official IFR result omitted observation_projected['sum']")
    projected = _float_vector(observation["sum"], "observation_projected['sum']")
    expected_length = len(prompt_ids) + len(generation_tokens)
    if len(projected) != expected_length:
        raise RuntimeError("official IFR projected attribution has the wrong coordinate length")
    prompt_masses = projected[: len(prompt_ids)]
    generation_masses = projected[len(prompt_ids) :]

    # The official default observation mask removes the reasoning and sink
    # content from the visible measure.  Generated tokens left outside those
    # two explicit content spans are serialization tokens, so their visible
    # mass belongs to the same aggregated FORMAT coordinate as prompt format.
    reasoning_positions = set(range(reasoning_span[0], reasoning_span[1] + 1))
    output_positions = set(range(output_span[0], output_span[1] + 1))
    hidden_positions = reasoning_positions | output_positions
    if any(generation_masses[position] != 0.0 for position in hidden_positions):
        raise RuntimeError(
            "official IFR observation mask exposed reasoning or output content mass"
        )
    generation_format_masses = [
        mass
        for position, mass in enumerate(generation_masses)
        if position not in hidden_positions
    ]

    field_ids: dict[str, list[int]] = {field.field_id: [] for field in fields}
    field_masses: dict[str, list[float]] = {field.field_id: [] for field in fields}
    skill_mass_values: list[float] = []
    format_mass_values: list[float] = list(generation_format_masses)
    for token_id, mass, (kind, field_id) in zip(
        prompt_ids,
        prompt_masses,
        sources,
        strict=True,
    ):
        if kind == "task_observation":
            if field_id is None:
                raise RuntimeError("task/observation token has no field_id")
            field_ids[field_id].append(token_id)
            field_masses[field_id].append(mass)
        elif kind == "skill":
            skill_mass_values.append(mass)
        elif kind == "format":
            format_mass_values.append(mass)
        else:
            raise RuntimeError(f"unknown provenance kind: {kind!r}")

    task_observation = tuple(
        FieldTokenMeasure(
            field_id=field_id,
            token_ids=tuple(field_ids[field_id]),
            masses=tuple(field_masses[field_id]),
        )
        for field_id in sorted(field_ids)
    )
    return DependencyMeasure(
        instance_id=instance_id,
        task_observation=task_observation,
        skill_mass=math.fsum(skill_mass_values),
        format_mass=math.fsum(format_mass_values),
        target_token_ids=target_ids,
        reasoning_span=reasoning_span,
        output_span=output_span,
        attributor_identity=id(attributor),
    )


def _measure_fields(measure: DependencyMeasure) -> dict[str, FieldTokenMeasure]:
    if not isinstance(measure, DependencyMeasure):
        raise TypeError("dependency distance requires DependencyMeasure values")
    fields = {field.field_id: field for field in measure.task_observation}
    if len(fields) != len(measure.task_observation):
        raise ValueError("DependencyMeasure contains duplicate task/observation field IDs")
    values = [
        *(mass for field in measure.task_observation for mass in field.masses),
        measure.skill_mass,
        measure.format_mass,
    ]
    if any(not math.isfinite(mass) or mass < 0.0 for mass in values):
        raise ValueError("DependencyMeasure masses must be finite and non-negative")
    if any(len(field.token_ids) != len(field.masses) for field in measure.task_observation):
        raise ValueError("DependencyMeasure field token IDs and masses are misaligned")
    return fields


def _paired_measure_values(
    old: DependencyMeasure,
    candidate: DependencyMeasure,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    old_fields = _measure_fields(old)
    candidate_fields = _measure_fields(candidate)
    if old.instance_id != candidate.instance_id:
        raise ValueError("old and candidate measures belong to different instances")
    if old.attributor_identity != candidate.attributor_identity:
        raise ValueError("old and candidate measures were not produced by the same attributor")
    if (
        old.target_token_ids != candidate.target_token_ids
        or old.reasoning_span != candidate.reasoning_span
        or old.output_span != candidate.output_span
    ):
        raise ValueError("old and candidate measures do not teacher-force the same raw completion")
    if old_fields.keys() != candidate_fields.keys():
        raise ValueError("old and candidate prompts expose different task/observation fields")

    old_values: list[float] = []
    candidate_values: list[float] = []
    for field_id in sorted(old_fields):
        old_field = old_fields[field_id]
        candidate_field = candidate_fields[field_id]
        if old_field.token_ids != candidate_field.token_ids:
            raise ValueError(
                f"task/observation field {field_id!r} has different old/candidate token coordinates"
            )
        old_values.extend(old_field.masses)
        candidate_values.extend(candidate_field.masses)
    old_values.extend((old.skill_mass, old.format_mass))
    candidate_values.extend((candidate.skill_mass, candidate.format_mass))
    return tuple(old_values), tuple(candidate_values)


def _bray_curtis_terms(
    old_values: Sequence[float],
    candidate_values: Sequence[float],
) -> tuple[float, float]:
    denominator = math.fsum(old_values) + math.fsum(candidate_values)
    numerator = math.fsum(
        abs(old_mass - candidate_mass)
        for old_mass, candidate_mass in zip(old_values, candidate_values, strict=True)
    )
    if numerator > denominator:
        raise ArithmeticError("finite non-negative Bray-Curtis numerator exceeds its denominator")
    return numerator, denominator


def instance_dependency_distance(old: DependencyMeasure, candidate: DependencyMeasure) -> float:
    """Bray-Curtis distance for one paired historical replay instance."""

    old_values, candidate_values = _paired_measure_values(old, candidate)
    numerator, denominator = _bray_curtis_terms(old_values, candidate_values)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def batch_dependency_distance(
    old: Sequence[DependencyMeasure],
    candidate: Sequence[DependencyMeasure],
) -> float:
    """Bray-Curtis distance after concatenating all paired instance measures."""

    old_measures = tuple(old)
    candidate_measures = tuple(candidate)
    if not old_measures:
        raise ValueError("dependency minibatch must be non-empty")
    if len(old_measures) != len(candidate_measures):
        raise ValueError("old and candidate dependency minibatches have different lengths")
    numerators: list[float] = []
    denominators: list[float] = []
    for old_measure, candidate_measure in zip(
        old_measures,
        candidate_measures,
        strict=True,
    ):
        old_values, candidate_values = _paired_measure_values(old_measure, candidate_measure)
        numerator, denominator = _bray_curtis_terms(old_values, candidate_values)
        numerators.append(numerator)
        denominators.append(denominator)

    total_denominator = math.fsum(denominators)
    if total_denominator == 0.0:
        return 0.0
    return math.fsum(numerators) / total_denominator


def passes_dependency_gate(*, distance: Real, epsilon_dep: Real) -> bool:
    """Apply the required explicit replay-dependency threshold."""

    values: dict[str, Real] = {"distance": distance, "epsilon_dep": epsilon_dep}
    converted: dict[str, float] = {}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real scalar")
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
        converted[name] = number
    return converted["distance"] <= converted["epsilon_dep"]
