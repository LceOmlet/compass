from __future__ import annotations

import csv
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import dspy


# These are provenance pins, not experiment identifiers.  The adapter accepts
# explicit pre-generated split paths and never invokes the owner generator.
CLUTRR_OWNER_COMMIT = "d045fae289d3746503677ceed7631c999202501e"
CLUTRR_BASELINE_OWNER_COMMIT = "303ed9a48f82a59b4eb34ac5bd5866f0d82c5552"


class ClutrrVariant(str, Enum):
    """Noise variants defined by the CLUTRR owner."""

    SUPPORTING = "supporting"
    IRRELEVANT = "irrelevant"
    DISCONNECTED = "disconnected"


# facebookresearch/clutrr defines these as task_2, task_3, and task_4.  Relation
# length remains data/config owned; this module does not choose three formal
# task-length combinations for the experiment.
OWNER_TASK_NUMBER_BY_VARIANT: Mapping[ClutrrVariant, int] = MappingProxyType(
    {
        ClutrrVariant.SUPPORTING: 2,
        ClutrrVariant.IRRELEVANT: 3,
        ClutrrVariant.DISCONNECTED: 4,
    }
)


class InferClutrrRelation(dspy.Signature):
    """Infer the queried kinship relation and return one relation label only."""

    story: str = dspy.InputField()
    query: str = dspy.InputField()
    relation: str = dspy.OutputField(
        desc="Exactly one CLUTRR relation label, with no explanation."
    )


program = dspy.Predict(InferClutrrRelation)


@dataclass(frozen=True, slots=True)
class ClutrrFixedSplitPaths:
    """Three already-materialized owner CSV splits.

    Validation must be prepared and frozen before optimization.  In
    particular, there is deliberately no optimizer-seed argument here.
    """

    train: Path
    validation: Path
    test: Path


@dataclass(frozen=True, slots=True)
class ClutrrDatasetSplits:
    train: tuple[dspy.Example, ...]
    validation: tuple[dspy.Example, ...]
    test: tuple[dspy.Example, ...]


_TASK_NAME = re.compile(r"^task_(?P<task_number>[1-9][0-9]*)\.(?P<length>[1-9][0-9]*)$")
_NUMERIC_LABEL = re.compile(r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)$")
_REQUIRED_COLUMNS = frozenset({"id", "story", "query", "task_name"})
_OWNER_VARIANT_NAME_BY_TASK_NUMBER = MappingProxyType(
    {
        1: "clean",
        2: ClutrrVariant.SUPPORTING.value,
        3: ClutrrVariant.IRRELEVANT.value,
        4: ClutrrVariant.DISCONNECTED.value,
    }
)


def _coerce_variant(variant: ClutrrVariant | str) -> ClutrrVariant:
    if isinstance(variant, ClutrrVariant):
        return variant
    try:
        return ClutrrVariant(variant)
    except ValueError as error:
        choices = ", ".join(item.value for item in ClutrrVariant)
        raise ValueError(f"CLUTRR variant must be one of: {choices}") from error


def _required_text(
    row: Mapping[str, str | None],
    *,
    field: str,
    path: Path,
    line_number: int,
) -> str:
    value = row.get(field)
    if value is None or not value.strip():
        raise ValueError(
            f"{path}: CSV line {line_number} has no value for {field!r}"
        )
    return value


def _required_relation_label(
    row: Mapping[str, str | None],
    *,
    label_column: str,
    path: Path,
    line_number: int,
) -> str:
    label = _required_text(
        row,
        field=label_column,
        path=path,
        line_number=line_number,
    )
    if _NUMERIC_LABEL.fullmatch(label.strip()):
        raise ValueError(
            f"{path}: CSV line {line_number} column {label_column!r} "
            "contains a numeric class id, not an owner relation label; "
            "select the textual owner label column explicitly"
        )
    return label


def _load_owner_csv(
    path: Path,
    *,
    variant: ClutrrVariant,
    relation_length: int | None,
    label_column: str,
    restrict_to_training_variant: bool,
) -> tuple[dspy.Example, ...]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = (_REQUIRED_COLUMNS | {label_column}).difference(columns)
        if missing:
            raise ValueError(
                f"{path}: owner CSV is missing columns {sorted(missing)}"
            )

        expected_task_number = OWNER_TASK_NUMBER_BY_VARIANT[variant]
        examples: list[dspy.Example] = []
        for line_number, row in enumerate(reader, start=2):
            task_name = _required_text(
                row,
                field="task_name",
                path=path,
                line_number=line_number,
            )
            match = _TASK_NAME.fullmatch(task_name)
            if match is None:
                raise ValueError(
                    f"{path}: CSV line {line_number} has invalid owner "
                    f"task_name {task_name!r}"
                )
            task_number = int(match.group("task_number"))
            row_relation_length = int(match.group("length"))
            if restrict_to_training_variant and task_number != expected_task_number:
                raise ValueError(
                    f"{path}: CSV line {line_number} belongs to {task_name!r}, "
                    f"not the owner {variant.value!r} variant"
                )
            if (
                restrict_to_training_variant
                and relation_length is not None
                and row_relation_length != relation_length
            ):
                raise ValueError(
                    f"{path}: CSV line {line_number} has relation length "
                    f"{row_relation_length}, expected {relation_length}"
                )

            example = dspy.Example(
                key=_required_text(
                    row,
                    field="id",
                    path=path,
                    line_number=line_number,
                ),
                story=_required_text(
                    row,
                    field="story",
                    path=path,
                    line_number=line_number,
                ),
                query=_required_text(
                    row,
                    field="query",
                    path=path,
                    line_number=line_number,
                ),
                relation=_required_relation_label(
                    row,
                    label_column=label_column,
                    path=path,
                    line_number=line_number,
                ),
                clutrr_variant=_OWNER_VARIANT_NAME_BY_TASK_NUMBER.get(
                    task_number,
                    f"task_{task_number}",
                ),
                clutrr_training_variant=variant.value,
                clutrr_task_name=task_name,
                clutrr_task_number=task_number,
                relation_length=row_relation_length,
                clutrr_label_column=label_column,
            ).with_inputs("story", "query")
            examples.append(example)
    return tuple(examples)


def load_fixed_clutrr_splits(
    paths: ClutrrFixedSplitPaths,
    *,
    variant: ClutrrVariant | str,
    relation_length: int | None = None,
    label_column: str = "target",
) -> ClutrrDatasetSplits:
    """Read frozen owner CSV files in their existing order.

    ``relation_length=None`` permits a pre-generated split containing multiple
    lengths of the same owner variant.  Supplying a length validates training
    and validation rows.  The owner robust test matrix is always preserved
    intact; this function never generates, filters, resamples, or reshuffles.

    The original owner CSV stores the textual label in ``target``.  Converted
    backings that store an integer in ``target`` must explicitly pass their
    textual owner column (for example, ``label_column="target_text"``); numeric
    and blank labels fail instead of being silently scored as relations.
    """

    resolved_variant = _coerce_variant(variant)
    if not isinstance(label_column, str) or not label_column.strip():
        raise TypeError("label_column must be a non-empty string")
    if relation_length is not None and (
        isinstance(relation_length, bool)
        or not isinstance(relation_length, int)
        or relation_length <= 0
    ):
        raise TypeError("relation_length must be a positive integer or None")
    return ClutrrDatasetSplits(
        train=_load_owner_csv(
            paths.train,
            variant=resolved_variant,
            relation_length=relation_length,
            label_column=label_column,
            restrict_to_training_variant=True,
        ),
        validation=_load_owner_csv(
            paths.validation,
            variant=resolved_variant,
            relation_length=relation_length,
            label_column=label_column,
            restrict_to_training_variant=True,
        ),
        test=_load_owner_csv(
            paths.test,
            variant=resolved_variant,
            relation_length=relation_length,
            label_column=label_column,
            # The robust owner package deliberately evaluates a model trained
            # on one noise condition against its complete task_1/2/3/4 test
            # matrix.  Preserve every row and its task_name for aggregation.
            restrict_to_training_variant=False,
        ),
    )


RelationOverlap = Callable[[list[list[str]], list[list[str]]], Any]


@dataclass(frozen=True, slots=True)
class ClutrrOwnerMetric:
    """Thin scalar adapter around the pinned baseline's relation_overlap."""

    relation_overlap: RelationOverlap

    @staticmethod
    def _gold_and_prediction(
        example: Any,
        prediction: Any,
    ) -> tuple[str, str]:
        gold = getattr(example, "relation")
        predicted = getattr(prediction, "relation")
        if not isinstance(gold, str) or not isinstance(predicted, str):
            raise TypeError("CLUTRR gold and predicted relation must be strings")
        return gold, predicted

    def score(self, example: Any, prediction: Any) -> float:
        gold, predicted = self._gold_and_prediction(example, prediction)
        # QualityMetric.relation_overlap expects batches of token sequences.
        # Treating each relation as exactly one token preserves its owner-defined
        # single-label accuracy semantics without local parsing or fuzzy match.
        return float(self.relation_overlap([[predicted]], [[gold]]))

    def __call__(
        self,
        example: Any,
        prediction: Any,
        trace: Any = None,
    ) -> float:
        del trace
        return self.score(example, prediction)

    def with_feedback(
        self,
        example: Any,
        prediction: Any,
        trace: Any = None,
    ) -> dspy.Prediction:
        del trace
        gold, predicted = self._gold_and_prediction(example, prediction)
        owner_score = float(
            self.relation_overlap([[predicted]], [[gold]])
        )
        return dspy.Prediction(
            score=owner_score,
            feedback=(
                f"Gold relation: {gold}\n"
                f"Predicted relation: {predicted}\n"
                f"Owner relation_overlap score: {owner_score}"
            ),
        )
