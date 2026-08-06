from __future__ import annotations

import csv
import inspect
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from bridge.mechanism_clutrr import (
    OWNER_TASK_NUMBER_BY_VARIANT,
    ClutrrFixedSplitPaths,
    ClutrrOwnerMetric,
    ClutrrVariant,
    load_fixed_clutrr_splits,
    program,
)


def _write_owner_csv(
    path: Path,
    *,
    task_number: int,
    relation_length: int = 3,
    ids: tuple[str, ...] = ("one", "two"),
    numeric_target: bool = False,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "id",
                "story",
                "query",
                "target",
                "target_text",
                "task_name",
                "text_query",
            ),
        )
        writer.writeheader()
        for index, identifier in enumerate(ids):
            writer.writerow(
                {
                    "id": identifier,
                    "story": f"owner story {index}",
                    "query": f"('person-{index}', 'relative-{index}')",
                    "target": str(4 if index == 0 else 8)
                    if numeric_target
                    else ("sister" if index == 0 else "uncle"),
                    "target_text": "sister" if index == 0 else "uncle",
                    "task_name": f"task_{task_number}.{relation_length}",
                    "text_query": f"What is the relation for row {index}?",
                }
            )


_REAL_HF_ROBUST_HEADER = (
    "\ufeff,id,story,query,target,target_text,clean_story,proof_state,f_comb,"
    "task_name,story_edges,edge_types,query_edge,genders,task_split"
)


def _write_hf_robust_fixture(
    path: Path,
    rows: tuple[tuple[str, str, str], ...],
) -> None:
    """Write the real CLUTRR/v1 robust backing header and essential cells."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(_REAL_HF_ROBUST_HEADER + "\n")
        writer = csv.writer(handle)
        for index, (identifier, task_name, label) in enumerate(rows):
            writer.writerow(
                (
                    index,
                    identifier,
                    f"owner story {identifier}",
                    f"('person-{identifier}', 'relative-{identifier}')",
                    index,
                    label,
                    f"clean owner story {identifier}",
                    "[]",
                    "mother-mother",
                    task_name,
                    "[]",
                    "[]",
                    "(0, 1)",
                    "person:female,relative:female",
                    "test" if identifier.startswith("test") else "train",
                )
            )


@pytest.mark.parametrize("variant", tuple(ClutrrVariant))
def test_fixed_owner_splits_support_each_variant_without_reordering(
    tmp_path: Path,
    variant: ClutrrVariant,
) -> None:
    task_number = OWNER_TASK_NUMBER_BY_VARIANT[variant]
    paths = ClutrrFixedSplitPaths(
        train=tmp_path / "train.csv",
        validation=tmp_path / "validation.csv",
        test=tmp_path / "test.csv",
    )
    for path in (paths.train, paths.validation, paths.test):
        _write_owner_csv(path, task_number=task_number)

    splits = load_fixed_clutrr_splits(
        paths,
        variant=variant,
        relation_length=3,
    )

    assert [example.key for example in splits.train] == ["one", "two"]
    assert [example.key for example in splits.validation] == ["one", "two"]
    assert [example.key for example in splits.test] == ["one", "two"]
    assert [example.relation for example in splits.train] == [
        "sister",
        "uncle",
    ]
    assert all(example.clutrr_variant == variant.value for example in splits.train)
    assert splits.train[0].inputs().toDict() == {
        "story": "owner story 0",
        "query": "('person-0', 'relative-0')",
    }


def test_fixed_split_loader_has_no_optimizer_seed_or_shuffle_surface() -> None:
    parameters = inspect.signature(load_fixed_clutrr_splits).parameters
    assert "optimizer_seed" not in parameters
    assert "seed" not in parameters


def test_fixed_split_loader_rejects_a_different_owner_variant(
    tmp_path: Path,
) -> None:
    path = tmp_path / "supporting.csv"
    _write_owner_csv(path, task_number=2)
    paths = ClutrrFixedSplitPaths(path, path, path)

    with pytest.raises(ValueError, match="not the owner 'irrelevant' variant"):
        load_fixed_clutrr_splits(
            paths,
            variant=ClutrrVariant.IRRELEVANT,
        )


def test_hf_backing_uses_explicit_text_label_and_never_numeric_target(
    tmp_path: Path,
) -> None:
    paths = ClutrrFixedSplitPaths(
        train=tmp_path / "train.csv",
        validation=tmp_path / "validation.csv",
        test=tmp_path / "test.csv",
    )
    _write_owner_csv(paths.train, task_number=2, numeric_target=True)
    _write_owner_csv(paths.validation, task_number=2, numeric_target=True)
    _write_owner_csv(paths.test, task_number=2, numeric_target=True)

    with pytest.raises(ValueError, match="contains a numeric class id"):
        load_fixed_clutrr_splits(
            paths,
            variant=ClutrrVariant.SUPPORTING,
        )

    splits = load_fixed_clutrr_splits(
        paths,
        variant=ClutrrVariant.SUPPORTING,
        label_column="target_text",
    )
    assert [example.relation for example in splits.train] == [
        "sister",
        "uncle",
    ]
    assert all(
        example.clutrr_label_column == "target_text"
        for example in splits.train
    )


def test_explicit_owner_label_column_rejects_an_empty_label(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty-label.csv"
    _write_hf_robust_fixture(
        path,
        (("train-empty", "task_2.3", ""),),
    )

    with pytest.raises(ValueError, match="no value for 'target_text'"):
        load_fixed_clutrr_splits(
            ClutrrFixedSplitPaths(path, path, path),
            variant=ClutrrVariant.SUPPORTING,
            label_column="target_text",
        )


def test_robust_test_split_preserves_all_owner_tasks_for_aggregation(
    tmp_path: Path,
) -> None:
    paths = ClutrrFixedSplitPaths(
        train=tmp_path / "train.csv",
        validation=tmp_path / "validation.csv",
        test=tmp_path / "test.csv",
    )
    _write_hf_robust_fixture(
        paths.train,
        (
            ("train-2", "task_2.2", "sister"),
            ("train-3", "task_2.3", "uncle"),
        ),
    )
    _write_hf_robust_fixture(
        paths.validation,
        (("train-val", "task_2.3", "mother"),),
    )
    _write_hf_robust_fixture(
        paths.test,
        (
            ("test-clean", "task_1.3", "father"),
            ("test-support", "task_2.3", "mother"),
            ("test-irrelevant", "task_3.3", "sister"),
            ("test-disconnected", "task_4.3", "uncle"),
        ),
    )

    splits = load_fixed_clutrr_splits(
        paths,
        variant=ClutrrVariant.SUPPORTING,
        label_column="target_text",
    )

    assert [example.key for example in splits.test] == [
        "test-clean",
        "test-support",
        "test-irrelevant",
        "test-disconnected",
    ]
    assert Counter(
        example.clutrr_task_name for example in splits.test
    ) == Counter(
        {
            "task_1.3": 1,
            "task_2.3": 1,
            "task_3.3": 1,
            "task_4.3": 1,
        }
    )
    assert Counter(
        example.clutrr_variant for example in splits.test
    ) == Counter(
        {
            "clean": 1,
            "supporting": 1,
            "irrelevant": 1,
            "disconnected": 1,
        }
    )
    assert [example.clutrr_task_number for example in splits.test] == [
        1,
        2,
        3,
        4,
    ]


def test_metric_delegates_exact_single_labels_to_owner_relation_overlap() -> None:
    calls: list[tuple[list[list[str]], list[list[str]]]] = []

    def relation_overlap(
        prediction: list[list[str]],
        hypothesis: list[list[str]],
    ) -> float:
        calls.append((prediction, hypothesis))
        return 0.75

    metric = ClutrrOwnerMetric(relation_overlap)
    example = SimpleNamespace(relation="sister")
    prediction = SimpleNamespace(relation="Sister.")

    assert metric(example, prediction) == 0.75
    assert calls == [([["Sister."]], [["sister"]])]


def test_feedback_contains_only_gold_prediction_and_owner_score() -> None:
    metric = ClutrrOwnerMetric(lambda _prediction, _hypothesis: 1.0)

    result = metric.with_feedback(
        SimpleNamespace(relation="mother"),
        SimpleNamespace(relation="mother"),
    )

    assert result.score == 1.0
    assert result.feedback == (
        "Gold relation: mother\n"
        "Predicted relation: mother\n"
        "Owner relation_overlap score: 1.0"
    )


def test_program_has_only_story_query_inputs_and_one_relation_output() -> None:
    assert tuple(program.signature.input_fields) == ("story", "query")
    assert tuple(program.signature.output_fields) == ("relation",)
