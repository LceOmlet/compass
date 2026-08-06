from __future__ import annotations

import json
import hashlib
from pathlib import Path

import dspy
import pytest

from bridge import mechanism_hitab as hitab


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _fake_owner_tree(tmp_path: Path) -> tuple[Path, dict]:
    owner_root = tmp_path / "HiTab"
    table_artifact = {
        "title": "Complete table",
        "top_root": {"name": "<TOP>", "children_dict": []},
        "left_root": {"name": "<LEFT>", "children_dict": []},
        "data": [[{"value": 3.0}, {"value": "三"}]],
        "texts": [["", "column"], ["row", "3"]],
        "merged_regions": [
            {
                "first_row": 0,
                "last_row": 0,
                "first_column": 0,
                "last_column": 1,
            }
        ],
    }
    _write_jsonl(
        owner_root / "data" / "processed_input" / "tables.jsonl",
        [{"name": "table-1", "kg": table_artifact, "props": ["derived"]}],
    )
    for filename, prefix in (
        ("train_samples.jsonl", "train"),
        ("dev_samples.jsonl", "dev"),
        ("test_samples.jsonl", "test"),
    ):
        _write_jsonl(
            owner_root / "data" / filename,
            [
                {
                    "id": f"{prefix}-1",
                    "table_id": "table-1",
                    "question": f"{prefix} question 1",
                    "answer": [3],
                    "linked_cells": {"gold": "must not leak"},
                    "aggregation": ["sum"],
                    "answer_formulas": ["=A1"],
                },
                {
                    "id": f"{prefix}-2",
                    "table_id": "table-1",
                    "question": f"{prefix} question 2",
                    "answer": ["三"],
                    "linked_cells": {"gold": "must not leak"},
                },
            ],
        )
    return owner_root, table_artifact


def _write_prepared_file(root: Path, relative: str, text: str) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _fake_prepared_root(tmp_path: Path) -> Path:
    root = tmp_path / "prepared"
    files: dict[str, dict] = {}
    table = {
        "name": "table-1",
        "kg": {"title": "safe", "data": [[{"value": 3}]]},
    }
    files["owner/data/processed_input/tables.jsonl"] = _write_prepared_file(
        root,
        "owner/data/processed_input/tables.jsonl",
        json.dumps(table) + "\n",
    )
    files["owner/qa/datadump/utils.py"] = _write_prepared_file(
        root,
        "owner/qa/datadump/utils.py",
        "def naive_str_to_float(value):\n    return value\n",
    )
    files["owner/qa/table/utils.py"] = _write_prepared_file(
        root,
        "owner/qa/table/utils.py",
        "def hmt_score(prediction, answer):\n    return float(prediction == answer)\n",
    )
    split_specs = (
        ("train_lite", 150),
        ("dev_lite", 300),
        ("test_official_full", 1584),
    )
    for split, count in split_specs:
        inputs = []
        labels = []
        for position in range(count):
            identity = f"{split}-{position}"
            inputs.append(
                json.dumps(
                    {
                        "id": identity,
                        "split_position": position,
                        "question": f"question {position}",
                        "table_id": "table-1",
                    }
                )
            )
            labels.append(
                json.dumps(
                    {
                        "id": identity,
                        "split_position": position,
                        "answer": "3",
                    }
                )
            )
        for suffix, rows in (("inputs", inputs), ("labels", labels)):
            relative = f"views/{split}_{suffix}.jsonl"
            files[relative] = _write_prepared_file(
                root,
                relative,
                "\n".join(rows) + "\n",
            )
    manifest = {
        "status": "frozen",
        "owner_commit": hitab.HITAB_OWNER_COMMIT,
        "files": files,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (root / "manifest.json.sha256").write_text(
        f"{manifest_hash}  manifest.json\n",
        encoding="utf-8",
    )
    return root


def test_loader_preserves_owner_splits_and_exposes_only_safe_inputs(
    tmp_path: Path,
) -> None:
    owner_root, table_artifact = _fake_owner_tree(tmp_path)

    benchmark = hitab.HiTabBenchmark(
        dataset_mode="lite",
        owner_root=owner_root,
    )

    assert benchmark.preserve_official_splits is True
    assert [example.question for example in benchmark.train_set] == [
        "train question 1",
        "train question 2",
    ]
    assert [example.question for example in benchmark.val_set] == [
        "dev question 1",
        "dev question 2",
    ]
    assert [example.question for example in benchmark.test_set] == [
        "test question 1",
        "test question 2",
    ]

    example = benchmark.train_set[0]
    assert example.inputs().keys() == ["question", "table"]
    assert example.labels().keys() == ["answer"]
    assert json.loads(example.table) == table_artifact
    assert "linked_cells" not in example.table
    assert "answer_formulas" not in example.table
    assert "aggregation" not in example.table

    named_predictors = list(hitab.program.named_predictors())
    assert len(named_predictors) == 1
    assert named_predictors[0][0] == "answer"


def test_metric_delegates_raw_prediction_to_dynamic_owner_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_root = tmp_path / "HiTab"
    score_module = owner_root / "qa" / "table" / "utils.py"
    score_module.parent.mkdir(parents=True)
    score_module.write_text(
        "def hmt_score(prediction, answer):\n"
        "    return 0.625 if prediction == 'raw [1]' and answer == [1] else 0.0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(hitab.HITAB_ROOT_ENV, str(owner_root))

    example = dspy.Example(answer=[1])
    prediction = dspy.Prediction(answer="raw [1]")

    assert hitab.hitab_metric(example, prediction) == 0.625
    assert hitab.hitab_metric_with_feedback(example, prediction) == {
        "score": 0.625,
        "feedback": "gold=[1]\npred='raw [1]'\nowner_score=0.625",
    }


def test_dynamic_owner_import_fails_instead_of_falling_back(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="microsoft/HiTab scorer is missing"):
        hitab.load_owner_hmt_score(tmp_path / "missing-owner")


def test_prepared_loader_uses_fixed_views_and_keeps_gold_out_of_inputs(
    tmp_path: Path,
) -> None:
    root = _fake_prepared_root(tmp_path)

    composition = hitab.build_prepared_hitab_owner_composition(root)

    assert (
        len(composition.splits.train),
        len(composition.splits.validation),
        len(composition.splits.test),
    ) == (150, 300, 1584)
    example = composition.splits.train[0]
    assert example.inputs().keys() == ["question", "table"]
    assert "answer" not in example.inputs()
    assert "linked_cells" not in example.table
    assert composition.metric(example, dspy.Prediction(answer="3")) == 1.0


def test_prepared_loader_rejects_input_label_identity_drift(tmp_path: Path) -> None:
    root = _fake_prepared_root(tmp_path)
    label_path = root / "views" / "train_lite_labels.jsonl"
    rows = label_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["id"] = "wrong-id"
    rows[0] = json.dumps(first)
    label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = label_path.read_bytes()
    manifest["files"]["views/train_lite_labels.jsonl"] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (root / "manifest.json.sha256").write_text(
        f"{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}  manifest.json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        hitab.load_prepared_hitab_splits(root)
