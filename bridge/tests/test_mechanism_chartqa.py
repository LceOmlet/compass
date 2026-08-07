from __future__ import annotations

import hashlib
import inspect
from io import BytesIO
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = PROJECT_ROOT / "upstreams"
sys.path.insert(
    0, str(UPSTREAM_ROOT / "skill-factory" / "upstream" / "lmms-eval")
)

import dspy
from dspy.utils.dummies import DummyLM
from lmms_eval.tasks.chartqa.utils import (
    chartqa_doc_to_text,
    chartqa_doc_to_visual,
    chartqa_process_results,
)
from PIL import Image
import bridge.mechanism_chartqa as mechanism_chartqa
from bridge.mechanism_chartqa import (
    build_prepared_chartqa_owner_composition,
    build_chartqa_owner_composition,
    chartqa_metric,
    chartqa_metric_with_feedback,
    chartqa_prompt_kwargs,
    load_prepared_chartqa_splits,
    load_chartqa_splits,
    load_chartqa_program_class,
    load_lmms_chartqa_test,
)


def test_composition_imports_the_two_pinned_owners() -> None:
    ChartQASingleStepProgram = load_chartqa_program_class(
        UPSTREAM_ROOT / "skill-factory"
    )
    assert Path(inspect.getfile(ChartQASingleStepProgram)).resolve() == (
        UPSTREAM_ROOT
        / "skill-factory"
        / "src"
        / "skill_factory"
        / "programs"
        / "chartqa.py"
    ).resolve()
    assert Path(inspect.getfile(chartqa_process_results)).resolve() == (
        UPSTREAM_ROOT
        / "skill-factory"
        / "upstream"
        / "lmms-eval"
        / "lmms_eval"
        / "tasks"
        / "chartqa"
        / "utils.py"
    ).resolve()


def _write_source_split(root: Path, split: str, answer: str) -> None:
    split_dir = root / "ChartQA Dataset" / split
    (split_dir / "png").mkdir(parents=True)
    Image.new("RGB", (4, 4), "white").save(split_dir / "png" / "chart.png")
    for source in ("human", "augmented"):
        (split_dir / f"{split}_{source}.json").write_text(
            json.dumps(
                [
                    {
                        "imgname": "chart.png",
                        "query": f"What is the {source} value?",
                        "label": answer,
                    }
                ]
            ),
            encoding="utf-8",
        )


def _test_doc(answer: str = "42") -> dict:
    return {
        "image": Image.new("RGB", (4, 4), "white"),
        "question": "What value is shown?",
        "answer": answer,
        "type": "human_test",
    }


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 4), "white").save(output, format="PNG")
    return output.getvalue()


class _PreparedParquetRows(list):
    def __init__(self, rows):
        super().__init__(rows)
        self.cast_calls: list[tuple[str, bool]] = []

    def cast_column(self, name, feature):
        self.cast_calls.append((name, feature.decode))
        return self


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal_prepared_manifest(root: Path, manifest: dict) -> None:
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (root / "manifest.json.sha256").write_text(
        f"{_sha256(manifest_path)}  manifest.json\n",
        encoding="utf-8",
    )


def _write_prepared_root(tmp_path: Path) -> tuple[Path, _PreparedParquetRows]:
    root = tmp_path / "prepared"
    for split in ("train", "val"):
        image_path = root / "images" / split / "chart.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), "white").save(image_path)

    train_inputs = [
        {
            "id": f"train-{position}",
            "split_position": position,
            "source": "human",
            "source_position": position,
            "question": f"Train question {position}?",
            "image": "images/train/chart.png",
        }
        for position in range(2)
    ]
    train_labels = [
        {
            "id": f"train-{position}",
            "split_position": position,
            "answer": str(position),
        }
        for position in range(2)
    ]
    val_inputs = [
        {
            "id": "val-0",
            "split_position": 0,
            "source": "human",
            "source_position": 0,
            "question": "Validation question?",
            "image": "images/val/chart.png",
        }
    ]
    val_labels = [{"id": "val-0", "split_position": 0, "answer": "3"}]
    _write_jsonl(root / "views" / "train_lite_inputs.jsonl", train_inputs)
    _write_jsonl(root / "views" / "train_lite_labels.jsonl", train_labels)
    _write_jsonl(root / "views" / "val_lite_inputs.jsonl", val_inputs)
    _write_jsonl(root / "views" / "val_lite_labels.jsonl", val_labels)

    parquet_path = root / "lmms_test" / "test.parquet"
    parquet_path.parent.mkdir(parents=True)
    parquet_path.touch()
    pinned_files = (
        "views/train_lite_inputs.jsonl",
        "views/train_lite_labels.jsonl",
        "views/val_lite_inputs.jsonl",
        "views/val_lite_labels.jsonl",
        "lmms_test/test.parquet",
    )
    _seal_prepared_manifest(
        root,
        {
            "benchmark": "ChartQA",
            "status": "frozen",
            "optimization_owner": {
                "commit": "044eabfc306abfe9340c5741f0093aefc5973d06"
            },
            "selection": {
                "train_lite": {
                    "count": 2,
                    "composition": {"human": 2},
                    "rule_id": "small-test-rule",
                },
                "val_lite": {
                    "count": 1,
                    "composition": {"human": 1},
                    "rule_id": "small-test-rule",
                },
            },
            "hygiene": {
                "split_hygiene_rule_id": "small-split-hygiene",
                "exact_duplicate_rule_id": "small-duplicate-hygiene",
            },
            "formal_test": {
                "parquet": "lmms_test/test.parquet",
                "parquet_sha256": _sha256(parquet_path),
                "row_count": 2,
                "dataset_revision": "small-test-revision",
                "lmms_eval_commit": "cb45ac4d4a667ea5ef89c7a148bff69b3489b981",
                "type_counts": {"human_test": 1, "augmented_test": 1},
            },
            "files": {
                path: {"sha256": _sha256(root / path)} for path in pinned_files
            },
        },
    )
    rows = _PreparedParquetRows(
        [
            {
                "image": {"bytes": _png_bytes(), "path": "owner.png"},
                "question": f"Test question {position}?",
                "answer": str(position + 4),
                "type": "human_test" if position == 0 else "augmented_test",
            }
            for position in range(2)
        ]
    )
    return root, rows


def _use_small_prepared_protocol(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(
        mechanism_chartqa,
        "CHARTQA_PREPARED_SPLIT_COUNTS",
        {"train": 2, "val": 1, "test": 2},
    )
    monkeypatch.setattr(
        mechanism_chartqa,
        "CHARTQA_PREPARED_COMPOSITION",
        {"train": {"human": 2}, "val": {"human": 1}},
    )
    monkeypatch.setattr(
        mechanism_chartqa,
        "CHARTQA_PREPARED_SELECTION_RULE_ID",
        "small-test-rule",
    )
    monkeypatch.setattr(
        mechanism_chartqa,
        "CHARTQA_PREPARED_SPLIT_HYGIENE_RULE_ID",
        "small-split-hygiene",
    )
    monkeypatch.setattr(
        mechanism_chartqa,
        "CHARTQA_PREPARED_EXACT_DUPLICATE_RULE_ID",
        "small-duplicate-hygiene",
    )
    monkeypatch.setattr(
        mechanism_chartqa,
        "CHARTQA_PREPARED_MANIFEST_SHA256",
        _sha256(root / "manifest.json"),
    )
    monkeypatch.setattr(
        mechanism_chartqa,
        "CHARTQA_TEST_DATASET_REVISION",
        "small-test-revision",
    )
    monkeypatch.setattr(
        mechanism_chartqa,
        "CHARTQA_TEST_PARQUET_SHA256",
        hashlib.sha256(b"").hexdigest(),
    )
    monkeypatch.setattr(
        mechanism_chartqa,
        "CHARTQA_TEST_TYPE_COUNTS",
        {"human_test": 1, "augmented_test": 1},
    )


def test_minimal_owner_doc_flows_through_program_and_metric() -> None:
    doc = _test_doc()
    prompt_kwargs = chartqa_prompt_kwargs()
    example = dspy.Example(
        image=dspy.Image(chartqa_doc_to_visual(doc)[0]),
        prompt=chartqa_doc_to_text(doc, prompt_kwargs),
        answer=doc["answer"],
        chartqa_type=doc["type"],
    ).with_inputs("image", "prompt")
    program = load_chartqa_program_class(UPSTREAM_ROOT / "skill-factory")(
        DummyLM([{"answer": "42"}])
    )

    prediction = program(**example.inputs())

    assert tuple(name for name, _ in program.named_predictors()) == ("predict",)
    assert chartqa_metric(example, prediction) == 1.0
    assert chartqa_process_results(doc, [prediction.answer])["relaxed_overall"] == 1.0
    assert chartqa_metric_with_feedback(example, prediction) == {
        "score": 1.0,
        "feedback": "gold='42'\npred='42'\nofficial_score=1.0",
    }


def test_source_train_val_and_lmms_test_remain_isolated(tmp_path: Path) -> None:
    _write_source_split(tmp_path, "train", "10")
    _write_source_split(tmp_path, "val", "20")
    poisoned_test = tmp_path / "ChartQA Dataset" / "test"
    poisoned_test.mkdir(parents=True)
    (poisoned_test / "test_human.json").write_text(
        "this source test must never be read",
        encoding="utf-8",
    )
    calls: list[tuple[str, str, dict]] = []

    def fake_load_dataset(dataset_path: str, *, split: str, **kwargs):
        calls.append((dataset_path, split, kwargs))
        return [_test_doc("30")]

    splits = load_chartqa_splits(
        tmp_path,
        load_dataset_fn=fake_load_dataset,
    )

    assert [example.answer for example in splits.train] == ["10", "10"]
    assert [example.answer for example in splits.validation] == ["20", "20"]
    assert [example.answer for example in splits.test] == ["30"]
    assert calls == [("lmms-lab/ChartQA", "test", {"token": True})]


def test_lmms_loader_uses_pinned_owner_prompt_and_visual() -> None:
    doc = _test_doc("7")

    examples = load_lmms_chartqa_test(
        load_dataset_fn=lambda *_args, **_kwargs: [doc]
    )

    assert len(examples) == 1
    assert examples[0].prompt == chartqa_doc_to_text(doc, chartqa_prompt_kwargs())
    assert examples[0].answer == "7"
    assert isinstance(examples[0].image, dspy.Image)


def test_composition_keeps_owner_program_and_one_predictor(tmp_path: Path) -> None:
    _write_source_split(tmp_path, "train", "10")
    _write_source_split(tmp_path, "val", "20")

    composition = build_chartqa_owner_composition(
        DummyLM([{"answer": "10"}]),
        tmp_path,
        program_owner_root=UPSTREAM_ROOT / "skill-factory",
        load_dataset_fn=lambda *_args, **_kwargs: [_test_doc("30")],
    )

    assert isinstance(
        composition.program,
        load_chartqa_program_class(UPSTREAM_ROOT / "skill-factory"),
    )
    assert tuple(name for name, _ in composition.program.named_predictors()) == (
        "predict",
    )
    assert composition.metric is chartqa_metric
    assert composition.metric_with_feedback is chartqa_metric_with_feedback


def test_prepared_views_and_local_parquet_preserve_owner_semantics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root, parquet_rows = _write_prepared_root(tmp_path)
    _use_small_prepared_protocol(monkeypatch, root)
    calls: list[tuple[str, dict]] = []

    def fake_load_dataset(dataset_name: str, **kwargs):
        calls.append((dataset_name, kwargs))
        return parquet_rows

    splits = load_prepared_chartqa_splits(
        root,
        load_dataset_fn=fake_load_dataset,
    )

    assert (len(splits.train), len(splits.validation), len(splits.test)) == (2, 1, 2)
    assert [example.key for example in splits.train] == ["train-0", "train-1"]
    assert [example.split_position for example in splits.test] == [0, 1]
    assert [example.answer for example in splits.test] == ["4", "5"]
    assert calls == [
        (
            "parquet",
            {
                "data_files": str((root / "lmms_test" / "test.parquet").resolve()),
                "split": "train",
            },
        )
    ]
    assert parquet_rows.cast_calls == [("image", False)]

    for example in (*splits.train, *splits.validation, *splits.test):
        assert set(dict(example.inputs())) == {"image", "prompt"}
        assert isinstance(example.image, dspy.Image)
    expected_train_prompt = chartqa_doc_to_text(
        {"question": "Train question 0?"},
        chartqa_prompt_kwargs(),
    )
    assert splits.train[0].prompt == expected_train_prompt
    assert splits.test[0].prompt == chartqa_doc_to_text(
        {"question": "Test question 0?"},
        chartqa_prompt_kwargs(),
    )
    assert chartqa_metric(
        splits.test[0],
        dspy.Prediction(answer="4"),
    ) == 1.0


def test_prepared_views_fail_on_input_label_identity_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root, parquet_rows = _write_prepared_root(tmp_path)
    _use_small_prepared_protocol(monkeypatch, root)
    labels_path = root / "views" / "train_lite_labels.jsonl"
    labels = [json.loads(line) for line in labels_path.read_text().splitlines()]
    labels[1]["id"] = "wrong-id"
    _write_jsonl(labels_path, labels)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["views/train_lite_labels.jsonl"]["sha256"] = _sha256(
        labels_path
    )
    _seal_prepared_manifest(root, manifest)
    _use_small_prepared_protocol(monkeypatch, root)

    try:
        load_prepared_chartqa_splits(
            root,
            load_dataset_fn=lambda *_args, **_kwargs: parquet_rows,
        )
    except ValueError as exc:
        assert "identity mismatch" in str(exc)
    else:
        raise AssertionError("mismatched prepared ChartQA views must fail")


def test_prepared_composition_uses_pinned_program_without_gold_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root, parquet_rows = _write_prepared_root(tmp_path)
    _use_small_prepared_protocol(monkeypatch, root)

    composition = build_prepared_chartqa_owner_composition(
        DummyLM([{"answer": "0"}]),
        root,
        program_owner_root=UPSTREAM_ROOT / "skill-factory",
        load_dataset_fn=lambda *_args, **_kwargs: parquet_rows,
    )

    assert isinstance(
        composition.program,
        load_chartqa_program_class(UPSTREAM_ROOT / "skill-factory"),
    )
    assert tuple(name for name, _ in composition.program.named_predictors()) == (
        "predict",
    )
    assert set(dict(composition.splits.train[0].inputs())) == {"image", "prompt"}
    assert composition.metric is chartqa_metric
    assert composition.metric_with_feedback is chartqa_metric_with_feedback
