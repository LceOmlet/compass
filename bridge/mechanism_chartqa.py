"""Thin ChartQA owner composition for COMPASS mechanism experiments.

The benchmark semantics stay with their existing owners:

* ``skill-factory`` owns the single-predictor DSPy program;
* pinned LMMS-Eval owns image normalization, prompting, and scoring;
* ``vis-nlp/ChartQA`` owns the train/validation file layout.

This module only maps the source release into the owner program's DSPy inputs
and keeps the LMMS test split separate from those optimization examples.
"""

from __future__ import annotations

import inspect
import hashlib
import importlib.util
from io import BytesIO
import json
import os
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import dspy
from datasets import Image as DatasetImage
from datasets import load_dataset
from lmms_eval import utils as lmms_utils
from lmms_eval.tasks.chartqa.utils import (
    chartqa_doc_to_text,
    chartqa_doc_to_visual,
    chartqa_process_results,
)
from PIL import Image as PILImage


SKILL_FACTORY_OWNER_COMMIT = "e23f82f9c7ed2eacfb9124b92143358a9953263c"
CHARTQA_OPTIMIZATION_OWNER_COMMIT = "044eabfc306abfe9340c5741f0093aefc5973d06"
LMMS_EVAL_OWNER_COMMIT = "cb45ac4d4a667ea5ef89c7a148bff69b3489b981"
CHARTQA_PREPARED_SPLIT_COUNTS = {"train": 150, "val": 300, "test": 2500}
SKILL_FACTORY_ROOT_ENV = "SKILL_FACTORY_ROOT"
_DEFAULT_SKILL_FACTORY_ROOT = (
    Path(__file__).resolve().parents[1] / "upstreams" / "skill-factory"
)


def _skill_factory_root(root: str | Path | None = None) -> Path:
    configured = root or os.environ.get(SKILL_FACTORY_ROOT_ENV)
    return Path(configured or _DEFAULT_SKILL_FACTORY_ROOT).expanduser().resolve()


def _verify_skill_factory_revision(root: Path) -> None:
    if not (root / ".git").exists():
        return
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cannot resolve pinned skill-factory checkout at {root}: "
            f"{result.stderr.strip()}"
        )
    revision = result.stdout.strip()
    if revision != SKILL_FACTORY_OWNER_COMMIT:
        raise RuntimeError(
            "skill-factory owner revision differs from the experiment pin: "
            f"expected={SKILL_FACTORY_OWNER_COMMIT}, actual={revision}"
        )


@lru_cache(maxsize=None)
def _load_chartqa_program_class(root_text: str) -> type[dspy.Module]:
    """Load only the pinned owner program file, without package side effects."""

    root = Path(root_text)
    _verify_skill_factory_revision(root)
    module_path = root / "src" / "skill_factory" / "programs" / "chartqa.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"the pinned skill-factory ChartQA program is missing: {module_path}"
        )
    module_name = (
        "_compass_chartqa_program_owner_"
        + hashlib.sha256(str(module_path).encode("utf-8")).hexdigest()[:16]
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import ChartQA owner program from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    program_class = getattr(module, "ChartQASingleStepProgram", None)
    if not isinstance(program_class, type) or not issubclass(program_class, dspy.Module):
        raise TypeError(
            "pinned skill-factory module has no DSPy ChartQASingleStepProgram"
        )
    return program_class


def load_chartqa_program_class(
    owner_root: str | Path | None = None,
) -> type[dspy.Module]:
    return _load_chartqa_program_class(str(_skill_factory_root(owner_root)))


@dataclass(frozen=True, slots=True)
class ChartQASplits:
    """Optimization splits from vis-nlp and the isolated LMMS official test."""

    train: tuple[dspy.Example, ...]
    validation: tuple[dspy.Example, ...]
    test: tuple[dspy.Example, ...]


@dataclass(frozen=True, slots=True)
class ChartQAOwnerComposition:
    """The owner program, data, and metric required by a COMPASS run."""

    program: dspy.Module
    splits: ChartQASplits
    metric: Callable[..., float]
    metric_with_feedback: Callable[..., dict[str, Any]]


@lru_cache(maxsize=1)
def _owner_task_config() -> Mapping[str, Any]:
    task_dir = Path(inspect.getfile(chartqa_doc_to_text)).resolve().parent
    return lmms_utils.load_yaml_config(
        yaml_path=str(task_dir / "chartqa.yaml"),
        mode="simple",
    )


def chartqa_prompt_kwargs(model_name: str = "default") -> dict[str, Any]:
    """Resolve prompt arguments from the pinned LMMS task configuration."""

    profiles = _owner_task_config()["lmms_eval_specific_kwargs"]
    selected = profiles.get(model_name, profiles["default"])
    return dict(selected)


def _dataset_root(root: str | Path) -> Path:
    root = Path(root).resolve()
    nested = root / "ChartQA Dataset"
    return nested if nested.is_dir() else root


def _example_from_owner_doc(
    doc: Mapping[str, Any],
    *,
    prompt_kwargs: Mapping[str, Any],
) -> dspy.Example:
    visual = chartqa_doc_to_visual(doc)[0]
    prompt = chartqa_doc_to_text(doc, dict(prompt_kwargs))
    return dspy.Example(
        image=dspy.Image(visual),
        prompt=prompt,
        answer=str(doc["answer"]),
        chartqa_type=str(doc["type"]),
    ).with_inputs("image", "prompt")


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"ChartQA view row must be an object: {path}:{line_number}"
                )
            records.append(record)
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _load_prepared_manifest(root: Path) -> Mapping[str, Any]:
    manifest_path = root / "manifest.json"
    checksum_path = root / "manifest.json.sha256"
    expected_manifest_hash = checksum_path.read_text(encoding="utf-8").split()[0]
    actual_manifest_hash = _sha256_file(manifest_path)
    if actual_manifest_hash != expected_manifest_hash:
        raise RuntimeError(
            "ChartQA prepared manifest checksum mismatch: "
            f"expected={expected_manifest_hash}, actual={actual_manifest_hash}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"ChartQA manifest must be an object: {manifest_path}")
    if manifest.get("benchmark") != "ChartQA" or manifest.get("status") != "frozen":
        raise ValueError(
            "prepared ChartQA root must contain a frozen ChartQA manifest"
        )
    if (
        manifest.get("optimization_owner", {}).get("commit")
        != CHARTQA_OPTIMIZATION_OWNER_COMMIT
    ):
        raise RuntimeError(
            "prepared ChartQA optimization owner differs from the experiment pin"
        )
    if (
        manifest.get("formal_test", {}).get("lmms_eval_commit")
        != LMMS_EVAL_OWNER_COMMIT
    ):
        raise RuntimeError(
            "prepared ChartQA LMMS-Eval owner differs from the experiment pin"
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise RuntimeError("prepared ChartQA manifest has no files mapping")
    required = (
        "views/train_lite_inputs.jsonl",
        "views/train_lite_labels.jsonl",
        "views/val_lite_inputs.jsonl",
        "views/val_lite_labels.jsonl",
        str(manifest["formal_test"]["parquet"]),
    )
    for relative_path in required:
        entry = files.get(relative_path)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("sha256"), str):
            raise RuntimeError(
                f"prepared ChartQA manifest does not pin {relative_path}"
            )
        if _sha256_file(root / relative_path) != entry["sha256"]:
            raise RuntimeError(
                f"prepared ChartQA file checksum mismatch: {relative_path}"
            )
    return manifest


def _prepared_relative_file(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("prepared ChartQA image must be a non-empty relative path")
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"prepared ChartQA image path must be relative: {relative}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"prepared ChartQA image path escapes its frozen root: {relative}"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"prepared ChartQA image is missing: {resolved}")
    return resolved


def _join_prepared_view(
    root: Path,
    split: str,
    *,
    expected_count: int,
) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    inputs_path = root / "views" / f"{split}_lite_inputs.jsonl"
    labels_path = root / "views" / f"{split}_lite_labels.jsonl"
    inputs = _read_jsonl(inputs_path)
    labels = _read_jsonl(labels_path)
    if len(inputs) != len(labels):
        raise ValueError(
            f"prepared ChartQA {split} input/label counts differ: "
            f"{len(inputs)} != {len(labels)}"
        )
    if len(inputs) != expected_count:
        raise ValueError(
            f"prepared ChartQA {split} count differs from frozen manifest: "
            f"{len(inputs)} != {expected_count}"
        )

    joined: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    seen_ids: set[str] = set()
    for expected_position, (input_row, label_row) in enumerate(zip(inputs, labels)):
        input_id = input_row.get("id")
        label_id = label_row.get("id")
        input_position = input_row.get("split_position")
        label_position = label_row.get("split_position")
        if (
            not isinstance(input_id, str)
            or input_id != label_id
            or input_position != label_position
            or input_position != expected_position
        ):
            raise ValueError(
                f"prepared ChartQA {split} input/label identity mismatch at "
                f"position {expected_position}: "
                f"input=({input_id!r}, {input_position!r}), "
                f"label=({label_id!r}, {label_position!r})"
            )
        if input_id in seen_ids:
            raise ValueError(
                f"prepared ChartQA {split} contains duplicate id {input_id!r}"
            )
        seen_ids.add(input_id)
        joined.append((input_row, label_row))
    return tuple(joined)


def load_prepared_chartqa_optimization_split(
    prepared_root: str | Path,
    split: str,
    *,
    model_name: str = "default",
) -> tuple[dspy.Example, ...]:
    """Load a frozen lite optimization split without exposing its label view."""

    if split not in {"train", "val"}:
        raise ValueError("prepared ChartQA optimization split must be 'train' or 'val'")
    root = Path(prepared_root).expanduser().resolve()
    manifest = _load_prepared_manifest(root)
    selection_name = f"{split}_lite"
    manifest_count = int(manifest["selection"][selection_name]["count"])
    expected_count = CHARTQA_PREPARED_SPLIT_COUNTS[split]
    if manifest_count != expected_count:
        raise RuntimeError(
            f"prepared ChartQA {split} manifest count differs from the frozen "
            f"protocol: {manifest_count} != {expected_count}"
        )
    prompt_kwargs = chartqa_prompt_kwargs(model_name)
    converted_images: dict[Path, dspy.Image] = {}
    examples: list[dspy.Example] = []

    for input_row, label_row in _join_prepared_view(
        root,
        split,
        expected_count=expected_count,
    ):
        image_path = _prepared_relative_file(root, input_row.get("image"))
        image = converted_images.get(image_path)
        chartqa_type = f"{input_row['source']}_{split}"
        owner_doc = {
            "question": input_row["question"],
            "answer": label_row["answer"],
            "type": chartqa_type,
        }
        if image is None:
            with PILImage.open(image_path) as source_image:
                visual_doc = {**owner_doc, "image": source_image}
                image = dspy.Image(chartqa_doc_to_visual(visual_doc)[0])
            converted_images[image_path] = image
        prompt = chartqa_doc_to_text(owner_doc, dict(prompt_kwargs))
        examples.append(
            dspy.Example(
                image=image,
                prompt=prompt,
                answer=str(label_row["answer"]),
                chartqa_type=chartqa_type,
                key=input_row["id"],
                split_position=input_row["split_position"],
                source=input_row["source"],
                source_position=input_row.get("source_position"),
            ).with_inputs("image", "prompt")
        )
    return tuple(examples)


def _parquet_image_bytes(value: Any, *, position: int) -> bytes:
    if isinstance(value, Mapping):
        value = value.get("bytes")
    if isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, bytes):
        raise ValueError(
            "prepared ChartQA test image must come from embedded parquet bytes "
            f"at position {position}"
        )
    return value


def load_prepared_lmms_chartqa_test(
    prepared_root: str | Path,
    *,
    model_name: str = "default",
    load_dataset_fn: Callable[..., Iterable[Mapping[str, Any]]] = load_dataset,
) -> tuple[dspy.Example, ...]:
    """Load the frozen full LMMS test parquet while retaining owner semantics."""

    root = Path(prepared_root).expanduser().resolve()
    manifest = _load_prepared_manifest(root)
    formal_test = manifest["formal_test"]
    parquet_path = _prepared_relative_file(root, formal_test["parquet"])
    manifest_count = int(formal_test["row_count"])
    expected_count = CHARTQA_PREPARED_SPLIT_COUNTS["test"]
    if manifest_count != expected_count:
        raise RuntimeError(
            "prepared ChartQA test manifest count differs from the frozen "
            f"protocol: {manifest_count} != {expected_count}"
        )
    rows = load_dataset_fn(
        "parquet",
        data_files=str(parquet_path),
        split="train",
    )
    cast_column = getattr(rows, "cast_column", None)
    if callable(cast_column):
        rows = cast_column("image", DatasetImage(decode=False))

    prompt_kwargs = chartqa_prompt_kwargs(model_name)
    examples: list[dspy.Example] = []
    for position, row in enumerate(rows):
        doc = dict(row)
        image_bytes = _parquet_image_bytes(doc.get("image"), position=position)
        with PILImage.open(BytesIO(image_bytes)) as source_image:
            owner_doc = {**doc, "image": source_image}
            example = _example_from_owner_doc(
                owner_doc,
                prompt_kwargs=prompt_kwargs,
            )
        examples.append(
            dspy.Example(
                **dict(example),
                key=f"chartqa:test:{position}",
                split_position=position,
            ).with_inputs("image", "prompt")
        )
    if len(examples) != expected_count:
        raise ValueError(
            "prepared ChartQA test count differs from frozen manifest: "
            f"{len(examples)} != {expected_count}"
        )
    return tuple(examples)


def load_prepared_chartqa_splits(
    prepared_root: str | Path,
    *,
    model_name: str = "default",
    load_dataset_fn: Callable[..., Iterable[Mapping[str, Any]]] = load_dataset,
) -> ChartQASplits:
    """Load the frozen 150/300 optimization views and full 2,500-row test."""

    return ChartQASplits(
        train=load_prepared_chartqa_optimization_split(
            prepared_root,
            "train",
            model_name=model_name,
        ),
        validation=load_prepared_chartqa_optimization_split(
            prepared_root,
            "val",
            model_name=model_name,
        ),
        test=load_prepared_lmms_chartqa_test(
            prepared_root,
            model_name=model_name,
            load_dataset_fn=load_dataset_fn,
        ),
    )


def load_vis_nlp_chartqa_split(
    root: str | Path,
    split: str,
    *,
    model_name: str = "default",
) -> tuple[dspy.Example, ...]:
    """Load one official vis-nlp optimization split (``train`` or ``val``)."""

    if split not in {"train", "val"}:
        raise ValueError("vis-nlp optimization split must be 'train' or 'val'")

    split_dir = _dataset_root(root) / split
    prompt_kwargs = chartqa_prompt_kwargs(model_name)
    examples: list[dspy.Example] = []
    converted_images: dict[Path, dspy.Image] = {}

    for source in ("human", "augmented"):
        records_path = split_dir / f"{split}_{source}.json"
        with records_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)

        for record in records:
            image_path = split_dir / "png" / record["imgname"]
            image = converted_images.get(image_path)
            doc = {
                "question": record["query"],
                "answer": record["label"],
                "type": f"{source}_{split}",
            }
            if image is None:
                with PILImage.open(image_path) as source_image:
                    visual_doc = {**doc, "image": source_image}
                    image = dspy.Image(chartqa_doc_to_visual(visual_doc)[0])
                converted_images[image_path] = image

            prompt = chartqa_doc_to_text(doc, prompt_kwargs)
            examples.append(
                dspy.Example(
                    image=image,
                    prompt=prompt,
                    answer=str(doc["answer"]),
                    chartqa_type=str(doc["type"]),
                ).with_inputs("image", "prompt")
            )

    return tuple(examples)


def load_lmms_chartqa_test(
    *,
    model_name: str = "default",
    load_dataset_fn: Callable[..., Iterable[Mapping[str, Any]]] = load_dataset,
) -> tuple[dspy.Example, ...]:
    """Load only the official LMMS-Eval ChartQA test split."""

    config = _owner_task_config()
    dataset_kwargs = dict(config.get("dataset_kwargs") or {})
    rows = load_dataset_fn(
        config["dataset_path"],
        split=config["test_split"],
        **dataset_kwargs,
    )
    prompt_kwargs = chartqa_prompt_kwargs(model_name)
    return tuple(
        _example_from_owner_doc(dict(row), prompt_kwargs=prompt_kwargs)
        for row in rows
    )


def load_chartqa_splits(
    vis_nlp_root: str | Path,
    *,
    model_name: str = "default",
    load_dataset_fn: Callable[..., Iterable[Mapping[str, Any]]] = load_dataset,
) -> ChartQASplits:
    """Compose official optimization data with the isolated official test."""

    return ChartQASplits(
        train=load_vis_nlp_chartqa_split(
            vis_nlp_root,
            "train",
            model_name=model_name,
        ),
        validation=load_vis_nlp_chartqa_split(
            vis_nlp_root,
            "val",
            model_name=model_name,
        ),
        test=load_lmms_chartqa_test(
            model_name=model_name,
            load_dataset_fn=load_dataset_fn,
        ),
    )


def chartqa_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Any = None,
) -> float:
    """Delegate the per-instance score to pinned LMMS-Eval."""

    del trace
    result = chartqa_process_results(
        {
            "answer": example.answer,
            "type": example.chartqa_type,
        },
        [prediction.answer],
    )
    return float(result["relaxed_overall"])


def chartqa_metric_with_feedback(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Any = None,
) -> dict[str, Any]:
    """Expose only gold, prediction, and the owner score to reflection."""

    score = chartqa_metric(example, prediction, trace)
    return {
        "score": score,
        "feedback": (
            f"gold={example.answer!r}\n"
            f"pred={prediction.answer!r}\n"
            f"official_score={score}"
        ),
    }


def build_chartqa_owner_composition(
    lm: dspy.BaseLM,
    vis_nlp_root: str | Path,
    *,
    program_owner_root: str | Path | None = None,
    model_name: str = "default",
    load_dataset_fn: Callable[..., Iterable[Mapping[str, Any]]] = load_dataset,
) -> ChartQAOwnerComposition:
    """Build the minimal owner-owned ChartQA surface used by COMPASS."""

    return ChartQAOwnerComposition(
        program=load_chartqa_program_class(program_owner_root)(lm),
        splits=load_chartqa_splits(
            vis_nlp_root,
            model_name=model_name,
            load_dataset_fn=load_dataset_fn,
        ),
        metric=chartqa_metric,
        metric_with_feedback=chartqa_metric_with_feedback,
    )


def build_prepared_chartqa_owner_composition(
    lm: dspy.BaseLM,
    prepared_root: str | Path,
    *,
    program_owner_root: str | Path | None = None,
    model_name: str = "default",
    load_dataset_fn: Callable[..., Iterable[Mapping[str, Any]]] = load_dataset,
) -> ChartQAOwnerComposition:
    """Build ChartQA from the frozen separated views and local test parquet."""

    return ChartQAOwnerComposition(
        program=load_chartqa_program_class(program_owner_root)(lm),
        splits=load_prepared_chartqa_splits(
            prepared_root,
            model_name=model_name,
            load_dataset_fn=load_dataset_fn,
        ),
        metric=chartqa_metric,
        metric_with_feedback=chartqa_metric_with_feedback,
    )
