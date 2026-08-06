from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import dspy
from gepa_artifact.benchmarks import dspy_program
from gepa_artifact.benchmarks.benchmark import Benchmark, BenchmarkMeta


HITAB_OWNER_COMMIT = "d179602662b490249baf068a76fbe4137029126e"
HITAB_ROOT_ENV = "HITAB_ROOT"
_DEFAULT_HITAB_ROOT = Path(__file__).resolve().parents[1] / "upstreams" / "hitab"


@dataclass(frozen=True, slots=True)
class HiTabSplits:
    train: tuple[dspy.Example, ...]
    validation: tuple[dspy.Example, ...]
    test: tuple[dspy.Example, ...]


@dataclass(frozen=True, slots=True)
class HiTabOwnerComposition:
    program: dspy.Module
    splits: HiTabSplits
    metric: Callable[..., float]
    metric_with_feedback: Callable[..., dict[str, Any]]
    manifest: Mapping[str, Any]


def _owner_root(owner_root: str | os.PathLike[str] | None = None) -> Path:
    configured = owner_root or os.environ.get(HITAB_ROOT_ENV)
    return Path(configured or _DEFAULT_HITAB_ROOT).expanduser().resolve()


def _verify_owner_revision(owner_root: Path) -> None:
    """Reject a populated Git checkout at a different owner revision."""

    if not (owner_root / ".git").exists():
        return
    result = subprocess.run(
        ["git", "-C", str(owner_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cannot resolve the pinned HiTab checkout at {owner_root}: "
            f"{result.stderr.strip()}"
        )
    revision = result.stdout.strip()
    if revision != HITAB_OWNER_COMMIT:
        raise RuntimeError(
            "HiTab owner revision differs from the experiment pin: "
            f"expected={HITAB_OWNER_COMMIT}, actual={revision}"
        )


@lru_cache(maxsize=None)
def _load_owner_hmt_score(owner_root_text: str) -> Callable[[Any, Any], float]:
    owner_root = Path(owner_root_text)
    _verify_owner_revision(owner_root)
    module_path = owner_root / "qa" / "table" / "utils.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            "the pinned microsoft/HiTab scorer is missing: "
            f"{module_path}"
        )

    module_name = (
        "_compass_hitab_owner_"
        + hashlib.sha256(str(module_path).encode("utf-8")).hexdigest()[:16]
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import HiTab scorer from {module_path}")
    module = importlib.util.module_from_spec(spec)

    # ``qa.table.utils`` imports another file from the same owner checkout.
    # The checkout is exposed only while that official module is executed.
    sys.path.insert(0, str(owner_root))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(owner_root))

    scorer = getattr(module, "hmt_score", None)
    if not callable(scorer):
        raise AttributeError(
            f"pinned microsoft/HiTab module has no callable hmt_score: {module_path}"
        )
    return scorer


def load_owner_hmt_score(
    owner_root: str | os.PathLike[str] | None = None,
) -> Callable[[Any, Any], float]:
    """Dynamically load the metric owned by the pinned HiTab checkout."""

    return _load_owner_hmt_score(str(_owner_root(owner_root)))


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_prepared_manifest(prepared_root: Path) -> Mapping[str, Any]:
    """Verify the immutable prepared view without redefining owner semantics."""

    manifest_path = prepared_root / "manifest.json"
    checksum_path = prepared_root / "manifest.json.sha256"
    expected_manifest_hash = checksum_path.read_text(encoding="utf-8").split()[0]
    actual_manifest_hash = _sha256_file(manifest_path)
    if actual_manifest_hash != expected_manifest_hash:
        raise RuntimeError(
            "HiTab prepared manifest checksum mismatch: "
            f"expected={expected_manifest_hash}, actual={actual_manifest_hash}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen":
        raise RuntimeError("HiTab prepared dataset manifest is not frozen")
    if manifest.get("owner_commit") != HITAB_OWNER_COMMIT:
        raise RuntimeError(
            "HiTab prepared dataset owner revision differs from the experiment pin"
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("HiTab prepared manifest has no files mapping")
    required = (
        "owner/data/processed_input/tables.jsonl",
        "owner/qa/datadump/utils.py",
        "owner/qa/table/utils.py",
        "views/train_lite_inputs.jsonl",
        "views/train_lite_labels.jsonl",
        "views/dev_lite_inputs.jsonl",
        "views/dev_lite_labels.jsonl",
        "views/test_official_full_inputs.jsonl",
        "views/test_official_full_labels.jsonl",
    )
    for relative_path in required:
        entry = files.get(relative_path)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("sha256"), str):
            raise RuntimeError(
                f"HiTab prepared manifest does not pin {relative_path}"
            )
        actual = _sha256_file(prepared_root / relative_path)
        if actual != entry["sha256"]:
            raise RuntimeError(
                f"HiTab prepared file checksum mismatch: {relative_path}"
            )
    return manifest


def _join_prepared_split(
    prepared_root: Path,
    input_filename: str,
    label_filename: str,
    tables: Mapping[str, Mapping[str, Any]],
) -> tuple[dspy.Example, ...]:
    inputs = _read_jsonl(prepared_root / "views" / input_filename)
    labels = _read_jsonl(prepared_root / "views" / label_filename)
    if len(inputs) != len(labels):
        raise ValueError(
            f"HiTab prepared inputs/labels length mismatch for {input_filename}"
        )
    examples: list[dspy.Example] = []
    for input_record, label_record in zip(inputs, labels, strict=True):
        identity = (input_record.get("id"), input_record.get("split_position"))
        label_identity = (
            label_record.get("id"),
            label_record.get("split_position"),
        )
        if identity != label_identity:
            raise ValueError(
                "HiTab prepared inputs/labels identity mismatch: "
                f"input={identity!r}, label={label_identity!r}"
            )
        table_id = str(input_record["table_id"])
        try:
            table_artifact = tables[table_id]
        except KeyError as error:
            raise ValueError(
                f"HiTab prepared input references missing table {table_id!r}"
            ) from error
        table_text = json.dumps(
            table_artifact,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        examples.append(
            dspy.Example(
                key=str(input_record["id"]),
                split_position=int(input_record["split_position"]),
                hitab_table_id=table_id,
                question=str(input_record["question"]),
                table=table_text,
                answer=label_record["answer"],
            ).with_inputs("question", "table")
        )
    return tuple(examples)


def load_prepared_hitab_splits(
    prepared_root: str | os.PathLike[str],
) -> tuple[HiTabSplits, Mapping[str, Any]]:
    """Load the fixed 150/300/full-test views from a verified prepared root."""

    root = Path(prepared_root).expanduser().resolve()
    manifest = _verify_prepared_manifest(root)
    tables = {
        str(record["name"]): record["kg"]
        for record in _read_jsonl(
            root / "owner" / "data" / "processed_input" / "tables.jsonl"
        )
    }
    splits = HiTabSplits(
        train=_join_prepared_split(
            root,
            "train_lite_inputs.jsonl",
            "train_lite_labels.jsonl",
            tables,
        ),
        validation=_join_prepared_split(
            root,
            "dev_lite_inputs.jsonl",
            "dev_lite_labels.jsonl",
            tables,
        ),
        test=_join_prepared_split(
            root,
            "test_official_full_inputs.jsonl",
            "test_official_full_labels.jsonl",
            tables,
        ),
    )
    expected_sizes = {"train": 150, "validation": 300, "test": 1584}
    actual_sizes = {
        "train": len(splits.train),
        "validation": len(splits.validation),
        "test": len(splits.test),
    }
    if actual_sizes != expected_sizes:
        raise RuntimeError(
            "HiTab prepared split sizes differ from the frozen protocol: "
            f"expected={expected_sizes}, actual={actual_sizes}"
        )
    return splits, manifest


def _load_table_artifacts(owner_root: Path) -> dict[str, Mapping[str, Any]]:
    table_path = owner_root / "data" / "processed_input" / "tables.jsonl"
    tables: dict[str, Mapping[str, Any]] = {}
    for record in _read_jsonl(table_path):
        name = str(record["name"])
        # The official preprocessing stores the complete source HMT artifact
        # under ``kg``. Sample-specific links/formulas never enter this object.
        tables[name] = record["kg"]
    return tables


def _load_split(
    owner_root: Path,
    filename: str,
    tables: Mapping[str, Mapping[str, Any]],
) -> list[dspy.Example]:
    split_path = owner_root / "data" / filename
    examples: list[dspy.Example] = []
    for sample in _read_jsonl(split_path):
        table_id = str(sample["table_id"])
        table_artifact = tables[table_id]
        table_text = json.dumps(
            table_artifact,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        examples.append(
            dspy.Example(
                question=sample["question"],
                table=table_text,
                answer=sample["answer"],
            ).with_inputs("question", "table")
        )
    return examples


class HiTabBenchmark(Benchmark):
    """HiTab's owner splits exposed through the GEPA artifact benchmark seam."""

    # The public registry uses this to forbid its legacy optimizer-seed
    # train/validation remix for this benchmark.
    preserve_official_splits = True

    def __init__(
        self,
        dataset_mode: str = "lite",
        *,
        owner_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.owner_root = _owner_root(owner_root)
        _verify_owner_revision(self.owner_root)
        super().__init__(dataset_mode=dataset_mode)

    def init_dataset(self) -> None:
        tables = _load_table_artifacts(self.owner_root)
        self.train_set = _load_split(
            self.owner_root,
            "train_samples.jsonl",
            tables,
        )
        self.val_set = _load_split(
            self.owner_root,
            "dev_samples.jsonl",
            tables,
        )
        self.test_set = _load_split(
            self.owner_root,
            "test_samples.jsonl",
            tables,
        )
        self.dataset = self.train_set + self.val_set + self.test_set

    def create_splits(self) -> None:
        # The owner already publishes train/dev/test. In particular, even the
        # framework's debugging mode must not merge or repartition them.
        return


class AnswerHiTab(dspy.Signature):
    """Answer from the complete HiTab table artifact and return the denotation."""

    question = dspy.InputField()
    table = dspy.InputField()
    answer = dspy.OutputField()


class HiTabProgram(dspy_program.LangProBeDSPyMetaProgram):
    """One owner-neutral predictor; COMPASS optimizes only its instruction."""

    def __init__(self) -> None:
        self.answer = dspy.Predict(AnswerHiTab)

    def forward(self, question: str, table: str) -> dspy.Prediction:
        return self.answer(question=question, table=table)


def _prediction_answer(prediction: Any) -> Any:
    if isinstance(prediction, Mapping):
        return prediction["answer"]
    return prediction.answer


def hitab_metric(
    example: Any,
    prediction: Any,
    trace: Any = None,
    *,
    owner_root: str | os.PathLike[str] | None = None,
) -> float:
    """Score the raw predictor value with HiTab's exact owner function."""

    del trace
    score = load_owner_hmt_score(owner_root)(
        _prediction_answer(prediction),
        example.answer,
    )
    return float(score)


def hitab_metric_with_feedback(
    example: Any,
    prediction: Any,
    trace: Any = None,
    *,
    owner_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return only gold, prediction, and the owner-computed score."""

    del trace
    predicted_answer = _prediction_answer(prediction)
    score = float(
        load_owner_hmt_score(owner_root)(predicted_answer, example.answer)
    )
    return {
        "score": score,
        "feedback": (
            f"gold={example.answer!r}\n"
            f"pred={predicted_answer!r}\n"
            f"owner_score={score!r}"
        ),
    }


def build_prepared_hitab_owner_composition(
    prepared_root: str | os.PathLike[str],
) -> HiTabOwnerComposition:
    """Bind the frozen prepared views to the pinned owner scorer."""

    root = Path(prepared_root).expanduser().resolve()
    splits, manifest = load_prepared_hitab_splits(root)
    metric_owner_root = root / "owner"
    return HiTabOwnerComposition(
        program=program,
        splits=splits,
        metric=partial(hitab_metric, owner_root=metric_owner_root),
        metric_with_feedback=partial(
            hitab_metric_with_feedback,
            owner_root=metric_owner_root,
        ),
        manifest=manifest,
    )


program = HiTabProgram()
benchmark_meta = BenchmarkMeta(
    benchmark=HiTabBenchmark,
    program=[program],
    metric=hitab_metric,
    name="HiTab",
    metric_with_feedback=hitab_metric_with_feedback,
)


__all__ = [
    "AnswerHiTab",
    "HITAB_OWNER_COMMIT",
    "HITAB_ROOT_ENV",
    "HiTabBenchmark",
    "HiTabOwnerComposition",
    "HiTabProgram",
    "HiTabSplits",
    "benchmark_meta",
    "build_prepared_hitab_owner_composition",
    "hitab_metric",
    "hitab_metric_with_feedback",
    "load_prepared_hitab_splits",
    "load_owner_hmt_score",
    "program",
]
