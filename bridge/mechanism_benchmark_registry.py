"""Lazy benchmark registry for the three COMPASS mechanism experiments.

This registry is deliberately separate from :mod:`paper_benchmark_registry`:
the latter remains the exact six-task GEPA artifact registry.  Each resolver
below only binds pinned owner data/program/metric surfaces to the common paper
runner; it does not implement benchmark semantics.
"""

from __future__ import annotations

import hashlib
import inspect
import importlib.util
import math
import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

from bridge.chartqa_protocol import FROZEN_SPLIT_FINGERPRINTS
from bridge.paper_benchmark_registry import OfficialDatasetSplits, split_fingerprint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

CHARTQA_ROOT_ENV = "CHARTQA_ROOT"
CHARTQA_PREPARED_ROOT_ENV = "CHARTQA_PREPARED_ROOT"
HITAB_PREPARED_ROOT_ENV = "HITAB_PREPARED_ROOT"
CLUTRR_HF_DATA_ROOT_ENV = "CLUTRR_HF_DATA_ROOT"
CLUTRR_BASELINE_ROOT_ENV = "CLUTRR_BASELINE_ROOT"

CHARTQA_OWNER_COMMIT = "044eabfc306abfe9340c5741f0093aefc5973d06"
CHARTQA_LMMS_OWNER_COMMIT = "cb45ac4d4a667ea5ef89c7a148bff69b3489b981"
CHARTQA_PROGRAM_OWNER_COMMIT = "e23f82f9c7ed2eacfb9124b92143358a9953263c"
CHARTQA_LMMS_OWNER_BLOBS: Mapping[str, str] = {
    "lmms_eval/tasks/chartqa/chartqa.yaml": (
        "2a42f4dfd808730b475117b193b57931300c12b4"
    ),
    "lmms_eval/tasks/chartqa/utils.py": "cdba4c4118311dcf5c175334bbf8c1f95ea29fa8",
}
CLUTRR_HF_DATA_COMMIT = "e5b496941e91abb7c319d2618a3ce96752bc4ab7"
CLUTRR_BASELINE_OWNER_COMMIT = "303ed9a48f82a59b4eb34ac5bd5866f0d82c5552"


CLUTRR_HF_VARIANT_DIRECTORY: Mapping[str, str] = {
    "clutrr_supporting": "rob_train_sup_23_test_all_23",
    "clutrr_irrelevant": "rob_train_irr_23_test_all_23",
    "clutrr_disconnected": "rob_train_disc_23_test_all_23",
}


@dataclass(frozen=True, slots=True)
class MechanismOwnerRoots:
    """Runtime locations of pinned owner data/metric checkouts."""

    chartqa: Path
    clutrr_hf_data: Path
    clutrr_baseline: Path
    hitab_prepared: Path | None = None
    chartqa_prepared: Path | None = None

    @classmethod
    def from_environment(cls) -> "MechanismOwnerRoots":
        return cls(
            chartqa=Path(
                os.environ.get(
                    CHARTQA_ROOT_ENV,
                    PROJECT_ROOT / "upstreams" / "chartqa",
                )
            ).expanduser().resolve(),
            clutrr_hf_data=Path(
                os.environ.get(
                    CLUTRR_HF_DATA_ROOT_ENV,
                    PROJECT_ROOT / "upstreams" / "clutrr-hf-data",
                )
            ).expanduser().resolve(),
            clutrr_baseline=Path(
                os.environ.get(
                    CLUTRR_BASELINE_ROOT_ENV,
                    PROJECT_ROOT / "upstreams" / "clutrr-baselines",
                )
            ).expanduser().resolve(),
            hitab_prepared=(
                Path(os.environ[HITAB_PREPARED_ROOT_ENV]).expanduser().resolve()
                if os.environ.get(HITAB_PREPARED_ROOT_ENV)
                else None
            ),
            chartqa_prepared=(
                Path(os.environ[CHARTQA_PREPARED_ROOT_ENV]).expanduser().resolve()
                if os.environ.get(CHARTQA_PREPARED_ROOT_ENV)
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class MechanismBenchmarkDefinition:
    task_id: str
    display_name: str
    benchmark_class_name: str


@dataclass(frozen=True, slots=True)
class ResolvedMechanismBenchmark:
    """Owner composition in the shape consumed by the shared paper runner."""

    task_id: str
    display_name: str
    benchmark_class_name: str
    program: Any
    metric: Callable[..., Any]
    metric_with_feedback: Callable[..., Any]
    splits: OfficialDatasetSplits
    provenance: Mapping[str, Any]
    custom_instruction_proposer: Any | None = None
    score_breakdown: Callable[[Any, Any], Mapping[str, Any]] | None = None
    num_threads: int | None = None
    feedback_fn_maps: None = None
    program_index: int = 0

    @property
    def benchmark_meta(self) -> "ResolvedMechanismBenchmark":
        """Expose the owner callbacks through the existing registry protocol."""

        return self

    @property
    def program_class_name(self) -> str:
        return getattr(self.program, "_name", self.program.__class__.__name__)

    @property
    def predictor_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.program.named_predictors())


_MECHANISM_DEFINITIONS: Mapping[str, MechanismBenchmarkDefinition] = {
    "hitab": MechanismBenchmarkDefinition(
        task_id="hitab",
        display_name="HiTab",
        benchmark_class_name="HiTabBenchmark",
    ),
    "chartqa": MechanismBenchmarkDefinition(
        task_id="chartqa",
        display_name="ChartQA",
        benchmark_class_name="ChartQAOwnerComposition",
    ),
    "clutrr_supporting": MechanismBenchmarkDefinition(
        task_id="clutrr_supporting",
        display_name="CLUTRR Supporting Noise",
        benchmark_class_name="ClutrrDatasetSplits",
    ),
    "clutrr_irrelevant": MechanismBenchmarkDefinition(
        task_id="clutrr_irrelevant",
        display_name="CLUTRR Irrelevant Noise",
        benchmark_class_name="ClutrrDatasetSplits",
    ),
    "clutrr_disconnected": MechanismBenchmarkDefinition(
        task_id="clutrr_disconnected",
        display_name="CLUTRR Disconnected Noise",
        benchmark_class_name="ClutrrDatasetSplits",
    ),
}


def load_mechanism_benchmark_specs() -> dict[str, MechanismBenchmarkDefinition]:
    """Return only the separately reviewed mechanism tasks."""

    return dict(_MECHANISM_DEFINITIONS)


def _verify_git_revision(root: Path, expected: str, owner: str) -> None:
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
            f"cannot resolve the pinned {owner} checkout at {root}: "
            f"{result.stderr.strip()}"
        )
    actual = result.stdout.strip()
    if actual != expected:
        raise RuntimeError(
            f"{owner} revision differs from the experiment pin: "
            f"expected={expected}, actual={actual}"
        )


def _verify_git_blobs(
    root: Path,
    expected_blobs: Mapping[str, str],
    owner: str,
) -> None:
    """Reject dirty owner files while respecting the checkout's Git filters."""

    if not (root / ".git").exists():
        raise RuntimeError(f"pinned {owner} checkout has no Git metadata: {root}")
    for relative_path, expected_blob in expected_blobs.items():
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "hash-object",
                f"--path={relative_path}",
                relative_path,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        actual_blob = result.stdout.strip()
        if result.returncode != 0 or actual_blob != expected_blob:
            raise RuntimeError(
                f"{owner} file differs from the pinned owner blob: "
                f"path={relative_path}, expected={expected_blob}, "
                f"actual={actual_blob or result.stderr.strip()}"
            )


def _freeze_split_parts(
    train: Any,
    validation: Any,
    test: Any,
) -> OfficialDatasetSplits:
    train = tuple(train)
    validation = tuple(validation)
    test = tuple(test)
    return OfficialDatasetSplits(
        train=train,
        validation=validation,
        test=test,
        fingerprints={
            "train": split_fingerprint(train),
            "validation": split_fingerprint(validation),
            "test": split_fingerprint(test),
        },
    )


def _freeze_splits(splits: Any) -> OfficialDatasetSplits:
    return _freeze_split_parts(splits.train, splits.validation, splits.test)


def _freeze_benchmark_sets(benchmark: Any) -> OfficialDatasetSplits:
    return _freeze_split_parts(
        benchmark.train_set,
        benchmark.val_set,
        benchmark.test_set,
    )


def clutrr_task_score_summary(
    examples: Any,
    scores: Any,
) -> dict[str, Any]:
    """Report owner-scored CLUTRR accuracy by owner task and macro mean."""

    examples = tuple(examples)
    values = tuple(float(score) for score in scores)
    if len(examples) != len(values):
        raise ValueError("CLUTRR examples and owner scores must have equal length")
    grouped: dict[int, list[float]] = {}
    for example, score in zip(examples, values, strict=True):
        task_number = int(example.clutrr_task_number)
        grouped.setdefault(task_number, []).append(score)
    per_task = {
        f"task_{task_number}": 100.0 * math.fsum(task_scores) / len(task_scores)
        for task_number, task_scores in sorted(grouped.items())
    }
    return {
        "macro_percent": math.fsum(per_task.values()) / len(per_task),
        "per_owner_task_percent": per_task,
    }


def chartqa_type_score_summary(
    examples: Any,
    scores: Any,
) -> dict[str, float]:
    """Aggregate owner-produced ChartQA scores without replacing its metric."""

    examples = tuple(examples)
    values = tuple(float(score) for score in scores)
    if len(examples) != len(values) or not values:
        raise ValueError(
            "ChartQA examples and owner scores must be non-empty and aligned"
        )
    grouped: dict[str, list[float]] = {"human": [], "augmented": []}
    for example, score in zip(examples, values, strict=True):
        source = str(example.chartqa_type).split("_", 1)[0]
        if source not in grouped:
            raise ValueError(f"unexpected ChartQA owner type: {example.chartqa_type!r}")
        grouped[source].append(score)
    if any(not group_scores for group_scores in grouped.values()):
        raise ValueError("ChartQA owner score vector is missing a required subgroup")
    return {
        "relaxed_overall_percent": 100.0 * math.fsum(values) / len(values),
        "relaxed_human_split_percent": (
            100.0 * math.fsum(grouped["human"]) / len(grouped["human"])
        ),
        "relaxed_augmented_split_percent": (
            100.0 * math.fsum(grouped["augmented"])
            / len(grouped["augmented"])
        ),
    }


@lru_cache(maxsize=None)
def _load_clutrr_relation_overlap_cached(root_text: str) -> Callable[..., Any]:
    root = Path(root_text)
    _verify_git_revision(
        root,
        CLUTRR_BASELINE_OWNER_COMMIT,
        "koustuvsinha/clutrr-baselines",
    )
    module_path = root / "codes" / "metric" / "quality_metric.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"the pinned CLUTRR baseline metric is missing: {module_path}"
        )
    module_name = (
        "_compass_clutrr_quality_metric_"
        + hashlib.sha256(str(module_path).encode("utf-8")).hexdigest()[:16]
    )
    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot import CLUTRR metric from {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    quality_metric = getattr(module, "QualityMetric", None)
    relation_overlap = getattr(quality_metric, "relation_overlap", None)
    if not callable(relation_overlap):
        raise AttributeError(
            "pinned CLUTRR baseline has no callable "
            f"QualityMetric.relation_overlap: {module_path}"
        )
    # relation_overlap does not read instance state. Binding only its unused
    # ``self`` argument preserves the exact owner method without constructing
    # the unrelated entity-overlap data dependency from QualityMetric.__init__.
    return partial(relation_overlap, None)


def _load_clutrr_relation_overlap(root: Path) -> Callable[..., Any]:
    return _load_clutrr_relation_overlap_cached(str(root.resolve()))


def _resolve_hitab(
    definition: MechanismBenchmarkDefinition,
    *,
    roots: MechanismOwnerRoots,
) -> ResolvedMechanismBenchmark:
    from bridge.mechanism_hitab import (
        HITAB_OWNER_COMMIT,
        build_prepared_hitab_owner_composition,
    )

    if roots.hitab_prepared is None:
        raise RuntimeError(
            f"{HITAB_PREPARED_ROOT_ENV} must point to the frozen HiTab prepared root"
    )
    composition = build_prepared_hitab_owner_composition(roots.hitab_prepared)
    return ResolvedMechanismBenchmark(
        task_id=definition.task_id,
        display_name=definition.display_name,
        benchmark_class_name=definition.benchmark_class_name,
        program=composition.program,
        metric=composition.metric,
        metric_with_feedback=composition.metric_with_feedback,
        splits=_freeze_splits(composition.splits),
        provenance={
            "dataset_and_metric": "microsoft/HiTab",
            "owner_commit": HITAB_OWNER_COMMIT,
            "prepared_manifest_sha256": hashlib.sha256(
                (roots.hitab_prepared / "manifest.json").read_bytes()
            ).hexdigest(),
            "split_policy": "fixed_first_150_train_first_300_dev_full_test",
        },
    )


def _resolve_chartqa(
    definition: MechanismBenchmarkDefinition,
    *,
    lm: Any,
    roots: MechanismOwnerRoots,
) -> ResolvedMechanismBenchmark:
    from dspy.teleprompt.gepa.instruction_proposal import (
        MultiModalInstructionProposer,
    )

    from bridge.mechanism_chartqa import (
        build_prepared_chartqa_owner_composition,
        chartqa_doc_to_text,
    )

    _verify_git_revision(
        roots.chartqa,
        CHARTQA_OWNER_COMMIT,
        "vis-nlp/ChartQA",
    )
    if roots.chartqa_prepared is None:
        raise RuntimeError(
            f"{CHARTQA_PREPARED_ROOT_ENV} must point to the frozen ChartQA "
            "prepared root"
        )
    lmms_root = Path(inspect.getfile(chartqa_doc_to_text)).resolve().parents[3]
    _verify_git_revision(
        lmms_root,
        CHARTQA_LMMS_OWNER_COMMIT,
        "lmms-eval",
    )
    _verify_git_blobs(lmms_root, CHARTQA_LMMS_OWNER_BLOBS, "lmms-eval")
    composition = build_prepared_chartqa_owner_composition(
        lm,
        roots.chartqa_prepared,
    )
    split_sizes = (
        len(composition.splits.train),
        len(composition.splits.validation),
        len(composition.splits.test),
    )
    if split_sizes != (150, 300, 2500):
        raise RuntimeError(
            "ChartQA prepared split sizes differ from the frozen protocol: "
            f"expected=(150, 300, 2500), actual={split_sizes}"
        )
    frozen_splits = _freeze_splits(composition.splits)
    if dict(frozen_splits.fingerprints) != FROZEN_SPLIT_FINGERPRINTS:
        raise RuntimeError(
            "ChartQA loaded split fingerprints differ from the frozen paper data: "
            f"expected={FROZEN_SPLIT_FINGERPRINTS}, "
            f"actual={dict(frozen_splits.fingerprints)}"
        )
    return ResolvedMechanismBenchmark(
        task_id=definition.task_id,
        display_name=definition.display_name,
        benchmark_class_name=definition.benchmark_class_name,
        program=composition.program,
        metric=composition.metric,
        metric_with_feedback=composition.metric_with_feedback,
        splits=frozen_splits,
        custom_instruction_proposer=MultiModalInstructionProposer(),
        score_breakdown=chartqa_type_score_summary,
        provenance={
            "optimization_splits": "vis-nlp/ChartQA",
            "optimization_owner_commit": CHARTQA_OWNER_COMMIT,
            "program": "skill-factory/ChartQASingleStepProgram",
            "program_owner_commit": CHARTQA_PROGRAM_OWNER_COMMIT,
            "test_metric": "lmms-eval/chartqa",
            "test_metric_owner_commit": CHARTQA_LMMS_OWNER_COMMIT,
            "test_metric_owner_blobs": dict(CHARTQA_LMMS_OWNER_BLOBS),
            "reflection_proposer": (
                "dspy.teleprompt.gepa.instruction_proposal."
                "MultiModalInstructionProposer"
            ),
            "prepared_manifest_sha256": hashlib.sha256(
                (roots.chartqa_prepared / "manifest.json").read_bytes()
            ).hexdigest(),
            "split_policy": (
                "balanced_75_75_train_150_150_val_and_full_isolated_test"
            ),
        },
    )


def _resolve_clutrr(
    definition: MechanismBenchmarkDefinition,
    *,
    roots: MechanismOwnerRoots,
) -> ResolvedMechanismBenchmark:
    from bridge.mechanism_clutrr import (
        CLUTRR_BASELINE_OWNER_COMMIT as ADAPTER_BASELINE_COMMIT,
        CLUTRR_OWNER_COMMIT,
        ClutrrFixedSplitPaths,
        ClutrrOwnerMetric,
        ClutrrVariant,
        load_fixed_clutrr_splits,
        program,
    )

    if ADAPTER_BASELINE_COMMIT != CLUTRR_BASELINE_OWNER_COMMIT:
        raise RuntimeError("CLUTRR adapter and registry baseline pins differ")
    _verify_git_revision(
        roots.clutrr_hf_data,
        CLUTRR_HF_DATA_COMMIT,
        "kliang5/CLUTRR_huggingface_dataset",
    )
    variant_by_task = {
        "clutrr_supporting": ClutrrVariant.SUPPORTING,
        "clutrr_irrelevant": ClutrrVariant.IRRELEVANT,
        "clutrr_disconnected": ClutrrVariant.DISCONNECTED,
    }
    variant = variant_by_task[definition.task_id]
    split_root = roots.clutrr_hf_data / CLUTRR_HF_VARIANT_DIRECTORY[
        definition.task_id
    ]
    splits = load_fixed_clutrr_splits(
        ClutrrFixedSplitPaths(
            train=split_root / "train.csv",
            validation=split_root / "validation.csv",
            test=split_root / "test.csv",
        ),
        variant=variant,
        relation_length=None,
        label_column="target_text",
    )
    # Mechanism runs use the existing paper-lite optimization envelope while
    # retaining the backing repository's fixed order.  The complete owner test
    # matrix remains isolated for final evaluation.
    metric = ClutrrOwnerMetric(
        _load_clutrr_relation_overlap(roots.clutrr_baseline)
    )
    return ResolvedMechanismBenchmark(
        task_id=definition.task_id,
        display_name=definition.display_name,
        benchmark_class_name=definition.benchmark_class_name,
        program=program,
        metric=metric,
        metric_with_feedback=metric.with_feedback,
        splits=_freeze_split_parts(
            splits.train[:150],
            splits.validation[:300],
            splits.test,
        ),
        provenance={
            "generator_owner": "facebookresearch/clutrr",
            "generator_owner_commit": CLUTRR_OWNER_COMMIT,
            "fixed_backing": CLUTRR_HF_VARIANT_DIRECTORY[definition.task_id],
            "fixed_backing_commit": CLUTRR_HF_DATA_COMMIT,
            "label_column": "target_text",
            "metric": "QualityMetric.relation_overlap",
            "metric_owner_commit": CLUTRR_BASELINE_OWNER_COMMIT,
            "split_policy": "fixed_backing_train_validation_test",
            "optimization_caps": {"train": 150, "validation": 300},
            "test_scope": "all_owner_tasks_1_to_4",
        },
        score_breakdown=clutrr_task_score_summary,
    )


def resolve_mechanism_benchmark(
    task_id: str,
    *,
    lm: Any,
    dataset_mode: str,
    roots: MechanismOwnerRoots | None = None,
) -> ResolvedMechanismBenchmark:
    """Bind one mechanism task without remixing its owner splits."""

    if dataset_mode != "lite":
        raise ValueError("mechanism experiments currently require dataset_mode='lite'")
    try:
        definition = _MECHANISM_DEFINITIONS[task_id]
    except KeyError as error:
        raise ValueError(f"unknown mechanism task: {task_id!r}") from error
    owner_roots = roots or MechanismOwnerRoots.from_environment()
    if task_id == "hitab":
        return _resolve_hitab(definition, roots=owner_roots)
    if task_id == "chartqa":
        return _resolve_chartqa(definition, lm=lm, roots=owner_roots)
    return _resolve_clutrr(definition, roots=owner_roots)


__all__ = [
    "CHARTQA_ROOT_ENV",
    "CHARTQA_PREPARED_ROOT_ENV",
    "CLUTRR_BASELINE_ROOT_ENV",
    "CLUTRR_HF_DATA_ROOT_ENV",
    "CLUTRR_HF_VARIANT_DIRECTORY",
    "HITAB_PREPARED_ROOT_ENV",
    "MechanismBenchmarkDefinition",
    "MechanismOwnerRoots",
    "ResolvedMechanismBenchmark",
    "chartqa_type_score_summary",
    "clutrr_task_score_summary",
    "load_mechanism_benchmark_specs",
    "resolve_mechanism_benchmark",
]
