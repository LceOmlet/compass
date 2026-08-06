from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "upstreams" / "skill-factory" / "src"))
sys.path.insert(
    0,
    str(PROJECT_ROOT / "upstreams" / "skill-factory" / "upstream" / "lmms-eval"),
)

import dspy

from bridge import mechanism_benchmark_registry as mechanism_registry
from bridge.mechanism_benchmark_registry import (
    CLUTRR_HF_VARIANT_DIRECTORY,
    MechanismOwnerRoots,
    clutrr_task_score_summary,
    load_mechanism_benchmark_specs,
    resolve_mechanism_benchmark,
)
from bridge.paper_benchmark_registry import load_official_benchmark_specs


def _splits(prefix: str) -> SimpleNamespace:
    return SimpleNamespace(
        train=(dspy.Example(key=f"{prefix}-train", value=1),),
        validation=(dspy.Example(key=f"{prefix}-validation", value=2),),
        test=(dspy.Example(key=f"{prefix}-test", value=3),),
    )


def test_mechanism_registry_is_separate_from_the_exact_six_paper_tasks() -> None:
    assert tuple(sorted(load_official_benchmark_specs())) == (
        "aime_2025",
        "hotpotqa",
        "hover",
        "ifbench",
        "livebench_math",
        "pupa",
    )
    assert tuple(sorted(load_mechanism_benchmark_specs())) == (
        "chartqa",
        "clutrr_disconnected",
        "clutrr_irrelevant",
        "clutrr_supporting",
        "hitab",
    )


def test_clutrr_registry_freezes_the_three_owner_robust_backings() -> None:
    assert CLUTRR_HF_VARIANT_DIRECTORY == {
        "clutrr_supporting": "rob_train_sup_23_test_all_23",
        "clutrr_irrelevant": "rob_train_irr_23_test_all_23",
        "clutrr_disconnected": "rob_train_disc_23_test_all_23",
    }


def test_chartqa_resolution_passes_the_run_lm_to_the_owner_program(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    owner_program = dspy.Predict("image, prompt -> answer")
    owner_splits = SimpleNamespace(
        train=tuple(dspy.Example(key=f"train-{index}") for index in range(150)),
        validation=tuple(
            dspy.Example(key=f"validation-{index}") for index in range(300)
        ),
        test=tuple(dspy.Example(key=f"test-{index}") for index in range(2500)),
    )

    def build_owner_composition(lm, prepared_root, **kwargs):
        captured.update(
            lm=lm,
            prepared_root=Path(prepared_root),
            kwargs=kwargs,
        )
        return SimpleNamespace(
            program=owner_program,
            splits=owner_splits,
            metric=lambda *_args, **_kwargs: 1.0,
            metric_with_feedback=lambda *_args, **_kwargs: {
                "score": 1.0,
                "feedback": "owner",
            },
        )

    fake_owner_module = ModuleType("bridge.mechanism_chartqa")
    fake_owner_module.build_prepared_chartqa_owner_composition = (
        build_owner_composition
    )
    fake_owner_module.chartqa_doc_to_text = lambda *_args, **_kwargs: "prompt"
    monkeypatch.setitem(
        sys.modules,
        "bridge.mechanism_chartqa",
        fake_owner_module,
    )
    real_verify_git_revision = mechanism_registry._verify_git_revision

    def verify_owner_revision(root: Path, expected: str, owner: str) -> None:
        # The fake ``chartqa_doc_to_text`` above lives in this test module, so
        # only its synthetic LMMS source location has no pinned checkout.
        if owner == "lmms-eval":
            return
        real_verify_git_revision(root, expected, owner)

    monkeypatch.setattr(
        mechanism_registry,
        "_verify_git_revision",
        verify_owner_revision,
    )
    lm = object()
    prepared_root = tmp_path / "chartqa-prepared"
    prepared_root.mkdir()
    (prepared_root / "manifest.json").write_text("{}", encoding="utf-8")
    roots = MechanismOwnerRoots(
        chartqa=tmp_path / "chartqa",
        clutrr_hf_data=tmp_path / "clutrr-data",
        clutrr_baseline=tmp_path / "clutrr-baseline",
        chartqa_prepared=prepared_root,
    )

    resolved = resolve_mechanism_benchmark(
        "chartqa",
        lm=lm,
        dataset_mode="lite",
        roots=roots,
    )

    assert captured == {
        "lm": lm,
        "prepared_root": roots.chartqa_prepared,
        "kwargs": {},
    }
    assert resolved.program is owner_program
    assert resolved.splits.train is owner_splits.train
    assert resolved.provenance["optimization_splits"] == "vis-nlp/ChartQA"
    assert resolved.provenance["test_metric"] == "lmms-eval/chartqa"


def test_hitab_resolution_uses_only_the_frozen_prepared_views(
    monkeypatch,
    tmp_path: Path,
) -> None:
    owner_program = dspy.Predict("question, table -> answer")
    owner_splits = SimpleNamespace(
        train=tuple(dspy.Example(key=f"train-{index}") for index in range(150)),
        validation=tuple(
            dspy.Example(key=f"validation-{index}") for index in range(300)
        ),
        test=tuple(dspy.Example(key=f"test-{index}") for index in range(1584)),
    )
    captured: dict[str, object] = {}

    def build_owner_composition(prepared_root):
        captured["prepared_root"] = Path(prepared_root)
        return SimpleNamespace(
            program=owner_program,
            splits=owner_splits,
            metric=lambda *_args, **_kwargs: 1.0,
            metric_with_feedback=lambda *_args, **_kwargs: {
                "score": 1.0,
                "feedback": "owner",
            },
        )

    fake_owner_module = ModuleType("bridge.mechanism_hitab")
    fake_owner_module.HITAB_OWNER_COMMIT = (
        "d179602662b490249baf068a76fbe4137029126e"
    )
    fake_owner_module.build_prepared_hitab_owner_composition = (
        build_owner_composition
    )
    monkeypatch.setitem(sys.modules, "bridge.mechanism_hitab", fake_owner_module)
    prepared_root = tmp_path / "hitab-prepared"
    prepared_root.mkdir()
    (prepared_root / "manifest.json").write_text("{}", encoding="utf-8")
    roots = MechanismOwnerRoots(
        chartqa=tmp_path / "chartqa",
        clutrr_hf_data=tmp_path / "clutrr-data",
        clutrr_baseline=tmp_path / "clutrr-baseline",
        hitab_prepared=prepared_root,
    )

    resolved = resolve_mechanism_benchmark(
        "hitab",
        lm=object(),
        dataset_mode="lite",
        roots=roots,
    )

    assert captured["prepared_root"] == prepared_root
    assert resolved.splits.train is owner_splits.train
    assert len(resolved.splits.test) == 1584
    assert "first_150" in resolved.provenance["split_policy"]


def test_clutrr_resolution_uses_text_labels_complete_test_and_owner_metric(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from bridge import mechanism_clutrr
    from bridge import mechanism_benchmark_registry as registry

    captured: dict[str, object] = {}
    owner_splits = SimpleNamespace(
        train=tuple(dspy.Example(key=f"train-{index}") for index in range(155)),
        validation=tuple(
            dspy.Example(key=f"validation-{index}") for index in range(305)
        ),
        test=tuple(dspy.Example(key=f"test-{index}") for index in range(4)),
    )
    owner_relation_overlap = object()

    def load_splits(paths, **kwargs):
        captured.update(paths=paths, kwargs=kwargs)
        return owner_splits

    class OwnerMetric:
        def __init__(self, relation_overlap):
            captured["relation_overlap"] = relation_overlap

        def __call__(self, *_args, **_kwargs):
            return 1.0

        def with_feedback(self, *_args, **_kwargs):
            return dspy.Prediction(score=1.0, feedback="owner")

    monkeypatch.setattr(
        mechanism_clutrr,
        "load_fixed_clutrr_splits",
        load_splits,
    )
    monkeypatch.setattr(mechanism_clutrr, "ClutrrOwnerMetric", OwnerMetric)
    monkeypatch.setattr(
        registry,
        "_load_clutrr_relation_overlap",
        lambda _root: owner_relation_overlap,
    )
    roots = MechanismOwnerRoots(
        chartqa=tmp_path / "chartqa",
        clutrr_hf_data=tmp_path / "clutrr-data",
        clutrr_baseline=tmp_path / "clutrr-baseline",
    )

    resolved = resolve_mechanism_benchmark(
        "clutrr_supporting",
        lm=object(),
        dataset_mode="lite",
        roots=roots,
    )

    expected_root = roots.clutrr_hf_data / "rob_train_sup_23_test_all_23"
    assert captured["paths"].train == expected_root / "train.csv"
    assert captured["paths"].validation == expected_root / "validation.csv"
    assert captured["paths"].test == expected_root / "test.csv"
    assert captured["kwargs"] == {
        "variant": mechanism_clutrr.ClutrrVariant.SUPPORTING,
        "relation_length": None,
        "label_column": "target_text",
    }
    assert captured["relation_overlap"] is owner_relation_overlap
    assert [example.key for example in resolved.splits.train] == [
        f"train-{index}" for index in range(150)
    ]
    assert [example.key for example in resolved.splits.validation] == [
        f"validation-{index}" for index in range(300)
    ]
    assert resolved.splits.test == owner_splits.test
    assert resolved.provenance["test_scope"] == "all_owner_tasks_1_to_4"


def test_mechanism_resolution_has_no_optimizer_seed_or_split_remix_surface() -> None:
    parameters = inspect.signature(resolve_mechanism_benchmark).parameters
    assert "optimizer_seed" not in parameters
    assert "seed" not in parameters


def test_clutrr_reporting_groups_owner_scores_without_replacing_them() -> None:
    examples = (
        SimpleNamespace(clutrr_task_number=1),
        SimpleNamespace(clutrr_task_number=1),
        SimpleNamespace(clutrr_task_number=2),
        SimpleNamespace(clutrr_task_number=3),
        SimpleNamespace(clutrr_task_number=4),
    )

    summary = clutrr_task_score_summary(
        examples,
        (1.0, 0.0, 1.0, 0.0, 1.0),
    )

    assert summary == {
        "macro_percent": 62.5,
        "per_owner_task_percent": {
            "task_1": 50.0,
            "task_2": 100.0,
            "task_3": 0.0,
            "task_4": 100.0,
        },
    }
