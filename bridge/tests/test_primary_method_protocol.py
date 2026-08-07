from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import dspy
import pytest

from bridge.primary_method_protocol import (
    MIPRO_HEAVY_OWNER_BUDGETS,
    canonical_sha256,
    compass_condition,
    make_gepa_feedback_metric,
    mipro_protocol,
    validate_primary_cell,
)
from experiments.paper.generate_primary_matrix import build_matrix
from experiments.paper.run_primary_method import (
    _compile_gepa,
    _compile_mipro,
    load_primary_config,
)


def _model() -> dict:
    return {
        "api_base": "https://api.example.invalid/v1",
        "api_key_env": "PAPER_TEST_API_KEY",
        "cache": True,
        "cache_in_memory": True,
        "enable_thinking": False,
        "max_tokens": 16384,
        "model": "openai/gpt-4.1-mini-2025-04-14",
        "model_type": "chat",
        "num_retries": 0,
        "temperature": 1.0,
    }


def _source_snapshot() -> dict:
    return {
        "root_head": "a" * 40,
        "file_sha256": {"bridge/example.py": "b" * 64},
        "submodules": {"upstreams/dspy": "c" * 40},
    }


def _baseline_config(tmp_path: Path, **overrides) -> dict:
    model = _model()
    config = {
        "schema_version": 1,
        "matrix_id": "primary_test_v1",
        "phase": "formal",
        "task_id": "ifbench",
        "method": "gepa",
        "dataset_mode": "lite",
        "optimizer_seed": 0,
        "logical_rollout_budget": 3593,
        "num_threads": 8,
        "model": model,
        "model_profile_sha256": canonical_sha256(model),
        "run_dir": str(tmp_path / "runs" / "ifbench_gepa"),
        "cache_dir": str(tmp_path / "cache" / "ifbench_gepa"),
        "source_snapshot": _source_snapshot(),
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    ("task_id", "budget"),
    [("ifbench", 3593), ("aime_2025", 1839)],
)
def test_formal_mipro_uses_only_owner_frozen_heavy_budget(
    task_id: str,
    budget: int,
) -> None:
    cell = validate_primary_cell(
        task_id=task_id,
        method="mipro",
        phase="formal",
        optimizer_seed=0,
        logical_rollout_budget=budget,
    )

    protocol = mipro_protocol(cell)

    assert protocol["init"]["auto"] == "heavy"
    assert protocol["init"]["seed"] == 0
    assert protocol["compile"] == {}
    assert (
        protocol["rollout_alignment"]["expected"]
        == (MIPRO_HEAVY_OWNER_BUDGETS[task_id])
    )


def test_formal_chartqa_mipro_fails_closed_without_owner_rollout_cap() -> None:
    with pytest.raises(ValueError, match="no max_metric_calls seam"):
        validate_primary_cell(
            task_id="chartqa",
            method="mipro",
            phase="formal",
            optimizer_seed=0,
            logical_rollout_budget=1152,
        )


def test_preflight_mipro_uses_two_official_trials_without_claiming_a_cap() -> None:
    cell = validate_primary_cell(
        task_id="chartqa",
        method="mipro",
        phase="preflight",
        optimizer_seed=0,
        logical_rollout_budget=60,
    )

    protocol = mipro_protocol(cell)

    assert protocol["init"]["auto"] is None
    assert protocol["init"]["num_candidates"] == 2
    assert protocol["compile"] == {"num_trials": 2}
    assert protocol["rollout_alignment"] == {
        "owner": "dspy.MIPROv2.trial_logs",
        "minimum": 60,
    }


def test_method_names_delegate_to_existing_compass_conditions() -> None:
    assert compass_condition("m0") == "mini_admission_reflection"
    assert compass_condition("compass") == "compass_reflection"
    with pytest.raises(ValueError, match="not a COMPASS-engine method"):
        compass_condition("gepa")


def test_gepa_feedback_bridge_passes_exact_owner_trace() -> None:
    seen: dict[str, object] = {}

    def owner_feedback(**kwargs):
        seen.update(kwargs)
        return {"score": 1.0, "feedback": "owner feedback"}

    predictor = dspy.Predict("question -> answer")
    program = SimpleNamespace(named_predictors=lambda: [("answer", predictor)])
    meta = SimpleNamespace(
        feedback_fn_maps=[{"answer": owner_feedback}],
        metric_with_feedback=None,
        metric=lambda gold, pred, trace=None: 1.0,
    )
    benchmark = SimpleNamespace(
        task_id="fake",
        program=program,
        predictor_names=("answer",),
        program_index=0,
        benchmark_meta=meta,
    )
    scalar_metric = MagicMock(return_value=1.0)
    metric = make_gepa_feedback_metric(benchmark, scalar_metric=scalar_metric)
    gold = dspy.Example(question="q", answer="a").with_inputs("question")
    pred = dspy.Prediction(answer="a")
    full_trace = [(predictor, {"question": "q"}, {"answer": "a"})]

    result = metric(
        gold,
        pred,
        full_trace,
        "answer",
        [(predictor, {"question": "q"}, {"answer": "a"})],
    )

    assert result.score == 1.0
    assert result.feedback == "owner feedback"
    assert seen == {
        "predictor_output": {"answer": "a"},
        "predictor_inputs": {"question": "q"},
        "module_inputs": gold,
        "module_outputs": pred,
        "captured_trace": full_trace,
    }


def test_primary_config_is_strict_and_contains_no_credential_value(
    tmp_path: Path,
) -> None:
    config = _baseline_config(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_primary_config(path)

    assert loaded["model"]["api_key_env"] == "PAPER_TEST_API_KEY"
    assert "api_key" not in loaded["model"]
    assert loaded["run_dir"] != loaded["cache_dir"]

    config["unexpected"] = True
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="config keys mismatch"):
        load_primary_config(path)


def test_formal_matrix_delegates_without_copying_compass_engine(tmp_path: Path) -> None:
    records, blocked = build_matrix(
        matrix_id="primary_test_v1",
        phase="formal",
        runtime_root=tmp_path,
        model=_model(),
        snapshot=_source_snapshot(),
        text_num_threads=16,
        chartqa_num_threads=4,
        max_candidate_workers=1,
        max_reflection_workers=1,
    )

    assert len(records) == 14
    assert blocked == [
        {
            "task_id": "chartqa",
            "method": "mipro",
            "phase": "formal",
            "status": "blocked_owner_interface",
            "reason": (
                "DSPy MIPROv2 has no max_metric_calls seam and the GEPA "
                "artifact has no 1152-rollout ChartQA protocol"
            ),
        }
    ]
    path_pairs = {
        (record["config"]["run_dir"], record["config"]["cache_dir"])
        for record in records
    }
    assert len(path_pairs) == len(records)
    assert all(run_dir != cache_dir for run_dir, cache_dir in path_pairs)

    by_key = {(record["task_id"], record["method"]): record for record in records}
    assert by_key[("ifbench", "m0")]["runner"].endswith(
        "experiments/paper/run_compass_reflection.py"
    )
    assert by_key[("chartqa", "compass")]["runner"].endswith(
        "experiments/mechanism/run_compass_reflection.py"
    )
    assert (
        by_key[("chartqa", "compass")]["config"]["optimizer"]["acceptance_mode"]
        == "strict_improvement"
    )
    assert (
        by_key[("chartqa", "compass")]["config"]["optimizer"]["max_metric_calls"]
        == 1152
    )
    m0_optimizer = by_key[("ifbench", "m0")]["config"]["optimizer"]
    assert m0_optimizer["reflection_minibatch_size"] == 3
    assert "proposal_minibatch_size" not in m0_optimizer
    assert "admission_minibatch_size" not in m0_optimizer


def test_preflight_matrix_fails_closed_when_text_runner_has_no_pilot_seam(
    tmp_path: Path,
) -> None:
    records, blocked = build_matrix(
        matrix_id="primary_test_v1",
        phase="preflight",
        runtime_root=tmp_path,
        model=_model(),
        snapshot=_source_snapshot(),
        text_num_threads=16,
        chartqa_num_threads=4,
        max_candidate_workers=1,
        max_reflection_workers=1,
    )

    assert len(records) == 8
    assert {(record["task_id"], record["method"]) for record in records} == {
        ("ifbench", "mipro"),
        ("ifbench", "gepa"),
        ("aime_2025", "mipro"),
        ("aime_2025", "gepa"),
        ("chartqa", "mipro"),
        ("chartqa", "gepa"),
        ("chartqa", "m0"),
        ("chartqa", "compass"),
    }
    assert {(record["task_id"], record["method"]) for record in blocked} == {
        ("ifbench", "m0"),
        ("ifbench", "compass"),
        ("aime_2025", "m0"),
        ("aime_2025", "compass"),
    }


def test_mipro_factory_calls_official_dspy_owner(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class FakeMIPRO:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def compile(self, program, **kwargs):
            calls["program"] = program
            calls["compile"] = kwargs
            return SimpleNamespace(trial_logs={1: {"total_eval_calls_so_far": 3593}})

    monkeypatch.setattr(dspy, "MIPROv2", FakeMIPRO)
    benchmark = SimpleNamespace(
        program=object(),
        splits=SimpleNamespace(train=(1, 2), validation=(3,), test=()),
    )
    config = _baseline_config(
        tmp_path,
        method="mipro",
        logical_rollout_budget=3593,
    )

    _, result = _compile_mipro(
        benchmark=benchmark,
        metric=object(),
        lm=object(),
        config=config,
        run_dir=tmp_path,
    )

    assert calls["init"]["auto"] == "heavy"
    assert calls["init"]["seed"] == 0
    assert calls["compile"]["trainset"] == [1, 2]
    assert calls["compile"]["valset"] == [3]
    assert result["owner"] == "dspy.MIPROv2"


def test_gepa_factory_forwards_multimodal_owner_proposer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    proposer = object()

    class FakeGEPA:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def compile(self, program, **kwargs):
            calls["program"] = program
            calls["compile"] = kwargs
            return SimpleNamespace(
                detailed_results=SimpleNamespace(total_metric_calls=1152)
            )

    monkeypatch.setattr(dspy, "GEPA", FakeGEPA)
    monkeypatch.setattr(
        "experiments.paper.run_primary_method.make_gepa_feedback_metric",
        lambda benchmark, scalar_metric: object(),
    )
    benchmark = SimpleNamespace(
        program=object(),
        splits=SimpleNamespace(train=(1,), validation=(2,), test=()),
        custom_instruction_proposer=proposer,
    )
    config = _baseline_config(
        tmp_path,
        task_id="chartqa",
        method="gepa",
        logical_rollout_budget=1152,
        num_threads=4,
    )

    _, result = _compile_gepa(
        benchmark=benchmark,
        metric=object(),
        lm=object(),
        config=config,
        run_dir=tmp_path,
    )

    assert calls["init"]["instruction_proposer"] is proposer
    assert calls["init"]["max_metric_calls"] == 1152
    assert calls["init"]["use_merge"] is False
    assert calls["compile"]["trainset"] == [1]
    assert calls["compile"]["valset"] == [2]
    assert result["owner"] == "dspy.GEPA"
    assert result["custom_instruction_proposer"] == "object"
