from __future__ import annotations

import importlib

import dspy
import pytest


_EVALUATION = importlib.import_module("experiments.paper.evaluate_aime_top1")


def test_method_argument_requires_version_and_path() -> None:
    version, path = _EVALUATION._parse_method("v57=F:/run")

    assert version == "v57"
    assert str(path).replace("\\", "/") == "F:/run"
    with pytest.raises(Exception, match="VERSION=RUN_DIR"):
        _EVALUATION._parse_method("v57")


def test_complete_official_answer_is_success_even_at_zero_score() -> None:
    assert _EVALUATION._prediction_succeeded(dspy.Prediction(answer="wrong"))
    assert not _EVALUATION._prediction_succeeded(dspy.Prediction())
    assert _EVALUATION._numeric_score(0, identity=("v57", 0)) == 0.0


def test_checkpoint_freezes_success_and_leaves_only_failures_pending() -> None:
    identities = {("v57", 0), ("v57", 1)}
    checkpoint = {"attempts": [], "scores": [], "pending": []}
    attempts = {("v57", 0): 1, ("v57", 1): 2}
    scores = {("v57", 0): 0.0}

    _EVALUATION._update_checkpoint(
        checkpoint,
        identities=identities,
        attempts=attempts,
        scores=scores,
    )

    assert checkpoint["pending"] == [
        {"version": "v57", "test_position": 1}
    ]
    assert checkpoint["total_metric_calls"] == 3
    assert checkpoint["retry_metric_calls"] == 1


def test_final_record_aggregates_official_numeric_scores() -> None:
    method = _EVALUATION.FrozenMethod(
        version="v57",
        run_dir=_EVALUATION.Path("F:/run"),
        selected_candidate_idx=15,
        candidate={"predict": "instruction"},
        candidate_sha256="candidate",
        selection_sha256="selection",
        model_config={},
        api_key_env="OPENAI_API_KEY",
        test_split_fingerprint="split",
    )

    records = _EVALUATION._final_records(
        methods=[method],
        test_size=2,
        attempts={("v57", 0): 1, ("v57", 1): 3},
        scores={("v57", 0): 1.0, ("v57", 1): 0.0},
    )

    assert records[0]["test_score"] == 0.5
    assert records[0]["test_score_percent"] == 50.0
    assert records[0]["retry_metric_calls"] == 2
    assert records[0]["execution_failures_retried"] == 1
