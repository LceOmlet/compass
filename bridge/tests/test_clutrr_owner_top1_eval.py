from __future__ import annotations

import math

import dspy

from experiments.mechanism import evaluate_clutrr_owner_top1 as subject


def test_prediction_success_requires_complete_relation_field() -> None:
    assert not subject._prediction_succeeded(dspy.Prediction())
    assert subject._prediction_succeeded(dspy.Prediction(relation="sister"))
    assert subject._prediction_succeeded(dspy.Prediction(relation=""))


def test_checkpoint_preserves_successes_and_retries_only_pending() -> None:
    identities = {("validation", 0), ("test", 0), ("test", 1)}
    checkpoint = {"attempts": [], "scores": [], "predictions": []}
    attempts = {("validation", 0): 1, ("test", 0): 2, ("test", 1): 1}
    scores = {("validation", 0): 0.0, ("test", 0): 1.0}
    predictions = {("validation", 0): "wrong", ("test", 0): "sister"}

    subject._update_checkpoint(
        checkpoint,
        identities=identities,
        attempts=attempts,
        scores=scores,
        predictions=predictions,
    )

    restored_attempts, restored_scores, restored_predictions = (
        subject._checkpoint_maps(checkpoint)
    )
    assert restored_attempts == attempts
    assert restored_scores == scores
    assert restored_predictions == predictions
    assert checkpoint["pending"] == [{"position": 1, "split": "test"}]
    assert checkpoint["total_metric_calls"] == 4
    assert checkpoint["retry_metric_calls"] == 1


def test_score_summary_uses_all_frozen_positions() -> None:
    attempts = {("test", 0): 1, ("test", 1): 3}
    scores = {("test", 0): 0.0, ("test", 1): 1.0}
    summary = subject._score_summary(
        "test", size=2, attempts=attempts, scores=scores
    )
    assert summary["aggregate_percent"] == 50.0
    assert summary["total_metric_calls"] == 4
    assert summary["retry_metric_calls"] == 2
    assert summary["execution_failures_retried"] == 1
    assert summary["max_attempts"] == 3


def test_candidate_digest_is_mapping_order_independent() -> None:
    assert subject._candidate_sha256({"a": "left", "b": "right"}) == (
        subject._candidate_sha256({"b": "right", "a": "left"})
    )


def test_numeric_score_accepts_owner_zero_and_rejects_nonfinite() -> None:
    assert subject._numeric_score(0.0, identity=("test", 0)) == 0.0
    try:
        subject._numeric_score(math.nan, identity=("test", 0))
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite owner score was accepted")
