from __future__ import annotations

import importlib

import dspy
import pytest

_RECOVERY = importlib.import_module(
    "experiments.14_ifbench_four_method_top1_recover_failures"
)
_TOP1_EVAL = importlib.import_module(
    "experiments.14_ifbench_four_method_top1_eval"
)
_EVALUATE_OWNER = importlib.import_module("bridge.dspy_evaluate")


def test_corrected_high_resolution_runs_freeze_owner_final_candidates() -> None:
    methods = _TOP1_EVAL.METHOD_SETS["v45-v46-owner-final"]

    assert [method["version"] for method in methods] == ["v45", "v46"]
    assert all(method["score_mode"] == "raw_frontier_rate" for method in methods)
    assert all(
        method["parent_selection_score_mode"] == "high_resolution"
        for method in methods
    )
    assert _RECOVERY.EXPECTED_TOP1_BY_METHOD_SET[
        "v45-v46-owner-final"
    ] == {"v45": 24, "v46": 246}


def test_parallelizer_error_parser_uses_explicit_example_identity() -> None:
    stderr = "\n".join(
        [
            "unrelated ERROR line",
            (
                "2026/08/05 14:44:04 ERROR dspy.utils.parallelizer: "
                "Error for Example({'key': '13', 'prompt': 'x', "
                "'top1_method_index': 2}) (input_keys={'prompt'}): timeout"
            ),
        ]
    )

    assert _RECOVERY._parse_parallelizer_error_identities(stderr) == [(2, "13")]


def test_parallelizer_error_parser_rejects_unidentified_error() -> None:
    stderr = (
        "ERROR dspy.utils.parallelizer: Error for Example({'key': '13'}) "
        "(input_keys={'prompt'}): timeout"
    )

    with pytest.raises(ValueError, match="could not parse"):
        _RECOVERY._parse_parallelizer_error_identities(stderr)


def test_original_failure_classification_ignores_superseded_error_event() -> None:
    base_scores = {
        (0, "a"): 0.0,
        (0, "b"): 1.0,
    }

    failures, superseded, counts = _RECOVERY._classify_original_failures(
        error_events=[(0, "a"), (0, "a"), (0, "b")],
        base_scores=base_scores,
        failure_score=0.0,
    )

    assert failures == {(0, "a")}
    assert superseded == [(0, "b")]
    assert counts[(0, "a")] == 2


def test_numeric_zero_from_complete_official_prediction_is_success() -> None:
    prediction = dspy.Prediction(response="")

    assert _RECOVERY._is_successful_prediction(prediction)
    assert _RECOVERY._numeric_score(0.0, identity=(0, "a")) == 0.0
    assert not _RECOVERY._is_successful_prediction(dspy.Prediction())


def test_merge_replaces_only_explicitly_recovered_identity() -> None:
    base_records = [
        {
            "version": "v43",
            "test_score": 0.5,
            "test_score_percent": 50.0,
            "per_instance_scores": [
                {"test_key": "a", "score": 0.0},
                {"test_key": "b", "score": 1.0},
            ],
        },
        {
            "version": "v44",
            "test_score": 0.0,
            "test_score_percent": 0.0,
            "per_instance_scores": [
                {"test_key": "a", "score": 0.0},
                {"test_key": "b", "score": 0.0},
            ],
        },
    ]

    merged = _RECOVERY._merge_recovered_scores(
        base_records=base_records,
        recovered_scores={(0, "a"): 0.5},
        attempts={(0, "a"): 3},
    )

    assert merged[0]["per_instance_scores"] == [
        {"test_key": "a", "score": 0.5},
        {"test_key": "b", "score": 1.0},
    ]
    assert merged[0]["test_score"] == 0.75
    assert merged[0]["execution_failures_recovered"] == 1
    assert merged[0]["recovery_metric_calls"] == 3
    assert merged[1]["per_instance_scores"] == base_records[1][
        "per_instance_scores"
    ]
    assert merged[1]["test_score"] == 0.0


def test_owner_scope_disables_only_dspy_tail_straggler_resubmission() -> None:
    captured: dict[str, object] = {}

    class FakeOfficialExecutor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs

    previous_official = _EVALUATE_OWNER._OFFICIAL_EVALUATE_PARALLEL_EXECUTOR
    previous_evaluate = _EVALUATE_OWNER._DSPY_EVALUATE.ParallelExecutor
    _EVALUATE_OWNER._OFFICIAL_EVALUATE_PARALLEL_EXECUTOR = FakeOfficialExecutor
    _EVALUATE_OWNER._DSPY_EVALUATE.ParallelExecutor = FakeOfficialExecutor
    try:
        with _EVALUATE_OWNER.without_parent_straggler_resubmission():
            executor = _EVALUATE_OWNER._DSPY_EVALUATE.ParallelExecutor(
                num_threads=3
            )
            assert isinstance(executor, FakeOfficialExecutor)
        assert _EVALUATE_OWNER._DSPY_EVALUATE.ParallelExecutor is FakeOfficialExecutor
    finally:
        _EVALUATE_OWNER._OFFICIAL_EVALUATE_PARALLEL_EXECUTOR = previous_official
        _EVALUATE_OWNER._DSPY_EVALUATE.ParallelExecutor = previous_evaluate

    assert captured["args"] == ()
    assert captured["kwargs"] == {
        "num_threads": 3,
        "timeout": 0,
        "straggler_limit": 0,
    }
