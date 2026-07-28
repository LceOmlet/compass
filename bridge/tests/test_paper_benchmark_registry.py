from __future__ import annotations

from types import SimpleNamespace

import pytest

from bridge.paper_benchmark_registry import (
    _canonical_feedback,
    load_official_benchmark_specs,
    split_fingerprint,
)


def test_official_registry_resolves_exact_six_paper_tasks() -> None:
    specs = load_official_benchmark_specs()

    assert tuple(sorted(specs)) == (
        "aime_2025",
        "hotpotqa",
        "hover",
        "ifbench",
        "livebench_math",
        "pupa",
    )
    assert {
        task_id: spec.max_metric_calls
        for task_id, spec in specs.items()
    } == {
        "hotpotqa": 6871,
        "ifbench": 3593,
        "hover": 7051,
        "pupa": 2426,
        "aime_2025": 1839,
        "livebench_math": 1839,
    }
    assert all(spec.predictor_names for spec in specs.values())


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"score": 0.25, "feedback": "canonical"}, {"score": 0.25, "feedback": "canonical"}),
        (
            {"feedback_score": 0.5, "feedback_text": "legacy"},
            {"score": 0.5, "feedback": "legacy"},
        ),
        (
            SimpleNamespace(score=0.75, feedback="object"),
            {"score": 0.75, "feedback": "object"},
        ),
    ],
)
def test_feedback_adapter_accepts_only_owner_shapes(value, expected) -> None:
    adapted = _canonical_feedback(lambda **_kwargs: value)
    assert adapted(anything=True) == expected


def test_feedback_adapter_rejects_unknown_shape() -> None:
    adapted = _canonical_feedback(lambda **_kwargs: {"value": 1})
    with pytest.raises(TypeError, match="official feedback"):
        adapted()


def test_split_fingerprint_is_deterministic_and_order_sensitive() -> None:
    examples = ({"id": 1, "text": "a"}, {"id": 2, "text": "b"})
    assert split_fingerprint(examples) == split_fingerprint(examples)
    assert split_fingerprint(examples) != split_fingerprint(tuple(reversed(examples)))
