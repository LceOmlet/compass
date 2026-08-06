from __future__ import annotations

import inspect

import dspy
import pytest

from bridge.mechanism_benchmark_registry import (
    MechanismBenchmarkDefinition,
    ResolvedMechanismBenchmark,
)
from bridge.paper_benchmark_registry import (
    OfficialDatasetSplits,
    canonical_feedback_map,
)
from experiments.paper.run_compass_reflection import (
    _load_benchmark_definition,
    main,
)


def test_common_runner_defaults_to_the_original_official_family() -> None:
    assert inspect.signature(main).parameters["benchmark_family"].default == (
        "official"
    )
    definition = _load_benchmark_definition(
        {"task_id": "ifbench"},
        benchmark_family="official",
    )
    assert definition.task_id == "ifbench"


def test_common_runner_resolves_mechanisms_only_through_separate_registry() -> None:
    definition = _load_benchmark_definition(
        {"task_id": "chartqa"},
        benchmark_family="mechanism",
    )
    assert isinstance(definition, MechanismBenchmarkDefinition)
    assert definition.task_id == "chartqa"

    with pytest.raises(ValueError, match="unknown paper task"):
        _load_benchmark_definition(
            {"task_id": "chartqa"},
            benchmark_family="official",
        )


def test_common_runner_rejects_an_unknown_registry_family() -> None:
    with pytest.raises(ValueError, match="unknown benchmark family"):
        _load_benchmark_definition(
            {"task_id": "ifbench"},
            benchmark_family="shadow",
        )


def test_mechanism_owner_feedback_uses_the_existing_canonical_seam() -> None:
    program = dspy.Predict("question -> answer")
    example = dspy.Example(question="q", answer="gold").with_inputs("question")
    prediction = dspy.Prediction(answer="pred")
    resolved = ResolvedMechanismBenchmark(
        task_id="mechanism",
        display_name="Mechanism",
        benchmark_class_name="OwnerComposition",
        program=program,
        metric=lambda *_args, **_kwargs: 0.25,
        metric_with_feedback=lambda owner_example, owner_prediction, _trace: {
            "score": 0.25,
            "feedback": f"{owner_example.answer}/{owner_prediction.answer}",
        },
        splits=OfficialDatasetSplits(
            train=(example,),
            validation=(example,),
            test=(example,),
            fingerprints={"train": "a", "validation": "b", "test": "c"},
        ),
        provenance={},
    )

    feedback = canonical_feedback_map(resolved)["self"](
        predictor_output=None,
        predictor_inputs=None,
        module_inputs=example,
        module_outputs=prediction,
        captured_trace=None,
    )

    assert feedback == {"score": 0.25, "feedback": "gold/pred"}
