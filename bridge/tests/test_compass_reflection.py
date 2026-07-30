from __future__ import annotations

import random
import threading
import time
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import dspy
import pytest
from gepa.core.adapter import EvaluationBatch
from gepa.core.engine import GEPAEngine
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.proposer.base import CandidateProposal, SubsampleEvaluation
from gepa.proposer.reflective_mutation.admission import AdmissionPlan
from gepa.strategies.acceptance import AcceptanceCriterion
from gepa.strategies.proposal_selection import AllImprovements

import bridge.b20_compass_reflection as compass_reflection
from bridge.b20_compass_reflection import (
    AlwaysAcceptAcceptance,
    SeedFallbackParetoCandidateSelector,
    SparseMinibatchEvaluationPolicy,
    SparseObservationDspyAdapter,
    select_reference_program_idx,
)


def _state() -> GEPAState:
    state = GEPAState(
        {"prompt": "seed"},
        ValsetEvaluation(
            outputs_by_val_id={0: "seed-0", 1: "seed-1"},
            scores_by_val_id={0: 1.0, 1: 0.0},
            objective_scores_by_val_id=None,
        ),
        frontier_type="instance",
    )
    state.program_candidates.append({"prompt": "child"})
    state.parent_program_for_candidate.append([0])
    state.program_birth_propose_ids.append(())
    state.prog_candidate_val_subscores.append({0: 1.0, 1: 1.0})
    state.prog_candidate_objective_scores.append({})
    state.named_predictor_id_to_update_next_for_program_candidate.append(0)
    state.num_metric_calls_by_discovery.append(0)
    state.pareto_front_valset = {0: 1.0, 1: 1.0}
    state.program_at_pareto_front_valset = {0: {0, 1}, 1: {1}}
    assert state.is_consistent()
    return state


def _adapter() -> SparseObservationDspyAdapter:
    return SparseObservationDspyAdapter(
        student_module=SimpleNamespace(),
        metric_fn=lambda *_args, **_kwargs: 0.0,
        feedback_map={},
        max_candidate_workers=2,
    )


def test_always_accept_uses_official_criterion_seam_for_worse_proposal() -> None:
    criterion = AlwaysAcceptAcceptance()
    proposal = SimpleNamespace(
        subsample_scores_before=[1.0, 1.0],
        subsample_scores_after=[0.0, 0.0],
    )

    assert isinstance(criterion, AcceptanceCriterion)
    assert criterion.should_accept(proposal, _state()) is True
    assert AllImprovements().select([proposal], _state(), criterion) == [proposal]


def test_default_acceptance_mode_still_rejects_worse_proposal() -> None:
    proposal = SimpleNamespace(
        subsample_scores_before=[1.0],
        subsample_scores_after=[0.0],
    )

    assert (
        compass_reflection._acceptance_criterion(
            "strict_improvement",
        ).should_accept(proposal, _state())
        is False
    )
    assert (
        compass_reflection.CompassReflectionEngineConfig.__dataclass_fields__[
            "acceptance_mode"
        ].default
        == "strict_improvement"
    )


def test_always_accept_commits_worse_admission_through_gepa_engine() -> None:
    state = GEPAState(
        {"prompt": "seed"},
        ValsetEvaluation(outputs_by_val_id={}, scores_by_val_id={}),
    )
    state.i = 0
    state.full_program_trace.append({"i": 1})
    engine = object.__new__(GEPAEngine)
    engine.acceptance_criterion = AlwaysAcceptAcceptance()
    engine.selection_strategy = AllImprovements()
    engine.logger = MagicMock()
    engine.adapter = MagicMock()
    engine.callbacks = None
    engine.merge_proposer = None
    engine._evaluate_programs_on_valset = MagicMock()
    engine._add_evaluated_program = MagicMock(return_value=(1, 0))
    engine._log_proposal_lm_calls = MagicMock()
    proposal = CandidateProposal(
        candidate={"prompt": "worse-child"},
        parent_program_ids=[0],
        subsample_indices=[11],
        subsample_scores_before=[1.0],
        subsample_scores_after=[0.0],
        eval_before=SubsampleEvaluation(
            scores=[1.0],
            outputs=["old"],
            trajectories=[{"trace": "old"}],
        ),
        eval_after=SubsampleEvaluation(
            scores=[0.0],
            outputs=["new"],
            trajectories=[{"trace": "new"}],
        ),
        admission_plan=AdmissionPlan(
            evaluation_ids=(11,),
            evaluation_batch=({"id": 11},),
            eval_before=EvaluationBatch(
                outputs=["old"],
                scores=[1.0],
                trajectories=[{"trace": "old"}],
            ),
            birth_propose_ids=(3,),
        ),
    )

    assert engine._run_reflective_batch([proposal], state)

    engine._evaluate_programs_on_valset.assert_not_called()
    call = engine._add_evaluated_program.call_args.kwargs
    assert call["birth_propose_ids"] == (3,)
    assert call["valset_evaluation"].scores_by_val_id == {11: 0.0}
    assert call["valset_evaluation"].outputs_by_val_id == {11: "new"}
    engine.adapter.commit_program_observations.assert_called_once()


def test_sparse_observations_only_replace_with_strictly_higher_reward() -> None:
    adapter = _adapter()
    evaluation = EvaluationBatch(
        outputs=["first"],
        scores=[0.5],
        trajectories=[{"example": object()}],
    )
    adapter.commit_program_observations(
        program_idx=3,
        evaluation_ids=(7,),
        evaluation=evaluation,
        committed_ids=(7,),
    )
    adapter.commit_program_observations(
        program_idx=3,
        evaluation_ids=(7,),
        evaluation=EvaluationBatch(
            outputs=["equal"],
            scores=[0.5],
            trajectories=[{"example": object()}],
        ),
        committed_ids=(7,),
    )

    fact = adapter.get_program_observation(3, 7)

    assert fact is not None
    assert fact.score == 0.5
    assert fact.output == "first"


def test_sparse_observation_state_round_trips_without_aliasing() -> None:
    adapter = _adapter()
    adapter.commit_program_observations(
        program_idx=0,
        evaluation_ids=(4,),
        evaluation=EvaluationBatch(
            outputs=["x"],
            scores=[1.0],
            trajectories=[{"example": "e"}],
        ),
        committed_ids=(4,),
    )
    persisted = adapter.get_adapter_state()
    restored = _adapter()
    restored.set_adapter_state(deepcopy(persisted))

    assert restored.get_program_observation(0, 4) == (
        adapter.get_program_observation(0, 4)
    )
    assert restored.get_adapter_state() is not persisted


def test_candidate_batch_concurrency_restores_submission_order() -> None:
    adapter = _adapter()

    def evaluate(
        batch: list[Any],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        assert capture_traces
        return EvaluationBatch(
            outputs=[candidate["prompt"]],
            scores=[float(batch[0])],
            trajectories=[{"example": batch[0]}],
        )

    adapter.evaluate = evaluate  # type: ignore[method-assign]
    results = adapter.batch_evaluate(
        [
            ({"prompt": "a"}, [1]),
            ({"prompt": "b"}, [2]),
            ({"prompt": "c"}, [3]),
        ]
    )

    assert [result.outputs for result in results] == [["a"], ["b"], ["c"]]
    assert [result.scores for result in results] == [[1.0], [2.0], [3.0]]


def test_candidate_batch_preserves_failed_slot_without_resubmission() -> None:
    adapter = _adapter()
    calls: list[str] = []

    def evaluate(
        batch: list[Any],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        assert capture_traces
        calls.append(candidate["prompt"])
        if candidate["prompt"] == "b":
            raise TimeoutError("candidate timed out")
        return EvaluationBatch(
            outputs=[candidate["prompt"]],
            scores=[float(batch[0])],
            trajectories=[{"example": batch[0]}],
        )

    adapter.evaluate = evaluate  # type: ignore[method-assign]
    results = adapter.batch_evaluate(
        [
            ({"prompt": "a"}, [1]),
            ({"prompt": "b"}, [2]),
            ({"prompt": "c"}, [3]),
        ]
    )

    assert [result.outputs if result is not None else None for result in results] == [
        ["a"],
        None,
        ["c"],
    ]
    assert sorted(calls) == ["a", "b", "c"]


def test_raw_feedback_reflection_is_concurrent_and_restores_task_order() -> None:
    adapter = object.__new__(SparseObservationDspyAdapter)
    adapter.max_reflection_workers = 3
    barrier = threading.Barrier(3)

    def propose_new_texts(
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        assert reflective_dataset["prompt"]
        assert components_to_update == ["prompt"]
        barrier.wait(timeout=5)
        index = int(candidate["prompt"])
        time.sleep((2 - index) * 0.01)
        return {"prompt": f"new-{index}"}

    adapter.propose_new_texts = propose_new_texts  # type: ignore[method-assign]
    results = adapter.propose_new_texts_batch(
        [
            (
                {"prompt": str(index)},
                {"prompt": [{"input": index, "feedback": "revise"}]},
                ["prompt"],
            )
            for index in range(3)
        ]
    )

    assert results == [
        {"prompt": "new-0"},
        {"prompt": "new-1"},
        {"prompt": "new-2"},
    ]


def test_raw_feedback_reflection_preserves_failed_slot_without_resubmission() -> None:
    adapter = object.__new__(SparseObservationDspyAdapter)
    adapter.max_reflection_workers = 3
    calls: list[str] = []

    def propose_new_texts(
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        assert reflective_dataset["prompt"]
        assert components_to_update == ["prompt"]
        calls.append(candidate["prompt"])
        if candidate["prompt"] == "1":
            raise TimeoutError("reflection timed out")
        return {"prompt": f"new-{candidate['prompt']}"}

    adapter.propose_new_texts = propose_new_texts  # type: ignore[method-assign]
    results = adapter.propose_new_texts_batch(
        [
            (
                {"prompt": str(index)},
                {"prompt": [{"input": index, "feedback": "revise"}]},
                ["prompt"],
            )
            for index in range(3)
        ]
    )

    assert results == [
        {"prompt": "new-0"},
        None,
        {"prompt": "new-2"},
    ]
    assert sorted(calls) == ["0", "1", "2"]


def test_raw_feedback_batch_uses_official_dspy_proposer_for_one_child() -> None:
    prompts: list[str] = []

    def reflection_lm(prompt: str) -> list[str]:
        prompts.append(prompt)
        return ["```only-child```"]

    adapter = SparseObservationDspyAdapter(
        student_module=SimpleNamespace(),
        metric_fn=lambda *_args, **_kwargs: 0.0,
        feedback_map={},
        reflection_lm=reflection_lm,
        max_reflection_workers=4,
    )
    result = adapter.propose_new_texts_batch(
        [
            (
                {"prompt": "seed"},
                {
                    "prompt": [
                        {
                            "Inputs": {"question": "q"},
                            "Generated Outputs": {"answer": "a"},
                            "Feedback": "revise",
                        }
                    ]
                },
                ["prompt"],
            )
        ]
    )

    assert result == [{"prompt": "only-child"}]
    assert len(prompts) == 1
    assert "seed" in prompts[0]
    assert "revise" in prompts[0]


def test_sparse_adapter_fail_fast_propagates_infrastructure_errors() -> None:
    class ExplodingProgram(dspy.Module):
        def forward(self, value: int) -> dspy.Prediction:
            raise RuntimeError(f"infrastructure failure for {value}")

    adapter = SparseObservationDspyAdapter(
        student_module=ExplodingProgram(),
        metric_fn=lambda *_args, **_kwargs: 0.0,
        feedback_map={},
        failure_score=0.0,
        num_threads=2,
        raise_on_error=True,
        max_candidate_workers=1,
    )
    batch = [dspy.Example(value=1).with_inputs("value")]

    with pytest.raises(Exception, match="cancelled due to errors"):
        adapter.evaluate(batch, {}, capture_traces=True)


def test_reference_selection_uses_frontier_rate_then_exposure() -> None:
    state = _state()

    selected = select_reference_program_idx(
        state,
        instance_id=0,
        sampled_parent_idx=0,
        rng=random.Random(0),
    )

    assert selected == 1


def test_sparse_final_selection_uses_rate_exposure_then_earliest() -> None:
    state = _state()
    policy = SparseMinibatchEvaluationPolicy()

    assert policy.get_best_program(state) == 1


def test_official_pareto_selector_falls_back_only_for_empty_seed_frontier() -> None:
    state = GEPAState(
        {"prompt": "seed"},
        ValsetEvaluation(
            outputs_by_val_id={},
            scores_by_val_id={},
            objective_scores_by_val_id=None,
        ),
        frontier_type="instance",
    )
    selector = SeedFallbackParetoCandidateSelector(random.Random(0))

    assert selector.select_candidate_idx(state) == 0

    state.commit_existing_program_evaluation(
        program_idx=0,
        valset_evaluation=ValsetEvaluation(
            outputs_by_val_id={5: "x"},
            scores_by_val_id={5: 1.0},
            objective_scores_by_val_id=None,
        ),
        run_dir=None,
        iteration=0,
    )
    assert selector.select_candidate_idx(state) == 0
