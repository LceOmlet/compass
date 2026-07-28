from __future__ import annotations

import random
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import dspy
import pytest
from gepa.core.adapter import EvaluationBatch
from gepa.core.state import GEPAState, ValsetEvaluation

from bridge.b20_compass_reflection import (
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
