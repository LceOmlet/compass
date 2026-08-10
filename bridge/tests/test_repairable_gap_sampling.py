from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from bridge.b19_reversible_parent_selection import parent_selection_snapshot
from bridge.b21_joint_linucb_scheduler import (
    JointAssignment,
    observed_arm_context,
    rectangular_max_weight_assignment,
    relaxed_repairable_gap,
)
from bridge.b22_repairable_gap_sampling import RepairableGapSamplingStrategy
from gepa.core.data_loader import ListDataLoader
from gepa.core.state import GEPAState, ValsetEvaluation


def _state(scores_by_program: list[dict[int, float]]) -> GEPAState:
    all_ids = sorted(
        {
            instance_id
            for scores in scores_by_program
            for instance_id in scores
        }
    )
    state = GEPAState(
        {"prompt": "skill-0"},
        ValsetEvaluation(
            outputs_by_val_id={
                instance_id: ""
                for instance_id in scores_by_program[0]
            },
            scores_by_val_id=dict(scores_by_program[0]),
            objective_scores_by_val_id=None,
        ),
        track_best_outputs=False,
        frontier_type="instance",
    )
    state.program_candidates = [
        {"prompt": f"skill-{program_idx}"}
        for program_idx in range(len(scores_by_program))
    ]
    state.parent_program_for_candidate = [[None] for _ in scores_by_program]
    state.program_birth_propose_ids = [()] + [
        None for _ in scores_by_program[1:]
    ]
    state.prog_candidate_val_subscores = [
        dict(scores) for scores in scores_by_program
    ]
    state.prog_candidate_objective_scores = [{} for _ in scores_by_program]
    state.named_predictor_id_to_update_next_for_program_candidate = [
        0 for _ in scores_by_program
    ]
    state.num_metric_calls_by_discovery = [0 for _ in scores_by_program]
    state.pareto_front_valset = {}
    state.program_at_pareto_front_valset = {}
    for instance_id in all_ids:
        observed = {
            program_idx: scores[instance_id]
            for program_idx, scores in enumerate(scores_by_program)
            if instance_id in scores and math.isfinite(scores[instance_id])
        }
        if not observed:
            continue
        best = max(observed.values())
        state.pareto_front_valset[instance_id] = best
        state.program_at_pareto_front_valset[instance_id] = {
            program_idx
            for program_idx, score in observed.items()
            if score == best
        }
    assert state.is_consistent()
    return state


def _failing_selector() -> SimpleNamespace:
    return SimpleNamespace(
        select_candidate_idx=lambda state: pytest.fail(
            "repairable-gap matching must not invoke scalar parent sampling"
        )
    )


def _fixed_batch_sampler(
    batches: tuple[tuple[int, ...], ...],
) -> SimpleNamespace:
    pending = iter(batches)
    return SimpleNamespace(
        next_minibatch_ids=lambda trainset, state: list(next(pending))
    )


def test_relaxed_h_is_the_same_canonical_quantity_used_by_linucb_context() -> None:
    state = _state(
        [
            {0: 0.2, 1: 0.8},
            {0: 0.7},
        ]
    )

    gap = relaxed_repairable_gap(
        state,
        active_program_indices=(0, 1),
        program_idx=0,
        minibatch_ids=(0, 1),
    )
    context = observed_arm_context(
        state,
        active_program_indices=(0, 1),
        program_idx=0,
        minibatch_ids=(0, 1),
        perfect_score=1.0,
    )

    assert gap == pytest.approx(0.25)
    assert context.repairable_gap == gap


def test_relaxed_h_counts_batch_positions_and_leaves_missing_unknown() -> None:
    state = _state(
        [
            {0: 0.0},
            {0: 1.0},
        ]
    )

    assert relaxed_repairable_gap(
        state,
        active_program_indices=(0, 1),
        program_idx=0,
        minibatch_ids=(0, 0, 1),
    ) == pytest.approx(2.0 / 3.0)
    assert relaxed_repairable_gap(
        state,
        active_program_indices=(0, 1),
        program_idx=1,
        minibatch_ids=(0, 0, 1),
    ) == 0.0


def test_exact_hungarian_maximizes_only_weight_then_stable_indices() -> None:
    weights = {
        (2, 0): 10.0,
        (2, 1): 9.0,
        (5, 0): 8.0,
        (5, 1): 0.0,
    }

    assert rectangular_max_weight_assignment(
        program_indices=(2, 5),
        minibatch_positions=(0, 1),
        weights=weights,
        total_minibatches=2,
    ) == (
        JointAssignment(program_idx=5, minibatch_position=0),
        JointAssignment(program_idx=2, minibatch_position=1),
    )

    assert rectangular_max_weight_assignment(
        program_indices=(2, 5),
        minibatch_positions=(0, 1, 2),
        weights={
            (program_idx, position): 0.0
            for program_idx in (2, 5)
            for position in (0, 1, 2)
        },
        total_minibatches=3,
    ) == (
        JointAssignment(program_idx=2, minibatch_position=0),
        JointAssignment(program_idx=5, minibatch_position=1),
    )


def test_strategy_uses_lex_active_set_and_repeated_h_matching() -> None:
    state = _state(
        [
            {0: 0.2, 1: 0.9, 2: 0.3},
            {0: 0.8, 1: 0.5, 2: 0.7},
        ]
    )
    state.sampling_strategy_state = {"unrelated": {"kept": True}}
    strategy = RepairableGapSamplingStrategy[int, str](
        top_n=2,
        minibatches_per_wave=3,
    )
    trainset = ListDataLoader(["zero", "one", "two"])

    tasks = strategy.sample_tasks(
        state,
        _failing_selector(),
        _fixed_batch_sampler(((0,), (1,), (2,))),
        trainset,
    )

    assert parent_selection_snapshot(
        state,
        top_n=2,
        score_mode="high_resolution_lexicographic",
    ).selection_active == (0, 1)
    assert [task.parent_idx for task in tasks] == [0, 1, 0]
    assert [task.minibatch_ids for task in tasks] == [[0], [1], [2]]
    assert [task.minibatch for task in tasks] == [["zero"], ["one"], ["two"]]
    assert [task.sampling_metadata["pass_index"] for task in tasks] == [0, 0, 1]
    assert [
        task.sampling_metadata["repairable_gap"] for task in tasks
    ] == pytest.approx([0.6, 0.4, 0.4])
    assert state.sampling_strategy_state == {"unrelated": {"kept": True}}
    assert not hasattr(strategy, "update_after_completed_admissions")
    assert all(
        "unit_draw" not in task.sampling_metadata
        and "sampling_probabilities" not in task.sampling_metadata
        and "ucb_score" not in task.sampling_metadata
        and "completed_arm_count_at_wave_start" not in task.sampling_metadata
        for task in tasks
    )
    json.dumps([task.sampling_metadata for task in tasks], allow_nan=False)


def test_all_zero_repeated_passes_restore_skills_without_capacity_state() -> None:
    state = _state(
        [
            {0: 0.4, 1: 0.4, 2: 0.4},
            {0: 0.4, 1: 0.4, 2: 0.4},
        ]
    )
    strategy = RepairableGapSamplingStrategy[int, str](
        top_n=1,
        minibatches_per_wave=5,
    )
    trainset = ListDataLoader(["zero", "one", "two"])

    tasks = strategy.sample_tasks(
        state,
        _failing_selector(),
        _fixed_batch_sampler(((0,), (1,), (2,), (0,), (1,))),
        trainset,
    )

    assert parent_selection_snapshot(
        state,
        top_n=1,
        score_mode="high_resolution_lexicographic",
    ).selection_active == (0, 1)
    assert [task.parent_idx for task in tasks] == [0, 1, 0, 1, 0]
    assert [task.sampling_metadata["minibatch_position"] for task in tasks] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert [task.sampling_metadata["pass_index"] for task in tasks] == [
        0,
        0,
        1,
        1,
        2,
    ]
