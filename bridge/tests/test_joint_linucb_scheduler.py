from __future__ import annotations

import math
import random
from fractions import Fraction
from itertools import permutations
from types import SimpleNamespace
from typing import Any

import pytest

from bridge.b19_reversible_parent_selection import parent_selection_snapshot
from bridge.b21_joint_linucb_scheduler import (
    JOINT_LINUCB_STATE_KEY,
    JointAssignment,
    JointLinUCBSamplingStrategy,
    linucb_score,
    observed_arm_context,
    rectangular_hungarian_assignment,
)
from gepa.core.data_loader import ListDataLoader
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.strategies.proposal_sampling import CompletedAdmissionOutcome


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


def _outcome(
    *,
    task_index: int,
    parent_idx: int,
    context: tuple[float, float, float, float],
    before: tuple[float, ...] = (0.0,),
    after: tuple[float, ...] = (0.0,),
    iteration: int = 1,
) -> CompletedAdmissionOutcome[int]:
    return CompletedAdmissionOutcome(
        iteration=iteration,
        task_index=task_index,
        proposal_id=f"{iteration}-{task_index}",
        parent_idx=parent_idx,
        minibatch_ids=(task_index,),
        admission_ids=tuple(range(len(before))),
        sampling_metadata={
            "scheduler": "joint_linucb",
            "schema_version": 1,
            "wave_iteration": iteration,
            "assignment_index": task_index,
            "parent_idx": parent_idx,
            "context": context,
        },
        scores_before=before,
        scores_after=after,
    )


def test_relaxed_observed_context_keeps_unknown_distinct_from_zero() -> None:
    state = _state(
        [
            {0: 0.2, 1: 0.8},
            {0: 0.7},
        ]
    )

    context_zero = observed_arm_context(
        state,
        active_program_indices=(0, 1),
        program_idx=0,
        minibatch_ids=(0, 1),
        perfect_score=1.0,
    )
    context_one = observed_arm_context(
        state,
        active_program_indices=(0, 1),
        program_idx=1,
        minibatch_ids=(0, 1),
        perfect_score=1.0,
    )

    assert context_zero.as_tuple() == pytest.approx((1.0, 0.25, 0.25, 0.5))
    assert context_one.as_tuple() == pytest.approx((1.0, 0.0, 0.25, 0.5))


def _brute_force_assignment(
    *,
    programs: tuple[int, ...],
    positions: tuple[int, ...],
    scores: dict[tuple[int, int], float],
    counts: dict[int, int],
) -> tuple[JointAssignment, ...]:
    candidates: list[tuple[JointAssignment, ...]] = []
    if len(positions) <= len(programs):
        for assigned_programs in permutations(programs, len(positions)):
            candidates.append(
                tuple(
                    JointAssignment(program_idx, position)
                    for position, program_idx in zip(
                        positions,
                        assigned_programs,
                        strict=True,
                    )
                )
            )
    else:
        for assigned_positions in permutations(positions, len(programs)):
            candidates.append(
                tuple(
                    sorted(
                        (
                            JointAssignment(program_idx, position)
                            for program_idx, position in zip(
                                programs,
                                assigned_positions,
                                strict=True,
                            )
                        ),
                        key=lambda item: (
                            item.minibatch_position,
                            item.program_idx,
                        ),
                    )
                )
            )

    def is_better(
        candidate: tuple[JointAssignment, ...],
        incumbent: tuple[JointAssignment, ...],
    ) -> bool:
        candidate_score = sum(
            (
                Fraction.from_float(
                    scores[(item.program_idx, item.minibatch_position)]
                )
                for item in candidate
            ),
            Fraction(0),
        )
        incumbent_score = sum(
            (
                Fraction.from_float(
                    scores[(item.program_idx, item.minibatch_position)]
                )
                for item in incumbent
            ),
            Fraction(0),
        )
        if candidate_score != incumbent_score:
            return candidate_score > incumbent_score
        candidate_count = sum(counts[item.program_idx] for item in candidate)
        incumbent_count = sum(counts[item.program_idx] for item in incumbent)
        if candidate_count != incumbent_count:
            return candidate_count < incumbent_count
        candidate_sequence = tuple(
            (item.minibatch_position, item.program_idx)
            for item in candidate
        )
        incumbent_sequence = tuple(
            (item.minibatch_position, item.program_idx)
            for item in incumbent
        )
        return candidate_sequence < incumbent_sequence

    best = candidates[0]
    for candidate in candidates[1:]:
        if is_better(candidate, best):
            best = candidate
    return tuple(
        sorted(
            best,
            key=lambda item: (item.minibatch_position, item.program_idx),
        )
    )


def test_rectangular_hungarian_matches_exact_brute_force_objectives() -> None:
    rng = random.Random(719)
    for program_count in range(1, 5):
        for minibatch_count in range(1, 5):
            programs = tuple(range(2, 2 + program_count))
            positions = tuple(range(minibatch_count))
            for _ in range(20):
                score_values = (-1.25, -0.5, 0.0, 0.1, 0.2, 0.3, 1.5)
                scores = {
                    (program_idx, position): rng.choice(score_values)
                    for program_idx in programs
                    for position in positions
                }
                counts = {
                    program_idx: rng.randrange(4)
                    for program_idx in programs
                }

                actual = rectangular_hungarian_assignment(
                    program_indices=programs,
                    minibatch_positions=positions,
                    scores=scores,
                    completed_arm_counts=counts,
                    total_minibatches=minibatch_count,
                )
                expected = _brute_force_assignment(
                    programs=programs,
                    positions=positions,
                    scores=scores,
                    counts=counts,
                )

                assert actual == expected


def test_tie_sequence_prefers_earliest_batches_then_program_indices() -> None:
    programs = (2, 5)
    positions = (0, 1, 2)
    scores = {
        (program_idx, position): 0.0
        for program_idx in programs
        for position in positions
    }

    assert rectangular_hungarian_assignment(
        program_indices=programs,
        minibatch_positions=positions,
        scores=scores,
        completed_arm_counts={},
        total_minibatches=3,
    ) == (
        JointAssignment(program_idx=2, minibatch_position=0),
        JointAssignment(program_idx=5, minibatch_position=1),
    )


def test_wave_uses_b19_active_set_and_restores_skills_between_passes() -> None:
    state = _state(
        [
            {0: 1.0, 1: 1.0, 2: 1.0},
            {0: 1.0, 1: 1.0, 2: 1.0},
        ]
    )
    strategy = JointLinUCBSamplingStrategy[int, str](
        top_n=1,
        minibatches_per_wave=3,
        perfect_score=1.0,
    )
    count_outcomes = tuple(
        [
            _outcome(
                task_index=task_index,
                parent_idx=0,
                context=(1.0, 0.0, 0.0, 0.0),
            )
            for task_index in range(5)
        ]
        + [
            _outcome(
                task_index=5,
                parent_idx=1,
                context=(1.0, 0.0, 0.0, 0.0),
            )
        ]
    )
    state.sampling_strategy_state = dict(
        strategy.update_after_completed_admissions(
            current_state={},
            outcomes=count_outcomes,
        )
    )
    batches = iter(([0], [1], [2]))
    batch_sampler = SimpleNamespace(
        next_minibatch_ids=lambda trainset, current_state: list(next(batches))
    )
    candidate_selector = SimpleNamespace(
        select_candidate_idx=lambda current_state: pytest.fail(
            "joint scheduler must not invoke scalar parent sampling"
        )
    )
    trainset = ListDataLoader(["zero", "one", "two"])

    tasks = strategy.sample_tasks(
        state,
        candidate_selector,
        batch_sampler,
        trainset,
    )

    assert parent_selection_snapshot(
        state,
        top_n=1,
        score_mode="high_resolution_lexicographic",
    ).selection_active == (0, 1)
    assert [task.parent_idx for task in tasks] == [0, 1, 1]
    assert [task.minibatch_ids for task in tasks] == [[0], [1], [2]]
    assert [task.minibatch for task in tasks] == [["zero"], ["one"], ["two"]]
    assert [task.sampling_metadata["pass_index"] for task in tasks] == [0, 0, 1]
    assert [
        task.sampling_metadata["completed_arm_count_at_wave_start"]
        for task in tasks
    ] == [5, 1, 1]


def test_completed_admissions_update_shared_statistics_with_signed_gain() -> None:
    strategy = JointLinUCBSamplingStrategy[int, Any](
        top_n=1,
        minibatches_per_wave=1,
        perfect_score=1.0,
    )
    first = _outcome(
        task_index=0,
        parent_idx=3,
        context=(1.0, 2.0, 0.0, 0.0),
        before=(0.5, 0.25),
        after=(0.25, 0.25),
    )
    second = _outcome(
        task_index=1,
        parent_idx=7,
        context=(1.0, 0.0, 1.0, 0.0),
        before=(0.0, 0.0),
        after=(1.0, 1.0),
    )

    forward = strategy.update_after_completed_admissions(
        current_state={"unrelated": {"kept": True}},
        outcomes=(first, second),
    )
    reverse = strategy.update_after_completed_admissions(
        current_state={"unrelated": {"kept": True}},
        outcomes=(second, first),
    )

    assert forward == reverse
    assert forward["unrelated"] == {"kept": True}
    checkpoint = forward[JOINT_LINUCB_STATE_KEY]
    assert checkpoint["covariance"] == [
        [3.0, 2.0, 1.0, 0.0],
        [2.0, 5.0, 0.0, 0.0],
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert checkpoint["response"] == pytest.approx(
        [0.875, -0.25, 1.0, 0.0]
    )
    assert checkpoint["completed_arm_counts"] == {3: 1, 7: 1}
    assert math.isfinite(
        linucb_score(
            (1.0, 0.5, 0.25, 0.0),
            covariance=checkpoint["covariance"],
            response=checkpoint["response"],
        )
    )


def test_absent_completed_outcomes_leave_checkpoint_unchanged() -> None:
    strategy = JointLinUCBSamplingStrategy[int, Any](
        top_n=1,
        minibatches_per_wave=1,
        perfect_score=1.0,
    )
    current = {"unrelated": {"kept": True}}

    assert strategy.update_after_completed_admissions(
        current_state=current,
        outcomes=(),
    ) == current


def test_completed_outcome_must_retain_owner_task_identity() -> None:
    strategy = JointLinUCBSamplingStrategy[int, Any](
        top_n=1,
        minibatches_per_wave=1,
        perfect_score=1.0,
    )
    outcome = _outcome(
        task_index=0,
        parent_idx=3,
        context=(1.0, 0.0, 0.0, 0.0),
    )
    changed = CompletedAdmissionOutcome(
        iteration=outcome.iteration,
        task_index=outcome.task_index,
        proposal_id=outcome.proposal_id,
        parent_idx=4,
        minibatch_ids=outcome.minibatch_ids,
        admission_ids=outcome.admission_ids,
        sampling_metadata=outcome.sampling_metadata,
        scores_before=outcome.scores_before,
        scores_after=outcome.scores_after,
    )

    with pytest.raises(ValueError, match="parent identity changed"):
        strategy.update_after_completed_admissions(
            current_state={},
            outcomes=(changed,),
        )
