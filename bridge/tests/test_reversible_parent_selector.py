from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from typing import Any

import pytest

from bridge.b19_reversible_parent_selection import (
    ReversibleMaskedExposureCorrectedCandidateSelector,
    parent_selection_snapshot,
    selection_active_parent_rates,
)
from gepa.core.state import GEPAState, ValsetEvaluation


def _state(
    rates: list[tuple[int, int]],
    parents: list[list[int | None]],
) -> GEPAState:
    max_exposure = max(exposure for _, exposure in rates)
    ids = range(max_exposure)
    state = GEPAState(
        {"prompt": "seed"},
        ValsetEvaluation(
            outputs_by_val_id={idx: "" for idx in ids},
            scores_by_val_id={idx: 0.0 for idx in ids},
            objective_scores_by_val_id=None,
        ),
        track_best_outputs=False,
        frontier_type="instance",
    )
    state.program_candidates = [
        {"prompt": f"skill-{idx}"}
        for idx in range(len(rates))
    ]
    state.parent_program_for_candidate = [list(value) for value in parents]
    state.program_birth_propose_ids = [() for _ in rates]
    state.prog_candidate_val_subscores = [
        {data_id: 0.0 for data_id in range(exposure)}
        for _, exposure in rates
    ]
    state.prog_candidate_objective_scores = [{} for _ in rates]
    state.named_predictor_id_to_update_next_for_program_candidate = [
        0 for _ in rates
    ]
    state.num_metric_calls_by_discovery = [0 for _ in rates]
    state.pareto_front_valset = {
        data_id: 0.0
        for data_id in range(max_exposure)
    }
    state.program_at_pareto_front_valset = {
        data_id: {
            candidate_idx
            for candidate_idx, (frontiers, _) in enumerate(rates)
            if data_id < frontiers
        }
        for data_id in range(max_exposure)
    }
    assert state.is_consistent()
    return state


def _set_rate(
    state: GEPAState,
    candidate_idx: int,
    *,
    frontiers: int,
    exposure: int,
) -> None:
    state.prog_candidate_val_subscores[candidate_idx] = {
        data_id: 0.0
        for data_id in range(exposure)
    }
    for data_id, front in state.program_at_pareto_front_valset.items():
        front.discard(candidate_idx)
        if data_id < frontiers:
            front.add(candidate_idx)
    assert state.is_consistent()


@pytest.mark.parametrize(
    ("rates", "parents", "expected"),
    [
        (
            [(4, 5), (3, 5), (4, 5)],
            [[None], [0], [1]],
            {2},
        ),
        (
            [(4, 5), (3, 5), (1, 2)],
            [[None], [0], [1]],
            {0, 1, 2},
        ),
        (
            [(4, 5), (9, 10), (7, 10)],
            [[None], [0], [0]],
            {1, 2},
        ),
    ],
)
def test_reversible_ancestor_mask_uses_transitive_descendants_only(
    rates: list[tuple[int, int]],
    parents: list[list[int | None]],
    expected: set[int],
) -> None:
    state = _state(rates, parents)

    active = selection_active_parent_rates(state, top_n=5)

    assert set(active) == expected


def test_ancestor_mask_precedes_tie_inclusive_global_top_five() -> None:
    state = _state(
        [
            (1, 10),
            (9, 10),
            (8, 10),
            (7, 10),
            (6, 10),
            (5, 10),
            (5, 10),
            (5, 10),
            (4, 10),
            (4, 10),
            (4, 10),
        ],
        [
            [None],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [9],
        ],
    )

    active = selection_active_parent_rates(state, top_n=5)

    assert active == {
        1: Fraction(9, 10),
        2: Fraction(4, 5),
        3: Fraction(7, 10),
        4: Fraction(3, 5),
        5: Fraction(1, 2),
        6: Fraction(1, 2),
        7: Fraction(1, 2),
    }


def test_both_masks_recompute_from_current_official_state() -> None:
    lineage_state = _state(
        [(8, 10), (3, 3)],
        [[None], [0]],
    )
    assert set(selection_active_parent_rates(lineage_state, top_n=5)) == {1}

    _set_rate(lineage_state, 1, frontiers=4, exposure=6)
    assert set(selection_active_parent_rates(lineage_state, top_n=5)) == {
        0,
        1,
    }

    top_state = _state(
        [(1, 10), (9, 10), (8, 10), (7, 10), (6, 10), (5, 10), (4, 10)],
        [[None], [0], [0], [0], [0], [0], [0]],
    )
    assert set(selection_active_parent_rates(top_state, top_n=5)) == {
        1,
        2,
        3,
        4,
        5,
    }

    _set_rate(top_state, 5, frontiers=3, exposure=10)
    assert set(selection_active_parent_rates(top_state, top_n=5)) == {
        1,
        2,
        3,
        4,
        6,
    }


class _RecordingRandom:
    def __init__(self) -> None:
        self.population: list[int] | None = None
        self.weights: list[float] | None = None

    def choices(
        self,
        population: list[int],
        *,
        weights: list[float],
        k: int,
    ) -> list[int]:
        assert k == 1
        self.population = list(population)
        self.weights = list(weights)
        return [population[-1]]


class _Observer:
    def __init__(self) -> None:
        self.candidates: list[dict[str, str]] | None = None

    def update_candidate_pool(self, candidates: Any) -> None:
        self.candidates = list(candidates)


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


def test_selector_samples_only_from_the_recomputed_active_set() -> None:
    state = _state(
        [
            (1, 10),
            (9, 10),
            (8, 10),
            (7, 10),
            (6, 10),
            (5, 10),
            (5, 10),
            (5, 10),
            (4, 10),
        ],
        [[None], [0], [0], [0], [0], [0], [0], [0], [0]],
    )
    before = deepcopy(
        (
            state.program_candidates,
            state.parent_program_for_candidate,
            state.prog_candidate_val_subscores,
            state.program_at_pareto_front_valset,
        )
    )
    rng = _RecordingRandom()
    observer = _Observer()
    selector = ReversibleMaskedExposureCorrectedCandidateSelector(
        rng,  # type: ignore[arg-type]
        observer,  # type: ignore[arg-type]
        _Logger(),
        top_n=5,
    )

    selected = selector.select_candidate_idx(state)

    assert selected == 7
    assert rng.population == [1, 2, 3, 4, 5, 6, 7]
    assert rng.weights == pytest.approx([0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5])
    assert observer.candidates == state.program_candidates
    assert before == (
        state.program_candidates,
        state.parent_program_for_candidate,
        state.prog_candidate_val_subscores,
        state.program_at_pareto_front_valset,
    )


def test_parent_selection_snapshot_exposes_lineage_and_boundary_ties() -> None:
    state = _state(
        [
            (1, 10),
            (9, 10),
            (8, 10),
            (7, 10),
            (6, 10),
            (5, 10),
            (5, 10),
            (5, 10),
            (4, 10),
        ],
        [[None], [0], [0], [0], [0], [0], [0], [0], [0]],
    )

    snapshot = parent_selection_snapshot(state, top_n=5)

    assert snapshot.lineage_active == (1, 2, 3, 4, 5, 6, 7, 8)
    assert snapshot.selection_active == (1, 2, 3, 4, 5, 6, 7)
    assert snapshot.top_n_cutoff == Fraction(5, 10)
    assert selection_active_parent_rates(state, top_n=5) == {
        idx: snapshot.rates[idx]
        for idx in snapshot.selection_active
    }
