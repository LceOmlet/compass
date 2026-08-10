from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from typing import Any

import pytest

from bridge.b19_reversible_parent_selection import (
    LexicographicHighResolutionScore,
    ReversibleMaskedExposureCorrectedCandidateSelector,
    candidate_selection_rate,
    common_clean_high_resolution_lexicographic_scores,
    common_clean_high_resolution_rates,
    evaluation_count,
    frontier_count,
    frontier_rate,
    high_resolution_frontier_credits,
    high_resolution_lexicographic_score,
    high_resolution_selection_rate,
    parent_selection_snapshot,
    select_top_candidate_idx,
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


def _state_with_fronts(
    fronts: list[set[int]],
    *,
    candidate_count: int,
    parents: list[list[int | None]],
) -> GEPAState:
    state = _state(
        [(0, len(fronts)) for _ in range(candidate_count)],
        parents,
    )
    state.program_at_pareto_front_valset = {
        data_id: set(front)
        for data_id, front in enumerate(fronts)
    }
    assert state.is_consistent()
    return state


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

    active = selection_active_parent_rates(
        state,
        top_n=5,
        score_mode="raw_frontier_rate",
    )

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

    active = selection_active_parent_rates(
        state,
        top_n=5,
        score_mode="raw_frontier_rate",
    )

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
    assert set(
        selection_active_parent_rates(
            lineage_state,
            top_n=5,
            score_mode="raw_frontier_rate",
        )
    ) == {1}

    _set_rate(lineage_state, 1, frontiers=4, exposure=6)
    assert set(
        selection_active_parent_rates(
            lineage_state,
            top_n=5,
            score_mode="raw_frontier_rate",
        )
    ) == {
        0,
        1,
    }

    top_state = _state(
        [(1, 10), (9, 10), (8, 10), (7, 10), (6, 10), (5, 10), (4, 10)],
        [[None], [0], [0], [0], [0], [0], [0]],
    )
    assert set(
        selection_active_parent_rates(
            top_state,
            top_n=5,
            score_mode="raw_frontier_rate",
        )
    ) == {
        1,
        2,
        3,
        4,
        5,
    }

    _set_rate(top_state, 5, frontiers=3, exposure=10)
    assert set(
        selection_active_parent_rates(
            top_state,
            top_n=5,
            score_mode="raw_frontier_rate",
        )
    ) == {
        1,
        2,
        3,
        4,
        6,
    }


def test_common_exposure_mask_removes_seed_structural_optimism() -> None:
    fronts = [{0} for _ in range(150)]
    fronts[147] = {1}
    fronts[148] = {1}
    state = _state_with_fronts(
        fronts,
        candidate_count=2,
        parents=[[None], [0]],
    )
    state.prog_candidate_val_subscores[1] = {
        data_id: 0.0
        for data_id in (147, 148, 149)
    }
    assert state.is_consistent()

    snapshot = parent_selection_snapshot(state, top_n=5)

    assert snapshot.rates == {
        0: Fraction(74, 75),
        1: Fraction(2, 3),
    }
    assert common_clean_high_resolution_rates(state, 0, 1) == (
        Fraction(1, 3),
        Fraction(2, 3),
    )
    assert snapshot.lineage_active == (1,)


def test_no_common_clean_exposure_supplies_no_masking_relation() -> None:
    state = _state_with_fronts(
        [{0}, {0}, {0}, {1}, {1}, {1}],
        candidate_count=2,
        parents=[[None], [0]],
    )
    state.prog_candidate_val_subscores[0] = {
        data_id: 0.0
        for data_id in (0, 1, 2)
    }
    state.prog_candidate_val_subscores[1] = {
        data_id: 0.0
        for data_id in (3, 4, 5)
    }
    assert state.is_consistent()

    assert common_clean_high_resolution_rates(state, 0, 1) is None
    assert parent_selection_snapshot(state, top_n=5).lineage_active == (0, 1)


def test_common_exposure_excludes_recursive_proposal_lineage_scores() -> None:
    state = _state_with_fronts(
        [{0}, {0}, {1}],
        candidate_count=2,
        parents=[[None], [0]],
    )
    state.prog_candidate_val_subscores[0] = {0: 0.0, 1: 0.0}
    state.prog_candidate_val_subscores[1] = {0: 0.0, 2: 0.0}
    state.program_birth_propose_ids = [(), (0,)]
    assert state.is_consistent()

    assert common_clean_high_resolution_rates(state, 0, 1) is None
    assert parent_selection_snapshot(state, top_n=5).lineage_active == (0, 1)


def test_common_exposure_uses_full_official_high_resolution_front() -> None:
    state = _state_with_fronts(
        [{0, 2, 3, 4}, {0, 2, 3, 4}, {1}],
        candidate_count=5,
        parents=[[None], [0], [0], [0], [0]],
    )
    for candidate_idx in (2, 3, 4):
        state.parent_program_for_candidate[candidate_idx] = [None]
        state.program_birth_propose_ids[candidate_idx] = None
    assert state.is_consistent()

    assert common_clean_high_resolution_rates(state, 0, 1) == (
        Fraction(1, 6),
        Fraction(1, 3),
    )
    assert parent_selection_snapshot(
        state,
        top_n=5,
        score_mode="raw_frontier_rate",
    ).lineage_active == (0, 1, 2, 3, 4)
    assert parent_selection_snapshot(state, top_n=5).lineage_active == (
        1,
        2,
        3,
        4,
    )


def test_common_exposure_ancestor_mask_is_reversible() -> None:
    state = _state_with_fronts(
        [{1}, {1}, {0}],
        candidate_count=2,
        parents=[[None], [0]],
    )
    assert parent_selection_snapshot(state, top_n=5).lineage_active == (1,)

    state.program_at_pareto_front_valset[0] = {0}
    state.program_at_pareto_front_valset[1] = {1}
    state.program_at_pareto_front_valset[2] = {0}
    assert parent_selection_snapshot(state, top_n=5).lineage_active == (0, 1)

    state.program_at_pareto_front_valset[0] = {1}
    assert parent_selection_snapshot(state, top_n=5).lineage_active == (1,)


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

    def choice(self, population: list[int]) -> int:
        self.population = list(population)
        self.weights = None
        return population[-1]


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
        score_mode="raw_frontier_rate",
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

    snapshot = parent_selection_snapshot(
        state,
        top_n=5,
        score_mode="raw_frontier_rate",
    )

    assert snapshot.lineage_active == (1, 2, 3, 4, 5, 6, 7, 8)
    assert snapshot.selection_active == (1, 2, 3, 4, 5, 6, 7)
    assert snapshot.top_n_cutoff == Fraction(5, 10)
    assert selection_active_parent_rates(
        state,
        top_n=5,
        score_mode="raw_frontier_rate",
    ) == {
        idx: snapshot.rates[idx]
        for idx in snapshot.selection_active
    }


def test_high_resolution_credit_conserves_one_unit_per_nonempty_front() -> None:
    state = _state(
        [(4, 4), (3, 4), (2, 4)],
        [[None], [0], [0]],
    )

    credits = high_resolution_frontier_credits(state)

    assert credits == {
        0: Fraction(13, 6),
        1: Fraction(7, 6),
        2: Fraction(2, 3),
    }
    assert sum(credits.values(), Fraction(0)) == 4
    assert high_resolution_selection_rate(state, 0) == Fraction(13, 24)
    assert high_resolution_selection_rate(state, 1) == Fraction(7, 24)
    assert high_resolution_selection_rate(state, 2) == Fraction(1, 6)


def test_high_resolution_is_default_and_raw_rate_remains_available() -> None:
    state = _state(
        [(4, 4), (3, 4), (2, 4)],
        [[None], [0], [0]],
    )

    default_snapshot = parent_selection_snapshot(state, top_n=5)
    explicit_snapshot = parent_selection_snapshot(
        state,
        top_n=5,
        score_mode="high_resolution",
    )
    raw_snapshot = parent_selection_snapshot(
        state,
        top_n=5,
        score_mode="raw_frontier_rate",
    )

    assert default_snapshot == explicit_snapshot
    assert default_snapshot.rates == {
        0: Fraction(13, 24),
        1: Fraction(7, 24),
        2: Fraction(1, 6),
    }
    assert default_snapshot.raw_rates == {
        0: Fraction(1, 1),
        1: Fraction(3, 4),
        2: Fraction(1, 2),
    }
    assert raw_snapshot.rates == raw_snapshot.raw_rates
    assert raw_snapshot.high_resolution_credits == {}


def test_legacy_high_resolution_selector_samples_proportional_to_shared_rate() -> None:
    state = _state(
        [(4, 4), (3, 4), (2, 4)],
        [[None], [0], [0]],
    )
    core_before = deepcopy(
        (
            state.program_candidates,
            state.parent_program_for_candidate,
            state.prog_candidate_val_subscores,
            state.program_at_pareto_front_valset,
        )
    )
    raw_before = {
        candidate_idx: (
            frontier_count(state, candidate_idx),
            evaluation_count(state, candidate_idx),
            frontier_rate(state, candidate_idx),
        )
        for candidate_idx in range(3)
    }
    state.full_program_trace.append({"i": 0})
    rng = _RecordingRandom()
    logger = _Logger()
    selector = ReversibleMaskedExposureCorrectedCandidateSelector(
        rng,  # type: ignore[arg-type]
        _Observer(),  # type: ignore[arg-type]
        logger,
        top_n=5,
        score_mode="high_resolution",
    )

    selected = selector.select_candidate_idx(state)

    assert selected == 2
    assert rng.population == [0, 1, 2]
    assert rng.weights == pytest.approx([13 / 24, 7 / 24, 1 / 6])
    assert core_before == (
        state.program_candidates,
        state.parent_program_for_candidate,
        state.prog_candidate_val_subscores,
        state.program_at_pareto_front_valset,
    )
    assert raw_before == {
        candidate_idx: (
            frontier_count(state, candidate_idx),
            evaluation_count(state, candidate_idx),
            frontier_rate(state, candidate_idx),
        )
        for candidate_idx in range(3)
    }
    assert "score_mode=high_resolution" in logger.messages[-1]
    assert "F_over_E=" in logger.messages[-1]
    assert "shared_credit=" in logger.messages[-1]
    assert "high_resolution_rate=" in logger.messages[-1]
    trace = state.full_program_trace[-1]["parent_selection"]
    assert trace["score_mode"] == "high_resolution"
    assert trace["raw_rates"][0] == {"numerator": 1, "denominator": 1}
    assert trace["high_resolution_credits"][0] == {
        "numerator": 13,
        "denominator": 6,
    }
    assert trace["rates"][0] == {"numerator": 13, "denominator": 24}


def test_high_resolution_lineage_mask_is_reversible_from_official_fronts() -> None:
    state = _state_with_fronts(
        [{0}, {1, 2}],
        candidate_count=3,
        parents=[[None], [0], [0]],
    )

    raw_snapshot = parent_selection_snapshot(
        state,
        top_n=5,
        score_mode="raw_frontier_rate",
    )
    shared_snapshot = parent_selection_snapshot(state, top_n=5)

    assert raw_snapshot.rates == {
        0: Fraction(1, 2),
        1: Fraction(1, 2),
        2: Fraction(1, 2),
    }
    assert raw_snapshot.lineage_active == (1, 2)
    assert shared_snapshot.rates == {
        0: Fraction(1, 2),
        1: Fraction(1, 4),
        2: Fraction(1, 4),
    }
    assert shared_snapshot.lineage_active == (0, 1, 2)

    state.program_at_pareto_front_valset[1] = {1}
    assert parent_selection_snapshot(state, top_n=5).lineage_active == (1,)

    state.program_at_pareto_front_valset[1] = {1, 2}
    assert parent_selection_snapshot(state, top_n=5).lineage_active == (0, 1, 2)


def test_high_resolution_top_five_keeps_exact_boundary_ties() -> None:
    state = _state_with_fronts(
        [{1}, {2}, {3}, {4}, {5, 6}, {7, 8, 9}],
        candidate_count=10,
        parents=[[None], *([[0]] * 9)],
    )

    shared_snapshot = parent_selection_snapshot(state, top_n=5)
    raw_snapshot = parent_selection_snapshot(
        state,
        top_n=5,
        score_mode="raw_frontier_rate",
    )

    assert shared_snapshot.top_n_cutoff == Fraction(1, 12)
    assert shared_snapshot.selection_active == (1, 2, 3, 4, 5, 6)
    assert shared_snapshot.rates[5] == shared_snapshot.rates[6]
    assert shared_snapshot.rates[7] == Fraction(1, 18)
    assert raw_snapshot.top_n_cutoff == Fraction(1, 6)
    assert raw_snapshot.selection_active == tuple(range(1, 10))


def test_lexicographic_high_resolution_does_not_multiply_hit_rate_and_resolution() -> None:
    state = _state_with_fronts(
        [
            {0, 2, 3, 4, 5},
            {0, 2, 3, 4, 5},
            {0, 2, 3, 4, 5},
            {1},
            {1},
        ],
        candidate_count=6,
        parents=[[None], *([[0]] * 5)],
    )
    for candidate_idx in range(1, 6):
        state.parent_program_for_candidate[candidate_idx] = [None]
        state.program_birth_propose_ids[candidate_idx] = None
    assert state.is_consistent()

    assert high_resolution_selection_rate(state, 0) == Fraction(3, 25)
    assert high_resolution_selection_rate(state, 1) == Fraction(2, 5)
    assert select_top_candidate_idx(state, score_mode="high_resolution") == 1

    assert high_resolution_lexicographic_score(
        state,
        0,
    ) == LexicographicHighResolutionScore(
        frontier_rate=Fraction(3, 5),
        tie_resolution=Fraction(1, 5),
    )
    assert high_resolution_lexicographic_score(
        state,
        1,
    ) == LexicographicHighResolutionScore(
        frontier_rate=Fraction(2, 5),
        tie_resolution=Fraction(1),
    )
    assert (
        select_top_candidate_idx(
            state,
            score_mode="high_resolution_lexicographic",
        )
        == 0
    )


def test_lexicographic_high_resolution_never_exposes_a_scalar_rate() -> None:
    state = _state([(0, 0)], [[None]])

    with pytest.raises(ValueError, match="no scalar selection rate"):
        candidate_selection_rate(
            state,
            0,
            score_mode="high_resolution_lexicographic",
        )


def test_lexicographic_high_resolution_uses_resolution_only_after_equal_hit_rate() -> None:
    state = _state_with_fronts(
        [{0, 2}, {0, 2}, {1}, {1}],
        candidate_count=3,
        parents=[[None], [0], [0]],
    )
    for candidate_idx in (1, 2):
        state.parent_program_for_candidate[candidate_idx] = [None]
        state.program_birth_propose_ids[candidate_idx] = None
    assert state.is_consistent()

    assert high_resolution_lexicographic_score(
        state,
        0,
    ) == LexicographicHighResolutionScore(
        frontier_rate=Fraction(1, 2),
        tie_resolution=Fraction(1, 2),
    )
    assert high_resolution_lexicographic_score(
        state,
        1,
    ) == LexicographicHighResolutionScore(
        frontier_rate=Fraction(1, 2),
        tie_resolution=Fraction(1),
    )
    assert (
        select_top_candidate_idx(
            state,
            score_mode="high_resolution_lexicographic",
        )
        == 1
    )


def test_lexicographic_high_resolution_masks_on_common_domain_tuple() -> None:
    state = _state_with_fronts(
        [{0, 2}, {0, 2}, {1}, {1}],
        candidate_count=3,
        parents=[[None], [0], [0]],
    )
    state.parent_program_for_candidate[2] = [None]
    state.program_birth_propose_ids[2] = None
    assert state.is_consistent()

    assert common_clean_high_resolution_lexicographic_scores(
        state,
        0,
        1,
    ) == (
        LexicographicHighResolutionScore(
            frontier_rate=Fraction(1, 2),
            tie_resolution=Fraction(1, 2),
        ),
        LexicographicHighResolutionScore(
            frontier_rate=Fraction(1, 2),
            tie_resolution=Fraction(1),
        ),
    )
    assert parent_selection_snapshot(
        state,
        top_n=5,
        score_mode="high_resolution_lexicographic",
    ).lineage_active == (1, 2)


def test_lexicographic_high_resolution_keeps_full_key_boundary_ties_and_samples_uniformly() -> None:
    state = _state_with_fronts(
        [{0}, {1}, {2, 3}, {2, 3}, {4, 5, 6}, {4, 5, 6}],
        candidate_count=7,
        parents=[[None], *([[0]] * 6)],
    )
    for candidate_idx in range(1, 7):
        state.parent_program_for_candidate[candidate_idx] = [None]
        state.program_birth_propose_ids[candidate_idx] = None
    assert state.is_consistent()
    state.full_program_trace.append({"i": 0})

    snapshot = parent_selection_snapshot(
        state,
        top_n=3,
        score_mode="high_resolution_lexicographic",
    )

    assert snapshot.selection_active == (2, 3, 4, 5, 6)
    assert snapshot.top_n_cutoff_score == LexicographicHighResolutionScore(
        frontier_rate=Fraction(1, 3),
        tie_resolution=Fraction(1, 3),
    )
    assert snapshot.selection_scores[4] == snapshot.selection_scores[5]
    assert snapshot.selection_scores[5] == snapshot.selection_scores[6]

    rng = _RecordingRandom()
    logger = _Logger()
    selector = ReversibleMaskedExposureCorrectedCandidateSelector(
        rng,  # type: ignore[arg-type]
        _Observer(),  # type: ignore[arg-type]
        logger,
        top_n=3,
        score_mode="high_resolution_lexicographic",
    )
    selected = selector.select_candidate_idx(state)

    assert selected == 6
    assert rng.population == [2, 3, 4, 5, 6]
    assert rng.weights is None
    assert "sampling_mode=uniform_over_lexicographic_top_n" in logger.messages[-1]
    trace = state.full_program_trace[-1]["parent_selection"]
    assert trace["sampling_mode"] == "uniform_over_lexicographic_top_n"
    assert trace["sampling_probabilities"] == {
        candidate_idx: {"numerator": 1, "denominator": 5}
        for candidate_idx in (2, 3, 4, 5, 6)
    }
    assert trace["selection_scores"][4] == {
        "frontier_rate": {"numerator": 1, "denominator": 3},
        "tie_resolution": {"numerator": 1, "denominator": 3},
    }
