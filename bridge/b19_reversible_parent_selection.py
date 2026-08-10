from __future__ import annotations

import random
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Protocol

from gepa.core.state import GEPAState
from gepa.logging.logger import LoggerProtocol


class CandidatePoolObserver(Protocol):
    def update_candidate_pool(
        self,
        candidates: Sequence[Mapping[str, str]],
    ) -> None: ...


SelectionScoreMode = Literal[
    "raw_frontier_rate",
    "high_resolution",
    "high_resolution_lexicographic",
]


@dataclass(frozen=True, order=True, slots=True)
class LexicographicHighResolutionScore:
    """Ordinal parent score that keeps hit rate and tie resolution separate."""

    frontier_rate: Fraction
    tie_resolution: Fraction


SelectionScore = Fraction | LexicographicHighResolutionScore


def _high_resolution_frontier_credit(
    front: Collection[int],
    candidate_idx: int,
) -> Fraction:
    if candidate_idx not in front:
        return Fraction(0)
    return Fraction(1, len(front))


def frontier_count(state: GEPAState, candidate_idx: int) -> int:
    return sum(
        candidate_idx in front
        for front in state.program_at_pareto_front_valset.values()
    )


def evaluation_count(state: GEPAState, candidate_idx: int) -> int:
    return state.get_program_average_val_subset(candidate_idx)[1]


def frontier_rate(state: GEPAState, candidate_idx: int) -> float:
    exposure = evaluation_count(state, candidate_idx)
    if exposure == 0:
        return 0.0
    return frontier_count(state, candidate_idx) / exposure


def high_resolution_frontier_credits(
    state: GEPAState,
) -> dict[int, Fraction]:
    """Share one selection-credit unit among each official clean frontier."""

    credits = {
        candidate_idx: Fraction(0)
        for candidate_idx in range(len(state.program_candidates))
    }
    for front in state.program_at_pareto_front_valset.values():
        if not front:
            continue
        for candidate_idx in front:
            credits[candidate_idx] += _high_resolution_frontier_credit(
                front,
                candidate_idx,
            )
    return credits


def high_resolution_selection_rate(
    state: GEPAState,
    candidate_idx: int,
) -> Fraction:
    """Return shared frontier credit per unchanged clean exposure."""

    exposure = evaluation_count(state, candidate_idx)
    if exposure == 0:
        return Fraction(0)
    return high_resolution_frontier_credits(state)[candidate_idx] / exposure


def high_resolution_lexicographic_score(
    state: GEPAState,
    candidate_idx: int,
) -> LexicographicHighResolutionScore:
    """Return ``(F/E, C/F)`` without multiplying the two dimensions.

    ``F/E`` is the clean frontier hit rate. ``C/F`` is the average shared
    credit conditional on a frontier hit, so it refines rather than reverses
    the primary hit-rate ordering.
    """

    exposure = evaluation_count(state, candidate_idx)
    frontiers = frontier_count(state, candidate_idx)
    if exposure == 0 or frontiers == 0:
        return LexicographicHighResolutionScore(Fraction(0), Fraction(0))
    credit = high_resolution_frontier_credits(state)[candidate_idx]
    return LexicographicHighResolutionScore(
        frontier_rate=Fraction(frontiers, exposure),
        tie_resolution=credit / frontiers,
    )


def candidate_selection_rate(
    state: GEPAState,
    candidate_idx: int,
    *,
    score_mode: SelectionScoreMode,
) -> Fraction:
    """Return the configured exact selection score on unchanged clean evidence."""

    if score_mode == "high_resolution_lexicographic":
        raise ValueError(
            "high_resolution_lexicographic has no scalar selection rate; "
            "use candidate_selection_score"
        )
    exposure = evaluation_count(state, candidate_idx)
    if exposure == 0:
        return Fraction(0)
    if score_mode == "raw_frontier_rate":
        return Fraction(frontier_count(state, candidate_idx), exposure)
    if score_mode == "high_resolution":
        return high_resolution_selection_rate(state, candidate_idx)
    raise ValueError(f"unsupported selection score mode: {score_mode!r}")


def candidate_selection_score(
    state: GEPAState,
    candidate_idx: int,
    *,
    score_mode: SelectionScoreMode,
) -> SelectionScore:
    """Return the exact configured scalar or lexicographic parent score."""

    if score_mode == "high_resolution_lexicographic":
        return high_resolution_lexicographic_score(state, candidate_idx)
    return candidate_selection_rate(
        state,
        candidate_idx,
        score_mode=score_mode,
    )


def select_top_candidate_idx(
    state: GEPAState,
    *,
    score_mode: SelectionScoreMode,
) -> int:
    """Select Top-1 by configured score, clean exposure, then earliest index."""

    eligible = tuple(
        candidate_idx
        for candidate_idx in range(len(state.program_candidates))
        if evaluation_count(state, candidate_idx) > 0
    )
    if not eligible:
        return 0
    return max(
        eligible,
        key=lambda candidate_idx: (
            candidate_selection_score(
                state,
                candidate_idx,
                score_mode=score_mode,
            ),
            evaluation_count(state, candidate_idx),
            -candidate_idx,
        ),
    )


def common_clean_high_resolution_rates(
    state: GEPAState,
    ancestor_idx: int,
    descendant_idx: int,
) -> tuple[Fraction, Fraction] | None:
    """Compare a lineage pair only on their common clean exposure domain.

    The returned tuple is ``(ancestor_rate, descendant_rate)``. Frontier
    ownership and high-resolution sharing remain the official global values
    for each common instance; the pair is not treated as a new frontier.
    ``None`` means that the pair has no jointly clean observation and therefore
    supplies no masking relation.
    """

    clean_ancestor_ids = {
        data_id
        for data_id in state.prog_candidate_val_subscores[ancestor_idx]
        if state.is_program_frontier_eligible(ancestor_idx, data_id)
    }
    clean_descendant_ids = {
        data_id
        for data_id in state.prog_candidate_val_subscores[descendant_idx]
        if state.is_program_frontier_eligible(descendant_idx, data_id)
    }
    common_ids = clean_ancestor_ids.intersection(clean_descendant_ids)
    if not common_ids:
        return None

    ancestor_credit = Fraction(0)
    descendant_credit = Fraction(0)
    for data_id in common_ids:
        front = state.program_at_pareto_front_valset.get(data_id, set())
        ancestor_credit += _high_resolution_frontier_credit(
            front,
            ancestor_idx,
        )
        descendant_credit += _high_resolution_frontier_credit(
            front,
            descendant_idx,
        )
    common_exposure = len(common_ids)
    return (
        ancestor_credit / common_exposure,
        descendant_credit / common_exposure,
    )


def common_clean_high_resolution_lexicographic_scores(
    state: GEPAState,
    ancestor_idx: int,
    descendant_idx: int,
) -> tuple[
    LexicographicHighResolutionScore,
    LexicographicHighResolutionScore,
] | None:
    """Return lineage scores on the pair's common clean exposure domain."""

    clean_ancestor_ids = {
        data_id
        for data_id in state.prog_candidate_val_subscores[ancestor_idx]
        if state.is_program_frontier_eligible(ancestor_idx, data_id)
    }
    clean_descendant_ids = {
        data_id
        for data_id in state.prog_candidate_val_subscores[descendant_idx]
        if state.is_program_frontier_eligible(descendant_idx, data_id)
    }
    common_ids = clean_ancestor_ids.intersection(clean_descendant_ids)
    if not common_ids:
        return None

    def score(candidate_idx: int) -> LexicographicHighResolutionScore:
        frontiers = 0
        credit = Fraction(0)
        for data_id in common_ids:
            front = state.program_at_pareto_front_valset.get(data_id, set())
            if candidate_idx in front:
                frontiers += 1
                credit += _high_resolution_frontier_credit(front, candidate_idx)
        return LexicographicHighResolutionScore(
            frontier_rate=Fraction(frontiers, len(common_ids)),
            tie_resolution=(credit / frontiers if frontiers else Fraction(0)),
        )

    return score(ancestor_idx), score(descendant_idx)


@dataclass(frozen=True, slots=True)
class ParentSelectionSnapshot:
    """Reporting view of the two recomputed reversible masks."""

    score_mode: SelectionScoreMode
    rates: Mapping[int, Fraction]
    selection_scores: Mapping[int, SelectionScore]
    raw_rates: Mapping[int, Fraction]
    high_resolution_credits: Mapping[int, Fraction]
    tie_resolutions: Mapping[int, Fraction]
    lineage_active: tuple[int, ...]
    selection_active: tuple[int, ...]
    top_n_cutoff: Fraction | None
    top_n_cutoff_score: SelectionScore | None
    sampling_mode: str
    sampling_probabilities: Mapping[int, Fraction]


def parent_selection_snapshot(
    state: GEPAState,
    *,
    top_n: int,
    score_mode: SelectionScoreMode = "high_resolution",
) -> ParentSelectionSnapshot:
    """Derive reversible ancestor and tie-inclusive global top-N masks."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if score_mode not in (
        "raw_frontier_rate",
        "high_resolution",
        "high_resolution_lexicographic",
    ):
        raise ValueError(f"unsupported selection score mode: {score_mode!r}")
    raw_rates = {
        candidate_idx: Fraction(frontiers, exposure)
        for candidate_idx in range(len(state.program_candidates))
        if (exposure := evaluation_count(state, candidate_idx)) > 0
        and (frontiers := frontier_count(state, candidate_idx)) > 0
    }
    high_resolution_credits = (
        high_resolution_frontier_credits(state)
        if score_mode in (
            "high_resolution",
            "high_resolution_lexicographic",
        )
        else {}
    )
    shared_rates = (
        {
            candidate_idx: high_resolution_credits[candidate_idx]
            / evaluation_count(state, candidate_idx)
            for candidate_idx in raw_rates
        }
        if high_resolution_credits
        else {}
    )
    tie_resolutions = (
        {
            candidate_idx: high_resolution_credits[candidate_idx]
            / frontier_count(state, candidate_idx)
            for candidate_idx in raw_rates
        }
        if high_resolution_credits
        else {}
    )
    if score_mode == "raw_frontier_rate":
        rates = raw_rates
        selection_scores: dict[int, SelectionScore] = dict(raw_rates)
        sampling_mode = "proportional_to_raw_frontier_rate"
    elif score_mode == "high_resolution":
        rates = shared_rates
        selection_scores = dict(shared_rates)
        sampling_mode = "proportional_to_shared_credit_rate"
    else:
        rates = {}
        selection_scores = {
            candidate_idx: LexicographicHighResolutionScore(
                frontier_rate=raw_rates[candidate_idx],
                tie_resolution=tie_resolutions[candidate_idx],
            )
            for candidate_idx in raw_rates
        }
        sampling_mode = "uniform_over_lexicographic_top_n"

    lineage_active = set(selection_scores)
    for descendant_idx, descendant_score in selection_scores.items():
        seen: set[int] = set()
        pending = [
            parent_idx
            for parent_idx in state.parent_program_for_candidate[descendant_idx]
            if parent_idx is not None
        ]
        while pending:
            ancestor_idx = pending.pop()
            if ancestor_idx in seen:
                continue
            seen.add(ancestor_idx)
            if ancestor_idx in selection_scores:
                if score_mode == "high_resolution":
                    common_rates = common_clean_high_resolution_rates(
                        state,
                        ancestor_idx,
                        descendant_idx,
                    )
                    masks_ancestor = (
                        common_rates is not None
                        and common_rates[1] >= common_rates[0]
                    )
                elif score_mode == "high_resolution_lexicographic":
                    common_scores = (
                        common_clean_high_resolution_lexicographic_scores(
                            state,
                            ancestor_idx,
                            descendant_idx,
                        )
                    )
                    masks_ancestor = (
                        common_scores is not None
                        and common_scores[1] >= common_scores[0]
                    )
                else:
                    masks_ancestor = (
                        descendant_score >= selection_scores[ancestor_idx]
                    )
                if masks_ancestor:
                    lineage_active.discard(ancestor_idx)
            pending.extend(
                parent_idx
                for parent_idx in state.parent_program_for_candidate[ancestor_idx]
                if parent_idx is not None
            )

    lineage_active_ids = tuple(sorted(lineage_active))
    cutoff_score: SelectionScore | None = None
    if len(lineage_active) > top_n:
        cutoff_score = sorted(
            (
                selection_scores[candidate_idx]
                for candidate_idx in lineage_active
            ),
            reverse=True,
        )[top_n - 1]
        lineage_active = {
            candidate_idx
            for candidate_idx in lineage_active
            if selection_scores[candidate_idx] >= cutoff_score
        }
    selection_active_ids = tuple(sorted(lineage_active))
    if not selection_active_ids:
        sampling_probabilities: dict[int, Fraction] = {}
    elif score_mode == "high_resolution_lexicographic":
        uniform_probability = Fraction(1, len(selection_active_ids))
        sampling_probabilities = {
            candidate_idx: uniform_probability
            for candidate_idx in selection_active_ids
        }
    else:
        total_rate = sum(
            rates[candidate_idx] for candidate_idx in selection_active_ids
        )
        sampling_probabilities = {
            candidate_idx: rates[candidate_idx] / total_rate
            for candidate_idx in selection_active_ids
        }
    cutoff = cutoff_score if isinstance(cutoff_score, Fraction) else None
    return ParentSelectionSnapshot(
        score_mode=score_mode,
        rates=rates,
        selection_scores=selection_scores,
        raw_rates=raw_rates,
        high_resolution_credits=high_resolution_credits,
        tie_resolutions=tie_resolutions,
        lineage_active=lineage_active_ids,
        selection_active=selection_active_ids,
        top_n_cutoff=cutoff,
        top_n_cutoff_score=cutoff_score,
        sampling_mode=sampling_mode,
        sampling_probabilities=sampling_probabilities,
    )


def selection_active_parent_rates(
    state: GEPAState,
    *,
    top_n: int,
    score_mode: SelectionScoreMode = "high_resolution",
) -> dict[int, Fraction]:
    """Return the final tie-inclusive proposal-parent sampling set."""

    if score_mode == "high_resolution_lexicographic":
        raise ValueError(
            "high_resolution_lexicographic has no scalar parent rate; "
            "use selection_active_parent_scores"
        )

    snapshot = parent_selection_snapshot(
        state,
        top_n=top_n,
        score_mode=score_mode,
    )
    return {
        candidate_idx: snapshot.rates[candidate_idx]
        for candidate_idx in snapshot.selection_active
    }


def selection_active_parent_scores(
    state: GEPAState,
    *,
    top_n: int,
    score_mode: SelectionScoreMode = "high_resolution_lexicographic",
) -> dict[int, SelectionScore]:
    """Return the final tie-inclusive scalar or lexicographic parent scores."""

    snapshot = parent_selection_snapshot(
        state,
        top_n=top_n,
        score_mode=score_mode,
    )
    return {
        candidate_idx: snapshot.selection_scores[candidate_idx]
        for candidate_idx in snapshot.selection_active
    }


class ReversibleMaskedExposureCorrectedCandidateSelector:
    """Apply reversible masks, then use the configured parent allocation."""

    def __init__(
        self,
        rng: random.Random,
        candidate_pool_observer: CandidatePoolObserver,
        logger: LoggerProtocol,
        *,
        top_n: int,
        score_mode: SelectionScoreMode = "high_resolution",
    ) -> None:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if score_mode not in (
            "raw_frontier_rate",
            "high_resolution",
            "high_resolution_lexicographic",
        ):
            raise ValueError(f"unsupported selection score mode: {score_mode!r}")
        self.rng = rng
        self.candidate_pool_observer = candidate_pool_observer
        self.logger = logger
        self.top_n = top_n
        self.score_mode = score_mode

    def select_candidate_idx(self, state: GEPAState) -> int:
        self.candidate_pool_observer.update_candidate_pool(state.program_candidates)
        snapshot = parent_selection_snapshot(
            state,
            top_n=self.top_n,
            score_mode=self.score_mode,
        )
        active_scores = {
            candidate_idx: snapshot.selection_scores[candidate_idx]
            for candidate_idx in snapshot.selection_active
        }
        if not active_scores:
            if state.full_program_trace:
                state.full_program_trace[-1]["parent_selection"] = (
                    _snapshot_record(snapshot, selected=0)
                )
            self.logger.log(
                "Reversible masked parent sampling: "
                f"score_mode={self.score_mode},"
                f"top_n={self.top_n},active=[],selected=0"
            )
            return 0
        eligible = list(active_scores)
        if self.score_mode == "high_resolution_lexicographic":
            selected = self.rng.choice(eligible)
        else:
            weights = [
                float(snapshot.rates[candidate_idx])
                for candidate_idx in eligible
            ]
            selected = self.rng.choices(eligible, weights=weights, k=1)[0]
        probabilities = tuple(
            float(snapshot.sampling_probabilities[candidate_idx])
            for candidate_idx in eligible
        )
        if state.full_program_trace:
            state.full_program_trace[-1]["parent_selection"] = _snapshot_record(
                snapshot,
                selected=selected,
            )
        entries = "; ".join(
            "candidate_idx="
            f"{candidate_idx},F={frontier_count(state, candidate_idx)},"
            f"E={evaluation_count(state, candidate_idx)},"
            f"F_over_E={float(snapshot.raw_rates[candidate_idx]):.17g},"
            + (
                "shared_credit="
                f"{float(snapshot.high_resolution_credits[candidate_idx]):.17g},"
                "high_resolution_rate="
                f"{float(snapshot.rates[candidate_idx]):.17g},"
                if snapshot.score_mode == "high_resolution"
                else ""
            )
            + (
                "shared_credit="
                f"{float(snapshot.high_resolution_credits[candidate_idx]):.17g},"
                "tie_resolution="
                f"{float(snapshot.tie_resolutions[candidate_idx]):.17g},"
                "lexicographic_key=("
                f"{float(snapshot.raw_rates[candidate_idx]):.17g},"
                f"{float(snapshot.tie_resolutions[candidate_idx]):.17g}),"
                if snapshot.score_mode == "high_resolution_lexicographic"
                else ""
            )
            + f"p={probability:.17g}"
            for candidate_idx, probability in zip(
                eligible,
                probabilities,
                strict=True,
            )
        )
        self.logger.log(
            "Reversible masked parent sampling: "
            f"score_mode={self.score_mode},"
            f"sampling_mode={snapshot.sampling_mode},"
            f"top_n={self.top_n},active={eligible}; {entries}; selected={selected}"
        )
        return selected


def _snapshot_record(
    snapshot: ParentSelectionSnapshot,
    *,
    selected: int,
) -> dict[str, object]:
    """Return a JSON-safe observational record without changing selection."""

    return {
        "score_mode": snapshot.score_mode,
        "sampling_mode": snapshot.sampling_mode,
        "sampling_probabilities": {
            candidate_idx: {
                "numerator": probability.numerator,
                "denominator": probability.denominator,
            }
            for candidate_idx, probability in (
                snapshot.sampling_probabilities.items()
            )
        },
        "rates": {
            candidate_idx: {
                "numerator": rate.numerator,
                "denominator": rate.denominator,
            }
            for candidate_idx, rate in snapshot.rates.items()
        },
        "selection_scores": {
            candidate_idx: _selection_score_record(score)
            for candidate_idx, score in snapshot.selection_scores.items()
        },
        "raw_rates": {
            candidate_idx: {
                "numerator": rate.numerator,
                "denominator": rate.denominator,
            }
            for candidate_idx, rate in snapshot.raw_rates.items()
        },
        "high_resolution_credits": {
            candidate_idx: {
                "numerator": credit.numerator,
                "denominator": credit.denominator,
            }
            for candidate_idx, credit in snapshot.high_resolution_credits.items()
        },
        "tie_resolutions": {
            candidate_idx: {
                "numerator": resolution.numerator,
                "denominator": resolution.denominator,
            }
            for candidate_idx, resolution in snapshot.tie_resolutions.items()
        },
        "lineage_active": list(snapshot.lineage_active),
        "selection_active": list(snapshot.selection_active),
        "top_n_cutoff": (
            {
                "numerator": snapshot.top_n_cutoff.numerator,
                "denominator": snapshot.top_n_cutoff.denominator,
            }
            if snapshot.top_n_cutoff is not None
            else None
        ),
        "top_n_cutoff_score": (
            _selection_score_record(snapshot.top_n_cutoff_score)
            if snapshot.top_n_cutoff_score is not None
            else None
        ),
        "selected": selected,
    }


def _selection_score_record(score: SelectionScore) -> dict[str, object]:
    if isinstance(score, LexicographicHighResolutionScore):
        return {
            "frontier_rate": {
                "numerator": score.frontier_rate.numerator,
                "denominator": score.frontier_rate.denominator,
            },
            "tie_resolution": {
                "numerator": score.tie_resolution.numerator,
                "denominator": score.tie_resolution.denominator,
            },
        }
    return {
        "scalar_rate": {
            "numerator": score.numerator,
            "denominator": score.denominator,
        }
    }
