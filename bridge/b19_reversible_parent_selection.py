from __future__ import annotations

import math
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
]


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


def candidate_selection_rate(
    state: GEPAState,
    candidate_idx: int,
    *,
    score_mode: SelectionScoreMode,
) -> Fraction:
    """Return the configured exact selection score on unchanged clean evidence."""

    exposure = evaluation_count(state, candidate_idx)
    if exposure == 0:
        return Fraction(0)
    if score_mode == "raw_frontier_rate":
        return Fraction(frontier_count(state, candidate_idx), exposure)
    if score_mode == "high_resolution":
        return high_resolution_selection_rate(state, candidate_idx)
    raise ValueError(f"unsupported selection score mode: {score_mode!r}")


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
            candidate_selection_rate(
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


@dataclass(frozen=True, slots=True)
class ParentSelectionSnapshot:
    """Reporting view of the two recomputed reversible masks."""

    score_mode: SelectionScoreMode
    rates: Mapping[int, Fraction]
    raw_rates: Mapping[int, Fraction]
    high_resolution_credits: Mapping[int, Fraction]
    lineage_active: tuple[int, ...]
    selection_active: tuple[int, ...]
    top_n_cutoff: Fraction | None


def parent_selection_snapshot(
    state: GEPAState,
    *,
    top_n: int,
    score_mode: SelectionScoreMode = "high_resolution",
) -> ParentSelectionSnapshot:
    """Derive reversible ancestor and tie-inclusive global top-N masks."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if score_mode not in ("raw_frontier_rate", "high_resolution"):
        raise ValueError(f"unsupported selection score mode: {score_mode!r}")
    raw_rates = {
        candidate_idx: Fraction(frontiers, exposure)
        for candidate_idx in range(len(state.program_candidates))
        if (exposure := evaluation_count(state, candidate_idx)) > 0
        and (frontiers := frontier_count(state, candidate_idx)) > 0
    }
    high_resolution_credits = (
        high_resolution_frontier_credits(state)
        if score_mode == "high_resolution"
        else {}
    )
    rates = (
        {
            candidate_idx: high_resolution_credits[candidate_idx]
            / evaluation_count(state, candidate_idx)
            for candidate_idx in raw_rates
        }
        if score_mode == "high_resolution"
        else raw_rates
    )
    lineage_active = set(rates)
    for descendant_idx, descendant_rate in rates.items():
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
            if ancestor_idx in rates:
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
                else:
                    masks_ancestor = descendant_rate >= rates[ancestor_idx]
                if masks_ancestor:
                    lineage_active.discard(ancestor_idx)
            pending.extend(
                parent_idx
                for parent_idx in state.parent_program_for_candidate[ancestor_idx]
                if parent_idx is not None
            )

    lineage_active_ids = tuple(sorted(lineage_active))
    cutoff: Fraction | None = None
    if len(lineage_active) > top_n:
        cutoff = sorted(
            (rates[candidate_idx] for candidate_idx in lineage_active),
            reverse=True,
        )[top_n - 1]
        lineage_active = {
            candidate_idx
            for candidate_idx in lineage_active
            if rates[candidate_idx] >= cutoff
        }
    return ParentSelectionSnapshot(
        score_mode=score_mode,
        rates=rates,
        raw_rates=raw_rates,
        high_resolution_credits=high_resolution_credits,
        lineage_active=lineage_active_ids,
        selection_active=tuple(sorted(lineage_active)),
        top_n_cutoff=cutoff,
    )


def selection_active_parent_rates(
    state: GEPAState,
    *,
    top_n: int,
    score_mode: SelectionScoreMode = "high_resolution",
) -> dict[int, Fraction]:
    """Return the final tie-inclusive proposal-parent sampling set."""

    snapshot = parent_selection_snapshot(
        state,
        top_n=top_n,
        score_mode=score_mode,
    )
    return {
        candidate_idx: snapshot.rates[candidate_idx]
        for candidate_idx in snapshot.selection_active
    }


class ReversibleMaskedExposureCorrectedCandidateSelector:
    """Apply reversible masks, then sample by the configured exposure rate."""

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
        if score_mode not in ("raw_frontier_rate", "high_resolution"):
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
        active_rates = {
            candidate_idx: snapshot.rates[candidate_idx]
            for candidate_idx in snapshot.selection_active
        }
        if not active_rates:
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
        eligible = list(active_rates)
        weights = [float(active_rates[candidate_idx]) for candidate_idx in eligible]
        total = math.fsum(weights)
        probabilities = tuple(weight / total for weight in weights)
        selected = self.rng.choices(eligible, weights=weights, k=1)[0]
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
                f"high_resolution_rate={weight:.17g},"
                if snapshot.score_mode == "high_resolution"
                else ""
            )
            + f"p={probability:.17g}"
            for candidate_idx, weight, probability in zip(
                eligible,
                weights,
                probabilities,
                strict=True,
            )
        )
        self.logger.log(
            "Reversible masked parent sampling: "
            f"score_mode={self.score_mode},"
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
        "rates": {
            candidate_idx: {
                "numerator": rate.numerator,
                "denominator": rate.denominator,
            }
            for candidate_idx, rate in snapshot.rates.items()
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
        "selected": selected,
    }
