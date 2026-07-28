from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Protocol

from gepa.core.state import GEPAState
from gepa.logging.logger import LoggerProtocol


class CandidatePoolObserver(Protocol):
    def update_candidate_pool(
        self,
        candidates: Sequence[Mapping[str, str]],
    ) -> None: ...


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


def selection_active_parent_rates(
    state: GEPAState,
    *,
    top_n: int,
) -> dict[int, Fraction]:
    """Derive reversible ancestor and tie-inclusive global top-N masks."""

    rates = {
        candidate_idx: Fraction(frontiers, exposure)
        for candidate_idx in range(len(state.program_candidates))
        if (exposure := evaluation_count(state, candidate_idx)) > 0
        and (frontiers := frontier_count(state, candidate_idx)) > 0
    }
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
            if (
                ancestor_idx in rates
                and descendant_rate >= rates[ancestor_idx]
            ):
                lineage_active.discard(ancestor_idx)
            pending.extend(
                parent_idx
                for parent_idx in state.parent_program_for_candidate[ancestor_idx]
                if parent_idx is not None
            )

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
    return {
        candidate_idx: rates[candidate_idx]
        for candidate_idx in sorted(lineage_active)
    }


class ReversibleMaskedExposureCorrectedCandidateSelector:
    """Apply reversible parent masks, then sample proportional to ``F_k/E_k``."""

    def __init__(
        self,
        rng: random.Random,
        candidate_pool_observer: CandidatePoolObserver,
        logger: LoggerProtocol,
        *,
        top_n: int,
    ) -> None:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        self.rng = rng
        self.candidate_pool_observer = candidate_pool_observer
        self.logger = logger
        self.top_n = top_n

    def select_candidate_idx(self, state: GEPAState) -> int:
        self.candidate_pool_observer.update_candidate_pool(state.program_candidates)
        active_rates = selection_active_parent_rates(
            state,
            top_n=self.top_n,
        )
        if not active_rates:
            self.logger.log(
                "Reversible masked parent sampling: active=[], selected=0"
            )
            return 0
        eligible = list(active_rates)
        weights = [float(active_rates[candidate_idx]) for candidate_idx in eligible]
        total = math.fsum(weights)
        probabilities = tuple(weight / total for weight in weights)
        selected = self.rng.choices(eligible, weights=weights, k=1)[0]
        entries = "; ".join(
            "candidate_idx="
            f"{candidate_idx},F={frontier_count(state, candidate_idx)},"
            f"E={evaluation_count(state, candidate_idx)},"
            f"F_over_E={weight:.17g},p={probability:.17g}"
            for candidate_idx, weight, probability in zip(
                eligible,
                weights,
                probabilities,
                strict=True,
            )
        )
        self.logger.log(
            "Reversible masked parent sampling: "
            f"top_n={self.top_n},active={eligible}; {entries}; selected={selected}"
        )
        return selected
