from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Generic, TypeVar

from bridge.b19_reversible_parent_selection import parent_selection_snapshot
from gepa.core.data_loader import ComparableHashable, DataId, DataLoader
from gepa.core.state import GEPAState
from gepa.proposer.reflective_mutation.base import CandidateSelector
from gepa.strategies.batch_sampler import BatchSampler
from gepa.strategies.proposal_sampling import (
    CompletedAdmissionOutcome,
    ProposalTask,
)


_DIMENSION = 4
_SCHEMA_VERSION = 1
JOINT_LINUCB_STATE_KEY = "b21_joint_linucb"
_SCHEDULER_LABEL = "joint_linucb"

_DataId = TypeVar("_DataId", bound=ComparableHashable)
_DataInst = TypeVar("_DataInst")


@dataclass(frozen=True, slots=True)
class ObservedArmContext:
    """The frozen ``(1, H, N, U)`` context for one proposal arm."""

    bias: float
    repairable_gap: float
    residual_gap: float
    observation_incompleteness: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.bias,
            self.repairable_gap,
            self.residual_gap,
            self.observation_incompleteness,
        )


@dataclass(frozen=True, slots=True)
class JointAssignment:
    """One program/minibatch edge selected by a rectangular matching pass."""

    program_idx: int
    minibatch_position: int


@dataclass(frozen=True, slots=True)
class _BanditCheckpoint:
    covariance: tuple[tuple[float, ...], ...]
    response: tuple[float, ...]
    completed_arm_counts: Mapping[int, int]


@dataclass(frozen=True, slots=True)
class _LexCost:
    """An additive lexicographically ordered cost used by Hungarian."""

    components: tuple[Fraction | int, ...]

    def __add__(self, other: _LexCost) -> _LexCost:
        self._require_same_dimension(other)
        return _LexCost(
            tuple(
                left + right
                for left, right in zip(
                    self.components,
                    other.components,
                    strict=True,
                )
            )
        )

    def __sub__(self, other: _LexCost) -> _LexCost:
        self._require_same_dimension(other)
        return _LexCost(
            tuple(
                left - right
                for left, right in zip(
                    self.components,
                    other.components,
                    strict=True,
                )
            )
        )

    def __lt__(self, other: _LexCost) -> bool:
        self._require_same_dimension(other)
        return self.components < other.components

    def _require_same_dimension(self, other: _LexCost) -> None:
        if len(self.components) != len(other.components):
            raise ValueError("lexicographic costs have different dimensions")


def relaxed_repairable_gap(
    state: GEPAState,
    *,
    active_program_indices: Sequence[int],
    program_idx: int,
    minibatch_ids: Sequence[DataId],
) -> float:
    """Return the canonical relaxed observed repairable gap ``H(s, b)``.

    Missing program-instance observations are unknown, not zero.  They add a
    zero term to the full-minibatch average instead of an imputed score.
    """

    _active, observations = _observed_scores_for_minibatch(
        state,
        active_program_indices=active_program_indices,
        program_idx=program_idx,
        minibatch_ids=minibatch_ids,
    )
    return _repairable_gap_from_observations(
        program_idx=program_idx,
        observations=observations,
    )


def observed_arm_context(
    state: GEPAState,
    *,
    active_program_indices: Sequence[int],
    program_idx: int,
    minibatch_ids: Sequence[DataId],
    perfect_score: float,
) -> ObservedArmContext:
    """Derive relaxed observed ``H``, ``N``, and ``U`` from owner state.

    Missing program-instance observations are unknown, not zero. They contribute
    only to ``U``. ``H`` and ``N`` average their observed positive gaps over the
    full minibatch, so unknown positions add no imputed reward or deficit.
    """

    active, observations = _observed_scores_for_minibatch(
        state,
        active_program_indices=active_program_indices,
        program_idx=program_idx,
        minibatch_ids=minibatch_ids,
    )
    target = _finite_float(perfect_score, label="perfect_score")

    residual_terms: list[float] = []
    incomplete_positions = 0
    for observed in observations:
        if len(observed) != len(active):
            incomplete_positions += 1
        if not observed:
            residual_terms.append(0.0)
            continue

        best_observed = max(observed.values())
        residual_terms.append(max(0.0, target - best_observed))

    denominator = len(minibatch_ids)
    return ObservedArmContext(
        bias=1.0,
        repairable_gap=_repairable_gap_from_observations(
            program_idx=program_idx,
            observations=observations,
        ),
        residual_gap=math.fsum(residual_terms) / denominator,
        observation_incompleteness=incomplete_positions / denominator,
    )


def _observed_scores_for_minibatch(
    state: GEPAState,
    *,
    active_program_indices: Sequence[int],
    program_idx: int,
    minibatch_ids: Sequence[DataId],
) -> tuple[tuple[int, ...], tuple[dict[int, float], ...]]:
    active = tuple(dict.fromkeys(int(idx) for idx in active_program_indices))
    if not active:
        raise ValueError("active_program_indices must not be empty")
    if program_idx not in active:
        raise ValueError("program_idx must belong to active_program_indices")
    if not minibatch_ids:
        raise ValueError("minibatch_ids must not be empty")

    observations: list[dict[int, float]] = []
    for instance_id in minibatch_ids:
        observed: dict[int, float] = {}
        for active_idx in active:
            raw_score = state.prog_candidate_val_subscores[active_idx].get(
                instance_id
            )
            if raw_score is None:
                continue
            score = float(raw_score)
            if math.isfinite(score):
                observed[active_idx] = score
        observations.append(observed)
    return active, tuple(observations)


def _repairable_gap_from_observations(
    *,
    program_idx: int,
    observations: Sequence[Mapping[int, float]],
) -> float:
    repairable_terms: list[float] = []
    for observed in observations:
        if not observed:
            repairable_terms.append(0.0)
            continue
        best_observed = max(observed.values())
        program_score = observed.get(program_idx)
        repairable_terms.append(
            max(0.0, best_observed - program_score)
            if program_score is not None
            else 0.0
        )
    return math.fsum(repairable_terms) / len(observations)


def linucb_score(
    context: Sequence[float],
    *,
    covariance: Sequence[Sequence[float]],
    response: Sequence[float],
) -> float:
    """Return ``x^T V^-1 q + sqrt(x^T V^-1 x)`` with ``beta=1``."""

    x = _finite_vector(context, label="context")
    if len(x) != _DIMENSION:
        raise ValueError(f"context must have dimension {_DIMENSION}")
    matrix = _finite_matrix(covariance, label="covariance")
    if len(matrix) != _DIMENSION or any(
        len(row) != _DIMENSION for row in matrix
    ):
        raise ValueError(f"covariance must be {_DIMENSION}x{_DIMENSION}")
    q = _finite_vector(response, label="response")
    if len(q) != _DIMENSION:
        raise ValueError(f"response must have dimension {_DIMENSION}")

    mean_weights = _solve_spd(matrix, q)
    uncertainty_weights = _solve_spd(matrix, x)
    mean = math.fsum(
        left * right for left, right in zip(x, mean_weights, strict=True)
    )
    uncertainty_squared = math.fsum(
        left * right
        for left, right in zip(x, uncertainty_weights, strict=True)
    )
    if uncertainty_squared < 0.0:
        raise ValueError("covariance produced negative LinUCB uncertainty")
    score = mean + math.sqrt(uncertainty_squared)
    if not math.isfinite(score):
        raise ValueError("LinUCB score must be finite")
    return score


def rectangular_hungarian_assignment(
    *,
    program_indices: Sequence[int],
    minibatch_positions: Sequence[int],
    scores: Mapping[tuple[int, int], float],
    completed_arm_counts: Mapping[int, int],
    total_minibatches: int,
) -> tuple[JointAssignment, ...]:
    """Solve one exact multi-objective rectangular Hungarian pass.

    The additive objective is ordered as follows, without scalarization:

    1. maximize total UCB score;
    2. minimize the sum of selected programs' completed-arm counts;
    3. minimize the assignment sequence sorted by minibatch position and then
       program index.

    The last objective is encoded as an additive indicator/program vector, so
    the Hungarian implementation never depends on solver return order.
    """

    programs = tuple(sorted(dict.fromkeys(int(idx) for idx in program_indices)))
    counts: dict[int, int] = {}
    for program_idx in programs:
        count = completed_arm_counts.get(program_idx, 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("completed arm counts must be non-negative integers")
        counts[program_idx] = count
    return _rectangular_exact_assignment(
        program_indices=programs,
        minibatch_positions=minibatch_positions,
        scores=scores,
        total_minibatches=total_minibatches,
        secondary_program_values={
            program_idx: -count for program_idx, count in counts.items()
        },
    )


def rectangular_max_weight_assignment(
    *,
    program_indices: Sequence[int],
    minibatch_positions: Sequence[int],
    weights: Mapping[tuple[int, int], float],
    total_minibatches: int,
) -> tuple[JointAssignment, ...]:
    """Maximize only total edge weight, then stably break exact ties.

    No capacity, exposure count, exploration bonus, or other policy term is
    present.  The stable tie objective prefers earlier minibatch positions and
    then lower program indices without scalarizing it into the edge weights.
    """

    return _rectangular_exact_assignment(
        program_indices=program_indices,
        minibatch_positions=minibatch_positions,
        scores=weights,
        total_minibatches=total_minibatches,
        secondary_program_values=None,
    )


def _rectangular_exact_assignment(
    *,
    program_indices: Sequence[int],
    minibatch_positions: Sequence[int],
    scores: Mapping[tuple[int, int], float],
    total_minibatches: int,
    secondary_program_values: Mapping[int, int] | None,
) -> tuple[JointAssignment, ...]:
    programs = tuple(sorted(dict.fromkeys(int(idx) for idx in program_indices)))
    positions = tuple(
        sorted(dict.fromkeys(int(position) for position in minibatch_positions))
    )
    if not programs or not positions:
        return ()
    if total_minibatches <= 0:
        raise ValueError("total_minibatches must be positive")
    if any(position < 0 or position >= total_minibatches for position in positions):
        raise ValueError("minibatch position is outside the wave")

    vector_dimension = (
        1
        + (1 if secondary_program_values is not None else 0)
        + 2 * total_minibatches
    )

    def edge_cost(program_idx: int, position: int) -> _LexCost:
        score = _finite_float(
            scores[(program_idx, position)],
            label="matching score",
        )
        assignment_key = [0] * (2 * total_minibatches)
        assignment_key[2 * position] = 1
        assignment_key[2 * position + 1] = -program_idx
        # Preserve the exact binary value of each finite float throughout all
        # Hungarian potential arithmetic. This makes a score tie an exact tie,
        # rather than a by-product of an intermediate summation order.
        secondary = (
            (secondary_program_values[program_idx],)
            if secondary_program_values is not None
            else ()
        )
        maximization_value: tuple[Fraction | int, ...] = (
            Fraction.from_float(score),
            *secondary,
            *assignment_key,
        )
        assert len(maximization_value) == vector_dimension
        return _LexCost(tuple(-component for component in maximization_value))

    rows_are_minibatches = len(positions) <= len(programs)
    if rows_are_minibatches:
        costs = [
            [edge_cost(program_idx, position) for program_idx in programs]
            for position in positions
        ]
    else:
        costs = [
            [edge_cost(program_idx, position) for position in positions]
            for program_idx in programs
        ]

    row_to_column = _hungarian_minimize(costs)
    selected: list[JointAssignment] = []
    for row_idx, column_idx in enumerate(row_to_column):
        if rows_are_minibatches:
            position = positions[row_idx]
            program_idx = programs[column_idx]
        else:
            program_idx = programs[row_idx]
            position = positions[column_idx]
        selected.append(
            JointAssignment(
                program_idx=program_idx,
                minibatch_position=position,
            )
        )
    return tuple(
        sorted(
            selected,
            key=lambda item: (item.minibatch_position, item.program_idx),
        )
    )


class JointLinUCBSamplingStrategy(Generic[_DataId, _DataInst]):
    """Deterministic whole-wave joint LinUCB proposal scheduling."""

    def __init__(
        self,
        *,
        top_n: int,
        minibatches_per_wave: int,
        perfect_score: float,
    ) -> None:
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
            raise TypeError("top_n must be a positive integer")
        if (
            isinstance(minibatches_per_wave, bool)
            or not isinstance(minibatches_per_wave, int)
            or minibatches_per_wave <= 0
        ):
            raise TypeError("minibatches_per_wave must be a positive integer")
        self.top_n = top_n
        self.minibatches_per_wave = minibatches_per_wave
        self.perfect_score = _finite_float(
            perfect_score,
            label="perfect_score",
        )

    def sample_tasks(
        self,
        state: GEPAState,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler[_DataId, _DataInst],
        trainset: DataLoader[_DataId, _DataInst],
    ) -> list[ProposalTask[_DataId, _DataInst]]:
        """Freeze one wave, solve repeated matchings, and return owner tasks."""

        # Parent eligibility is owned by b19. The legacy candidate selector is
        # intentionally not sampled because a joint assignment is the one and
        # only allocation policy for this explicit strategy.
        del candidate_selector
        snapshot = parent_selection_snapshot(
            state,
            top_n=self.top_n,
            score_mode="high_resolution_lexicographic",
        )
        active = snapshot.selection_active
        if not active:
            return []

        checkpoint = _decode_checkpoint(state.sampling_strategy_state)
        minibatches: list[tuple[_DataId, ...]] = []
        for _ in range(self.minibatches_per_wave):
            minibatch_ids = tuple(
                batch_sampler.next_minibatch_ids(trainset, state)
            )
            if not minibatch_ids:
                raise ValueError("epoch sampler returned an empty minibatch")
            minibatches.append(minibatch_ids)

        contexts: dict[tuple[int, int], ObservedArmContext] = {}
        scores: dict[tuple[int, int], float] = {}
        for program_idx in active:
            for position, minibatch_ids in enumerate(minibatches):
                context = observed_arm_context(
                    state,
                    active_program_indices=active,
                    program_idx=program_idx,
                    minibatch_ids=minibatch_ids,
                    perfect_score=self.perfect_score,
                )
                contexts[(program_idx, position)] = context
                scores[(program_idx, position)] = linucb_score(
                    context.as_tuple(),
                    covariance=checkpoint.covariance,
                    response=checkpoint.response,
                )

        remaining_positions = list(range(len(minibatches)))
        scheduled: list[
            tuple[int, JointAssignment, ObservedArmContext, float]
        ] = []
        pass_index = 0
        while remaining_positions:
            pass_assignments = rectangular_hungarian_assignment(
                program_indices=active,
                minibatch_positions=remaining_positions,
                scores=scores,
                completed_arm_counts=checkpoint.completed_arm_counts,
                total_minibatches=len(minibatches),
            )
            if not pass_assignments:
                raise RuntimeError("joint matching did not cover a remaining minibatch")
            for assignment in pass_assignments:
                scheduled.append(
                    (
                        pass_index,
                        assignment,
                        contexts[
                            (
                                assignment.program_idx,
                                assignment.minibatch_position,
                            )
                        ],
                        scores[
                            (
                                assignment.program_idx,
                                assignment.minibatch_position,
                            )
                        ],
                    )
                )
            matched_positions = {
                assignment.minibatch_position
                for assignment in pass_assignments
            }
            remaining_positions = [
                position
                for position in remaining_positions
                if position not in matched_positions
            ]
            pass_index += 1

        tasks: list[ProposalTask[_DataId, _DataInst]] = []
        for assignment_index, (
            assignment_pass,
            assignment,
            context,
            score,
        ) in enumerate(scheduled):
            minibatch_ids = minibatches[assignment.minibatch_position]
            metadata: dict[str, Any] = {
                "scheduler": _SCHEDULER_LABEL,
                "schema_version": _SCHEMA_VERSION,
                "wave_iteration": state.i + 1,
                "assignment_index": assignment_index,
                "pass_index": assignment_pass,
                "minibatch_position": assignment.minibatch_position,
                "parent_idx": assignment.program_idx,
                "active_program_indices": tuple(active),
                "context": context.as_tuple(),
                "ucb_score": score,
                "completed_arm_count_at_wave_start": (
                    checkpoint.completed_arm_counts.get(
                        assignment.program_idx,
                        0,
                    )
                ),
            }
            tasks.append(
                ProposalTask(
                    parent_idx=assignment.program_idx,
                    parent_candidate=state.program_candidates[
                        assignment.program_idx
                    ],
                    minibatch_ids=list(minibatch_ids),
                    minibatch=trainset.fetch(minibatch_ids),
                    sampling_metadata=metadata,
                )
            )
        return tasks

    def update_after_completed_admissions(
        self,
        *,
        current_state: Mapping[str, Any],
        outcomes: tuple[CompletedAdmissionOutcome[_DataId], ...],
    ) -> Mapping[str, Any]:
        """Atomically aggregate complete official admission outcomes.

        Each completed arm contributes one signed arithmetic-mean admission
        delta, one outer product, and one completed-arm count. Missing outcomes
        are absent from ``outcomes`` and therefore contribute nothing.
        """

        if not outcomes:
            return dict(current_state)
        checkpoint = _decode_checkpoint(current_state)
        ordered = tuple(
            sorted(
                outcomes,
                key=lambda outcome: (outcome.task_index, outcome.parent_idx),
            )
        )
        task_indices = [outcome.task_index for outcome in ordered]
        if len(task_indices) != len(set(task_indices)):
            raise ValueError("completed admission outcomes repeat a task index")

        contexts: list[tuple[float, ...]] = []
        rewards: list[float] = []
        count_increments: dict[int, int] = {}
        wave_iteration: int | None = None
        for outcome in ordered:
            metadata = outcome.sampling_metadata
            if metadata.get("scheduler") != _SCHEDULER_LABEL:
                raise ValueError("completed outcome belongs to another scheduler")
            if metadata.get("schema_version") != _SCHEMA_VERSION:
                raise ValueError("completed outcome has another scheduler schema")
            if metadata.get("assignment_index") != outcome.task_index:
                raise ValueError("completed outcome task identity changed")
            if metadata.get("parent_idx") != outcome.parent_idx:
                raise ValueError("completed outcome parent identity changed")
            metadata_iteration = metadata.get("wave_iteration")
            if metadata_iteration != outcome.iteration:
                raise ValueError("completed outcome wave identity changed")
            if wave_iteration is None:
                wave_iteration = outcome.iteration
            elif wave_iteration != outcome.iteration:
                raise ValueError("completed outcomes span more than one wave")

            context = _finite_vector(
                metadata.get("context", ()),
                label="completed arm context",
            )
            if len(context) != _DIMENSION:
                raise ValueError(
                    f"completed arm context must have dimension {_DIMENSION}"
                )
            before = _finite_vector(
                outcome.scores_before,
                label="admission scores_before",
            )
            after = _finite_vector(
                outcome.scores_after,
                label="admission scores_after",
            )
            if (
                not before
                or len(before) != len(after)
                or len(before) != len(outcome.admission_ids)
            ):
                raise ValueError("completed admission outcome is not aligned")
            reward = math.fsum(after) / len(after) - math.fsum(before) / len(
                before
            )
            contexts.append(context)
            rewards.append(reward)
            count_increments[outcome.parent_idx] = (
                count_increments.get(outcome.parent_idx, 0) + 1
            )

        covariance = [list(row) for row in checkpoint.covariance]
        response = list(checkpoint.response)
        for row_idx in range(_DIMENSION):
            response[row_idx] += math.fsum(
                context[row_idx] * reward
                for context, reward in zip(contexts, rewards, strict=True)
            )
            for column_idx in range(_DIMENSION):
                covariance[row_idx][column_idx] += math.fsum(
                    context[row_idx] * context[column_idx]
                    for context in contexts
                )

        completed_counts = dict(checkpoint.completed_arm_counts)
        for program_idx, increment in count_increments.items():
            completed_counts[program_idx] = (
                completed_counts.get(program_idx, 0) + increment
            )

        replacement = dict(current_state)
        replacement[JOINT_LINUCB_STATE_KEY] = _encode_checkpoint(
            _BanditCheckpoint(
                covariance=tuple(tuple(row) for row in covariance),
                response=tuple(response),
                completed_arm_counts=completed_counts,
            )
        )
        return replacement


def _fresh_checkpoint() -> _BanditCheckpoint:
    return _BanditCheckpoint(
        covariance=tuple(
            tuple(1.0 if row_idx == column_idx else 0.0 for column_idx in range(_DIMENSION))
            for row_idx in range(_DIMENSION)
        ),
        response=(0.0,) * _DIMENSION,
        completed_arm_counts={},
    )


def _decode_checkpoint(root: Mapping[str, Any]) -> _BanditCheckpoint:
    if not isinstance(root, Mapping):
        raise TypeError("sampling strategy checkpoint must be a mapping")
    raw = root.get(JOINT_LINUCB_STATE_KEY)
    if raw is None:
        return _fresh_checkpoint()
    if not isinstance(raw, Mapping):
        raise TypeError("joint LinUCB checkpoint must be a mapping")
    if raw.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("joint LinUCB checkpoint schema changed")

    covariance = _finite_matrix(raw.get("covariance", ()), label="covariance")
    response = _finite_vector(raw.get("response", ()), label="response")
    if len(covariance) != _DIMENSION or any(
        len(row) != _DIMENSION for row in covariance
    ):
        raise ValueError(f"covariance must be {_DIMENSION}x{_DIMENSION}")
    if len(response) != _DIMENSION:
        raise ValueError(f"response must have dimension {_DIMENSION}")
    for row_idx in range(_DIMENSION):
        for column_idx in range(_DIMENSION):
            if covariance[row_idx][column_idx] != covariance[column_idx][row_idx]:
                raise ValueError("covariance must be symmetric")
    # The same solve used for scoring validates positive definiteness now,
    # rather than after an opaque checkpoint is accepted.
    _solve_spd(covariance, (0.0,) * _DIMENSION)

    raw_counts = raw.get("completed_arm_counts", {})
    if not isinstance(raw_counts, Mapping):
        raise TypeError("completed_arm_counts must be a mapping")
    completed_counts: dict[int, int] = {}
    for raw_program_idx, raw_count in raw_counts.items():
        if (
            isinstance(raw_program_idx, bool)
            or not isinstance(raw_program_idx, int)
            or raw_program_idx < 0
        ):
            raise ValueError("completed arm program indices must be non-negative integers")
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 0
        ):
            raise ValueError("completed arm counts must be non-negative integers")
        completed_counts[raw_program_idx] = raw_count
    return _BanditCheckpoint(
        covariance=covariance,
        response=response,
        completed_arm_counts=completed_counts,
    )


def _encode_checkpoint(checkpoint: _BanditCheckpoint) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "covariance": [list(row) for row in checkpoint.covariance],
        "response": list(checkpoint.response),
        "completed_arm_counts": dict(
            sorted(checkpoint.completed_arm_counts.items())
        ),
    }


def _hungarian_minimize(costs: Sequence[Sequence[_LexCost]]) -> tuple[int, ...]:
    """Assign every row to a distinct column using rectangular Hungarian."""

    if not costs or not costs[0]:
        return ()
    row_count = len(costs)
    column_count = len(costs[0])
    if row_count > column_count:
        raise ValueError("Hungarian requires rows <= columns")
    if any(len(row) != column_count for row in costs):
        raise ValueError("Hungarian costs must be rectangular")
    dimension = len(costs[0][0].components)
    if any(
        len(cost.components) != dimension
        for row in costs
        for cost in row
    ):
        raise ValueError("Hungarian costs have inconsistent dimensions")

    zero = _LexCost((0,) * dimension)
    row_potential = [zero for _ in range(row_count + 1)]
    column_potential = [zero for _ in range(column_count + 1)]
    column_to_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        column_to_row[0] = row
        current_column = 0
        minimum: list[_LexCost | None] = [None] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[current_column] = True
            current_row = column_to_row[current_column]
            delta: _LexCost | None = None
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = (
                    costs[current_row - 1][column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if minimum[column] is None or reduced < minimum[column]:
                    minimum[column] = reduced
                    predecessor[column] = current_column
                candidate = minimum[column]
                assert candidate is not None
                if delta is None or candidate < delta:
                    delta = candidate
                    next_column = column
            if delta is None:
                raise RuntimeError("Hungarian could not extend its alternating path")
            for column in range(column_count + 1):
                if used[column]:
                    assigned_row = column_to_row[column]
                    row_potential[assigned_row] = (
                        row_potential[assigned_row] + delta
                    )
                    column_potential[column] = (
                        column_potential[column] - delta
                    )
                elif column > 0 and minimum[column] is not None:
                    minimum[column] = minimum[column] - delta
            current_column = next_column
            if column_to_row[current_column] == 0:
                break

        while current_column != 0:
            previous_column = predecessor[current_column]
            column_to_row[current_column] = column_to_row[previous_column]
            current_column = previous_column

    row_to_column = [-1] * row_count
    for column in range(1, column_count + 1):
        assigned_row = column_to_row[column]
        if assigned_row != 0:
            row_to_column[assigned_row - 1] = column - 1
    if any(column < 0 for column in row_to_column):
        raise RuntimeError("Hungarian returned an incomplete assignment")
    return tuple(row_to_column)


def _solve_spd(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> tuple[float, ...]:
    """Solve a small symmetric positive-definite system by Cholesky."""

    dimension = len(matrix)
    if dimension == 0 or len(vector) != dimension:
        raise ValueError("SPD system has incompatible dimensions")
    if any(len(row) != dimension for row in matrix):
        raise ValueError("SPD matrix must be square")

    lower = [[0.0] * dimension for _ in range(dimension)]
    for row in range(dimension):
        for column in range(row + 1):
            residual = matrix[row][column] - math.fsum(
                lower[row][inner] * lower[column][inner]
                for inner in range(column)
            )
            if row == column:
                if residual <= 0.0 or not math.isfinite(residual):
                    raise ValueError("covariance must be positive definite")
                lower[row][column] = math.sqrt(residual)
            else:
                lower[row][column] = residual / lower[column][column]

    forward = [0.0] * dimension
    for row in range(dimension):
        forward[row] = (
            vector[row]
            - math.fsum(
                lower[row][column] * forward[column]
                for column in range(row)
            )
        ) / lower[row][row]

    result = [0.0] * dimension
    for row in range(dimension - 1, -1, -1):
        result[row] = (
            forward[row]
            - math.fsum(
                lower[column][row] * result[column]
                for column in range(row + 1, dimension)
            )
        ) / lower[row][row]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("SPD solve produced a non-finite result")
    return tuple(result)


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_vector(values: Any, *, label: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    return tuple(
        _finite_float(value, label=label)
        for value in values
    )


def _finite_matrix(values: Any, *, label: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of rows")
    return tuple(
        _finite_vector(row, label=label)
        for row in values
    )
