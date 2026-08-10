from __future__ import annotations

from typing import Any, Generic, TypeVar

from bridge.b19_reversible_parent_selection import parent_selection_snapshot
from bridge.b21_joint_linucb_scheduler import (
    relaxed_repairable_gap,
    rectangular_max_weight_assignment,
)
from gepa.core.data_loader import ComparableHashable, DataLoader
from gepa.core.state import GEPAState
from gepa.proposer.reflective_mutation.base import CandidateSelector
from gepa.strategies.batch_sampler import BatchSampler
from gepa.strategies.proposal_sampling import ProposalTask


_SCHEMA_VERSION = 1
_SCHEDULER_LABEL = "repairable_gap"

_DataId = TypeVar("_DataId", bound=ComparableHashable)
_DataInst = TypeVar("_DataInst")


class RepairableGapSamplingStrategy(Generic[_DataId, _DataInst]):
    """Jointly match lex-active skills to minibatches using only relaxed H."""

    def __init__(
        self,
        *,
        top_n: int,
        minibatches_per_wave: int,
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

    def sample_tasks(
        self,
        state: GEPAState,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler[_DataId, _DataInst],
        trainset: DataLoader[_DataId, _DataInst],
    ) -> list[ProposalTask[_DataId, _DataInst]]:
        """Freeze one H-weighted wave and return its exact repeated matching."""

        # Parent eligibility is wholly owned by the b19 lexicographic selector.
        # The legacy scalar candidate selector must not become a second policy.
        del candidate_selector
        snapshot = parent_selection_snapshot(
            state,
            top_n=self.top_n,
            score_mode="high_resolution_lexicographic",
        )
        active = snapshot.selection_active
        if not active:
            return []

        minibatches: list[tuple[_DataId, ...]] = []
        for _ in range(self.minibatches_per_wave):
            minibatch_ids = tuple(
                batch_sampler.next_minibatch_ids(trainset, state)
            )
            if not minibatch_ids:
                raise ValueError("epoch sampler returned an empty minibatch")
            minibatches.append(minibatch_ids)

        gaps = {
            (program_idx, minibatch_position): relaxed_repairable_gap(
                state,
                active_program_indices=active,
                program_idx=program_idx,
                minibatch_ids=minibatch_ids,
            )
            for program_idx in active
            for minibatch_position, minibatch_ids in enumerate(minibatches)
        }

        remaining_positions = list(range(len(minibatches)))
        scheduled: list[tuple[int, int, int]] = []
        pass_index = 0
        while remaining_positions:
            pass_assignments = rectangular_max_weight_assignment(
                program_indices=active,
                minibatch_positions=remaining_positions,
                weights=gaps,
                total_minibatches=len(minibatches),
            )
            if not pass_assignments:
                raise RuntimeError(
                    "repairable-gap matching did not cover a remaining minibatch"
                )
            for assignment in pass_assignments:
                scheduled.append(
                    (
                        pass_index,
                        assignment.program_idx,
                        assignment.minibatch_position,
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
            parent_idx,
            minibatch_position,
        ) in enumerate(scheduled):
            minibatch_ids = minibatches[minibatch_position]
            metadata: dict[str, Any] = {
                "scheduler": _SCHEDULER_LABEL,
                "schema_version": _SCHEMA_VERSION,
                "wave_iteration": state.i + 1,
                "assignment_index": assignment_index,
                "pass_index": assignment_pass,
                "minibatch_position": minibatch_position,
                "parent_idx": parent_idx,
                "active_program_indices": tuple(active),
                "repairable_gap": gaps[(parent_idx, minibatch_position)],
            }
            tasks.append(
                ProposalTask(
                    parent_idx=parent_idx,
                    parent_candidate=state.program_candidates[parent_idx],
                    minibatch_ids=list(minibatch_ids),
                    minibatch=trainset.fetch(minibatch_ids),
                    sampling_metadata=metadata,
                )
            )
        return tasks
