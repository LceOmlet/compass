from __future__ import annotations

import math
import random
from collections.abc import Hashable, Iterable, Mapping
from numbers import Real
from typing import Generic, TypeVar

InstanceId = TypeVar("InstanceId", bound=Hashable)


class SparseSkillState(Generic[InstanceId]):
    """Sparse instance frontiers keyed by an exact, immutable candidate identity."""

    def __init__(self, *, seed_skill: Hashable) -> None:
        self._seed_skill = self._skill_identity(seed_skill)
        self._observed_scores: dict[Hashable, dict[InstanceId, float]] = {
            self._seed_skill: {}
        }
        self._evaluated_instances: dict[Hashable, set[InstanceId]] = {
            self._seed_skill: set()
        }
        self._fronts: dict[InstanceId, set[Hashable]] = {}

    @staticmethod
    def _skill_identity(value: Hashable) -> Hashable:
        if not isinstance(value, Hashable):
            raise TypeError("skill identity must be hashable")
        return value

    @staticmethod
    def _scores(
        scores_by_instance: Mapping[InstanceId, Real],
    ) -> dict[InstanceId, float]:
        if not isinstance(scores_by_instance, Mapping):
            raise TypeError("scores_by_instance must be a mapping")

        result: dict[InstanceId, float] = {}
        for instance_id, score in scores_by_instance.items():
            if isinstance(score, bool) or not isinstance(score, Real):
                raise TypeError("each observed score must be a finite real number")
            value = float(score)
            if not math.isfinite(value):
                raise ValueError("each observed score must be finite")
            result[instance_id] = value
        return result

    @property
    def seed_skill(self) -> Hashable:
        return self._seed_skill

    def is_known(self, skill: Hashable) -> bool:
        return self._skill_identity(skill) in self._observed_scores

    def evaluation_count(self, skill: Hashable) -> int:
        return len(self._evaluated_instances[self._skill_identity(skill)])

    def record_evaluations(
        self,
        *,
        skill: Hashable,
        instance_ids: Iterable[InstanceId],
    ) -> None:
        skill = self._skill_identity(skill)
        if skill not in self._evaluated_instances:
            raise KeyError(skill)
        self._evaluated_instances[skill].update(instance_ids)

    def frontier_count(self, skill: Hashable) -> int:
        skill = self._skill_identity(skill)
        if skill not in self._observed_scores:
            raise KeyError(skill)
        return sum(skill in front for front in self._fronts.values())

    def frontier_rate(self, skill: Hashable) -> float:
        evaluated = self.evaluation_count(skill)
        if evaluated == 0:
            return 0.0
        return self.frontier_count(skill) / evaluated

    def frontier_pool(self) -> tuple[Hashable, ...]:
        return tuple(
            skill
            for skill in self._observed_scores
            if self.frontier_count(skill) > 0
        )

    def front_mapping(self) -> dict[InstanceId, set[Hashable]]:
        return {
            instance_id: set(front)
            for instance_id, front in self._fronts.items()
        }

    def select_parent(self, *, rng: random.Random) -> Hashable:
        if not isinstance(rng, random.Random):
            raise TypeError("rng must be an explicit random.Random instance")

        eligible: list[Hashable] = []
        weights: list[float] = []
        for skill in self._observed_scores:
            evaluated = self.evaluation_count(skill)
            frontiers = self.frontier_count(skill)
            if evaluated > 0 and frontiers > 0:
                eligible.append(skill)
                weights.append(frontiers / evaluated)

        if not eligible:
            return self._seed_skill
        return rng.choices(eligible, weights=weights, k=1)[0]

    def _recompute_front(self, instance_id: InstanceId) -> None:
        observations = {
            skill: scores[instance_id]
            for skill, scores in self._observed_scores.items()
            if instance_id in scores
        }
        if not observations:
            self._fronts.pop(instance_id, None)
            return
        best = max(observations.values())
        self._fronts[instance_id] = {
            skill for skill, score in observations.items() if score == best
        }

    def _commit_known(
        self,
        skill: Hashable,
        scores_by_instance: Mapping[InstanceId, float],
    ) -> None:
        observed = self._observed_scores[skill]
        for instance_id, score in scores_by_instance.items():
            observed[instance_id] = max(
                observed.get(instance_id, float("-inf")),
                score,
            )
        self._evaluated_instances[skill].update(scores_by_instance)
        for instance_id in scores_by_instance:
            self._recompute_front(instance_id)

    def _candidate_owns_front(
        self,
        *,
        instance_id: InstanceId,
        score: float,
    ) -> bool:
        previous = [
            observed[instance_id]
            for observed in self._observed_scores.values()
            if instance_id in observed
        ]
        return not previous or score >= max(previous)

    def commit_existing_observations(
        self,
        *,
        skill: Hashable,
        scores_by_instance: Mapping[InstanceId, Real],
    ) -> None:
        skill = self._skill_identity(skill)
        if skill not in self._observed_scores:
            raise KeyError(skill)
        self._commit_known(skill, self._scores(scores_by_instance))

    def candidate_front_count(
        self,
        *,
        skill: Hashable,
        scores_by_instance: Mapping[InstanceId, Real],
    ) -> int:
        skill = self._skill_identity(skill)
        if skill in self._observed_scores:
            raise ValueError("an existing skill is not a new candidate")
        scores = self._scores(scores_by_instance)
        return sum(
            self._candidate_owns_front(instance_id=instance_id, score=score)
            for instance_id, score in scores.items()
        )

    def commit_candidate(
        self,
        *,
        skill: Hashable,
        scores_by_instance: Mapping[InstanceId, Real],
    ) -> bool:
        skill = self._skill_identity(skill)
        if skill in self._observed_scores:
            raise ValueError("an existing skill must use commit_existing_observations")
        scores = self._scores(scores_by_instance)
        if not self.candidate_front_count(
            skill=skill,
            scores_by_instance=scores,
        ):
            return False

        self._observed_scores[skill] = dict(scores)
        self._evaluated_instances[skill] = set(scores)
        for instance_id in scores:
            self._recompute_front(instance_id)
        return True
