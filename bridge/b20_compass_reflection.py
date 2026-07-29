from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from pathlib import Path
from typing import Any, Literal, TypeVar

from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from gepa import optimize
from gepa.core.adapter import EvaluationBatch
from gepa.core.data_loader import DataLoader
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.logging.logger import Logger, LoggerProtocol
from gepa.proposer.reflective_mutation.admission import (
    AdmissionPlan,
    commit_adapter_observations,
)
from gepa.strategies.acceptance import StrictImprovementAcceptance
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import ParetoCandidateSelector
from gepa.strategies.proposal_sampling import (
    IndependentSampling,
    SingleMutationSampling,
)
from gepa.strategies.proposal_selection import AllImprovements

from bridge.b19_reversible_parent_selection import (
    ReversibleMaskedExposureCorrectedCandidateSelector,
    evaluation_count,
    frontier_count,
    frontier_rate,
)

DataId = Hashable
BatchItemT = TypeVar("BatchItemT")
ReflectionCondition = Literal[
    "mini_admission_reflection",
    "compass_reflection",
]

_LOGGER = logging.getLogger(__name__)


def _run_batch_item(
    operation: Callable[..., BatchItemT],
    *args: Any,
    stage: str,
) -> BatchItemT | None:
    """Return one batch item result, preserving an ordinary failure as an empty slot."""

    try:
        return operation(*args)
    except MemoryError:
        raise
    except Exception:
        _LOGGER.exception("%s item failed; this item will not be submitted", stage)
        return None


@dataclass(frozen=True, slots=True)
class SparseObservation:
    """One official DSPy execution bound to a program-instance maximum."""

    score: float
    output: Any
    trajectory: Mapping[str, Any]


class SparseObservationDspyAdapter(DspyAdapter):
    """Add sparse observation persistence and ordered whole-epoch concurrency."""

    _STATE_KEY = "compass_sparse_admission"
    _SCHEMA_VERSION = 1

    def __init__(
        self,
        *args: Any,
        max_candidate_workers: int = 1,
        max_reflection_workers: int = 1,
        **kwargs: Any,
    ):
        for name, value in (
            ("max_candidate_workers", max_candidate_workers),
            ("max_reflection_workers", max_reflection_workers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TypeError(f"{name} must be a positive integer")
        super().__init__(*args, **kwargs)
        self.max_candidate_workers = max_candidate_workers
        self.max_reflection_workers = max_reflection_workers
        self._observation_facts: dict[tuple[int, Hashable], SparseObservation] = {}

    def evaluate(
        self,
        batch: Sequence[Any],
        candidate: Mapping[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        if not batch:
            return EvaluationBatch(
                outputs=[],
                scores=[],
                trajectories=[] if capture_traces else None,
                objective_scores=None,
                num_metric_calls=0,
            )
        return super().evaluate(
            list(batch),
            dict(candidate),
            capture_traces=capture_traces,
        )

    def batch_evaluate(
        self,
        items: list[tuple[dict[str, str], list[Any]]],
    ) -> list[EvaluationBatch | None]:
        """Evaluate independent items, preserving failed positions without resubmission."""

        if not items:
            return []
        if len(items) == 1 or self.max_candidate_workers == 1:
            return [
                _run_batch_item(
                    self.evaluate,
                    batch,
                    candidate,
                    True,
                    stage="DSPy candidate evaluation",
                )
                for candidate, batch in items
            ]

        results: list[EvaluationBatch | None] = [None] * len(items)
        workers = min(self.max_candidate_workers, len(items))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_batch_item,
                    self.evaluate,
                    batch,
                    candidate,
                    True,
                    stage="DSPy candidate evaluation",
                ): item_index
                for item_index, (candidate, batch) in enumerate(items)
            }
            for future in as_completed(futures):
                item_index = futures[future]
                results[item_index] = future.result()
        return results

    def propose_new_texts_batch(
        self,
        jobs: list[
            tuple[
                dict[str, str],
                Mapping[str, Sequence[Mapping[str, Any]]],
                list[str],
            ]
        ],
    ) -> list[dict[str, str] | None]:
        """Call the official DSPy proposer once per job and preserve failed slots."""

        if not jobs:
            return []

        def propose(
            job: tuple[
                dict[str, str],
                Mapping[str, Sequence[Mapping[str, Any]]],
                list[str],
            ],
        ) -> dict[str, str]:
            candidate, reflective_dataset, components = job
            return dict(
                self.propose_new_texts(
                    candidate,
                    dict(reflective_dataset),
                    components,
                )
            )

        if len(jobs) == 1 or self.max_reflection_workers == 1:
            return [
                _run_batch_item(
                    propose,
                    job,
                    stage="DSPy reflection",
                )
                for job in jobs
            ]

        results: list[dict[str, str] | None] = [None] * len(jobs)
        workers = min(self.max_reflection_workers, len(jobs))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_batch_item,
                    propose,
                    job,
                    stage="DSPy reflection",
                ): job_index
                for job_index, job in enumerate(jobs)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results

    def commit_program_observations(
        self,
        *,
        program_idx: int,
        evaluation_ids: Sequence[Hashable],
        evaluation: EvaluationBatch,
        committed_ids: Sequence[Hashable],
    ) -> None:
        """Persist only facts whose scalar maximum was committed by GEPA."""

        ids = tuple(evaluation_ids)
        outputs = tuple(evaluation.outputs)
        scores = tuple(evaluation.scores)
        trajectories = tuple(evaluation.trajectories or ())
        if not (len(ids) == len(outputs) == len(scores) == len(trajectories)):
            raise RuntimeError("official observation vectors are not instance-aligned")
        committed = set(committed_ids)
        unknown = committed.difference(ids)
        if unknown:
            raise RuntimeError(
                "official state committed an ID outside the evaluated batch"
            )

        for instance_id, output, raw_score, trajectory in zip(
            ids,
            outputs,
            scores,
            trajectories,
            strict=True,
        ):
            if instance_id not in committed:
                continue
            score = _finite_score(raw_score)
            if not isinstance(trajectory, Mapping):
                raise RuntimeError("official DSPy trajectory is not a mapping")
            key = (int(program_idx), instance_id)
            previous = self._observation_facts.get(key)
            if previous is not None and score <= previous.score:
                continue
            self._observation_facts[key] = SparseObservation(
                score=score,
                output=output,
                trajectory=dict(trajectory),
            )

    def get_program_observation(
        self,
        program_idx: int,
        instance_id: Hashable,
    ) -> SparseObservation | None:
        return self._observation_facts.get((int(program_idx), instance_id))

    def get_adapter_state(self) -> dict[str, Any]:
        return {
            self._STATE_KEY: {
                "schema_version": self._SCHEMA_VERSION,
                "observation_facts": dict(self._observation_facts),
            }
        }

    def set_adapter_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("adapter state must be a mapping")
        unexpected = set(state).difference({self._STATE_KEY})
        if unexpected:
            raise RuntimeError(
                "unrecognized adapter state outside the COMPASS namespace: "
                f"{sorted(unexpected)}"
            )
        payload = state.get(self._STATE_KEY)
        if payload is None:
            self._observation_facts = {}
            return
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != self._SCHEMA_VERSION
        ):
            raise RuntimeError("unsupported COMPASS admission adapter-state schema")
        raw_facts = payload.get("observation_facts")
        if not isinstance(raw_facts, Mapping):
            raise RuntimeError("COMPASS admission facts are not a mapping")
        facts: dict[tuple[int, Hashable], SparseObservation] = {}
        for key, fact in raw_facts.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or isinstance(key[0], bool)
                or not isinstance(key[0], int)
                or not isinstance(fact, SparseObservation)
            ):
                raise RuntimeError("COMPASS admission fact entry is malformed")
            facts[(key[0], key[1])] = fact
        self._observation_facts = facts


def _finite_score(score: Real) -> float:
    if isinstance(score, bool) or not isinstance(score, Real):
        raise TypeError("GEPA score must be numeric")
    value = float(score)
    if not math.isfinite(value):
        raise ValueError("GEPA score must be finite")
    return value


def select_reference_program_idx(
    state: GEPAState,
    *,
    instance_id: DataId,
    sampled_parent_idx: int,
    rng: random.Random,
) -> int:
    """Select one history-fixed instance-frontier reference."""

    owners = tuple(sorted(state.program_at_pareto_front_valset.get(instance_id, ())))
    if not owners:
        return sampled_parent_idx

    rates = {
        candidate_idx: Fraction(
            frontier_count(state, candidate_idx),
            evaluation_count(state, candidate_idx),
        )
        for candidate_idx in owners
        if evaluation_count(state, candidate_idx) > 0
    }
    if not rates:
        raise RuntimeError(
            "an admission frontier owner has no evaluated instance exposure"
        )
    best_rate = max(rates.values())
    rate_tied = tuple(idx for idx in owners if rates.get(idx) == best_rate)
    best_exposure = max(evaluation_count(state, idx) for idx in rate_tied)
    exact_tied = tuple(
        idx for idx in rate_tied if evaluation_count(state, idx) == best_exposure
    )
    return (
        exact_tied[0] if len(exact_tied) == 1 else rng.choice(tuple(sorted(exact_tied)))
    )


class MiniAdmissionHook:
    """Prepare history-fixed references for an independent admit minibatch."""

    def __init__(
        self,
        *,
        adapter: SparseObservationDspyAdapter,
        rng: random.Random,
        run_dir: Path,
        logger: LoggerProtocol,
    ) -> None:
        self.adapter = adapter
        self.rng = rng
        self.run_dir = run_dir
        self.logger = logger

    @staticmethod
    def _validate_disjoint_instances(
        propose_evaluation: EvaluationBatch,
        admission_batch: Sequence[Any],
    ) -> None:
        trajectories = tuple(propose_evaluation.trajectories or ())
        propose_examples = tuple(
            trajectory.get("example")
            for trajectory in trajectories
            if isinstance(trajectory, Mapping) and "example" in trajectory
        )
        if len(propose_examples) != len(trajectories):
            raise RuntimeError(
                "official proposal trajectories do not identify every input instance"
            )
        if any(
            proposed is admitted
            for proposed in propose_examples
            for admitted in admission_batch
        ):
            raise RuntimeError("B_propose and B_admit overlap within one round")

    def _evaluate_missing_reference_groups(
        self,
        *,
        state: GEPAState,
        ids: tuple[DataId, ...],
        batch: tuple[Any, ...],
        reference_program_indices: tuple[int, ...],
    ) -> None:
        missing_by_program: dict[int, list[int]] = defaultdict(list)
        for position, (instance_id, program_idx) in enumerate(
            zip(ids, reference_program_indices, strict=True)
        ):
            known_score = state.prog_candidate_val_subscores[program_idx].get(
                instance_id
            )
            fact = self.adapter.get_program_observation(program_idx, instance_id)
            if fact is not None:
                if known_score is None or fact.score != known_score:
                    raise RuntimeError(
                        "persisted reference observation is not bound to the "
                        "official maximum reward"
                    )
                continue
            missing_by_program[program_idx].append(position)

        program_groups = tuple(sorted(missing_by_program))
        items = [
            (
                state.program_candidates[program_idx],
                [batch[position] for position in missing_by_program[program_idx]],
            )
            for program_idx in program_groups
        ]
        evaluations = self.adapter.batch_evaluate(items)
        if len(evaluations) != len(program_groups):
            raise RuntimeError("reference candidate batches are not aligned")

        for program_idx, evaluation in zip(
            program_groups,
            evaluations,
            strict=True,
        ):
            positions = missing_by_program[program_idx]
            group_ids = tuple(ids[position] for position in positions)
            if (
                len(evaluation.outputs) != len(group_ids)
                or len(evaluation.scores) != len(group_ids)
                or len(evaluation.trajectories or ()) != len(group_ids)
            ):
                raise RuntimeError(
                    "official missing-reference evaluation is not instance-aligned"
                )
            if evaluation.objective_scores:
                raise RuntimeError(
                    "sparse instance admission does not support objective scores"
                )
            metric_calls = (
                evaluation.num_metric_calls
                if evaluation.num_metric_calls is not None
                else len(group_ids)
            )
            state.increment_evals(metric_calls)

            if state.evaluation_cache is not None:
                state.evaluation_cache.put_batch(
                    state.program_candidates[program_idx],
                    group_ids,
                    evaluation.outputs,
                    evaluation.scores,
                    None,
                )

            observed = ValsetEvaluation(
                outputs_by_val_id=dict(zip(group_ids, evaluation.outputs, strict=True)),
                scores_by_val_id={
                    instance_id: _finite_score(score)
                    for instance_id, score in zip(
                        group_ids,
                        evaluation.scores,
                        strict=True,
                    )
                },
                objective_scores_by_val_id=None,
            )
            committed = state.commit_existing_program_evaluation(
                program_idx=program_idx,
                valset_evaluation=observed,
                run_dir=str(self.run_dir),
                iteration=state.i + 1,
            )
            bind_ids = list(committed)
            for instance_id, score in observed.scores_by_val_id.items():
                if (
                    self.adapter.get_program_observation(
                        program_idx,
                        instance_id,
                    )
                    is None
                    and state.prog_candidate_val_subscores[program_idx].get(instance_id)
                    == score
                ):
                    bind_ids.append(instance_id)
            commit_adapter_observations(
                self.adapter,
                program_idx=program_idx,
                evaluation_ids=group_ids,
                evaluation=evaluation,
                committed_ids=tuple(dict.fromkeys(bind_ids)),
            )
            for instance_id in group_ids:
                fact = self.adapter.get_program_observation(
                    program_idx,
                    instance_id,
                )
                current_score = state.prog_candidate_val_subscores[program_idx].get(
                    instance_id
                )
                if fact is None or fact.score != current_score:
                    raise RuntimeError(
                        "official reference execution did not produce a "
                        "maximum-bound observation fact"
                    )

    def prepare(
        self,
        *,
        state: GEPAState,
        parent_program_idx: int,
        parent_candidate: Mapping[str, str],
        components_to_update: Sequence[str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        propose_evaluation: EvaluationBatch,
        admission_ids: Sequence[DataId],
        admission_batch: Sequence[Any],
    ) -> AdmissionPlan | None:
        del parent_candidate, components_to_update, reflective_dataset
        ids = tuple(admission_ids)
        batch = tuple(admission_batch)
        if len(ids) != len(batch):
            raise RuntimeError("official admit IDs and instances are misaligned")
        self._validate_disjoint_instances(propose_evaluation, batch)

        reference_program_indices = tuple(
            select_reference_program_idx(
                state,
                instance_id=instance_id,
                sampled_parent_idx=parent_program_idx,
                rng=self.rng,
            )
            for instance_id in ids
        )
        self._evaluate_missing_reference_groups(
            state=state,
            ids=ids,
            batch=batch,
            reference_program_indices=reference_program_indices,
        )

        facts = tuple(
            self.adapter.get_program_observation(program_idx, instance_id)
            for program_idx, instance_id in zip(
                reference_program_indices,
                ids,
                strict=True,
            )
        )
        if any(fact is None for fact in facts):
            raise RuntimeError("admission reference fact is unavailable")
        concrete_facts = tuple(fact for fact in facts if fact is not None)
        eval_before = EvaluationBatch(
            outputs=[fact.output for fact in concrete_facts],
            scores=[fact.score for fact in concrete_facts],
            trajectories=[dict(fact.trajectory) for fact in concrete_facts],
            objective_scores=None,
            num_metric_calls=0,
        )

        if state.full_program_trace:
            state.full_program_trace[-1].setdefault(
                "admission_references",
                [],
            ).append(
                {
                    "parent_idx": parent_program_idx,
                    "admission_ids": list(ids),
                    "reference_program_indices": list(reference_program_indices),
                    "reference_scores": list(eval_before.scores),
                }
            )
        self.logger.log(
            "Admission references: "
            f"parent={parent_program_idx},ids={list(ids)},"
            f"owners={list(reference_program_indices)},"
            f"scores={list(eval_before.scores)}"
        )
        return AdmissionPlan(
            evaluation_ids=ids,
            evaluation_batch=batch,
            eval_before=eval_before,
        )


class CleanMiniAdmissionHook(MiniAdmissionHook):
    """Exclude the prospective child's recursive ``B_propose`` lineage."""

    def frontier_ineligible_ids(
        self,
        *,
        state: GEPAState,
        parent_program_idx: int,
        propose_ids: Sequence[DataId],
    ) -> Sequence[DataId]:
        return tuple(
            state.get_prospective_frontier_ineligible_ids(
                parent_program_idx,
                propose_ids,
            )
        )


class SparseMinibatchEvaluationPolicy:
    """Use admission evidence only and select by clean ``(F/E, E, -idx)``."""

    def get_seed_eval_batch(self, loader: DataLoader) -> list[DataId]:
        del loader
        return []

    def get_eval_batch(
        self,
        loader: DataLoader,
        state: GEPAState,
        target_program_idx: int | None = None,
    ) -> list[DataId]:
        del loader, state, target_program_idx
        raise RuntimeError(
            "admission proposals must bypass validation-policy candidate evaluation"
        )

    def get_best_program(self, state: GEPAState) -> int:
        eligible = tuple(
            candidate_idx
            for candidate_idx in range(len(state.program_candidates))
            if evaluation_count(state, candidate_idx) > 0
        )
        if not eligible:
            return 0
        return max(
            eligible,
            key=lambda idx: (
                frontier_rate(state, idx),
                evaluation_count(state, idx),
                -idx,
            ),
        )

    def get_valset_score(self, program_idx: int, state: GEPAState) -> float:
        return frontier_rate(state, program_idx)


class SeedFallbackParetoCandidateSelector:
    """Use official Pareto selection after the empty sparse-seed frontier."""

    def __init__(self, rng: random.Random) -> None:
        self._delegate = ParetoCandidateSelector(rng)

    def select_candidate_idx(self, state: GEPAState) -> int:
        if not any(state.get_pareto_front_mapping().values()):
            return 0
        return self._delegate.select_candidate_idx(state)


class _NoOpCandidatePoolObserver:
    def update_candidate_pool(
        self,
        candidates: Sequence[Mapping[str, str]],
    ) -> None:
        del candidates


@dataclass(frozen=True, slots=True)
class CompassReflectionEngineConfig:
    run_dir: Path
    condition: ReflectionCondition
    seed: int
    reflection_minibatch_size: int
    parent_top_n: int
    max_metric_calls: int
    perfect_score: float
    failure_score: float
    num_threads: int | None
    max_candidate_workers: int
    skip_perfect_score: bool
    add_format_failure_as_feedback: bool
    track_best_outputs: bool
    display_progress_bar: bool
    raise_on_exception: bool
    use_cloudpickle: bool
    epoch_parallel_enabled: bool = False
    max_reflection_workers: int = 1


@dataclass(frozen=True, slots=True)
class CompassReflectionRun:
    result: Any
    adapter: SparseObservationDspyAdapter
    evaluation_policy: SparseMinibatchEvaluationPolicy


def proposal_sampling_strategy(
    *,
    trainset_size: int,
    minibatch_size: int,
    epoch_parallel_enabled: bool,
) -> SingleMutationSampling | IndependentSampling:
    """Use one proposal task or one independent task per epoch minibatch."""

    if not epoch_parallel_enabled:
        return SingleMutationSampling()
    proposal_tasks = (trainset_size + minibatch_size - 1) // minibatch_size
    if proposal_tasks <= 0:
        raise ValueError("whole-epoch proposal sampling requires a non-empty trainset")
    return IndependentSampling(proposal_tasks)


def run_compass_reflection_engine(
    *,
    program: Any,
    metric_fn: Callable[..., Any],
    feedback_map: dict[str, Callable[..., Mapping[str, Any]]],
    trainset: list[Any],
    reflection_lm: Any,
    config: CompassReflectionEngineConfig,
) -> CompassReflectionRun:
    """Run reflection through the official GEPA engine and DSPy adapter."""

    if config.condition not in (
        "mini_admission_reflection",
        "compass_reflection",
    ):
        raise ValueError(f"unsupported reflection condition: {config.condition!r}")
    seed_candidate = {
        name: predictor.signature.instructions
        for name, predictor in program.named_predictors()
    }
    if not seed_candidate:
        raise RuntimeError("official program exposes no named predictors")

    logger = Logger(str(config.run_dir / "run_log.txt"))
    adapter_rng = random.Random(config.seed)
    strategy_rng = random.Random(config.seed)
    sampler = EpochShuffledBatchSampler(
        minibatch_size=config.reflection_minibatch_size,
        rng=strategy_rng,
        iteration_is_epoch=config.epoch_parallel_enabled,
    )
    sampling_strategy = proposal_sampling_strategy(
        trainset_size=len(trainset),
        minibatch_size=config.reflection_minibatch_size,
        epoch_parallel_enabled=config.epoch_parallel_enabled,
    )
    if config.epoch_parallel_enabled:
        assert isinstance(sampling_strategy, IndependentSampling)
        logger.log(
            "Whole-epoch proposal wave enabled: "
            f"trainset_size={len(trainset)}, "
            f"minibatch_size={config.reflection_minibatch_size}, "
            f"n_tasks={sampling_strategy.n}, "
            f"max_candidate_workers={config.max_candidate_workers}, "
            f"max_reflection_workers={config.max_reflection_workers}"
        )
    adapter = SparseObservationDspyAdapter(
        student_module=program,
        metric_fn=metric_fn,
        feedback_map=feedback_map,
        failure_score=config.failure_score,
        num_threads=config.num_threads,
        add_format_failure_as_feedback=config.add_format_failure_as_feedback,
        rng=adapter_rng,
        reflection_lm=reflection_lm,
        warn_on_score_mismatch=True,
        reflection_minibatch_size=config.reflection_minibatch_size,
        raise_on_error=config.raise_on_exception,
        max_candidate_workers=config.max_candidate_workers,
        max_reflection_workers=config.max_reflection_workers,
    )
    hook_class = (
        MiniAdmissionHook
        if config.condition == "mini_admission_reflection"
        else CleanMiniAdmissionHook
    )
    admission_hook = hook_class(
        adapter=adapter,
        rng=strategy_rng,
        run_dir=config.run_dir,
        logger=logger,
    )
    if config.condition == "mini_admission_reflection":
        selector = SeedFallbackParetoCandidateSelector(strategy_rng)
    else:
        selector = ReversibleMaskedExposureCorrectedCandidateSelector(
            strategy_rng,
            _NoOpCandidatePoolObserver(),
            logger,
            top_n=config.parent_top_n,
        )
    evaluation_policy = SparseMinibatchEvaluationPolicy()

    result = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=trainset,
        adapter=adapter,
        task_lm=None,
        evaluator=None,
        reflection_lm=None,
        candidate_selection_strategy=selector,
        frontier_type="instance",
        skip_perfect_score=config.skip_perfect_score,
        batch_sampler=sampler,
        reflection_minibatch_size=None,
        perfect_score=config.perfect_score,
        reflection_prompt_template=None,
        custom_candidate_proposer=None,
        module_selector="round_robin",
        use_merge=False,
        max_metric_calls=config.max_metric_calls,
        logger=logger,
        run_dir=str(config.run_dir),
        callbacks=None,
        use_wandb=False,
        use_mlflow=False,
        track_best_outputs=config.track_best_outputs,
        display_progress_bar=config.display_progress_bar,
        use_cloudpickle=config.use_cloudpickle,
        cache_evaluation=True,
        seed=config.seed,
        raise_on_exception=config.raise_on_exception,
        val_evaluation_policy=evaluation_policy,
        acceptance_criterion=StrictImprovementAcceptance(),
        sampling_strategy=sampling_strategy,
        selection_strategy=AllImprovements(),
        reflection_strategy=None,
        admission_hook=admission_hook,
    )
    return CompassReflectionRun(
        result=result,
        adapter=adapter,
        evaluation_policy=evaluation_policy,
    )


__all__ = [
    "CleanMiniAdmissionHook",
    "CompassReflectionEngineConfig",
    "CompassReflectionRun",
    "MiniAdmissionHook",
    "SeedFallbackParetoCandidateSelector",
    "SparseMinibatchEvaluationPolicy",
    "SparseObservation",
    "SparseObservationDspyAdapter",
    "proposal_sampling_strategy",
    "run_compass_reflection_engine",
    "select_reference_program_idx",
]
