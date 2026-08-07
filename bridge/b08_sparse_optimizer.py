from __future__ import annotations

import math
import os
import traceback
from collections.abc import Callable, Hashable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Protocol

import dspy
from dspy.teleprompt.bootstrap_finetune import FailedPrediction
from dspy.utils.parallelizer import ParallelExecutor

from gepa_artifact.gepa.gepa_utils import capture_module_trace_with_feedback
from gepa_artifact.gepa.gepa_utils import GEPAState
from gepa_artifact.gepa.instruction_proposal import ProposeNewInstructionModule

from .b01_aime_capture import CapturedRollout
from .b03_token_replay import HistoricalReplayContextError
from .b06_dependency_gate import (
    DependencyMeasureUnavailableError,
    passes_dependency_gate,
)
from .b07_sparse_state import SparseSkillState
from .b11_terminal_likelihood import AIMEHistoricalReplayScorer
from .b13_reflection_budget import OfficialReflectionInputBudget
from .dspy_evaluate import without_parent_straggler_resubmission


class PreparedReplayDependency(Protocol):
    def distance(self, candidate_skill: str) -> Real: ...


class PreparedParentAnalysis(Protocol):
    rollout_credits: tuple[tuple[CapturedRollout, Any], ...]
    dependency: PreparedReplayDependency


class ReplayDependencyDistance(Protocol):
    """Compute the paired historical-replay distance through the b06 bridge."""

    def __call__(
        self,
        *,
        parent_skill: str,
        candidate_skill: str,
        rollouts: tuple[CapturedRollout, ...],
    ) -> Real: ...

    def prepare(
        self,
        *,
        parent_skill: str,
        rollouts: tuple[CapturedRollout, ...],
    ) -> PreparedReplayDependency: ...

    def prepare_parent_analysis(
        self,
        *,
        historical_replay: AIMEHistoricalReplayScorer,
        parent_skill: str,
        rollouts: tuple[CapturedRollout, ...],
    ) -> PreparedParentAnalysis: ...


@dataclass(frozen=True, slots=True)
class SparseOptimizerResult:
    frontier_pool: tuple[Hashable, ...]
    front_mapping: dict[int, set[Hashable]]


ProgramIdentity = tuple[tuple[str, str], ...]


class AIMESparseDependencyOptimizer:
    """Sparse AIME loop composed from the pinned Artifact and project bridges."""

    def __init__(
        self,
        *,
        gepa: Any,
        current_minibatch: Callable[[], tuple[CapturedRollout, ...]],
        dependency_distance: ReplayDependencyDistance,
        historical_replay: AIMEHistoricalReplayScorer,
        reflection_budget: OfficialReflectionInputBudget | None,
        reflection_lm: Any,
        n_candidates: Integral,
        epsilon_dep: Real,
    ) -> None:
        if not callable(current_minibatch):
            raise TypeError("current_minibatch must be the capture bridge callback")
        if not callable(dependency_distance):
            raise TypeError("dependency_distance must be callable")
        if not isinstance(historical_replay, AIMEHistoricalReplayScorer):
            raise TypeError("historical_replay must be AIMEHistoricalReplayScorer")
        if reflection_budget is not None and not isinstance(
            reflection_budget,
            OfficialReflectionInputBudget,
        ):
            raise TypeError("reflection_budget must be OfficialReflectionInputBudget or None")
        if not callable(reflection_lm):
            raise TypeError("reflection_lm must be the configured external proposal LM")
        if isinstance(n_candidates, bool) or not isinstance(n_candidates, Integral) or n_candidates <= 0:
            raise ValueError("n_candidates must be a positive integer")
        if getattr(gepa, "num_iters", None) is not None:
            raise ValueError("sparse mode does not reinterpret official num_iters")
        if getattr(gepa, "max_evals_per_trainval_instance", None) is not None:
            raise ValueError(
                "sparse mode does not reinterpret official max_evals_per_trainval_instance"
            )
        max_metric_calls = getattr(gepa, "max_metric_calls", None)
        if (
            isinstance(max_metric_calls, bool)
            or not isinstance(max_metric_calls, Integral)
            or max_metric_calls < 0
        ):
            raise ValueError("sparse mode requires official max_metric_calls as a non-negative integer")
        if getattr(gepa, "gepa_state", None) is not None:
            raise RuntimeError("sparse and official full-evaluation state cannot share one GEPA run")
        failure_score = getattr(gepa, "failure_score", None)
        if (
            isinstance(failure_score, bool)
            or not isinstance(failure_score, Real)
            or not math.isfinite(float(failure_score))
        ):
            raise ValueError("the official AIME failure_score must be a finite real number")

        # Validate the required, default-free threshold through its owning bridge.
        passes_dependency_gate(distance=0.0, epsilon_dep=epsilon_dep)

        if getattr(gepa, "num_threads", None) is None:
            gepa.num_threads = os.cpu_count()

        self._gepa = gepa
        self._current_minibatch = current_minibatch
        self._dependency_distance = dependency_distance
        self._historical_replay = historical_replay
        self._reflection_budget = reflection_budget
        self._reflection_lm = reflection_lm
        self._n_candidates = int(n_candidates)
        self._epsilon_dep = epsilon_dep
        self._max_metric_calls = int(max_metric_calls)
        self._metric_calls = 0
        self._runtime: GEPAState | None = None

    def _propose_candidates(self, proposer: ProposeNewInstructionModule) -> tuple[str, ...]:
        executor = ParallelExecutor(
            num_threads=self._n_candidates,
            max_errors=self._n_candidates,
            disable_progress_bar=True,
            provide_traceback=True,
            timeout=0,
            straggler_limit=0,
        )
        proposals = executor.execute(lambda _: proposer.compile(), range(self._n_candidates))
        candidates: list[str] = []
        for proposal in proposals:
            if proposal is None:
                continue
            if not isinstance(proposal, Mapping):
                raise TypeError("the official proposer must return a mapping")
            new_instruction = proposal.get("new_instruction")
            if not isinstance(new_instruction, str):
                raise TypeError("the official fenced extractor must return one instruction string")
            candidates.append(new_instruction)
        return tuple(candidates)

    def _discard_parent_analysis(self, future: Future[PreparedParentAnalysis]) -> None:
        """Wait for an unneeded analysis without changing proposal skip semantics."""

        try:
            future.result()
        except Exception as error:
            iteration = self._runtime.i + 1 if self._runtime is not None else 0
            self._gepa.logger.log(
                f"Iteration {iteration}: "
                f"Discarded parent analysis failed after proposal path ended: {error}"
            )

    @staticmethod
    def _named_predictors(program: dspy.Module) -> tuple[tuple[str, Any], ...]:
        predictors = tuple(program.named_predictors())
        if not predictors:
            raise ValueError("the official program must expose at least one predictor")
        names: list[str] = []
        for name, predictor in predictors:
            if not isinstance(name, str) or not name:
                raise TypeError("the official predictor name must be non-empty text")
            signature = getattr(predictor, "signature", None)
            instruction = getattr(signature, "instructions", None)
            if not isinstance(instruction, str):
                raise TypeError("the official predictor instruction must be text")
            names.append(name)
        if len(set(names)) != len(names):
            raise RuntimeError("the official program exposes duplicate predictor names")
        return predictors

    @classmethod
    def _program_identity(cls, program: dspy.Module) -> ProgramIdentity:
        return tuple(
            (name, predictor.signature.instructions)
            for name, predictor in cls._named_predictors(program)
        )

    @classmethod
    def _predictor(cls, program: dspy.Module, predictor_name: str) -> Any:
        matches = [
            predictor
            for name, predictor in cls._named_predictors(program)
            if name == predictor_name
        ]
        if len(matches) != 1:
            raise RuntimeError("selected predictor is absent or ambiguous in the official program")
        return matches[0]

    @staticmethod
    def _instruction(identity: ProgramIdentity, predictor_name: str) -> str:
        matches = [instruction for name, instruction in identity if name == predictor_name]
        if len(matches) != 1:
            raise RuntimeError("selected predictor is absent or ambiguous in program identity")
        return matches[0]

    @staticmethod
    def _candidate_identity(
        parent_identity: ProgramIdentity,
        predictor_name: str,
        new_instruction: str,
    ) -> ProgramIdentity:
        if not isinstance(new_instruction, str):
            raise TypeError("candidate instruction must be text")
        changed = False
        candidate: list[tuple[str, str]] = []
        for name, instruction in parent_identity:
            if name == predictor_name:
                candidate.append((name, new_instruction))
                changed = True
            else:
                candidate.append((name, instruction))
        if not changed:
            raise RuntimeError("selected predictor is absent from parent program identity")
        return tuple(candidate)

    @staticmethod
    def _score_mapping(
        instance_ids: Sequence[int],
        scores: Sequence[Real],
    ) -> dict[int, float]:
        ids = tuple(instance_ids)
        values = tuple(scores)
        if len(ids) != len(values):
            raise RuntimeError("scores do not align with the original minibatch instance IDs")
        if len(set(ids)) != len(ids):
            raise RuntimeError("one minibatch cannot contain duplicate instance IDs")

        mapped: dict[int, float] = {}
        for instance_id, score in zip(ids, values, strict=True):
            if isinstance(score, bool) or not isinstance(score, Real):
                raise TypeError("the official metric must return real scores")
            value = float(score)
            if not math.isfinite(value):
                raise ValueError("the official metric must return finite scores")
            mapped[instance_id] = value
        return mapped

    @classmethod
    def _parent_feedback_score_mapping(
        cls,
        *,
        selected_ids: Sequence[int],
        selected_examples: Sequence[dspy.Example],
        dataset_with_feedback: Sequence[Mapping[str, Any]],
        subsample_scores: Sequence[Real],
        subsample_score: Real,
        failure_score: Real,
    ) -> tuple[dict[int, float], tuple[int, ...]]:
        """Restore scores omitted by the official feedback capture to selected IDs."""

        ids = tuple(selected_ids)
        examples = tuple(selected_examples)
        feedback = tuple(dataset_with_feedback)
        scores = tuple(subsample_scores)
        if len(ids) != len(examples):
            raise RuntimeError("selected examples do not align with selected instance IDs")
        if len(set(ids)) != len(ids):
            raise RuntimeError("one minibatch cannot contain duplicate instance IDs")
        if len(feedback) != len(scores):
            raise RuntimeError("official feedback records do not align with returned scores")

        selected_inputs: list[dict[str, Any]] = []
        for example in examples:
            if not isinstance(example, dspy.Example):
                raise TypeError("selected minibatch entries must be DSPy Examples")
            example_inputs = example.inputs()
            items = getattr(example_inputs, "items", None)
            if not callable(items):
                raise TypeError("selected DSPy Example inputs must expose items()")
            selected_inputs.append(dict(items()))

        for left in range(len(selected_inputs)):
            for right in range(left + 1, len(selected_inputs)):
                if selected_inputs[left] == selected_inputs[right]:
                    raise RuntimeError(
                        "selected minibatch contains duplicate predictor inputs"
                    )

        matched_positions: list[int] = []
        seen_positions: set[int] = set()
        for record in feedback:
            if not isinstance(record, Mapping):
                raise TypeError("official feedback records must be mappings")
            returned_inputs = record.get("inputs")
            if not isinstance(returned_inputs, Mapping):
                raise TypeError("official feedback record omitted predictor inputs")
            exact_inputs = dict(returned_inputs)
            matches = [
                position
                for position, expected_inputs in enumerate(selected_inputs)
                if exact_inputs == expected_inputs
            ]
            if not matches:
                raise RuntimeError("official feedback contains unknown predictor inputs")
            if len(matches) != 1:
                raise RuntimeError("official feedback predictor inputs are ambiguous")
            position = matches[0]
            if position in seen_positions:
                raise RuntimeError("official feedback repeats one selected instance")
            if matched_positions and position <= matched_positions[-1]:
                raise RuntimeError("official feedback changed selected minibatch order")
            seen_positions.add(position)
            matched_positions.append(position)

        feedback_ids = tuple(ids[position] for position in matched_positions)
        returned_scores = cls._score_mapping(feedback_ids, scores)
        if (
            isinstance(subsample_score, bool)
            or not isinstance(subsample_score, Real)
            or not math.isfinite(float(subsample_score))
        ):
            raise TypeError("the official aggregate minibatch score must be a finite real number")
        if float(subsample_score) != sum(returned_scores.values()):
            raise RuntimeError("official aggregate minibatch score differs from returned scores")
        if (
            isinstance(failure_score, bool)
            or not isinstance(failure_score, Real)
            or not math.isfinite(float(failure_score))
        ):
            raise TypeError("the official failure score must be a finite real number")

        restored = {instance_id: float(failure_score) for instance_id in ids}
        restored.update(returned_scores)
        return restored, feedback_ids

    def _captured_parent_rollouts(
        self,
        *,
        parent_skill: str,
        selected_ids: tuple[int, ...],
        expected_count: int | None,
    ) -> tuple[CapturedRollout, ...]:
        if expected_count == 0:
            return ()
        try:
            rollouts = tuple(self._current_minibatch())
        except RuntimeError as error:
            if "no captured" in str(error).lower():
                if expected_count is None:
                    return ()
                raise RuntimeError(
                    "official feedback contains parsed predictions but none were captured"
                ) from error
            raise
        if not rollouts:
            if expected_count is None:
                return ()
            raise RuntimeError("the official feedback batch produced no captured parent rollout")
        if expected_count is not None and len(rollouts) != expected_count:
            raise RuntimeError("captured parent rollouts do not match official parsed predictions")

        selected = set(selected_ids)
        seen: set[int] = set()
        for rollout in rollouts:
            if not isinstance(rollout, CapturedRollout):
                raise TypeError("current_minibatch must return CapturedRollout values")
            if self._runtime is None or rollout.iteration != self._runtime.i:
                raise RuntimeError("captured rollout belongs to a different sparse iteration")
            if rollout.skill_text != parent_skill:
                raise RuntimeError("captured rollout skill differs from the selected parent")
            if rollout.instance_index not in selected:
                raise RuntimeError("captured rollout lies outside the official selected minibatch")
            if rollout.instance_index in seen:
                raise RuntimeError("the same valid instance was captured twice")
            seen.add(rollout.instance_index)
        return rollouts

    def _candidate_program(
        self,
        *,
        parent_program: dspy.Module,
        predictor_name: str,
        new_instruction: str,
    ) -> dspy.Module:
        parent_identity = self._program_identity(parent_program)
        parent_lm = parent_program.get_lm()
        candidate = parent_program.deepcopy()
        candidate.set_lm(parent_lm)
        if tuple(name for name, _ in self._named_predictors(candidate)) != tuple(
            name for name, _ in parent_identity
        ):
            raise RuntimeError("candidate deepcopy changed official predictor order or identity")
        predictor = self._predictor(candidate, predictor_name)
        predictor.signature = predictor.signature.with_instructions(new_instruction)
        expected = self._candidate_identity(
            parent_identity,
            predictor_name,
            new_instruction,
        )
        if self._program_identity(candidate) != expected:
            raise RuntimeError("candidate mutation changed more than the selected predictor")
        return candidate

    def _evaluate_candidate(
        self,
        *,
        candidate: dspy.Module,
        minibatch_examples: list[dspy.Example],
    ) -> tuple[tuple[Real, ...], tuple[dspy.Prediction | None, ...]]:
        examples = tuple(minibatch_examples)
        executor = ParallelExecutor(
            num_threads=self._gepa.num_threads,
            max_errors=len(examples) + 1,
            disable_progress_bar=True,
            provide_traceback=True,
            compare_results=True,
            timeout=0,
            straggler_limit=0,
        )

        def evaluate_position(position: int) -> tuple[int, dspy.Prediction, Real]:
            example = examples[position]
            prediction = candidate(**example.inputs())
            score = self._gepa.metric_fn(example, prediction)
            return position, prediction, score

        self._metric_calls += len(examples)
        raw_results = executor.execute(evaluate_position, range(len(examples)))
        if len(raw_results) != len(examples):
            raise RuntimeError("official ParallelExecutor returned a misaligned candidate result vector")

        scores: list[Real] = []
        predictions: list[dspy.Prediction | None] = []
        for expected_position, result in enumerate(raw_results):
            if result is None:
                scores.append(self._gepa.failure_score)
                predictions.append(None)
                continue
            if not isinstance(result, tuple) or len(result) != 3:
                raise TypeError("candidate evaluation must return position, prediction, and score")
            actual_position, prediction, score = result
            if actual_position != expected_position:
                raise RuntimeError("official ParallelExecutor changed candidate minibatch order")
            scores.append(score)
            predictions.append(prediction)
        return tuple(scores), tuple(predictions)

    @property
    def metric_calls(self) -> int:
        return self._metric_calls

    def optimize(
        self,
        *,
        base_program: dspy.Module,
        trainset: Sequence[dspy.Example],
    ) -> SparseOptimizerResult:
        examples = tuple(trainset)
        if not examples:
            raise ValueError("trainset must be non-empty")

        seed_identity = self._program_identity(base_program)
        predictor_names = tuple(name for name, _ in seed_identity)
        self._runtime = GEPAState(
            base_program,
            (None, [], []),
            (float(self._gepa.failure_score), [], []),
            self._gepa.seed,
            track_scores_on=self._gepa.track_scores_on,
            run_linearized_gepa=self._gepa.run_linearized_gepa,
        )
        if tuple(self._runtime.list_of_named_predictors) != predictor_names:
            raise RuntimeError("official GEPAState changed the predictor order")
        self._gepa.gepa_state = self._runtime
        state: SparseSkillState[int] = SparseSkillState(seed_skill=seed_identity)
        program_indices: dict[ProgramIdentity, int] = {seed_identity: 0}

        while self._metric_calls < self._max_metric_calls:
            runtime = self._runtime
            runtime.i += 1
            runtime.full_program_trace.append({"i": runtime.i})

            parent_identity = state.select_parent(rng=runtime.rng1)
            if not isinstance(parent_identity, tuple) or parent_identity not in program_indices:
                raise RuntimeError("sparse parent identity is not a known complete program")
            parent_program_index = program_indices[parent_identity]
            parent_program = runtime.program_candidates[parent_program_index]
            if self._program_identity(parent_program) != parent_identity:
                raise RuntimeError("stored program no longer matches its immutable identity")
            if tuple(name for name, _ in parent_identity) != predictor_names:
                raise RuntimeError("program candidate changed official predictor order")

            predictor_to_update_id = (
                runtime.named_predictor_id_to_update_next_for_program_candidate[
                    parent_program_index
                ]
            )
            runtime.full_program_trace[-1]["selected_program_candidate"] = parent_program_index
            runtime.full_program_trace[-1]["predictor_to_update_id"] = predictor_to_update_id
            runtime.named_predictor_id_to_update_next_for_program_candidate[
                parent_program_index
            ] = (predictor_to_update_id + 1) % len(runtime.list_of_named_predictors)
            predictor_name = predictor_names[predictor_to_update_id]
            parent_predictor = self._predictor(parent_program, predictor_name)
            parent_skill = self._instruction(parent_identity, predictor_name)
            feedback_functions: Mapping[str, Callable[..., Any]] = (
                self._gepa.named_predictor_to_feedback_fn_map
            )
            if predictor_name not in feedback_functions:
                self._gepa.logger.log(
                    f"Iteration {runtime.i + 1}: Predictor {predictor_name} not in feedback map, skipping"
                )
                continue
            feedback_function = feedback_functions[predictor_name]

            selected_ids = tuple(
                self._gepa.select_training_sample_and_update_shuffled_trainset(
                    list(examples),
                    runtime.i,
                )
            )
            if not selected_ids:
                raise RuntimeError("the official GEPA sampler returned an empty minibatch")
            if any(
                isinstance(instance_id, bool)
                or not isinstance(instance_id, Integral)
                or instance_id < 0
                or instance_id >= len(examples)
                for instance_id in selected_ids
            ):
                raise RuntimeError("the official GEPA sampler returned an invalid instance ID")
            selected_ids = tuple(int(instance_id) for instance_id in selected_ids)
            runtime.full_program_trace[-1]["subsample_ids"] = list(selected_ids)

            self._metric_calls += len(selected_ids)
            state.record_evaluations(skill=parent_identity, instance_ids=selected_ids)
            selected_examples = [examples[index] for index in selected_ids]
            with without_parent_straggler_resubmission():
                dataset_with_feedback, subsample_score, subsample_scores = (
                    capture_module_trace_with_feedback(
                        parent_predictor,
                        parent_program,
                        selected_examples,
                        self._gepa.metric_fn,
                        self._gepa.logger,
                        runtime,
                        self._gepa.skip_perfect_score,
                        self._gepa.perfect_score,
                        failure_score=self._gepa.failure_score,
                        format_failure_score=self._gepa.failure_score,
                        feedback_func=feedback_function,
                        add_format_failure_as_feedback=self._gepa.add_format_failure_as_feedback,
                        num_threads=self._gepa.num_threads,
                    )
                )

            finish_parent_batch = getattr(feedback_function, "finish_parent_batch", None)
            if callable(finish_parent_batch):
                finish_parent_batch()
            feedback_was_empty = (
                dataset_with_feedback is None
                and subsample_score is None
                and subsample_scores is None
            )
            if not feedback_was_empty and (
                dataset_with_feedback is None
                or subsample_score is None
                or subsample_scores is None
            ):
                raise RuntimeError("official feedback capture returned a partial empty result")

            expected_rollouts = None if feedback_was_empty else sum(
                not isinstance(item["generated_output"], FailedPrediction)
                for item in dataset_with_feedback
            )
            rollouts = self._captured_parent_rollouts(
                parent_skill=parent_skill,
                selected_ids=selected_ids,
                expected_count=expected_rollouts,
            )

            if feedback_was_empty:
                parent_score_map = {
                    instance_id: float(self._gepa.failure_score)
                    for instance_id in selected_ids
                }
                parent_score_map.update(
                    (rollout.instance_index, float(rollout.reward))
                    for rollout in rollouts
                )
                state.commit_existing_observations(
                    skill=parent_identity,
                    scores_by_instance=parent_score_map,
                )
                continue

            failed_predictions = sum(
                isinstance(item["generated_output"], FailedPrediction)
                for item in dataset_with_feedback
            )
            if failed_predictions:
                rewrite_failed_feedback = getattr(
                    feedback_function,
                    "rewrite_failed_prediction_feedback",
                    None,
                )
                if callable(rewrite_failed_feedback):
                    rewrite_failed_feedback(dataset_with_feedback)
                discard_failed = getattr(
                    feedback_function,
                    "discard_failed_predictions",
                    None,
                )
                if callable(discard_failed):
                    discard_failed(
                        predictor=parent_predictor,
                        dataset_with_feedback=dataset_with_feedback,
                    )

            score_values = tuple(subsample_scores)
            if len(rollouts) == len(score_values) and all(
                float(rollout.reward) == float(score)
                for rollout, score in zip(rollouts, score_values, strict=True)
            ):
                returned_scores = self._score_mapping(
                    tuple(rollout.instance_index for rollout in rollouts),
                    score_values,
                )
                if float(subsample_score) != sum(returned_scores.values()):
                    raise RuntimeError("official aggregate minibatch score differs from returned scores")
                parent_score_map = {
                    instance_id: float(self._gepa.failure_score)
                    for instance_id in selected_ids
                }
                parent_score_map.update(returned_scores)
                feedback_instance_ids = tuple(rollout.instance_index for rollout in rollouts)
            else:
                parent_score_map, feedback_instance_ids = self._parent_feedback_score_mapping(
                    selected_ids=selected_ids,
                    selected_examples=selected_examples,
                    dataset_with_feedback=dataset_with_feedback,
                    subsample_scores=score_values,
                    subsample_score=subsample_score,
                    failure_score=self._gepa.failure_score,
                )
            state.commit_existing_observations(
                skill=parent_identity,
                scores_by_instance=parent_score_map,
            )
            for rollout in rollouts:
                if float(rollout.reward) != parent_score_map[rollout.instance_index]:
                    raise RuntimeError("captured parent reward differs from the official minibatch score")
            if not rollouts:
                continue

            if self._reflection_budget is None:
                reflection_samples = tuple(dataset_with_feedback)
            else:
                reflection_selection = self._reflection_budget.select(
                    base_program=parent_predictor,
                    dataset_with_feedback=dataset_with_feedback,
                )
                for position in reflection_selection.excluded_positions:
                    self._gepa.logger.log(
                        f"Iteration {runtime.i + 1}: "
                        f"Reflection excluded_instance_id={feedback_instance_ids[position]}, "
                        f"reason=individual_context_exceeded"
                    )
                if not reflection_selection.samples:
                    continue
                if not reflection_selection.combined_fits:
                    self._gepa.logger.log(
                        f"Iteration {runtime.i + 1}: "
                        f"Reflection batch_context_exceeded=True, mutation_skipped=True"
                    )
                    continue
                reflection_samples = tuple(reflection_selection.samples)

            proposer = ProposeNewInstructionModule(
                base_program=parent_predictor,
                instruction_lm=self._reflection_lm,
                dataset_with_feedback=list(reflection_samples),
                knowledgebase_qe=self._gepa.knowledgebase_qe,
            )
            with ThreadPoolExecutor(max_workers=1) as parent_analysis_executor:
                parent_analysis_future = parent_analysis_executor.submit(
                    self._dependency_distance.prepare_parent_analysis,
                    historical_replay=self._historical_replay,
                    parent_skill=parent_skill,
                    rollouts=rollouts,
                )
                try:
                    proposed = self._propose_candidates(proposer)
                except Exception as error:
                    self._gepa.logger.log(
                        f"Iteration {runtime.i + 1}: Exception during instruction proposal: {error}"
                    )
                    self._gepa.logger.log(traceback.format_exc())
                    self._discard_parent_analysis(parent_analysis_future)
                    continue

                candidate_instructions: list[str] = []
                candidate_identities: list[ProgramIdentity] = []
                seen: set[ProgramIdentity] = set()
                for instruction in proposed:
                    identity = self._candidate_identity(
                        parent_identity,
                        predictor_name,
                        instruction,
                    )
                    if identity in seen or state.is_known(identity):
                        continue
                    seen.add(identity)
                    candidate_instructions.append(instruction)
                    candidate_identities.append(identity)
                if not candidate_instructions:
                    self._discard_parent_analysis(parent_analysis_future)
                    continue

                scorable_candidates: list[str] = []
                scorable_identities: list[ProgramIdentity] = []
                scorable_original_indices: list[int] = []
                for index, (candidate_skill, identity) in enumerate(
                    zip(candidate_instructions, candidate_identities, strict=True)
                ):
                    try:
                        self._historical_replay.validate_candidate_context(candidate_skill, rollouts)
                    except HistoricalReplayContextError as error:
                        self._gepa.logger.log(
                            f"Iteration {runtime.i + 1}: "
                            f"Terminal candidate_index={index}, context_exceeded=True, "
                            f"rejected=True, detail={error}"
                        )
                        continue
                    scorable_candidates.append(candidate_skill)
                    scorable_identities.append(identity)
                    scorable_original_indices.append(index)
                if not scorable_candidates:
                    self._discard_parent_analysis(parent_analysis_future)
                    continue

                try:
                    parent_analysis = parent_analysis_future.result()
                except DependencyMeasureUnavailableError as error:
                    self._gepa.logger.log(
                        f"Iteration {runtime.i + 1}: "
                        f"Dependency unavailable=True, mutation_skipped=True, detail={error}"
                    )
                    continue

            replay_scores = self._historical_replay.scores_from_credits(
                tuple(scorable_candidates),
                parent_analysis.rollout_credits,
            )
            if len(replay_scores) != len(scorable_candidates) or any(
                not math.isfinite(score) for score in replay_scores
            ):
                raise RuntimeError("terminal teacher-forcing scores are invalid or misaligned")
            ranked = sorted(
                range(len(scorable_candidates)),
                key=lambda index: (-replay_scores[index], index),
            )
            prepared_dependency = parent_analysis.dependency
            selected_index: int | None = None
            dependency_unavailable = False
            for index in ranked:
                try:
                    distance = prepared_dependency.distance(scorable_candidates[index])
                except DependencyMeasureUnavailableError as error:
                    self._gepa.logger.log(
                        f"Iteration {runtime.i + 1}: "
                        f"Terminal candidate_index={scorable_original_indices[index]}, "
                        f"dependency_unavailable=True, mutation_skipped=True, detail={error}"
                    )
                    dependency_unavailable = True
                    break
                gate_passed = passes_dependency_gate(
                    distance=distance,
                    epsilon_dep=self._epsilon_dep,
                )
                self._gepa.logger.log(
                    f"Iteration {runtime.i + 1}: "
                    f"Terminal candidate_index={scorable_original_indices[index]}, "
                    f"teacher_forcing_score={replay_scores[index]:.17g}, "
                    f"dependency_distance={float(distance):.17g}, "
                    f"epsilon_dep={float(self._epsilon_dep):.17g}, "
                    f"gate_passed={gate_passed}"
                )
                if gate_passed:
                    selected_index = index
                    break
            if dependency_unavailable:
                continue
            if selected_index is None:
                continue

            new_instruction = scorable_candidates[selected_index]
            candidate_identity = scorable_identities[selected_index]
            candidate = self._candidate_program(
                parent_program=parent_program,
                predictor_name=predictor_name,
                new_instruction=new_instruction,
            )
            if self._program_identity(candidate) != candidate_identity:
                raise RuntimeError("candidate program identity differs from the selected proposal")

            candidate_examples = [examples[index] for index in selected_ids]
            candidate_scores, candidate_predictions = self._evaluate_candidate(
                candidate=candidate,
                minibatch_examples=candidate_examples,
            )
            failed_candidate_positions = frozenset(
                index
                for index, prediction in enumerate(candidate_predictions)
                if prediction is None
            )
            discard_candidate = getattr(
                feedback_function,
                "discard_program_rollouts",
                None,
            )
            if callable(discard_candidate):
                discard_candidate(
                    program=candidate,
                    examples=candidate_examples,
                    predictions=candidate_predictions,
                )
            candidate_score_map = self._score_mapping(selected_ids, candidate_scores)
            candidate_total = math.fsum(candidate_score_map.values())
            parent_total = math.fsum(parent_score_map.values())
            self._gepa.logger.log(
                f"Iteration {runtime.i + 1}: "
                f"Sparse candidate minibatch_score={candidate_total:.17g}/{len(selected_ids)}, "
                f"failed_instances={len(failed_candidate_positions)}"
            )
            reward_delta = candidate_total - parent_total
            if reward_delta <= 0.0:
                self._gepa.logger.log(
                    f"Iteration {runtime.i + 1}: "
                    f"Sparse candidate reward_delta={reward_delta:.17g}, accepted=False"
                )
                continue

            accepted = state.commit_candidate(
                skill=candidate_identity,
                scores_by_instance=candidate_score_map,
            )
            self._gepa.logger.log(
                f"Iteration {runtime.i + 1}: "
                f"Sparse candidate reward_delta={reward_delta:.17g}, accepted={accepted}"
            )
            if accepted:
                candidate_index, _ = runtime.update_state_with_new_program(
                    parent_program_idx=[parent_program_index],
                    new_program=candidate,
                    trainset_score=None,
                    trainset_outputs=[],
                    trainset_subscores=[],
                    valset_score=float(self._gepa.failure_score),
                    valset_outputs=[],
                    valset_subscores=[],
                    run_dir=self._gepa.run_dir,
                    track_scores_on=self._gepa.track_scores_on,
                    num_metric_calls_by_discovery_of_new_program=self._metric_calls,
                )
                if candidate_index != len(runtime.program_candidates) - 1:
                    raise RuntimeError("official GEPAState returned a nonterminal candidate index")
                if self._program_identity(runtime.program_candidates[candidate_index]) != candidate_identity:
                    raise RuntimeError("official GEPAState changed the accepted program identity")
                program_indices[candidate_identity] = candidate_index
                if not runtime.is_consistent():
                    raise RuntimeError("official GEPAState scheduler projection is inconsistent")

        return SparseOptimizerResult(
            frontier_pool=state.frontier_pool(),
            front_mapping=state.front_mapping(),
        )
