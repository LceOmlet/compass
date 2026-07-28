from __future__ import annotations

import random
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Protocol

import dspy
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from gepa import optimize
from gepa.core.adapter import EvaluationBatch
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.logging.logger import Logger, LoggerProtocol
from gepa.proposer.reflective_mutation.admission import (
    AdmissionPlan,
    commit_adapter_observations,
)
from gepa.strategies.acceptance import StrictImprovementAcceptance
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.strategies.proposal_sampling import SingleMutationSampling
from gepa.strategies.proposal_selection import AllImprovements
from gepa_artifact.benchmarks.IFBench import (
    IFBenchCoT2StageProgram,
    feedback_fn_map as official_feedback_fn_map,
    metric as official_metric,
)

from bridge.b17_dspy_terminal_analysis import (
    DSPyObservationFact,
    DSPyReferenceObservation,
    DSPyTerminalAnalysisBridge,
    DSPyTerminalAnalysisCaptureMixin,
)
from bridge.b19_reversible_parent_selection import (
    ReversibleMaskedExposureCorrectedCandidateSelector,
)
from bridge.b20_compass_reflection import (
    SparseMinibatchEvaluationPolicy,
    select_reference_program_idx,
)
from bridge.terminal_reflection import TerminalAnalysisUnavailableError

DataId = Hashable


class TerminalProposalFn(Protocol):
    """Injected terminal-likelihood/FlashTrace proposer.

    The official DSPy adapter owns invocation of this public ``ProposalFn``
    extension point.  This module deliberately does not implement proposal
    generation, teacher forcing, attribution, or dependency gating.
    """

    def __call__(
        self,
        *,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]: ...

    def update_candidate_pool(
        self,
        candidates: Sequence[Mapping[str, str]],
    ) -> None: ...

    def set_logger(self, logger: LoggerProtocol) -> None: ...


IFBenchForwardProgram = IFBenchCoT2StageProgram


class SeedlessDspyAdapter(DspyAdapter):
    """Permit GEPA's explicit empty seed evaluation without running DSPy."""

    def evaluate(self, batch, candidate, capture_traces=False):
        if batch:
            return super().evaluate(batch, candidate, capture_traces)
        return EvaluationBatch(
            outputs=[],
            scores=[],
            trajectories=[] if capture_traces else None,
            num_metric_calls=0,
        )


class TerminalAnalysisDspyAdapter(
    DSPyTerminalAnalysisCaptureMixin,
    SeedlessDspyAdapter,
):
    """Observe official adapter calls through the existing b17 mixin."""


def seed_candidate_from_program(program: IFBenchForwardProgram) -> dict[str, str]:
    """Read the official candidate component names and instructions verbatim."""

    return {
        name: predictor.signature.instructions
        for name, predictor in program.named_predictors()
    }


def canonical_ifbench_feedback_map(
    program: IFBenchForwardProgram,
) -> dict[str, Callable[..., dict[str, Any]]]:
    """Map the artifact's legacy feedback keys to canonical DSPy keys.

    The artifact returns ``feedback_score``/``feedback_text`` while the
    current official ``DspyAdapter`` consumes ``score``/``feedback``.
    Predictor names must already agree exactly; no fuzzy name mapping is used.
    """

    predictor_names = tuple(name for name, _ in program.named_predictors())
    if set(predictor_names) != set(official_feedback_fn_map):
        raise ValueError(
            "official IFBench predictor names and feedback keys differ: "
            f"predictors={predictor_names!r}, feedback={tuple(official_feedback_fn_map)!r}"
        )

    def adapt(delegate: Callable[..., Any]) -> Callable[..., dict[str, Any]]:
        def canonical_feedback(**kwargs: Any) -> dict[str, Any]:
            result = delegate(**kwargs)
            if not isinstance(result, Mapping):
                raise TypeError("official IFBench feedback must return a mapping")
            if "feedback_score" not in result or "feedback_text" not in result:
                raise KeyError(
                    "official IFBench feedback must contain feedback_score and feedback_text"
                )
            return {
                "score": result["feedback_score"],
                "feedback": result["feedback_text"],
            }

        return canonical_feedback

    return {
        name: adapt(official_feedback_fn_map[name])
        for name in predictor_names
    }


def _finite_score(score: Real) -> float:
    value = float(score)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("GEPA score must be finite")
    return value


class IFBenchAdmissionHook:
    """Prepare heterogeneous old-frontier references for one admit minibatch."""

    def __init__(
        self,
        *,
        adapter: TerminalAnalysisDspyAdapter,
        analysis_bridge: DSPyTerminalAnalysisBridge,
        rng: random.Random,
        run_dir: Path,
        logger: LoggerProtocol,
    ) -> None:
        self.adapter = adapter
        self.analysis_bridge = analysis_bridge
        self.rng = rng
        self.run_dir = run_dir
        self.logger = logger

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

    def _reference_program_idx(
        self,
        *,
        state: GEPAState,
        instance_id: DataId,
        sampled_parent_idx: int,
    ) -> int:
        try:
            return select_reference_program_idx(
                state,
                instance_id=instance_id,
                sampled_parent_idx=sampled_parent_idx,
                rng=self.rng,
            )
        except RuntimeError as error:
            raise TerminalAnalysisUnavailableError(str(error)) from error

    def _stage_signatures(self, candidate: Mapping[str, str]) -> dict[str, Any]:
        program = self.adapter.build_program(dict(candidate))
        predictors = {
            name: predictor
            for name, predictor in program.named_predictors()
        }
        expected = {
            "generate_response_module.predict",
            "ensure_correct_response_module.predict",
        }
        if set(predictors) != expected:
            raise TerminalAnalysisUnavailableError(
                "official IFBench program does not expose exactly its two predictors"
            )
        return {name: predictors[name].signature for name in expected}

    def _ensure_reference_fact(
        self,
        *,
        state: GEPAState,
        program_idx: int,
        instance_id: DataId,
        instance: Any,
    ) -> DSPyObservationFact:
        known_score = state.prog_candidate_val_subscores[program_idx].get(instance_id)
        fact = self.adapter.get_program_observation(program_idx, instance_id)
        if fact is not None:
            if known_score is None or fact.score != known_score:
                raise TerminalAnalysisUnavailableError(
                    "persisted reference observation is not bound to the official maximum reward"
                )
            return fact

        candidate = state.program_candidates[program_idx]
        evaluated = self.adapter.evaluate([instance], candidate, capture_traces=True)
        metric_calls = (
            evaluated.num_metric_calls
            if evaluated.num_metric_calls is not None
            else 1
        )
        state.increment_evals(metric_calls)
        if len(evaluated.scores) != 1 or len(evaluated.outputs) != 1:
            raise TerminalAnalysisUnavailableError(
                "official missing-reference evaluation is not one-instance aligned"
            )
        if state.evaluation_cache is not None:
            objective_scores = (
                list(evaluated.objective_scores)
                if evaluated.objective_scores
                else None
            )
            state.evaluation_cache.put_batch(
                candidate,
                [instance_id],
                evaluated.outputs,
                evaluated.scores,
                objective_scores,
            )

        observed = ValsetEvaluation(
            outputs_by_val_id={instance_id: evaluated.outputs[0]},
            scores_by_val_id={instance_id: _finite_score(evaluated.scores[0])},
            objective_scores_by_val_id=(
                {instance_id: dict(evaluated.objective_scores[0])}
                if evaluated.objective_scores
                else None
            ),
        )
        committed = state.commit_existing_program_evaluation(
            program_idx=program_idx,
            valset_evaluation=observed,
            run_dir=str(self.run_dir),
            iteration=state.i + 1,
        )
        current_score = state.prog_candidate_val_subscores[program_idx].get(instance_id)
        observed_score = observed.scores_by_val_id[instance_id]
        bind_ids: Sequence[DataId]
        if instance_id in committed:
            bind_ids = committed
        elif current_score == observed_score:
            # A checkpoint may contain the exact scalar maximum but no older
            # trajectory fact. An equal re-execution can fill that missing
            # atomic observation without replacing an existing equal fact.
            bind_ids = (instance_id,)
        else:
            raise TerminalAnalysisUnavailableError(
                "re-executed frontier reference did not reproduce its maximum reward"
            )
        commit_adapter_observations(
            self.adapter,
            program_idx=program_idx,
            evaluation_ids=(instance_id,),
            evaluation=evaluated,
            committed_ids=bind_ids,
        )
        fact = self.adapter.get_program_observation(program_idx, instance_id)
        if fact is None or fact.score != current_score:
            raise TerminalAnalysisUnavailableError(
                "official reference execution did not produce a max-bound observation fact"
            )
        return fact

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
        del propose_evaluation
        components = tuple(components_to_update)
        if len(components) != 1:
            raise TerminalAnalysisUnavailableError(
                "official IFBench admission requires its round-robin single component"
            )
        ids = tuple(admission_ids)
        batch = tuple(admission_batch)
        if len(ids) != len(batch):
            raise RuntimeError("official admit IDs and data instances are misaligned")

        references: list[DSPyReferenceObservation] = []
        outputs: list[Any] = []
        scores: list[float] = []
        trajectories: list[Any] = []
        signatures: dict[int, dict[str, Any]] = {}
        try:
            for instance_id, instance in zip(ids, batch, strict=True):
                program_idx = self._reference_program_idx(
                    state=state,
                    instance_id=instance_id,
                    sampled_parent_idx=parent_program_idx,
                )
                candidate = state.program_candidates[program_idx]
                fact = self._ensure_reference_fact(
                    state=state,
                    program_idx=program_idx,
                    instance_id=instance_id,
                    instance=instance,
                )
                if program_idx not in signatures:
                    signatures[program_idx] = self._stage_signatures(candidate)
                stage_signatures = signatures[program_idx]
                references.append(
                    DSPyReferenceObservation(
                        program_idx=program_idx,
                        instance_id=instance_id,
                        candidate=dict(candidate),
                        fact=fact,
                        stage_signatures=stage_signatures,
                    )
                )
                outputs.append(fact.output)
                scores.append(fact.score)
                trajectories.append(fact.trajectory)
        except TerminalAnalysisUnavailableError as error:
            self.logger.log(
                f"Admission references unavailable; mutation skipped: {error}"
            )
            return None

        eval_before = EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=None,
            num_metric_calls=0,
        )
        self.analysis_bridge.bind_admission_references(
            parent_candidate=parent_candidate,
            component=components[0],
            reflective_dataset=reflective_dataset,
            references=tuple(references),
        )
        return AdmissionPlan(
            evaluation_ids=ids,
            evaluation_batch=batch,
            eval_before=eval_before,
        )


@dataclass(frozen=True)
class OfficialIFBenchEngineConfig:
    run_dir: Path
    seed: int
    reflection_minibatch_size: int
    parent_top_n: int
    max_metric_calls: int
    perfect_score: float
    failure_score: float
    num_threads: int | None
    skip_perfect_score: bool
    add_format_failure_as_feedback: bool
    track_best_outputs: bool
    display_progress_bar: bool
    raise_on_exception: bool
    use_cloudpickle: bool


@dataclass(frozen=True)
class OfficialIFBenchRun:
    result: Any


def run_official_ifbench_engine(
    *,
    trainset: list[Any],
    terminal_proposal: TerminalProposalFn,
    terminal_analysis_bridge: DSPyTerminalAnalysisBridge,
    config: OfficialIFBenchEngineConfig,
) -> OfficialIFBenchRun:
    """Run IFBench through official GEPA/DSPy orchestration only."""

    program = IFBenchForwardProgram()
    seed_candidate = seed_candidate_from_program(program)
    feedback_map = canonical_ifbench_feedback_map(program)
    logger = Logger(str(config.run_dir / "run_log.txt"))
    terminal_proposal.set_logger(logger)

    # Match the official DSPy GEPA wrapper: adapter trace sampling and GEPA
    # strategies start from the same seed but advance independent RNG streams.
    adapter_rng = random.Random(config.seed)
    strategy_rng = random.Random(config.seed)
    sampler = EpochShuffledBatchSampler(
        minibatch_size=config.reflection_minibatch_size,
        rng=strategy_rng,
    )

    adapter = TerminalAnalysisDspyAdapter(
        student_module=program,
        metric_fn=official_metric,
        feedback_map=feedback_map,
        failure_score=config.failure_score,
        num_threads=config.num_threads,
        add_format_failure_as_feedback=config.add_format_failure_as_feedback,
        rng=adapter_rng,
        custom_instruction_proposer=terminal_proposal,
        warn_on_score_mismatch=True,
        reflection_minibatch_size=config.reflection_minibatch_size,
    )
    adapter.install_terminal_analysis_bridge(terminal_analysis_bridge)
    admission_hook = IFBenchAdmissionHook(
        adapter=adapter,
        analysis_bridge=terminal_analysis_bridge,
        rng=strategy_rng,
        run_dir=config.run_dir,
        logger=logger,
    )
    selector = ReversibleMaskedExposureCorrectedCandidateSelector(
        strategy_rng,
        terminal_proposal,
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
        sampling_strategy=SingleMutationSampling(),
        selection_strategy=AllImprovements(),
        reflection_strategy=None,
        admission_hook=admission_hook,
    )
    return OfficialIFBenchRun(result=result)
