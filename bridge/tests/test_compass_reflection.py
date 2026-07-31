from __future__ import annotations

import random
import threading
import time
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import dspy
import pytest
from gepa.core.adapter import EvaluationBatch
from gepa.core.engine import GEPAEngine
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.proposer.base import CandidateProposal, SubsampleEvaluation
from gepa.proposer.reflective_mutation.admission import AdmissionPlan
from gepa.strategies.acceptance import AcceptanceCriterion
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.strategies.proposal_selection import AllImprovements

import bridge.b20_compass_reflection as compass_reflection
from bridge.b19_reversible_parent_selection import (
    frontier_rate,
    high_resolution_selection_rate,
)
from bridge.b20_compass_reflection import (
    AlwaysAcceptAcceptance,
    CompassReflectionEngineConfig,
    SeedFallbackParetoCandidateSelector,
    SparseMinibatchEvaluationPolicy,
    SparseObservationDspyAdapter,
    build_split_admission_loaders,
    resolve_minibatch_sizes,
    run_compass_reflection_engine,
    select_reference_program_idx,
)


def _state() -> GEPAState:
    state = GEPAState(
        {"prompt": "seed"},
        ValsetEvaluation(
            outputs_by_val_id={0: "seed-0", 1: "seed-1"},
            scores_by_val_id={0: 1.0, 1: 0.0},
            objective_scores_by_val_id=None,
        ),
        frontier_type="instance",
    )
    state.program_candidates.append({"prompt": "child"})
    state.parent_program_for_candidate.append([0])
    state.program_birth_propose_ids.append(())
    state.prog_candidate_val_subscores.append({0: 1.0, 1: 1.0})
    state.prog_candidate_objective_scores.append({})
    state.named_predictor_id_to_update_next_for_program_candidate.append(0)
    state.num_metric_calls_by_discovery.append(0)
    state.pareto_front_valset = {0: 1.0, 1: 1.0}
    state.program_at_pareto_front_valset = {0: {0, 1}, 1: {1}}
    assert state.is_consistent()
    return state


def _adapter() -> SparseObservationDspyAdapter:
    return SparseObservationDspyAdapter(
        student_module=SimpleNamespace(),
        metric_fn=lambda *_args, **_kwargs: 0.0,
        feedback_map={},
        max_candidate_workers=2,
    )


def _engine_config(
    tmp_path: Path,
    **batch_sizes: int | None,
) -> CompassReflectionEngineConfig:
    return CompassReflectionEngineConfig(
        run_dir=tmp_path,
        condition="compass_reflection",
        seed=7,
        parent_top_n=5,
        max_metric_calls=100,
        perfect_score=1.0,
        failure_score=0.0,
        num_threads=1,
        max_candidate_workers=1,
        skip_perfect_score=True,
        add_format_failure_as_feedback=False,
        track_best_outputs=True,
        display_progress_bar=False,
        raise_on_exception=True,
        use_cloudpickle=True,
        **batch_sizes,
    )


def test_split_admission_loaders_share_train_ids_and_offset_validation() -> None:
    train = [object(), object()]
    validation = [object(), object(), object()]

    proposal_loader, admission_loader = build_split_admission_loaders(
        train,
        validation,
    )

    assert proposal_loader.all_ids() == [0, 1]
    assert admission_loader.all_ids() == [0, 1, 2, 3, 4]
    assert proposal_loader.fetch([0, 1])[0] is train[0]
    assert admission_loader.fetch([0, 1])[1] is train[1]
    assert admission_loader.fetch([2, 4]) == [
        validation[0],
        validation[2],
    ]


def test_split_admission_sampler_excludes_recursive_train_provenance() -> None:
    train = [{"origin": "train", "id": idx} for idx in range(5)]
    validation = [{"origin": "validation", "id": idx} for idx in range(3)]
    _, admission_loader = build_split_admission_loaders(train, validation)
    state = GEPAState(
        {"prompt": "seed"},
        ValsetEvaluation(
            outputs_by_val_id={},
            scores_by_val_id={},
            objective_scores_by_val_id=None,
        ),
        frontier_type="instance",
    )
    child_one = state.update_state_with_new_program(
        parent_program_idx=[0],
        new_program={"prompt": "child-1"},
        valset_evaluation=ValsetEvaluation(
            outputs_by_val_id={4: "child-1"},
            scores_by_val_id={4: 1.0},
            objective_scores_by_val_id=None,
        ),
        run_dir=None,
        num_metric_calls_by_discovery_of_new_program=0,
        birth_propose_ids=(0,),
    )
    child_two = state.update_state_with_new_program(
        parent_program_idx=[child_one],
        new_program={"prompt": "child-2"},
        valset_evaluation=ValsetEvaluation(
            outputs_by_val_id={4: "child-2"},
            scores_by_val_id={4: 1.0},
            objective_scores_by_val_id=None,
        ),
        run_dir=None,
        num_metric_calls_by_discovery_of_new_program=0,
        birth_propose_ids=(1,),
    )
    excluded = state.get_prospective_frontier_ineligible_ids(
        child_two,
        (2,),
    )
    sampler = EpochShuffledBatchSampler(
        minibatch_size=5,
        rng=random.Random(11),
    )

    sampled_batches = [
        sampler.next_minibatch_ids(
            admission_loader,
            state,
            excluded_ids=excluded,
        )
        for _ in range(3)
    ]
    sampled_ids = {
        data_id for batch_ids in sampled_batches for data_id in batch_ids
    }
    admission_batch = admission_loader.fetch(sorted(sampled_ids))

    assert excluded == frozenset({0, 1, 2})
    assert all(
        excluded.isdisjoint(batch_ids)
        for batch_ids in sampled_batches
    )
    assert sampled_ids == {3, 4, 5, 6, 7}
    assert sum(item["origin"] == "train" for item in admission_batch) == 2
    assert sum(item["origin"] == "validation" for item in admission_batch) == 3


def test_minibatch_config_resolves_legacy_and_split_modes(
    tmp_path: Path,
) -> None:
    assert resolve_minibatch_sizes(
        _engine_config(tmp_path, reflection_minibatch_size=3)
    ) == (3, 3, False)
    assert resolve_minibatch_sizes(
        _engine_config(
            tmp_path,
            proposal_minibatch_size=2,
            admission_minibatch_size=5,
        )
    ) == (2, 5, True)

    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_minibatch_sizes(
            _engine_config(
                tmp_path,
                reflection_minibatch_size=3,
                proposal_minibatch_size=2,
                admission_minibatch_size=5,
            )
        )


def test_split_engine_uses_owner_admission_loader_sampler_and_rng(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = [object(), object(), object()]
    validation = [object(), object()]
    adapter = object()
    captured: dict[str, Any] = {}
    program = SimpleNamespace(
        named_predictors=lambda: [
            (
                "prompt",
                SimpleNamespace(
                    signature=SimpleNamespace(instructions="seed")
                ),
            )
        ]
    )

    monkeypatch.setattr(
        compass_reflection,
        "SparseObservationDspyAdapter",
        lambda **_kwargs: adapter,
    )

    def optimize_stub(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(compass_reflection, "optimize", optimize_stub)

    run = run_compass_reflection_engine(
        program=program,
        metric_fn=lambda *_args, **_kwargs: 0.0,
        feedback_map={},
        trainset=train,
        validation_set=validation,
        reflection_lm=object(),
        config=_engine_config(
            tmp_path,
            proposal_minibatch_size=2,
            admission_minibatch_size=4,
        ),
    )

    proposal_loader = captured["trainset"]
    admission_loader = captured["admission_set"]
    assert run.adapter is adapter
    assert captured["valset"] is admission_loader
    assert proposal_loader.all_ids() == [0, 1, 2]
    assert admission_loader.all_ids() == [0, 1, 2, 3, 4]
    assert proposal_loader.fetch([1])[0] is train[1]
    assert admission_loader.fetch([1])[0] is train[1]
    assert admission_loader.fetch([3])[0] is validation[0]
    assert captured["batch_sampler"].minibatch_size == 2
    assert captured["admission_batch_sampler"].minibatch_size == 4
    assert (
        captured["batch_sampler"].rng
        is not captured["admission_batch_sampler"].rng
    )
    assert (
        captured["admission_hook"].rng
        is not captured["admission_batch_sampler"].rng
    )
    assert captured["admission_hook"].rng is not captured["batch_sampler"].rng


def test_legacy_engine_keeps_train_only_shared_sampler_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = [object(), object(), object()]
    validation = [object()]
    captured: dict[str, Any] = {}
    program = SimpleNamespace(
        named_predictors=lambda: [
            (
                "prompt",
                SimpleNamespace(
                    signature=SimpleNamespace(instructions="seed")
                ),
            )
        ]
    )
    monkeypatch.setattr(
        compass_reflection,
        "SparseObservationDspyAdapter",
        lambda **_kwargs: object(),
    )

    def optimize_stub(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(compass_reflection, "optimize", optimize_stub)

    run_compass_reflection_engine(
        program=program,
        metric_fn=lambda *_args, **_kwargs: 0.0,
        feedback_map={},
        trainset=train,
        validation_set=validation,
        reflection_lm=object(),
        config=_engine_config(tmp_path, reflection_minibatch_size=3),
    )

    assert captured["trainset"] is train
    assert captured["valset"] is train
    assert captured["admission_set"] is None
    assert captured["admission_batch_sampler"] is None
    assert captured["admission_hook"].rng is captured["batch_sampler"].rng


def test_split_admission_rejects_non_compass_condition(tmp_path: Path) -> None:
    config = replace(
        _engine_config(
            tmp_path,
            proposal_minibatch_size=2,
            admission_minibatch_size=4,
        ),
        condition="mini_admission_reflection",
    )
    program = SimpleNamespace(
        named_predictors=lambda: [
            (
                "prompt",
                SimpleNamespace(
                    signature=SimpleNamespace(instructions="seed")
                ),
            )
        ]
    )

    with pytest.raises(ValueError, match="recursive proposal-lineage exclusion"):
        run_compass_reflection_engine(
            program=program,
            metric_fn=lambda *_args, **_kwargs: 0.0,
            feedback_map={},
            trainset=[object(), object()],
            validation_set=[object()],
            reflection_lm=object(),
            config=config,
        )


def _raw_and_high_resolution_ranking_diverge_state() -> GEPAState:
    fronts = (
        {0, 1, 2, 3, 4, 5},
        {0, 6, 7, 8, 9},
        {0, 10, 11, 12, 13},
        {1},
    )
    state = GEPAState(
        {"prompt": "seed"},
        ValsetEvaluation(
            outputs_by_val_id={idx: "" for idx in range(len(fronts))},
            scores_by_val_id={idx: 0.0 for idx in range(len(fronts))},
            objective_scores_by_val_id=None,
        ),
        frontier_type="instance",
    )
    candidate_count = 14
    state.program_candidates = [
        {"prompt": f"skill-{idx}"}
        for idx in range(candidate_count)
    ]
    state.parent_program_for_candidate = [
        [None],
        *([[0]] * (candidate_count - 1)),
    ]
    state.program_birth_propose_ids = [() for _ in range(candidate_count)]
    state.prog_candidate_val_subscores = [
        {data_id: 0.0 for data_id in range(len(fronts))}
        for _ in range(candidate_count)
    ]
    state.prog_candidate_objective_scores = [{} for _ in range(candidate_count)]
    state.named_predictor_id_to_update_next_for_program_candidate = [
        0 for _ in range(candidate_count)
    ]
    state.num_metric_calls_by_discovery = [0 for _ in range(candidate_count)]
    state.pareto_front_valset = {
        data_id: 0.0
        for data_id in range(len(fronts))
    }
    state.program_at_pareto_front_valset = {
        data_id: set(front)
        for data_id, front in enumerate(fronts)
    }
    assert state.is_consistent()
    return state


def test_always_accept_uses_official_criterion_seam_for_worse_proposal() -> None:
    criterion = AlwaysAcceptAcceptance()
    proposal = SimpleNamespace(
        subsample_scores_before=[1.0, 1.0],
        subsample_scores_after=[0.0, 0.0],
    )

    assert isinstance(criterion, AcceptanceCriterion)
    assert criterion.should_accept(proposal, _state()) is True
    assert AllImprovements().select([proposal], _state(), criterion) == [proposal]


def test_default_acceptance_mode_still_rejects_worse_proposal() -> None:
    proposal = SimpleNamespace(
        subsample_scores_before=[1.0],
        subsample_scores_after=[0.0],
    )

    assert (
        compass_reflection._acceptance_criterion(
            "strict_improvement",
        ).should_accept(proposal, _state())
        is False
    )
    assert (
        compass_reflection.CompassReflectionEngineConfig.__dataclass_fields__[
            "acceptance_mode"
        ].default
        == "strict_improvement"
    )


def test_always_accept_commits_worse_admission_through_gepa_engine() -> None:
    state = GEPAState(
        {"prompt": "seed"},
        ValsetEvaluation(outputs_by_val_id={}, scores_by_val_id={}),
    )
    state.i = 0
    state.full_program_trace.append({"i": 1})
    engine = object.__new__(GEPAEngine)
    engine.acceptance_criterion = AlwaysAcceptAcceptance()
    engine.selection_strategy = AllImprovements()
    engine.logger = MagicMock()
    engine.adapter = MagicMock()
    engine.callbacks = None
    engine.merge_proposer = None
    engine._evaluate_programs_on_valset = MagicMock()
    engine._add_evaluated_program = MagicMock(return_value=(1, 0))
    engine._log_proposal_lm_calls = MagicMock()
    proposal = CandidateProposal(
        candidate={"prompt": "worse-child"},
        parent_program_ids=[0],
        subsample_indices=[11],
        subsample_scores_before=[1.0],
        subsample_scores_after=[0.0],
        eval_before=SubsampleEvaluation(
            scores=[1.0],
            outputs=["old"],
            trajectories=[{"trace": "old"}],
        ),
        eval_after=SubsampleEvaluation(
            scores=[0.0],
            outputs=["new"],
            trajectories=[{"trace": "new"}],
        ),
        admission_plan=AdmissionPlan(
            evaluation_ids=(11,),
            evaluation_batch=({"id": 11},),
            eval_before=EvaluationBatch(
                outputs=["old"],
                scores=[1.0],
                trajectories=[{"trace": "old"}],
            ),
            birth_propose_ids=(3,),
        ),
    )

    assert engine._run_reflective_batch([proposal], state)

    engine._evaluate_programs_on_valset.assert_not_called()
    call = engine._add_evaluated_program.call_args.kwargs
    assert call["birth_propose_ids"] == (3,)
    assert call["valset_evaluation"].scores_by_val_id == {11: 0.0}
    assert call["valset_evaluation"].outputs_by_val_id == {11: "new"}
    engine.adapter.commit_program_observations.assert_called_once()


def test_sparse_observations_only_replace_with_strictly_higher_reward() -> None:
    adapter = _adapter()
    evaluation = EvaluationBatch(
        outputs=["first"],
        scores=[0.5],
        trajectories=[{"example": object()}],
    )
    adapter.commit_program_observations(
        program_idx=3,
        evaluation_ids=(7,),
        evaluation=evaluation,
        committed_ids=(7,),
    )
    adapter.commit_program_observations(
        program_idx=3,
        evaluation_ids=(7,),
        evaluation=EvaluationBatch(
            outputs=["equal"],
            scores=[0.5],
            trajectories=[{"example": object()}],
        ),
        committed_ids=(7,),
    )

    fact = adapter.get_program_observation(3, 7)

    assert fact is not None
    assert fact.score == 0.5
    assert fact.output == "first"


def test_sparse_observation_state_round_trips_without_aliasing() -> None:
    adapter = _adapter()
    adapter.commit_program_observations(
        program_idx=0,
        evaluation_ids=(4,),
        evaluation=EvaluationBatch(
            outputs=["x"],
            scores=[1.0],
            trajectories=[{"example": "e"}],
        ),
        committed_ids=(4,),
    )
    persisted = adapter.get_adapter_state()
    restored = _adapter()
    restored.set_adapter_state(deepcopy(persisted))

    assert restored.get_program_observation(0, 4) == (
        adapter.get_program_observation(0, 4)
    )
    assert restored.get_adapter_state() is not persisted


def test_candidate_batch_concurrency_restores_submission_order() -> None:
    adapter = _adapter()

    def evaluate(
        batch: list[Any],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        assert capture_traces
        return EvaluationBatch(
            outputs=[candidate["prompt"]],
            scores=[float(batch[0])],
            trajectories=[{"example": batch[0]}],
        )

    adapter.evaluate = evaluate  # type: ignore[method-assign]
    results = adapter.batch_evaluate(
        [
            ({"prompt": "a"}, [1]),
            ({"prompt": "b"}, [2]),
            ({"prompt": "c"}, [3]),
        ]
    )

    assert [result.outputs for result in results] == [["a"], ["b"], ["c"]]
    assert [result.scores for result in results] == [[1.0], [2.0], [3.0]]


def test_candidate_batch_preserves_failed_slot_without_resubmission() -> None:
    adapter = _adapter()
    calls: list[str] = []

    def evaluate(
        batch: list[Any],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        assert capture_traces
        calls.append(candidate["prompt"])
        if candidate["prompt"] == "b":
            raise TimeoutError("candidate timed out")
        return EvaluationBatch(
            outputs=[candidate["prompt"]],
            scores=[float(batch[0])],
            trajectories=[{"example": batch[0]}],
        )

    adapter.evaluate = evaluate  # type: ignore[method-assign]
    results = adapter.batch_evaluate(
        [
            ({"prompt": "a"}, [1]),
            ({"prompt": "b"}, [2]),
            ({"prompt": "c"}, [3]),
        ]
    )

    assert [result.outputs if result is not None else None for result in results] == [
        ["a"],
        None,
        ["c"],
    ]
    assert sorted(calls) == ["a", "b", "c"]


def test_raw_feedback_reflection_is_concurrent_and_restores_task_order() -> None:
    adapter = object.__new__(SparseObservationDspyAdapter)
    adapter.max_reflection_workers = 3
    barrier = threading.Barrier(3)

    def propose_new_texts(
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        assert reflective_dataset["prompt"]
        assert components_to_update == ["prompt"]
        barrier.wait(timeout=5)
        index = int(candidate["prompt"])
        time.sleep((2 - index) * 0.01)
        return {"prompt": f"new-{index}"}

    adapter.propose_new_texts = propose_new_texts  # type: ignore[method-assign]
    results = adapter.propose_new_texts_batch(
        [
            (
                {"prompt": str(index)},
                {"prompt": [{"input": index, "feedback": "revise"}]},
                ["prompt"],
            )
            for index in range(3)
        ]
    )

    assert results == [
        {"prompt": "new-0"},
        {"prompt": "new-1"},
        {"prompt": "new-2"},
    ]


def test_raw_feedback_reflection_preserves_failed_slot_without_resubmission() -> None:
    adapter = object.__new__(SparseObservationDspyAdapter)
    adapter.max_reflection_workers = 3
    calls: list[str] = []

    def propose_new_texts(
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        assert reflective_dataset["prompt"]
        assert components_to_update == ["prompt"]
        calls.append(candidate["prompt"])
        if candidate["prompt"] == "1":
            raise TimeoutError("reflection timed out")
        return {"prompt": f"new-{candidate['prompt']}"}

    adapter.propose_new_texts = propose_new_texts  # type: ignore[method-assign]
    results = adapter.propose_new_texts_batch(
        [
            (
                {"prompt": str(index)},
                {"prompt": [{"input": index, "feedback": "revise"}]},
                ["prompt"],
            )
            for index in range(3)
        ]
    )

    assert results == [
        {"prompt": "new-0"},
        None,
        {"prompt": "new-2"},
    ]
    assert sorted(calls) == ["0", "1", "2"]


def test_raw_feedback_batch_uses_official_dspy_proposer_for_one_child() -> None:
    prompts: list[str] = []

    def reflection_lm(prompt: str) -> list[str]:
        prompts.append(prompt)
        return ["```only-child```"]

    adapter = SparseObservationDspyAdapter(
        student_module=SimpleNamespace(),
        metric_fn=lambda *_args, **_kwargs: 0.0,
        feedback_map={},
        reflection_lm=reflection_lm,
        max_reflection_workers=4,
    )
    result = adapter.propose_new_texts_batch(
        [
            (
                {"prompt": "seed"},
                {
                    "prompt": [
                        {
                            "Inputs": {"question": "q"},
                            "Generated Outputs": {"answer": "a"},
                            "Feedback": "revise",
                        }
                    ]
                },
                ["prompt"],
            )
        ]
    )

    assert result == [{"prompt": "only-child"}]
    assert len(prompts) == 1
    assert "seed" in prompts[0]
    assert "revise" in prompts[0]


def test_sparse_adapter_fail_fast_propagates_infrastructure_errors() -> None:
    class ExplodingProgram(dspy.Module):
        def forward(self, value: int) -> dspy.Prediction:
            raise RuntimeError(f"infrastructure failure for {value}")

    adapter = SparseObservationDspyAdapter(
        student_module=ExplodingProgram(),
        metric_fn=lambda *_args, **_kwargs: 0.0,
        feedback_map={},
        failure_score=0.0,
        num_threads=2,
        raise_on_error=True,
        max_candidate_workers=1,
    )
    batch = [dspy.Example(value=1).with_inputs("value")]

    with pytest.raises(Exception, match="cancelled due to errors"):
        adapter.evaluate(batch, {}, capture_traces=True)


def test_failed_missing_reference_skips_only_affected_proposal_task(
    tmp_path: Path,
) -> None:
    state = GEPAState(
        {"prompt": "seed"},
        ValsetEvaluation(
            outputs_by_val_id={},
            scores_by_val_id={},
            objective_scores_by_val_id=None,
        ),
        frontier_type="instance",
    )
    state.total_num_evals = 0
    adapter = MagicMock()
    adapter.get_program_observation.return_value = None
    adapter.batch_evaluate.return_value = [None]
    logger = MagicMock()
    hook = compass_reflection.MiniAdmissionHook(
        adapter=adapter,
        rng=random.Random(0),
        run_dir=tmp_path,
        logger=logger,
    )

    with pytest.raises(RuntimeError, match="admission reference evaluation failed"):
        hook._evaluate_missing_reference_groups(
            state=state,
            ids=(3,),
            batch=(object(),),
            reference_program_indices=(0,),
        )

    logger.log.assert_called_once()
    assert "skipping only the affected proposal task" in logger.log.call_args.args[0]
    assert state.total_num_evals == 0


def test_reference_selection_uses_frontier_rate_then_exposure() -> None:
    state = _state()

    selected = select_reference_program_idx(
        state,
        instance_id=0,
        sampled_parent_idx=0,
        rng=random.Random(0),
    )

    assert selected == 1


def test_sparse_final_selection_uses_rate_exposure_then_earliest() -> None:
    state = _state()
    policy = SparseMinibatchEvaluationPolicy()

    assert policy.get_best_program(state) == 1


def test_admission_reference_and_final_selection_remain_on_raw_frontier_rate() -> None:
    state = _raw_and_high_resolution_ranking_diverge_state()

    assert frontier_rate(state, 0) == pytest.approx(3 / 4)
    assert frontier_rate(state, 1) == pytest.approx(1 / 2)
    assert high_resolution_selection_rate(state, 0) == Fraction(17, 120)
    assert high_resolution_selection_rate(state, 1) == Fraction(7, 24)
    assert high_resolution_selection_rate(
        state,
        1,
    ) > high_resolution_selection_rate(state, 0)

    assert select_reference_program_idx(
        state,
        instance_id=0,
        sampled_parent_idx=1,
        rng=random.Random(0),
    ) == 0
    assert SparseMinibatchEvaluationPolicy().get_best_program(state) == 0


def test_official_pareto_selector_falls_back_only_for_empty_seed_frontier() -> None:
    state = GEPAState(
        {"prompt": "seed"},
        ValsetEvaluation(
            outputs_by_val_id={},
            scores_by_val_id={},
            objective_scores_by_val_id=None,
        ),
        frontier_type="instance",
    )
    selector = SeedFallbackParetoCandidateSelector(random.Random(0))

    assert selector.select_candidate_idx(state) == 0

    state.commit_existing_program_evaluation(
        program_idx=0,
        valset_evaluation=ValsetEvaluation(
            outputs_by_val_id={5: "x"},
            scores_by_val_id={5: 1.0},
            objective_scores_by_val_id=None,
        ),
        run_dir=None,
        iteration=0,
    )
    assert selector.select_candidate_idx(state) == 0
