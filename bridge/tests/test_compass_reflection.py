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
from dspy.utils.dummies import DummyLM
from gepa.core.adapter import EvaluationBatch
from gepa.core.engine import GEPAEngine
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.proposer.base import CandidateProposal, SubsampleEvaluation
from gepa.proposer.reflective_mutation.admission import AdmissionPlan
from gepa.strategies.acceptance import AcceptanceCriterion
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.strategies.proposal_selection import AllImprovements
from gepa.utils import MaxCandidateProposalsStopper
from gepa_artifact.benchmarks.IFBench.ifbench_program import (
    IFBenchCoT2StageProgram,
)

import bridge.b20_compass_reflection as compass_reflection
import bridge.request_deadline as request_deadline_module
from bridge import dci_compass
from bridge.b19_reversible_parent_selection import (
    frontier_rate,
    high_resolution_selection_rate,
    select_top_candidate_idx,
)
from bridge.b20_compass_reflection import (
    AlwaysAcceptAcceptance,
    CompassReflectionEngineConfig,
    SeedFallbackParetoCandidateSelector,
    SparseMinibatchEvaluationPolicy,
    SparseObservationDspyAdapter,
    SparseObservationTrackingAdapter,
    build_split_admission_loaders,
    resolve_minibatch_sizes,
    run_compass_gepa_adapter_engine,
    run_compass_reflection_engine,
    select_reference_program_idx,
)
from bridge.dci_agent_lite import DciAgentLiteConfig
from bridge.dci_compass import DciCompassConfig
from bridge.request_deadline import remaining_request_seconds


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
    sampled_ids = {data_id for batch_ids in sampled_batches for data_id in batch_ids}
    admission_batch = admission_loader.fetch(sorted(sampled_ids))

    assert excluded == frozenset({0, 1, 2})
    assert all(excluded.isdisjoint(batch_ids) for batch_ids in sampled_batches)
    assert sampled_ids == {3, 4, 5, 6, 7}
    assert sum(item["origin"] == "train" for item in admission_batch) == 2
    assert sum(item["origin"] == "validation" for item in admission_batch) == 3


def test_mini_admission_sampler_excludes_only_current_propose_ids() -> None:
    train = [object() for _ in range(4)]
    validation = [object() for _ in range(2)]
    _, admission_loader = build_split_admission_loaders(train, validation)
    state = GEPAState(
        {"prompt": "seed"},
        ValsetEvaluation(outputs_by_val_id={}, scores_by_val_id={}),
        frontier_type="instance",
    )
    child = state.update_state_with_new_program(
        parent_program_idx=[0],
        new_program={"prompt": "child"},
        valset_evaluation=ValsetEvaluation(
            outputs_by_val_id={},
            scores_by_val_id={},
        ),
        run_dir=None,
        num_metric_calls_by_discovery_of_new_program=0,
        birth_propose_ids=(0,),
    )
    hook = object.__new__(compass_reflection.MiniAdmissionHook)
    excluded = hook.frontier_ineligible_ids(
        state=state,
        parent_program_idx=child,
        propose_ids=(2,),
    )
    sampler = EpochShuffledBatchSampler(
        minibatch_size=5,
        rng=random.Random(13),
    )

    sampled = sampler.next_minibatch_ids(
        admission_loader,
        state,
        excluded_ids=excluded,
    )

    assert excluded == (2,)
    assert set(sampled) == {0, 1, 3, 4, 5}


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


def test_epoch_sampling_can_be_windowed_to_five_minibatches() -> None:
    strategy = compass_reflection.proposal_sampling_strategy(
        trainset_size=150,
        minibatch_size=3,
        epoch_parallel_enabled=True,
        proposal_tasks_per_iteration=5,
    )

    assert isinstance(strategy, compass_reflection.IndependentSampling)
    assert strategy.n == 5


def test_joint_linucb_sampling_is_explicit_and_uses_the_same_epoch_window() -> None:
    assert (
        CompassReflectionEngineConfig.__dataclass_fields__[
            "proposal_sampling_mode"
        ].default
        == "independent"
    )

    strategy = compass_reflection.proposal_sampling_strategy(
        trainset_size=150,
        minibatch_size=3,
        epoch_parallel_enabled=True,
        proposal_tasks_per_iteration=5,
        proposal_sampling_mode="joint_linucb",
        parent_top_n=7,
        perfect_score=1.0,
    )

    assert isinstance(strategy, compass_reflection.JointLinUCBSamplingStrategy)
    assert strategy.minibatches_per_wave == 5
    assert strategy.top_n == 7
    assert strategy.perfect_score == 1.0


def test_joint_linucb_rejects_an_implicit_single_task_or_scalar_parent_mode(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="epoch_parallel_enabled=true"):
        compass_reflection.proposal_sampling_strategy(
            trainset_size=6,
            minibatch_size=3,
            epoch_parallel_enabled=False,
            proposal_sampling_mode="joint_linucb",
            parent_top_n=5,
            perfect_score=1.0,
        )

    config = replace(
        _engine_config(
            tmp_path,
            proposal_minibatch_size=3,
            admission_minibatch_size=3,
        ),
        proposal_sampling_mode="joint_linucb",
        epoch_parallel_enabled=True,
        parent_selection_score_mode="high_resolution",
    )
    with pytest.raises(ValueError, match="high_resolution_lexicographic"):
        compass_reflection._prepare_compass_engine(
            trainset=[object() for _ in range(6)],
            validation_set=[object() for _ in range(3)],
            config=config,
        )


def test_split_engine_uses_owner_admission_loader_sampler_and_rng(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = [object(), object(), object()]
    validation = [object(), object()]
    adapter = object()
    adapter_kwargs: dict[str, Any] = {}
    captured: dict[str, Any] = {}
    instruction_proposer = object()
    program = SimpleNamespace(
        named_predictors=lambda: [
            (
                "prompt",
                SimpleNamespace(signature=SimpleNamespace(instructions="seed")),
            )
        ]
    )

    def make_adapter(**kwargs: Any) -> object:
        adapter_kwargs.update(kwargs)
        return adapter

    monkeypatch.setattr(
        compass_reflection,
        "SparseObservationDspyAdapter",
        make_adapter,
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
        custom_instruction_proposer=instruction_proposer,
        config=replace(
            _engine_config(
                tmp_path,
                proposal_minibatch_size=2,
                admission_minibatch_size=4,
            ),
            evaluation_straggler_timeout=0,
        ),
    )

    proposal_loader = captured["trainset"]
    admission_loader = captured["admission_set"]
    assert run.adapter is adapter
    assert adapter_kwargs["evaluation_timeout"] == 0
    assert adapter_kwargs["custom_instruction_proposer"] is instruction_proposer
    assert captured["valset"] is admission_loader
    assert proposal_loader.all_ids() == [0, 1, 2]
    assert admission_loader.all_ids() == [0, 1, 2, 3, 4]
    assert proposal_loader.fetch([1])[0] is train[1]
    assert admission_loader.fetch([1])[0] is train[1]
    assert admission_loader.fetch([3])[0] is validation[0]
    assert captured["batch_sampler"].minibatch_size == 2
    assert captured["admission_batch_sampler"].minibatch_size == 4
    assert captured["batch_sampler"].rng is not captured["admission_batch_sampler"].rng
    assert captured["admission_hook"].rng is not captured["admission_batch_sampler"].rng
    assert captured["admission_hook"].rng is not captured["batch_sampler"].rng
    assert captured["stop_callbacks"] is None


def test_candidate_proposal_budget_uses_official_stopper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    program = SimpleNamespace(
        named_predictors=lambda: [
            (
                "prompt",
                SimpleNamespace(signature=SimpleNamespace(instructions="seed")),
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
        trainset=[object(), object(), object()],
        reflection_lm=object(),
        config=replace(
            _engine_config(tmp_path, reflection_minibatch_size=1),
            max_candidate_proposals=128,
        ),
    )

    stopper = captured["stop_callbacks"]
    assert isinstance(stopper, MaxCandidateProposalsStopper)
    assert stopper.max_proposals == 128
    assert isinstance(
        captured["sampling_strategy"],
        compass_reflection.SingleMutationSampling,
    )


def test_candidate_proposal_budget_is_default_inert_and_rejects_epoch_parallel(
    tmp_path: Path,
) -> None:
    assert (
        CompassReflectionEngineConfig.__dataclass_fields__[
            "max_candidate_proposals"
        ].default
        is None
    )
    config = replace(
        _engine_config(tmp_path, reflection_minibatch_size=1),
        max_candidate_proposals=128,
        epoch_parallel_enabled=True,
    )
    program = SimpleNamespace(
        named_predictors=lambda: [
            (
                "prompt",
                SimpleNamespace(signature=SimpleNamespace(instructions="seed")),
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="official stopper counts optimizer iterations",
    ):
        run_compass_reflection_engine(
            program=program,
            metric_fn=lambda *_args, **_kwargs: 0.0,
            feedback_map={},
            trainset=[object()],
            reflection_lm=object(),
            config=config,
        )


def test_split_engine_uses_five_minibatch_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = [object() for _ in range(30)]
    validation = [object() for _ in range(10)]
    captured: dict[str, Any] = {}
    program = SimpleNamespace(
        named_predictors=lambda: [
            (
                "prompt",
                SimpleNamespace(signature=SimpleNamespace(instructions="seed")),
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
        config=replace(
            _engine_config(
                tmp_path,
                proposal_minibatch_size=3,
                admission_minibatch_size=3,
            ),
            epoch_parallel_enabled=True,
            proposal_tasks_per_iteration=5,
        ),
    )

    strategy = captured["sampling_strategy"]
    proposal_sampler = captured["batch_sampler"]
    admission_sampler = captured["admission_batch_sampler"]
    assert isinstance(strategy, compass_reflection.IndependentSampling)
    assert strategy.n == 5
    assert proposal_sampler.iteration_is_epoch is False
    assert proposal_sampler.minibatches_per_iteration == 5
    assert admission_sampler.iteration_is_epoch is True


def test_split_engine_wires_joint_linucb_as_the_only_proposal_scheduler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    program = SimpleNamespace(
        named_predictors=lambda: [
            (
                "prompt",
                SimpleNamespace(signature=SimpleNamespace(instructions="seed")),
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
        trainset=[object() for _ in range(30)],
        validation_set=[object() for _ in range(10)],
        reflection_lm=object(),
        config=replace(
            _engine_config(
                tmp_path,
                proposal_minibatch_size=3,
                admission_minibatch_size=3,
            ),
            parent_selection_score_mode="high_resolution_lexicographic",
            proposal_sampling_mode="joint_linucb",
            epoch_parallel_enabled=True,
            proposal_tasks_per_iteration=5,
        ),
    )

    strategy = captured["sampling_strategy"]
    sampler = captured["batch_sampler"]
    assert isinstance(strategy, compass_reflection.JointLinUCBSamplingStrategy)
    assert strategy.minibatches_per_wave == 5
    assert sampler.minibatches_per_iteration == 5
    assert sampler.iteration_is_epoch is False


def test_dci_engine_wires_official_windowed_epoch_source_and_admission_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = [object() for _ in range(30)]
    validation = [object() for _ in range(10)]
    adapter = object()
    admission_hook = object()
    adapter_kwargs: dict[str, Any] = {}
    hook_kwargs: dict[str, Any] = {}
    captured: dict[str, Any] = {}
    program = SimpleNamespace(
        named_predictors=lambda: [
            (
                "prompt",
                SimpleNamespace(signature=SimpleNamespace(instructions="seed")),
            )
        ]
    )

    def make_adapter(**kwargs: Any) -> object:
        adapter_kwargs.update(kwargs)
        return adapter

    def make_admission_hook(**kwargs: Any) -> object:
        hook_kwargs.update(kwargs)
        return admission_hook

    monkeypatch.setattr(
        dci_compass,
        "DciSparseObservationDspyAdapter",
        make_adapter,
    )
    monkeypatch.setattr(
        dci_compass,
        "DciAdmissionHook",
        make_admission_hook,
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
        config=replace(
            _engine_config(
                tmp_path,
                proposal_minibatch_size=1,
                admission_minibatch_size=3,
            ),
            epoch_parallel_enabled=True,
            proposal_tasks_per_iteration=5,
            dci_config=DciCompassConfig(
                agent=DciAgentLiteConfig(
                    runner_command=("dci-agent-lite",),
                    package_dir=tmp_path / "package",
                    agent_dir=tmp_path / "agent",
                    provider="openai",
                    model="model",
                    system_prompt_file=tmp_path / "system.txt",
                ),
            ),
        ),
    )

    proposal_sampler = captured["batch_sampler"]
    assert isinstance(proposal_sampler, EpochShuffledBatchSampler)
    assert adapter_kwargs["proposal_batch_sampler"] is proposal_sampler
    assert proposal_sampler.minibatch_size == 1
    assert proposal_sampler.iteration_is_epoch is False
    assert proposal_sampler.minibatches_per_iteration == 5
    assert hook_kwargs["admission_size"] == 3
    assert captured["admission_hook"] is admission_hook


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
                SimpleNamespace(signature=SimpleNamespace(instructions="seed")),
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


def test_split_mini_admission_uses_pareto_without_recursive_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
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
                SimpleNamespace(signature=SimpleNamespace(instructions="seed")),
            )
        ]
    )
    monkeypatch.setattr(
        compass_reflection,
        "SparseObservationDspyAdapter",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        compass_reflection,
        "optimize",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    run_compass_reflection_engine(
        program=program,
        metric_fn=lambda *_args, **_kwargs: 0.0,
        feedback_map={},
        trainset=[object(), object()],
        validation_set=[object()],
        reflection_lm=object(),
        config=config,
    )

    assert isinstance(captured["admission_hook"], compass_reflection.MiniAdmissionHook)
    assert not isinstance(
        captured["admission_hook"],
        compass_reflection.CleanMiniAdmissionHook,
    )
    assert isinstance(
        captured["candidate_selection_strategy"],
        SeedFallbackParetoCandidateSelector,
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
        {"prompt": f"skill-{idx}"} for idx in range(candidate_count)
    ]
    state.parent_program_for_candidate = [
        [None],
        *([[0]] * (candidate_count - 1)),
    ]
    state.program_birth_propose_ids = [() for _ in range(candidate_count)]
    state.prog_candidate_val_subscores = [
        {data_id: 0.0 for data_id in range(len(fronts))} for _ in range(candidate_count)
    ]
    state.prog_candidate_objective_scores = [{} for _ in range(candidate_count)]
    state.named_predictor_id_to_update_next_for_program_candidate = [
        0 for _ in range(candidate_count)
    ]
    state.num_metric_calls_by_discovery = [0 for _ in range(candidate_count)]
    state.pareto_front_valset = {data_id: 0.0 for data_id in range(len(fronts))}
    state.program_at_pareto_front_valset = {
        data_id: set(front) for data_id, front in enumerate(fronts)
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


def test_generic_tracking_adapter_restores_opaque_owner_trace_once() -> None:
    example = object()
    owner_trace = object()
    owner = SimpleNamespace(
        evaluate=MagicMock(
            return_value=EvaluationBatch(
                outputs=[object()],
                scores=[0.5],
                trajectories=[owner_trace],
                num_metric_calls=1,
            )
        ),
        make_reflective_dataset=MagicMock(return_value={"prompt": []}),
    )
    adapter = SparseObservationTrackingAdapter(owner)

    evaluation = adapter.evaluate(
        [example],
        {"prompt": "seed"},
        capture_traces=True,
    )
    reflective = adapter.make_reflective_dataset(
        {"prompt": "seed"},
        evaluation,
        ["prompt"],
    )

    owner.evaluate.assert_called_once()
    owner.make_reflective_dataset.assert_called_once()
    restored = owner.make_reflective_dataset.call_args.args[1]
    assert restored.trajectories[0] is owner_trace
    assert reflective == {"prompt": []}
    with pytest.raises(RuntimeError, match="overlap"):
        compass_reflection.MiniAdmissionHook._validate_disjoint_instances(
            evaluation,
            [example],
        )


def test_generic_tracking_adapter_uses_official_batch_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = SimpleNamespace(
        evaluate=MagicMock(
            side_effect=lambda batch, _candidate, capture_traces: EvaluationBatch(
                outputs=list(batch),
                scores=[1.0] * len(batch),
                trajectories=[object() for _ in batch] if capture_traces else None,
            )
        ),
        make_reflective_dataset=MagicMock(),
    )
    official_fallback = MagicMock(wraps=compass_reflection.default_batch_evaluate)
    monkeypatch.setattr(
        compass_reflection,
        "default_batch_evaluate",
        official_fallback,
    )
    adapter = SparseObservationTrackingAdapter(owner)

    evaluations = adapter.batch_evaluate(
        [({"prompt": "a"}, [1]), ({"prompt": "b"}, [2])]
    )

    official_fallback.assert_called_once()
    assert owner.evaluate.call_count == 2
    assert [evaluation.outputs for evaluation in evaluations] == [[1], [2]]


def test_generic_tracking_adapter_calls_owner_batch_once() -> None:
    owner_traces = [object(), object()]
    items = [({"prompt": "a"}, [1]), ({"prompt": "b"}, [2])]
    owner = SimpleNamespace(
        evaluate=MagicMock(side_effect=AssertionError("must not call evaluate")),
        batch_evaluate=MagicMock(
            return_value=[
                EvaluationBatch(
                    outputs=[1],
                    scores=[1.0],
                    trajectories=[owner_traces[0]],
                ),
                EvaluationBatch(
                    outputs=[2],
                    scores=[0.0],
                    trajectories=[owner_traces[1]],
                ),
            ]
        ),
        make_reflective_dataset=MagicMock(return_value={"prompt": []}),
    )
    adapter = SparseObservationTrackingAdapter(owner)

    evaluations = adapter.batch_evaluate(items)
    adapter.make_reflective_dataset(
        items[0][0],
        evaluations[0],
        ["prompt"],
    )

    owner.batch_evaluate.assert_called_once_with(items)
    owner.evaluate.assert_not_called()
    restored = owner.make_reflective_dataset.call_args.args[1]
    assert restored.trajectories[0] is owner_traces[0]


def test_generic_tracking_adapter_namespaces_owner_state() -> None:
    owner_state = {"nested": {"values": [1]}}
    owner = SimpleNamespace(
        get_adapter_state=MagicMock(return_value=owner_state),
        set_adapter_state=MagicMock(),
    )
    adapter = SparseObservationTrackingAdapter(owner)

    persisted = adapter.get_adapter_state()
    owner_state["nested"]["values"].append(2)
    adapter.set_adapter_state(deepcopy(persisted))

    assert persisted["owner_adapter"] == {"nested": {"values": [1]}}
    owner.set_adapter_state.assert_called_once_with({"nested": {"values": [1]}})


def test_generic_engine_passes_owner_adapter_to_shared_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    owner = SimpleNamespace()
    reflection_lm = object()
    proposer = MagicMock()
    monkeypatch.setattr(
        compass_reflection,
        "optimize",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    run = run_compass_gepa_adapter_engine(
        seed_candidate={"prompt": "seed"},
        adapter=owner,
        trainset=[object(), object()],
        validation_set=None,
        reflection_lm=reflection_lm,
        custom_candidate_proposer=proposer,
        config=_engine_config(tmp_path, reflection_minibatch_size=1),
    )

    assert isinstance(run.adapter, SparseObservationTrackingAdapter)
    assert run.adapter.delegate is owner
    assert captured["adapter"] is run.adapter
    assert captured["reflection_lm"] is reflection_lm
    assert captured["custom_candidate_proposer"] is proposer


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


def test_built_program_preserves_official_predictors_and_shares_one_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"value": 100.0}
    observed: list[float | None] = []

    monkeypatch.setattr(
        request_deadline_module,
        "monotonic",
        lambda: now["value"],
    )

    class DeadlineProbeLM(DummyLM):
        def forward(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> Any:
            observed.append(remaining_request_seconds())
            now["value"] += 7.0
            return super().forward(prompt=prompt, messages=messages, **kwargs)

    student = IFBenchCoT2StageProgram()
    candidate = {
        name: predictor.signature.instructions
        for name, predictor in student.named_predictors()
    }

    def feedback(**_kwargs: Any) -> dict[str, Any]:
        return {"score": 0.0, "feedback": "revise"}

    adapter = SparseObservationDspyAdapter(
        student_module=student,
        metric_fn=lambda _example, _prediction, _trace=None: 0.0,
        feedback_map={name: feedback for name in candidate},
        num_threads=1,
        rollout_timeout_seconds=600,
    )

    program = adapter.build_program(candidate)
    assert [name for name, _ in program.named_predictors()] == list(candidate)

    lm = DeadlineProbeLM(
        [
            {"reasoning": "draft reasoning", "response": "draft"},
            {"reasoning": "final reasoning", "final_response": "final"},
        ]
    )
    batch = [dspy.Example(prompt="follow the constraints").with_inputs("prompt")]
    with dspy.context(lm=lm):
        evaluation = adapter.evaluate(batch, candidate, capture_traces=True)

    assert observed == pytest.approx([600.0, 593.0])
    assert remaining_request_seconds() is None
    reflective_dataset = adapter.make_reflective_dataset(
        candidate,
        evaluation,
        list(candidate),
    )
    assert list(reflective_dataset) == list(candidate)


def test_built_program_without_rollout_deadline_uses_official_program_unchanged() -> (
    None
):
    student = IFBenchCoT2StageProgram()
    candidate = {
        name: predictor.signature.instructions
        for name, predictor in student.named_predictors()
    }
    adapter = object.__new__(SparseObservationDspyAdapter)
    adapter.student = student
    adapter.rollout_timeout_seconds = None

    program = adapter.build_program(candidate)

    assert "forward" not in program.__dict__
    assert [name for name, _ in program.named_predictors()] == list(candidate)


def test_proposal_deadline_discards_only_the_expired_parallel_job() -> None:
    adapter = object.__new__(SparseObservationDspyAdapter)
    adapter.max_reflection_workers = 2
    adapter.proposal_timeout_seconds = 0.02

    def propose_new_texts(
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        assert reflective_dataset["prompt"]
        assert components_to_update == ["prompt"]
        if candidate["prompt"] == "slow":
            time.sleep(0.03)
        assert remaining_request_seconds() is not None
        return {"prompt": f"new-{candidate['prompt']}"}

    adapter.propose_new_texts = propose_new_texts  # type: ignore[method-assign]
    results = adapter.propose_new_texts_batch(
        [
            (
                {"prompt": name},
                {"prompt": [{"feedback": "revise"}]},
                ["prompt"],
            )
            for name in ("slow", "fast")
        ]
    )

    assert results == [None, {"prompt": "new-fast"}]


def test_expected_lm_timeout_uses_failure_score_without_cancelling_sibling() -> None:
    class TimeoutProgram(dspy.Module):
        def forward(self, value: str) -> dspy.Prediction:
            if value == "timeout":
                raise dspy.LMTimeoutError("expected deadline")
            return dspy.Prediction(value=value)

    adapter = SparseObservationDspyAdapter(
        student_module=TimeoutProgram(),
        metric_fn=lambda _example, _prediction, _trace=None: 1.0,
        feedback_map={},
        failure_score=0.0,
        num_threads=2,
        raise_on_error=True,
        nonfatal_evaluation_exceptions=(dspy.LMTimeoutError,),
    )
    batch = [
        dspy.Example(value=value).with_inputs("value")
        for value in ("timeout", "success")
    ]

    evaluation = adapter.evaluate(batch, {}, capture_traces=True)

    assert evaluation.scores == [0.0, 1.0]
    assert len(evaluation.outputs) == 2
    assert len(evaluation.trajectories or ()) == 2


def test_non_timeout_evaluation_error_remains_fail_fast() -> None:
    class BrokenProgram(dspy.Module):
        def forward(self, value: str) -> dspy.Prediction:
            del value
            raise ValueError("implementation bug")

    adapter = SparseObservationDspyAdapter(
        student_module=BrokenProgram(),
        metric_fn=lambda _example, _prediction, _trace=None: 1.0,
        feedback_map={},
        failure_score=0.0,
        num_threads=1,
        raise_on_error=True,
        nonfatal_evaluation_exceptions=(dspy.LMTimeoutError,),
    )
    batch = [dspy.Example(value="broken").with_inputs("value")]

    with pytest.raises(Exception, match="cancelled"):
        adapter.evaluate(batch, {}, capture_traces=True)


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


def test_admission_reference_and_sparse_policy_remain_on_raw_frontier_rate() -> None:
    state = _raw_and_high_resolution_ranking_diverge_state()

    assert frontier_rate(state, 0) == pytest.approx(3 / 4)
    assert frontier_rate(state, 1) == pytest.approx(1 / 2)
    assert high_resolution_selection_rate(state, 0) == Fraction(17, 120)
    assert high_resolution_selection_rate(state, 1) == Fraction(7, 24)
    assert high_resolution_selection_rate(
        state,
        1,
    ) > high_resolution_selection_rate(state, 0)

    assert (
        select_reference_program_idx(
            state,
            instance_id=0,
            sampled_parent_idx=1,
            rng=random.Random(0),
        )
        == 0
    )
    assert SparseMinibatchEvaluationPolicy().get_best_program(state) == 0
    assert (
        select_top_candidate_idx(
            state,
            score_mode="raw_frontier_rate",
        )
        == 0
    )
    assert (
        select_top_candidate_idx(
            state,
            score_mode="high_resolution",
        )
        == 1
    )


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
