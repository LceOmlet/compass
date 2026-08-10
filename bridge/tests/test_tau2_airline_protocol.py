from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bridge import tau2_airline_protocol as subject


def fake_tasks(ids):
    return tuple(SimpleNamespace(id=task_id) for task_id in ids)


def test_official_trial_seeds_are_frozen_from_pinned_tau2_runner():
    assert subject.TAU2_OPTIMIZATION_TRIAL_SEED == 626729
    assert subject.TAU2_TEST_TRIAL_SEEDS == (
        626729,
        373753,
        361454,
        1567,
    )


def test_frozen_source_rejects_dirty_owner_worktree(monkeypatch, tmp_path):
    monkeypatch.setattr(subject, "_git_head", lambda root: subject.TAU2_AIRLINE_COMMIT)
    monkeypatch.setattr(
        subject,
        "_git_worktree_status",
        lambda root: " M src/tau2/runner/batch.py",
    )
    with pytest.raises(RuntimeError, match="clean index and worktree"):
        subject.verify_frozen_tau2_airline_source(tmp_path)


def test_optimization_view_requires_explicit_complete_partition():
    train = fake_tasks(subject.TAU2_AIRLINE_TRAIN_IDS)
    proposal = subject.TAU2_AIRLINE_TRAIN_IDS[:-6]
    validation = subject.TAU2_AIRLINE_TRAIN_IDS[-6:]
    view = subject.freeze_optimization_view(
        train,
        proposal_task_ids=proposal,
        validation_task_ids=validation,
    )
    assert tuple(str(task.id) for task in view.proposal) == proposal
    assert tuple(str(task.id) for task in view.validation) == validation

    with pytest.raises(ValueError, match="partition all 30"):
        subject.freeze_optimization_view(
            train,
            proposal_task_ids=proposal[:-1],
            validation_task_ids=validation,
        )
    with pytest.raises(ValueError, match="disjoint"):
        subject.freeze_optimization_view(
            train,
            proposal_task_ids=proposal,
            validation_task_ids=(proposal[-1], *validation),
        )
    with pytest.raises(ValueError, match="owner order"):
        subject.freeze_optimization_view(
            train,
            proposal_task_ids=tuple(reversed(proposal)),
            validation_task_ids=validation,
        )


def test_paper_optimization_view_uses_every_fifth_owner_position():
    train = fake_tasks(subject.TAU2_AIRLINE_TRAIN_IDS)
    view = subject.frozen_tau2_airline_optimization_view(train)
    assert subject.TAU2_AIRLINE_VALIDATION_IDS == (
        "5",
        "12",
        "21",
        "34",
        "41",
        "49",
    )
    assert tuple(str(task.id) for task in view.validation) == (
        "5",
        "12",
        "21",
        "34",
        "41",
        "49",
    )
    assert len(view.proposal) == 24
    assert len(view.validation) == 6
    assert {task.id for task in view.proposal}.isdisjoint(
        task.id for task in view.validation
    )


def test_runtime_config_uses_official_text_protocol_and_models():
    config = subject.build_tau2_airline_text_config(
        api_base="https://example.test/v1",
        task_split_name="train",
        num_trials=1,
    )
    assert config.domain == "airline"
    assert config.agent == "llm_agent"
    assert config.user == "user_simulator"
    assert config.llm_agent == "openai/gpt-4.1-mini-2025-04-14"
    assert config.llm_user == "openai/gpt-4.1-2025-04-14"
    assert config.llm_args_agent == {
        "temperature": 0.0,
        "api_base": "https://example.test/v1",
    }
    assert config.llm_args_user == {
        "temperature": 0.0,
        "api_base": "https://example.test/v1",
    }
    assert "api_key" not in config.model_dump_json()
    assert config.max_steps == 200
    assert config.max_errors == 10
    assert config.timeout is None
    assert config.max_concurrency == 5
    assert config.seed == 300


def test_native_gepa_uses_generic_adapter_seam(monkeypatch, tmp_path):
    view = subject.Tau2AirlineOptimizationView(
        proposal=fake_tasks(subject.TAU2_AIRLINE_TRAIN_IDS[:-6]),
        validation=fake_tasks(subject.TAU2_AIRLINE_TRAIN_IDS[-6:]),
    )
    adapter = object()
    monkeypatch.setattr(subject, "Tau2GEPAAdapter", lambda *args, **kwargs: adapter)
    captured = {}
    result = SimpleNamespace(
        best_idx=1,
        candidates=[
            {"agent_instruction": "seed"},
            {"agent_instruction": "best"},
        ],
    )

    def fake_optimize(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(subject, "optimize", fake_optimize)
    run_config = subject.build_tau2_airline_text_config(
        api_base="https://example.test/v1",
        task_split_name="train",
        num_trials=1,
    )
    run = subject.run_tau2_airline_optimization(
        method="gepa",
        view=view,
        run_config=run_config,
        reflection_lm="reflection",
        run_dir=tmp_path,
        settings=subject.Tau2AirlineOptimizationSettings(
            parent_selection_score_mode="raw"
        ),
    )
    assert captured["adapter"] is adapter
    assert captured["task_lm"] is None
    assert captured["evaluator"] is None
    assert captured["reflection_lm"] == "reflection"
    assert captured["max_metric_calls"] == 600
    assert captured["reflection_minibatch_size"] == 3
    assert len(captured["trainset"]) == 24
    assert len(captured["valset"]) == 6
    assert {example.seed for example in captured["trainset"]} == {626729}
    assert run.selected_candidate_idx == 1
    assert run.selected_candidate == {"agent_instruction": "best"}


def test_preflight_uses_frozen_96_rollout_budget(monkeypatch, tmp_path):
    view = subject.Tau2AirlineOptimizationView(
        proposal=fake_tasks(subject.TAU2_AIRLINE_PROPOSAL_IDS),
        validation=fake_tasks(subject.TAU2_AIRLINE_VALIDATION_IDS),
    )
    monkeypatch.setattr(subject, "Tau2GEPAAdapter", lambda *args, **kwargs: object())
    captured = {}
    result = SimpleNamespace(
        best_idx=0,
        candidates=[{"agent_instruction": "seed"}],
    )

    def fake_optimize(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(subject, "optimize", fake_optimize)
    run_config = subject.build_tau2_airline_text_config(
        api_base="https://example.test/v1",
        task_split_name="train",
        num_trials=1,
    )
    subject.run_tau2_airline_optimization(
        method="gepa",
        view=view,
        run_config=run_config,
        reflection_lm="reflection",
        run_dir=tmp_path,
        settings=subject.Tau2AirlineOptimizationSettings(
            parent_selection_score_mode="raw_frontier_rate",
            phase="preflight",
            max_metric_calls=subject.TAU2_PREFLIGHT_ROLLOUT_BUDGET,
        ),
    )
    assert captured["max_metric_calls"] == 96


@pytest.mark.parametrize(
    ("method", "condition"),
    [("m0", "mini_admission_reflection"), ("compass", "compass_reflection")],
)
def test_compass_family_uses_same_adapter_engine(
    monkeypatch,
    tmp_path,
    method,
    condition,
):
    view = subject.Tau2AirlineOptimizationView(
        proposal=fake_tasks(subject.TAU2_AIRLINE_TRAIN_IDS[:-6]),
        validation=fake_tasks(subject.TAU2_AIRLINE_TRAIN_IDS[-6:]),
    )
    adapter = object()
    monkeypatch.setattr(subject, "Tau2GEPAAdapter", lambda *args, **kwargs: adapter)
    captured = {}
    result = SimpleNamespace(
        candidates=[
            {"agent_instruction": "seed"},
            {"agent_instruction": "selected"},
        ]
    )
    evaluation_policy = SimpleNamespace(get_best_program=Mock(return_value=1))

    def fake_engine(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            result=result,
            evaluation_policy=evaluation_policy,
        )

    monkeypatch.setattr(subject, "run_compass_gepa_adapter_engine", fake_engine)
    monkeypatch.setattr(subject.GEPAState, "load", lambda path: "state")
    run_config = subject.build_tau2_airline_text_config(
        api_base="https://example.test/v1",
        task_split_name="train",
        num_trials=1,
    )
    run = subject.run_tau2_airline_optimization(
        method=method,
        view=view,
        run_config=run_config,
        reflection_lm="reflection",
        run_dir=tmp_path,
        settings=subject.Tau2AirlineOptimizationSettings(
            parent_selection_score_mode="high_resolution"
        ),
    )
    assert captured["adapter"] is adapter
    assert captured["config"].condition == condition
    assert captured["config"].proposal_minibatch_size == 3
    assert captured["config"].admission_minibatch_size == 3
    assert captured["config"].max_metric_calls == 600
    assert captured["config"].proposal_sampling_mode == "independent"
    assert captured["config"].epoch_parallel_enabled is False
    assert run.selected_candidate_idx == 1
    assert run.selected_candidate == {"agent_instruction": "selected"}
    evaluation_policy.get_best_program.assert_called_once_with("state")


def test_tau2_compass_can_explicitly_bind_joint_linucb_without_changing_defaults(
    monkeypatch,
    tmp_path,
):
    view = subject.Tau2AirlineOptimizationView(
        proposal=fake_tasks(subject.TAU2_AIRLINE_TRAIN_IDS[:-6]),
        validation=fake_tasks(subject.TAU2_AIRLINE_TRAIN_IDS[-6:]),
    )
    monkeypatch.setattr(subject, "Tau2GEPAAdapter", lambda *args, **kwargs: object())
    captured = {}

    def fake_engine(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            result=SimpleNamespace(candidates=[{"agent_instruction": "seed"}]),
            evaluation_policy=SimpleNamespace(get_best_program=Mock(return_value=0)),
        )

    monkeypatch.setattr(subject, "run_compass_gepa_adapter_engine", fake_engine)
    monkeypatch.setattr(subject.GEPAState, "load", lambda path: "state")
    run_config = subject.build_tau2_airline_text_config(
        api_base="https://example.test/v1",
        task_split_name="train",
        num_trials=1,
    )

    subject.run_tau2_airline_optimization(
        method="compass",
        view=view,
        run_config=run_config,
        reflection_lm="reflection",
        run_dir=tmp_path,
        settings=subject.Tau2AirlineOptimizationSettings(
            parent_selection_score_mode="high_resolution_lexicographic",
            proposal_sampling_mode="joint_linucb",
            epoch_parallel_enabled=True,
        ),
    )

    assert captured["config"].proposal_sampling_mode == "joint_linucb"
    assert captured["config"].epoch_parallel_enabled is True


def test_mipro_is_not_shadow_adapted(tmp_path):
    with pytest.raises(ValueError, match="MIPROv2 has no generic GEPAAdapter"):
        subject.run_tau2_airline_optimization(
            method="mipro",  # type: ignore[arg-type]
            view=subject.Tau2AirlineOptimizationView((), ()),
            run_config=SimpleNamespace(),
            reflection_lm=object(),
            run_dir=tmp_path,
            settings=subject.Tau2AirlineOptimizationSettings(
                parent_selection_score_mode="raw"
            ),
        )


def test_optimizer_refuses_unverified_nonempty_resume(tmp_path):
    (tmp_path / "gepa_state.bin").write_bytes(b"existing")
    config = subject.build_tau2_airline_text_config(
        api_base="https://example.test/v1",
        task_split_name="train",
        num_trials=1,
    )
    view = subject.Tau2AirlineOptimizationView(
        proposal=fake_tasks(subject.TAU2_AIRLINE_TRAIN_IDS[:-6]),
        validation=fake_tasks(subject.TAU2_AIRLINE_TRAIN_IDS[-6:]),
    )
    with pytest.raises(FileExistsError, match="strict manifest"):
        subject.run_tau2_airline_optimization(
            method="gepa",
            view=view,
            run_config=config,
            reflection_lm=object(),
            run_dir=tmp_path,
            settings=subject.Tau2AirlineOptimizationSettings(
                parent_selection_score_mode="raw"
            ),
        )


def test_final_evaluation_delegates_to_owner_and_validates_identities(
    monkeypatch,
    tmp_path,
):
    test_tasks = fake_tasks(subject.TAU2_AIRLINE_TEST_IDS)
    simulations = [
        SimpleNamespace(
            task_id=task_id,
            trial=trial,
            seed=seed,
            termination_reason=subject.TerminationReason.USER_STOP,
            reward_info=object(),
        )
        for trial, seed in enumerate(subject.TAU2_TEST_TRIAL_SEEDS)
        for task_id in reversed(subject.TAU2_AIRLINE_TEST_IDS)
    ]
    results = SimpleNamespace(simulations=simulations)
    metrics = object()
    captured = {}
    monkeypatch.setattr(
        subject,
        "register_tau2_fixed_candidate_agent",
        lambda candidate: "fixed-agent",
    )

    def fake_run_tasks(config, tasks, **kwargs):
        captured["config"] = config
        captured["tasks"] = tasks
        captured.update(kwargs)
        return results

    monkeypatch.setattr(subject, "run_tasks", fake_run_tasks)
    monkeypatch.setattr(subject, "compute_metrics", lambda owner_results: metrics)
    config = subject.build_tau2_airline_text_config(
        api_base="https://example.test/v1",
        task_split_name="train",
        num_trials=1,
    )
    evaluation = subject.run_tau2_airline_final_evaluation(
        candidate={"agent_instruction": "frozen"},
        test_tasks=test_tasks,
        run_config=config,
        save_path=tmp_path / "results.json",
        save_dir=tmp_path,
        console_display=False,
    )
    assert captured["config"].agent == "fixed-agent"
    assert captured["config"].num_trials == 4
    assert captured["evaluation_type"] is subject.EvaluationType.ALL
    assert captured["results_format"] == "json"
    assert [str(task.id) for task in captured["tasks"]] == list(
        subject.TAU2_AIRLINE_TEST_IDS
    )
    assert evaluation.results is results
    assert evaluation.metrics is metrics


def test_final_evaluation_resume_requires_external_verification_and_owner_resume(
    tmp_path,
):
    save_path = tmp_path / "results.json"
    save_path.write_text("{}", encoding="utf-8")
    config = subject.build_tau2_airline_text_config(
        api_base="https://example.test/v1",
        task_split_name="train",
        num_trials=1,
    )
    kwargs = {
        "candidate": {"agent_instruction": "frozen"},
        "test_tasks": fake_tasks(subject.TAU2_AIRLINE_TEST_IDS),
        "run_config": config,
        "save_path": save_path,
        "save_dir": tmp_path,
        "console_display": False,
    }
    with pytest.raises(FileExistsError, match="strict manifest"):
        subject.run_tau2_airline_final_evaluation(**kwargs)
    with pytest.raises(ValueError, match="auto_resume=True"):
        subject.run_tau2_airline_final_evaluation(
            **kwargs,
            verified_resume=True,
        )


def test_final_resume_checkpoint_requires_exact_owner_info_and_tasks(
    monkeypatch,
    tmp_path,
):
    expected_info = SimpleNamespace(digest="same")
    previous = SimpleNamespace(
        info=SimpleNamespace(digest="same"),
        tasks=[SimpleNamespace(id="2", digest="task")],
    )
    monkeypatch.setattr(subject.Results, "load", lambda path: previous)
    monkeypatch.setattr(subject, "get_info", lambda config: expected_info)
    monkeypatch.setattr(
        subject,
        "get_pydantic_hash",
        lambda value, exclude=None: value.digest,
    )
    official_test = [SimpleNamespace(id="2", digest="task")]

    subject._validate_final_resume_checkpoint(
        tmp_path / "results.json",
        final_config=object(),
        official_test=official_test,
    )

    previous.info.digest = "different"
    with pytest.raises(RuntimeError, match="run config changed"):
        subject._validate_final_resume_checkpoint(
            tmp_path / "results.json",
            final_config=object(),
            official_test=official_test,
        )
    previous.info.digest = "same"
    previous.tasks[0].digest = "different"
    with pytest.raises(RuntimeError, match="tasks changed"):
        subject._validate_final_resume_checkpoint(
            tmp_path / "results.json",
            final_config=object(),
            official_test=official_test,
        )


def test_final_evaluation_rejects_unresolved_infrastructure_failure():
    identities = [
        SimpleNamespace(
            task_id=task_id,
            trial=trial,
            seed=seed,
            termination_reason=(
                subject.TerminationReason.INFRASTRUCTURE_ERROR
                if task_id == "2" and trial == 0
                else subject.TerminationReason.USER_STOP
            ),
            reward_info=object(),
        )
        for trial, seed in enumerate(subject.TAU2_TEST_TRIAL_SEEDS)
        for task_id in subject.TAU2_AIRLINE_TEST_IDS
    ]
    with pytest.raises(RuntimeError, match="unresolved infrastructure failures"):
        subject._validate_final_results(SimpleNamespace(simulations=identities))
