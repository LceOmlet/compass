from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from bridge import tau2_airline_protocol as protocol
from experiments.paper import run_tau2_airline as subject


def fake_tasks(ids):
    return tuple(SimpleNamespace(id=task_id) for task_id in ids)


def fake_splits():
    return protocol.Tau2AirlineTaskSplits(
        train=fake_tasks(protocol.TAU2_AIRLINE_TRAIN_IDS),
        test=fake_tasks(protocol.TAU2_AIRLINE_TEST_IDS),
    )


def args(tmp_path, *, method="seed", phase="preflight", resume=False):
    budget = subject._budget_for(method, phase)
    return Namespace(
        cache_dir=str(tmp_path / "cache"),
        logical_rollout_budget=budget,
        matrix_id="tau2_test_v1",
        method=method,
        phase=phase,
        run_dir=str(tmp_path / "run"),
        tau2_root=str(tmp_path / "tau2"),
        api_base=subject.PRIMARY_API_BASE,
        api_key_env="TEST_TAU2_API_KEY",
        resume=resume,
        task_id="tau2_airline",
        quiet=True,
    )


def test_identity_freezes_partition_budget_and_never_serializes_secret(
    monkeypatch,
    tmp_path,
):
    secret = "credential-value-must-never-be-persisted"
    monkeypatch.setenv("TEST_TAU2_API_KEY", secret)
    identity = subject._build_run_identity(
        method="compass",
        phase="formal",
        tau2_root=tmp_path,
        api_base=subject.PRIMARY_API_BASE,
        api_key_env="TEST_TAU2_API_KEY",
    )
    serialized = json.dumps(identity)
    assert secret not in serialized
    assert identity["task"]["validation_ids"] == [
        "5",
        "12",
        "21",
        "34",
        "41",
        "49",
    ]
    assert len(identity["task"]["proposal_ids"]) == 24
    assert identity["optimizer"]["logical_rollout_budget"] == 600
    assert identity["optimizer"]["parent_selection_score_mode"] == (
        "high_resolution_lexicographic"
    )


def test_preflight_uses_official_96_soft_threshold_and_does_not_touch_test(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("TEST_TAU2_API_KEY", "not-persisted")
    monkeypatch.setattr(
        subject, "load_frozen_tau2_airline_splits", lambda root: fake_splits()
    )
    monkeypatch.setattr(subject, "build_tau2_reflection_lm", lambda **kwargs: object())
    captured = {}

    def fake_optimize(**kwargs):
        captured.update(kwargs)
        usage = subject.summarize_tau2_resource_usage([])
        usage["logical_episodes"] = 101
        usage["execution_failures"] = 101
        kwargs["resource_usage_callback"](usage)
        return SimpleNamespace(
            selected_candidate_idx=2,
            selected_candidate={"agent_instruction": "pilot candidate"},
            result=SimpleNamespace(
                candidates=[{}, {}, {}],
                total_metric_calls=101,
                num_full_val_evals=0,
            ),
        )

    monkeypatch.setattr(subject, "run_tau2_airline_optimization", fake_optimize)
    monkeypatch.setattr(
        subject,
        "run_tau2_airline_final_evaluation",
        lambda **kwargs: pytest.fail("preflight must not touch official test"),
    )
    result = subject._run(args(tmp_path, method="gepa", phase="preflight"))
    assert captured["settings"].phase == "preflight"
    assert captured["settings"].max_metric_calls == 96
    assert len(captured["view"].proposal) == 24
    assert len(captured["view"].validation) == 6
    assert result["status"] == "completed"
    assert result["evaluation"]["status"] == "not_run"
    assert result["optimization"]["max_metric_calls_threshold"] == 96
    assert result["optimization"]["total_metric_calls"] == 101
    assert result["optimization"]["metric_call_overshoot"] == 5
    assert (
        result["optimization"]["budget_semantics"]
        == "official_soft_iteration_boundary"
    )
    assert (
        result["optimization"]["owner_episode_resource_usage"]["logical_episodes"]
        == 101
    )


class FakeMetrics:
    avg_reward = 0.625

    def model_dump(self, *, mode):
        assert mode == "json"
        return {
            "avg_reward": self.avg_reward,
            "total_simulations": 80,
            "infra_error_count": 0,
        }


def fake_owner_simulations(count=80):
    return [
        SimpleNamespace(
            agent_cost=0.01,
            user_cost=0.02,
            get_messages=list,
        )
        for _ in range(count)
    ]


def test_formal_seed_uses_official_final_evaluator_and_writes_atomic_result(
    monkeypatch,
    tmp_path,
):
    secret = "not-persisted-anywhere"
    monkeypatch.setenv("TEST_TAU2_API_KEY", secret)
    monkeypatch.setattr(
        subject, "load_frozen_tau2_airline_splits", lambda root: fake_splits()
    )
    captured = {}

    def fake_evaluation(**kwargs):
        captured.update(kwargs)
        kwargs["save_path"].write_text('{"official": true}\n', encoding="utf-8")
        return SimpleNamespace(
            results=SimpleNamespace(simulations=fake_owner_simulations()),
            metrics=FakeMetrics(),
        )

    monkeypatch.setattr(subject, "run_tau2_airline_final_evaluation", fake_evaluation)
    run_args = args(tmp_path, method="seed", phase="formal")
    result = subject._run(run_args)
    run_dir = tmp_path / "run"
    assert result["evaluation"]["role"] == "official_seed_evaluation"
    assert result["evaluation"]["num_simulations"] == 80
    assert result["evaluation"]["test_score_percent"] == 62.5
    assert captured["run_config"].auto_resume is False
    assert captured["verified_resume"] is False
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == (
        "completed"
    )
    assert json.loads((run_dir / "final_result.json").read_text())["status"] == (
        "completed"
    )
    for path in run_dir.rglob("*.json"):
        assert secret not in path.read_text(encoding="utf-8")


def test_resume_delegates_existing_evaluation_checkpoint_to_owner_auto_resume(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("TEST_TAU2_API_KEY", "not-persisted")
    monkeypatch.setattr(
        subject, "load_frozen_tau2_airline_splits", lambda root: fake_splits()
    )
    calls = []

    def interrupted_evaluation(**kwargs):
        calls.append(kwargs)
        kwargs["save_path"].write_text('{"checkpoint": true}\n', encoding="utf-8")
        raise RuntimeError("infrastructure failure")

    monkeypatch.setattr(
        subject,
        "run_tau2_airline_final_evaluation",
        interrupted_evaluation,
    )
    with pytest.raises(RuntimeError, match="infrastructure failure"):
        subject._run(args(tmp_path, method="seed", phase="formal"))
    assert (
        json.loads((tmp_path / "run" / "manifest.json").read_text())["exception_type"]
        == "builtins.RuntimeError"
    )

    def resumed_evaluation(**kwargs):
        calls.append(kwargs)
        assert kwargs["run_config"].auto_resume is True
        assert kwargs["verified_resume"] is True
        kwargs["save_path"].write_text('{"completed": true}\n', encoding="utf-8")
        return SimpleNamespace(
            results=SimpleNamespace(simulations=fake_owner_simulations()),
            metrics=FakeMetrics(),
        )

    monkeypatch.setattr(
        subject,
        "run_tau2_airline_final_evaluation",
        resumed_evaluation,
    )
    result = subject._run(args(tmp_path, method="seed", phase="formal", resume=True))
    assert result["status"] == "completed"
    assert len(calls) == 2


def test_run_directory_requires_exact_identity_for_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subject, "load_frozen_tau2_airline_splits", lambda root: fake_splits()
    )
    subject._run(args(tmp_path, method="seed", phase="preflight"))
    with pytest.raises(FileExistsError, match="non-empty"):
        subject._run(args(tmp_path, method="seed", phase="preflight"))
    with pytest.raises(RuntimeError, match="identity changed"):
        subject._run(args(tmp_path, method="compass", phase="preflight", resume=True))


def test_cli_does_not_offer_mipro():
    with pytest.raises(SystemExit):
        subject._parser().parse_args(
            [
                "--method",
                "mipro",
                "--phase",
                "formal",
                "--run-dir",
                "run",
                "--tau2-root",
                "tau2",
            ]
        )


def test_strict_config_entrypoint_matches_direct_arguments(tmp_path):
    config = {
        "schema_version": 1,
        "matrix_id": "tau2_test_v1",
        "task_id": "tau2_airline",
        "method": "gepa",
        "phase": "preflight",
        "logical_rollout_budget": 96,
        "run_dir": str(tmp_path / "run"),
        "cache_dir": str(tmp_path / "cache"),
        "tau2_root": str(tmp_path / "tau2"),
        "api_base": subject.PRIMARY_API_BASE,
        "api_key_env": "TEST_TAU2_API_KEY",
        "quiet": True,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    resolved = subject._resolve_cli_args(
        subject._parser().parse_args(["--config", str(path)])
    )

    assert resolved.method == "gepa"
    assert resolved.phase == "preflight"
    assert resolved.run_dir == str(tmp_path / "run")
    assert resolved.quiet is True
    assert resolved.resume is False


def test_config_entrypoint_rejects_protocol_overrides_and_extra_keys(tmp_path):
    config = {
        "schema_version": 1,
        "matrix_id": "tau2_test_v1",
        "task_id": "tau2_airline",
        "method": "gepa",
        "phase": "preflight",
        "logical_rollout_budget": 96,
        "run_dir": str(tmp_path / "run"),
        "cache_dir": str(tmp_path / "cache"),
        "tau2_root": str(tmp_path / "tau2"),
        "api_base": subject.PRIMARY_API_BASE,
        "api_key_env": "TEST_TAU2_API_KEY",
        "quiet": True,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be combined"):
        subject._resolve_cli_args(
            subject._parser().parse_args(["--config", str(path), "--method", "compass"])
        )

    config["api_key"] = "forbidden"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="keys mismatch"):
        subject.load_tau2_paper_config(path)
