from __future__ import annotations

import json
from pathlib import Path

import pytest

from bridge.request_deadline import DeadlineAwareLM

from experiments.paper.run_compass_reflection import (
    _create_lm,
    _method_config_kwargs,
    _minibatch_config_kwargs,
    _require_aime_gepa_protocol,
    _require_frozen_protocol,
    _require_resume_identity,
    load_run_config,
)


def _config(tmp_path: Path) -> dict:
    return {
        "cache_dir": str(tmp_path / "cache"),
        "condition": "compass_reflection",
        "dataset_mode": "lite",
        "model": {
            "api_base": "https://example.invalid/v1",
            "api_key_env": "TEST_API_KEY",
            "cache": True,
            "cache_in_memory": True,
            "enable_thinking": True,
            "max_tokens": 16384,
            "model": "openai/test",
            "model_type": "chat",
            "num_retries": 0,
            "serving_max_model_len": 40960,
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
        },
        "optimizer": {
            "add_format_failure_as_feedback": False,
            "display_progress_bar": False,
            "evaluation_straggler_timeout": 0,
            "failure_score": 0,
            "max_candidate_workers": 3,
            "max_metric_calls": 3593,
            "num_threads": 32,
            "parent_top_n": 5,
            "perfect_score": 1,
            "raise_on_exception": True,
            "reflection_minibatch_size": 3,
            "skip_perfect_score": True,
            "track_best_outputs": True,
            "use_cloudpickle": True,
        },
        "optimizer_seed": 0,
        "run_dir": str(tmp_path / "run"),
        "source_snapshot": {"commit": "abc"},
        "task_id": "ifbench",
    }


def test_runner_config_accepts_only_the_frozen_protocol(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config(tmp_path)), encoding="utf-8")

    config = load_run_config(path)
    _require_frozen_protocol(config, budget=3593)

    assert config["condition"] == "compass_reflection"


def test_resume_identity_binds_config_and_dataset() -> None:
    task = {
        "task_id": "hitab",
        "split_sizes": {"train": 150, "validation": 300, "test": 1584},
        "split_fingerprints": {"train": "a", "validation": "b", "test": "c"},
    }
    manifest = {"config_sha256": "frozen", "task": task}

    _require_resume_identity(
        manifest,
        config_sha256="frozen",
        task_manifest=task,
    )
    with pytest.raises(RuntimeError, match="manifest/config"):
        _require_resume_identity(
            manifest,
            config_sha256="different",
            task_manifest=task,
        )
    with pytest.raises(RuntimeError, match="dataset identity"):
        _require_resume_identity(
            manifest,
            config_sha256="frozen",
            task_manifest={**task, "split_fingerprints": {"train": "changed"}},
        )


def test_runner_config_accepts_profile_owned_output_cap(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["model"]["max_tokens"] = 32768
    config["model"]["serving_max_model_len"] = 131072
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_run_config(path)
    _require_frozen_protocol(loaded, budget=3593)
    lm = _create_lm(loaded["model"], api_key="test-key")

    assert lm.kwargs["max_tokens"] == 32768
    assert isinstance(lm, DeadlineAwareLM)


def test_runner_forwards_whole_operation_timeouts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["optimizer"].update(
        {
            "rollout_timeout_seconds": 600,
            "proposal_timeout_seconds": 600,
        }
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_run_config(path)

    assert _method_config_kwargs(loaded["optimizer"])[
        "rollout_timeout_seconds"
    ] == 600
    assert _method_config_kwargs(loaded["optimizer"])[
        "proposal_timeout_seconds"
    ] == 600


@pytest.mark.parametrize("value", [0, -1, True, float("inf")])
def test_runner_rejects_invalid_whole_operation_timeout(
    tmp_path: Path,
    value: object,
) -> None:
    config = _config(tmp_path)
    config["optimizer"]["rollout_timeout_seconds"] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(TypeError, match="rollout_timeout_seconds"):
        load_run_config(path)


@pytest.mark.parametrize("max_tokens", [0, -1, True, 1.5])
def test_runner_config_rejects_invalid_output_cap(
    tmp_path: Path,
    max_tokens: object,
) -> None:
    config = _config(tmp_path)
    config["model"]["max_tokens"] = max_tokens
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(TypeError, match="model.max_tokens"):
        load_run_config(path)


def test_runner_config_rejects_budget_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["optimizer"]["max_metric_calls"] = 100
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_run_config(path)
    with pytest.raises(ValueError, match="max_metric_calls"):
        _require_frozen_protocol(loaded, budget=3593)


def test_runner_config_rejects_deployment_context_equal_to_output_cap(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["model"]["serving_max_model_len"] = 16384
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_run_config(path)
    with pytest.raises(ValueError, match="must exceed model.max_tokens"):
        _require_frozen_protocol(loaded, budget=3593)


def test_runner_config_accepts_explicit_split_minibatches(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    del config["optimizer"]["reflection_minibatch_size"]
    config["optimizer"]["proposal_minibatch_size"] = 2
    config["optimizer"]["admission_minibatch_size"] = 5
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_run_config(path)
    _require_frozen_protocol(loaded, budget=3593)

    assert _minibatch_config_kwargs(loaded["optimizer"]) == {
        "reflection_minibatch_size": None,
        "proposal_minibatch_size": 2,
        "admission_minibatch_size": 5,
    }


def test_runner_config_rejects_mixed_legacy_and_split_minibatches(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["optimizer"]["proposal_minibatch_size"] = 2
    config["optimizer"]["admission_minibatch_size"] = 5
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be combined"):
        load_run_config(path)


def test_runner_config_rejects_split_batches_for_mini_admission(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["condition"] = "mini_admission_reflection"
    del config["optimizer"]["reflection_minibatch_size"]
    config["optimizer"]["proposal_minibatch_size"] = 2
    config["optimizer"]["admission_minibatch_size"] = 5
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="requires condition='compass_reflection'"):
        load_run_config(path)


def test_runner_forwards_existing_compass_method_controls(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["optimizer"].update(
        {
            "acceptance_mode": "always_accept",
            "epoch_parallel_enabled": True,
            "max_reflection_workers": 5,
            "parent_selection_score_mode": "raw_frontier_rate",
            "proposal_tasks_per_iteration": 5,
        }
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_run_config(path)

    assert _method_config_kwargs(loaded["optimizer"]) == {
        "acceptance_mode": "always_accept",
        "epoch_parallel_enabled": True,
        "evaluation_straggler_timeout": 0,
        "max_reflection_workers": 5,
        "parent_selection_score_mode": "raw_frontier_rate",
        "proposal_tasks_per_iteration": 5,
    }


def test_runner_forwards_optional_official_candidate_proposal_budget(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["optimizer"]["max_candidate_proposals"] = 128
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_run_config(path)

    assert loaded["optimizer"]["max_candidate_proposals"] == 128
    assert _method_config_kwargs(loaded["optimizer"])[
        "max_candidate_proposals"
    ] == 128


def test_runner_rejects_candidate_proposal_budget_with_epoch_parallel(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["optimizer"].update(
        {
            "epoch_parallel_enabled": True,
            "max_candidate_proposals": 128,
        }
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="epoch_parallel_enabled=false"):
        load_run_config(path)


def test_runner_builds_concrete_pinned_dci_boundary_config(tmp_path: Path) -> None:
    config = _config(tmp_path)
    del config["optimizer"]["reflection_minibatch_size"]
    config["optimizer"].update(
        {
            "proposal_minibatch_size": 1,
            "admission_minibatch_size": 5,
            "proposal_timeout_seconds": 1200,
            "dci": {
                "agent_dir": str(tmp_path / "pi" / ".pi" / "agent"),
                "max_turns": 20,
                "model": "gpt-5.4-nano",
                "package_dir": str(tmp_path / "pi" / "packages" / "coding-agent"),
                "proposal_evidence_size": 3,
                "provider": "openai",
                "runner_command": ["uv", "run", "dci-agent-lite"],
                "tools": "read,bash",
            },
        }
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_run_config(path)
    dci = _method_config_kwargs(loaded["optimizer"])["dci_config"]

    assert dci.proposal_evidence_size == 3
    assert dci.agent.runner_command == ("uv", "run", "dci-agent-lite")
    assert dci.agent.max_turns == 20
    assert dci.agent.run_timeout_seconds == 1200
    assert dci.agent.system_prompt_file.name == "dci_subproblem_free_text.txt"


def test_aime_gepa_protocol_rejects_decoding_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["task_id"] = "aime_2025"
    config["optimizer"]["max_metric_calls"] = 1839
    config["model"]["temperature"] = 0.7

    with pytest.raises(ValueError, match="model.temperature"):
        _require_aime_gepa_protocol(config)

    config["model"]["temperature"] = 0.6
    _require_aime_gepa_protocol(config)

    config["model"]["max_tokens"] = 32768
    with pytest.raises(ValueError, match="model.max_tokens"):
        _require_aime_gepa_protocol(config)


def test_runner_forwards_owner_request_timeout_to_dspy(tmp_path: Path) -> None:
    model = _config(tmp_path)["model"]
    model["timeout"] = 600
    lm = _create_lm(model, api_key="test-key")

    assert lm.kwargs["timeout"] == 600


def test_runner_forwards_explicit_response_schema_capability(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["model"]["supports_response_schema"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_run_config(path)
    lm = _create_lm(loaded["model"], api_key="test-key")

    assert lm.supports_response_schema is True
    assert "supports_response_schema" not in lm.kwargs


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_runner_rejects_invalid_response_schema_capability(
    tmp_path: Path,
    value: object,
) -> None:
    config = _config(tmp_path)
    config["model"]["supports_response_schema"] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(TypeError, match="supports_response_schema"):
        load_run_config(path)


def test_runner_forwards_owner_repetition_detection_without_reimplementation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    owner_config = {
        "max_pattern_size": 20,
        "min_pattern_size": 3,
        "min_count": 4,
    }
    config["model"]["repetition_detection"] = owner_config
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_run_config(path)
    lm = _create_lm(loaded["model"], api_key="test-key")

    assert lm.kwargs["extra_body"]["repetition_detection"] == owner_config


def test_runner_rejects_nonpositive_request_timeout(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["model"]["timeout"] = 0
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(TypeError, match="model.timeout"):
        load_run_config(path)
