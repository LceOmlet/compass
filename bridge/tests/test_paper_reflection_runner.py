from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.paper.run_compass_reflection import (
    _method_config_kwargs,
    _minibatch_config_kwargs,
    _require_aime_gepa_protocol,
    _require_frozen_protocol,
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
        "max_reflection_workers": 5,
        "parent_selection_score_mode": "raw_frontier_rate",
        "proposal_tasks_per_iteration": 5,
    }


def test_aime_gepa_protocol_rejects_decoding_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["task_id"] = "aime_2025"
    config["optimizer"]["max_metric_calls"] = 1839
    config["model"]["temperature"] = 0.7

    with pytest.raises(ValueError, match="model.temperature"):
        _require_aime_gepa_protocol(config)

    config["model"]["temperature"] = 0.6
    _require_aime_gepa_protocol(config)
