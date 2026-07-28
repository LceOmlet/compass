from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.paper.run_compass_reflection import (
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
