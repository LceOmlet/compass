from __future__ import annotations

from pathlib import Path

import pytest

from experiments.paper.generate_reflection_configs import (
    TASK_BUDGETS,
    build_run_config,
)


def _profile() -> dict:
    return {
        "api_base": "http://127.0.0.1:18080/v1",
        "api_key_env": "LOCAL_OPENAI_API_KEY",
        "cache": True,
        "cache_in_memory": True,
        "enable_thinking": True,
        "max_tokens": 16384,
        "model": "openai/Qwen/Qwen3-8B",
        "model_type": "chat",
        "num_retries": 0,
        "serving_max_model_len": 40960,
        "temperature": 0.6,
    }


def test_generated_config_preserves_frozen_batch_and_budget() -> None:
    slug, config = build_run_config(
        task_id="hover",
        condition="compass_reflection",
        seed=2,
        tag="20260728_v1",
        model_profile_name="qwen3_8b_local_vllm",
        model_profile=_profile(),
        remote_root=Path("/shared/reflection-bridge"),
        snapshot={"root_head": "abc"},
    )

    assert slug.endswith("hover_seed2_20260728_v1")
    assert config["optimizer"]["max_metric_calls"] == TASK_BUDGETS["hover"]
    assert config["optimizer"]["reflection_minibatch_size"] == 3
    assert config["optimizer"]["max_candidate_workers"] == 3
    assert config["optimizer_seed"] == 2
    assert config["run_dir"].endswith(slug)
    assert config["cache_dir"].endswith(f"cache_{slug}")


def test_generated_config_rejects_unsafe_tag() -> None:
    with pytest.raises(ValueError, match="tag"):
        build_run_config(
            task_id="ifbench",
            condition="compass_reflection",
            seed=0,
            tag="../overwrite",
            model_profile_name="qwen3_8b_local_vllm",
            model_profile=_profile(),
            remote_root=Path("/shared/reflection-bridge"),
            snapshot={},
        )
