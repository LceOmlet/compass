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


def test_generated_config_can_enable_split_admission_budget() -> None:
    _, config = build_run_config(
        task_id="ifbench",
        condition="compass_reflection",
        seed=0,
        tag="20260731_split",
        model_profile_name="qwen3_8b_local_vllm",
        model_profile=_profile(),
        remote_root=Path("/shared/reflection-bridge"),
        snapshot={"root_head": "abc"},
        proposal_minibatch_size=2,
        admission_minibatch_size=5,
    )

    optimizer = config["optimizer"]
    assert "reflection_minibatch_size" not in optimizer
    assert optimizer["proposal_minibatch_size"] == 2
    assert optimizer["admission_minibatch_size"] == 5


def test_generated_config_requires_both_split_batch_sizes() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        build_run_config(
            task_id="ifbench",
            condition="compass_reflection",
            seed=0,
            tag="20260731_split",
            model_profile_name="qwen3_8b_local_vllm",
            model_profile=_profile(),
            remote_root=Path("/shared/reflection-bridge"),
            snapshot={},
            proposal_minibatch_size=2,
        )


def test_generated_split_admission_requires_compass_condition() -> None:
    with pytest.raises(ValueError, match="requires condition='compass_reflection'"):
        build_run_config(
            task_id="ifbench",
            condition="mini_admission_reflection",
            seed=0,
            tag="20260731_split",
            model_profile_name="qwen3_8b_local_vllm",
            model_profile=_profile(),
            remote_root=Path("/shared/reflection-bridge"),
            snapshot={},
            proposal_minibatch_size=2,
            admission_minibatch_size=5,
        )
