from __future__ import annotations

from pathlib import Path

import pytest

from experiments.paper.generate_reflection_configs import (
    TASK_BUDGETS,
    build_run_config,
    load_model_profiles,
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


def test_qwen35_profile_uses_supported_long_context_without_changing_dci_output() -> None:
    profiles = load_model_profiles(
        Path("experiments/paper/model_profiles.json")
    )
    profile = profiles["qwen3_5_9b_local_vllm_18035"]

    assert profile["max_tokens"] == 32768
    assert profile["serving_max_model_len"] == 131072
    assert profile["serving_backend"] == (
        "vllm-metax-0.17.0+gd10261.d20260409.maca3.5.3.20.torch2.8"
    )
    assert profile["supports_response_schema"] is True
    assert profile["repetition_detection"] == {
        "max_pattern_size": 1024,
        "min_pattern_size": 3,
        "min_count": 4,
    }

    dci_models = Path(
        "experiments/paper/dci_agent_qwen3_5_9b/models.json"
    ).read_text(encoding="utf-8")
    assert '"contextWindow": 131072' in dci_models
    assert '"maxTokens": 4096' in dci_models


def test_gpt_ge_profile_pins_dated_gpt41_mini_without_embedding_a_key() -> None:
    profile = load_model_profiles(
        Path("experiments/paper/model_profiles.json")
    )["gpt_4_1_mini_gpt_ge"]

    assert profile == {
        "api_base": "https://api.gpt.ge/v1",
        "api_key_env": "COMPASS_GPT_GE_API_KEY",
        "cache": True,
        "cache_in_memory": True,
        "enable_thinking": False,
        "max_tokens": 16384,
        "model": "openai/gpt-4.1-mini-2025-04-14",
        "model_type": "chat",
        "num_retries": 0,
        "temperature": 1.0,
    }


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


def test_generated_aime_config_matches_gepa_and_current_four_cell_controls() -> None:
    profiles = load_model_profiles(
        Path("experiments/paper/model_profiles.json")
    )
    profile = profiles["qwen3_8b_local_vllm_18000"]
    _, config = build_run_config(
        task_id="aime_2025",
        condition="compass_reflection",
        seed=0,
        tag="v50_high_resolution_always_accept_window5",
        model_profile_name="qwen3_8b_local_vllm_18000",
        model_profile=profile,
        remote_root=Path("F:/compass-aime-local"),
        snapshot={"root_head": "abc"},
        proposal_minibatch_size=3,
        admission_minibatch_size=3,
        acceptance_mode="always_accept",
        epoch_parallel_enabled=True,
        max_candidate_workers=5,
        max_reflection_workers=5,
        parent_selection_score_mode="high_resolution",
        proposal_tasks_per_iteration=5,
    )

    optimizer = config["optimizer"]
    assert config["optimizer_seed"] == 0
    assert optimizer["max_metric_calls"] == 1839
    assert optimizer["num_threads"] == 32
    assert optimizer["proposal_minibatch_size"] == 3
    assert optimizer["admission_minibatch_size"] == 3
    assert optimizer["acceptance_mode"] == "always_accept"
    assert optimizer["parent_selection_score_mode"] == "high_resolution"
    assert optimizer["epoch_parallel_enabled"] is True
    assert optimizer["proposal_tasks_per_iteration"] == 5
    assert optimizer["max_candidate_workers"] == 5
    assert optimizer["max_reflection_workers"] == 5
    assert config["model"]["timeout"] == 6000
    assert profile == {
        "api_base": "http://127.0.0.1:18000/v1",
        "api_key_env": "OPENAI_API_KEY",
        "cache": True,
        "cache_in_memory": True,
        "checkpoint": (
            "/mnt/geogpt-doc-new/deepresearch/"
            "tool_jepa_qwen35_9b/models/Qwen3-8B"
        ),
        "enable_thinking": True,
        "max_tokens": 16384,
        "model": "openai/Qwen3-8B",
        "model_type": "chat",
        "num_retries": 0,
        "serving_backend": "vllm",
        "serving_max_model_len": 40960,
        "temperature": 0.6,
        "timeout": 6000,
        "top_k": 20,
        "top_p": 0.95,
    }


def test_generated_aime_window_one_limits_only_outer_minibatch_concurrency() -> None:
    profile = load_model_profiles(
        Path("experiments/paper/model_profiles.json")
    )["qwen3_8b_local_vllm_18000"]
    _, config = build_run_config(
        task_id="aime_2025",
        condition="compass_reflection",
        seed=0,
        tag="v51_raw_strict_window1_timeout6000",
        model_profile_name="qwen3_8b_local_vllm_18000",
        model_profile=profile,
        remote_root=Path("F:/compass-aime-local"),
        snapshot={"root_head": "abc"},
        proposal_minibatch_size=3,
        admission_minibatch_size=3,
        epoch_parallel_enabled=False,
        max_candidate_workers=1,
        max_reflection_workers=1,
    )

    assert config["optimizer"]["proposal_tasks_per_iteration"] is None
    assert config["optimizer"]["epoch_parallel_enabled"] is False
    assert config["optimizer"]["max_candidate_workers"] == 1
    assert config["optimizer"]["max_reflection_workers"] == 1
    assert config["optimizer"]["num_threads"] == 32
    assert config["model"]["timeout"] == 6000


def test_ifbench_qwen35_profile_uses_1200_second_request_timeout() -> None:
    profile = load_model_profiles(
        Path("experiments/paper/model_profiles.json")
    )["qwen3_5_9b_local_vllm_18035"]

    assert profile["timeout"] == 1200


def test_generated_ifbench_config_uses_1200_second_operation_deadlines() -> None:
    _, config = build_run_config(
        task_id="ifbench",
        condition="compass_reflection",
        seed=0,
        tag="timeout1200",
        model_profile_name="qwen3_5_9b_local_vllm_18035",
        model_profile=load_model_profiles(
            Path("experiments/paper/model_profiles.json")
        )["qwen3_5_9b_local_vllm_18035"],
        remote_root=Path("F:/COMPASS_evolution"),
        snapshot={},
        rollout_timeout_seconds=1200,
        proposal_timeout_seconds=1200,
    )

    assert config["model"]["timeout"] == 1200
    assert config["optimizer"]["rollout_timeout_seconds"] == 1200
    assert config["optimizer"]["proposal_timeout_seconds"] == 1200
    assert config["optimizer"]["evaluation_straggler_timeout"] == 0


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
