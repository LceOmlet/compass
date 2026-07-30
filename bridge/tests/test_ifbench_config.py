from __future__ import annotations

import runpy
from pathlib import Path

import dspy
import pytest
import yaml

from bridge.siliconflow_lm import SiliconFlowLM

_ROOT = Path(__file__).resolve().parents[2]
_ENTRY = _ROOT / "experiments" / "11_ifbench_sparse_dependency_training.py"
_RAW_ENTRY = _ROOT / "experiments" / "11_ifbench_sparse_raw_feedback_training.py"


def _entry_namespace() -> dict[str, object]:
    return runpy.run_path(str(_ENTRY))


@pytest.mark.parametrize(
    "config_name",
    [
        "11_ifbench_siliconflow_sparse_single_gpu_v23_resume_v22_no_tf_full_epoch_parallel.json",
        "11_ifbench_siliconflow_sparse_single_gpu_v24_resume_v23_no_tf_full_epoch_parallel.json",
        "11_ifbench_siliconflow_sparse_single_gpu_v25_resume_v24_no_tf_full_epoch_parallel.json",
        "11_ifbench_siliconflow_sparse_single_gpu_v26_resume_v25_no_tf_full_epoch_parallel.json",
    ],
)
def test_full_epoch_configs_enable_whole_epoch_parallelism(config_name: str) -> None:
    namespace = _entry_namespace()
    config = namespace["load_config"](_ROOT / "experiments" / config_name)
    namespace["_require_official_configuration"](config)

    assert config["epoch_parallel"] == {
        "enabled": True,
        "max_candidate_workers": 50,
        "max_reflection_workers": 50,
    }


def test_legacy_config_keeps_single_task_defaults() -> None:
    namespace = _entry_namespace()
    config = namespace["load_config"](
        _ROOT
        / "experiments"
        / "11_ifbench_siliconflow_sparse_single_gpu_v22_resume_v21_no_tf_epoch_parallel.json"
    )
    namespace["_require_official_configuration"](config)

    assert config["epoch_parallel"] == {
        "enabled": False,
        "max_candidate_workers": 1,
        "max_reflection_workers": 1,
    }


def test_v27_raw_feedback_has_one_candidate_and_no_local_model_config() -> None:
    namespace = runpy.run_path(str(_RAW_ENTRY))
    config = namespace["load_config"](
        _ROOT
        / "experiments"
        / "11_ifbench_siliconflow_v27_raw_feedback_no_local_model_full_epoch_parallel.json"
    )
    namespace["_require_configuration"](config)

    assert config["deployment"] == {
        "run_dir": (
            "/mnt/geogpt-doc-new/deepresearch/gepa-multi-skill/"
            "reflection-bridge-v27-no-local-model/runs/"
            "11_ifbench_raw_feedback_k1_20260729_v27_no_local_model_"
            "full_epoch_parallel"
        )
    }
    assert config["remote_lm"]["n"] == 1
    assert config["epoch_parallel"] == {
        "enabled": True,
        "max_candidate_workers": 50,
        "max_reflection_workers": 50,
    }
    lm = namespace["_build_remote_lm"](config["remote_lm"], api_key="test-key")
    assert type(lm) is dspy.LM
    source = _RAW_ENTRY.read_text(encoding="utf-8")
    assert "load_model_and_tokenizer" not in source
    assert "ExactTokenOffloadedFlashTrace" not in source


def test_v34_rate_limit_keepalive_has_per_request_timeout() -> None:
    namespace = runpy.run_path(str(_RAW_ENTRY))
    config = namespace["load_config"](
        _ROOT
        / "experiments"
        / "11_ifbench_siliconflow_v34_raw_feedback_rate_limit_retry.json"
    )
    namespace["_require_configuration"](config)

    assert config["remote_lm"]["num_retries"] == 0
    assert config["remote_lm"]["rollout_timeout_seconds"] == 600
    assert config["epoch_parallel"] == {
        "enabled": True,
        "max_candidate_workers": 50,
        "max_reflection_workers": 50,
    }
    lm = namespace["_build_remote_lm"](config["remote_lm"], api_key="test-key")
    assert isinstance(lm, SiliconFlowLM)
    assert lm.rollout_timeout_seconds == 600


def test_v35_dual_account_router_only_changes_transport_endpoint() -> None:
    namespace = runpy.run_path(str(_RAW_ENTRY))
    v34 = namespace["load_config"](
        _ROOT
        / "experiments"
        / "11_ifbench_siliconflow_v34_raw_feedback_rate_limit_retry.json"
    )
    v35 = namespace["load_config"](
        _ROOT
        / "experiments"
        / "11_ifbench_siliconflow_v35_dual_account_rate_limit_failover.json"
    )
    namespace["_require_configuration"](v35)

    assert {
        key: value
        for key, value in v35.items()
        if key not in {"deployment", "remote_lm"}
    } == {
        key: value
        for key, value in v34.items()
        if key not in {"deployment", "remote_lm"}
    }
    assert {
        key: value
        for key, value in v35["remote_lm"].items()
        if key not in {"api_base", "api_key_env", "model"}
    } == {
        key: value
        for key, value in v34["remote_lm"].items()
        if key not in {"api_base", "api_key_env", "model"}
    }
    assert v35["remote_lm"]["api_base"] == "http://127.0.0.1:40035/v1"
    assert v35["remote_lm"]["api_key_env"] == "COMPASS_LITELLM_PROXY_KEY"
    assert v35["remote_lm"]["model"] == "openai/compass-qwen3-8b"
    assert v35["remote_lm"]["num_retries"] == 0
    assert v35["remote_lm"]["rollout_timeout_seconds"] == 600
    lm = namespace["_build_remote_lm"](v35["remote_lm"], api_key="test-key")
    assert isinstance(lm, SiliconFlowLM)


def test_v36_unconditional_admission_only_changes_acceptance_and_run_dir() -> None:
    namespace = runpy.run_path(str(_RAW_ENTRY))
    v35 = namespace["load_config"](
        _ROOT
        / "experiments"
        / "11_ifbench_siliconflow_v35_dual_account_rate_limit_failover.json"
    )
    v36 = namespace["load_config"](
        _ROOT
        / "experiments"
        / "11_ifbench_siliconflow_v36_unconditional_admission_local.json"
    )
    namespace["_require_configuration"](v35)
    namespace["_require_configuration"](v36)

    assert v35["official_gepa"].get(
        "acceptance_mode",
        "strict_improvement",
    ) == "strict_improvement"
    assert v36["official_gepa"]["acceptance_mode"] == "always_accept"
    assert v36["remote_lm"] == v35["remote_lm"]
    assert v36["parent_selection"] == v35["parent_selection"]
    assert v36["epoch_parallel"] == v35["epoch_parallel"]
    assert {
        key: value
        for key, value in v36["official_gepa"].items()
        if key != "acceptance_mode"
    } == v35["official_gepa"]


def test_v35_router_uses_official_least_busy_rate_limit_failover() -> None:
    router_path = (
        _ROOT / "experiments" / "11_ifbench_litellm_two_account_router_v35.yaml"
    )
    router_text = router_path.read_text(encoding="utf-8")
    router = yaml.safe_load(router_text)

    assert [entry["model_name"] for entry in router["model_list"]] == [
        "compass-qwen3-8b",
        "compass-qwen3-8b",
    ]
    assert [entry["litellm_params"]["api_key"] for entry in router["model_list"]] == [
        "os.environ/SILICONFLOW_API_KEY_PRIMARY",
        "os.environ/SILICONFLOW_API_KEY_SECONDARY",
    ]
    assert router["router_settings"] == {
        "routing_strategy": "least-busy",
        "num_retries": 0,
        "retry_policy": {"RateLimitErrorRetries": 1},
        "max_fallbacks": 0,
    }
    assert "disable_cooldowns" not in router["router_settings"]
    assert router["litellm_settings"] == {
        "num_retries": 0,
        "telemetry": False,
    }
    assert "sk-" not in router_text
