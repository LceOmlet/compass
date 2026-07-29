from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import yaml

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
    source = _RAW_ENTRY.read_text(encoding="utf-8")
    assert "load_model_and_tokenizer" not in source
    assert "ExactTokenOffloadedFlashTrace" not in source


def test_v28_retry_only_changes_the_unique_deployment_path() -> None:
    namespace = runpy.run_path(str(_RAW_ENTRY))
    v27 = namespace["load_config"](
        _ROOT
        / "experiments"
        / "11_ifbench_siliconflow_v27_raw_feedback_no_local_model_full_epoch_parallel.json"
    )
    v28 = namespace["load_config"](
        _ROOT
        / "experiments"
        / "11_ifbench_siliconflow_v28_raw_feedback_no_local_model_full_epoch_parallel.json"
    )
    namespace["_require_configuration"](v28)

    assert {key: value for key, value in v28.items() if key != "deployment"} == {
        key: value for key, value in v27.items() if key != "deployment"
    }
    assert v28["deployment"]["run_dir"].endswith(
        "11_ifbench_raw_feedback_k1_20260729_v28_no_local_model_full_epoch_parallel"
    )


def test_v29_retry_only_changes_the_unique_deployment_path() -> None:
    namespace = runpy.run_path(str(_RAW_ENTRY))
    v28 = namespace["load_config"](
        _ROOT
        / "experiments"
        / "11_ifbench_siliconflow_v28_raw_feedback_no_local_model_full_epoch_parallel.json"
    )
    v29 = namespace["load_config"](
        _ROOT
        / "experiments"
        / "11_ifbench_siliconflow_v29_raw_feedback_no_local_model_full_epoch_parallel.json"
    )
    namespace["_require_configuration"](v29)

    assert {key: value for key, value in v29.items() if key != "deployment"} == {
        key: value for key, value in v28.items() if key != "deployment"
    }
    assert v29["deployment"]["run_dir"].endswith(
        "11_ifbench_raw_feedback_k1_20260729_v29_no_local_model_full_epoch_parallel"
    )


def test_v30_two_account_router_only_changes_runtime_endpoint() -> None:
    namespace = runpy.run_path(str(_RAW_ENTRY))
    v29 = namespace["load_config"](
        _ROOT
        / "experiments"
        / "11_ifbench_siliconflow_v29_raw_feedback_no_local_model_full_epoch_parallel.json"
    )
    v30 = namespace["load_config"](
        _ROOT
        / "experiments"
        / "11_ifbench_siliconflow_v30_two_account_router_no_local_model_full_epoch_parallel.json"
    )
    namespace["_require_configuration"](v30)

    assert {
        key: value
        for key, value in v30.items()
        if key not in {"deployment", "remote_lm"}
    } == {
        key: value
        for key, value in v29.items()
        if key not in {"deployment", "remote_lm"}
    }
    assert {
        key: value
        for key, value in v30["remote_lm"].items()
        if key not in {"api_base", "api_key_env", "model"}
    } == {
        key: value
        for key, value in v29["remote_lm"].items()
        if key not in {"api_base", "api_key_env", "model"}
    }
    assert v30["deployment"]["run_dir"].endswith(
        "11_ifbench_raw_feedback_k1_20260729_v30_no_local_model_"
        "full_epoch_parallel_two_account_router"
    )
    assert v30["remote_lm"]["api_base"] == "http://127.0.0.1:40029/v1"
    assert v30["remote_lm"]["api_key_env"] == "COMPASS_LITELLM_PROXY_KEY"
    assert v30["remote_lm"]["model"] == "openai/compass-qwen3-8b"


def test_v30_router_has_two_secret_free_deployments_and_no_retries() -> None:
    router_path = (
        _ROOT / "experiments" / "11_ifbench_litellm_two_account_router_v30.yaml"
    )
    router = yaml.safe_load(router_path.read_text(encoding="utf-8"))

    assert [entry["model_name"] for entry in router["model_list"]] == [
        "compass-qwen3-8b",
        "compass-qwen3-8b",
    ]
    assert [entry["litellm_params"]["api_key"] for entry in router["model_list"]] == [
        "os.environ/SILICONFLOW_API_KEY_PRIMARY",
        "os.environ/SILICONFLOW_API_KEY_SECONDARY",
    ]
    assert router["router_settings"] == {
        "routing_strategy": "simple-shuffle",
        "num_retries": 0,
        "max_fallbacks": 0,
        "disable_cooldowns": True,
    }
    assert router["litellm_settings"]["num_retries"] == 0
    assert "sk-" not in router_path.read_text(encoding="utf-8")
