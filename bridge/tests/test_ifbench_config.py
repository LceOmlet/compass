from __future__ import annotations

import runpy
from pathlib import Path

import dspy
import pytest

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
