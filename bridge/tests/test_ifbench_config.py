from __future__ import annotations

import runpy
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_ENTRY = _ROOT / "experiments" / "11_ifbench_sparse_dependency_training.py"


def _entry_namespace() -> dict[str, object]:
    return runpy.run_path(str(_ENTRY))


@pytest.mark.parametrize(
    "config_name",
    [
        "11_ifbench_siliconflow_sparse_single_gpu_v23_resume_v22_no_tf_full_epoch_parallel.json",
        "11_ifbench_siliconflow_sparse_single_gpu_v24_resume_v23_no_tf_full_epoch_parallel.json",
        "11_ifbench_siliconflow_sparse_single_gpu_v25_resume_v24_no_tf_full_epoch_parallel.json",
    ],
)
def test_full_epoch_configs_enable_whole_epoch_parallelism(config_name: str) -> None:
    namespace = _entry_namespace()
    config = namespace["load_config"](
        _ROOT / "experiments" / config_name
    )
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
