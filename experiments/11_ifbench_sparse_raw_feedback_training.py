from __future__ import annotations

import argparse
import json
import math
import os
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Python 3.12 resolves Windows platform metadata through WMI.  Resolve it once
# on the main thread so concurrent OpenAI clients only read the stdlib cache
# instead of entering WMI/RPC together.
if os.name == "nt":
    platform.platform()

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from gepa_artifact.benchmarks.IFBench import IFBench

from bridge.b20_compass_reflection import (
    CompassReflectionEngineConfig,
    run_compass_reflection_engine,
)
from bridge.paper_benchmark_registry import (
    canonical_feedback_map,
    load_official_benchmark_specs,
)
from bridge.siliconflow_lm import SiliconFlowLM

CONFIG_KEYS = {
    "deployment": {"run_dir"},
    "remote_lm": {
        "api_base",
        "api_key_env",
        "cache",
        "cache_dir_env",
        "cache_in_memory",
        "enable_thinking",
        "max_tokens",
        "model",
        "model_type",
        "n",
        "num_retries",
        "temperature",
        "top_k",
        "top_p",
    },
    "parent_selection": {"top_n"},
    "epoch_parallel": {
        "enabled",
        "max_candidate_workers",
        "max_reflection_workers",
    },
    "official_gepa": {
        "add_format_failure_as_feedback",
        "dataset_mode",
        "display_progress_bar",
        "failure_score",
        "max_metric_calls",
        "num_threads",
        "perfect_score",
        "raise_on_exception",
        "reflection_minibatch_size",
        "seed",
        "skip_perfect_score",
        "track_best_outputs",
        "use_cloudpickle",
    },
}
OPTIONAL_CONFIG_KEYS = {
    "remote_lm": {"rollout_timeout_seconds"},
    "parent_selection": {"mode"},
    "official_gepa": {"acceptance_mode"},
}


def _exact_mapping(
    value: Any,
    name: str,
    keys: set[str],
    *,
    optional_keys: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    missing = keys.difference(value)
    extra = set(value).difference(keys | (optional_keys or set()))
    if missing or extra:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return dict(value)


def load_config(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = _exact_mapping(raw, "config", set(CONFIG_KEYS))
    return {
        section: _exact_mapping(
            root[section],
            section,
            keys,
            optional_keys=OPTIONAL_CONFIG_KEYS.get(section),
        )
        for section, keys in CONFIG_KEYS.items()
    }


def _require_configuration(config: dict[str, dict[str, Any]]) -> None:
    remote = config["remote_lm"]
    parent_selection = config["parent_selection"]
    epoch_parallel = config["epoch_parallel"]
    official = config["official_gepa"]
    required_remote = {
        "cache": True,
        "cache_in_memory": True,
        "enable_thinking": True,
        "max_tokens": 16384,
        "n": 1,
        "num_retries": 0,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
    }
    required_official = {
        "add_format_failure_as_feedback": False,
        "dataset_mode": "lite",
        "display_progress_bar": False,
        "failure_score": 0,
        "max_metric_calls": 3593,
        "perfect_score": 1,
        "raise_on_exception": True,
        "reflection_minibatch_size": 3,
        "seed": 0,
        "skip_perfect_score": True,
        "track_best_outputs": True,
    }
    for name, expected in required_remote.items():
        if remote[name] != expected:
            raise ValueError(
                f"remote_lm.{name} must equal the required value {expected!r}"
            )
    rollout_timeout_seconds = remote.get("rollout_timeout_seconds")
    if rollout_timeout_seconds is not None and (
        isinstance(rollout_timeout_seconds, bool)
        or not isinstance(rollout_timeout_seconds, (int, float))
        or not math.isfinite(rollout_timeout_seconds)
        or rollout_timeout_seconds <= 0
    ):
        raise TypeError(
            "remote_lm.rollout_timeout_seconds must be a positive JSON number"
        )
    for name, expected in required_official.items():
        if official[name] != expected:
            raise ValueError(
                f"official_gepa.{name} must equal the required value {expected!r}"
            )
    if not isinstance(official["use_cloudpickle"], bool):
        raise TypeError("official_gepa.use_cloudpickle must be a JSON boolean")
    acceptance_mode = official.get("acceptance_mode", "strict_improvement")
    if acceptance_mode not in {"strict_improvement", "always_accept"}:
        raise ValueError(
            "official_gepa.acceptance_mode must be "
            "'strict_improvement' or 'always_accept'"
        )
    num_threads = official["num_threads"]
    if num_threads is not None and (type(num_threads) is not int or num_threads <= 0):
        raise TypeError(
            "official_gepa.num_threads must be null or a positive JSON integer"
        )
    if type(parent_selection["top_n"]) is not int or parent_selection["top_n"] <= 0:
        raise TypeError("parent_selection.top_n must be a positive JSON integer")
    if parent_selection.get("mode", "high_resolution") not in {
        "raw_frontier_rate",
        "high_resolution",
    }:
        raise ValueError(
            "parent_selection.mode must be "
            "'raw_frontier_rate' or 'high_resolution'"
        )
    if epoch_parallel["enabled"] is not True:
        raise ValueError("epoch_parallel.enabled must be true")
    for name in ("max_candidate_workers", "max_reflection_workers"):
        value = epoch_parallel[name]
        if type(value) is not int or value <= 0:
            raise TypeError(f"epoch_parallel.{name} must be a positive JSON integer")


def _prepare_run_dir(run_dir: Path, *, resume_existing: bool) -> None:
    if not resume_existing:
        run_dir.mkdir(parents=True, exist_ok=False)
        return
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"resume run directory does not exist or is not a directory: {run_dir}"
        )
    state_path = run_dir / "gepa_state.bin"
    if not state_path.is_file():
        raise FileNotFoundError(
            f"resume run directory is missing the official GEPA checkpoint: {state_path}"
        )


def _build_remote_lm(remote: Mapping[str, Any], *, api_key: str) -> dspy.LM:
    lm_kwargs = {
        "model": remote["model"],
        "model_type": remote["model_type"],
        "temperature": remote["temperature"],
        "max_tokens": remote["max_tokens"],
        "cache": remote["cache"],
        "cache_in_memory": remote["cache_in_memory"],
        "num_retries": remote["num_retries"],
        "api_base": remote["api_base"],
        "api_key": api_key,
        "n": remote["n"],
        "top_p": remote["top_p"],
        "extra_body": {
            "top_k": remote["top_k"],
            "chat_template_kwargs": {"enable_thinking": remote["enable_thinking"]},
        },
    }
    rollout_timeout_seconds = remote.get("rollout_timeout_seconds")
    if rollout_timeout_seconds is None:
        return dspy.LM(**lm_kwargs)
    return SiliconFlowLM(
        **lm_kwargs,
        rollout_timeout_seconds=rollout_timeout_seconds,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="resume from gepa_state.bin in the configured existing run directory",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    _require_configuration(config)

    deployment = config["deployment"]
    remote = config["remote_lm"]
    epoch_parallel = config["epoch_parallel"]
    official = config["official_gepa"]

    api_key_env = remote["api_key_env"]
    if not isinstance(api_key_env, str) or not api_key_env:
        raise ValueError("remote_lm.api_key_env must be non-empty text")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(
            f"required API key environment variable is unset: {api_key_env}"
        )
    cache_dir_env = remote["cache_dir_env"]
    if not isinstance(cache_dir_env, str) or not cache_dir_env:
        raise ValueError("remote_lm.cache_dir_env must be non-empty text")
    cache_dir = os.environ.get(cache_dir_env)
    if not cache_dir:
        raise ValueError(
            f"required cache directory environment variable is unset: {cache_dir_env}"
        )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    run_dir = Path(deployment["run_dir"]).resolve()
    _prepare_run_dir(run_dir, resume_existing=args.resume_existing)

    lm = _build_remote_lm(remote, api_key=api_key)
    dspy.configure(lm=lm, adapter=ChatAdapter())

    spec = load_official_benchmark_specs()["ifbench"]
    benchmark = IFBench(dataset_mode=official["dataset_mode"])
    run_compass_reflection_engine(
        program=spec.program,
        metric_fn=spec.benchmark_meta.metric,
        feedback_map=canonical_feedback_map(spec),
        trainset=benchmark.train_set,
        reflection_lm=lm,
        config=CompassReflectionEngineConfig(
            run_dir=run_dir,
            condition="compass_reflection",
            seed=official["seed"],
            reflection_minibatch_size=official["reflection_minibatch_size"],
            parent_top_n=config["parent_selection"]["top_n"],
            parent_selection_score_mode=config["parent_selection"].get(
                "mode",
                "high_resolution",
            ),
            max_metric_calls=official["max_metric_calls"],
            perfect_score=official["perfect_score"],
            failure_score=official["failure_score"],
            num_threads=official["num_threads"],
            max_candidate_workers=epoch_parallel["max_candidate_workers"],
            skip_perfect_score=official["skip_perfect_score"],
            add_format_failure_as_feedback=official["add_format_failure_as_feedback"],
            track_best_outputs=official["track_best_outputs"],
            display_progress_bar=official["display_progress_bar"],
            raise_on_exception=official["raise_on_exception"],
            use_cloudpickle=official["use_cloudpickle"],
            epoch_parallel_enabled=epoch_parallel["enabled"],
            max_reflection_workers=epoch_parallel["max_reflection_workers"],
            acceptance_mode=official.get(
                "acceptance_mode",
                "strict_improvement",
            ),
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
