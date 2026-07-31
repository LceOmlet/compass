from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from gepa.core.state import GEPAState

from bridge.b20_compass_reflection import (
    CompassReflectionEngineConfig,
    run_compass_reflection_engine,
)
from bridge.minibatch_config import minibatch_config_kwargs
from bridge.paper_benchmark_registry import (
    canonical_feedback_map,
    instantiate_official_splits,
    load_official_benchmark_specs,
)

ROOT_KEYS = {
    "cache_dir",
    "condition",
    "dataset_mode",
    "model",
    "optimizer",
    "optimizer_seed",
    "run_dir",
    "source_snapshot",
    "task_id",
}
OPTIMIZER_REQUIRED_KEYS = {
    "add_format_failure_as_feedback",
    "display_progress_bar",
    "failure_score",
    "max_candidate_workers",
    "max_metric_calls",
    "num_threads",
    "parent_top_n",
    "perfect_score",
    "raise_on_exception",
    "skip_perfect_score",
    "track_best_outputs",
    "use_cloudpickle",
}
OPTIMIZER_BATCH_KEYS = {
    "reflection_minibatch_size",
    "proposal_minibatch_size",
    "admission_minibatch_size",
}
OPTIMIZER_METHOD_KEYS = {
    "acceptance_mode",
    "epoch_parallel_enabled",
    "max_reflection_workers",
    "parent_selection_score_mode",
    "proposal_tasks_per_iteration",
}
MODEL_REQUIRED_KEYS = {
    "api_key_env",
    "cache",
    "cache_in_memory",
    "enable_thinking",
    "max_tokens",
    "model",
    "model_type",
    "num_retries",
    "temperature",
}
MODEL_OPTIONAL_KEYS = {
    "api_base",
    "checkpoint",
    "serving_backend",
    "serving_max_model_len",
    "top_k",
    "top_p",
    "timeout",
}


def _exact_mapping(
    value: Any,
    *,
    name: str,
    keys: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    missing = keys.difference(value)
    extra = set(value).difference(keys)
    if missing or extra:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    return dict(value)


def _model_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("model must be a JSON object")
    missing = MODEL_REQUIRED_KEYS.difference(value)
    extra = set(value).difference(MODEL_REQUIRED_KEYS | MODEL_OPTIONAL_KEYS)
    if missing or extra:
        raise ValueError(
            f"model keys mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    model = dict(value)
    timeout = model.get("timeout")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise TypeError("model.timeout must be a positive finite number")
    return model


def _optimizer_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("optimizer must be a JSON object")
    missing = OPTIMIZER_REQUIRED_KEYS.difference(value)
    extra = set(value).difference(
        OPTIMIZER_REQUIRED_KEYS
        | OPTIMIZER_BATCH_KEYS
        | OPTIMIZER_METHOD_KEYS
    )
    if missing or extra:
        raise ValueError(
            f"optimizer keys mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    optimizer = dict(value)
    minibatch_config_kwargs(optimizer, namespace="optimizer")
    acceptance_mode = optimizer.get(
        "acceptance_mode",
        "strict_improvement",
    )
    if acceptance_mode not in {"strict_improvement", "always_accept"}:
        raise ValueError(
            "optimizer.acceptance_mode must be 'strict_improvement' "
            "or 'always_accept'"
        )
    score_mode = optimizer.get(
        "parent_selection_score_mode",
        "high_resolution",
    )
    if score_mode not in {"raw_frontier_rate", "high_resolution"}:
        raise ValueError(
            "optimizer.parent_selection_score_mode must be "
            "'raw_frontier_rate' or 'high_resolution'"
        )
    epoch_parallel_enabled = optimizer.get(
        "epoch_parallel_enabled",
        False,
    )
    if not isinstance(epoch_parallel_enabled, bool):
        raise TypeError("optimizer.epoch_parallel_enabled must be a JSON boolean")
    proposal_tasks = optimizer.get("proposal_tasks_per_iteration")
    if proposal_tasks is not None and (
        isinstance(proposal_tasks, bool)
        or not isinstance(proposal_tasks, int)
        or proposal_tasks <= 0
    ):
        raise TypeError(
            "optimizer.proposal_tasks_per_iteration must be a positive integer"
        )
    if proposal_tasks is not None and not epoch_parallel_enabled:
        raise ValueError(
            "optimizer.proposal_tasks_per_iteration requires "
            "optimizer.epoch_parallel_enabled=true"
        )
    max_reflection_workers = optimizer.get("max_reflection_workers", 1)
    if (
        isinstance(max_reflection_workers, bool)
        or not isinstance(max_reflection_workers, int)
        or max_reflection_workers <= 0
    ):
        raise TypeError(
            "optimizer.max_reflection_workers must be a positive integer"
        )
    return optimizer


def _minibatch_config_kwargs(
    optimizer: Mapping[str, Any],
) -> dict[str, int | None]:
    return minibatch_config_kwargs(optimizer, namespace="optimizer")


def _method_config_kwargs(optimizer: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve explicitly configured COMPASS method choices without copying GEPA."""

    return {
        "acceptance_mode": optimizer.get(
            "acceptance_mode",
            "strict_improvement",
        ),
        "epoch_parallel_enabled": optimizer.get(
            "epoch_parallel_enabled",
            False,
        ),
        "max_reflection_workers": optimizer.get(
            "max_reflection_workers",
            1,
        ),
        "parent_selection_score_mode": optimizer.get(
            "parent_selection_score_mode",
            "high_resolution",
        ),
        "proposal_tasks_per_iteration": optimizer.get(
            "proposal_tasks_per_iteration"
        ),
    }


def load_run_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = _exact_mapping(raw, name="config", keys=ROOT_KEYS)
    config["optimizer"] = _optimizer_mapping(config["optimizer"])
    config["model"] = _model_mapping(config["model"])
    if not isinstance(config["source_snapshot"], Mapping):
        raise TypeError("source_snapshot must be a JSON object")
    if config["condition"] not in (
        "mini_admission_reflection",
        "compass_reflection",
    ):
        raise ValueError("unsupported reflection condition")
    if (
        "proposal_minibatch_size" in config["optimizer"]
        and config["condition"] != "compass_reflection"
    ):
        raise ValueError(
            "split train/validation admission requires "
            "condition='compass_reflection'"
        )
    if (
        isinstance(config["optimizer_seed"], bool)
        or not isinstance(config["optimizer_seed"], int)
        or config["optimizer_seed"] < 0
    ):
        raise TypeError("optimizer_seed must be a non-negative integer")
    if config["dataset_mode"] != "lite":
        raise ValueError("paper experiments require dataset_mode='lite'")
    return config


def _require_frozen_protocol(
    config: Mapping[str, Any],
    *,
    budget: int,
) -> None:
    optimizer = config["optimizer"]
    expected = {
        "add_format_failure_as_feedback": False,
        "display_progress_bar": False,
        "failure_score": 0,
        "max_metric_calls": budget,
        "parent_top_n": 5,
        "perfect_score": 1,
        "raise_on_exception": True,
        "skip_perfect_score": True,
        "track_best_outputs": True,
        "use_cloudpickle": True,
    }
    for key, expected_value in expected.items():
        if optimizer[key] != expected_value:
            raise ValueError(
                f"optimizer.{key} must equal the frozen value "
                f"{expected_value!r}"
            )
    for key in ("num_threads", "max_candidate_workers"):
        value = optimizer[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TypeError(f"optimizer.{key} must be a positive integer")

    model = config["model"]
    expected_model = {
        "cache": True,
        "cache_in_memory": True,
        "max_tokens": 16384,
        "num_retries": 0,
    }
    for key, expected_value in expected_model.items():
        if model[key] != expected_value:
            raise ValueError(
                f"model.{key} must equal the frozen value {expected_value!r}"
            )
    if not isinstance(model["enable_thinking"], bool):
        raise TypeError("model.enable_thinking must be a JSON boolean")
    serving_max_model_len = model.get("serving_max_model_len")
    if serving_max_model_len is not None:
        if (
            isinstance(serving_max_model_len, bool)
            or not isinstance(serving_max_model_len, int)
        ):
            raise TypeError("model.serving_max_model_len must be an integer")
        if serving_max_model_len <= model["max_tokens"]:
            raise ValueError(
                "model.serving_max_model_len must exceed model.max_tokens "
                "so non-empty prompts fit in the deployed context window"
            )


def _require_aime_gepa_protocol(config: Mapping[str, Any]) -> None:
    """Reject AIME settings that drift from the official GEPA comparison."""

    expected_optimizer = {
        "max_metric_calls": 1839,
        "num_threads": 32,
    }
    for key, expected_value in expected_optimizer.items():
        if config["optimizer"][key] != expected_value:
            raise ValueError(
                f"AIME GEPA parity requires optimizer.{key}="
                f"{expected_value!r}"
            )
    if config["optimizer_seed"] != 0:
        raise ValueError("AIME GEPA parity requires optimizer_seed=0")

    expected_model = {
        "enable_thinking": True,
        "max_tokens": 16384,
        "model_type": "chat",
        "num_retries": 0,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
    }
    for key, expected_value in expected_model.items():
        if config["model"].get(key) != expected_value:
            raise ValueError(
                f"AIME GEPA parity requires model.{key}={expected_value!r}"
            )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    try:
        return dict(value)
    except (TypeError, ValueError):
        return repr(value)


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _create_lm(model_config: Mapping[str, Any], *, api_key: str) -> dspy.LM:
    kwargs: dict[str, Any] = {
        "model": model_config["model"],
        "model_type": model_config["model_type"],
        "temperature": model_config["temperature"],
        "max_tokens": model_config["max_tokens"],
        "cache": model_config["cache"],
        "cache_in_memory": model_config["cache_in_memory"],
        "num_retries": model_config["num_retries"],
        "api_key": api_key,
    }
    api_base = model_config.get("api_base")
    if api_base is not None:
        kwargs["api_base"] = api_base
    timeout = model_config.get("timeout")
    if timeout is not None:
        kwargs["timeout"] = timeout
    top_p = model_config.get("top_p")
    if top_p is not None:
        kwargs["top_p"] = top_p
    extra_body: dict[str, Any] = {}
    top_k = model_config.get("top_k")
    if top_k is not None:
        extra_body["top_k"] = top_k
    if model_config["enable_thinking"]:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    if extra_body:
        kwargs["extra_body"] = extra_body
    return dspy.LM(**kwargs)


def _lm_usage(lm: Any) -> dict[str, float | int]:
    cost = 0.0
    input_tokens = 0
    output_tokens = 0
    for record in getattr(lm, "history", ()):
        if not isinstance(record, Mapping):
            continue
        cost += float(record.get("cost") or 0.0)
        usage = record.get("usage")
        if isinstance(usage, Mapping):
            input_tokens += int(usage.get("prompt_tokens") or 0)
            output_tokens += int(usage.get("completion_tokens") or 0)
    return {
        "cost": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "calls": len(getattr(lm, "history", ())),
    }


def _score_summary(scores: Sequence[Any]) -> dict[str, Any]:
    values = [float(score) for score in scores]
    if not values or any(not math.isfinite(value) for value in values):
        raise RuntimeError("held-out evaluation returned missing or non-finite scores")
    return {
        "aggregate_percent": 100.0 * math.fsum(values) / len(values),
        "count": len(values),
        "scores": values,
    }


def main() -> int:
    from gepa_artifact.utils.metric_logger import (
        CounterWithLock,
        MetricWithLogger,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = load_run_config(config_path)

    specs = load_official_benchmark_specs()
    task_id = config["task_id"]
    if task_id not in specs:
        raise ValueError(f"unknown paper task: {task_id!r}")
    spec = specs[task_id]
    _require_frozen_protocol(config, budget=spec.max_metric_calls)
    if task_id == "aime_2025":
        _require_aime_gepa_protocol(config)

    api_key_env = config["model"]["api_key_env"]
    if not isinstance(api_key_env, str) or not api_key_env:
        raise TypeError("model.api_key_env must be non-empty text")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"required API key environment variable is unset: {api_key_env}"
        )

    cache_dir = Path(config["cache_dir"]).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    dspy_cache_dir = cache_dir / ".dspy_cache"
    dspy_cache_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "DSPY_CACHEDIR",
        "DSP_CACHEDIR",
        "DSPY_NOTEBOOK_CACHEDIR",
        "DSP_NOTEBOOK_CACHEDIR",
    ):
        os.environ[name] = str(dspy_cache_dir)

    run_dir = Path(config["run_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    start_wall = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    splits = instantiate_official_splits(
        spec,
        optimizer_seed=config["optimizer_seed"],
        dataset_mode=config["dataset_mode"],
    )
    owner_thread_cap = spec.benchmark_meta.num_threads
    requested_threads = config["optimizer"]["num_threads"]
    effective_threads = (
        min(requested_threads, owner_thread_cap)
        if owner_thread_cap is not None
        else requested_threads
    )
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": started_at,
        "config_path": str(config_path),
        "config_sha256": _config_hash(config),
        "condition": config["condition"],
        "task": {
            "task_id": task_id,
            "display_name": spec.display_name,
            "benchmark_class": spec.benchmark_class_name,
            "program_class": spec.program_class_name,
            "predictor_names": list(spec.predictor_names),
            "dataset_mode": config["dataset_mode"],
            "split_sizes": {
                "train": len(splits.train),
                "validation": len(splits.validation),
                "test": len(splits.test),
            },
            "split_fingerprints": dict(splits.fingerprints),
        },
        "optimizer": {
            **dict(config["optimizer"]),
            "effective_num_threads": effective_threads,
            "owner_num_threads_cap": owner_thread_cap,
        },
        "model": {
            key: value
            for key, value in config["model"].items()
            if key != "api_key_env"
        },
        "api_key_env_name": api_key_env,
        "cache_dir": str(cache_dir),
        "run_dir": str(run_dir),
        "source_snapshot": dict(config["source_snapshot"]),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    _atomic_write_json(run_dir / "manifest.json", manifest)
    _atomic_write_json(run_dir / "config.json", config)

    lm = _create_lm(config["model"], api_key=api_key)
    chat_adapter = ChatAdapter()
    dspy.configure(lm=lm, adapter=chat_adapter)
    counter = CounterWithLock()
    try:
        with MetricWithLogger(
            metric_fn=spec.benchmark_meta.metric,
            run_dir=str(run_dir),
            counter_with_lock=counter,
            train_dataset=list(splits.train),
            val_dataset=list(splits.validation),
            test_dataset=list(splits.test),
            log_trace=False,
            log_example=False,
            log_prediction=True,
        ) as metric:
            engine_config = CompassReflectionEngineConfig(
                run_dir=run_dir,
                condition=config["condition"],
                seed=config["optimizer_seed"],
                parent_top_n=config["optimizer"]["parent_top_n"],
                max_metric_calls=config["optimizer"]["max_metric_calls"],
                perfect_score=config["optimizer"]["perfect_score"],
                failure_score=config["optimizer"]["failure_score"],
                num_threads=effective_threads,
                max_candidate_workers=config["optimizer"][
                    "max_candidate_workers"
                ],
                skip_perfect_score=config["optimizer"]["skip_perfect_score"],
                add_format_failure_as_feedback=config["optimizer"][
                    "add_format_failure_as_feedback"
                ],
                track_best_outputs=config["optimizer"]["track_best_outputs"],
                display_progress_bar=config["optimizer"][
                    "display_progress_bar"
                ],
                raise_on_exception=config["optimizer"]["raise_on_exception"],
                use_cloudpickle=config["optimizer"]["use_cloudpickle"],
                **_minibatch_config_kwargs(config["optimizer"]),
                **_method_config_kwargs(config["optimizer"]),
            )
            run = run_compass_reflection_engine(
                program=spec.program,
                metric_fn=metric,
                feedback_map=canonical_feedback_map(spec),
                trainset=list(splits.train),
                validation_set=list(splits.validation),
                reflection_lm=lm,
                config=engine_config,
            )

            state = GEPAState.load(str(run_dir))
            selected_idx = run.evaluation_policy.get_best_program(state)
            selected_candidate = state.program_candidates[selected_idx]
            selection = {
                "selected_candidate_idx": selected_idx,
                "selection_rule": "max(F/E, E, -candidate_idx)",
                "frontier_count": sum(
                    selected_idx in front
                    for front in state.program_at_pareto_front_valset.values()
                ),
                "clean_exposure": state.get_program_average_val_subset(
                    selected_idx
                )[1],
                "candidate": selected_candidate,
            }
            _atomic_write_json(run_dir / "selected_candidate.json", selection)

            validation_evaluation = run.adapter.evaluate(
                list(splits.validation),
                selected_candidate,
                capture_traces=False,
            )
            test_evaluation = run.adapter.evaluate(
                list(splits.test),
                selected_candidate,
                capture_traces=False,
            )
            completed_at = datetime.now(timezone.utc).isoformat()
            final_result = {
                "schema_version": 1,
                "status": "completed",
                "completed_at_utc": completed_at,
                "condition": config["condition"],
                "task_id": task_id,
                "optimizer_seed": config["optimizer_seed"],
                "selected_candidate_idx": selected_idx,
                "candidate_pool_size": len(state.program_candidates),
                "optimization_iterations": state.i + 1,
                "optimization_metric_calls": state.total_num_evals,
                "validation": _score_summary(validation_evaluation.scores),
                "test": _score_summary(test_evaluation.scores),
                "metric_counter": {
                    "total": counter.step_counter,
                    "train": counter.train_counter,
                    "validation": counter.val_counter,
                    "test": counter.test_counter,
                },
                "lm_usage": _lm_usage(lm),
                "wall_time_seconds": time.time() - start_wall,
            }
            _atomic_write_json(run_dir / "final_result.json", final_result)
            manifest["status"] = "completed"
            manifest["completed_at_utc"] = completed_at
            manifest["final_result"] = "final_result.json"
            _atomic_write_json(run_dir / "manifest.json", manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["exception"] = {
            "class": f"{error.__class__.__module__}.{error.__class__.__qualname__}",
            "message": str(error),
        }
        _atomic_write_json(run_dir / "manifest.json", manifest)
        raise
    finally:
        dspy.configure(lm=None, adapter=None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
