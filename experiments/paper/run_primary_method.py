"""Run one create-only Seed, DSPy MIPROv2, or native DSPy GEPA cell.

M0 and COMPASS remain owned by ``run_compass_reflection.py`` and are emitted
to that runner by the primary matrix generator.  This file does not copy their
optimizer loop.  It exists only for the three program-level owner methods that
the existing COMPASS runner does not execute.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import dspy
from dspy.adapters.chat_adapter import ChatAdapter

from bridge.primary_method_protocol import (
    canonical_sha256,
    make_gepa_feedback_metric,
    mipro_protocol,
    validate_primary_cell,
)
from experiments.paper.run_compass_reflection import (
    _atomic_write_json,
    _configure_run_cache,
    _create_lm,
    _lm_usage,
    _score_summary,
    _verify_local_source_snapshot,
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
CONFIG_KEYS: Final = {
    "cache_dir",
    "dataset_mode",
    "logical_rollout_budget",
    "matrix_id",
    "method",
    "model",
    "model_profile_sha256",
    "num_threads",
    "optimizer_seed",
    "phase",
    "run_dir",
    "schema_version",
    "source_snapshot",
    "task_id",
}
MODEL_KEYS: Final = {
    "api_base",
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
MODEL_OPTIONAL_KEYS: Final = {"timeout"}


def _exact_mapping(value: Any, *, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    missing = keys.difference(value)
    extra = set(value).difference(keys)
    if missing or extra:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return dict(value)


def load_primary_config(path: Path) -> dict[str, Any]:
    """Load one strict, credential-free primary baseline config."""

    config = _exact_mapping(
        json.loads(path.read_text(encoding="utf-8")),
        name="config",
        keys=CONFIG_KEYS,
    )
    if config["schema_version"] != 1:
        raise ValueError("primary config schema_version must equal 1")
    if not isinstance(config["matrix_id"], str) or not config["matrix_id"]:
        raise TypeError("matrix_id must be non-empty text")
    if config["dataset_mode"] != "lite":
        raise ValueError("primary runner requires dataset_mode='lite'")
    cell = validate_primary_cell(
        task_id=config["task_id"],
        method=config["method"],
        phase=config["phase"],
        optimizer_seed=config["optimizer_seed"],
        logical_rollout_budget=config["logical_rollout_budget"],
    )
    if cell.method not in {"seed", "mipro", "gepa"}:
        raise ValueError(
            f"{cell.method} is owned by experiments/paper/run_compass_reflection.py"
        )
    threads = config["num_threads"]
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise TypeError("num_threads must be a positive integer")

    model = _exact_mapping(
        config["model"],
        name="model",
        keys=set(config["model"]),
    )
    missing_model = MODEL_KEYS.difference(model)
    extra_model = set(model).difference(MODEL_KEYS | MODEL_OPTIONAL_KEYS)
    if missing_model or extra_model:
        raise ValueError(
            "model keys mismatch; "
            f"missing={sorted(missing_model)}, extra={sorted(extra_model)}"
        )
    if model["cache"] is not True or model["cache_in_memory"] is not True:
        raise ValueError("formal method caches must use DSPy's isolated cache owner")
    if model["num_retries"] != 0:
        raise ValueError("model.num_retries must remain owner-frozen at zero")
    if not isinstance(model["api_key_env"], str) or not model["api_key_env"]:
        raise TypeError("model.api_key_env must be non-empty text")
    for name in ("model", "model_type"):
        if not isinstance(model[name], str) or not model[name]:
            raise TypeError(f"model.{name} must be non-empty text")
    if canonical_sha256(model) != config["model_profile_sha256"]:
        raise RuntimeError("model profile hash differs from the frozen config")
    config["model"] = model

    source_snapshot = config["source_snapshot"]
    if not isinstance(source_snapshot, Mapping):
        raise TypeError("source_snapshot must be a JSON object")
    for name in ("run_dir", "cache_dir"):
        if not isinstance(config[name], str) or not config[name]:
            raise TypeError(f"{name} must be non-empty text")
    if Path(config["run_dir"]).resolve() == Path(config["cache_dir"]).resolve():
        raise ValueError("run_dir and cache_dir must be distinct")
    return config


def _require_source_identity(snapshot: Mapping[str, Any]) -> None:
    """Check the root revision and the explicitly frozen semantic files."""

    expected_head = snapshot.get("root_head")
    if not isinstance(expected_head, str) or len(expected_head) != 40:
        raise TypeError("source_snapshot.root_head must be a Git commit")
    actual_head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_head != expected_head:
        raise RuntimeError(
            "source revision differs from the frozen primary matrix: "
            f"expected={expected_head}, actual={actual_head}"
        )
    _verify_local_source_snapshot(snapshot)


def _resolve_benchmark(config: Mapping[str, Any], *, lm: dspy.BaseLM) -> Any:
    task_id = config["task_id"]
    if task_id == "chartqa":
        from bridge.mechanism_benchmark_registry import (
            resolve_mechanism_benchmark,
        )

        return resolve_mechanism_benchmark(
            "chartqa",
            lm=lm,
            dataset_mode=config["dataset_mode"],
        )

    from bridge.paper_benchmark_registry import (
        instantiate_official_splits,
        load_official_benchmark_specs,
    )

    spec = load_official_benchmark_specs()[task_id]
    splits = instantiate_official_splits(
        spec,
        optimizer_seed=config["optimizer_seed"],
        dataset_mode=config["dataset_mode"],
    )
    # Keep the official spec immutable; this thin view only binds its owner
    # splits so all three method runners share one shape.
    return _ResolvedOfficialBenchmark(spec, splits)


class _ResolvedOfficialBenchmark:
    def __init__(self, spec: Any, splits: Any) -> None:
        self._spec = spec
        self.task_id = spec.task_id
        self.splits = splits
        self.program = spec.program
        self.metric = spec.benchmark_meta.metric
        self.metric_with_feedback = spec.benchmark_meta.metric_with_feedback
        self.benchmark_meta = spec.benchmark_meta
        self.display_name = spec.display_name
        self.benchmark_class_name = spec.benchmark_class_name
        self.program_class_name = spec.program_class_name
        self.predictor_names = spec.predictor_names
        self.custom_instruction_proposer = None
        self.score_breakdown = None
        self.num_threads = spec.benchmark_meta.num_threads
        self.program_index = spec.program_index


def _optimization_count(counter: Any) -> int:
    return int(counter.train_counter) + int(counter.val_counter)


def _latest_mipro_trial_rollouts(program: Any) -> int:
    trial_logs = getattr(program, "trial_logs", None)
    if not isinstance(trial_logs, Mapping) or not trial_logs:
        raise RuntimeError("official MIPROv2 did not expose trial_logs")
    counts = [
        int(record["total_eval_calls_so_far"])
        for record in trial_logs.values()
        if isinstance(record, Mapping) and "total_eval_calls_so_far" in record
    ]
    if not counts:
        raise RuntimeError("official MIPROv2 trial_logs contain no rollout counts")
    return max(counts)


def _compile_mipro(
    *,
    benchmark: Any,
    metric: Any,
    lm: dspy.BaseLM,
    config: Mapping[str, Any],
    run_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    cell = validate_primary_cell(
        task_id=config["task_id"],
        method="mipro",
        phase=config["phase"],
        optimizer_seed=config["optimizer_seed"],
        logical_rollout_budget=config["logical_rollout_budget"],
    )
    protocol = mipro_protocol(cell)
    init = protocol["init"]
    optimizer = dspy.MIPROv2(
        metric=metric,
        prompt_model=lm,
        task_model=lm,
        auto=init["auto"],
        num_candidates=init.get("num_candidates"),
        num_threads=config["num_threads"],
        max_errors=(len(benchmark.splits.train) + len(benchmark.splits.validation))
        * 100,
        seed=init["seed"],
        log_dir=str(run_dir / "optimizer_logs"),
        track_stats=True,
    )
    optimized = optimizer.compile(
        benchmark.program,
        trainset=list(benchmark.splits.train),
        valset=list(benchmark.splits.validation),
        **protocol["compile"],
    )
    trial_rollouts = _latest_mipro_trial_rollouts(optimized)
    if cell.phase == "preflight" and trial_rollouts < cell.logical_rollout_budget:
        raise RuntimeError(
            "MIPROv2 preflight did not reach its minimum rollout coverage: "
            f"required={cell.logical_rollout_budget}, actual={trial_rollouts}"
        )
    return optimized, {
        "owner": "dspy.MIPROv2",
        "protocol": protocol,
        "trial_rollouts": trial_rollouts,
    }


def _compile_gepa(
    *,
    benchmark: Any,
    metric: Any,
    lm: dspy.BaseLM,
    config: Mapping[str, Any],
    run_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    gepa_metric = make_gepa_feedback_metric(benchmark, scalar_metric=metric)
    optimizer = dspy.GEPA(
        metric=gepa_metric,
        reflection_lm=lm,
        max_metric_calls=config["logical_rollout_budget"],
        reflection_minibatch_size=3,
        candidate_selection_strategy="pareto",
        skip_perfect_score=True,
        add_format_failure_as_feedback=False,
        instruction_proposer=getattr(
            benchmark,
            "custom_instruction_proposer",
            None,
        ),
        use_merge=False,
        max_merge_invocations=None,
        num_threads=config["num_threads"],
        failure_score=0.0,
        perfect_score=1.0,
        log_dir=str(run_dir),
        track_stats=True,
        use_wandb=False,
        track_best_outputs=True,
        seed=config["optimizer_seed"],
        gepa_kwargs={"use_cloudpickle": True},
    )
    optimized = optimizer.compile(
        benchmark.program,
        trainset=list(benchmark.splits.train),
        valset=list(benchmark.splits.validation),
    )
    details = getattr(optimized, "detailed_results", None)
    actual = getattr(details, "total_metric_calls", None)
    if actual is None:
        raise RuntimeError("native DSPy GEPA did not expose total_metric_calls")
    actual = int(actual)
    expected = int(config["logical_rollout_budget"])
    if actual != expected:
        raise RuntimeError(
            "native GEPA did not exactly consume the frozen rollout allocation; "
            "no filler evaluations are permitted: "
            f"expected={expected}, actual={actual}"
        )
    return optimized, {
        "owner": "dspy.GEPA",
        "total_metric_calls": actual,
        "use_merge": False,
        "custom_instruction_proposer": (
            type(getattr(benchmark, "custom_instruction_proposer", None)).__name__
            if getattr(benchmark, "custom_instruction_proposer", None) is not None
            else None
        ),
    }


def _evaluate_program(
    *,
    program: Any,
    examples: Sequence[Any],
    metric: Any,
    num_threads: int,
) -> dict[str, Any]:
    result = dspy.Evaluate(
        devset=list(examples),
        metric=metric,
        num_threads=num_threads,
        display_progress=False,
        display_table=False,
        max_errors=max(1, len(examples) * 10),
        provide_traceback=True,
        failure_score=0.0,
        timeout=0,
        straggler_limit=0,
    )(program)
    return _score_summary([score for _, _, score in result.results])


def run(config_path: Path) -> int:
    from gepa_artifact.utils.metric_logger import CounterWithLock, MetricWithLogger

    config = load_primary_config(config_path)
    _require_source_identity(config["source_snapshot"])
    run_dir = Path(config["run_dir"]).resolve()
    cache_dir = Path(config["cache_dir"]).resolve()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    if cache_dir.exists():
        raise FileExistsError(cache_dir)

    api_key_env = config["model"]["api_key_env"]
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"required API key environment variable is unset: {api_key_env}"
        )

    _configure_run_cache(cache_dir, config["model"])
    lm = _create_lm(config["model"], api_key=api_key)
    dspy.configure(lm=lm, adapter=ChatAdapter())
    benchmark = _resolve_benchmark(config, lm=lm)
    owner_cap = getattr(benchmark, "num_threads", None)
    effective_threads = (
        min(config["num_threads"], owner_cap)
        if owner_cap is not None
        else config["num_threads"]
    )
    task_manifest = {
        "task_id": config["task_id"],
        "display_name": benchmark.display_name,
        "benchmark_class": benchmark.benchmark_class_name,
        "program_class": benchmark.program_class_name,
        "predictor_names": list(benchmark.predictor_names),
        "dataset_mode": config["dataset_mode"],
        "split_sizes": {
            "train": len(benchmark.splits.train),
            "validation": len(benchmark.splits.validation),
            "test": len(benchmark.splits.test),
        },
        "split_fingerprints": dict(benchmark.splits.fingerprints),
    }
    provenance = getattr(benchmark, "provenance", None)
    if isinstance(provenance, Mapping):
        task_manifest["owner_provenance"] = dict(provenance)

    run_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": started,
        "matrix_id": config["matrix_id"],
        "phase": config["phase"],
        "method": config["method"],
        "config_path": str(config_path),
        "config_sha256": canonical_sha256(config),
        "model_profile_sha256": config["model_profile_sha256"],
        "source_snapshot_sha256": canonical_sha256(config["source_snapshot"]),
        "task": task_manifest,
        "logical_rollout_budget": config["logical_rollout_budget"],
        "num_threads": {
            "requested": config["num_threads"],
            "owner_cap": owner_cap,
            "effective": effective_threads,
        },
        "run_dir": str(run_dir),
        "cache_dir": str(cache_dir),
        "api_key_env_name": api_key_env,
        "model": {
            key: value for key, value in config["model"].items() if key != "api_key_env"
        },
        "runtime": {"python": sys.version, "platform": platform.platform()},
    }
    _atomic_write_json(run_dir / "config.json", config)
    _atomic_write_json(run_dir / "manifest.json", manifest)

    counter = CounterWithLock()
    start_wall = time.time()
    try:
        with MetricWithLogger(
            metric_fn=benchmark.metric,
            run_dir=str(run_dir),
            counter_with_lock=counter,
            train_dataset=list(benchmark.splits.train),
            val_dataset=list(benchmark.splits.validation),
            test_dataset=list(benchmark.splits.test),
            log_trace=False,
            log_example=False,
            log_prediction=True,
        ) as metric:
            if config["method"] == "seed":
                optimized = benchmark.program
                method_result = {"owner": "unoptimized owner program"}
            elif config["method"] == "mipro":
                optimized, method_result = _compile_mipro(
                    benchmark=benchmark,
                    metric=metric,
                    lm=lm,
                    config=config,
                    run_dir=run_dir,
                )
            else:
                optimized, method_result = _compile_gepa(
                    benchmark=benchmark,
                    metric=metric,
                    lm=lm,
                    config=config,
                    run_dir=run_dir,
                )

            optimization_metric_calls = _optimization_count(counter)
            if config["method"] != "seed":
                expected = int(config["logical_rollout_budget"])
                if (
                    config["phase"] == "formal"
                    and optimization_metric_calls != expected
                ):
                    raise RuntimeError(
                        "formal optimization did not exactly consume its frozen "
                        "logical-rollout allocation; no filler evaluations are "
                        f"permitted: expected={expected}, "
                        f"actual={optimization_metric_calls}"
                    )
                if (
                    config["phase"] == "preflight"
                    and optimization_metric_calls < expected
                ):
                    raise RuntimeError(
                        "preflight did not reach its minimum logical-rollout "
                        f"coverage: required={expected}, "
                        f"actual={optimization_metric_calls}"
                    )
            result: dict[str, Any] = {
                "schema_version": 1,
                "status": (
                    "completed"
                    if config["phase"] == "formal"
                    else "preflight_completed"
                ),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "task_id": config["task_id"],
                "method": config["method"],
                "phase": config["phase"],
                "optimizer_seed": config["optimizer_seed"],
                "logical_rollout_budget": config["logical_rollout_budget"],
                "optimization_metric_calls_logged": optimization_metric_calls,
                "method_result": method_result,
                "lm_usage": _lm_usage(lm),
                "wall_time_seconds": time.time() - start_wall,
            }
            if config["phase"] == "formal":
                result["validation"] = _evaluate_program(
                    program=optimized,
                    examples=benchmark.splits.validation,
                    metric=metric,
                    num_threads=effective_threads,
                )
                result["test"] = _evaluate_program(
                    program=optimized,
                    examples=benchmark.splits.test,
                    metric=metric,
                    num_threads=effective_threads,
                )
                score_breakdown = getattr(benchmark, "score_breakdown", None)
                if score_breakdown is not None:
                    result["task_score_breakdown"] = {
                        "validation": score_breakdown(
                            benchmark.splits.validation,
                            result["validation"]["scores"],
                        ),
                        "test": score_breakdown(
                            benchmark.splits.test,
                            result["test"]["scores"],
                        ),
                    }
            if config["method"] != "seed":
                optimized.save(str(run_dir / "optimized_program"), save_program=True)
            result["metric_counter"] = {
                "total": counter.step_counter,
                "train": counter.train_counter,
                "validation": counter.val_counter,
                "test": counter.test_counter,
            }
            _atomic_write_json(run_dir / "final_result.json", result)
            manifest["status"] = result["status"]
            manifest["completed_at_utc"] = result["completed_at_utc"]
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate identity and owner support without loading data or calling an LM",
    )
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = load_primary_config(config_path)
    _require_source_identity(config["source_snapshot"])
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "task_id": config["task_id"],
                    "method": config["method"],
                    "phase": config["phase"],
                    "config_sha256": canonical_sha256(config),
                },
                sort_keys=True,
            )
        )
        return 0
    return run(config_path)


if __name__ == "__main__":
    raise SystemExit(main())
