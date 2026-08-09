from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from gepa.core.state import GEPAState

from bridge.b19_reversible_parent_selection import evaluation_count, frontier_count
from bridge.b20_compass_reflection import SparseMinibatchEvaluationPolicy
from bridge.mechanism_benchmark_registry import resolve_mechanism_benchmark
from bridge.paper_benchmark_registry import canonical_feedback_map
from experiments.paper.run_compass_reflection import (
    _configure_run_cache,
    _create_lm,
    _lm_usage,
)


Identity = tuple[str, int]

CONFIG_KEYS = {
    "api_key_env",
    "cache_dir",
    "expected",
    "model",
    "num_threads",
    "output_dir",
    "schema_version",
    "source_run_dir",
}
EXPECTED_KEYS = {
    "candidate_pool_size",
    "candidate_sha256",
    "old_selected_candidate_idx",
    "optimization_metric_calls",
    "owner_selected_candidate_idx",
    "source_candidates_sha256",
    "source_config_sha256",
    "source_final_result_sha256",
    "source_gepa_state_sha256",
    "source_manifest_config_sha256",
    "source_manifest_sha256",
    "source_root_head",
    "split_fingerprints",
    "split_sizes",
    "task_id",
}
MODEL_KEYS = {
    "api_base",
    "cache",
    "cache_in_memory",
    "enable_thinking",
    "max_tokens",
    "model",
    "model_type",
    "num_retries",
    "temperature",
    "timeout",
    "top_k",
    "top_p",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_sha256(candidate: Mapping[str, str]) -> str:
    payload = json.dumps(
        dict(candidate),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _config_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _load_config(path: Path) -> dict[str, Any]:
    config = _exact_mapping(_read_json(path), name="config", keys=CONFIG_KEYS)
    if config["schema_version"] != 1:
        raise ValueError("config.schema_version must equal 1")
    config["expected"] = _exact_mapping(
        config["expected"],
        name="config.expected",
        keys=EXPECTED_KEYS,
    )
    config["model"] = _exact_mapping(
        config["model"],
        name="config.model",
        keys=MODEL_KEYS,
    )
    if config["model"]["cache"] is not True:
        raise ValueError("CLUTRR correction must preserve cache=true")
    if config["model"]["cache_in_memory"] is not True:
        raise ValueError("CLUTRR correction must preserve cache_in_memory=true")
    if config["model"]["num_retries"] != 0:
        raise ValueError("CLUTRR correction must preserve model.num_retries=0")
    threads = config["num_threads"]
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise TypeError("config.num_threads must be a positive integer")
    api_key_env = config["api_key_env"]
    if not isinstance(api_key_env, str) or not api_key_env:
        raise TypeError("config.api_key_env must be non-empty text")
    return config


def _verify_source_artifacts(
    config: Mapping[str, Any],
) -> tuple[GEPAState, dict[str, str], Mapping[str, Any]]:
    source = Path(config["source_run_dir"]).resolve(strict=True)
    expected = config["expected"]
    artifact_hashes = {
        "gepa_state.bin": expected["source_gepa_state_sha256"],
        "candidates.json": expected["source_candidates_sha256"],
        "config.json": expected["source_config_sha256"],
        "manifest.json": expected["source_manifest_sha256"],
        "final_result.json": expected["source_final_result_sha256"],
    }
    for filename, expected_sha256 in artifact_hashes.items():
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256_file(path) != expected_sha256:
            raise RuntimeError(f"source artifact changed: {path}")

    source_config = _read_json(source / "config.json")
    source_manifest = _read_json(source / "manifest.json")
    source_final = _read_json(source / "final_result.json")
    if source_manifest.get("status") != "completed":
        raise RuntimeError("source mechanism run is not completed")
    if source_final.get("status") != "completed":
        raise RuntimeError("source final result is not completed")
    if source_config.get("task_id") != expected["task_id"]:
        raise RuntimeError("source config task identity changed")
    if source_config.get("dataset_mode") != "lite":
        raise RuntimeError("source config is not the frozen lite protocol")
    source_snapshot = source_manifest.get("source_snapshot")
    if not isinstance(source_snapshot, Mapping):
        raise TypeError("source manifest has no source snapshot")
    if source_snapshot.get("root_head") != expected["source_root_head"]:
        raise RuntimeError("source root HEAD identity changed")
    if source_manifest.get("config_sha256") != expected[
        "source_manifest_config_sha256"
    ]:
        raise RuntimeError("source manifest config identity changed")

    task = source_manifest.get("task")
    if not isinstance(task, Mapping) or task.get("task_id") != expected["task_id"]:
        raise RuntimeError("source task manifest identity changed")
    if task.get("split_sizes") != expected["split_sizes"]:
        raise RuntimeError("source split sizes changed")
    if task.get("split_fingerprints") != expected["split_fingerprints"]:
        raise RuntimeError("source split fingerprints changed")
    if int(source_final.get("selected_candidate_idx", -1)) != expected[
        "old_selected_candidate_idx"
    ]:
        raise RuntimeError("source invalid selection record changed")
    if int(source_final.get("optimization_metric_calls", -1)) != expected[
        "optimization_metric_calls"
    ]:
        raise RuntimeError("source optimization budget evidence changed")

    state = GEPAState.load(str(source))
    if len(state.program_candidates) != expected["candidate_pool_size"]:
        raise RuntimeError("source candidate pool size changed")
    if int(state.total_num_evals) != expected["optimization_metric_calls"]:
        raise RuntimeError("source GEPA state metric-call count changed")
    selected_idx = int(SparseMinibatchEvaluationPolicy().get_best_program(state))
    if selected_idx != expected["owner_selected_candidate_idx"]:
        raise RuntimeError("owner evaluation policy selection changed")
    raw_candidate = state.program_candidates[selected_idx]
    if not isinstance(raw_candidate, Mapping):
        raise TypeError("owner-selected candidate is not a mapping")
    candidate = {str(name): str(instruction) for name, instruction in raw_candidate.items()}
    if _candidate_sha256(candidate) != expected["candidate_sha256"]:
        raise RuntimeError("owner-selected candidate content changed")
    return state, candidate, source_manifest


def _identity_record(identity: Identity) -> dict[str, Any]:
    split, position = identity
    return {"position": position, "split": split}


def _new_checkpoint(
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    candidate: Mapping[str, str],
) -> dict[str, Any]:
    expected = config["expected"]
    identities = {
        (split, position)
        for split in ("validation", "test")
        for position in range(int(expected["split_sizes"][split]))
    }
    return {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha256,
        "source_run_dir": str(Path(config["source_run_dir"]).resolve()),
        "source_gepa_state_sha256": expected["source_gepa_state_sha256"],
        "task_id": expected["task_id"],
        "split_sizes": dict(expected["split_sizes"]),
        "split_fingerprints": dict(expected["split_fingerprints"]),
        "selected_candidate_idx": expected["owner_selected_candidate_idx"],
        "candidate_sha256": _candidate_sha256(candidate),
        "retry_policy": "empty_prediction_until_complete",
        "selection_frozen_before_test": True,
        "test_used_for_selection": False,
        "rounds_completed": 0,
        "attempts": [],
        "scores": [],
        "predictions": [],
        "pending": [
            _identity_record(identity) for identity in sorted(identities)
        ],
        "rounds": [],
        "lm_usage": {
            "calls": 0,
            "cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
        "elapsed_seconds": 0.0,
    }


def _checkpoint_maps(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[Identity, int], dict[Identity, float], dict[Identity, str]]:
    attempts = {
        (str(item["split"]), int(item["position"])): int(item["attempts"])
        for item in checkpoint.get("attempts", [])
    }
    scores = {
        (str(item["split"]), int(item["position"])): float(item["score"])
        for item in checkpoint.get("scores", [])
    }
    predictions = {
        (str(item["split"]), int(item["position"])): str(item["relation"])
        for item in checkpoint.get("predictions", [])
    }
    if set(scores) != set(predictions):
        raise RuntimeError("checkpoint score/prediction identities differ")
    return attempts, scores, predictions


def _update_checkpoint(
    checkpoint: dict[str, Any],
    *,
    identities: set[Identity],
    attempts: Mapping[Identity, int],
    scores: Mapping[Identity, float],
    predictions: Mapping[Identity, str],
) -> None:
    if set(attempts).difference(identities):
        raise RuntimeError("checkpoint contains unknown attempt identities")
    if set(scores) != set(predictions):
        raise RuntimeError("checkpoint score/prediction identities differ")
    if set(scores).difference(identities):
        raise RuntimeError("checkpoint contains unknown score identities")
    checkpoint["attempts"] = [
        {**_identity_record(identity), "attempts": int(attempts[identity])}
        for identity in sorted(attempts)
    ]
    checkpoint["scores"] = [
        {**_identity_record(identity), "score": float(scores[identity])}
        for identity in sorted(scores)
    ]
    checkpoint["predictions"] = [
        {**_identity_record(identity), "relation": predictions[identity]}
        for identity in sorted(predictions)
    ]
    pending = identities.difference(scores)
    checkpoint["pending"] = [
        _identity_record(identity) for identity in sorted(pending)
    ]
    checkpoint["total_metric_calls"] = sum(attempts.values())
    checkpoint["retry_metric_calls"] = sum(
        max(0, attempts.get(identity, 0) - 1) for identity in identities
    )


def _prediction_succeeded(prediction: dspy.Prediction) -> bool:
    return "relation" in prediction and isinstance(prediction.relation, str)


def _numeric_score(value: Any, *, identity: Identity) -> float:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, Real):
        raise TypeError(f"owner metric returned non-numeric score for {identity!r}")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"owner metric returned non-finite score for {identity!r}")
    return score


def _usage_delta(
    current: Mapping[str, float | int],
    previous: Mapping[str, float | int],
) -> dict[str, float | int]:
    return {
        "calls": int(current["calls"]) - int(previous["calls"]),
        "cost": float(current["cost"]) - float(previous["cost"]),
        "input_tokens": int(current["input_tokens"])
        - int(previous["input_tokens"]),
        "output_tokens": int(current["output_tokens"])
        - int(previous["output_tokens"]),
    }


def _add_usage(
    aggregate: dict[str, float | int],
    delta: Mapping[str, float | int],
) -> None:
    aggregate["calls"] = int(aggregate["calls"]) + int(delta["calls"])
    aggregate["cost"] = float(aggregate["cost"]) + float(delta["cost"])
    aggregate["input_tokens"] = int(aggregate["input_tokens"]) + int(
        delta["input_tokens"]
    )
    aggregate["output_tokens"] = int(aggregate["output_tokens"]) + int(
        delta["output_tokens"]
    )


def _score_summary(
    split: str,
    *,
    size: int,
    attempts: Mapping[Identity, int],
    scores: Mapping[Identity, float],
) -> dict[str, Any]:
    values = [float(scores[(split, position)]) for position in range(size)]
    aggregate = math.fsum(values) / size
    split_calls = sum(attempts[(split, position)] for position in range(size))
    return {
        "aggregate_percent": 100.0 * aggregate,
        "count": size,
        "execution_failures_retried": sum(
            attempts[(split, position)] > 1 for position in range(size)
        ),
        "max_attempts": max(attempts[(split, position)] for position in range(size)),
        "retry_metric_calls": split_calls - size,
        "scores": values,
        "total_metric_calls": split_calls,
    }


def _validate_resume(
    checkpoint: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    candidate_sha256: str,
) -> None:
    expected = config["expected"]
    checks = {
        "config_sha256": config_sha256,
        "source_run_dir": str(Path(config["source_run_dir"]).resolve()),
        "source_gepa_state_sha256": expected["source_gepa_state_sha256"],
        "task_id": expected["task_id"],
        "split_sizes": expected["split_sizes"],
        "split_fingerprints": expected["split_fingerprints"],
        "selected_candidate_idx": expected["owner_selected_candidate_idx"],
        "candidate_sha256": candidate_sha256,
    }
    for key, expected_value in checks.items():
        if checkpoint.get(key) != expected_value:
            raise RuntimeError(f"resume identity mismatch: {key}")
    if checkpoint.get("status") == "completed":
        raise RuntimeError("completed CLUTRR correction cannot be resumed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve(strict=True)
    config = _load_config(config_path)
    config_sha256 = _config_sha256(config)
    state, candidate, source_manifest = _verify_source_artifacts(config)
    expected = config["expected"]

    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"unset API-key environment: {config['api_key_env']}"
        )
    _configure_run_cache(Path(config["cache_dir"]).resolve(), config["model"])
    lm = _create_lm(config["model"], api_key=api_key)
    dspy.configure(lm=lm, adapter=ChatAdapter())
    spec = resolve_mechanism_benchmark(
        expected["task_id"],
        lm=lm,
        dataset_mode="lite",
    )
    splits = {
        "validation": tuple(spec.splits.validation),
        "test": tuple(spec.splits.test),
    }
    if dict(spec.splits.fingerprints) != expected["split_fingerprints"]:
        raise RuntimeError("current pinned owner split fingerprints changed")
    for split, examples in splits.items():
        if len(examples) != int(expected["split_sizes"][split]):
            raise RuntimeError(f"current pinned owner {split} size changed")

    adapter = DspyAdapter(
        student_module=spec.program,
        metric_fn=spec.benchmark_meta.metric,
        feedback_map=canonical_feedback_map(spec),
        failure_score=0.0,
        num_threads=config["num_threads"],
        raise_on_error=False,
    )
    program = adapter.build_program(candidate)

    output_dir = Path(config["output_dir"]).resolve()
    checkpoint_path = output_dir / "evaluation_checkpoint.json"
    manifest_path = output_dir / "manifest.json"
    if args.resume:
        if not checkpoint_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("--resume requires checkpoint and manifest")
        checkpoint = _read_json(checkpoint_path)
        _validate_resume(
            checkpoint,
            config=config,
            config_sha256=config_sha256,
            candidate_sha256=_candidate_sha256(candidate),
        )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = _new_checkpoint(
            config=config,
            config_sha256=config_sha256,
            candidate=candidate,
        )
        _write_json(checkpoint_path, checkpoint)
        _write_json(output_dir / "config.json", config)
        _write_json(
            manifest_path,
            {
                "schema_version": 1,
                "status": "running",
                "started_at_utc": checkpoint["started_at_utc"],
                "config_path": str(config_path),
                "config_sha256": config_sha256,
                "source_run_dir": str(Path(config["source_run_dir"]).resolve()),
                "source_manifest_sha256": expected["source_manifest_sha256"],
                "source_gepa_state_sha256": expected[
                    "source_gepa_state_sha256"
                ],
                "source_root_head": expected["source_root_head"],
                "task": source_manifest["task"],
                "old_selected_candidate_idx": expected[
                    "old_selected_candidate_idx"
                ],
                "selected_candidate_idx": expected[
                    "owner_selected_candidate_idx"
                ],
                "candidate_sha256": _candidate_sha256(candidate),
                "candidate_pool_size": len(state.program_candidates),
                "optimization_metric_calls": state.total_num_evals,
                "owner_selection": {
                    "frontier_count": frontier_count(
                        state, expected["owner_selected_candidate_idx"]
                    ),
                    "clean_exposure": evaluation_count(
                        state, expected["owner_selected_candidate_idx"]
                    ),
                    "selection_rule": (
                        "owner SparseMinibatchEvaluationPolicy.get_best_program"
                    ),
                },
                "model": dict(config["model"]),
                "api_key_env_name": config["api_key_env"],
                "num_threads": config["num_threads"],
                "retry_policy": "empty_prediction_until_complete",
                "selection_frozen_before_test": True,
                "test_used_for_selection": False,
            },
        )

    identities = {
        (split, position)
        for split, examples in splits.items()
        for position in range(len(examples))
    }
    attempts, scores, predictions = _checkpoint_maps(checkpoint)
    if set(attempts).difference(identities) or set(scores).difference(identities):
        raise RuntimeError("checkpoint contains an unknown CLUTRR identity")

    process_started = time.time()
    previous_usage = _lm_usage(lm)
    try:
        while identities.difference(scores):
            round_number = int(checkpoint["rounds_completed"]) + 1
            attempted = 0
            succeeded = 0
            round_started = time.time()
            for split in ("validation", "test"):
                positions = sorted(
                    position
                    for current_split, position in identities.difference(scores)
                    if current_split == split
                )
                if not positions:
                    continue
                print(
                    "CLUTRR_OWNER_FINAL_BATCH",
                    f"round={round_number}",
                    f"split={split}",
                    f"pending={len(positions)}",
                    flush=True,
                )
                batch = [splits[split][position] for position in positions]
                evaluator = dspy.Evaluate(
                    devset=batch,
                    metric=spec.benchmark_meta.metric,
                    num_threads=config["num_threads"],
                    display_progress=True,
                    max_errors=max(1, len(batch) * 100),
                    provide_traceback=True,
                    failure_score=0.0,
                    timeout=0,
                    straggler_limit=0,
                )
                evaluation = evaluator(program)
                if len(evaluation.results) != len(positions):
                    raise RuntimeError("official DSPy Evaluate returned incomplete results")
                for position, result in zip(
                    positions, evaluation.results, strict=True
                ):
                    example, prediction, raw_score = result
                    if example is not splits[split][position]:
                        raise RuntimeError("official DSPy Evaluate changed alignment")
                    identity = (split, position)
                    attempts[identity] = attempts.get(identity, 0) + 1
                    attempted += 1
                    if _prediction_succeeded(prediction):
                        scores[identity] = _numeric_score(
                            raw_score, identity=identity
                        )
                        predictions[identity] = prediction.relation
                        succeeded += 1

            current_usage = _lm_usage(lm)
            _add_usage(
                checkpoint["lm_usage"],
                _usage_delta(current_usage, previous_usage),
            )
            previous_usage = current_usage
            checkpoint["elapsed_seconds"] = float(
                checkpoint.get("elapsed_seconds", 0.0)
            ) + (time.time() - process_started)
            process_started = time.time()
            checkpoint["rounds_completed"] = round_number
            checkpoint["rounds"].append(
                {
                    "attempted": attempted,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": time.time() - round_started,
                    "remaining": len(identities.difference(scores)),
                    "round": round_number,
                    "succeeded": succeeded,
                }
            )
            _update_checkpoint(
                checkpoint,
                identities=identities,
                attempts=attempts,
                scores=scores,
                predictions=predictions,
            )
            _write_json(checkpoint_path, checkpoint)
            print(
                "CLUTRR_OWNER_FINAL_ROUND",
                f"round={round_number}",
                f"succeeded={succeeded}",
                f"remaining={len(identities.difference(scores))}",
                flush=True,
            )

        validation = _score_summary(
            "validation",
            size=len(splits["validation"]),
            attempts=attempts,
            scores=scores,
        )
        test = _score_summary(
            "test",
            size=len(splits["test"]),
            attempts=attempts,
            scores=scores,
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        score_breakdown = spec.score_breakdown
        if score_breakdown is None:
            raise RuntimeError("pinned CLUTRR benchmark has no score breakdown")
        final_result = {
            "schema_version": 1,
            "status": "completed",
            "completed_at_utc": completed_at,
            "task_id": expected["task_id"],
            "source_run_dir": str(Path(config["source_run_dir"]).resolve()),
            "source_final_result_sha256": expected[
                "source_final_result_sha256"
            ],
            "source_gepa_state_sha256": expected["source_gepa_state_sha256"],
            "old_selected_candidate_idx": expected[
                "old_selected_candidate_idx"
            ],
            "selected_candidate_idx": expected[
                "owner_selected_candidate_idx"
            ],
            "candidate_sha256": _candidate_sha256(candidate),
            "candidate_pool_size": len(state.program_candidates),
            "optimization_metric_calls": state.total_num_evals,
            "selection_frozen_before_test": True,
            "test_used_for_selection": False,
            "remaining_execution_failure_count": 0,
            "retry_metric_calls": checkpoint["retry_metric_calls"],
            "retry_rounds": checkpoint["rounds_completed"],
            "validation": validation,
            "test": test,
            "task_score_breakdown": {
                "validation": score_breakdown(
                    splits["validation"], validation["scores"]
                ),
                "test": score_breakdown(
                    splits["test"], test["scores"]
                ),
            },
            "lm_usage": dict(checkpoint["lm_usage"]),
            "elapsed_seconds": checkpoint["elapsed_seconds"],
        }
        _write_json(output_dir / "final_result.json", final_result)
        checkpoint["status"] = "completed"
        checkpoint["completed_at_utc"] = completed_at
        _write_json(checkpoint_path, checkpoint)
        manifest = _read_json(manifest_path)
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = completed_at
        manifest["final_result"] = "final_result.json"
        _write_json(manifest_path, manifest)
    except BaseException as error:
        manifest = _read_json(manifest_path)
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["exception"] = {
            "class": f"{error.__class__.__module__}.{error.__class__.__qualname__}",
            "message": str(error),
        }
        _write_json(manifest_path, manifest)
        raise
    finally:
        dspy.configure(lm=None, adapter=None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
