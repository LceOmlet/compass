from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

from bridge.paper_benchmark_registry import (
    canonical_feedback_map,
    instantiate_official_splits,
    load_official_benchmark_specs,
)
from experiments.paper.run_compass_reflection import _create_lm


Identity = tuple[str, int]


@dataclass(frozen=True, slots=True)
class FrozenMethod:
    version: str
    run_dir: Path
    selected_candidate_idx: int
    candidate: Mapping[str, str]
    candidate_sha256: str
    selection_sha256: str
    model_config: Mapping[str, Any]
    api_key_env: str
    test_split_fingerprint: str


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_sha256(candidate: Mapping[str, str]) -> str:
    return _sha256_bytes(
        json.dumps(
            dict(candidate),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--method must be VERSION=RUN_DIR")
    version, raw_path = value.split("=", 1)
    version = version.strip()
    raw_path = raw_path.strip()
    if not version or not raw_path:
        raise argparse.ArgumentTypeError("--method must be VERSION=RUN_DIR")
    return version, Path(raw_path)


def _load_method(
    version: str,
    run_dir: Path,
    *,
    predictor_names: Sequence[str],
) -> FrozenMethod:
    run_dir = run_dir.resolve(strict=True)
    selection_path = run_dir / "selected_candidate.json"
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.json"
    for path in (selection_path, manifest_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    selection = _read_json(selection_path)
    manifest = _read_json(manifest_path)
    config = _read_json(config_path)
    if config.get("task_id") != "aime_2025":
        raise ValueError(f"{version} is not an AIME-2025 run")
    if config.get("dataset_mode") != "lite":
        raise ValueError(f"{version} does not use the frozen lite dataset")
    if config.get("optimizer_seed") != 0:
        raise ValueError(f"{version} does not use optimizer_seed=0")

    selected_idx = selection.get("selected_candidate_idx")
    if isinstance(selected_idx, bool) or not isinstance(selected_idx, int):
        raise TypeError(f"{version} has no integer selected candidate index")
    candidate_raw = selection.get("candidate")
    if not isinstance(candidate_raw, Mapping):
        raise TypeError(f"{version} selected candidate is not a mapping")
    candidate = dict(candidate_raw)
    if set(candidate) != set(predictor_names) or any(
        not isinstance(name, str)
        or not isinstance(instruction, str)
        or not instruction
        for name, instruction in candidate.items()
    ):
        raise ValueError(
            f"{version} selected candidate does not match official predictors"
        )

    task = manifest.get("task")
    if not isinstance(task, Mapping):
        raise ValueError(f"{version} manifest has no task record")
    fingerprints = task.get("split_fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise ValueError(f"{version} manifest has no split fingerprints")
    test_fingerprint = fingerprints.get("test")
    if not isinstance(test_fingerprint, str) or not test_fingerprint:
        raise ValueError(f"{version} manifest has no test fingerprint")

    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise TypeError(f"{version} model config is not a mapping")
    api_key_env = model_config.get("api_key_env")
    if not isinstance(api_key_env, str) or not api_key_env:
        raise ValueError(f"{version} has no API-key environment name")

    return FrozenMethod(
        version=version,
        run_dir=run_dir,
        selected_candidate_idx=selected_idx,
        candidate=candidate,
        candidate_sha256=_candidate_sha256(candidate),
        selection_sha256=_sha256_file(selection_path),
        model_config=dict(model_config),
        api_key_env=api_key_env,
        test_split_fingerprint=test_fingerprint,
    )


def _identity_record(identity: Identity) -> dict[str, Any]:
    version, test_position = identity
    return {"version": version, "test_position": test_position}


def _identity_sort_key(identity: Identity) -> tuple[str, int]:
    return identity


def _new_checkpoint(
    *,
    methods: Sequence[FrozenMethod],
    test_size: int,
    test_split_fingerprint: str,
) -> dict[str, Any]:
    identities = {
        (method.version, test_position)
        for method in methods
        for test_position in range(test_size)
    }
    return {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "retry_policy": "execution_failures_until_numeric_success",
        "selection_frozen_before_test": True,
        "test_used_for_selection": False,
        "dataset": "AIME-2025",
        "dataset_mode": "lite",
        "split": "official_test",
        "test_size": test_size,
        "test_split_fingerprint": test_split_fingerprint,
        "methods": [
            {
                "version": method.version,
                "run_dir": str(method.run_dir),
                "selected_candidate_idx": method.selected_candidate_idx,
                "candidate_sha256": method.candidate_sha256,
                "selection_sha256": method.selection_sha256,
            }
            for method in methods
        ],
        "rounds_completed": 0,
        "attempts": [],
        "scores": [],
        "pending": [
            _identity_record(identity)
            for identity in sorted(identities, key=_identity_sort_key)
        ],
        "method_batches": [],
        "rounds": [],
    }


def _checkpoint_maps(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[Identity, int], dict[Identity, float]]:
    attempts = {
        (str(item["version"]), int(item["test_position"])): int(item["attempts"])
        for item in checkpoint.get("attempts", [])
    }
    scores = {
        (str(item["version"]), int(item["test_position"])): float(item["score"])
        for item in checkpoint.get("scores", [])
    }
    return attempts, scores


def _update_checkpoint(
    checkpoint: dict[str, Any],
    *,
    identities: set[Identity],
    attempts: Mapping[Identity, int],
    scores: Mapping[Identity, float],
) -> None:
    if set(attempts).difference(identities):
        raise ValueError("checkpoint attempts contain an unknown identity")
    if set(scores).difference(identities):
        raise ValueError("checkpoint scores contain an unknown identity")
    pending = identities.difference(scores)
    checkpoint["attempts"] = [
        {
            **_identity_record(identity),
            "attempts": int(attempts.get(identity, 0)),
        }
        for identity in sorted(identities, key=_identity_sort_key)
        if attempts.get(identity, 0)
    ]
    checkpoint["scores"] = [
        {
            **_identity_record(identity),
            "score": float(scores[identity]),
        }
        for identity in sorted(scores, key=_identity_sort_key)
    ]
    checkpoint["pending"] = [
        _identity_record(identity)
        for identity in sorted(pending, key=_identity_sort_key)
    ]
    checkpoint["total_metric_calls"] = sum(attempts.values())
    checkpoint["retry_metric_calls"] = sum(
        max(0, attempts.get(identity, 0) - 1) for identity in identities
    )


def _prediction_succeeded(prediction: dspy.Prediction) -> bool:
    return "answer" in prediction


def _numeric_score(value: Any, *, identity: Identity) -> float:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, Real):
        raise TypeError(f"official metric returned non-numeric score for {identity!r}")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"official metric returned non-finite score for {identity!r}")
    return score


def _validate_resume(
    checkpoint: Mapping[str, Any],
    *,
    methods: Sequence[FrozenMethod],
    test_size: int,
    test_split_fingerprint: str,
) -> None:
    expected_methods = [
        {
            "version": method.version,
            "run_dir": str(method.run_dir),
            "selected_candidate_idx": method.selected_candidate_idx,
            "candidate_sha256": method.candidate_sha256,
            "selection_sha256": method.selection_sha256,
        }
        for method in methods
    ]
    if checkpoint.get("methods") != expected_methods:
        raise ValueError("frozen method artifacts changed after evaluation started")
    if int(checkpoint.get("test_size", -1)) != test_size:
        raise ValueError("official AIME test size changed")
    if checkpoint.get("test_split_fingerprint") != test_split_fingerprint:
        raise ValueError("official AIME test split changed")


def _final_records(
    *,
    methods: Sequence[FrozenMethod],
    test_size: int,
    attempts: Mapping[Identity, int],
    scores: Mapping[Identity, float],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for method in methods:
        per_instance = [
            {
                "test_position": position,
                "score": float(scores[(method.version, position)]),
                "attempts": int(attempts[(method.version, position)]),
            }
            for position in range(test_size)
        ]
        aggregate = math.fsum(item["score"] for item in per_instance) / test_size
        total_calls = sum(item["attempts"] for item in per_instance)
        records.append(
            {
                "version": method.version,
                "run_dir": str(method.run_dir),
                "selected_candidate_idx": method.selected_candidate_idx,
                "candidate_sha256": method.candidate_sha256,
                "test_score": aggregate,
                "test_score_percent": 100.0 * aggregate,
                "test_size": test_size,
                "total_metric_calls": total_calls,
                "retry_metric_calls": total_calls - test_size,
                "execution_failures_retried": sum(
                    item["attempts"] > 1 for item in per_instance
                ),
                "max_attempts": max(item["attempts"] for item in per_instance),
                "per_instance_scores": per_instance,
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        type=_parse_method,
        metavar="VERSION=RUN_DIR",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-threads", default=32, type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.num_threads <= 0:
        raise ValueError("--num-threads must be positive")

    raw_methods: list[tuple[str, Path]] = args.method
    versions = [version for version, _ in raw_methods]
    if len(set(versions)) != len(versions):
        raise ValueError("--method versions must be unique")

    spec = load_official_benchmark_specs()["aime_2025"]
    methods = [
        _load_method(version, run_dir, predictor_names=spec.predictor_names)
        for version, run_dir in raw_methods
    ]
    first_model = dict(methods[0].model_config)
    first_api_key_env = methods[0].api_key_env
    first_test_fingerprint = methods[0].test_split_fingerprint
    for method in methods[1:]:
        if dict(method.model_config) != first_model:
            raise ValueError("frozen methods use different model configurations")
        if method.api_key_env != first_api_key_env:
            raise ValueError("frozen methods use different API-key environments")
        if method.test_split_fingerprint != first_test_fingerprint:
            raise ValueError("frozen methods recorded different AIME test splits")

    splits = instantiate_official_splits(
        spec,
        optimizer_seed=0,
        dataset_mode="lite",
    )
    test = tuple(splits.test)
    if splits.fingerprints["test"] != first_test_fingerprint:
        raise ValueError("current official AIME test split differs from the runs")
    test_size = len(test)
    if not test_size:
        raise ValueError("official AIME test split is empty")

    output_dir = args.output_dir.resolve()
    checkpoint_path = output_dir / "evaluation_checkpoint.json"
    manifest_path = output_dir / "manifest.json"
    if args.resume:
        if not checkpoint_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("--resume requires an existing checkpoint")
        checkpoint = _read_json(checkpoint_path)
        _validate_resume(
            checkpoint,
            methods=methods,
            test_size=test_size,
            test_split_fingerprint=splits.fingerprints["test"],
        )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = _new_checkpoint(
            methods=methods,
            test_size=test_size,
            test_split_fingerprint=splits.fingerprints["test"],
        )
        _write_json(checkpoint_path, checkpoint)
        _write_json(
            manifest_path,
            {
                "schema_version": 1,
                "status": "running",
                "started_at_utc": checkpoint["started_at_utc"],
                "dataset": "AIME-2025",
                "dataset_mode": "lite",
                "split": "official_test",
                "test_size": test_size,
                "test_split_fingerprint": splits.fingerprints["test"],
                "num_threads": args.num_threads,
                "retry_policy": "execution_failures_until_numeric_success",
                "cache": False,
                "selection_frozen_before_test": True,
                "test_used_for_selection": False,
                "methods": checkpoint["methods"],
            },
        )

    identities = {
        (method.version, position)
        for method in methods
        for position in range(test_size)
    }
    attempts, scores = _checkpoint_maps(checkpoint)
    if set(attempts).difference(identities) or set(scores).difference(identities):
        raise ValueError("checkpoint contains an unknown method or test position")

    api_key = os.environ.get(first_api_key_env)
    if not api_key:
        raise RuntimeError(f"unset API-key environment: {first_api_key_env}")
    evaluation_model = dict(first_model)
    evaluation_model["cache"] = False
    evaluation_model["cache_in_memory"] = False
    lm = _create_lm(evaluation_model, api_key=api_key)
    dspy.configure(lm=lm, adapter=ChatAdapter())
    adapter = DspyAdapter(
        student_module=spec.program,
        metric_fn=spec.benchmark_meta.metric,
        feedback_map=canonical_feedback_map(spec),
        failure_score=0.0,
        num_threads=args.num_threads,
        raise_on_error=False,
    )
    programs = {
        method.version: adapter.build_program(dict(method.candidate))
        for method in methods
    }

    started_at = time.time()
    try:
        while identities.difference(scores):
            round_number = int(checkpoint["rounds_completed"]) + 1
            round_attempted = 0
            round_succeeded = 0
            for method in methods:
                positions = sorted(
                    position
                    for version, position in identities.difference(scores)
                    if version == method.version
                )
                if not positions:
                    continue
                print(
                    "AIME_TOP1_BATCH",
                    f"round={round_number}",
                    f"version={method.version}",
                    f"pending={len(positions)}",
                    flush=True,
                )
                batch = [test[position] for position in positions]
                evaluator = dspy.Evaluate(
                    devset=batch,
                    metric=spec.benchmark_meta.metric,
                    num_threads=args.num_threads,
                    display_progress=True,
                    max_errors=max(1, len(batch) * 100),
                    provide_traceback=True,
                    failure_score=0.0,
                    timeout=0,
                    straggler_limit=0,
                )
                evaluation = evaluator(programs[method.version])
                if len(evaluation.results) != len(positions):
                    raise RuntimeError("official Evaluate returned incomplete results")

                succeeded = 0
                for position, result in zip(positions, evaluation.results, strict=True):
                    example, prediction, raw_score = result
                    if example is not test[position]:
                        raise RuntimeError("official Evaluate changed example alignment")
                    identity = (method.version, position)
                    attempts[identity] = attempts.get(identity, 0) + 1
                    if _prediction_succeeded(prediction):
                        scores[identity] = _numeric_score(raw_score, identity=identity)
                        succeeded += 1
                    elif float(raw_score) != 0.0:
                        raise RuntimeError(
                            "empty execution-failure prediction has non-failure score"
                        )

                round_attempted += len(positions)
                round_succeeded += succeeded
                checkpoint["method_batches"].append(
                    {
                        "round": round_number,
                        "version": method.version,
                        "attempted": len(positions),
                        "succeeded": succeeded,
                        "remaining_for_method": sum(
                            1
                            for version, _ in identities.difference(scores)
                            if version == method.version
                        ),
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                _update_checkpoint(
                    checkpoint,
                    identities=identities,
                    attempts=attempts,
                    scores=scores,
                )
                _write_json(checkpoint_path, checkpoint)
                print(
                    "AIME_TOP1_BATCH_RESULT",
                    f"round={round_number}",
                    f"version={method.version}",
                    f"succeeded={succeeded}",
                    f"remaining={len(positions) - succeeded}",
                    flush=True,
                )

            checkpoint["rounds_completed"] = round_number
            checkpoint["rounds"].append(
                {
                    "round": round_number,
                    "attempted": round_attempted,
                    "succeeded": round_succeeded,
                    "remaining": len(identities.difference(scores)),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            _update_checkpoint(
                checkpoint,
                identities=identities,
                attempts=attempts,
                scores=scores,
            )
            _write_json(checkpoint_path, checkpoint)
            print(
                "AIME_TOP1_ROUND_RESULT",
                f"round={round_number}",
                f"succeeded={round_succeeded}",
                f"remaining={len(identities.difference(scores))}",
                flush=True,
            )

        records = _final_records(
            methods=methods,
            test_size=test_size,
            attempts=attempts,
            scores=scores,
        )
        for record in records:
            _write_json(output_dir / f"{record['version']}_top1.json", record)
            print(
                "AIME_TOP1_METHOD_RESULT",
                record["version"],
                f"candidate_idx={record['selected_candidate_idx']}",
                f"test_score_percent={record['test_score_percent']:.12g}",
                f"retry_calls={record['retry_metric_calls']}",
                flush=True,
            )

        completed_at = datetime.now(timezone.utc).isoformat()
        checkpoint["status"] = "completed"
        checkpoint["completed_at_utc"] = completed_at
        _update_checkpoint(
            checkpoint,
            identities=identities,
            attempts=attempts,
            scores=scores,
        )
        _write_json(checkpoint_path, checkpoint)
        final_result = {
            "schema_version": 1,
            "status": "completed",
            "completed_at_utc": completed_at,
            "elapsed_seconds_this_process": time.time() - started_at,
            "selection_frozen_before_test": True,
            "test_used_for_selection": False,
            "dataset": "AIME-2025",
            "dataset_mode": "lite",
            "split": "official_test",
            "test_size": test_size,
            "test_split_fingerprint": splits.fingerprints["test"],
            "retry_policy": "execution_failures_until_numeric_success",
            "remaining_execution_failure_count": 0,
            "total_metric_calls": sum(attempts.values()),
            "retry_metric_calls": sum(
                max(0, attempts.get(identity, 0) - 1) for identity in identities
            ),
            "retry_rounds": int(checkpoint["rounds_completed"]),
            "records": records,
        }
        _write_json(output_dir / "final_result.json", final_result)
        manifest = _read_json(manifest_path)
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = completed_at
        manifest["final_result"] = "final_result.json"
        manifest["evaluation_checkpoint"] = "evaluation_checkpoint.json"
        _write_json(manifest_path, manifest)
    finally:
        dspy.configure(lm=None, adapter=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
