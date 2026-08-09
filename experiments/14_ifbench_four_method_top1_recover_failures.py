from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import re
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from gepa.core.state import GEPAState
from gepa_artifact.benchmarks.IFBench import (
    IFBench,
    IFBenchCoT2StageProgram,
)
from gepa_artifact.benchmarks.IFBench import (
    feedback_fn_map as official_feedback_fn_map,
)
from gepa_artifact.benchmarks.IFBench import (
    metric as official_metric,
)

Identity = tuple[int, str]

EXPECTED_TOP1_BY_METHOD_SET = {
    "v43-v46": {
        "v43": 78,
        "v44": 341,
        "v45": 15,
        "v46": 417,
    },
    "v45-v46-owner-final": {
        "v45": 24,
        "v46": 246,
    },
}

_PARALLELIZER_ERROR_MARKER = "ERROR dspy.utils.parallelizer: Error for Example("
_KEY_PATTERN = re.compile(r"['\"]key['\"]\s*:\s*['\"]([^'\"]+)['\"]")
_METHOD_PATTERN = re.compile(
    r"['\"]top1_method_index['\"]\s*:\s*(\d+)"
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_record(identity: Identity) -> dict[str, Any]:
    method_index, test_key = identity
    return {
        "method_index": method_index,
        "test_key": test_key,
    }


def _identity_sort_key(identity: Identity) -> tuple[int, tuple[int, Any]]:
    method_index, test_key = identity
    try:
        key_part: tuple[int, Any] = (0, int(test_key))
    except ValueError:
        key_part = (1, test_key)
    return method_index, key_part


def _parse_parallelizer_error_identities(stderr_text: str) -> list[Identity]:
    """Extract identities from DSPy's explicit per-example error records."""

    identities: list[Identity] = []
    for line_number, line in enumerate(stderr_text.splitlines(), start=1):
        if _PARALLELIZER_ERROR_MARKER not in line:
            continue
        key_match = _KEY_PATTERN.search(line)
        method_match = _METHOD_PATTERN.search(line)
        if key_match is None or method_match is None:
            raise ValueError(
                "could not parse an official parallelizer error identity "
                f"at stderr line {line_number}"
            )
        identities.append(
            (int(method_match.group(1)), str(key_match.group(1)))
        )
    return identities


def _base_score_index(
    records: Iterable[Mapping[str, Any]],
) -> dict[Identity, float]:
    indexed: dict[Identity, float] = {}
    for method_index, record in enumerate(records):
        for item in record["per_instance_scores"]:
            identity = (method_index, str(item["test_key"]))
            if identity in indexed:
                raise ValueError(f"duplicate base score identity: {identity!r}")
            score = float(item["score"])
            if not math.isfinite(score):
                raise ValueError(f"non-finite base score for {identity!r}")
            indexed[identity] = score
    return indexed


def _classify_original_failures(
    *,
    error_events: Iterable[Identity],
    base_scores: Mapping[Identity, float],
    failure_score: float,
) -> tuple[set[Identity], list[Identity], Counter[Identity]]:
    """Keep only error identities whose persisted result is failure_score.

    DSPy may make one tail-straggler submission after the original call.  An
    error event whose persisted score is non-failure was superseded by the
    successful sibling call and must not be retried or replaced.
    """

    counts = Counter(error_events)
    unknown = sorted(set(counts).difference(base_scores), key=_identity_sort_key)
    if unknown:
        raise ValueError(f"stderr contains unknown test identities: {unknown!r}")

    failures: set[Identity] = set()
    superseded: list[Identity] = []
    for identity in counts:
        if base_scores[identity] == failure_score:
            failures.add(identity)
        else:
            superseded.append(identity)
    superseded.sort(key=_identity_sort_key)
    return failures, superseded, counts


def _is_successful_prediction(prediction: dspy.Prediction) -> bool:
    # The official IFBench program returns Prediction(response=...) only after
    # both model stages complete.  DSPy Evaluate returns Prediction() solely
    # for an exception and pairs it with failure_score.
    return "response" in prediction


def _numeric_score(value: Any, *, identity: Identity) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"official metric returned a non-numeric score for {identity!r}")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"official metric returned a non-finite score for {identity!r}")
    return score


def _merge_recovered_scores(
    *,
    base_records: list[Mapping[str, Any]],
    recovered_scores: Mapping[Identity, float],
    attempts: Mapping[Identity, int],
) -> list[dict[str, Any]]:
    merged_records = copy.deepcopy(base_records)
    replaced: set[Identity] = set()

    for method_index, record in enumerate(merged_records):
        original_score = float(record["test_score"])
        method_attempts = 0
        for item in record["per_instance_scores"]:
            identity = (method_index, str(item["test_key"]))
            if identity not in recovered_scores:
                continue
            item["score"] = float(recovered_scores[identity])
            replaced.add(identity)
            method_attempts += int(attempts[identity])

        instance_scores = record["per_instance_scores"]
        if not instance_scores:
            raise ValueError(f"empty per-instance scores for method {method_index}")
        score = sum(float(item["score"]) for item in instance_scores) / len(
            instance_scores
        )
        record["original_test_score"] = original_score
        record["original_test_score_percent"] = 100.0 * original_score
        record["test_score"] = score
        record["test_score_percent"] = 100.0 * score
        record["execution_failures_recovered"] = sum(
            1 for identity in recovered_scores if identity[0] == method_index
        )
        record["recovery_metric_calls"] = method_attempts

    missing = set(recovered_scores).difference(replaced)
    if missing:
        raise ValueError(
            "recovered scores do not exist in the base result: "
            f"{sorted(missing, key=_identity_sort_key)!r}"
        )
    return merged_records


def _validate_base_result(
    *,
    base_final: Mapping[str, Any],
    base_manifest: Mapping[str, Any],
    method_set: str,
    expected_candidates: Mapping[str, int],
    test_keys: set[str],
    config_path: Path,
) -> list[Mapping[str, Any]]:
    if base_final.get("status") != "completed":
        raise ValueError("base final_result is not completed")
    if base_manifest.get("status") != "completed":
        raise ValueError("base manifest is not completed")
    if base_manifest.get("method_set") != method_set:
        raise ValueError("base manifest method-set mismatch")
    if Path(str(base_manifest.get("config"))).resolve() != config_path.resolve():
        raise ValueError("base manifest config does not match --config")
    if base_final.get("dataset") != "IFBench":
        raise ValueError("base result is not IFBench")
    if base_final.get("dataset_mode") != "lite":
        raise ValueError("base result is not IFBench lite")
    if base_final.get("split") != "official_test":
        raise ValueError("base result is not official_test")

    records = list(base_final.get("records", []))
    if len(records) != len(expected_candidates):
        raise ValueError("base result does not contain the four frozen methods")
    seen_versions: set[str] = set()
    for method_index, record in enumerate(records):
        version = str(record.get("version"))
        if version in seen_versions or version not in expected_candidates:
            raise ValueError(f"unexpected or duplicate method version: {version!r}")
        seen_versions.add(version)
        if int(record.get("selected_candidate_idx")) != expected_candidates[version]:
            raise ValueError(f"frozen candidate mismatch for {version}")
        instance_scores = record.get("per_instance_scores", [])
        if len(instance_scores) != len(test_keys):
            raise ValueError(f"incomplete base scores for {version}")
        record_keys = [str(item["test_key"]) for item in instance_scores]
        if len(set(record_keys)) != len(record_keys):
            raise ValueError(f"duplicate test key in base scores for {version}")
        if set(record_keys) != test_keys:
            raise ValueError(f"official test keys mismatch for {version}")
        recomputed = sum(float(item["score"]) for item in instance_scores) / len(
            instance_scores
        )
        if not math.isclose(
            recomputed,
            float(record["test_score"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"base aggregate mismatch for {version}")

        manifest_method = base_manifest["methods"][method_index]
        if manifest_method["version"] != version:
            raise ValueError(f"base manifest record order mismatch for {version}")
        if int(manifest_method["selected_candidate_idx"]) != expected_candidates[version]:
            raise ValueError(f"base manifest candidate mismatch for {version}")

    if seen_versions != set(expected_candidates):
        raise ValueError("base result is missing a frozen method")
    return records


def _build_programs(
    *,
    records: list[Mapping[str, Any]],
    methods: tuple[Mapping[str, Any], ...],
    run_root: Path,
    num_threads: int,
    failure_score: float,
) -> list[dspy.Module]:
    if len(records) != len(methods):
        raise ValueError("method descriptor count mismatch")
    candidates: list[Mapping[str, str]] = []
    for record, method in zip(records, methods, strict=True):
        if record["version"] != method["version"]:
            raise ValueError("method order differs from the base result")
        run_dir = run_root / str(method["run_name"])
        if str(record["run_dir"]) != str(run_dir):
            raise ValueError(f"run directory mismatch for {record['version']}")
        state = GEPAState.load(str(run_dir))
        candidate_index = int(record["selected_candidate_idx"])
        candidate = state.program_candidates[candidate_index]
        evaluation_entry = importlib.import_module(
            "experiments.14_ifbench_four_method_top1_eval"
        )
        if evaluation_entry._candidate_digest(candidate) != record["candidate_sha256"]:
            raise ValueError(f"candidate digest mismatch for {record['version']}")
        candidates.append(candidate)

    adapter = DspyAdapter(
        student_module=IFBenchCoT2StageProgram(),
        metric_fn=official_metric,
        feedback_map=official_feedback_fn_map,
        failure_score=failure_score,
        num_threads=num_threads,
    )
    return [adapter.build_program(candidate) for candidate in candidates]


def _routed_examples(
    *,
    identities: Iterable[Identity],
    test_fields: Mapping[str, Mapping[str, Any]],
) -> list[dspy.Example]:
    examples: list[dspy.Example] = []
    for method_index, test_key in sorted(identities, key=_identity_sort_key):
        fields = copy.deepcopy(test_fields[test_key])
        fields["top1_method_index"] = method_index
        examples.append(
            dspy.Example(**fields).with_inputs("prompt", "top1_method_index")
        )
    return examples


def _new_checkpoint(
    *,
    base_output_dir: Path,
    base_final_sha256: str,
    original_failures: set[Identity],
    error_event_counts: Counter[Identity],
    superseded_error_identities: list[Identity],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_output_dir": str(base_output_dir),
        "base_final_result_sha256": base_final_sha256,
        "original_execution_failures": [
            _identity_record(identity)
            for identity in sorted(original_failures, key=_identity_sort_key)
        ],
        "original_parallelizer_error_events": sum(error_event_counts.values()),
        "duplicate_parallelizer_error_events": sum(
            count - 1 for count in error_event_counts.values()
        ),
        "superseded_error_identities": [
            _identity_record(identity) for identity in superseded_error_identities
        ],
        "rounds_completed": 0,
        "retry_metric_calls": 0,
        "attempts": [],
        "recovered_scores": [],
        "pending": [
            _identity_record(identity)
            for identity in sorted(original_failures, key=_identity_sort_key)
        ],
        "rounds": [],
    }


def _checkpoint_maps(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[Identity, int], dict[Identity, float]]:
    attempts = {
        (int(item["method_index"]), str(item["test_key"])): int(item["attempts"])
        for item in checkpoint.get("attempts", [])
    }
    recovered = {
        (int(item["method_index"]), str(item["test_key"])): float(item["score"])
        for item in checkpoint.get("recovered_scores", [])
    }
    return attempts, recovered


def _update_checkpoint(
    checkpoint: dict[str, Any],
    *,
    attempts: Mapping[Identity, int],
    recovered: Mapping[Identity, float],
    original_failures: set[Identity],
) -> None:
    pending = original_failures.difference(recovered)
    checkpoint["attempts"] = [
        {
            **_identity_record(identity),
            "attempts": int(attempts.get(identity, 0)),
        }
        for identity in sorted(original_failures, key=_identity_sort_key)
    ]
    checkpoint["recovered_scores"] = [
        {
            **_identity_record(identity),
            "score": float(recovered[identity]),
        }
        for identity in sorted(recovered, key=_identity_sort_key)
    ]
    checkpoint["pending"] = [
        _identity_record(identity)
        for identity in sorted(pending, key=_identity_sort_key)
    ]
    checkpoint["retry_metric_calls"] = sum(attempts.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-output-dir", required=True, type=Path)
    parser.add_argument("--base-stderr-log", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--num-threads", default=32, type=int)
    parser.add_argument(
        "--method-set",
        choices=tuple(EXPECTED_TOP1_BY_METHOD_SET),
        default="v43-v46",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.num_threads <= 0:
        raise ValueError("--num-threads must be positive")
    base_output_dir = args.base_output_dir.resolve()
    base_final_path = base_output_dir / "final_result.json"
    base_manifest_path = base_output_dir / "manifest.json"
    if not base_final_path.is_file() or not base_manifest_path.is_file():
        raise FileNotFoundError("the completed base result and manifest are required")
    if not args.base_stderr_log.is_file():
        raise FileNotFoundError(args.base_stderr_log)

    evaluation_entry = importlib.import_module(
        "experiments.14_ifbench_four_method_top1_eval"
    )
    methods = evaluation_entry.METHOD_SETS[args.method_set]
    config = _load_json(args.config)
    remote = copy.deepcopy(config["remote_lm"])
    official = config["official_gepa"]
    if official["dataset_mode"] != "lite":
        raise ValueError("the recovery protocol requires IFBench lite")
    failure_score = float(official["failure_score"])

    benchmark = IFBench(dataset_mode=official["dataset_mode"])
    test_fields: dict[str, Mapping[str, Any]] = {}
    for example in benchmark.test_set:
        test_key = str(example.key)
        if test_key in test_fields:
            raise ValueError(f"duplicate official test key: {test_key!r}")
        test_fields[test_key] = copy.deepcopy(example.toDict())

    base_final = _load_json(base_final_path)
    base_manifest = _load_json(base_manifest_path)
    base_records = _validate_base_result(
        base_final=base_final,
        base_manifest=base_manifest,
        method_set=args.method_set,
        expected_candidates=EXPECTED_TOP1_BY_METHOD_SET[args.method_set],
        test_keys=set(test_fields),
        config_path=args.config,
    )
    base_scores = _base_score_index(base_records)
    error_events = _parse_parallelizer_error_identities(
        args.base_stderr_log.read_text(encoding="utf-8", errors="strict")
    )
    original_failures, superseded_errors, error_event_counts = (
        _classify_original_failures(
            error_events=error_events,
            base_scores=base_scores,
            failure_score=failure_score,
        )
    )

    output_dir = args.output_dir.resolve()
    checkpoint_path = output_dir / "recovery_checkpoint.json"
    manifest_path = output_dir / "manifest.json"
    if args.resume:
        if not output_dir.is_dir() or not checkpoint_path.is_file():
            raise FileNotFoundError("--resume requires an existing recovery checkpoint")
        checkpoint = _load_json(checkpoint_path)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = _new_checkpoint(
            base_output_dir=base_output_dir,
            base_final_sha256=_sha256(base_final_path),
            original_failures=original_failures,
            error_event_counts=error_event_counts,
            superseded_error_identities=superseded_errors,
        )
        _write_json(checkpoint_path, checkpoint)
        _write_json(
            manifest_path,
            {
                "schema_version": 1,
                "status": "running",
                "started_at_utc": checkpoint["started_at_utc"],
                "method_set": args.method_set,
                "base_output_dir": str(base_output_dir),
                "base_stderr_log": str(args.base_stderr_log.resolve()),
                "config": str(args.config.resolve()),
                "num_threads": args.num_threads,
                "retry_policy": "execution_failures_until_numeric_success",
                "cache": False,
                "selection_frozen_before_test": True,
                "test_used_for_selection": False,
            },
        )

    if checkpoint["base_final_result_sha256"] != _sha256(base_final_path):
        raise ValueError("base final_result changed after recovery started")
    checkpoint_failures = {
        (int(item["method_index"]), str(item["test_key"]))
        for item in checkpoint["original_execution_failures"]
    }
    if checkpoint_failures != original_failures:
        raise ValueError("original execution-failure set changed")
    attempts, recovered = _checkpoint_maps(checkpoint)
    if set(attempts).difference(original_failures):
        raise ValueError("checkpoint contains attempts for a non-failure identity")
    if set(recovered).difference(original_failures):
        raise ValueError("checkpoint recovered a non-failure identity")

    remote["cache"] = False
    remote["cache_in_memory"] = False
    api_key_env = str(remote["api_key_env"])
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"unset API key environment: {api_key_env}")
    training_entry = importlib.import_module(
        "experiments.11_ifbench_sparse_raw_feedback_training"
    )
    evaluate_owner = importlib.import_module("bridge.dspy_evaluate")
    without_parent_straggler_resubmission = (
        evaluate_owner.without_parent_straggler_resubmission
    )
    lm = training_entry._build_remote_lm(remote, api_key=api_key)
    _ = lm.supported_params
    dspy.configure(lm=lm, adapter=ChatAdapter())

    programs = _build_programs(
        records=base_records,
        methods=methods,
        run_root=evaluation_entry.RUN_ROOT,
        num_threads=args.num_threads,
        failure_score=failure_score,
    )
    router = evaluation_entry.ProgramRouter(programs)

    started_at = time.time()
    while True:
        pending = original_failures.difference(recovered)
        if not pending:
            break
        round_number = int(checkpoint["rounds_completed"]) + 1
        examples = _routed_examples(identities=pending, test_fields=test_fields)
        print(
            "RECOVERY_ROUND",
            f"round={round_number}",
            f"pending={len(examples)}",
            flush=True,
        )
        evaluator = dspy.Evaluate(
            devset=examples,
            metric=official_metric,
            num_threads=args.num_threads,
            display_progress=True,
            max_errors=max(1, len(examples) * 10),
            provide_traceback=True,
            failure_score=failure_score,
        )
        with without_parent_straggler_resubmission():
            evaluation = evaluator(router)

        succeeded_this_round = 0
        for example, prediction, raw_score in evaluation.results:
            identity = (int(example.top1_method_index), str(example.key))
            if identity not in pending:
                raise RuntimeError(f"unexpected retry result identity: {identity!r}")
            attempts[identity] = attempts.get(identity, 0) + 1
            if _is_successful_prediction(prediction):
                recovered[identity] = _numeric_score(raw_score, identity=identity)
                succeeded_this_round += 1
            elif float(raw_score) != failure_score:
                raise RuntimeError(
                    "DSPy returned an empty failure prediction with a non-failure score "
                    f"for {identity!r}"
                )

        checkpoint["rounds_completed"] = round_number
        checkpoint["rounds"].append(
            {
                "round": round_number,
                "attempted": len(examples),
                "succeeded": succeeded_this_round,
                "remaining": len(original_failures.difference(recovered)),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _update_checkpoint(
            checkpoint,
            attempts=attempts,
            recovered=recovered,
            original_failures=original_failures,
        )
        _write_json(checkpoint_path, checkpoint)
        print(
            "RECOVERY_ROUND_RESULT",
            f"round={round_number}",
            f"succeeded={succeeded_this_round}",
            f"remaining={len(original_failures.difference(recovered))}",
            flush=True,
        )

    merged_records = _merge_recovered_scores(
        base_records=base_records,
        recovered_scores=recovered,
        attempts=attempts,
    )
    for record in merged_records:
        _write_json(output_dir / f"{record['version']}_top1.json", record)
        print(
            "RECOVERED_METHOD_RESULT",
            record["version"],
            f"candidate_idx={record['selected_candidate_idx']}",
            f"test_score_percent={record['test_score_percent']:.12g}",
            f"recovered={record['execution_failures_recovered']}",
            f"retry_calls={record['recovery_metric_calls']}",
            flush=True,
        )

    completed_at = datetime.now(timezone.utc).isoformat()
    checkpoint["status"] = "completed"
    checkpoint["completed_at_utc"] = completed_at
    _update_checkpoint(
        checkpoint,
        attempts=attempts,
        recovered=recovered,
        original_failures=original_failures,
    )
    _write_json(checkpoint_path, checkpoint)

    final_result = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": completed_at,
        "elapsed_seconds_this_process": time.time() - started_at,
        "selection_frozen_before_test": True,
        "test_used_for_selection": False,
        "dataset": "IFBench",
        "dataset_mode": "lite",
        "split": "official_test",
        "test_size": len(test_fields),
        "base_output_dir": str(base_output_dir),
        "base_final_result_sha256": checkpoint["base_final_result_sha256"],
        "retry_policy": "execution_failures_until_numeric_success",
        "original_execution_failure_count": len(original_failures),
        "recovered_execution_failure_count": len(recovered),
        "remaining_execution_failure_count": 0,
        "retry_metric_calls": sum(attempts.values()),
        "retry_rounds": int(checkpoint["rounds_completed"]),
        "records": merged_records,
    }
    _write_json(output_dir / "final_result.json", final_result)
    manifest = _load_json(manifest_path)
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = completed_at
    manifest["final_result"] = "final_result.json"
    manifest["recovery_checkpoint"] = "recovery_checkpoint.json"
    _write_json(manifest_path, manifest)
    print(
        "RECOVERY_FINAL_RESULT",
        json.dumps(
            {
                record["version"]: {
                    "candidate_idx": record["selected_candidate_idx"],
                    "test_score_percent": record["test_score_percent"],
                    "recovered": record["execution_failures_recovered"],
                    "retry_calls": record["recovery_metric_calls"],
                }
                for record in merged_records
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
