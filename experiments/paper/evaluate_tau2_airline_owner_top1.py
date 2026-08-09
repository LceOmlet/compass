"""Evaluate the owner-selected Top-1 from one frozen tau2 Airline run.

This is a recovery boundary for a completed optimization whose final candidate
was recorded with the proposal-parent score instead of GEPA's evaluation
policy.  It never optimizes, rewrites the source run, or owns tau2 evaluation
and retry behavior.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gepa.core.state import GEPAState
from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import compute_metrics

from bridge.b19_reversible_parent_selection import evaluation_count, frontier_count
from bridge.b20_compass_reflection import SparseMinibatchEvaluationPolicy
from bridge.paper_source_snapshot import build_frozen_upstream_snapshot
from bridge.tau2_airline_protocol import (
    TAU2_AGENT_MODEL,
    TAU2_AIRLINE_TEST_IDS,
    TAU2_MAX_CONCURRENCY,
    TAU2_TEST_TRIAL_SEEDS,
    TAU2_USER_MODEL,
    build_tau2_airline_text_config,
    load_frozen_tau2_airline_splits,
    run_tau2_airline_final_evaluation,
)
from bridge.tau2_gepa_adapter import (
    TAU2_FIXED_CANDIDATE_AGENT_PREFIX,
    candidate_sha256,
    summarize_tau2_resource_usage,
)
from experiments.paper.run_tau2_airline import (
    PRIMARY_API_BASE,
    _activate_api_key,
    _atomic_write_json,
    _canonical_sha256,
    _git_head,
    _read_json_object,
    _sha256_file,
    _utc_now,
)

SCHEMA_VERSION = 1
SEMANTIC_SOURCE_FILES = (
    "bridge/b19_reversible_parent_selection.py",
    "bridge/b20_compass_reflection.py",
    "bridge/paper_source_snapshot.py",
    "bridge/tau2_airline_protocol.py",
    "bridge/tau2_gepa_adapter.py",
    "experiments/paper/run_tau2_airline.py",
)
CONFIG_KEYS = {
    "api_base",
    "api_key_env",
    "expected_candidate_idx",
    "expected_candidate_sha256",
    "expected_source_candidates_sha256",
    "expected_source_identity_sha256",
    "expected_source_recorded_candidate_idx",
    "expected_source_state_sha256",
    "output_dir",
    "quiet",
    "schema_version",
    "source_run",
    "tau2_root",
}


@dataclass(frozen=True, slots=True)
class SourceSelection:
    candidate_idx: int
    candidate: dict[str, str]
    candidate_sha256: str
    frontier_count: int
    clean_exposure: int
    source_identity: dict[str, Any]


def _checkpoint_identity(identity_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "identity_sha256": identity_sha256,
        "owner_evaluation_checkpoint": "evaluation/results.json",
        "retry_owner": "tau2.runner.checkpoint.auto_resume",
    }


def _selection_record(
    *, identity_sha256: str, selection: SourceSelection
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "identity_sha256": identity_sha256,
        "selected_candidate_idx": selection.candidate_idx,
        "candidate": selection.candidate,
        "candidate_sha256": selection.candidate_sha256,
        "frontier_count": selection.frontier_count,
        "clean_exposure": selection.clean_exposure,
        "selection_rule": "max(raw_frontier_rate, clean_exposure, -candidate_idx)",
    }


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


@contextmanager
def _exclusive_output_lock(output_dir: Path) -> Iterator[None]:
    """Hold one OS-released lock for a correction output across owner calls."""

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.lock"
    handle = lock_path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(
                f"tau2 owner-top1 output is already active: {output_dir}"
            ) from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def load_config(path: Path) -> dict[str, Any]:
    config = _read_json_object(path)
    if set(config) != CONFIG_KEYS:
        raise ValueError(
            "tau2 owner-top1 config keys mismatch; "
            f"missing={sorted(CONFIG_KEYS - set(config))}, "
            f"extra={sorted(set(config) - CONFIG_KEYS)}"
        )
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("tau2 owner-top1 schema_version must equal 1")
    if config["api_base"] != PRIMARY_API_BASE:
        raise ValueError(f"tau2 owner-top1 api_base must be {PRIMARY_API_BASE}")
    for key in (
        "api_key_env",
        "expected_candidate_sha256",
        "expected_source_candidates_sha256",
        "expected_source_identity_sha256",
        "expected_source_state_sha256",
        "output_dir",
        "source_run",
        "tau2_root",
    ):
        if not isinstance(config[key], str) or not config[key].strip():
            raise TypeError(f"tau2 owner-top1 {key} must be non-empty text")
    for key in ("expected_candidate_idx", "expected_source_recorded_candidate_idx"):
        if isinstance(config[key], bool) or not isinstance(config[key], int):
            raise TypeError(f"tau2 owner-top1 {key} must be an integer")
        if config[key] < 0:
            raise ValueError(f"tau2 owner-top1 {key} must be non-negative")
    for key in (
        "expected_candidate_sha256",
        "expected_source_candidates_sha256",
        "expected_source_identity_sha256",
        "expected_source_state_sha256",
    ):
        if len(config[key]) != 64:
            raise ValueError(f"tau2 owner-top1 {key} must be SHA256 text")
    if not isinstance(config["quiet"], bool):
        raise TypeError("tau2 owner-top1 quiet must be a JSON boolean")
    source_run = Path(config["source_run"]).resolve()
    output_dir = Path(config["output_dir"]).resolve()
    if _paths_overlap(source_run, output_dir):
        raise ValueError("source_run and output_dir must be distinct and non-nested")
    return config


def _verified_source_selection(config: Mapping[str, Any]) -> SourceSelection:
    source_run = Path(str(config["source_run"])).resolve()
    identity_path = source_run / "run_identity.json"
    manifest_path = source_run / "manifest.json"
    final_path = source_run / "final_result.json"
    state_path = source_run / "optimizer" / "gepa_state.bin"
    candidates_path = source_run / "optimizer" / "candidates.json"
    for path in (
        identity_path,
        manifest_path,
        final_path,
        state_path,
        candidates_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"tau2 source artifact is missing: {path}")

    source_identity = _read_json_object(identity_path)
    source_identity_sha256 = _canonical_sha256(source_identity)
    if source_identity_sha256 != config["expected_source_identity_sha256"]:
        raise RuntimeError("tau2 source run identity changed")
    manifest = _read_json_object(manifest_path)
    final_result = _read_json_object(final_path)
    if (
        manifest.get("status") != "completed"
        or final_result.get("status") != "completed"
    ):
        raise RuntimeError("tau2 source optimization is not completed")
    if manifest.get("identity_sha256") != source_identity_sha256:
        raise RuntimeError("tau2 source manifest identity changed")
    if final_result.get("identity_sha256") != source_identity_sha256:
        raise RuntimeError("tau2 source final-result identity changed")
    if final_result.get("method") != "compass" or final_result.get("phase") != "formal":
        raise RuntimeError("tau2 source is not the formal COMPASS run")
    if (
        final_result.get("selected_candidate_idx")
        != config["expected_source_recorded_candidate_idx"]
    ):
        raise RuntimeError("tau2 source recorded-candidate identity changed")
    if _sha256_file(state_path) != config["expected_source_state_sha256"]:
        raise RuntimeError("tau2 source GEPA state changed")
    if _sha256_file(candidates_path) != config["expected_source_candidates_sha256"]:
        raise RuntimeError("tau2 source candidate export changed")

    state = GEPAState.load(str(state_path.parent))
    selected_idx = SparseMinibatchEvaluationPolicy().get_best_program(state)
    if selected_idx != config["expected_candidate_idx"]:
        raise RuntimeError("tau2 owner final candidate changed")
    candidate_export = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not isinstance(candidate_export, list):
        raise TypeError("tau2 candidate export must be a list")
    if selected_idx >= len(candidate_export):
        raise RuntimeError("tau2 owner candidate is absent from candidate export")
    raw_candidate = state.program_candidates[selected_idx]
    if not isinstance(raw_candidate, Mapping):
        raise TypeError("tau2 owner candidate must be a mapping")
    candidate = {str(key): str(value) for key, value in raw_candidate.items()}
    if candidate_export[selected_idx] != candidate:
        raise RuntimeError("tau2 state candidate differs from owner export")
    digest = candidate_sha256(candidate)
    if digest != config["expected_candidate_sha256"]:
        raise RuntimeError("tau2 owner candidate content changed")
    frontiers = frontier_count(state, selected_idx)
    exposure = evaluation_count(state, selected_idx)
    if exposure <= 0:
        raise RuntimeError("tau2 owner candidate has no clean exposure")
    return SourceSelection(
        candidate_idx=selected_idx,
        candidate=candidate,
        candidate_sha256=digest,
        frontier_count=frontiers,
        clean_exposure=exposure,
        source_identity=source_identity,
    )


def _build_identity(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    selection: SourceSelection,
) -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    project_root = runner_path.parents[2]
    source_run = Path(str(config["source_run"])).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "role": "tau2_airline_owner_final_selection_correction",
        "source": {
            "run_dir": str(source_run),
            "identity_sha256": config["expected_source_identity_sha256"],
            "state_sha256": config["expected_source_state_sha256"],
            "candidates_sha256": config["expected_source_candidates_sha256"],
            "recorded_candidate_idx": config["expected_source_recorded_candidate_idx"],
            "compass_git_head": selection.source_identity["source"]["compass_git_head"],
        },
        "selection": {
            "owner": (
                "bridge.b20_compass_reflection."
                "SparseMinibatchEvaluationPolicy.get_best_program"
            ),
            "rule": "max(raw_frontier_rate, clean_exposure, -candidate_idx)",
            "selected_candidate_idx": selection.candidate_idx,
            "candidate_sha256": selection.candidate_sha256,
            "frontier_count": selection.frontier_count,
            "clean_exposure": selection.clean_exposure,
            "raw_frontier_rate": (selection.frontier_count / selection.clean_exposure),
            "test_used_for_selection": False,
        },
        "evaluator": {
            "compass_git_head": _git_head(project_root),
            "runner_sha256": _sha256_file(runner_path),
            "config_sha256": _sha256_file(config_path),
            "semantic_source_sha256": {
                relative: _sha256_file(project_root / relative)
                for relative in SEMANTIC_SOURCE_FILES
            },
            "frozen_upstreams": build_frozen_upstream_snapshot(project_root),
        },
        "task": {
            "benchmark": "tau2_airline",
            "official_test_ids": list(TAU2_AIRLINE_TEST_IDS),
            "official_test_trial_seeds": list(TAU2_TEST_TRIAL_SEEDS),
            "num_simulations": len(TAU2_AIRLINE_TEST_IDS) * len(TAU2_TEST_TRIAL_SEEDS),
        },
        "models": {
            "agent": TAU2_AGENT_MODEL,
            "user_simulator": TAU2_USER_MODEL,
            "api_base": config["api_base"],
            "api_key_env": config["api_key_env"],
        },
        "runtime": {
            "max_concurrency": TAU2_MAX_CONCURRENCY,
            "retry_owner": "tau2.runner.checkpoint.auto_resume",
            "cache": "disabled",
        },
    }


def _expected_final_identities() -> set[tuple[str, int, int]]:
    return {
        (str(task_id), trial, seed)
        for trial, seed in enumerate(TAU2_TEST_TRIAL_SEEDS)
        for task_id in TAU2_AIRLINE_TEST_IDS
    }


def _validate_owner_results_artifact(
    results_path: Path,
    *,
    selection: SourceSelection,
    require_complete: bool,
) -> dict[str, Any]:
    """Validate checkpoint identity only; tau2 still owns retry and metrics."""

    results = _read_json_object(results_path)
    info = results.get("info")
    if not isinstance(info, Mapping):
        raise TypeError("tau2 owner checkpoint info must be an object")
    agent_info = info.get("agent_info")
    if not isinstance(agent_info, Mapping):
        raise TypeError("tau2 owner checkpoint agent_info must be an object")
    expected_implementation = (
        TAU2_FIXED_CANDIDATE_AGENT_PREFIX + selection.candidate_sha256
    )
    if agent_info.get("implementation") != expected_implementation:
        raise RuntimeError("tau2 owner checkpoint candidate identity changed")
    if agent_info.get("llm") != TAU2_AGENT_MODEL:
        raise RuntimeError("tau2 owner checkpoint agent model changed")
    if info.get("num_trials") != len(TAU2_TEST_TRIAL_SEEDS):
        raise RuntimeError("tau2 owner checkpoint trial count changed")

    simulations = results.get("simulations")
    if not isinstance(simulations, list):
        raise TypeError("tau2 owner checkpoint simulations must be a list")
    identities: list[tuple[str, int, int]] = []
    unresolved: list[tuple[str, int, int]] = []
    for simulation in simulations:
        if not isinstance(simulation, Mapping):
            raise TypeError("tau2 owner checkpoint simulation must be an object")
        trial = simulation.get("trial")
        seed = simulation.get("seed")
        if isinstance(trial, bool) or not isinstance(trial, int):
            raise TypeError("tau2 owner checkpoint simulation trial is invalid")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("tau2 owner checkpoint simulation seed is invalid")
        identity = (str(simulation.get("task_id")), trial, seed)
        identities.append(identity)
        if (
            simulation.get("termination_reason") == "infrastructure_error"
            or simulation.get("reward_info") is None
        ):
            unresolved.append(identity)
    if len(set(identities)) != len(identities):
        raise RuntimeError("tau2 owner checkpoint contains duplicate identities")
    expected = _expected_final_identities()
    actual = set(identities)
    if not actual.issubset(expected):
        raise RuntimeError("tau2 owner checkpoint contains foreign identities")
    if require_complete:
        if actual != expected:
            raise RuntimeError("tau2 owner completed results are incomplete")
        if unresolved:
            raise RuntimeError(
                "tau2 owner completed results contain infrastructure failures"
            )
    return results


def _official_result_summaries(
    results: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    owner_results = Results.model_validate(results)
    metrics = compute_metrics(owner_results).model_dump(mode="json")
    usage = summarize_tau2_resource_usage(owner_results.simulations)
    return metrics, usage


def _validate_completed_output(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    identity_sha256: str,
    selection: SourceSelection,
) -> dict[str, Any]:
    if manifest.get("status") != "completed":
        raise RuntimeError("tau2 owner-top1 manifest is not completed")
    if manifest.get("final_result") != "final_result.json":
        raise RuntimeError("tau2 owner-top1 final-result pointer changed")
    final_path = output_dir / "final_result.json"
    results_path = output_dir / "evaluation" / "results.json"
    for path in (final_path, results_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"tau2 owner-top1 completed artifact is missing: {path}"
            )
    final_result = _read_json_object(final_path)
    if final_result.get("status") != "completed":
        raise RuntimeError("tau2 owner-top1 final result is not completed")
    if final_result.get("identity_sha256") != identity_sha256:
        raise RuntimeError("tau2 owner-top1 final result identity changed")
    if final_result.get("selected_candidate_idx") != selection.candidate_idx:
        raise RuntimeError("tau2 owner-top1 final candidate index changed")
    if final_result.get("selected_candidate_sha256") != selection.candidate_sha256:
        raise RuntimeError("tau2 owner-top1 final candidate content changed")
    if final_result.get("selection") != identity["selection"]:
        raise RuntimeError("tau2 owner-top1 final selection record changed")
    evaluation = final_result.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("status") != "completed":
        raise RuntimeError("tau2 owner-top1 final evaluation is not completed")
    if Path(str(evaluation.get("results_path"))).resolve() != results_path.resolve():
        raise RuntimeError("tau2 owner-top1 final results path changed")
    if evaluation.get("results_sha256") != _sha256_file(results_path):
        raise RuntimeError("tau2 owner-top1 final results content changed")
    results = _validate_owner_results_artifact(
        results_path,
        selection=selection,
        require_complete=True,
    )
    simulations = results["simulations"]
    if evaluation.get("num_simulations") != len(simulations):
        raise RuntimeError("tau2 owner-top1 final simulation count changed")
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("tau2 owner-top1 final metrics must be an object")
    official_metrics, official_usage = _official_result_summaries(results)
    if dict(metrics) != official_metrics:
        raise RuntimeError("tau2 owner-top1 final official metrics changed")
    if evaluation.get("owner_resource_usage") != official_usage:
        raise RuntimeError("tau2 owner-top1 final resource usage changed")
    avg_reward = metrics.get("avg_reward")
    score = evaluation.get("test_score_percent")
    if not isinstance(avg_reward, (int, float)) or isinstance(avg_reward, bool):
        raise TypeError("tau2 owner-top1 avg_reward must be numeric")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise TypeError("tau2 owner-top1 test_score_percent must be numeric")
    if abs(float(score) - float(avg_reward) * 100.0) > 1e-9:
        raise RuntimeError("tau2 owner-top1 final score aggregation changed")
    return final_result


def _verified_owner_resume(
    results_path: Path,
    *,
    resume: bool,
    selection: SourceSelection,
) -> bool:
    if not resume:
        return False
    if not results_path.is_file():
        return False
    _validate_owner_results_artifact(
        results_path,
        selection=selection,
        require_complete=False,
    )
    return True


def _initialize_output(
    output_dir: Path,
    *,
    identity: Mapping[str, Any],
    selection: SourceSelection,
    resume: bool,
) -> tuple[dict[str, Any], str]:
    identity_sha256 = _canonical_sha256(identity)
    identity_path = output_dir / "run_identity.json"
    checkpoint_path = output_dir / "checkpoint_identity.json"
    manifest_path = output_dir / "manifest.json"
    selection_path = output_dir / "selected_candidate.json"
    if output_dir.exists():
        if not resume:
            raise FileExistsError(
                f"tau2 owner-top1 output already exists: {output_dir}"
            )
        for path in (identity_path, checkpoint_path, manifest_path, selection_path):
            if not path.is_file():
                raise FileNotFoundError(f"incomplete tau2 owner-top1 output: {path}")
        if _read_json_object(identity_path) != dict(identity):
            raise RuntimeError("tau2 owner-top1 identity changed; refusing resume")
        if _read_json_object(checkpoint_path) != _checkpoint_identity(identity_sha256):
            raise RuntimeError(
                "tau2 owner-top1 checkpoint identity changed; refusing resume"
            )
        manifest = _read_json_object(manifest_path)
        if manifest.get("identity_sha256") != identity_sha256:
            raise RuntimeError("tau2 owner-top1 manifest identity changed")
        if _read_json_object(selection_path) != _selection_record(
            identity_sha256=identity_sha256,
            selection=selection,
        ):
            raise RuntimeError("tau2 owner-top1 selection changed; refusing resume")
        if manifest.get("status") not in {"running", "failed", "completed"}:
            raise RuntimeError("tau2 owner-top1 manifest status is invalid")
        if manifest.get("status") != "completed":
            if (output_dir / "final_result.json").exists():
                raise RuntimeError(
                    "tau2 owner-top1 incomplete manifest has a final result"
                )
            manifest["status"] = "running"
            manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
            manifest["resumed_at_utc"] = _utc_now()
            manifest.pop("failed_at_utc", None)
            manifest.pop("exception_type", None)
    else:
        if resume:
            raise FileNotFoundError(
                "--resume requires an existing tau2 owner-top1 output"
            )
        output_dir.mkdir(parents=True, exist_ok=False)
        _atomic_write_json(identity_path, identity)
        _atomic_write_json(checkpoint_path, _checkpoint_identity(identity_sha256))
        _atomic_write_json(
            selection_path,
            _selection_record(identity_sha256=identity_sha256, selection=selection),
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "started_at_utc": _utc_now(),
            "identity_sha256": identity_sha256,
            "resume_count": 0,
            "final_result": None,
        }
    _atomic_write_json(manifest_path, manifest)
    return manifest, identity_sha256


def _run_locked(
    config_path: Path,
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    selection = _verified_source_selection(config)
    tau2_root = Path(config["tau2_root"]).resolve()
    splits = load_frozen_tau2_airline_splits(tau2_root)
    identity = _build_identity(
        config=config,
        config_path=config_path,
        selection=selection,
    )
    manifest, identity_sha256 = _initialize_output(
        output_dir,
        identity=identity,
        selection=selection,
        resume=resume,
    )
    if manifest.get("status") == "completed":
        return _validate_completed_output(
            output_dir,
            manifest=manifest,
            identity=identity,
            identity_sha256=identity_sha256,
            selection=selection,
        )

    try:
        evaluation_dir = output_dir / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        results_path = evaluation_dir / "results.json"
        owner_resume = _verified_owner_resume(
            results_path,
            resume=resume,
            selection=selection,
        )
        _activate_api_key(str(config["api_key_env"]))
        run_config = build_tau2_airline_text_config(
            api_base=str(config["api_base"]),
            task_split_name="train",
            num_trials=1,
            auto_resume=owner_resume,
        )
        evaluation = run_tau2_airline_final_evaluation(
            candidate=selection.candidate,
            test_tasks=splits.test,
            run_config=run_config,
            save_path=results_path,
            save_dir=evaluation_dir,
            console_display=not bool(config["quiet"]),
            verified_resume=owner_resume,
        )
        metrics = evaluation.metrics.model_dump(mode="json")
        final_result = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "completed_at_utc": _utc_now(),
            "identity_sha256": identity_sha256,
            "role": "tau2_airline_owner_final_selection_correction",
            "source_run": str(Path(config["source_run"]).resolve()),
            "selected_candidate_idx": selection.candidate_idx,
            "selected_candidate_sha256": selection.candidate_sha256,
            "selection": dict(identity["selection"]),
            "evaluation": {
                "status": "completed",
                "role": "official_final_evaluation",
                "results_path": str(results_path),
                "results_sha256": _sha256_file(results_path),
                "num_simulations": len(evaluation.results.simulations),
                "metrics": metrics,
                "test_score_percent": float(evaluation.metrics.avg_reward) * 100.0,
                "owner_resource_usage": summarize_tau2_resource_usage(
                    evaluation.results.simulations
                ),
            },
        }
        _atomic_write_json(output_dir / "final_result.json", final_result)
        manifest = _read_json_object(output_dir / "manifest.json")
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = final_result["completed_at_utc"]
        manifest["final_result"] = "final_result.json"
        _atomic_write_json(output_dir / "manifest.json", manifest)
        return final_result
    except BaseException as error:
        manifest = _read_json_object(output_dir / "manifest.json")
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = _utc_now()
        manifest["exception_type"] = (
            f"{error.__class__.__module__}.{error.__class__.__qualname__}"
        )
        _atomic_write_json(output_dir / "manifest.json", manifest)
        raise


def _run(config_path: Path, *, resume: bool) -> dict[str, Any]:
    config = load_config(config_path)
    output_dir = Path(config["output_dir"]).resolve()
    with _exclusive_output_lock(output_dir):
        return _run_locked(
            config_path,
            config=config,
            output_dir=output_dir,
            resume=resume,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the owner-selected Top-1 from a frozen tau2 run"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = _run(args.config.resolve(strict=True), resume=bool(args.resume))
    print(
        "TAU2_OWNER_TOP1_FINAL",
        {
            "candidate_idx": result["selected_candidate_idx"],
            "test_score_percent": result["evaluation"]["test_score_percent"],
            "num_simulations": result["evaluation"]["num_simulations"],
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
