"""Run the frozen paper protocol for tau2-bench Airline.

This CLI is deliberately an orchestration boundary.  Tau2 owns each episode,
its multi-turn state, tools, environment, reward, checkpoint, and
infrastructure-failure resume.  GEPA or the existing COMPASS adapter engine
owns optimization.  This file only freezes identities, selects the supported
owner path, and atomically records paper artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from bridge.paper_source_snapshot import build_frozen_upstream_snapshot
from bridge.tau2_airline_protocol import (
    TAU2_AGENT_MODEL,
    TAU2_AIRLINE_COMMIT,
    TAU2_AIRLINE_FILE_SHA256,
    TAU2_AIRLINE_PROPOSAL_IDS,
    TAU2_AIRLINE_TEST_IDS,
    TAU2_AIRLINE_TRAIN_IDS,
    TAU2_AIRLINE_VALIDATION_IDS,
    TAU2_AIRLINE_VERSION,
    TAU2_MAX_CONCURRENCY,
    TAU2_OPTIMIZATION_ROLLOUT_BUDGET,
    TAU2_PREFLIGHT_ROLLOUT_BUDGET,
    TAU2_REFLECTION_MODEL,
    TAU2_TEST_TRIAL_SEEDS,
    TAU2_USER_MODEL,
    Tau2AdapterMethod,
    Tau2AirlineOptimizationSettings,
    Tau2ExperimentPhase,
    build_tau2_airline_text_config,
    build_tau2_reflection_lm,
    frozen_tau2_airline_optimization_view,
    load_frozen_tau2_airline_splits,
    run_tau2_airline_final_evaluation,
    run_tau2_airline_optimization,
)
from bridge.tau2_gepa_adapter import (
    candidate_sha256,
    summarize_tau2_resource_usage,
    tau2_seed_candidate,
)

Tau2PaperMethod = Literal["seed", "gepa", "m0", "compass"]

SCHEMA_VERSION = 1
PRIMARY_API_BASE = "https://api.gpt.ge/v1"
PRIMARY_API_KEY_ENV = "COMPASS_GPT_GE_API_KEY"
PROPOSAL_MINIBATCH_SIZE = 3
ADMISSION_MINIBATCH_SIZE = 3
PARENT_TOP_N = 5
CONFIG_KEYS = {
    "api_base",
    "api_key_env",
    "cache_dir",
    "logical_rollout_budget",
    "matrix_id",
    "method",
    "phase",
    "quiet",
    "run_dir",
    "schema_version",
    "task_id",
    "tau2_root",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a JSON object: {path}")
    return dict(value)


def load_tau2_paper_config(path: Path) -> dict[str, Any]:
    """Load one strict, credential-free launcher binding.

    The config only freezes CLI inputs.  Algorithm, benchmark, retry, metric,
    and checkpoint behavior remain owned by the protocol and upstream tau2.
    """

    config = _read_json_object(path)
    if set(config) != CONFIG_KEYS:
        raise ValueError(
            "tau2 paper config keys mismatch; "
            f"missing={sorted(CONFIG_KEYS - set(config))}, "
            f"extra={sorted(set(config) - CONFIG_KEYS)}"
        )
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("tau2 paper config schema_version must equal 1")
    if config["method"] not in ("seed", "gepa", "m0", "compass"):
        raise ValueError("tau2 paper config method is unsupported")
    if config["phase"] not in ("preflight", "formal"):
        raise ValueError("tau2 paper config phase is unsupported")
    if config["api_base"] != PRIMARY_API_BASE:
        raise ValueError(f"tau2 paper api_base must be {PRIMARY_API_BASE}")
    if not isinstance(config["api_key_env"], str) or not config["api_key_env"].strip():
        raise TypeError("tau2 paper config api_key_env must be non-empty text")
    if not isinstance(config["quiet"], bool):
        raise TypeError("tau2 paper config quiet must be a JSON boolean")
    if config["task_id"] != "tau2_airline":
        raise ValueError("tau2 paper config task_id must equal tau2_airline")
    if not isinstance(config["matrix_id"], str) or not config["matrix_id"].strip():
        raise TypeError("tau2 paper config matrix_id must be non-empty text")
    expected_budget = _budget_for(config["method"], config["phase"])
    if config["logical_rollout_budget"] != expected_budget:
        raise ValueError(
            "tau2 paper config logical_rollout_budget differs from protocol"
        )
    for name in ("run_dir", "cache_dir", "tau2_root"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise TypeError(f"tau2 paper config {name} must be non-empty text")
    if Path(config["run_dir"]).resolve() == Path(config["cache_dir"]).resolve():
        raise ValueError("tau2 paper run_dir and cache_dir must be distinct")
    return config


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (
        json.dumps(
            dict(value),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _budget_for(method: Tau2PaperMethod, phase: Tau2ExperimentPhase) -> int:
    if method == "seed":
        return 0
    return (
        TAU2_PREFLIGHT_ROLLOUT_BUDGET
        if phase == "preflight"
        else TAU2_OPTIMIZATION_ROLLOUT_BUDGET
    )


def _selection_mode(method: Tau2PaperMethod) -> str:
    return "high_resolution" if method == "compass" else "raw_frontier_rate"


def _build_run_identity(
    *,
    method: Tau2PaperMethod,
    phase: Tau2ExperimentPhase,
    tau2_root: Path,
    api_base: str,
    api_key_env: str,
    matrix_id: str = "direct_tau2_airline_v1",
    cache_dir: Path | None = None,
    task_id: str = "tau2_airline",
    logical_rollout_budget: int | None = None,
) -> dict[str, Any]:
    if method not in ("seed", "gepa", "m0", "compass"):
        raise ValueError("tau2 paper method must be seed, gepa, m0, or compass")
    if phase not in ("preflight", "formal"):
        raise ValueError("tau2 paper phase must be preflight or formal")
    if api_base != PRIMARY_API_BASE:
        raise ValueError(f"tau2 paper api_base must be {PRIMARY_API_BASE}")
    if not api_key_env.strip():
        raise ValueError("api_key_env must be non-empty")
    if task_id != "tau2_airline":
        raise ValueError("tau2 task_id must equal tau2_airline")
    if not matrix_id.strip():
        raise ValueError("tau2 matrix_id must be non-empty")

    runner_path = Path(__file__).resolve()
    source_root = runner_path.parents[2]
    protocol_path = source_root / "bridge" / "tau2_airline_protocol.py"
    adapter_path = source_root / "bridge" / "tau2_gepa_adapter.py"
    method_source_files = {
        "tau2_airline_protocol.py": protocol_path,
        "tau2_gepa_adapter.py": adapter_path,
    }
    if method in ("m0", "compass"):
        method_source_files.update(
            {
                "b19_reversible_parent_selection.py": (
                    source_root / "bridge" / "b19_reversible_parent_selection.py"
                ),
                "b20_compass_reflection.py": (
                    source_root / "bridge" / "b20_compass_reflection.py"
                ),
                "minibatch_config.py": (source_root / "bridge" / "minibatch_config.py"),
            }
        )
    budget = _budget_for(method, phase)
    if logical_rollout_budget is not None and logical_rollout_budget != budget:
        raise ValueError("tau2 logical rollout budget differs from protocol")
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix_id": matrix_id,
        "benchmark": "tau2_airline",
        "phase": phase,
        "method": method,
        "optimizer_seed": 0,
        "source": {
            "compass_git_head": _git_head(source_root),
            "runner_sha256": _sha256_file(runner_path),
            "method_source_sha256": {
                name: _sha256_file(path) for name, path in method_source_files.items()
            },
            "frozen_upstreams": build_frozen_upstream_snapshot(source_root),
            "tau2_root": str(tau2_root.resolve()),
            "tau2_version": TAU2_AIRLINE_VERSION,
            "tau2_git_commit": TAU2_AIRLINE_COMMIT,
            "tau2_data_sha256": dict(TAU2_AIRLINE_FILE_SHA256),
        },
        "task": {
            "domain": "airline",
            "official_train_ids": list(TAU2_AIRLINE_TRAIN_IDS),
            "proposal_ids": list(TAU2_AIRLINE_PROPOSAL_IDS),
            "validation_ids": list(TAU2_AIRLINE_VALIDATION_IDS),
            "official_test_ids": list(TAU2_AIRLINE_TEST_IDS),
            "partition_rule": "one_based_owner_position_mod_5_equals_0_is_validation",
            "official_test_trials": len(TAU2_TEST_TRIAL_SEEDS),
            "official_test_trial_seeds": list(TAU2_TEST_TRIAL_SEEDS),
        },
        "models": {
            "agent": TAU2_AGENT_MODEL,
            "user_simulator": TAU2_USER_MODEL,
            "reflection": TAU2_REFLECTION_MODEL,
            "api_base": api_base,
            "api_key_env": api_key_env,
        },
        "cache": {
            "namespace": (str(cache_dir.resolve()) if cache_dir is not None else None),
            "external_lm_cache": "disabled",
            "optimizer_evaluation_cache": "official_gepa_run_local_state",
        },
        "optimizer": {
            "logical_rollout_budget": budget,
            "proposal_minibatch_size": PROPOSAL_MINIBATCH_SIZE,
            "admission_minibatch_size": ADMISSION_MINIBATCH_SIZE,
            "parent_top_n": PARENT_TOP_N,
            "parent_selection_score_mode": _selection_mode(method),
            "acceptance": "strict_improvement",
            "max_concurrency": TAU2_MAX_CONCURRENCY,
            "epoch_parallel": False,
            "teacher_forcing": False,
            "flash_trace": False,
        },
        "evaluation": {
            "preflight_test_policy": "not_run",
            "formal_test_policy": "official_tau2_all_four_trials",
            "failure_resume_owner": "tau2.runner.checkpoint.auto_resume",
        },
    }


def _checkpoint_identity(identity_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "identity_sha256": identity_sha256,
        "optimizer_checkpoint": "optimizer/gepa_state.bin",
        "selected_candidate": "selected_candidate.json",
        "owner_evaluation_checkpoint": "evaluation/results.json",
        "retry_owner": "tau2.runner.checkpoint.auto_resume",
    }


def _new_manifest(
    *, identity: Mapping[str, Any], identity_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at_utc": _utc_now(),
        "identity_sha256": identity_sha256,
        "method": identity["method"],
        "phase": identity["phase"],
        "resume_count": 0,
        "final_result": None,
    }


def _validate_existing_run(
    run_dir: Path,
    *,
    expected_identity: Mapping[str, Any],
    expected_identity_sha256: str,
) -> dict[str, Any]:
    identity_path = run_dir / "run_identity.json"
    checkpoint_path = run_dir / "checkpoint_identity.json"
    manifest_path = run_dir / "manifest.json"
    for path in (identity_path, checkpoint_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete tau2 paper run identity: {path}")
    stored_identity = _read_json_object(identity_path)
    if stored_identity != dict(expected_identity):
        raise RuntimeError("tau2 paper run identity changed; refusing resume")
    stored_checkpoint = _read_json_object(checkpoint_path)
    if stored_checkpoint != _checkpoint_identity(expected_identity_sha256):
        raise RuntimeError("tau2 paper checkpoint identity changed; refusing resume")
    manifest = _read_json_object(manifest_path)
    if manifest.get("identity_sha256") != expected_identity_sha256:
        raise RuntimeError("tau2 paper manifest identity changed; refusing resume")
    if manifest.get("method") != expected_identity["method"]:
        raise RuntimeError("tau2 paper manifest method changed; refusing resume")
    if manifest.get("phase") != expected_identity["phase"]:
        raise RuntimeError("tau2 paper manifest phase changed; refusing resume")
    return manifest


def _initialize_run(
    run_dir: Path,
    *,
    identity: Mapping[str, Any],
    resume: bool,
) -> tuple[dict[str, Any], str]:
    identity_sha256 = _canonical_sha256(identity)
    run_dir.mkdir(parents=True, exist_ok=True)
    contents = list(run_dir.iterdir())
    if not contents:
        if resume:
            raise FileNotFoundError("--resume requires an existing tau2 paper run")
        _atomic_write_json(run_dir / "run_identity.json", identity)
        _atomic_write_json(
            run_dir / "checkpoint_identity.json",
            _checkpoint_identity(identity_sha256),
        )
        manifest = _new_manifest(
            identity=identity,
            identity_sha256=identity_sha256,
        )
        _atomic_write_json(run_dir / "manifest.json", manifest)
        return manifest, identity_sha256
    if not resume:
        raise FileExistsError(
            "tau2 paper run_dir is non-empty; use --resume only after verifying "
            "this exact run identity"
        )
    manifest = _validate_existing_run(
        run_dir,
        expected_identity=identity,
        expected_identity_sha256=identity_sha256,
    )
    manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
    manifest["resumed_at_utc"] = _utc_now()
    if manifest.get("status") != "completed":
        manifest["status"] = "running"
        manifest.pop("failed_at_utc", None)
        manifest.pop("exception_type", None)
    _atomic_write_json(run_dir / "manifest.json", manifest)
    return manifest, identity_sha256


def _activate_api_key(api_key_env: str) -> None:
    value = os.environ.get(api_key_env)
    if value is None or not value.strip():
        raise RuntimeError(
            f"required API credential environment is unset: {api_key_env}"
        )
    # Tau2's pinned OpenAI/LiteLLM owner reads OPENAI_API_KEY.  The value is
    # kept process-local and is never copied into a config or artifact.
    os.environ["OPENAI_API_KEY"] = value


def _load_selected_candidate(
    path: Path,
    *,
    method: Tau2PaperMethod,
    identity_sha256: str,
) -> tuple[int, dict[str, str], str, dict[str, Any]]:
    record = _read_json_object(path)
    if record.get("method") != method:
        raise RuntimeError("selected candidate method identity changed")
    if record.get("identity_sha256") != identity_sha256:
        raise RuntimeError("selected candidate run identity changed")
    selected_idx = record.get("selected_candidate_idx")
    if isinstance(selected_idx, bool) or not isinstance(selected_idx, int):
        raise TypeError("selected candidate index must be an integer")
    raw_candidate = record.get("candidate")
    if not isinstance(raw_candidate, Mapping):
        raise TypeError("selected candidate must be a JSON object")
    candidate = {str(key): str(value) for key, value in raw_candidate.items()}
    actual_sha256 = candidate_sha256(candidate)
    if record.get("candidate_sha256") != actual_sha256:
        raise RuntimeError("selected candidate content identity changed")
    optimization = record.get("optimization")
    if not isinstance(optimization, Mapping):
        raise TypeError("selected candidate optimization summary must be an object")
    return selected_idx, candidate, actual_sha256, dict(optimization)


def _write_selected_candidate(
    path: Path,
    *,
    method: Tau2PaperMethod,
    identity_sha256: str,
    selected_candidate_idx: int,
    candidate: Mapping[str, str],
    optimization_summary: Mapping[str, Any],
) -> tuple[dict[str, str], str]:
    frozen_candidate = dict(candidate)
    digest = candidate_sha256(frozen_candidate)
    _atomic_write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "identity_sha256": identity_sha256,
            "method": method,
            "selected_candidate_idx": selected_candidate_idx,
            "candidate": frozen_candidate,
            "candidate_sha256": digest,
            "optimization": dict(optimization_summary),
        },
    )
    return frozen_candidate, digest


def _completed_result_if_valid(
    run_dir: Path,
    *,
    manifest: Mapping[str, Any],
    identity_sha256: str,
) -> dict[str, Any] | None:
    if manifest.get("status") != "completed":
        return None
    final_path = run_dir / "final_result.json"
    if not final_path.is_file():
        raise RuntimeError("completed tau2 manifest has no final_result.json")
    final_result = _read_json_object(final_path)
    if final_result.get("status") != "completed":
        raise RuntimeError("completed tau2 manifest points to incomplete final result")
    if final_result.get("identity_sha256") != identity_sha256:
        raise RuntimeError("tau2 final result identity changed")
    return final_result


def _optimization_summary(
    result: Any,
    *,
    max_metric_calls_threshold: int,
    owner_resource_usage: Mapping[str, Any],
    reflection_lm: Any,
) -> dict[str, Any]:
    candidates = getattr(result, "candidates", None)
    total_metric_calls = getattr(result, "total_metric_calls", None)
    if (
        isinstance(total_metric_calls, bool)
        or not isinstance(total_metric_calls, int)
        or total_metric_calls < 0
    ):
        raise RuntimeError("official GEPA result has no valid total_metric_calls")
    if total_metric_calls < max_metric_calls_threshold:
        raise RuntimeError(
            "official GEPA stopped before reaching its frozen soft metric-call "
            "threshold: "
            f"threshold={max_metric_calls_threshold}, actual={total_metric_calls}"
        )
    return {
        "total_metric_calls": total_metric_calls,
        "max_metric_calls_threshold": max_metric_calls_threshold,
        "metric_call_overshoot": total_metric_calls - max_metric_calls_threshold,
        "budget_semantics": "official_soft_iteration_boundary",
        "num_candidates": len(candidates) if isinstance(candidates, Sequence) else None,
        "num_full_val_evals": getattr(result, "num_full_val_evals", None),
        "owner_episode_resource_usage": dict(owner_resource_usage),
        "reflection_lm_resource_usage": {
            "prompt_tokens": int(getattr(reflection_lm, "total_tokens_in", 0)),
            "completion_tokens": int(getattr(reflection_lm, "total_tokens_out", 0)),
            "cost_usd": float(getattr(reflection_lm, "total_cost", 0.0)),
        },
    }


def _resource_usage_recorder(
    path: Path,
    *,
    identity_sha256: str,
) -> tuple[Any, Any]:
    """Persist additive owner-emitted resource deltas after each adapter batch."""

    lock = threading.Lock()
    zero = summarize_tau2_resource_usage([])

    def load_usage() -> dict[str, int | float]:
        if not path.is_file():
            return dict(zero)
        record = _read_json_object(path)
        if record.get("identity_sha256") != identity_sha256:
            raise RuntimeError("optimization resource ledger identity changed")
        stored = record.get("usage")
        if not isinstance(stored, Mapping) or set(stored) != set(zero):
            raise RuntimeError("optimization resource ledger schema changed")
        return {key: cast(int | float, stored[key]) for key in zero}

    def record(delta: Mapping[str, int | float]) -> None:
        if set(delta) != set(zero):
            raise RuntimeError("adapter resource delta schema changed")
        with lock:
            current = load_usage()
            merged = {key: current[key] + delta[key] for key in zero}
            _atomic_write_json(
                path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "identity_sha256": identity_sha256,
                    "usage": merged,
                },
            )

    return record, load_usage


def _run(args: argparse.Namespace) -> dict[str, Any]:
    method = cast(Tau2PaperMethod, args.method)
    phase = cast(Tau2ExperimentPhase, args.phase)
    run_dir = Path(args.run_dir).resolve()
    cache_dir_text = getattr(args, "cache_dir", None)
    cache_dir = (
        Path(cache_dir_text).resolve()
        if cache_dir_text is not None
        else run_dir.with_name(run_dir.name + ".cache")
    )
    tau2_root = Path(args.tau2_root).resolve()

    # Owner split/source verification happens before any API credential is
    # activated or any paid call can be made.
    splits = load_frozen_tau2_airline_splits(tau2_root)
    view = frozen_tau2_airline_optimization_view(splits.train)
    identity = _build_run_identity(
        method=method,
        phase=phase,
        tau2_root=tau2_root,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        matrix_id=getattr(args, "matrix_id", "direct_tau2_airline_v1"),
        cache_dir=cache_dir,
        task_id=getattr(args, "task_id", "tau2_airline"),
        logical_rollout_budget=getattr(args, "logical_rollout_budget", None),
    )
    manifest, identity_sha256 = _initialize_run(
        run_dir,
        identity=identity,
        resume=bool(args.resume),
    )
    completed = _completed_result_if_valid(
        run_dir,
        manifest=manifest,
        identity_sha256=identity_sha256,
    )
    if completed is not None:
        return completed

    selected_path = run_dir / "selected_candidate.json"
    try:
        if selected_path.is_file():
            selected_idx, selected_candidate, selected_sha256, optimization_summary = (
                _load_selected_candidate(
                    selected_path,
                    method=method,
                    identity_sha256=identity_sha256,
                )
            )
        elif method == "seed":
            selected_idx = 0
            selected_candidate, selected_sha256 = _write_selected_candidate(
                selected_path,
                method=method,
                identity_sha256=identity_sha256,
                selected_candidate_idx=selected_idx,
                candidate=tau2_seed_candidate(),
                optimization_summary={
                    "total_metric_calls": 0,
                    "num_candidates": 1,
                    "num_full_val_evals": 0,
                    "owner_episode_resource_usage": (summarize_tau2_resource_usage([])),
                    "reflection_lm_resource_usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cost_usd": 0.0,
                    },
                },
            )
            optimization_summary = {
                "total_metric_calls": 0,
                "num_candidates": 1,
                "num_full_val_evals": 0,
                "owner_episode_resource_usage": summarize_tau2_resource_usage([]),
                "reflection_lm_resource_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                },
            }
        else:
            _activate_api_key(args.api_key_env)
            run_config = build_tau2_airline_text_config(
                api_base=args.api_base,
                task_split_name="train",
                num_trials=1,
            )
            reflection_lm = build_tau2_reflection_lm(api_base=args.api_base)
            record_usage, load_usage = _resource_usage_recorder(
                run_dir / "optimization_resource_usage.json",
                identity_sha256=identity_sha256,
            )
            optimization = run_tau2_airline_optimization(
                method=cast(Tau2AdapterMethod, method),
                view=view,
                run_config=run_config,
                reflection_lm=reflection_lm,
                run_dir=run_dir / "optimizer",
                settings=Tau2AirlineOptimizationSettings(
                    parent_selection_score_mode=cast(Any, _selection_mode(method)),
                    phase=phase,
                    optimizer_seed=0,
                    max_metric_calls=_budget_for(method, phase),
                    proposal_minibatch_size=PROPOSAL_MINIBATCH_SIZE,
                    admission_minibatch_size=ADMISSION_MINIBATCH_SIZE,
                    parent_top_n=PARENT_TOP_N,
                    max_concurrency=TAU2_MAX_CONCURRENCY,
                    display_progress_bar=False,
                    use_cloudpickle=True,
                ),
                verified_resume=bool(args.resume),
                resource_usage_callback=record_usage,
            )
            selected_idx = optimization.selected_candidate_idx
            optimization_summary = _optimization_summary(
                optimization.result,
                max_metric_calls_threshold=_budget_for(method, phase),
                owner_resource_usage=load_usage(),
                reflection_lm=reflection_lm,
            )
            selected_candidate, selected_sha256 = _write_selected_candidate(
                selected_path,
                method=method,
                identity_sha256=identity_sha256,
                selected_candidate_idx=selected_idx,
                candidate=optimization.selected_candidate,
                optimization_summary=optimization_summary,
            )

        evaluation_record: dict[str, Any]
        if phase == "preflight":
            evaluation_record = {
                "status": "not_run",
                "reason": "preflight_contract_excludes_official_test",
            }
        else:
            _activate_api_key(args.api_key_env)
            evaluation_dir = run_dir / "evaluation"
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            results_path = evaluation_dir / "results.json"
            owner_resume = bool(args.resume and results_path.exists())
            run_config = build_tau2_airline_text_config(
                api_base=args.api_base,
                task_split_name="train",
                num_trials=1,
                auto_resume=owner_resume,
            )
            evaluation = run_tau2_airline_final_evaluation(
                candidate=selected_candidate,
                test_tasks=splits.test,
                run_config=run_config,
                save_path=results_path,
                save_dir=evaluation_dir,
                console_display=not args.quiet,
                verified_resume=owner_resume,
            )
            metrics = evaluation.metrics.model_dump(mode="json")
            evaluation_record = {
                "status": "completed",
                "role": (
                    "official_seed_evaluation"
                    if method == "seed"
                    else "official_final_evaluation"
                ),
                "results_path": str(results_path),
                "results_sha256": _sha256_file(results_path),
                "num_simulations": len(evaluation.results.simulations),
                "metrics": metrics,
                "test_score_percent": float(evaluation.metrics.avg_reward) * 100.0,
                "owner_resource_usage": summarize_tau2_resource_usage(
                    evaluation.results.simulations
                ),
            }

        final_result = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "completed_at_utc": _utc_now(),
            "identity_sha256": identity_sha256,
            "method": method,
            "phase": phase,
            "selected_candidate_idx": selected_idx,
            "selected_candidate_sha256": selected_sha256,
            "optimization": dict(optimization_summary),
            "evaluation": evaluation_record,
        }
        _atomic_write_json(run_dir / "final_result.json", final_result)
        manifest = _read_json_object(run_dir / "manifest.json")
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = final_result["completed_at_utc"]
        manifest["final_result"] = "final_result.json"
        _atomic_write_json(run_dir / "manifest.json", manifest)
        return final_result
    except BaseException as error:
        manifest_path = run_dir / "manifest.json"
        if manifest_path.is_file():
            failed_manifest = _read_json_object(manifest_path)
            failed_manifest["status"] = "failed"
            failed_manifest["failed_at_utc"] = _utc_now()
            # Persist only the type.  Provider exception strings can contain
            # request material and do not belong in the identity artifact.
            failed_manifest["exception_type"] = (
                f"{error.__class__.__module__}.{error.__class__.__qualname__}"
            )
            _atomic_write_json(manifest_path, failed_manifest)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen tau2-bench Airline paper protocol"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--method", choices=("seed", "gepa", "m0", "compass"))
    parser.add_argument(
        "--phase",
        choices=("preflight", "formal"),
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--matrix-id")
    parser.add_argument("--task-id")
    parser.add_argument("--logical-rollout-budget", type=int)
    parser.add_argument("--tau2-root")
    parser.add_argument("--api-base")
    parser.add_argument("--api-key-env")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _resolve_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is not None:
        direct = {
            name: getattr(args, name)
            for name in (
                "method",
                "phase",
                "run_dir",
                "cache_dir",
                "matrix_id",
                "task_id",
                "logical_rollout_budget",
                "tau2_root",
                "api_base",
                "api_key_env",
            )
            if getattr(args, name) is not None
        }
        if direct or args.quiet:
            raise ValueError(
                "--config cannot be combined with direct protocol arguments"
            )
        config = load_tau2_paper_config(args.config.resolve(strict=True))
        return argparse.Namespace(**config, resume=bool(args.resume))

    missing = [
        name
        for name in ("method", "phase", "run_dir", "tau2_root")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "direct tau2 paper invocation is missing: " + ", ".join(missing)
        )
    args.api_base = args.api_base or PRIMARY_API_BASE
    args.api_key_env = args.api_key_env or PRIMARY_API_KEY_ENV
    args.cache_dir = args.cache_dir or str(
        Path(args.run_dir).with_name(Path(args.run_dir).name + ".cache")
    )
    args.matrix_id = args.matrix_id or "direct_tau2_airline_v1"
    args.task_id = args.task_id or "tau2_airline"
    args.logical_rollout_budget = (
        args.logical_rollout_budget
        if args.logical_rollout_budget is not None
        else _budget_for(args.method, args.phase)
    )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _resolve_cli_args(_parser().parse_args(argv))
    result = _run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "method": result["method"],
                "phase": result["phase"],
                "identity_sha256": result["identity_sha256"],
                "final_result": str(Path(args.run_dir).resolve() / "final_result.json"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
