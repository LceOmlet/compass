"""Create the credential-free IFBench/AIME/ChartQA primary run matrix.

The matrix is create-only and separates every task/method/phase cache and run
directory.  It delegates M0/COMPASS to their existing runner and emits Seed,
MIPROv2, and native GEPA cells for ``run_primary_method.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from bridge.paper_source_snapshot import build_frozen_upstream_snapshot
from bridge.primary_method_protocol import (
    PREFLIGHT_MINIMUM_ROLLOUTS,
    PRIMARY_METHODS,
    PRIMARY_TASK_BUDGETS,
    canonical_sha256,
    compass_condition,
    source_file_hashes,
    validate_primary_cell,
)
from experiments.paper.generate_reflection_configs import load_model_profiles

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
PRIMARY_RUNNER: Final = PROJECT_ROOT / "experiments/paper/run_primary_method.py"
COMPASS_TEXT_RUNNER: Final = (
    PROJECT_ROOT / "experiments/paper/run_compass_reflection.py"
)
COMPASS_CHARTQA_RUNNER: Final = (
    PROJECT_ROOT / "experiments/mechanism/run_compass_reflection.py"
)
DEFAULT_MATRIX_ID: Final = "primary_gpt41mini_seed0_v1"
DEFAULT_OUTPUT_DIR: Final = (
    PROJECT_ROOT / "experiments/paper/generated" / DEFAULT_MATRIX_ID
)
DEFAULT_FORMAL_ROOT: Final = Path("F:/compass-primary-local")
DEFAULT_PREFLIGHT_ROOT: Final = Path("F:/compass-paper-preflight")
DEFAULT_PYTHON: Final = Path(
    "F:/compass-ifbench-local/.venv-no-torch-py312/Scripts/python.exe"
)
DEFAULT_CHARTQA_PYTHON: Final = (
    PROJECT_ROOT.parent / "skill-factory/.venv/Scripts/python.exe"
)
MODEL_PROFILE: Final = "gpt_4_1_mini_gpt_ge"
_SAFE_NAME: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

SOURCE_FILES: Final[tuple[str, ...]] = (
    "bridge/b19_reversible_parent_selection.py",
    "bridge/b20_compass_reflection.py",
    "bridge/chartqa_protocol.py",
    "bridge/mechanism_benchmark_registry.py",
    "bridge/mechanism_chartqa.py",
    "bridge/minibatch_config.py",
    "bridge/paper_benchmark_registry.py",
    "bridge/paper_source_snapshot.py",
    "bridge/primary_method_protocol.py",
    "bridge/request_deadline.py",
    "experiments/mechanism/run_compass_reflection.py",
    "experiments/paper/generate_primary_matrix.py",
    "experiments/paper/model_profiles.json",
    "experiments/paper/run_compass_reflection.py",
    "experiments/paper/run_primary_method.py",
    "patches/dspy-working-tree.patch",
    "patches/gepa-working-tree.patch",
    "patches/gepa-artifact-working-tree.patch",
    "upstreams.lock.json",
)


def _portable(path: Path) -> str:
    return path.as_posix()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json_new(path: Path, value: Any) -> None:
    _write_new(path, _json_bytes(value))


def _write_jsonl_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_new(
        path,
        b"".join(
            (
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            for row in rows
        ),
    )


def source_snapshot() -> dict[str, Any]:
    root_head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "root_head": root_head,
        "file_sha256": source_file_hashes(PROJECT_ROOT, SOURCE_FILES),
        "submodules": build_frozen_upstream_snapshot(PROJECT_ROOT),
    }


def _budget(*, task_id: str, method: str, phase: str) -> int:
    if method == "seed":
        return 0
    if phase == "formal":
        return PRIMARY_TASK_BUDGETS[task_id]
    return PREFLIGHT_MINIMUM_ROLLOUTS[task_id]


def _baseline_config(
    *,
    matrix_id: str,
    task_id: str,
    method: str,
    phase: str,
    runtime_root: Path,
    model: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    num_threads: int,
) -> dict[str, Any]:
    budget = _budget(task_id=task_id, method=method, phase=phase)
    validate_primary_cell(
        task_id=task_id,
        method=method,
        phase=phase,
        optimizer_seed=0,
        logical_rollout_budget=budget,
    )
    slug = f"{matrix_id}_{phase}_{task_id}_{method}_seed0"
    return {
        "schema_version": 1,
        "matrix_id": matrix_id,
        "phase": phase,
        "task_id": task_id,
        "method": method,
        "dataset_mode": "lite",
        "optimizer_seed": 0,
        "logical_rollout_budget": budget,
        "num_threads": num_threads,
        "model": dict(model),
        "model_profile_sha256": canonical_sha256(model),
        "run_dir": _portable(runtime_root / "runs" / slug),
        "cache_dir": _portable(runtime_root / "cache" / slug),
        "source_snapshot": dict(snapshot),
    }


def _compass_config(
    *,
    matrix_id: str,
    task_id: str,
    method: str,
    phase: str,
    runtime_root: Path,
    model: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    num_threads: int,
    max_candidate_workers: int,
    max_reflection_workers: int,
) -> dict[str, Any]:
    budget = _budget(task_id=task_id, method=method, phase=phase)
    validate_primary_cell(
        task_id=task_id,
        method=method,
        phase=phase,
        optimizer_seed=0,
        logical_rollout_budget=budget,
    )
    slug = f"{matrix_id}_{phase}_{task_id}_{method}_seed0"
    optimizer = {
        "acceptance_mode": "strict_improvement",
        "add_format_failure_as_feedback": False,
        "display_progress_bar": False,
        "epoch_parallel_enabled": False,
        "evaluation_straggler_timeout": 0,
        "failure_score": 0,
        "max_candidate_workers": max_candidate_workers,
        "max_metric_calls": budget,
        "max_reflection_workers": max_reflection_workers,
        "num_threads": num_threads,
        "parent_selection_score_mode": (
            "high_resolution_lexicographic"
            if method == "compass"
            else "high_resolution"
        ),
        "parent_top_n": 5,
        "perfect_score": 1,
        "raise_on_exception": True,
        "skip_perfect_score": True,
        "track_best_outputs": True,
        "use_cloudpickle": True,
    }
    if method == "m0":
        optimizer["reflection_minibatch_size"] = 3
    else:
        optimizer["proposal_minibatch_size"] = 3
        optimizer["admission_minibatch_size"] = 3

    return {
        "cache_dir": _portable(runtime_root / "cache" / slug),
        "condition": compass_condition(method),
        "dataset_mode": "lite",
        "model": dict(model),
        "optimizer": optimizer,
        "optimizer_seed": 0,
        "run_dir": _portable(runtime_root / "runs" / slug),
        "source_snapshot": dict(snapshot),
        "task_id": task_id,
    }


def build_matrix(
    *,
    matrix_id: str,
    phase: str,
    runtime_root: Path,
    model: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    text_num_threads: int,
    chartqa_num_threads: int,
    max_candidate_workers: int,
    max_reflection_workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build runnable cells and separately recorded owner-interface gaps."""

    if not _SAFE_NAME.fullmatch(matrix_id):
        raise ValueError("matrix_id must be path-safe")
    if phase not in {"preflight", "formal"}:
        raise ValueError("phase must be 'preflight' or 'formal'")
    records: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    methods = PRIMARY_METHODS if phase == "formal" else PRIMARY_METHODS[1:]
    for task_id in PRIMARY_TASK_BUDGETS:
        for method in methods:
            if (
                phase == "preflight"
                and task_id != "chartqa"
                and method in {"m0", "compass"}
            ):
                blocked.append(
                    {
                        "task_id": task_id,
                        "method": method,
                        "phase": phase,
                        "status": "blocked_owner_interface",
                        "reason": (
                            "the existing official text COMPASS runner freezes the "
                            "full paper budget and exposes no disjoint pilot-budget "
                            "seam; this matrix will not duplicate its optimizer loop"
                        ),
                    }
                )
                continue
            if task_id == "chartqa" and method == "mipro" and phase == "formal":
                blocked.append(
                    {
                        "task_id": task_id,
                        "method": method,
                        "phase": phase,
                        "status": "blocked_owner_interface",
                        "reason": (
                            "DSPy MIPROv2 has no max_metric_calls seam and the "
                            "GEPA artifact has no 1152-rollout ChartQA protocol"
                        ),
                    }
                )
                continue
            threads = chartqa_num_threads if task_id == "chartqa" else text_num_threads
            if method in {"seed", "mipro", "gepa"}:
                config = _baseline_config(
                    matrix_id=matrix_id,
                    task_id=task_id,
                    method=method,
                    phase=phase,
                    runtime_root=runtime_root,
                    model=model,
                    snapshot=snapshot,
                    num_threads=threads,
                )
                runner = PRIMARY_RUNNER
            else:
                config = _compass_config(
                    matrix_id=matrix_id,
                    task_id=task_id,
                    method=method,
                    phase=phase,
                    runtime_root=runtime_root,
                    model=model,
                    snapshot=snapshot,
                    num_threads=threads,
                    max_candidate_workers=max_candidate_workers,
                    max_reflection_workers=max_reflection_workers,
                )
                runner = (
                    COMPASS_CHARTQA_RUNNER
                    if task_id == "chartqa"
                    else COMPASS_TEXT_RUNNER
                )
            slug = f"{matrix_id}_{phase}_{task_id}_{method}_seed0"
            records.append(
                {
                    "slug": slug,
                    "task_id": task_id,
                    "method": method,
                    "phase": phase,
                    "logical_rollout_budget": _budget(
                        task_id=task_id,
                        method=method,
                        phase=phase,
                    ),
                    "runner": _portable(runner),
                    "config": config,
                }
            )
    return records, blocked


def _queue_entry(
    record: Mapping[str, Any],
    *,
    config_path: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    config = record["config"]
    task_id = record["task_id"]
    python = DEFAULT_CHARTQA_PYTHON if task_id == "chartqa" else DEFAULT_PYTHON
    required_env = [config["model"]["api_key_env"]]
    if task_id == "chartqa":
        required_env.extend(
            ["CHARTQA_PREPARED_ROOT", "CHARTQA_ROOT", "SKILL_FACTORY_ROOT"]
        )
    slug = record["slug"]
    return {
        "slug": slug,
        "task_id": task_id,
        "method": record["method"],
        "phase": record["phase"],
        "logical_rollout_budget": record["logical_rollout_budget"],
        "config_file": _portable(config_path),
        "config_sha256": canonical_sha256(config),
        "run_dir": config["run_dir"],
        "cache_dir": config["cache_dir"],
        "argv": [
            _portable(python),
            record["runner"],
            "--config",
            _portable(config_path),
        ],
        "cwd": _portable(PROJECT_ROOT),
        "pythonpath": [
            _portable(PROJECT_ROOT),
            _portable(PROJECT_ROOT / "upstreams/dspy"),
            _portable(PROJECT_ROOT / "upstreams/gepa/src"),
            _portable(PROJECT_ROOT / "upstreams/gepa-artifact"),
        ],
        "required_env": required_env,
        "stdout_log": _portable(runtime_root / "logs" / f"{slug}.stdout.log"),
        "stderr_log": _portable(runtime_root / "logs" / f"{slug}.stderr.log"),
        "pid_file": _portable(runtime_root / "logs" / f"{slug}.pid"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-id", default=DEFAULT_MATRIX_ID)
    parser.add_argument("--phase", choices=("preflight", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--text-num-threads", type=int, default=32)
    parser.add_argument("--chartqa-num-threads", type=int, default=8)
    parser.add_argument("--max-candidate-workers", type=int, default=1)
    parser.add_argument("--max-reflection-workers", type=int, default=1)
    parser.add_argument(
        "--model-profiles",
        type=Path,
        default=PROJECT_ROOT / "experiments/paper/model_profiles.json",
    )
    args = parser.parse_args()
    for name in (
        "text_num_threads",
        "chartqa_num_threads",
        "max_candidate_workers",
        "max_reflection_workers",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be positive")
    runtime_root = args.runtime_root or (
        DEFAULT_FORMAL_ROOT if args.phase == "formal" else DEFAULT_PREFLIGHT_ROOT
    )
    profiles = load_model_profiles(args.model_profiles)
    model = profiles[MODEL_PROFILE]
    snapshot = source_snapshot()
    records, blocked = build_matrix(
        matrix_id=args.matrix_id,
        phase=args.phase,
        runtime_root=runtime_root,
        model=model,
        snapshot=snapshot,
        text_num_threads=args.text_num_threads,
        chartqa_num_threads=args.chartqa_num_threads,
        max_candidate_workers=args.max_candidate_workers,
        max_reflection_workers=args.max_reflection_workers,
    )

    output_dir = args.output_dir
    config_dir = output_dir / "configs"
    queue: list[dict[str, Any]] = []
    for record in records:
        config_path = config_dir / f"{record['slug']}.json"
        _write_json_new(config_path, record["config"])
        queue.append(
            _queue_entry(
                record,
                config_path=config_path,
                runtime_root=runtime_root,
            )
        )
    _write_jsonl_new(output_dir / "launch_queue.jsonl", queue)
    manifest = {
        "schema_version": 1,
        "matrix_id": args.matrix_id,
        "phase": args.phase,
        "model_profile": MODEL_PROFILE,
        "model_profile_sha256": canonical_sha256(model),
        "source_snapshot": snapshot,
        "source_snapshot_sha256": canonical_sha256(snapshot),
        "runnable_cell_count": len(records),
        "blocked_cell_count": len(blocked),
        "blocked_cells": blocked,
        "cells": [
            {key: value for key, value in record.items() if key != "config"}
            | {
                "config_file": f"configs/{record['slug']}.json",
                "config_sha256": canonical_sha256(record["config"]),
            }
            for record in records
        ],
    }
    _write_json_new(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": _portable(output_dir),
                "runnable": len(records),
                "blocked": len(blocked),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
