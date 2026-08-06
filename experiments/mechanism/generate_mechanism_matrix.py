"""Generate the frozen HiTab/ChartQA/CLUTRR COMPASS mechanism matrix.

The generated directory is create-only.  It contains one validated runner
configuration per experiment, an immutable manifest with content hashes, and
separate text/visual launch queues.  Queue entries name API-key environment
variables but never contain credential values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
RUNNER: Final = PROJECT_ROOT / "experiments" / "mechanism" / "run_compass_reflection.py"
DEFAULT_MATRIX_ID: Final = "hitab_chartqa_clutrr_mechanism_v1"
DEFAULT_OUTPUT_DIR: Final = (
    PROJECT_ROOT / "experiments" / "mechanism" / "generated" / DEFAULT_MATRIX_ID
)
DEFAULT_RUNTIME_ROOT: Final = Path("F:/compass-mechanism-local")
DEFAULT_TEXT_API_BASE: Final = "http://127.0.0.1:40038/v1"
DEFAULT_VISUAL_API_BASE: Final = "http://127.0.0.1:40039/v1"
DEFAULT_CLUTRR_API_BASE: Final = "http://127.0.0.1:18000/v1"
DEFAULT_API_KEY_ENV: Final = "COMPASS_LITELLM_PROXY_KEY"
DEFAULT_CLUTRR_API_KEY_ENV: Final = "COMPASS_VLLM_API_KEY"
DEFAULT_CLUTRR_MODEL: Final = "openai/Qwen3-8B"
DEFAULT_HITAB_PYTHON: Final = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_CHARTQA_PYTHON: Final = (
    PROJECT_ROOT.parent / "skill-factory" / ".venv" / "Scripts" / "python.exe"
)
DEFAULT_CLUTRR_PYTHON: Final = Path(
    "F:/compass-ifbench-local/.venv-no-torch-py312/Scripts/python.exe"
)
DEFAULT_HITAB_RUNTIME_DEPS: Final = Path(
    "F:/compass-hitab-local/smoke-runtime-deps-py312"
)

CANDIDATE_PROPOSALS: Final = 128
PROPOSAL_MINIBATCH_SIZE: Final = 3
ADMISSION_MINIBATCH_SIZE: Final = 3
# Each admission item may first require one clean rollout of its frozen
# reference owner before the candidate rollout.  That reference work is part
# of the optimizer's official metric-call accounting.
ADMISSION_REFERENCE_CALLS_PER_CANDIDATE: Final = ADMISSION_MINIBATCH_SIZE
OPTIMIZATION_METRIC_CALL_CAP: Final = CANDIDATE_PROPOSALS * (
    PROPOSAL_MINIBATCH_SIZE
    + ADMISSION_REFERENCE_CALLS_PER_CANDIDATE
    + ADMISSION_MINIBATCH_SIZE
)
FINAL_VALIDATION_SIZE: Final = 300

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_ENV = re.compile(r"^[A-Z_][A-Z0-9_]*$")


@dataclass(frozen=True, slots=True)
class TaskProtocol:
    task_id: str
    seeds: tuple[int, ...]
    resource_class: str
    final_test_size: int


TASK_PROTOCOLS: Final[tuple[TaskProtocol, ...]] = (
    TaskProtocol("hitab", (0,), "text", 1584),
    TaskProtocol("chartqa", (0,), "visual", 2500),
    TaskProtocol("clutrr_supporting", (0, 1, 2), "text", 447),
    TaskProtocol("clutrr_irrelevant", (0, 1, 2), "text", 444),
    TaskProtocol("clutrr_disconnected", (0, 1, 2), "text", 445),
)

METHOD_CELLS: Final[tuple[tuple[str, str, str], ...]] = (
    ("raw_strict", "raw_frontier_rate", "strict_improvement"),
    ("raw_always", "raw_frontier_rate", "always_accept"),
    ("highres_strict", "high_resolution", "strict_improvement"),
    ("highres_always", "high_resolution", "always_accept"),
)

SOURCE_FILES: Final[tuple[str, ...]] = (
    "bridge/b19_reversible_parent_selection.py",
    "bridge/b20_compass_reflection.py",
    "bridge/minibatch_config.py",
    "bridge/mechanism_benchmark_registry.py",
    "bridge/mechanism_chartqa.py",
    "bridge/mechanism_clutrr.py",
    "bridge/mechanism_hitab.py",
    "bridge/paper_benchmark_registry.py",
    "bridge/request_deadline.py",
    "experiments/mechanism/generate_mechanism_matrix.py",
    "experiments/mechanism/run_compass_reflection.py",
    "experiments/paper/run_compass_reflection.py",
    "patches/dspy-working-tree.patch",
    "patches/gepa-artifact-working-tree.patch",
    "patches/gepa-working-tree.patch",
    "scripts/run-mechanism-matrix.sh",
    "upstreams.lock.json",
)

TASK_REQUIRED_ENV: Final[Mapping[str, tuple[str, ...]]] = {
    "hitab": ("HITAB_PREPARED_ROOT",),
    "chartqa": (
        "CHARTQA_PREPARED_ROOT",
        "CHARTQA_ROOT",
        "SKILL_FACTORY_ROOT",
    ),
    "clutrr_supporting": (
        "CLUTRR_HF_DATA_ROOT",
        "CLUTRR_BASELINE_ROOT",
    ),
    "clutrr_irrelevant": (
        "CLUTRR_HF_DATA_ROOT",
        "CLUTRR_BASELINE_ROOT",
    ),
    "clutrr_disconnected": (
        "CLUTRR_HF_DATA_ROOT",
        "CLUTRR_BASELINE_ROOT",
    ),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _portable_path(path: Path) -> str:
    """Keep generated paths directly usable from Git Bash and Windows Python."""

    return path.as_posix()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _write_json_new(path: Path, payload: Any) -> None:
    _write_new(path, _json_bytes(payload))


def _write_jsonl_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(
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
    )
    _write_new(path, payload)


def source_snapshot() -> dict[str, Any]:
    """Fingerprint the local semantic owners used by the generated configs."""

    missing = [relative for relative in SOURCE_FILES if not (PROJECT_ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"mechanism source files are missing: {missing}")
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "root_head": result.stdout.strip(),
        "file_sha256": {
            relative: _sha256_file(PROJECT_ROOT / relative)
            for relative in SOURCE_FILES
        },
    }


def metric_call_accounting(task: TaskProtocol) -> dict[str, int]:
    """Return the frozen optimization cap and separately scoped final calls."""

    final_calls = FINAL_VALIDATION_SIZE + task.final_test_size
    return {
        "candidate_proposals": CANDIDATE_PROPOSALS,
        "proposal_calls_per_candidate": PROPOSAL_MINIBATCH_SIZE,
        "admission_reference_calls_per_candidate": (
            ADMISSION_REFERENCE_CALLS_PER_CANDIDATE
        ),
        "admission_calls_per_candidate": ADMISSION_MINIBATCH_SIZE,
        "optimization_metric_call_cap": OPTIMIZATION_METRIC_CALL_CAP,
        # SparseMinibatchEvaluationPolicy.get_seed_eval_batch returns [].
        "sparse_seed_evaluation_calls": 0,
        # The shared runner evaluates the frozen winner after optimize returns.
        "final_validation_calls_outside_optimizer_cap": FINAL_VALIDATION_SIZE,
        "final_test_calls_outside_optimizer_cap": task.final_test_size,
        "planned_total_calls_ceiling": OPTIMIZATION_METRIC_CALL_CAP + final_calls,
    }


def _model_config(
    *,
    task_id: str,
    resource_class: str,
    text_api_base: str,
    visual_api_base: str,
    clutrr_api_base: str,
    text_api_key_env: str,
    visual_api_key_env: str,
    clutrr_api_key_env: str,
    clutrr_model: str,
) -> dict[str, Any]:
    if resource_class not in {"text", "visual"}:
        raise ValueError(f"unknown resource class: {resource_class!r}")
    if task_id.startswith("clutrr_"):
        api_base = clutrr_api_base
        api_key_env = clutrr_api_key_env
        model = clutrr_model
    else:
        api_base = text_api_base if resource_class == "text" else visual_api_base
        api_key_env = (
            text_api_key_env if resource_class == "text" else visual_api_key_env
        )
        model = (
            "openai/compass-qwen3-8b"
            if resource_class == "text"
            else "openai/compass-qwen3-vl-8b-thinking"
        )
    if not isinstance(api_base, str) or not api_base.strip():
        raise TypeError(f"{resource_class} API base must be non-empty text")
    if not _SAFE_ENV.fullmatch(api_key_env):
        raise ValueError(f"unsafe API-key environment name: {api_key_env!r}")
    if not isinstance(model, str) or not model.strip():
        raise TypeError(f"{task_id} model must be non-empty text")
    return {
        "api_base": api_base,
        "api_key_env": api_key_env,
        "cache": True,
        "cache_in_memory": True,
        # Qwen3-VL-8B-Thinking is intrinsically a thinking checkpoint and its
        # owner endpoint rejects the optional ``enable_thinking`` parameter.
        "enable_thinking": resource_class != "visual",
        "max_tokens": 16384,
        "model": model,
        "model_type": "chat",
        "num_retries": 0,
        "temperature": 0.6,
        "timeout": 1200,
        "top_k": 20,
        "top_p": 0.95,
    }


def build_run_config(
    *,
    matrix_id: str,
    task: TaskProtocol,
    seed: int,
    cell_slug: str,
    selection_mode: str,
    acceptance_mode: str,
    runtime_root: Path,
    snapshot: Mapping[str, Any],
    text_api_base: str = DEFAULT_TEXT_API_BASE,
    visual_api_base: str = DEFAULT_VISUAL_API_BASE,
    clutrr_api_base: str = DEFAULT_CLUTRR_API_BASE,
    text_api_key_env: str = DEFAULT_API_KEY_ENV,
    visual_api_key_env: str = DEFAULT_API_KEY_ENV,
    clutrr_api_key_env: str = DEFAULT_CLUTRR_API_KEY_ENV,
    clutrr_model: str = DEFAULT_CLUTRR_MODEL,
) -> tuple[str, dict[str, Any]]:
    """Build one config without changing selection or admission semantics."""

    if not _SAFE_NAME.fullmatch(matrix_id):
        raise ValueError("matrix_id must be path-safe")
    if not _SAFE_NAME.fullmatch(cell_slug):
        raise ValueError("cell_slug must be path-safe")
    if selection_mode not in {"raw_frontier_rate", "high_resolution"}:
        raise ValueError("unsupported parent-selection score mode")
    if acceptance_mode not in {"strict_improvement", "always_accept"}:
        raise ValueError("unsupported admission mode")
    if seed not in task.seeds:
        raise ValueError(f"seed {seed} is not frozen for {task.task_id}")

    slug = f"{matrix_id}_{task.task_id}_seed{seed}_{cell_slug}"
    model = _model_config(
        task_id=task.task_id,
        resource_class=task.resource_class,
        text_api_base=text_api_base,
        visual_api_base=visual_api_base,
        clutrr_api_base=clutrr_api_base,
        text_api_key_env=text_api_key_env,
        visual_api_key_env=visual_api_key_env,
        clutrr_api_key_env=clutrr_api_key_env,
        clutrr_model=clutrr_model,
    )
    config = {
        "cache_dir": _portable_path(runtime_root / "cache" / slug),
        "condition": "compass_reflection",
        "dataset_mode": "lite",
        "model": model,
        "optimizer": {
            "acceptance_mode": acceptance_mode,
            "add_format_failure_as_feedback": False,
            "admission_minibatch_size": ADMISSION_MINIBATCH_SIZE,
            "display_progress_bar": False,
            "epoch_parallel_enabled": False,
            "evaluation_straggler_timeout": 0,
            "failure_score": 0,
            "max_candidate_proposals": CANDIDATE_PROPOSALS,
            "max_candidate_workers": 1,
            "max_metric_calls": OPTIMIZATION_METRIC_CALL_CAP,
            "max_reflection_workers": 1,
            "num_threads": 32,
            "parent_selection_score_mode": selection_mode,
            "parent_top_n": 5,
            "perfect_score": 1,
            "proposal_minibatch_size": PROPOSAL_MINIBATCH_SIZE,
            "proposal_timeout_seconds": 1200,
            "raise_on_exception": True,
            "rollout_timeout_seconds": 1200,
            "skip_perfect_score": True,
            "track_best_outputs": True,
            "use_cloudpickle": True,
        },
        "optimizer_seed": seed,
        "run_dir": _portable_path(runtime_root / "runs" / slug),
        "source_snapshot": dict(snapshot),
        "task_id": task.task_id,
    }
    return slug, config


def build_matrix(
    *,
    matrix_id: str,
    runtime_root: Path,
    snapshot: Mapping[str, Any],
    text_api_base: str = DEFAULT_TEXT_API_BASE,
    visual_api_base: str = DEFAULT_VISUAL_API_BASE,
    clutrr_api_base: str = DEFAULT_CLUTRR_API_BASE,
    text_api_key_env: str = DEFAULT_API_KEY_ENV,
    visual_api_key_env: str = DEFAULT_API_KEY_ENV,
    clutrr_api_key_env: str = DEFAULT_CLUTRR_API_KEY_ENV,
    clutrr_model: str = DEFAULT_CLUTRR_MODEL,
) -> list[dict[str, Any]]:
    """Build all 44 frozen task/seed/method records in stable order."""

    records: list[dict[str, Any]] = []
    for task in TASK_PROTOCOLS:
        for seed in task.seeds:
            for cell_slug, selection_mode, acceptance_mode in METHOD_CELLS:
                slug, config = build_run_config(
                    matrix_id=matrix_id,
                    task=task,
                    seed=seed,
                    cell_slug=cell_slug,
                    selection_mode=selection_mode,
                    acceptance_mode=acceptance_mode,
                    runtime_root=runtime_root,
                    snapshot=snapshot,
                    text_api_base=text_api_base,
                    visual_api_base=visual_api_base,
                    clutrr_api_base=clutrr_api_base,
                    text_api_key_env=text_api_key_env,
                    visual_api_key_env=visual_api_key_env,
                    clutrr_api_key_env=clutrr_api_key_env,
                    clutrr_model=clutrr_model,
                )
                records.append(
                    {
                        "slug": slug,
                        "task_id": task.task_id,
                        "seed": seed,
                        "cell": cell_slug,
                        "selection_mode": selection_mode,
                        "acceptance_mode": acceptance_mode,
                        "resource_class": task.resource_class,
                        "metric_calls": metric_call_accounting(task),
                        "config": config,
                    }
                )
    if len(records) != 44:
        raise AssertionError(f"frozen mechanism matrix must have 44 runs, got {len(records)}")
    return records


def _queue_entry(
    record: Mapping[str, Any],
    *,
    config_path: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    config = record["config"]
    slug = record["slug"]
    key_env = config["model"]["api_key_env"]
    required_env = tuple(dict.fromkeys((key_env, *TASK_REQUIRED_ENV[record["task_id"]])))
    common_pythonpath = (
        PROJECT_ROOT,
        PROJECT_ROOT / "upstreams" / "dspy",
        PROJECT_ROOT / "upstreams" / "gepa" / "src",
        PROJECT_ROOT / "upstreams" / "gepa-artifact",
    )
    if record["task_id"] == "chartqa":
        python_executable = DEFAULT_CHARTQA_PYTHON
        pythonpath = (
            *common_pythonpath,
            PROJECT_ROOT
            / "upstreams"
            / "skill-factory"
            / "upstream"
            / "lmms-eval",
        )
    elif str(record["task_id"]).startswith("clutrr_"):
        python_executable = DEFAULT_CLUTRR_PYTHON
        pythonpath = common_pythonpath
    else:
        python_executable = DEFAULT_HITAB_PYTHON
        pythonpath = (*common_pythonpath, DEFAULT_HITAB_RUNTIME_DEPS)
    return {
        "slug": slug,
        "task_id": record["task_id"],
        "seed": record["seed"],
        "cell": record["cell"],
        "resource_class": record["resource_class"],
        "api_base": config["model"]["api_base"],
        "api_key_env": key_env,
        "required_env": list(required_env),
        "cwd": _portable_path(PROJECT_ROOT),
        "argv": (
            _portable_path(python_executable),
            _portable_path(RUNNER),
            "--config",
            _portable_path(config_path.resolve()),
        ),
        "pythonpath": [_portable_path(path) for path in pythonpath],
        "stdout_log": _portable_path(runtime_root / "logs" / f"{slug}.stdout.log"),
        "stderr_log": _portable_path(runtime_root / "logs" / f"{slug}.stderr.log"),
        "pid_file": _portable_path(runtime_root / "logs" / f"{slug}.pid"),
        "run_dir": config["run_dir"],
        "cache_dir": config["cache_dir"],
    }


def write_matrix(
    output_dir: Path,
    *,
    matrix_id: str = DEFAULT_MATRIX_ID,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    snapshot: Mapping[str, Any] | None = None,
    text_api_base: str = DEFAULT_TEXT_API_BASE,
    visual_api_base: str = DEFAULT_VISUAL_API_BASE,
    clutrr_api_base: str = DEFAULT_CLUTRR_API_BASE,
    text_api_key_env: str = DEFAULT_API_KEY_ENV,
    visual_api_key_env: str = DEFAULT_API_KEY_ENV,
    clutrr_api_key_env: str = DEFAULT_CLUTRR_API_KEY_ENV,
    clutrr_model: str = DEFAULT_CLUTRR_MODEL,
) -> Path:
    """Create a content-addressed matrix directory and refuse overwrites."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    frozen_snapshot = dict(snapshot) if snapshot is not None else source_snapshot()
    records = build_matrix(
        matrix_id=matrix_id,
        runtime_root=runtime_root,
        snapshot=frozen_snapshot,
        text_api_base=text_api_base,
        visual_api_base=visual_api_base,
        clutrr_api_base=clutrr_api_base,
        text_api_key_env=text_api_key_env,
        visual_api_key_env=visual_api_key_env,
        clutrr_api_key_env=clutrr_api_key_env,
        clutrr_model=clutrr_model,
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    config_entries: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    for record in records:
        relative_config = Path("configs") / f"{record['slug']}.json"
        config_path = output_dir / relative_config
        _write_json_new(config_path, record["config"])
        queue_entry = _queue_entry(
            record,
            config_path=config_path,
            runtime_root=runtime_root,
        )
        queue_entry["config_file"] = relative_config.as_posix()
        queue_entry["config_sha256"] = _sha256_file(config_path)
        queue.append(queue_entry)
        config_entries.append(
            {
                key: record[key]
                for key in (
                    "slug",
                    "task_id",
                    "seed",
                    "cell",
                    "selection_mode",
                    "acceptance_mode",
                    "resource_class",
                    "metric_calls",
                )
            }
            | {
                "config_file": relative_config.as_posix(),
                "config_sha256": _sha256_file(config_path),
            }
        )

    queue_paths = {
        "all": output_dir / "launch_queue_all.jsonl",
        "text": output_dir / "launch_queue_text.jsonl",
        "visual": output_dir / "launch_queue_visual.jsonl",
    }
    _write_jsonl_new(queue_paths["all"], queue)
    for resource_class in ("text", "visual"):
        _write_jsonl_new(
            queue_paths[resource_class],
            [row for row in queue if row["resource_class"] == resource_class],
        )

    manifest = {
        "schema_version": 1,
        "matrix_id": matrix_id,
        "artifact_policy": "create_only_no_overwrite",
        "run_count": len(records),
        "resource_counts": {
            resource_class: sum(
                record["resource_class"] == resource_class for record in records
            )
            for resource_class in ("text", "visual")
        },
        "protocol": {
            "method_cells": [
                {
                    "cell": cell,
                    "parent_selection_score_mode": selection,
                    "acceptance_mode": acceptance,
                }
                for cell, selection, acceptance in METHOD_CELLS
            ],
            "candidate_budget_owner": "gepa.utils.MaxCandidateProposalsStopper",
            "max_candidate_proposals": CANDIDATE_PROPOSALS,
            "epoch_parallel_enabled": False,
            "proposal_minibatch_size": PROPOSAL_MINIBATCH_SIZE,
            "admission_minibatch_size": ADMISSION_MINIBATCH_SIZE,
            "reflection_minibatch_size": None,
            "max_candidate_workers": 1,
            "max_reflection_workers": 1,
            "parent_top_n": 5,
            "dci": False,
            "teacher_forcing": False,
            "flashtrace": False,
            "rollout_timeout_seconds": 1200,
            "proposal_timeout_seconds": 1200,
            "failure_score": 0,
            "perfect_score": 1,
            "skip_perfect_score": True,
            "raise_on_exception": True,
            "optimization_metric_call_cap": OPTIMIZATION_METRIC_CALL_CAP,
            "optimization_cap_accounting": (
                "128 * (3 propose + up to 3 admit-reference + 3 admit) = 1152"
            ),
            "final_evaluation_scope": "outside_optimizer_cap",
        },
        "tasks": [
            {
                "task_id": task.task_id,
                "seeds": list(task.seeds),
                "resource_class": task.resource_class,
                "final_validation_size": FINAL_VALIDATION_SIZE,
                "final_test_size": task.final_test_size,
                "metric_calls": metric_call_accounting(task),
            }
            for task in TASK_PROTOCOLS
        ],
        "source_snapshot": frozen_snapshot,
        "runs": config_entries,
        "launch_queues": {
            name: {
                "file": path.name,
                "sha256": _sha256_file(path),
                "count": (
                    len(queue)
                    if name == "all"
                    else sum(row["resource_class"] == name for row in queue)
                ),
            }
            for name, path in queue_paths.items()
        },
        "credential_policy": "api_key_environment_names_only",
    }
    manifest_path = output_dir / "matrix_manifest.json"
    _write_json_new(manifest_path, manifest)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--matrix-id", default=DEFAULT_MATRIX_ID)
    parser.add_argument("--text-api-base", default=DEFAULT_TEXT_API_BASE)
    parser.add_argument("--visual-api-base", default=DEFAULT_VISUAL_API_BASE)
    parser.add_argument("--clutrr-api-base", default=DEFAULT_CLUTRR_API_BASE)
    parser.add_argument("--text-api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--visual-api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument(
        "--clutrr-api-key-env",
        default=DEFAULT_CLUTRR_API_KEY_ENV,
    )
    parser.add_argument("--clutrr-model", default=DEFAULT_CLUTRR_MODEL)
    args = parser.parse_args()
    manifest = write_matrix(
        args.output_dir,
        matrix_id=args.matrix_id,
        runtime_root=args.runtime_root,
        text_api_base=args.text_api_base,
        visual_api_base=args.visual_api_base,
        clutrr_api_base=args.clutrr_api_base,
        text_api_key_env=args.text_api_key_env,
        visual_api_key_env=args.visual_api_key_env,
        clutrr_api_key_env=args.clutrr_api_key_env,
        clutrr_model=args.clutrr_model,
    )
    print(
        json.dumps(
            {"manifest": _portable_path(manifest), "run_count": 44},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
