from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.minibatch_config import minibatch_config_kwargs  # noqa: E402
from bridge.paper_source_snapshot import (  # noqa: E402
    build_frozen_upstream_snapshot,
)

DEFAULT_REMOTE_ROOT = Path(
    "/mnt/geogpt-doc-new/deepresearch/gepa-multi-skill/reflection-bridge"
)
TASK_BUDGETS = {
    "hotpotqa": 6871,
    "hover": 7051,
    "ifbench": 3593,
    "pupa": 2426,
    "aime_2025": 1839,
    "livebench_math": 1839,
}
REFLECTION_CONDITIONS = {
    "mini_admission_reflection",
    "compass_reflection",
}
SNAPSHOT_FILES = (
    "bridge/b16_official_gepa_ifbench.py",
    "bridge/b19_reversible_parent_selection.py",
    "bridge/b20_compass_reflection.py",
    "bridge/b21_joint_linucb_scheduler.py",
    "bridge/b22_repairable_gap_sampling.py",
    "bridge/dci_agent_lite.py",
    "bridge/dci_compass.py",
    "bridge/dci_docker_isolation.py",
    "bridge/minibatch_config.py",
    "bridge/paper_benchmark_registry.py",
    "bridge/paper_source_snapshot.py",
    "bridge/prompts/dci_subproblem_free_text.txt",
    "bridge/request_deadline.py",
    "docker/dci-sandbox/Dockerfile",
    "docker/dci-sandbox/entrypoint.sh",
    "experiments/paper/dci_agent_qwen3_5_9b/models.json",
    "experiments/paper/generate_reflection_configs.py",
    "experiments/paper/model_profiles.json",
    "experiments/paper/run_compass_reflection.py",
    "experiments/paper/serve_qwen3_8b_vllm_metax.sh",
    "scripts/start_aime_v47_v50_gepa_parity_local.ps1",
    "scripts/start_aime_v51_v54_window1_timeout6000_local.ps1",
)
TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _run_git(*args: str, cwd: Path = PROJECT_ROOT) -> bytes:
    return subprocess.run(
        ("git", *args),
        cwd=cwd,
        check=True,
        capture_output=True,
    ).stdout


def source_snapshot() -> dict[str, Any]:
    file_hashes: dict[str, str] = {}
    for relative in SNAPSHOT_FILES:
        path = PROJECT_ROOT / relative
        file_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "root_head": _run_git("rev-parse", "HEAD").decode().strip(),
        "submodules": build_frozen_upstream_snapshot(PROJECT_ROOT),
        "file_sha256": file_hashes,
    }


def load_model_profiles(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not raw:
        raise TypeError("model profile file must contain a non-empty object")
    profiles: dict[str, dict[str, Any]] = {}
    for name, profile in raw.items():
        if not isinstance(name, str) or not name or not isinstance(profile, Mapping):
            raise TypeError("model profiles must map non-empty names to objects")
        profiles[name] = dict(profile)
    return profiles


def build_run_config(
    *,
    task_id: str,
    condition: str,
    seed: int,
    tag: str,
    model_profile_name: str,
    model_profile: Mapping[str, Any],
    remote_root: Path,
    snapshot: Mapping[str, Any],
    proposal_minibatch_size: int | None = None,
    admission_minibatch_size: int | None = None,
    acceptance_mode: str = "strict_improvement",
    epoch_parallel_enabled: bool = False,
    max_candidate_workers: int = 3,
    max_reflection_workers: int = 1,
    parent_selection_score_mode: str = "high_resolution",
    proposal_sampling_mode: str = "independent",
    proposal_tasks_per_iteration: int | None = None,
    rollout_timeout_seconds: float | None = None,
    proposal_timeout_seconds: float | None = None,
) -> tuple[str, dict[str, Any]]:
    if task_id not in TASK_BUDGETS:
        raise ValueError(f"unknown paper task: {task_id!r}")
    if condition not in REFLECTION_CONDITIONS:
        raise ValueError(f"unknown reflection condition: {condition!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TypeError("seed must be a non-negative integer")
    if not TAG_PATTERN.fullmatch(tag):
        raise ValueError("tag must contain only letters, digits, dot, dash, underscore")
    if not TAG_PATTERN.fullmatch(model_profile_name):
        raise ValueError("model profile name is not path-safe")
    if acceptance_mode not in {"strict_improvement", "always_accept"}:
        raise ValueError("unknown acceptance mode")
    if parent_selection_score_mode not in {
        "raw_frontier_rate",
        "high_resolution",
        "high_resolution_lexicographic",
    }:
        raise ValueError("unknown parent-selection score mode")
    if proposal_sampling_mode not in {
        "independent",
        "joint_linucb",
        "repairable_gap",
    }:
        raise ValueError("unknown proposal-sampling mode")
    if not isinstance(epoch_parallel_enabled, bool):
        raise TypeError("epoch_parallel_enabled must be a boolean")
    if proposal_sampling_mode in {"joint_linucb", "repairable_gap"}:
        if not epoch_parallel_enabled:
            raise ValueError(
                f"{proposal_sampling_mode} requires epoch_parallel_enabled=true"
            )
        if parent_selection_score_mode != "high_resolution_lexicographic":
            raise ValueError(
                f"{proposal_sampling_mode} requires "
                "parent_selection_score_mode='high_resolution_lexicographic'"
            )
    for name, value in (
        ("max_candidate_workers", max_candidate_workers),
        ("max_reflection_workers", max_reflection_workers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TypeError(f"{name} must be a positive integer")
    if proposal_tasks_per_iteration is not None and (
        isinstance(proposal_tasks_per_iteration, bool)
        or not isinstance(proposal_tasks_per_iteration, int)
        or proposal_tasks_per_iteration <= 0
    ):
        raise TypeError("proposal_tasks_per_iteration must be a positive integer")
    if proposal_tasks_per_iteration is not None and not epoch_parallel_enabled:
        raise ValueError(
            "proposal_tasks_per_iteration requires epoch_parallel_enabled=true"
        )
    for name, value in (
        ("rollout_timeout_seconds", rollout_timeout_seconds),
        ("proposal_timeout_seconds", proposal_timeout_seconds),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise TypeError(f"{name} must be a positive number")
    if (
        proposal_minibatch_size is not None
        or admission_minibatch_size is not None
    ) and condition != "compass_reflection":
        raise ValueError(
            "split train/validation admission requires "
            "condition='compass_reflection'"
        )
    raw_minibatch_config = (
        {"reflection_minibatch_size": 3}
        if proposal_minibatch_size is None
        and admission_minibatch_size is None
        else {
            name: value
            for name, value in (
                ("proposal_minibatch_size", proposal_minibatch_size),
                ("admission_minibatch_size", admission_minibatch_size),
            )
            if value is not None
        }
    )
    minibatch_config = {
        name: value
        for name, value in minibatch_config_kwargs(
            raw_minibatch_config,
            namespace="optimizer",
        ).items()
        if value is not None
    }

    sampling_suffix = {
        "independent": "",
        "joint_linucb": "_joint_linucb",
        "repairable_gap": "_repairable_gap",
    }[proposal_sampling_mode]
    slug = (
        f"paper_{model_profile_name}_{condition}_{task_id}_"
        f"seed{seed}_{tag}{sampling_suffix}"
    )
    config = {
        "cache_dir": str(remote_root / f"cache_{slug}"),
        "condition": condition,
        "dataset_mode": "lite",
        "model": dict(model_profile),
        "optimizer": {
            "add_format_failure_as_feedback": False,
            "display_progress_bar": False,
            "failure_score": 0,
            "acceptance_mode": acceptance_mode,
            "epoch_parallel_enabled": epoch_parallel_enabled,
            "evaluation_straggler_timeout": 0,
            "max_candidate_workers": max_candidate_workers,
            "max_metric_calls": TASK_BUDGETS[task_id],
            "max_reflection_workers": max_reflection_workers,
            "num_threads": 32,
            "parent_selection_score_mode": parent_selection_score_mode,
            "parent_top_n": 5,
            "perfect_score": 1,
            "raise_on_exception": True,
            "skip_perfect_score": True,
            "track_best_outputs": True,
            "use_cloudpickle": True,
            "proposal_tasks_per_iteration": proposal_tasks_per_iteration,
            **(
                {"proposal_sampling_mode": proposal_sampling_mode}
                if proposal_sampling_mode != "independent"
                else {}
            ),
            **(
                {"rollout_timeout_seconds": rollout_timeout_seconds}
                if rollout_timeout_seconds is not None
                else {}
            ),
            **(
                {"proposal_timeout_seconds": proposal_timeout_seconds}
                if proposal_timeout_seconds is not None
                else {}
            ),
            **minibatch_config,
        },
        "optimizer_seed": seed,
        "run_dir": str(remote_root / "runs" / slug),
        "source_snapshot": dict(snapshot),
        "task_id": task_id,
    }
    return slug, config


def _write_new_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", required=True)
    parser.add_argument(
        "--condition",
        required=True,
        choices=sorted(REFLECTION_CONDITIONS),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(TASK_BUDGETS),
        default=tuple(TASK_BUDGETS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0,))
    parser.add_argument("--proposal-minibatch-size", type=int)
    parser.add_argument("--admission-minibatch-size", type=int)
    parser.add_argument(
        "--acceptance-mode",
        choices=("strict_improvement", "always_accept"),
        default="strict_improvement",
    )
    parser.add_argument(
        "--parent-selection-score-mode",
        choices=(
            "raw_frontier_rate",
            "high_resolution",
            "high_resolution_lexicographic",
        ),
        default="high_resolution",
    )
    parser.add_argument("--epoch-parallel-enabled", action="store_true")
    parser.add_argument(
        "--proposal-sampling-mode",
        choices=("independent", "joint_linucb", "repairable_gap"),
        default="independent",
    )
    parser.add_argument("--proposal-tasks-per-iteration", type=int)
    parser.add_argument("--rollout-timeout-seconds", type=float)
    parser.add_argument("--proposal-timeout-seconds", type=float)
    parser.add_argument("--max-candidate-workers", type=int, default=3)
    parser.add_argument("--max-reflection-workers", type=int, default=1)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--model-profiles",
        type=Path,
        default=PROJECT_ROOT / "experiments/paper/model_profiles.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/paper/generated",
    )
    parser.add_argument(
        "--remote-root",
        type=Path,
        default=DEFAULT_REMOTE_ROOT,
    )
    args = parser.parse_args()

    profiles = load_model_profiles(args.model_profiles)
    if args.model_profile not in profiles:
        raise ValueError(f"unknown model profile: {args.model_profile!r}")
    snapshot = source_snapshot()
    written: list[str] = []
    for task_id in args.tasks:
        for seed in args.seeds:
            slug, config = build_run_config(
                task_id=task_id,
                condition=args.condition,
                seed=seed,
                tag=args.tag,
                model_profile_name=args.model_profile,
                model_profile=profiles[args.model_profile],
                remote_root=args.remote_root,
                snapshot=snapshot,
                proposal_minibatch_size=args.proposal_minibatch_size,
                admission_minibatch_size=args.admission_minibatch_size,
                acceptance_mode=args.acceptance_mode,
                epoch_parallel_enabled=args.epoch_parallel_enabled,
                max_candidate_workers=args.max_candidate_workers,
                max_reflection_workers=args.max_reflection_workers,
                parent_selection_score_mode=(
                    args.parent_selection_score_mode
                ),
                proposal_sampling_mode=args.proposal_sampling_mode,
                proposal_tasks_per_iteration=(
                    args.proposal_tasks_per_iteration
                ),
                rollout_timeout_seconds=args.rollout_timeout_seconds,
                proposal_timeout_seconds=args.proposal_timeout_seconds,
            )
            path = args.output_dir / f"{slug}.json"
            _write_new_json(path, config)
            written.append(str(path))
    print(json.dumps({"written": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
