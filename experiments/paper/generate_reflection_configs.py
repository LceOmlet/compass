from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.minibatch_config import minibatch_config_kwargs  # noqa: E402

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
    "bridge/minibatch_config.py",
    "bridge/paper_benchmark_registry.py",
    "experiments/paper/generate_reflection_configs.py",
    "experiments/paper/model_profiles.json",
    "experiments/paper/run_compass_reflection.py",
    "experiments/paper/serve_qwen3_8b_vllm_metax.sh",
)
SUBMODULES = (
    "upstreams/dspy",
    "upstreams/gepa",
    "upstreams/gepa-artifact",
)
TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _run_git(*args: str, cwd: Path = PROJECT_ROOT, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        ("git", *args),
        cwd=cwd,
        input=input_bytes,
        check=True,
        capture_output=True,
    ).stdout


def _git_blob_hash(payload: bytes) -> str:
    return _run_git("hash-object", "--stdin", input_bytes=payload).decode().strip()


def source_snapshot() -> dict[str, Any]:
    file_hashes: dict[str, str] = {}
    for relative in SNAPSHOT_FILES:
        path = PROJECT_ROOT / relative
        file_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    submodules: dict[str, dict[str, str]] = {}
    for relative in SUBMODULES:
        path = PROJECT_ROOT / relative
        submodules[relative] = {
            "head": _run_git("rev-parse", "HEAD", cwd=path).decode().strip(),
            "diff_git_blob": _git_blob_hash(
                _run_git("diff", "--binary", cwd=path)
            ),
        }

    return {
        "root_head": _run_git("rev-parse", "HEAD").decode().strip(),
        "root_diff_git_blob": _git_blob_hash(
            _run_git("diff", "--binary")
        ),
        "submodules": submodules,
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

    slug = (
        f"paper_{model_profile_name}_{condition}_{task_id}_"
        f"seed{seed}_{tag}"
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
            "max_candidate_workers": 3,
            "max_metric_calls": TASK_BUDGETS[task_id],
            "num_threads": 32,
            "parent_top_n": 5,
            "perfect_score": 1,
            "raise_on_exception": True,
            "skip_perfect_score": True,
            "track_best_outputs": True,
            "use_cloudpickle": True,
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
            )
            path = args.output_dir / f"{slug}.json"
            _write_new_json(path, config)
            written.append(str(path))
    print(json.dumps({"written": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
