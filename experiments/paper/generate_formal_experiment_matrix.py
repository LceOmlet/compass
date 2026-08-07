"""Freeze the paper experiment ledger into monitor- and launcher-facing queues.

This module is deliberately an orchestration boundary.  It never imports or
reimplements an optimizer, benchmark metric, retry loop, or result parser.
Actual commands arrive through a separately frozen binding file and must name
an existing Python runner plus a content-hashed config.  Missing bindings stay
visible as blocked rows and cannot be launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from bridge.paper_source_snapshot import (
    build_frozen_upstream_snapshot,
    verify_project_source_snapshot,
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER: Final = (
    PROJECT_ROOT / "experiments/paper/formal_experiment_ledger_v1.json"
)
DEFAULT_OUTPUT: Final = (
    PROJECT_ROOT / "experiments/paper/generated/compass_paper_primary_ablation_v1"
)
SAFE_ID: Final = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SAFE_ENV: Final = re.compile(r"^[A-Z_][A-Z0-9_]*$")
SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
PHASES: Final = ("preflight", "formal")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _portable(path: Path) -> str:
    return path.as_posix()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
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


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _reject_secret_values(value: Any, *, path: str = "root") -> None:
    if isinstance(value, str):
        lowered = value.lower()
        if value.startswith("sk-") or "bearer sk-" in lowered:
            raise ValueError(f"credential-like value is forbidden at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_secret_values(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secret_values(item, path=f"{path}[{index}]")


def _require_sha256(value: Any, *, name: str) -> None:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{name} must be an exact lowercase SHA-256 digest")


def load_ledger(path: Path = DEFAULT_LEDGER) -> dict[str, Any]:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(ledger, Mapping):
        raise TypeError("formal experiment ledger must be a JSON object")
    ledger = dict(ledger)
    _reject_secret_values(ledger)
    if ledger.get("schema_version") != 1:
        raise ValueError("unsupported formal experiment ledger schema")
    matrix_id = ledger.get("matrix_id")
    if not isinstance(matrix_id, str) or not SAFE_ID.fullmatch(matrix_id):
        raise ValueError("matrix_id must be a safe lowercase identifier")
    if ledger.get("credential_policy") != "environment_variable_names_only":
        raise ValueError("credential policy must keep values outside artifacts")
    semantic = ledger.get("semantic_policy")
    required_semantics = {
        "official_benchmark_program_metric_and_retry_owners_remain_unchanged",
        "no_shadow_optimizer_or_metric",
        "no_runtime_watchdog_or_proactive_interruption",
        "no_dynamic_worker_change",
        "no_new_optimizer_gate",
        "preflight_and_formal_state_and_cache_are_disjoint",
    }
    if not isinstance(semantic, Mapping) or any(
        semantic.get(name) is not True for name in required_semantics
    ):
        raise ValueError("ledger semantic policy is incomplete")
    for scope, root in ledger.get("runtime_roots", {}).items():
        if scope not in {"primary", "ablation"}:
            raise ValueError(f"unknown runtime scope: {scope}")
        if not isinstance(root, str) or not root.startswith("F:/"):
            raise ValueError(f"{scope} runtime root must be on F:")
    for name, profile in ledger.get("model_panels", {}).items():
        if not SAFE_ID.fullmatch(name) or not isinstance(profile, Mapping):
            raise ValueError("invalid model panel")
        if not SAFE_ENV.fullmatch(str(profile.get("api_key_env", ""))):
            raise ValueError("model panel must name a safe credential environment")
        if "api_key" in profile:
            raise ValueError("credential values cannot be serialized")
    _validate_task_protocols(ledger)
    _validate_cells(ledger)
    return ledger


def _validate_task_protocols(ledger: Mapping[str, Any]) -> None:
    tasks = ledger.get("task_protocols")
    if not isinstance(tasks, Mapping) or set(tasks) != {
        "ifbench",
        "aime_2025",
        "chartqa",
        "tau2_airline",
        "clutrr_irrelevant",
    }:
        raise ValueError("task protocols must contain the frozen five tasks")
    for task_id, protocol in tasks.items():
        if not isinstance(protocol, Mapping):
            raise TypeError(f"task protocol must be an object: {task_id}")
        scope = protocol.get("scope")
        if scope not in {"primary", "ablation"}:
            raise ValueError(f"invalid task scope: {task_id}")
        if protocol.get("resource_class") not in ledger["resource_classes"]:
            raise ValueError(f"unknown resource class for {task_id}")
        if protocol.get("model_panel") not in ledger["model_panels"]:
            raise ValueError(f"unknown model panel for {task_id}")
        split = protocol.get("split_identity")
        if not isinstance(split, Mapping):
            raise TypeError(f"missing split identity: {task_id}")
        fingerprints = split.get("fingerprints")
        if fingerprints is not None:
            if set(fingerprints) != {"train", "validation", "test"}:
                raise ValueError(f"split fingerprints are incomplete: {task_id}")
            for split_name, digest in fingerprints.items():
                _require_sha256(digest, name=f"{task_id}.{split_name}")
    tau = tasks["tau2_airline"]["split_identity"]
    train_ids = tau["train_ids"]
    proposal_ids = tau["proposal_ids"]
    validation_ids = tau["validation_ids"]
    test_ids = tau["test_ids"]
    if (
        len(train_ids) != 30
        or len(test_ids) != 20
        or len(proposal_ids) != 24
        or len(validation_ids) != 6
        or set(proposal_ids).intersection(validation_ids)
        or set(proposal_ids).union(validation_ids) != set(train_ids)
        or set(train_ids).intersection(test_ids)
    ):
        raise ValueError("tau2 Airline split/optimization identities are invalid")


def _validate_cells(ledger: Mapping[str, Any]) -> None:
    methods = ledger["methods"]
    tasks = ledger["task_protocols"]
    primary = ledger.get("primary_cells")
    if not isinstance(primary, list) or len(primary) != 20:
        raise ValueError("primary matrix must contain exactly 20 cells")
    identities: list[tuple[str, str]] = []
    dispositions = Counter()
    for cell in primary:
        if cell.get("task") not in tasks or tasks[cell["task"]]["scope"] != "primary":
            raise ValueError("primary cell references an unknown primary task")
        if cell.get("method") not in methods:
            raise ValueError("primary cell references an unknown method")
        disposition = cell.get("disposition")
        if disposition not in {"required_new", "reusable", "not_applicable"}:
            raise ValueError("unknown primary disposition")
        dispositions[disposition] += 1
        identities.append((cell["task"], cell["method"]))
    if len(set(identities)) != 20:
        raise ValueError("primary cells must be unique")
    if dispositions != {"required_new": 17, "reusable": 2, "not_applicable": 1}:
        raise ValueError(f"primary dispositions changed: {dict(dispositions)}")
    tau_mipro = next(
        cell
        for cell in primary
        if cell["task"] == "tau2_airline" and cell["method"] == "miprov2"
    )
    if tau_mipro["disposition"] != "not_applicable":
        raise ValueError("tau2 MIPROv2 must remain explicitly not applicable")
    ablation = ledger.get("ablation_factorial")
    if (
        not isinstance(ablation, Mapping)
        or ablation.get("task") != "clutrr_irrelevant"
        or ablation.get("method") != "compass"
        or ablation.get("seeds") != [0, 1, 2]
        or len(ablation.get("cells", ())) != 4
    ):
        raise ValueError("CLUTRR ablation factorial changed")


def expand_cells(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = ledger["task_protocols"]
    methods = ledger["methods"]
    rows: list[dict[str, Any]] = []
    for source in ledger["primary_cells"]:
        task_id = source["task"]
        method_id = source["method"]
        task = tasks[task_id]
        seed = task["optimizer_seed"]
        rows.append(
            {
                "cell_id": f"primary.{task_id}.{method_id}.seed{seed}",
                "scope": "primary",
                "task_id": task_id,
                "method_id": method_id,
                "seed": seed,
                "disposition": source["disposition"],
                "reason": source.get("reason"),
                "published_reference_percent": source.get(
                    "published_reference_percent"
                ),
                "optimization": methods[method_id]["optimization"],
                "owner_interface": methods[method_id]["owner_interface"],
                "resource_class": task["resource_class"],
                "capacity_group": ledger["resource_classes"][task["resource_class"]][
                    "capacity_group"
                ],
                "model_panel": task["model_panel"],
                "optimization_rollout_budget": (
                    task["optimization_rollout_budget"]
                    if methods[method_id]["optimization"]
                    else 0
                ),
                "preflight_rollout_budget": (
                    task["preflight_rollout_budget"]
                    if methods[method_id]["optimization"]
                    and source["disposition"] == "required_new"
                    else 0
                ),
                "task_protocol": task,
                "ablation_controls": None,
            }
        )
    factorial = ledger["ablation_factorial"]
    task = tasks[factorial["task"]]
    for seed in factorial["seeds"]:
        for control in factorial["cells"]:
            rows.append(
                {
                    "cell_id": (
                        f"ablation.{factorial['task']}.{control['name']}.seed{seed}"
                    ),
                    "scope": "ablation",
                    "task_id": factorial["task"],
                    "method_id": factorial["method"],
                    "seed": seed,
                    "disposition": "required_new",
                    "reason": None,
                    "published_reference_percent": None,
                    "optimization": True,
                    "owner_interface": methods[factorial["method"]]["owner_interface"],
                    "resource_class": task["resource_class"],
                    "capacity_group": ledger["resource_classes"][
                        task["resource_class"]
                    ]["capacity_group"],
                    "model_panel": task["model_panel"],
                    "optimization_rollout_budget": task["ablation_budget"][
                        "max_metric_calls"
                    ],
                    "preflight_rollout_budget": (
                        task["ablation_budget"][
                            "preflight_task_rollouts_seed0_per_cell"
                        ]
                        if seed == 0
                        else 0
                    ),
                    "task_protocol": task,
                    "ablation_controls": dict(control),
                }
            )
    if len(rows) != 32 or len({row["cell_id"] for row in rows}) != 32:
        raise AssertionError("expanded paper matrix must contain 32 unique cells")
    return rows


def _slug(cell_id: str, phase: str) -> str:
    return f"{cell_id.replace('.', '_')}_{phase}"


def _paths(
    ledger: Mapping[str, Any], cell: Mapping[str, Any], phase: str
) -> dict[str, str]:
    root = Path(ledger["runtime_roots"][cell["scope"]])
    slug = _slug(cell["cell_id"], phase)
    phase_root = root / phase
    return {
        "slug": slug,
        "run_dir": _portable(phase_root / "runs" / slug),
        "cache_dir": _portable(phase_root / "cache" / slug),
        "stdout_log": _portable(phase_root / "logs" / f"{slug}.stdout.log"),
        "stderr_log": _portable(phase_root / "logs" / f"{slug}.stderr.log"),
        "pid_file": _portable(phase_root / "logs" / f"{slug}.pid"),
        "release_file": _portable(root / "releases" / f"{cell['cell_id']}.json"),
    }


def _preflight_source_cell_id(cell: Mapping[str, Any]) -> str | None:
    if not cell["optimization"]:
        return None
    if cell["scope"] == "ablation" and cell["seed"] != 0:
        return cell["cell_id"].rsplit("seed", 1)[0] + "seed0"
    return str(cell["cell_id"])


def _load_bindings(
    path: Path | None, *, matrix_id: str
) -> tuple[dict[str, Any], str | None]:
    if path is None:
        return {}, None
    payload = path.read_bytes()
    raw = json.loads(payload.decode("utf-8"))
    _reject_secret_values(raw)
    if raw.get("schema_version") != 1 or raw.get("matrix_id") != matrix_id:
        raise ValueError("binding file identity differs from the ledger")
    bindings = raw.get("bindings")
    if not isinstance(bindings, Mapping):
        raise TypeError("bindings must be a JSON object")
    return dict(bindings), _sha256(payload)


def _resolve_path(text: str, *, base: Path) -> Path:
    path = Path(text)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _validate_binding(
    raw: Any,
    *,
    cell: Mapping[str, Any],
    phase: str,
    rollout_budget: int,
    expected_paths: Mapping[str, str],
    matrix_id: str,
    model_api_key_env: str,
    binding_base: Path,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError("one phase binding must be an object")
    required = {
        "cwd",
        "argv",
        "pythonpath",
        "required_env",
        "config_sha256",
        "runner_sha256",
    }
    if set(raw) != required:
        raise ValueError(
            f"binding keys differ; missing={sorted(required - set(raw))}, "
            f"extra={sorted(set(raw) - required)}"
        )
    argv = raw["argv"]
    if (
        not isinstance(argv, list)
        or len(argv) != 4
        or argv[2] != "--config"
        or any(not isinstance(part, str) or not part for part in argv)
    ):
        raise ValueError("binding argv must be [python, runner.py, --config, config]")
    forbidden = {"powershell", "powershell.exe", "pwsh", "pwsh.exe", "cmd", "cmd.exe"}
    if Path(argv[0]).name.lower() in forbidden:
        raise ValueError("PowerShell and cmd runtimes are forbidden")
    runner = _resolve_path(argv[1], base=binding_base)
    config = _resolve_path(argv[3], base=binding_base)
    if runner.suffix.lower() != ".py" or not runner.is_file():
        raise FileNotFoundError(f"bound Python runner is unavailable: {runner}")
    if config.suffix.lower() != ".json" or not config.is_file():
        raise FileNotFoundError(f"bound config is unavailable: {config}")
    _require_sha256(raw["config_sha256"], name="binding.config_sha256")
    if _sha256_file(config) != raw["config_sha256"]:
        raise ValueError(f"bound config hash mismatch: {config}")
    _require_sha256(raw["runner_sha256"], name="binding.runner_sha256")
    if _sha256_file(runner) != raw["runner_sha256"]:
        raise ValueError(f"bound runner hash mismatch: {runner}")
    config_value = json.loads(config.read_text(encoding="utf-8"))
    _reject_secret_values(config_value)
    if not isinstance(config_value, Mapping):
        raise TypeError("bound config must contain one JSON object")
    config_task = config_value.get("task_id")
    if config_task != cell["task_id"]:
        raise ValueError(
            "bound config task differs from queue cell: "
            f"expected={cell['task_id']}, actual={config_task}"
        )
    config_method = config_value.get("method")
    if config_method == "mipro":
        config_method = "miprov2"
    if config_method is None:
        config_method = {
            "mini_admission_reflection": "m0",
            "compass_reflection": "compass",
        }.get(config_value.get("condition"))
    if config_method != cell["method_id"]:
        raise ValueError(
            "bound config method differs from queue cell: "
            f"expected={cell['method_id']}, actual={config_method}"
        )
    config_phase = config_value.get("phase")
    if config_phase != phase:
        raise ValueError(
            "bound config phase differs from queue row: "
            f"expected={phase}, actual={config_phase}"
        )
    config_matrix = config_value.get("matrix_id")
    if config_matrix != matrix_id:
        raise ValueError("bound config matrix_id differs from the ledger")
    optimizer = config_value.get("optimizer")
    config_budget = config_value.get("logical_rollout_budget")
    if config_budget is None and isinstance(optimizer, Mapping):
        config_budget = optimizer.get("max_metric_calls")
    if config_budget != rollout_budget:
        raise ValueError(
            "bound config rollout budget differs from queue row: "
            f"expected={rollout_budget}, actual={config_budget}"
        )
    for name in ("run_dir", "cache_dir"):
        configured = config_value.get(name)
        if not isinstance(configured, str) or not configured:
            raise TypeError(f"bound config {name} must be non-empty text")
        if Path(configured).resolve() != Path(expected_paths[name]).resolve():
            raise ValueError(
                f"bound config {name} differs from queue row: "
                f"expected={expected_paths[name]}, actual={configured}"
            )
    config_api_key_env = config_value.get("api_key_env")
    model = config_value.get("model")
    if config_api_key_env is None and isinstance(model, Mapping):
        config_api_key_env = model.get("api_key_env")
    if config_api_key_env != model_api_key_env:
        raise ValueError("bound config credential env differs from model panel")
    required_env = raw["required_env"]
    if (
        not isinstance(required_env, list)
        or model_api_key_env not in required_env
        or any(
            not isinstance(name, str) or not SAFE_ENV.fullmatch(name)
            for name in required_env
        )
    ):
        raise ValueError(
            "binding required_env must include the model credential env name"
        )
    pythonpath = raw["pythonpath"]
    if (
        not isinstance(pythonpath, list)
        or not pythonpath
        or any(not isinstance(item, str) or not item for item in pythonpath)
    ):
        raise ValueError("binding pythonpath must be a non-empty string array")
    cwd = _resolve_path(raw["cwd"], base=binding_base)
    if not cwd.is_dir():
        raise FileNotFoundError(f"binding cwd is unavailable: {cwd}")
    return {
        "cwd": _portable(cwd),
        "argv": [argv[0], _portable(runner), "--config", _portable(config)],
        "pythonpath": list(pythonpath),
        "required_env": list(required_env),
        "config_sha256": raw["config_sha256"],
        "runner_sha256": raw["runner_sha256"],
    }


def build_phase_rows(
    ledger: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    bindings: Mapping[str, Any],
    binding_base: Path,
) -> list[dict[str, Any]]:
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    rows: list[dict[str, Any]] = []
    for cell in cells:
        if cell["disposition"] != "required_new":
            continue
        if phase == "preflight" and cell["preflight_rollout_budget"] == 0:
            continue
        paths = _paths(ledger, cell, phase)
        rollout_budget = (
            cell["preflight_rollout_budget"]
            if phase == "preflight"
            else cell["optimization_rollout_budget"]
        )
        raw_binding = bindings.get(cell["cell_id"], {}).get(phase)
        profile = ledger["model_panels"][cell["model_panel"]]
        binding = (
            _validate_binding(
                raw_binding,
                cell=cell,
                phase=phase,
                rollout_budget=rollout_budget,
                expected_paths=paths,
                matrix_id=ledger["matrix_id"],
                model_api_key_env=profile["api_key_env"],
                binding_base=binding_base,
            )
            if raw_binding is not None
            else None
        )
        blocked_reasons = []
        if binding is None:
            blocked_reasons.append("missing_frozen_runner_config_binding")
        if phase == "formal" and cell["optimization"]:
            blocked_reasons.append("preflight_release_record_required_at_launch")
        rows.append(
            {
                **{
                    key: cell[key]
                    for key in (
                        "cell_id",
                        "scope",
                        "task_id",
                        "method_id",
                        "seed",
                        "resource_class",
                        "capacity_group",
                        "model_panel",
                        "optimization",
                        "ablation_controls",
                    )
                },
                "phase": phase,
                "rollout_budget": rollout_budget,
                "paths": paths,
                "binding": binding,
                "launch_binding_ready": binding is not None,
                "blocked_reasons": blocked_reasons,
                "formal_release_required": phase == "formal" and cell["optimization"],
                "preflight_source_cell_id": _preflight_source_cell_id(cell),
                "task_identity": {
                    "owner": cell["task_protocol"]["owner"],
                    "split_identity": cell["task_protocol"]["split_identity"],
                    "model": profile,
                    "method_id": cell["method_id"],
                    "phase": phase,
                    "rollout_budget": rollout_budget,
                    "config_sha256": (
                        binding["config_sha256"] if binding is not None else None
                    ),
                    "runner_sha256": (
                        binding["runner_sha256"] if binding is not None else None
                    ),
                },
            }
        )
    return rows


def _source_snapshot(ledger_path: Path) -> dict[str, Any]:
    files = (
        ledger_path,
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts/run-paper-formal-matrix.sh",
        PROJECT_ROOT / "experiments/paper/published_gepa_seed0.json",
        PROJECT_ROOT / "experiments/paper/chartqa_balanced_v2_data_lock.json",
        PROJECT_ROOT / "bridge/paper_benchmark_registry.py",
        PROJECT_ROOT / "bridge/mechanism_benchmark_registry.py",
        PROJECT_ROOT / "bridge/paper_source_snapshot.py",
        PROJECT_ROOT / "bridge/tau2_airline_protocol.py",
        PROJECT_ROOT / "bridge/tau2_gepa_adapter.py",
        PROJECT_ROOT / "experiments/paper/generate_primary_matrix.py",
        PROJECT_ROOT / "experiments/paper/run_compass_reflection.py",
        PROJECT_ROOT / "experiments/paper/run_primary_method.py",
        PROJECT_ROOT / "experiments/paper/run_tau2_airline.py",
        PROJECT_ROOT / "experiments/mechanism/run_compass_reflection.py",
        PROJECT_ROOT / "patches/dspy-working-tree.patch",
        PROJECT_ROOT / "patches/gepa-working-tree.patch",
        PROJECT_ROOT / "patches/gepa-artifact-working-tree.patch",
        PROJECT_ROOT / "upstreams.lock.json",
        PROJECT_ROOT / "scripts/run_with_dpapi_secret.py",
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"matrix source snapshot is incomplete: {missing}")
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "root_head": head,
        "file_sha256": {
            path.relative_to(PROJECT_ROOT).as_posix(): _sha256_file(path)
            for path in files
        },
        "submodules": build_frozen_upstream_snapshot(PROJECT_ROOT),
    }


def write_matrix(
    output_dir: Path,
    *,
    ledger_path: Path = DEFAULT_LEDGER,
    bindings_path: Path | None = None,
    snapshot: Mapping[str, Any] | None = None,
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    ledger_path = ledger_path.resolve(strict=True)
    ledger = load_ledger(ledger_path)
    cells = expand_cells(ledger)
    bindings, binding_sha256 = _load_bindings(
        bindings_path.resolve(strict=True) if bindings_path else None,
        matrix_id=ledger["matrix_id"],
    )
    known = {cell["cell_id"] for cell in cells}
    unknown_bindings = set(bindings) - known
    if unknown_bindings:
        raise ValueError(f"bindings contain unknown cells: {sorted(unknown_bindings)}")
    binding_base = bindings_path.resolve().parent if bindings_path else PROJECT_ROOT
    preflight = build_phase_rows(
        ledger,
        cells,
        phase="preflight",
        bindings=bindings,
        binding_base=binding_base,
    )
    formal = build_phase_rows(
        ledger,
        cells,
        phase="formal",
        bindings=bindings,
        binding_base=binding_base,
    )
    monitor = [
        {
            "cell_id": row["cell_id"],
            "scope": row["scope"],
            "task_id": row["task_id"],
            "method_id": row["method_id"],
            "phase": row["phase"],
            "capacity_group": row["capacity_group"],
            "paths": row["paths"],
            "expected_identity": row["task_identity"],
            "read_only_policy": (
                "inspect the real command tree, method/phase/budget and frozen "
                "runner/config hashes, manifest/final_result, log freshness, owner "
                "identity and explicit tracebacks; do not mutate a healthy run"
            ),
        }
        for row in (*preflight, *formal)
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "cells.jsonl": _jsonl_bytes(cells),
        "preflight_queue.jsonl": _jsonl_bytes(preflight),
        "formal_queue.jsonl": _jsonl_bytes(formal),
        "monitor_queue.jsonl": _jsonl_bytes(monitor),
    }
    for name, payload in artifacts.items():
        _write_new(output_dir / name, payload)
    frozen_snapshot = (
        dict(snapshot) if snapshot is not None else _source_snapshot(ledger_path)
    )
    manifest = {
        "schema_version": 1,
        "matrix_id": ledger["matrix_id"],
        "artifact_policy": ledger["artifact_policy"],
        "ledger_file": ledger_path.relative_to(PROJECT_ROOT).as_posix(),
        "ledger_sha256": _sha256_file(ledger_path),
        "bindings_sha256": binding_sha256,
        "source_snapshot": frozen_snapshot,
        "cell_counts": {
            "all": len(cells),
            "required_new": sum(
                cell["disposition"] == "required_new" for cell in cells
            ),
            "reusable": sum(cell["disposition"] == "reusable" for cell in cells),
            "not_applicable": sum(
                cell["disposition"] == "not_applicable" for cell in cells
            ),
            "preflight": len(preflight),
            "formal": len(formal),
        },
        "capacity_groups": ledger["capacity_groups"],
        "financial_envelopes_usd": ledger["financial_envelopes_usd"],
        "artifacts": {
            name: {"sha256": _sha256(payload), "count": payload.count(b"\n")}
            for name, payload in artifacts.items()
        },
        "credential_policy": ledger["credential_policy"],
        "semantic_policy": ledger["semantic_policy"],
    }
    manifest_path = output_dir / "matrix_manifest.json"
    _write_new(manifest_path, _json_bytes(manifest))
    return manifest_path


def verify_matrix(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _reject_secret_values(manifest)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported generated matrix schema")
    source_snapshot = manifest.get("source_snapshot")
    if not isinstance(source_snapshot, Mapping):
        raise TypeError("generated matrix has no source snapshot")
    if "root_head" not in source_snapshot or "submodules" not in source_snapshot:
        raise ValueError("generated matrix source snapshot is incomplete")
    verify_project_source_snapshot(PROJECT_ROOT, source_snapshot)
    for name, meta in manifest.get("artifacts", {}).items():
        path = manifest_path.parent / name
        payload = path.read_bytes()
        if _sha256(payload) != meta.get("sha256"):
            raise RuntimeError(f"generated matrix artifact hash mismatch: {name}")
        if payload.count(b"\n") != meta.get("count"):
            raise RuntimeError(f"generated matrix artifact count mismatch: {name}")
    return manifest


def _load_phase_queue(
    manifest_path: Path, phase: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = verify_matrix(manifest_path)
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    name = f"{phase}_queue.jsonl"
    rows = [
        json.loads(line)
        for line in (manifest_path.parent / name)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    return manifest, rows


def _verify_formal_release(row: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if not row["formal_release_required"]:
        return
    path = Path(row["paths"]["release_file"])
    if not path.is_file():
        raise RuntimeError(
            f"formal cell has no approved preflight release: {row['cell_id']} ({path})"
        )
    release = json.loads(path.read_text(encoding="utf-8"))
    _reject_secret_values(release)
    expected = {
        "schema_version": 1,
        "status": "approved",
        "matrix_id": manifest["matrix_id"],
        "cell_id": row["cell_id"],
        "ledger_sha256": manifest["ledger_sha256"],
        "formal_config_sha256": row["binding"]["config_sha256"],
        "preflight_source_cell_id": row["preflight_source_cell_id"],
    }
    mismatches = {
        key: {"expected": value, "actual": release.get(key)}
        for key, value in expected.items()
        if release.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"preflight release identity mismatch: {mismatches}")


def emit_launch_records(
    manifest_path: Path,
    *,
    phase: str,
    capacity_group: str,
    cell_ids: Sequence[str] = (),
) -> int:
    manifest, rows = _load_phase_queue(manifest_path, phase)
    if capacity_group not in manifest["capacity_groups"]:
        raise ValueError(f"unknown capacity group: {capacity_group}")
    selected = [
        row
        for row in rows
        if row["capacity_group"] == capacity_group
        and (not cell_ids or row["cell_id"] in cell_ids)
    ]
    if not selected:
        raise RuntimeError("no queue rows match the requested launch selection")
    if cell_ids and set(cell_ids) != {row["cell_id"] for row in selected}:
        raise RuntimeError("one or more requested cell IDs are absent from the queue")
    blocked = [row["cell_id"] for row in selected if row["binding"] is None]
    if blocked:
        raise RuntimeError(
            f"selected cells have no frozen runner/config binding: {blocked}"
        )
    for row in selected:
        if phase == "formal":
            _verify_formal_release(row, manifest)
        binding = row["binding"]
        runner_path = Path(binding["argv"][1])
        config_path = Path(binding["argv"][3])
        if _sha256_file(runner_path) != binding["runner_sha256"]:
            raise RuntimeError(f"bound runner changed after freeze: {row['cell_id']}")
        if _sha256_file(config_path) != binding["config_sha256"]:
            raise RuntimeError(f"bound config changed after freeze: {row['cell_id']}")
        fields = (
            row["paths"]["slug"],
            binding["cwd"],
            row["paths"]["stdout_log"],
            row["paths"]["stderr_log"],
            row["paths"]["pid_file"],
            row["paths"]["run_dir"],
            row["paths"]["cache_dir"],
            *binding["argv"],
            ";".join(binding["pythonpath"]),
            ",".join(binding["required_env"]),
        )
        if any("\0" in field for field in fields):
            raise ValueError(f"NUL in launch binding: {row['cell_id']}")
        sys.stdout.buffer.write("\0".join(fields).encode("utf-8") + b"\0")
    return 0


def audit_owner_splits() -> int:
    """Print owner-produced split fingerprints without making model calls."""

    from bridge.mechanism_benchmark_registry import resolve_mechanism_benchmark
    from bridge.paper_benchmark_registry import (
        instantiate_official_splits,
        load_official_benchmark_specs,
    )

    specs = load_official_benchmark_specs()
    result: dict[str, Any] = {}
    for task_id in ("ifbench", "aime_2025"):
        splits = instantiate_official_splits(
            specs[task_id],
            optimizer_seed=0,
            dataset_mode="lite",
        )
        result[task_id] = {
            "sizes": {
                "train": len(splits.train),
                "validation": len(splits.validation),
                "test": len(splits.test),
            },
            "fingerprints": dict(splits.fingerprints),
        }
    clutrr = resolve_mechanism_benchmark(
        "clutrr_irrelevant",
        lm=None,
        dataset_mode="lite",
    ).splits
    result["clutrr_irrelevant"] = {
        "sizes": {
            "train": len(clutrr.train),
            "validation": len(clutrr.validation),
            "test": len(clutrr.test),
        },
        "fingerprints": dict(clutrr.fingerprints),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    generate.add_argument("--bindings", type=Path)
    generate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True, type=Path)
    emit = subparsers.add_parser("emit")
    emit.add_argument("--manifest", required=True, type=Path)
    emit.add_argument("--phase", required=True, choices=PHASES)
    emit.add_argument("--capacity-group", required=True)
    emit.add_argument("--cell-id", action="append", default=[])
    capacity = subparsers.add_parser("capacity-limit")
    capacity.add_argument("--manifest", required=True, type=Path)
    capacity.add_argument("--capacity-group", required=True)
    subparsers.add_parser("audit-owner-splits")
    args = parser.parse_args()
    if args.command == "generate":
        path = write_matrix(
            args.output_dir,
            ledger_path=args.ledger,
            bindings_path=args.bindings,
        )
        print(json.dumps({"manifest": _portable(path), "status": "frozen"}))
        return 0
    if args.command == "verify":
        manifest = verify_matrix(args.manifest)
        print(json.dumps({"matrix_id": manifest["matrix_id"], "status": "verified"}))
        return 0
    if args.command == "emit":
        return emit_launch_records(
            args.manifest,
            phase=args.phase,
            capacity_group=args.capacity_group,
            cell_ids=args.cell_id,
        )
    if args.command == "capacity-limit":
        manifest = verify_matrix(args.manifest)
        try:
            limit = manifest["capacity_groups"][args.capacity_group][
                "max_parallel_runs"
            ]
        except KeyError as error:
            raise ValueError(
                f"unknown capacity group: {args.capacity_group}"
            ) from error
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("frozen capacity limit must be a positive integer")
        print(limit)
        return 0
    return audit_owner_splits()


if __name__ == "__main__":
    raise SystemExit(main())
