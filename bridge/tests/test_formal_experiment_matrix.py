from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from experiments.paper.generate_formal_experiment_matrix import (
    DEFAULT_LEDGER,
    _validate_binding,
    build_phase_rows,
    emit_launch_records,
    expand_cells,
    load_ledger,
    verify_matrix,
    write_matrix,
)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_ledger_expands_exact_required_reusable_and_na_cells() -> None:
    ledger = load_ledger(DEFAULT_LEDGER)
    cells = expand_cells(ledger)

    assert len(cells) == 32
    assert Counter(cell["scope"] for cell in cells) == {"primary": 20, "ablation": 12}
    assert Counter(cell["disposition"] for cell in cells) == {
        "required_new": 29,
        "reusable": 2,
        "not_applicable": 1,
    }
    assert {cell["cell_id"] for cell in cells if cell["disposition"] == "reusable"} == {
        "primary.ifbench.seed.seed0",
        "primary.aime_2025.seed.seed0",
    }
    tau_mipro = next(
        cell
        for cell in cells
        if cell["cell_id"] == "primary.tau2_airline.miprov2.seed0"
    )
    assert tau_mipro["disposition"] == "not_applicable"
    assert "GEPAAdapter" in tau_mipro["reason"]


def test_exact_budgets_splits_models_and_ablation_controls_are_frozen() -> None:
    ledger = load_ledger(DEFAULT_LEDGER)
    cells = expand_cells(ledger)
    tasks = ledger["task_protocols"]

    assert {
        task: tasks[task]["optimization_rollout_budget"]
        for task in ("ifbench", "aime_2025", "chartqa", "tau2_airline")
    } == {
        "ifbench": 3593,
        "aime_2025": 1839,
        "chartqa": 1152,
        "tau2_airline": 600,
    }
    assert tasks["chartqa"]["task_decoding"] == {
        "temperature": 0.0,
        "max_tokens": 16,
        "owner_provenance_do_sample": False,
        "structured_image_url": True,
    }
    tau = tasks["tau2_airline"]["split_identity"]
    assert len(tau["proposal_ids"]) == 24
    assert tau["validation_ids"] == ["5", "12", "21", "34", "41", "49"]
    assert not set(tau["proposal_ids"]) & set(tau["validation_ids"])
    assert set(tau["proposal_ids"]) | set(tau["validation_ids"]) == set(
        tau["train_ids"]
    )
    ablation = tasks["clutrr_irrelevant"]["ablation_budget"]
    assert ablation == {
        "complete_train_passes": 5,
        "proposal_minibatch_size": 3,
        "serial_proposal_iterations": 250,
        "proposal_instance_exposures": 750,
        "max_candidate_proposals": 250,
        "max_metric_calls": 2250,
        "epoch_parallel_enabled": False,
        "preflight_task_rollouts_seed0_per_cell": 60,
    }
    assert all(
        cell["optimization_rollout_budget"] == 2250
        for cell in cells
        if cell["scope"] == "ablation"
    )
    assert ledger["model_panels"]["gpt_4_1_mini_gpt_ge"]["task_model"] == (
        "openai/gpt-4.1-mini-2025-04-14"
    )


def test_generated_create_only_queues_are_monitorable_and_fail_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "matrix"
    manifest_path = write_matrix(output)
    manifest = verify_matrix(manifest_path)
    cells = _rows(output / "cells.jsonl")
    preflight = _rows(output / "preflight_queue.jsonl")
    formal = _rows(output / "formal_queue.jsonl")
    monitor = _rows(output / "monitor_queue.jsonl")

    assert manifest["cell_counts"] == {
        "all": 32,
        "required_new": 29,
        "reusable": 2,
        "not_applicable": 1,
        "preflight": 19,
        "formal": 29,
    }
    assert len(cells) == 32
    assert len(preflight) == 19
    assert len(formal) == 29
    assert len(monitor) == 48
    assert all(row["binding"] is None for row in (*preflight, *formal))
    assert all(row["launch_binding_ready"] is False for row in (*preflight, *formal))
    assert all(
        row["paths"][name].startswith("F:/")
        for row in (*preflight, *formal)
        for name in (
            "run_dir",
            "cache_dir",
            "stdout_log",
            "stderr_log",
            "pid_file",
            "release_file",
        )
    )
    assert {(row["cell_id"], row["phase"]) for row in monitor} == {
        (row["cell_id"], row["phase"]) for row in (*preflight, *formal)
    }
    seed2_highres = next(
        row
        for row in formal
        if row["cell_id"] == "ablation.clutrr_irrelevant.highres_strict.seed2"
    )
    assert seed2_highres["preflight_source_cell_id"] == (
        "ablation.clutrr_irrelevant.highres_strict.seed0"
    )
    payload = b"".join(path.read_bytes() for path in sorted(output.iterdir()))
    assert b"sk-" not in payload
    with pytest.raises(RuntimeError, match="no frozen runner/config binding"):
        emit_launch_records(
            manifest_path,
            phase="preflight",
            capacity_group="gpt_ge_account",
            cell_ids=["primary.ifbench.miprov2.seed0"],
        )
    with pytest.raises(FileExistsError):
        write_matrix(
            output,
            snapshot={"root_head": "test", "file_sha256": {}},
        )


def test_phase_rows_keep_preflight_and_formal_paths_disjoint() -> None:
    ledger = load_ledger(DEFAULT_LEDGER)
    cells = expand_cells(ledger)
    preflight = build_phase_rows(
        ledger,
        cells,
        phase="preflight",
        bindings={},
        binding_base=DEFAULT_LEDGER.parent,
    )
    formal = build_phase_rows(
        ledger,
        cells,
        phase="formal",
        bindings={},
        binding_base=DEFAULT_LEDGER.parent,
    )
    preflight_paths = {
        row["paths"][name]
        for row in preflight
        for name in ("run_dir", "cache_dir", "stdout_log", "stderr_log")
    }
    formal_paths = {
        row["paths"][name]
        for row in formal
        for name in ("run_dir", "cache_dir", "stdout_log", "stderr_log")
    }
    assert preflight_paths.isdisjoint(formal_paths)


def test_manifest_hash_verification_detects_queue_tampering(tmp_path: Path) -> None:
    output = tmp_path / "matrix"
    manifest = write_matrix(output)
    queue = output / "formal_queue.jsonl"
    queue.write_bytes(queue.read_bytes() + b"{}\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_matrix(manifest)


def test_published_references_match_the_frozen_owner_transcription() -> None:
    ledger = load_ledger(DEFAULT_LEDGER)
    published = json.loads(
        (DEFAULT_LEDGER.parent / "published_gepa_seed0.json").read_text(
            encoding="utf-8"
        )
    )["rows"]["gpt_4_1_mini"]
    for cell in ledger["primary_cells"]:
        if "published_reference_percent" not in cell:
            continue
        assert (
            cell["published_reference_percent"]
            == published[cell["method"]][cell["task"]]
        )


def test_launcher_is_bash_only_and_has_no_retry_watchdog_or_secret_loader() -> None:
    launcher = (
        Path(__file__).resolve().parents[2] / "scripts/run-paper-formal-matrix.sh"
    ).read_text(encoding="utf-8")
    assert launcher.startswith("#!/usr/bin/env bash\n")
    lowered = launcher.lower()
    assert "powershell" not in lowered
    assert "cmd.exe" not in lowered
    assert "retry" not in lowered
    assert "watchdog" not in lowered
    assert "api_key" not in lowered
    assert "sk-" not in launcher


def test_binding_freezes_runner_and_config_content(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    config = tmp_path / "config.json"
    runner.write_text("print('owner runner')\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"
    config.write_text(
        json.dumps(
            {
                "api_key_env": "COMPASS_GPT_GE_API_KEY",
                "cache_dir": str(cache_dir),
                "logical_rollout_budget": 96,
                "matrix_id": "paper_test_v1",
                "method": "gepa",
                "phase": "preflight",
                "run_dir": str(run_dir),
                "task_id": "ifbench",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    binding = {
        "cwd": str(tmp_path),
        "argv": ["python", str(runner), "--config", str(config)],
        "pythonpath": [str(tmp_path)],
        "required_env": ["COMPASS_GPT_GE_API_KEY"],
        "config_sha256": digest(config),
        "runner_sha256": digest(runner),
    }
    frozen = _validate_binding(
        binding,
        cell={"task_id": "ifbench", "method_id": "gepa"},
        phase="preflight",
        rollout_budget=96,
        expected_paths={"run_dir": str(run_dir), "cache_dir": str(cache_dir)},
        matrix_id="paper_test_v1",
        model_api_key_env="COMPASS_GPT_GE_API_KEY",
        binding_base=tmp_path,
    )
    assert frozen["runner_sha256"] == digest(runner)

    runner.write_text("print('drift')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runner hash mismatch"):
        _validate_binding(
            binding,
            cell={"task_id": "ifbench", "method_id": "gepa"},
            phase="preflight",
            rollout_budget=96,
            expected_paths={"run_dir": str(run_dir), "cache_dir": str(cache_dir)},
            matrix_id="paper_test_v1",
            model_api_key_env="COMPASS_GPT_GE_API_KEY",
            binding_base=tmp_path,
        )


def test_hash_correct_binding_for_another_cell_is_rejected(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    config = tmp_path / "config.json"
    runner.write_text("print('owner runner')\n", encoding="utf-8")
    config.write_text(
        json.dumps(
            {
                "api_key_env": "COMPASS_GPT_GE_API_KEY",
                "cache_dir": str(tmp_path / "wrong-cache"),
                "logical_rollout_budget": 60,
                "matrix_id": "paper_test_v1",
                "method": "mipro",
                "phase": "preflight",
                "run_dir": str(tmp_path / "wrong-run"),
                "task_id": "aime_2025",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    binding = {
        "cwd": str(tmp_path),
        "argv": ["python", str(runner), "--config", str(config)],
        "pythonpath": [str(tmp_path)],
        "required_env": ["COMPASS_GPT_GE_API_KEY"],
        "config_sha256": digest(config),
        "runner_sha256": digest(runner),
    }
    with pytest.raises(ValueError, match="task differs"):
        _validate_binding(
            binding,
            cell={"task_id": "ifbench", "method_id": "gepa"},
            phase="preflight",
            rollout_budget=96,
            expected_paths={
                "run_dir": str(tmp_path / "right-run"),
                "cache_dir": str(tmp_path / "right-cache"),
            },
            matrix_id="paper_test_v1",
            model_api_key_env="COMPASS_GPT_GE_API_KEY",
            binding_base=tmp_path,
        )
