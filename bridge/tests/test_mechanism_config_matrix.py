from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from experiments.mechanism.generate_mechanism_matrix import (
    ADMISSION_MINIBATCH_SIZE,
    ADMISSION_REFERENCE_CALLS_PER_CANDIDATE,
    CANDIDATE_PROPOSALS,
    METHOD_CELLS,
    OPTIMIZATION_METRIC_CALL_CAP,
    PROPOSAL_MINIBATCH_SIZE,
    TASK_PROTOCOLS,
    build_matrix,
    metric_call_accounting,
    write_matrix,
)
from experiments.paper.run_compass_reflection import (
    _require_frozen_protocol,
    _verify_local_source_snapshot,
    load_run_config,
)


def _records() -> list[dict]:
    return build_matrix(
        matrix_id="test_mechanism_v1",
        runtime_root=Path("F:/mechanism-test"),
        snapshot={"root_head": "test", "file_sha256": {}},
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_matrix_has_exact_frozen_44_run_factorial() -> None:
    records = _records()

    assert len(records) == 44
    assert len({record["slug"] for record in records}) == 44
    assert Counter(record["task_id"] for record in records) == {
        "hitab": 4,
        "chartqa": 4,
        "clutrr_supporting": 12,
        "clutrr_irrelevant": 12,
        "clutrr_disconnected": 12,
    }
    grouped: dict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
    for record in records:
        grouped[(record["task_id"], record["seed"])].add(
            (record["selection_mode"], record["acceptance_mode"])
        )
    expected_cells = {(selection, acceptance) for _, selection, acceptance in METHOD_CELLS}
    assert grouped
    assert all(cells == expected_cells for cells in grouped.values())
    assert set(grouped) == {
        (task.task_id, seed) for task in TASK_PROTOCOLS for seed in task.seeds
    }


def test_every_config_preserves_budget_batches_and_only_the_four_cell_controls() -> None:
    records = _records()
    invariant_optimizers = []
    for record in records:
        config = record["config"]
        optimizer = config["optimizer"]
        assert config["condition"] == "compass_reflection"
        assert config["dataset_mode"] == "lite"
        assert optimizer["max_candidate_proposals"] == CANDIDATE_PROPOSALS == 128
        assert optimizer["max_metric_calls"] == OPTIMIZATION_METRIC_CALL_CAP == 1152
        assert optimizer["proposal_minibatch_size"] == PROPOSAL_MINIBATCH_SIZE == 3
        assert optimizer["admission_minibatch_size"] == ADMISSION_MINIBATCH_SIZE == 3
        assert "reflection_minibatch_size" not in optimizer
        assert optimizer["epoch_parallel_enabled"] is False
        assert optimizer["max_candidate_workers"] == 1
        assert optimizer["max_reflection_workers"] == 1
        assert optimizer["parent_top_n"] == 5
        assert optimizer["rollout_timeout_seconds"] == 1200
        assert optimizer["proposal_timeout_seconds"] == 1200
        assert optimizer["failure_score"] == 0
        assert optimizer["perfect_score"] == 1
        assert optimizer["skip_perfect_score"] is True
        assert optimizer["raise_on_exception"] is True
        assert "dci" not in optimizer
        assert not any(
            "teacher" in key.lower() or "flashtrace" in key.lower()
            for key in optimizer
        )
        invariant_optimizers.append(
            {
                key: value
                for key, value in optimizer.items()
                if key not in {"acceptance_mode", "parent_selection_score_mode"}
            }
        )
    assert all(item == invariant_optimizers[0] for item in invariant_optimizers)


def test_metric_call_cap_and_post_optimization_evaluations_are_separate() -> None:
    expected_totals = {
        "hitab": 3036,
        "chartqa": 3952,
        "clutrr_supporting": 1899,
        "clutrr_irrelevant": 1896,
        "clutrr_disconnected": 1897,
    }
    assert ADMISSION_REFERENCE_CALLS_PER_CANDIDATE == 3
    assert OPTIMIZATION_METRIC_CALL_CAP == 128 * (3 + 3 + 3)
    for task in TASK_PROTOCOLS:
        accounting = metric_call_accounting(task)
        assert accounting["optimization_metric_call_cap"] == 1152
        assert accounting["admission_reference_calls_per_candidate"] == 3
        assert accounting["sparse_seed_evaluation_calls"] == 0
        assert accounting["final_validation_calls_outside_optimizer_cap"] == 300
        assert accounting["final_test_calls_outside_optimizer_cap"] == task.final_test_size
        assert accounting["planned_total_calls_ceiling"] == expected_totals[task.task_id]


def test_three_compute_routes_are_separate_and_credentials_are_env_only() -> None:
    records = _records()
    for record in records:
        model = record["config"]["model"]
        assert model["timeout"] == 1200
        assert model["num_retries"] == 0
        assert "api_key" not in model
        if record["resource_class"] == "visual":
            assert model["api_key_env"] == "COMPASS_LITELLM_PROXY_KEY"
            assert model["api_base"] == "http://127.0.0.1:40039/v1"
            assert model["model"] == "openai/compass-qwen3-vl-8b-thinking"
            assert model["enable_thinking"] is False
        elif record["task_id"].startswith("clutrr_"):
            assert model["api_key_env"] == "COMPASS_VLLM_API_KEY"
            assert model["api_base"] == "http://127.0.0.1:18000/v1"
            assert model["model"] == "openai/Qwen3-8B"
            assert model["enable_thinking"] is True
        else:
            assert model["api_key_env"] == "COMPASS_LITELLM_PROXY_KEY"
            assert model["api_base"] == "http://127.0.0.1:40038/v1"
            assert model["model"] == "openai/compass-qwen3-8b"
            assert model["enable_thinking"] is True


def test_create_only_output_has_hashed_manifest_and_launch_queues(tmp_path: Path) -> None:
    output = tmp_path / "matrix"
    manifest_path = write_matrix(
        output,
        matrix_id="test_mechanism_v1",
        runtime_root=Path("F:/mechanism-test"),
        snapshot={"root_head": "test", "file_sha256": {}},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_queue = _read_jsonl(output / "launch_queue_all.jsonl")
    text_queue = _read_jsonl(output / "launch_queue_text.jsonl")
    visual_queue = _read_jsonl(output / "launch_queue_visual.jsonl")

    assert manifest["run_count"] == len(all_queue) == 44
    assert manifest["resource_counts"] == {"text": 40, "visual": 4}
    assert len(text_queue) == 40
    assert len(visual_queue) == 4
    assert all(row["resource_class"] == "text" for row in text_queue)
    assert all(row["resource_class"] == "visual" for row in visual_queue)
    assert len({row["run_dir"] for row in all_queue}) == 44
    assert len({row["cache_dir"] for row in all_queue}) == 44
    assert len({row["stdout_log"] for row in all_queue}) == 44
    assert all(
        (
            "COMPASS_VLLM_API_KEY"
            if row["task_id"].startswith("clutrr_")
            else "COMPASS_LITELLM_PROXY_KEY"
        )
        in row["required_env"]
        for row in all_queue
    )
    assert all(Path(row["argv"][0]).name == "python.exe" for row in all_queue)
    assert all(row["pythonpath"] for row in all_queue)
    assert all(
        row["cwd"] == row["pythonpath"][0]
        and row["argv"][1].endswith(
            "/experiments/mechanism/run_compass_reflection.py"
        )
        and row["argv"][2] == "--config"
        for row in all_queue
    )
    assert {
        row["argv"][0]
        for row in all_queue
        if row["task_id"].startswith("clutrr_")
    } == {
        "F:/compass-ifbench-local/.venv-no-torch-py312/Scripts/python.exe"
    }
    assert all(
        "\\" not in value
        for row in all_queue
        for value in (
            row["cwd"],
            row["argv"][0],
            row["argv"][1],
            row["argv"][3],
            row["stdout_log"],
            row["stderr_log"],
            row["pid_file"],
            row["run_dir"],
            row["cache_dir"],
        )
    )

    for queue_name, queue_meta in manifest["launch_queues"].items():
        queue_path = output / queue_meta["file"]
        assert hashlib.sha256(queue_path.read_bytes()).hexdigest() == queue_meta["sha256"]
        assert queue_meta["count"] == {"all": 44, "text": 40, "visual": 4}[queue_name]
    for run in manifest["runs"]:
        config_path = output / run["config_file"]
        assert hashlib.sha256(config_path.read_bytes()).hexdigest() == run["config_sha256"]

    serialized = output.read_bytes() if output.is_file() else b"".join(
        path.read_bytes() for path in sorted(output.rglob("*")) if path.is_file()
    )
    assert b"sk-" not in serialized
    with pytest.raises(FileExistsError):
        write_matrix(
            output,
            matrix_id="test_mechanism_v1",
            runtime_root=Path("F:/mechanism-test"),
            snapshot={"root_head": "test", "file_sha256": {}},
        )


def test_generated_configs_are_accepted_by_the_shared_runner_schema(tmp_path: Path) -> None:
    output = tmp_path / "matrix"
    write_matrix(
        output,
        matrix_id="test_mechanism_v1",
        runtime_root=Path("F:/mechanism-test"),
        snapshot={"root_head": "test", "file_sha256": {}},
    )
    for path in sorted((output / "configs").glob("*.json")):
        config = load_run_config(path)
        _require_frozen_protocol(config, budget=OPTIMIZATION_METRIC_CALL_CAP)


def test_generator_rejects_unsafe_matrix_and_secret_like_env_names() -> None:
    with pytest.raises(ValueError, match="matrix_id"):
        build_matrix(
            matrix_id="../overwrite",
            runtime_root=Path("F:/mechanism-test"),
            snapshot={},
        )
    with pytest.raises(ValueError, match="environment"):
        build_matrix(
            matrix_id="safe",
            runtime_root=Path("F:/mechanism-test"),
            snapshot={},
            text_api_key_env="sk-secret-value",
        )


def test_mechanism_source_snapshot_rejects_drift_and_path_escape() -> None:
    relative = "bridge/minibatch_config.py"
    source = Path(__file__).resolve().parents[2] / relative
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    _verify_local_source_snapshot({"file_sha256": {relative: digest}})
    with pytest.raises(RuntimeError, match="differs from the frozen matrix"):
        _verify_local_source_snapshot({"file_sha256": {relative: "0" * 64}})
    with pytest.raises(ValueError, match="escapes the project root"):
        _verify_local_source_snapshot(
            {"file_sha256": {"../outside.py": "0" * 64}}
        )
