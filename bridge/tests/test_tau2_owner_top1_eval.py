from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments.paper import evaluate_tau2_airline_owner_top1 as subject


def _selection() -> subject.SourceSelection:
    candidate = {"agent_instruction": "owner best"}
    return subject.SourceSelection(
        candidate_idx=4,
        candidate=candidate,
        candidate_sha256=subject.candidate_sha256(candidate),
        frontier_count=14,
        clean_exposure=14,
        source_identity={"source": {"compass_git_head": "a" * 40}},
    )


def _owner_results(
    selection: subject.SourceSelection,
    *,
    complete: bool,
) -> dict[str, object]:
    identities = sorted(subject._expected_final_identities())
    if not complete:
        identities = identities[:1]
    return {
        "info": {
            "agent_info": {
                "implementation": (
                    subject.TAU2_FIXED_CANDIDATE_AGENT_PREFIX
                    + selection.candidate_sha256
                ),
                "llm": subject.TAU2_AGENT_MODEL,
            },
            "num_trials": len(subject.TAU2_TEST_TRIAL_SEEDS),
        },
        "simulations": [
            {
                "task_id": task_id,
                "trial": trial,
                "seed": seed,
                "termination_reason": "user_stop",
                "reward_info": {"reward": 1.0},
            }
            for task_id, trial, seed in identities
        ],
    }


def test_source_selection_delegates_to_owner_policy(monkeypatch, tmp_path) -> None:
    source_run = tmp_path / "source"
    optimizer_dir = source_run / "optimizer"
    optimizer_dir.mkdir(parents=True)
    source_identity = {"source": {"compass_git_head": "a" * 40}}
    identity_sha256 = subject._canonical_sha256(source_identity)
    (source_run / "run_identity.json").write_text(
        json.dumps(source_identity), encoding="utf-8"
    )
    (source_run / "manifest.json").write_text(
        json.dumps({"status": "completed", "identity_sha256": identity_sha256}),
        encoding="utf-8",
    )
    (source_run / "final_result.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "identity_sha256": identity_sha256,
                "method": "compass",
                "phase": "formal",
                "selected_candidate_idx": 0,
            }
        ),
        encoding="utf-8",
    )
    state_path = optimizer_dir / "gepa_state.bin"
    state_path.write_bytes(b"state")
    candidates = [
        {"agent_instruction": "seed"},
        {"agent_instruction": "owner best"},
    ]
    candidates_path = optimizer_dir / "candidates.json"
    candidates_path.write_text(json.dumps(candidates), encoding="utf-8")
    state = SimpleNamespace(program_candidates=candidates)
    owner_get_best = Mock(return_value=1)
    monkeypatch.setattr(subject.GEPAState, "load", lambda path: state)
    monkeypatch.setattr(
        subject,
        "SparseMinibatchEvaluationPolicy",
        lambda: SimpleNamespace(get_best_program=owner_get_best),
    )
    monkeypatch.setattr(subject, "frontier_count", lambda state, idx: 5)
    monkeypatch.setattr(subject, "evaluation_count", lambda state, idx: 6)
    config = {
        "source_run": str(source_run),
        "expected_source_identity_sha256": identity_sha256,
        "expected_source_recorded_candidate_idx": 0,
        "expected_source_state_sha256": subject._sha256_file(state_path),
        "expected_source_candidates_sha256": subject._sha256_file(candidates_path),
        "expected_candidate_idx": 1,
        "expected_candidate_sha256": subject.candidate_sha256(candidates[1]),
    }

    selection = subject._verified_source_selection(config)

    assert selection.candidate_idx == 1
    assert selection.frontier_count == 5
    assert selection.clean_exposure == 6
    owner_get_best.assert_called_once_with(state)


def test_output_is_create_only_and_strictly_resumable(tmp_path) -> None:
    output_dir = tmp_path / "output"
    identity = {"schema_version": 1, "role": "test"}
    selection = subject.SourceSelection(
        candidate_idx=4,
        candidate={"agent_instruction": "owner best"},
        candidate_sha256=subject.candidate_sha256({"agent_instruction": "owner best"}),
        frontier_count=14,
        clean_exposure=14,
        source_identity={"source": {"compass_git_head": "a" * 40}},
    )

    manifest, identity_sha256 = subject._initialize_output(
        output_dir,
        identity=identity,
        selection=selection,
        resume=False,
    )

    assert manifest["status"] == "running"
    assert json.loads((output_dir / "run_identity.json").read_text()) == identity
    assert (
        json.loads((output_dir / "selected_candidate.json").read_text())[
            "identity_sha256"
        ]
        == identity_sha256
    )
    with pytest.raises(FileExistsError):
        subject._initialize_output(
            output_dir,
            identity=identity,
            selection=selection,
            resume=False,
        )

    resumed, resumed_identity = subject._initialize_output(
        output_dir,
        identity=identity,
        selection=selection,
        resume=True,
    )

    assert resumed_identity == identity_sha256
    assert resumed["resume_count"] == 1


def test_resume_requires_an_existing_output(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="--resume requires"):
        subject._initialize_output(
            tmp_path / "absent",
            identity={"schema_version": 1},
            selection=_selection(),
            resume=True,
        )


def test_resume_rechecks_selection_and_checkpoint_identity(tmp_path) -> None:
    output_dir = tmp_path / "output"
    identity = {"schema_version": 1, "role": "test"}
    selection = _selection()
    subject._initialize_output(
        output_dir,
        identity=identity,
        selection=selection,
        resume=False,
    )
    identity_sha256 = subject._canonical_sha256(identity)
    selection_path = output_dir / "selected_candidate.json"
    record = json.loads(selection_path.read_text(encoding="utf-8"))
    record["selected_candidate_idx"] = 0
    selection_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="selection changed"):
        subject._initialize_output(
            output_dir,
            identity=identity,
            selection=selection,
            resume=True,
        )

    selection_path.write_text(
        json.dumps(
            subject._selection_record(
                identity_sha256=identity_sha256,
                selection=selection,
            )
        ),
        encoding="utf-8",
    )
    checkpoint_path = output_dir / "checkpoint_identity.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["retry_owner"] = "other"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint identity changed"):
        subject._initialize_output(
            output_dir,
            identity=identity,
            selection=selection,
            resume=True,
        )


def test_load_config_rejects_nested_source_and_output(tmp_path) -> None:
    source = tmp_path / "source"
    config = {
        "api_base": subject.PRIMARY_API_BASE,
        "api_key_env": "API_KEY",
        "expected_candidate_idx": 4,
        "expected_candidate_sha256": "a" * 64,
        "expected_source_candidates_sha256": "b" * 64,
        "expected_source_identity_sha256": "c" * 64,
        "expected_source_recorded_candidate_idx": 0,
        "expected_source_state_sha256": "d" * 64,
        "output_dir": str(source / "correction"),
        "quiet": True,
        "schema_version": 1,
        "source_run": str(source),
        "tau2_root": str(tmp_path / "tau2"),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="non-nested"):
        subject.load_config(path)


def test_owner_resume_requires_candidate_bound_checkpoint(tmp_path) -> None:
    selection = _selection()
    results_path = tmp_path / "results.json"
    assert not subject._verified_owner_resume(
        results_path,
        resume=True,
        selection=selection,
    )

    payload = _owner_results(selection, complete=False)
    payload["info"]["agent_info"]["implementation"] = "other_candidate"
    results_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="candidate identity changed"):
        subject._verified_owner_resume(
            results_path,
            resume=True,
            selection=selection,
        )

    results_path.write_text(
        json.dumps(_owner_results(selection, complete=False)), encoding="utf-8"
    )
    assert subject._verified_owner_resume(
        results_path,
        resume=True,
        selection=selection,
    )


def test_completed_output_rechecks_candidate_results_and_score(
    monkeypatch, tmp_path
) -> None:
    output_dir = tmp_path / "output"
    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True)
    selection = _selection()
    identity = {
        "schema_version": 1,
        "selection": {
            "selected_candidate_idx": selection.candidate_idx,
            "candidate_sha256": selection.candidate_sha256,
        },
    }
    identity_sha256 = subject._canonical_sha256(identity)
    results_path = evaluation_dir / "results.json"
    results_path.write_text(
        json.dumps(_owner_results(selection, complete=True)), encoding="utf-8"
    )
    final_result = {
        "status": "completed",
        "identity_sha256": identity_sha256,
        "selected_candidate_idx": selection.candidate_idx,
        "selected_candidate_sha256": selection.candidate_sha256,
        "selection": identity["selection"],
        "evaluation": {
            "status": "completed",
            "results_path": str(results_path.resolve()),
            "results_sha256": subject._sha256_file(results_path),
            "num_simulations": len(subject._expected_final_identities()),
            "metrics": {"avg_reward": 0.525},
            "test_score_percent": 52.5,
            "owner_resource_usage": {"logical_episodes": 80},
        },
    }
    final_path = output_dir / "final_result.json"
    final_path.write_text(json.dumps(final_result), encoding="utf-8")
    manifest = {"status": "completed", "final_result": "final_result.json"}
    monkeypatch.setattr(
        subject,
        "_official_result_summaries",
        lambda results: (
            {"avg_reward": 0.525},
            {"logical_episodes": 80},
        ),
    )

    actual = subject._validate_completed_output(
        output_dir,
        manifest=manifest,
        identity=identity,
        identity_sha256=identity_sha256,
        selection=selection,
    )
    assert actual["evaluation"]["test_score_percent"] == 52.5

    final_result["selected_candidate_idx"] = 0
    final_path.write_text(json.dumps(final_result), encoding="utf-8")
    with pytest.raises(RuntimeError, match="candidate index changed"):
        subject._validate_completed_output(
            output_dir,
            manifest=manifest,
            identity=identity,
            identity_sha256=identity_sha256,
            selection=selection,
        )


def test_output_lock_rejects_a_second_process_scope(tmp_path) -> None:
    output_dir = tmp_path / "output"
    with (
        subject._exclusive_output_lock(output_dir),
        pytest.raises(RuntimeError, match="already active"),
        subject._exclusive_output_lock(output_dir),
    ):
        raise AssertionError("second lock unexpectedly acquired")


def test_semantic_identity_covers_direct_runtime_dependencies() -> None:
    assert set(subject.SEMANTIC_SOURCE_FILES) == {
        "bridge/b19_reversible_parent_selection.py",
        "bridge/b20_compass_reflection.py",
        "bridge/paper_source_snapshot.py",
        "bridge/tau2_airline_protocol.py",
        "bridge/tau2_gepa_adapter.py",
        "experiments/paper/run_tau2_airline.py",
    }
