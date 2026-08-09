from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments.paper import evaluate_tau2_airline_owner_top1 as subject


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
        candidate_sha256=subject.candidate_sha256(
            {"agent_instruction": "owner best"}
        ),
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
    assert json.loads((output_dir / "selected_candidate.json").read_text())[
        "identity_sha256"
    ] == identity_sha256
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
