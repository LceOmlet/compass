from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from bridge.dci_agent_lite import (
    DciAgentLiteConfig,
    DciAgentLiteFinalAdapter,
    DciBoundaryError,
    load_dci_result,
)
from bridge.request_deadline import request_deadline


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "isolated-corpus"
    (corpus / "selection").mkdir(parents=True)
    return corpus


def test_result_preserves_free_text_and_orders_selected_ids_by_allowlist(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    (corpus / "selection" / "d2").touch()
    (corpus / "selection" / "d1").touch()
    final = "  Bind the intermediate entity before the terminal check.\n"

    result = load_dci_result(
        final,
        corpus_dir=corpus,
        artifact_dir=tmp_path / "artifacts",
        id_by_token={"d1": 1, "d2": 2, "d3": 3},
    )

    assert result.definition == final
    assert result.selected_ids == (1, 2)


def test_result_rejects_unknown_or_non_marker_workspace_entries(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    (corpus / "selection" / "unknown").touch()
    with pytest.raises(DciBoundaryError, match="unknown DataId"):
        load_dci_result(
            "a subproblem",
            corpus_dir=corpus,
            artifact_dir=tmp_path / "artifacts",
            id_by_token={"d1": 1},
        )

    (corpus / "selection" / "unknown").unlink()
    (corpus / "selection" / "nested").mkdir()
    with pytest.raises(DciBoundaryError, match="only direct marker files"):
        load_dci_result(
            "a subproblem",
            corpus_dir=corpus,
            artifact_dir=tmp_path / "artifacts",
            id_by_token={"d1": 1},
        )


def test_adapter_invokes_official_cli_once_and_reads_owned_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    output_dir = tmp_path / "artifacts" / "run-0"
    prompt = tmp_path / "system.txt"
    prompt.write_text("free text", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output_dir.mkdir(parents=True)
        (output_dir / "final.txt").write_text(
            "Missing terminal verification.\nReuse the verified binding.",
            encoding="utf-8",
        )
        (corpus / "selection" / "d1").touch()
        return subprocess.CompletedProcess(command, 0, stdout="ignored", stderr="")

    monkeypatch.setattr("bridge.dci_agent_lite.subprocess.run", fake_run)
    adapter = DciAgentLiteFinalAdapter(
        DciAgentLiteConfig(
            runner_command=("python", "-m", "dci.benchmark.pi_rpc_runner"),
            package_dir=tmp_path / "pi-package",
            agent_dir=tmp_path / "pi-agent",
            provider="openai",
            model="model",
            system_prompt_file=prompt,
        )
    )

    result = adapter.run(
        rendered_question="inspect the frozen corpus",
        corpus_cwd=corpus,
        output_dir=output_dir,
        id_by_token={"d1": 1},
    )

    assert result.definition == (
        "Missing terminal verification.\nReuse the verified binding."
    )
    assert result.selected_ids == (1,)
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:3] == ["python", "-m", "dci.benchmark.pi_rpc_runner"]
    assert kwargs["input"] == "inspect the frozen corpus"
    assert kwargs["shell"] is False


def test_adapter_refuses_a_prepopulated_selection_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    (corpus / "selection" / "d1").touch()
    monkeypatch.setattr(
        "bridge.dci_agent_lite.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("runner must not be called"),
    )
    adapter = DciAgentLiteFinalAdapter(
        DciAgentLiteConfig(
            runner_command=("dci-agent-lite",),
            package_dir=tmp_path / "package",
            agent_dir=tmp_path / "agent",
            provider="openai",
            model="model",
            system_prompt_file=tmp_path / "prompt.txt",
        )
    )

    with pytest.raises(DciBoundaryError, match="must start empty"):
        adapter.run(
            rendered_question="inspect",
            corpus_cwd=corpus,
            output_dir=tmp_path / "output",
            id_by_token={"d1": 1},
        )


def test_adapter_passes_the_shared_absolute_deadline_to_the_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    output_dir = tmp_path / "artifacts" / "run-deadline"
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        output_dir.mkdir(parents=True)
        (output_dir / "final.txt").write_text("subproblem", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("bridge.dci_agent_lite.subprocess.run", fake_run)
    monkeypatch.setattr("bridge.dci_agent_lite.time.time", lambda: 1000.0)
    adapter = DciAgentLiteFinalAdapter(
        DciAgentLiteConfig(
            runner_command=("python", "-m", "bridge.runner", "--"),
            package_dir=tmp_path / "package",
            agent_dir=tmp_path / "agent",
            provider="openai",
            model="model",
            system_prompt_file=tmp_path / "prompt.txt",
            run_timeout_seconds=600,
        )
    )

    with request_deadline(600):
        adapter.run(
            rendered_question="inspect",
            corpus_cwd=corpus,
            output_dir=output_dir,
            id_by_token={},
        )

    command = calls[0]
    option = command.index("--deadline-unix-seconds")
    boundary = command.index("--")
    assert option < boundary
    assert 1599.0 < float(command[option + 1]) <= 1600.0
