from __future__ import annotations

import io
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from bridge.dci_docker_isolation import (
    DciIsolationError,
    build_docker_command,
    main,
    parse_invocation,
)


_IMAGE = "sha256:" + "a" * 64


def _argv(
    tmp_path: Path,
    *,
    pass_environment: tuple[str, ...] = (),
    timeout_seconds: float | None = None,
    deadline_unix_seconds: float | None = None,
) -> list[str]:
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "selection").mkdir(exist_ok=True)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("system", encoding="utf-8")
    agent = tmp_path / "agent"
    agent.mkdir(exist_ok=True)
    (agent / "models.json").write_text("{}", encoding="utf-8")
    launcher = ["--image", _IMAGE]
    if timeout_seconds is not None:
        launcher.extend(["--timeout-seconds", str(timeout_seconds)])
    if deadline_unix_seconds is not None:
        launcher.extend(
            ["--deadline-unix-seconds", str(deadline_unix_seconds)]
        )
    for name in pass_environment:
        launcher.extend(["--pass-env", name])
    return [
        *launcher,
        "--",
        "--provider",
        "openai",
        "--model",
        "model",
        "--package-dir",
        str(tmp_path / "ignored-host-pi"),
        "--agent-dir",
        str(agent),
        "--cwd",
        str(corpus),
        "--tools",
        "read,bash",
        "--system-prompt-file",
        str(prompt),
        "--output-dir",
        str(tmp_path / "output"),
        "--max-turns",
        "20",
    ]


def test_command_mounts_only_exact_inputs_and_forwards_named_environment(
    tmp_path: Path,
) -> None:
    launcher, invocation = parse_invocation(
        _argv(tmp_path, pass_environment=("OPENAI_API_KEY",))
    )
    command = build_docker_command(
        launcher,
        invocation,
        environment={"OPENAI_API_KEY": "secret", "HELDOUT_PATH": "hidden"},
    )

    assert command[:8] == [
        "docker",
        "run",
        "--rm",
        "--init",
        "--read-only",
        "--pull",
        "never",
        "-i",
    ]
    assert command[8:12] == [
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
    ]
    mount_values = [
        command[position + 1]
        for position, value in enumerate(command[:-1])
        if value == "--mount"
    ]
    assert any("dst=/corpus,readonly" in item for item in mount_values)
    assert any(
        "dst=/corpus/selection" in item and "readonly" not in item
        for item in mount_values
    )
    assert any("dst=/input/system_prompt.txt,readonly" in item for item in mount_values)
    assert any("dst=/output" in item and "readonly" not in item for item in mount_values)
    assert any("dst=/agent-seed/models.json,readonly" in item for item in mount_values)
    assert not any("dst=/agent," in item for item in mount_values)
    assert not any("ignored-host-pi" in item for item in command)
    assert not any("HELDOUT_PATH" in item for item in command)
    assert "OPENAI_API_KEY" in command
    assert "secret" not in command
    assert command[-19:] == [
        _IMAGE,
        "--provider",
        "openai",
        "--model",
        "model",
        "--package-dir",
        "/opt/pi-mono/packages/coding-agent",
        "--agent-dir",
        "/agent",
        "--cwd",
        "/corpus",
        "--tools",
        "read,bash",
        "--system-prompt-file",
        "/input/system_prompt.txt",
        "--output-dir",
        "/output",
        "--max-turns",
        "20",
    ]


def test_requires_immutable_image_and_present_explicit_environment(
    tmp_path: Path,
) -> None:
    argv = _argv(tmp_path)
    argv[1] = "compass-dci:latest"
    with pytest.raises(DciIsolationError, match="immutable sha256"):
        parse_invocation(argv)

    launcher, invocation = parse_invocation(
        _argv(tmp_path, pass_environment=("OPENAI_API_KEY",))
    )
    with pytest.raises(DciIsolationError, match="OPENAI_API_KEY"):
        build_docker_command(launcher, invocation, environment={})


def test_rejects_nonempty_artifact_directory(tmp_path: Path) -> None:
    argv = _argv(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    launcher, invocation = parse_invocation(argv)

    with pytest.raises(DciIsolationError, match="must be empty"):
        build_docker_command(launcher, invocation, environment={})


def test_main_invokes_docker_once_without_a_host_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "bridge.dci_docker_isolation.sys.stdin",
        io.StringIO("question"),
    )
    monkeypatch.setattr(
        "bridge.dci_docker_isolation.subprocess.run",
        fake_run,
    )

    assert main(_argv(tmp_path)) == 0
    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["input"] == "question"
    assert kwargs["text"] is True
    assert kwargs["shell"] is False


def test_main_times_out_only_the_exact_container_and_removes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    cidfile: Path | None = None

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal cidfile
        calls.append((command, kwargs))
        if command[1] == "run":
            cidfile = Path(command[command.index("--cidfile") + 1])
            cidfile.write_text("a" * 64, encoding="ascii")
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "bridge.dci_docker_isolation.sys.stdin",
        io.StringIO("question"),
    )
    monkeypatch.setattr(
        "bridge.dci_docker_isolation.subprocess.run",
        fake_run,
    )

    assert main(_argv(tmp_path, timeout_seconds=600)) == 2
    assert len(calls) == 2
    assert calls[0][1]["timeout"] == 600
    assert calls[1][0] == ["docker", "rm", "-f", "a" * 64]
    assert cidfile is not None and not cidfile.exists()


def test_main_converts_shared_absolute_deadline_at_container_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(_command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "bridge.dci_docker_isolation.sys.stdin",
        io.StringIO("question"),
    )
    monkeypatch.setattr(
        "bridge.dci_docker_isolation.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "bridge.dci_docker_isolation.time.time",
        lambda: 1000.0,
    )

    assert main(_argv(tmp_path, deadline_unix_seconds=1600)) == 0
    assert calls[0]["timeout"] == 600
