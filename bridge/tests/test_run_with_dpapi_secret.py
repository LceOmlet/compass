from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_with_dpapi_secret.py"
SPEC = importlib.util.spec_from_file_location("run_with_dpapi_secret", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def test_secret_is_in_child_environment_but_not_argv(monkeypatch, tmp_path):
    secret_file = tmp_path / "secret.dpapi"
    secret_file.write_bytes(b"ciphertext")
    monkeypatch.setattr(subject, "decrypt_dpapi_blob", lambda path: "private-value")
    captured = {}

    def fake_run(command, *, env, check):
        captured["command"] = command
        captured["env"] = env
        captured["check"] = check
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(subject.subprocess, "run", fake_run)
    result = subject.run_with_secret(
        secret_file=secret_file,
        env_names=["COMPASS_GPT_GE_API_KEY", "OPENAI_API_KEY"],
        command=["python", "runner.py", "--config", "formal.json"],
    )
    assert result == 7
    assert captured["env"]["COMPASS_GPT_GE_API_KEY"] == "private-value"
    assert captured["env"]["OPENAI_API_KEY"] == "private-value"
    assert "private-value" not in captured["command"]
    assert captured["check"] is False


@pytest.mark.parametrize("name", ["", "1KEY", "BAD-NAME", "A=B"])
def test_invalid_environment_name_is_rejected(monkeypatch, name):
    monkeypatch.setattr(subject, "decrypt_dpapi_blob", lambda path: "unused")
    with pytest.raises(ValueError, match="env-name"):
        subject.run_with_secret(
            secret_file=Path("unused"),
            env_names=[name],
            command=["python"],
        )


def test_empty_command_is_rejected(monkeypatch):
    monkeypatch.setattr(subject, "decrypt_dpapi_blob", lambda path: "unused")
    with pytest.raises(ValueError, match="command"):
        subject.run_with_secret(
            secret_file=Path("unused"),
            env_names=["COMPASS_GPT_GE_API_KEY"],
            command=[],
        )


def test_duplicate_environment_names_are_rejected(monkeypatch):
    monkeypatch.setattr(subject, "decrypt_dpapi_blob", lambda path: "unused")
    with pytest.raises(ValueError, match="unique"):
        subject.run_with_secret(
            secret_file=Path("unused"),
            env_names=["OPENAI_API_KEY", "OPENAI_API_KEY"],
            command=["python"],
        )
