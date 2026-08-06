from __future__ import annotations

import math
import subprocess
import time
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from bridge.request_deadline import remaining_request_seconds


DataIdT = TypeVar("DataIdT", bound=Hashable)


class DciBoundaryError(RuntimeError):
    """The official DCI process did not produce a valid boundary value."""


@dataclass(frozen=True, slots=True)
class DciAgentLiteConfig:
    """Concrete invocation of the pinned official DCI-Agent-Lite CLI."""

    runner_command: tuple[str, ...]
    package_dir: Path
    agent_dir: Path
    provider: str
    model: str
    system_prompt_file: Path
    tools: str = "read,bash"
    max_turns: int | None = None
    run_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.runner_command or any(not part for part in self.runner_command):
            raise ValueError("runner_command must contain non-empty arguments")
        for name, value in (
            ("provider", self.provider),
            ("model", self.model),
            ("tools", self.tools),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.max_turns is not None and (
            isinstance(self.max_turns, bool)
            or not isinstance(self.max_turns, int)
            or self.max_turns <= 0
        ):
            raise TypeError("max_turns must be a positive integer or None")
        if self.run_timeout_seconds is not None and (
            isinstance(self.run_timeout_seconds, bool)
            or not isinstance(self.run_timeout_seconds, (int, float))
            or not math.isfinite(self.run_timeout_seconds)
            or self.run_timeout_seconds <= 0
        ):
            raise TypeError(
                "run_timeout_seconds must be a positive number or None"
            )


@dataclass(frozen=True, slots=True)
class DciResult(Generic[DataIdT]):
    """One official free-text result plus its explicit selected identities."""

    definition: str
    selected_ids: tuple[DataIdT, ...]
    corpus_dir: Path
    artifact_dir: Path


def load_dci_result(
    final_text: str,
    *,
    corpus_dir: Path,
    artifact_dir: Path,
    id_by_token: Mapping[str, DataIdT],
) -> DciResult[DataIdT]:
    """Keep the official final text verbatim and read explicit member markers."""

    if not final_text.strip():
        raise DciBoundaryError("official DCI final.txt is empty")
    selection_dir = corpus_dir / "selection"
    if not selection_dir.is_dir():
        raise DciBoundaryError("DCI selection workspace is missing")

    selected_tokens: set[str] = set()
    for marker in selection_dir.iterdir():
        if marker.is_symlink() or not marker.is_file():
            raise DciBoundaryError(
                "DCI selection workspace may contain only direct marker files"
            )
        token = marker.name
        if token not in id_by_token:
            raise DciBoundaryError(
                "DCI selection workspace contains an unknown DataId token"
            )
        selected_tokens.add(token)

    selected_ids = tuple(
        data_id
        for token, data_id in id_by_token.items()
        if token in selected_tokens
    )
    return DciResult(
        definition=final_text,
        selected_ids=selected_ids,
        corpus_dir=corpus_dir,
        artifact_dir=artifact_dir,
    )


class DciAgentLiteFinalAdapter(Generic[DataIdT]):
    """Invoke the official CLI and adapt only its authoritative artifacts."""

    def __init__(self, config: DciAgentLiteConfig) -> None:
        self.config = config

    def run(
        self,
        *,
        rendered_question: str,
        corpus_cwd: Path,
        output_dir: Path,
        id_by_token: Mapping[str, DataIdT],
    ) -> DciResult[DataIdT]:
        if not rendered_question.strip():
            raise ValueError("rendered_question must be non-empty")
        selection_dir = corpus_cwd / "selection"
        if not selection_dir.is_dir():
            raise DciBoundaryError("DCI selection workspace is missing")
        if any(selection_dir.iterdir()):
            raise DciBoundaryError("DCI selection workspace must start empty")

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        runner_command = list(self.config.runner_command)
        if self.config.run_timeout_seconds is not None:
            remaining = remaining_request_seconds()
            timeout_seconds = float(self.config.run_timeout_seconds)
            if remaining is not None:
                timeout_seconds = min(timeout_seconds, remaining)
            try:
                boundary = runner_command.index("--")
            except ValueError as exc:
                raise DciBoundaryError(
                    "a deadline-aware DCI runner command must contain '--'"
                ) from exc
            runner_command[boundary:boundary] = [
                "--deadline-unix-seconds",
                repr(time.time() + timeout_seconds),
            ]

        command: list[str] = [
            *runner_command,
            "--provider",
            self.config.provider,
            "--model",
            self.config.model,
            "--package-dir",
            str(self.config.package_dir),
            "--agent-dir",
            str(self.config.agent_dir),
            "--cwd",
            str(corpus_cwd),
            "--tools",
            self.config.tools,
            "--system-prompt-file",
            str(self.config.system_prompt_file),
            "--output-dir",
            str(output_dir),
        ]
        if self.config.max_turns is not None:
            command.extend(["--max-turns", str(self.config.max_turns)])

        completed = subprocess.run(
            command,
            input=rendered_question,
            text=True,
            capture_output=True,
            check=False,
            shell=False,
        )
        if completed.returncode != 0:
            raise DciBoundaryError(
                "official DCI-Agent-Lite exited non-zero; inspect its artifact directory"
            )

        final_path = output_dir / "final.txt"
        if not final_path.is_file():
            raise DciBoundaryError("official DCI run did not write final.txt")
        return load_dci_result(
            final_path.read_text(encoding="utf-8"),
            corpus_dir=corpus_cwd,
            artifact_dir=output_dir,
            id_by_token=id_by_token,
        )


__all__ = [
    "DciAgentLiteConfig",
    "DciAgentLiteFinalAdapter",
    "DciBoundaryError",
    "DciResult",
    "load_dci_result",
]
