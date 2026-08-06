from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


_DIGEST_REFERENCE = re.compile(
    r"(?:^sha256:|@sha256:)[0-9a-f]{64}$",
    flags=re.ASCII,
)
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$", flags=re.ASCII)
_CONTAINER_ID = re.compile(r"^[0-9a-f]{12,64}$", flags=re.ASCII)
_DOCKER_CLEANUP_TIMEOUT_SECONDS = 30


class DciIsolationError(RuntimeError):
    """The requested process cannot be run inside the exact Docker boundary."""


@dataclass(frozen=True, slots=True)
class DockerLauncherConfig:
    image: str
    docker_executable: str
    pass_environment: tuple[str, ...]
    timeout_seconds: float | None
    deadline_unix_seconds: float | None


@dataclass(frozen=True, slots=True)
class OfficialDciInvocation:
    provider: str
    model: str
    package_dir: Path
    agent_dir: Path
    corpus_dir: Path
    tools: str
    system_prompt_file: Path
    output_dir: Path
    max_turns: int | None


def _split_launcher_and_official_args(
    argv: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        boundary = argv.index("--")
    except ValueError as exc:
        raise DciIsolationError(
            "Docker launcher arguments must end with '--' before official DCI arguments"
        ) from exc
    launcher = tuple(argv[:boundary])
    official = tuple(argv[boundary + 1 :])
    if not official:
        raise DciIsolationError("official DCI arguments are missing")
    return launcher, official


def parse_invocation(
    argv: Sequence[str],
) -> tuple[DockerLauncherConfig, OfficialDciInvocation]:
    launcher_args, official_args = _split_launcher_and_official_args(tuple(argv))

    launcher_parser = argparse.ArgumentParser(
        prog="python -m bridge.dci_docker_isolation",
        allow_abbrev=False,
    )
    launcher_parser.add_argument("--image", required=True)
    launcher_parser.add_argument("--docker-executable", default="docker")
    launcher_parser.add_argument("--timeout-seconds", type=float)
    launcher_parser.add_argument("--deadline-unix-seconds", type=float)
    launcher_parser.add_argument(
        "--pass-env",
        action="append",
        default=[],
        dest="pass_environment",
    )
    launcher = launcher_parser.parse_args(launcher_args)
    if not _DIGEST_REFERENCE.search(launcher.image):
        raise DciIsolationError(
            "--image must be an immutable sha256 image ID or repository digest"
        )
    environment_names = tuple(dict.fromkeys(launcher.pass_environment))
    if any(not _ENVIRONMENT_NAME.fullmatch(name) for name in environment_names):
        raise DciIsolationError("--pass-env values must be environment variable names")
    if launcher.timeout_seconds is not None and (
        not math.isfinite(launcher.timeout_seconds)
        or launcher.timeout_seconds <= 0
    ):
        raise DciIsolationError("--timeout-seconds must be positive and finite")
    if launcher.deadline_unix_seconds is not None and (
        not math.isfinite(launcher.deadline_unix_seconds)
        or launcher.deadline_unix_seconds <= 0
    ):
        raise DciIsolationError(
            "--deadline-unix-seconds must be positive and finite"
        )
    if (
        launcher.timeout_seconds is not None
        and launcher.deadline_unix_seconds is not None
    ):
        raise DciIsolationError(
            "use only one of --timeout-seconds and --deadline-unix-seconds"
        )

    official_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    official_parser.add_argument("--provider", required=True)
    official_parser.add_argument("--model", required=True)
    official_parser.add_argument("--package-dir", type=Path, required=True)
    official_parser.add_argument("--agent-dir", type=Path, required=True)
    official_parser.add_argument("--cwd", type=Path, required=True, dest="corpus_dir")
    official_parser.add_argument("--tools", required=True)
    official_parser.add_argument("--system-prompt-file", type=Path, required=True)
    official_parser.add_argument("--output-dir", type=Path, required=True)
    official_parser.add_argument("--max-turns", type=int)
    official = official_parser.parse_args(official_args)
    if official.max_turns is not None and official.max_turns <= 0:
        raise DciIsolationError("--max-turns must be positive")

    return (
        DockerLauncherConfig(
            image=launcher.image,
            docker_executable=launcher.docker_executable,
            pass_environment=environment_names,
            timeout_seconds=launcher.timeout_seconds,
            deadline_unix_seconds=launcher.deadline_unix_seconds,
        ),
        OfficialDciInvocation(
            provider=official.provider,
            model=official.model,
            package_dir=official.package_dir,
            agent_dir=official.agent_dir,
            corpus_dir=official.corpus_dir,
            tools=official.tools,
            system_prompt_file=official.system_prompt_file,
            output_dir=official.output_dir,
            max_turns=official.max_turns,
        ),
    )


def _existing_directory(path: Path, *, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise DciIsolationError(f"{label} does not exist") from exc
    if not resolved.is_dir():
        raise DciIsolationError(f"{label} must be a directory")
    return resolved


def _existing_file(path: Path, *, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise DciIsolationError(f"{label} does not exist") from exc
    if not resolved.is_file():
        raise DciIsolationError(f"{label} must be a file")
    return resolved


def _empty_output_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir():
        raise DciIsolationError("output directory must be a directory")
    if any(resolved.iterdir()):
        raise DciIsolationError("output directory must be empty")
    return resolved


def _bind_mount(source: Path, destination: str, *, readonly: bool) -> str:
    source_text = str(source)
    if "," in source_text:
        raise DciIsolationError("Docker bind source paths cannot contain commas")
    value = f"type=bind,src={source_text},dst={destination}"
    return f"{value},readonly" if readonly else value


def build_docker_command(
    launcher: DockerLauncherConfig,
    invocation: OfficialDciInvocation,
    *,
    environment: Mapping[str, str] | None = None,
    cidfile: Path | None = None,
) -> list[str]:
    host_environment = os.environ if environment is None else environment
    missing_environment = tuple(
        name for name in launcher.pass_environment if name not in host_environment
    )
    if missing_environment:
        raise DciIsolationError(
            "requested environment variables are missing: "
            + ", ".join(missing_environment)
        )

    corpus_dir = _existing_directory(invocation.corpus_dir, label="corpus directory")
    selection_dir = _existing_directory(
        corpus_dir / "selection",
        label="selection workspace",
    )
    system_prompt_file = _existing_file(
        invocation.system_prompt_file,
        label="system prompt file",
    )
    output_dir = _empty_output_directory(invocation.output_dir)

    command = [
        launcher.docker_executable,
        "run",
        "--rm",
        "--init",
        "--read-only",
        "--pull",
        "never",
        "-i",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--mount",
        _bind_mount(corpus_dir, "/corpus", readonly=True),
        "--mount",
        _bind_mount(selection_dir, "/corpus/selection", readonly=False),
        "--mount",
        _bind_mount(system_prompt_file, "/input/system_prompt.txt", readonly=True),
        "--mount",
        _bind_mount(output_dir, "/output", readonly=False),
        "--tmpfs",
        "/tmp:rw,nosuid,nodev",
        "--tmpfs",
        "/agent:rw,nosuid,nodev",
        "--workdir",
        "/corpus",
        "--env",
        "HOME=/tmp/home",
        "--env",
        "PI_CODING_AGENT_DIR=/agent",
    ]
    if cidfile is not None:
        command[8:8] = ["--cidfile", str(cidfile)]
    for name in launcher.pass_environment:
        command.extend(["--env", name])

    agent_dir = invocation.agent_dir.expanduser()
    if agent_dir.exists():
        if not agent_dir.is_dir():
            raise DciIsolationError("agent directory must be a directory")
        for filename in ("models.json", "settings.json"):
            candidate = agent_dir / filename
            if candidate.exists():
                source = _existing_file(candidate, label=filename)
                command.extend(
                    [
                        "--mount",
                        _bind_mount(
                            source,
                            f"/agent-seed/{filename}",
                            readonly=True,
                        ),
                    ]
                )

    command.extend(
        [
            launcher.image,
            "--provider",
            invocation.provider,
            "--model",
            invocation.model,
            "--package-dir",
            "/opt/pi-mono/packages/coding-agent",
            "--agent-dir",
            "/agent",
            "--cwd",
            "/corpus",
            "--tools",
            invocation.tools,
            "--system-prompt-file",
            "/input/system_prompt.txt",
            "--output-dir",
            "/output",
        ]
    )
    if invocation.max_turns is not None:
        command.extend(["--max-turns", str(invocation.max_turns)])
    return command


def _force_remove_container(
    launcher: DockerLauncherConfig,
    cidfile: Path,
) -> None:
    try:
        container_id = cidfile.read_text(encoding="ascii").strip()
    except OSError:
        return
    if not _CONTAINER_ID.fullmatch(container_id):
        return
    subprocess.run(
        [launcher.docker_executable, "rm", "-f", container_id],
        check=False,
        capture_output=True,
        shell=False,
        timeout=_DOCKER_CLEANUP_TIMEOUT_SECONDS,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        launcher, invocation = parse_invocation(
            tuple(sys.argv[1:] if argv is None else argv)
        )
        output_parent = invocation.output_dir.expanduser().resolve().parent
        cidfile = output_parent / (
            f".{invocation.output_dir.name}.{uuid.uuid4().hex}.docker.cid"
        )
        command = build_docker_command(
            launcher,
            invocation,
            cidfile=cidfile,
        )
        question = sys.stdin.read()
        timeout_seconds = launcher.timeout_seconds
        if launcher.deadline_unix_seconds is not None:
            timeout_seconds = launcher.deadline_unix_seconds - time.time()
            if timeout_seconds <= 0:
                print(
                    "DCI isolation error: operation deadline exceeded",
                    file=sys.stderr,
                )
                return 2
        try:
            completed = subprocess.run(
                command,
                input=question,
                text=True,
                check=False,
                shell=False,
                timeout=timeout_seconds,
            )
            return int(completed.returncode)
        except subprocess.TimeoutExpired:
            _force_remove_container(launcher, cidfile)
            print(
                "DCI isolation error: operation deadline exceeded",
                file=sys.stderr,
            )
            return 2
        except BaseException:
            _force_remove_container(launcher, cidfile)
            raise
        finally:
            cidfile.unlink(missing_ok=True)
    except DciIsolationError as exc:
        print(f"DCI isolation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DciIsolationError",
    "DockerLauncherConfig",
    "OfficialDciInvocation",
    "build_docker_command",
    "main",
    "parse_invocation",
]
