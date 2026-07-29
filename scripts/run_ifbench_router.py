#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType

_REQUIRED_ENV = (
    "COMPASS_LITELLM_PROXY_KEY",
    "DSPY_CACHEDIR",
    "LITELLM_MASTER_KEY",
    "SILICONFLOW_API_KEY_PRIMARY",
    "SILICONFLOW_API_KEY_SECONDARY",
)


class _TerminationRequested(Exception):
    def __init__(self, signum: int) -> None:
        super().__init__(f"termination requested by signal {signum}")
        self.signum = signum


def _handle_termination(signum: int, _frame: FrameType | None) -> None:
    raise _TerminationRequested(signum)


def _require_environment() -> dict[str, str]:
    missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise ValueError(f"required environment variables are unset: {missing}")
    if (
        os.environ["COMPASS_LITELLM_PROXY_KEY"]
        != os.environ["LITELLM_MASTER_KEY"]
    ):
        raise ValueError(
            "COMPASS_LITELLM_PROXY_KEY and LITELLM_MASTER_KEY must match"
        )
    return dict(os.environ)


def _load_run_dir(config_path: Path) -> Path:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = raw.get("deployment", {}).get("run_dir")
    if not isinstance(run_dir, str) or not run_dir:
        raise ValueError("config deployment.run_dir must be non-empty text")
    return Path(run_dir)


def _wait_for_proxy(
    process: subprocess.Popen[bytes],
    *,
    host: str,
    port: int,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            process.wait()
            raise RuntimeError(
                f"LiteLLM proxy exited before readiness with code {returncode}"
            )
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(
        f"LiteLLM proxy did not listen on {host}:{port} within "
        f"{timeout_seconds:g} seconds"
    )


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _terminate_process_group(
    process: subprocess.Popen[bytes] | None,
    *,
    grace_seconds: float,
) -> None:
    if process is None:
        return

    process_exited = process.poll() is not None
    if process_exited:
        process.wait()
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        if not process_exited:
            process.wait()
        return

    deadline = time.monotonic() + grace_seconds
    if not process_exited:
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            pass

    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            process.wait()
            return
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))

    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _monitor(
    proxy: subprocess.Popen[bytes],
    training: subprocess.Popen[bytes],
    *,
    poll_seconds: float = 0.2,
) -> int:
    while True:
        training_returncode = training.poll()
        if training_returncode is not None:
            training.wait()
            return training_returncode
        proxy_returncode = proxy.poll()
        if proxy_returncode is not None:
            proxy.wait()
            raise RuntimeError(
                f"LiteLLM proxy exited while training with code {proxy_returncode}"
            )
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run IFBench and its LiteLLM router with process-group cleanup. "
            "Secrets must be supplied through environment variables."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--router-config", required=True, type=Path)
    parser.add_argument("--project-root", default=Path.cwd(), type=Path)
    parser.add_argument("--python", default=sys.executable, type=Path)
    parser.add_argument("--litellm", required=True, type=Path)
    parser.add_argument("--proxy-host", default="127.0.0.1")
    parser.add_argument("--proxy-port", default=40029, type=int)
    parser.add_argument("--proxy-ready-timeout", default=60.0, type=float)
    parser.add_argument("--shutdown-grace", default=30.0, type=float)
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()

    if os.name != "posix":
        raise RuntimeError("this process-group supervisor requires POSIX")
    if args.proxy_port <= 0 or args.proxy_port > 65535:
        raise ValueError("--proxy-port must be between 1 and 65535")
    if args.proxy_ready_timeout <= 0:
        raise ValueError("--proxy-ready-timeout must be positive")
    if args.shutdown_grace <= 0:
        raise ValueError("--shutdown-grace must be positive")

    project_root = args.project_root.resolve()
    config_path = args.config.resolve()
    router_config_path = args.router_config.resolve()
    run_dir = _load_run_dir(config_path)
    run_parent = run_dir.parent
    run_parent.mkdir(parents=True, exist_ok=True)
    training_log_path = run_parent / f"{run_dir.name}.launch.log"
    proxy_log_path = run_parent / f"{run_dir.name}.proxy.launch.log"
    environment = _require_environment()
    if _port_is_open(args.proxy_host, args.proxy_port):
        raise RuntimeError(
            f"refusing to start because {args.proxy_host}:{args.proxy_port} "
            "is already in use"
        )

    training_command = [
        str(args.python.resolve()),
        str(project_root / "experiments" / "11_ifbench_sparse_raw_feedback_training.py"),
        "--config",
        str(config_path),
    ]
    if args.resume_existing:
        training_command.append("--resume-existing")
    proxy_command = [
        str(args.litellm.resolve()),
        "--config",
        str(router_config_path),
        "--host",
        args.proxy_host,
        "--port",
        str(args.proxy_port),
    ]

    proxy: subprocess.Popen[bytes] | None = None
    training: subprocess.Popen[bytes] | None = None
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    for signum in previous_handlers:
        signal.signal(signum, _handle_termination)

    with (
        proxy_log_path.open("ab", buffering=0) as proxy_log,
        training_log_path.open("ab", buffering=0) as training_log,
    ):
        try:
            proxy = subprocess.Popen(
                proxy_command,
                cwd=project_root,
                env=environment,
                stdout=proxy_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _wait_for_proxy(
                proxy,
                host=args.proxy_host,
                port=args.proxy_port,
                timeout_seconds=args.proxy_ready_timeout,
            )
            training = subprocess.Popen(
                training_command,
                cwd=project_root,
                env=environment,
                stdout=training_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            return _monitor(proxy, training)
        except _TerminationRequested as exc:
            return 128 + exc.signum
        finally:
            for signum in previous_handlers:
                signal.signal(signum, signal.SIG_IGN)
            try:
                _terminate_process_group(
                    training,
                    grace_seconds=args.shutdown_grace,
                )
                _terminate_process_group(
                    proxy,
                    grace_seconds=args.shutdown_grace,
                )
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
