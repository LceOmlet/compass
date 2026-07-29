from __future__ import annotations

import os
import runpy
import signal
import subprocess
from pathlib import Path
from unittest.mock import Mock, call

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_ENTRY = _ROOT / "scripts" / "run_ifbench_router.py"


def _namespace() -> dict[str, object]:
    return runpy.run_path(str(_ENTRY))


def test_environment_requires_secret_values_without_putting_them_in_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _namespace()
    for name in namespace["_REQUIRED_ENV"]:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="required environment variables"):
        namespace["_require_environment"]()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups only")
def test_terminate_process_group_escalates_and_reaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _namespace()
    process = Mock()
    process.pid = 27182
    process.poll.return_value = None
    process.wait.side_effect = [
        subprocess.TimeoutExpired(cmd="proxy", timeout=0),
        -signal.SIGKILL,
    ]
    killpg = Mock()
    monkeypatch.setattr(os, "killpg", killpg)

    namespace["_terminate_process_group"](process, grace_seconds=0)

    assert killpg.call_args_list == [
        call(process.pid, signal.SIGTERM),
        call(process.pid, 0),
        call(process.pid, signal.SIGKILL),
    ]
    assert process.wait.call_args_list == [
        call(timeout=0),
        call(),
    ]


def test_monitor_returns_training_failure_after_reaping() -> None:
    namespace = _namespace()
    proxy = Mock()
    proxy.poll.return_value = None
    training = Mock()
    training.poll.return_value = 1

    assert namespace["_monitor"](proxy, training, poll_seconds=0) == 1
    training.wait.assert_called_once_with()
    proxy.wait.assert_not_called()


def test_monitor_reaps_proxy_failure() -> None:
    namespace = _namespace()
    proxy = Mock()
    proxy.poll.return_value = 9
    training = Mock()
    training.poll.return_value = None

    with pytest.raises(RuntimeError, match="proxy exited while training"):
        namespace["_monitor"](proxy, training, poll_seconds=0)

    proxy.wait.assert_called_once_with()
