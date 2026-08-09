from __future__ import annotations

import threading
from contextlib import contextmanager
from importlib import import_module
from typing import Any

from dspy.utils.parallelizer import ParallelExecutor

_DSPY_EVALUATE = import_module("dspy.evaluate.evaluate")
_OFFICIAL_EVALUATE_PARALLEL_EXECUTOR = _DSPY_EVALUATE.ParallelExecutor
_PARENT_EVALUATION_LOCK = threading.RLock()


def _parent_parallel_executor(*args: Any, **kwargs: Any) -> ParallelExecutor:
    # Current DSPy Evaluate always forwards its configured timeout and
    # straggler_limit (including the official defaults) to ParallelExecutor.
    # This owner scope intentionally replaces only those two scheduler values;
    # every other official executor argument is preserved unchanged.
    kwargs.pop("timeout", None)
    kwargs.pop("straggler_limit", None)
    return _OFFICIAL_EVALUATE_PARALLEL_EXECUTOR(
        *args,
        timeout=0,
        straggler_limit=0,
        **kwargs,
    )


@contextmanager
def without_parent_straggler_resubmission():
    """Run official DSPy Evaluate without its tail-straggler duplicate call."""

    with _PARENT_EVALUATION_LOCK:
        previous = _DSPY_EVALUATE.ParallelExecutor
        if previous not in (
            _OFFICIAL_EVALUATE_PARALLEL_EXECUTOR,
            _parent_parallel_executor,
        ):
            raise RuntimeError(
                "another component replaced DSPy Evaluate ParallelExecutor"
            )
        _DSPY_EVALUATE.ParallelExecutor = _parent_parallel_executor
        try:
            yield
        finally:
            _DSPY_EVALUATE.ParallelExecutor = previous


__all__ = ["without_parent_straggler_resubmission"]
