from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from concurrent.futures import Future
from contextlib import contextmanager
from numbers import Integral
from typing import Any

from dspy.utils import parallelizer as dspy_parallelizer


_OFFICIAL_THREAD_POOL_EXECUTOR = dspy_parallelizer.ThreadPoolExecutor
_PATCH_LOCK = threading.RLock()


class _SuppressedRepeatedResubmission(RuntimeError):
    pass


class _AtMostOneResubmissionExecutor(_OFFICIAL_THREAD_POOL_EXECUTOR):
    """Preserve official execution while enforcing its stated per-index retry bound."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._submission_lock = threading.RLock()
        self._real_submit_count: dict[int, int] = {}
        self._item_identity: dict[int, Any] = {}

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        qualname = getattr(fn, "__qualname__", "")
        if (
            kwargs
            or len(args) != 4
            or getattr(fn, "__module__", None) != dspy_parallelizer.__name__
            or not qualname.endswith("ParallelExecutor._execute_parallel.<locals>.worker")
        ):
            raise RuntimeError("official DSPy parallel submit contract changed")
        parent_overrides, submission_id, index, item = args
        if not isinstance(parent_overrides, Mapping):
            raise RuntimeError("official DSPy parent overrides are not a mapping")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in (submission_id, index)
        ):
            raise RuntimeError("official DSPy submission and item indices must be integers")
        item_index = int(index)

        with self._submission_lock:
            if item_index in self._item_identity and self._item_identity[item_index] is not item:
                raise RuntimeError("official DSPy reused one item index for another object")
            count = self._real_submit_count.get(item_index, 0)
            if count >= 2:
                placeholder: Future[Any] = Future()
                placeholder.set_exception(
                    _SuppressedRepeatedResubmission(
                        "official DSPy attempted more than one resubmission for an item"
                    )
                )
                return placeholder
            future = super().submit(fn, *args)
            self._item_identity[item_index] = item
            self._real_submit_count[item_index] = count + 1
            return future


@contextmanager
def bounded_dspy_straggler_resubmission() -> Iterator[None]:
    """Scope the official executor to its documented one-resubmit-per-item intent."""

    with _PATCH_LOCK:
        previous = dspy_parallelizer.ThreadPoolExecutor
        if previous is not _OFFICIAL_THREAD_POOL_EXECUTOR:
            raise RuntimeError("another component replaced the official DSPy thread executor")
        dspy_parallelizer.ThreadPoolExecutor = _AtMostOneResubmissionExecutor
        try:
            yield
        finally:
            if dspy_parallelizer.ThreadPoolExecutor is not _AtMostOneResubmissionExecutor:
                dspy_parallelizer.ThreadPoolExecutor = previous
                raise RuntimeError("official DSPy thread executor changed inside bounded scope")
            dspy_parallelizer.ThreadPoolExecutor = previous


__all__ = ["bounded_dspy_straggler_resubmission"]
