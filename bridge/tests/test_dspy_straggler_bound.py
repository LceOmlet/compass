from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from dspy.utils import parallelizer as dspy_parallelizer

from bridge.b18_dspy_straggler_bound import (
    _AtMostOneResubmissionExecutor,
    _OFFICIAL_THREAD_POOL_EXECUTOR,
    bounded_dspy_straggler_resubmission,
)


def test_bounded_executor_scopes_can_overlap_without_serializing_rollouts() -> None:
    barrier = threading.Barrier(3)

    def enter_scope() -> object:
        with bounded_dspy_straggler_resubmission():
            assert (
                dspy_parallelizer.ThreadPoolExecutor
                is _AtMostOneResubmissionExecutor
            )
            barrier.wait(timeout=5)
            return dspy_parallelizer.ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=3) as executor:
        observed = list(executor.map(lambda _index: enter_scope(), range(3)))

    assert observed == [_AtMostOneResubmissionExecutor] * 3
    assert dspy_parallelizer.ThreadPoolExecutor is _OFFICIAL_THREAD_POOL_EXECUTOR
