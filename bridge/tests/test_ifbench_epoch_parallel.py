from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
from gepa.core.adapter import EvaluationBatch
from gepa.core.data_loader import ListDataLoader
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.strategies.proposal_sampling import (
    IndependentSampling,
    SingleMutationSampling,
)

from bridge.b14_flashtrace_token_ids import (
    _ExactTokenStateMixin,
    _WeightedSinkScopeMixin,
)
from bridge.b16_official_gepa_ifbench import (
    TerminalAnalysisDspyAdapter,
    _proposal_sampling_strategy,
)
from bridge.b17_dspy_terminal_analysis import (
    DSPyParentAnalysisBuilder,
    DSPyTerminalAnalysisBridge,
)
from bridge.terminal_reflection import TerminalAnalysisUnavailableError


def test_full_epoch_sampling_is_exactly_fifty_three_item_tasks() -> None:
    strategy = _proposal_sampling_strategy(
        trainset_size=150,
        minibatch_size=3,
        epoch_parallel_enabled=True,
    )
    assert isinstance(strategy, IndependentSampling)
    assert strategy.n == 50

    loader = ListDataLoader(list(range(150)))
    sampler = EpochShuffledBatchSampler(
        minibatch_size=3,
        rng=random.Random(0),
    )
    state = SimpleNamespace(
        i=204,
        program_candidates=[{"prompt": "parent"}],
    )

    class Selector:
        def __init__(self) -> None:
            self.calls = 0

        def select_candidate_idx(self, selected_state: Any) -> int:
            assert selected_state is state
            self.calls += 1
            return 0

    selector = Selector()
    tasks = strategy.sample_tasks(state, selector, sampler, loader)

    assert selector.calls == 50
    assert len(tasks) == 50
    assert all(len(task.minibatch_ids) == 3 for task in tasks)
    assert sorted(
        instance_id
        for task in tasks
        for instance_id in task.minibatch_ids
    ) == list(range(150))


def test_single_task_sampling_remains_the_default() -> None:
    strategy = _proposal_sampling_strategy(
        trainset_size=150,
        minibatch_size=3,
        epoch_parallel_enabled=False,
    )
    assert isinstance(strategy, SingleMutationSampling)


def test_terminal_adapter_candidate_batch_is_concurrent_and_ordered() -> None:
    adapter = object.__new__(TerminalAnalysisDspyAdapter)
    adapter.max_candidate_workers = 3
    barrier = threading.Barrier(3)

    def evaluate(
        batch: list[Any],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        assert capture_traces
        barrier.wait(timeout=5)
        return EvaluationBatch(
            outputs=[candidate["prompt"]],
            scores=[float(batch[0])],
            trajectories=[{"example": batch[0]}],
        )

    adapter.evaluate = evaluate  # type: ignore[method-assign]
    results = adapter.batch_evaluate(
        [
            ({"prompt": "a"}, [1]),
            ({"prompt": "b"}, [2]),
            ({"prompt": "c"}, [3]),
        ]
    )

    assert [result.outputs for result in results] == [["a"], ["b"], ["c"]]
    assert [result.scores for result in results] == [[1.0], [2.0], [3.0]]


def test_terminal_analysis_bridge_keeps_multiple_exact_jobs_bound() -> None:
    builder = object.__new__(DSPyParentAnalysisBuilder)
    prepared_a = object()
    prepared_b = object()

    def prepare(
        *,
        component: str,
        references: tuple[Any, ...],
        known_candidates: frozenset[Any],
    ) -> object:
        assert component == "component"
        assert known_candidates
        return references[0]

    builder.prepare = prepare  # type: ignore[method-assign]
    dataset_a = {"component": ({"job": "a"},)}
    dataset_b = {"component": ({"job": "b"},)}
    parent_a = {"component": "parent-a"}
    parent_b = {"component": "parent-b"}

    with ThreadPoolExecutor(max_workers=2) as executor:
        bridge = DSPyTerminalAnalysisBridge(builder=builder, executor=executor)
        bridge.update_candidate_pool((parent_a, parent_b))
        bridge.bind_admission_references(
            parent_candidate=parent_a,
            component="component",
            reflective_dataset=dataset_a,
            references=(prepared_a,),  # type: ignore[arg-type]
        )
        bridge.bind_admission_references(
            parent_candidate=parent_b,
            component="component",
            reflective_dataset=dataset_b,
            references=(prepared_b,),  # type: ignore[arg-type]
        )

        assert (
            bridge.take(
                parent_candidate=parent_a,
                component="component",
                reflective_dataset=dataset_a,
            )
            is prepared_a
        )
        assert (
            bridge.take(
                parent_candidate=parent_b,
                component="component",
                reflective_dataset=dataset_b,
            )
            is prepared_b
        )
        with pytest.raises(
            TerminalAnalysisUnavailableError,
            match="different parent/component",
        ):
            bridge.take(
                parent_candidate=parent_b,
                component="component",
                reflective_dataset=dataset_a,
            )

        bridge.discard(
            parent_candidate=parent_a,
            component="component",
            reflective_dataset=dataset_a,
        )
        assert (
            bridge.take(
                parent_candidate=parent_b,
                component="component",
                reflective_dataset=dataset_b,
            )
            is prepared_b
        )
        bridge.clear()


def _assert_shared_flashtrace_scope_waits(
    first_scope: Any,
    second_scope: Any,
) -> None:
    first_entered = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()

    def first() -> None:
        with first_scope():
            first_entered.set()
            assert release_first.wait(timeout=5)

    def second() -> None:
        assert first_entered.wait(timeout=5)
        second_attempted.set()
        with second_scope():
            second_entered.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(first)
        second_future = executor.submit(second)
        assert first_entered.wait(timeout=5)
        assert second_attempted.wait(timeout=5)
        assert not second_entered.wait(timeout=0.1)
        release_first.set()
        first_future.result(timeout=5)
        second_future.result(timeout=5)
    assert second_entered.is_set()


def test_exact_token_scope_queues_concurrent_epoch_tasks() -> None:
    owner = _ExactTokenStateMixin()
    first_state = object()
    second_state = object()
    _assert_shared_flashtrace_scope_waits(
        lambda: owner._exact_token_scope(first_state),  # type: ignore[arg-type]
        lambda: owner._exact_token_scope(second_state),  # type: ignore[arg-type]
    )
    assert owner._active_exact_token_state is None


def test_weighted_sink_scope_queues_concurrent_epoch_tasks() -> None:
    owner = _WeightedSinkScopeMixin()
    _assert_shared_flashtrace_scope_waits(
        lambda: owner.weighted_sink_scope((1.0,)),
        lambda: owner.weighted_sink_scope((2.0,)),
    )
    assert owner._active_external_sink_weights is None


def test_unweighted_exact_task_cannot_observe_another_tasks_sink_weights() -> None:
    class Owner(_WeightedSinkScopeMixin, _ExactTokenStateMixin):
        pass

    owner = Owner()
    weighted_entered = threading.Event()
    unweighted_attempted = threading.Event()
    release_weighted = threading.Event()
    observed: list[tuple[float, ...] | None] = []

    def weighted() -> None:
        with owner.weighted_sink_scope((1.0, 2.0)):
            weighted_entered.set()
            assert release_weighted.wait(timeout=5)

    def unweighted() -> None:
        assert weighted_entered.wait(timeout=5)
        unweighted_attempted.set()
        with owner._exact_token_scope(object()):  # type: ignore[arg-type]
            observed.append(owner._active_external_sink_weights)

    with ThreadPoolExecutor(max_workers=2) as executor:
        weighted_future = executor.submit(weighted)
        unweighted_future = executor.submit(unweighted)
        assert weighted_entered.wait(timeout=5)
        assert unweighted_attempted.wait(timeout=5)
        release_weighted.set()
        weighted_future.result(timeout=5)
        unweighted_future.result(timeout=5)

    assert observed == [None]
