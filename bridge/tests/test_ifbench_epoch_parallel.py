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
