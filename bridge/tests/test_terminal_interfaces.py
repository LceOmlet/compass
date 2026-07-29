from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from gepa.proposer.reflective_mutation.reflection_lm import ReflectionProposal

from bridge.b03_token_replay import TokenReplayUtility
from bridge.terminal_reflection import (
    PreparedTerminalAnalysis,
    TerminalAnalysisUnavailableError,
    TerminalLikelihoodReflectionLM,
)


class _PreparedAnalysis(PreparedTerminalAnalysis):
    def __init__(self) -> None:
        self.scored: tuple[Mapping[str, str], ...] = ()
        self.dependency_calls: list[Mapping[str, str]] = []
        self.known_calls: list[Mapping[str, str]] = []

    def is_known_candidate(self, candidate: Mapping[str, str]) -> bool:
        self.known_calls.append(dict(candidate))
        return candidate == {"first": "known", "second": "unchanged"}

    def teacher_forcing_scores(
        self,
        *,
        component: str,
        candidate_programs: tuple[Mapping[str, str], ...],
    ) -> tuple[float, ...]:
        assert component == "first"
        self.scored = tuple(dict(candidate) for candidate in candidate_programs)
        values = {"unavailable": 10.0, "selected": 9.0, "rejected": 8.0}
        return tuple(values[candidate["first"]] for candidate in candidate_programs)

    def dependency_distance(
        self,
        *,
        component: str,
        candidate_program: Mapping[str, str],
    ) -> float:
        assert component == "first"
        self.dependency_calls.append(dict(candidate_program))
        if candidate_program["first"] == "unavailable":
            raise TerminalAnalysisUnavailableError("candidate render is unavailable")
        return 0.25 if candidate_program["first"] == "selected" else 0.9


class _AnalysisProvider:
    def __init__(self, prepared: _PreparedAnalysis) -> None:
        self.prepared = prepared
        self.take_calls = 0
        self.discard_calls = 0

    def take(
        self,
        *,
        parent_candidate: Mapping[str, str],
        component: str,
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> PreparedTerminalAnalysis:
        assert parent_candidate == {"first": "old", "second": "unchanged"}
        assert component == "first"
        assert reflective_dataset["first"]
        self.take_calls += 1
        return self.prepared

    def update_candidate_pool(self, candidates: Sequence[Mapping[str, str]]) -> None:
        del candidates

    def discard(
        self,
        *,
        parent_candidate: Mapping[str, str],
        component: str,
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> None:
        del parent_candidate, component, reflective_dataset
        self.discard_calls += 1


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


def test_terminal_selection_passes_full_programs_and_skips_one_unavailable_candidate() -> None:
    completions = (
        "```unavailable```",
        "```selected```",
        "```selected```",
        "```known```",
        "```rejected```",
    )
    prepared = _PreparedAnalysis()
    provider = _AnalysisProvider(prepared)
    logger = _Logger()
    reflection = TerminalLikelihoodReflectionLM(
        complete=lambda _prompt, *, n: completions if n == len(completions) else (),
        analysis_provider=provider,
        n_candidates=len(completions),
        epsilon_dep=0.8,
        logger=logger,
    )

    proposal, returned = reflection.reflect(
        candidate={"first": "old", "second": "unchanged"},
        reflective_dataset={"first": ({"input": "x", "feedback": "y"},)},
        components_to_update=["first"],
    )

    assert returned is reflection
    assert proposal.new_texts == {"first": "selected"}
    assert prepared.scored == (
        {"first": "unavailable", "second": "unchanged"},
        {"first": "selected", "second": "unchanged"},
        {"first": "rejected", "second": "unchanged"},
    )
    assert prepared.dependency_calls == [
        {"first": "unavailable", "second": "unchanged"},
        {"first": "selected", "second": "unchanged"},
    ]
    assert proposal.metadata["proposal_duplicate_count"] == 2
    assert proposal.metadata["teacher_forcing_enabled"] is True
    assert provider.take_calls == 1
    assert provider.discard_calls == 1
    assert any(
        "candidate_index=0" in message
        and "dependency_unavailable=True" in message
        and "candidate_skipped=True" in message
        for message in logger.messages
    )


def test_teacher_forcing_free_selection_skips_scoring_and_keeps_dependency_gate() -> None:
    prepared = _PreparedAnalysis()
    provider = _AnalysisProvider(prepared)
    logger = _Logger()
    completion_calls: list[int] = []

    def complete(_prompt: str, *, n: int) -> tuple[str, ...]:
        completion_calls.append(n)
        return ("```selected```",)

    reflection = TerminalLikelihoodReflectionLM(
        complete=complete,
        analysis_provider=provider,
        n_candidates=1,
        epsilon_dep=0.8,
        teacher_forcing_enabled=False,
        logger=logger,
    )

    proposal, _ = reflection.reflect(
        candidate={"first": "old", "second": "unchanged"},
        reflective_dataset={"first": ({"input": "x", "feedback": "y"},)},
        components_to_update=["first"],
    )

    assert completion_calls == [1]
    assert prepared.scored == ()
    assert prepared.dependency_calls == [
        {"first": "selected", "second": "unchanged"}
    ]
    assert proposal.new_texts == {"first": "selected"}
    assert proposal.metadata["teacher_forcing_enabled"] is False
    assert "teacher_forcing_score" not in proposal.metadata
    assert any(
        "teacher_forcing_enabled=False" in message
        and "gate_passed=True" in message
        for message in logger.messages
    )


def test_teacher_forcing_free_selection_can_reject_the_only_candidate() -> None:
    prepared = _PreparedAnalysis()
    provider = _AnalysisProvider(prepared)
    reflection = TerminalLikelihoodReflectionLM(
        complete=lambda _prompt, *, n: ("```rejected```",) if n == 1 else (),
        analysis_provider=provider,
        n_candidates=1,
        epsilon_dep=0.8,
        teacher_forcing_enabled=False,
    )

    proposal, _ = reflection.reflect(
        candidate={"first": "old", "second": "unchanged"},
        reflective_dataset={"first": ({"input": "x", "feedback": "y"},)},
        components_to_update=["first"],
    )

    assert prepared.scored == ()
    assert proposal.new_texts == {}


def test_teacher_forcing_free_selection_requires_one_candidate() -> None:
    with pytest.raises(
        ValueError,
        match="requires n_candidates=1",
    ):
        TerminalLikelihoodReflectionLM(
            complete=lambda _prompt, *, n: ("```selected```",) * n,
            analysis_provider=_AnalysisProvider(_PreparedAnalysis()),
            n_candidates=3,
            epsilon_dep=0.8,
            teacher_forcing_enabled=False,
        )


def test_terminal_reflect_many_runs_concurrently_and_restores_job_order() -> None:
    reflection = TerminalLikelihoodReflectionLM(
        complete=lambda _prompt, *, n: ("```unused```",) * n,
        analysis_provider=_AnalysisProvider(_PreparedAnalysis()),
        n_candidates=1,
        epsilon_dep=0.8,
        teacher_forcing_enabled=False,
        max_reflection_workers=3,
    )
    barrier = threading.Barrier(3)

    def reflect(
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[ReflectionProposal, TerminalLikelihoodReflectionLM]:
        del reflective_dataset, components_to_update
        barrier.wait(timeout=5)
        index = int(candidate["first"])
        time.sleep((2 - index) * 0.01)
        return ReflectionProposal(new_texts={"first": f"new-{index}"}), reflection

    reflection.reflect = reflect  # type: ignore[method-assign]
    jobs = [
        (
            {"first": str(index), "second": "unchanged"},
            {"first": ({"input": index},)},
            ["first"],
        )
        for index in range(3)
    ]

    results = reflection.reflect_many(jobs)

    assert [proposal.new_texts["first"] for proposal, _ in results] == [
        "new-0",
        "new-1",
        "new-2",
    ]


def test_replay_scores_allow_heterogeneous_original_skills() -> None:
    replay = object.__new__(TokenReplayUtility)
    replay._captured_prompt_ids = lambda rollout: (1, len(rollout.skill_text))
    replay._prompt_ids = lambda rollout, candidate: (2, len(rollout.skill_text), len(candidate))
    likelihood_batches = iter(
        (
            ((-1.0,), (-2.0,)),
            ((-3.0,), (-4.0,)),
        )
    )
    replay._packed_selected_log_likelihoods = lambda **_kwargs: next(likelihood_batches)

    def rollout(skill: str, token: int) -> SimpleNamespace:
        return SimpleNamespace(
            skill_text=skill,
            token_record=SimpleNamespace(completion_token_ids=(token, 0)),
        )

    def credit(token: int, reward: float) -> SimpleNamespace:
        return SimpleNamespace(
            rollout_token_ids=(token,),
            reasoning_positions=(0,),
            answer_positions=(),
            token_weights=(1.0,),
            advantage=reward,
        )

    scores = replay.scores(
        ("candidate-a", "candidate-b"),
        (
            (rollout("old-a", 11), credit(11, 1.0)),
            (rollout("old-b", 12), credit(12, -1.0)),
        ),
    )

    assert scores == (1.0, 1.0)
