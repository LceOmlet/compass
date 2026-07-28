from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

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
    assert provider.take_calls == 1
    assert provider.discard_calls == 1
    assert any(
        "candidate_index=0" in message
        and "dependency_unavailable=True" in message
        and "candidate_skipped=True" in message
        for message in logger.messages
    )


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

