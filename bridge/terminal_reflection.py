from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Protocol

from gepa.proposer.reflective_mutation.reflection_lm import ReflectionProposal
from gepa.strategies.instruction_proposal import InstructionProposalSignature


class TerminalAnalysisUnavailableError(RuntimeError):
    """The old-rollout terminal comparison cannot be computed exactly."""


class PreparedTerminalAnalysis(Protocol):
    """Old-reference analysis prepared for one official GEPA admission task."""

    def is_known_candidate(self, candidate: Mapping[str, str]) -> bool: ...

    def teacher_forcing_scores(
        self,
        *,
        component: str,
        candidate_programs: tuple[Mapping[str, str], ...],
    ) -> tuple[float, ...]: ...

    def dependency_distance(
        self,
        *,
        component: str,
        candidate_program: Mapping[str, str],
    ) -> float: ...


class TerminalAnalysisProvider(Protocol):
    """Return the analysis bound to the current reflection/admission job."""

    def take(
        self,
        *,
        parent_candidate: Mapping[str, str],
        component: str,
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> PreparedTerminalAnalysis: ...

    def update_candidate_pool(
        self,
        candidates: Sequence[Mapping[str, str]],
    ) -> None: ...

    def discard(
        self,
        *,
        parent_candidate: Mapping[str, str],
        component: str,
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> None: ...


class CandidateCompletionFn(Protocol):
    """Generate independent completions for one official reflection prompt."""

    def __call__(
        self,
        prompt: str | list[dict[str, Any]],
        *,
        n: int,
    ) -> Sequence[str]: ...


@dataclass(frozen=True, slots=True)
class TerminalSelection:
    original_index: int
    candidate_text: str
    raw_completion: str
    teacher_forcing_score: float
    dependency_distance: float
    duplicate_count: int


class TerminalLikelihoodReflectionLM:
    """Official reflection rendering plus project-specific terminal selection.

    The official GEPA proposer remains the owner of the proposal lifecycle.  This
    implementation only replaces its reflection edge: it obtains ``n_candidates``
    full completions for the exact official prompt, overlays each onto the sampled
    parent, scores the full children on fixed admission references, applies the
    old-reference dependency gate in descending
    teacher-forcing order, and returns one full instruction to GEPA.
    """

    def __init__(
        self,
        *,
        complete: CandidateCompletionFn,
        analysis_provider: TerminalAnalysisProvider,
        n_candidates: Integral,
        epsilon_dep: Real,
        logger: Any | None = None,
    ) -> None:
        if isinstance(n_candidates, bool) or not isinstance(n_candidates, Integral):
            raise TypeError("n_candidates must be an integer")
        if int(n_candidates) <= 0:
            raise ValueError("n_candidates must be positive")
        if isinstance(epsilon_dep, bool) or not isinstance(epsilon_dep, Real):
            raise TypeError("epsilon_dep must be numeric")
        epsilon = float(epsilon_dep)
        if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon_dep must lie in [0, 1]")
        self._complete = complete
        self._analysis_provider = analysis_provider
        self._n_candidates = int(n_candidates)
        self._epsilon_dep = epsilon
        self._logger = logger

    def _log(self, message: str) -> None:
        if self._logger is not None:
            self._logger.log(message)

    def set_logger(self, logger: Any) -> None:
        """Use the exact logger shared by the official GEPA engine."""

        if logger is None or not callable(getattr(logger, "log", None)):
            raise TypeError("logger must expose log")
        if self._logger is not None and self._logger is not logger:
            raise RuntimeError("terminal reflection logger is already bound")
        self._logger = logger

    @staticmethod
    def _complete_program(
        parent: Mapping[str, str],
        *,
        component: str,
        text: str,
    ) -> dict[str, str]:
        if component not in parent:
            raise KeyError(f"component {component!r} is absent from the parent candidate")
        candidate = dict(parent)
        candidate[component] = text
        return candidate

    def _select(
        self,
        *,
        parent_candidate: Mapping[str, str],
        component: str,
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        raw_completions: Sequence[str],
    ) -> TerminalSelection | None:
        if len(raw_completions) != self._n_candidates:
            raise RuntimeError(
                "the proposal LM returned "
                f"{len(raw_completions)} completions for n={self._n_candidates}"
            )
        extracted: list[
            tuple[int, str, str, dict[str, str], tuple[tuple[str, str], ...]]
        ] = []
        for index, raw in enumerate(raw_completions):
            if not isinstance(raw, str):
                raise TypeError("every reflection completion must be text")
            text = InstructionProposalSignature.output_extractor(raw)["new_instruction"]
            if not isinstance(text, str):
                raise TypeError("the official extractor did not return text")
            program = self._complete_program(
                parent_candidate,
                component=component,
                text=text,
            )
            candidate_key = tuple(sorted(program.items()))
            extracted.append((index, text, raw, program, candidate_key))

        multiplicity = Counter(candidate_key for *_, candidate_key in extracted)
        analysis = self._analysis_provider.take(
            parent_candidate=parent_candidate,
            component=component,
            reflective_dataset=reflective_dataset,
        )
        unique: list[
            tuple[int, str, str, dict[str, str], tuple[tuple[str, str], ...]]
        ] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for index, text, raw, program, candidate_key in extracted:
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            if analysis.is_known_candidate(program):
                continue
            unique.append((index, text, raw, program, candidate_key))
        if not unique:
            return None

        scores = analysis.teacher_forcing_scores(
            component=component,
            candidate_programs=tuple(program for *_, program, _key in unique),
        )
        if len(scores) != len(unique):
            raise RuntimeError("teacher-forcing scores are misaligned with candidates")
        numeric_scores = tuple(float(value) for value in scores)
        if any(not math.isfinite(value) for value in numeric_scores):
            raise RuntimeError("teacher-forcing scores must be finite")

        for position, (original_index, _text, _raw, _program, candidate_key) in enumerate(unique):
            self._log(
                "Terminal teacher_forcing candidate_index="
                f"{original_index}, score={numeric_scores[position]:.17g}, "
                f"duplicate_count={multiplicity[candidate_key]}"
            )

        ranked = sorted(
            range(len(unique)),
            key=lambda position: (-numeric_scores[position], unique[position][0]),
        )
        for position in ranked:
            original_index, text, raw, program, candidate_key = unique[position]
            try:
                distance = float(
                    analysis.dependency_distance(
                        component=component,
                        candidate_program=program,
                    )
                )
            except TerminalAnalysisUnavailableError as error:
                self._log(
                    "Terminal candidate_index="
                    f"{original_index}, dependency_unavailable=True, "
                    f"candidate_skipped=True, reason={error}"
                )
                continue
            if not math.isfinite(distance) or not 0.0 <= distance <= 1.0:
                raise RuntimeError("dependency distance must lie in [0, 1]")
            passed = distance <= self._epsilon_dep
            self._log(
                "Terminal candidate_index="
                f"{original_index}, teacher_forcing_score={numeric_scores[position]:.17g}, "
                f"dependency_distance={distance:.17g}, "
                f"epsilon_dep={self._epsilon_dep:.17g}, gate_passed={passed}"
            )
            if passed:
                self._log(
                    "Terminal selected candidate_index="
                    f"{original_index}, teacher_forcing_score={numeric_scores[position]:.17g}, "
                    f"dependency_distance={distance:.17g}"
                )
                return TerminalSelection(
                    original_index=original_index,
                    candidate_text=text,
                    raw_completion=raw,
                    teacher_forcing_score=numeric_scores[position],
                    dependency_distance=distance,
                    duplicate_count=multiplicity[candidate_key],
                )
        self._log("Terminal selection found no candidate inside the dependency gate")
        return None

    def reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[ReflectionProposal, "TerminalLikelihoodReflectionLM"]:
        if len(components_to_update) != 1:
            raise RuntimeError(
                "terminal-likelihood mode requires the configured official single-component mutation"
            )
        component = components_to_update[0]
        records = reflective_dataset.get(component)
        if not records:
            self._analysis_provider.discard(
                parent_candidate=candidate,
                component=component,
                reflective_dataset=reflective_dataset,
            )
            return ReflectionProposal(new_texts={}), self
        prompt = InstructionProposalSignature.prompt_renderer(
            {
                "current_instruction_doc": candidate[component],
                "dataset_with_feedback": records,
                "prompt_template": None,
            }
        )
        try:
            raw = tuple(self._complete(prompt, n=self._n_candidates))
            selected = self._select(
                parent_candidate=candidate,
                component=component,
                reflective_dataset=reflective_dataset,
                raw_completions=raw,
            )
        except BaseException:
            # GEPA owns proposal retry.  Keep the immutable parent analysis
            # bound to this exact reflection job so its official per-task
            # fallback can make another attempt without rebuilding the old
            # rollout analysis.
            raise
        self._analysis_provider.discard(
            parent_candidate=candidate,
            component=component,
            reflective_dataset=reflective_dataset,
        )
        if selected is None:
            return ReflectionProposal(new_texts={}, prompts={component: prompt}), self
        proposal = ReflectionProposal(
            new_texts={component: selected.candidate_text},
            prompts={component: prompt},
            raw_lm_outputs={component: selected.raw_completion},
            metadata={
                "terminal_candidate_index": selected.original_index,
                "teacher_forcing_score": selected.teacher_forcing_score,
                "dependency_distance": selected.dependency_distance,
                "proposal_duplicate_count": selected.duplicate_count,
            },
        )
        return proposal, self

    def __call__(
        self,
        *,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Implement the official DSPy adapter's public ``ProposalFn`` hook."""

        proposal, _ = self.reflect(
            candidate=candidate,
            reflective_dataset=reflective_dataset,
            components_to_update=components_to_update,
        )
        return proposal.new_texts

    def update_candidate_pool(
        self,
        candidates: Sequence[Mapping[str, str]],
    ) -> None:
        """Receive the official pool snapshot used for pre-execution deduplication."""

        self._analysis_provider.update_candidate_pool(candidates)
