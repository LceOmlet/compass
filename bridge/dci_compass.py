from __future__ import annotations

import json
import random
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from dspy.adapters.utils import serialize_for_json
from gepa.core.adapter import EvaluationBatch
from gepa.core.data_loader import DataLoader
from gepa.core.state import GEPAState
from gepa.proposer.reflective_mutation.admission import (
    AdmissionPlan,
    PostProposalAdmissionRequest,
)
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler

from bridge.b20_compass_reflection import (
    AdmissionReferenceSnapshot,
    CleanMiniAdmissionHook,
    SparseObservation,
    SparseObservationDspyAdapter,
    _finite_score,
)
from bridge.dci_agent_lite import (
    DciAgentLiteConfig,
    DciAgentLiteFinalAdapter,
    DciResult,
)


DataId = Hashable
DCI_COLD_START_EPOCHS = 1


@dataclass(frozen=True, slots=True)
class DciCompassConfig:
    """DCI-specific representation settings; GEPA remains the optimizer owner."""

    agent: DciAgentLiteConfig
    proposal_evidence_size: int = 3

    def __post_init__(self) -> None:
        if (
            isinstance(self.proposal_evidence_size, bool)
            or not isinstance(self.proposal_evidence_size, int)
            or self.proposal_evidence_size < 2
        ):
            raise TypeError("proposal_evidence_size must be an integer of at least 2")


@dataclass(frozen=True, slots=True)
class LogicalRollout:
    rollout_ref: str
    phase: str
    iteration: int
    parent_program_idx: int
    candidate: Mapping[str, str]
    data_id: DataId
    score: float
    output: Any
    trajectory: Mapping[str, Any]
    proposal_id: str | None


@dataclass(frozen=True, slots=True)
class DciSubproblem:
    subproblem_id: str
    definition: str
    created_iteration: int
    seed_data_id: DataId
    member_ids: tuple[DataId, ...]
    failure_ids: tuple[DataId, ...]
    success_ids: tuple[DataId, ...]
    corpus_snapshot_dir: Path
    artifact_dir: Path

    @property
    def instance_count(self) -> int:
        return len(self.member_ids)

    @property
    def failure_count(self) -> int:
        return len(self.failure_ids)

    @property
    def success_count(self) -> int:
        return len(self.success_ids)


@dataclass(frozen=True, slots=True)
class _DciBatchContext:
    iteration: int
    corpus: tuple[LogicalRollout, ...]
    catalog: tuple[DciSubproblem, ...]
    frontier_rollout_refs: Mapping[DataId, tuple[str, ...]]
    cold_start: bool = False


class DciReflectiveDataset(dict[str, list[dict[str, Any]]]):
    """Official DSPy reflective records plus exact DCI boundary data."""

    def __init__(
        self,
        records: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        seed_rollout: LogicalRollout,
    ) -> None:
        super().__init__(
            (name, [dict(item) for item in items])
            for name, items in records.items()
        )
        self.seed_rollout = seed_rollout
        self.dci_result: DciResult[DataId] | None = None
        self.subproblem_id: str | None = None
        self.member_ids: tuple[DataId, ...] = ()
        self.failure_ids: tuple[DataId, ...] = ()
        self.success_ids: tuple[DataId, ...] = ()
        self.selected_evidence_ids: tuple[DataId, ...] = ()
        self.skipped_no_dci_evidence = False


class DciSparseObservationDspyAdapter(SparseObservationDspyAdapter):
    """Adapt official DSPy rollouts and DCI finals without owning optimization."""

    _DCI_STATE_KEY = "compass_dci"
    _DCI_SCHEMA_VERSION = 2

    def __init__(
        self,
        *args: Any,
        proposal_loader: DataLoader,
        admission_loader: DataLoader,
        proposal_batch_sampler: EpochShuffledBatchSampler,
        dci_config: DciCompassConfig,
        dci_root: Path,
        dci_seed: int,
        perfect_score: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.proposal_loader = proposal_loader
        self.admission_loader = admission_loader
        self.proposal_batch_sampler = proposal_batch_sampler
        self.dci_config = dci_config
        self.dci_root = dci_root
        self.dci_seed = int(dci_seed)
        self.perfect_score = _finite_score(perfect_score)
        self.dci_boundary = DciAgentLiteFinalAdapter(dci_config.agent)

        admission_ids = tuple(admission_loader.all_ids())
        if len(admission_ids) != len(set(admission_ids)):
            raise RuntimeError("official admission loader IDs must be unique")
        self._token_by_id = {
            data_id: f"d{position:08d}"
            for position, data_id in enumerate(admission_ids)
        }
        self._id_by_token = {
            token: data_id for data_id, token in self._token_by_id.items()
        }
        proposal_ids = set(proposal_loader.all_ids())
        if not proposal_ids.issubset(self._token_by_id):
            raise RuntimeError("proposal IDs must belong to the admission universe")

        self._logical_rollouts: list[LogicalRollout] = []
        self._rollout_by_ref: dict[str, LogicalRollout] = {}
        self._rollout_ref_by_trajectory_identity: dict[int, str] = {}
        self._fact_rollout_ref: dict[tuple[int, DataId], str] = {}
        self._subproblems: list[DciSubproblem] = []
        self._active_batch_context: _DciBatchContext | None = None
        self.proposal_skipped_no_dci_evidence = 0

    @property
    def logical_rollouts(self) -> tuple[LogicalRollout, ...]:
        return tuple(self._logical_rollouts)

    @property
    def subproblems(self) -> tuple[DciSubproblem, ...]:
        return tuple(self._subproblems)

    def bind_proposal_batch_context(
        self,
        *,
        state: GEPAState,
        iteration: int,
        tasks: Sequence[Any],
    ) -> None:
        """Freeze existing owner state before the batch executes any seed rollout."""

        if any(len(tuple(task.minibatch_ids)) != 1 for task in tasks):
            raise RuntimeError("DCI requires exactly one seed instance per proposal task")

        frontier_refs: dict[DataId, tuple[str, ...]] = {}
        for data_id, owners in state.program_at_pareto_front_valset.items():
            refs: list[str] = []
            for owner in sorted(owners):
                ref = self._fact_rollout_ref.get((int(owner), data_id))
                if ref is None:
                    raise RuntimeError(
                        "DCI cannot resume from frontier evidence that predates its "
                        "append-only rollout corpus"
                    )
                refs.append(ref)
            if refs:
                frontier_refs[data_id] = tuple(refs)

        self._active_batch_context = _DciBatchContext(
            iteration=int(iteration),
            corpus=tuple(self._logical_rollouts),
            catalog=tuple(self._subproblems),
            frontier_rollout_refs=frontier_refs,
            cold_start=(
                self.proposal_batch_sampler.epoch < DCI_COLD_START_EPOCHS
            ),
        )

    def record_logical_rollouts(
        self,
        *,
        phase: str,
        iteration: int,
        parent_program_idx: int,
        candidate: Mapping[str, str],
        evaluation_ids: Sequence[DataId],
        evaluation: EvaluationBatch,
        proposal_id: str | None = None,
    ) -> None:
        ids = tuple(evaluation_ids)
        outputs = tuple(evaluation.outputs)
        scores = tuple(evaluation.scores)
        trajectories = tuple(evaluation.trajectories or ())
        if not (len(ids) == len(outputs) == len(scores) == len(trajectories)):
            raise RuntimeError("logical rollout vectors are not instance-aligned")

        for data_id, output, raw_score, trajectory in zip(
            ids,
            outputs,
            scores,
            trajectories,
            strict=True,
        ):
            if data_id not in self._token_by_id:
                raise RuntimeError("logical rollout ID is outside the admission universe")
            if not isinstance(trajectory, Mapping):
                raise RuntimeError("logical rollout trajectory must be a mapping")
            rollout_ref = f"r{len(self._logical_rollouts):012d}"
            record = LogicalRollout(
                rollout_ref=rollout_ref,
                phase=phase,
                iteration=int(iteration),
                parent_program_idx=int(parent_program_idx),
                candidate=dict(candidate),
                data_id=data_id,
                score=_finite_score(raw_score),
                output=output,
                trajectory=dict(trajectory),
                proposal_id=proposal_id,
            )
            self._logical_rollouts.append(record)
            self._rollout_by_ref[rollout_ref] = record
            self._rollout_ref_by_trajectory_identity[id(trajectory)] = rollout_ref

    def commit_program_observations(self, **kwargs: Any) -> None:
        super().commit_program_observations(**kwargs)
        program_idx = int(kwargs["program_idx"])
        ids = tuple(kwargs["evaluation_ids"])
        trajectories = tuple(kwargs["evaluation"].trajectories or ())
        committed = set(kwargs["committed_ids"])
        for data_id, trajectory in zip(ids, trajectories, strict=True):
            if data_id not in committed:
                continue
            rollout_ref = self._rollout_ref_by_trajectory_identity.get(id(trajectory))
            if rollout_ref is None:
                raise RuntimeError(
                    "committed observation has no completed logical rollout record"
                )
            self._fact_rollout_ref[(program_idx, data_id)] = rollout_ref

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> DciReflectiveDataset:
        trajectories = tuple(eval_batch.trajectories or ())
        if len(trajectories) != 1:
            raise RuntimeError("DCI proposal minibatch must contain one seed rollout")
        rollout_ref = self._rollout_ref_by_trajectory_identity.get(id(trajectories[0]))
        if rollout_ref is None:
            raise RuntimeError("seed rollout was not bound to its stable DataId")
        records = super().make_reflective_dataset(
            candidate,
            eval_batch,
            components_to_update,
        )
        return DciReflectiveDataset(
            records,
            seed_rollout=self._rollout_by_ref[rollout_ref],
        )

    def _write_corpus_snapshot(
        self,
        *,
        dataset: DciReflectiveDataset,
        context: _DciBatchContext,
    ) -> tuple[Path, Path, Mapping[str, DataId]]:
        token = self._token_by_id[dataset.seed_rollout.data_id]
        unique = uuid4().hex
        corpus_dir = (
            self.dci_root
            / "corpora"
            / f"i{context.iteration:06d}-{token}-{unique}"
        )
        artifact_dir = (
            self.dci_root
            / "runs"
            / f"i{context.iteration:06d}-{token}-{unique}"
        )
        corpus_dir.mkdir(parents=True, exist_ok=False)
        (corpus_dir / "selection").mkdir()

        frontier_refs = {
            ref
            for refs in context.frontier_rollout_refs.values()
            for ref in refs
        }
        rollout_documents_dir = corpus_dir / "rollouts"
        rollout_documents_dir.mkdir()
        payloads_by_token: dict[str, list[dict[str, Any]]] = {}
        for record in context.corpus:
            data_id_token = self._token_by_id[record.data_id]
            payloads_by_token.setdefault(data_id_token, []).append(
                self._rollout_payload(
                    record,
                    is_frontier=record.rollout_ref in frontier_refs,
                )
            )
        for data_id_token, payloads in payloads_by_token.items():
            (rollout_documents_dir / f"{data_id_token}.json").write_text(
                json.dumps(
                    payloads,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

        (corpus_dir / "seed.json").write_text(
            json.dumps(
                self._rollout_payload(dataset.seed_rollout, is_frontier=False),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (corpus_dir / "subproblems.json").write_text(
            json.dumps(
                [
                    {
                        "subproblem_id": item.subproblem_id,
                        "definition": item.definition,
                        "created_iteration": item.created_iteration,
                        "seed_data_id": self._token_by_id[item.seed_data_id],
                        "member_data_ids": [
                            self._token_by_id[data_id]
                            for data_id in item.member_ids
                        ],
                        "failure_data_ids": [
                            self._token_by_id[data_id]
                            for data_id in item.failure_ids
                        ],
                        "success_data_ids": [
                            self._token_by_id[data_id]
                            for data_id in item.success_ids
                        ],
                        "instance_count": item.instance_count,
                        "failure_count": item.failure_count,
                        "success_count": item.success_count,
                        "corpus_snapshot": item.corpus_snapshot_dir.name,
                    }
                    for item in context.catalog
                ],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        allowed_ids = set(context.frontier_rollout_refs)
        allowed_ids.add(dataset.seed_rollout.data_id)
        allowed_id_by_token = {
            self._token_by_id[data_id]: data_id
            for data_id in self.admission_loader.all_ids()
            if data_id in allowed_ids
        }
        (corpus_dir / "allowed_data_ids.json").write_text(
            json.dumps(
                list(allowed_id_by_token),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return corpus_dir, artifact_dir, allowed_id_by_token

    def _rollout_payload(
        self,
        record: LogicalRollout,
        *,
        is_frontier: bool,
    ) -> dict[str, Any]:
        return {
            "rollout_ref": record.rollout_ref,
            "data_id": self._token_by_id[record.data_id],
            "phase": record.phase,
            "iteration": record.iteration,
            "parent_program_idx": record.parent_program_idx,
            "proposal_id": record.proposal_id,
            "candidate": serialize_for_json(dict(record.candidate)),
            "score": record.score,
            "is_frontier": is_frontier,
            "example": serialize_for_json(record.trajectory.get("example")),
            "output": serialize_for_json(record.output),
            "trajectory": serialize_for_json(dict(record.trajectory)),
        }

    def _render_question(
        self,
        *,
        dataset: DciReflectiveDataset,
        context: _DciBatchContext,
        allowed_id_by_token: Mapping[str, DataId],
    ) -> str:
        historical_data_ids = {record.data_id for record in context.corpus}
        propose_parent_count = sum(
            record.phase == "propose_parent" for record in context.corpus
        )
        admit_candidate_count = sum(
            record.phase == "admit_candidate" for record in context.corpus
        )
        return (
            "Inspect the rollouts/ document directory, seed.json, optional "
            "subproblems.json, and "
            "allowed_data_ids.json in the current directory.\n\n"
            "Corpus guide: seed.json contains the current fresh seed rollout. "
            f"rollouts/ contains {len(historical_data_ids)} instance documents "
            f"holding all {len(context.corpus)} previously completed training "
            f"rollout records, including {propose_parent_count} proposal-parent "
            f"and {admit_candidate_count} admission-candidate records; held-out "
            "reporting data is absent. Each rollouts/<DataId-token>.json document "
            "contains every historical rollout for one task instance in stable "
            "rollout order, so one document can contain multiple rollout records. "
            f"allowed_data_ids.json contains all {len(allowed_id_by_token)} valid "
            "marker tokens for this interaction, including the seed; use exact "
            "tokens from that file. Count a subproblem instance by unique DataId, "
            "not by rollout rows or text hits. A matching instance has clean-frontier "
            "evidence facing the same concrete requirement or observable blocking "
            "step as the seed; it may be unresolved or a successful contrast.\n\n"
            f"The seed DataId token is {self._token_by_id[dataset.seed_rollout.data_id]!r} "
            "and is already an implicit member. Search the complete frozen corpus "
            "with several targeted `rg -l <pattern> rollouts/` queries, read the "
            "candidate instance documents, and run follow-up searches for missing "
            "angles. Do not concatenate or dump every rollout document into the "
            "model context. An instance remains eligible even when it belongs to "
            "a historical subproblem; overlapping discoveries are allowed. "
            "For every additional clean-frontier instance that belongs to this "
            "subproblem, create a direct marker file with "
            "`touch selection/<DataId-token>`. Do not mark an instance merely because "
            "you inspected it or used it as an unrelated counterexample. "
            "Finish with an ordinary free-text definition and useful reasoning for "
            "the new subproblem; no JSON or fixed response schema is required."
        )

    def _partition_selected_members(
        self,
        *,
        dataset: DciReflectiveDataset,
        result: DciResult[DataId],
        context: _DciBatchContext,
    ) -> None:
        seed_id = dataset.seed_rollout.data_id
        dataset.member_ids = tuple(dict.fromkeys((seed_id, *result.selected_ids)))
        failures: list[DataId] = []
        successes: list[DataId] = []
        for data_id in dataset.member_ids:
            rollout_refs = context.frontier_rollout_refs.get(data_id)
            if not rollout_refs:
                continue
            score = self._rollout_by_ref[rollout_refs[0]].score
            if score >= self.perfect_score:
                successes.append(data_id)
            else:
                failures.append(data_id)
        dataset.failure_ids = tuple(failures)
        dataset.success_ids = tuple(successes)

    def _select_evidence(
        self,
        *,
        dataset: DciReflectiveDataset,
        context: _DciBatchContext,
    ) -> tuple[tuple[str, DataId, LogicalRollout], ...]:
        seed_id = dataset.seed_rollout.data_id
        failures = [
            data_id for data_id in dataset.failure_ids if data_id != seed_id
        ]
        successes = [
            data_id for data_id in dataset.success_ids if data_id != seed_id
        ]
        rng = random.Random(
            f"dci-evidence:{self.dci_seed}:{context.iteration}:"
            f"{self._token_by_id[seed_id]}"
        )
        selected: list[tuple[str, DataId, LogicalRollout]] = []
        slots = self.dci_config.proposal_evidence_size - 1
        for _ in range(slots):
            if failures and successes:
                polarity = "failure" if rng.random() < 0.5 else "success"
            elif failures:
                polarity = "failure"
            elif successes:
                polarity = "success"
            else:
                break
            pool = failures if polarity == "failure" else successes
            position = rng.randrange(len(pool))
            data_id = pool.pop(position)
            rollout_ref = context.frontier_rollout_refs[data_id][0]
            selected.append(
                (polarity, data_id, self._rollout_by_ref[rollout_ref])
            )
        return tuple(selected)

    def _augment_reflective_dataset(
        self,
        *,
        dataset: DciReflectiveDataset,
        result: DciResult[DataId],
        selected: Sequence[tuple[str, DataId, LogicalRollout]],
    ) -> dict[str, list[dict[str, Any]]]:
        context_text = f"DCI subproblem:\n{result.definition}"

        augmented: dict[str, list[dict[str, Any]]] = {}
        for component, items in dataset.items():
            component_items = [dict(item) for item in items]
            for item in component_items:
                feedback = str(item.get("Feedback", ""))
                prefix = f"{feedback}\n\n" if feedback else ""
                item["Feedback"] = f"{prefix}{context_text}"
            for polarity, data_id, record in selected:
                component_items.append(
                    {
                        "Inputs": {
                            "DCI evidence DataId": self._token_by_id[data_id],
                            "Historical input": serialize_for_json(
                                record.trajectory.get("example")
                            ),
                        },
                        "Generated Outputs": serialize_for_json(record.output),
                        "Feedback": (
                            f"{polarity} contrast for: {result.definition}. "
                            f"Observed reward: {record.score}."
                        ),
                    }
                )
            augmented[component] = component_items
        return augmented

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        if not isinstance(reflective_dataset, DciReflectiveDataset):
            raise RuntimeError("DCI proposal lost its exact reflective dataset binding")
        context = self._active_batch_context
        if context is None or context.iteration != reflective_dataset.seed_rollout.iteration:
            raise RuntimeError("DCI proposal has no matching frozen batch context")
        if context.cold_start:
            return {}

        corpus_dir, artifact_dir, allowed_id_by_token = self._write_corpus_snapshot(
            dataset=reflective_dataset,
            context=context,
        )
        result = self.dci_boundary.run(
            rendered_question=self._render_question(
                dataset=reflective_dataset,
                context=context,
                allowed_id_by_token=allowed_id_by_token,
            ),
            corpus_cwd=corpus_dir,
            output_dir=artifact_dir,
            id_by_token=allowed_id_by_token,
        )
        reflective_dataset.dci_result = result
        self._partition_selected_members(
            dataset=reflective_dataset,
            result=result,
            context=context,
        )
        selected = self._select_evidence(
            dataset=reflective_dataset,
            context=context,
        )
        reflective_dataset.selected_evidence_ids = tuple(
            data_id for _, data_id, _ in selected
        )
        if not selected:
            reflective_dataset.skipped_no_dci_evidence = True
            return {}

        return dict(
            super().propose_new_texts(
                candidate,
                self._augment_reflective_dataset(
                    dataset=reflective_dataset,
                    result=result,
                    selected=selected,
                ),
                components_to_update,
            )
        )

    def propose_new_texts_batch(
        self,
        jobs: list[
            tuple[
                dict[str, str],
                Mapping[str, Sequence[Mapping[str, Any]]],
                list[str],
            ]
        ],
    ) -> list[dict[str, str] | None]:
        results = super().propose_new_texts_batch(jobs)
        for _, raw_dataset, _ in jobs:
            if not isinstance(raw_dataset, DciReflectiveDataset):
                raise RuntimeError("DCI batch lost its reflective dataset binding")
            if raw_dataset.skipped_no_dci_evidence:
                self.proposal_skipped_no_dci_evidence += 1
            dci_result = raw_dataset.dci_result
            if dci_result is None:
                continue
            subproblem = DciSubproblem(
                subproblem_id=f"sp-{len(self._subproblems):06d}",
                definition=dci_result.definition,
                created_iteration=raw_dataset.seed_rollout.iteration,
                seed_data_id=raw_dataset.seed_rollout.data_id,
                member_ids=raw_dataset.member_ids,
                failure_ids=raw_dataset.failure_ids,
                success_ids=raw_dataset.success_ids,
                corpus_snapshot_dir=dci_result.corpus_dir,
                artifact_dir=dci_result.artifact_dir,
            )
            self._subproblems.append(subproblem)
            raw_dataset.subproblem_id = subproblem.subproblem_id
        return results

    def get_adapter_state(self) -> dict[str, Any]:
        state = super().get_adapter_state()
        state[self._DCI_STATE_KEY] = {
            "schema_version": self._DCI_SCHEMA_VERSION,
            "logical_rollouts": list(self._logical_rollouts),
            "fact_rollout_ref": dict(self._fact_rollout_ref),
            "subproblems": list(self._subproblems),
            "proposal_skipped_no_dci_evidence": (
                self.proposal_skipped_no_dci_evidence
            ),
        }
        return state

    def set_adapter_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("adapter state must be a mapping")
        unexpected = set(state).difference({self._STATE_KEY, self._DCI_STATE_KEY})
        if unexpected:
            raise RuntimeError(f"unrecognized DCI adapter state: {sorted(unexpected)}")
        super().set_adapter_state(
            {self._STATE_KEY: state.get(self._STATE_KEY)}
            if self._STATE_KEY in state
            else {}
        )
        payload = state.get(self._DCI_STATE_KEY)
        if payload is None:
            self._logical_rollouts = []
            self._rollout_by_ref = {}
            self._fact_rollout_ref = {}
            self._subproblems = []
            self.proposal_skipped_no_dci_evidence = 0
            return
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != self._DCI_SCHEMA_VERSION
        ):
            raise RuntimeError("unsupported COMPASS DCI adapter-state schema")
        logical_rollouts = payload.get("logical_rollouts")
        fact_rollout_ref = payload.get("fact_rollout_ref")
        subproblems = payload.get("subproblems")
        skipped = payload.get("proposal_skipped_no_dci_evidence")
        if (
            not isinstance(logical_rollouts, list)
            or any(not isinstance(item, LogicalRollout) for item in logical_rollouts)
            or not isinstance(fact_rollout_ref, Mapping)
            or not isinstance(subproblems, list)
            or any(not isinstance(item, DciSubproblem) for item in subproblems)
            or isinstance(skipped, bool)
            or not isinstance(skipped, int)
            or skipped < 0
        ):
            raise RuntimeError("COMPASS DCI adapter state is malformed")
        self._logical_rollouts = list(logical_rollouts)
        self._rollout_by_ref = {
            item.rollout_ref: item for item in self._logical_rollouts
        }
        self._rollout_ref_by_trajectory_identity = {}
        self._fact_rollout_ref = dict(fact_rollout_ref)
        self._subproblems = list(subproblems)
        self.proposal_skipped_no_dci_evidence = skipped
        self._active_batch_context = None


class DciAdmissionHook(CleanMiniAdmissionHook):
    """Return exact DCI scope through the existing admission/reference owner."""

    def __init__(self, *args: Any, admission_size: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if (
            isinstance(admission_size, bool)
            or not isinstance(admission_size, int)
            or admission_size <= 0
        ):
            raise TypeError("admission_size must be a positive integer")
        self.admission_size = admission_size

    def _select_admission_ids(
        self,
        *,
        dataset: DciReflectiveDataset,
        admission_set: DataLoader,
        excluded: set[DataId],
    ) -> tuple[DataId, ...]:
        universe = tuple(admission_set.all_ids())
        universe_set = set(universe)
        unknown_members = set(dataset.member_ids).difference(universe_set)
        if unknown_members:
            raise RuntimeError("DCI subproblem member is outside admission universe")

        eligible_members = tuple(
            dict.fromkeys(
                data_id
                for data_id in dataset.member_ids
                if data_id not in excluded
            )
        )
        if len(eligible_members) > self.admission_size:
            selected = list(self.rng.sample(eligible_members, self.admission_size))
        else:
            selected = list(eligible_members)

        deficit = self.admission_size - len(selected)
        if deficit:
            selected_set = set(selected)
            fill_pool = tuple(
                data_id
                for data_id in universe
                if data_id not in excluded and data_id not in selected_set
            )
            if len(fill_pool) < deficit:
                raise ValueError(
                    "Cannot sample a full DCI admission minibatch after "
                    "applying recursive proposal exclusions."
                )
            selected.extend(self.rng.sample(fill_pool, deficit))
        return tuple(selected)

    def bind_proposal_batch_context(
        self,
        *,
        state: GEPAState,
        iteration: int,
        tasks: Sequence[Any],
    ) -> None:
        """Freeze references before this window executes any parent rollout."""

        del tasks
        self._reference_snapshot_iteration = int(iteration)
        self._reference_snapshot = self.capture_reference_snapshot(state)

    def prepare_after_proposals(
        self,
        *,
        state: GEPAState,
        admission_set: DataLoader,
        requests: Sequence[PostProposalAdmissionRequest],
    ) -> Sequence[AdmissionPlan | None]:
        snapshot = getattr(self, "_reference_snapshot", None)
        snapshot_iteration = getattr(
            self,
            "_reference_snapshot_iteration",
            None,
        )
        if (
            not isinstance(snapshot, AdmissionReferenceSnapshot)
            or snapshot_iteration != state.i + 1
        ):
            raise RuntimeError("DCI admission has no matching pre-batch reference snapshot")

        prepared: list[
            tuple[
                PostProposalAdmissionRequest,
                DciReflectiveDataset,
                tuple[DataId, ...],
                tuple[Any, ...],
                tuple[int, ...],
                tuple[SparseObservation | None, ...],
            ]
            | None
        ] = []
        for request in requests:
            dataset = request.reflective_dataset
            if not isinstance(dataset, DciReflectiveDataset):
                raise RuntimeError("DCI admission lost its proposal result binding")
            if dataset.dci_result is None or dataset.subproblem_id is None:
                raise RuntimeError("DCI admission request has no canonical subproblem")

            excluded = set(
                state.get_prospective_frontier_ineligible_ids(
                    request.parent_program_idx,
                    request.birth_propose_ids,
                )
            )
            ids = self._select_admission_ids(
                dataset=dataset,
                admission_set=admission_set,
                excluded=excluded,
            )
            batch = tuple(admission_set.fetch(ids))
            reference_program_indices = self.bind_snapshot_reference_program_indices(
                snapshot=snapshot,
                instance_ids=ids,
                sampled_parent_idx=request.parent_program_idx,
            )
            raw_facts = tuple(
                snapshot.facts.get((program_idx, instance_id))
                for program_idx, instance_id in zip(
                    reference_program_indices,
                    ids,
                    strict=True,
                )
            )
            prepared.append(
                (
                    request,
                    dataset,
                    ids,
                    batch,
                    reference_program_indices,
                    raw_facts,
                )
            )

        for item in prepared:
            if item is None:
                continue
            _, _, ids, batch, reference_program_indices, raw_facts = item
            if any(fact is None for fact in raw_facts):
                self._evaluate_missing_reference_groups(
                    state=state,
                    ids=ids,
                    batch=batch,
                    reference_program_indices=reference_program_indices,
                )

        plans: list[AdmissionPlan | None] = []
        for item in prepared:
            if item is None:
                plans.append(None)
                continue
            request, dataset, ids, batch, reference_program_indices, raw_facts = item
            resolved_facts = tuple(
                snapshot_fact
                if snapshot_fact is not None
                else self.adapter.get_program_observation(program_idx, instance_id)
                for snapshot_fact, program_idx, instance_id in zip(
                    raw_facts,
                    reference_program_indices,
                    ids,
                    strict=True,
                )
            )
            if any(fact is None for fact in resolved_facts):
                raise RuntimeError("DCI admission reference fact is unavailable")
            facts = tuple(fact for fact in resolved_facts if fact is not None)
            plans.append(
                self.prepare_with_bound_references(
                    state=state,
                    parent_program_idx=request.parent_program_idx,
                    parent_candidate=dict(request.parent_candidate),
                    components_to_update=request.components_to_update,
                    reflective_dataset=dataset,
                    propose_evaluation=request.propose_evaluation,
                    admission_ids=ids,
                    admission_batch=batch,
                    reference_program_indices=reference_program_indices,
                    reference_facts=facts,
                )
            )
        return plans


__all__ = [
    "DciAdmissionHook",
    "DciCompassConfig",
    "DciReflectiveDataset",
    "DciSparseObservationDspyAdapter",
    "DciSubproblem",
    "LogicalRollout",
]
