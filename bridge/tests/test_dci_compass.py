from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gepa.core.adapter import EvaluationBatch
from gepa.core.data_loader import ListDataLoader
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.proposer.reflective_mutation.admission import (
    PostProposalAdmissionRequest,
)
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler

from bridge.b20_compass_reflection import (
    AdmissionReferenceSnapshot,
    SparseObservation,
    SparseObservationDspyAdapter,
)
from bridge.dci_agent_lite import DciAgentLiteConfig, DciResult
from bridge.dci_compass import (
    DciAdmissionHook,
    DciCompassConfig,
    DciReflectiveDataset,
    DciSparseObservationDspyAdapter,
    LogicalRollout,
    _DciBatchContext,
)


def _config(tmp_path: Path) -> DciCompassConfig:
    return DciCompassConfig(
        agent=DciAgentLiteConfig(
            runner_command=("dci-agent-lite",),
            package_dir=tmp_path / "package",
            agent_dir=tmp_path / "agent",
            provider="openai",
            model="model",
            system_prompt_file=tmp_path / "system.txt",
        )
    )


def _adapter(tmp_path: Path) -> DciSparseObservationDspyAdapter:
    data = [{"q": index} for index in range(6)]
    return DciSparseObservationDspyAdapter(
        student_module=SimpleNamespace(),
        metric_fn=lambda *_args, **_kwargs: 0.0,
        feedback_map={},
        proposal_loader=ListDataLoader(data[:2]),
        admission_loader=ListDataLoader(data),
        proposal_batch_sampler=EpochShuffledBatchSampler(
            minibatch_size=1,
            rng=random.Random(0),
        ),
        dci_config=_config(tmp_path),
        dci_root=tmp_path / "dci",
        dci_seed=7,
        perfect_score=1.0,
    )


def _rollout(
    ref: str,
    data_id: int,
    *,
    iteration: int = 1,
    score: float = 0.0,
) -> LogicalRollout:
    return LogicalRollout(
        rollout_ref=ref,
        phase="propose_parent",
        iteration=iteration,
        parent_program_idx=0,
        candidate={"prompt": "seed"},
        data_id=data_id,
        score=score,
        output={"answer": data_id},
        trajectory={"example": {"q": data_id}},
        proposal_id=None,
    )


def _result(
    tmp_path: Path,
    name: str,
    *,
    definition: str,
    selected_ids: tuple[int, ...],
) -> DciResult[int]:
    return DciResult(
        definition=definition,
        selected_ids=selected_ids,
        corpus_dir=tmp_path / f"corpus-{name}",
        artifact_dir=tmp_path / f"artifacts-{name}",
    )


def _admission_dataset(
    member_ids: tuple[int, ...],
) -> DciReflectiveDataset:
    dataset = DciReflectiveDataset(
        {"prompt": [{"Feedback": "seed"}]},
        seed_rollout=_rollout("seed", 0),
    )
    dataset.dci_result = DciResult(
        definition="terminal check",
        selected_ids=tuple(data_id for data_id in member_ids if data_id != 0),
        corpus_dir=Path("corpus"),
        artifact_dir=Path("artifacts"),
    )
    dataset.member_ids = member_ids
    dataset.failure_ids = tuple(data_id for data_id in member_ids if data_id != 0)
    dataset.success_ids = ()
    dataset.subproblem_id = "sp-000000"
    return dataset


def _admission_request(
    dataset: DciReflectiveDataset,
) -> PostProposalAdmissionRequest:
    return PostProposalAdmissionRequest(
        parent_program_idx=4,
        parent_candidate={"prompt": "parent"},
        proposed_candidate={"prompt": "child"},
        components_to_update=("prompt",),
        reflective_dataset=dataset,
        propose_evaluation=EvaluationBatch(
            outputs=["seed"],
            scores=[0.0],
            trajectories=[{"example": {"q": 0}}],
        ),
        birth_propose_ids=(0,),
        reflection_metadata={},
    )


def _admission_hook(
    tmp_path: Path,
    *,
    seed: int = 0,
    admission_size: int = 3,
) -> DciAdmissionHook:
    return DciAdmissionHook(
        adapter=MagicMock(spec=SparseObservationDspyAdapter),
        rng=random.Random(seed),
        run_dir=tmp_path,
        logger=MagicMock(),
        admission_size=admission_size,
    )


def test_corpus_snapshot_uses_lossless_instance_documents(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    older = _rollout("r-older", 2, score=0.25)
    newer = LogicalRollout(
        rollout_ref="r-newer",
        phase="admit_candidate",
        iteration=2,
        parent_program_idx=1,
        candidate={"prompt": "candidate\nwith detail"},
        data_id=2,
        score=1.0,
        output={"answer": "multi\nline"},
        trajectory={"example": {"q": 2}, "trace": ["first", "second"]},
        proposal_id="proposal-2",
    )
    other = _rollout("r-other", 3, score=0.5)
    seed = _rollout("r-seed", 0, iteration=3)
    context = _DciBatchContext(
        iteration=3,
        corpus=(older, newer, other),
        catalog=(),
        frontier_rollout_refs={
            2: (newer.rollout_ref,),
            3: (other.rollout_ref,),
        },
    )
    dataset = DciReflectiveDataset(
        {"prompt": [{"Feedback": "seed"}]},
        seed_rollout=seed,
    )

    corpus_dir, _, allowed_id_by_token = adapter._write_corpus_snapshot(
        dataset=dataset,
        context=context,
    )

    rollout_documents = corpus_dir / "rollouts"
    assert sorted(path.name for path in rollout_documents.iterdir()) == [
        "d00000002.json",
        "d00000003.json",
    ]
    assert not (corpus_dir / "rollouts.jsonl").exists()
    assert json.loads(
        (rollout_documents / "d00000002.json").read_text(encoding="utf-8")
    ) == [
        adapter._rollout_payload(older, is_frontier=False),
        adapter._rollout_payload(newer, is_frontier=True),
    ]
    assert json.loads(
        (rollout_documents / "d00000003.json").read_text(encoding="utf-8")
    ) == [adapter._rollout_payload(other, is_frontier=True)]
    assert tuple(allowed_id_by_token.values()) == (0, 2, 3)

    question = adapter._render_question(
        dataset=dataset,
        context=context,
        allowed_id_by_token=allowed_id_by_token,
    )
    assert "rg -l <pattern> rollouts/" in question
    assert "all 3 previously completed training rollout records" in question
    assert "2 instance documents" in question
    assert "overlapping discoveries are allowed" in question
    assert "Create one new subproblem" not in question


def test_dci_system_prompt_uses_targeted_official_corpus_interaction() -> None:
    prompt = (
        Path(__file__).parents[1] / "prompts" / "dci_subproblem_free_text.txt"
    ).read_text(encoding="utf-8")

    assert "rg -l <pattern> rollouts/" in prompt
    assert "read only candidate" in prompt
    assert "whole corpus into the model context" in prompt
    assert "prior membership never makes an instance ineligible" in prompt
    assert "always discovers one new subproblem" not in prompt


def test_rollout_sink_round_trips_separately_from_maximum_facts(tmp_path: Path):
    adapter = _adapter(tmp_path)
    trajectory = {"example": {"q": 0}}
    evaluation = EvaluationBatch(
        outputs=[{"answer": 0}],
        scores=[0.25],
        trajectories=[trajectory],
    )
    adapter.record_logical_rollouts(
        phase="propose_parent",
        iteration=1,
        parent_program_idx=0,
        candidate={"prompt": "seed"},
        evaluation_ids=(0,),
        evaluation=evaluation,
    )
    adapter.commit_program_observations(
        program_idx=0,
        evaluation_ids=(0,),
        evaluation=evaluation,
        committed_ids=(0,),
    )

    restored = _adapter(tmp_path / "restored")
    restored.set_adapter_state(adapter.get_adapter_state())

    assert len(restored.logical_rollouts) == 1
    assert restored.logical_rollouts[0].data_id == 0
    assert restored.get_program_observation(0, 0) is not None


def test_dci_batch_appends_every_discovery_in_canonical_order_without_merge(
    monkeypatch,
    tmp_path: Path,
):
    adapter = _adapter(tmp_path)
    first = DciReflectiveDataset(
        {"prompt": [{"Feedback": "seed"}]},
        seed_rollout=_rollout("r0", 0),
    )
    second = DciReflectiveDataset(
        {"prompt": [{"Feedback": "seed"}]},
        seed_rollout=_rollout("r1", 1),
    )
    for name, dataset in (("first", first), ("second", second)):
        dataset.dci_result = _result(
            tmp_path,
            name,
            definition="Same   terminal check",
            selected_ids=(2,),
        )
        dataset.member_ids = (dataset.seed_rollout.data_id, 2)
        dataset.failure_ids = (2,)
        dataset.success_ids = ()
    calls = []

    def official_batch(self, jobs):
        calls.append(jobs)
        return [{"prompt": "a"}, {"prompt": "b"}]

    monkeypatch.setattr(
        SparseObservationDspyAdapter,
        "propose_new_texts_batch",
        official_batch,
    )
    jobs = [
        ({"prompt": "seed"}, first, ["prompt"]),
        ({"prompt": "seed"}, second, ["prompt"]),
    ]

    assert adapter.propose_new_texts_batch(jobs) == [
        {"prompt": "a"},
        {"prompt": "b"},
    ]
    assert len(calls) == 1
    assert first.subproblem_id == "sp-000000"
    assert second.subproblem_id == "sp-000001"
    assert len(adapter.subproblems) == 2
    assert adapter.subproblems[0].definition == adapter.subproblems[1].definition
    assert adapter.subproblems[0].member_ids == (0, 2)
    assert adapter.subproblems[0].instance_count == 2
    assert adapter.subproblems[0].failure_count == 1
    assert adapter.subproblems[0].corpus_snapshot_dir.name == "corpus-first"


def test_dci_batch_context_defaults_to_non_cold_start() -> None:
    context = _DciBatchContext(
        iteration=1,
        corpus=(),
        catalog=(),
        frontier_rollout_refs={},
    )

    assert context.cold_start is False


def test_dci_cold_start_follows_the_official_proposal_sampler_epoch(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    state = MagicMock()
    state.program_at_pareto_front_valset = {}
    task = SimpleNamespace(minibatch_ids=(0,))

    adapter.proposal_batch_sampler.epoch = 0
    adapter.bind_proposal_batch_context(
        state=state,
        iteration=1,
        tasks=(task,),
    )
    assert adapter._active_batch_context is not None
    assert adapter._active_batch_context.cold_start is True

    adapter.proposal_batch_sampler.epoch = 1
    adapter.bind_proposal_batch_context(
        state=state,
        iteration=2,
        tasks=(task,),
    )
    assert adapter._active_batch_context is not None
    assert adapter._active_batch_context.cold_start is False


def test_cold_start_skips_dci_and_does_not_count_no_evidence(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    seed = _rollout("r-cold-start", 0, iteration=1)
    adapter._logical_rollouts = [seed]
    adapter._rollout_by_ref = {seed.rollout_ref: seed}
    adapter._active_batch_context = _DciBatchContext(
        iteration=1,
        corpus=(),
        catalog=(),
        frontier_rollout_refs={},
        cold_start=True,
    )
    dataset = DciReflectiveDataset(
        {"prompt": [{"Feedback": "seed feedback"}]},
        seed_rollout=seed,
    )
    adapter.dci_boundary = MagicMock()
    job = ({"prompt": "seed"}, dataset, ["prompt"])

    assert adapter.propose_new_texts(*job) == {}
    assert adapter.propose_new_texts_batch([job]) == [{}]
    adapter.dci_boundary.run.assert_not_called()
    assert dataset.dci_result is None
    assert dataset.skipped_no_dci_evidence is False
    assert adapter.proposal_skipped_no_dci_evidence == 0
    assert adapter.subproblems == ()


def test_dci_admission_samples_three_members_without_replacement(
    tmp_path: Path,
) -> None:
    seed = 19
    hook = _admission_hook(tmp_path, seed=seed)
    hook.prepare_with_bound_references = MagicMock(return_value="plan")
    facts = {
        (7, data_id): SparseObservation(
            data_id / 10,
            f"old-{data_id}",
            {"example": {"q": data_id}},
        )
        for data_id in range(1, 6)
    }
    hook._reference_snapshot = AdmissionReferenceSnapshot(
        owners_by_instance={data_id: (7,) for data_id in range(1, 6)},
        frontier_counts={7: 5},
        evaluation_counts={7: 5},
        facts=facts,
    )
    hook._reference_snapshot_iteration = 12
    dataset = _admission_dataset((0, 1, 2, 3, 4, 5))
    metadata_before = (
        dataset.member_ids,
        dataset.failure_ids,
        dataset.success_ids,
        dataset.dci_result,
    )
    state = MagicMock()
    state.i = 11
    state.get_prospective_frontier_ineligible_ids.return_value = (0,)
    loader = ListDataLoader([{"q": index} for index in range(7)])
    expected_ids = tuple(random.Random(seed).sample((1, 2, 3, 4, 5), 3))

    plans = hook.prepare_after_proposals(
        state=state,
        admission_set=loader,
        requests=[_admission_request(dataset)],
    )

    assert plans == ["plan"]
    kwargs = hook.prepare_with_bound_references.call_args.kwargs
    assert kwargs["admission_ids"] == expected_ids
    assert len(set(kwargs["admission_ids"])) == 3
    assert kwargs["reference_facts"] == tuple(
        facts[(7, data_id)] for data_id in expected_ids
    )
    assert (
        dataset.member_ids,
        dataset.failure_ids,
        dataset.success_ids,
        dataset.dci_result,
    ) == metadata_before


def test_dci_admission_keeps_members_then_globally_fills_to_three(
    tmp_path: Path,
) -> None:
    seed = 23
    hook = _admission_hook(tmp_path, seed=seed)
    hook.prepare_with_bound_references = MagicMock(return_value="plan")
    facts = {
        (7, data_id): SparseObservation(
            data_id / 10,
            f"old-{data_id}",
            {"example": {"q": data_id}},
        )
        for data_id in range(1, 7)
    }
    hook._reference_snapshot = AdmissionReferenceSnapshot(
        owners_by_instance={data_id: (7,) for data_id in range(1, 7)},
        frontier_counts={7: 6},
        evaluation_counts={7: 6},
        facts=facts,
    )
    hook._reference_snapshot_iteration = 12
    dataset = _admission_dataset((0, 1, 2))
    metadata_before = (
        dataset.member_ids,
        dataset.failure_ids,
        dataset.success_ids,
        dataset.dci_result,
    )
    state = MagicMock()
    state.i = 11
    state.get_prospective_frontier_ineligible_ids.return_value = (0, 2)
    loader = ListDataLoader([{"q": index} for index in range(7)])
    expected_ids = (
        1,
        *random.Random(seed).sample((3, 4, 5, 6), 2),
    )

    plans = hook.prepare_after_proposals(
        state=state,
        admission_set=loader,
        requests=[_admission_request(dataset)],
    )

    assert plans == ["plan"]
    kwargs = hook.prepare_with_bound_references.call_args.kwargs
    assert kwargs["admission_ids"] == expected_ids
    assert len(set(kwargs["admission_ids"])) == 3
    assert {0, 2}.isdisjoint(kwargs["admission_ids"])
    assert kwargs["admission_batch"] == tuple(
        {"q": data_id} for data_id in expected_ids
    )
    assert (
        dataset.member_ids,
        dataset.failure_ids,
        dataset.success_ids,
        dataset.dci_result,
    ) == metadata_before


def test_dci_admission_fails_when_global_legal_pool_cannot_fill_three(
    tmp_path: Path,
) -> None:
    hook = _admission_hook(tmp_path)
    hook._reference_snapshot = AdmissionReferenceSnapshot(
        owners_by_instance={},
        frontier_counts={},
        evaluation_counts={},
        facts={},
    )
    hook._reference_snapshot_iteration = 12
    dataset = _admission_dataset((0, 1))
    state = MagicMock()
    state.i = 11
    state.get_prospective_frontier_ineligible_ids.return_value = (0, 2)

    with pytest.raises(ValueError, match="full DCI admission minibatch"):
        hook.prepare_after_proposals(
            state=state,
            admission_set=ListDataLoader([{"q": index} for index in range(4)]),
            requests=[_admission_request(dataset)],
        )


def test_dci_admission_evaluates_missing_snapshot_fact_and_keeps_alignment(
    tmp_path: Path,
) -> None:
    hook = _admission_hook(tmp_path)
    hook.prepare_with_bound_references = MagicMock(return_value="plan")
    hook._evaluate_missing_reference_groups = MagicMock()
    fact_1 = SparseObservation(0.25, "old-1", {"example": {"q": 1}})
    fact_2 = SparseObservation(0.5, "old-2", {"example": {"q": 2}})
    fact_3 = SparseObservation(0.75, "fresh-3", {"example": {"q": 3}})
    hook.adapter.get_program_observation.return_value = fact_3
    hook._reference_snapshot = AdmissionReferenceSnapshot(
        owners_by_instance={1: (7,), 2: (7,), 3: (7,)},
        frontier_counts={7: 3},
        evaluation_counts={7: 3},
        facts={(7, 1): fact_1, (7, 2): fact_2},
    )
    hook._reference_snapshot_iteration = 12
    dataset = _admission_dataset((0, 1, 2, 3))
    state = MagicMock()
    state.i = 11
    state.get_prospective_frontier_ineligible_ids.return_value = (0,)
    loader = ListDataLoader([{"q": index} for index in range(4)])

    plans = hook.prepare_after_proposals(
        state=state,
        admission_set=loader,
        requests=[_admission_request(dataset)],
    )

    assert plans == ["plan"]
    hook._evaluate_missing_reference_groups.assert_called_once_with(
        state=state,
        ids=(1, 2, 3),
        batch=({"q": 1}, {"q": 2}, {"q": 3}),
        reference_program_indices=(7, 7, 7),
    )
    kwargs = hook.prepare_with_bound_references.call_args.kwargs
    assert kwargs["admission_ids"] == (1, 2, 3)
    assert kwargs["reference_program_indices"] == (7, 7, 7)
    assert kwargs["reference_facts"] == (fact_1, fact_2, fact_3)


def test_dci_admission_binds_every_reference_before_building_any_plan():
    hook = object.__new__(DciAdmissionHook)
    hook.admission_size = 1
    hook.rng = random.Random(0)
    events: list[str] = []
    fact_1 = SparseObservation(0.25, "old-1", {"example": {"q": 1}})
    fact_2 = SparseObservation(0.5, "old-2", {"example": {"q": 2}})
    hook._reference_snapshot = AdmissionReferenceSnapshot(
        owners_by_instance={1: (4,), 2: (5,)},
        frontier_counts={4: 1, 5: 1},
        evaluation_counts={4: 1, 5: 1},
        facts={(4, 1): fact_1, (5, 2): fact_2},
    )
    hook._reference_snapshot_iteration = 3

    def bind(*, instance_ids, **_kwargs):
        data_id = tuple(instance_ids)[0]
        events.append(f"bind-{data_id}")
        return (4 if data_id == 1 else 5,)

    def prepare(**kwargs):
        data_id = tuple(kwargs["admission_ids"])[0]
        events.append(f"prepare-{data_id}")
        return f"plan-{data_id}"

    hook.bind_snapshot_reference_program_indices = MagicMock(side_effect=bind)
    hook.prepare_with_bound_references = MagicMock(side_effect=prepare)

    def request(seed_id: int, member_id: int) -> PostProposalAdmissionRequest:
        dataset = DciReflectiveDataset(
            {"prompt": [{"Feedback": "seed"}]},
            seed_rollout=_rollout(f"seed-{seed_id}", seed_id),
        )
        dataset.dci_result = DciResult(
            definition=f"subproblem-{seed_id}",
            selected_ids=(member_id,),
            corpus_dir=Path(f"corpus-{seed_id}"),
            artifact_dir=Path(f"artifacts-{seed_id}"),
        )
        dataset.member_ids = (seed_id, member_id)
        dataset.failure_ids = (member_id,)
        dataset.subproblem_id = f"sp-{seed_id}"
        return PostProposalAdmissionRequest(
            parent_program_idx=0,
            parent_candidate={"prompt": "parent"},
            proposed_candidate={"prompt": "child"},
            components_to_update=("prompt",),
            reflective_dataset=dataset,
            propose_evaluation=EvaluationBatch(
                outputs=["seed"],
                scores=[0.0],
                trajectories=[{"example": {"q": seed_id}}],
            ),
            birth_propose_ids=(seed_id,),
            reflection_metadata={},
        )

    state = MagicMock()
    state.i = 2
    state.get_prospective_frontier_ineligible_ids.side_effect = (
        lambda _parent, propose_ids: tuple(propose_ids)
    )
    loader = ListDataLoader([{"q": index} for index in range(4)])

    plans = hook.prepare_after_proposals(
        state=state,
        admission_set=loader,
        requests=(request(0, 1), request(3, 2)),
    )

    assert plans == ["plan-1", "plan-2"]
    assert events == ["bind-1", "bind-2", "prepare-1", "prepare-2"]


def test_dci_admission_snapshot_keeps_the_pre_batch_maximum_fact(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    state = GEPAState(
        {"prompt": "seed"},
        ValsetEvaluation(
            outputs_by_val_id={2: "old-output"},
            scores_by_val_id={2: 0.25},
            objective_scores_by_val_id=None,
        ),
        frontier_type="instance",
    )
    old_evaluation = EvaluationBatch(
        outputs=["old-output"],
        scores=[0.25],
        trajectories=[{"example": {"q": 2}, "version": "old"}],
    )
    adapter.record_logical_rollouts(
        phase="propose_parent",
        iteration=0,
        parent_program_idx=0,
        candidate={"prompt": "seed"},
        evaluation_ids=(2,),
        evaluation=old_evaluation,
    )
    adapter.commit_program_observations(
        program_idx=0,
        evaluation_ids=(2,),
        evaluation=old_evaluation,
        committed_ids=(2,),
    )
    hook = DciAdmissionHook(
        adapter=adapter,
        rng=random.Random(0),
        run_dir=tmp_path,
        logger=MagicMock(),
        admission_size=3,
    )

    hook.bind_proposal_batch_context(state=state, iteration=1, tasks=())

    state.commit_existing_program_evaluation(
        program_idx=0,
        valset_evaluation=ValsetEvaluation(
            outputs_by_val_id={2: "new-output"},
            scores_by_val_id={2: 1.0},
            objective_scores_by_val_id=None,
        ),
        run_dir=None,
        iteration=1,
    )
    new_evaluation = EvaluationBatch(
        outputs=["new-output"],
        scores=[1.0],
        trajectories=[{"example": {"q": 2}, "version": "new"}],
    )
    adapter.record_logical_rollouts(
        phase="propose_parent",
        iteration=1,
        parent_program_idx=0,
        candidate={"prompt": "seed"},
        evaluation_ids=(2,),
        evaluation=new_evaluation,
    )
    adapter.commit_program_observations(
        program_idx=0,
        evaluation_ids=(2,),
        evaluation=new_evaluation,
        committed_ids=(2,),
    )

    assert hook._reference_snapshot.facts[(0, 2)].score == 0.25
    assert hook._reference_snapshot.facts[(0, 2)].output == "old-output"
    assert adapter.get_program_observation(0, 2).score == 1.0


def test_dci_proposal_derives_strata_from_frontier_scores_then_uses_official_dspy(
    monkeypatch,
    tmp_path: Path,
):
    adapter = _adapter(tmp_path)
    success = _rollout("r-success", 2, score=1.0)
    failure = _rollout("r-failure", 3, score=0.25)
    seed = _rollout("r-seed", 0, iteration=2)
    adapter._logical_rollouts = [success, failure, seed]
    adapter._rollout_by_ref = {
        item.rollout_ref: item for item in adapter._logical_rollouts
    }
    adapter._active_batch_context = _DciBatchContext(
        iteration=2,
        corpus=(success, failure),
        catalog=(),
        frontier_rollout_refs={
            2: (success.rollout_ref,),
            3: (failure.rollout_ref,),
        },
    )
    dataset = DciReflectiveDataset(
        {"prompt": [{"Feedback": "seed feedback"}]},
        seed_rollout=seed,
    )
    adapter.dci_boundary = MagicMock()
    definition = "  terminal check\nReuse the verified binding.\n"
    adapter.dci_boundary.run.return_value = _result(
        tmp_path,
        "proposal",
        definition=definition,
        selected_ids=(2, 3),
    )
    captured = {}

    def official_proposal(self, candidate, reflective_dataset, components):
        captured["dataset"] = reflective_dataset
        captured["components"] = components
        return {"prompt": "improved"}

    monkeypatch.setattr(
        SparseObservationDspyAdapter,
        "propose_new_texts",
        official_proposal,
    )

    result = adapter.propose_new_texts(
        {"prompt": "seed"},
        dataset,
        ["prompt"],
    )

    assert result == {"prompt": "improved"}
    call = adapter.dci_boundary.run.call_args
    assert "existing_subproblem_ids" not in call.kwargs
    assert (call.kwargs["corpus_cwd"] / "selection").is_dir()
    assert dataset.member_ids == (0, 2, 3)
    assert dataset.failure_ids == (3,)
    assert dataset.success_ids == (2,)
    assert set(dataset.selected_evidence_ids) == {2, 3}
    assert len(captured["dataset"]["prompt"]) == 3
    assert captured["dataset"]["prompt"][0]["Feedback"].endswith(definition)


@pytest.mark.parametrize(
    ("coins", "expected"),
    [
        ((0.1, 0.2), (("failure", 1), ("failure", 2))),
        ((0.1, 0.9), (("failure", 1), ("success", 3))),
        ((0.9, 0.8), (("success", 3), ("success", 4))),
    ],
)
def test_each_remaining_evidence_slot_independently_chooses_a_stratum(
    monkeypatch,
    tmp_path: Path,
    coins: tuple[float, float],
    expected: tuple[tuple[str, int], tuple[str, int]],
) -> None:
    adapter = _adapter(tmp_path)
    records = {
        data_id: _rollout(
            f"r-{data_id}",
            data_id,
            score=0.0 if data_id < 3 else 1.0,
        )
        for data_id in (1, 2, 3, 4)
    }
    adapter._rollout_by_ref = {
        record.rollout_ref: record for record in records.values()
    }
    context = _DciBatchContext(
        iteration=1,
        corpus=tuple(records.values()),
        catalog=(),
        frontier_rollout_refs={
            data_id: (record.rollout_ref,)
            for data_id, record in records.items()
        },
    )
    dataset = DciReflectiveDataset(
        {"prompt": [{"Feedback": "seed"}]},
        seed_rollout=_rollout("seed", 0),
    )
    dataset.failure_ids = (1, 2)
    dataset.success_ids = (3, 4)

    class FakeRandom:
        def __init__(self) -> None:
            self._coins = iter(coins)

        def random(self) -> float:
            return next(self._coins)

        def randrange(self, size: int) -> int:
            assert size > 0
            return 0

    monkeypatch.setattr(
        "bridge.dci_compass.random.Random",
        lambda _seed: FakeRandom(),
    )

    selected = adapter._select_evidence(dataset=dataset, context=context)

    assert tuple((polarity, data_id) for polarity, data_id, _ in selected) == expected


def test_repeated_seed_counts_when_frozen_frontier_exists_but_is_not_resampled(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    prior_seed_frontier = _rollout("r-prior-seed", 0, score=0.25)
    success = _rollout("r-success", 2, score=1.0)
    current_seed = _rollout("r-current-seed", 0, iteration=2, score=0.0)
    adapter._rollout_by_ref = {
        record.rollout_ref: record
        for record in (prior_seed_frontier, success, current_seed)
    }
    context = _DciBatchContext(
        iteration=2,
        corpus=(prior_seed_frontier, success),
        catalog=(),
        frontier_rollout_refs={
            0: (prior_seed_frontier.rollout_ref,),
            2: (success.rollout_ref,),
        },
    )
    dataset = DciReflectiveDataset(
        {"prompt": [{"Feedback": "seed"}]},
        seed_rollout=current_seed,
    )
    result = _result(
        tmp_path,
        "repeated-seed",
        definition="constraint verification",
        selected_ids=(2,),
    )

    adapter._partition_selected_members(
        dataset=dataset,
        result=result,
        context=context,
    )
    selected = adapter._select_evidence(dataset=dataset, context=context)

    assert dataset.member_ids == (0, 2)
    assert dataset.failure_ids == (0,)
    assert dataset.success_ids == (2,)
    assert tuple(data_id for _, data_id, _ in selected) == (2,)


def test_adapter_state_rejects_obsolete_json_boundary_schema(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    state = adapter.get_adapter_state()
    state[adapter._DCI_STATE_KEY]["schema_version"] = 1

    with pytest.raises(RuntimeError, match="unsupported COMPASS DCI"):
        _adapter(tmp_path / "restored-old-schema").set_adapter_state(state)


def test_no_extra_dci_evidence_skips_only_proposal_and_keeps_discovery(
    monkeypatch,
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    seed = _rollout("r-seed", 0, iteration=2)
    adapter._logical_rollouts = [seed]
    adapter._rollout_by_ref = {seed.rollout_ref: seed}
    adapter._active_batch_context = _DciBatchContext(
        iteration=2,
        corpus=(),
        catalog=(),
        frontier_rollout_refs={},
    )
    dataset = DciReflectiveDataset(
        {"prompt": [{"Feedback": "seed feedback"}]},
        seed_rollout=seed,
    )
    adapter.dci_boundary = MagicMock()
    adapter.dci_boundary.run.return_value = _result(
        tmp_path,
        "no-evidence",
        definition="An unresolved seed-only bottleneck.",
        selected_ids=(),
    )

    assert adapter.propose_new_texts(
        {"prompt": "seed"},
        dataset,
        ["prompt"],
    ) == {}
    assert dataset.skipped_no_dci_evidence is True
    assert dataset.member_ids == (0,)

    monkeypatch.setattr(
        SparseObservationDspyAdapter,
        "propose_new_texts_batch",
        lambda self, jobs: [{} for _ in jobs],
    )
    adapter.propose_new_texts_batch(
        [({"prompt": "seed"}, dataset, ["prompt"])]
    )

    assert adapter.proposal_skipped_no_dci_evidence == 1
    assert len(adapter.subproblems) == 1
    assert adapter.subproblems[0].member_ids == (0,)
    assert adapter.subproblems[0].failure_count == 0
    assert adapter.subproblems[0].success_count == 0

    restored = _adapter(tmp_path / "restored-with-discovery")
    restored.set_adapter_state(adapter.get_adapter_state())
    assert restored.proposal_skipped_no_dci_evidence == 1
    assert restored.subproblems == adapter.subproblems
