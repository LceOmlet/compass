"""Frozen tau2-bench Airline protocol and official Adapter wiring.

This module deliberately leaves the optimizer's proposal/validation view
explicit.  Tau2 v1.0.1 owns only the 30-task train split and the 20-task test
split; it does not own a train-internal validation partition.  Callers must
therefore freeze that partition rather than inheriting an accidental local
choice.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from gepa import optimize
from gepa.core.state import GEPAState
from gepa.lm import LM
from tau2.data_model.simulation import Results, TerminationReason, TextRunConfig
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType
from tau2.metrics.agent_metrics import AgentMetrics, compute_metrics
from tau2.runner.batch import run_tasks
from tau2.runner.helpers import get_info, get_tasks
from tau2.utils.pydantic_utils import get_pydantic_hash

from bridge.b19_reversible_parent_selection import SelectionScoreMode
from bridge.b20_compass_reflection import (
    CompassReflectionEngineConfig,
    run_compass_gepa_adapter_engine,
)
from bridge.tau2_gepa_adapter import (
    Tau2Example,
    Tau2GEPAAdapter,
    Tau2ResourceUsageCallback,
    register_tau2_fixed_candidate_agent,
    tau2_seed_candidate,
)

Tau2AdapterMethod = Literal["gepa", "m0", "compass"]
Tau2ExperimentPhase = Literal["preflight", "formal"]

TAU2_AIRLINE_VERSION = "v1.0.1"
TAU2_AIRLINE_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
TAU2_AIRLINE_TRAIN_IDS = (
    "0",
    "1",
    "3",
    "4",
    "5",
    "7",
    "9",
    "10",
    "11",
    "12",
    "14",
    "15",
    "17",
    "20",
    "21",
    "23",
    "27",
    "28",
    "33",
    "34",
    "36",
    "38",
    "39",
    "40",
    "41",
    "42",
    "43",
    "46",
    "47",
    "49",
)
TAU2_AIRLINE_TEST_IDS = (
    "2",
    "6",
    "8",
    "13",
    "16",
    "18",
    "19",
    "22",
    "24",
    "25",
    "26",
    "29",
    "30",
    "31",
    "32",
    "35",
    "37",
    "44",
    "45",
    "48",
)
TAU2_AIRLINE_VALIDATION_IDS = tuple(
    task_id
    for owner_position, task_id in enumerate(TAU2_AIRLINE_TRAIN_IDS, start=1)
    if owner_position % 5 == 0
)
TAU2_AIRLINE_PROPOSAL_IDS = tuple(
    task_id
    for task_id in TAU2_AIRLINE_TRAIN_IDS
    if task_id not in set(TAU2_AIRLINE_VALIDATION_IDS)
)
TAU2_AIRLINE_FILE_SHA256 = {
    "data/tau2/domains/airline/db.json": (
        "7184914bd3720d93f1160a09bb2724c3a5601d8ca39d02d371cbbfa62626f7e2"
    ),
    "data/tau2/domains/airline/policy.md": (
        "7db375ddcce5061cf9f48a5aabe1fc3e52aa2936d6a65b7c5b62e3c7ca54cfe4"
    ),
    "data/tau2/domains/airline/split_tasks.json": (
        "8249d7af4c654a48e5fa776993c3416c3b68c7213a26d00ec89e976c28363e97"
    ),
    "data/tau2/domains/airline/tasks.json": (
        "34dfcf0622792d0d0fa32aa491d21582e1b5d3ec1948e7743bae881885f878ea"
    ),
}

TAU2_RUN_SEED = 300
TAU2_OPTIMIZATION_TRIAL_SEED = 626729
TAU2_TEST_TRIAL_SEEDS = (626729, 373753, 361454, 1567)
TAU2_OPTIMIZATION_ROLLOUT_BUDGET = 600
TAU2_PREFLIGHT_ROLLOUT_BUDGET = 96
TAU2_MAX_CONCURRENCY = 5
TAU2_AGENT_MODEL = "openai/gpt-4.1-mini-2025-04-14"
TAU2_USER_MODEL = "openai/gpt-4.1-2025-04-14"
TAU2_REFLECTION_MODEL = TAU2_AGENT_MODEL


@dataclass(frozen=True, slots=True)
class Tau2AirlineTaskSplits:
    train: tuple[Task, ...]
    test: tuple[Task, ...]


@dataclass(frozen=True, slots=True)
class Tau2AirlineOptimizationView:
    proposal: tuple[Task, ...]
    validation: tuple[Task, ...]


@dataclass(frozen=True, slots=True)
class Tau2AirlineOptimizationSettings:
    parent_selection_score_mode: SelectionScoreMode
    phase: Tau2ExperimentPhase = "formal"
    optimizer_seed: int = 0
    max_metric_calls: int = TAU2_OPTIMIZATION_ROLLOUT_BUDGET
    proposal_minibatch_size: int = 3
    admission_minibatch_size: int = 3
    parent_top_n: int = 5
    max_concurrency: int = TAU2_MAX_CONCURRENCY
    display_progress_bar: bool = False
    use_cloudpickle: bool = True


@dataclass(frozen=True, slots=True)
class Tau2AirlineOptimizationRun:
    method: Tau2AdapterMethod
    result: Any
    adapter: Tau2GEPAAdapter
    selected_candidate_idx: int
    selected_candidate: dict[str, str]


@dataclass(frozen=True, slots=True)
class Tau2AirlineFinalEvaluation:
    results: Results
    metrics: AgentMetrics


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_worktree_status(root: Path) -> str:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_frozen_tau2_airline_source(tau2_root: Path) -> None:
    """Fail closed when the pinned owner source or Airline data has changed."""

    root = tau2_root.resolve()
    if _git_head(root) != TAU2_AIRLINE_COMMIT:
        raise RuntimeError(
            f"tau2-bench must be {TAU2_AIRLINE_VERSION} at {TAU2_AIRLINE_COMMIT}"
        )
    dirty = _git_worktree_status(root)
    if dirty:
        raise RuntimeError(
            "frozen tau2-bench source must have a clean index and worktree; "
            f"found changes:\n{dirty}"
        )
    for relative_path, expected in TAU2_AIRLINE_FILE_SHA256.items():
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen tau2 Airline file: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                "frozen tau2 Airline file hash mismatch: "
                f"{relative_path}: expected={expected}, actual={actual}"
            )


def _validate_owner_task_order(
    tasks: Sequence[Task],
    *,
    expected_ids: tuple[str, ...],
    split_name: str,
) -> tuple[Task, ...]:
    loaded = tuple(tasks)
    actual_ids = tuple(str(task.id) for task in loaded)
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"official tau2 Airline {split_name} task identity/order changed: "
            f"expected={list(expected_ids)}, actual={list(actual_ids)}"
        )
    return loaded


def load_frozen_tau2_airline_splits(tau2_root: Path) -> Tau2AirlineTaskSplits:
    """Load both owner splits only after source and identity verification."""

    verify_frozen_tau2_airline_source(tau2_root)
    train = _validate_owner_task_order(
        get_tasks("airline", task_split_name="train"),
        expected_ids=TAU2_AIRLINE_TRAIN_IDS,
        split_name="train",
    )
    test = _validate_owner_task_order(
        get_tasks("airline", task_split_name="test"),
        expected_ids=TAU2_AIRLINE_TEST_IDS,
        split_name="test",
    )
    if set(TAU2_AIRLINE_TRAIN_IDS).intersection(TAU2_AIRLINE_TEST_IDS):
        raise RuntimeError("frozen tau2 Airline train/test splits overlap")
    return Tau2AirlineTaskSplits(train=train, test=test)


def freeze_optimization_view(
    train_tasks: Sequence[Task],
    *,
    proposal_task_ids: Sequence[str],
    validation_task_ids: Sequence[str],
) -> Tau2AirlineOptimizationView:
    """Validate one explicit, disjoint partition of the official train split."""

    official = _validate_owner_task_order(
        train_tasks,
        expected_ids=TAU2_AIRLINE_TRAIN_IDS,
        split_name="train",
    )
    proposal_ids = tuple(str(task_id) for task_id in proposal_task_ids)
    validation_ids = tuple(str(task_id) for task_id in validation_task_ids)
    _validate_optimization_partition_ids(proposal_ids, validation_ids)
    by_id = {str(task.id): task for task in official}
    return Tau2AirlineOptimizationView(
        proposal=tuple(by_id[task_id] for task_id in proposal_ids),
        validation=tuple(by_id[task_id] for task_id in validation_ids),
    )


def frozen_tau2_airline_optimization_view(
    train_tasks: Sequence[Task],
) -> Tau2AirlineOptimizationView:
    """Return the paper's deterministic 24/6 owner-order train view.

    Every fifth task in the official owner order is validation-only.  The
    other 24 tasks are proposal-only.  Keeping this rule here makes preflight
    and formal runs share one auditable partition without asking the model or
    a launcher to infer one.
    """

    return freeze_optimization_view(
        train_tasks,
        proposal_task_ids=TAU2_AIRLINE_PROPOSAL_IDS,
        validation_task_ids=TAU2_AIRLINE_VALIDATION_IDS,
    )


def _validate_optimization_partition_ids(
    proposal_ids: tuple[str, ...],
    validation_ids: tuple[str, ...],
) -> None:
    if not proposal_ids or not validation_ids:
        raise ValueError("proposal and validation task IDs must both be non-empty")
    if len(set(proposal_ids)) != len(proposal_ids):
        raise ValueError("proposal task IDs contain duplicates")
    if len(set(validation_ids)) != len(validation_ids):
        raise ValueError("validation task IDs contain duplicates")
    if set(proposal_ids).intersection(validation_ids):
        raise ValueError("proposal and validation task IDs must be disjoint")
    if set(proposal_ids).union(validation_ids) != set(TAU2_AIRLINE_TRAIN_IDS):
        raise ValueError(
            "proposal and validation task IDs must partition all 30 official "
            "Airline train tasks"
        )
    proposal_set = set(proposal_ids)
    validation_set = set(validation_ids)
    owner_proposal_ids = tuple(
        task_id for task_id in TAU2_AIRLINE_TRAIN_IDS if task_id in proposal_set
    )
    owner_validation_ids = tuple(
        task_id for task_id in TAU2_AIRLINE_TRAIN_IDS if task_id in validation_set
    )
    if proposal_ids != owner_proposal_ids or validation_ids != owner_validation_ids:
        raise ValueError(
            "proposal and validation task IDs must each preserve official owner order"
        )


def _validate_runtime_config(
    config: TextRunConfig,
    *,
    task_split_name: Literal["train", "test"],
    num_trials: int,
) -> None:
    expected = {
        "domain": "airline",
        "task_set_name": "airline",
        "task_split_name": task_split_name,
        "agent": "llm_agent",
        "user": "user_simulator",
        "llm_agent": TAU2_AGENT_MODEL,
        "llm_user": TAU2_USER_MODEL,
        "num_trials": num_trials,
        "max_steps": 200,
        "max_errors": 10,
        "timeout": None,
        "max_concurrency": TAU2_MAX_CONCURRENCY,
        "seed": TAU2_RUN_SEED,
        "max_retries": 3,
        "retry_delay": 1.0,
        "auto_review": False,
        "hallucination_retries": 0,
        "enforce_communication_protocol": False,
        "verbose_logs": False,
    }
    mismatches = {
        key: {"expected": value, "actual": getattr(config, key)}
        for key, value in expected.items()
        if getattr(config, key) != value
    }
    if mismatches:
        raise ValueError(f"tau2 Airline runtime config mismatch: {mismatches}")
    expected_lm_arg_keys = {"temperature", "api_base"}
    for owner, args in (
        ("agent", config.llm_args_agent),
        ("user", config.llm_args_user),
    ):
        if set(args) != expected_lm_arg_keys:
            raise ValueError(
                f"tau2 Airline {owner} LM args must contain exactly "
                "temperature and api_base; credentials must come from the "
                "process environment so owner checkpoints cannot serialize them"
            )
        if args["temperature"] != 0.0:
            raise ValueError(f"tau2 Airline {owner} temperature must be 0")
        if not isinstance(args["api_base"], str) or not args["api_base"].strip():
            raise ValueError(f"tau2 Airline {owner} api_base must be non-empty")


def build_tau2_airline_text_config(
    *,
    api_base: str,
    task_split_name: Literal["train", "test"],
    num_trials: int,
    agent_name: str = "llm_agent",
    auto_resume: bool = False,
    max_concurrency: int = TAU2_MAX_CONCURRENCY,
) -> TextRunConfig:
    """Construct the official half-duplex Airline model/runtime configuration.

    Credentials are intentionally absent.  LiteLLM resolves them from the
    process environment, preventing tau2's owner checkpoint metadata from
    serializing a plaintext secret.
    """

    if not api_base.strip():
        raise ValueError("api_base must be non-empty")
    if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
        raise TypeError("max_concurrency must be an integer")
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    return TextRunConfig(
        domain="airline",
        task_set_name="airline",
        task_split_name=task_split_name,
        agent=agent_name,
        user="user_simulator",
        llm_agent=TAU2_AGENT_MODEL,
        llm_args_agent={
            "temperature": 0.0,
            "api_base": api_base,
        },
        llm_user=TAU2_USER_MODEL,
        llm_args_user={
            "temperature": 0.0,
            "api_base": api_base,
        },
        num_trials=num_trials,
        max_steps=200,
        max_errors=10,
        timeout=None,
        max_concurrency=max_concurrency,
        seed=TAU2_RUN_SEED,
        max_retries=3,
        retry_delay=1.0,
        auto_resume=auto_resume,
        auto_review=False,
        hallucination_retries=0,
        enforce_communication_protocol=False,
        verbose_logs=False,
    )


def build_tau2_reflection_lm(*, api_base: str) -> LM:
    """Use GEPA's official generic reflection-LM seam for Airline."""

    if not api_base.strip():
        raise ValueError("api_base must be non-empty")
    return LM(
        TAU2_REFLECTION_MODEL,
        temperature=1.0,
        max_tokens=16_384,
        num_retries=0,
        api_base=api_base,
    )


def _optimization_examples(tasks: Sequence[Task]) -> list[Tau2Example]:
    return [Tau2Example(task=task, seed=TAU2_OPTIMIZATION_TRIAL_SEED) for task in tasks]


def run_tau2_airline_optimization(
    *,
    method: Tau2AdapterMethod,
    view: Tau2AirlineOptimizationView,
    run_config: TextRunConfig,
    reflection_lm: Any,
    run_dir: Path,
    settings: Tau2AirlineOptimizationSettings,
    verified_resume: bool = False,
    resource_usage_callback: Tau2ResourceUsageCallback | None = None,
) -> Tau2AirlineOptimizationRun:
    """Route each supported method through its official Adapter seam."""

    if method not in ("gepa", "m0", "compass"):
        raise ValueError(
            "tau2 Adapter runner supports only gepa, m0, and compass; "
            "DSPy MIPROv2 has no generic GEPAAdapter seam"
        )
    _validate_runtime_config(run_config, task_split_name="train", num_trials=1)
    proposal_ids = tuple(str(task.id) for task in view.proposal)
    validation_ids = tuple(str(task.id) for task in view.validation)
    _validate_optimization_partition_ids(proposal_ids, validation_ids)
    expected_budget = {
        "preflight": TAU2_PREFLIGHT_ROLLOUT_BUDGET,
        "formal": TAU2_OPTIMIZATION_ROLLOUT_BUDGET,
    }.get(settings.phase)
    if expected_budget is None:
        raise ValueError("tau2 Airline phase must be 'preflight' or 'formal'")
    if settings.max_metric_calls != expected_budget:
        raise ValueError(
            f"{settings.phase} tau2 Airline optimization budget must be "
            f"{expected_budget}"
        )
    if settings.max_concurrency != TAU2_MAX_CONCURRENCY:
        raise ValueError("formal tau2 Airline concurrency must be 5")
    if settings.proposal_minibatch_size != 3:
        raise ValueError("formal tau2 Airline proposal minibatch size must be 3")
    if settings.admission_minibatch_size != 3:
        raise ValueError("formal tau2 Airline admission minibatch size must be 3")

    if run_dir.exists() and any(run_dir.iterdir()) and not verified_resume:
        raise FileExistsError(
            "tau2 optimizer run_dir is non-empty; a caller must verify its strict "
            "manifest before setting verified_resume=True"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter = Tau2GEPAAdapter(
        run_config.model_copy(update={"num_trials": 1}),
        max_workers=settings.max_concurrency,
        resource_usage_callback=resource_usage_callback,
    )
    proposal_examples = _optimization_examples(view.proposal)
    validation_examples = _optimization_examples(view.validation)
    seed_candidate = tau2_seed_candidate()

    if method == "gepa":
        result = optimize(
            seed_candidate=seed_candidate,
            trainset=proposal_examples,
            valset=validation_examples,
            adapter=adapter,
            task_lm=None,
            evaluator=None,
            reflection_lm=reflection_lm,
            candidate_selection_strategy="pareto",
            frontier_type="instance",
            skip_perfect_score=True,
            batch_sampler="epoch_shuffled",
            reflection_minibatch_size=settings.proposal_minibatch_size,
            perfect_score=1.0,
            use_merge=False,
            max_metric_calls=settings.max_metric_calls,
            run_dir=str(run_dir),
            track_best_outputs=True,
            display_progress_bar=settings.display_progress_bar,
            use_cloudpickle=settings.use_cloudpickle,
            cache_evaluation=True,
            seed=settings.optimizer_seed,
            raise_on_exception=True,
            acceptance_criterion="strict_improvement",
        )
        selected_idx = result.best_idx
    else:
        condition = (
            "mini_admission_reflection" if method == "m0" else "compass_reflection"
        )
        engine_config = CompassReflectionEngineConfig(
            run_dir=run_dir,
            condition=condition,
            seed=settings.optimizer_seed,
            parent_top_n=settings.parent_top_n,
            max_metric_calls=settings.max_metric_calls,
            perfect_score=1.0,
            failure_score=0.0,
            num_threads=settings.max_concurrency,
            max_candidate_workers=1,
            skip_perfect_score=True,
            add_format_failure_as_feedback=False,
            track_best_outputs=True,
            display_progress_bar=settings.display_progress_bar,
            raise_on_exception=True,
            use_cloudpickle=settings.use_cloudpickle,
            parent_selection_score_mode=settings.parent_selection_score_mode,
            epoch_parallel_enabled=False,
            max_reflection_workers=1,
            acceptance_mode="strict_improvement",
            proposal_minibatch_size=settings.proposal_minibatch_size,
            admission_minibatch_size=settings.admission_minibatch_size,
        )
        compass_run = run_compass_gepa_adapter_engine(
            seed_candidate=seed_candidate,
            adapter=adapter,
            trainset=proposal_examples,
            validation_set=validation_examples,
            reflection_lm=reflection_lm,
            config=engine_config,
        )
        result = compass_run.result
        state = GEPAState.load(str(run_dir))
        selected_idx = compass_run.evaluation_policy.get_best_program(state)

    selected_candidate = dict(result.candidates[selected_idx])
    return Tau2AirlineOptimizationRun(
        method=method,
        result=result,
        adapter=adapter,
        selected_candidate_idx=selected_idx,
        selected_candidate=selected_candidate,
    )


def _expected_final_identities() -> set[tuple[str, int, int]]:
    return {
        (task_id, trial, seed)
        for trial, seed in enumerate(TAU2_TEST_TRIAL_SEEDS)
        for task_id in TAU2_AIRLINE_TEST_IDS
    }


def _validate_final_results(results: Results) -> None:
    identities: list[tuple[str, int, int]] = []
    infrastructure_failures: list[tuple[str, int, int]] = []
    for simulation in results.simulations:
        if simulation.trial is None or simulation.seed is None:
            raise RuntimeError("official tau2 result is missing trial or seed identity")
        identity = (
            str(simulation.task_id),
            int(simulation.trial),
            int(simulation.seed),
        )
        identities.append(identity)
        if (
            simulation.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
            or simulation.reward_info is None
        ):
            infrastructure_failures.append(identity)
    if len(set(identities)) != len(identities):
        raise RuntimeError("official tau2 final results contain duplicate identities")
    expected = _expected_final_identities()
    actual = set(identities)
    if actual != expected:
        raise RuntimeError(
            "official tau2 final result identity mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if infrastructure_failures:
        raise RuntimeError(
            "official tau2 final evaluation has unresolved infrastructure failures: "
            f"{sorted(infrastructure_failures)}"
        )


def _validate_final_resume_checkpoint(
    save_path: Path,
    *,
    final_config: TextRunConfig,
    official_test: Sequence[Task],
) -> None:
    """Apply tau2's own compatibility identity before allowing auto-resume."""

    previous = Results.load(save_path)
    expected_info = get_info(final_config)
    exclude_fields = {"environment_info": {"policy"}}
    if get_pydantic_hash(previous.info, exclude=exclude_fields) != get_pydantic_hash(
        expected_info, exclude=exclude_fields
    ):
        raise RuntimeError("tau2 final resume checkpoint run config changed")
    previous_tasks = {str(task.id): task for task in previous.tasks}
    expected_tasks = {str(task.id): task for task in official_test}
    if set(previous_tasks) != set(expected_tasks):
        raise RuntimeError("tau2 final resume checkpoint task set changed")
    changed = [
        task_id
        for task_id, expected_task in expected_tasks.items()
        if get_pydantic_hash(previous_tasks[task_id])
        != get_pydantic_hash(expected_task)
    ]
    if changed:
        raise RuntimeError(
            f"tau2 final resume checkpoint tasks changed: {sorted(changed)}"
        )


def run_tau2_airline_final_evaluation(
    *,
    candidate: dict[str, str],
    test_tasks: Sequence[Task],
    run_config: TextRunConfig,
    save_path: Path,
    save_dir: Path,
    console_display: bool = True,
    verified_resume: bool = False,
) -> Tau2AirlineFinalEvaluation:
    """Evaluate one frozen candidate only through tau2 owner entry points.

    A caller may repeat this call with ``verified_resume=True`` and an
    ``auto_resume`` config after verifying its external manifest.  Tau2's
    official checkpoint code then removes only infrastructure failures and
    resubmits them while preserving completed simulations.
    """

    official_test = _validate_owner_task_order(
        test_tasks,
        expected_ids=TAU2_AIRLINE_TEST_IDS,
        split_name="test",
    )
    _validate_runtime_config(run_config, task_split_name="train", num_trials=1)
    if save_path.exists() and not verified_resume:
        raise FileExistsError(
            "tau2 final save_path already exists; a caller must verify its strict "
            "manifest before setting verified_resume=True"
        )
    if save_path.exists() and not run_config.auto_resume:
        raise ValueError("verified tau2 final resume requires auto_resume=True")
    fixed_agent_name = register_tau2_fixed_candidate_agent(candidate)
    final_config = run_config.model_copy(
        update={
            "agent": fixed_agent_name,
            "task_split_name": "test",
            "task_ids": list(TAU2_AIRLINE_TEST_IDS),
            "num_trials": 4,
            "seed": TAU2_RUN_SEED,
            "max_concurrency": TAU2_MAX_CONCURRENCY,
        }
    )
    if save_path.exists():
        _validate_final_resume_checkpoint(
            save_path,
            final_config=final_config,
            official_test=official_test,
        )
    results = run_tasks(
        final_config,
        list(official_test),
        save_path=save_path,
        save_dir=save_dir,
        evaluation_type=EvaluationType.ALL,
        console_display=console_display,
        results_format="json",
    )
    _validate_final_results(results)
    return Tau2AirlineFinalEvaluation(
        results=results,
        metrics=compute_metrics(results),
    )


__all__ = [
    "TAU2_AGENT_MODEL",
    "TAU2_AIRLINE_COMMIT",
    "TAU2_AIRLINE_FILE_SHA256",
    "TAU2_AIRLINE_PROPOSAL_IDS",
    "TAU2_AIRLINE_TEST_IDS",
    "TAU2_AIRLINE_TRAIN_IDS",
    "TAU2_AIRLINE_VALIDATION_IDS",
    "TAU2_AIRLINE_VERSION",
    "TAU2_MAX_CONCURRENCY",
    "TAU2_OPTIMIZATION_ROLLOUT_BUDGET",
    "TAU2_OPTIMIZATION_TRIAL_SEED",
    "TAU2_PREFLIGHT_ROLLOUT_BUDGET",
    "TAU2_REFLECTION_MODEL",
    "TAU2_RUN_SEED",
    "TAU2_TEST_TRIAL_SEEDS",
    "TAU2_USER_MODEL",
    "Tau2AdapterMethod",
    "Tau2AirlineFinalEvaluation",
    "Tau2AirlineOptimizationRun",
    "Tau2AirlineOptimizationSettings",
    "Tau2AirlineOptimizationView",
    "Tau2AirlineTaskSplits",
    "Tau2ExperimentPhase",
    "build_tau2_airline_text_config",
    "build_tau2_reflection_lm",
    "freeze_optimization_view",
    "frozen_tau2_airline_optimization_view",
    "load_frozen_tau2_airline_splits",
    "run_tau2_airline_final_evaluation",
    "run_tau2_airline_optimization",
    "verify_frozen_tau2_airline_source",
]
