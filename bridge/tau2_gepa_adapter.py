"""Thin GEPA adapter for the pinned tau2-bench text-agent protocol.

The adapter owns no task semantics.  Each example is executed by tau2's
``run_single_task`` entry point, which in turn owns the environment, agent
loop, user simulator, tools, and evaluator.  The only local extension is an
agent factory whose system instruction is bound to one immutable GEPA
candidate before the official agent is constructed.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, TypedDict

from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT, LLMAgent
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType
from tau2.registry import registry
from tau2.runner.batch import run_single_task

TAU2_AGENT_INSTRUCTION_COMPONENT = "agent_instruction"
TAU2_CANDIDATE_AGENT_NAME = "compass_candidate_llm_agent"
TAU2_FIXED_CANDIDATE_AGENT_PREFIX = "compass_fixed_candidate_llm_agent_"


@dataclass(frozen=True, slots=True)
class Tau2Example:
    """One official tau2 task paired with its frozen simulation seed."""

    task: Task
    seed: int


@dataclass(frozen=True, slots=True)
class Tau2ExecutionFailure:
    """A submitted episode for which the owner runner raised an exception."""

    task_id: str
    seed: int
    error_type: str
    error_message: str


Tau2RolloutOutput = SimulationRun | Tau2ExecutionFailure


class Tau2Trajectory(TypedDict):
    """Opaque owner trajectory retained for reflection and sparse admission."""

    example: Tau2Example
    task_id: str
    seed: int
    candidate_sha256: str
    output: Tau2RolloutOutput


@dataclass(frozen=True, slots=True)
class _CandidateBinding:
    agent_instruction: str
    candidate_sha256: str


_candidate_binding: ContextVar[_CandidateBinding | None] = ContextVar(
    "compass_tau2_candidate_binding",
    default=None,
)
_fixed_candidate_factories: dict[str, Any] = {}
_fixed_candidate_factories_lock = threading.Lock()


def _validate_candidate(candidate: Mapping[str, str]) -> str:
    if set(candidate) != {TAU2_AGENT_INSTRUCTION_COMPONENT}:
        raise ValueError(
            "tau2 candidates must contain exactly one 'agent_instruction' component"
        )
    instruction = candidate[TAU2_AGENT_INSTRUCTION_COMPONENT]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("tau2 agent_instruction must be a non-empty string")
    return instruction


def candidate_sha256(candidate: Mapping[str, str]) -> str:
    """Return the canonical identity of one candidate without changing behavior."""

    _validate_candidate(candidate)
    payload = json.dumps(
        dict(candidate),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tau2_seed_candidate() -> dict[str, str]:
    """Expose the byte-identical instruction shipped by tau2's official agent."""

    return {TAU2_AGENT_INSTRUCTION_COMPONENT: AGENT_INSTRUCTION}


class CandidateBoundLLMAgent(LLMAgent):
    """Official LLMAgent with only its instruction component replaced."""

    def __init__(
        self,
        *,
        tools: list[Any],
        domain_policy: str,
        llm: str,
        llm_args: dict[str, Any] | None,
        agent_instruction: str,
    ) -> None:
        if not isinstance(agent_instruction, str) or not agent_instruction.strip():
            raise ValueError("agent_instruction must be a non-empty string")
        self._candidate_agent_instruction = agent_instruction
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=self._candidate_agent_instruction,
        )


def _candidate_agent_factory(
    tools: list[Any],
    domain_policy: str,
    **kwargs: Any,
) -> CandidateBoundLLMAgent:
    binding = _candidate_binding.get()
    if binding is None:
        raise RuntimeError(
            "tau2 candidate agent was constructed outside a bound rollout"
        )
    return CandidateBoundLLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        agent_instruction=binding.agent_instruction,
    )


def register_tau2_candidate_agent() -> str:
    """Register the stateless candidate factory without replacing owner entries."""

    existing = registry.get_agent_factory(TAU2_CANDIDATE_AGENT_NAME)
    if existing is None:
        registry.register_agent_factory(
            _candidate_agent_factory,
            TAU2_CANDIDATE_AGENT_NAME,
        )
    elif existing is not _candidate_agent_factory:
        raise RuntimeError(
            "tau2 candidate agent name is already owned by another factory"
        )
    return TAU2_CANDIDATE_AGENT_NAME


def register_tau2_fixed_candidate_agent(candidate: Mapping[str, str]) -> str:
    """Bind one frozen candidate to an official registry factory.

    The optimizer uses a ``ContextVar`` because different candidates can run
    concurrently.  Tau2's official ``run_tasks`` final evaluator creates its
    own worker threads, so a frozen candidate instead needs an immutable
    factory that can be resolved by name in every owner worker.
    """

    instruction = _validate_candidate(candidate)
    agent_name = TAU2_FIXED_CANDIDATE_AGENT_PREFIX + candidate_sha256(candidate)
    with _fixed_candidate_factories_lock:
        owned_factory = _fixed_candidate_factories.get(agent_name)
        registered_factory = registry.get_agent_factory(agent_name)
        if owned_factory is not None:
            if registered_factory is not owned_factory:
                raise RuntimeError(
                    "tau2 fixed candidate agent name is no longer owned by "
                    "its registered factory"
                )
            return agent_name
        if registered_factory is not None:
            raise RuntimeError(
                "tau2 fixed candidate agent name is already owned by another factory"
            )

        def fixed_candidate_agent_factory(
            tools: list[Any],
            domain_policy: str,
            **kwargs: Any,
        ) -> CandidateBoundLLMAgent:
            return CandidateBoundLLMAgent(
                tools=tools,
                domain_policy=domain_policy,
                llm=kwargs.get("llm"),
                llm_args=kwargs.get("llm_args"),
                agent_instruction=instruction,
            )

        _fixed_candidate_factories[agent_name] = fixed_candidate_agent_factory
        registry.register_agent_factory(fixed_candidate_agent_factory, agent_name)
    return agent_name


class Tau2GEPAAdapter(GEPAAdapter[Tau2Example, Tau2Trajectory, Tau2RolloutOutput]):
    """Execute prompt candidates through tau2's official single-task runner."""

    def __init__(self, run_config: Any, *, max_workers: int | None = None) -> None:
        if getattr(run_config, "domain", None) != "airline":
            raise ValueError("Tau2GEPAAdapter requires domain='airline'")
        configured_workers = getattr(run_config, "max_concurrency", None)
        workers = configured_workers if max_workers is None else max_workers
        if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
            raise TypeError("max_workers must be a positive integer")
        self.run_config = run_config
        self.max_workers = workers
        self.agent_name = register_tau2_candidate_agent()

    def _run_one(
        self,
        example: Tau2Example,
        binding: _CandidateBinding,
    ) -> Tau2RolloutOutput:
        token = _candidate_binding.set(binding)
        try:
            config = self.run_config.model_copy(
                update={
                    "agent": self.agent_name,
                    "task_ids": [example.task.id],
                }
            )
            simulation = run_single_task(
                config,
                example.task,
                seed=example.seed,
                evaluation_type=EvaluationType.ALL,
            )
            if simulation.reward_info is None:
                return Tau2ExecutionFailure(
                    task_id=str(example.task.id),
                    seed=example.seed,
                    error_type="MissingRewardInfo",
                    error_message=(
                        "tau2 run_single_task returned without official reward_info"
                    ),
                )
            return simulation
        # The owner entry point can surface heterogeneous provider, runner, and
        # environment exceptions.  They all have the same adapter-level meaning:
        # this submitted episode failed, while sibling episodes remain valid.
        except Exception as exc:  # noqa: BLE001
            return Tau2ExecutionFailure(
                task_id=str(example.task.id),
                seed=example.seed,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        finally:
            _candidate_binding.reset(token)

    @staticmethod
    def _score(output: Tau2RolloutOutput) -> float:
        if isinstance(output, Tau2ExecutionFailure):
            return 0.0
        if output.reward_info is None:
            return 0.0
        return float(output.reward_info.reward)

    def evaluate(
        self,
        batch: list[Tau2Example],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Tau2Trajectory, Tau2RolloutOutput]:
        instruction = _validate_candidate(candidate)
        digest = candidate_sha256(candidate)
        if not batch:
            return EvaluationBatch(
                outputs=[],
                scores=[],
                trajectories=[] if capture_traces else None,
                objective_scores=None,
                num_metric_calls=0,
            )
        task_ids = [str(example.task.id) for example in batch]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("one tau2 evaluation batch cannot repeat a task id")
        binding = _CandidateBinding(
            agent_instruction=instruction,
            candidate_sha256=digest,
        )
        outputs: list[Tau2RolloutOutput | None] = [None] * len(batch)
        workers = min(self.max_workers, len(batch))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._run_one, example, binding): position
                for position, example in enumerate(batch)
            }
            for future in as_completed(futures):
                outputs[futures[future]] = future.result()
        aligned_outputs = [output for output in outputs if output is not None]
        if len(aligned_outputs) != len(batch):
            raise RuntimeError("tau2 owner outputs are not instance-aligned")
        scores = [self._score(output) for output in aligned_outputs]
        trajectories = None
        if capture_traces:
            trajectories = [
                Tau2Trajectory(
                    example=example,
                    task_id=str(example.task.id),
                    seed=example.seed,
                    candidate_sha256=digest,
                    output=output,
                )
                for example, output in zip(
                    batch,
                    aligned_outputs,
                    strict=True,
                )
            ]
        return EvaluationBatch(
            outputs=aligned_outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=None,
            num_metric_calls=len(batch),
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Tau2Trajectory, Tau2RolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        _validate_candidate(candidate)
        digest = candidate_sha256(candidate)
        unexpected = set(components_to_update).difference(
            {TAU2_AGENT_INSTRUCTION_COMPONENT}
        )
        if unexpected:
            raise ValueError(f"unknown tau2 components requested: {sorted(unexpected)}")
        if eval_batch.trajectories is None:
            raise ValueError("tau2 reflection requires captured owner trajectories")
        if not (
            len(eval_batch.outputs)
            == len(eval_batch.scores)
            == len(eval_batch.trajectories)
        ):
            raise RuntimeError("tau2 reflection vectors are not instance-aligned")

        records: list[dict[str, Any]] = []
        for trajectory in eval_batch.trajectories:
            if trajectory["candidate_sha256"] != digest:
                raise RuntimeError("tau2 trajectory belongs to another candidate")
            output = trajectory["output"]
            generated: dict[str, Any]
            feedback: dict[str, Any]
            if isinstance(output, Tau2ExecutionFailure):
                failure = asdict(output)
                generated = {"execution_failure": failure}
                feedback = {"execution_failure": failure}
            else:
                generated = {
                    "messages": [
                        message.model_dump(mode="json")
                        for message in output.get_messages()
                    ],
                    "termination_reason": output.termination_reason.value,
                }
                feedback = {
                    "reward_info": (
                        None
                        if output.reward_info is None
                        else output.reward_info.model_dump(mode="json")
                    )
                }
            records.append(
                {
                    "Inputs": {
                        "task_id": trajectory["task_id"],
                        "seed": trajectory["seed"],
                    },
                    "Generated Outputs": generated,
                    "Feedback": json.dumps(
                        feedback,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    "candidate_sha256": digest,
                }
            )
        return {component: list(records) for component in components_to_update}


__all__ = [
    "TAU2_AGENT_INSTRUCTION_COMPONENT",
    "TAU2_CANDIDATE_AGENT_NAME",
    "TAU2_FIXED_CANDIDATE_AGENT_PREFIX",
    "CandidateBoundLLMAgent",
    "Tau2Example",
    "Tau2ExecutionFailure",
    "Tau2GEPAAdapter",
    "Tau2Trajectory",
    "candidate_sha256",
    "register_tau2_candidate_agent",
    "register_tau2_fixed_candidate_agent",
    "tau2_seed_candidate",
]
