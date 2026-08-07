from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from bridge import tau2_gepa_adapter as subject


class FakeConfig:
    domain = "airline"
    max_concurrency = 3

    def __init__(self, **values):
        self.values = dict(values)
        self.agent = values.get("agent", "llm_agent")

    def model_copy(self, *, update):
        return FakeConfig(**{**self.values, **update})


def fake_task(task_id: str):
    return SimpleNamespace(id=task_id)


def fake_simulation(task_id: str, reward: float, marker: str):
    message = SimpleNamespace(
        model_dump=lambda **_: {"role": "assistant", "content": marker}
    )
    reward_info = SimpleNamespace(
        reward=reward,
        model_dump=lambda **_: {"reward": reward, "reward_basis": ["DB"]},
    )
    return SimpleNamespace(
        task_id=task_id,
        reward_info=reward_info,
        termination_reason=SimpleNamespace(value="user_stop"),
        get_messages=lambda: [message],
    )


def test_seed_candidate_is_byte_identical_to_owner_instruction():
    assert subject.tau2_seed_candidate() == {
        subject.TAU2_AGENT_INSTRUCTION_COMPONENT: subject.AGENT_INSTRUCTION
    }


def test_candidate_agent_changes_only_owner_instruction_component():
    agent = subject.CandidateBoundLLMAgent(
        tools=[],
        domain_policy="OWNER POLICY",
        llm="owner/model",
        llm_args={"temperature": 0},
        agent_instruction="LEARNED INSTRUCTION",
    )
    assert agent.system_prompt == subject.SYSTEM_PROMPT.format(
        domain_policy="OWNER POLICY",
        agent_instruction="LEARNED INSTRUCTION",
    )
    assert (
        subject.CandidateBoundLLMAgent.generate_next_message
        is subject.LLMAgent.generate_next_message
    )
    assert agent.tools == []
    assert agent.domain_policy == "OWNER POLICY"
    assert agent.llm == "owner/model"
    assert agent.llm_args == {"temperature": 0}


def test_candidate_hash_is_canonical_and_unicode_sensitive():
    left = {subject.TAU2_AGENT_INSTRUCTION_COMPONENT: "处理航班"}
    same = dict(reversed(list(left.items())))
    changed = {subject.TAU2_AGENT_INSTRUCTION_COMPONENT: "处理航班。"}
    assert subject.candidate_sha256(left) == subject.candidate_sha256(same)
    assert subject.candidate_sha256(left) != subject.candidate_sha256(changed)


def test_factory_fails_closed_without_a_bound_candidate():
    subject.register_tau2_candidate_agent()
    factory = subject.registry.get_agent_factory(subject.TAU2_CANDIDATE_AGENT_NAME)
    with pytest.raises(RuntimeError, match="outside a bound rollout"):
        factory([], "policy", llm="model", llm_args={})


def test_fixed_candidate_factory_is_stable_and_context_free():
    candidate = {subject.TAU2_AGENT_INSTRUCTION_COMPONENT: "frozen instruction"}
    agent_name = subject.register_tau2_fixed_candidate_agent(candidate)
    assert agent_name == (
        subject.TAU2_FIXED_CANDIDATE_AGENT_PREFIX + subject.candidate_sha256(candidate)
    )
    factory = subject.registry.get_agent_factory(agent_name)
    assert subject.register_tau2_fixed_candidate_agent(dict(candidate)) == agent_name
    assert subject.registry.get_agent_factory(agent_name) is factory

    result: dict[str, object] = {}

    def construct_in_owner_worker():
        result["agent"] = factory(
            [],
            "owner policy",
            llm="owner/model",
            llm_args={"temperature": 0},
        )

    worker = threading.Thread(target=construct_in_owner_worker)
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    agent = result["agent"]
    assert agent.system_prompt == subject.SYSTEM_PROMPT.format(
        domain_policy="owner policy",
        agent_instruction="frozen instruction",
    )
    assert agent.llm == "owner/model"
    assert agent.llm_args == {"temperature": 0}


def test_fixed_candidate_factory_does_not_reuse_another_candidate():
    left = {subject.TAU2_AGENT_INSTRUCTION_COMPONENT: "left"}
    right = {subject.TAU2_AGENT_INSTRUCTION_COMPONENT: "right"}
    left_name = subject.register_tau2_fixed_candidate_agent(left)
    right_name = subject.register_tau2_fixed_candidate_agent(right)
    assert left_name != right_name
    left_agent = subject.registry.get_agent_factory(left_name)(
        [], "policy", llm="model", llm_args={}
    )
    right_agent = subject.registry.get_agent_factory(right_name)(
        [], "policy", llm="model", llm_args={}
    )
    assert "left" in left_agent.system_prompt
    assert "right" not in left_agent.system_prompt
    assert "right" in right_agent.system_prompt
    assert "left" not in right_agent.system_prompt


def test_evaluate_uses_official_entry_and_preserves_input_order(monkeypatch):
    barrier = threading.Barrier(3)
    seen: list[tuple[str, int, object, str]] = []

    def fake_run(config, task, *, seed, evaluation_type):
        barrier.wait(timeout=2)
        factory = subject.registry.get_agent_factory(config.agent)
        agent = factory([], "policy", llm="model", llm_args={"temperature": 0})
        if task.id == "slow":
            time.sleep(0.02)
        seen.append((task.id, seed, evaluation_type, agent.system_prompt))
        reward = {"slow": 0.25, "fast": 1.0, "middle": 0.5}[task.id]
        return fake_simulation(task.id, reward, agent.system_prompt)

    monkeypatch.setattr(subject, "run_single_task", fake_run)
    adapter = subject.Tau2GEPAAdapter(FakeConfig(), max_workers=3)
    batch = [
        subject.Tau2Example(fake_task("slow"), 11),
        subject.Tau2Example(fake_task("fast"), 12),
        subject.Tau2Example(fake_task("middle"), 13),
    ]
    candidate = {subject.TAU2_AGENT_INSTRUCTION_COMPONENT: "candidate alpha"}
    result = adapter.evaluate(batch, candidate, capture_traces=True)

    assert [output.task_id for output in result.outputs] == [
        "slow",
        "fast",
        "middle",
    ]
    assert result.scores == [0.25, 1.0, 0.5]
    assert result.num_metric_calls == 3
    assert result.objective_scores is None
    assert [item[0] for item in seen] != ["slow", "fast", "middle"]
    assert {item[1] for item in seen} == {11, 12, 13}
    assert {item[2] for item in seen} == {subject.EvaluationType.ALL}
    assert all("candidate alpha" in item[3] for item in seen)
    assert [trajectory["example"] for trajectory in result.trajectories] == batch
    assert subject._candidate_binding.get() is None


def test_concurrent_adapters_do_not_cross_candidate_bindings(monkeypatch):
    barrier = threading.Barrier(2)

    def fake_run(config, task, *, seed, evaluation_type):
        del seed, evaluation_type
        factory = subject.registry.get_agent_factory(config.agent)
        barrier.wait(timeout=2)
        agent = factory([], "policy", llm="model", llm_args={})
        return fake_simulation(task.id, 1.0, agent.system_prompt)

    monkeypatch.setattr(subject, "run_single_task", fake_run)
    adapter = subject.Tau2GEPAAdapter(FakeConfig(), max_workers=1)
    outputs: dict[str, str] = {}

    def evaluate(marker: str):
        result = adapter.evaluate(
            [subject.Tau2Example(fake_task(marker), 1)],
            {subject.TAU2_AGENT_INSTRUCTION_COMPONENT: marker},
        )
        outputs[marker] = result.outputs[0].get_messages()[0].model_dump()["content"]

    threads = [
        threading.Thread(target=evaluate, args=("candidate-A",)),
        threading.Thread(target=evaluate, args=("candidate-B",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()

    assert "candidate-A" in outputs["candidate-A"]
    assert "candidate-B" not in outputs["candidate-A"]
    assert "candidate-B" in outputs["candidate-B"]
    assert "candidate-A" not in outputs["candidate-B"]


def test_one_execution_failure_does_not_discard_success(monkeypatch):
    def fake_run(config, task, *, seed, evaluation_type):
        del config, seed, evaluation_type
        if task.id == "failed":
            raise TimeoutError("owner timeout")
        return fake_simulation(task.id, 0.75, "success")

    monkeypatch.setattr(subject, "run_single_task", fake_run)
    adapter = subject.Tau2GEPAAdapter(FakeConfig(), max_workers=2)
    result = adapter.evaluate(
        [
            subject.Tau2Example(fake_task("ok"), 1),
            subject.Tau2Example(fake_task("failed"), 2),
        ],
        {subject.TAU2_AGENT_INSTRUCTION_COMPONENT: "candidate"},
        capture_traces=False,
    )
    assert result.scores == [0.75, 0.0]
    assert result.trajectories is None
    assert result.num_metric_calls == 2
    assert isinstance(result.outputs[1], subject.Tau2ExecutionFailure)
    assert result.outputs[1].error_type == "TimeoutError"


def test_missing_owner_reward_is_an_execution_failure(monkeypatch):
    def fake_run(config, task, *, seed, evaluation_type):
        del config, seed, evaluation_type
        simulation = fake_simulation(task.id, 0.5, "unevaluated")
        simulation.reward_info = None
        return simulation

    monkeypatch.setattr(subject, "run_single_task", fake_run)
    adapter = subject.Tau2GEPAAdapter(FakeConfig(), max_workers=1)
    result = adapter.evaluate(
        [subject.Tau2Example(fake_task("missing-reward"), 3)],
        {subject.TAU2_AGENT_INSTRUCTION_COMPONENT: "candidate"},
        capture_traces=True,
    )

    assert result.scores == [0.0]
    assert isinstance(result.outputs[0], subject.Tau2ExecutionFailure)
    assert result.outputs[0].error_type == "MissingRewardInfo"
    reflective = adapter.make_reflective_dataset(
        {subject.TAU2_AGENT_INSTRUCTION_COMPONENT: "candidate"},
        result,
        [subject.TAU2_AGENT_INSTRUCTION_COMPONENT],
    )
    record = reflective[subject.TAU2_AGENT_INSTRUCTION_COMPONENT][0]
    assert "execution_failure" in record["Generated Outputs"]


def test_reflective_dataset_uses_only_owner_messages_and_reward(monkeypatch):
    def fake_run(config, task, *, seed, evaluation_type):
        del config, seed, evaluation_type
        return fake_simulation(task.id, 0.5, "owner transcript")

    monkeypatch.setattr(subject, "run_single_task", fake_run)
    adapter = subject.Tau2GEPAAdapter(FakeConfig(), max_workers=1)
    candidate = {subject.TAU2_AGENT_INSTRUCTION_COMPONENT: "candidate"}
    evaluation = adapter.evaluate(
        [subject.Tau2Example(fake_task("task-1"), 7)],
        candidate,
        capture_traces=True,
    )
    reflective = adapter.make_reflective_dataset(
        candidate,
        evaluation,
        [subject.TAU2_AGENT_INSTRUCTION_COMPONENT],
    )
    record = reflective[subject.TAU2_AGENT_INSTRUCTION_COMPONENT][0]
    assert record["Inputs"] == {"task_id": "task-1", "seed": 7}
    assert record["Generated Outputs"] == {
        "messages": [{"role": "assistant", "content": "owner transcript"}],
        "termination_reason": "user_stop",
    }
    assert json.loads(record["Feedback"]) == {
        "reward_info": {"reward": 0.5, "reward_basis": ["DB"]}
    }
    assert "evaluation_criteria" not in json.dumps(record)


def test_adapter_rejects_wrong_candidate_shape_and_duplicate_tasks(monkeypatch):
    adapter = subject.Tau2GEPAAdapter(FakeConfig(), max_workers=1)
    with pytest.raises(ValueError, match="exactly one"):
        adapter.evaluate([], {"other": "text"})
    with pytest.raises(ValueError, match="repeat a task id"):
        adapter.evaluate(
            [
                subject.Tau2Example(fake_task("same"), 1),
                subject.Tau2Example(fake_task("same"), 2),
            ],
            {subject.TAU2_AGENT_INSTRUCTION_COMPONENT: "candidate"},
        )
