from __future__ import annotations

import argparse
import copy
import json
import os
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cloudpickle
import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from gepa_artifact.benchmarks.IFBench import (
    IFBench,
    IFBenchCoT2StageProgram,
    feedback_fn_map as official_feedback_fn_map,
    metric as official_metric,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _gepa_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    with (snapshot_dir / "gepa_state.bin").open("rb") as handle:
        state = cloudpickle.load(handle)
    if not isinstance(state, Mapping):
        raise TypeError("frozen GEPA state must be a mapping")

    candidates = _load_json(snapshot_dir / "gepa_candidates.json")
    val_scores = [
        float(sum(candidate_scores.values()))
        for candidate_scores in state["prog_candidate_val_subscores"]
    ]
    best_idx = max(range(len(val_scores)), key=val_scores.__getitem__)
    discovery_rollouts = int(state["num_metric_calls_by_discovery"][best_idx])
    if len(candidates) != len(val_scores):
        raise ValueError("GEPA candidates and validation scores are not aligned")

    return {
        "candidate": candidates[best_idx],
        "best_idx": best_idx,
        "discovery_rollouts": discovery_rollouts,
        "validation_score_sum": val_scores[best_idx],
        "validation_size": len(state["prog_candidate_val_subscores"][best_idx]),
    }


def _v12_snapshot(snapshot_dir: Path) -> list[dict[str, Any]]:
    candidates = _load_json(snapshot_dir / "v12_candidates.json")
    run_log = _load_json(snapshot_dir / "v12_run_log.json")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("v12 snapshot must contain seed plus non-seed candidates")
    if not isinstance(run_log, list):
        raise TypeError("v12 run log must be a list")

    discovery_rollouts: dict[int, int] = {}
    admission_rollouts = 0
    for entry in run_log:
        if "admission_ids" in entry:
            admission_rollouts += len(entry["admission_ids"])
        if "new_program_idx" in entry:
            skill_idx = int(entry["new_program_idx"])
            parent_rollouts = sum(
                len(previous["subsample_ids"])
                for previous in run_log
                if int(previous["i"]) <= int(entry["i"])
            )
            discovery_rollouts[skill_idx] = (
                parent_rollouts + admission_rollouts
            )

    expected = set(range(1, len(candidates)))
    if set(discovery_rollouts) != expected:
        raise ValueError(
            "v12 run log does not identify every non-seed skill discovery"
        )

    return [
        {
            "candidate": candidates[skill_idx],
            "skill_idx": skill_idx,
            "discovery_rollouts": discovery_rollouts[skill_idx],
        }
        for skill_idx in range(1, len(candidates))
    ]


def _build_lm(remote: Mapping[str, Any]) -> dspy.LM:
    api_key_env = str(remote["api_key_env"])
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"unset API key environment: {api_key_env}")

    lm = dspy.LM(
        model=remote["model"],
        model_type=remote["model_type"],
        temperature=remote["temperature"],
        max_tokens=remote["max_tokens"],
        cache=remote["cache"],
        cache_in_memory=remote["cache_in_memory"],
        num_retries=remote["num_retries"],
        api_base=remote["api_base"],
        api_key=api_key,
        n=remote["n"],
        top_p=remote["top_p"],
        extra_body={
            "top_k": remote["top_k"],
            "chat_template_kwargs": {
                "enable_thinking": remote["enable_thinking"],
            },
        },
    )
    _ = lm.supported_params
    return lm


class ProgramRouter(dspy.Module):
    def __init__(self, programs: list[dspy.Module]):
        super().__init__()
        self._programs = programs

    def forward(
        self,
        prompt: str,
        oracle_program_index: int,
    ) -> dspy.Prediction:
        return self._programs[int(oracle_program_index)](prompt=prompt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    snapshot_dir = args.snapshot_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    config = _load_json(snapshot_dir / "gepa_config.json")
    remote = config["remote_lm"]
    official = config["official_gepa"]
    final_eval = config["final_evaluation"]

    gepa = _gepa_snapshot(snapshot_dir)
    v12 = _v12_snapshot(snapshot_dir)
    descriptors = [
        {
            "kind": "gepa_validation_best",
            "label": f"gepa_program_{gepa['best_idx']}",
            **gepa,
        },
        *[
            {
                "kind": "v12_nonseed",
                "label": f"v12_skill_{item['skill_idx']}",
                **item,
            }
            for item in v12
        ],
    ]

    lm = _build_lm(remote)
    dspy.configure(lm=lm, adapter=ChatAdapter())
    benchmark = IFBench(dataset_mode=official["dataset_mode"])

    adapter = DspyAdapter(
        student_module=IFBenchCoT2StageProgram(),
        metric_fn=official_metric,
        feedback_map=official_feedback_fn_map,
        failure_score=official["failure_score"],
        num_threads=official["num_threads"],
    )
    programs = [
        adapter.build_program(descriptor["candidate"])
        for descriptor in descriptors
    ]

    routed_testset: list[dspy.Example] = []
    for program_index in range(len(programs)):
        for example in benchmark.test_set:
            fields = copy.deepcopy(example.toDict())
            fields["oracle_program_index"] = program_index
            routed_testset.append(
                dspy.Example(**fields).with_inputs(
                    "prompt",
                    "oracle_program_index",
                )
            )

    manifest = {
        "comparison_contract": {
            "gepa_comparator": "frozen validation-best single program",
            "seed_excluded": True,
            "selection": "maximum test score among all frozen non-seed v12 skills",
            "test_cost_excluded_from_optimization_cost": True,
            "speed_unit": "one complete logical IFBench instance trajectory",
        },
        "dspy_num_threads": dspy.settings.num_threads,
        "n_programs": len(programs),
        "programs": [
            {
                key: value
                for key, value in descriptor.items()
                if key != "candidate"
            }
            for descriptor in descriptors
        ],
        "test_size": len(benchmark.test_set),
        "total_test_rollouts": len(routed_testset),
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(
        "Frozen IFBench oracle evaluation:",
        f"programs={len(programs)}",
        f"test={len(benchmark.test_set)}",
        f"logical_test_rollouts={len(routed_testset)}",
        f"global_threads={dspy.settings.num_threads}",
        flush=True,
    )

    started_at = time.time()
    evaluator = dspy.Evaluate(
        devset=routed_testset,
        metric=official_metric,
        num_threads=final_eval["num_threads"],
        display_progress=final_eval["display_progress"],
        max_errors=(
            len(routed_testset)
            * int(final_eval["max_errors_per_instance"])
        ),
        provide_traceback=final_eval["provide_traceback"],
        failure_score=official["failure_score"],
    )
    evaluation = evaluator(ProgramRouter(programs))

    grouped_scores: dict[int, list[float]] = defaultdict(list)
    for example, _prediction, score in evaluation.results:
        grouped_scores[int(example.oracle_program_index)].append(float(score))

    gepa_score = (
        sum(grouped_scores[0]) / len(grouped_scores[0])
    )
    records: list[dict[str, Any]] = []
    for program_index, descriptor in enumerate(descriptors):
        scores = grouped_scores[program_index]
        if len(scores) != len(benchmark.test_set):
            raise RuntimeError("official evaluation returned an incomplete group")
        raw_score = sum(scores) / len(scores)
        record = {
            key: value
            for key, value in descriptor.items()
            if key != "candidate"
        }
        record.update(
            {
                "program_index": program_index,
                "test_score": raw_score,
                "test_score_percent": 100.0 * raw_score,
                "test_rollouts": len(scores),
            }
        )
        if descriptor["kind"] == "v12_nonseed":
            record.update(
                {
                    "delta_vs_gepa": raw_score - gepa_score,
                    "rollout_speedup_vs_gepa": (
                        gepa["discovery_rollouts"]
                        / descriptor["discovery_rollouts"]
                    ),
                    "rollouts_saved_vs_gepa": (
                        gepa["discovery_rollouts"]
                        - descriptor["discovery_rollouts"]
                    ),
                }
            )
        records.append(record)
        _write_json(output_dir / f"{descriptor['label']}.json", record)
        print("PROGRAM_RESULT", json.dumps(record, sort_keys=True), flush=True)

    v12_records = [
        record for record in records if record["kind"] == "v12_nonseed"
    ]
    selected = max(v12_records, key=lambda item: item["test_score"])
    final_result = {
        "elapsed_seconds": time.time() - started_at,
        "gepa": records[0],
        "informal_test_oracle_winner": selected,
        "records": records,
        "selection_bias_warning": (
            "The v12 winner was selected on the official test set; this is an "
            "informal test-oracle estimate, not an unbiased generalization result."
        ),
    }
    _write_json(output_dir / "final_result.json", final_result)
    print(
        "FINAL_RESULT",
        json.dumps(final_result, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
