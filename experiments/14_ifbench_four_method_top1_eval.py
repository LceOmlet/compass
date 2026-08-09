from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import time
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from gepa.core.state import GEPAState
from gepa_artifact.benchmarks.IFBench import (
    IFBench,
    IFBenchCoT2StageProgram,
)
from gepa_artifact.benchmarks.IFBench import (
    feedback_fn_map as official_feedback_fn_map,
)
from gepa_artifact.benchmarks.IFBench import (
    metric as official_metric,
)

from bridge.b19_reversible_parent_selection import (
    evaluation_count,
    frontier_count,
    high_resolution_frontier_credits,
    high_resolution_selection_rate,
)
from bridge.b20_compass_reflection import SparseMinibatchEvaluationPolicy

RUNTIME_ROOT = Path(r"F:\compass-ifbench-local")
RUN_ROOT = RUNTIME_ROOT / "runs"
REMOTE_CONFIG = (
    RUNTIME_ROOT
    / "config"
    / "11_ifbench_siliconflow_v37_high_resolution_selection_strict_local.json"
)
METHODS = (
    {
        "label": "raw_strict",
        "version": "v35",
        "admission": "strict_improvement",
        "score_mode": "raw_frontier_rate",
        "run_name": "11_ifbench_raw_feedback_k1_20260730_v35_local_dual_account_r2",
    },
    {
        "label": "raw_always_accept",
        "version": "v36",
        "admission": "always_accept",
        "score_mode": "raw_frontier_rate",
        "run_name": "11_ifbench_raw_feedback_k1_20260730_v36_unconditional_admission",
    },
    {
        "label": "high_resolution_strict",
        "version": "v37",
        "admission": "strict_improvement",
        "score_mode": "high_resolution",
        "run_name": "11_ifbench_raw_feedback_k1_20260730_v37_high_resolution_selection_strict",
    },
    {
        "label": "high_resolution_always_accept",
        "version": "v38",
        "admission": "always_accept",
        "score_mode": "high_resolution",
        "run_name": "11_ifbench_raw_feedback_k1_20260730_v38_high_resolution_selection_always_accept",
    },
)
METHOD_SETS = {
    "v35-v38": METHODS,
    "v43-v46": (
        {
            "label": "raw_strict_window5",
            "version": "v43",
            "admission": "strict_improvement",
            "score_mode": "raw_frontier_rate",
            "run_name": (
                "11_ifbench_raw_feedback_k1_20260731_"
                "v43_split_1to1_raw_strict_window5"
            ),
        },
        {
            "label": "raw_always_accept_window5",
            "version": "v44",
            "admission": "always_accept",
            "score_mode": "raw_frontier_rate",
            "run_name": (
                "11_ifbench_raw_feedback_k1_20260731_"
                "v44_split_1to1_raw_always_accept_window5"
            ),
        },
        {
            "label": "high_resolution_strict_window5",
            "version": "v45",
            "admission": "strict_improvement",
            "score_mode": "high_resolution",
            "run_name": (
                "11_ifbench_raw_feedback_k1_20260731_"
                "v45_split_1to1_high_resolution_strict_window5"
            ),
        },
        {
            "label": "high_resolution_always_accept_window5",
            "version": "v46",
            "admission": "always_accept",
            "score_mode": "high_resolution",
            "run_name": (
                "11_ifbench_raw_feedback_k1_20260731_"
                "v46_split_1to1_high_resolution_always_accept_window5"
            ),
        },
    ),
    "v45-v46-owner-final": (
        {
            "label": "high_resolution_strict_window5_owner_final",
            "version": "v45",
            "admission": "strict_improvement",
            "score_mode": "raw_frontier_rate",
            "parent_selection_score_mode": "high_resolution",
            "run_name": (
                "11_ifbench_raw_feedback_k1_20260731_"
                "v45_split_1to1_high_resolution_strict_window5"
            ),
        },
        {
            "label": "high_resolution_always_accept_window5_owner_final",
            "version": "v46",
            "admission": "always_accept",
            "score_mode": "raw_frontier_rate",
            "parent_selection_score_mode": "high_resolution",
            "run_name": (
                "11_ifbench_raw_feedback_k1_20260731_"
                "v46_split_1to1_high_resolution_always_accept_window5"
            ),
        },
    ),
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _fraction_record(value: Fraction) -> dict[str, int]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def _candidate_digest(candidate: Mapping[str, str]) -> str:
    encoded = json.dumps(
        candidate,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _select_top1(
    state: GEPAState,
    *,
    score_mode: str,
) -> tuple[int, dict[str, Any]]:
    eligible = tuple(
        candidate_idx
        for candidate_idx in range(len(state.program_candidates))
        if evaluation_count(state, candidate_idx) > 0
    )
    if not eligible:
        return 0, {
            "frontier_count": 0,
            "clean_exposure": 0,
            "raw_frontier_rate": 0.0,
            "high_resolution_credit": {"numerator": 0, "denominator": 1},
            "high_resolution_rate": {"numerator": 0, "denominator": 1},
        }

    high_resolution_credits = high_resolution_frontier_credits(state)
    if score_mode == "raw_frontier_rate":
        selected_idx = SparseMinibatchEvaluationPolicy().get_best_program(state)
    elif score_mode == "high_resolution":
        score = lambda candidate_idx: high_resolution_selection_rate(
            state,
            candidate_idx,
        )
        selected_idx = max(
            eligible,
            key=lambda candidate_idx: (
                score(candidate_idx),
                evaluation_count(state, candidate_idx),
                -candidate_idx,
            ),
        )
    else:
        raise ValueError(f"unsupported score mode: {score_mode!r}")
    frontiers = frontier_count(state, selected_idx)
    exposure = evaluation_count(state, selected_idx)
    high_resolution_rate = high_resolution_selection_rate(state, selected_idx)
    return selected_idx, {
        "frontier_count": frontiers,
        "clean_exposure": exposure,
        "raw_frontier_rate": frontiers / exposure,
        "high_resolution_credit": _fraction_record(
            high_resolution_credits[selected_idx]
        ),
        "high_resolution_rate": _fraction_record(high_resolution_rate),
        "high_resolution_rate_float": float(high_resolution_rate),
    }


class ProgramRouter(dspy.Module):
    def __init__(self, programs: list[dspy.Module]):
        super().__init__()
        self._programs = programs

    def forward(
        self,
        prompt: str,
        top1_method_index: int,
    ) -> dspy.Prediction:
        return self._programs[int(top1_method_index)](prompt=prompt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-threads", default=32, type=int)
    parser.add_argument(
        "--method-set",
        choices=tuple(METHOD_SETS),
        default="v35-v38",
    )
    parser.add_argument("--config", default=REMOTE_CONFIG, type=Path)
    args = parser.parse_args()

    if args.num_threads <= 0:
        raise ValueError("--num-threads must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    methods = METHOD_SETS[args.method_set]
    config = _load_json(args.config)
    remote = config["remote_lm"]
    official = config["official_gepa"]
    if official["dataset_mode"] != "lite":
        raise ValueError("the frozen four-method protocol requires IFBench lite")

    api_key_env = str(remote["api_key_env"])
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"unset API key environment: {api_key_env}")

    training_entry = importlib.import_module(
        "experiments.11_ifbench_sparse_raw_feedback_training"
    )
    lm = training_entry._build_remote_lm(remote, api_key=api_key)
    _ = lm.supported_params
    dspy.configure(lm=lm, adapter=ChatAdapter())

    descriptors: list[dict[str, Any]] = []
    candidates: list[Mapping[str, str]] = []
    for method in methods:
        run_dir = RUN_ROOT / method["run_name"]
        state = GEPAState.load(str(run_dir))
        selected_idx, score_record = _select_top1(
            state,
            score_mode=method["score_mode"],
        )
        candidate = state.program_candidates[selected_idx]
        descriptors.append(
            {
                **method,
                "run_dir": str(run_dir),
                "selected_candidate_idx": selected_idx,
                "selection_rule": (
                    f"max({method['score_mode']}, clean_exposure, "
                    "-candidate_idx)"
                ),
                "candidate_pool_size": len(state.program_candidates),
                "optimization_metric_calls": state.total_num_evals,
                "candidate_sha256": _candidate_digest(candidate),
                **score_record,
            }
        )
        candidates.append(candidate)

    benchmark = IFBench(dataset_mode=official["dataset_mode"])
    adapter = DspyAdapter(
        student_module=IFBenchCoT2StageProgram(),
        metric_fn=official_metric,
        feedback_map=official_feedback_fn_map,
        failure_score=official["failure_score"],
        num_threads=args.num_threads,
    )
    programs = [adapter.build_program(candidate) for candidate in candidates]

    routed_testset: list[dspy.Example] = []
    for method_index in range(len(programs)):
        for example in benchmark.test_set:
            fields = copy.deepcopy(example.toDict())
            fields["top1_method_index"] = method_index
            routed_testset.append(
                dspy.Example(**fields).with_inputs(
                    "prompt",
                    "top1_method_index",
                )
            )

    started_at = time.time()
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_frozen_before_test": True,
        "test_used_for_selection": False,
        "dataset": "IFBench",
        "dataset_mode": official["dataset_mode"],
        "split": "official_test",
        "test_size": len(benchmark.test_set),
        "n_methods": len(descriptors),
        "logical_test_rollouts": len(routed_testset),
        "num_threads": args.num_threads,
        "method_set": args.method_set,
        "config": str(args.config.resolve()),
        "methods": descriptors,
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(
        "Four-method IFBench top-1 evaluation:",
        f"methods={len(programs)}",
        f"test={len(benchmark.test_set)}",
        f"logical_test_rollouts={len(routed_testset)}",
        f"threads={args.num_threads}",
        flush=True,
    )
    for descriptor in descriptors:
        print(
            "TOP1",
            descriptor["version"],
            f"candidate_idx={descriptor['selected_candidate_idx']}",
            f"F={descriptor['frontier_count']}",
            f"E={descriptor['clean_exposure']}",
            f"F_over_E={descriptor['raw_frontier_rate']:.17g}",
            "high_resolution_rate="
            f"{descriptor['high_resolution_rate_float']:.17g}",
            flush=True,
        )

    evaluator = dspy.Evaluate(
        devset=routed_testset,
        metric=official_metric,
        num_threads=args.num_threads,
        display_progress=True,
        max_errors=len(routed_testset) * 10,
        provide_traceback=True,
        failure_score=official["failure_score"],
    )
    evaluation = evaluator(ProgramRouter(programs))

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for example, _prediction, score in evaluation.results:
        grouped[int(example.top1_method_index)].append(
            {
                "test_key": str(example.key),
                "score": float(score),
            }
        )

    records: list[dict[str, Any]] = []
    for method_index, descriptor in enumerate(descriptors):
        instance_records = grouped[method_index]
        if len(instance_records) != len(benchmark.test_set):
            raise RuntimeError(
                f"incomplete official test group for {descriptor['version']}: "
                f"{len(instance_records)} != {len(benchmark.test_set)}"
            )
        raw_score = sum(item["score"] for item in instance_records) / len(
            instance_records
        )
        record = {
            **descriptor,
            "test_score": raw_score,
            "test_score_percent": 100.0 * raw_score,
            "test_rollouts": len(instance_records),
            "per_instance_scores": instance_records,
        }
        records.append(record)
        _write_json(output_dir / f"{descriptor['version']}_top1.json", record)
        print(
            "METHOD_RESULT",
            descriptor["version"],
            f"candidate_idx={descriptor['selected_candidate_idx']}",
            f"test_score_percent={100.0 * raw_score:.12g}",
            flush=True,
        )

    finished_at = datetime.now(timezone.utc).isoformat()
    final_result = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": finished_at,
        "elapsed_seconds": time.time() - started_at,
        "selection_frozen_before_test": True,
        "test_used_for_selection": False,
        "dataset": "IFBench",
        "dataset_mode": official["dataset_mode"],
        "split": "official_test",
        "test_size": len(benchmark.test_set),
        "records": records,
    }
    _write_json(output_dir / "final_result.json", final_result)
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = finished_at
    manifest["final_result"] = "final_result.json"
    _write_json(output_dir / "manifest.json", manifest)
    print(
        "FINAL_RESULT",
        json.dumps(
            {
                record["version"]: {
                    "candidate_idx": record["selected_candidate_idx"],
                    "test_score_percent": record["test_score_percent"],
                }
                for record in records
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
