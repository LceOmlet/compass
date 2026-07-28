from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from gepa_artifact.benchmarks.IFBench import (
    IFBench,
    IFBenchCoT2StageProgram,
    feedback_fn_map as official_feedback_fn_map,
    metric as official_metric,
)


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise TypeError("config must be a JSON object")
    return dict(config)


def _official_gepa_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> float | dspy.Prediction:
    if pred_name is None:
        return official_metric(gold, pred, trace)
    if pred_name not in official_feedback_fn_map:
        raise KeyError(f"unknown official IFBench predictor: {pred_name!r}")
    if not pred_trace or len(pred_trace) != 1:
        raise ValueError("official DSPy GEPA must provide one selected predictor call")
    _predictor, predictor_inputs, predictor_output = pred_trace[0]
    feedback = official_feedback_fn_map[pred_name](
        predictor_output=predictor_output,
        predictor_inputs=predictor_inputs,
        module_inputs=gold,
        module_outputs=pred,
        captured_trace=trace,
    )
    return dspy.Prediction(
        score=feedback["feedback_score"],
        feedback=feedback["feedback_text"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = _load_config(args.config)

    run_dir = Path(config["run_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    remote = config["remote_lm"]
    official = config["official_gepa"]
    final_eval = config["final_evaluation"]

    api_key = os.environ.get(remote["api_key_env"])
    if not api_key:
        raise ValueError(f"unset API key environment: {remote['api_key_env']}")

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
    # Materialize DSPy's official LiteLLM client once before its 32 evaluation
    # workers can race on the client's lazy import.
    _ = lm.supported_params
    dspy.configure(lm=lm, adapter=ChatAdapter())

    benchmark = IFBench(dataset_mode=official["dataset_mode"])
    program = IFBenchCoT2StageProgram()
    optimizer = dspy.GEPA(
        metric=_official_gepa_metric,
        max_metric_calls=official["max_metric_calls"],
        reflection_minibatch_size=official["reflection_minibatch_size"],
        candidate_selection_strategy=official["candidate_selection_strategy"],
        reflection_lm=lm,
        skip_perfect_score=official["skip_perfect_score"],
        add_format_failure_as_feedback=official[
            "add_format_failure_as_feedback"
        ],
        instruction_proposer=None,
        component_selector=official["component_selector"],
        use_merge=official["use_merge"],
        max_merge_invocations=official["max_merge_invocations"],
        num_threads=official["num_threads"],
        failure_score=official["failure_score"],
        perfect_score=official["perfect_score"],
        log_dir=str(run_dir),
        track_stats=official["track_stats"],
        use_wandb=False,
        track_best_outputs=official["track_best_outputs"],
        warn_on_score_mismatch=official["warn_on_score_mismatch"],
        use_mlflow=False,
        seed=official["seed"],
        gepa_kwargs={
            "frontier_type": official["frontier_type"],
            "batch_sampler": official["batch_sampler"],
            "val_evaluation_policy": official["val_evaluation_policy"],
            "acceptance_criterion": official["acceptance_criterion"],
            "cache_evaluation": official["cache_evaluation"],
            "use_cloudpickle": official["use_cloudpickle"],
        },
    )
    print(
        "Official GEPA IFBench:",
        f"train={len(benchmark.train_set)}",
        f"val={len(benchmark.val_set)}",
        f"test={len(benchmark.test_set)}",
        flush=True,
    )
    optimized_program = optimizer.compile(
        program,
        trainset=benchmark.train_set,
        valset=benchmark.val_set,
    )

    evaluator = dspy.Evaluate(
        devset=benchmark.test_set,
        metric=official_metric,
        num_threads=final_eval["num_threads"],
        display_progress=final_eval["display_progress"],
        max_errors=len(benchmark.test_set) * final_eval["max_errors_per_instance"],
        provide_traceback=final_eval["provide_traceback"],
    )
    test_score = evaluator(optimized_program)
    result = {
        "test_score": float(test_score),
        "best_idx": optimized_program.detailed_results.best_idx,
        "total_metric_calls": optimized_program.detailed_results.total_metric_calls,
        "num_full_val_evals": optimized_program.detailed_results.num_full_val_evals,
    }
    (run_dir / "final_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Official GEPA final result: {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
