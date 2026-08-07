"""Frozen protocol helpers for the primary COMPASS paper methods.

This module is deliberately orchestration-only.  Benchmark programs, task
metrics, prompt proposal, optimizer state, and multimodal rendering remain
owned by their pinned upstream implementations.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import dspy

from bridge.paper_benchmark_registry import canonical_feedback_map

PrimaryTask = Literal["ifbench", "aime_2025", "chartqa"]
PrimaryMethod = Literal["seed", "mipro", "gepa", "m0", "compass"]
RunPhase = Literal["preflight", "formal"]

PRIMARY_TASK_BUDGETS: Final[Mapping[str, int]] = {
    "ifbench": 3593,
    "aime_2025": 1839,
    "chartqa": 1152,
}
PREFLIGHT_MINIMUM_ROLLOUTS: Final[Mapping[str, int]] = {
    "ifbench": 96,
    "aime_2025": 60,
    "chartqa": 60,
}

# These are the only two exact MIPROv2-Heavy invocation totals frozen by the
# official GEPA artifact.  ChartQA is not in that artifact and current DSPy
# MIPROv2 exposes trials/candidates, not a max_metric_calls seam.  Guessing a
# trial count that happens to approach 1152 would not be rollout matching.
MIPRO_HEAVY_OWNER_BUDGETS: Final[Mapping[str, int]] = {
    "ifbench": 3593,
    "aime_2025": 1839,
}

PRIMARY_METHODS: Final[tuple[str, ...]] = (
    "seed",
    "mipro",
    "gepa",
    "m0",
    "compass",
)
PRIMARY_TASKS: Final[tuple[str, ...]] = tuple(PRIMARY_TASK_BUDGETS)
RUN_PHASES: Final[tuple[str, ...]] = ("preflight", "formal")


@dataclass(frozen=True, slots=True)
class PrimaryCell:
    """One validated task/method/phase identity from a frozen matrix."""

    task_id: PrimaryTask
    method: PrimaryMethod
    phase: RunPhase
    optimizer_seed: int
    logical_rollout_budget: int


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible protocol object without interpreting it."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_primary_cell(
    *,
    task_id: str,
    method: str,
    phase: str,
    optimizer_seed: int,
    logical_rollout_budget: int,
) -> PrimaryCell:
    """Validate the frozen primary-suite budget without inventing a cap."""

    if task_id not in PRIMARY_TASKS:
        raise ValueError(f"unsupported primary task: {task_id!r}")
    if method not in PRIMARY_METHODS:
        raise ValueError(f"unsupported primary method: {method!r}")
    if phase not in RUN_PHASES:
        raise ValueError(f"unsupported primary phase: {phase!r}")
    if isinstance(optimizer_seed, bool) or optimizer_seed != 0:
        raise ValueError("required primary cells use optimizer_seed=0")
    if (
        isinstance(logical_rollout_budget, bool)
        or not isinstance(logical_rollout_budget, int)
        or logical_rollout_budget < 0
    ):
        raise TypeError("logical_rollout_budget must be a non-negative integer")

    if method == "seed":
        if phase != "formal":
            raise ValueError("Seed is evaluation-only and has no optimizer preflight")
        if logical_rollout_budget != 0:
            raise ValueError("Seed has no optimization rollout budget")
    elif phase == "formal":
        expected = PRIMARY_TASK_BUDGETS[task_id]
        if logical_rollout_budget != expected:
            raise ValueError(
                f"formal {task_id}/{method} requires exactly {expected} "
                "optimization task rollouts"
            )
        if method == "mipro" and task_id not in MIPRO_HEAVY_OWNER_BUDGETS:
            raise ValueError(
                "formal ChartQA MIPROv2 is blocked: official DSPy MIPROv2 "
                "has no max_metric_calls seam and the official GEPA artifact "
                "does not define a 1152-rollout ChartQA MIPROv2 protocol"
            )
    else:
        minimum = PREFLIGHT_MINIMUM_ROLLOUTS[task_id]
        if logical_rollout_budget < minimum:
            raise ValueError(
                f"{task_id} preflight must cover at least {minimum} logical rollouts"
            )

    return PrimaryCell(
        task_id=task_id,  # type: ignore[arg-type]
        method=method,  # type: ignore[arg-type]
        phase=phase,  # type: ignore[arg-type]
        optimizer_seed=optimizer_seed,
        logical_rollout_budget=logical_rollout_budget,
    )


def compass_condition(method: str) -> str:
    """Map paper row names to the existing COMPASS engine contracts."""

    if method == "m0":
        return "mini_admission_reflection"
    if method == "compass":
        return "compass_reflection"
    raise ValueError(f"{method!r} is not a COMPASS-engine method")


def mipro_protocol(cell: PrimaryCell) -> dict[str, Any]:
    """Return only arguments accepted by the official DSPy MIPROv2 owner."""

    if cell.method != "mipro":
        raise ValueError("MIPRO protocol requested for a non-MIPRO cell")
    if cell.phase == "formal":
        owner_budget = MIPRO_HEAVY_OWNER_BUDGETS.get(cell.task_id)
        if owner_budget != cell.logical_rollout_budget:
            raise ValueError("formal MIPRO cell has no exact owner rollout protocol")
        return {
            "init": {
                "auto": "heavy",
                "max_errors_policy": "train_plus_validation_times_100",
                "seed": cell.optimizer_seed,
            },
            "compile": {},
            "rollout_alignment": {
                "owner": "gepa-artifact/get_max_invocations",
                "expected": owner_budget,
            },
        }

    # The disjoint pilot exercises official MIPRO initialization and two
    # official Optuna trials.  Its actual owner-reported rollout count must be
    # at least the preflight minimum; it is not treated as a formal cap.
    return {
        "init": {
            "auto": None,
            "num_candidates": 2,
            "max_errors_policy": "train_plus_validation_times_100",
            "seed": cell.optimizer_seed,
        },
        "compile": {"num_trials": 2},
        "rollout_alignment": {
            "owner": "dspy.MIPROv2.trial_logs",
            "minimum": cell.logical_rollout_budget,
        },
    }


def make_gepa_feedback_metric(
    benchmark: Any,
    *,
    scalar_metric: Callable[[Any, Any, Any], Any] | None = None,
) -> Callable[[Any, Any, Any, str | None, Any], dspy.Prediction]:
    """Bridge owner feedback into DSPy GEPA's documented five-argument seam.

    ``canonical_feedback_map`` remains the owner of predictor-specific
    feedback.  This adapter only passes DSPy's exact trace objects and returns
    DSPy's public ``Prediction(score, feedback)`` type.
    """

    feedback_map = canonical_feedback_map(benchmark)
    metric = scalar_metric or benchmark.benchmark_meta.metric

    def gepa_metric(
        gold: Any,
        pred: Any,
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
    ) -> dspy.Prediction:
        score = float(metric(gold, pred, trace))
        feedback = f"This trajectory got a score of {score}."
        if pred_name is not None:
            if pred_name not in feedback_map:
                raise KeyError(f"unknown owner predictor feedback key: {pred_name!r}")
            if not isinstance(pred_trace, Sequence) or len(pred_trace) != 1:
                raise RuntimeError(
                    "DSPy GEPA did not provide its documented one-predictor trace"
                )
            _, predictor_inputs, predictor_output = pred_trace[0]
            owner_output = feedback_map[pred_name](
                predictor_output=predictor_output,
                predictor_inputs=predictor_inputs,
                module_inputs=gold,
                module_outputs=pred,
                captured_trace=trace,
            )
            owner_score = float(owner_output["score"])
            if owner_score != score:
                raise RuntimeError(
                    "owner scalar metric and predictor feedback score disagree: "
                    f"metric={score}, feedback={owner_score}"
                )
            feedback = str(owner_output["feedback"])
        return dspy.Prediction(score=score, feedback=feedback)

    return gepa_metric


def source_file_hashes(
    project_root: Path,
    relative_paths: Sequence[str],
) -> dict[str, str]:
    """Hash only explicitly named semantic files for a create-only matrix."""

    hashes: dict[str, str] = {}
    for relative in relative_paths:
        path = (project_root / relative).resolve()
        try:
            path.relative_to(project_root.resolve())
        except ValueError as error:
            raise ValueError(
                f"source path escapes project root: {relative!r}"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(f"source file is missing: {relative}")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


__all__ = [
    "MIPRO_HEAVY_OWNER_BUDGETS",
    "PREFLIGHT_MINIMUM_ROLLOUTS",
    "PRIMARY_METHODS",
    "PRIMARY_TASKS",
    "PRIMARY_TASK_BUDGETS",
    "PrimaryCell",
    "canonical_sha256",
    "compass_condition",
    "make_gepa_feedback_metric",
    "mipro_protocol",
    "source_file_hashes",
    "validate_primary_cell",
]
