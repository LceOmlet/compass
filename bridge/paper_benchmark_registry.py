from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OfficialBenchmarkSpec:
    """One paper task resolved from the official GEPA artifact registry."""

    task_id: str
    display_name: str
    benchmark_meta: Any
    program_index: int
    max_metric_calls: int

    @property
    def program(self) -> Any:
        return self.benchmark_meta.program[self.program_index]

    @property
    def benchmark_class_name(self) -> str:
        return self.benchmark_meta.benchmark.__name__

    @property
    def program_class_name(self) -> str:
        return getattr(self.program, "_name", self.program.__class__.__name__)

    @property
    def predictor_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.program.named_predictors())


@dataclass(frozen=True, slots=True)
class OfficialDatasetSplits:
    train: tuple[Any, ...]
    validation: tuple[Any, ...]
    test: tuple[Any, ...]
    fingerprints: Mapping[str, str]


_TASKS_BY_BENCHMARK_CLASS = {
    "HotpotQABench": ("hotpotqa", "HotpotQA", 6871),
    "IFBench": ("ifbench", "IFBench", 3593),
    "hoverBench": ("hover", "HoVer", 7051),
    "Papillon": ("pupa", "PUPA", 2426),
    "AIMEBench": ("aime_2025", "AIME-2025", 1839),
    "LiveBenchMathBench": ("livebench_math", "LiveBench-Math", 1839),
}


def _official_benchmark_metas() -> tuple[Any, ...]:
    """Import the six owner-provided ``BenchmarkMeta`` lists verbatim."""

    from gepa_artifact.benchmarks.AIME import benchmark as aime_metas
    from gepa_artifact.benchmarks.IFBench import benchmark as ifbench_metas
    from gepa_artifact.benchmarks.hotpotQA import benchmark as hotpot_metas
    from gepa_artifact.benchmarks.hover import benchmark as hover_metas
    from gepa_artifact.benchmarks.livebench_math import benchmark as livebench_metas
    from gepa_artifact.benchmarks.papillon import benchmark as pupa_metas

    # This is the same concatenation used by the artifact's
    # ``scripts.experiment_configs.get_benchmarks``. No benchmark program,
    # metric, feedback function, or dataset implementation is copied here.
    return tuple(
        hover_metas
        + hotpot_metas
        + pupa_metas
        + ifbench_metas
        + livebench_metas
        + aime_metas
    )


def load_official_benchmark_specs() -> dict[str, OfficialBenchmarkSpec]:
    """Resolve the paper tasks and reject silent owner-registry drift."""

    specs: dict[str, OfficialBenchmarkSpec] = {}
    seen_classes: set[str] = set()
    for meta in _official_benchmark_metas():
        benchmark_class_name = meta.benchmark.__name__
        if benchmark_class_name not in _TASKS_BY_BENCHMARK_CLASS:
            raise RuntimeError(
                "the official GEPA artifact registry contains an unreviewed "
                f"benchmark: {benchmark_class_name!r}"
            )
        if benchmark_class_name in seen_classes:
            raise RuntimeError(
                "the official GEPA artifact registry contains duplicate "
                f"benchmark metadata for {benchmark_class_name!r}"
            )
        seen_classes.add(benchmark_class_name)
        task_id, display_name, budget = _TASKS_BY_BENCHMARK_CLASS[
            benchmark_class_name
        ]
        if len(meta.program) != 1:
            raise RuntimeError(
                f"paper task {task_id!r} no longer has exactly one official program"
            )
        program = meta.program[0]
        predictor_names = tuple(name for name, _ in program.named_predictors())
        if not predictor_names or len(set(predictor_names)) != len(predictor_names):
            raise RuntimeError(
                f"paper task {task_id!r} has invalid official predictor names"
            )
        specs[task_id] = OfficialBenchmarkSpec(
            task_id=task_id,
            display_name=display_name,
            benchmark_meta=meta,
            program_index=0,
            max_metric_calls=budget,
        )

    expected_classes = set(_TASKS_BY_BENCHMARK_CLASS)
    if seen_classes != expected_classes:
        raise RuntimeError(
            "the official GEPA artifact registry does not match the six frozen "
            f"paper tasks; missing={sorted(expected_classes - seen_classes)}, "
            f"extra={sorted(seen_classes - expected_classes)}"
        )
    return specs


def _canonical_feedback(delegate: Callable[..., Any]) -> Callable[..., dict[str, Any]]:
    """Normalize the artifact's legacy feedback keys for current DSPy GEPA."""

    def canonical(**kwargs: Any) -> dict[str, Any]:
        result = delegate(**kwargs)
        if isinstance(result, Mapping):
            if "score" in result and "feedback" in result:
                return {
                    "score": result["score"],
                    "feedback": result["feedback"],
                }
            if "feedback_score" in result and "feedback_text" in result:
                return {
                    "score": result["feedback_score"],
                    "feedback": result["feedback_text"],
                }
        if hasattr(result, "score") and hasattr(result, "feedback"):
            return {
                "score": result.score,
                "feedback": result.feedback,
            }
        raise TypeError(
            "official feedback must expose score/feedback or "
            "feedback_score/feedback_text"
        )

    return canonical


def canonical_feedback_map(
    spec: OfficialBenchmarkSpec,
) -> dict[str, Callable[..., dict[str, Any]]]:
    """Reuse the owner feedback map or its owner-defined default construction."""

    meta = spec.benchmark_meta
    program = spec.program
    predictor_names = spec.predictor_names
    maps = meta.feedback_fn_maps
    if maps is not None and maps[spec.program_index] is not None:
        owner_map = maps[spec.program_index]
        if set(owner_map) != set(predictor_names):
            raise RuntimeError(
                f"official feedback keys for {spec.task_id!r} differ from its "
                f"predictors: feedback={tuple(owner_map)}, predictors={predictor_names}"
            )
        return {
            name: _canonical_feedback(owner_map[name])
            for name in predictor_names
        }

    metric_with_feedback = meta.metric_with_feedback
    if metric_with_feedback is None:
        raise RuntimeError(
            f"official paper task {spec.task_id!r} provides no feedback owner"
        )

    def owner_default_feedback(
        predictor_output: Any,
        predictor_inputs: Any,
        module_inputs: Any,
        module_outputs: Any,
        captured_trace: Any,
    ) -> Any:
        del predictor_output, predictor_inputs, captured_trace
        # This matches the default construction in the artifact's
        # ``scripts.run_experiments`` exactly.
        return metric_with_feedback(module_inputs, module_outputs, None)

    adapted = _canonical_feedback(owner_default_feedback)
    return {name: adapted for name in predictor_names}


def seed_candidate_from_program(program: Any) -> dict[str, str]:
    """Read official component names and instructions without rewriting them."""

    candidate = {
        name: predictor.signature.instructions
        for name, predictor in program.named_predictors()
    }
    if not candidate or any(
        not isinstance(name, str)
        or not name
        or not isinstance(instruction, str)
        for name, instruction in candidate.items()
    ):
        raise RuntimeError("official program does not expose valid seed instructions")
    return candidate


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        return _jsonable(dict(value))
    except (TypeError, ValueError):
        return {
            "__class__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "__repr__": repr(value),
        }


def split_fingerprint(examples: Sequence[Any]) -> str:
    """Hash split content and order for a run manifest."""

    digest = hashlib.sha256()
    for index, example in enumerate(examples):
        payload = json.dumps(
            {
                "index": index,
                "example": _jsonable(example),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def instantiate_official_splits(
    spec: OfficialBenchmarkSpec,
    *,
    optimizer_seed: int,
    dataset_mode: str = "lite",
) -> OfficialDatasetSplits:
    """Instantiate owner data and reproduce its nonzero-seed train/val shuffle."""

    if isinstance(optimizer_seed, bool) or not isinstance(optimizer_seed, int):
        raise TypeError("optimizer_seed must be an integer")
    benchmark = spec.benchmark_meta.benchmark(dataset_mode=dataset_mode)
    train = list(benchmark.train_set)
    validation = list(benchmark.val_set)
    test = list(benchmark.test_set)

    # This is the official artifact protocol in ``scripts.run_experiments``.
    if optimizer_seed != 0:
        train_size = len(train)
        combined = train + validation
        random.Random(optimizer_seed).shuffle(combined)
        train = combined[:train_size]
        validation = combined[train_size:]

    return OfficialDatasetSplits(
        train=tuple(train),
        validation=tuple(validation),
        test=tuple(test),
        fingerprints={
            "train": split_fingerprint(train),
            "validation": split_fingerprint(validation),
            "test": split_fingerprint(test),
        },
    )
