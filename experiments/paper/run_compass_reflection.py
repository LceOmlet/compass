from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from gepa.core.state import GEPAState

from bridge.b19_reversible_parent_selection import (
    candidate_selection_rate,
    evaluation_count,
    frontier_count,
    high_resolution_frontier_credits,
    high_resolution_selection_rate,
)
from bridge.b20_compass_reflection import (
    CompassReflectionEngineConfig,
    run_compass_reflection_engine,
)
from bridge.minibatch_config import minibatch_config_kwargs
from bridge.paper_benchmark_registry import (
    canonical_feedback_map,
    instantiate_official_splits,
    load_official_benchmark_specs,
)
from bridge.paper_source_snapshot import verify_project_source_snapshot
from bridge.request_deadline import DeadlineAwareLM

ROOT_KEYS = {
    "cache_dir",
    "condition",
    "dataset_mode",
    "model",
    "optimizer",
    "optimizer_seed",
    "run_dir",
    "source_snapshot",
    "task_id",
}
OPTIMIZER_REQUIRED_KEYS = {
    "add_format_failure_as_feedback",
    "display_progress_bar",
    "failure_score",
    "max_candidate_workers",
    "max_metric_calls",
    "num_threads",
    "parent_top_n",
    "perfect_score",
    "raise_on_exception",
    "skip_perfect_score",
    "track_best_outputs",
    "use_cloudpickle",
}
OPTIMIZER_BATCH_KEYS = {
    "reflection_minibatch_size",
    "proposal_minibatch_size",
    "admission_minibatch_size",
}
OPTIMIZER_METHOD_KEYS = {
    "acceptance_mode",
    "dci",
    "epoch_parallel_enabled",
    "evaluation_straggler_timeout",
    "max_candidate_proposals",
    "max_reflection_workers",
    "parent_selection_score_mode",
    "proposal_timeout_seconds",
    "proposal_tasks_per_iteration",
    "rollout_timeout_seconds",
}
DCI_MAPPING_KEYS = {
    "agent_dir",
    "max_turns",
    "model",
    "package_dir",
    "proposal_evidence_size",
    "provider",
    "runner_command",
    "tools",
}
MODEL_REQUIRED_KEYS = {
    "api_key_env",
    "cache",
    "cache_in_memory",
    "enable_thinking",
    "max_tokens",
    "model",
    "model_type",
    "num_retries",
    "temperature",
}
MODEL_OPTIONAL_KEYS = {
    "api_base",
    "checkpoint",
    "repetition_detection",
    "serving_backend",
    "serving_max_model_len",
    "supports_response_schema",
    "top_k",
    "top_p",
    "timeout",
}


def _exact_mapping(
    value: Any,
    *,
    name: str,
    keys: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    missing = keys.difference(value)
    extra = set(value).difference(keys)
    if missing or extra:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return dict(value)


def _model_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("model must be a JSON object")
    missing = MODEL_REQUIRED_KEYS.difference(value)
    extra = set(value).difference(MODEL_REQUIRED_KEYS | MODEL_OPTIONAL_KEYS)
    if missing or extra:
        raise ValueError(
            f"model keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    model = dict(value)
    max_tokens = model["max_tokens"]
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise TypeError("model.max_tokens must be a positive integer")
    timeout = model.get("timeout")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise TypeError("model.timeout must be a positive finite number")
    if "supports_response_schema" in model and not isinstance(
        model["supports_response_schema"],
        bool,
    ):
        raise TypeError("model.supports_response_schema must be a JSON boolean")
    repetition_detection = model.get("repetition_detection")
    if repetition_detection is not None:
        if not isinstance(repetition_detection, Mapping):
            raise TypeError("model.repetition_detection must be a JSON object")
        model["repetition_detection"] = dict(repetition_detection)
    return model


def _optimizer_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("optimizer must be a JSON object")
    missing = OPTIMIZER_REQUIRED_KEYS.difference(value)
    extra = set(value).difference(
        OPTIMIZER_REQUIRED_KEYS | OPTIMIZER_BATCH_KEYS | OPTIMIZER_METHOD_KEYS
    )
    if missing or extra:
        raise ValueError(
            f"optimizer keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    optimizer = dict(value)
    minibatch_config_kwargs(optimizer, namespace="optimizer")
    acceptance_mode = optimizer.get(
        "acceptance_mode",
        "strict_improvement",
    )
    if acceptance_mode not in {"strict_improvement", "always_accept"}:
        raise ValueError(
            "optimizer.acceptance_mode must be 'strict_improvement' or 'always_accept'"
        )
    score_mode = optimizer.get(
        "parent_selection_score_mode",
        "high_resolution",
    )
    if score_mode not in {
        "raw_frontier_rate",
        "high_resolution",
        "high_resolution_lexicographic",
    }:
        raise ValueError(
            "optimizer.parent_selection_score_mode must be "
            "'raw_frontier_rate', 'high_resolution', or "
            "'high_resolution_lexicographic'"
        )
    epoch_parallel_enabled = optimizer.get(
        "epoch_parallel_enabled",
        False,
    )
    if not isinstance(epoch_parallel_enabled, bool):
        raise TypeError("optimizer.epoch_parallel_enabled must be a JSON boolean")
    proposal_tasks = optimizer.get("proposal_tasks_per_iteration")
    if proposal_tasks is not None and (
        isinstance(proposal_tasks, bool)
        or not isinstance(proposal_tasks, int)
        or proposal_tasks <= 0
    ):
        raise TypeError(
            "optimizer.proposal_tasks_per_iteration must be a positive integer"
        )
    if proposal_tasks is not None and not epoch_parallel_enabled:
        raise ValueError(
            "optimizer.proposal_tasks_per_iteration requires "
            "optimizer.epoch_parallel_enabled=true"
        )
    max_candidate_proposals = optimizer.get("max_candidate_proposals")
    if max_candidate_proposals is not None and (
        isinstance(max_candidate_proposals, bool)
        or not isinstance(max_candidate_proposals, int)
        or max_candidate_proposals <= 0
    ):
        raise TypeError("optimizer.max_candidate_proposals must be a positive integer")
    if max_candidate_proposals is not None and epoch_parallel_enabled:
        raise ValueError(
            "optimizer.max_candidate_proposals requires "
            "optimizer.epoch_parallel_enabled=false"
        )
    max_reflection_workers = optimizer.get("max_reflection_workers", 1)
    if (
        isinstance(max_reflection_workers, bool)
        or not isinstance(max_reflection_workers, int)
        or max_reflection_workers <= 0
    ):
        raise TypeError("optimizer.max_reflection_workers must be a positive integer")
    for name in ("rollout_timeout_seconds", "proposal_timeout_seconds"):
        timeout = optimizer.get(name)
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise TypeError(f"optimizer.{name} must be a positive finite number")
    evaluation_straggler_timeout = optimizer.get(
        "evaluation_straggler_timeout",
        120,
    )
    if (
        isinstance(evaluation_straggler_timeout, bool)
        or not isinstance(evaluation_straggler_timeout, int)
        or evaluation_straggler_timeout < 0
    ):
        raise TypeError(
            "optimizer.evaluation_straggler_timeout must be a non-negative integer"
        )
    dci = optimizer.get("dci")
    if dci is not None:
        dci = _exact_mapping(dci, name="optimizer.dci", keys=DCI_MAPPING_KEYS)
        command = dci["runner_command"]
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(part, str) or not part for part in command)
        ):
            raise TypeError(
                "optimizer.dci.runner_command must be a non-empty string array"
            )
        for name in ("agent_dir", "model", "package_dir", "provider", "tools"):
            if not isinstance(dci[name], str) or not dci[name].strip():
                raise TypeError(f"optimizer.dci.{name} must be a non-empty string")
        max_turns = dci["max_turns"]
        if max_turns is not None and (
            isinstance(max_turns, bool)
            or not isinstance(max_turns, int)
            or max_turns <= 0
        ):
            raise TypeError(
                "optimizer.dci.max_turns must be a positive integer or null"
            )
        evidence_size = dci["proposal_evidence_size"]
        if (
            isinstance(evidence_size, bool)
            or not isinstance(evidence_size, int)
            or evidence_size < 2
        ):
            raise TypeError("optimizer.dci.proposal_evidence_size must be at least 2")
        optimizer["dci"] = dci
    return optimizer


def _minibatch_config_kwargs(
    optimizer: Mapping[str, Any],
) -> dict[str, int | None]:
    return minibatch_config_kwargs(optimizer, namespace="optimizer")


def _method_config_kwargs(optimizer: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve explicitly configured COMPASS method choices without copying GEPA."""

    method = {
        "acceptance_mode": optimizer.get(
            "acceptance_mode",
            "strict_improvement",
        ),
        "epoch_parallel_enabled": optimizer.get(
            "epoch_parallel_enabled",
            False,
        ),
        "evaluation_straggler_timeout": optimizer.get(
            "evaluation_straggler_timeout",
            120,
        ),
        "max_reflection_workers": optimizer.get(
            "max_reflection_workers",
            1,
        ),
        "parent_selection_score_mode": optimizer.get(
            "parent_selection_score_mode",
            "high_resolution",
        ),
        "proposal_tasks_per_iteration": optimizer.get("proposal_tasks_per_iteration"),
    }
    for name in ("rollout_timeout_seconds", "proposal_timeout_seconds"):
        if optimizer.get(name) is not None:
            method[name] = optimizer[name]
    if optimizer.get("max_candidate_proposals") is not None:
        method["max_candidate_proposals"] = optimizer["max_candidate_proposals"]
    raw_dci = optimizer.get("dci")
    if raw_dci is not None:
        from bridge.dci_agent_lite import DciAgentLiteConfig
        from bridge.dci_compass import DciCompassConfig

        method["dci_config"] = DciCompassConfig(
            agent=DciAgentLiteConfig(
                runner_command=tuple(raw_dci["runner_command"]),
                package_dir=Path(raw_dci["package_dir"]),
                agent_dir=Path(raw_dci["agent_dir"]),
                provider=raw_dci["provider"],
                model=raw_dci["model"],
                system_prompt_file=(
                    Path(__file__).resolve().parents[2]
                    / "bridge"
                    / "prompts"
                    / "dci_subproblem_free_text.txt"
                ),
                tools=raw_dci["tools"],
                max_turns=raw_dci["max_turns"],
                run_timeout_seconds=optimizer.get("proposal_timeout_seconds"),
            ),
            proposal_evidence_size=raw_dci["proposal_evidence_size"],
        )
    return method


def load_run_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = _exact_mapping(raw, name="config", keys=ROOT_KEYS)
    config["optimizer"] = _optimizer_mapping(config["optimizer"])
    config["model"] = _model_mapping(config["model"])
    if not isinstance(config["source_snapshot"], Mapping):
        raise TypeError("source_snapshot must be a JSON object")
    if config["condition"] not in (
        "mini_admission_reflection",
        "compass_reflection",
    ):
        raise ValueError("unsupported reflection condition")
    if (
        "proposal_minibatch_size" in config["optimizer"]
        and config["condition"] != "compass_reflection"
    ):
        raise ValueError(
            "split train/validation admission requires condition='compass_reflection'"
        )
    if (
        isinstance(config["optimizer_seed"], bool)
        or not isinstance(config["optimizer_seed"], int)
        or config["optimizer_seed"] < 0
    ):
        raise TypeError("optimizer_seed must be a non-negative integer")
    if config["dataset_mode"] != "lite":
        raise ValueError("paper experiments require dataset_mode='lite'")
    return config


def _require_frozen_protocol(
    config: Mapping[str, Any],
    *,
    budget: int,
) -> None:
    optimizer = config["optimizer"]
    expected = {
        "add_format_failure_as_feedback": False,
        "display_progress_bar": False,
        "evaluation_straggler_timeout": 0,
        "failure_score": 0,
        "max_metric_calls": budget,
        "parent_top_n": 5,
        "perfect_score": 1,
        "raise_on_exception": True,
        "skip_perfect_score": True,
        "track_best_outputs": True,
        "use_cloudpickle": True,
    }
    for key, expected_value in expected.items():
        if optimizer[key] != expected_value:
            raise ValueError(
                f"optimizer.{key} must equal the frozen value {expected_value!r}"
            )
    for key in ("num_threads", "max_candidate_workers"):
        value = optimizer[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TypeError(f"optimizer.{key} must be a positive integer")

    model = config["model"]
    expected_model = {
        "cache": True,
        "cache_in_memory": True,
        "num_retries": 0,
    }
    for key, expected_value in expected_model.items():
        if model[key] != expected_value:
            raise ValueError(
                f"model.{key} must equal the frozen value {expected_value!r}"
            )
    if not isinstance(model["enable_thinking"], bool):
        raise TypeError("model.enable_thinking must be a JSON boolean")
    serving_max_model_len = model.get("serving_max_model_len")
    if serving_max_model_len is not None:
        if isinstance(serving_max_model_len, bool) or not isinstance(
            serving_max_model_len, int
        ):
            raise TypeError("model.serving_max_model_len must be an integer")
        if serving_max_model_len <= model["max_tokens"]:
            raise ValueError(
                "model.serving_max_model_len must exceed model.max_tokens "
                "so non-empty prompts fit in the deployed context window"
            )


def _require_aime_gepa_protocol(config: Mapping[str, Any]) -> None:
    """Reject AIME settings that drift from the official GEPA comparison."""

    expected_optimizer = {
        "max_metric_calls": 1839,
        "num_threads": 32,
    }
    for key, expected_value in expected_optimizer.items():
        if config["optimizer"][key] != expected_value:
            raise ValueError(
                f"AIME GEPA parity requires optimizer.{key}={expected_value!r}"
            )
    if config["optimizer_seed"] != 0:
        raise ValueError("AIME GEPA parity requires optimizer_seed=0")

    if config["model"].get("model") == "openai/gpt-4.1-mini-2025-04-14":
        expected_model = {
            "api_base": "https://api.gpt.ge/v1",
            "enable_thinking": False,
            "max_tokens": 16384,
            "model_type": "chat",
            "num_retries": 0,
            "temperature": 1.0,
        }
    else:
        expected_model = {
            "enable_thinking": True,
            "max_tokens": 16384,
            "model_type": "chat",
            "num_retries": 0,
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
        }
    for key, expected_value in expected_model.items():
        if config["model"].get(key) != expected_value:
            raise ValueError(
                f"AIME GEPA parity requires model.{key}={expected_value!r}"
            )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    try:
        return dict(value)
    except (TypeError, ValueError):
        return repr(value)


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_local_source_snapshot(snapshot: Mapping[str, Any]) -> None:
    """Verify the local semantic files frozen by a mechanism matrix."""
    verify_project_source_snapshot(Path(__file__).resolve().parents[2], snapshot)


def _require_resume_identity(
    previous_manifest: Mapping[str, Any],
    *,
    config_sha256: str,
    task_manifest: Mapping[str, Any],
) -> None:
    """Bind an official GEPA checkpoint to its persisted config and dataset."""

    if previous_manifest.get("config_sha256") != config_sha256:
        raise RuntimeError("persisted manifest/config identity mismatch")
    previous_task = previous_manifest.get("task")
    if not isinstance(previous_task, Mapping):
        raise TypeError("persisted run manifest has no task identity")
    if dict(previous_task) != dict(task_manifest):
        raise RuntimeError("resume dataset identity differs from the persisted run")


def _create_lm(model_config: Mapping[str, Any], *, api_key: str) -> dspy.LM:
    kwargs: dict[str, Any] = {
        "model": model_config["model"],
        "model_type": model_config["model_type"],
        "temperature": model_config["temperature"],
        "max_tokens": model_config["max_tokens"],
        "cache": model_config["cache"],
        "cache_in_memory": model_config["cache_in_memory"],
        "num_retries": model_config["num_retries"],
        "api_key": api_key,
    }
    api_base = model_config.get("api_base")
    if api_base is not None:
        kwargs["api_base"] = api_base
    timeout = model_config.get("timeout")
    if timeout is not None:
        kwargs["timeout"] = timeout
    supports_response_schema = model_config.get("supports_response_schema")
    if supports_response_schema is True:
        kwargs["supports_response_schema"] = True
    top_p = model_config.get("top_p")
    if top_p is not None:
        kwargs["top_p"] = top_p
    extra_body: dict[str, Any] = {}
    top_k = model_config.get("top_k")
    if top_k is not None:
        extra_body["top_k"] = top_k
    repetition_detection = model_config.get("repetition_detection")
    if repetition_detection is not None:
        extra_body["repetition_detection"] = dict(repetition_detection)
    if model_config["enable_thinking"]:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    if extra_body:
        kwargs["extra_body"] = extra_body
    return DeadlineAwareLM(**kwargs)


def _configure_run_cache(
    cache_dir: Path,
    model_config: Mapping[str, Any],
) -> Path:
    """Point DSPy's official cache owner at this run's isolated directory."""

    dspy_cache_dir = cache_dir / ".dspy_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dspy_cache_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "DSPY_CACHEDIR",
        "DSP_CACHEDIR",
        "DSPY_NOTEBOOK_CACHEDIR",
        "DSP_NOTEBOOK_CACHEDIR",
    ):
        os.environ[name] = str(dspy_cache_dir)
    dspy.configure_cache(
        enable_disk_cache=model_config["cache"],
        enable_memory_cache=model_config["cache_in_memory"],
        disk_cache_dir=str(dspy_cache_dir),
    )
    return dspy_cache_dir


def _lm_usage(lm: Any) -> dict[str, float | int]:
    cost = 0.0
    input_tokens = 0
    output_tokens = 0
    for record in getattr(lm, "history", ()):
        if not isinstance(record, Mapping):
            continue
        cost += float(record.get("cost") or 0.0)
        usage = record.get("usage")
        if isinstance(usage, Mapping):
            input_tokens += int(usage.get("prompt_tokens") or 0)
            output_tokens += int(usage.get("completion_tokens") or 0)
    return {
        "cost": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "calls": len(getattr(lm, "history", ())),
    }


def _score_summary(scores: Sequence[Any]) -> dict[str, Any]:
    values = [float(score) for score in scores]
    if not values or any(not math.isfinite(value) for value in values):
        raise RuntimeError("held-out evaluation returned missing or non-finite scores")
    return {
        "aggregate_percent": 100.0 * math.fsum(values) / len(values),
        "count": len(values),
        "scores": values,
    }


def _fraction_record(value: Any) -> dict[str, int]:
    return {
        "numerator": int(value.numerator),
        "denominator": int(value.denominator),
    }


def _owner_selected_candidate_idx(run: Any, state: GEPAState) -> int:
    """Keep final-program ownership with GEPA's evaluation policy."""
    return int(run.evaluation_policy.get_best_program(state))


def _load_benchmark_definition(
    config: Mapping[str, Any],
    *,
    benchmark_family: str,
) -> Any:
    task_id = config["task_id"]
    if benchmark_family == "official":
        specs = load_official_benchmark_specs()
        if task_id not in specs:
            raise ValueError(f"unknown paper task: {task_id!r}")
        return specs[task_id]
    if benchmark_family == "mechanism":
        from bridge.mechanism_benchmark_registry import (
            load_mechanism_benchmark_specs,
        )

        specs = load_mechanism_benchmark_specs()
        if task_id not in specs:
            raise ValueError(f"unknown mechanism task: {task_id!r}")
        return specs[task_id]
    raise ValueError(f"unknown benchmark family: {benchmark_family!r}")


def main(*, benchmark_family: str = "official") -> int:
    from gepa_artifact.utils.metric_logger import (
        CounterWithLock,
        MetricWithLogger,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the exact stored config through GEPA's official run_dir seam",
    )
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = load_run_config(config_path)

    task_id = config["task_id"]
    definition = _load_benchmark_definition(
        config,
        benchmark_family=benchmark_family,
    )
    _verify_local_source_snapshot(config["source_snapshot"])
    budget = (
        definition.max_metric_calls
        if benchmark_family == "official"
        else config["optimizer"]["max_metric_calls"]
    )
    _require_frozen_protocol(config, budget=budget)
    if benchmark_family == "official" and task_id == "aime_2025":
        _require_aime_gepa_protocol(config)

    api_key_env = config["model"]["api_key_env"]
    if not isinstance(api_key_env, str) or not api_key_env:
        raise TypeError("model.api_key_env must be non-empty text")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"required API key environment variable is unset: {api_key_env}"
        )

    cache_dir = Path(config["cache_dir"]).resolve()

    run_dir = Path(config["run_dir"]).resolve()
    previous_manifest: Mapping[str, Any] | None = None
    if args.resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"resume requires the existing run directory: {run_dir}"
            )
        state_path = run_dir / "gepa_state.bin"
        stored_config_path = run_dir / "config.json"
        stored_manifest_path = run_dir / "manifest.json"
        for required in (state_path, stored_config_path, stored_manifest_path):
            if not required.is_file():
                raise FileNotFoundError(
                    f"resume requires the persisted run artifact: {required}"
                )
        stored_config = json.loads(stored_config_path.read_text(encoding="utf-8"))
        if _config_hash(stored_config) != _config_hash(config):
            raise RuntimeError("resume config differs from the persisted run config")
        loaded_manifest = json.loads(stored_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_manifest, Mapping):
            raise TypeError("persisted run manifest must be a JSON object")
        if loaded_manifest.get("status") == "completed":
            raise RuntimeError("completed mechanism runs cannot be resumed")
        previous_manifest = loaded_manifest
    elif run_dir.exists():
        raise FileExistsError(run_dir)

    _configure_run_cache(cache_dir, config["model"])
    start_wall = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    lm = _create_lm(config["model"], api_key=api_key)
    if benchmark_family == "official":
        spec = definition
        splits = instantiate_official_splits(
            spec,
            optimizer_seed=config["optimizer_seed"],
            dataset_mode=config["dataset_mode"],
        )
        owner_provenance: Mapping[str, Any] | None = None
    else:
        from bridge.mechanism_benchmark_registry import (
            resolve_mechanism_benchmark,
        )

        spec = resolve_mechanism_benchmark(
            task_id,
            lm=lm,
            dataset_mode=config["dataset_mode"],
        )
        splits = spec.splits
        owner_provenance = spec.provenance
    owner_thread_cap = spec.benchmark_meta.num_threads
    requested_threads = config["optimizer"]["num_threads"]
    effective_threads = (
        min(requested_threads, owner_thread_cap)
        if owner_thread_cap is not None
        else requested_threads
    )
    task_manifest = {
        "task_id": task_id,
        "display_name": spec.display_name,
        "benchmark_class": spec.benchmark_class_name,
        "program_class": spec.program_class_name,
        "predictor_names": list(spec.predictor_names),
        "dataset_mode": config["dataset_mode"],
        "split_sizes": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
        "split_fingerprints": dict(splits.fingerprints),
    }
    if owner_provenance is not None:
        task_manifest["owner_provenance"] = dict(owner_provenance)
    config_sha256 = _config_hash(config)
    if previous_manifest is not None:
        _require_resume_identity(
            previous_manifest,
            config_sha256=config_sha256,
            task_manifest=task_manifest,
        )
    if not args.resume:
        run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": (
            previous_manifest.get("started_at_utc", started_at)
            if previous_manifest is not None
            else started_at
        ),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "condition": config["condition"],
        "task": task_manifest,
        "optimizer": {
            **dict(config["optimizer"]),
            "effective_num_threads": effective_threads,
            "owner_num_threads_cap": owner_thread_cap,
        },
        "model": {
            key: value for key, value in config["model"].items() if key != "api_key_env"
        },
        "api_key_env_name": api_key_env,
        "cache_dir": str(cache_dir),
        "run_dir": str(run_dir),
        "source_snapshot": dict(config["source_snapshot"]),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    if previous_manifest is not None:
        manifest["resume_count"] = int(previous_manifest.get("resume_count", 0)) + 1
        manifest["resumed_at_utc"] = started_at
    _atomic_write_json(run_dir / "manifest.json", manifest)
    if not args.resume:
        _atomic_write_json(run_dir / "config.json", config)

    chat_adapter = ChatAdapter()
    dspy.configure(lm=lm, adapter=chat_adapter)
    counter = CounterWithLock()
    try:
        with MetricWithLogger(
            metric_fn=spec.benchmark_meta.metric,
            run_dir=str(run_dir),
            counter_with_lock=counter,
            train_dataset=list(splits.train),
            val_dataset=list(splits.validation),
            test_dataset=list(splits.test),
            log_trace=False,
            log_example=False,
            log_prediction=True,
        ) as metric:
            engine_config = CompassReflectionEngineConfig(
                run_dir=run_dir,
                condition=config["condition"],
                seed=config["optimizer_seed"],
                parent_top_n=config["optimizer"]["parent_top_n"],
                max_metric_calls=config["optimizer"]["max_metric_calls"],
                perfect_score=config["optimizer"]["perfect_score"],
                failure_score=config["optimizer"]["failure_score"],
                num_threads=effective_threads,
                max_candidate_workers=config["optimizer"]["max_candidate_workers"],
                skip_perfect_score=config["optimizer"]["skip_perfect_score"],
                add_format_failure_as_feedback=config["optimizer"][
                    "add_format_failure_as_feedback"
                ],
                track_best_outputs=config["optimizer"]["track_best_outputs"],
                display_progress_bar=config["optimizer"]["display_progress_bar"],
                raise_on_exception=config["optimizer"]["raise_on_exception"],
                use_cloudpickle=config["optimizer"]["use_cloudpickle"],
                **_minibatch_config_kwargs(config["optimizer"]),
                **_method_config_kwargs(config["optimizer"]),
            )
            run = run_compass_reflection_engine(
                program=spec.program,
                metric_fn=metric,
                feedback_map=canonical_feedback_map(spec),
                trainset=list(splits.train),
                validation_set=list(splits.validation),
                reflection_lm=lm,
                config=engine_config,
                custom_instruction_proposer=getattr(
                    spec,
                    "custom_instruction_proposer",
                    None,
                ),
            )

            state = GEPAState.load(str(run_dir))
            parent_score_mode = config["optimizer"][
                "parent_selection_score_mode"
            ]
            selected_idx = _owner_selected_candidate_idx(run, state)
            selected_candidate = state.program_candidates[selected_idx]
            selected_frontiers = frontier_count(state, selected_idx)
            selected_exposure = evaluation_count(state, selected_idx)
            selected_high_resolution_credit = high_resolution_frontier_credits(state)[
                selected_idx
            ]
            selected_high_resolution_rate = high_resolution_selection_rate(
                state,
                selected_idx,
            )
            selection = {
                "selected_candidate_idx": selected_idx,
                "selection_rule": (
                    "max(raw_frontier_rate, clean_exposure, -candidate_idx)"
                ),
                "parent_selection_score_mode": parent_score_mode,
                "selection_score": _fraction_record(
                    candidate_selection_rate(
                        state,
                        selected_idx,
                        score_mode="raw_frontier_rate",
                    )
                ),
                "frontier_count": selected_frontiers,
                "clean_exposure": selected_exposure,
                "raw_frontier_rate": (
                    selected_frontiers / selected_exposure if selected_exposure else 0.0
                ),
                "high_resolution_credit": _fraction_record(
                    selected_high_resolution_credit
                ),
                "high_resolution_rate": _fraction_record(selected_high_resolution_rate),
                "high_resolution_rate_float": float(selected_high_resolution_rate),
                "candidate": selected_candidate,
            }
            _atomic_write_json(run_dir / "selected_candidate.json", selection)

            validation_evaluation = run.adapter.evaluate(
                list(splits.validation),
                selected_candidate,
                capture_traces=False,
            )
            test_evaluation = run.adapter.evaluate(
                list(splits.test),
                selected_candidate,
                capture_traces=False,
            )
            completed_at = datetime.now(timezone.utc).isoformat()
            final_result = {
                "schema_version": 1,
                "status": "completed",
                "completed_at_utc": completed_at,
                "condition": config["condition"],
                "task_id": task_id,
                "optimizer_seed": config["optimizer_seed"],
                "selected_candidate_idx": selected_idx,
                "candidate_pool_size": len(state.program_candidates),
                "optimization_iterations": state.i + 1,
                "optimization_metric_calls": state.total_num_evals,
                "validation": _score_summary(validation_evaluation.scores),
                "test": _score_summary(test_evaluation.scores),
                "metric_counter": {
                    "total": counter.step_counter,
                    "train": counter.train_counter,
                    "validation": counter.val_counter,
                    "test": counter.test_counter,
                },
                "lm_usage": _lm_usage(lm),
                "wall_time_seconds": time.time() - start_wall,
            }
            score_breakdown = getattr(spec, "score_breakdown", None)
            if score_breakdown is not None:
                final_result["task_score_breakdown"] = {
                    "validation": score_breakdown(
                        splits.validation,
                        validation_evaluation.scores,
                    ),
                    "test": score_breakdown(
                        splits.test,
                        test_evaluation.scores,
                    ),
                }
            _atomic_write_json(run_dir / "final_result.json", final_result)
            manifest["status"] = "completed"
            manifest["completed_at_utc"] = completed_at
            manifest["final_result"] = "final_result.json"
            _atomic_write_json(run_dir / "manifest.json", manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["exception"] = {
            "class": f"{error.__class__.__module__}.{error.__class__.__qualname__}",
            "message": str(error),
        }
        _atomic_write_json(run_dir / "manifest.json", manifest)
        raise
    finally:
        dspy.configure(lm=None, adapter=None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
