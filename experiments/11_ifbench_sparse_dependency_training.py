from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from flashtrace import load_model_and_tokenizer

from bridge.b14_flashtrace_token_ids import (
    ExactTokenOffloadedFlashTrace,
    ExactTokenOffloadedLLMIFRAttribution,
)
from bridge.b16_official_gepa_ifbench import (
    OfficialIFBenchEngineConfig,
    run_official_ifbench_engine,
)
from bridge.b17_dspy_terminal_analysis import (
    DSPyParentAnalysisBuilder,
    DSPyTerminalAnalysisBridge,
)
from bridge.dspy_token_provenance import (
    ActualDSPyTokenProvenance,
    ExactTraceDataLineageResolver,
)
from bridge.terminal_reflection import TerminalLikelihoodReflectionLM
from gepa_artifact.benchmarks.IFBench import IFBench


CONFIG_KEYS = {
    "deployment": {
        "attn_implementation",
        "checkpoint",
        "device_map",
        "dtype",
        "run_dir",
        "trust_remote_code",
    },
    "remote_lm": {
        "api_base",
        "api_key_env",
        "cache",
        "cache_dir_env",
        "cache_in_memory",
        "enable_thinking",
        "max_tokens",
        "model",
        "model_type",
        "n",
        "num_retries",
        "temperature",
        "top_k",
        "top_p",
    },
    "flashtrace": {
        "chunk_tokens",
        "credit_hops",
        "dependency_hops",
        "recompute_attention",
        "renorm_threshold_default",
        "show_progress",
        "sink_chunk_tokens",
        "use_chat_template",
    },
    "terminal_likelihood": {"n_candidates"},
    "dependency": {"epsilon_dep"},
    "parent_selection": {"top_n"},
    "official_gepa": {
        "add_format_failure_as_feedback",
        "dataset_mode",
        "display_progress_bar",
        "failure_score",
        "max_metric_calls",
        "num_threads",
        "perfect_score",
        "raise_on_exception",
        "reflection_minibatch_size",
        "seed",
        "skip_perfect_score",
        "track_best_outputs",
        "use_cloudpickle",
    },
}


def _exact_mapping(value: Any, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    missing = keys.difference(value)
    extra = set(value).difference(keys)
    if missing or extra:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return dict(value)


def load_config(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = _exact_mapping(raw, "config", set(CONFIG_KEYS))
    return {
        section: _exact_mapping(root[section], section, keys)
        for section, keys in CONFIG_KEYS.items()
    }


def _require_official_configuration(config: dict[str, dict[str, Any]]) -> None:
    remote = config["remote_lm"]
    flashtrace = config["flashtrace"]
    parent_selection = config["parent_selection"]
    official = config["official_gepa"]
    required_remote = {
        "cache": True,
        "cache_in_memory": True,
        "enable_thinking": True,
        "max_tokens": 16384,
        "n": 1,
        "num_retries": 0,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
    }
    required_official = {
        "add_format_failure_as_feedback": False,
        "dataset_mode": "lite",
        "display_progress_bar": False,
        "failure_score": 0,
        "max_metric_calls": 3593,
        "num_threads": None,
        "perfect_score": 1,
        "raise_on_exception": True,
        "reflection_minibatch_size": 3,
        "seed": 0,
        "skip_perfect_score": True,
        "track_best_outputs": True,
    }
    for name, expected in required_remote.items():
        if remote[name] != expected:
            raise ValueError(f"remote_lm.{name} must equal the official value {expected!r}")
    for name, expected in required_official.items():
        if official[name] != expected:
            raise ValueError(
                f"official_gepa.{name} must equal the official value {expected!r}"
            )
    if not isinstance(official["use_cloudpickle"], bool):
        raise TypeError("official_gepa.use_cloudpickle must be a JSON boolean")
    if flashtrace["credit_hops"] != 1 or flashtrace["dependency_hops"] != 1:
        raise ValueError("the configured exact FlashTrace bridges require hops=1")
    if (
        type(parent_selection["top_n"]) is not int
        or parent_selection["top_n"] <= 0
    ):
        raise TypeError("parent_selection.top_n must be a positive JSON integer")


def _official_completion_batch(lm: Any, prompt: str, *, n: int) -> list[str]:
    """Bind ``n`` while delegating output normalization to official DSPy."""

    if not isinstance(prompt, str):
        raise TypeError("the official GEPA reflection renderer must return text")
    view = SimpleNamespace(reflection_lm=lambda value: lm(value, n=n))
    return DspyAdapter.stripped_lm_call(view, prompt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    _require_official_configuration(config)

    deployment = config["deployment"]
    remote = config["remote_lm"]
    flashtrace = config["flashtrace"]
    terminal = config["terminal_likelihood"]
    dependency = config["dependency"]
    official = config["official_gepa"]

    api_key_env = remote["api_key_env"]
    if not isinstance(api_key_env, str) or not api_key_env:
        raise ValueError("remote_lm.api_key_env must be non-empty text")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"required API key environment variable is unset: {api_key_env}")
    cache_dir_env = remote["cache_dir_env"]
    if not isinstance(cache_dir_env, str) or not cache_dir_env:
        raise ValueError("remote_lm.cache_dir_env must be non-empty text")
    cache_dir = os.environ.get(cache_dir_env)
    if not cache_dir:
        raise ValueError(f"required cache directory environment variable is unset: {cache_dir_env}")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = Path(deployment["checkpoint"]).resolve(strict=True)
    run_dir = Path(deployment["run_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    model, tokenizer = load_model_and_tokenizer(
        str(checkpoint),
        device_map=deployment["device_map"],
        dtype=deployment["dtype"],
        trust_remote_code=deployment["trust_remote_code"],
        attn_implementation=deployment["attn_implementation"],
    )
    tracer = ExactTokenOffloadedFlashTrace(
        model,
        tokenizer,
        chunk_tokens=flashtrace["chunk_tokens"],
        sink_chunk_tokens=flashtrace["sink_chunk_tokens"],
        recompute_attention=flashtrace["recompute_attention"],
        use_chat_template=flashtrace["use_chat_template"],
    )
    dependency_attributor = ExactTokenOffloadedLLMIFRAttribution(
        model,
        tokenizer,
        chunk_tokens=flashtrace["chunk_tokens"],
        sink_chunk_tokens=flashtrace["sink_chunk_tokens"],
        renorm_threshold_default=flashtrace["renorm_threshold_default"],
        show_progress=flashtrace["show_progress"],
        recompute_attention=flashtrace["recompute_attention"],
        use_chat_template=flashtrace["use_chat_template"],
    )

    chat_adapter = ChatAdapter()
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
            "chat_template_kwargs": {"enable_thinking": remote["enable_thinking"]},
        },
    )
    dspy.configure(lm=lm, adapter=chat_adapter)

    chat_template_kwargs = {
        "add_generation_prompt": True,
        "continue_final_message": False,
        "enable_thinking": remote["enable_thinking"],
    }
    provenance = ActualDSPyTokenProvenance(
        tokenizer=tokenizer,
        chat_template_kwargs=chat_template_kwargs,
    )
    builder = DSPyParentAnalysisBuilder(
        model=model,
        tracer=tracer,
        attributor=dependency_attributor,
        tokenizer=tokenizer,
        chat_adapter=chat_adapter,
        provenance=provenance,
        lineage_resolver=ExactTraceDataLineageResolver(),
        chat_template_kwargs=chat_template_kwargs,
    )

    benchmark = IFBench(dataset_mode=official["dataset_mode"])
    engine_config = OfficialIFBenchEngineConfig(
        run_dir=run_dir,
        seed=official["seed"],
        reflection_minibatch_size=official["reflection_minibatch_size"],
        parent_top_n=config["parent_selection"]["top_n"],
        max_metric_calls=official["max_metric_calls"],
        perfect_score=official["perfect_score"],
        failure_score=official["failure_score"],
        num_threads=official["num_threads"],
        skip_perfect_score=official["skip_perfect_score"],
        add_format_failure_as_feedback=official["add_format_failure_as_feedback"],
        track_best_outputs=official["track_best_outputs"],
        display_progress_bar=official["display_progress_bar"],
        raise_on_exception=official["raise_on_exception"],
        use_cloudpickle=official["use_cloudpickle"],
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        analysis_bridge = DSPyTerminalAnalysisBridge(
            builder=builder,
            executor=executor,
        )
        terminal_proposal = TerminalLikelihoodReflectionLM(
            complete=lambda prompt, *, n: _official_completion_batch(
                lm,
                prompt,
                n=n,
            ),
            analysis_provider=analysis_bridge,
            n_candidates=terminal["n_candidates"],
            epsilon_dep=dependency["epsilon_dep"],
        )
        run_official_ifbench_engine(
            trainset=benchmark.train_set,
            terminal_proposal=terminal_proposal,
            terminal_analysis_bridge=analysis_bridge,
            config=engine_config,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
