from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import dspy
import torch
from dspy.clients.lm_local_arbor import ArborProvider
from flashtrace import FlashTrace, load_model_and_tokenizer
from flashtrace.attribution import LLMIFRAttribution
from transformers import AutoConfig, AutoTokenizer

from bridge.b01_aime_capture import ArborTokenCallback, ArborTokenRecord
from bridge.b02_flashtrace_credit import (
    _encode_with_offsets,
    _native_completion_layout,
    build_flashtrace_credit,
)
from bridge.b12_qwen_native_aime import (
    QWEN3_MATH_BOXED_INSTRUCTION,
    TOKEN_ID_MODE_LOCAL_CANONICAL,
    TOKEN_ID_MODE_PROVIDER_EXACT,
    QwenNativeAIMEAdapter,
    QwenNativeAIMELMDispatcher,
    build_qwen_native_token_layout,
)
from bridge.b14_flashtrace_token_ids import (
    ExactTokenOffloadedFlashTrace,
    ExactTokenOffloadedLLMIFRAttribution,
    shared_exact_token_capture,
    token_surfaces,
)
from gepa_artifact.benchmarks.AIME import AIMEBench
from gepa_artifact.benchmarks.AIME.AIME_program import program_cot


def _load_existing_parity_helpers() -> ModuleType:
    path = Path(__file__).with_name("07_flashtrace_offload_parity.py")
    spec = importlib.util.spec_from_file_location("_flashtrace_offload_parity_07", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the existing FlashTrace parity helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_PARITY = _load_existing_parity_helpers()
_aggregate_outputs = _PARITY._aggregate_outputs
_compare_numeric = _PARITY._compare_numeric
_measure = _PARITY._measure

# End-to-end numerical parity only: BF16 official execution versus the
# exact-ID + streamed/offloaded bridge.  Training and gates do not use this.
_END_TO_END_RTOL = torch.finfo(torch.bfloat16).eps / 2
_END_TO_END_ATOL = 3e-5


def _call_with_exact_model_input(
    *,
    model: Any,
    expected_ids: Sequence[int],
    call: Callable[[], Any],
) -> Any:
    expected = tuple(int(token_id) for token_id in expected_ids)
    observed: list[tuple[int, ...]] = []

    def capture_input(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise RuntimeError("the exact-ID model forward omitted one [1, sequence] input_ids tensor")
        observed.append(tuple(int(value) for value in input_ids[0].detach().cpu().tolist()))

    handle = model.register_forward_pre_hook(capture_input, with_kwargs=True)
    try:
        result = call()
    finally:
        handle.remove()
    if not observed:
        raise RuntimeError("the exact-ID attribution path did not call the task model")
    if len(observed) != 1:
        raise RuntimeError("the exact-ID attribution path repeated the full model capture")
    if any(actual != expected for actual in observed):
        raise RuntimeError("the task model did not receive the exact captured token-ID sequence")
    return result


def _config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("config must be a JSON object")
    deployment = value.get("deployment")
    native_task = value.get("native_task")
    if not isinstance(deployment, dict) or not isinstance(native_task, dict):
        raise TypeError("config must contain deployment and native_task objects")
    return value


def _dispatcher(
    *,
    config: Mapping[str, Any],
    tokenizer: Any,
    context_window_tokens: int,
    callback: ArborTokenCallback,
    token_id_mode: str,
) -> tuple[QwenNativeAIMELMDispatcher, QwenNativeAIMEAdapter]:
    deployment = config["deployment"]
    native_task = config["native_task"]
    if native_task["enable_thinking"] is not True:
        raise ValueError("native_task.enable_thinking must be true")
    if native_task["math_instruction"] != QWEN3_MATH_BOXED_INSTRUCTION:
        raise ValueError("native_task.math_instruction changed from the pinned Qwen recommendation")
    checkpoint = Path(deployment["checkpoint"]).resolve(strict=True)
    adapter = QwenNativeAIMEAdapter(math_instruction=native_task["math_instruction"])
    dispatcher = QwenNativeAIMELMDispatcher(
        task_tokenizer=tokenizer,
        task_context_window_tokens=context_window_tokens,
        task_competition_max_output_tokens=native_task["competition_max_output_tokens"],
        reflection_max_output_tokens=native_task["competition_max_output_tokens"],
        token_id_mode=token_id_mode,
        task_lm_config={
            "model": f"openai/{checkpoint}",
            "model_type": "chat",
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 16384,
            "cache": deployment["cache"],
            "cache_in_memory": deployment["cache_in_memory"],
            "num_retries": 0,
            "provider": ArborProvider(),
            "api_base": deployment["api_base"],
            "api_key": deployment["api_key"],
            "n": 1,
            "extra_body": {"top_k": 20},
        },
        current_minibatch=lambda: (),
        reflection_call=None,
    )
    dspy.configure(lm=dispatcher, adapter=adapter, callbacks=[callback])
    return dispatcher, adapter


def _capture_real_rollout(
    *,
    index: int,
    trainset: Sequence[Any],
    dispatcher: QwenNativeAIMELMDispatcher,
    adapter: QwenNativeAIMEAdapter,
    callback: ArborTokenCallback,
) -> ArborTokenRecord:
    if index < 0 or index >= len(trainset):
        raise IndexError(f"AIME index {index} is out of range")
    inputs = dict(trainset[index].inputs())
    program = program_cot.deepcopy()
    program.set_lm(dispatcher)
    prediction = program(**inputs)
    messages = adapter.format(program.predict.signature, [], inputs)
    native_output = getattr(prediction, "native_output", None)
    if not isinstance(native_output, str):
        raise RuntimeError("the real Arbor parity Prediction omitted native output")
    return callback.take_for_prediction(messages, native_output)


def _roundtrip_fixture(
    *,
    index: int,
    tokenizer: Any,
    record: ArborTokenRecord,
) -> tuple[dict[str, Any] | None, str | None]:
    prompt = record.prompt_text
    layout = record.token_layout
    try:
        encoded_prompt_ids, _ = _encode_with_offsets(tokenizer, prompt, "real Arbor prompt")
    except RuntimeError as error:
        return None, str(error)
    if encoded_prompt_ids != record.prompt_token_ids:
        return None, "real Arbor prompt text re-encodes to different token IDs"

    try:
        completion_ids, completion_offsets = _encode_with_offsets(
            tokenizer,
            record.completion_text,
            "real Arbor completion",
        )
    except RuntimeError as error:
        return None, str(error)
    if completion_ids != layout.nonterminal_token_ids:
        return None, "real Arbor completion text re-encodes to different token IDs"

    text_layout = _native_completion_layout(
        completion=record.completion_text,
        offsets=completion_offsets,
        native_reasoning=record.native_reasoning_text,
        native_output=record.native_output_text,
    )
    if (
        text_layout.history_positions != layout.history_positions
        or text_layout.reasoning_positions != layout.reasoning_positions
        or text_layout.output_positions != layout.output_positions
    ):
        return None, "official text offsets and exact Qwen token-ID spans differ"

    history_ids = tuple(layout.nonterminal_token_ids[position] for position in layout.history_positions)
    output_ids = tuple(layout.nonterminal_token_ids[position] for position in layout.output_positions)
    history_text = tokenizer.decode(
        list(history_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    credit_prompt = prompt + history_text
    exact_credit_prompt_ids = record.prompt_token_ids + history_ids
    try:
        text_credit_prompt_ids, _ = _encode_with_offsets(
            tokenizer,
            credit_prompt,
            "real Arbor FlashTrace credit prompt",
        )
        text_output_ids, _ = _encode_with_offsets(
            tokenizer,
            record.native_output_text,
            "real Arbor FlashTrace output target",
        )
    except RuntimeError as error:
        return None, str(error)
    if text_credit_prompt_ids != exact_credit_prompt_ids:
        return None, "FlashTrace credit prompt text re-encodes to different token IDs"
    if text_output_ids != output_ids:
        return None, "FlashTrace output target text re-encodes to different token IDs"

    return {
        "index": index,
        "prompt": prompt,
        "prompt_ids": record.prompt_token_ids,
        "target": record.completion_text,
        "generation_ids": layout.full_token_ids,
        "reasoning_span": (layout.reasoning_positions[0], layout.reasoning_positions[-1]),
        "output_span": (layout.output_positions[0], layout.output_positions[-1]),
        "credit_prompt": credit_prompt,
        "credit_prompt_ids": exact_credit_prompt_ids,
        "credit_target": record.native_output_text,
        "credit_generation_ids": output_ids + (layout.full_token_ids[layout.eos_position],),
        "credit_output_span": (0, len(output_ids) - 1),
        "total_tokens": len(record.prompt_token_ids) + len(layout.full_token_ids),
    }, None


def _fixture_payload(
    index: int,
    record: ArborTokenRecord,
    *,
    text_path_failure: str,
) -> dict[str, Any]:
    return {
        "instance_index": index,
        "text_path_failure": text_path_failure,
        "prompt_messages": list(record.prompt_messages),
        "completion_text": record.completion_text,
        "native_reasoning_text": record.native_reasoning_text,
        "native_output_text": record.native_output_text,
        "finish_reason": record.finish_reason,
        "prompt_token_ids": list(record.prompt_token_ids),
        "completion_token_ids": list(record.completion_token_ids),
    }


def _save_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _metadata_shape(value: Any) -> Any:
    if torch.is_tensor(value):
        return ("tensor", str(value.dtype), tuple(value.shape))
    if isinstance(value, Mapping):
        return {key: _metadata_shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return ("list", tuple(_metadata_shape(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_metadata_shape(item) for item in value))
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is not None:
        return (
            type(value).__name__,
            tuple((name, _metadata_shape(getattr(value, name))) for name in fields),
        )
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if value is None:
        return "none"
    return type(value).__name__


def _compare_token_contract(
    *,
    reference: Any,
    candidate: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    generation_ids: Sequence[int],
) -> None:
    if len(reference.prompt_tokens) != len(prompt_ids):
        raise RuntimeError("official text path returned the wrong prompt token count")
    if len(reference.generation_tokens) != len(generation_ids):
        raise RuntimeError("official text path returned the wrong generation token count")
    if tuple(candidate.prompt_tokens) != token_surfaces(tokenizer, prompt_ids):
        raise RuntimeError("exact-ID path changed the one-label-per-prompt-ID contract")
    if tuple(candidate.generation_tokens) != token_surfaces(tokenizer, generation_ids):
        raise RuntimeError("exact-ID path changed the one-label-per-generation-ID contract")
    if _metadata_shape(reference.metadata) != _metadata_shape(candidate.metadata):
        raise RuntimeError("official and exact-ID metadata structures differ")
    reference_ifr = reference.metadata.get("ifr")
    candidate_ifr = candidate.metadata.get("ifr")
    if not isinstance(reference_ifr, Mapping) or not isinstance(candidate_ifr, Mapping):
        raise RuntimeError("official or exact-ID result omitted IFR metadata")
    for key in (
        "type",
        "sink_span_generation",
        "sink_span_absolute",
        "thinking_span_generation",
        "thinking_span_absolute",
        "all_gen_span_generation",
        "all_gen_span_absolute",
        "renorm_threshold",
        "n_hops",
    ):
        if reference_ifr.get(key) != candidate_ifr.get(key):
            raise RuntimeError(f"official and exact-ID IFR metadata differ at {key}")


def _compare_credit(reference: Any, candidate: Any) -> dict[str, float]:
    if (
        reference.output_span != candidate.output_span
        or reference.reasoning_span != candidate.reasoning_span
        or reference.method != candidate.method
    ):
        raise RuntimeError("official and exact-ID FlashTrace spans or method differ")
    score_error = _compare_numeric(
        torch.tensor(reference.scores, dtype=torch.float32),
        torch.tensor(candidate.scores, dtype=torch.float32),
        "credit.scores",
        rtol=_END_TO_END_RTOL,
        atol=_END_TO_END_ATOL,
    )
    raw_error = _compare_numeric(
        _aggregate_outputs(reference.metadata["ifr"]["raw"]),
        _aggregate_outputs(candidate.metadata["ifr"]["raw"]),
        "credit.raw",
        rtol=torch.finfo(torch.bfloat16).eps / 2,
        atol=1e-5,
    )
    return {
        "max_absolute": max(score_error["max_absolute"], raw_error["max_absolute"]),
        "relative_l1": max(score_error["relative_l1"], raw_error["relative_l1"]),
    }


def _compare_dependency(reference: Any, candidate: Any) -> dict[str, float]:
    matrix_error = _compare_numeric(
        reference.attribution_matrix,
        candidate.attribution_matrix,
        "dependency.matrix",
        rtol=_END_TO_END_RTOL,
        atol=_END_TO_END_ATOL,
    )
    raw_error = _compare_numeric(
        _aggregate_outputs(reference.metadata["ifr"]["raw"]),
        _aggregate_outputs(candidate.metadata["ifr"]["raw"]),
        "dependency.raw",
        rtol=torch.finfo(torch.bfloat16).eps / 2,
        atol=1e-5,
    )
    return {
        "max_absolute": max(matrix_error["max_absolute"], raw_error["max_absolute"]),
        "relative_l1": max(matrix_error["relative_l1"], raw_error["relative_l1"]),
    }


def _capture_mode(
    *,
    output: Path,
    tokenizer: Any,
    trainset: Sequence[Any],
    dispatcher: QwenNativeAIMELMDispatcher,
    adapter: QwenNativeAIMEAdapter,
    callback: ArborTokenCallback,
) -> int:
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    for index in range(len(trainset)):
        try:
            record = _capture_real_rollout(
                index=index,
                trainset=trainset,
                dispatcher=dispatcher,
                adapter=adapter,
                callback=callback,
            )
        except ValueError as error:
            print(
                json.dumps(
                    {
                        "invalid_qwen_rollout_aime_index": index,
                        "official_failure_score": 0,
                        "reason": str(error),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue
        _, reason = _roundtrip_fixture(index=index, tokenizer=tokenizer, record=record)
        if reason is None:
            print(json.dumps({"roundtrip_aime_index": index}), flush=True)
            continue
        try:
            _save_exclusive(
                output,
                _fixture_payload(index, record, text_path_failure=reason),
            )
        except FileExistsError:
            print(
                json.dumps({"existing_first_nonroundtrip": str(output)}),
                flush=True,
            )
            return 0
        print(
            json.dumps(
                {
                    "saved_first_nonroundtrip": str(output),
                    "aime_index": index,
                    "text_path_failure": reason,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    raise RuntimeError("all real AIME train rollouts were roundtrippable; no fixture was written")


def _replay_nonroundtrip_mode(
    *,
    fixture_path: Path,
    model: Any,
    tokenizer: Any,
) -> int:
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("the non-roundtrip fixture must be a JSON object")
    required = {
        "instance_index",
        "text_path_failure",
        "prompt_messages",
        "completion_text",
        "native_reasoning_text",
        "native_output_text",
        "finish_reason",
        "prompt_token_ids",
        "completion_token_ids",
    }
    if not required.issubset(payload):
        missing = sorted(required.difference(payload))
        raise ValueError(f"the non-roundtrip fixture omitted fields: {missing}")
    if payload["finish_reason"] != "stop":
        raise ValueError("the replay fixture must contain one completed Arbor rollout")
    if not isinstance(payload["instance_index"], int) or isinstance(payload["instance_index"], bool):
        raise TypeError("fixture instance_index must be an integer")
    if not isinstance(payload["text_path_failure"], str) or not payload["text_path_failure"]:
        raise TypeError("fixture text_path_failure must be non-empty text")
    prompt_messages_value = payload["prompt_messages"]
    if not isinstance(prompt_messages_value, list) or not prompt_messages_value:
        raise TypeError("fixture prompt_messages must be one non-empty list")
    prompt_messages: tuple[dict[str, Any], ...] = tuple(
        dict(message) for message in prompt_messages_value
    )
    for message in prompt_messages:
        if set(message) != {"role", "content"} or not all(
            isinstance(message[key], str) for key in ("role", "content")
        ):
            raise TypeError("fixture prompt messages must contain textual role and content")

    def integer_ids(field: str) -> tuple[int, ...]:
        value = payload[field]
        if not isinstance(value, list) or not value:
            raise TypeError(f"fixture {field} must be one non-empty ID list")
        if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in value):
            raise TypeError(f"fixture {field} must contain only integer token IDs")
        return tuple(value)

    prompt_ids = integer_ids("prompt_token_ids")
    completion_ids = integer_ids("completion_token_ids")
    completion_text = payload["completion_text"]
    reasoning_text = payload["native_reasoning_text"]
    output_text = payload["native_output_text"]
    if not all(isinstance(value, str) and value for value in (completion_text, reasoning_text, output_text)):
        raise TypeError("fixture completion, reasoning, and output fields must be non-empty text")
    layout = build_qwen_native_token_layout(
        tokenizer=tokenizer,
        completion_token_ids=completion_ids,
        raw_completion_text=completion_text,
        native_reasoning_text=reasoning_text,
        native_output_text=output_text,
    )
    prompt_text = payload.get("prompt_text")
    if prompt_text is None:
        prompt_text = tokenizer.apply_chat_template(
            [dict(message) for message in prompt_messages],
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
            enable_thinking=True,
        )
    if not isinstance(prompt_text, str) or not prompt_text:
        raise TypeError("fixture prompt_text must be non-empty text")
    record = ArborTokenRecord(
        prompt_messages=prompt_messages,
        prompt_text=prompt_text,
        completion_text=completion_text,
        native_reasoning_text=reasoning_text,
        native_output_text=output_text,
        finish_reason=payload["finish_reason"],
        prompt_token_ids=prompt_ids,
        completion_token_ids=completion_ids,
        token_layout=layout,
    )
    _, replay_failure = _roundtrip_fixture(
        index=int(payload["instance_index"]),
        tokenizer=tokenizer,
        record=record,
    )
    if replay_failure is None:
        raise RuntimeError("the requested fixture is roundtrippable and is not a non-roundtrip case")
    if replay_failure != payload["text_path_failure"]:
        raise RuntimeError("the saved and replayed text-interface failures differ")

    credit = ExactTokenOffloadedFlashTrace(
        model,
        tokenizer,
        chunk_tokens=128,
        sink_chunk_tokens=32,
        recompute_attention=True,
        use_chat_template=False,
    )
    dependency = ExactTokenOffloadedLLMIFRAttribution(
        model,
        tokenizer,
        chunk_tokens=128,
        sink_chunk_tokens=32,
        renorm_threshold_default=0.0,
        show_progress=False,
        recompute_attention=True,
        use_chat_template=False,
    )
    expected_model_ids = prompt_ids + completion_ids
    credit_result, credit_perf = _measure(
        lambda: _call_with_exact_model_input(
            model=model,
            expected_ids=expected_model_ids,
            call=lambda: build_flashtrace_credit(
                tracer=credit,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                prompt_token_ids=prompt_ids,
                completion_text=completion_text,
                token_layout=layout,
                native_reasoning_text=reasoning_text,
                native_output_text=output_text,
                advantage=1.0,
                capture_lease=None,
            ),
        )
    )
    if credit_result.rollout_token_ids != layout.nonterminal_token_ids:
        raise RuntimeError("exact credit changed the captured non-EOS rollout IDs")
    if not torch.isfinite(torch.tensor(credit_result.token_weights)).all():
        raise RuntimeError("exact credit returned non-finite token weights")

    dependency_result, dependency_perf = _measure(
        lambda: _call_with_exact_model_input(
            model=model,
            expected_ids=expected_model_ids,
            call=lambda: dependency.calculate_ifr_multi_hop_ids(
                prompt_text,
                prompt_ids=prompt_ids,
                target=completion_text,
                generation_ids=layout.full_token_ids,
                sink_span=(layout.output_positions[0], layout.output_positions[-1]),
                thinking_span=(layout.reasoning_positions[0], layout.reasoning_positions[-1]),
                n_hops=1,
                renorm_threshold=0.0,
                observation_mask=None,
                capture_lease=None,
            ),
        )
    )
    if tuple(dependency_result.prompt_tokens) != token_surfaces(tokenizer, prompt_ids):
        raise RuntimeError("exact dependency changed the captured prompt-ID coordinates")
    if tuple(dependency_result.generation_tokens) != token_surfaces(tokenizer, completion_ids):
        raise RuntimeError("exact dependency changed the captured completion-ID coordinates")
    dependency_matrix = torch.as_tensor(dependency_result.attribution_matrix)
    if not torch.isfinite(dependency_matrix).all():
        raise RuntimeError("exact dependency returned non-finite attribution")
    metadata = dependency_result.metadata
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("ifr"), Mapping):
        raise RuntimeError("exact dependency omitted official IFR metadata")
    if metadata["ifr"].get("n_hops") != 1:
        raise RuntimeError("exact dependency did not execute the requested one-hop path")

    report = {
        "fixture": str(fixture_path),
        "aime_index": payload["instance_index"],
        "nonroundtrip": True,
        "official_text_interface_called": False,
        "captured_model_input_tokens": len(expected_model_ids),
        "model_input_exact": {"credit": True, "dependency": True},
        "credit": {
            "reasoning_tokens": len(credit_result.reasoning_positions),
            "answer_tokens": len(credit_result.answer_positions),
            "token_weight_sum": float(sum(credit_result.token_weights)),
            "performance": credit_perf,
        },
        "dependency": {
            "matrix_shape": list(dependency_matrix.shape),
            "n_hops": 1,
            "performance": dependency_perf,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


def _parity_mode(
    *,
    fixtures: Sequence[Mapping[str, Any]],
    model: Any,
    tokenizer: Any,
) -> int:
    official_credit = FlashTrace(
        model,
        tokenizer,
        chunk_tokens=128,
        sink_chunk_tokens=32,
        recompute_attention=True,
        use_chat_template=False,
    )
    exact_credit = ExactTokenOffloadedFlashTrace(
        model,
        tokenizer,
        chunk_tokens=128,
        sink_chunk_tokens=32,
        recompute_attention=True,
        use_chat_template=False,
    )
    official_dependency = LLMIFRAttribution(
        model,
        tokenizer,
        chunk_tokens=128,
        sink_chunk_tokens=32,
        renorm_threshold_default=0.0,
        show_progress=False,
        recompute_attention=True,
        use_chat_template=False,
    )
    exact_dependency = ExactTokenOffloadedLLMIFRAttribution(
        model,
        tokenizer,
        chunk_tokens=128,
        sink_chunk_tokens=32,
        renorm_threshold_default=0.0,
        show_progress=False,
        recompute_attention=True,
        use_chat_template=False,
    )

    report: list[dict[str, Any]] = []
    for fixture in fixtures:
        official_credit_result, official_credit_perf = _measure(
            lambda fixture=fixture: official_credit.trace(
                prompt=fixture["credit_prompt"],
                target=fixture["credit_target"],
                output_span=fixture["credit_output_span"],
                hops=1,
                method="flashtrace",
                renorm_threshold=0.0,
            )
        )
        exact_credit_result, exact_credit_perf = _measure(
            lambda fixture=fixture: _call_with_exact_model_input(
                model=model,
                expected_ids=fixture["credit_prompt_ids"] + fixture["credit_generation_ids"],
                call=lambda: exact_credit.trace_ids(
                    prompt=fixture["credit_prompt"],
                    prompt_ids=fixture["credit_prompt_ids"],
                    target=fixture["credit_target"],
                    generation_ids=fixture["credit_generation_ids"],
                    output_span=fixture["credit_output_span"],
                    hops=1,
                    method="flashtrace",
                    renorm_threshold=0.0,
                    capture_lease=None,
                ),
            )
        )
        _compare_token_contract(
            reference=official_credit_result,
            candidate=exact_credit_result,
            tokenizer=tokenizer,
            prompt_ids=fixture["credit_prompt_ids"],
            generation_ids=fixture["credit_generation_ids"],
        )
        credit_error = _compare_credit(official_credit_result, exact_credit_result)

        official_dependency_result, official_dependency_perf = _measure(
            lambda fixture=fixture: official_dependency.calculate_ifr_multi_hop(
                fixture["prompt"],
                target=fixture["target"],
                sink_span=fixture["output_span"],
                thinking_span=fixture["reasoning_span"],
                n_hops=1,
                renorm_threshold=0.0,
                observation_mask=None,
            )
        )
        exact_dependency_result, exact_dependency_perf = _measure(
            lambda fixture=fixture: _call_with_exact_model_input(
                model=model,
                expected_ids=fixture["prompt_ids"] + fixture["generation_ids"],
                call=lambda: exact_dependency.calculate_ifr_multi_hop_ids(
                    fixture["prompt"],
                    prompt_ids=fixture["prompt_ids"],
                    target=fixture["target"],
                    generation_ids=fixture["generation_ids"],
                    sink_span=fixture["output_span"],
                    thinking_span=fixture["reasoning_span"],
                    n_hops=1,
                    renorm_threshold=0.0,
                    observation_mask=None,
                    capture_lease=None,
                ),
            )
        )
        _compare_token_contract(
            reference=official_dependency_result,
            candidate=exact_dependency_result,
            tokenizer=tokenizer,
            prompt_ids=fixture["prompt_ids"],
            generation_ids=fixture["generation_ids"],
        )
        dependency_error = _compare_dependency(
            official_dependency_result,
            exact_dependency_result,
        )

        credit_sink_length = (
            fixture["credit_output_span"][1]
            - fixture["credit_output_span"][0]
            + 1
        )

        def uniform_credit_call(
            fixture: Mapping[str, Any] = fixture,
        ) -> tuple[Any, Any]:
            with shared_exact_token_capture(model) as capture_lease:
                unweighted = exact_credit.trace_ids(
                    prompt=fixture["credit_prompt"],
                    prompt_ids=fixture["credit_prompt_ids"],
                    target=fixture["credit_target"],
                    generation_ids=fixture["credit_generation_ids"],
                    output_span=fixture["credit_output_span"],
                    hops=1,
                    method="flashtrace",
                    renorm_threshold=0.0,
                    capture_lease=capture_lease,
                )
                with exact_credit.weighted_sink_scope((1.0,) * credit_sink_length):
                    weighted = exact_credit.trace_ids(
                        prompt=fixture["credit_prompt"],
                        prompt_ids=fixture["credit_prompt_ids"],
                        target=fixture["credit_target"],
                        generation_ids=fixture["credit_generation_ids"],
                        output_span=fixture["credit_output_span"],
                        hops=1,
                        method="flashtrace",
                        renorm_threshold=0.0,
                        capture_lease=capture_lease,
                    )
            return unweighted, weighted

        (uniform_credit_reference, uniform_credit_weighted), uniform_credit_perf = _measure(
            lambda: _call_with_exact_model_input(
                model=model,
                expected_ids=(
                    fixture["credit_prompt_ids"]
                    + fixture["credit_generation_ids"]
                ),
                call=uniform_credit_call,
            )
        )
        uniform_credit_error = _compare_credit(
            uniform_credit_reference,
            uniform_credit_weighted,
        )

        dependency_sink_length = fixture["output_span"][1] - fixture["output_span"][0] + 1

        def uniform_dependency_call(
            fixture: Mapping[str, Any] = fixture,
        ) -> tuple[Any, Any]:
            with shared_exact_token_capture(model) as capture_lease:
                unweighted = exact_dependency.calculate_ifr_multi_hop_ids(
                    fixture["prompt"],
                    prompt_ids=fixture["prompt_ids"],
                    target=fixture["target"],
                    generation_ids=fixture["generation_ids"],
                    sink_span=fixture["output_span"],
                    thinking_span=fixture["reasoning_span"],
                    n_hops=1,
                    renorm_threshold=0.0,
                    observation_mask=None,
                    capture_lease=capture_lease,
                )
                weighted = exact_dependency.calculate_ifr_multi_hop_weighted_sink_ids(
                    fixture["prompt"],
                    prompt_ids=fixture["prompt_ids"],
                    target=fixture["target"],
                    generation_ids=fixture["generation_ids"],
                    sink_span=fixture["output_span"],
                    thinking_span=fixture["reasoning_span"],
                    n_hops=1,
                    renorm_threshold=0.0,
                    observation_mask=None,
                    capture_lease=capture_lease,
                    sink_weights=(1.0,) * dependency_sink_length,
                )
            return unweighted, weighted

        (
            uniform_dependency_reference,
            uniform_dependency_weighted,
        ), uniform_dependency_perf = _measure(
            lambda: _call_with_exact_model_input(
                model=model,
                expected_ids=fixture["prompt_ids"] + fixture["generation_ids"],
                call=uniform_dependency_call,
            )
        )
        uniform_dependency_error = _compare_dependency(
            uniform_dependency_reference,
            uniform_dependency_weighted,
        )

        full_ids = fixture["prompt_ids"] + fixture["generation_ids"]
        if fixture["credit_prompt_ids"] + fixture["credit_generation_ids"] != full_ids:
            raise RuntimeError("credit and dependency views do not share one full token sequence")

        def shared_call(fixture: Mapping[str, Any] = fixture) -> tuple[Any, Any]:
            with shared_exact_token_capture(model) as capture_lease:
                shared_credit = exact_credit.trace_ids(
                    prompt=fixture["credit_prompt"],
                    prompt_ids=fixture["credit_prompt_ids"],
                    target=fixture["credit_target"],
                    generation_ids=fixture["credit_generation_ids"],
                    output_span=fixture["credit_output_span"],
                    hops=1,
                    method="flashtrace",
                    renorm_threshold=0.0,
                    capture_lease=capture_lease,
                )
                shared_dependency = exact_dependency.calculate_ifr_multi_hop_ids(
                    fixture["prompt"],
                    prompt_ids=fixture["prompt_ids"],
                    target=fixture["target"],
                    generation_ids=fixture["generation_ids"],
                    sink_span=fixture["output_span"],
                    thinking_span=fixture["reasoning_span"],
                    n_hops=1,
                    renorm_threshold=0.0,
                    observation_mask=None,
                    capture_lease=capture_lease,
                )
            return shared_credit, shared_dependency

        (shared_credit_result, shared_dependency_result), shared_perf = _measure(
            lambda: _call_with_exact_model_input(
                model=model,
                expected_ids=full_ids,
                call=shared_call,
            )
        )
        shared_credit_error = _compare_credit(exact_credit_result, shared_credit_result)
        shared_dependency_error = _compare_dependency(
            exact_dependency_result,
            shared_dependency_result,
        )
        report.append(
            {
                "aime_index": fixture["index"],
                "total_tokens": fixture["total_tokens"],
                "credit": {
                    "official_text": official_credit_perf,
                    "exact_ids": exact_credit_perf,
                    "error": credit_error,
                },
                "dependency": {
                    "official_text": official_dependency_perf,
                    "exact_ids": exact_dependency_perf,
                    "error": dependency_error,
                },
                "shared_parent_capture": {
                    "separate_forward_calls": 2,
                    "shared_forward_calls": 1,
                    "separate_seconds": (
                        exact_credit_perf["seconds"] + exact_dependency_perf["seconds"]
                    ),
                    "shared": shared_perf,
                    "credit_error": shared_credit_error,
                    "dependency_error": shared_dependency_error,
                },
                "uniform_weighted_sink": {
                    "credit": {
                        "performance": uniform_credit_perf,
                        "error": uniform_credit_error,
                    },
                    "dependency": {
                        "performance": uniform_dependency_perf,
                        "error": uniform_dependency_error,
                    },
                },
            }
        )
        print(json.dumps({"measurement": report[-1]}, ensure_ascii=False), flush=True)
    print(json.dumps({"results": report}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--parity-index", action="append", type=int)
    mode.add_argument("--capture-first-nonroundtrip", type=Path)
    mode.add_argument("--replay-nonroundtrip", type=Path)
    mode.add_argument("--canonical-rollout-index", type=int)
    mode.add_argument("--canonical-parity-index", type=int)
    args = parser.parse_args()

    config = _config(args.config)
    deployment = config["deployment"]
    checkpoint = Path(deployment["checkpoint"]).resolve(strict=True)
    needs_model = (
        args.parity_index is not None
        or args.replay_nonroundtrip is not None
        or args.canonical_parity_index is not None
    )
    if not needs_model:
        model = None
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint),
            trust_remote_code=True,
            local_files_only=True,
        )
        model_config = AutoConfig.from_pretrained(
            str(checkpoint),
            trust_remote_code=True,
            local_files_only=True,
        )
    else:
        model, tokenizer = load_model_and_tokenizer(
            str(checkpoint),
            device_map=deployment["device_map"],
            dtype="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation=deployment["attn_implementation"],
        )
        model_config = model.config
    context_window_tokens = getattr(model_config, "max_position_embeddings", None)
    if isinstance(context_window_tokens, bool) or not isinstance(context_window_tokens, int):
        raise TypeError("the Qwen task model must expose integer max_position_embeddings")

    if args.replay_nonroundtrip is not None:
        assert model is not None
        return _replay_nonroundtrip_mode(
            fixture_path=args.replay_nonroundtrip,
            model=model,
            tokenizer=tokenizer,
        )

    benchmark = AIMEBench(dataset_mode="lite")
    trainset = benchmark.train_set
    token_id_mode = (
        TOKEN_ID_MODE_LOCAL_CANONICAL
        if args.canonical_rollout_index is not None or args.canonical_parity_index is not None
        else TOKEN_ID_MODE_PROVIDER_EXACT
    )
    callback = ArborTokenCallback(
        tokenizer=tokenizer,
        token_id_mode=token_id_mode,
    )
    dispatcher, adapter = _dispatcher(
        config=config,
        tokenizer=tokenizer,
        context_window_tokens=context_window_tokens,
        callback=callback,
        token_id_mode=token_id_mode,
    )

    if args.canonical_parity_index is not None:
        record = _capture_real_rollout(
            index=args.canonical_parity_index,
            trainset=trainset,
            dispatcher=dispatcher,
            adapter=adapter,
            callback=callback,
        )
        fixture, reason = _roundtrip_fixture(
            index=args.canonical_parity_index,
            tokenizer=tokenizer,
            record=record,
        )
        if fixture is None:
            raise RuntimeError(f"canonical AIME rollout is not locally roundtrippable: {reason}")
        assert model is not None
        return _parity_mode(fixtures=[fixture], model=model, tokenizer=tokenizer)

    if args.canonical_rollout_index is not None:
        record = _capture_real_rollout(
            index=args.canonical_rollout_index,
            trainset=trainset,
            dispatcher=dispatcher,
            adapter=adapter,
            callback=callback,
        )
        expected_prompt_ids = tuple(
            tokenizer.apply_chat_template(
                list(record.prompt_messages),
                tokenize=True,
                add_generation_prompt=True,
                continue_final_message=False,
                enable_thinking=True,
            )
        )
        if record.prompt_token_ids != expected_prompt_ids:
            raise RuntimeError("canonical rollout changed the locally rendered prompt IDs")
        if record.completion_token_ids != record.token_layout.full_token_ids:
            raise RuntimeError("canonical rollout completion IDs and token layout disagree")
        print(
            json.dumps(
                {
                    "aime_index": args.canonical_rollout_index,
                    "token_id_mode": TOKEN_ID_MODE_LOCAL_CANONICAL,
                    "prompt_tokens": len(record.prompt_token_ids),
                    "completion_tokens": len(record.completion_token_ids),
                    "reasoning_tokens": len(record.token_layout.reasoning_positions),
                    "output_tokens": len(record.token_layout.output_positions),
                    "finish_reason": record.finish_reason,
                }
            )
        )
        return 0

    if args.parity_index is None:
        return _capture_mode(
            output=args.capture_first_nonroundtrip,
            tokenizer=tokenizer,
            trainset=trainset,
            dispatcher=dispatcher,
            adapter=adapter,
            callback=callback,
        )

    if len(set(args.parity_index)) != len(args.parity_index):
        raise ValueError("--parity-index values must be distinct")
    fixtures: list[dict[str, Any]] = []
    for index in args.parity_index:
        record = _capture_real_rollout(
            index=index,
            trainset=trainset,
            dispatcher=dispatcher,
            adapter=adapter,
            callback=callback,
        )
        fixture, reason = _roundtrip_fixture(index=index, tokenizer=tokenizer, record=record)
        if fixture is None:
            raise RuntimeError(f"AIME index {index} cannot use official text parity: {reason}")
        fixtures.append(fixture)
    assert model is not None
    return _parity_mode(fixtures=fixtures, model=model, tokenizer=tokenizer)


if __name__ == "__main__":
    raise SystemExit(main())
