from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from dspy.adapters.chat_adapter import ChatAdapter

from .b01_aime_capture import CapturedRollout
from .b02_flashtrace_credit import FlashTraceCredit, build_flashtrace_credit
from .b03_token_replay import TokenReplayUtility
from .b10_flashtrace_offload import OffloadedCaptureLease


FAILURE_SCORE = 0.0
PERFECT_SCORE = 1.0


class AIMEHistoricalReplayScorer:
    """FlashTrace credit and task-model scoring on captured parent rollouts."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        teacher_forcing_replica_model: Any | None,
        teacher_forcing_replica_tokenizer: Any | None,
        tracer: Any,
        chat_adapter: ChatAdapter,
        nonroundtrip_fixture_path: Path,
    ) -> None:
        if not isinstance(chat_adapter, ChatAdapter):
            raise TypeError("chat_adapter must be the shared official ChatAdapter instance")
        if getattr(tracer, "model", None) is not model or getattr(tracer, "tokenizer", None) is not tokenizer:
            raise RuntimeError("FlashTrace must reuse the exact shared model and tokenizer")
        primary_config = getattr(model, "config", None)
        primary_devices = {parameter.device for parameter in model.parameters()}
        if len(primary_devices) != 1 or next(iter(primary_devices)).type != "cuda":
            raise RuntimeError("the primary task model must be fully materialized on one CUDA device")
        single_model = (
            teacher_forcing_replica_model is None
            and teacher_forcing_replica_tokenizer is None
        )
        if (teacher_forcing_replica_model is None) != (
            teacher_forcing_replica_tokenizer is None
        ):
            raise ValueError("the replica model and tokenizer must both be configured or both omitted")
        if not single_model:
            if teacher_forcing_replica_model is model:
                raise RuntimeError("the configured replica must be an independent model instance")
            if teacher_forcing_replica_tokenizer is tokenizer:
                raise RuntimeError("the configured replica tokenizer must come from the second load")
            if tokenizer.get_vocab() != teacher_forcing_replica_tokenizer.get_vocab():
                raise RuntimeError(
                    "the two official task tokenizers do not have the same token-ID vocabulary"
                )
            for field in ("chat_template", "eos_token_id", "pad_token_id"):
                if getattr(tokenizer, field, None) != getattr(
                    teacher_forcing_replica_tokenizer,
                    field,
                    None,
                ):
                    raise RuntimeError(f"the two official task tokenizers differ in {field}")
            replica_config = getattr(teacher_forcing_replica_model, "config", None)
            for field in (
                "model_type",
                "vocab_size",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "max_position_embeddings",
                "_attn_implementation",
            ):
                if getattr(primary_config, field, None) != getattr(replica_config, field, None):
                    raise RuntimeError(f"the two official task models differ in config.{field}")
            replica_devices = {
                parameter.device for parameter in teacher_forcing_replica_model.parameters()
            }
            if len(replica_devices) != 1 or next(iter(replica_devices)).type != "cuda":
                raise RuntimeError(
                    "the teacher-forcing replica must be fully materialized on one CUDA device"
                )
            if not primary_devices.isdisjoint(replica_devices):
                raise RuntimeError("teacher-forcing model replicas must use distinct CUDA devices")
        self._model = model
        self._tokenizer = tokenizer
        self._tracer = tracer
        if not isinstance(nonroundtrip_fixture_path, Path):
            raise TypeError("nonroundtrip_fixture_path must be an explicit pathlib.Path")
        if not nonroundtrip_fixture_path.parent.is_dir():
            raise FileNotFoundError(nonroundtrip_fixture_path.parent)
        self._nonroundtrip_fixture_path = nonroundtrip_fixture_path
        self._nonroundtrip_fixture_captured = nonroundtrip_fixture_path.exists()
        self._eos_token_ids = self._read_eos_token_ids(model)
        self._replay = TokenReplayUtility(model=model, tokenizer=tokenizer, adapter=chat_adapter)
        self._replica_model = teacher_forcing_replica_model
        self._replica_replay = (
            None
            if single_model
            else TokenReplayUtility(
                model=teacher_forcing_replica_model,
                tokenizer=teacher_forcing_replica_tokenizer,
                adapter=chat_adapter,
            )
        )
        self._credit_cache: dict[tuple[Any, ...], FlashTraceCredit] = {}

    def _capture_first_nonroundtrip(self, rollout: CapturedRollout) -> None:
        if self._nonroundtrip_fixture_captured:
            return
        record = rollout.token_record
        encoded = self._tokenizer(
            record.completion_text,
            add_special_tokens=False,
        )["input_ids"]
        canonical_ids = tuple(int(token_id) for token_id in encoded)
        if canonical_ids == record.token_layout.nonterminal_token_ids:
            return
        payload = {
            "instance_index": rollout.instance_index,
            "text_path_failure": (
                "real Arbor completion text re-encodes to different token IDs"
            ),
            "prompt_messages": list(record.prompt_messages),
            "prompt_text": record.prompt_text,
            "prompt_token_ids": list(record.prompt_token_ids),
            "completion_token_ids": list(record.completion_token_ids),
            "completion_text": record.completion_text,
            "native_reasoning_text": record.native_reasoning_text,
            "native_output_text": record.native_output_text,
            "finish_reason": record.finish_reason,
        }
        try:
            with self._nonroundtrip_fixture_path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
        except FileExistsError:
            pass
        self._nonroundtrip_fixture_captured = True

    @staticmethod
    def _read_eos_token_ids(model: Any) -> tuple[int, ...]:
        generation_config = getattr(model, "generation_config", None)
        if generation_config is None:
            raise TypeError("the shared model must expose generation_config")
        configured = getattr(generation_config, "eos_token_id", None)
        if isinstance(configured, bool):
            raise TypeError("model generation_config.eos_token_id must contain integer IDs")
        if isinstance(configured, Integral):
            values = (configured,)
        elif isinstance(configured, (list, tuple)):
            values = tuple(configured)
        else:
            raise TypeError("model generation_config.eos_token_id must be an integer or sequence")
        if not values or any(
            isinstance(token_id, bool) or not isinstance(token_id, Integral) or token_id < 0
            for token_id in values
        ):
            raise ValueError("model generation_config.eos_token_id must contain non-negative integers")
        eos_ids = tuple(int(token_id) for token_id in values)
        if len(set(eos_ids)) != len(eos_ids):
            raise ValueError("model generation_config.eos_token_id must contain distinct IDs")
        return eos_ids

    @staticmethod
    def advantage(reward: Real) -> float:
        if isinstance(reward, bool) or not isinstance(reward, Real) or not math.isfinite(float(reward)):
            raise TypeError("the official reward must be a finite real number")
        return 2.0 * (float(reward) - FAILURE_SCORE) / (PERFECT_SCORE - FAILURE_SCORE) - 1.0

    def _credit(
        self,
        rollout: CapturedRollout,
        advantage: float,
        *,
        capture_lease: OffloadedCaptureLease | None,
    ) -> FlashTraceCredit:
        if rollout.token_record.native_reasoning_text != rollout.native_reasoning_text:
            raise RuntimeError("captured native reasoning differs between rollout and token record")
        if rollout.token_record.native_output_text != rollout.native_output_text:
            raise RuntimeError("captured native output differs between rollout and token record")
        captured_ids = tuple(rollout.token_record.completion_token_ids)
        if not captured_ids or captured_ids[-1] not in self._eos_token_ids:
            raise RuntimeError("the captured rollout must end in one model EOS token ID")
        layout = rollout.token_record.token_layout
        if layout.full_token_ids != captured_ids:
            raise RuntimeError("captured completion IDs differ from the shared Qwen token layout")
        self._capture_first_nonroundtrip(rollout)
        cache_key = (
            tuple(rollout.token_record.prompt_token_ids),
            captured_ids,
            rollout.token_record.completion_text,
            rollout.native_reasoning_text,
            rollout.native_output_text,
            rollout.skill_text,
            advantage,
        )
        cached = self._credit_cache.get(cache_key)
        if cached is not None:
            return cached
        credit = build_flashtrace_credit(
            tracer=self._tracer,
            tokenizer=self._tokenizer,
            prompt_text=rollout.token_record.prompt_text,
            prompt_token_ids=rollout.token_record.prompt_token_ids,
            completion_text=rollout.token_record.completion_text,
            token_layout=layout,
            native_reasoning_text=rollout.native_reasoning_text,
            native_output_text=rollout.native_output_text,
            advantage=advantage,
            capture_lease=capture_lease,
        )
        self._credit_cache[cache_key] = credit
        return credit

    def rollout_credits(
        self,
        minibatch: tuple[CapturedRollout, ...],
    ) -> tuple[tuple[CapturedRollout, FlashTraceCredit], ...]:
        weighted: list[tuple[CapturedRollout, FlashTraceCredit]] = []
        for rollout in minibatch:
            advantage = self.advantage(rollout.reward)
            if advantage != 0.0:
                weighted.append(
                    (
                        rollout,
                        self._credit(
                            rollout,
                            advantage,
                            capture_lease=None,
                        ),
                    )
                )
        if not weighted:
            raise RuntimeError("sum(abs(advantage)) is zero; skip this mutation")
        return tuple(weighted)

    @property
    def replay(self) -> TokenReplayUtility:
        return self._replay

    @property
    def teacher_forcing_replica_replay(self) -> TokenReplayUtility | None:
        return self._replica_replay

    @property
    def model(self) -> Any:
        return self._model

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        return self._eos_token_ids

    def scores(
        self,
        candidate_skills: tuple[str, ...],
        minibatch: tuple[CapturedRollout, ...],
    ) -> tuple[float, ...]:
        return self.scores_from_credits(candidate_skills, self.rollout_credits(minibatch))

    def credit_for_rollout(
        self,
        rollout: CapturedRollout,
        *,
        capture_lease: OffloadedCaptureLease | None,
    ) -> tuple[CapturedRollout, FlashTraceCredit] | None:
        advantage = self.advantage(rollout.reward)
        if advantage == 0.0:
            return None
        return (
            rollout,
            self._credit(
                rollout,
                advantage,
                capture_lease=capture_lease,
            ),
        )

    def scores_from_credits(
        self,
        candidate_skills: tuple[str, ...],
        rollout_credits: tuple[tuple[CapturedRollout, FlashTraceCredit], ...],
    ) -> tuple[float, ...]:
        candidates = tuple(candidate_skills)
        if not candidates:
            raise ValueError("candidate_skills must not be empty")
        if len(candidates) == 1 or self._replica_replay is None:
            return self._replay.scores(candidates, rollout_credits)

        split = (len(candidates) + 1) // 2
        with ThreadPoolExecutor(max_workers=2) as executor:
            primary = executor.submit(
                self._replay.scores,
                candidates[:split],
                rollout_credits,
            )
            replica = executor.submit(
                self._replica_replay.scores,
                candidates[split:],
                rollout_credits,
            )
            primary_scores = primary.result()
            replica_scores = replica.result()
        return primary_scores + replica_scores

    def validate_candidate_context(
        self,
        candidate_skill: str,
        minibatch: tuple[CapturedRollout, ...],
    ) -> None:
        if not isinstance(candidate_skill, str):
            raise TypeError("candidate_skill must be text")
        for rollout in minibatch:
            self._replay.validate_historical_context(rollout, candidate_skill)
