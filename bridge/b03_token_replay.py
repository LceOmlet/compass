from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.nn.functional as F
from dspy.adapters.chat_adapter import ChatAdapter

if TYPE_CHECKING:
    from .b01_aime_capture import CapturedRollout


class TokenCredit(Protocol):
    rollout_token_ids: tuple[int, ...]
    reasoning_positions: tuple[int, ...]
    answer_positions: tuple[int, ...]
    token_weights: tuple[float, ...]
    advantage: Real


class HistoricalReplayContextError(ValueError):
    """A candidate prompt plus one captured old rollout exceeds model context."""


class TeacherForcingBatchOutOfMemoryError(MemoryError):
    """A configured teacher-forcing batch cannot fit on the task-model device."""


class TokenReplayUtility:
    """Evaluate the project token-credit utility with the frozen task model."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        adapter: ChatAdapter,
        max_batch_size: Integral = 1,
    ) -> None:
        chat_template = getattr(tokenizer, "chat_template", None)
        if not isinstance(chat_template, str) or not chat_template:
            raise ValueError("the task tokenizer must expose its locked chat template")
        input_embeddings = model.get_input_embeddings()
        input_device = input_embeddings.weight.device
        if input_device.type == "meta":
            raise ValueError("the task model input embedding has no materialized device")
        output_embeddings = model.get_output_embeddings()
        output_device = output_embeddings.weight.device
        if output_device.type == "meta":
            raise ValueError("the task model output embedding has no materialized device")
        if not isinstance(adapter, ChatAdapter):
            raise TypeError("adapter must be the shared official ChatAdapter instance")
        model_config = getattr(model, "config", None)
        context_window = getattr(model_config, "max_position_embeddings", None)
        if isinstance(context_window, bool) or not isinstance(context_window, int) or context_window <= 0:
            raise TypeError("the task model must expose positive integer max_position_embeddings")
        if getattr(model_config, "model_type", None) != "qwen3":
            raise TypeError("packed replay requires the pinned official Qwen3 task model")
        if getattr(model_config, "_attn_implementation", None) != "flash_attention_2":
            raise RuntimeError("packed replay requires the official FlashAttention2 implementation")
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, Integral)
            or int(max_batch_size) <= 0
        ):
            raise TypeError("max_batch_size must be a positive integer")

        self._model = model
        self._tokenizer = tokenizer
        self._chat_template = chat_template
        self._input_device = input_device
        self._output_device = output_device
        self._adapter = adapter
        self._context_window = context_window
        self._max_batch_size = int(max_batch_size)
        self._old_likelihoods: dict[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
            tuple[float, ...],
        ] = {}

    @staticmethod
    def _integer_tuple(values: Any, field: str) -> tuple[int, ...]:
        result = tuple(values)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
            raise TypeError(f"{field} must contain integer values")
        return result

    def _prompt_ids(self, rollout: CapturedRollout, skill: str) -> tuple[int, ...]:
        signature = rollout.signature
        if getattr(signature, "instructions", None) != rollout.skill_text:
            raise RuntimeError("the captured signature no longer contains the captured original skill")
        with_instructions = getattr(signature, "with_instructions", None)
        if not callable(with_instructions):
            raise TypeError("the captured DSPy signature does not support with_instructions")

        messages = self._adapter.format(
            signature=with_instructions(skill),
            demos=list(rollout.demos),
            inputs=dict(rollout.predictor_inputs),
        )
        encoded = self._tokenizer.apply_chat_template(
            messages,
            chat_template=self._chat_template,
            tokenize=True,
            add_generation_prompt=True,
            continue_final_message=False,
            enable_thinking=True,
        )
        prompt_ids = self._integer_tuple(encoded, "chat-template prompt IDs")
        if not prompt_ids:
            raise ValueError("the rendered task prompt must contain at least one token")
        return prompt_ids

    def _captured_prompt_ids(self, rollout: CapturedRollout) -> tuple[int, ...]:
        prompt_ids = self._integer_tuple(
            rollout.token_record.prompt_token_ids,
            "captured canonical prompt token IDs",
        )
        if not prompt_ids:
            raise ValueError("the captured canonical prompt must contain at least one token")
        return prompt_ids

    @staticmethod
    def _credit_values(
        credit: TokenCredit,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...], float]:
        rollout_ids = TokenReplayUtility._integer_tuple(credit.rollout_token_ids, "rollout_token_ids")
        if not rollout_ids:
            raise ValueError("rollout_token_ids must contain the non-terminal rollout")

        reasoning = TokenReplayUtility._integer_tuple(credit.reasoning_positions, "reasoning_positions")
        answer = TokenReplayUtility._integer_tuple(credit.answer_positions, "answer_positions")
        positions = reasoning + answer
        if not positions or positions != tuple(sorted(positions)) or len(set(positions)) != len(positions):
            raise ValueError("reasoning and answer positions must form one non-empty ordered selection")
        if positions[0] < 0 or positions[-1] >= len(rollout_ids):
            raise IndexError("a selected rollout position is outside rollout_token_ids")

        all_weights = tuple(credit.token_weights)
        if len(all_weights) != len(rollout_ids):
            raise ValueError("token_weights must have one entry per non-terminal rollout token")
        full_weights: list[float] = []
        for value in all_weights:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise TypeError("token_weights must contain finite real numbers")
            if value < 0:
                raise ValueError("token_weights must be non-negative")
            full_weights.append(float(value))
        if not math.isclose(math.fsum(full_weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("token_weights must sum to one")
        if any(full_weights[position] != 0 for position in set(range(len(rollout_ids))).difference(positions)):
            raise ValueError("format-token weights must be zero")

        active_positions: list[int] = []
        weights: list[float] = []
        for position in positions:
            value = full_weights[position]
            if value > 0:
                active_positions.append(position)
                weights.append(value)
        if not active_positions:
            raise ValueError("at least one credited rollout token must have positive weight")

        advantage = credit.advantage
        if isinstance(advantage, bool) or not isinstance(advantage, Real) or not math.isfinite(float(advantage)):
            raise TypeError("advantage must be a finite real number")
        return rollout_ids, tuple(active_positions), tuple(weights), float(advantage)

    def _selected_log_likelihoods(
        self,
        *,
        prompt_ids: tuple[int, ...],
        rollout_ids: tuple[int, ...],
        positions: tuple[int, ...],
    ) -> tuple[float, ...]:
        sequence_ids = prompt_ids + rollout_ids
        if len(sequence_ids) > self._context_window:
            raise HistoricalReplayContextError(
                f"historical teacher-forcing sequence has {len(sequence_ids)} tokens, "
                f"exceeding model context {self._context_window}"
            )
        input_ids = torch.tensor([sequence_ids], dtype=torch.long, device=self._input_device)
        attention_mask = torch.ones_like(input_ids)
        target_indices = torch.tensor(
            [len(prompt_ids) + position for position in positions],
            dtype=torch.long,
            device=self._output_device,
        )
        prediction_indices = target_indices - 1

        with torch.inference_mode():
            output = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=prediction_indices,
                use_cache=False,
                return_dict=True,
            )
            logits = getattr(output, "logits", None)
            if (
                not isinstance(logits, torch.Tensor)
                or logits.ndim != 3
                or logits.shape[:2] != (1, len(positions))
            ):
                raise RuntimeError("the task model forward did not return the requested causal-LM logits")
            targets = torch.tensor(
                [rollout_ids[position] for position in positions],
                dtype=torch.long,
                device=logits.device,
            )
            losses = F.cross_entropy(logits[0].float(), targets, reduction="none")
        return tuple(float(value) for value in (-losses).to(torch.float64).cpu().tolist())

    def _packed_selected_log_likelihoods(
        self,
        *,
        prompt_id_rows: tuple[tuple[int, ...], ...],
        rollout_ids: tuple[int, ...],
        positions: tuple[int, ...],
    ) -> tuple[tuple[float, ...], ...]:
        """Score candidate prompts in bounded batches without changing token coordinates."""

        if not prompt_id_rows:
            raise ValueError("prompt_id_rows must not be empty")
        if not positions:
            raise ValueError("positions must not be empty")
        likelihood_rows: list[tuple[float, ...]] = []
        for start in range(0, len(prompt_id_rows), self._max_batch_size):
            chunk = prompt_id_rows[start : start + self._max_batch_size]
            try:
                if len(chunk) == 1:
                    likelihood_rows.append(
                        self._selected_log_likelihoods(
                            prompt_ids=chunk[0],
                            rollout_ids=rollout_ids,
                            positions=positions,
                        )
                    )
                    continue
                likelihood_rows.extend(
                    self._batched_selected_log_likelihoods(
                        prompt_id_rows=chunk,
                        rollout_ids=rollout_ids,
                        positions=positions,
                    )
                )
            except torch.OutOfMemoryError as error:
                if self._max_batch_size == 1:
                    raise
                raise TeacherForcingBatchOutOfMemoryError(
                    "teacher-forcing batch exhausted device memory; "
                    "the configured batch size is not reduced or retried"
                ) from error
        return tuple(likelihood_rows)

    def _batched_selected_log_likelihoods(
        self,
        *,
        prompt_id_rows: tuple[tuple[int, ...], ...],
        rollout_ids: tuple[int, ...],
        positions: tuple[int, ...],
    ) -> tuple[tuple[float, ...], ...]:
        """Left-pad prompts so every historical rollout keeps its original positions."""

        if len(prompt_id_rows) <= 1:
            raise ValueError("a true replay batch requires at least two prompt rows")
        if len(prompt_id_rows) > self._max_batch_size:
            raise ValueError("prompt batch exceeds max_batch_size")
        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if (
            isinstance(pad_token_id, bool)
            or not isinstance(pad_token_id, Integral)
            or int(pad_token_id) < 0
        ):
            raise TypeError("batched replay requires a non-negative integer pad_token_id")

        max_prompt_length = max(len(prompt_ids) for prompt_ids in prompt_id_rows)
        sequence_length = max_prompt_length + len(rollout_ids)
        for prompt_ids in prompt_id_rows:
            if len(prompt_ids) + len(rollout_ids) > self._context_window:
                raise HistoricalReplayContextError(
                    "historical teacher-forcing sequence has "
                    f"{len(prompt_ids) + len(rollout_ids)} tokens, "
                    f"exceeding model context {self._context_window}"
                )

        input_rows: list[tuple[int, ...]] = []
        attention_rows: list[tuple[int, ...]] = []
        position_rows: list[tuple[int, ...]] = []
        for prompt_ids in prompt_id_rows:
            padding = max_prompt_length - len(prompt_ids)
            content = prompt_ids + rollout_ids
            input_rows.append((int(pad_token_id),) * padding + content)
            attention_rows.append((0,) * padding + (1,) * len(content))
            position_rows.append((0,) * padding + tuple(range(len(content))))
        if any(len(row) != sequence_length for row in input_rows):
            raise RuntimeError("batched replay input rows are misaligned")

        input_ids = torch.tensor(input_rows, dtype=torch.long, device=self._input_device)
        attention_mask = torch.tensor(
            attention_rows,
            dtype=torch.long,
            device=self._input_device,
        )
        position_ids = torch.tensor(
            position_rows,
            dtype=torch.long,
            device=self._input_device,
        )
        target_indices = torch.tensor(
            [max_prompt_length + position for position in positions],
            dtype=torch.long,
            device=self._output_device,
        )
        prediction_indices = target_indices - 1

        with torch.inference_mode():
            output = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                logits_to_keep=prediction_indices,
                use_cache=False,
                return_dict=True,
            )
            logits = getattr(output, "logits", None)
            expected_shape = (len(prompt_id_rows), len(positions))
            if (
                not isinstance(logits, torch.Tensor)
                or logits.ndim != 3
                or logits.shape[:2] != expected_shape
            ):
                raise RuntimeError("the task model forward did not return batched causal-LM logits")
            targets = torch.tensor(
                [rollout_ids[position] for position in positions],
                dtype=torch.long,
                device=logits.device,
            ).expand(len(prompt_id_rows), -1)
            # Preserve the true candidate-batched forward while avoiding one
            # simultaneous FP32 copy of every candidate's selected logits.
            loss_rows = [
                F.cross_entropy(
                    logits[index].float(),
                    targets[index],
                    reduction="none",
                )
                for index in range(len(prompt_id_rows))
            ]
            losses = torch.stack(loss_rows, dim=0)
        return tuple(
            tuple(float(value) for value in row)
            for row in (-losses).to(torch.float64).cpu().tolist()
        )

    def validate_historical_context(self, rollout: CapturedRollout, skill: str) -> None:
        prompt_ids = self._prompt_ids(rollout, skill)
        completion_ids = self._integer_tuple(
            rollout.token_record.completion_token_ids,
            "captured completion token IDs",
        )
        total = len(prompt_ids) + len(completion_ids)
        if total > self._context_window:
            raise HistoricalReplayContextError(
                f"candidate prompt plus captured rollout has {total} tokens, "
                f"exceeding model context {self._context_window}"
            )

    def scores(
        self,
        candidate_skills: Sequence[str],
        rollout_credits: Sequence[tuple[CapturedRollout, TokenCredit]],
    ) -> tuple[float, ...]:
        """Score candidates up to the old-skill constant, which preserves their exact order."""

        candidates = tuple(candidate_skills)
        if not candidates or any(not isinstance(candidate, str) for candidate in candidates):
            raise TypeError("candidate_skills must be a non-empty sequence of text values")
        if not rollout_credits:
            raise ValueError("rollout_credits must not be empty")

        denominator_terms: list[float] = []
        numerator_terms: list[list[float]] = [[] for _ in candidates]
        for rollout, credit in rollout_credits:
            rollout_ids, positions, weights, advantage = self._credit_values(credit)
            captured_completion_ids = tuple(rollout.token_record.completion_token_ids)
            if not captured_completion_ids or captured_completion_ids[:-1] != rollout_ids:
                raise RuntimeError("credit rollout IDs do not equal the captured completion before terminal EOS")
            self._captured_prompt_ids(rollout)

            likelihood_rows = self._packed_selected_log_likelihoods(
                prompt_id_rows=tuple(
                    self._prompt_ids(rollout, candidate)
                    for candidate in candidates
                ),
                rollout_ids=rollout_ids,
                positions=positions,
            )
            for index, likelihoods in enumerate(likelihood_rows):
                weighted = math.fsum(
                    weight * likelihood
                    for weight, likelihood in zip(weights, likelihoods, strict=True)
                )
                numerator_terms[index].append(advantage * weighted)
            denominator_terms.append(abs(advantage))

        denominator = math.fsum(denominator_terms)
        if denominator <= 0:
            raise ValueError("sum(abs(advantage)) must be positive")
        return tuple(math.fsum(terms) / denominator for terms in numerator_terms)

    def utility(
        self,
        candidate_skill: str,
        rollout_credits: Sequence[tuple[CapturedRollout, TokenCredit]],
    ) -> float:
        if not isinstance(candidate_skill, str):
            raise TypeError("candidate_skill must be text")
        if not rollout_credits:
            raise ValueError("rollout_credits must not be empty")

        original_skills = {rollout.skill_text for rollout, _ in rollout_credits}
        if len(original_skills) != 1:
            raise ValueError("one replay utility call requires one common original skill")

        denominator_terms: list[float] = []
        numerator_terms: list[float] = []
        for rollout, credit in rollout_credits:
            rollout_ids, positions, weights, advantage = self._credit_values(credit)
            captured_completion_ids = tuple(rollout.token_record.completion_token_ids)
            if not captured_completion_ids or captured_completion_ids[:-1] != rollout_ids:
                raise RuntimeError("credit rollout IDs do not equal the captured completion before terminal EOS")
            old_prompt_ids = self._captured_prompt_ids(rollout)

            cache_key = (old_prompt_ids, rollout_ids, positions)
            old_likelihoods = self._old_likelihoods.get(cache_key)
            if old_likelihoods is None:
                old_likelihoods = self._selected_log_likelihoods(
                    prompt_ids=old_prompt_ids,
                    rollout_ids=rollout_ids,
                    positions=positions,
                )
                self._old_likelihoods[cache_key] = old_likelihoods

            if candidate_skill == rollout.skill_text:
                candidate_likelihoods = old_likelihoods
            else:
                candidate_prompt_ids = self._prompt_ids(rollout, candidate_skill)
                candidate_likelihoods = self._selected_log_likelihoods(
                    prompt_ids=candidate_prompt_ids,
                    rollout_ids=rollout_ids,
                    positions=positions,
                )

            weighted_delta = math.fsum(
                weight * (candidate - old)
                for weight, candidate, old in zip(weights, candidate_likelihoods, old_likelihoods, strict=True)
            )
            numerator_terms.append(advantage * weighted_delta)
            denominator_terms.append(abs(advantage))

        denominator = math.fsum(denominator_terms)
        if denominator <= 0:
            raise ValueError("sum(abs(advantage)) must be positive")
        return math.fsum(numerator_terms) / denominator
