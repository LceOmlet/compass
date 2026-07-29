from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dspy.adapters.chat_adapter import ChatAdapter

from bridge.b03_token_replay import (
    TeacherForcingBatchOutOfMemoryError,
    TokenReplayUtility,
)


class _Tokenizer:
    chat_template = "locked-test-template"
    pad_token_id = 0


class _DeterministicModel:
    def __init__(self, *, fail_on_batch: bool = False) -> None:
        self.config = SimpleNamespace(
            max_position_embeddings=128,
            model_type="qwen3",
            _attn_implementation="flash_attention_2",
        )
        self._input_embeddings = SimpleNamespace(weight=torch.zeros(1))
        self._output_embeddings = SimpleNamespace(weight=torch.zeros(1))
        self.fail_on_batch = fail_on_batch
        self.calls: list[dict[str, torch.Tensor]] = []

    def get_input_embeddings(self) -> Any:
        return self._input_embeddings

    def get_output_embeddings(self) -> Any:
        return self._output_embeddings

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        position_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del use_cache, return_dict
        positions = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .unsqueeze(0)
            .expand_as(input_ids)
            if position_ids is None
            else position_ids
        )
        self.calls.append(
            {
                "input_ids": input_ids.detach().clone(),
                "attention_mask": attention_mask.detach().clone(),
                "position_ids": positions.detach().clone(),
                "logits_to_keep": logits_to_keep.detach().clone(),
            }
        )
        if self.fail_on_batch and input_ids.shape[0] > 1:
            raise torch.OutOfMemoryError("synthetic true-batch OOM")

        selected_tokens = input_ids.index_select(1, logits_to_keep)
        selected_positions = positions.index_select(1, logits_to_keep)
        centers = (selected_tokens + selected_positions) % 32
        vocabulary = torch.arange(32, dtype=torch.float32).view(1, 1, -1)
        logits = -(vocabulary - centers.unsqueeze(-1).float()).abs()
        return SimpleNamespace(logits=logits)


def _replay(model: _DeterministicModel, *, max_batch_size: int) -> TokenReplayUtility:
    return TokenReplayUtility(
        model=model,
        tokenizer=_Tokenizer(),
        adapter=ChatAdapter(),
        max_batch_size=max_batch_size,
    )


def test_true_batch_preserves_candidate_rows_and_token_coordinates() -> None:
    prompt_rows = ((10, 11, 12), (13,), (14, 15))
    rollout_ids = (5, 6, 7)
    credited_positions = (0, 2)
    sequential_model = _DeterministicModel()
    batched_model = _DeterministicModel()

    sequential = _replay(
        sequential_model,
        max_batch_size=1,
    )._packed_selected_log_likelihoods(
        prompt_id_rows=prompt_rows,
        rollout_ids=rollout_ids,
        positions=credited_positions,
    )
    batched = _replay(
        batched_model,
        max_batch_size=3,
    )._packed_selected_log_likelihoods(
        prompt_id_rows=prompt_rows,
        rollout_ids=rollout_ids,
        positions=credited_positions,
    )

    # The deterministic fake proves row/coordinate identity. Real low-precision
    # kernels may use different reductions for batch=1 and batch=3.
    torch.testing.assert_close(
        torch.tensor(batched),
        torch.tensor(sequential),
        rtol=0.0,
        atol=0.0,
    )
    sequential_order = sorted(
        range(len(sequential)),
        key=lambda index: (-sum(sequential[index]), index),
    )
    batched_order = sorted(
        range(len(batched)),
        key=lambda index: (-sum(batched[index]), index),
    )
    assert batched_order == sequential_order
    assert len(sequential_model.calls) == 3
    assert len(batched_model.calls) == 1

    call = batched_model.calls[0]
    assert call["input_ids"].tolist() == [
        [10, 11, 12, 5, 6, 7],
        [0, 0, 13, 5, 6, 7],
        [0, 14, 15, 5, 6, 7],
    ]
    assert call["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1],
    ]
    assert call["position_ids"].tolist() == [
        [0, 1, 2, 3, 4, 5],
        [0, 0, 0, 1, 2, 3],
        [0, 0, 1, 2, 3, 4],
    ]
    assert call["logits_to_keep"].tolist() == [2, 4]


def test_true_batch_computes_loss_one_candidate_row_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _DeterministicModel()
    replay = _replay(model, max_batch_size=3)
    original_cross_entropy = torch.nn.functional.cross_entropy
    input_shapes: list[tuple[int, ...]] = []

    def recorded_cross_entropy(
        input: torch.Tensor,
        target: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        input_shapes.append(tuple(input.shape))
        return original_cross_entropy(input, target, *args, **kwargs)

    monkeypatch.setattr(
        "bridge.b03_token_replay.F.cross_entropy",
        recorded_cross_entropy,
    )

    scores = replay._packed_selected_log_likelihoods(
        prompt_id_rows=((10, 11, 12), (13,), (14, 15)),
        rollout_ids=(5, 6, 7),
        positions=(0, 2),
    )

    assert len(scores) == 3
    assert input_shapes == [(2, 32), (2, 32), (2, 32)]
    assert len(model.calls) == 1
    assert model.calls[0]["input_ids"].shape[0] == 3


def test_true_batch_oom_propagates_without_fallback() -> None:
    model = _DeterministicModel(fail_on_batch=True)
    replay = _replay(model, max_batch_size=3)

    with pytest.raises(
        TeacherForcingBatchOutOfMemoryError,
        match="batch exhausted device memory",
    ) as caught:
        replay._packed_selected_log_likelihoods(
            prompt_id_rows=((10, 11), (12,), (13, 14)),
            rollout_ids=(5, 6),
            positions=(0,),
        )

    assert isinstance(caught.value.__cause__, torch.OutOfMemoryError)
    assert len(model.calls) == 1
    assert model.calls[0]["input_ids"].shape[0] == 3


def test_true_batch_chunks_without_reordering() -> None:
    model = _DeterministicModel()
    replay = _replay(model, max_batch_size=3)

    scores = replay._packed_selected_log_likelihoods(
        prompt_id_rows=((10,), (11,), (12,), (13,)),
        rollout_ids=(5, 6),
        positions=(0,),
    )

    assert len(scores) == 4
    assert [call["input_ids"].shape[0] for call in model.calls] == [3, 1]
