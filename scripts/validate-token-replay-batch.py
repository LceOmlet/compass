from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from dspy.adapters.chat_adapter import ChatAdapter
from flashtrace import load_model_and_tokenizer

from bridge.b03_token_replay import TokenReplayUtility


def _prompt_ids(tokenizer: object, instruction: str) -> tuple[int, ...]:
    encoded = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    f"Instruction: {instruction}\n"
                    "Return the exact answer after reasoning carefully."
                ),
            }
        ],
        chat_template=tokenizer.chat_template,
        tokenize=True,
        add_generation_prompt=True,
        continue_final_message=False,
        enable_thinking=True,
    )
    return tuple(int(token_id) for token_id in encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        str(args.checkpoint.resolve(strict=True)),
        device_map={"": 0},
        dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    prompt_rows = tuple(
        _prompt_ids(tokenizer, instruction)
        for instruction in (
            "Be concise.",
            "Check every condition, then be concise.",
            "First enumerate all applicable constraints, verify them, and then answer concisely.",
        )
    )
    rollout_ids = tuple(
        int(token_id)
        for token_id in tokenizer(
            "We need compare the constraints carefully. The final answer is 42.",
            add_special_tokens=False,
        )["input_ids"]
    )
    if len(rollout_ids) < 3:
        raise RuntimeError("validation rollout tokenization is unexpectedly short")
    credited_positions = (0, len(rollout_ids) // 2, len(rollout_ids) - 1)
    adapter = ChatAdapter()
    sequential = TokenReplayUtility(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        max_batch_size=1,
    )._packed_selected_log_likelihoods(
        prompt_id_rows=prompt_rows,
        rollout_ids=rollout_ids,
        positions=credited_positions,
    )
    batched = TokenReplayUtility(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        max_batch_size=3,
    )._packed_selected_log_likelihoods(
        prompt_id_rows=prompt_rows,
        rollout_ids=rollout_ids,
        positions=credited_positions,
    )

    sequential_tensor = torch.tensor(sequential, dtype=torch.float64)
    batched_tensor = torch.tensor(batched, dtype=torch.float64)
    if not torch.isfinite(sequential_tensor).all():
        raise RuntimeError("batch-one teacher forcing returned a non-finite likelihood")
    if not torch.isfinite(batched_tensor).all():
        raise RuntimeError("true-batch teacher forcing returned a non-finite likelihood")
    sequential_scores = tuple(sum(row) for row in sequential)
    batched_scores = tuple(sum(row) for row in batched)
    sequential_order = tuple(
        sorted(
            range(len(sequential_scores)),
            key=lambda index: (-sequential_scores[index], index),
        )
    )
    batched_order = tuple(
        sorted(
            range(len(batched_scores)),
            key=lambda index: (-batched_scores[index], index),
        )
    )
    diagnostics = {
        "dtype": str(model.get_output_embeddings().weight.dtype),
        "prompt_lengths": [len(row) for row in prompt_rows],
        "rollout_length": len(rollout_ids),
        "credited_positions": list(credited_positions),
        "sequential_likelihoods": sequential,
        "batched_likelihoods": batched,
        "max_absolute_difference": float(
            torch.max(torch.abs(sequential_tensor - batched_tensor))
        ),
        "sequential_scores": sequential_scores,
        "batched_scores": batched_scores,
        "sequential_candidate_order": sequential_order,
        "batched_candidate_order": batched_order,
        "candidate_order_changed": batched_order != sequential_order,
    }
    print(json.dumps(diagnostics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
