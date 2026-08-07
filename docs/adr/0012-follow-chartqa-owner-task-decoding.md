# Follow the pinned ChartQA owner decoding for task execution

ChartQA task rollout, validation, and test calls use the pinned LMMS-Eval generation settings (`temperature=0`, `max_new_tokens=16`, `do_sample=False`) through the owner program's official DSPy `Predict.update_config` seam, while proposal and reflection calls use the paper GPT-4.1 Mini profile (`temperature=1`, `max_tokens=16384`). The OpenAI-compatible boundary maps `max_new_tokens` to `max_tokens`, represents deterministic decoding with `temperature=0`, and records rather than forwards the unsupported `do_sample` field.
