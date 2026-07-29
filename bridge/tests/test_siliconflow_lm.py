from __future__ import annotations

from unittest import mock

import dspy
import pytest

from bridge.siliconflow_lm import SiliconFlowLM


def _lm(*, rollout_timeout_seconds: float = 600) -> SiliconFlowLM:
    return SiliconFlowLM(
        "openai/Qwen/Qwen3-8B",
        api_base="https://api.siliconflow.cn/v1",
        api_key="test-key",
        cache=False,
        num_retries=0,
        rollout_timeout_seconds=rollout_timeout_seconds,
    )


def test_rate_limit_retries_the_same_logical_request() -> None:
    messages = [{"role": "user", "content": "same request"}]
    response = object()
    rate_limits = [
        dspy.LMRateLimitError(
            "TPM limit",
            model="openai/Qwen/Qwen3-8B",
            provider="openai",
            status=429,
            retry_after=2.5,
        ),
        dspy.LMRateLimitError(
            "TPM limit",
            model="openai/Qwen/Qwen3-8B",
            provider="openai",
            status=429,
            retry_after=None,
        ),
        dspy.LMRateLimitError(
            "TPM limit",
            model="openai/Qwen/Qwen3-8B",
            provider="openai",
            status=429,
            retry_after=None,
        ),
    ]

    with (
        mock.patch.object(
            dspy.LM,
            "forward",
            autospec=True,
            side_effect=[*rate_limits, response],
        ) as owner_forward,
        mock.patch("bridge.siliconflow_lm.time.sleep") as sleep,
    ):
        assert _lm().forward(messages=messages, rollout_id=17) is response

    assert owner_forward.call_count == 4
    for call in owner_forward.call_args_list:
        assert call.kwargs["messages"] == messages
        assert call.kwargs["rollout_id"] == 17
        assert call.kwargs["timeout"] == 600
    assert sleep.call_args_list == [
        mock.call(2.5),
        mock.call(60.0),
        mock.call(60.0),
    ]


def test_provider_timeout_is_not_retried() -> None:
    timeout = dspy.LMTimeoutError("provider request timed out")

    with (
        mock.patch.object(
            dspy.LM,
            "forward",
            autospec=True,
            side_effect=timeout,
        ) as owner_forward,
        mock.patch("bridge.siliconflow_lm.time.sleep") as sleep,
    ):
        with pytest.raises(dspy.LMTimeoutError) as exc_info:
            _lm().forward(prompt="request")

    assert exc_info.value is timeout
    owner_forward.assert_called_once()
    sleep.assert_not_called()


def test_inner_litellm_retry_must_stay_disabled() -> None:
    with pytest.raises(ValueError, match="num_retries=0"):
        SiliconFlowLM(
            "openai/Qwen/Qwen3-8B",
            rollout_timeout_seconds=600,
            num_retries=1,
        )
