from __future__ import annotations

import logging
import math
import time
from typing import Any

import dspy

_LOGGER = logging.getLogger(__name__)
_DEFAULT_RATE_LIMIT_WAIT_SECONDS = 60.0


class SiliconFlowLM(dspy.LM):
    """Keep one unchanged SiliconFlow rollout alive across rate-limit responses."""

    def __init__(
        self,
        *args: Any,
        rollout_timeout_seconds: float,
        num_retries: int = 0,
        **kwargs: Any,
    ) -> None:
        if (
            isinstance(rollout_timeout_seconds, bool)
            or not isinstance(rollout_timeout_seconds, (int, float))
            or not math.isfinite(rollout_timeout_seconds)
            or rollout_timeout_seconds <= 0
        ):
            raise TypeError("rollout_timeout_seconds must be a positive finite number")
        if num_retries != 0:
            raise ValueError(
                "SiliconFlowLM requires num_retries=0 so LiteLLM does not duplicate "
                "the rate-limit keepalive"
            )
        super().__init__(*args, num_retries=0, **kwargs)
        self.rollout_timeout_seconds = float(rollout_timeout_seconds)

    @staticmethod
    def _rate_limit_wait_seconds(error: dspy.LMRateLimitError) -> float:
        retry_after = error.retry_after
        if (
            isinstance(retry_after, (int, float))
            and not isinstance(retry_after, bool)
            and math.isfinite(retry_after)
            and retry_after > 0
        ):
            return float(retry_after)
        return _DEFAULT_RATE_LIMIT_WAIT_SECONDS

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        request_kwargs = dict(kwargs)
        request_kwargs["timeout"] = self.rollout_timeout_seconds

        while True:
            try:
                return super().forward(
                    prompt=prompt,
                    messages=messages,
                    **request_kwargs,
                )
            except dspy.LMRateLimitError as error:
                wait_seconds = self._rate_limit_wait_seconds(error)
                _LOGGER.warning(
                    "SiliconFlow rate limit; keeping the same logical rollout alive "
                    "and retrying in %.3fs",
                    wait_seconds,
                )
                time.sleep(wait_seconds)
