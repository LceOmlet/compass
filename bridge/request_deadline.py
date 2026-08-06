from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar, Token
from time import monotonic
from typing import Any, Iterator

import dspy


_REQUEST_DEADLINE: ContextVar[float | None] = ContextVar(
    "compass_request_deadline",
    default=None,
)


def _positive_seconds(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise TypeError(f"{name} must be a positive finite number")
    return float(value)


@contextmanager
def request_deadline(seconds: int | float) -> Iterator[None]:
    """Apply one monotonic wall-clock budget to all nested LM calls."""

    duration = _positive_seconds(seconds, name="request deadline")
    deadline = monotonic() + duration
    current = _REQUEST_DEADLINE.get()
    if current is not None:
        deadline = min(deadline, current)
    token: Token[float | None] = _REQUEST_DEADLINE.set(deadline)
    try:
        yield
    finally:
        _REQUEST_DEADLINE.reset(token)


def remaining_request_seconds() -> float | None:
    """Return the active budget, failing before another request after expiry."""

    deadline = _REQUEST_DEADLINE.get()
    if deadline is None:
        return None
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise dspy.LMTimeoutError("COMPASS operation deadline exceeded")
    return remaining


class DeadlineAwareLM(dspy.LM):
    """Use DSPy's per-call timeout seam to share one operation deadline."""

    def __init__(
        self,
        *args: Any,
        supports_response_schema: bool = False,
        **kwargs: Any,
    ) -> None:
        if not isinstance(supports_response_schema, bool):
            raise TypeError("supports_response_schema must be a boolean")
        self._declared_supports_response_schema = supports_response_schema
        super().__init__(*args, **kwargs)

    @property
    def supports_response_schema(self) -> bool:
        """Use an explicit endpoint capability when LiteLLM cannot identify it."""

        return (
            self._declared_supports_response_schema
            or super().supports_response_schema
        )

    def _deadline_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        remaining = remaining_request_seconds()
        if remaining is None:
            return kwargs
        configured = kwargs.get("timeout", self.kwargs.get("timeout"))
        if configured is not None:
            configured = _positive_seconds(configured, name="LM timeout")
            remaining = min(remaining, configured)
        return {**kwargs, "timeout": remaining}

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        return super().forward(
            prompt=prompt,
            messages=messages,
            **self._deadline_kwargs(dict(kwargs)),
        )

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        return await super().aforward(
            prompt=prompt,
            messages=messages,
            **self._deadline_kwargs(dict(kwargs)),
        )


__all__ = [
    "DeadlineAwareLM",
    "remaining_request_seconds",
    "request_deadline",
]
