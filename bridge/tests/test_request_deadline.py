from __future__ import annotations

import asyncio
import math
from typing import Any

import dspy
import pytest

import bridge.request_deadline as deadline_module
from bridge.request_deadline import (
    DeadlineAwareLM,
    remaining_request_seconds,
    request_deadline,
)


def test_request_deadline_uses_earliest_nested_deadline_and_restores_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"value": 100.0}
    monkeypatch.setattr(deadline_module, "monotonic", lambda: now["value"])

    assert remaining_request_seconds() is None
    with request_deadline(10):
        assert remaining_request_seconds() == pytest.approx(10.0)

        now["value"] = 101.0
        with request_deadline(20):
            assert remaining_request_seconds() == pytest.approx(9.0)

        assert remaining_request_seconds() == pytest.approx(9.0)

    assert remaining_request_seconds() is None


@pytest.mark.parametrize("seconds", [0, -1, True, math.inf, math.nan, "600"])
def test_request_deadline_rejects_non_positive_or_non_finite_values(
    seconds: object,
) -> None:
    with pytest.raises(TypeError, match="positive finite number"):
        with request_deadline(seconds):  # type: ignore[arg-type]
            pass


def test_expired_deadline_fails_before_another_lm_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"value": 10.0}
    monkeypatch.setattr(deadline_module, "monotonic", lambda: now["value"])
    parent_called = False

    def fake_forward(self: dspy.LM, *args: Any, **kwargs: Any) -> object:
        nonlocal parent_called
        parent_called = True
        return object()

    monkeypatch.setattr(dspy.LM, "forward", fake_forward)
    lm = DeadlineAwareLM("openai/test", timeout=600, cache=False)

    with request_deadline(3):
        now["value"] = 13.0
        with pytest.raises(dspy.LMTimeoutError, match="operation deadline exceeded"):
            lm.forward(prompt="too late")

    assert parent_called is False


def test_deadline_aware_lm_passes_the_smaller_remaining_sync_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"value": 100.0}
    monkeypatch.setattr(deadline_module, "monotonic", lambda: now["value"])
    calls: list[dict[str, Any]] = []

    def fake_forward(
        self: dspy.LM,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        calls.append({"prompt": prompt, "messages": messages, **kwargs})
        return "ok"

    monkeypatch.setattr(dspy.LM, "forward", fake_forward)
    lm = DeadlineAwareLM("openai/test", timeout=600, cache=False)

    with request_deadline(30):
        now["value"] = 101.5
        assert lm.forward(prompt="first") == "ok"
        assert lm.forward(prompt="second", timeout=5) == "ok"

    assert calls[0]["timeout"] == pytest.approx(28.5)
    assert calls[1]["timeout"] == pytest.approx(5.0)


def test_deadline_aware_lm_preserves_calls_without_an_operation_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_forward(self: dspy.LM, **kwargs: Any) -> str:
        calls.append(kwargs)
        return "ok"

    monkeypatch.setattr(dspy.LM, "forward", fake_forward)
    lm = DeadlineAwareLM("openai/test", timeout=600, cache=False)

    assert lm.forward(prompt="plain") == "ok"
    assert calls == [{"prompt": "plain", "messages": None}]


def test_deadline_aware_lm_uses_explicit_response_schema_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dspy.LM,
        "supports_response_schema",
        property(lambda self: False),
    )

    inferred = DeadlineAwareLM("openai/test", cache=False)
    explicit = DeadlineAwareLM(
        "openai/test",
        cache=False,
        supports_response_schema=True,
    )

    assert inferred.supports_response_schema is False
    assert explicit.supports_response_schema is True
    assert "supports_response_schema" not in explicit.kwargs


def test_deadline_aware_lm_rejects_invalid_response_schema_capability() -> None:
    with pytest.raises(TypeError, match="supports_response_schema"):
        DeadlineAwareLM(
            "openai/test",
            cache=False,
            supports_response_schema="yes",  # type: ignore[arg-type]
        )


def test_deadline_aware_lm_passes_remaining_async_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"value": 50.0}
    monkeypatch.setattr(deadline_module, "monotonic", lambda: now["value"])
    calls: list[dict[str, Any]] = []

    async def fake_aforward(
        self: dspy.LM,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        calls.append({"prompt": prompt, "messages": messages, **kwargs})
        return "ok"

    monkeypatch.setattr(dspy.LM, "aforward", fake_aforward)
    lm = DeadlineAwareLM("openai/test", timeout=600, cache=False)

    async def invoke() -> str:
        with request_deadline(12):
            now["value"] = 52.0
            return await lm.aforward(prompt="async")

    assert asyncio.run(invoke()) == "ok"
    assert calls[0]["timeout"] == pytest.approx(10.0)
