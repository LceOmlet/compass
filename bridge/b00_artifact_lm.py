from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

import dspy

if TYPE_CHECKING:
    from .b01_aime_capture import CapturedRollout


class ReflectionCall(Protocol):
    def __call__(
        self,
        *,
        prompt: str,
        lm_kwargs: dict[str, Any],
        minibatch: tuple[CapturedRollout, ...],
    ) -> list[str]: ...


class ArtifactLMDispatcher(dspy.LM):
    """Route the two call shapes used by the pinned Artifact runtime."""

    _REQUIRED_TASK_CONFIG = frozenset(
        {
            "model",
            "model_type",
            "temperature",
            "top_p",
            "max_tokens",
            "cache",
            "cache_in_memory",
            "num_retries",
            "provider",
            "extra_body",
        }
    )

    def __init__(
        self,
        *,
        task_lm_config: Mapping[str, Any],
        current_minibatch: Callable[[], tuple[CapturedRollout, ...]],
        reflection_call: ReflectionCall | None,
    ) -> None:
        missing = self._REQUIRED_TASK_CONFIG.difference(task_lm_config)
        if missing:
            raise ValueError(f"task_lm_config is missing explicit keys: {sorted(missing)}")

        config = dict(task_lm_config)
        self._current_minibatch = current_minibatch
        self._reflection_call = reflection_call
        super().__init__(**config)

    def _task_kwargs(
        self,
        call_kwargs: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del messages
        effective_n = call_kwargs.get("n", self.kwargs.get("n", 1))
        if isinstance(effective_n, bool) or effective_n != 1:
            raise ValueError("the first official AIME bridge requires n=1")

        base_extra = self.kwargs.get("extra_body")
        call_extra = call_kwargs.get("extra_body")
        if base_extra is not None and not isinstance(base_extra, Mapping):
            raise TypeError("configured extra_body must be a mapping")
        if call_extra is not None and not isinstance(call_extra, Mapping):
            raise TypeError("call extra_body must be a mapping")

        merged_extra: dict[str, Any] = {}
        if base_extra is not None:
            merged_extra.update(base_extra)
        if call_extra is not None:
            merged_extra.update(call_extra)
        if "return_token_ids" in merged_extra and merged_extra["return_token_ids"] is not True:
            raise ValueError("return_token_ids conflicts with the required Arbor token capture")
        merged_extra["return_token_ids"] = True

        routed = dict(call_kwargs)
        routed["extra_body"] = merged_extra
        return routed

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[str] | list[dict[str, Any]]:
        if (prompt is None) == (messages is None):
            raise ValueError("exactly one of prompt or messages must be supplied")

        if messages is not None:
            return super().__call__(messages=messages, **self._task_kwargs(kwargs, messages))

        if not isinstance(prompt, str):
            raise TypeError("the Artifact reflection prompt must be a string")
        if self._reflection_call is None:
            result = super().__call__(prompt=prompt, **kwargs)
            if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], str):
                raise TypeError("the official task LM reflection must return one completion")
            return result
        result = self._reflection_call(
            prompt=prompt,
            lm_kwargs=dict(kwargs),
            minibatch=self._current_minibatch(),
        )
        if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], str):
            raise TypeError("reflection_call must return the official one-completion list[str]")
        return result

    async def acall(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[str] | list[dict[str, Any]]:
        if prompt is not None or messages is None:
            raise ValueError("the pinned Artifact reflection path is synchronous and positional")
        return await super().acall(messages=messages, **self._task_kwargs(kwargs, messages))
