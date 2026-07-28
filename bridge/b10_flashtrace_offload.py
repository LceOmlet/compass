from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from flashtrace import FlashTrace
from flashtrace import attribution as flashtrace_attribution
from flashtrace import core as flashtrace_core
from flashtrace import tracer as flashtrace_tracer
from flashtrace.attribution import LLMIFRAttribution
from flashtrace.improved import LLMIFRAttributionBoth
from torch.utils.hooks import RemovableHandle


_OFFICIAL_ATTACH_HOOKS = flashtrace_core.attach_hooks
_OFFICIAL_RECOMPUTE_LAYER_ATTENTION = flashtrace_core.recompute_layer_attention
_OFFICIAL_APPLY_ROTARY_POS_EMB = flashtrace_core._apply_rotary_pos_emb
_BACKEND_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class _CaptureRequest:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    input_ids_dtype: torch.dtype
    attention_mask_dtype: torch.dtype
    input_ids_device: torch.device
    attention_mask_device: torch.device
    recompute_attention: bool

    @classmethod
    def from_call(
        cls,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        recompute_attention: bool,
    ) -> _CaptureRequest:
        if not torch.is_tensor(input_ids) or not torch.is_tensor(attention_mask):
            raise TypeError("official FlashTrace capture inputs must be tensors")
        return cls(
            input_ids=input_ids.detach().to(device="cpu").clone(),
            attention_mask=attention_mask.detach().to(device="cpu").clone(),
            input_ids_dtype=input_ids.dtype,
            attention_mask_dtype=attention_mask.dtype,
            input_ids_device=input_ids.device,
            attention_mask_device=attention_mask.device,
            recompute_attention=recompute_attention,
        )

    def matches(self, other: _CaptureRequest) -> bool:
        return (
            self.input_ids_dtype == other.input_ids_dtype
            and self.attention_mask_dtype == other.attention_mask_dtype
            and self.input_ids_device == other.input_ids_device
            and self.attention_mask_device == other.attention_mask_device
            and self.recompute_attention is other.recompute_attention
            and self.input_ids.shape == other.input_ids.shape
            and self.attention_mask.shape == other.attention_mask.shape
            and torch.equal(self.input_ids, other.input_ids)
            and torch.equal(self.attention_mask, other.attention_mask)
        )


class OffloadedCaptureLease:
    """Own one exact official capture for at most two sequential consumers."""

    __slots__ = (
        "_active",
        "_capture",
        "_consumer_count",
        "_in_capture",
        "_lock",
        "_model",
        "_owner_thread",
        "_request",
    )

    def __init__(self, model: nn.Module) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("capture lease model must be a torch module")
        self._model = model
        self._lock = threading.RLock()
        self._active = False
        self._owner_thread: int | None = None
        self._request: _CaptureRequest | None = None
        self._capture: Any = None
        self._consumer_count = 0
        self._in_capture = False

    def __enter__(self) -> OffloadedCaptureLease:
        with self._lock:
            if self._active:
                raise RuntimeError("the offloaded capture lease is already active")
            self._active = True
            self._owner_thread = threading.get_ident()
            self._request = None
            self._capture = None
            self._consumer_count = 0
            self._in_capture = False
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        with self._lock:
            if not self._active:
                raise RuntimeError("the offloaded capture lease is not active")
            if self._owner_thread != threading.get_ident():
                raise RuntimeError("the offloaded capture lease must exit on its owner thread")
            if self._in_capture:
                raise RuntimeError("cannot release an active official capture")
            self._capture = None
            self._request = None
            self._consumer_count = 0
            self._owner_thread = None
            self._active = False

    def _capture_or_reuse(
        self,
        *,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        recompute_attention: bool,
        capture: Callable[[], Any],
    ) -> Any:
        with self._lock:
            if not self._active:
                raise RuntimeError("the offloaded capture lease must be entered before use")
            if self._owner_thread != threading.get_ident():
                raise RuntimeError("offloaded capture consumers must run sequentially on the owner thread")
            if self._in_capture:
                raise RuntimeError("concurrent or recursive offloaded capture consumption is forbidden")
            if model is not self._model:
                raise RuntimeError("offloaded capture consumers must share the exact model instance")
            if recompute_attention is not True:
                raise RuntimeError("shared offloaded capture requires recompute_attention=True")
            if self._consumer_count >= 2:
                raise RuntimeError("an offloaded capture lease supports at most two consumers")

            request = _CaptureRequest.from_call(
                input_ids=input_ids,
                attention_mask=attention_mask,
                recompute_attention=recompute_attention,
            )
            if self._request is not None:
                if not self._request.matches(request):
                    raise RuntimeError(
                        "offloaded capture reuse requires identical full input IDs and attention mask"
                    )
                if self._capture is None:
                    raise RuntimeError("the offloaded capture lease lost its official capture")
                self._consumer_count += 1
                return self._capture

            if not callable(capture):
                raise TypeError("capture must call the existing official/offload capture path")
            self._in_capture = True
            try:
                captured = capture()
            finally:
                self._in_capture = False
            if not isinstance(captured, tuple) or len(captured) != 4:
                raise RuntimeError("official FlashTrace returned an unexpected capture tuple")
            self._request = request
            self._capture = captured
            self._consumer_count = 1
            return captured


class _OffloadedActivation:
    __slots__ = ("_cpu_tensor", "_source_device")

    def __init__(self, tensor: torch.Tensor) -> None:
        if not torch.is_tensor(tensor):
            raise TypeError("the official activation cache entry must be a tensor")
        self._source_device = tensor.device
        self._cpu_tensor = torch.empty_like(
            tensor,
            device="cpu",
            pin_memory=True,
        )
        self._cpu_tensor.copy_(tensor.detach(), non_blocking=True)

    def __getitem__(self, index: int) -> torch.Tensor:
        if index != 0:
            raise IndexError("the official sentence aggregate only supports batch index zero")
        return self._cpu_tensor[0].to(
            device=self._source_device,
            non_blocking=True,
        )


def _offload_slot(
    cache: dict[str, list[Any]],
    name: str,
    layer_index: int,
) -> None:
    value = cache[name][layer_index]
    if not torch.is_tensor(value):
        raise RuntimeError(f"official FlashTrace did not populate {name}[{layer_index}]")
    cache[name][layer_index] = _OffloadedActivation(value)


def _attach_offload_hooks(
    layers: Sequence[nn.Module],
    model_dtype: torch.dtype,
) -> tuple[dict[str, list[Any]], list[RemovableHandle]]:
    cache, hooks = _OFFICIAL_ATTACH_HOOKS(layers, model_dtype)

    def forward_hook(name: str, layer_index: int):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            del module, inputs, output
            _offload_slot(cache, name, layer_index)

        return hook

    def forward_pre_hook(name: str, layer_index: int):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            del module, inputs
            _offload_slot(cache, name, layer_index)

        return hook

    for layer_index, layer in enumerate(layers):
        hooks.append(
            layer.input_layernorm.register_forward_hook(
                forward_hook("pre_attn_resid", layer_index)
            )
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                forward_pre_hook("mid_resid", layer_index)
            )
        )
        hooks.append(
            layer.mlp.register_forward_hook(forward_hook("mlp_out", layer_index))
        )
        hooks.append(
            layer.register_forward_hook(forward_hook("post_resid", layer_index))
        )

    return cache, hooks


class _SinkAttentionExpression:
    __slots__ = (
        "_owner",
        "_query_start",
        "_query_end",
        "_key_end",
        "_factors",
    )

    def __init__(
        self,
        owner: _SinkRowAttention,
        query_start: int,
        query_end: int,
        key_end: int,
        factors: tuple[torch.Tensor, ...] = (),
    ) -> None:
        self._owner = owner
        self._query_start = query_start
        self._query_end = query_end
        self._key_end = key_end
        self._factors = factors

    @property
    def dtype(self) -> torch.dtype:
        return self._owner.dtype

    def __mul__(self, factor: torch.Tensor) -> _SinkAttentionExpression:
        if not torch.is_tensor(factor):
            return NotImplemented
        return _SinkAttentionExpression(
            self._owner,
            self._query_start,
            self._query_end,
            self._key_end,
            self._factors + (factor,),
        )

    def __rmul__(self, factor: torch.Tensor) -> _SinkAttentionExpression:
        return self.__mul__(factor)

    def sum(self, dim: int | None = None, **kwargs: Any) -> torch.Tensor:
        if dim != 1 or kwargs:
            raise RuntimeError("streamed FlashTrace attention only supports sum(dim=1)")
        return self._owner.sum_rows(
            query_start=self._query_start,
            query_end=self._query_end,
            key_end=self._key_end,
            factors=self._factors,
        )


class _SinkRowAttention:
    __slots__ = ("_x_prev", "_layer_weights", "_rotary_emb", "_params")

    def __init__(
        self,
        x_prev: torch.Tensor,
        layer_weights: dict[str, torch.Tensor | nn.Module],
        rotary_emb: nn.Module,
        params: Any,
    ) -> None:
        self._x_prev = x_prev
        self._layer_weights = layer_weights
        self._rotary_emb = rotary_emb
        self._params = params
        if int(params.sink_chunk_tokens) <= 0:
            raise ValueError("official sink_chunk_tokens must be positive")

    @property
    def dtype(self) -> torch.dtype:
        return self._params.model_dtype

    def __getitem__(self, key: tuple[slice, slice, slice]) -> _SinkAttentionExpression:
        if not isinstance(key, tuple) or len(key) != 3:
            raise RuntimeError("streamed attention requires [heads, queries, keys]")
        head_slice, query_slice, key_slice = key
        if head_slice != slice(None, None, None):
            raise RuntimeError("streamed attention requires all query heads")
        if not isinstance(query_slice, slice) or not isinstance(key_slice, slice):
            raise RuntimeError("streamed attention requires contiguous query/key slices")
        if query_slice.step not in (None, 1) or key_slice.step not in (None, 1):
            raise RuntimeError("streamed attention does not support strided slices")
        sequence_length = int(self._x_prev.shape[0])
        query_start = 0 if query_slice.start is None else int(query_slice.start)
        query_end = sequence_length if query_slice.stop is None else int(query_slice.stop)
        key_start = 0 if key_slice.start is None else int(key_slice.start)
        key_end = sequence_length if key_slice.stop is None else int(key_slice.stop)
        if key_start != 0:
            raise RuntimeError("official sentence aggregation must use a zero-based key prefix")
        if not (0 <= query_start < query_end <= sequence_length):
            raise RuntimeError("invalid streamed attention query slice")
        if not (0 < key_end <= sequence_length):
            raise RuntimeError("invalid streamed attention key slice")
        return _SinkAttentionExpression(self, query_start, query_end, key_end)

    def _query_and_key(self) -> tuple[torch.Tensor, torch.Tensor]:
        x_prev = self._x_prev
        layer_weights = self._layer_weights
        params = self._params
        device = x_prev.device
        model_dtype = params.model_dtype
        sequence_length = int(x_prev.shape[0])

        input_norm = layer_weights["in_ln"]
        query_weight = layer_weights["q_w"].to(device=device, non_blocking=True)
        key_weight = layer_weights["k_w"].to(device=device, non_blocking=True)
        x_normed = input_norm(x_prev.unsqueeze(0)).squeeze(0).to(model_dtype)

        query = torch.matmul(x_normed, query_weight.T)
        key = torch.matmul(x_normed, key_weight.T)
        query_bias = layer_weights.get("q_bias")
        key_bias = layer_weights.get("k_bias")
        if query_bias is not None:
            query = query + query_bias.to(device=device, non_blocking=True)
        if key_bias is not None:
            key = key + key_bias.to(device=device, non_blocking=True)

        query = query.view(
            sequence_length,
            int(params.n_heads_q),
            int(params.head_dim),
        ).transpose(0, 1).unsqueeze(0)
        key = key.view(
            sequence_length,
            int(params.n_kv_heads),
            int(params.head_dim),
        ).transpose(0, 1).unsqueeze(0)

        position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)
        cos, sin = self._rotary_emb(key, position_ids)
        cos = cos.to(device=device, non_blocking=False)
        sin = sin.to(device=device, non_blocking=False)
        query, key = _OFFICIAL_APPLY_ROTARY_POS_EMB(query, key, cos, sin)
        key = key.repeat_interleave(int(params.group_size), dim=1)
        return query, key

    def sum_rows(
        self,
        *,
        query_start: int,
        query_end: int,
        key_end: int,
        factors: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        query, key = self._query_and_key()
        params = self._params
        device = query.device
        model_dtype = params.model_dtype
        sequence_length = int(key.shape[2])
        query_count = query_end - query_start
        key_transposed = key.transpose(2, 3)
        accumulator = torch.zeros(
            (int(params.n_heads_q), key_end),
            device=device,
            dtype=torch.float32,
        )

        for factor in factors:
            if factor.device != device or factor.dtype != model_dtype:
                raise RuntimeError("official alpha factors changed device or dtype")
            if factor.ndim != 3 or factor.shape[0] != 1:
                raise RuntimeError("official alpha factors must have shape [1, P, 1|J]")
            if int(factor.shape[1]) != query_count or int(factor.shape[2]) not in (1, key_end):
                raise RuntimeError("official alpha factor shape does not match the sink span")

        chunk_tokens = int(params.sink_chunk_tokens)
        scale = math.sqrt(int(params.head_dim))
        for block_start in range(query_start, query_end, chunk_tokens):
            block_end = min(query_end, block_start + chunk_tokens)
            attention_scores = torch.matmul(
                query[:, :, block_start:block_end],
                key_transposed,
            ) / scale
            causal_mask = torch.triu(
                torch.full(
                    (block_end - block_start, sequence_length),
                    float("-inf"),
                    device=device,
                    dtype=attention_scores.dtype,
                ),
                diagonal=block_start + 1,
            )
            attention_scores = attention_scores + causal_mask.unsqueeze(0).unsqueeze(0)
            attention = torch.nn.functional.softmax(
                attention_scores,
                dim=-1,
                dtype=torch.float32,
            ).to(model_dtype)[0]
            weighted = attention[:, :, :key_end]
            local_start = block_start - query_start
            local_end = block_end - query_start
            for factor in factors:
                weighted = weighted * factor[:, local_start:local_end]
            accumulator.add_(weighted.sum(dim=1, dtype=torch.float32))

        return accumulator.to(model_dtype)


def _recompute_sink_row_attention(
    x_prev: torch.Tensor,
    layer_weights: dict[str, torch.Tensor | nn.Module],
    rotary_emb: nn.Module,
    params: Any,
) -> _SinkRowAttention:
    return _SinkRowAttention(x_prev, layer_weights, rotary_emb, params)


@contextmanager
def _patched_attribute_capture() -> Iterator[None]:
    with _BACKEND_LOCK:
        previous = flashtrace_attribution.attach_hooks
        if previous not in (_OFFICIAL_ATTACH_HOOKS, _attach_offload_hooks):
            raise RuntimeError("another component replaced official FlashTrace activation hooks")
        flashtrace_attribution.attach_hooks = _attach_offload_hooks
        try:
            yield
        finally:
            flashtrace_attribution.attach_hooks = previous


@contextmanager
def _capture_last_logit_only(model: nn.Module) -> Iterator[None]:
    def add_official_qwen_argument(
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        del module
        if "logits_to_keep" in kwargs:
            raise RuntimeError("official FlashTrace unexpectedly set logits_to_keep")
        forwarded = dict(kwargs)
        forwarded["logits_to_keep"] = 1
        return args, forwarded

    handle = model.register_forward_pre_hook(
        add_official_qwen_argument,
        with_kwargs=True,
    )
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def _patched_streaming_attention() -> Iterator[None]:
    with _BACKEND_LOCK:
        previous = flashtrace_core.recompute_layer_attention
        if previous not in (
            _OFFICIAL_RECOMPUTE_LAYER_ATTENTION,
            _recompute_sink_row_attention,
        ):
            raise RuntimeError("another component replaced official FlashTrace attention recomputation")
        flashtrace_core.recompute_layer_attention = _recompute_sink_row_attention
        try:
            yield
        finally:
            flashtrace_core.recompute_layer_attention = previous


class _OffloadCaptureMixin:
    _active_capture_lease: OffloadedCaptureLease | None = None

    @contextmanager
    def _capture_lease_scope(
        self,
        capture_lease: OffloadedCaptureLease | None,
    ) -> Iterator[None]:
        if capture_lease is not None and not isinstance(capture_lease, OffloadedCaptureLease):
            raise TypeError("capture_lease must be an OffloadedCaptureLease")
        if self._active_capture_lease is not None:
            raise RuntimeError("an offloaded capture lease is already bound to this engine")
        self._active_capture_lease = capture_lease
        try:
            yield
        finally:
            self._active_capture_lease = None

    def _capture_model_state_uncached(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        recompute_attention: bool,
    ) -> Any:
        with _patched_attribute_capture(), _capture_last_logit_only(self.model):
            return super()._capture_model_state(
                input_ids,
                attention_mask,
                recompute_attention=recompute_attention,
            )

    def _capture_model_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        recompute_attention: bool = False,
    ) -> Any:
        if recompute_attention is not True:
            raise RuntimeError("the offload backend requires official recompute_attention=True")
        capture_lease = self._active_capture_lease
        if capture_lease is None:
            return self._capture_model_state_uncached(
                input_ids,
                attention_mask,
                recompute_attention=recompute_attention,
            )
        return capture_lease._capture_or_reuse(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            recompute_attention=recompute_attention,
            capture=lambda: self._capture_model_state_uncached(
                input_ids,
                attention_mask,
                recompute_attention=recompute_attention,
            ),
        )


class OffloadedLLMIFRAttribution(_OffloadCaptureMixin, LLMIFRAttribution):
    """Official one-hop IFR with an exact offloaded/streamed compute backend."""

    def __init__(self, *args: Any, recompute_attention: bool = True, **kwargs: Any) -> None:
        if recompute_attention is not True:
            raise ValueError("OffloadedLLMIFRAttribution requires recompute_attention=True")
        super().__init__(*args, recompute_attention=True, **kwargs)

    def calculate_ifr_multi_hop(self, *args: Any, n_hops: int = 1, **kwargs: Any) -> Any:
        if int(n_hops) != 1:
            raise ValueError("the dependency backend is defined only for n_hops=1")
        with _patched_streaming_attention():
            return super().calculate_ifr_multi_hop(*args, n_hops=1, **kwargs)


class _OffloadedLLMIFRAttributionBoth(_OffloadCaptureMixin, LLMIFRAttributionBoth):
    def calculate_ifr_multi_hop_both(
        self,
        *args: Any,
        n_hops: int = 1,
        **kwargs: Any,
    ) -> Any:
        if int(n_hops) != 1:
            raise ValueError("the rollout-credit backend is defined only for hops=1")
        with _patched_streaming_attention():
            return super().calculate_ifr_multi_hop_both(*args, n_hops=1, **kwargs)


class OffloadedFlashTrace(FlashTrace):
    """Official FlashTrace facade restricted to the adopted hops=1 credit path."""

    def __init__(self, *args: Any, recompute_attention: bool = True, **kwargs: Any) -> None:
        if recompute_attention is not True:
            raise ValueError("OffloadedFlashTrace requires recompute_attention=True")
        super().__init__(*args, recompute_attention=True, **kwargs)

    def trace(
        self,
        *,
        prompt: str,
        target: str | None = None,
        output_span: tuple[int, int] | None = None,
        reasoning_span: tuple[int, int] | None = None,
        hops: int = 1,
        method: str = "flashtrace",
        renorm_threshold: float | None = None,
    ) -> Any:
        if method != "flashtrace" or int(hops) != 1:
            raise ValueError("the rollout-credit backend supports only method='flashtrace', hops=1")
        with _BACKEND_LOCK:
            previous = flashtrace_tracer.LLMIFRAttributionBoth
            if previous not in (LLMIFRAttributionBoth, _OffloadedLLMIFRAttributionBoth):
                raise RuntimeError("another component replaced the official FlashTrace engine")
            flashtrace_tracer.LLMIFRAttributionBoth = _OffloadedLLMIFRAttributionBoth
            try:
                return super().trace(
                    prompt=prompt,
                    target=target,
                    output_span=output_span,
                    reasoning_span=reasoning_span,
                    hops=1,
                    method="flashtrace",
                    renorm_threshold=renorm_threshold,
                )
            finally:
                flashtrace_tracer.LLMIFRAttributionBoth = previous


__all__ = [
    "OffloadedCaptureLease",
    "OffloadedFlashTrace",
    "OffloadedLLMIFRAttribution",
]
