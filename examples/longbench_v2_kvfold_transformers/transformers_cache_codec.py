# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

"""Apply KVfold to post-RoPE Qwen3 K/V through ``DynamicCache.update``.

Qwen3 calls ``DynamicCache.update`` after applying RoPE to the key tensor.  A
context-local monkeypatch therefore avoids copying a version-specific
``Qwen3Attention.forward`` while still replacing the exact K/V values used by
the current attention operation and stored in the cache.

The offline Transformers model is layer-sharded rather than tensor parallel.
For Qwen3-32B, this module explicitly divides its eight KV heads into four
contiguous two-head shards so each native payload matches the production TP4
layout ``[K/V, token=128, local_head=2, channel=128]``.
"""

from __future__ import annotations

import functools
import inspect
import operator
import threading
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

from kvfold_codec import KvfoldCodec, RoundtripStats

_PATCH_LOCK = threading.Lock()
_ACTIVE_PATCH: KvfoldCacheProcessor | None = None


class TransformersCacheCodecError(RuntimeError):
    """Raised when the offline Qwen3 cache path violates its test contract."""


class KvfoldCacheProcessor:
    """Temporarily round-trip Qwen3-32B K/V before DynamicCache stores it.

    Args:
        codec: Canonical codec name or an alias accepted by ``KvfoldCodec``.
        library: Optional path to ``libkvfold_longbench_bridge.so``.
        token_blocks_per_batch: Maximum number of 128-token blocks copied to
            the CPU in one native call. Each token block produces four codec
            blocks because the offline eight-head tensor simulates TP4.

    The hook is deliberately strict: it accepts BF16, batch-one Qwen3-32B K/V
    only. It never changes the input K/V tensors in place. If an update contains
    at least one globally aligned full block, cloned K/V tensors are passed to
    the original ``DynamicCache.update`` implementation.
    """

    BLOCK_SIZE = 128
    EXPECTED_BATCH_SIZE = 1
    EXPECTED_KV_HEADS = 8
    EXPECTED_HEAD_DIM = 128
    SIMULATED_TP_SIZE = 4
    LOCAL_KV_HEADS = EXPECTED_KV_HEADS // SIMULATED_TP_SIZE
    VALUES_PER_CODEC_BLOCK = 2 * BLOCK_SIZE * LOCAL_KV_HEADS * EXPECTED_HEAD_DIM
    STATS_SCHEMA_VERSION = 1

    _COUNTER_FIELDS = (
        "calls",
        "calls_without_full_block",
        "failures",
        "input_tokens",
        "processed_tokens",
        "skipped_tokens",
        "skipped_leading_tokens",
        "skipped_trailing_tokens",
        "token_blocks",
        "blocks",
        "values",
        "raw_bytes",
        "record_bytes",
        "primary_blocks",
        "fallback_blocks",
        "exceptions",
        "m1_groups",
        "m0_groups",
    )

    _INTEGER_DTYPES = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }

    def __init__(
        self,
        codec: str,
        library: str | Path | None = None,
        token_blocks_per_batch: int = 32,
    ) -> None:
        try:
            batch_size = operator.index(token_blocks_per_batch)
        except TypeError as error:
            raise TypeError("token_blocks_per_batch must be an integer") from error
        if batch_size <= 0:
            raise ValueError("token_blocks_per_batch must be positive")

        self.codec = KvfoldCodec(codec=codec, library=library)
        self.token_blocks_per_batch = batch_size
        self._stats: Counter[str] = Counter()
        self._layer_stats: defaultdict[int, Counter[str]] = defaultdict(Counter)
        self._stats_lock = threading.Lock()
        self._dynamic_cache_class: type | None = None
        self._original_update: Any = None
        self._wrapped_update: Any = None
        self._installed = False

        expected_record = self.codec.record_bytes(self.VALUES_PER_CODEC_BLOCK)
        if expected_record <= 0:
            raise TransformersCacheCodecError(
                f"{self.codec.codec} does not support the Qwen3-32B TP4 payload: "
                f"values={self.VALUES_PER_CODEC_BLOCK}"
            )
        if expected_record != self.codec.spec.record_bytes_per_block:
            raise TransformersCacheCodecError(
                "codec catalog/native record-size mismatch: "
                f"catalog={self.codec.spec.record_bytes_per_block}, "
                f"native={expected_record}"
            )
        self.record_bytes_per_block = expected_record

    @property
    def installed(self) -> bool:
        """Whether this instance currently owns the global monkeypatch."""

        return self._installed

    @staticmethod
    def _layer_index(layer_idx: Any) -> int:
        try:
            result = operator.index(layer_idx)
        except TypeError as error:
            raise TransformersCacheCodecError(
                f"layer_idx must be an integer, got {type(layer_idx).__name__}"
            ) from error
        if result < 0:
            raise TransformersCacheCodecError(
                f"layer_idx must be non-negative, got {result}"
            )
        return result

    def _validate_kv(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TransformersCacheCodecError(
                "key_states and value_states must be tensors"
            )
        expected_shape = (
            self.EXPECTED_BATCH_SIZE,
            self.EXPECTED_KV_HEADS,
            "sequence",
            self.EXPECTED_HEAD_DIM,
        )
        if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
            raise TransformersCacheCodecError(
                "expected matching Qwen3 K/V shaped [B,H,S,D], got "
                f"key={tuple(key.shape)}, value={tuple(value.shape)}"
            )
        if (
            key.shape[0] != self.EXPECTED_BATCH_SIZE
            or key.shape[1] != self.EXPECTED_KV_HEADS
            or key.shape[3] != self.EXPECTED_HEAD_DIM
        ):
            raise TransformersCacheCodecError(
                f"expected Qwen3-32B K/V shape {expected_shape}, got {tuple(key.shape)}"
            )
        if key.shape[2] <= 0:
            raise TransformersCacheCodecError(
                "K/V update must contain at least one token"
            )
        if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            raise TransformersCacheCodecError(
                f"KVfold requires BF16 K/V, got key={key.dtype}, value={value.dtype}"
            )
        if key.device != value.device:
            raise TransformersCacheCodecError(
                f"K/V device mismatch: key={key.device}, value={value.device}"
            )
        if key.requires_grad or value.requires_grad:
            raise TransformersCacheCodecError(
                "KVfold Transformers hook is inference-only; K/V must not require gradients"
            )

    def _validate_positions(
        self,
        cache: Any,
        layer_idx: int,
        sequence_length: int,
        cache_kwargs: Mapping[str, Any] | None,
    ) -> tuple[int, int, int]:
        get_seq_length = getattr(cache, "get_seq_length", None)
        if not callable(get_seq_length):
            raise TransformersCacheCodecError(
                "patched DynamicCache has no callable get_seq_length"
            )
        try:
            cached_length = operator.index(get_seq_length(layer_idx))
        except TypeError:
            cached_length = operator.index(get_seq_length())
        if cached_length < 0:
            raise TransformersCacheCodecError(
                f"DynamicCache returned a negative sequence length: {cached_length}"
            )

        if cache_kwargs is not None and not isinstance(cache_kwargs, Mapping):
            raise TransformersCacheCodecError("cache_kwargs must be a mapping or None")
        positions = (
            cache_kwargs.get("cache_position")
            if isinstance(cache_kwargs, Mapping)
            else None
        )
        if positions is None:
            # Current Qwen3 implementations call DynamicCache.update() without
            # cache_kwargs. For this batch-one, strictly sequential evaluator,
            # the current per-layer cache length is the unambiguous start.
            start = cached_length
        else:
            if not isinstance(positions, torch.Tensor):
                raise TransformersCacheCodecError("cache_position must be a tensor")
            if positions.ndim != 1 or positions.numel() != sequence_length:
                raise TransformersCacheCodecError(
                    "cache_position must be one-dimensional and match K/V "
                    f"sequence length: shape={tuple(positions.shape)}, "
                    f"sequence_length={sequence_length}"
                )
            if positions.dtype not in self._INTEGER_DTYPES:
                raise TransformersCacheCodecError(
                    f"cache_position must use an integer dtype, got {positions.dtype}"
                )

            positions_cpu = positions.detach().to(device="cpu", dtype=torch.int64)
            start = int(positions_cpu[0])
            if start < 0:
                raise TransformersCacheCodecError(
                    f"cache_position must be non-negative, got start={start}"
                )
            expected = torch.arange(start, start + sequence_length, dtype=torch.int64)
            if not torch.equal(positions_cpu, expected):
                raise TransformersCacheCodecError(
                    "cache_position must be contiguous and strictly increasing by one"
                )
            if cached_length != start:
                raise TransformersCacheCodecError(
                    "cache_position does not follow the current layer cache: "
                    f"layer={layer_idx}, cached_length={cached_length}, "
                    f"start={start}"
                )

        leading = (-start) % self.BLOCK_SIZE
        leading = min(leading, sequence_length)
        aligned_tokens = (
            (sequence_length - leading) // self.BLOCK_SIZE * self.BLOCK_SIZE
        )
        trailing = sequence_length - leading - aligned_tokens
        return leading, aligned_tokens, trailing

    def _make_payload(
        self, key: torch.Tensor, value: torch.Tensor, token_blocks: int
    ) -> torch.Tensor:
        token_count = token_blocks * self.BLOCK_SIZE
        expected = (
            self.EXPECTED_BATCH_SIZE,
            self.EXPECTED_KV_HEADS,
            token_count,
            self.EXPECTED_HEAD_DIM,
        )
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise TransformersCacheCodecError(
                f"internal K/V chunk shape mismatch: expected={expected}, "
                f"key={tuple(key.shape)}, value={tuple(value.shape)}"
            )

        def shard(tensor: torch.Tensor) -> torch.Tensor:
            # [B, 8H, T, D] -> [B, token_block, TP4, 128T, 2H, D]
            return (
                tensor.view(
                    self.EXPECTED_BATCH_SIZE,
                    self.SIMULATED_TP_SIZE,
                    self.LOCAL_KV_HEADS,
                    token_blocks,
                    self.BLOCK_SIZE,
                    self.EXPECTED_HEAD_DIM,
                )
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(
                    token_blocks * self.SIMULATED_TP_SIZE,
                    self.BLOCK_SIZE,
                    self.LOCAL_KV_HEADS,
                    self.EXPECTED_HEAD_DIM,
                )
            )

        return torch.stack((shard(key), shard(value)), dim=1).contiguous()

    def _decode_payload(
        self, decoded: torch.Tensor, token_blocks: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codec_blocks = token_blocks * self.SIMULATED_TP_SIZE
        expected = (
            codec_blocks,
            2,
            self.BLOCK_SIZE,
            self.LOCAL_KV_HEADS,
            self.EXPECTED_HEAD_DIM,
        )
        if (
            decoded.device.type != "cpu"
            or decoded.dtype != torch.bfloat16
            or not decoded.is_contiguous()
            or tuple(decoded.shape) != expected
        ):
            raise TransformersCacheCodecError(
                f"native codec returned an invalid tensor: expected={expected}, "
                f"shape={tuple(decoded.shape)}, dtype={decoded.dtype}, "
                f"device={decoded.device}, contiguous={decoded.is_contiguous()}"
            )

        def join(tensor: torch.Tensor) -> torch.Tensor:
            # [B, token_block, TP4, 128T, 2H, D] -> [B, 8H, T, D]
            return (
                tensor.view(
                    self.EXPECTED_BATCH_SIZE,
                    token_blocks,
                    self.SIMULATED_TP_SIZE,
                    self.BLOCK_SIZE,
                    self.LOCAL_KV_HEADS,
                    self.EXPECTED_HEAD_DIM,
                )
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(
                    self.EXPECTED_BATCH_SIZE,
                    self.EXPECTED_KV_HEADS,
                    token_blocks * self.BLOCK_SIZE,
                    self.EXPECTED_HEAD_DIM,
                )
            )

        return join(decoded[:, 0]), join(decoded[:, 1])

    def _validate_native_stats(self, stats: RoundtripStats, codec_blocks: int) -> None:
        if stats.codec != self.codec.codec:
            raise TransformersCacheCodecError(
                f"native codec mismatch: expected={self.codec.codec}, got={stats.codec}"
            )
        expected_values = codec_blocks * self.VALUES_PER_CODEC_BLOCK
        if (
            stats.blocks != codec_blocks
            or stats.values != expected_values
            or stats.raw_bytes != expected_values * 2
            or stats.record_bytes != codec_blocks * self.record_bytes_per_block
            or stats.primary_blocks + stats.fallback_blocks != codec_blocks
        ):
            raise TransformersCacheCodecError(
                "native codec returned inconsistent accounting: "
                f"blocks={stats.blocks}/{codec_blocks}, "
                f"values={stats.values}/{expected_values}, "
                f"raw_bytes={stats.raw_bytes}/{expected_values * 2}, "
                f"record_bytes={stats.record_bytes}/"
                f"{codec_blocks * self.record_bytes_per_block}, "
                f"modes={stats.primary_blocks}+{stats.fallback_blocks}"
            )
        if (
            stats.primary_mode != self.codec.spec.primary_mode
            or stats.fallback_mode != self.codec.spec.fallback_mode
        ):
            raise TransformersCacheCodecError(
                "native codec mode labels do not match the codec catalog"
            )

    @staticmethod
    def _native_counter(stats: RoundtripStats) -> Counter[str]:
        return Counter(
            blocks=stats.blocks,
            values=stats.values,
            raw_bytes=stats.raw_bytes,
            record_bytes=stats.record_bytes,
            primary_blocks=stats.primary_blocks,
            fallback_blocks=stats.fallback_blocks,
            exceptions=stats.exceptions,
            m1_groups=stats.m1_groups,
            m0_groups=stats.m0_groups,
        )

    def _roundtrip_update(
        self,
        cache: Any,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Mapping[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_kv(key, value)
        leading, aligned_tokens, trailing = self._validate_positions(
            cache, layer_idx, int(key.shape[2]), cache_kwargs
        )
        token_blocks = aligned_tokens // self.BLOCK_SIZE

        update_stats: Counter[str] = Counter(
            calls=1,
            input_tokens=int(key.shape[2]),
            processed_tokens=aligned_tokens,
            skipped_tokens=leading + trailing,
            skipped_leading_tokens=leading,
            skipped_trailing_tokens=trailing,
            token_blocks=token_blocks,
        )
        if token_blocks == 0:
            update_stats["calls_without_full_block"] = 1
            self._commit_stats(layer_idx, update_stats)
            return key, value

        patched_key = key.clone()
        patched_value = value.clone()
        processed_blocks = 0
        for first_block in range(0, token_blocks, self.token_blocks_per_batch):
            chunk_blocks = min(self.token_blocks_per_batch, token_blocks - first_block)
            first_token = leading + first_block * self.BLOCK_SIZE
            last_token = first_token + chunk_blocks * self.BLOCK_SIZE
            key_cpu = (
                key[:, :, first_token:last_token, :]
                .detach()
                .to(device="cpu", copy=True)
                .contiguous()
            )
            value_cpu = (
                value[:, :, first_token:last_token, :]
                .detach()
                .to(device="cpu", copy=True)
                .contiguous()
            )
            payload = self._make_payload(key_cpu, value_cpu, chunk_blocks)
            decoded, native_stats = self.codec.roundtrip_blocks(payload)
            codec_blocks = chunk_blocks * self.SIMULATED_TP_SIZE
            self._validate_native_stats(native_stats, codec_blocks)
            decoded_key, decoded_value = self._decode_payload(decoded, chunk_blocks)
            patched_key[:, :, first_token:last_token, :].copy_(
                decoded_key.to(device=key.device)
            )
            patched_value[:, :, first_token:last_token, :].copy_(
                decoded_value.to(device=value.device)
            )
            update_stats.update(self._native_counter(native_stats))
            processed_blocks += chunk_blocks

        if processed_blocks != token_blocks:
            raise TransformersCacheCodecError(
                f"internal token-block mismatch: expected={token_blocks}, "
                f"processed={processed_blocks}"
            )
        self._commit_stats(layer_idx, update_stats)
        return patched_key, patched_value

    def _commit_stats(self, layer_idx: int, values: Counter[str]) -> None:
        with self._stats_lock:
            self._stats.update(values)
            self._layer_stats[layer_idx].update(values)

    def _record_failure(self, layer_idx: int | None) -> None:
        with self._stats_lock:
            self._stats["failures"] += 1
            if layer_idx is not None:
                self._layer_stats[layer_idx]["failures"] += 1

    def _make_wrapper(self, original_update: Any):
        @functools.wraps(original_update)
        def wrapped_update(
            cache: Any,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Mapping[str, Any] | None = None,
        ):
            normalized_layer: int | None = None
            try:
                normalized_layer = self._layer_index(layer_idx)
                patched_key, patched_value = self._roundtrip_update(
                    cache,
                    key_states,
                    value_states,
                    normalized_layer,
                    cache_kwargs,
                )
            except Exception:
                self._record_failure(normalized_layer)
                raise
            try:
                if cache_kwargs is None:
                    return original_update(
                        cache,
                        patched_key,
                        patched_value,
                        normalized_layer,
                    )
                return original_update(
                    cache,
                    patched_key,
                    patched_value,
                    normalized_layer,
                    cache_kwargs,
                )
            except Exception:
                self._record_failure(normalized_layer)
                raise

        return wrapped_update

    def __enter__(self) -> KvfoldCacheProcessor:
        global _ACTIVE_PATCH

        try:
            from transformers.cache_utils import DynamicCache
        except ImportError as error:
            raise TransformersCacheCodecError(
                "Transformers is required for the offline DynamicCache hook"
            ) from error

        with _PATCH_LOCK:
            if _ACTIVE_PATCH is not None:
                raise TransformersCacheCodecError(
                    "another DynamicCache codec patch is already active"
                )
            if self._installed:
                raise TransformersCacheCodecError("this cache patch is already active")
            original_update = getattr(DynamicCache, "update", None)
            if not callable(original_update):
                raise TransformersCacheCodecError(
                    "transformers.cache_utils.DynamicCache.update is not callable"
                )
            wrapper = self._make_wrapper(original_update)
            DynamicCache.update = wrapper
            self._dynamic_cache_class = DynamicCache
            self._original_update = original_update
            self._wrapped_update = wrapper
            self._installed = True
            _ACTIVE_PATCH = self
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        global _ACTIVE_PATCH

        restoration_error = None
        with _PATCH_LOCK:
            if not self._installed or _ACTIVE_PATCH is not self:
                restoration_error = "this instance does not own the active cache patch"
            elif self._dynamic_cache_class.update is not self._wrapped_update:
                restoration_error = (
                    "DynamicCache.update changed while the KVfold patch was active"
                )
            else:
                self._dynamic_cache_class.update = self._original_update
            if _ACTIVE_PATCH is self:
                _ACTIVE_PATCH = None
            self._installed = False
        if restoration_error is not None and exc_type is None:
            raise TransformersCacheCodecError(restoration_error)
        return False

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable, internally consistent stats snapshot."""

        with self._stats_lock:
            totals = Counter(self._stats)
            layers = {
                str(layer): dict(sorted(values.items()))
                for layer, values in sorted(self._layer_stats.items())
            }
        counters = {key: int(totals[key]) for key in self._COUNTER_FIELDS}
        layer_blocks = {
            layer: int(values.get("blocks", 0)) for layer, values in layers.items()
        }
        return {
            "stats_schema_version": self.STATS_SCHEMA_VERSION,
            "codec": self.codec.codec,
            "codec_id": self.codec.codec_id,
            "codec_source_id": self.codec.source_id,
            "primary_mode": self.codec.spec.primary_mode,
            "fallback_mode": self.codec.spec.fallback_mode,
            "mode_counts": {
                self.codec.spec.primary_mode: int(totals["primary_blocks"]),
                self.codec.spec.fallback_mode: int(totals["fallback_blocks"]),
            },
            "block_size": self.BLOCK_SIZE,
            "simulated_tp_size": self.SIMULATED_TP_SIZE,
            "global_kv_heads": self.EXPECTED_KV_HEADS,
            "local_kv_heads": self.LOCAL_KV_HEADS,
            "head_dim": self.EXPECTED_HEAD_DIM,
            "values_per_codec_block": self.VALUES_PER_CODEC_BLOCK,
            "record_bytes_per_block": self.record_bytes_per_block,
            "token_blocks_per_batch": self.token_blocks_per_batch,
            "installed": self._installed,
            "layer_count": len(layers),
            "layer_blocks": layer_blocks,
            "layers": layers,
            "counters": counters,
            **counters,
        }


@contextmanager
def patch_dynamic_cache(
    processor: KvfoldCacheProcessor,
) -> Iterator[KvfoldCacheProcessor]:
    """Temporarily patch ``DynamicCache.update`` with ``processor``."""

    if not isinstance(processor, KvfoldCacheProcessor):
        raise TypeError("processor must be a KvfoldCacheProcessor")
    with processor:
        yield processor


def new_dynamic_cache(config: Any):
    """Create the native DynamicCache across old and new Transformers APIs."""

    try:
        from transformers.cache_utils import DynamicCache
    except ImportError as error:
        raise TransformersCacheCodecError(
            "Transformers is required for the offline DynamicCache hook"
        ) from error
    parameters = inspect.signature(DynamicCache.__init__).parameters
    if "config" in parameters:
        return DynamicCache(config=config)
    return DynamicCache()


__all__ = [
    "KvfoldCacheProcessor",
    "TransformersCacheCodecError",
    "new_dynamic_cache",
    "patch_dynamic_cache",
]
