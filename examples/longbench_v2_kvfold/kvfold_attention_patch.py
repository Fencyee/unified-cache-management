# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

"""Monkeypatch vLLM-Ascend to round-trip newly generated K/V through KVfold."""

from __future__ import annotations

import atexit
import functools
import hashlib
import inspect
import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from wrapt import register_post_import_hook

_ATTENTION_PATCH_MARKER = "_kvfold_longbench_original_forward"
_WORKER_SHUTDOWN_PATCH_MARKER = "_kvfold_longbench_original_shutdown"
_STAT_COUNTER_FIELDS = (
    "calls",
    "calls_without_full_block",
    "blocks",
    "values",
    "raw_bytes",
    "record_bytes",
    "primary_blocks",
    "fallback_blocks",
    "exceptions",
    "m1_groups",
    "m0_groups",
    "processed_tokens",
    "skipped_tail_tokens",
    "failures",
)
_stats = Counter()
_layer_blocks = Counter()
_stats_lock = threading.Lock()
_stats_write_lock = threading.Lock()
_codec = None
_installed = False
_final_stats_written = False
_reported_shape = False
_reported_runtime = False
_server_config_path = ""
_server_config_sha256 = ""
_worker_rank = None
_expected_layer_count = 0


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _log(message: str) -> None:
    print(f"[KVFOLD-ATTN pid={os.getpid()}] {message}", file=sys.stderr, flush=True)


def _get_codec():
    global _codec
    if _codec is None:
        from kvfold_codec import KvfoldCodec

        _codec = KvfoldCodec()
    return _codec


def _server_config_identity() -> tuple[str, str]:
    configured = os.environ.get("KVFOLD_SERVER_CONFIG", "").strip()
    if not configured:
        raise ValueError("KVFOLD_SERVER_CONFIG must name the config used to start vLLM")
    path = Path(configured).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"KVfold server config does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return str(path), digest.hexdigest()


def _model_architectures(instance: Any) -> set[str]:
    model_config = getattr(getattr(instance, "vllm_config", None), "model_config", None)
    architectures: set[str] = set()
    architecture = getattr(model_config, "architecture", None)
    if isinstance(architecture, str):
        architectures.add(architecture)
    hf_config = getattr(model_config, "hf_config", None)
    values = getattr(hf_config, "architectures", None)
    if values:
        architectures.update(str(value) for value in values)
    return architectures


def _has_kv_cache(kv_cache: Any) -> bool:
    if kv_cache is None:
        return False
    if isinstance(kv_cache, (list, tuple)):
        return len(kv_cache) >= 2 and all(
            int(getattr(item, "numel", lambda: 0)()) > 0 for item in kv_cache[:2]
        )
    return int(getattr(kv_cache, "numel", lambda: 0)()) > 0


def _cache_layout(kv_cache: Any) -> tuple[int, int, int]:
    import torch

    if isinstance(kv_cache, (list, tuple)):
        caches = kv_cache[:2]
    elif int(getattr(kv_cache, "ndim", 0)) == 5 and int(kv_cache.shape[0]) == 2:
        caches = (kv_cache[0], kv_cache[1])
    else:
        raise ValueError(f"unsupported KV cache container: {type(kv_cache).__name__}")
    if len(caches) != 2 or any(int(getattr(cache, "ndim", 0)) != 4 for cache in caches):
        raise ValueError(
            "expected K/V cache tensors shaped [blocks, block, heads, dim]"
        )
    shapes = [tuple(int(value) for value in cache.shape) for cache in caches]
    if shapes[0] != shapes[1]:
        raise ValueError(f"K/V cache shape mismatch: {shapes}")
    if any(cache.dtype != torch.bfloat16 for cache in caches):
        raise TypeError(
            "KVfold requires BF16 paged K/V cache, got "
            f"{[str(cache.dtype) for cache in caches]}"
        )
    return shapes[0][1], shapes[0][2], shapes[0][3]


def _validate_slot_mapping(
    attn_metadata: Any, full_tokens: int, block_size: int
) -> None:
    """Require every selected group to be one complete physical cache block."""
    if full_tokens == 0:
        return
    import torch

    slot_mapping = getattr(attn_metadata, "slot_mapping", None)
    if slot_mapping is None:
        raise ValueError("attention metadata has no slot_mapping")
    slots = slot_mapping.detach().reshape(-1)
    if int(slots.numel()) < full_tokens:
        raise ValueError(
            f"slot_mapping is shorter than selected tokens: {slots.numel()} < {full_tokens}"
        )
    slots = slots[:full_tokens].to(device="cpu", dtype=torch.int64, copy=True)
    groups = slots.view(-1, block_size)
    expected_offsets = torch.arange(block_size, dtype=torch.int64).view(1, -1)
    offsets = torch.remainder(groups, block_size)
    block_ids = torch.div(groups, block_size, rounding_mode="floor")
    contiguous = bool(torch.equal(offsets, expected_offsets.expand_as(offsets)))
    same_block = bool(torch.all(block_ids == block_ids[:, :1]).item())
    if not contiguous or not same_block or bool(torch.any(groups < 0).item()):
        raise ValueError(
            "selected tokens do not form complete physical cache blocks; "
            "run serial requests with block-aligned chunked prefill"
        )


def _validate_runtime(
    instance: Any,
    key: Any,
    value: Any,
    kv_cache: Any,
    block_size: int,
    expected_tp_size: int,
    expected_local_kv_heads: int,
    expected_head_size: int,
) -> None:
    global _reported_runtime
    vllm_config = getattr(instance, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    if getattr(model_config, "enforce_eager", None) is not True:
        raise RuntimeError("KVfold attention test requires vLLM --enforce-eager")
    cache_config = getattr(vllm_config, "cache_config", None)
    if getattr(cache_config, "enable_prefix_caching", None) is not False:
        raise RuntimeError(
            "KVfold attention test requires prefix caching to be disabled"
        )
    if getattr(vllm_config, "speculative_config", None) is not None:
        raise RuntimeError(
            "KVfold attention test does not support speculative decoding"
        )
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    max_num_batched_tokens = getattr(scheduler_config, "max_num_batched_tokens", None)
    if (
        not isinstance(max_num_batched_tokens, int)
        or max_num_batched_tokens <= 0
        or max_num_batched_tokens % block_size != 0
    ):
        raise ValueError(
            "max_num_batched_tokens must be a positive multiple of "
            f"{block_size}, got {max_num_batched_tokens}"
        )
    max_num_seqs = getattr(scheduler_config, "max_num_seqs", None)
    if max_num_seqs != 1:
        raise ValueError(
            f"KVfold attention test requires max_num_seqs=1, got {max_num_seqs}"
        )
    for name in ("max_num_scheduled_tokens", "long_prefill_token_threshold"):
        scheduler_value = getattr(scheduler_config, name, None)
        if scheduler_value not in (None, 0) and (
            not isinstance(scheduler_value, int) or scheduler_value % block_size != 0
        ):
            raise ValueError(
                f"{name} must be zero or a multiple of {block_size}, got {scheduler_value}"
            )
    parallel_config = getattr(vllm_config, "parallel_config", None)
    actual_tp_size = getattr(parallel_config, "tensor_parallel_size", None)
    if actual_tp_size != expected_tp_size:
        raise ValueError(
            f"expected TP={expected_tp_size}, got tensor_parallel_size={actual_tp_size}"
        )
    actual_cache_block, cache_heads, cache_head_size = _cache_layout(kv_cache)
    if actual_cache_block != block_size:
        raise ValueError(
            f"configured block_size={block_size}, but KV cache uses {actual_cache_block}"
        )
    if cache_heads != expected_local_kv_heads or cache_head_size != expected_head_size:
        raise ValueError(
            "unexpected paged K/V cache shape: "
            f"local_heads={cache_heads}, head_size={cache_head_size}, "
            f"expected local_heads={expected_local_kv_heads}, "
            f"head_size={expected_head_size}"
        )
    actual_heads = int(key.shape[1])
    actual_head_size = int(key.shape[2])
    if (
        actual_heads != expected_local_kv_heads
        or actual_head_size != expected_head_size
        or tuple(key.shape[1:]) != tuple(value.shape[1:])
    ):
        raise ValueError(
            "unexpected Qwen3 K/V shape: "
            f"key={tuple(key.shape)}, value={tuple(value.shape)}, "
            f"expected local_heads={expected_local_kv_heads}, head_size={expected_head_size}"
        )
    if not _reported_runtime:
        _reported_runtime = True
        _log(
            f"runtime validated: eager=True, TP={actual_tp_size}, "
            f"cache_block={actual_cache_block}, local_kv_heads={actual_heads}, "
            f"head_size={actual_head_size}, max_num_batched_tokens={max_num_batched_tokens}, "
            "max_num_seqs=1, prefix_cache=False, speculative=False"
        )


def _roundtrip_kv(
    key,
    value,
    attn_metadata: Any,
    actual_tokens: int,
    block_size: int,
    layer_name: str,
):
    global _reported_shape
    import torch

    if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        raise TypeError(
            f"KVfold requires BF16 K/V, got key={key.dtype}, value={value.dtype}"
        )
    if key.ndim != 3 or value.ndim != 3:
        raise ValueError(
            f"expected 3-D K/V, got key={tuple(key.shape)}, value={tuple(value.shape)}"
        )
    if key.shape[1:] != value.shape[1:]:
        raise ValueError(
            f"K/V shape mismatch: key={tuple(key.shape)}, value={tuple(value.shape)}"
        )

    usable_tokens = int(actual_tokens)
    if (
        usable_tokens < 0
        or usable_tokens > int(key.shape[0])
        or usable_tokens > int(value.shape[0])
    ):
        raise ValueError(
            f"invalid num_actual_tokens={usable_tokens} for "
            f"key={tuple(key.shape)}, value={tuple(value.shape)}"
        )
    full_tokens = usable_tokens // block_size * block_size
    skipped_tokens = usable_tokens - full_tokens
    if full_tokens == 0:
        with _stats_lock:
            _stats["calls_without_full_block"] += 1
            _stats["skipped_tail_tokens"] += skipped_tokens
        return key, value

    _validate_slot_mapping(attn_metadata, full_tokens, block_size)

    key_cpu = key[:full_tokens].detach().to(device="cpu", copy=True).contiguous()
    value_cpu = value[:full_tokens].detach().to(device="cpu", copy=True).contiguous()
    codec = _get_codec()

    local_heads = int(key.shape[1])
    head_size = int(key.shape[2])
    block_count = full_tokens // block_size
    key_blocks = key_cpu.view(block_count, block_size, local_heads, head_size)
    value_blocks = value_cpu.view(block_count, block_size, local_heads, head_size)
    # Match UCM's block payload: one native token/head/channel K block followed
    # by one native token/head/channel V block. Batch all blocks across the C
    # boundary to avoid one Python/ctypes transition per 128 tokens.
    payloads = torch.stack((key_blocks, value_blocks), dim=1).contiguous()
    decoded, block_stats = codec.roundtrip_blocks(payloads)
    decoded_key = decoded[:, 0].reshape(full_tokens, local_heads, head_size)
    decoded_value = decoded[:, 1].reshape(full_tokens, local_heads, head_size)
    aggregate = Counter(
        blocks=block_stats.blocks,
        values=block_stats.values,
        raw_bytes=block_stats.raw_bytes,
        record_bytes=block_stats.record_bytes,
        primary_blocks=block_stats.primary_blocks,
        fallback_blocks=block_stats.fallback_blocks,
        exceptions=block_stats.exceptions,
        m1_groups=block_stats.m1_groups,
        m0_groups=block_stats.m0_groups,
    )

    patched_key = key.clone()
    patched_value = value.clone()
    patched_key[:full_tokens].copy_(decoded_key)
    patched_value[:full_tokens].copy_(decoded_value)
    with _stats_lock:
        _stats.update(aggregate)
        _stats["calls"] += 1
        _stats["processed_tokens"] += full_tokens
        _stats["skipped_tail_tokens"] += skipped_tokens
        _layer_blocks[layer_name] += block_count
        first_shape = not _reported_shape
        _reported_shape = True
        calls = _stats["calls"]
    if first_shape:
        factor = aggregate["raw_bytes"] / aggregate["record_bytes"]
        _log(
            f"first {codec.codec} block path active: "
            f"K/V={tuple(key.shape)}, actual_tokens={usable_tokens}, block_size={block_size}, "
            f"local_kv_heads={local_heads}, head_size={head_size}, "
            f"raw/record={factor:.6f}x"
        )
    if calls == 1 or calls % 10000 == 0:
        _write_stats(log_summary=False)
    return patched_key, patched_value


def _detect_worker_rank() -> int:
    global _worker_rank
    if _worker_rank is not None:
        return _worker_rank
    try:
        import torch.distributed as distributed

        if distributed.is_available() and distributed.is_initialized():
            _worker_rank = int(distributed.get_rank())
            return _worker_rank
    except Exception:
        pass
    for variable in ("RANK", "LOCAL_RANK"):
        value = os.environ.get(variable, "").strip()
        if value:
            try:
                _worker_rank = int(value)
                return _worker_rank
            except ValueError:
                pass
    _worker_rank = -1
    return _worker_rank


def _write_stats(log_summary: bool = True, final: bool = False) -> None:
    global _final_stats_written

    if not _installed:
        return
    with _stats_write_lock:
        # Once the shutdown path has published a final snapshot, no delayed
        # periodic writer may replace it with final=false.
        if _final_stats_written:
            return
        with _stats_lock:
            if final and _codec is None and not _stats and not _layer_blocks:
                return
            snapshot = {
                field: int(_stats.get(field, 0)) for field in _STAT_COUNTER_FIELDS
            }
            snapshot.update(
                {
                    field: int(value)
                    for field, value in _stats.items()
                    if field not in snapshot
                }
            )
            layer_blocks = dict(sorted(_layer_blocks.items()))
        codec_spec = getattr(_codec, "spec", None)
        primary_mode = str(getattr(codec_spec, "primary_mode", "not-loaded"))
        fallback_mode = str(getattr(codec_spec, "fallback_mode", "not-loaded"))
        snapshot.update(
            {
                "stats_schema_version": 2,
                "final": final,
                "pid": os.getpid(),
                "rank": _detect_worker_rank(),
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "bridge": str(getattr(_codec, "path", "not-loaded")),
                "codec": str(getattr(_codec, "codec", "not-loaded")),
                "codec_id": int(getattr(_codec, "codec_id", 0)),
                "codec_source_id": str(getattr(_codec, "source_id", "not-loaded")),
                "primary_mode": primary_mode,
                "fallback_mode": fallback_mode,
                "mode_counts": {
                    primary_mode: int(snapshot["primary_blocks"]),
                    fallback_mode: int(snapshot["fallback_blocks"]),
                },
                "expected_layer_count": _expected_layer_count,
                "layer_count": len(layer_blocks),
                "layer_names": list(layer_blocks),
                "layer_blocks": layer_blocks,
                "run_id": os.environ.get("KVFOLD_RUN_ID", ""),
                "server_config": _server_config_path,
                "server_config_sha256": _server_config_sha256,
            }
        )
        if log_summary:
            _log("summary: " + json.dumps(snapshot, sort_keys=True))
        stats_dir = os.environ.get("KVFOLD_STATS_DIR")
        if stats_dir:
            path = Path(stats_dir)
            path.mkdir(parents=True, exist_ok=True)
            target = path / f"kvfold-attention-{os.getpid()}.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8"
            )
            temporary.replace(target)
        if final:
            _final_stats_written = True


def _write_final_stats() -> None:
    # sitecustomize also runs in non-worker utility processes when users leave
    # KVFOLD_ATTN_ENABLE=1 in their shell. _write_stats() suppresses the
    # misleading not-loaded final snapshot in that case.
    _write_stats(final=True)


def _patch_worker_shutdown_module(module) -> None:
    """Persist the final snapshot before a vLLM model worker tears down."""

    cls = getattr(module, "WorkerProc", None)
    if cls is None:
        raise RuntimeError("vllm multiproc executor has no WorkerProc")
    if hasattr(cls, _WORKER_SHUTDOWN_PATCH_MARKER):
        return

    original = cls.shutdown
    actual_parameters = list(inspect.signature(original).parameters)
    if actual_parameters != ["self"]:
        raise RuntimeError(
            "unsupported WorkerProc.shutdown signature: "
            f"expected=['self'], actual={actual_parameters}"
        )

    @functools.wraps(original)
    def wrapped(self):
        # Write before the original shutdown: NPU/distributed cleanup may block
        # long enough for the parent process to terminate this worker.
        try:
            _write_final_stats()
        except BaseException as error:
            # Statistics must never prevent vLLM from releasing its resources.
            _log(f"failed to write final stats before worker shutdown: {error}")
        return original(self)

    setattr(cls, _WORKER_SHUTDOWN_PATCH_MARKER, original)
    cls.shutdown = wrapped
    _log("installed final-stats hook on WorkerProc.shutdown")


def _patch_attention_module(module) -> None:
    global _expected_layer_count, _server_config_path, _server_config_sha256

    cls = getattr(module, "AscendAttentionBackendImpl", None)
    if cls is None:
        raise RuntimeError(
            "vllm_ascend.attention.attention_v1 has no AscendAttentionBackendImpl"
        )
    if hasattr(cls, _ATTENTION_PATCH_MARKER):
        return

    original = cls.forward
    expected_parameters = [
        "self",
        "layer",
        "query",
        "key",
        "value",
        "kv_cache",
        "attn_metadata",
        "output",
        "output_scale",
        "output_block_scale",
    ]
    actual_parameters = list(inspect.signature(original).parameters)
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            "unsupported AscendAttentionBackendImpl.forward signature: "
            f"expected={expected_parameters}, actual={actual_parameters}"
        )
    block_size = int(os.environ.get("KVFOLD_TOKEN_BLOCK_SIZE", "128"))
    if block_size <= 0:
        raise ValueError(f"KVFOLD_TOKEN_BLOCK_SIZE must be positive, got {block_size}")
    strict = _env_bool("KVFOLD_ATTN_STRICT", True)
    include_subclasses = _env_bool("KVFOLD_ATTN_INCLUDE_SUBCLASSES", False)
    target_architecture = os.environ.get(
        "KVFOLD_TARGET_ARCHITECTURE", "Qwen3ForCausalLM"
    )
    expected_tp_size = int(os.environ.get("KVFOLD_EXPECTED_TP_SIZE", "4"))
    expected_local_kv_heads = int(os.environ.get("KVFOLD_EXPECTED_LOCAL_KV_HEADS", "2"))
    expected_head_size = int(os.environ.get("KVFOLD_EXPECTED_HEAD_SIZE", "128"))
    _expected_layer_count = int(os.environ.get("KVFOLD_EXPECTED_LAYER_COUNT", "64"))
    run_id = os.environ.get("KVFOLD_RUN_ID", "").strip()
    if (
        min(
            expected_tp_size,
            expected_local_kv_heads,
            expected_head_size,
            _expected_layer_count,
        )
        <= 0
    ):
        raise ValueError("KVfold expected TP/head/layer settings must all be positive")
    if not run_id:
        raise ValueError("KVFOLD_RUN_ID must be a non-empty, unique run label")
    _server_config_path, _server_config_sha256 = _server_config_identity()
    # Fail during worker initialization instead of reaching the first
    # LongBench request with a missing or ABI-incompatible native bridge.
    _get_codec()

    @functools.wraps(original)
    def wrapped(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        has_inputs = (
            key is not None
            and value is not None
            and attn_metadata is not None
            and _has_kv_cache(kv_cache)
        )
        if has_inputs:
            try:
                if not include_subclasses and type(self) is not cls:
                    raise RuntimeError(
                        f"unsupported attention subclass: {type(self).__qualname__}"
                    )
                architectures = _model_architectures(self)
                if target_architecture not in architectures:
                    raise RuntimeError(
                        f"expected architecture={target_architecture}, got {sorted(architectures)}"
                    )
                if not hasattr(attn_metadata, "num_actual_tokens"):
                    raise RuntimeError("attention metadata has no num_actual_tokens")
                actual_tokens = int(attn_metadata.num_actual_tokens)
                layer_name = str(getattr(layer, "layer_name", "")).strip()
                if not layer_name:
                    raise RuntimeError("attention layer has no stable layer_name")
                _validate_runtime(
                    self,
                    key,
                    value,
                    kv_cache,
                    block_size,
                    expected_tp_size,
                    expected_local_kv_heads,
                    expected_head_size,
                )
                key, value = _roundtrip_kv(
                    key,
                    value,
                    attn_metadata,
                    actual_tokens,
                    block_size,
                    layer_name,
                )
            except Exception as error:
                with _stats_lock:
                    _stats["failures"] += 1
                _write_stats(log_summary=False)
                _log(
                    f"roundtrip failed at layer={getattr(layer, 'layer_name', '?')}: {error}"
                )
                if strict:
                    raise
        return original(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

    setattr(cls, _ATTENTION_PATCH_MARKER, original)
    cls.forward = wrapped
    _log(
        "installed on AscendAttentionBackendImpl.forward; "
        f"codec={_get_codec().codec}, block_size={block_size}, strict={strict}, "
        f"target={target_architecture}, "
        f"expected_tp={expected_tp_size}, expected_local_kv_heads={expected_local_kv_heads}, "
        f"expected_head_size={expected_head_size}, expected_layers={_expected_layer_count}, "
        f"include_subclasses={include_subclasses}, "
        f"run_id={run_id}, config_sha256={_server_config_sha256}, "
        f"source={_get_codec().source_id}"
    )


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True
    register_post_import_hook(
        _patch_worker_shutdown_module, "vllm.v1.executor.multiproc_executor"
    )
    register_post_import_hook(
        _patch_attention_module, "vllm_ascend.attention.attention_v1"
    )
    atexit.register(_write_final_stats)
    _log("post-import hook registered")
