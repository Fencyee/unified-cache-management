# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

"""ctypes wrapper for the self-contained KVfold BF16 codec bridge."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from codec_catalog import CodecSpec, resolve_codec


class KvfoldCodecError(RuntimeError):
    """Raised when the native KVfold codec rejects a block."""


class _NativeStats(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("codec_id", ctypes.c_uint32),
        ("blocks", ctypes.c_uint64),
        ("values", ctypes.c_uint64),
        ("raw_bytes", ctypes.c_uint64),
        ("record_bytes", ctypes.c_uint64),
        ("primary_blocks", ctypes.c_uint64),
        ("fallback_blocks", ctypes.c_uint64),
        ("exceptions", ctypes.c_uint64),
        ("m1_groups", ctypes.c_uint64),
        ("m0_groups", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class RoundtripStats:
    codec: str
    primary_mode: str
    fallback_mode: str
    blocks: int
    values: int
    raw_bytes: int
    record_bytes: int
    primary_blocks: int
    fallback_blocks: int
    exceptions: int
    m1_groups: int
    m0_groups: int

    @property
    def mode_counts(self) -> dict[str, int]:
        return {
            self.primary_mode: self.primary_blocks,
            self.fallback_mode: self.fallback_blocks,
        }


def _default_library() -> Path:
    return Path(__file__).resolve().parent / "build" / "libkvfold_longbench_bridge.so"


class KvfoldCodec:
    ABI_VERSION = 2

    def __init__(
        self,
        codec: str | None = None,
        library: str | os.PathLike[str] | None = None,
    ) -> None:
        selected = codec or os.environ.get("KVFOLD_CODEC", "")
        try:
            self.spec: CodecSpec = resolve_codec(selected)
        except ValueError as error:
            raise KvfoldCodecError(str(error)) from error
        self.codec = self.spec.name
        self.codec_id = self.spec.codec_id

        path = Path(
            library or os.environ.get("KVFOLD_CODEC_BRIDGE", _default_library())
        )
        if not path.is_file():
            raise KvfoldCodecError(
                f"KVfold bridge not found: {path}. Run build_bridge.sh first."
            )
        self.path = path.resolve()
        self._lib = ctypes.CDLL(str(self.path))
        self._configure_api()
        actual_abi = int(self._lib.kvfold_longbench_abi_version())
        if actual_abi != self.ABI_VERSION:
            raise KvfoldCodecError(
                f"KVfold bridge ABI mismatch: expected {self.ABI_VERSION}, got {actual_abi}"
            )
        native_name = self._decode(self._lib.kvfold_longbench_codec_name(self.codec_id))
        if native_name != self.codec:
            raise KvfoldCodecError(
                f"KVfold codec ID mismatch: requested={self.codec}, native={native_name}"
            )
        self.source_id = self._decode(
            self._lib.kvfold_longbench_codec_source_id(self.codec_id)
        )
        if self.source_id in {"", "unknown", "invalid"}:
            raise KvfoldCodecError(
                f"KVfold bridge has an invalid source ID: {self.source_id!r}"
            )

    @staticmethod
    def _decode(value: bytes | None) -> str:
        return value.decode("utf-8", errors="replace") if value else ""

    def _configure_api(self) -> None:
        self._lib.kvfold_longbench_abi_version.argtypes = []
        self._lib.kvfold_longbench_abi_version.restype = ctypes.c_uint32
        self._lib.kvfold_longbench_codec_name.argtypes = [ctypes.c_uint32]
        self._lib.kvfold_longbench_codec_name.restype = ctypes.c_char_p
        self._lib.kvfold_longbench_codec_source_id.argtypes = [ctypes.c_uint32]
        self._lib.kvfold_longbench_codec_source_id.restype = ctypes.c_char_p
        self._lib.kvfold_longbench_record_bytes.argtypes = [
            ctypes.c_uint32,
            ctypes.c_size_t,
        ]
        self._lib.kvfold_longbench_record_bytes.restype = ctypes.c_size_t
        self._lib.kvfold_longbench_error_name.argtypes = [ctypes.c_int]
        self._lib.kvfold_longbench_error_name.restype = ctypes.c_char_p
        self._lib.kvfold_longbench_roundtrip_bf16_blocks.argtypes = [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(_NativeStats),
        ]
        self._lib.kvfold_longbench_roundtrip_bf16_blocks.restype = ctypes.c_int

    def record_bytes(self, values: int) -> int:
        return int(self._lib.kvfold_longbench_record_bytes(self.codec_id, int(values)))

    def _error_name(self, error: int) -> str:
        return self._decode(self._lib.kvfold_longbench_error_name(error))

    def roundtrip(self, source: torch.Tensor) -> tuple[torch.Tensor, RoundtripStats]:
        return self._roundtrip_blocks(source, 1, source.numel())

    def roundtrip_blocks(
        self, source: torch.Tensor
    ) -> tuple[torch.Tensor, RoundtripStats]:
        if source.ndim < 2:
            raise KvfoldCodecError(
                "batched KVfold source must have a block dimension, "
                f"got shape={tuple(source.shape)}"
            )
        blocks = int(source.shape[0])
        values_per_block = source.numel() // blocks if blocks else 0
        return self._roundtrip_blocks(source, blocks, values_per_block)

    def _roundtrip_blocks(
        self, source: torch.Tensor, blocks: int, values_per_block: int
    ) -> tuple[torch.Tensor, RoundtripStats]:
        if source.device.type != "cpu":
            raise KvfoldCodecError("KVfold native codec accepts CPU tensors only")
        if source.dtype != torch.bfloat16:
            raise KvfoldCodecError(f"expected torch.bfloat16, got {source.dtype}")
        if not source.is_contiguous():
            raise KvfoldCodecError("KVfold source tensor must be contiguous")
        values = source.numel()
        if (
            blocks <= 0
            or values_per_block <= 0
            or values != blocks * values_per_block
            or values_per_block % 32 != 0
        ):
            raise KvfoldCodecError(
                "each KVfold block must contain a positive multiple of 32 values: "
                f"shape={tuple(source.shape)}, blocks={blocks}, "
                f"values_per_block={values_per_block}"
            )

        decoded = torch.empty_like(source)
        native = _NativeStats()
        native.struct_size = ctypes.sizeof(_NativeStats)
        error = int(
            self._lib.kvfold_longbench_roundtrip_bf16_blocks(
                self.codec_id,
                ctypes.c_void_p(source.data_ptr()),
                ctypes.c_void_p(decoded.data_ptr()),
                blocks,
                values_per_block,
                ctypes.byref(native),
            )
        )
        if error != 0:
            raise KvfoldCodecError(
                f"KVfold {self.codec} roundtrip failed: "
                f"error={error} ({self._error_name(error)}), blocks={blocks}, "
                f"values_per_block={values_per_block}, "
                f"record_bytes={self.record_bytes(values_per_block)}"
            )
        if native.codec_id != self.codec_id:
            raise KvfoldCodecError(
                f"native stats codec mismatch: expected={self.codec_id}, "
                f"got={native.codec_id}"
            )
        return decoded, RoundtripStats(
            codec=self.codec,
            primary_mode=self.spec.primary_mode,
            fallback_mode=self.spec.fallback_mode,
            blocks=int(native.blocks),
            values=int(native.values),
            raw_bytes=int(native.raw_bytes),
            record_bytes=int(native.record_bytes),
            primary_blocks=int(native.primary_blocks),
            fallback_blocks=int(native.fallback_blocks),
            exceptions=int(native.exceptions),
            m1_groups=int(native.m1_groups),
            m0_groups=int(native.m0_groups),
        )
