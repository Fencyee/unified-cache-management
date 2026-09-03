# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

"""Single Python catalog for every codec exposed by the native bridge."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodecSpec:
    name: str
    codec_id: int
    display_name: str
    primary_mode: str
    fallback_mode: str
    record_bytes_per_block: int
    aliases: tuple[str, ...] = ()


CODECS = (
    CodecSpec(
        name="tunstall_bf16_r160",
        codec_id=3,
        display_name="Tunstall BF16 R160",
        primary_mode="high_precision",
        fallback_mode="quantized",
        record_bytes_per_block=81920,
        aliases=("r160_tunstall",),
    ),
    CodecSpec(
        name="tunstall_bf16_r200",
        codec_id=2,
        display_name="Tunstall BF16 R200",
        primary_mode="tunstall",
        fallback_mode="fp8_fallback",
        record_bytes_per_block=65536,
        aliases=("r200",),
    ),
    CodecSpec(
        name="r160_base_bf16",
        codec_id=1,
        display_name="Base15 BF16 R160",
        primary_mode="base15",
        fallback_mode="unsupported",
        record_bytes_per_block=86016,
        aliases=("r160", "r160_base15"),
    ),
)

_BY_NAME = {
    candidate: spec for spec in CODECS for candidate in (spec.name, *spec.aliases)
}


def accepted_codec_names() -> tuple[str, ...]:
    """Return canonical names followed by supported command-line aliases."""
    canonical = tuple(spec.name for spec in CODECS)
    aliases = tuple(alias for spec in CODECS for alias in spec.aliases)
    return canonical + aliases


def resolve_codec(value: str) -> CodecSpec:
    """Resolve a canonical name or compatibility alias to one codec spec."""
    normalized = value.strip().lower().replace("-", "_")
    spec = _BY_NAME.get(normalized)
    if spec is None:
        expected = ", ".join(spec.name for spec in CODECS)
        raise ValueError(f"unknown KVfold codec {value!r}; expected one of: {expected}")
    return spec
