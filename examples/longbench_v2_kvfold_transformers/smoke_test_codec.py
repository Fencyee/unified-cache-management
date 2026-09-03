#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

"""Deterministic CPU smoke tests for every registered KVfold codec."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from codec_catalog import CODECS, accepted_codec_names, resolve_codec
from kvfold_codec import KvfoldCodec, KvfoldCodecError, RoundtripStats

BLOCK_SHAPE = (2, 128, 2, 128)
VALUES_PER_BLOCK = 65536
RAW_BYTES = VALUES_PER_BLOCK * 2


def make_source(
    exponents: int | torch.Tensor, force_m5_index: int | None = None
) -> torch.Tensor:
    index = torch.arange(VALUES_PER_BLOCK, dtype=torch.int32)
    if isinstance(exponents, int):
        exponent = torch.full_like(index, exponents)
    else:
        exponent = exponents.to(dtype=torch.int32).reshape(-1)
        if int(exponent.numel()) != VALUES_PER_BLOCK:
            raise ValueError("exponent fixture has the wrong size")
    sign = (index & 1) << 15
    mantissa = (index * 37 + 11) & 0x7F
    if force_m5_index is not None:
        mantissa[force_m5_index] |= 0x20
    bits = (sign | (exponent << 7) | mantissa).to(torch.int16)
    return bits.view(torch.bfloat16).reshape(BLOCK_SHAPE).contiguous()


def tensor_bits(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.view(torch.int16).to(torch.int32) & 0xFFFF).reshape(-1)


def bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare BF16 bit patterns, including NaNs with identical payloads."""
    return bool(torch.equal(left.view(torch.int16), right.view(torch.int16)))


def roundtrip(
    codec: KvfoldCodec,
    source: torch.Tensor,
    expected_record: int,
    expected_modes: tuple[int, int],
) -> tuple[torch.Tensor, RoundtripStats]:
    before = source.clone()
    decoded, stats = codec.roundtrip(source)
    if not bitwise_equal(source, before):
        raise AssertionError(f"{codec.codec} modified its input tensor")
    if decoded.shape != source.shape or decoded.dtype != torch.bfloat16:
        raise AssertionError(f"{codec.codec} changed output shape or dtype")
    expected = {
        "blocks": 1,
        "values": VALUES_PER_BLOCK,
        "raw_bytes": RAW_BYTES,
        "record_bytes": expected_record,
        "primary_blocks": expected_modes[0],
        "fallback_blocks": expected_modes[1],
    }
    for field, value in expected.items():
        if getattr(stats, field) != value:
            raise AssertionError(
                f"{codec.codec} returned {field}={getattr(stats, field)}, expected={value}"
            )
    decoded_again, stats_again = codec.roundtrip(source)
    if not bitwise_equal(decoded, decoded_again) or stats != stats_again:
        raise AssertionError(f"{codec.codec} roundtrip is not deterministic")
    return decoded, stats


def smoke_r160_base_bf16(codec: KvfoldCodec) -> None:
    source = make_source(127)
    decoded, stats = roundtrip(codec, source, 86016, (1, 0))
    source_bits = tensor_bits(source)
    expected = source_bits & 0xFFFC
    expected[: stats.m1_groups * 8] |= source_bits[: stats.m1_groups * 8] & 0x2
    expected[: stats.m0_groups * 8] |= source_bits[: stats.m0_groups * 8] & 0x1
    if not torch.equal(tensor_bits(decoded), expected):
        raise AssertionError("r160_base_bf16 reconstructed unexpected BF16 bits")
    if (stats.exceptions, stats.m1_groups, stats.m0_groups) != (0, 4095, 0):
        raise AssertionError(
            "unexpected Base15 auxiliary counts: "
            f"exceptions={stats.exceptions}, M1={stats.m1_groups}, M0={stats.m0_groups}"
        )

    overflow_source = make_source(torch.arange(VALUES_PER_BLOCK) % 256)
    try:
        codec.roundtrip(overflow_source)
    except KvfoldCodecError:
        pass
    else:
        raise AssertionError(
            "r160_base_bf16 did not reject an overflowing exception stream"
        )
    print(
        "r160_base_bf16 smoke: OK, "
        f"record={stats.record_bytes}, factor={RAW_BYTES / stats.record_bytes:.6f}x, "
        f"exceptions={stats.exceptions}, M1={stats.m1_groups}, M0={stats.m0_groups}"
    )


def smoke_tunstall_bf16_r160(codec: KvfoldCodec) -> None:
    high_source = make_source(127)
    high_decoded, high_stats = roundtrip(codec, high_source, 81920, (1, 0))
    if not bitwise_equal(high_source, high_decoded):
        raise AssertionError(
            "tunstall_bf16_r160 high-precision fixture is not bit-exact"
        )

    mode_index = 3 * VALUES_PER_BLOCK // 4
    exponents = torch.arange(VALUES_PER_BLOCK) % 256
    quant_source = make_source(exponents, force_m5_index=mode_index)
    quant_decoded, quant_stats = roundtrip(codec, quant_source, 81920, (0, 1))
    source_bits = tensor_bits(quant_source)
    clamped_exponents = exponents.clamp(112, 143)
    expected = (
        (source_bits & 0x8000) | (clamped_exponents << 7) | (source_bits & 0x78) | 0x3
    )
    expected[mode_index] &= ~0x20
    if not torch.equal(tensor_bits(quant_decoded), expected):
        raise AssertionError("tunstall_bf16_r160 quantized fixture is unexpected")

    mixed = torch.stack((high_source, quant_source))
    mixed_decoded, mixed_stats = codec.roundtrip_blocks(mixed)
    if (mixed_stats.primary_blocks, mixed_stats.fallback_blocks) != (1, 1):
        raise AssertionError("tunstall_bf16_r160 mixed mode counts are incorrect")
    if not bitwise_equal(mixed_decoded[0], high_decoded) or not bitwise_equal(
        mixed_decoded[1], quant_decoded
    ):
        raise AssertionError("tunstall_bf16_r160 batched blocks changed results")
    print(
        "tunstall_bf16_r160 smoke: OK, "
        f"record={high_stats.record_bytes}, factor={RAW_BYTES / high_stats.record_bytes:.6f}x, "
        "high_precision=1, quantized=1"
    )


def smoke_tunstall_bf16_r200(codec: KvfoldCodec) -> None:
    index = torch.arange(VALUES_PER_BLOCK, dtype=torch.int64)
    # A deterministic, model-like low-entropy exponent distribution exercises
    # the normal mark16 decoder path. A constant stream is tested separately
    # below to cover the shortest legal mark stream and its padding boundary.
    pseudo_random = (index * 40503 + 12345) & 0xFFFF
    primary_exponents = torch.where(
        pseudo_random < 26000,
        127,
        torch.where(
            pseudo_random < 46000,
            126,
            torch.where(
                pseudo_random < 56000,
                125,
                torch.where(
                    pseudo_random < 61000,
                    124,
                    torch.where(pseudo_random < 63500, 123, 122),
                ),
            ),
        ),
    )
    primary_source = make_source(primary_exponents)
    primary_decoded, primary_stats = roundtrip(codec, primary_source, 65536, (1, 0))
    changed_primary = (
        (tensor_bits(primary_source) ^ tensor_bits(primary_decoded)) & 0xFFF0
    ).count_nonzero()
    if int(changed_primary) != 0:
        raise AssertionError("tunstall_bf16_r200 changed retained Tunstall bits")

    # A constant tensor produces a very short Tunstall mark stream and leaves
    # substantial padding in the payload half. This guards the in-place
    # decoder against mistaking that padding for the optional M3 stream.
    low_entropy_source = torch.ones(BLOCK_SHAPE, dtype=torch.bfloat16)
    low_entropy_decoded, _ = roundtrip(codec, low_entropy_source, 65536, (1, 0))
    changed_low_entropy = (
        (tensor_bits(low_entropy_source) ^ tensor_bits(low_entropy_decoded)) & 0xFFF0
    ).count_nonzero()
    if int(changed_low_entropy) != 0:
        raise AssertionError("tunstall_bf16_r200 changed low-entropy retained bits")

    mode_index = VALUES_PER_BLOCK // 2
    fallback_source = make_source(112 + index % 32, force_m5_index=mode_index)
    fallback_decoded, fallback_stats = roundtrip(codec, fallback_source, 65536, (0, 1))
    source_bits = tensor_bits(fallback_source)
    expected = (source_bits & 0xFFE0) | 0x10
    expected[mode_index] &= ~0x20
    if not torch.equal(tensor_bits(fallback_decoded), expected):
        raise AssertionError("tunstall_bf16_r200 FP8 fallback fixture is unexpected")

    mixed = torch.stack((primary_source, fallback_source))
    mixed_decoded, mixed_stats = codec.roundtrip_blocks(mixed)
    if (mixed_stats.primary_blocks, mixed_stats.fallback_blocks) != (1, 1):
        raise AssertionError("tunstall_bf16_r200 mixed mode counts are incorrect")
    if not bitwise_equal(mixed_decoded[0], primary_decoded) or not bitwise_equal(
        mixed_decoded[1], fallback_decoded
    ):
        raise AssertionError("tunstall_bf16_r200 batched blocks changed results")
    print(
        "tunstall_bf16_r200 smoke: OK, "
        f"record={primary_stats.record_bytes}, "
        f"factor={RAW_BYTES / primary_stats.record_bytes:.6f}x, "
        "tunstall=1, fp8_fallback=1"
    )


SMOKE_CASES: dict[str, Callable[[KvfoldCodec], None]] = {
    "tunstall_bf16_r160": smoke_tunstall_bf16_r160,
    "tunstall_bf16_r200": smoke_tunstall_bf16_r200,
    "r160_base_bf16": smoke_r160_base_bf16,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--codec", choices=("all", *accepted_codec_names()), default="all"
    )
    args = parser.parse_args()
    selected = CODECS if args.codec == "all" else (resolve_codec(args.codec),)
    for spec in selected:
        codec = KvfoldCodec(spec.name)
        SMOKE_CASES[spec.name](codec)
        print(f"  source={codec.source_id}")


if __name__ == "__main__":
    main()
