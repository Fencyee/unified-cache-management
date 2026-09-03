#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

"""Compare paired LongBench V2 outputs from the pure Transformers runner."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from codec_catalog import CODECS, resolve_codec

COUNTER_FIELDS = (
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


def parse_codec(value: str) -> str:
    try:
        return resolve_codec(value).name
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and KVfold pure-Transformers LongBench runs."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--codec",
        type=parse_codec,
        choices=tuple(spec.name for spec in CODECS),
        required=True,
    )
    parser.add_argument("--candidate-stats", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-generation-failures", action="store_true")
    return parser.parse_args()


def metadata_path(result: Path) -> Path:
    return result.with_suffix(result.suffix + ".meta.json")


def stats_path(result: Path) -> Path:
    return result.with_suffix(result.suffix + ".codec-stats.json")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        item_id = str(row["_id"])
        if item_id in rows:
            raise ValueError(f"duplicate _id={item_id} in {path}:{line_number}")
        rows[item_id] = row
    return rows


def _accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return math.nan
    return sum(row.get("pred") == row.get("answer") for row in rows) / len(rows)


def _mcnemar_exact_pvalue(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value) for value in range(min(left_only, right_only) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_summary(
    baseline: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    if baseline_ids != candidate_ids:
        raise ValueError(
            "result ID sets differ: "
            f"baseline_only={sorted(baseline_ids - candidate_ids)[:10]}, "
            f"candidate_only={sorted(candidate_ids - baseline_ids)[:10]}"
        )

    paired = []
    for item_id, left in baseline.items():
        right = candidate[item_id]
        for field in ("answer", "domain", "difficulty", "length", "prompt_sha256"):
            if left.get(field) != right.get(field):
                raise ValueError(f"_id={item_id} differs in {field}")
        paired.append((left, right))

    correctness = [
        (
            left.get("pred") == left.get("answer"),
            right.get("pred") == right.get("answer"),
        )
        for left, right in paired
    ]
    both_correct = sum(left and right for left, right in correctness)
    baseline_only = sum(left and not right for left, right in correctness)
    candidate_only = sum(not left and right for left, right in correctness)
    both_wrong = len(paired) - both_correct - baseline_only - candidate_only
    answer_flips = sum(left.get("pred") != right.get("pred") for left, right in paired)
    parsed_answer_flips = sum(
        left.get("pred") in {"A", "B", "C", "D"}
        and right.get("pred") in {"A", "B", "C", "D"}
        and left.get("pred") != right.get("pred")
        for left, right in paired
    )

    strata: dict[str, dict[str, dict[str, float | int]]] = {}
    for field in ("difficulty", "length", "domain"):
        groups: defaultdict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = (
            defaultdict(list)
        )
        for left, right in paired:
            groups[str(left[field])].append((left, right))
        strata[field] = {}
        for value, values in sorted(groups.items()):
            left_accuracy = _accuracy([item[0] for item in values])
            right_accuracy = _accuracy([item[1] for item in values])
            strata[field][value] = {
                "count": len(values),
                "baseline_accuracy": left_accuracy,
                "candidate_accuracy": right_accuracy,
                "delta_percentage_points": (right_accuracy - left_accuracy) * 100,
            }

    baseline_rows = [left for left, _ in paired]
    candidate_rows = [right for _, right in paired]
    baseline_accuracy = _accuracy(baseline_rows)
    candidate_accuracy = _accuracy(candidate_rows)
    return {
        "count": len(paired),
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "delta_percentage_points": (candidate_accuracy - baseline_accuracy) * 100,
        "both_correct": both_correct,
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "both_wrong": both_wrong,
        "answer_flips": answer_flips,
        "answer_flip_rate": answer_flips / len(paired) if paired else math.nan,
        "parsed_answer_flips": parsed_answer_flips,
        "parsed_answer_flip_rate": (
            parsed_answer_flips / len(paired) if paired else math.nan
        ),
        "mcnemar_exact_pvalue": _mcnemar_exact_pvalue(baseline_only, candidate_only),
        "baseline_parse_failures": sum(
            row.get("pred") is None for row in baseline_rows
        ),
        "candidate_parse_failures": sum(
            row.get("pred") is None for row in candidate_rows
        ),
        "baseline_request_failures": sum(
            bool(row.get("error")) for row in baseline_rows
        ),
        "candidate_request_failures": sum(
            bool(row.get("error")) for row in candidate_rows
        ),
        "strata": strata,
    }


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_summary(summary: dict[str, Any]) -> None:
    label = resolve_codec(str(summary["codec"])).display_name
    print(f"# LongBench V2 Transformers: Baseline vs {label}\n")
    print(f"| Metric | Baseline | {label} | Delta |")
    print("|---|---:|---:|---:|")
    print(
        f"| Overall ({summary['count']} samples) | "
        f"{_percent(summary['baseline_accuracy'])} | "
        f"{_percent(summary['candidate_accuracy'])} | "
        f"{summary['delta_percentage_points']:+.2f} pp |"
    )
    print("\n| Paired outcome | Count |")
    print("|---|---:|")
    print(f"| Both correct | {summary['both_correct']} |")
    print(f"| Baseline only correct | {summary['baseline_only_correct']} |")
    print(f"| {label} only correct | {summary['candidate_only_correct']} |")
    print(f"| Both wrong | {summary['both_wrong']} |")
    print(
        f"| Prediction changes | {summary['answer_flips']} "
        f"({_percent(summary['answer_flip_rate'])}) |"
    )
    print(f"| McNemar exact p-value | {summary['mcnemar_exact_pvalue']:.6g} |")

    for field in ("difficulty", "length", "domain"):
        print(f"\n## By {field}\n")
        print(f"| Group | N | Baseline | {label} | Delta |")
        print("|---|---:|---:|---:|---:|")
        for value, item in summary["strata"][field].items():
            print(
                f"| {value} | {item['count']} | "
                f"{_percent(item['baseline_accuracy'])} | "
                f"{_percent(item['candidate_accuracy'])} | "
                f"{item['delta_percentage_points']:+.2f} pp |"
            )


def validate_metadata(
    baseline_result: Path, candidate_result: Path, codec: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = load_json(metadata_path(baseline_result))
    candidate = load_json(metadata_path(candidate_result))
    expected_fields = (
        "runner",
        "dataset",
        "dataset_revision",
        "dataset_json_sha256",
        "prompt_template_sha256",
        "selected_ids",
        "model",
        "model_config_sha256",
        "tokenizer_config_sha256",
        "chat_template_sha256",
        "transformers_version",
        "torch_version",
        "torch_npu_version",
        "dtype",
        "device_map",
        "max_memory",
        "hf_device_map",
        "attention_implementation",
        "rope",
        "max_position_embeddings",
        "max_input_tokens",
        "max_new_tokens",
        "codec_token_blocks_per_batch",
        "subset",
        "seed",
        "enable_thinking",
        "block_size",
        "simulated_tp_size",
        "expected_layers",
        "expected_kv_heads",
        "expected_head_dim",
    )
    mismatches = [
        field
        for field in expected_fields
        if baseline.get(field) != candidate.get(field)
    ]
    if mismatches:
        raise ValueError(
            "baseline/candidate Transformers metadata differs in " f"{mismatches}"
        )
    if baseline.get("runner") != "pure-transformers-dynamic-cache-v1":
        raise ValueError(f"unsupported Transformers runner: {baseline.get('runner')}")
    selected_ids = baseline.get("selected_ids")
    if (
        not isinstance(selected_ids, list)
        or not selected_ids
        or any(not isinstance(item_id, str) or not item_id for item_id in selected_ids)
        or len(set(selected_ids)) != len(selected_ids)
    ):
        raise ValueError("metadata selected_ids must be a non-empty unique string list")
    if baseline.get("codec") != "none":
        raise ValueError("baseline metadata must contain codec=none")
    spec = resolve_codec(codec)
    if candidate.get("codec") != spec.name:
        raise ValueError(
            f"candidate metadata codec mismatch: expected={spec.name}, "
            f"got={candidate.get('codec')}"
        )
    if int(candidate.get("codec_id", 0)) != spec.codec_id:
        raise ValueError("candidate metadata codec ID does not match the catalog")
    if not str(candidate.get("codec_source_id", "")).strip():
        raise ValueError("candidate metadata is missing codec_source_id")
    if (
        baseline.get("codec_id") is not None
        or baseline.get("codec_source_id") is not None
    ):
        raise ValueError("baseline metadata must not identify a codec implementation")
    baseline_run_id = str(baseline.get("run_id", "")).strip()
    candidate_run_id = str(candidate.get("run_id", "")).strip()
    if not baseline_run_id or not candidate_run_id:
        raise ValueError("both runs must have non-empty run_id values")
    if baseline_run_id == candidate_run_id:
        raise ValueError("baseline and candidate must use different run IDs")
    return baseline, candidate


def validate_candidate_stats(
    path: Path,
    metadata: dict[str, Any],
    codec: str,
    candidate_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row = load_json(path)
    spec = resolve_codec(codec)
    if row.get("final") is not True:
        raise ValueError("candidate codec stats are not a final snapshot")
    if int(row.get("stats_schema_version", 0)) != 1:
        raise ValueError(
            f"unsupported Transformers codec stats schema: {row.get('stats_schema_version')}"
        )
    if row.get("run_id") != metadata.get("run_id"):
        raise ValueError("candidate result/stats run_id mismatch")
    if row.get("codec") != spec.name or int(row.get("codec_id", 0)) != spec.codec_id:
        raise ValueError("candidate codec stats identify a different codec")
    if row.get("codec_source_id") != metadata.get("codec_source_id"):
        raise ValueError("candidate result/stats codec source mismatch")
    expected_samples = len(metadata["selected_ids"])
    if (
        int(row.get("samples", -1)) != expected_samples
        or int(row.get("samples_with_codec", -1)) != expected_samples
        or int(row.get("invalid_samples", -1)) != 0
    ):
        raise ValueError(
            "candidate codec stats do not cover every selected sample: "
            f"expected={expected_samples}, samples={row.get('samples')}, "
            f"valid={row.get('samples_with_codec')}, "
            f"invalid={row.get('invalid_samples')}"
        )
    failures = int(row.get("failures", 0))
    blocks = int(row.get("blocks", 0))
    primary = int(row.get("primary_blocks", 0))
    fallback = int(row.get("fallback_blocks", 0))
    if failures:
        raise ValueError(f"candidate codec recorded {failures} failures")
    if blocks <= 0 or primary + fallback != blocks:
        raise ValueError(
            "candidate mode counts do not cover every codec block: "
            f"blocks={blocks}, primary={primary}, fallback={fallback}"
        )
    expected_layers = int(metadata["expected_layers"])
    layer_blocks = row.get("layer_blocks")
    if not isinstance(layer_blocks, dict):
        raise ValueError("candidate codec stats are missing layer_blocks")
    normalized_layers = {str(key): int(value) for key, value in layer_blocks.items()}
    expected_layer_names = {str(layer) for layer in range(expected_layers)}
    if (
        int(row.get("layer_count", 0)) != expected_layers
        or len(normalized_layers) != expected_layers
        or set(normalized_layers) != expected_layer_names
        or any(value <= 0 for value in normalized_layers.values())
        or sum(normalized_layers.values()) != blocks
    ):
        raise ValueError(
            "candidate did not exercise every expected Qwen3 attention layer"
        )
    values_per_block = (
        2
        * int(metadata["block_size"])
        * (int(metadata["expected_kv_heads"]) // int(metadata["simulated_tp_size"]))
        * int(metadata["expected_head_dim"])
    )
    if int(row.get("values", 0)) != blocks * values_per_block:
        raise ValueError("candidate values/block accounting is inconsistent")
    if int(row.get("raw_bytes", 0)) != blocks * values_per_block * 2:
        raise ValueError("candidate raw-byte accounting is inconsistent")
    expected_record = spec.record_bytes_per_block
    if (
        int(row.get("record_bytes_per_block", 0)) != expected_record
        or int(row.get("record_bytes", 0)) != blocks * expected_record
    ):
        raise ValueError("candidate record-byte accounting is inconsistent")
    expected_modes = {
        spec.primary_mode: primary,
        spec.fallback_mode: fallback,
    }
    if (
        row.get("primary_mode") != spec.primary_mode
        or row.get("fallback_mode") != spec.fallback_mode
        or row.get("mode_counts") != expected_modes
    ):
        raise ValueError("candidate codec mode labels/counts are inconsistent")
    expected_identity = {
        "block_size": int(metadata["block_size"]),
        "simulated_tp_size": int(metadata["simulated_tp_size"]),
        "global_kv_heads": int(metadata["expected_kv_heads"]),
        "local_kv_heads": int(metadata["expected_kv_heads"])
        // int(metadata["simulated_tp_size"]),
        "head_dim": int(metadata["expected_head_dim"]),
        "values_per_codec_block": values_per_block,
        "token_blocks_per_batch": int(metadata["codec_token_blocks_per_batch"]),
    }
    mismatches = [
        key
        for key, expected in expected_identity.items()
        if int(row.get(key, -1)) != expected
    ]
    if mismatches:
        raise ValueError(
            f"candidate codec stats layout differs from metadata in {mismatches}"
        )
    counters = row.get("counters")
    if not isinstance(counters, dict):
        raise ValueError("candidate codec stats are missing counters")
    accounting_fields = (
        "failures",
        "blocks",
        "values",
        "raw_bytes",
        "record_bytes",
        "primary_blocks",
        "fallback_blocks",
    )
    if any(
        int(counters.get(field, -1)) != int(row.get(field, -2))
        for field in accounting_fields
    ):
        raise ValueError("candidate top-level and nested counters disagree")

    aggregate = {field: 0 for field in COUNTER_FIELDS}
    aggregate_layers = {str(layer): 0 for layer in range(expected_layers)}
    for item_id, result in candidate_rows.items():
        sample = result.get("codec_stats")
        if not isinstance(sample, dict):
            raise ValueError(f"_id={item_id} is missing per-sample codec_stats")
        input_tokens = int(result.get("input_prompt_tokens", -1))
        blocks_per_layer = (input_tokens // int(metadata["block_size"])) * int(
            metadata["simulated_tp_size"]
        )
        sample_blocks = expected_layers * blocks_per_layer
        sample_layers = sample.get("layer_blocks")
        if not isinstance(sample_layers, dict):
            raise ValueError(f"_id={item_id} is missing per-layer codec coverage")
        normalized_sample_layers = {
            str(key): int(value) for key, value in sample_layers.items()
        }
        if (
            input_tokens <= 0
            or blocks_per_layer <= 0
            or int(sample.get("layer_count", 0)) != expected_layers
            or set(normalized_sample_layers) != expected_layer_names
            or any(
                value != blocks_per_layer for value in normalized_sample_layers.values()
            )
            or int(sample.get("blocks", 0)) != sample_blocks
            or int(sample.get("failures", 0)) != 0
        ):
            raise ValueError(
                f"_id={item_id} does not cover every complete block in all layers"
            )
        sample_primary = int(sample.get("primary_blocks", 0))
        sample_fallback = int(sample.get("fallback_blocks", 0))
        if (
            sample_primary + sample_fallback != sample_blocks
            or sample.get("mode_counts")
            != {
                spec.primary_mode: sample_primary,
                spec.fallback_mode: sample_fallback,
            }
            or int(sample.get("values", 0)) != sample_blocks * values_per_block
            or int(sample.get("raw_bytes", 0)) != sample_blocks * values_per_block * 2
            or int(sample.get("record_bytes", 0)) != sample_blocks * expected_record
        ):
            raise ValueError(f"_id={item_id} has inconsistent codec accounting")
        for field in COUNTER_FIELDS:
            aggregate[field] += int(sample.get(field, 0))
        for layer, count in normalized_sample_layers.items():
            aggregate_layers[layer] += count

    if any(
        int(counters.get(field, -1)) != aggregate[field] for field in COUNTER_FIELDS
    ):
        raise ValueError("candidate final counters do not equal per-sample counters")
    if normalized_layers != aggregate_layers:
        raise ValueError("candidate final layer coverage does not equal result rows")
    return row


def main() -> None:
    args = parse_args()
    _, candidate_metadata = validate_metadata(args.baseline, args.candidate, args.codec)
    baseline_rows = load_rows(args.baseline)
    candidate_rows = load_rows(args.candidate)
    selected_ids = {str(item_id) for item_id in candidate_metadata["selected_ids"]}
    if set(baseline_rows) != selected_ids or set(candidate_rows) != selected_ids:
        raise ValueError(
            "result IDs do not exactly match metadata selected_ids: "
            f"expected={len(selected_ids)}, baseline={len(baseline_rows)}, "
            f"candidate={len(candidate_rows)}"
        )
    for item_id in set(baseline_rows) & set(candidate_rows):
        if baseline_rows[item_id].get("input_ids_sha256") != candidate_rows[
            item_id
        ].get("input_ids_sha256"):
            raise ValueError(f"_id={item_id} used different model input token IDs")
        if int(baseline_rows[item_id].get("input_prompt_tokens", -1)) != int(
            candidate_rows[item_id].get("input_prompt_tokens", -2)
        ):
            raise ValueError(f"_id={item_id} used different input token counts")
    summary = paired_summary(baseline_rows, candidate_rows)
    summary["codec"] = args.codec
    if not args.allow_partial and summary["count"] != 503:
        raise ValueError(
            f"complete comparison requires 503 samples, got {summary['count']}"
        )
    failures = (
        summary["baseline_request_failures"] + summary["candidate_request_failures"]
    )
    if failures and not args.allow_generation_failures:
        raise ValueError(f"comparison contains {failures} generation failures")
    runtime = validate_candidate_stats(
        args.candidate_stats or stats_path(args.candidate),
        candidate_metadata,
        args.codec,
        candidate_rows,
    )
    summary["candidate_codec_stats"] = runtime
    print_summary(summary)
    label = resolve_codec(args.codec).display_name
    print("\n## Offline codec coverage\n")
    print("| Codec | Blocks | Primary | Fallback | Source ID |")
    print("|---|---:|---:|---:|---|")
    print(
        f"| {label} | {runtime['blocks']} | {runtime['primary_blocks']} | "
        f"{runtime['fallback_blocks']} | `{runtime['codec_source_id']}` |"
    )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
