#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

"""Paired comparison of baseline and KVfold candidate LongBench v2 outputs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from codec_catalog import CODECS, resolve_codec


def parse_codec(value: str) -> str:
    try:
        return resolve_codec(value).name
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--codec",
        type=parse_codec,
        choices=tuple(spec.name for spec in CODECS),
        required=True,
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--candidate-stats-dir", type=Path)
    parser.add_argument("--expected-workers", type=int, default=4)
    parser.add_argument("--expected-layers", type=int, default=64)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-request-failures", action="store_true")
    parser.add_argument("--skip-runtime-check", action="store_true")
    return parser.parse_args()


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


def accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return math.nan
    return sum(row.get("pred") == row.get("answer") for row in rows) / len(rows)


def load_metadata(result: Path) -> dict[str, Any]:
    path = result.with_suffix(result.suffix + ".meta.json")
    if not path.is_file():
        raise FileNotFoundError(f"result metadata is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_metadata(
    baseline_path: Path, candidate_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = load_metadata(baseline_path)
    candidate = load_metadata(candidate_path)
    fields = (
        "dataset",
        "dataset_revision",
        "dataset_json_sha256",
        "prompt_template_sha256",
        "selected_ids",
        "server_config_sha256",
        "served_model_name",
        "tokenizer",
        "subset",
        "max_input_tokens",
        "max_tokens",
        "temperature",
        "seed",
        "enable_thinking",
    )
    mismatches = [
        field for field in fields if baseline.get(field) != candidate.get(field)
    ]
    if mismatches:
        raise ValueError(f"baseline/candidate metadata differs in {mismatches}")
    baseline_run_id = str(baseline.get("run_id", "")).strip()
    candidate_run_id = str(candidate.get("run_id", "")).strip()
    if not baseline_run_id or not candidate_run_id:
        raise ValueError(
            "baseline and candidate metadata must contain non-empty run_id values"
        )
    if baseline_run_id == candidate_run_id:
        raise ValueError("baseline and candidate must use different run_id values")
    return baseline, candidate


def validate_candidate_stats(
    path: Path,
    expected_workers: int,
    expected_run_id: str,
    expected_server_config_sha256: str,
    expected_codec: str,
    expected_layers: int,
) -> dict[str, Any]:
    spec = resolve_codec(expected_codec)
    if expected_workers <= 0:
        raise ValueError("expected candidate worker count must be positive")
    if expected_layers <= 0:
        raise ValueError("expected candidate layer count must be positive")
    if not expected_run_id:
        raise ValueError("candidate result metadata has an empty run_id")
    snapshots = []
    for stats_file in sorted(path.glob("kvfold-attention-*.json")):
        row = json.loads(stats_file.read_text(encoding="utf-8"))
        if int(row.get("blocks", 0)) > 0 or int(row.get("failures", 0)) > 0:
            snapshots.append(row)
    if len(snapshots) != expected_workers:
        raise ValueError(
            f"expected exactly {expected_workers} active candidate worker stats, got {len(snapshots)}"
        )
    if any(row.get("final") is not True for row in snapshots):
        raise ValueError(
            "candidate stats contain an intermediate snapshot; stop every worker "
            "gracefully before running the comparison"
        )
    ranks = {int(row.get("rank", -1)) for row in snapshots}
    expected_ranks = set(range(expected_workers))
    if ranks != expected_ranks:
        raise ValueError(
            f"candidate worker ranks mismatch: expected={expected_ranks}, got={ranks}"
        )
    run_ids = {str(row.get("run_id", "")) for row in snapshots}
    if run_ids != {expected_run_id}:
        raise ValueError(
            f"candidate stats run_id mismatch: expected={expected_run_id}, got={run_ids}"
        )
    config_ids = {str(row.get("server_config_sha256", "")) for row in snapshots}
    if config_ids != {expected_server_config_sha256}:
        raise ValueError(
            "candidate worker/client server config mismatch: "
            f"expected={expected_server_config_sha256}, got={config_ids}"
        )
    codecs = {str(row.get("codec", "")) for row in snapshots}
    if codecs != {spec.name}:
        raise ValueError(
            f"candidate worker codec mismatch: expected={spec.name}, got={codecs}"
        )
    codec_ids = {int(row.get("codec_id", 0)) for row in snapshots}
    if codec_ids != {spec.codec_id}:
        raise ValueError(
            f"candidate worker codec ID mismatch: expected={spec.codec_id}, got={codec_ids}"
        )
    schema_versions = {int(row.get("stats_schema_version", 0)) for row in snapshots}
    if schema_versions != {2}:
        raise ValueError(f"unsupported candidate stats schemas: {schema_versions}")
    failures = sum(int(row.get("failures", 0)) for row in snapshots)
    if failures:
        raise ValueError(
            f"candidate worker stats contain {failures} roundtrip failures"
        )
    if any(int(row.get("blocks", 0)) <= 0 for row in snapshots):
        raise ValueError("a candidate worker did not process any complete cache block")
    source_ids = {str(row.get("codec_source_id", "unknown")) for row in snapshots}
    if len(source_ids) != 1 or source_ids & {"unknown", "not-loaded"}:
        raise ValueError(
            f"candidate workers used inconsistent codec sources: {source_ids}"
        )

    expected_layer_counts = {
        int(row.get("expected_layer_count", 0)) for row in snapshots
    }
    if expected_layer_counts != {expected_layers}:
        raise ValueError(
            "candidate worker expected-layer configuration mismatch: "
            f"expected={expected_layers}, got={expected_layer_counts}"
        )
    layer_maps = []
    for row in snapshots:
        layer_names = row.get("layer_names")
        layer_blocks = row.get("layer_blocks")
        if not isinstance(layer_names, list) or not isinstance(layer_blocks, dict):
            raise ValueError("candidate worker stats do not contain layer coverage")
        normalized = {str(name): int(count) for name, count in layer_blocks.items()}
        if (
            int(row.get("layer_count", 0)) != expected_layers
            or len(layer_names) != expected_layers
            or len(normalized) != expected_layers
            or set(layer_names) != set(normalized)
            or any(count <= 0 for count in normalized.values())
            or sum(normalized.values()) != int(row.get("blocks", 0))
        ):
            raise ValueError(
                "candidate worker did not cover every expected attention layer: "
                f"rank={row.get('rank')}, expected={expected_layers}, "
                f"layer_count={row.get('layer_count')}, blocks={row.get('blocks')}"
            )
        layer_maps.append(normalized)
    first_layer_map = layer_maps[0]
    if any(layer_map != first_layer_map for layer_map in layer_maps[1:]):
        raise ValueError("candidate TP workers have different per-layer block counts")

    workload_fields = (
        "calls",
        "calls_without_full_block",
        "blocks",
        "values",
        "raw_bytes",
        "record_bytes",
        "processed_tokens",
        "skipped_tail_tokens",
    )
    workloads = {
        tuple(int(row.get(field, 0)) for field in workload_fields) for row in snapshots
    }
    if len(workloads) != 1:
        raise ValueError(
            f"candidate TP workers processed different workloads: fields={workload_fields}"
        )
    values_per_block = 2 * 128 * 2 * 128
    for row in snapshots:
        worker_blocks = int(row["blocks"])
        if (
            int(row.get("values", 0)) != worker_blocks * values_per_block
            or int(row.get("raw_bytes", 0)) != worker_blocks * values_per_block * 2
            or int(row.get("record_bytes", 0))
            != worker_blocks * spec.record_bytes_per_block
            or int(row.get("processed_tokens", 0)) != worker_blocks * 128
        ):
            raise ValueError(
                "candidate worker byte/block accounting does not match the "
                f"Qwen3-32B TP4 layout: rank={row.get('rank')}"
            )

    blocks = sum(int(row["blocks"]) for row in snapshots)
    primary_blocks = sum(int(row.get("primary_blocks", 0)) for row in snapshots)
    fallback_blocks = sum(int(row.get("fallback_blocks", 0)) for row in snapshots)
    if primary_blocks + fallback_blocks != blocks:
        raise ValueError(
            "worker mode counts do not cover every processed block: "
            f"blocks={blocks}, {spec.primary_mode}={primary_blocks}, "
            f"{spec.fallback_mode}={fallback_blocks}"
        )
    for row in snapshots:
        expected_modes = {
            spec.primary_mode: int(row.get("primary_blocks", 0)),
            spec.fallback_mode: int(row.get("fallback_blocks", 0)),
        }
        if (
            row.get("primary_mode") != spec.primary_mode
            or row.get("fallback_mode") != spec.fallback_mode
        ):
            raise ValueError("candidate worker mode labels do not match codec catalog")
        if row.get("mode_counts") != expected_modes:
            raise ValueError(
                f"candidate worker mode_counts mismatch: expected={expected_modes}, "
                f"got={row.get('mode_counts')}"
            )
    return {
        "active_workers": len(snapshots),
        "ranks": sorted(ranks),
        "layers": expected_layers,
        "layer_names": sorted(first_layer_map),
        "per_layer_blocks": first_layer_map,
        "blocks": blocks,
        "failures": failures,
        "exceptions": sum(int(row.get("exceptions", 0)) for row in snapshots),
        "m1_groups": sum(int(row.get("m1_groups", 0)) for row in snapshots),
        "m0_groups": sum(int(row.get("m0_groups", 0)) for row in snapshots),
        "primary_mode": spec.primary_mode,
        "fallback_mode": spec.fallback_mode,
        "primary_blocks": primary_blocks,
        "fallback_blocks": fallback_blocks,
        "mode_counts": {
            spec.primary_mode: primary_blocks,
            spec.fallback_mode: fallback_blocks,
        },
        "source_id": next(iter(source_ids)),
        "run_id": expected_run_id,
        "server_config_sha256": expected_server_config_sha256,
        "codec": spec.name,
    }


def mcnemar_exact_pvalue(left_only: int, right_only: int) -> float:
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
        only_baseline = sorted(baseline_ids - candidate_ids)
        only_candidate = sorted(candidate_ids - baseline_ids)
        raise ValueError(
            "result ID sets differ: "
            f"baseline_only={only_baseline[:10]}, candidate_only={only_candidate[:10]}"
        )

    paired = []
    for item_id in baseline:
        left = baseline[item_id]
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
    both_correct = sum(
        left_correct and right_correct for left_correct, right_correct in correctness
    )
    baseline_only = sum(
        left_correct and not right_correct
        for left_correct, right_correct in correctness
    )
    candidate_only = sum(
        not left_correct and right_correct
        for left_correct, right_correct in correctness
    )
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
        groups = defaultdict(list)
        for left, right in paired:
            groups[str(left[field])].append((left, right))
        strata[field] = {}
        for value, values in sorted(groups.items()):
            left_rows = [item[0] for item in values]
            right_rows = [item[1] for item in values]
            base_acc = accuracy(left_rows)
            candidate_acc = accuracy(right_rows)
            strata[field][value] = {
                "count": len(values),
                "baseline_accuracy": base_acc,
                "candidate_accuracy": candidate_acc,
                "delta_percentage_points": (candidate_acc - base_acc) * 100,
            }

    baseline_rows = [left for left, _ in paired]
    candidate_rows = [right for _, right in paired]
    baseline_accuracy = accuracy(baseline_rows)
    candidate_accuracy = accuracy(candidate_rows)
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
        "mcnemar_exact_pvalue": mcnemar_exact_pvalue(baseline_only, candidate_only),
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


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_summary(summary: dict[str, Any]) -> None:
    spec = resolve_codec(str(summary["codec"]))
    label = spec.display_name
    print(f"# LongBench v2 paired accuracy comparison: Baseline vs {label}\n")
    print(f"| Metric | Baseline | {label} | Delta |")
    print("|---|---:|---:|---:|")
    print(
        f"| Overall ({summary['count']} samples) | "
        f"{percent(summary['baseline_accuracy'])} | {percent(summary['candidate_accuracy'])} | "
        f"{summary['delta_percentage_points']:+.2f} pp |"
    )
    print("\n| Paired outcome | Count |")
    print("|---|---:|")
    print(f"| Both correct | {summary['both_correct']} |")
    print(
        f"| Baseline only correct ({label} regressions) | {summary['baseline_only_correct']} |"
    )
    print(f"| {label} only correct | {summary['candidate_only_correct']} |")
    print(f"| Both wrong | {summary['both_wrong']} |")
    print(
        f"| Any prediction change (including parse failure) | {summary['answer_flips']} "
        f"({percent(summary['answer_flip_rate'])}) |"
    )
    print(
        f"| Parsed A-D answer flips | {summary['parsed_answer_flips']} "
        f"({percent(summary['parsed_answer_flip_rate'])}) |"
    )
    print(
        f"| Baseline parse/request failures | {summary['baseline_parse_failures']} / "
        f"{summary['baseline_request_failures']} |"
    )
    print(
        f"| {label} parse/request failures | {summary['candidate_parse_failures']} / "
        f"{summary['candidate_request_failures']} |"
    )
    print(f"| McNemar exact p-value | {summary['mcnemar_exact_pvalue']:.6g} |")
    if "candidate_runtime" in summary:
        runtime = summary["candidate_runtime"]
        print(
            f"| {label} runtime workers / blocks / failures | {runtime['active_workers']} / "
            f"{runtime['blocks']} / {runtime['failures']} |"
        )
        print(
            f"| {label} TP ranks / covered layers | "
            f"{runtime['ranks']} / {runtime['layers']} |"
        )
        print(f"| {label} codec source | `{runtime['source_id']}` |")
        print(
            f"| {label} {runtime['primary_mode']} / {runtime['fallback_mode']} blocks | "
            f"{runtime['primary_blocks']} / {runtime['fallback_blocks']} |"
        )
        if summary["codec"] == "r160_base_bf16":
            print(
                f"| {label} exceptions / M1 groups / M0 groups | "
                f"{runtime['exceptions']} / {runtime['m1_groups']} / "
                f"{runtime['m0_groups']} |"
            )

    for field in ("difficulty", "length", "domain"):
        print(f"\n## By {field}\n")
        print(f"| Group | N | Baseline | {label} | Delta |")
        print("|---|---:|---:|---:|---:|")
        for value, row in summary["strata"][field].items():
            print(
                f"| {value} | {row['count']} | {percent(row['baseline_accuracy'])} | "
                f"{percent(row['candidate_accuracy'])} | {row['delta_percentage_points']:+.2f} pp |"
            )


def main() -> None:
    args = parse_args()
    _, candidate_metadata = validate_metadata(args.baseline, args.candidate)
    summary = paired_summary(load_rows(args.baseline), load_rows(args.candidate))
    summary["codec"] = args.codec
    if not args.allow_partial and summary["count"] != 503:
        raise ValueError(
            f"complete comparison requires 503 samples, got {summary['count']}"
        )
    total_request_failures = (
        summary["baseline_request_failures"] + summary["candidate_request_failures"]
    )
    if total_request_failures and not args.allow_request_failures:
        raise ValueError(
            f"comparison contains {total_request_failures} request failures"
        )
    if not args.skip_runtime_check:
        if args.candidate_stats_dir is None:
            raise ValueError(
                "--candidate-stats-dir is required unless --skip-runtime-check is used"
            )
        summary["candidate_runtime"] = validate_candidate_stats(
            args.candidate_stats_dir,
            args.expected_workers,
            str(candidate_metadata.get("run_id", "")),
            str(candidate_metadata.get("server_config_sha256", "")),
            args.codec,
            args.expected_layers,
        )
    print_summary(summary)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
