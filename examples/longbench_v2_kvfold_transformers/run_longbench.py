#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

"""Run LongBench V2 directly with Transformers and an optional KVfold hook."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codec_catalog import CODECS, resolve_codec
from longbench_common import (
    DATASET_NAME,
    DATASET_REVISION,
    PROMPT_TEMPLATE,
    extract_answer,
    file_sha256,
    load_dataset_json,
    load_ids,
    make_prompt,
    read_result_rows,
    select_rows,
    text_sha256,
)
from transformers_cache_codec import KvfoldCacheProcessor, patch_dynamic_cache
from transformers_model import (
    TransformersQwen3Model,
    add_transformers_model_arguments,
    apply_chat_template_non_thinking,
    options_from_namespace,
)


def parse_codec(value: str) -> str:
    if value.strip().lower() in {"none", "baseline"}:
        return "none"
    try:
        return resolve_codec(value).name
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pure Transformers LongBench V2 accuracy runner; no vLLM, HTTP "
            "service, UCM Store, or NPU performance measurement is involved."
        )
    )
    add_transformers_model_arguments(parser)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--codec",
        type=parse_codec,
        choices=("none", *(spec.name for spec in CODECS)),
        default="none",
        help="Use none for Baseline, or select one registered codec",
    )
    parser.add_argument("--codec-library", type=Path)
    parser.add_argument(
        "--codec-token-blocks-per-batch",
        type=int,
        default=32,
        help="Maximum 128-token groups copied to CPU per codec call",
    )
    parser.add_argument(
        "--subset", choices=("full", "smoke", "stratified"), default="full"
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--difficulty", action="append", default=[])
    parser.add_argument("--length", action="append", default=[])
    parser.add_argument("--max-input-tokens", type=int, default=120_000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-generation-failures", action="store_true")
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=1,
        help="Call gc.collect/torch.npu.empty_cache every N completed samples; 0 disables",
    )
    return parser.parse_args()


def _model_file_hash(model_path: Path, name: str) -> str | None:
    candidate = model_path / name
    return file_sha256(candidate) if candidate.is_file() else None


def _jsonable_device_map(device_map: Any) -> dict[str, str] | None:
    if not isinstance(device_map, dict):
        try:
            device_map = dict(device_map)
        except (TypeError, ValueError):
            return None
    return {str(key): str(value) for key, value in sorted(device_map.items())}


def _token_ids_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _render_token_ids(tokenizer: Any, prompt: str) -> tuple[str, list[int]]:
    rendered = apply_chat_template_non_thinking(tokenizer, prompt)
    encoded = tokenizer(rendered, add_special_tokens=False)
    token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise RuntimeError("offline evaluator only supports one prompt at a time")
        token_ids = token_ids[0]
    return rendered, [int(value) for value in token_ids]


def truncate_for_chat_template(
    tokenizer: Any, prompt: str, max_input_tokens: int
) -> tuple[str, int, list[int], bool]:
    """Head/tail truncate the user prompt while preserving the chat wrapper."""

    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")
    raw_ids = [
        int(value) for value in tokenizer.encode(prompt, add_special_tokens=False)
    ]
    _, full_ids = _render_token_ids(tokenizer, prompt)
    if len(full_ids) <= max_input_tokens:
        return prompt, len(raw_ids), full_ids, False

    budget = min(len(raw_ids), max_input_tokens)
    while budget > 0:
        left = budget // 2
        right = budget - left
        selected_ids = raw_ids[:left]
        if right:
            selected_ids += raw_ids[-right:]
        candidate = tokenizer.decode(selected_ids, skip_special_tokens=True)
        _, final_ids = _render_token_ids(tokenizer, candidate)
        if len(final_ids) <= max_input_tokens:
            return candidate, len(raw_ids), final_ids, True
        budget -= max(1, len(final_ids) - max_input_tokens)
    raise RuntimeError("chat template overhead exceeds max_input_tokens")


def _counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    fields = tuple(after.get("counters", {}))
    counters = {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in fields}
    before_layers = before.get("layer_blocks", {})
    after_layers = after.get("layer_blocks", {})
    layers = {
        str(layer): int(after_layers.get(layer, 0)) - int(before_layers.get(layer, 0))
        for layer in set(before_layers) | set(after_layers)
    }
    layers = {layer: value for layer, value in sorted(layers.items()) if value}
    return {
        **counters,
        "layer_count": len(layers),
        "layer_blocks": layers,
        "mode_counts": {
            str(after["primary_mode"]): counters.get("primary_blocks", 0),
            str(after["fallback_mode"]): counters.get("fallback_blocks", 0),
        },
    }


def _validate_sample_codec_stats(stats: dict[str, Any], input_tokens: int) -> None:
    expected_layers = 64
    token_blocks_per_layer = input_tokens // KvfoldCacheProcessor.BLOCK_SIZE
    codec_blocks_per_layer = (
        token_blocks_per_layer * KvfoldCacheProcessor.SIMULATED_TP_SIZE
    )
    expected_blocks = expected_layers * codec_blocks_per_layer
    layer_blocks = stats.get("layer_blocks", {})
    expected_layer_names = {str(layer) for layer in range(expected_layers)}
    if (
        token_blocks_per_layer <= 0
        or stats.get("layer_count") != expected_layers
        or int(stats.get("blocks", 0)) != expected_blocks
        or len(layer_blocks) != expected_layers
        or set(layer_blocks) != expected_layer_names
        or any(int(value) != codec_blocks_per_layer for value in layer_blocks.values())
        or int(stats.get("failures", 0)) != 0
    ):
        raise RuntimeError(
            "KVfold hook did not cover every complete block in every Qwen3 layer: "
            f"input_tokens={input_tokens}, expected_layers={expected_layers}, "
            f"expected_blocks={expected_blocks}, got_layers={stats.get('layer_count')}, "
            f"got_blocks={stats.get('blocks')}"
        )


def _aggregate_codec_rows(
    rows: list[dict[str, Any]],
    processor: KvfoldCacheProcessor,
    run_id: str,
    final: bool,
) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    layer_blocks: defaultdict[str, int] = defaultdict(int)
    samples_with_codec = 0
    invalid_samples = 0
    identity = processor.snapshot()
    counter_fields = tuple(identity["counters"])
    for row in rows:
        sample = row.get("codec_stats")
        if not isinstance(sample, dict):
            invalid_samples += 1
            continue
        try:
            _validate_sample_codec_stats(sample, int(row["input_prompt_tokens"]))
        except (KeyError, TypeError, ValueError, RuntimeError):
            invalid_samples += 1
            continue
        samples_with_codec += 1
        for field in counter_fields:
            totals[field] += int(sample.get(field, 0))
        for layer, count in sample.get("layer_blocks", {}).items():
            layer_blocks[str(layer)] += int(count)

    if final and (invalid_samples or samples_with_codec != len(rows)):
        raise RuntimeError(
            "cannot finalize codec stats because result rows are missing valid "
            f"per-sample coverage: valid={samples_with_codec}, invalid={invalid_samples}, "
            f"rows={len(rows)}"
        )
    counters = {field: int(totals[field]) for field in counter_fields}
    return {
        "stats_schema_version": 1,
        "final": final,
        "run_id": run_id,
        "codec": identity["codec"],
        "codec_id": identity["codec_id"],
        "codec_source_id": identity["codec_source_id"],
        "primary_mode": identity["primary_mode"],
        "fallback_mode": identity["fallback_mode"],
        "mode_counts": {
            identity["primary_mode"]: counters["primary_blocks"],
            identity["fallback_mode"]: counters["fallback_blocks"],
        },
        "block_size": identity["block_size"],
        "simulated_tp_size": identity["simulated_tp_size"],
        "global_kv_heads": identity["global_kv_heads"],
        "local_kv_heads": identity["local_kv_heads"],
        "head_dim": identity["head_dim"],
        "values_per_codec_block": identity["values_per_codec_block"],
        "record_bytes_per_block": identity["record_bytes_per_block"],
        "token_blocks_per_batch": identity["token_blocks_per_batch"],
        "samples": len(rows),
        "samples_with_codec": samples_with_codec,
        "invalid_samples": invalid_samples,
        "layer_count": len(layer_blocks),
        "layer_blocks": dict(
            sorted(layer_blocks.items(), key=lambda item: int(item[0]))
        ),
        "counters": counters,
        **counters,
    }


def _metadata(
    args: argparse.Namespace,
    backend: TransformersQwen3Model,
    selected: list[dict[str, Any]],
    processor: KvfoldCacheProcessor | None,
) -> dict[str, Any]:
    try:
        import torch_npu

        torch_npu_version = getattr(torch_npu, "__version__", "unknown")
    except ImportError:
        torch_npu_version = None
    model_path = Path(args.model).resolve()
    options = backend.options
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner": "pure-transformers-dynamic-cache-v1",
        "run_id": args.run_id,
        "codec": args.codec,
        "codec_id": processor.codec.codec_id if processor else None,
        "codec_source_id": processor.codec.source_id if processor else None,
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "dataset_json": str(args.dataset_json.resolve()),
        "dataset_json_sha256": file_sha256(args.dataset_json),
        "prompt_template_sha256": text_sha256(PROMPT_TEMPLATE),
        "selected_ids": [str(row["_id"]) for row in selected],
        "model": str(model_path),
        "model_config_sha256": _model_file_hash(model_path, "config.json"),
        "tokenizer_config_sha256": _model_file_hash(
            model_path, "tokenizer_config.json"
        ),
        "chat_template_sha256": text_sha256(
            str(getattr(backend.tokenizer, "chat_template", ""))
        ),
        "transformers_version": backend.transformers.__version__,
        "torch_version": backend.torch.__version__,
        "torch_npu_version": torch_npu_version,
        "dtype": "bfloat16",
        "device_map": str(args.device_map),
        "max_memory": {
            str(key): str(value) for key, value in (options.max_memory or {}).items()
        },
        "hf_device_map": _jsonable_device_map(
            getattr(backend.model, "hf_device_map", None)
        ),
        "attention_implementation": "sdpa",
        "rope": {
            "rope_type": "yarn",
            "factor": options.yarn_factor,
            "original_max_position_embeddings": options.original_max_position_embeddings,
            "rope_theta": options.rope_theta,
        },
        "max_position_embeddings": options.max_position_embeddings,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "codec_token_blocks_per_batch": args.codec_token_blocks_per_batch,
        "subset": args.subset,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "enable_thinking": False,
        "block_size": KvfoldCacheProcessor.BLOCK_SIZE,
        "simulated_tp_size": KvfoldCacheProcessor.SIMULATED_TP_SIZE,
        "expected_layers": 64,
        "expected_kv_heads": KvfoldCacheProcessor.EXPECTED_KV_HEADS,
        "expected_head_dim": KvfoldCacheProcessor.EXPECTED_HEAD_DIM,
    }


def _validate_or_write_metadata(
    path: Path, metadata: dict[str, Any], resume: bool
) -> None:
    if resume and path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError(f"run metadata is not a JSON object: {path}")
        keys = set(metadata) - {"created_at"}
        mismatches = [
            key for key in sorted(keys) if existing.get(key) != metadata.get(key)
        ]
        if mismatches:
            raise ValueError(
                "refusing to resume a different Transformers run; metadata "
                f"differs in {mismatches}"
            )
        return
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _cleanup_device_cache(backend: TransformersQwen3Model) -> None:
    gc.collect()
    npu = getattr(backend.torch, "npu", None)
    if npu is not None and hasattr(npu, "empty_cache"):
        npu.empty_cache()


def _rewrite_result_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".resume.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
    temporary.replace(path)


def _has_unterminated_tail(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as stream:
        stream.seek(-1, 2)
        return stream.read(1) != b"\n"


def _partition_resume_rows(
    rows: list[dict[str, Any]], candidate: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reusable = []
    retry = []
    for row in rows:
        if row.get("error"):
            retry.append(row)
            continue
        if candidate:
            sample = row.get("codec_stats")
            try:
                if not isinstance(sample, dict):
                    raise ValueError("missing codec_stats")
                _validate_sample_codec_stats(sample, int(row["input_prompt_tokens"]))
            except (KeyError, TypeError, ValueError, RuntimeError):
                retry.append(row)
                continue
        reusable.append(row)
    return reusable, retry


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0 or args.max_input_tokens <= 0:
        raise ValueError("token limits must be positive")
    if args.max_input_tokens + args.max_new_tokens > args.max_position_embeddings:
        raise ValueError(
            "max input + max output exceeds max_position_embeddings: "
            f"{args.max_input_tokens}+{args.max_new_tokens}>{args.max_position_embeddings}"
        )
    if args.empty_cache_every < 0:
        raise ValueError("empty-cache-every cannot be negative")
    if not args.run_id.strip():
        raise ValueError("run-id cannot be empty")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    stats_file = args.output.with_suffix(args.output.suffix + ".codec-stats.json")
    if not args.resume:
        existing_artifacts = [
            path for path in (args.output, meta_path, stats_file) if path.exists()
        ]
        if existing_artifacts:
            raise FileExistsError(
                "result artifact already exists; use --resume or a new output: "
                + ", ".join(str(path) for path in existing_artifacts)
            )
    elif args.output.exists() and not meta_path.is_file():
        raise FileNotFoundError(
            "cannot resume an existing result without its metadata sidecar: "
            f"{meta_path}"
        )
    all_rows = load_dataset_json(args.dataset_json)
    selected = select_rows(
        all_rows,
        subset=args.subset,
        seed=args.seed,
        max_samples=args.max_samples,
        ids=load_ids(args.ids_file),
        domains=tuple(args.domain),
        difficulties=tuple(args.difficulty),
        lengths=tuple(args.length),
    )
    if not selected:
        raise ValueError("no LongBench V2 samples matched the selection")
    unfiltered_full = (
        args.subset == "full"
        and args.max_samples == 0
        and args.ids_file is None
        and not args.domain
        and not args.difficulty
        and not args.length
    )
    if unfiltered_full and len(selected) != 503:
        raise ValueError(
            f"full LongBench V2 must contain 503 rows, got {len(selected)}"
        )

    normalize_result_tail = args.resume and _has_unterminated_tail(args.output)
    existing = (
        read_result_rows(args.output, allow_truncated_final_line=True)
        if args.resume
        else []
    )
    completed_ids = {str(row["_id"]) for row in existing}
    selected_ids = {str(row["_id"]) for row in selected}
    if not completed_ids.issubset(selected_ids):
        raise ValueError("existing output contains IDs outside the selected dataset")

    random.seed(args.seed)
    backend = TransformersQwen3Model(options_from_namespace(args))
    backend.transformers.set_seed(args.seed)
    processor = None
    if args.codec != "none":
        processor = KvfoldCacheProcessor(
            args.codec,
            library=args.codec_library,
            token_blocks_per_batch=args.codec_token_blocks_per_batch,
        )

    metadata = _metadata(args, backend, selected, processor)
    _validate_or_write_metadata(meta_path, metadata, args.resume)
    if normalize_result_tail:
        print(
            "Resume: normalizing an unterminated final JSONL record",
            flush=True,
        )
        _rewrite_result_rows(args.output, existing)
    existing, retry_rows = _partition_resume_rows(existing, processor is not None)
    if retry_rows:
        print(
            f"Resume: removing {len(retry_rows)} failed or incomplete rows so they "
            "will be retried",
            flush=True,
        )
        _rewrite_result_rows(args.output, existing)
    completed_ids = {str(row["_id"]) for row in existing}
    pending = [row for row in selected if str(row["_id"]) not in completed_ids]
    print(
        f"LongBench V2 Transformers: codec={args.codec}, selected={len(selected)}, "
        f"completed={len(existing)}, pending={len(pending)}, output={args.output}"
    )

    completed_run = False
    patch_context = patch_dynamic_cache(processor) if processor else nullcontext()
    try:
        with patch_context, args.output.open("a", encoding="utf-8") as output:
            for ordinal, item in enumerate(pending, 1):
                prompt, original_tokens, input_ids, truncated = (
                    truncate_for_chat_template(
                        backend.tokenizer, make_prompt(item), args.max_input_tokens
                    )
                )
                before = processor.snapshot() if processor else None
                started = time.perf_counter()
                response = ""
                error = None
                generated_tokens = 0
                try:
                    result = backend.generate_greedy(
                        prompt, max_new_tokens=args.max_new_tokens
                    )
                    if result.input_tokens != len(input_ids):
                        raise RuntimeError(
                            "tokenization changed between validation and generation: "
                            f"expected={len(input_ids)}, got={result.input_tokens}"
                        )
                    response = result.text
                    generated_tokens = result.output_tokens
                except KeyboardInterrupt:
                    raise
                except Exception as exception:  # Keep failure in the denominator.
                    error = f"{type(exception).__name__}: {exception}"
                duration = time.perf_counter() - started
                after = processor.snapshot() if processor else None
                sample_stats = (
                    _counter_delta(before, after) if before and after else None
                )
                if processor is not None and error is None:
                    try:
                        _validate_sample_codec_stats(sample_stats, len(input_ids))
                    except Exception as exception:
                        error = f"{type(exception).__name__}: {exception}"
                prediction = extract_answer(response)
                row = {
                    "_id": str(item["_id"]),
                    "domain": item["domain"],
                    "sub_domain": item["sub_domain"],
                    "difficulty": item["difficulty"],
                    "length": item["length"],
                    "question": item["question"],
                    "choice_A": item["choice_A"],
                    "choice_B": item["choice_B"],
                    "choice_C": item["choice_C"],
                    "choice_D": item["choice_D"],
                    "answer": item["answer"],
                    "response": response,
                    "pred": prediction,
                    "judge": prediction == item["answer"],
                    "error": error,
                    "duration_s": round(duration, 6),
                    "original_prompt_tokens": original_tokens,
                    "input_prompt_tokens": len(input_ids),
                    "output_tokens": generated_tokens,
                    "truncated": truncated,
                    "prompt_sha256": text_sha256(prompt),
                    "input_ids_sha256": _token_ids_sha256(input_ids),
                    "codec_stats": sample_stats,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                status = "OK" if error is None else error
                print(
                    f"[{len(existing) + ordinal}/{len(selected)}] id={item['_id']} "
                    f"tokens={len(input_ids)} pred={prediction} answer={item['answer']} "
                    f"duration={duration:.2f}s status={status}",
                    flush=True,
                )
                if error and not args.allow_generation_failures:
                    raise RuntimeError(
                        "generation failed; fix the environment and resume with the same command"
                    )
                if args.empty_cache_every and ordinal % args.empty_cache_every == 0:
                    _cleanup_device_cache(backend)
        completed_run = True
    finally:
        if processor is not None:
            current_rows = read_result_rows(args.output)
            can_finalize = (
                completed_run
                and len(current_rows) == len(selected)
                and (
                    args.allow_generation_failures
                    or not any(row.get("error") for row in current_rows)
                )
            )
            stats = _aggregate_codec_rows(
                current_rows,
                processor,
                args.run_id,
                can_finalize,
            )
            stats_file.write_text(
                json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    rows = read_result_rows(args.output)
    correct = sum(row.get("pred") == row.get("answer") for row in rows)
    parse_failures = sum(row.get("pred") is None for row in rows)
    generation_failures = sum(bool(row.get("error")) for row in rows)
    print(
        f"Done: {correct}/{len(rows)} correct ({correct / len(rows) * 100:.2f}%), "
        f"parse_failures={parse_failures}, generation_failures={generation_failures}"
    )


if __name__ == "__main__":
    main()
