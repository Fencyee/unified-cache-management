#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

"""Run the official LongBench v2 zero-shot prompt against a local vLLM API."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DATASET_NAME = "THUDM/LongBench-v2"
DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
DATASET_SPLIT = "train"
PROMPT_TEMPLATE = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: \"The correct answer is (insert answer here)\"."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential LongBench v2 evaluator for paired KVfold tests."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="token-abc123")
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument(
        "--run-id",
        required=True,
        help="Unique label for this baseline or candidate server run",
    )
    parser.add_argument(
        "--server-config",
        type=Path,
        required=True,
        help="The config.properties used to start this server",
    )
    parser.add_argument(
        "--tokenizer", required=True, help="Local Qwen3 tokenizer/model path"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument(
        "--subset", choices=("full", "smoke", "stratified"), default="full"
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--difficulty", action="append", default=[])
    parser.add_argument("--length", action="append", default=[])
    parser.add_argument("--max-input-tokens", type=int, default=120000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-request-failures", action="store_true")
    return parser.parse_args()


def _load_local(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, list):
        raise ValueError(
            f"dataset JSON must contain a list, got {type(value).__name__}"
        )
    return value


def load_data(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset_json:
        return _load_local(args.dataset_json)
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        revision=args.dataset_revision,
    )
    return [dict(item) for item in dataset]


def _select_one_per_group(
    items: list[dict[str, Any]], fields: tuple[str, ...], seed: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, item in enumerate(items):
        groups[tuple(str(item[field]) for field in fields)].append((index, item))
    rng = random.Random(seed)
    selected = [rng.choice(groups[key]) for key in sorted(groups)]
    return [item for _, item in sorted(selected, key=lambda pair: pair[0])]


def select_data(
    items: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    ids = None
    if args.ids_file:
        ids = {
            line.strip()
            for line in args.ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    filtered = []
    for item in items:
        if ids is not None and str(item["_id"]) not in ids:
            continue
        if args.domain and item["domain"] not in args.domain:
            continue
        if args.difficulty and item["difficulty"] not in args.difficulty:
            continue
        if args.length and item["length"] not in args.length:
            continue
        filtered.append(item)

    if args.subset == "smoke":
        filtered = _select_one_per_group(filtered, ("domain", "difficulty"), args.seed)
    elif args.subset == "stratified":
        filtered = _select_one_per_group(
            filtered, ("domain", "difficulty", "length"), args.seed
        )
    if args.max_samples > 0:
        filtered = filtered[: args.max_samples]
    return filtered


def make_prompt(item: dict[str, Any]) -> str:
    replacements = {
        "$DOC$": item["context"].strip(),
        "$Q$": item["question"].strip(),
        "$C_A$": item["choice_A"].strip(),
        "$C_B$": item["choice_B"].strip(),
        "$C_C$": item["choice_C"].strip(),
        "$C_D$": item["choice_D"].strip(),
    }
    prompt = PROMPT_TEMPLATE
    for marker, value in replacements.items():
        prompt = prompt.replace(marker, value)
    return prompt


def truncate_prompt(prompt: str, tokenizer, max_tokens: int) -> tuple[str, int, int]:
    token_ids = tokenizer.encode(prompt)
    original_tokens = len(token_ids)
    if original_tokens <= max_tokens:
        return prompt, original_tokens, original_tokens
    left = max_tokens // 2
    right = max_tokens - left
    token_ids = token_ids[:left] + token_ids[-right:]
    return (
        tokenizer.decode(token_ids, skip_special_tokens=True),
        original_tokens,
        len(token_ids),
    )


def extract_answer(response: str | None) -> str | None:
    if not response:
        return None
    normalized = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", normalized)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", normalized)
    return match.group(1) if match else None


def query(
    client: Any, prompt: str, args: argparse.Namespace
) -> tuple[str, str | None, int, float]:
    last_error = None
    start = time.perf_counter()
    for attempt in range(1, args.retries + 1):
        try:
            completion = client.chat.completions.create(
                model=args.served_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                seed=args.seed,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            response = completion.choices[0].message.content
            if not response:
                raise RuntimeError("model returned an empty response")
            return response.strip(), None, attempt, time.perf_counter() - start
        except KeyboardInterrupt:
            raise
        except Exception as error:  # Keep every failed sample in the denominator.
            last_error = f"{type(error).__name__}: {error}"
            if attempt < args.retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    return "", last_error, args.retries, time.perf_counter() - start


def read_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        item_id = str(row["_id"])
        if item_id in completed:
            raise ValueError(f"duplicate _id={item_id} in {path}:{line_number}")
        completed.add(item_id)
    return completed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(
    args: argparse.Namespace, selected: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_NAME,
        "dataset_revision": args.dataset_revision,
        "dataset_json": str(args.dataset_json.resolve()) if args.dataset_json else None,
        "dataset_json_sha256": (
            _file_sha256(args.dataset_json) if args.dataset_json else None
        ),
        "prompt_template_sha256": hashlib.sha256(
            PROMPT_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "selected_ids": [str(item["_id"]) for item in selected],
        "base_url": args.base_url,
        "run_id": args.run_id,
        "server_config": str(args.server_config.resolve()),
        "server_config_sha256": _file_sha256(args.server_config),
        "served_model_name": args.served_model_name,
        "tokenizer": args.tokenizer,
        "subset": args.subset,
        "max_input_tokens": args.max_input_tokens,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "enable_thinking": False,
    }
    return metadata


def validate_or_write_metadata(
    args: argparse.Namespace, selected: Iterable[dict[str, Any]]
) -> None:
    metadata = _metadata(args, selected)
    path = args.output.with_suffix(args.output.suffix + ".meta.json")
    if args.resume and args.output.exists():
        if not path.is_file():
            raise FileNotFoundError(f"resume metadata is missing: {path}")
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = set(metadata) - {"created_at"}
        mismatches = [
            key for key in sorted(comparable) if existing.get(key) != metadata.get(key)
        ]
        if mismatches:
            raise ValueError(
                "refusing to mix different configurations with --resume; "
                f"metadata differs in {mismatches}"
            )
        return
    if path.exists():
        raise FileExistsError(f"metadata already exists: {path}")
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()

    from openai import OpenAI
    from tqdm import tqdm
    from transformers import AutoTokenizer

    if args.max_input_tokens <= 0 or args.max_tokens <= 0 or args.retries <= 0:
        raise ValueError("token limits and retries must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(
            f"output already exists: {args.output}; use --resume or a new file"
        )

    items = select_data(load_data(args), args)
    if not items:
        raise ValueError("no LongBench v2 samples matched the selection")
    unfiltered_full_run = (
        args.subset == "full"
        and args.max_samples == 0
        and args.ids_file is None
        and not args.domain
        and not args.difficulty
        and not args.length
    )
    if unfiltered_full_run and len(items) != 503:
        raise ValueError(
            f"full LongBench v2 run must contain 503 samples, got {len(items)}"
        )
    completed = read_completed(args.output) if args.resume else set()
    selected_ids = {str(item["_id"]) for item in items}
    if not completed.issubset(selected_ids):
        unexpected = sorted(completed - selected_ids)
        raise ValueError(f"existing output contains unselected IDs: {unexpected[:10]}")
    validate_or_write_metadata(args, items)
    pending = [item for item in items if str(item["_id"]) not in completed]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.request_timeout,
    )
    print(
        f"LongBench v2: selected={len(items)}, completed={len(completed)}, "
        f"pending={len(pending)}, output={args.output}"
    )

    with args.output.open("a", encoding="utf-8") as output:
        for item in tqdm(pending, desc="LongBench v2"):
            prompt, original_tokens, input_tokens = truncate_prompt(
                make_prompt(item), tokenizer, args.max_input_tokens
            )
            response, error, attempts, duration = query(client, prompt, args)
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
                "attempts": attempts,
                "duration_s": round(duration, 6),
                "original_prompt_tokens": original_tokens,
                "input_prompt_tokens": input_tokens,
                "truncated": input_tokens < original_tokens,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()

    rows = [
        json.loads(line)
        for line in args.output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    correct = sum(row.get("pred") == row.get("answer") for row in rows)
    parse_failures = sum(row.get("pred") is None for row in rows)
    request_failures = sum(bool(row.get("error")) for row in rows)
    print(
        f"Done: {correct}/{len(rows)} correct ({correct / len(rows) * 100:.2f}%), "
        f"parse_failures={parse_failures}, request_failures={request_failures}"
    )
    if request_failures and not args.allow_request_failures:
        raise RuntimeError(
            f"{request_failures} requests failed; use a fresh output after fixing the service"
        )


if __name__ == "__main__":
    main()
