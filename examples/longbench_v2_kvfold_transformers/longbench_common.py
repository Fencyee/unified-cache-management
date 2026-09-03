#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

"""Shared, dependency-light helpers for the Transformers LongBench runner."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

DATASET_NAME = "THUDM/LongBench-v2"
DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_dataset_json(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"LongBench V2 data file not found: {path}")
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    else:
        with path.open(encoding="utf-8") as stream:
            rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError(f"dataset JSON must contain a list, got {type(rows).__name__}")
    required = {
        "_id",
        "domain",
        "sub_domain",
        "difficulty",
        "length",
        "question",
        "choice_A",
        "choice_B",
        "choice_C",
        "choice_D",
        "answer",
        "context",
    }
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"dataset row {index} is not an object")
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"dataset row {index} is missing fields: {missing}")
        item_id = str(row["_id"])
        if not item_id or item_id in seen_ids:
            raise ValueError(f"dataset row {index} has an empty or duplicate _id")
        seen_ids.add(item_id)
        if row["answer"] not in {"A", "B", "C", "D"}:
            raise ValueError(f"dataset row {index} has an invalid answer")
    return rows


def _select_one_per_group(
    rows: list[dict[str, Any]], fields: tuple[str, ...], seed: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[tuple(str(row[field]) for field in fields)].append((index, row))
    rng = random.Random(seed)
    selected = [rng.choice(groups[key]) for key in sorted(groups)]
    return [row for _, row in sorted(selected, key=lambda pair: pair[0])]


def select_rows(
    rows: list[dict[str, Any]],
    *,
    subset: str,
    seed: int,
    max_samples: int = 0,
    ids: set[str] | None = None,
    domains: tuple[str, ...] = (),
    difficulties: tuple[str, ...] = (),
    lengths: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if ids is not None and str(row["_id"]) not in ids:
            continue
        if domains and str(row["domain"]) not in domains:
            continue
        if difficulties and str(row["difficulty"]) not in difficulties:
            continue
        if lengths and str(row["length"]) not in lengths:
            continue
        selected.append(row)

    if subset == "smoke":
        selected = _select_one_per_group(selected, ("domain", "difficulty"), seed)
    elif subset == "stratified":
        selected = _select_one_per_group(
            selected, ("domain", "difficulty", "length"), seed
        )
    elif subset != "full":
        raise ValueError(f"unknown subset: {subset}")
    if max_samples > 0:
        selected = selected[:max_samples]
    return selected


def make_prompt(row: dict[str, Any]) -> str:
    replacements = {
        "$DOC$": str(row["context"]).strip(),
        "$Q$": str(row["question"]).strip(),
        "$C_A$": str(row["choice_A"]).strip(),
        "$C_B$": str(row["choice_B"]).strip(),
        "$C_C$": str(row["choice_C"]).strip(),
        "$C_D$": str(row["choice_D"]).strip(),
    }
    prompt = PROMPT_TEMPLATE
    for marker, value in replacements.items():
        prompt = prompt.replace(marker, value)
    return prompt


def extract_answer(response: str | None) -> str | None:
    if not response:
        return None
    normalized = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", normalized)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", normalized)
    return match.group(1) if match else None


def read_result_rows(
    path: Path, *, allow_truncated_final_line: bool = False
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    seen: set[str] = set()
    lines = path.read_bytes().splitlines(keepends=True)
    for line_number, raw_line in enumerate(lines, 1):
        if not raw_line.strip():
            continue
        is_unterminated_tail = line_number == len(lines) and not raw_line.endswith(
            (b"\n", b"\r")
        )
        try:
            line = raw_line.decode("utf-8")
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            if allow_truncated_final_line and is_unterminated_tail:
                break
            raise
        item_id = str(row["_id"])
        if item_id in seen:
            raise ValueError(f"duplicate _id={item_id} in {path}:{line_number}")
        seen.add(item_id)
        rows.append(row)
    return rows


def load_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
