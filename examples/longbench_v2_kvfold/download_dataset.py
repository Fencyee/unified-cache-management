#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

"""Download and freeze the 503-row LongBench v2 evaluation set as JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DATASET_NAME = "THUDM/LongBench-v2"
DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/longbench_v2.json"))
    parser.add_argument("--revision", default=DATASET_REVISION)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, split="train", revision=args.revision)
    rows = [dict(item) for item in dataset]
    if len(rows) != 503:
        raise RuntimeError(f"expected 503 LongBench v2 rows, got {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, ensure_ascii=False)
    digest = hashlib.sha256()
    with args.output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    print(
        f"saved={args.output} rows={len(rows)} bytes={args.output.stat().st_size} "
        f"sha256={digest.hexdigest()}"
    )


if __name__ == "__main__":
    main()
