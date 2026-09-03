# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

"""Automatically install the KVfold attention hook in every vLLM worker."""

import os

if os.environ.get("KVFOLD_ATTN_ENABLE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    from kvfold_attention_patch import install

    install()
