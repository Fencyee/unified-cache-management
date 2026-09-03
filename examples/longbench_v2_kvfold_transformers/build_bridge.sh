#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CXX_BIN="${CXX:-g++}"
OUTPUT_DIR="${SCRIPT_DIR}/build"
OUTPUT_FILE="${OUTPUT_DIR}/libkvfold_longbench_bridge.so"
CODEC_DIR="${SCRIPT_DIR}/native/codec"

for command_name in "${CXX_BIN}" sha256sum awk; do
    command -v "${command_name}" >/dev/null 2>&1 || {
        echo "ERROR: required command not found: ${command_name}" >&2
        exit 1
    }
done

combined_hash() {
    sha256sum "$@" | awk '{print $1}' | sha256sum | awk '{print $1}'
}

R160_BASE_SOURCE_ID="r160-base15:$(combined_hash "${CODEC_DIR}/r160_base_bf16.cc" "${CODEC_DIR}/r160_base_bf16.h" "${CODEC_DIR}/tunstall.h")"
TUNSTALL_R160_SOURCE_ID="r160-tunstall:$(combined_hash "${CODEC_DIR}/tunstall_bf16_r160.cc" "${CODEC_DIR}/tunstall_bf16_r160.h" "${CODEC_DIR}/tunstall.cc" "${CODEC_DIR}/tunstall.h")"
TUNSTALL_R200_SOURCE_ID="r200-tunstall:$(combined_hash "${CODEC_DIR}/tunstall_bf16_r200.cc" "${CODEC_DIR}/tunstall_bf16_r200.h" "${CODEC_DIR}/tunstall.cc" "${CODEC_DIR}/tunstall.h")"

mkdir -p "${OUTPUT_DIR}"

CXX_ARGS=(
    -std=c++17
    -O3
    -DNDEBUG
    -march=native
    -fPIC
    -fvisibility=hidden
    -shared
    -Wall
    -Wextra
    -Wl,-Bsymbolic
    -Wl,-z,defs
    "-DKVFOLD_R160_BASE_SOURCE_ID=\"${R160_BASE_SOURCE_ID}\""
    "-DKVFOLD_TUNSTALL_R160_SOURCE_ID=\"${TUNSTALL_R160_SOURCE_ID}\""
    "-DKVFOLD_TUNSTALL_R200_SOURCE_ID=\"${TUNSTALL_R200_SOURCE_ID}\""
    "-I${CODEC_DIR}"
    "${SCRIPT_DIR}/native/kvfold_codec_bridge.cc"
    "${CODEC_DIR}/r160_base_bf16.cc"
    "${CODEC_DIR}/tunstall_bf16_r160.cc"
    "${CODEC_DIR}/tunstall_bf16_r200.cc"
    "${CODEC_DIR}/tunstall.cc"
    -o
    "${OUTPUT_FILE}"
)
"${CXX_BIN}" "${CXX_ARGS[@]}"

echo "Built ${OUTPUT_FILE}"
echo "R160 Base15 source: ${R160_BASE_SOURCE_ID}"
echo "R160 Tunstall source: ${TUNSTALL_R160_SOURCE_ID}"
echo "R200 Tunstall source: ${TUNSTALL_R200_SOURCE_ID}"
if command -v nm >/dev/null 2>&1; then
    nm -D "${OUTPUT_FILE}" | grep ' kvfold_longbench_' || true
fi
