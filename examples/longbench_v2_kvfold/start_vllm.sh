#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${1:-${KVFOLD_SERVER_CONFIG:-${SCRIPT_DIR}/qwen3_32b_longbench.properties}}"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: server config not found: ${CONFIG_FILE}" >&2
    exit 1
fi
CONFIG_FILE="$(cd "$(dirname "${CONFIG_FILE}")" && pwd)/$(basename "${CONFIG_FILE}")"

# The properties file is part of this trusted experiment directory and uses
# ordinary Bash assignments/exports.
source "${CONFIG_FILE}"

required_variables=(
    model served_model_name server_host server_port tp_size dp_size pp_size
    max_model_len max_num_batched_tokens max_num_seqs block_size
    gpu_memory_utilization distributed_executor_backend
)
for variable_name in "${required_variables[@]}"; do
    if [[ -z "${!variable_name:-}" ]]; then
        echo "ERROR: required config variable is empty: ${variable_name}" >&2
        exit 1
    fi
done
command -v vllm >/dev/null 2>&1 || {
    echo "ERROR: vllm command not found" >&2
    exit 1
}

if [[ "${enforce_eager:-}" != "true" ||
      "${enable_prefix_caching:-}" != "false" ||
      "${enable_speculative_decoding:-}" != "false" ||
      "${max_num_seqs}" != "1" ||
      "${block_size}" != "128" ||
      $((max_num_batched_tokens % block_size)) -ne 0 ]]; then
    echo "ERROR: this experiment requires eager=true, prefix/speculative=false," >&2
    echo "       max_num_seqs=1, block_size=128, and block-aligned scheduling" >&2
    exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export KVFOLD_SERVER_CONFIG="${CONFIG_FILE}"

enabled="${KVFOLD_ATTN_ENABLE:-0}"
case "${enabled,,}" in
    1|true|yes|on) enabled=true ;;
    0|false|no|off) enabled=false ;;
    *)
        echo "ERROR: KVFOLD_ATTN_ENABLE must be 0 or 1" >&2
        exit 1
        ;;
esac

mode=baseline
if [[ "${enabled}" == "true" ]]; then
    export KVFOLD_ATTN_ENABLE=1
    selected_codec="${KVFOLD_CODEC:-}"
    if ! canonical_codec="$(KVFOLD_ATTN_ENABLE=0 python3 -c \
        'import sys; from codec_catalog import resolve_codec; print(resolve_codec(sys.argv[1]).name)' \
        "${selected_codec}" 2>/dev/null)"; then
        available_codecs="$(KVFOLD_ATTN_ENABLE=0 python3 -c \
            'from codec_catalog import CODECS; print(", ".join(spec.name for spec in CODECS))')"
        echo "ERROR: KVFOLD_CODEC must select one of: ${available_codecs}" >&2
        exit 1
    fi
    export KVFOLD_CODEC="${canonical_codec}"
    if [[ -z "${KVFOLD_RUN_ID:-}" ]]; then
        echo "ERROR: KVFOLD_RUN_ID must be a unique non-empty label" >&2
        exit 1
    fi
    export KVFOLD_CODEC_BRIDGE="${KVFOLD_CODEC_BRIDGE:-${SCRIPT_DIR}/build/libkvfold_longbench_bridge.so}"
    if [[ ! -f "${KVFOLD_CODEC_BRIDGE}" ]]; then
        echo "ERROR: native bridge not found: ${KVFOLD_CODEC_BRIDGE}" >&2
        echo "Run: bash ${SCRIPT_DIR}/build_bridge.sh" >&2
        exit 1
    fi
    export KVFOLD_TOKEN_BLOCK_SIZE="${KVFOLD_TOKEN_BLOCK_SIZE:-128}"
    export KVFOLD_TARGET_ARCHITECTURE="${KVFOLD_TARGET_ARCHITECTURE:-Qwen3ForCausalLM}"
    export KVFOLD_EXPECTED_TP_SIZE="${KVFOLD_EXPECTED_TP_SIZE:-4}"
    export KVFOLD_EXPECTED_LOCAL_KV_HEADS="${KVFOLD_EXPECTED_LOCAL_KV_HEADS:-2}"
    export KVFOLD_EXPECTED_HEAD_SIZE="${KVFOLD_EXPECTED_HEAD_SIZE:-128}"
    export KVFOLD_EXPECTED_LAYER_COUNT="${KVFOLD_EXPECTED_LAYER_COUNT:-64}"
    export KVFOLD_ATTN_STRICT="${KVFOLD_ATTN_STRICT:-1}"
    export KVFOLD_STATS_DIR="${KVFOLD_STATS_DIR:-${SCRIPT_DIR}/results/${KVFOLD_RUN_ID}-worker-stats}"
    if [[ -e "${KVFOLD_STATS_DIR}" ]]; then
        echo "ERROR: stats directory already exists: ${KVFOLD_STATS_DIR}" >&2
        exit 1
    fi
    mkdir -p "$(dirname "${KVFOLD_STATS_DIR}")"
    mkdir "${KVFOLD_STATS_DIR}"
    mode="${KVFOLD_CODEC}"
else
    export KVFOLD_ATTN_ENABLE=0
fi

log_path="${vllm_log_path:-${SCRIPT_DIR}/logs}"
if [[ "${log_path}" != /* ]]; then
    log_path="${SCRIPT_DIR}/${log_path#./}"
fi
mkdir -p "${log_path}"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${log_path}/vllm_${mode}_${timestamp}.log"
config_sha256="$(sha256sum "${CONFIG_FILE}" | awk '{print $1}')"

CMD=(
    vllm serve "${model}"
    --served-model-name "${served_model_name}"
    --host "${server_host}"
    --port "${server_port}"
    --tensor-parallel-size "${tp_size}"
    --data-parallel-size "${dp_size}"
    --pipeline-parallel-size "${pp_size}"
    --max-model-len "${max_model_len}"
    --max-num-batched-tokens "${max_num_batched_tokens}"
    --max-num-seqs "${max_num_seqs}"
    --block-size "${block_size}"
    --gpu-memory-utilization "${gpu_memory_utilization}"
    --distributed-executor-backend "${distributed_executor_backend}"
    --seed "${seed:-2026}"
    --trust-remote-code
    --enforce-eager
    --no-enable-prefix-caching
)

if [[ -n "${quantization:-}" && "${quantization}" != "NONE" ]]; then
    CMD+=(--quantization "${quantization}")
fi
if [[ "${enable_expert_parallel:-false}" == "true" ]]; then
    CMD+=(--enable-expert-parallel)
fi
if [[ "${async_scheduling:-false}" == "true" ]]; then
    CMD+=(--async-scheduling)
fi
if [[ "${enable_rope_scaling:-false}" == "true" ]]; then
    for variable_name in rope_theta rope_type factor original_max_position_embeddings; do
        if [[ -z "${!variable_name:-}" ]]; then
            echo "ERROR: required YaRN config variable is empty: ${variable_name}" >&2
            exit 1
        fi
    done
    hf_overrides='{"rope_parameters":{"rope_theta":'${rope_theta}',"rope_type":"'${rope_type}'","factor":'${factor}',"original_max_position_embeddings":'${original_max_position_embeddings}'},"max_model_len":'${max_model_len}'}'
    CMD+=(--hf-overrides "${hf_overrides}")
fi

additional_config='{}'
if [[ -n "${enable_ascend_scheduler:-}" || -n "${enable_torchair_graph:-}" ]]; then
    additional_config='{"ascend_scheduler_config":{"enabled":'${enable_ascend_scheduler:-false}'},"torchair_graph_config":{"enabled":'${enable_torchair_graph:-false}'}}'
    CMD+=(--additional-config "${additional_config}")
fi

{
    echo "===== KVfold LongBench server ====="
    echo "mode              : ${mode}"
    echo "run_id            : ${KVFOLD_RUN_ID:-baseline}"
    echo "config            : ${CONFIG_FILE}"
    echo "config_sha256     : ${config_sha256}"
    echo "bridge            : ${KVFOLD_CODEC_BRIDGE:-disabled}"
    echo "stats_dir         : ${KVFOLD_STATS_DIR:-disabled}"
    echo "log               : ${log_file}"
    echo "command           : ${CMD[*]}"
    echo "==================================="
    "${CMD[@]}"
} 2>&1 | tee "${log_file}"
