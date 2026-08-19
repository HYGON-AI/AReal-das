#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export AREAL_ENV_PROFILE="${AREAL_ENV_PROFILE:-qwen}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the local model directory.}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
N_NODES="${N_NODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
ACTOR_BACKEND="${ACTOR_BACKEND:-megatron:d1p1t4}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-sglang:d1p1t4}"
WEIGHT_UPDATE_MODE="${WEIGHT_UPDATE_MODE:-xccl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gsm8k-qwen3-8b-hcu}"
TRIAL_NAME="${TRIAL_NAME:-grpo-megatron-tp4-sglang-tp4-fa3}"
TIMESTAMP="${TIMESTAMP:-$(date '+%Y%m%d-%H%M%S')}"
LOG_DIR="${LOG_DIR:-${AREAL_RUNS_ROOT}/${EXPERIMENT_NAME}-${TRIAL_NAME}-${TIMESTAMP}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train.log}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-1}"
N_SAMPLES="${N_SAMPLES:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TOTAL_TRAIN_EPOCHS="${TOTAL_TRAIN_EPOCHS:-1}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.4}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-65536}"
SGLANG_PAGE_SIZE="${SGLANG_PAGE_SIZE:-64}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-fa3}"
ACTOR_ATTN_IMPL="${ACTOR_ATTN_IMPL:-sdpa}"

SGLANG_DISABLE_CUSTOM_ALL_REDUCE="${SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-True}"
ROLLOUT_SETUP_TIMEOUT="${ROLLOUT_SETUP_TIMEOUT:-900}"

CLUSTER_CONFIG=(
  "scheduler.type=ray" "experiment_name=${EXPERIMENT_NAME}" "trial_name=${TRIAL_NAME}"
  "cluster.n_nodes=${N_NODES}" "cluster.n_gpus_per_node=${N_GPUS_PER_NODE}"
  "cluster.fileroot=${FILER_ROOT}" "cluster.name_resolve.nfs_record_root=${NAME_RESOLVE_ROOT}"
)
DATA_CONFIG=(
  "++train_dataset.scheduling_spec=null" "++valid_dataset.scheduling_spec=null"
  "train_dataset.batch_size=${TRAIN_BATCH_SIZE}" "valid_dataset.batch_size=${VALID_BATCH_SIZE}"
)
ACTOR_CONFIG=(
  "actor.backend=${ACTOR_BACKEND}" "actor.path=${MODEL_PATH}" "actor.weight_update_mode=${WEIGHT_UPDATE_MODE}"
  "++actor.attn_impl=${ACTOR_ATTN_IMPL}"
)
ROLLOUT_CONFIG=(
  "rollout.backend=${ROLLOUT_BACKEND}"
  "+rollout.setup_timeout=${ROLLOUT_SETUP_TIMEOUT}"
  "gconfig.n_samples=${N_SAMPLES}"
  "gconfig.max_new_tokens=${MAX_NEW_TOKENS}"
)
SGLANG_CONFIG=(
  "sglang.model_path=${MODEL_PATH}"
  "tokenizer_path=${TOKENIZER_PATH}"

  "sglang.mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}"

  "++sglang.chunked_prefill_size=${SGLANG_CHUNKED_PREFILL_SIZE}"
  "++sglang.page_size=${SGLANG_PAGE_SIZE}"

  "++sglang.disable_radix_cache=True"
  "++sglang.disable_cuda_graph=True"
  "++sglang.disable_cuda_graph_padding=True"
  "++sglang.disable_overlap_schedule=True"

  "+sglang.disable_custom_all_reduce=${SGLANG_DISABLE_CUSTOM_ALL_REDUCE}"

  "++sglang.attention_backend=${SGLANG_ATTENTION_BACKEND}"
)
TRAINER_CONFIG=("total_train_epochs=${TOTAL_TRAIN_EPOCHS}")

grpo_prepare_run "${N_NODES}" "$((N_NODES * N_GPUS_PER_NODE))"
grpo_print_summary
grpo_launch "${CLUSTER_CONFIG[@]}" "${DATA_CONFIG[@]}" "${ACTOR_CONFIG[@]}" \
  "${ROLLOUT_CONFIG[@]}" "${SGLANG_CONFIG[@]}" "${TRAINER_CONFIG[@]}"
