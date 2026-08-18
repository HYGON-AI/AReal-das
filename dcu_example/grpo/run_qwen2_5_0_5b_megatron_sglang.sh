#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export AREAL_ENV_PROFILE="${AREAL_ENV_PROFILE:-qwen}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

# ============================================================================
# Qwen2.5-0.5B-Instruct / AReaL v1.0.4 GRPO smoke configuration
#
# Upstream AReaL v1.0.4 facts used here:
#   - examples/math/gsm8k_grpo.yaml uses Qwen2.5-Instruct as the GSM8K baseline.
#   - gconfig.n_samples defaults to 4 and gconfig.max_new_tokens to 1024 there.
#   - MegatronEngine bridge_type defaults to mbridge; we set it explicitly.
#   - SGLangConfig fields used below all exist in v1.0.4.
#
# Qwen2.5-0.5B has model_type=qwen2, 14 attention heads and 2 KV heads.
# TP2 is therefore a clean first parallel setting (14/2=7, 2/2=1).
# This script intentionally does NOT copy local-only SGLang keys such as
# sglang.page_size, because page_size is not part of upstream v1.0.4 SGLangConfig.
# ============================================================================

MODEL_PATH="${MODEL_PATH:-/workspace/Qwen2.5-0.5B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"

N_NODES="${N_NODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# 2 GPUs for Megatron actor + 2 GPUs for SGLang rollout.
# The remaining 4 GPUs stay idle in this correctness-first smoke configuration.
ACTOR_BACKEND="${ACTOR_BACKEND:-megatron:d1p1t2}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-sglang:d1p1t2}"
WEIGHT_UPDATE_MODE="${WEIGHT_UPDATE_MODE:-xccl}"
MEGATRON_BRIDGE_TYPE="${MEGATRON_BRIDGE_TYPE:-mbridge}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-gsm8k-qwen2-5-0-5b-dcu}"
TRIAL_NAME="${TRIAL_NAME:-grpo-megatron-tp2-sglang-tp2}"
TIMESTAMP="${TIMESTAMP:-$(date '+%Y%m%d-%H%M%S')}"
LOG_DIR="${LOG_DIR:-${AREAL_RUNS_ROOT}/${EXPERIMENT_NAME}-${TRIAL_NAME}-${TIMESTAMP}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train.log}"

# Keep the first adaptation run small. Override from the shell when needed.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-1}"
N_SAMPLES="${N_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TOTAL_TRAIN_EPOCHS="${TOTAL_TRAIN_EPOCHS:-1}"
# Empty means follow total_train_epochs. Example smoke run: TOTAL_TRAIN_STEPS=20.
TOTAL_TRAIN_STEPS="${TOTAL_TRAIN_STEPS:-10}"

# DCU-safe SGLang settings. All keys below exist in upstream AReaL v1.0.4.
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.4}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:--1}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-fa3}"
SGLANG_DISABLE_CUSTOM_ALL_REDUCE="${SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-True}"
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-True}"
SGLANG_DISABLE_CUDA_GRAPH_PADDING="${SGLANG_DISABLE_CUDA_GRAPH_PADDING:-True}"
SGLANG_DISABLE_OVERLAP_SCHEDULE="${SGLANG_DISABLE_OVERLAP_SCHEDULE:-True}"
ROLLOUT_SETUP_TIMEOUT="${ROLLOUT_SETUP_TIMEOUT:-900}"

ACTOR_ATTN_IMPL="${ACTOR_ATTN_IMPL:-sdpa}"

CLUSTER_CONFIG=(
  "scheduler.type=ray"
  "experiment_name=${EXPERIMENT_NAME}"
  "trial_name=${TRIAL_NAME}"
  "cluster.n_nodes=${N_NODES}"
  "cluster.n_gpus_per_node=${N_GPUS_PER_NODE}"
  "cluster.fileroot=${FILER_ROOT}"
  "cluster.name_resolve.nfs_record_root=${NAME_RESOLVE_ROOT}"
)

DATA_CONFIG=(
  "++train_dataset.scheduling_spec=null"
  "++valid_dataset.scheduling_spec=null"
  "train_dataset.batch_size=${TRAIN_BATCH_SIZE}"
  "valid_dataset.batch_size=${VALID_BATCH_SIZE}"
)

ACTOR_CONFIG=(
  "actor.backend=${ACTOR_BACKEND}"
  "actor.path=${MODEL_PATH}"
  "actor.weight_update_mode=${WEIGHT_UPDATE_MODE}"
  "++actor.megatron.bridge_type=${MEGATRON_BRIDGE_TYPE}"
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
  "++sglang.disable_radix_cache=True"
  "++sglang.disable_cuda_graph=${SGLANG_DISABLE_CUDA_GRAPH}"
  "++sglang.disable_cuda_graph_padding=${SGLANG_DISABLE_CUDA_GRAPH_PADDING}"
  "++sglang.disable_overlap_schedule=${SGLANG_DISABLE_OVERLAP_SCHEDULE}"
  "+sglang.disable_custom_all_reduce=${SGLANG_DISABLE_CUSTOM_ALL_REDUCE}"
  "++sglang.attention_backend=${SGLANG_ATTENTION_BACKEND}"
)

TRAINER_CONFIG=(
  "total_train_epochs=${TOTAL_TRAIN_EPOCHS}"
)
if [[ -n "${TOTAL_TRAIN_STEPS}" ]]; then
  TRAINER_CONFIG+=("++total_train_steps=${TOTAL_TRAIN_STEPS}")
fi

grpo_prepare_run "${N_NODES}" "$((N_NODES * N_GPUS_PER_NODE))"
grpo_print_summary
grpo_launch \
  "${CLUSTER_CONFIG[@]}" \
  "${DATA_CONFIG[@]}" \
  "${ACTOR_CONFIG[@]}" \
  "${ROLLOUT_CONFIG[@]}" \
  "${SGLANG_CONFIG[@]}" \
  "${TRAINER_CONFIG[@]}"
