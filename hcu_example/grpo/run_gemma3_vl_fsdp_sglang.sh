#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

# Launcher metadata consumed by grpo/run.sh without sourcing this file.
HCU_LAUNCHER_FAMILY=gemma3
HCU_LAUNCHER_VARIANT=vl
HCU_LAUNCHER_ACTOR_BACKEND=fsdp
HCU_LAUNCHER_ROLLOUT_BACKEND=sglang
HCU_LAUNCHER_PROFILE=gemma3

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Use the dedicated Gemma3 correctness profile. run.sh reads the launcher
# metadata before starting Ray, so the same environment is inherited by the
# driver, Ray workers, FSDP actor, and SGLang subprocesses.
export AREAL_ENV_PROFILE="${AREAL_ENV_PROFILE:-gemma3}"
export AREAL_TRAIN_BACKEND="fsdp"

# IMPORTANT: Gemma3-4B is a VLM here. Use Geometry3K GRPO, not GSM8K.
export GRPO_ENTRYPOINT="${GRPO_ENTRYPOINT:-examples/vlm/geometry3k_grpo.py}"
export GRPO_CONFIG="${GRPO_CONFIG:-examples/vlm/geometry3k_grpo.yaml}"

source "${SCRIPT_DIR}/common.sh"

# ============================================================
# Model / cluster
# ============================================================
MODEL_PATH="${MODEL_PATH:-/workspace/gemma3_4b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"

N_NODES="${N_NODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# 8-card single-node correctness baseline:
#   actor   : FSDP DP4       -> GPUs 0-3
#   rollout : 1 replica TP4 -> GPUs 4-7
#
# This intentionally mirrors the standalone start_sglang_gemma3.sh that has
# already been validated on this machine. Do not change TP while debugging
# correctness; TP2 and TP4 are not numerically/kernel-shape equivalent.
ACTOR_BACKEND="${ACTOR_BACKEND:-fsdp:d4p1t1}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-sglang:d1p1t4}"
WEIGHT_UPDATE_MODE="${WEIGHT_UPDATE_MODE:-xccl}"

# ============================================================
# Experiment paths
# ============================================================
EXPERIMENT_NAME="${EXPERIMENT_NAME:-geometry3k-gemma3-4b-fsdp}"
TRIAL_NAME="${TRIAL_NAME:-fsdp4-sglang-tp4-standalone-aligned}"
TIMESTAMP="${TIMESTAMP:-$(date '+%Y%m%d-%H%M%S')}"

AREAL_RUNS_ROOT="${AREAL_RUNS_ROOT:-/workspace/areal_runs}"
FILER_ROOT="${FILER_ROOT:-${AREAL_RUNS_ROOT}/experiments}"
NAME_RESOLVE_ROOT="${NAME_RESOLVE_ROOT:-${AREAL_RUNS_ROOT}/name_resolve}"
LOG_DIR="${LOG_DIR:-${AREAL_RUNS_ROOT}/${EXPERIMENT_NAME}-${TRIAL_NAME}-${TIMESTAMP}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train.log}"

mkdir -p "${LOG_DIR}" "${FILER_ROOT}" "${NAME_RESOLVE_ROOT}"

# ============================================================
# Geometry3K / GRPO
# ============================================================
DATASET_PATH="${DATASET_PATH:-hiyouga/geometry3k}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-4}"
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-4096}"

N_SAMPLES="${N_SAMPLES:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
ROLLOUT_MAX_CONCURRENT="${ROLLOUT_MAX_CONCURRENT:-1}"
ROLLOUT_SETUP_TIMEOUT="${ROLLOUT_SETUP_TIMEOUT:-1200}"

TOTAL_TRAIN_EPOCHS="${TOTAL_TRAIN_EPOCHS:-1}"
TOTAL_TRAIN_STEPS="${TOTAL_TRAIN_STEPS:-2}"

# ============================================================
# FSDP actor
# ============================================================
ACTOR_LR="${ACTOR_LR:-1.0e-6}"
ACTOR_MAX_TOKENS_PER_MB="${ACTOR_MAX_TOKENS_PER_MB:-4096}"

# Gemma3 mixes sliding-window and full attention. For the first correctness
# run use the Transformers SDPA path on the actor side.
ACTOR_ATTN_IMPL="${ACTOR_ATTN_IMPL:-sdpa}"
FSDP_MEMORY_EFFICIENT_LOAD="${FSDP_MEMORY_EFFICIENT_LOAD:-true}"
FSDP_OFFLOAD_PARAMS="${FSDP_OFFLOAD_PARAMS:-false}"

# ============================================================
# SGLang rollout -- copied from the validated standalone Gemma3 policy
# ============================================================
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-8192}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-1}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.80}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:--1}"
SGLANG_PAGE_SIZE="${SGLANG_PAGE_SIZE:-64}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-triton}"
SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-auto}"
SGLANG_ENABLE_NAN_DETECTION="${SGLANG_ENABLE_NAN_DETECTION:-True}"
# Gemma3 VLM / SWA correctness baseline.
SGLANG_DISABLE_RADIX_CACHE="${SGLANG_DISABLE_RADIX_CACHE:-True}"
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-True}"
SGLANG_DISABLE_CUSTOM_ALL_REDUCE="${SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-False}"
# ============================================================
# HCU / Ray environment note
# ============================================================
# IMPORTANT:
# run.sh --restart-ray starts the Ray cluster BEFORE this launcher is executed.
# Therefore DO NOT export cluster-wide HCU/RCCL/torch environment variables here.
# Doing so makes the driver environment differ from the already-running Ray
# worker environment and common.sh will intentionally fail its preflight check.
#
# Put cluster-wide variables in the selected AREAL_ENV_PROFILE (gemma3), or
# export them in the shell BEFORE invoking run.sh so Ray inherits them.
#
# The gemma3 profile mirrors the validated standalone HCU/HIP/SGLang environment
# and explicitly removes Qwen/MLA-oriented fast paths.

# ============================================================
# Safety check: keep the rollout topology equal to standalone while debugging
# ============================================================
if [[ "${ROLLOUT_BACKEND}" =~ t([0-9]+) ]]; then
  ROLLOUT_TP="${BASH_REMATCH[1]}"
  if [[ "${ROLLOUT_TP}" != "4" ]]; then
    echo "[WARN] Standalone-validated Gemma3 baseline uses TP4; current ROLLOUT_BACKEND=${ROLLOUT_BACKEND}." >&2
    echo "[WARN] Changing TP changes weight sharding and HCU kernel shapes; use TP4 for the A/B correctness run." >&2
  fi
fi

# ============================================================
# AReaL configuration arrays
# ============================================================
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
  "train_dataset.path=${DATASET_PATH}"
  "valid_dataset.path=${DATASET_PATH}"
  "+train_dataset.max_length=${TRAIN_MAX_LENGTH}"
  "train_dataset.batch_size=${TRAIN_BATCH_SIZE}"
  "valid_dataset.batch_size=${VALID_BATCH_SIZE}"
)

ACTOR_CONFIG=(
  "actor.backend=${ACTOR_BACKEND}"
  "actor.path=${MODEL_PATH}"
  "actor.dtype=bfloat16"
  "actor.disable_dropout=true"
  "actor.gradient_checkpointing=true"
  "actor.optimizer.type=adam"
  "actor.optimizer.lr=${ACTOR_LR}"
  "actor.eps_clip=0.2"
  "+actor.weight_update_mode=${WEIGHT_UPDATE_MODE}"
  "actor.mb_spec.max_tokens_per_mb=${ACTOR_MAX_TOKENS_PER_MB}"
  "+actor.mb_spec.packing_algorithm=ffd"
  "++actor.mask_no_eos_with_zero=True"
  "++actor.attn_impl=${ACTOR_ATTN_IMPL}"
  "++actor.fsdp.memory_efficient_load=${FSDP_MEMORY_EFFICIENT_LOAD}"
  "++actor.fsdp.offload_params=${FSDP_OFFLOAD_PARAMS}"
)

ROLLOUT_CONFIG=(
  "rollout.backend=${ROLLOUT_BACKEND}"
  "rollout.max_concurrent_rollouts=${ROLLOUT_MAX_CONCURRENT}"
  "gconfig.n_samples=${N_SAMPLES}"
  "gconfig.max_new_tokens=${MAX_NEW_TOKENS}"
  "+rollout.setup_timeout=${ROLLOUT_SETUP_TIMEOUT}"
)

SGLANG_CONFIG=(
  "sglang.model_path=${MODEL_PATH}"
  "tokenizer_path=${TOKENIZER_PATH}"
  "sglang.dtype=bfloat16"
  # Intentional AReaL protocol difference from standalone: Geometry3K's VLM
  # workflow sends pre-tokenized input_ids + image_data. Keeping this True also
  # avoids the previously observed Gemma3 0-image-token vs 256-embedding mismatch.
  "sglang.skip_tokenizer_init=True"
  "sglang.context_length=${SGLANG_CONTEXT_LENGTH}"
  "sglang.max_running_requests=${SGLANG_MAX_RUNNING_REQUESTS}"
  "sglang.mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}"
  "sglang.enable_multimodal=True"
  "++sglang.kv_cache_dtype=${SGLANG_KV_CACHE_DTYPE}"
  "++sglang.chunked_prefill_size=${SGLANG_CHUNKED_PREFILL_SIZE}"
  "++sglang.page_size=${SGLANG_PAGE_SIZE}"
  "++sglang.disable_radix_cache=${SGLANG_DISABLE_RADIX_CACHE}"
  "++sglang.disable_cuda_graph=${SGLANG_DISABLE_CUDA_GRAPH}"
  "++sglang.attention_backend=${SGLANG_ATTENTION_BACKEND}"
  "+sglang.disable_custom_all_reduce=${SGLANG_DISABLE_CUSTOM_ALL_REDUCE}"
  "++sglang.enable_nan_detection=${SGLANG_ENABLE_NAN_DETECTION}"
)

TRAINER_CONFIG=(
  "total_train_epochs=${TOTAL_TRAIN_EPOCHS}"
  "++total_train_steps=${TOTAL_TRAIN_STEPS}"
  "saver.freq_epochs=null"
  "recover.freq_epochs=null"
  "evaluator.freq_epochs=null"
)

# ============================================================
# Prepare / launch
# ============================================================
export MODEL_PATH TOKENIZER_PATH
export N_NODES N_GPUS_PER_NODE
export ACTOR_BACKEND ROLLOUT_BACKEND WEIGHT_UPDATE_MODE
export EXPERIMENT_NAME TRIAL_NAME TIMESTAMP LOG_DIR LOG_FILE
export FILER_ROOT NAME_RESOLVE_ROOT

# common.sh validates Ray/resources/model paths and sets up the shared runtime.
grpo_prepare_run "${N_NODES}" "$((N_NODES * N_GPUS_PER_NODE))"
grpo_print_summary

grpo_launch \
  "${CLUSTER_CONFIG[@]}" \
  "${DATA_CONFIG[@]}" \
  "${ACTOR_CONFIG[@]}" \
  "${ROLLOUT_CONFIG[@]}" \
  "${SGLANG_CONFIG[@]}" \
  "${TRAINER_CONFIG[@]}"
