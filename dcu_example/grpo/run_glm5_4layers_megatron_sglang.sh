#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export AREAL_ENV_PROFILE="${AREAL_ENV_PROFILE:-glm5}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

# ==============================================================================
# Model / dataset
# ==============================================================================
MODEL_PATH="${MODEL_PATH:-/workspace/GLM-5-4Layers}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
DATASET_PATH="${DATASET_PATH:-openai/gsm8k}"

# ==============================================================================
# Cluster
# ==============================================================================
N_NODES="${N_NODES:-2}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# Actor: attention d1/p2/t4, FFN d1/p2/t1/e4
# Rollout: SGLang d1/p1/t8
ACTOR_BACKEND="${ACTOR_BACKEND:-megatron:(attn:d1p2t4|ffn:d1p2t1e4)}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-sglang:d1p1t8}"

BRIDGE_TYPE="${BRIDGE_TYPE:-megatron-bridge}"
USE_MBRIDGE_SAVE="${USE_MBRIDGE_SAVE:-false}"
WEIGHT_UPDATE_MODE="${WEIGHT_UPDATE_MODE:-xccl}"

# ==============================================================================
# Experiment
# ==============================================================================
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gsm8k-glm5-4layer-dcu-2nodes-megatron}"
TRIAL_NAME="${TRIAL_NAME:-tp8-ep8-smoke}"
TIMESTAMP="${TIMESTAMP:-$(date '+%Y%m%d-%H%M%S')}"
LOG_DIR="${LOG_DIR:-${AREAL_RUNS_ROOT}/${EXPERIMENT_NAME}-${TRIAL_NAME}-${TIMESTAMP}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train.log}"

# ==============================================================================
# GRPO / rollout
# ==============================================================================
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-1}"

# IMPORTANT:
# Actor uses PP=2. In Megatron forward/compute_logp, AReaL enforces at least
# 2 * PP = 4 micro-batches. With train_batch_size=1, use n_samples=4 so one
# prompt yields four rollout trajectories and the allocator has >=4 sequences.
N_SAMPLES="${N_SAMPLES:-4}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TOTAL_TRAIN_EPOCHS="${TOTAL_TRAIN_EPOCHS:-1}"
TOTAL_TRAIN_STEPS="${TOTAL_TRAIN_STEPS:-10}"

# ==============================================================================
# Actor micro-batching
# ==============================================================================
ACTOR_N_MBS="${ACTOR_N_MBS:-4}"

# IMPORTANT:
# This is the TOTAL token capacity of one Actor micro-batch before CP splitting.
# It must be >= every individual prompt+response trajectory length.
# The previous value 128 caused:
#   RuntimeError: Values [1088] is larger than capacity 128
# Keep it aligned with SGLang context length for a safe upper bound.
ACTOR_MB_TOKENS="${ACTOR_MB_TOKENS:-2048}"

ACTOR_LR="${ACTOR_LR:-1.0e-6}"

# ==============================================================================
# SGLang
# ==============================================================================
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.45}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-2048}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-1024}"
SGLANG_PAGE_SIZE="${SGLANG_PAGE_SIZE:-64}"
SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-fp8_e4m3}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-1}"

# ==============================================================================
# Preflight checks
# ==============================================================================
infer_actor_parallel_size() {
  local backend="$1"
  local kind="$2"

  # Prefer the attention branch for heterogeneous Megatron backends, e.g.:
  # megatron:(attn:d1p2t4|ffn:d1p2t1e4)
  if [[ "${backend}" =~ attn:d([0-9]+)p([0-9]+)t([0-9]+) ]]; then
    case "${kind}" in
      dp) printf '%s\n' "${BASH_REMATCH[1]}" ;;
      pp) printf '%s\n' "${BASH_REMATCH[2]}" ;;
      tp) printf '%s\n' "${BASH_REMATCH[3]}" ;;
      *) return 1 ;;
    esac
    return 0
  fi

  # Plain Megatron backend, e.g. megatron:d1p2t4
  if [[ "${backend}" =~ megatron:d([0-9]+)p([0-9]+)t([0-9]+) ]]; then
    case "${kind}" in
      dp) printf '%s\n' "${BASH_REMATCH[1]}" ;;
      pp) printf '%s\n' "${BASH_REMATCH[2]}" ;;
      tp) printf '%s\n' "${BASH_REMATCH[3]}" ;;
      *) return 1 ;;
    esac
    return 0
  fi

  return 1
}

ACTOR_DP_SIZE="$(infer_actor_parallel_size "${ACTOR_BACKEND}" dp || printf '1\n')"
ACTOR_PP_SIZE="$(infer_actor_parallel_size "${ACTOR_BACKEND}" pp || printf '1\n')"
ACTOR_TP_SIZE="$(infer_actor_parallel_size "${ACTOR_BACKEND}" tp || printf '1\n')"

if (( ACTOR_PP_SIZE > 1 )); then
  MIN_FORWARD_MBS=$((2 * ACTOR_PP_SIZE))
else
  MIN_FORWARD_MBS=1
fi

EFFECTIVE_TRAJECTORIES=$((TRAIN_BATCH_SIZE * N_SAMPLES))
MIN_REQUIRED_TRAJECTORIES=$((MIN_FORWARD_MBS * ACTOR_DP_SIZE))

if (( EFFECTIVE_TRAJECTORIES < MIN_REQUIRED_TRAJECTORIES )); then
  cat >&2 <<ERR
ERROR: rollout trajectory count is too small for the Actor pipeline configuration.
  ACTOR_BACKEND              = ${ACTOR_BACKEND}
  Actor DP / PP / TP         = ${ACTOR_DP_SIZE} / ${ACTOR_PP_SIZE} / ${ACTOR_TP_SIZE}
  TRAIN_BATCH_SIZE           = ${TRAIN_BATCH_SIZE}
  N_SAMPLES                  = ${N_SAMPLES}
  effective trajectories     = ${EFFECTIVE_TRAJECTORIES}
  required trajectories      >= ${MIN_REQUIRED_TRAJECTORIES}

AReaL Megatron forward uses at least 2 * PP micro-batches when PP > 1.
Increase N_SAMPLES or TRAIN_BATCH_SIZE before launching.
ERR
  exit 1
fi

if (( ACTOR_MB_TOKENS <= MAX_NEW_TOKENS )); then
  cat >&2 <<ERR
ERROR: ACTOR_MB_TOKENS=${ACTOR_MB_TOKENS} is not large enough for
MAX_NEW_TOKENS=${MAX_NEW_TOKENS} plus prompt tokens.
Increase ACTOR_MB_TOKENS or reduce MAX_NEW_TOKENS.
ERR
  exit 1
fi

if (( ACTOR_MB_TOKENS < SGLANG_CONTEXT_LENGTH )); then
  cat >&2 <<WARN
WARNING: ACTOR_MB_TOKENS=${ACTOR_MB_TOKENS} is smaller than
SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH}.
A rollout whose prompt+response length exceeds ACTOR_MB_TOKENS will fail during
Actor compute_logp micro-batch allocation. The safe setting is:
  ACTOR_MB_TOKENS >= SGLANG_CONTEXT_LENGTH
WARN
fi

if (( ACTOR_N_MBS < MIN_FORWARD_MBS )); then
  echo "INFO: ACTOR_N_MBS=${ACTOR_N_MBS}, but PP=${ACTOR_PP_SIZE} makes AReaL forward use at least ${MIN_FORWARD_MBS} micro-batches." >&2
fi

# ==============================================================================
# Hydra configs
# ==============================================================================
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
  "+train_dataset.scheduling_spec=null"
  "+valid_dataset.scheduling_spec=null"
  "train_dataset.path=${DATASET_PATH}"
  "train_dataset.batch_size=${TRAIN_BATCH_SIZE}"
  "train_dataset.num_workers=0"
  "valid_dataset.path=${DATASET_PATH}"
  "valid_dataset.batch_size=${VALID_BATCH_SIZE}"
  "valid_dataset.num_workers=0"
)

ACTOR_CONFIG=(
  "actor.backend='${ACTOR_BACKEND}'"
  "actor.path=${MODEL_PATH}"
  "actor.dtype=bfloat16"
  "actor.gradient_checkpointing=true"
  "actor.weight_update_mode=${WEIGHT_UPDATE_MODE}"
  "++actor.setup_timeout=3600"
  "++actor.request_timeout=3600"
  "++actor.megatron.bridge_type=${BRIDGE_TYPE}"
  "++actor.megatron.use_mbridge_save=${USE_MBRIDGE_SAVE}"
  "++actor.megatron.recompute_granularity=full"
  "++actor.megatron.recompute_method=uniform"
  "++actor.megatron.recompute_num_layers=1"
  "++actor.megatron.moe_router_dtype=fp32"
  "++actor.megatron.moe_token_dispatcher_type=alltoall"
  "++actor.megatron.moe_permute_fusion=false"
  "++actor.megatron.moe_shared_expert_overlap=false"
  "actor.optimizer.type=adam_bf16"
  "actor.optimizer.lr=${ACTOR_LR}"
  "actor.eps_clip=0.2"
  "actor.ppo_n_minibatches=1"
  "actor.kl_ctl=0.0"
  "++actor.mb_spec.n_mbs=${ACTOR_N_MBS}"
  "++actor.mb_spec.max_tokens_per_mb=${ACTOR_MB_TOKENS}"
)

ROLLOUT_CONFIG=(
  "rollout.backend=${ROLLOUT_BACKEND}"
  "++rollout.setup_timeout=1800"
  "++rollout.request_timeout=3600"
  "gconfig.n_samples=${N_SAMPLES}"
  "gconfig.min_new_tokens=0"
  "gconfig.max_new_tokens=${MAX_NEW_TOKENS}"
  "gconfig.greedy=false"
  "gconfig.temperature=1.0"
)

SGLANG_CONFIG=(
  "tokenizer_path=${TOKENIZER_PATH}"
  "sglang.model_path=${MODEL_PATH}"
  "sglang.dtype=bfloat16"
  "sglang.mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}"
  "sglang.context_length=${SGLANG_CONTEXT_LENGTH}"
  "sglang.max_running_requests=${SGLANG_MAX_RUNNING_REQUESTS}"
  "++sglang.kv_cache_dtype=${SGLANG_KV_CACHE_DTYPE}"
  "+sglang.nsa_prefill_backend=flashmla_auto"
  "+sglang.nsa_decode_backend=flashmla_kv"
  "++sglang.chunked_prefill_size=${SGLANG_CHUNKED_PREFILL_SIZE}"
  "++sglang.page_size=${SGLANG_PAGE_SIZE}"
  "++sglang.disable_radix_cache=true"
  "++sglang.disable_cuda_graph=true"
  "++sglang.disable_cuda_graph_padding=true"
  "++sglang.disable_overlap_schedule=true"
  "++sglang.attention_backend=null"
)

TRAINER_CONFIG=(
  "total_train_epochs=${TOTAL_TRAIN_EPOCHS}"
  "++total_train_steps=${TOTAL_TRAIN_STEPS}"
  "saver.freq_epochs=null"
  "recover.freq_epochs=null"
  "evaluator.freq_epochs=null"
)

# Disable automatic NUMA balancing. Ignore the failure when the container does
# not have permission to change the host sysctl.
sysctl -w kernel.numa_balancing=0 >/dev/null 2>&1 || true

# ==============================================================================
# Launch
# ==============================================================================
echo "========================================================================"
echo "GLM-5 GRPO launch configuration"
echo "  Actor backend          : ${ACTOR_BACKEND}"
echo "  Actor DP / PP / TP     : ${ACTOR_DP_SIZE} / ${ACTOR_PP_SIZE} / ${ACTOR_TP_SIZE}"
echo "  Train batch size       : ${TRAIN_BATCH_SIZE}"
echo "  N samples              : ${N_SAMPLES}"
echo "  Effective trajectories : ${EFFECTIVE_TRAJECTORIES}"
echo "  Actor n_mbs            : ${ACTOR_N_MBS}"
echo "  Actor MB tokens        : ${ACTOR_MB_TOKENS}"
echo "  Max new tokens         : ${MAX_NEW_TOKENS}"
echo "  SGLang context length  : ${SGLANG_CONTEXT_LENGTH}"
echo "  Weight update mode     : ${WEIGHT_UPDATE_MODE}"
echo "========================================================================"

grpo_prepare_run "${N_NODES}" "$((N_NODES * N_GPUS_PER_NODE))"
grpo_print_summary

grpo_launch \
  "${CLUSTER_CONFIG[@]}" \
  "${DATA_CONFIG[@]}" \
  "${ACTOR_CONFIG[@]}" \
  "${ROLLOUT_CONFIG[@]}" \
  "${SGLANG_CONFIG[@]}" \
  "${TRAINER_CONFIG[@]}"