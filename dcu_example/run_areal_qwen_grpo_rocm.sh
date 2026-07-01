#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AREAL_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${AREAL_HOME}"
# =========================
# User Config
# =========================
export PYTHONPATH=/home/dcu_megatron/Megatron-Bridge/src:/home/dcu_megatron/Megatron-LM:/home/sglang_v0.5.10/python:$PYTHONPATH
# export HIP_VISIBLE_DEVICES=0,1,2,3,4
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4

MODEL_PATH=/model/qwen3/Qwen3-1.7B
TOKENIZER_PATH=${MODEL_PATH}

EXPERIMENT_NAME=gsm8k-rocm-smoke
TRIAL_NAME=trial-qwen-grpo-fa3

N_GPUS_PER_NODE=5

# Parallel config
ROLLOUT_BACKEND=sglang:d1p1t2
ACTOR_BACKEND=megatron:d1p1t1

# Dataset / training config
TRAIN_BATCH_SIZE=4
VALID_BATCH_SIZE=4
N_SAMPLES=4
MAX_NEW_TOKENS=256
TOTAL_TRAIN_EPOCHS=1

# SGLang config
SGLANG_MEM_FRACTION_STATIC=0.5
SGLANG_CHUNKED_PREFILL_SIZE=65536
SGLANG_PAGE_SIZE=64
SGLANG_ATTENTION_BACKEND=fa3

# Actor config
ACTOR_ATTN_IMPL=sdpa

# =========================
# Environment
# =========================

export USE_DCU_CUSTOM_ALLREDUCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000

export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3

# Conservative ROCm path for AReaL embedded SGLang
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=01
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1


# =========================
# Cleanup Old Workers
# =========================

pkill -f areal.infra.rpc.guard || true
pkill -f "sglang" || true
rm -rf /tmp/areal/name_resolve

# =========================
# Run AReaL GRPO
# =========================

python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=local \
  experiment_name=${EXPERIMENT_NAME} \
  trial_name=${TRIAL_NAME} \
  cluster.n_nodes=1 \
  cluster.n_gpus_per_node=${N_GPUS_PER_NODE} \
  rollout.backend=${ROLLOUT_BACKEND} \
  actor.backend=${ACTOR_BACKEND} \
  actor.path=${MODEL_PATH} \
  sglang.model_path=${MODEL_PATH} \
  tokenizer_path=${TOKENIZER_PATH} \
  train_dataset.batch_size=${TRAIN_BATCH_SIZE} \
  valid_dataset.batch_size=${VALID_BATCH_SIZE} \
  gconfig.n_samples=${N_SAMPLES} \
  gconfig.max_new_tokens=${MAX_NEW_TOKENS} \
  sglang.mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC} \
  +sglang.chunked_prefill_size=${SGLANG_CHUNKED_PREFILL_SIZE} \
  +sglang.page_size=${SGLANG_PAGE_SIZE} \
  +sglang.disable_radix_cache=True \
  +sglang.disable_cuda_graph=True \
  +sglang.disable_cuda_graph_padding=True \
  +sglang.disable_overlap_schedule=True \
  +sglang.attention_backend=${SGLANG_ATTENTION_BACKEND} \
  total_train_epochs=${TOTAL_TRAIN_EPOCHS} \
  +actor.attn_impl=${ACTOR_ATTN_IMPL}