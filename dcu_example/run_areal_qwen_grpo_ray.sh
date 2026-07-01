#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AREAL_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${AREAL_HOME}"
# =========================
# User Config
# =========================
export PYTHONPATH=${AREAL_HOME}:/home/dcu_megatron/Megatron-Bridge/src:/home/dcu_megatron/Megatron-LM:/home/sglang_v0.5.10/python:$PYTHONPATH
unset CUDA_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES

MODEL_PATH=/model/qwen3/Qwen3-8B
TOKENIZER_PATH=${MODEL_PATH}

EXPERIMENT_NAME=gsm8k-rocm-ray
TRIAL_NAME=trial-qwen3-ray-fa3

#nhb: Capture the first complete PPO update on the Megatron actor.
export AREAL_TORCH_PROF_DIR=/home/areal_runs/profiles/${TRIAL_NAME}-$(date "+%Y%m%d-%H%M%S")
export AREAL_TORCH_PROF_RANK=0
#nhb: Profile the third PPO update after two warmup updates.
export AREAL_TORCH_PROF_UPDATE=2


mkdir -p "${AREAL_TORCH_PROF_DIR}"

N_GPUS_PER_NODE=8

ROLLOUT_BACKEND=sglang:d1p1t4
ACTOR_BACKEND=megatron:d1p1t4

TRAIN_BATCH_SIZE=1
VALID_BATCH_SIZE=1
N_SAMPLES=2
MAX_NEW_TOKENS=64
TOTAL_TRAIN_EPOCHS=1

SGLANG_MEM_FRACTION_STATIC=0.4
SGLANG_CHUNKED_PREFILL_SIZE=65536
SGLANG_PAGE_SIZE=64
SGLANG_ATTENTION_BACKEND=fa3

ACTOR_ATTN_IMPL=sdpa

# If 1, script restarts local Ray head before training.
# If you already started Ray manually, set this to 0.
RESTART_RAY=1

# =========================
# SGLang Env
# =========================

export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3

export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1

# =========================
# Cleanup Old AReaL/SGLang
# =========================

pkill -f areal.infra.rpc.guard || true
pkill -f "sglang" || true
rm -rf /tmp/areal/name_resolve
export RAY_DEDUP_LOGS=0
# =========================
# Ray Setup
# =========================

if [ "${RESTART_RAY}" = "1" ]; then
  ray stop -f || true

  HOST_IP=$(hostname -I | awk '{print $1}')

  ray start --head \
    --node-ip-address=${HOST_IP} \
    --num-gpus=${N_GPUS_PER_NODE} \
    --num-cpus=128 \
    --disable-usage-stats
fi

ray status

python - <<'PY'
import ray
ray.init(address="auto")
print("Ray cluster resources:")
print(ray.cluster_resources())
ray.shutdown()
PY

# =========================
# Run AReaL with Ray Scheduler
# =========================
#  +train_dataset.scheduling_spec=null \
 # +valid_dataset.scheduling_spec=null \
 
time=$(date "+%Y%m%d-%H%M%S")
logpath=/home/areal_runs/logs/ray-${EXPERIMENT_NAME}-${TRIAL_NAME}

python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=ray \
  experiment_name=${EXPERIMENT_NAME} \
  trial_name=${TRIAL_NAME} \
  cluster.n_nodes=1 \
  cluster.n_gpus_per_node=${N_GPUS_PER_NODE} \
   +train_dataset.scheduling_spec=null \
 +valid_dataset.scheduling_spec=null \
  rollout.backend=${ROLLOUT_BACKEND} \
  actor.backend=${ACTOR_BACKEND} \
  actor.path=${MODEL_PATH} \
    actor.weight_update_mode=xccl \
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
  +actor.attn_impl=${ACTOR_ATTN_IMPL} 2>&1 | tee ${logpath}-${time}.log