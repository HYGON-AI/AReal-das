#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AREAL_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${AREAL_HOME}"
# =========================
# Mode
# =========================
# Usage:
#   bash /home/run_areal_ray_2nodes.sh head
#   bash /home/run_areal_ray_2nodes.sh worker <HEAD_IP>
#   bash /home/run_areal_ray_2nodes.sh train <HEAD_IP>

MODE=${1:-}
HEAD_IP=${2:-}

# =========================
# User Config
# =========================

AREAL_DIR=/home/AReaL-1.0.4
SGLANG_SRC=/home/sglang_v0.5.10/python
VENV_DIR=/opt/areal-venv-py31115

MODEL_PATH=/model/qwen3/Qwen3-1.7B
TOKENIZER_PATH=${MODEL_PATH}

EXPERIMENT_NAME=gsm8k-rocm-ray-2nodes
TRIAL_NAME=trial-qwen3-ray-2nodes-fa3

VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_NODES=2
N_GPUS_PER_NODE=8

ROLLOUT_BACKEND=sglang:d1p1t2
ACTOR_BACKEND=fsdp:d1p1t1

TRAIN_BATCH_SIZE=4
VALID_BATCH_SIZE=4
N_SAMPLES=4
MAX_NEW_TOKENS=256
TOTAL_TRAIN_EPOCHS=1

SGLANG_MEM_FRACTION_STATIC=0.5
SGLANG_CHUNKED_PREFILL_SIZE=65536
SGLANG_PAGE_SIZE=64
SGLANG_ATTENTION_BACKEND=fa3
ACTOR_ATTN_IMPL=sdpa

# /home 是共享目录，所以这里放 /home 下
FILERoot=/home/areal_runs/experiments
NAME_RESOLVE_ROOT=/home/areal_runs/name_resolve

# =========================
# Common Env
# =========================

source ${VENV_DIR}/bin/activate
cd ${AREAL_DIR}

export PYTHONPATH=${SGLANG_SRC}:${PYTHONPATH}

unset ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=${VISIBLE_DEVICES}
export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}

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

mkdir -p ${FILERoot}
mkdir -p ${NAME_RESOLVE_ROOT}

sysctl -w kernel.numa_balancing=0 || true

# =========================
# Ray Head
# =========================

if [ "${MODE}" = "head" ]; then
  ray stop -f || true
  pkill -f areal.infra.rpc.guard || true
  pkill -f sglang || true

  THIS_IP=$(hostname -I | awk '{print $1}')

  echo "Starting Ray head on ${THIS_IP}"

  ray start --head \
    --node-ip-address=${THIS_IP} \
    --port=6379 \
    --num-gpus=${N_GPUS_PER_NODE} \
    --disable-usage-stats

  echo
  echo "Head started."
  echo "Run this on worker node:"
  echo "  bash /home/run_areal_ray_2nodes.sh worker ${THIS_IP}"
  echo
  echo "After worker joined, run this on head:"
  echo "  bash /home/run_areal_ray_2nodes.sh train ${THIS_IP}"
  exit 0
fi

# =========================
# Ray Worker
# =========================

if [ "${MODE}" = "worker" ]; then
  if [ -z "${HEAD_IP}" ]; then
    echo "Usage: bash /home/run_areal_ray_2nodes.sh worker <HEAD_IP>"
    exit 1
  fi

  ray stop -f || true
  pkill -f areal.infra.rpc.guard || true
  pkill -f sglang || true

  THIS_IP=$(hostname -I | awk '{print $1}')

  echo "Starting Ray worker ${THIS_IP}, connecting to ${HEAD_IP}:6379"

  ray start \
    --address=${HEAD_IP}:6379 \
    --node-ip-address=${THIS_IP} \
    --num-gpus=${N_GPUS_PER_NODE} \
    --disable-usage-stats

  ray status --address=${HEAD_IP}:6379
  exit 0
fi

# =========================
# Train On Head
# =========================

if [ "${MODE}" = "train" ]; then
  if [ -z "${HEAD_IP}" ]; then
    HEAD_IP=$(hostname -I | awk '{print $1}')
  fi

  export RAY_ADDRESS=${HEAD_IP}:6379

  ray status --address=${RAY_ADDRESS}

  python - <<PY
import ray
ray.init(address="${RAY_ADDRESS}")
print("Ray cluster resources:")
print(ray.cluster_resources())
ray.shutdown()
PY

  rm -rf ${NAME_RESOLVE_ROOT}
  mkdir -p ${NAME_RESOLVE_ROOT}

  python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=ray \
    experiment_name=${EXPERIMENT_NAME} \
    trial_name=${TRIAL_NAME} \
    cluster.n_nodes=${N_NODES} \
    cluster.n_gpus_per_node=${N_GPUS_PER_NODE} \
    cluster.fileroot=${FILERoot} \
    cluster.name_resolve.nfs_record_root=${NAME_RESOLVE_ROOT} \
    +train_dataset.scheduling_spec=null \
    +valid_dataset.scheduling_spec=null \
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

  exit 0
fi

echo "Usage:"
echo "  bash /home/run_areal_ray_2nodes.sh head"
echo "  bash /home/run_areal_ray_2nodes.sh worker <HEAD_IP>"
echo "  bash /home/run_areal_ray_2nodes.sh train <HEAD_IP>"
exit 1