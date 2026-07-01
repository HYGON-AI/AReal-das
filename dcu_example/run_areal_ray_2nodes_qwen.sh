#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AREAL_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${AREAL_HOME}"
MODE=${1:-}
HEAD_IP=${2:-}

# =========================
# User Config
# =========================
AREAL_HOME=/home/AReaL-1.0.4
SGLANG_HOME=/home/sglang_v0.5.10/python
VENV=/opt/areal-venv-py31115

MODEL_PATH=/model/qwen3/Qwen3-14B
TOKENIZER_PATH=${MODEL_PATH}

EXPERIMENT_NAME=gsm8k-rocm-ray-2nodes
TRIAL_NAME=trial-qwen35-4b-xccl-2nodes

N_NODES=2
N_GPUS_PER_NODE=8
N_CPUS_PER_NODE=128

# 强制 actor 独占一个 8 卡节点，rollout 独占另一个 8 卡节点
ACTOR_BACKEND=megatron:d1p1t8
ROLLOUT_BACKEND=sglang:d1p1t8

TRAIN_BATCH_SIZE=2
VALID_BATCH_SIZE=2
N_SAMPLES=2
MAX_NEW_TOKENS=256

SGLANG_MEM_FRACTION_STATIC=0.5
SGLANG_CHUNKED_PREFILL_SIZE=65536
SGLANG_PAGE_SIZE=64
SGLANG_ATTENTION_BACKEND=fa3

ACTOR_ATTN_IMPL=sdpa

# /home 两节点共享的话，用 /home 下路径
FILERoot=/home/areal_runs/experiments
NAME_RESOLVE_ROOT=/home/areal_runs/name_resolve

# =========================
# Common Env
# =========================
source ${VENV}/bin/activate
cd ${AREAL_HOME}

# export PYTHONPATH=${SGLANG_HOME}:${PYTHONPATH}
export PYTHONPATH=${AREAL_HOME}:/home/dcu_megatron/Megatron-Bridge/src:/home/dcu_megatron/Megatron-LM:/home/sglang_v0.5.10/python:$PYTHONPATH

# Ray 模式不要全局暴露 CUDA/HIP 全卡
unset CUDA_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES

export RAY_DEDUP_LOGS=0

# SGLang / HCU env
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

export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512

mkdir -p ${FILERoot} ${NAME_RESOLVE_ROOT}

sysctl -w kernel.numa_balancing=0 || true

# =========================
# Modes
# =========================
if [ "${MODE}" = "head" ]; then
  HOST_IP=$(hostname -I | awk '{print $1}')

  ray stop -f || true

  ray start --head \
    --node-ip-address=${HOST_IP} \
    --port=6379 \
    --num-gpus=${N_GPUS_PER_NODE} \
    --num-cpus=${N_CPUS_PER_NODE} \
    --disable-usage-stats

  echo "Head started: ${HOST_IP}:6379"
  echo "On worker node, run:"
  echo "bash $0 worker ${HOST_IP}"
  exit 0
fi

if [ "${MODE}" = "worker" ]; then
  if [ -z "${HEAD_IP}" ]; then
    echo "Usage: bash $0 worker <HEAD_IP>"
    exit 1
  fi

  ray stop -f || true

  ray start \
    --address=${HEAD_IP}:6379 \
    --num-gpus=${N_GPUS_PER_NODE} \
    --num-cpus=${N_CPUS_PER_NODE}

  ray status
  exit 0
fi

if [ "${MODE}" = "status" ]; then
  ray status
  python - <<'PY'
import ray
ray.init(address="auto")
print(ray.cluster_resources())
ray.shutdown()
PY
  exit 0
fi

if [ "${MODE}" = "stop" ]; then
  ray stop -f || true
  pkill -f areal.infra.rpc.guard || true
  pkill -f sglang || true
  pkill -f RayRPCServer || true
  exit 0
fi

if [ "${MODE}" = "train" ]; then
  ray status

  python - <<'PY'
import ray
ray.init(address="auto")
print("Ray cluster resources:")
print(ray.cluster_resources())
ray.shutdown()
PY


time=$(date "+%Y%m%d-%H%M%S")
logpath=/home/areal_runs/logs/${EXPERIMENT_NAME}-${TRIAL_NAME}

mkdir -p /home/areal_runs/logs

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
    actor.optimizer.lr=1.0e-6 \
    actor.eps_clip=0.2 \
    +actor.mask_no_eos_with_zero=True \
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
    actor.weight_update_mode=xccl \
    total_train_epochs=1 \
    saver.freq_epochs=null \
    recover.freq_epochs=null \
    evaluator.freq_epochs=null \
    +actor.attn_impl=${ACTOR_ATTN_IMPL} \
    2>&1 | tee ${logpath}-${time}.log

  exit 0
fi

echo "Usage:"
echo "  bash $0 head"
echo "  bash $0 worker <HEAD_IP>"
echo "  bash $0 status"
echo "  bash $0 train"
echo "  bash $0 stop"
exit 1