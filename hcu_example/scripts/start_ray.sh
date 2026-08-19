#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common/common_env.sh"

areal_preflight_common

NODE_IP="${1:-${NODE_IP:-$(hostname -I | awk '{print $1}')}}"
NUM_GPUS="${NUM_GPUS:-${N_GPUS_PER_NODE:-8}}"
NUM_CPUS="${NUM_CPUS:-${N_CPUS_PER_NODE:-128}}"
STOP_EXISTING_RAY="${STOP_EXISTING_RAY:-1}"

if [[ -z "${NODE_IP}" ]]; then
  echo "[ERROR] Unable to determine NODE_IP. Usage: NODE_IP=<ip> bash $0 or bash $0 <ip>" >&2
  exit 1
fi

if [[ "${STOP_EXISTING_RAY}" == "1" ]]; then
  ray stop -f || true
fi

ray_args=(
  --head
  --node-ip-address="${NODE_IP}"
  --port="${RAY_PORT}"
  --num-gpus="${NUM_GPUS}"
  --num-cpus="${NUM_CPUS}"
  --temp-dir="${RAY_TMPDIR}"
  --min-worker-port="${RAY_MIN_WORKER_PORT}"
  --max-worker-port="${RAY_MAX_WORKER_PORT}"
  --object-manager-port="${RAY_OBJECT_MANAGER_PORT}"
  --node-manager-port="${RAY_NODE_MANAGER_PORT}"
  --disable-usage-stats
)

if [[ -n "${RAY_RESOURCES:-}" ]]; then
  ray_args+=(--resources="${RAY_RESOURCES}")
fi

echo "[INFO] Starting Ray head ${NODE_IP}:${RAY_PORT} with ${NUM_GPUS} GPUs, ${NUM_CPUS} CPUs"
echo "[INFO] RAY_TMPDIR=${RAY_TMPDIR}"
echo "[INFO] AREAL_ENV_PROFILE=${AREAL_ENV_PROFILE}"
echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"
echo "[INFO] PYTHONPATH=${PYTHONPATH}"
echo "[INFO] SGLANG_HOME=${SGLANG_HOME}"
ray start "${ray_args[@]}"

sleep 3
export RAY_ADDRESS="${NODE_IP}:${RAY_PORT}"
ray status --address="${RAY_ADDRESS}"

echo "[INFO] Validating Ray worker environment inherited from the head daemon..."
areal_validate_ray_worker_env

echo "[OK] Ray head started and environment validated: ${RAY_ADDRESS}"
echo "Worker command: bash ${SCRIPT_DIR}/start_ray_worker.sh ${NODE_IP} <worker-ip>"
