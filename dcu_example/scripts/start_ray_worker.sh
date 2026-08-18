#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common/common_env.sh"

areal_preflight_common

HEAD_IP="${1:-${HEAD_IP:-}}"
WORKER_IP="${2:-${WORKER_IP:-$(hostname -I | awk '{print $1}')}}"
NUM_GPUS="${NUM_GPUS:-${N_GPUS_PER_NODE:-8}}"
NUM_CPUS="${NUM_CPUS:-${N_CPUS_PER_NODE:-128}}"
STOP_EXISTING_RAY="${STOP_EXISTING_RAY:-1}"

if [[ -z "${HEAD_IP}" ]]; then
  echo "Usage: bash $0 <HEAD_IP> [WORKER_IP]" >&2
  exit 1
fi

if [[ -z "${WORKER_IP}" ]]; then
  echo "[ERROR] Unable to determine worker IP. Pass it as the second argument or set WORKER_IP." >&2
  exit 1
fi

if [[ "${STOP_EXISTING_RAY}" == "1" ]]; then
  ray stop -f || true
fi

ray_args=(
  --address="${HEAD_IP}:${RAY_PORT}"
  --node-ip-address="${WORKER_IP}"
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

echo "[INFO] Joining Ray cluster ${HEAD_IP}:${RAY_PORT} from ${WORKER_IP} with ${NUM_GPUS} GPUs, ${NUM_CPUS} CPUs"
echo "[INFO] RAY_TMPDIR=${RAY_TMPDIR}"
echo "[INFO] AREAL_ENV_PROFILE=${AREAL_ENV_PROFILE}"
echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"
echo "[INFO] PYTHONPATH=${PYTHONPATH}"
echo "[INFO] SGLANG_HOME=${SGLANG_HOME}"
ray start "${ray_args[@]}"

export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
ray status --address="${RAY_ADDRESS}"
echo "[INFO] Validating Ray worker environments across the cluster..."
areal_validate_ray_worker_env
echo "[OK] Worker joined and cluster environment validated: ${WORKER_IP} -> ${RAY_ADDRESS}"
