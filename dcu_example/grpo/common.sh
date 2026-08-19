#!/usr/bin/env bash
# Shared helpers for DCU AReaL GRPO examples.

GRPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export AREAL_RAY_MANAGED_DEVICES=1
# shellcheck disable=SC1091
source "${GRPO_DIR}/../common/common_env.sh"
cd "${AREAL_ROOT}"

GRPO_ENTRYPOINT="${GRPO_ENTRYPOINT:-examples/math/gsm8k_rl.py}"
GRPO_CONFIG="${GRPO_CONFIG:-examples/math/gsm8k_grpo.yaml}"
CLEAN_BEFORE_TRAIN="${CLEAN_BEFORE_TRAIN:-1}"

_grpo_default_host_ip() {
  hostname -I 2>/dev/null | awk '{print $1}'
}

grpo_resolve_ray_address() {
  if [[ -z "${RAY_ADDRESS:-}" ]]; then
    local host_ip
    host_ip="${HOST_IP:-$(_grpo_default_host_ip)}"
    [[ -n "${host_ip}" ]] || {
      echo "[ERROR] RAY_ADDRESS is not set and the local host IP could not be detected." >&2
      return 2
    }
    export RAY_ADDRESS="${host_ip}:${RAY_PORT}"
  fi
  echo "[INFO] RAY_ADDRESS=${RAY_ADDRESS}"
}

grpo_validate_files() {
  [[ -d "${MODEL_PATH}" ]] || {
    echo "[ERROR] Model directory does not exist: ${MODEL_PATH}" >&2
    return 2
  }
  if [[ "${TOKENIZER_PATH:-}" == /* && ! -d "${TOKENIZER_PATH}" ]]; then
    echo "[ERROR] Tokenizer directory does not exist: ${TOKENIZER_PATH}" >&2
    return 2
  fi
  [[ "${N_NODES}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[ERROR] N_NODES must be a positive integer, got: ${N_NODES}" >&2
    return 2
  }
  [[ "${N_GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[ERROR] N_GPUS_PER_NODE must be a positive integer, got: ${N_GPUS_PER_NODE}" >&2
    return 2
  }
  [[ -f "${AREAL_ROOT}/${GRPO_ENTRYPOINT}" ]] || {
    echo "[ERROR] GRPO entrypoint does not exist: ${AREAL_ROOT}/${GRPO_ENTRYPOINT}" >&2
    return 2
  }
  [[ -f "${AREAL_ROOT}/${GRPO_CONFIG}" ]] || {
    echo "[ERROR] GRPO config does not exist: ${AREAL_ROOT}/${GRPO_CONFIG}" >&2
    return 2
  }
}


# Validate the outer rollout batch against FSDP data-parallel dispatch.
# TrainController partitions the list returned by rollout.prepare_batch() across
# actor DP ranks. gconfig.n_samples lives *inside* each rollout item and does not
# increase the number of dispatchable outer items. Therefore TRAIN_BATCH_SIZE
# must be at least the FSDP DP size and divisible by it.
grpo_validate_fsdp_batching() {
  [[ "${ACTOR_BACKEND:-}" == fsdp:* ]] || return 0

  local dp_size=1
  if [[ "${ACTOR_BACKEND}" =~ d([0-9]+) ]]; then
    dp_size="${BASH_REMATCH[1]}"
  fi

  if [[ ! "${TRAIN_BATCH_SIZE:-}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] TRAIN_BATCH_SIZE must be a positive integer for FSDP, got: ${TRAIN_BATCH_SIZE:-unset}" >&2
    return 2
  fi

  if (( TRAIN_BATCH_SIZE < dp_size )); then
    echo "[ERROR] TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} is smaller than FSDP DP size ${dp_size}." >&2
    echo "        actor.backend=${ACTOR_BACKEND}" >&2
    echo "        Use TRAIN_BATCH_SIZE >= ${dp_size}." >&2
    return 2
  fi

  if (( TRAIN_BATCH_SIZE % dp_size != 0 )); then
    echo "[ERROR] TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} must be divisible by FSDP DP size ${dp_size}." >&2
    echo "        actor.backend=${ACTOR_BACKEND}" >&2
    echo "        Examples: ${dp_size}, $((dp_size * 2)), $((dp_size * 3)), ..." >&2
    return 2
  fi

  if [[ "${VALID_BATCH_SIZE:-}" =~ ^[1-9][0-9]*$ ]] && (( VALID_BATCH_SIZE % dp_size != 0 )); then
    echo "[WARN] VALID_BATCH_SIZE=${VALID_BATCH_SIZE} is not divisible by FSDP DP size ${dp_size}." >&2
    echo "       Evaluation may pad batches, but using a multiple of ${dp_size} is safer." >&2
  fi

  echo "[OK] FSDP batch dispatch check: dp=${dp_size}, train_batch=${TRAIN_BATCH_SIZE}, valid_batch=${VALID_BATCH_SIZE:-unset}"
}

grpo_validate_ray_cluster() {
  local expected_nodes="${1:?expected nodes required}"
  local expected_gpus="${2:?expected GPUs required}"
  ray status --address="${RAY_ADDRESS}"
  EXPECTED_NODES="${expected_nodes}" EXPECTED_GPUS="${expected_gpus}" \
  "${PYTHON_BIN}" - <<'PY'
import os
import ray
address = os.environ["RAY_ADDRESS"]
expected_nodes = int(os.environ["EXPECTED_NODES"])
expected_gpus = float(os.environ["EXPECTED_GPUS"])
ray.init(address=address)
resources = ray.cluster_resources()
nodes = [node for node in ray.nodes() if node.get("Alive")]
print("Ray cluster resources:")
print(resources)
print("Alive nodes:", len(nodes))
for node in nodes:
    print("node:", node.get("NodeManagerAddress"), "resources:", node.get("Resources"))
if len(nodes) != expected_nodes:
    raise RuntimeError(f"Expected {expected_nodes} Ray node(s), got {len(nodes)}")
if resources.get("GPU", 0) != expected_gpus:
    raise RuntimeError(f"Expected {expected_gpus:g} GPUs/DCUs, got {resources.get('GPU', 0)}")
ray.shutdown()
print("[OK] Ray cluster resource check passed")
PY
}

grpo_prepare_run() {
  local expected_nodes="${1:?expected nodes required}"
  local expected_gpus="${2:?expected GPUs required}"
  areal_preflight_common
  grpo_validate_files
  grpo_validate_fsdp_batching
  grpo_resolve_ray_address
  mkdir -p "${LOG_DIR}" "${FILER_ROOT}" "${NAME_RESOLVE_ROOT}"
  if [[ "${CLEAN_BEFORE_TRAIN}" == "1" ]]; then
    echo "===== Cleaning old local AReaL/SGLang processes ====="
    CLEAN_NAME_RESOLVE=1 STOP_RAY=0 bash "${GRPO_DIR}/../scripts/cleanup_areal.sh"
  fi
  echo "===== Python / source environment ====="
  areal_print_python_env
  grpo_validate_ray_cluster "${expected_nodes}" "${expected_gpus}"
  areal_save_env_snapshot "${LOG_DIR}/runtime_env.txt"
  areal_validate_ray_worker_env | tee -a "${LOG_DIR}/runtime_env.txt"
}

grpo_print_summary() {
  echo "============================================================"
  echo "Experiment:      ${EXPERIMENT_NAME}"
  echo "Trial:           ${TRIAL_NAME}"
  echo "Model:           ${MODEL_PATH}"
  echo "Actor backend:   ${ACTOR_BACKEND}"
  echo "Rollout backend: ${ROLLOUT_BACKEND}"
  echo "Nodes:           ${N_NODES}"
  echo "DCUs/node:       ${N_GPUS_PER_NODE}"
  echo "Train batch:     ${TRAIN_BATCH_SIZE}"
  echo "Valid batch:     ${VALID_BATCH_SIZE}"
  echo "Samples:         ${N_SAMPLES}"
  echo "Max new tokens:  ${MAX_NEW_TOKENS}"
  echo "Ray:             ${RAY_ADDRESS}"
  echo "Log:             ${LOG_FILE}"
  echo "============================================================"
}

grpo_launch() {
  local -a overrides=("$@")
  set +e
  "${PYTHON_BIN}" "${GRPO_ENTRYPOINT}" --config "${GRPO_CONFIG}" \
    "${overrides[@]}" 2>&1 | tee "${LOG_FILE}"
  local status=${PIPESTATUS[0]}
  set -e
  echo
  echo "============================================================"
  echo "Training exit code: ${status}"
  echo "Log file: ${LOG_FILE}"
  echo "============================================================"
  return "${status}"
}
