#!/usr/bin/env bash
set -Eeuo pipefail

# export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
# export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
AREAL_ROOT_RESOLVED="${AREAL_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

usage() {
  cat <<'USAGE'
AReaL DCU GRPO launcher

Model/backend discovery:
  bash run.sh --list
  bash run.sh --search=qwen3
  bash run.sh --model=qwen3_8b --backends
  bash run.sh --model=qwen3_8b --backend=fsdp --info
  bash run.sh --check-fsdp

Training:
  bash run.sh --model=qwen3_8b --backend=fsdp [options]
  bash run.sh --model=qwen3_8b_fsdp_sglang [options]     # legacy alias

Single-node Ray + training:
  bash run.sh --model=qwen3_8b --backend=fsdp --restart-ray

Multi-node Ray lifecycle (run on each physical node):
  # head node
  bash run.sh --ray-head --model=qwen3_30b_a3b_4layers --backend=fsdp \
    --ray-address=10.16.1.48:6379

  # worker node
  bash run.sh --ray-worker --model=qwen3_30b_a3b_4layers --backend=fsdp \
    --ray-address=10.16.1.48:6379 --worker-ip=10.16.1.61

  # then launch training from the head node
  bash run.sh --model=qwen3_30b_a3b_4layers --backend=fsdp \
    --ray-address=10.16.1.48:6379

Options:
  --model=<name>          Base model key, e.g. qwen3_8b.
                          Legacy <model>_<backend>_sglang names are also accepted.
  --backend=<name>        Actor backend: fsdp or megatron.
  --list                  List discovered model keys and available actor backends.
  --search=<pattern>      Search model keys by substring.
  --backends              Show available backends for --model.
  --info                  Show resolved defaults/readiness for a model/backend.
  --check-fsdp            Static-audit every discovered FSDP launcher.
  --dry-run               Resolve and print the launch without starting training.

Training overrides:
  --ray-address=<ip:port> Existing Ray cluster address.
  --model-path=<path>     Override MODEL_PATH.
  --tokenizer-path=<path> Override TOKENIZER_PATH.
  --nodes=<N>             Override N_NODES.
  --gpus-per-node=<N>     Override N_GPUS_PER_NODE.
  --no-cleanup            Do not kill old local AReaL/SGLang processes.
  --restart-ray           Single-node training only: recreate local Ray head first.

Ray-only actions:
  --ray-head              Start/restart a Ray head on this node and exit.
  --ray-worker            Join this node to --ray-address and exit.
  --ray-status            Show/validate Ray cluster status and exit.
  --worker-ip=<ip>        Worker node IP for --ray-worker (auto-detected otherwise).
  --head-ip=<ip>          Head bind IP for --ray-head/--restart-ray.
  --profile=<name>        Ray environment profile if --model is omitted:
                          qwen, qwen35, glm5, base.

  -h, --help              Show this help.
USAGE
}

script_for() {
  local model="$1" backend="$2"
  printf '%s/run_%s_%s_sglang.sh\n' "${SCRIPT_DIR}" "${model}" "${backend}"
}

backend_from_filename() {
  local base
  base="$(basename "$1")"
  if [[ "${base}" == run_*_fsdp_sglang.sh ]]; then
    echo fsdp
  elif [[ "${base}" == run_*_megatron_sglang.sh ]]; then
    echo megatron
  else
    return 1
  fi
}

model_from_filename() {
  local base backend
  base="$(basename "$1")"
  backend="$(backend_from_filename "$1")" || return 1
  base="${base#run_}"
  base="${base%_${backend}_sglang.sh}"
  echo "${base}"
}

discover_models() {
  local f
  shopt -s nullglob
  for f in "${SCRIPT_DIR}"/run_*_fsdp_sglang.sh "${SCRIPT_DIR}"/run_*_megatron_sglang.sh; do
    model_from_filename "${f}"
  done | sort -u
  shopt -u nullglob
}

backends_for_model() {
  local model="$1" backend f
  for backend in fsdp megatron; do
    f="$(script_for "${model}" "${backend}")"
    [[ -f "${f}" ]] && echo "${backend}"
  done
}

model_exists() {
  local model="$1"
  [[ -n "$(backends_for_model "${model}")" ]]
}

extract_default() {
  local file="$1" var="$2" line
  line="$(grep -m1 -E "^(export[[:space:]]+)?${var}=" "${file}" 2>/dev/null || true)"
  if [[ "${line}" =~ :-([^\}]*)\} ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  else
    printf '%s\n' ""
  fi
}

extract_profile() {
  local file="$1" p
  p="$(extract_default "${file}" AREAL_ENV_PROFILE)"
  if [[ -n "${p}" ]]; then
    echo "${p}"
    return
  fi
  case "$(model_from_filename "${file}")" in
    glm5*) echo glm5 ;;
    qwen3_5*) echo qwen35 ;;
    *) echo qwen ;;
  esac
}

parallel_dim() {
  local spec="$1" dim="$2"
  if [[ "${spec}" =~ ${dim}([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo 1
  fi
}

simple_world_size() {
  local spec="$1" d p t e c
  d="$(parallel_dim "${spec}" d)"
  p="$(parallel_dim "${spec}" p)"
  t="$(parallel_dim "${spec}" t)"
  e="$(parallel_dim "${spec}" e)"
  c="$(parallel_dim "${spec}" c)"
  echo $((d * p * t * e * c))
}

fsdp_adapter_state() {
  local model="$1"
  local engine="${AREAL_ROOT_RESOLVED}/areal/engine/fsdp_engine.py"
  case "${model}" in
    qwen3_30b_a3b_4layers)
      if [[ ! -f "${engine}" ]]; then
        echo "requires-adapter"
      elif grep -q 'def _iter_rollout_weight_tensors' "${engine}"; then
        echo "adapter-present"
      else
        echo "needs-patch"
      fi
      ;;
    glm5_4layers)
      if [[ ! -f "${engine}" ]]; then
        echo "requires-adapter"
      elif grep -q 'def _iter_rollout_weight_tensors' "${engine}" && \
           grep -q 'model_type == "glm_moe_dsa"' "${engine}"; then
        echo "adapter-present"
      else
        echo "needs-patch"
      fi
      ;;
    *) echo "not-required" ;;
  esac
}

fsdp_audit_one() {
  local model="$1" file actor train_batch valid_batch dp nodes gpus rollout total_gpu actor_world rollout_world adapter
  file="$(script_for "${model}" fsdp)"
  [[ -f "${file}" ]] || return 1

  actor="$(extract_default "${file}" ACTOR_BACKEND)"
  rollout="$(extract_default "${file}" ROLLOUT_BACKEND)"
  train_batch="$(extract_default "${file}" TRAIN_BATCH_SIZE)"
  valid_batch="$(extract_default "${file}" VALID_BATCH_SIZE)"
  nodes="$(extract_default "${file}" N_NODES)"
  gpus="$(extract_default "${file}" N_GPUS_PER_NODE)"
  dp="$(parallel_dim "${actor}" d)"
  actor_world="$(simple_world_size "${actor}")"
  rollout_world="$(simple_world_size "${rollout}")"
  total_gpu=$((nodes * gpus))
  adapter="$(fsdp_adapter_state "${model}")"

  local status="OK" notes=()
  if ! bash -n "${file}" >/dev/null 2>&1; then
    status="ERROR"
    notes+=("shell-syntax")
  fi
  if [[ "${actor}" != fsdp:* ]]; then
    status="ERROR"
    notes+=("actor-not-fsdp:${actor}")
  fi
  if [[ ! "${train_batch}" =~ ^[1-9][0-9]*$ ]]; then
    status="ERROR"
    notes+=("invalid-train-batch:${train_batch}")
  elif (( train_batch < dp || train_batch % dp != 0 )); then
    status="ERROR"
    notes+=("train_batch=${train_batch},dp=${dp}")
  fi
  if [[ "${valid_batch}" =~ ^[1-9][0-9]*$ ]] && (( valid_batch % dp != 0 )); then
    [[ "${status}" == "OK" ]] && status="WARN"
    notes+=("valid_batch_not_multiple_of_dp")
  fi
  if (( actor_world + rollout_world > total_gpu )); then
    status="ERROR"
    notes+=("gpu_budget=$((actor_world + rollout_world))>${total_gpu}")
  fi
  if [[ "${adapter}" == "needs-patch" || "${adapter}" == "requires-adapter" ]]; then
    [[ "${status}" == "OK" ]] && status="PATCH"
    notes+=("${adapter}")
  fi

  printf '%-30s %-6s dp=%-2s batch=%-3s actor=%-16s rollout=%-16s nodes=%sx%s' \
    "${model}" "${status}" "${dp}" "${train_batch}" "${actor}" "${rollout}" "${nodes}" "${gpus}"
  if ((${#notes[@]})); then
    printf '  [%s]' "$(IFS=,; echo "${notes[*]}")"
  fi
  echo

  [[ "${status}" != "ERROR" ]]
}

backend_status() {
  local model="$1" backend="$2" file adapter
  file="$(script_for "${model}" "${backend}")"
  [[ -f "${file}" ]] || { echo unsupported; return; }
  if ! bash -n "${file}" >/dev/null 2>&1; then
    echo invalid-shell
    return
  fi
  if [[ "${backend}" == fsdp ]]; then
    adapter="$(fsdp_adapter_state "${model}")"
    case "${adapter}" in
      needs-patch|requires-adapter) echo "${adapter}"; return ;;
    esac
    local actor batch dp
    actor="$(extract_default "${file}" ACTOR_BACKEND)"
    batch="$(extract_default "${file}" TRAIN_BATCH_SIZE)"
    dp="$(parallel_dim "${actor}" d)"
    if [[ ! "${batch}" =~ ^[1-9][0-9]*$ ]] || (( batch < dp || batch % dp != 0 )); then
      echo invalid-batch
      return
    fi
  fi
  echo configured
}

list_models() {
  local model backends=() b
  printf '%-32s %s\n' "MODEL" "ACTOR BACKENDS"
  printf '%-32s %s\n' "--------------------------------" "-------------------------"
  while IFS= read -r model; do
    [[ -n "${model}" ]] || continue
    backends=()
    while IFS= read -r b; do
      [[ -n "${b}" ]] && backends+=("${b}")
    done < <(backends_for_model "${model}")
    printf '%-32s %s\n' "${model}" "$(IFS=,; echo "${backends[*]}")"
  done < <(discover_models)
}

show_backends() {
  local model="$1" b
  model_exists "${model}" || { echo "[ERROR] Unknown model: ${model}" >&2; return 2; }
  echo "Model: ${model}"
  while IFS= read -r b; do
    [[ -n "${b}" ]] || continue
    printf '  %-10s status=%s script=%s\n' "${b}" "$(backend_status "${model}" "${b}")" "$(basename "$(script_for "${model}" "${b}")")"
  done < <(backends_for_model "${model}")
}

show_info_one() {
  local model="$1" backend="$2" file
  file="$(script_for "${model}" "${backend}")"
  [[ -f "${file}" ]] || { echo "[ERROR] ${model} does not support backend ${backend} in this package." >&2; return 2; }
  echo "============================================================"
  echo "Model key:        ${model}"
  echo "Backend:          ${backend}"
  echo "Status:           $(backend_status "${model}" "${backend}")"
  echo "Script:           ${file}"
  echo "Profile:          $(extract_profile "${file}")"
  echo "Model path:       $(extract_default "${file}" MODEL_PATH)"
  echo "Actor backend:    $(extract_default "${file}" ACTOR_BACKEND)"
  echo "Rollout backend:  $(extract_default "${file}" ROLLOUT_BACKEND)"
  echo "Nodes:            $(extract_default "${file}" N_NODES)"
  echo "GPUs/node:        $(extract_default "${file}" N_GPUS_PER_NODE)"
  echo "Train batch:      $(extract_default "${file}" TRAIN_BATCH_SIZE)"
  echo "Valid batch:      $(extract_default "${file}" VALID_BATCH_SIZE)"
  echo "N samples:        $(extract_default "${file}" N_SAMPLES)"
  if [[ "${backend}" == fsdp ]]; then
    echo "FSDP adapter:     $(fsdp_adapter_state "${model}")"
  fi
  echo "============================================================"
}

local_primary_ip() {
  hostname -I 2>/dev/null | awk '{print $1}'
}

is_local_ip() {
  local wanted="$1" ip
  while IFS= read -r ip; do
    [[ "${ip}" == "${wanted}" ]] && return 0
  done < <(hostname -I 2>/dev/null | tr ' ' '\n' | sed '/^$/d')
  [[ "${wanted}" == "127.0.0.1" || "${wanted}" == "localhost" ]]
}

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
MODEL=""
BACKEND=""
SEARCH=""
RAY_ADDRESS_ARG=""
WORKER_IP_ARG=""
HEAD_IP_ARG=""
PROFILE_ARG=""
NODES_ARG=""
GPUS_PER_NODE_ARG=""
DO_LIST=0
DO_BACKENDS=0
DO_INFO=0
DO_CHECK_FSDP=0
DO_DRY_RUN=0
RESTART_RAY=0
RAY_ACTION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model=*) MODEL="${1#*=}" ;;
    --backend=*) BACKEND="${1#*=}" ;;
    --search=*) SEARCH="${1#*=}" ;;
    --list) DO_LIST=1 ;;
    --backends) DO_BACKENDS=1 ;;
    --info) DO_INFO=1 ;;
    --check-fsdp) DO_CHECK_FSDP=1 ;;
    --dry-run) DO_DRY_RUN=1 ;;
    --ray-address=*) RAY_ADDRESS_ARG="${1#*=}" ;;
    --model-path=*) export MODEL_PATH="${1#*=}" ;;
    --tokenizer-path=*) export TOKENIZER_PATH="${1#*=}" ;;
    --nodes=*) NODES_ARG="${1#*=}"; export N_NODES="${NODES_ARG}" ;;
    --gpus-per-node=*) GPUS_PER_NODE_ARG="${1#*=}"; export N_GPUS_PER_NODE="${GPUS_PER_NODE_ARG}" ;;
    --no-cleanup) export CLEAN_BEFORE_TRAIN=0 ;;
    --restart-ray) RESTART_RAY=1 ;;
    --ray-head) RAY_ACTION=head ;;
    --ray-worker) RAY_ACTION=worker ;;
    --ray-status) RAY_ACTION=status ;;
    --worker-ip=*) WORKER_IP_ARG="${1#*=}" ;;
    --head-ip=*) HEAD_IP_ARG="${1#*=}" ;;
    --profile=*) PROFILE_ARG="${1#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# Backward-compatible aliases such as --model=qwen3_8b_fsdp_sglang.
if [[ "${MODEL}" == *_fsdp_sglang ]]; then
  legacy_backend=fsdp
  MODEL="${MODEL%_fsdp_sglang}"
  if [[ -n "${BACKEND}" && "${BACKEND}" != "${legacy_backend}" ]]; then
    echo "[ERROR] Legacy model alias implies fsdp but --backend=${BACKEND} was supplied." >&2
    exit 2
  fi
  BACKEND=fsdp
elif [[ "${MODEL}" == *_megatron_sglang ]]; then
  legacy_backend=megatron
  MODEL="${MODEL%_megatron_sglang}"
  if [[ -n "${BACKEND}" && "${BACKEND}" != "${legacy_backend}" ]]; then
    echo "[ERROR] Legacy model alias implies megatron but --backend=${BACKEND} was supplied." >&2
    exit 2
  fi
  BACKEND=megatron
fi

if [[ "${DO_LIST}" == 1 ]]; then
  list_models
  exit 0
fi

if [[ -n "${SEARCH}" ]]; then
  found=0
  while IFS= read -r model; do
    if [[ "${model}" == *"${SEARCH}"* ]]; then
      found=1
      mapfile -t search_backends < <(backends_for_model "${model}")
      printf '%-32s %s\n' "${model}" "$(IFS=,; echo "${search_backends[*]}")"
    fi
  done < <(discover_models)
  [[ "${found}" == 1 ]] || { echo "No model matched: ${SEARCH}"; exit 1; }
  exit 0
fi

if [[ "${DO_CHECK_FSDP}" == 1 ]]; then
  echo "===== FSDP launcher static audit ====="
  failed=0
  while IFS= read -r model; do
    [[ -f "$(script_for "${model}" fsdp)" ]] || continue
    fsdp_audit_one "${model}" || failed=1
  done < <(discover_models)
  echo
  echo "Legend: OK=configured, PATCH=core adapter required, WARN=non-fatal static warning, ERROR=launcher invalid."
  [[ "${failed}" == 0 ]] || exit 2
  exit 0
fi

if [[ -n "${MODEL}" ]] && ! model_exists "${MODEL}"; then
  echo "[ERROR] Unknown model: ${MODEL}" >&2
  echo >&2
  list_models >&2
  exit 2
fi

if [[ "${DO_BACKENDS}" == 1 ]]; then
  [[ -n "${MODEL}" ]] || { echo "[ERROR] --backends requires --model=<name>." >&2; exit 2; }
  show_backends "${MODEL}"
  exit 0
fi

if [[ "${DO_INFO}" == 1 ]]; then
  [[ -n "${MODEL}" ]] || { echo "[ERROR] --info requires --model=<name>." >&2; exit 2; }
  if [[ -n "${BACKEND}" ]]; then
    show_info_one "${MODEL}" "${BACKEND}"
  else
    while IFS= read -r b; do
      [[ -n "${b}" ]] && show_info_one "${MODEL}" "${b}"
    done < <(backends_for_model "${MODEL}")
  fi
  exit 0
fi

# Resolve backend for training or model-aware Ray operations.
if [[ -n "${MODEL}" && -z "${BACKEND}" ]]; then
  mapfile -t available_backends < <(backends_for_model "${MODEL}")
  if ((${#available_backends[@]} == 1)); then
    BACKEND="${available_backends[0]}"
  elif [[ -z "${RAY_ACTION}" ]]; then
    echo "[ERROR] Model ${MODEL} has multiple backends; pass --backend=fsdp or --backend=megatron." >&2
    show_backends "${MODEL}" >&2
    exit 2
  fi
fi

if [[ -n "${BACKEND}" && "${BACKEND}" != fsdp && "${BACKEND}" != megatron ]]; then
  echo "[ERROR] Unsupported actor backend: ${BACKEND}. Use fsdp or megatron." >&2
  exit 2
fi

SELECTED_SCRIPT=""
if [[ -n "${MODEL}" && -n "${BACKEND}" ]]; then
  SELECTED_SCRIPT="$(script_for "${MODEL}" "${BACKEND}")"
  [[ -f "${SELECTED_SCRIPT}" ]] || {
    echo "[ERROR] ${MODEL} does not have backend ${BACKEND}." >&2
    show_backends "${MODEL}" >&2
    exit 2
  }
fi

if [[ -n "${PROFILE_ARG}" ]]; then
  PROFILE="${PROFILE_ARG}"
elif [[ -n "${SELECTED_SCRIPT}" ]]; then
  PROFILE="$(extract_profile "${SELECTED_SCRIPT}")"
elif [[ -n "${MODEL}" ]]; then
  first_backend="$(backends_for_model "${MODEL}" | head -1)"
  PROFILE="$(extract_profile "$(script_for "${MODEL}" "${first_backend}")")"
else
  PROFILE=""
fi

# -----------------------------------------------------------------------------
# Ray-only actions
# -----------------------------------------------------------------------------
if [[ -n "${RAY_ACTION}" ]]; then
  [[ -n "${PROFILE}" ]] || {
    echo "[ERROR] ${RAY_ACTION} action needs --model=<name> or --profile=<qwen|qwen35|glm5|base>." >&2
    exit 2
  }
  export AREAL_ENV_PROFILE="${PROFILE}"

  if [[ -n "${RAY_ADDRESS_ARG}" ]]; then
    [[ "${RAY_ADDRESS_ARG}" == *:* ]] || { echo "[ERROR] --ray-address must be ip:port." >&2; exit 2; }
    RAY_HEAD_IP="${RAY_ADDRESS_ARG%:*}"
    export RAY_PORT="${RAY_ADDRESS_ARG##*:}"
    export RAY_ADDRESS="${RAY_ADDRESS_ARG}"
  else
    RAY_HEAD_IP="${HEAD_IP_ARG:-$(local_primary_ip)}"
    export RAY_PORT="${RAY_PORT:-6379}"
    export RAY_ADDRESS="${RAY_HEAD_IP}:${RAY_PORT}"
  fi

  default_gpus=8
  default_nodes=""
  if [[ -n "${SELECTED_SCRIPT}" ]]; then
    default_gpus="$(extract_default "${SELECTED_SCRIPT}" N_GPUS_PER_NODE)"
    default_nodes="$(extract_default "${SELECTED_SCRIPT}" N_NODES)"
  elif [[ -n "${MODEL}" ]]; then
    first_backend="$(backends_for_model "${MODEL}" | head -1)"
    tmp_script="$(script_for "${MODEL}" "${first_backend}")"
    default_gpus="$(extract_default "${tmp_script}" N_GPUS_PER_NODE)"
    default_nodes="$(extract_default "${tmp_script}" N_NODES)"
  fi
  NUM_GPUS_RESOLVED="${N_GPUS_PER_NODE:-${default_gpus:-8}}"

  case "${RAY_ACTION}" in
    head)
      if [[ -n "${HEAD_IP_ARG}" ]]; then
        RAY_HEAD_IP="${HEAD_IP_ARG}"
        export RAY_ADDRESS="${RAY_HEAD_IP}:${RAY_PORT}"
      fi
      if ! is_local_ip "${RAY_HEAD_IP}"; then
        echo "[ERROR] Ray head IP ${RAY_HEAD_IP} is not an IP of this node." >&2
        echo "        Local addresses: $(hostname -I 2>/dev/null || true)" >&2
        exit 2
      fi
      echo "===== Starting Ray head ====="
      echo "profile=${AREAL_ENV_PROFILE} address=${RAY_ADDRESS} gpus=${NUM_GPUS_RESOLVED}"
      STOP_EXISTING_RAY=1 NUM_GPUS="${NUM_GPUS_RESOLVED}" \
        bash "${EXAMPLE_ROOT}/scripts/start_ray.sh" "${RAY_HEAD_IP}"
      if [[ -n "${default_nodes}" && "${default_nodes}" -gt 1 ]]; then
        echo
        echo "[INFO] This model defaults to ${default_nodes} nodes. Start each worker with:"
        echo "  bash run.sh --ray-worker --model=${MODEL}${BACKEND:+ --backend=${BACKEND}} --ray-address=${RAY_ADDRESS} --worker-ip=<worker-ip>"
      fi
      exit 0
      ;;
    worker)
      [[ -n "${RAY_ADDRESS_ARG}" ]] || { echo "[ERROR] --ray-worker requires --ray-address=<head-ip:port>." >&2; exit 2; }
      WORKER_IP="${WORKER_IP_ARG:-$(local_primary_ip)}"
      [[ -n "${WORKER_IP}" ]] || { echo "[ERROR] Could not determine worker IP." >&2; exit 2; }
      if ! is_local_ip "${WORKER_IP}"; then
        echo "[ERROR] Worker IP ${WORKER_IP} is not an IP of this node." >&2
        echo "        Local addresses: $(hostname -I 2>/dev/null || true)" >&2
        exit 2
      fi
      echo "===== Joining Ray worker ====="
      echo "profile=${AREAL_ENV_PROFILE} head=${RAY_ADDRESS} worker=${WORKER_IP} gpus=${NUM_GPUS_RESOLVED}"
      STOP_EXISTING_RAY=1 NUM_GPUS="${NUM_GPUS_RESOLVED}" \
        bash "${EXAMPLE_ROOT}/scripts/start_ray_worker.sh" "${RAY_HEAD_IP}" "${WORKER_IP}"
      exit 0
      ;;
    status)
      [[ -n "${RAY_ADDRESS}" ]] || { echo "[ERROR] Ray address unavailable." >&2; exit 2; }
      if [[ -n "${MODEL}" && -n "${default_nodes}" ]]; then
        status_nodes="${N_NODES:-${default_nodes}}"
        status_gpus_per_node="${N_GPUS_PER_NODE:-${default_gpus}}"
        EXPECTED_NODES="${status_nodes}" \
        EXPECTED_GPUS="$((status_nodes * status_gpus_per_node))" \
        GPUS_PER_NODE="${status_gpus_per_node}" \
          bash "${EXAMPLE_ROOT}/scripts/ray_status.sh" "${RAY_HEAD_IP}"
      else
        EXPECTED_NODES="" EXPECTED_GPUS="" GPUS_PER_NODE="" \
          bash "${EXAMPLE_ROOT}/scripts/ray_status.sh" "${RAY_HEAD_IP}"
      fi
      exit 0
      ;;
  esac
fi

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
[[ -n "${MODEL}" ]] || { echo "[ERROR] Missing --model=<name>." >&2; usage >&2; exit 2; }
[[ -n "${BACKEND}" ]] || { echo "[ERROR] Missing --backend for ${MODEL}." >&2; exit 2; }

export AREAL_ENV_PROFILE="${PROFILE}"
[[ -n "${RAY_ADDRESS_ARG}" ]] && export RAY_ADDRESS="${RAY_ADDRESS_ARG}"

DEFAULT_N_NODES="$(extract_default "${SELECTED_SCRIPT}" N_NODES)"
DEFAULT_GPUS_PER_NODE="$(extract_default "${SELECTED_SCRIPT}" N_GPUS_PER_NODE)"
REQUESTED_N_NODES="${N_NODES:-${DEFAULT_N_NODES}}"
REQUESTED_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${DEFAULT_GPUS_PER_NODE}}"

if [[ "${BACKEND}" == fsdp ]]; then
  st="$(backend_status "${MODEL}" fsdp)"
  if [[ "${st}" == invalid-batch || "${st}" == invalid-shell ]]; then
    echo "[ERROR] FSDP launcher static status is ${st}. Run: bash run.sh --check-fsdp" >&2
    exit 2
  fi
  if [[ "${st}" == needs-patch || "${st}" == requires-adapter ]]; then
    echo "[ERROR] ${MODEL} FSDP requires the AReaL FSDP adapter before launch." >&2
    echo "        AREAL_ROOT=${AREAL_ROOT_RESOLVED} bash ${EXAMPLE_ROOT}/patches/apply_fsdp_qwen3moe_glm5_sglang.sh" >&2
    exit 2
  fi
fi

if [[ "${RESTART_RAY}" == 1 ]]; then
  if [[ "${REQUESTED_N_NODES}" != 1 ]]; then
    echo "[ERROR] --restart-ray is single-node only; ${MODEL}/${BACKEND} requests N_NODES=${REQUESTED_N_NODES}." >&2
    echo "        Multi-node sequence:" >&2
    echo "          1) head:   bash run.sh --ray-head --model=${MODEL} --backend=${BACKEND} --ray-address=<head-ip>:6379" >&2
    echo "          2) worker: bash run.sh --ray-worker --model=${MODEL} --backend=${BACKEND} --ray-address=<head-ip>:6379 --worker-ip=<worker-ip>" >&2
    echo "          3) train:  bash run.sh --model=${MODEL} --backend=${BACKEND} --ray-address=<head-ip>:6379" >&2
    exit 2
  fi

  if [[ -n "${RAY_ADDRESS_ARG}" ]]; then
    [[ "${RAY_ADDRESS_ARG}" == *:* ]] || { echo "[ERROR] --ray-address must be ip:port." >&2; exit 2; }
    RAY_HEAD_IP="${RAY_ADDRESS_ARG%:*}"
    export RAY_PORT="${RAY_ADDRESS_ARG##*:}"
  else
    RAY_HEAD_IP="${HEAD_IP_ARG:-$(local_primary_ip)}"
    export RAY_PORT="${RAY_PORT:-6379}"
  fi
  [[ -n "${HEAD_IP_ARG}" ]] && RAY_HEAD_IP="${HEAD_IP_ARG}"

  if ! is_local_ip "${RAY_HEAD_IP}"; then
    echo "[ERROR] --restart-ray head IP ${RAY_HEAD_IP} is not local to this node." >&2
    echo "        Local addresses: $(hostname -I 2>/dev/null || true)" >&2
    exit 2
  fi

  export RAY_ADDRESS="${RAY_HEAD_IP}:${RAY_PORT}"
  echo "===== Restarting single-node Ray for ${MODEL}/${BACKEND} ====="
  echo "profile=${AREAL_ENV_PROFILE} address=${RAY_ADDRESS} gpus=${REQUESTED_GPUS_PER_NODE}"
  STOP_EXISTING_RAY=1 NUM_GPUS="${REQUESTED_GPUS_PER_NODE}" \
    bash "${EXAMPLE_ROOT}/scripts/start_ray.sh" "${RAY_HEAD_IP}"
fi

if [[ "${DO_DRY_RUN}" == 1 ]]; then
  echo "===== Dry run ====="
  echo "model=${MODEL}"
  echo "backend=${BACKEND}"
  echo "script=${SELECTED_SCRIPT}"
  echo "profile=${AREAL_ENV_PROFILE}"
  echo "nodes=${REQUESTED_N_NODES}"
  echo "gpus_per_node=${REQUESTED_GPUS_PER_NODE}"
  echo "ray_address=${RAY_ADDRESS:-auto}"
  echo "MODEL_PATH=${MODEL_PATH:-<script-default>}"
  echo "TOKENIZER_PATH=${TOKENIZER_PATH:-<script-default>}"
  exit 0
fi

exec bash "${SELECTED_SCRIPT}"
