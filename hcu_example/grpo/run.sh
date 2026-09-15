#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

# export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
# export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'USAGE'
AReaL HCU GRPO launcher

Model/backend discovery:
  bash run.sh --list
  bash run.sh --search=qwen3
  bash run.sh --model=qwen3 --variant=dense --backend=fsdp --rollout=sglang --backends
  bash run.sh --model=qwen3 --variant=dense --backend=fsdp --rollout=sglang --info
  bash run.sh --check-fsdp

Training:
  bash run.sh --model=qwen3 --variant=dense --backend=fsdp --rollout=sglang [options]
  bash run.sh --model=qwen3_8b_fsdp_sglang [options]     # legacy alias

Single-node Ray + training:
  bash run.sh --model=qwen3 --variant=dense --backend=fsdp --rollout=sglang --restart-ray

Multi-node Ray lifecycle (run on each physical node):
  # head node
  bash run.sh --ray-head --model=qwen3 --variant=moe --backend=megatron --rollout=sglang \
    --ray-address=<head-node-ip>:6379

  # worker node
  bash run.sh --ray-worker --model=qwen3 --variant=moe --backend=megatron --rollout=sglang \
    --ray-address=<head-node-ip>:6379 --worker-ip=<worker-node-ip>

  # then launch training from the head node
  bash run.sh --model=qwen3 --variant=moe --backend=megatron --rollout=sglang \
    --ray-address=<head-node-ip>:6379

Options:
  --model=<name>          Model family, e.g. qwen3, or a legacy model key.
                          Legacy <model>_<backend>_sglang names are also accepted.
  --variant=<name>        Required for training: dense, moe, vl, or vl_moe.
  --backend=<name>        Actor backend: fsdp or megatron.
  --rollout=<name>        Rollout backend: sglang or vllm.
  --list                  List family/variant/backend/rollout launchers.
  --search=<pattern>      Search family, variant, backend, rollout, or legacy keys.
  --backends              Show available backends for the selected model.
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

normalize_family() {
  local value="${1,,}"
  value="${value//-/_}"
  value="${value//./_}"
  case "${value}" in
    qwen25) echo qwen2_5 ;;
    qwen35) echo qwen3_5 ;;
    *) echo "${value}" ;;
  esac
}

canonical_launcher_id() {
  local value="$1"
  case "${value}" in
    qwen2_5_0_5b) echo qwen2_5_dense ;;
    qwen3_1_7b|qwen3_8b) echo qwen3_dense ;;
    qwen3_vl_4b) echo qwen3_vl ;;
    qwen3_30b_a3b_4layers) echo qwen3_moe ;;
    glm5_4layers) echo glm5_moe ;;
    *) echo "${value}" ;;
  esac
}

legacy_model_key() {
  case "$1" in
    qwen2_5_dense) echo qwen2_5 ;;
    qwen3_dense) echo qwen3 ;;
    qwen3_vl) echo qwen3_vl ;;
    qwen3_moe) echo qwen3_moe ;;
    qwen3_5_dense) echo qwen3_5 ;;
    glm5_moe) echo glm5 ;;
    *) echo "$1" ;;
  esac
}

script_for() {
  local model backend rollout
  model="$(canonical_launcher_id "$1")"
  backend="$2"
  rollout="${3:-sglang}"
  if [[ "${backend}" == fsdp || "${backend}" == megatron ]] &&
     [[ "${rollout}" == sglang || "${rollout}" == vllm ]]; then
    printf '%s/run_%s_%s_%s.sh\n' "${SCRIPT_DIR}" "${model}" "${backend}" "${rollout}"
  else
    return 1
  fi
}

backend_from_filename() {
  local base
  base="$(basename "$1")"
  case "${base}" in
    run_*_fsdp_sglang.sh|run_*_fsdp_vllm.sh) echo fsdp ;;
    run_*_megatron_sglang.sh|run_*_megatron_vllm.sh) echo megatron ;;
    *) return 1 ;;
  esac
}

rollout_from_filename() {
  local base
  base="$(basename "$1")"
  case "${base}" in
    run_*_fsdp_sglang.sh|run_*_megatron_sglang.sh) echo sglang ;;
    run_*_fsdp_vllm.sh|run_*_megatron_vllm.sh) echo vllm ;;
    *) return 1 ;;
  esac
}

model_from_filename() {
  local base backend rollout
  base="$(basename "$1")"
  backend="$(backend_from_filename "$1")" || return 1
  rollout="$(rollout_from_filename "$1")" || return 1
  base="${base#run_}"
  base="${base%_${backend}_${rollout}.sh}"
  echo "${base}"
}

launcher_metadata() {
  local file="$1" key="$2"
  grep -m1 -E "^${key}=" "${file}" 2>/dev/null | cut -d= -f2- | tr -d '[:space:]' || true
}

# Launcher metadata is declarative and mandatory: run.sh reads these keys with
# grep instead of sourcing the launcher, which would execute it. A launcher
# missing a key is a bug in that launcher, not something to guess around.
require_launcher_metadata() {
  local file="$1" key="$2" value
  value="$(launcher_metadata "${file}" "${key}")"
  if [[ -z "${value}" ]]; then
    echo "[ERROR] ${file##*/} is missing required launcher metadata ${key}." >&2
    echo "        Add '${key}=<value>' near the top of the launcher." >&2
    return 2
  fi
  printf '%s\n' "${value}"
}

launcher_family() {
  local value
  value="$(require_launcher_metadata "$1" HCU_LAUNCHER_FAMILY)" || return 2
  normalize_family "${value}"
}

launcher_variant() {
  require_launcher_metadata "$1" HCU_LAUNCHER_VARIANT
}

launcher_actor_backend() {
  require_launcher_metadata "$1" HCU_LAUNCHER_ACTOR_BACKEND
}

launcher_rollout_backend() {
  require_launcher_metadata "$1" HCU_LAUNCHER_ROLLOUT_BACKEND
}

extract_default() {
  local file="$1" var="$2" line
  line="$(grep -m1 -E "^(export[[:space:]]+)?${var}=" "${file}" 2>/dev/null || true)"
  if [[ "${line}" =~ :-([^}]*)} ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  else
    printf '%s\n' ""
  fi
}

extract_profile() {
  require_launcher_metadata "$1" HCU_LAUNCHER_PROFILE
}

collect_launcher_matches() {
  MATCHED_FILES=()
  local file family variant backend rollout requested_family requested_id
  requested_family="$(normalize_family "${MODEL:-}")"
  requested_id="$(canonical_launcher_id "${MODEL:-}")"
  shopt -s nullglob
  for file in "${SCRIPT_DIR}"/run_*_fsdp_*.sh "${SCRIPT_DIR}"/run_*_megatron_*.sh; do
    family="$(launcher_family "${file}")"
    variant="$(launcher_variant "${file}")"
    backend="$(launcher_actor_backend "${file}")"
    rollout="$(launcher_rollout_backend "${file}")"
    [[ -n "${MODEL:-}" && "${requested_id}" == "$(model_from_filename "${file}")" ]] ||
      [[ -z "${MODEL:-}" ]] || [[ "${requested_family}" == "${family}" ]] || continue
    [[ -z "${VARIANT_ARG:-}" || "${VARIANT_ARG}" == "${variant}" ]] || continue
    [[ -z "${BACKEND}" || "${BACKEND}" == "${backend}" ]] || continue
    [[ -z "${ROLLOUT_ARG:-}" || "${ROLLOUT_ARG}" == "${rollout}" ]] || continue
    MATCHED_FILES+=("${file}")
  done
  shopt -u nullglob
}

resolve_launcher() {
  collect_launcher_matches
  if ((${#MATCHED_FILES[@]} == 0)); then
    echo "[ERROR] No launcher matches model=${MODEL:-<auto>} variant=${VARIANT_ARG:-<auto>} backend=${BACKEND:-<auto>} rollout=${ROLLOUT_ARG:-<auto>}." >&2
    echo "        Use: bash run.sh --list" >&2
    return 2
  fi
  if ((${#MATCHED_FILES[@]} > 1)); then
    echo "[ERROR] Launcher selection is ambiguous:" >&2
    printf '        %s\n' "${MATCHED_FILES[@]##*/}" >&2
    echo "        Add --variant, --backend, or --rollout." >&2
    return 2
  fi
  SELECTED_SCRIPT="${MATCHED_FILES[0]}"
  MODEL="$(model_from_filename "${SELECTED_SCRIPT}")"
  BACKEND="$(launcher_actor_backend "${SELECTED_SCRIPT}")"
  ROLLOUT_ARG="$(launcher_rollout_backend "${SELECTED_SCRIPT}")"
  RESOLVED_FAMILY="$(launcher_family "${SELECTED_SCRIPT}")"
  RESOLVED_VARIANT="$(launcher_variant "${SELECTED_SCRIPT}")"
  RESOLVED_MODEL_KEY="$(legacy_model_key "${MODEL}")"
  PROFILE="$(extract_profile "${SELECTED_SCRIPT}")"
  if [[ -n "${PROFILE_ARG:-}" && "${PROFILE_ARG}" != "${PROFILE}" ]]; then
    echo "[WARN] --profile=${PROFILE_ARG} overrides launcher profile=${PROFILE}." >&2
    PROFILE="${PROFILE_ARG}"
  fi
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

fsdp_audit_one() {
  local file="$1" model actor train_batch valid_batch dp nodes gpus rollout total_gpu actor_world rollout_world
  model="$(model_from_filename "${file}")"

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

  printf '%-30s %-6s dp=%-2s batch=%-3s actor=%-16s rollout=%-16s nodes=%sx%s' \
    "${model}_$(launcher_rollout_backend "${file}")" \
    "${status}" "${dp}" "${train_batch}" "${actor}" "${rollout}" "${nodes}" "${gpus}"
  if ((${#notes[@]})); then
    printf '  [%s]' "$(IFS=,; echo "${notes[*]}")"
  fi
  echo

  [[ "${status}" != "ERROR" ]]
}

backend_status() {
  local model="$1" backend="$2" rollout="${3:-sglang}" file
  file="$(script_for "${model}" "${backend}" "${rollout}")"
  [[ -f "${file}" ]] || { echo unsupported; return; }
  if ! bash -n "${file}" >/dev/null 2>&1; then
    echo invalid-shell
    return
  fi
  if [[ "${backend}" == fsdp ]]; then
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
  local model file family variant actor rollout legacy
  printf '%-12s %-8s %-10s %-10s %-24s %s\n' \
    "FAMILY" "VARIANT" "ACTOR" "ROLLOUT" "MODEL" "LEGACY KEY"
  printf '%-12s %-8s %-10s %-10s %-24s %s\n' \
    "------------" "--------" "----------" "----------" "------------------------" "----------"
  # Iterate launcher files, not model keys: a family/variant with both an FSDP
  # and a Megatron launcher must show one row per launcher.
  shopt -s nullglob
  for file in "${SCRIPT_DIR}"/run_*_fsdp_*.sh "${SCRIPT_DIR}"/run_*_megatron_*.sh; do
    model="$(model_from_filename "${file}")" || continue
    family="$(launcher_family "${file}")"
    variant="$(launcher_variant "${file}")"
    actor="$(launcher_actor_backend "${file}")"
    rollout="$(launcher_rollout_backend "${file}")"
    legacy="$(legacy_model_key "${model}")"
    if [[ -n "${SEARCH:-}" &&
          "${family} ${variant} ${actor} ${rollout} ${model} ${legacy}" != *"${SEARCH,,}"* ]]; then
      continue
    fi
    printf '%-12s %-8s %-10s %-10s %-24s %s\n' \
      "${family}" "${variant}" "${actor}" "${rollout}" "${model}" "${legacy}"
  done | sort
  shopt -u nullglob
}

show_backends() {
  local requested="$1" file
  MODEL="${requested}"
  collect_launcher_matches
  ((${#MATCHED_FILES[@]})) || {
    echo "[ERROR] Unknown model or selector: ${requested}" >&2
    return 2
  }
  echo "Model selector: ${requested}"
  for file in "${MATCHED_FILES[@]}"; do
    printf '  actor=%-10s rollout=%-10s variant=%-8s status=%s script=%s\n' \
      "$(launcher_actor_backend "${file}")" "$(launcher_rollout_backend "${file}")" \
      "$(launcher_variant "${file}")" \
      "$(backend_status "$(model_from_filename "${file}")" "$(backend_from_filename "${file}")" "$(launcher_rollout_backend "${file}")")" \
      "$(basename "${file}")"
  done
}

show_info_file() {
  local file="$1" model backend
  model="$(model_from_filename "${file}")"
  backend="$(backend_from_filename "${file}")"
  echo "============================================================"
  echo "Model key:        $(legacy_model_key "${model}")"
  echo "Family:           $(launcher_family "${file}")"
  echo "Variant:          $(launcher_variant "${file}")"
  echo "Actor backend:    $(launcher_actor_backend "${file}")"
  echo "Rollout backend:  $(launcher_rollout_backend "${file}")"
  echo "Status:           $(backend_status "${model}" "${backend}" "$(launcher_rollout_backend "${file}")")"
  echo "Script:           ${file}"
  echo "Profile:          $(extract_profile "${file}")"
  echo "Model path:       $(extract_default "${file}" MODEL_PATH)"
  echo "Actor spec:       $(extract_default "${file}" ACTOR_BACKEND)"
  echo "Rollout spec:     $(extract_default "${file}" ROLLOUT_BACKEND)"
  echo "Nodes:            $(extract_default "${file}" N_NODES)"
  echo "GPUs/node:        $(extract_default "${file}" N_GPUS_PER_NODE)"
  echo "Train batch:      $(extract_default "${file}" TRAIN_BATCH_SIZE)"
  echo "Valid batch:      $(extract_default "${file}" VALID_BATCH_SIZE)"
  echo "N samples:        $(extract_default "${file}" N_SAMPLES)"
  echo "============================================================"
}

show_info_selected() {
  collect_launcher_matches
  ((${#MATCHED_FILES[@]})) || {
    echo "[ERROR] Unknown model or selector: ${MODEL}" >&2
    return 2
  }
  for file in "${MATCHED_FILES[@]}"; do
    show_info_file "${file}"
  done
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
VARIANT_ARG=""
ROLLOUT_ARG=""
SELECTED_SCRIPT=""
RESOLVED_FAMILY=""
RESOLVED_VARIANT=""
RESOLVED_MODEL_KEY=""
PROFILE=""
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
    --variant=*) VARIANT_ARG="${1#*=}" ;;
    --rollout=*) ROLLOUT_ARG="${1#*=}" ;;
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
case "${MODEL}" in
  *_fsdp_sglang|*_fsdp_vllm|*_megatron_sglang|*_megatron_vllm)
    alias_name="${MODEL}"
    MODEL="${alias_name%_*_*}"
    alias_rollout="${alias_name##*_}"
    alias_backend="${alias_name%_${alias_rollout}}"
    alias_backend="${alias_backend##*_}"
    if [[ -n "${BACKEND}" && "${BACKEND}" != "${alias_backend}" ]]; then
      echo "[ERROR] Legacy model alias implies ${alias_backend} but --backend=${BACKEND} was supplied." >&2
      exit 2
    fi
    if [[ -n "${ROLLOUT_ARG}" && "${ROLLOUT_ARG}" != "${alias_rollout}" ]]; then
      echo "[ERROR] Legacy model alias implies ${alias_rollout} but --rollout=${ROLLOUT_ARG} was supplied." >&2
      exit 2
    fi
    BACKEND="${alias_backend}"
    ROLLOUT_ARG="${alias_rollout}"
    ;;
esac

if [[ "${DO_LIST}" == 1 ]]; then
  list_models
  exit 0
fi

if [[ -n "${SEARCH}" ]]; then
  # list_models already applies SEARCH; count data rows rather than re-grepping
  # the rendered table, whose header would match generic patterns.
  search_output="$(list_models)"
  printf '%s\n' "${search_output}"
  if (( $(printf '%s\n' "${search_output}" | tail -n +3 | grep -c .) == 0 )); then
    echo "No model matched: ${SEARCH}"
    exit 1
  fi
  exit 0
fi

if [[ "${DO_CHECK_FSDP}" == 1 ]]; then
  echo "===== FSDP launcher static audit ====="
  failed=0
  shopt -s nullglob
  for file in "${SCRIPT_DIR}"/run_*_fsdp_*.sh; do
    fsdp_audit_one "${file}" || failed=1
  done
  shopt -u nullglob
  echo
  echo "Legend: OK=configured, WARN=non-fatal static warning, ERROR=launcher invalid."
  [[ "${failed}" == 0 ]] || exit 2
  exit 0
fi

# Validate selector values before resolution so a typo reports the bad value
# instead of a generic "no launcher matches".
if [[ -n "${VARIANT_ARG}" && "${VARIANT_ARG}" != dense && "${VARIANT_ARG}" != moe &&
      "${VARIANT_ARG}" != vl && "${VARIANT_ARG}" != vl_moe ]]; then
  echo "[ERROR] Unsupported variant: ${VARIANT_ARG}. Use dense, moe, vl, or vl_moe." >&2
  exit 2
fi
if [[ -n "${BACKEND}" && "${BACKEND}" != fsdp && "${BACKEND}" != megatron ]]; then
  echo "[ERROR] Unsupported actor backend: ${BACKEND}. Use fsdp or megatron." >&2
  exit 2
fi
if [[ -n "${ROLLOUT_ARG}" && "${ROLLOUT_ARG}" != sglang && "${ROLLOUT_ARG}" != vllm ]]; then
  echo "[ERROR] Unsupported rollout backend: ${ROLLOUT_ARG}. Use sglang or vllm." >&2
  exit 2
fi

if [[ "${DO_BACKENDS}" == 1 ]]; then
  [[ -n "${MODEL}" ]] || { echo "[ERROR] --backends requires --model=<family-or-key>." >&2; exit 2; }
  show_backends "${MODEL}"
  exit $?
fi

if [[ "${DO_INFO}" == 1 ]]; then
  [[ -n "${MODEL}" ]] || { echo "[ERROR] --info requires --model=<family-or-key>." >&2; exit 2; }
  show_info_selected
  exit $?
fi

# Resolve exactly one launcher before training or model-aware Ray operations.
# Selection is explicit only: family, variant, actor backend and rollout backend
# all come from the command line or a legacy model key.  Model weights are never
# inspected, so pointing a launcher at the wrong checkpoint fails at load time
# rather than being silently rerouted here.
if [[ -n "${MODEL}" ]]; then
  [[ -n "${VARIANT_ARG}" ]] || {
    echo "[ERROR] --variant is required. Use dense, moe, vl, or vl_moe." >&2
    echo "        Run: bash run.sh --list" >&2
    exit 2
  }
  resolve_launcher || exit $?
fi


# resolve_launcher() sets PROFILE from launcher metadata when --model is given;
# --profile only applies to model-less Ray actions.
if [[ -z "${PROFILE}" ]]; then
  PROFILE="${PROFILE_ARG}"
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
      echo "  bash run.sh --ray-worker --model=${RESOLVED_FAMILY} --variant=${RESOLVED_VARIANT} --backend=${BACKEND} --rollout=${ROLLOUT_ARG} --ray-address=${RAY_ADDRESS} --worker-ip=<worker-ip>"
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
  st="$(backend_status "${MODEL}" fsdp "${ROLLOUT_ARG:-sglang}")"
  if [[ "${st}" == invalid-batch || "${st}" == invalid-shell ]]; then
    echo "[ERROR] FSDP launcher static status is ${st}. Run: bash run.sh --check-fsdp" >&2
    exit 2
  fi
fi

if [[ "${RESTART_RAY}" == 1 && "${DO_DRY_RUN}" != 1 ]]; then
  if [[ "${REQUESTED_N_NODES}" != 1 ]]; then
    echo "[ERROR] --restart-ray is single-node only; ${MODEL}/${BACKEND} requests N_NODES=${REQUESTED_N_NODES}." >&2
    echo "        Multi-node sequence:" >&2
    echo "          1) head:   bash run.sh --ray-head --model=${RESOLVED_FAMILY} --variant=${RESOLVED_VARIANT} --backend=${BACKEND} --rollout=${ROLLOUT_ARG} --ray-address=<head-ip>:6379" >&2
    echo "          2) worker: bash run.sh --ray-worker --model=${RESOLVED_FAMILY} --variant=${RESOLVED_VARIANT} --backend=${BACKEND} --rollout=${ROLLOUT_ARG} --ray-address=<head-ip>:6379 --worker-ip=<worker-ip>" >&2
    echo "          3) train:  bash run.sh --model=${RESOLVED_FAMILY} --variant=${RESOLVED_VARIANT} --backend=${BACKEND} --rollout=${ROLLOUT_ARG} --ray-address=<head-ip>:6379" >&2
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
  VALIDATE_RAY_WORKER_ENV=0 STOP_EXISTING_RAY=1 NUM_GPUS="${REQUESTED_GPUS_PER_NODE}" \
    bash "${EXAMPLE_ROOT}/scripts/start_ray.sh" "${RAY_HEAD_IP}"
fi

if [[ "${DO_DRY_RUN}" == 1 ]]; then
  echo "===== Dry run ====="
  echo "model=${RESOLVED_MODEL_KEY:-${MODEL}}"
  echo "family=${RESOLVED_FAMILY:-<auto>}"
  echo "variant=${RESOLVED_VARIANT:-<auto>}"
  echo "backend=${BACKEND}"
  echo "rollout=${ROLLOUT_ARG:-<auto>}"
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
