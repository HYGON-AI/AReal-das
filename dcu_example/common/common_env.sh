#!/usr/bin/env bash

# Shared runtime environment for AReaL DCU examples.
# Source this file BEFORE starting Ray so raylet/worker processes inherit the
# same Python, SGLang and HCU/DCU environment as the training driver.

AREAL_COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export AREAL_EXAMPLES_ROOT="${AREAL_EXAMPLES_ROOT:-$(cd "${AREAL_COMMON_DIR}/.." && pwd)}"

# -----------------------------------------------------------------------------
# Source / Python path resolution
# -----------------------------------------------------------------------------
# Override policy:
#
#   1. Values exported by the caller always win.
#   2. Alias variables are used when the canonical variable is not exported.
#   3. The original DCU paths are used only as defaults.
#
# Examples:
#
#   export AREAL_HOME=/data/AReaL-1.0.4
#   export VENV=/data/venvs/areal
#   export MEGATRON_HOME=/data/dcu_megatron
#   export SGLANG_ROOT=/data/sglang
#   source dcu_example/common/common_env.sh
#
# SGLANG_HOME means the Python source directory (normally <SGLANG_ROOT>/python).
# MEGATRON_HOME means the directory containing Megatron-LM and Megatron-Bridge.

if [[ -n "${AREAL_HOME:-}" ]]; then
  :
elif [[ -n "${AREAL_ROOT:-}" ]]; then
  export AREAL_HOME="${AREAL_ROOT}"
else
  export AREAL_HOME="$(cd "${AREAL_EXAMPLES_ROOT}/.." && pwd)"
fi
export AREAL_ROOT="${AREAL_ROOT:-${AREAL_HOME}}"
export BASE_DIR="${BASE_DIR:-$(cd "${AREAL_HOME}/.." && pwd)}"

# VENV is the canonical variable. VENV_PATH is accepted as an alias so either
# of the following works before sourcing this file:
#
#   export VENV=<VENV_PATH>
#   export VENV_PATH=<VENV_PATH>
if [[ -n "${VENV:-}" ]]; then
  :
elif [[ -n "${VENV_PATH:-}" ]]; then
  export VENV="${VENV_PATH}"
else
  export VENV="/opt/areal-venv-py31115"
fi
export VENV_PATH="${VENV}"
export PYTHON_BIN="${PYTHON_BIN:-${VENV}/bin/python}"

# MEGATRON_HOME is canonical; MEGATRON_ROOT is accepted as an alias.
if [[ -n "${MEGATRON_HOME:-}" ]]; then
  :
elif [[ -n "${MEGATRON_ROOT:-}" ]]; then
  export MEGATRON_HOME="${MEGATRON_ROOT}"
else
  export MEGATRON_HOME="/workspace/dcu_megatron"
fi
export MEGATRON_ROOT="${MEGATRON_HOME}"

# SGLANG_HOME is canonical for the Python source tree.  If only SGLANG_ROOT is
# exported, SGLANG_HOME is derived as <SGLANG_ROOT>/python.  If SGLANG_HOME is
# exported directly, SGLANG_ROOT is derived from its parent directory.
if [[ -n "${SGLANG_HOME:-}" ]]; then
  export SGLANG_ROOT="${SGLANG_ROOT:-$(dirname -- "${SGLANG_HOME}")}"
else
  export SGLANG_ROOT="${SGLANG_ROOT:-/workspace/sglang}"
  export SGLANG_HOME="${SGLANG_ROOT}/python"
fi

# DTK environment can also be moved/overridden without editing this file.
export DTK_ENV="${DTK_ENV:-/opt/dtk/env.sh}"
if [[ -f "${DTK_ENV}" ]]; then
  # shellcheck disable=SC1091
  source "${DTK_ENV}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Required Python not found: ${PYTHON_BIN}" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -d "${MEGATRON_HOME}/Megatron-LM" ]]; then
  echo "[ERROR] Megatron-LM not found: ${MEGATRON_HOME}/Megatron-LM" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -d "${SGLANG_HOME}/sglang" ]]; then
  echo "[ERROR] SGLang Python source not found: ${SGLANG_HOME}/sglang" >&2
  return 1 2>/dev/null || exit 1
fi

# Activate the resolved virtualenv before Ray starts.  Because VENV can be
# exported before sourcing this file, Ray head/worker and the training driver
# all use the same caller-selected Python environment.
# shellcheck disable=SC1090
source "${VENV}/bin/activate"

# Rebuild PYTHONPATH from the resolved source variables.  We intentionally do
# not append an inherited PYTHONPATH here: stale SGLang/Megatron/AReaL trees are
# a frequent source of mixed installations inside Ray workers.  If an extra
# source directory is intentionally required, put it in AREAL_EXTRA_PYTHONPATH.
export PATH="${VENV}/bin:${PATH}"
_AREAL_PYTHONPATH_PARTS=(
  "${MEGATRON_HOME}"
  "${AREAL_HOME}"
  "${MEGATRON_HOME}/Megatron-Bridge/src"
  "${MEGATRON_HOME}/Megatron-LM"
  "${SGLANG_HOME}"
)
if [[ -n "${AREAL_EXTRA_PYTHONPATH:-}" ]]; then
  _AREAL_PYTHONPATH_PARTS+=("${AREAL_EXTRA_PYTHONPATH}")
fi
export PYTHONPATH="$(IFS=:; echo "${_AREAL_PYTHONPATH_PARTS[*]}")"
unset _AREAL_PYTHONPATH_PARTS

# Ray-managed examples must not inherit a global device mask. Local scheduler
# examples set AREAL_RAY_MANAGED_DEVICES=0 before sourcing this file.
if [[ "${AREAL_RAY_MANAGED_DEVICES:-1}" == "1" ]]; then
  unset CUDA_VISIBLE_DEVICES || true
  unset HIP_VISIBLE_DEVICES || true
  unset ROCR_VISIBLE_DEVICES || true
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# Some DCU/PyTorch builds still read the legacy CUDA-named alias on ROCm/HIP.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_ALLOC_CONF}}"

# -----------------------------------------------------------------------------
# Runtime / cache / logs
# -----------------------------------------------------------------------------
export TMPDIR="${TMPDIR:-/dev/shm}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"

# Keep run outputs, transient coordination files, and compiler caches separate.
export AREAL_RUNS_ROOT="${AREAL_RUNS_ROOT:-/workspace/areal_runs}"
export AREAL_RUNTIME_ROOT="${AREAL_RUNTIME_ROOT:-/workspace/areal_runtime}"
export AREAL_CACHE_ROOT="${AREAL_CACHE_ROOT:-/workspace/areal_cache}"

export RUN_ROOT="${RUN_ROOT:-${AREAL_RUNTIME_ROOT}}"
export FILER_ROOT="${FILER_ROOT:-${AREAL_RUNTIME_ROOT}/experiments}"
export FILERoot="${FILERoot:-${FILER_ROOT}}"
export NAME_RESOLVE_ROOT="${NAME_RESOLVE_ROOT:-${AREAL_RUNTIME_ROOT}/name_resolve}"
export LOG_ROOT="${LOG_ROOT:-${AREAL_RUNS_ROOT}}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${AREAL_CACHE_ROOT}/xdg}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${AREAL_CACHE_ROOT}/torchinductor_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${AREAL_CACHE_ROOT}/triton_cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${AREAL_CACHE_ROOT}/torch_extensions}"

# -----------------------------------------------------------------------------
# Ray
# -----------------------------------------------------------------------------
export RAY_PORT="${RAY_PORT:-6379}"
export RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-20000}"
export RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-20299}"
export RAY_OBJECT_MANAGER_PORT="${RAY_OBJECT_MANAGER_PORT:-8076}"
export RAY_NODE_MANAGER_PORT="${RAY_NODE_MANAGER_PORT:-8077}"
export RAY_TMPDIR="${RAY_TMPDIR:-/dev/shm/ray_tmp_areal}"

case "${RAY_TMPDIR}" in
  /home/*|/workspace/*)
    echo "[WARN] RAY_TMPDIR=${RAY_TMPDIR} looks shared. Prefer a node-local path such as /dev/shm/ray_tmp_areal." >&2
    ;;
esac

# -----------------------------------------------------------------------------
# DCU / SGLang common environment
# -----------------------------------------------------------------------------
# These are common to the two known-working Qwen launchers. Model-specific
# SGLang CLI options (context length, chunked prefill size, attention backend,
# NSA/MLA backend, etc.) intentionally stay in each run script.
export USE_DCU_CUSTOM_ALLREDUCE="${USE_DCU_CUSTOM_ALLREDUCE:-1}"
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD="${SGL_CHUNKED_PREFIX_CACHE_THRESHOLD:-0}"
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD="${SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD:-0}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-1200}"
export GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=0x40000}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export HIP_KERNEL_BATCH_CEILING="${HIP_KERNEL_BATCH_CEILING:-100}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-3}"

export SGLANG_KVALLOC_KERNEL="${SGLANG_KVALLOC_KERNEL:-1}"
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS="${SGLANG_ASSIGN_EXTEND_CACHE_LOCS:-1}"
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL="${SGLANG_ASSIGN_REQ_TO_TOKEN_POOL:-1}"
export SGLANG_GET_LAST_LOC="${SGLANG_GET_LAST_LOC:-1}"
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON="${SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON:-1}"
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES="${SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES:-1}"

export HIP_H2D_DISABLE_COPY_BUFFER="${HIP_H2D_DISABLE_COPY_BUFFER:-0}"
export HIP_D2H_DISABLE_COPY_BUFFER="${HIP_D2H_DISABLE_COPY_BUFFER:-0}"
export HIP_H2D_DIRECT_COPY_THRESHOLD="${HIP_H2D_DIRECT_COPY_THRESHOLD:-32768}"
export HIP_H2D_HSAAPI_COPY_THRESHOLD="${HIP_H2D_HSAAPI_COPY_THRESHOLD:-32768}"
export HIP_D2H_DIRECT_COPY_THRESHOLD="${HIP_D2H_DIRECT_COPY_THRESHOLD:-512}"
export HIP_D2H_HSAAPI_COPY_THRESHOLD="${HIP_D2H_HSAAPI_COPY_THRESHOLD:-512}"

# -----------------------------------------------------------------------------
# Model family profile
# -----------------------------------------------------------------------------
AREAL_ENV_PROFILE="${AREAL_ENV_PROFILE:-qwen}"
export AREAL_ENV_PROFILE

case "${AREAL_ENV_PROFILE}" in
  base)
    ;;
  qwen)
    # Required by the known-working Qwen3 SGLang launchers in this package.
    export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
    export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO="${SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO:-1}"
    export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
    export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
    ;;
  qwen35)
    # Qwen3.5 uses the same SGLang speculative-v2 environment as Qwen3, but
    # the current DCU Megatron GatedDeltaNet path must not enter torch.compile.
    export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
    export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO="${SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO:-1}"
    export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
    export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
    ;;
  glm5)
    # GLM5/MLA path is intentionally different from Qwen's speculative-v2 path.
    unset SGLANG_ENABLE_SPEC_V2 || true
    unset SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO || true
    export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
    export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
    export HF_HOME="${HF_HOME:-${BASE_DIR}/hf_cache}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
    export HSA_KERNARG_POOL_SIZE="${HSA_KERNARG_POOL_SIZE:-8388608}"
    export ROC_AQL_QUEUE_SIZE="${ROC_AQL_QUEUE_SIZE:-131072}"
    export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"
    export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-8}"
    export ALLREDUCE_STREAM_WITH_COMPUTE="${ALLREDUCE_STREAM_WITH_COMPUTE:-1}"
    ;;
  *)
    echo "[ERROR] Unknown AREAL_ENV_PROFILE=${AREAL_ENV_PROFILE}. Use base, qwen, qwen35, or glm5." >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

mkdir -p \
  "${TMPDIR}" \
  "${RAY_TMPDIR}" \
  "${AREAL_RUNS_ROOT}" \
  "${FILER_ROOT}" \
  "${NAME_RESOLVE_ROOT}" \
  "${XDG_CACHE_HOME}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}"

if [[ "${AREAL_ENV_PROFILE}" == "glm5" ]]; then
  mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"
fi

# Common preflight validation. This is intentionally called by Ray start and
# training launchers rather than during every `source common_env.sh`, so helper
# scripts remain cheap to source.
areal_preflight_common() {
  local failed=0

  echo "===== AReaL DCU preflight ====="

  for dir in \
    "${AREAL_HOME}" \
    "${MEGATRON_HOME}/Megatron-LM" \
    "${MEGATRON_HOME}/Megatron-Bridge/src" \
    "${SGLANG_HOME}/sglang" \
    "${VENV}"
  do
    if [[ ! -e "${dir}" ]]; then
      echo "[ERROR] Required path does not exist: ${dir}" >&2
      failed=1
    fi
  done

  for cmd in ray hostname awk sed grep; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      echo "[ERROR] Required command not found in PATH: ${cmd}" >&2
      failed=1
    fi
  done

  if [[ ! -e /dev/kfd ]]; then
    echo "[ERROR] /dev/kfd is not visible. Start the container with --device=/dev/kfd." >&2
    failed=1
  fi
  if [[ ! -e /dev/dri ]]; then
    echo "[ERROR] /dev/dri is not visible. Start the container with --device=/dev/dri." >&2
    failed=1
  fi

  for dir in "${TMPDIR}" "${RAY_TMPDIR}" "${AREAL_RUNS_ROOT}" "${AREAL_RUNTIME_ROOT}" "${AREAL_CACHE_ROOT}"; do
    mkdir -p "${dir}" 2>/dev/null || true
    if [[ ! -d "${dir}" || ! -w "${dir}" ]]; then
      echo "[ERROR] Runtime path is not writable: ${dir}" >&2
      failed=1
    fi
  done

  "${PYTHON_BIN}" - <<'PY_INNER' || failed=1
import sys
try:
    import torch
except Exception as exc:
    print(f"[ERROR] torch import failed: {exc!r}", file=sys.stderr)
    raise SystemExit(1)

print(f"[INFO] torch={torch.__version__}, hip={getattr(torch.version, 'hip', None)}")
if getattr(torch.version, "hip", None) is None:
    print("[ERROR] This example expects a ROCm/HIP PyTorch build.", file=sys.stderr)
    raise SystemExit(1)
count = torch.cuda.device_count()
print(f"[INFO] visible DCU/GPU count before Ray: {count}")
if count <= 0:
    print("[ERROR] PyTorch cannot see any DCU/GPU device.", file=sys.stderr)
    raise SystemExit(1)
PY_INNER

  if [[ "${failed}" != "0" ]]; then
    echo "[ERROR] AReaL DCU preflight failed." >&2
    return 2
  fi

  echo "[OK] AReaL DCU preflight passed."
}

# Lightweight validation helper. It verifies that driver and Ray daemons can
# resolve the intended virtualenv and source trees without importing SGLang's
# heavy runtime/plugin stack.
areal_print_python_env() {
  echo "VENV:            ${VENV}"
  echo "Python:          ${PYTHON_BIN}"
  echo "AReaL:           ${AREAL_HOME}"
  echo "Megatron:        ${MEGATRON_HOME}"
  echo "SGLang source:   ${SGLANG_HOME}"
  echo "Ray:             $(command -v ray 2>/dev/null || echo not-found)"
  "${PYTHON_BIN}" - <<'PY'
import os, sys
print("sys.executable:  ", sys.executable)
print("PYTHONPATH:      ", os.environ.get("PYTHONPATH", ""))
try:
    import torch
    print("torch:           ", torch.__version__, "hip=", torch.version.hip)
except Exception as exc:
    print("torch import ERROR:", repr(exc))
try:
    import areal
    print("areal:           ", getattr(areal, "__file__", None))
except Exception as exc:
    print("areal import ERROR:", repr(exc))
try:
    import importlib.util
    spec = importlib.util.find_spec("sglang")
    print("sglang spec:     ", None if spec is None else spec.origin)
    try:
        from importlib.metadata import version
        print("sglang version:  ", version("sglang"))
    except Exception:
        pass
except Exception as exc:
    print("sglang lookup ERROR:", repr(exc))
PY
}

# Save a non-secret runtime snapshot for reproducibility/debugging. Deliberately
# excludes tokens, credentials and the full environment.
areal_save_env_snapshot() {
  local out="${1:?usage: areal_save_env_snapshot <file>}"
  mkdir -p "$(dirname "${out}")"
  {
    echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
    areal_print_python_env
    echo "---- selected environment ----"
    for name in \
      AREAL_ENV_PROFILE AREAL_EXAMPLES_ROOT AREAL_HOME AREAL_ROOT BASE_DIR \
      VENV VENV_PATH PYTHON_BIN DTK_ENV \
      MEGATRON_HOME MEGATRON_ROOT SGLANG_HOME SGLANG_ROOT AREAL_EXTRA_PYTHONPATH \
      PATH PYTHONPATH \
      CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES \
      RAY_ADDRESS RAY_PORT RAY_TMPDIR RAY_DEDUP_LOGS RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO \
      AREAL_RUNS_ROOT AREAL_RUNTIME_ROOT AREAL_CACHE_ROOT FILER_ROOT NAME_RESOLVE_ROOT \
      TMPDIR XDG_CACHE_HOME TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR TORCH_EXTENSIONS_DIR \
      PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF \
      USE_DCU_CUSTOM_ALLREDUCE SGL_CHUNKED_PREFIX_CACHE_THRESHOLD \
      SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT \
      SGLANG_SET_CPU_AFFINITY SGLANG_ENABLE_SPEC_V2 SGLANG_KVALLOC_KERNEL \
      SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO SGLANG_ASSIGN_EXTEND_CACHE_LOCS \
      SGLANG_ASSIGN_REQ_TO_TOKEN_POOL SGLANG_GET_LAST_LOC \
      SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES \
      GLIBC_TUNABLES HIP_KERNEL_BATCH_CEILING GPU_MAX_HW_QUEUES \
      HIP_H2D_DISABLE_COPY_BUFFER HIP_D2H_DISABLE_COPY_BUFFER \
      HIP_H2D_DIRECT_COPY_THRESHOLD HIP_H2D_HSAAPI_COPY_THRESHOLD \
      HIP_D2H_DIRECT_COPY_THRESHOLD HIP_D2H_HSAAPI_COPY_THRESHOLD \
      TORCH_COMPILE_DISABLE TORCHDYNAMO_DISABLE HSA_KERNARG_POOL_SIZE ROC_AQL_QUEUE_SIZE \
      NCCL_MAX_NCHANNELS NCCL_MIN_NCHANNELS ALLREDUCE_STREAM_WITH_COMPUTE
    do
      printf '%s=%s\n' "${name}" "${!name-}"
    done
  } | tee "${out}"
}

# Validate the actual Python environment of Ray workers, not only the driver.
# This catches the common case where RESTART_RAY=0 reuses a raylet that was
# started from a different venv/PYTHONPATH. Requires RAY_ADDRESS to be set.
areal_validate_ray_worker_env() {
  [[ -n "${RAY_ADDRESS:-}" ]] || {
    echo "[ERROR] RAY_ADDRESS is required for areal_validate_ray_worker_env" >&2
    return 2
  }

  EXPECTED_VENV="${VENV}" \
  EXPECTED_SGLANG_HOME="${SGLANG_HOME}" \
  EXPECTED_MEGATRON_LM="${MEGATRON_HOME}/Megatron-LM" \
  EXPECTED_AREAL_ENV_PROFILE="${AREAL_ENV_PROFILE}" \
  EXPECTED_USE_DCU_CUSTOM_ALLREDUCE="${USE_DCU_CUSTOM_ALLREDUCE:-}" \
  EXPECTED_SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-}" \
  EXPECTED_SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-}" \
  EXPECTED_SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO="${SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO:-}" \
  EXPECTED_TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-}" \
  EXPECTED_TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-}" \
  EXPECTED_HSA_KERNARG_POOL_SIZE="${HSA_KERNARG_POOL_SIZE:-}" \
  EXPECTED_ROC_AQL_QUEUE_SIZE="${ROC_AQL_QUEUE_SIZE:-}" \
  EXPECTED_NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-}" \
  EXPECTED_NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-}" \
  EXPECTED_ALLREDUCE_STREAM_WITH_COMPUTE="${ALLREDUCE_STREAM_WITH_COMPUTE:-}" \
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
import pathlib
import socket
import sys
import ray

address = os.environ["RAY_ADDRESS"]
expected_venv = pathlib.Path(os.environ["EXPECTED_VENV"]).absolute()
expected_sglang = pathlib.Path(os.environ["EXPECTED_SGLANG_HOME"]).resolve()
expected_megatron_lm = pathlib.Path(os.environ["EXPECTED_MEGATRON_LM"]).resolve()

ray.init(address=address)

@ray.remote(num_cpus=0)
def probe():
    spec = importlib.util.find_spec("sglang")
    return {
        "host": socket.gethostname(),
        "python": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "sys_path": list(sys.path),
        "sglang_spec": None if spec is None else spec.origin,
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hip_visible": os.environ.get("HIP_VISIBLE_DEVICES"),
        "rocr_visible": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "selected_env": {
            name: os.environ.get(name)
            for name in [
                "AREAL_ENV_PROFILE",
                "USE_DCU_CUSTOM_ALLREDUCE",
                "SGLANG_SET_CPU_AFFINITY",
                "SGLANG_ENABLE_SPEC_V2",
                "SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO",
                "TORCH_COMPILE_DISABLE",
                "TORCHDYNAMO_DISABLE",
                "HSA_KERNARG_POOL_SIZE",
                "ROC_AQL_QUEUE_SIZE",
                "NCCL_MAX_NCHANNELS",
                "NCCL_MIN_NCHANNELS",
                "ALLREDUCE_STREAM_WITH_COMPUTE",
            ]
        },
    }

alive = [n for n in ray.nodes() if n.get("Alive")]
refs = []
for node in alive:
    addr = node.get("NodeManagerAddress")
    node_resource = f"node:{addr}"
    resources = node.get("Resources", {})
    if node_resource in resources:
        refs.append((addr, probe.options(resources={node_resource: 0.001}).remote()))
    else:
        # Fallback for unusual Ray resource naming. This still checks at least
        # one worker, but cannot hard-pin the task to this node.
        refs.append((addr, probe.remote()))

results = []
for requested_addr, ref in refs:
    result = ray.get(ref)
    result["requested_node"] = requested_addr
    results.append(result)

ray.shutdown()

expected_selected_env = {
    "AREAL_ENV_PROFILE": os.environ.get("EXPECTED_AREAL_ENV_PROFILE", ""),
    "USE_DCU_CUSTOM_ALLREDUCE": os.environ.get("EXPECTED_USE_DCU_CUSTOM_ALLREDUCE", ""),
    "SGLANG_SET_CPU_AFFINITY": os.environ.get("EXPECTED_SGLANG_SET_CPU_AFFINITY", ""),
    "SGLANG_ENABLE_SPEC_V2": os.environ.get("EXPECTED_SGLANG_ENABLE_SPEC_V2", ""),
    "SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO": os.environ.get("EXPECTED_SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO", ""),
    "TORCH_COMPILE_DISABLE": os.environ.get("EXPECTED_TORCH_COMPILE_DISABLE", ""),
    "TORCHDYNAMO_DISABLE": os.environ.get("EXPECTED_TORCHDYNAMO_DISABLE", ""),
    "HSA_KERNARG_POOL_SIZE": os.environ.get("EXPECTED_HSA_KERNARG_POOL_SIZE", ""),
    "ROC_AQL_QUEUE_SIZE": os.environ.get("EXPECTED_ROC_AQL_QUEUE_SIZE", ""),
    "NCCL_MAX_NCHANNELS": os.environ.get("EXPECTED_NCCL_MAX_NCHANNELS", ""),
    "NCCL_MIN_NCHANNELS": os.environ.get("EXPECTED_NCCL_MIN_NCHANNELS", ""),
    "ALLREDUCE_STREAM_WITH_COMPUTE": os.environ.get("EXPECTED_ALLREDUCE_STREAM_WITH_COMPUTE", ""),
}

errors = []
for r in results:
    print("Ray worker env:", r)
    exe = pathlib.Path(r["python"]).absolute()
    paths = []
    for x in r["sys_path"]:
        if not x:
            continue
        try:
            paths.append(pathlib.Path(x).resolve())
        except Exception:
            pass

    try:
        exe.relative_to(expected_venv)
    except ValueError:
        errors.append(
            f"node {r['requested_node']}: python={exe} is not under VENV={expected_venv}"
        )

    if expected_sglang not in paths:
        errors.append(
            f"node {r['requested_node']}: SGLANG_HOME={expected_sglang} not present in sys.path"
        )
    if expected_megatron_lm not in paths:
        errors.append(
            f"node {r['requested_node']}: Megatron-LM={expected_megatron_lm} not present in sys.path"
        )

    actual_env = r.get("selected_env", {})
    for name, expected_value in expected_selected_env.items():
        if expected_value == "":
            continue
        actual_value = actual_env.get(name)
        if actual_value != expected_value:
            errors.append(
                f"node {r['requested_node']}: {name}={actual_value!r}, expected {expected_value!r}"
            )

if errors:
    print("[ERROR] Ray worker Python environment mismatch:")
    for e in errors:
        print("  -", e)
    print("Ray was started from a different environment. Stop and restart Ray on every node with the same AREAL_ENV_PROFILE and this examples package.")
    print("Single-node example:")
    print("  AREAL_ENV_PROFILE=%s bash scripts/stop_ray.sh" % os.environ.get("EXPECTED_AREAL_ENV_PROFILE", "qwen"))
    print("  AREAL_ENV_PROFILE=%s NUM_GPUS=8 NUM_CPUS=128 bash scripts/start_ray.sh <HEAD_IP>" % os.environ.get("EXPECTED_AREAL_ENV_PROFILE", "qwen"))
    raise SystemExit(3)

print(f"[OK] Ray worker Python environment validated on {len(results)} node(s).")
PY
}