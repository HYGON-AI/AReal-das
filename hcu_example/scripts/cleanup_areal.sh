#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common/common_env.sh"

CLEAN_NAME_RESOLVE="${CLEAN_NAME_RESOLVE:-1}"
STOP_RAY="${STOP_RAY:-0}"

if [[ "${STOP_RAY}" == "1" ]]; then
  ray stop -f || true
fi

pkill -TERM -f 'python(3)? -m areal\.infra\.rpc\.(guard|rpc_server)' || true
pkill -TERM -f 'RayRPCServer' || true
pkill -TERM -f 'python(3)? -m sglang\.launch_server' || true
pkill -TERM -f 'python(3)? .*gsm8k_rl\.py' || true

if [[ "${CLEAN_NAME_RESOLVE}" == "1" ]]; then
  rm -rf "${NAME_RESOLVE_ROOT}"
  mkdir -p "${NAME_RESOLVE_ROOT}"
fi

echo "[OK] Cleaned AReaL/SGLang processes on this node."
