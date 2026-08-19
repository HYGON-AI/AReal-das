#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Choose one of: base, qwen, qwen35, glm5.
export AREAL_ENV_PROFILE="${AREAL_ENV_PROFILE:-qwen}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common/common_env.sh"

areal_preflight_common
areal_print_python_env

echo
cat <<INFO
Selected profile: ${AREAL_ENV_PROFILE}
RAY_PORT:        ${RAY_PORT}
RAY_TMPDIR:      ${RAY_TMPDIR}
FILER_ROOT:      ${FILER_ROOT}
NAME_RESOLVE:    ${NAME_RESOLVE_ROOT}
AREAL_RUNS_ROOT: ${AREAL_RUNS_ROOT}
INFO

echo "[OK] Local environment check completed."
