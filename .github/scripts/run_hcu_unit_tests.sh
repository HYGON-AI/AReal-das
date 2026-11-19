#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
exec "${HCU_CI_PYTHON:-python3}" -m pytest \
    -q -rs --maxfail=1 --durations=20 \
    -o cache_dir=/tmp/pytest-cache \
    --junitxml="${HCU_CI_JUNIT_XML:-pytest-cpu.xml}" \
    tests/unit
