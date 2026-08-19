#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common/common_env.sh"

HEAD_IP="${1:-${HEAD_IP:-127.0.0.1}}"
RAY_ADDRESS="${RAY_ADDRESS:-${HEAD_IP}:${RAY_PORT}}"
EXPECTED_NODES="${EXPECTED_NODES:-}"
EXPECTED_GPUS="${EXPECTED_GPUS:-}"
MIN_CPUS_PER_NODE="${MIN_CPUS_PER_NODE:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-}"

ray status --address="${RAY_ADDRESS}"

"${PYTHON_BIN}" - <<PY
import os
import ray

address = "${RAY_ADDRESS}"
expected_nodes = os.environ.get("EXPECTED_NODES")
expected_gpus = os.environ.get("EXPECTED_GPUS")
min_cpus_per_node = float(os.environ.get("MIN_CPUS_PER_NODE", "0") or 0)
gpus_per_node = os.environ.get("GPUS_PER_NODE")

ray.init(address=address)
resources = ray.cluster_resources()
nodes = [node for node in ray.nodes() if node["Alive"]]

print("Ray cluster resources:")
print(resources)
print("Alive nodes:", len(nodes))

for node in nodes:
    node_resources = node.get("Resources", {})
    addr = node.get("NodeManagerAddress")
    cpu = node_resources.get("CPU", 0)
    gpu = node_resources.get("GPU", 0)
    print("node:", addr, "CPU:", cpu, "GPU:", gpu, "resources:", node_resources)
    if min_cpus_per_node and cpu < min_cpus_per_node:
        raise RuntimeError(f"Node {addr} CPU insufficient: {cpu} < {min_cpus_per_node}")
    if gpus_per_node and gpu != float(gpus_per_node):
        raise RuntimeError(f"Node {addr} should register {gpus_per_node} GPUs, got {gpu}")

if expected_nodes and len(nodes) != int(expected_nodes):
    raise RuntimeError(f"Expected {expected_nodes} Ray nodes, got {len(nodes)}")

if expected_gpus and resources.get("GPU", 0) != float(expected_gpus):
    raise RuntimeError(f"Expected {expected_gpus} GPUs/HCUs, got {resources.get('GPU', 0)}")

@ray.remote(num_gpus=0)
def env_probe():
    import os
    result = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "RAY_DEDUP_LOGS": os.environ.get("RAY_DEDUP_LOGS"),
    }
    try:
        import torch
        result["torch_cuda_device_count"] = torch.cuda.device_count()
        result["torch_hip"] = getattr(torch.version, "hip", None)
    except Exception as exc:  # pragma: no cover - diagnostic only
        result["torch_error"] = repr(exc)
    return result

print("Env probe:")
print(ray.get(env_probe.remote()))
ray.shutdown()
print("[OK] Ray cluster status check passed")
PY
