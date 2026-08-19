# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0
export USE_OPTIMIZED_MODEL="${USE_OPTIMIZED_MODEL:-1}"
# Use the optimized VLLM Ascend model by default. Set USE_OPTIMIZED_MODEL=0
# only for workloads that are incompatible with the optimized implementation.

python examples/vlm_npu/geometry3k_grpo.py \
    --config examples/vlm_npu/qwen2_5_vl_3b_geometry3k_grpo.yaml \
    scheduler.type=local