# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0
export USE_OPTIMIZED_MODEL="${USE_OPTIMIZED_MODEL:-1}"
# Use the optimized VLLM Ascend model by default. Set USE_OPTIMIZED_MODEL=0
# only for workloads that are incompatible with the optimized implementation.

python examples/vlm_npu/virl39k_grpo.py \
   --config examples/vlm_npu/qwen2_5_vl_3b_geometry3k_grpo.yaml \
   scheduler.type=ray \
   experiment_name=virl39k-grpo-multiNode \
   rollout.backend=vllm:d32 \
   actor.backend=fsdp:d16 \
   cluster.n_nodes=3 \
   cluster.n_gpus_per_node=16 \
   train_dataset.path=data/ViRL39K \
   actor.mb_spec.max_tokens_per_mb=16384 \
   rollout.max_concurrent_rollouts=128 \
   rollout.max_head_offpolicyness=4 \
   train_dataset.batch_size=528 \
   valid_dataset.batch_size=528 \
   gconfig.max_new_tokens=16384 \
   gconfig.n_samples=5 \
   saver.freq_epochs=1 \
   total_train_epochs=5
