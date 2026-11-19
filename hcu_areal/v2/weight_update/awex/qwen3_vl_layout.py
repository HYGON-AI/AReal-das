# SPDX-License-Identifier: Apache-2.0
"""VLM weight layout operations for AReaL v2 AWEX FSDP <-> SGLang.

The SGLang vision QKV TP parameter is rank-major:
  [Q(tp0), K(tp0), V(tp0), Q(tp1), K(tp1), V(tp1), ...]
The Hugging Face / FSDP global checkpoint QKV parameter is component-major:
  [Q(all heads), K(all heads), V(all heads)].

Do not only rename these tensors: transfer plans operate on physical row offsets.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

_QWEN3_VL_TYPES = frozenset({"qwen3_vl"})
_VISION_QKV = re.compile(r"^model\.visual\.blocks\.\d+\.attn\.qkv\.(weight|bias)$")


def is_qwen3_vl_type(model_type: str | None) -> bool:
    return model_type in _QWEN3_VL_TYPES


def normalize_qwen3_vl_train_name(name: str) -> str:
    """Normalize FSDP HF wrapper names without touching unrelated models."""
    if name.startswith("model.model."):
        name = "model." + name[len("model.model.") :]
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model.") :]
    if name.startswith("visual."):
        return "model." + name
    return name


def normalize_qwen3_vl_visual_infer_name(name: str) -> str:
    """Normalize SGLang vision-module names to HF checkpoint names."""
    if not name.startswith("visual."):
        raise ValueError(f"Not an SGLang vision parameter: {name}")
    return "model." + name.replace(".attn.qkv_proj.", ".attn.qkv.")


def is_qwen3_vl_vision_qkv(name: str) -> bool:
    return bool(_VISION_QKV.fullmatch(name))


def pack_vision_qkv_rank_major(full_hf_qkv: Tensor, tp_size: int) -> Tensor:
    """Reorder HF [Q,K,V] rows into SGLang TP-rank-major [Q_i,K_i,V_i].

    Works for vision QKV weight [3*hidden, hidden] and bias [3*hidden].
    The caller separately shards this full packed tensor into FSDP source shards.
    """
    import torch

    if tp_size < 1:
        raise ValueError(f"Invalid inference TP size: {tp_size}")
    if not full_hf_qkv.ndim:
        raise ValueError("Vision QKV must be at least a 1-D tensor")
    if full_hf_qkv.shape[0] % (3 * tp_size):
        raise ValueError(
            f"Vision QKV rows={full_hf_qkv.shape[0]} cannot be split into "
            f"Q/K/V and tp_size={tp_size}; padded/dummy heads require "
            "a model-specific layout converter"
        )
    q, k, v = full_hf_qkv.chunk(3, dim=0)
    q_shards, k_shards, v_shards = (
        tensor.chunk(tp_size, dim=0) for tensor in (q, k, v)
    )
    return torch.cat(
        [
            torch.cat((q_shards[i], k_shards[i], v_shards[i]), dim=0)
            for i in range(tp_size)
        ],
        dim=0,
    ).contiguous()


def qwen3_vl_vision_sharding_dim(name: str) -> int | None:
    """Return the SGLang vision TP shard dimension, or None if replicated.

    Derived from Qwen3VL SGLang VisionAttention, VisionMLP and PatchMerger.
    This contract is restricted to Qwen3-VL dense, no DP-vision encoder.
    """
    if not name.startswith("model.visual."):
        raise ValueError(f"Not a Qwen3-VL vision parameter: {name}")
    if name.startswith("model.visual.patch_embed."):
        return None
    if name == "model.visual.pos_embed.weight":
        return 0
    if name.endswith(
        (
            ".norm.weight",
            ".norm.bias",
            ".norm1.weight",
            ".norm1.bias",
            ".norm2.weight",
            ".norm2.bias",
        )
    ):
        return None
    if name.endswith((".attn.qkv.weight", ".attn.qkv.bias")):
        return 0
    if name.endswith(".attn.proj.weight"):
        return 1
    if name.endswith(".attn.proj.bias"):
        return None
    if name.endswith((".linear_fc1.weight", ".linear_fc1.bias")):
        return 0
    if name.endswith(".linear_fc2.weight"):
        return 1
    if name.endswith(".linear_fc2.bias"):
        return None
    raise ValueError(f"Unknown Qwen3-VL visual tensor sharding layout: {name}")
