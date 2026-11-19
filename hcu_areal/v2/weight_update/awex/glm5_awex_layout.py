# SPDX-License-Identifier: Apache-2.0

"""GLM-5 MLA/DSA layouts; import-light rules for the runtime patch."""

from __future__ import annotations

import os
import re


def is_glm5(adapter, role):
    if os.getenv("HCU_AREAL_DISABLE_GLM5_AWEX", "").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    ):
        return False
    if role == "train":
        config = getattr(getattr(adapter, "_engine", None), "hf_config", None)
    elif role == "infer":
        config = getattr(adapter._get_model(), "config", None)
    else:
        raise ValueError(f"Unknown AWEX role: {role}")
    return getattr(config, "model_type", None) == "glm_moe_dsa"


_FUSED_A = re.compile(
    r"^model\.layers\.\d+\.self_attn\.fused_qkv_a_proj_with_mqa\.weight$"
)


def glm5_inference_parts(name, tensor, config):
    """Return writable views of the replicated [q_a; kv_a] projection."""
    if not _FUSED_A.fullmatch(name):
        return None
    q = int(config.q_lora_rank)
    kv = int(config.kv_lora_rank) + int(config.qk_rope_head_dim)
    if tensor.ndim != 2 or tuple(tensor.shape) != (q + kv, int(config.hidden_size)):
        raise ValueError(f"Invalid GLM fused Q/KV shape: {name} {tuple(tensor.shape)}")
    return [
        (name.replace("fused_qkv_a_proj_with_mqa", "q_a_proj"), tensor.narrow(0, 0, q)),
        (
            name.replace("fused_qkv_a_proj_with_mqa", "kv_a_proj_with_mqa"),
            tensor.narrow(0, q, kv),
        ),
    ]


class Glm5ShardingProxy:
    """Override replicated MLA/DSA tensors; preserve the base TP/EP rules."""

    def __init__(self, base):
        self._base = base

    def get_sharding_strategy(self, name, **kwargs):
        if (
            ".self_attn.q_a_proj." in name
            or ".self_attn.kv_a_proj_with_mqa." in name
            or ".self_attn.indexer." in name
            or name.endswith(".mlp.gate.e_score_correction_bias")
        ):
            from awex.sharding.param_sharding import ShardingType

            return ShardingType.NO_SHARDING, 0, 1
        return self._base.get_sharding_strategy(name, **kwargs)

    def __getattr__(self, name):
        return getattr(self._base, name)


def glm5_mapping_registry(hf_config):
    # Import only inside a GLM Megatron worker, after its environment is set.
    from megatron.bridge.models.glm_moe_dsa.glm5_bridge import GLM5Bridge

    # The Bridge uses the actual layer and MTP counts to construct mappings.
    # Construct once per export, without loading another model or checkpoint.
    if hf_config is None:
        raise ValueError("GLM AWEX mapping requires the model's HF config")
    bridge = GLM5Bridge.__new__(GLM5Bridge)
    bridge.hf_config = hf_config
    return bridge.mapping_registry()


def glm5_convert_gathered(name, tensor, registry):
    """Apply the existing GLM bridge's names/layouts after AWEX's TP gather."""
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        GatedMLPMapping,
    )

    while name.startswith("module."):
        name = name[len("module.") :]
    mapping = registry.megatron_to_hf_lookup(name)
    if isinstance(mapping, GatedMLPMapping):
        if tensor.shape[0] % 2:
            raise ValueError(f"Invalid GLM gate/up shape: {name} {tuple(tensor.shape)}")
        gate, up = tensor.chunk(2, dim=0)
        return [(mapping.hf_param["gate"], gate), (mapping.hf_param["up"], up)]
    if isinstance(mapping, AutoMapping):
        if mapping.permute_dims is not None:
            tensor = tensor.permute(*mapping.permute_dims)
        return [(mapping.hf_param, tensor)]
    raise ValueError(f"Unsupported GLM AWEX mapping: {name} {type(mapping).__name__}")


def glm5_wire_tensor(name, tensor):
    """Match SGLang's indexer LayerNorm storage without changing train params.

    nsa_indexer.py stores k_norm in FP32, or BF16 when the HIP AITER path is
    active. BF16 training values can be represented exactly in the FP32 wire
    format. All other weights retain their original dtype and storage.
    """
    if not re.fullmatch(
        r"model\.layers\.\d+\.self_attn\.indexer\.k_norm\.(weight|bias)", name
    ):
        return tensor
    import torch

    aiter = os.getenv("SGLANG_USE_AITER", "0").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )
    dtype = torch.bfloat16 if aiter and torch.version.hip is not None else torch.float32
    return tensor.to(dtype=dtype)


def assert_glm5_transfer_metadata(infer_meta, train_meta):
    """Reject missing/extra names, shape and dtype mismatches before NCCL."""

    def index(entries, role):
        result = {}
        for item in entries:
            if item.name in result:
                raise ValueError(f"GLM AWEX duplicate {role} name: {item.name}")
            result[item.name] = item
        return result

    inf, trn = index(infer_meta, "infer"), index(train_meta, "train")
    common = sorted(set(inf) & set(trn))
    missing, extra = sorted(set(trn) - set(inf)), sorted(set(inf) - set(trn))
    shapes = [
        (n, tuple(trn[n].global_shape), tuple(inf[n].global_shape))
        for n in common
        if tuple(trn[n].global_shape) != tuple(inf[n].global_shape)
    ]
    dtypes = [
        (n, str(trn[n].dtype), str(inf[n].dtype))
        for n in common
        if str(trn[n].dtype) != str(inf[n].dtype)
    ]
    if missing or extra or shapes or dtypes:
        raise ValueError(
            f"GLM AWEX metadata preflight FAILED: train={len(trn)} infer={len(inf)} "
            f"missing={len(missing)} {missing[:8]} extra={len(extra)} {extra[:8]} "
            f"shape_mismatch={len(shapes)} {shapes[:8]} dtype_mismatch={len(dtypes)} {dtypes[:8]}"
        )
    if not trn:
        raise ValueError("GLM AWEX metadata must not be empty")
    return len(trn)
