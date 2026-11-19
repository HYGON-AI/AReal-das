# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V3/R1 BF16 AWEX layouts; only the requested architecture opts in."""

from __future__ import annotations

from functools import lru_cache
import os
import re


def is_deepseek(adapter, role):
    if os.getenv("HCU_AREAL_DISABLE_DEEPSEEK_AWEX", "").strip().lower() in (
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
    return getattr(config, "model_type", None) == "deepseek_v3"


def deepseek_inference_parts(name, tensor, config):
    """Writable, replicated views: fused [q_a; kv_a], without copying storage."""
    if not re.fullmatch(
        r"model\.layers\.\d+\.self_attn\.fused_qkv_a_proj_with_mqa\.weight", name
    ):
        return None
    q = int(config.q_lora_rank)
    kv = int(config.kv_lora_rank) + int(config.qk_rope_head_dim)
    if tensor.ndim != 2 or tuple(tensor.shape) != (q + kv, int(config.hidden_size)):
        raise ValueError(
            f"Invalid DeepSeek fused Q/KV shape: {name} {tuple(tensor.shape)}"
        )
    return [
        (name.replace("fused_qkv_a_proj_with_mqa", "q_a_proj"), tensor.narrow(0, 0, q)),
        (
            name.replace("fused_qkv_a_proj_with_mqa", "kv_a_proj_with_mqa"),
            tensor.narrow(0, q, kv),
        ),
    ]


def is_deepseek_runtime_kv_scale(name):
    """SGLang-only KV-cache scales are not trainable model weights."""
    return bool(
        re.fullmatch(
            r"model\.layers\.\d+\.self_attn\.attn_(?:mha|mqa)\.[kv]_scale", name
        )
    )


class DeepseekShardingProxy:
    def __init__(self, base):
        self._base = base

    def get_sharding_strategy(self, name, **kwargs):
        if name.endswith(".weight_scale_inv"):
            name = name[
                : -len("_scale_inv")
            ]  # Block scales follow the corresponding weight axis.
        if (
            ".self_attn.q_a_proj." in name
            or ".self_attn.kv_a_proj_with_mqa." in name
            or name.endswith(".mlp.gate.e_score_correction_bias")
        ):
            from awex.sharding.param_sharding import ShardingType

            return ShardingType.NO_SHARDING, 0, 1
        return self._base.get_sharding_strategy(name, **kwargs)

    def __getattr__(self, name):
        return getattr(self._base, name)


@lru_cache(maxsize=1)
def deepseek_mapping_registry():
    from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge

    return DeepSeekV3Bridge.__new__(DeepSeekV3Bridge).mapping_registry()


def deepseek_convert_gathered(name, tensor):
    """Use native DeepSeek Bridge mappings after the existing TP gather."""
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        GatedMLPMapping,
    )

    while name.startswith("module."):
        name = name[len("module.") :]
    mapping = deepseek_mapping_registry().megatron_to_hf_lookup(name)
    if isinstance(mapping, GatedMLPMapping):
        if tensor.shape[0] % 2:
            raise ValueError(
                f"Invalid DeepSeek gate/up shape: {name} {tuple(tensor.shape)}"
            )
        gate, up = tensor.chunk(2, dim=0)
        return [(mapping.hf_param["gate"], gate), (mapping.hf_param["up"], up)]
    if isinstance(mapping, AutoMapping):
        if mapping.permute_dims is not None:
            tensor = tensor.permute(*mapping.permute_dims)
        return [(mapping.hf_param, tensor)]
    raise ValueError(
        f"Unsupported DeepSeek AWEX mapping: {name} {type(mapping).__name__}"
    )


def assert_deepseek_transfer_metadata(infer_meta, train_meta):
    """Require exact names/global shapes/dtypes before creating transfer groups."""

    def index(entries, role):
        result = {}
        for item in entries:
            if item.name in result:
                raise ValueError(f"DeepSeek AWEX duplicate {role} name: {item.name}")
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
            f"DeepSeek AWEX metadata preflight FAILED: train={len(trn)} infer={len(inf)} "
            f"missing={len(missing)} {missing[:8]} extra={len(extra)} {extra[:8]} "
            f"shape_mismatch={len(shapes)} {shapes[:8]} dtype_mismatch={len(dtypes)} {dtypes[:8]}"
        )
    if not trn:
        raise ValueError("DeepSeek AWEX metadata must not be empty")
    return len(trn)
