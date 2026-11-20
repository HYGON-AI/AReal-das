# SPDX-License-Identifier: Apache-2.0

"""Weight views for DeepSeek V4's ROCm block-FP8 rollout."""

from __future__ import annotations

import re

_FP8_BLOCK_SIZE = 128
_Q_LORA_RANK = 1024
_EXPERT = re.compile(
    r"(model\.layers\.\d+\.mlp\.experts)\.w(13|2)_weight(_scale_inv)?$"
)
_SHARDED = {
    "embed_tokens": 0,
    "lm_head": 0,
    "self_attn.wq_b": 0,
    "self_attn.wo_a": 0,
    "self_attn.wo_b": 1,
    "gate_proj": 0,
    "up_proj": 0,
    "down_proj": 1,
}


def is_deepseek_v4(adapter, role):
    model = (
        getattr(adapter, "_engine", None) if role == "train" else adapter._get_model()
    )
    config = getattr(model, "hf_config" if role == "train" else "config", None)
    return getattr(config, "model_type", None) == "deepseek_v4"


def _runtime_name(name):
    if name.startswith("embed."):
        return "model.embed_tokens." + name.removeprefix("embed.")
    if name.startswith("head."):
        return "lm_head." + name.removeprefix("head.")
    if name.startswith(("hc_head_", "norm.")):
        return "model." + name
    name = "model." + name
    for source, target in (
        (".attn.", ".self_attn."),
        (".ffn.", ".mlp."),
        (".attn_norm.", ".input_layernorm."),
        (".ffn_norm.", ".post_attention_layernorm."),
        (".indexer.scorer.weights_proj.", ".indexer.weights_proj."),
        (".gate.bias", ".gate.e_score_correction_bias"),
        (".w1.", ".gate_proj."),
        (".w2.", ".down_proj."),
        (".w3.", ".up_proj."),
    ):
        name = name.replace(source, target)
    return name


def _is_static(name):
    return name.endswith((".tid2eid", ".attn_mqa.k_scale", ".attn_mqa.v_scale"))


def _quantized(name):
    if not name.endswith(".weight"):
        return False
    module = name.removesuffix(".weight")
    if ".mlp." in module:
        return module.endswith((".gate_proj", ".up_proj", ".down_proj"))
    return (
        module.endswith((".wq_a", ".wkv", ".wq_b", ".wo_b"))
        and ".self_attn." in module
        and ".compressor." not in module
    )


def export_parts(name, weight, compress_ratios, vocab_size):
    """Convert the Bridge's checkpoint names to the writable AWEX views."""
    import torch

    if _is_static(name):
        return
    name = _runtime_name(name)
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        weight = weight[:vocab_size]
    elif name.endswith(".wo_a.weight"):
        weight = weight.flatten(0, -2)
    elif name.endswith(".compressor.ape"):
        layer = int(name.split(".")[2])
        if compress_ratios[layer] == 4:
            # SGLang reorders the overlapping compressor APE after checkpoint load.
            weight = torch.cat(weight.chunk(2, dim=-1), dim=0).reshape_as(weight)

    if _quantized(name):
        from areal.engine.megatron_utils.fp8.ue8m0 import per_block_cast_to_fp8_ue8m0

        # V4's source scales are powers of two. Retain that quantization grid
        # even though the HCU rollout stores the expanded scales as FP32.
        packed, scales = per_block_cast_to_fp8_ue8m0(weight.to(torch.bfloat16))
        yield name, packed.view(torch.uint8)
        yield name + "_scale_inv", scales
        return

    fp32 = (
        ".hc_" in name
        or ".compressor." in name
        and name.endswith((".ape", ".norm.weight"))
        or name.endswith((".attn_sink", ".e_score_correction_bias"))
    )
    yield name, weight.to(torch.float32 if fp32 else torch.bfloat16)


def inference_parts(name, weight):
    """Return aliases directly backed by SGLang's live tensor storage."""
    import torch

    if _is_static(name):
        return []
    if weight.dtype == torch.float8_e4m3fn:
        weight = weight.view(torch.uint8)
    module = name.removesuffix("_scale_inv").removesuffix(".weight")
    if module.endswith((".wqkv_a", ".wq_b", ".wo_b")) or ".shared_experts." in module:
        # hipBLASLt stores these linears transposed, including square wo_b
        # matrices whose shape does not reveal the layout.
        weight = weight.transpose(0, 1)
    expert = _EXPERT.fullmatch(name)
    if expert is not None:
        prefix, projection, scale = expert.groups()
        suffix = ".weight" + (scale or "")
        if projection == "2":
            return [
                (f"{prefix}.{i}.down_proj{suffix}", value)
                for i, value in enumerate(weight.unbind(0))
            ]
        return [
            (f"{prefix}.{i}.{part}{suffix}", value)
            for i, values in enumerate(weight.unbind(0))
            for part, value in zip(("gate_proj", "up_proj"), values.chunk(2, dim=0))
        ]
    for fused, first, second, first_rows in (
        (".wqkv_a.", ".wq_a.", ".wkv.", _Q_LORA_RANK),
        (".compressor.wkv_gate.", ".compressor.wkv.", ".compressor.wgate.", None),
        (
            ".shared_experts.gate_up_proj.",
            ".shared_experts.gate_proj.",
            ".shared_experts.up_proj.",
            None,
        ),
    ):
        if fused in name:
            rows = first_rows or weight.shape[0] // 2
            if first_rows and name.endswith("_scale_inv"):
                rows //= _FP8_BLOCK_SIZE
            left, right = weight.split((rows, weight.shape[0] - rows), dim=0)
            return [
                (name.replace(fused, first), left),
                (name.replace(fused, second), right),
            ]
    return [(name, weight)]


class DeepSeekV4Sharding:
    def __init__(self, tp_size):
        self.tp_size = tp_size

    def get_sharding_strategy(self, name):
        from awex.sharding.param_sharding import ShardingType

        module = name.removesuffix("_scale_inv").removesuffix(".weight")
        if ".indexer." in module:
            return ShardingType.NO_SHARDING, 0, 1
        prefix, _, projection = module.rpartition(".")
        key = f"self_attn.{projection}" if prefix.endswith(".self_attn") else projection
        axis = _SHARDED.get(key)
        if axis is None or self.tp_size == 1:
            return ShardingType.NO_SHARDING, 0, 1
        return ShardingType.TP_SHARDING, axis, self.tp_size


def validate_rollout(adapter, rank_info):
    import torch

    model = adapter._get_model()
    args = adapter._scheduler.server_args
    if (
        rank_info.tp_size != 2
        or args.ep_size != 1
        or args.pp_size != 1
        or args.enable_dp_attention
        or args.enable_dp_lm_head
        or getattr(args, "speculative_algorithm", None)
        or getattr(model, "num_fused_shared_experts", 0)
        or model.config.q_lora_rank != _Q_LORA_RANK
        or model.model.layers[0].mlp.experts.w13_weight.dtype != torch.float8_e4m3fn
    ):
        raise NotImplementedError(
            "DeepSeek V4 AWEX requires SGLang TP2, EP1, PP1 and the HCU FP4-to-FP8 rollout"
        )
