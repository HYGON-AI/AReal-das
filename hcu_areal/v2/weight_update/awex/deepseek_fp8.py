# SPDX-License-Identifier: Apache-2.0

"""DeepSeek block-FP8 export and writable SGLang runtime views.

AWEX transports FP8 storage as uint8 and block scales as FP32. This module
never changes the precision of the destination model or treats a float8 cast
as dequantization. Packed AITER/DeepGEMM/MX formats require other layouts.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re

TARGET_ENV = "HCU_AREAL_DEEPSEEK_FP8_CONFIG"


def validate_quant_config(config):
    if config is None:
        return None
    if not isinstance(config, dict):
        name = (
            config.get_name() if callable(getattr(config, "get_name", None)) else None
        )
        if name != "fp8" or getattr(config, "use_mxfp8", False):
            raise NotImplementedError(
                "DeepSeek AWEX supports unquantized or block-FP8 e4m3 weights only"
            )
        config = dict(
            quant_method=name,
            fmt="e4m3",
            activation_scheme=getattr(config, "activation_scheme", None),
            weight_block_size=getattr(config, "weight_block_size", None),
            ignored_layers=getattr(config, "ignored_layers", []),
        )
    if (
        config.get("quant_method") != "fp8"
        or config.get("fmt", "e4m3") != "e4m3"
        or config.get("activation_scheme") != "dynamic"
    ):
        raise NotImplementedError(
            "DeepSeek AWEX requires dynamic block-FP8 e4m3 quantization"
        )
    block = config.get("weight_block_size")
    if (
        not isinstance(block, (list, tuple))
        or len(block) != 2
        or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in block)
    ):
        raise NotImplementedError(
            "DeepSeek AWEX FP8 requires two positive block dimensions"
        )
    return dict(config, weight_block_size=list(block))


def target_config(engine):
    raw = os.getenv(TARGET_ENV)
    if raw is not None:
        raw = raw.strip()
        # JSON is convenient for Ray env_vars; paths allow long exclusion lists.
        config = json.loads(
            raw if raw.startswith(("{", "null")) else Path(raw).read_text()
        )
    else:
        config = getattr(engine, "quantization_config", None)
    return validate_quant_config(config)


def float8_dtypes():
    import torch

    return (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


_LINEAR = re.compile(
    r"model\.layers\.\d+\.(?:self_attn\.(?:q_proj|q_a_proj|q_b_proj|"
    r"kv_a_proj_with_mqa|kv_b_proj|o_proj)|mlp\.(?:(?:experts\.\d+|"
    r"shared_experts)\.)?(?:gate_proj|up_proj|down_proj))\.weight$"
)


def quantized_name(name, config):
    if config is None or not _LINEAR.fullmatch(name):
        return False
    prefix = name[: -len(".weight")]
    aliases = {prefix, prefix.removeprefix("model.")}
    if ".self_attn.q_a_proj" in prefix or ".self_attn.kv_a_proj_with_mqa" in prefix:
        aliases |= {
            re.sub(r"(q_a_proj|kv_a_proj_with_mqa)$", "fused_qkv_a_proj_with_mqa", x)
            for x in tuple(aliases)
        }
    if prefix.endswith((".gate_proj", ".up_proj")):
        aliases |= {
            re.sub(r"(gate_proj|up_proj)$", "gate_up_proj", x) for x in tuple(aliases)
        }
    ignored = config.get("ignored_layers") or config.get("modules_to_not_convert") or []
    return not any(
        alias == skip or alias.startswith(skip + ".")
        for alias in aliases
        for skip in ignored
    )


def dequantize_blocks(weight, scale, block, dtype=None):
    """Scale values have dequantization semantics, including *_scale_inv."""
    import torch

    dtype = dtype or torch.bfloat16
    if weight.dtype not in float8_dtypes():
        raise TypeError("Expected actual E4M3 float8 storage with a matching scale")
    if scale.numel() == 1:
        return (weight.float() * scale.float()).to(dtype)
    if weight.ndim != 2 or tuple(scale.shape) != tuple(
        math.ceil(size / step) for size, step in zip(weight.shape, block)
    ):
        raise ValueError("FP8 source weight and scale block geometry do not match")
    expanded = (
        scale.float().repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)
    )
    return (weight.float() * expanded[: weight.shape[0], : weight.shape[1]]).to(dtype)


def source_scale_map(models):
    """Resolve plain PyTorch FP8 scales before expert names become global IDs."""
    result = {}
    for model in models if isinstance(models, (list, tuple)) else [models]:
        if not callable(getattr(model, "named_parameters", None)):
            continue
        params = dict(model.named_parameters())
        values = (
            dict(model.named_buffers())
            if callable(getattr(model, "named_buffers", None))
            else {}
        )
        values.update(params)
        for name, parameter in params.items():
            if parameter.dtype not in float8_dtypes():
                continue
            for candidate in (
                name + "_scale_inv",
                name + "_scale",
                re.sub(r"weight(\d+)$", r"weight_scale_inv\1", name),
            ):
                if candidate in values and candidate != name:
                    result[id(parameter)] = values[candidate]
                    break
    return result


def compute_source(parameter, config, scales):
    """Native TE tensors keep their own dequantization; plain FP8 needs scales."""
    if parameter.dtype not in float8_dtypes():
        return parameter
    from megatron.core.fp8_utils import is_float8tensor

    if is_float8tensor(parameter):
        return parameter
    data = getattr(parameter, "_rowwise_data", parameter)
    scale = getattr(parameter, "_rowwise_scale_inv", None)
    if scale is None:
        scale = scales.get(id(parameter))
    if scale is None:
        raise ValueError(
            "Plain FP8 training parameter has no paired scale; casting alone is invalid"
        )
    if config is None:
        raise ValueError(
            "Plain FP8 training export requires source quantization_config"
        )
    value = dequantize_blocks(data, scale, config["weight_block_size"])
    for attr in ("tensor_model_parallel", "partition_dim", "partition_stride"):
        if hasattr(parameter, attr):
            setattr(value, attr, getattr(parameter, attr))
    return value


def export_weight(name, tensor, config):
    if not quantized_name(name, config):
        return [(name, tensor)]
    import torch

    if tensor.device.type == "cpu":
        block = config["weight_block_size"]
        rows, columns = tensor.shape
        padded = torch.nn.functional.pad(
            tensor.float(), (0, (-columns) % block[1], 0, (-rows) % block[0])
        )
        tiles = padded.view(
            math.ceil(rows / block[0]),
            block[0],
            math.ceil(columns / block[1]),
            block[1],
        )
        scales = (
            tiles.abs().amax(dim=(1, 3)).clamp(min=1e-10)
            / torch.finfo(torch.float8_e4m3fn).max
        )
        inverse = (
            scales.reciprocal()
            .repeat_interleave(block[0], 0)
            .repeat_interleave(block[1], 1)
        )
        quantized = (
            (tensor.float() * inverse[:rows, :columns])
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        pairs = [
            (name, quantized),
            (name[: -len(".weight")] + ".weight_scale_inv", scales),
        ]
    else:
        from areal.engine.megatron_utils.fp8.quantize import _quantize_param

        pairs = _quantize_param(
            name, tensor, config["weight_block_size"], scale_dtype=torch.float32
        )
    weight, scale = pairs[0][1], pairs[1][1]
    # The wire format is always checkpoint E4M3FN. Receiving runtimes which
    # normalize to FNUZ commit both bytes and doubled scales before their ACK.
    return [(name, weight.view(torch.uint8)), (pairs[1][0], scale)]


def runtime_plan(adapter):
    """Validate and cache raw block-FP8 runtime layouts, including DCU transpose."""
    import torch

    model = adapter._get_model()
    config = validate_quant_config(getattr(model, "quant_config", None))
    if config is None:
        return None
    server_args = getattr(getattr(adapter, "_scheduler", None), "server_args", None)
    if int(getattr(server_args, "ep_size", 1)) != 1:
        raise NotImplementedError(
            "DeepSeek FP8 AWEX currently requires inference EP1; training EP is supported"
        )
    cached = getattr(adapter, "_hcu_deepseek_fp8_plan", None)
    if cached is not None and cached["model"] is model and cached["config"] == config:
        return cached
    from sglang.srt.layers.quantization import fp8 as native

    if getattr(native, "_use_aiter", False) or getattr(
        native, "SGLANG_USE_AITER_FP8_ASM_MOE", False
    ):
        raise NotImplementedError(
            "DeepSeek FP8 AWEX requires unshuffled weights: disable AITER FP8 preshuffle"
        )
    modules = {}
    for prefix, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if method is None or not getattr(method, "block_quant", False):
            continue
        validate_quant_config(method.quant_config)
        if getattr(method, "use_mxfp8", False) or getattr(method, "use_marlin", False):
            raise NotImplementedError(
                "Packed MXFP8/Marlin weights need a separate AWEX layout"
            )
        is_linear = hasattr(module, "weight") and module.weight.dtype in float8_dtypes()
        is_moe = (
            hasattr(module, "w13_weight") and module.w13_weight.dtype in float8_dtypes()
        )
        if not (is_linear or is_moe):
            continue
        transpose = (
            is_linear
            and getattr(getattr(method, "w8a8_block_fp8_linear", None), "__name__", "")
            == "hipblaslt_w8a8_block_fp8_linear"
        )
        attrs = ("weight",) if is_linear else ("w13_weight", "w2_weight")
        for attr in attrs:
            weight = getattr(module, attr)
            scale = getattr(module, attr + "_scale_inv", None)
            canonical = weight.T if transpose else weight
            canonical_scale = scale.T if transpose and scale is not None else scale
            block = config["weight_block_size"]
            expected = (
                *canonical.shape[:-2],
                math.ceil(canonical.shape[-2] / block[0]),
                math.ceil(canonical.shape[-1] / block[1]),
            )
            if (
                scale is None
                or scale.dtype != torch.float32
                or getattr(scale, "format_ue8m0", False)
                or tuple(canonical_scale.shape) != expected
            ):
                raise ValueError(
                    f"Unsupported DeepSeek FP8 scale geometry/packing: {prefix}.{attr}"
                )
            modules[prefix + "." + attr] = (module, attr, transpose)
    cached = dict(model=model, config=config, modules=modules, staged={})
    adapter._hcu_deepseek_fp8_plan = cached
    return cached


def inference_parts(adapter, name, tensor, original):
    """Expose matching uint8 weight views and FP32 scale views to AWEX."""
    import torch

    plan = runtime_plan(adapter)
    if plan is None:
        if tensor.dtype in float8_dtypes():
            raise ValueError(
                f"FP8 storage requires quantization_config and a paired scale layout: {name}"
            )
        return None
    scale = name.endswith("_scale_inv")
    base = name[: -len("_scale_inv")] if scale else name
    entry = plan["modules"].get(base)
    if entry is None:
        if tensor.dtype in float8_dtypes():
            raise ValueError(f"Unrecognized FP8 storage layout: {name}")
        return None
    module, attr, transpose = entry
    value = tensor.T if transpose else tensor
    if getattr(module, attr).dtype == torch.float8_e4m3fnuz:
        key = name
        if key not in plan["staged"]:
            plan["staged"][key] = torch.empty_like(
                value, dtype=torch.float32 if scale else torch.uint8
            )
        value = plan["staged"][key]
    if not scale:
        value = value.view(torch.uint8)
    suffix = "weight_scale_inv" if scale else "weight"
    if ".fused_qkv_a_proj_with_mqa." in name:
        cfg = plan["model"].config
        q, kv = int(cfg.q_lora_rank), int(cfg.kv_lora_rank + cfg.qk_rope_head_dim)
        if scale:
            block = plan["config"]["weight_block_size"][0]
            if q % block:
                raise ValueError("Fused Q/KV-A split crosses a quantization block")
            q, kv = q // block, math.ceil(kv / block)
        if value.shape[0] != q + kv:
            raise ValueError("Fused Q/KV-A FP8 weight/scale shape mismatch")
        return [
            (name.replace("fused_qkv_a_proj_with_mqa", "q_a_proj"), value[:q]),
            (
                name.replace("fused_qkv_a_proj_with_mqa", "kv_a_proj_with_mqa"),
                value[q:],
            ),
        ]
    if attr in ("w13_weight", "w2_weight"):
        count = int(plan["model"].config.n_routed_experts)
        prefix = name[: name.rfind(".experts.")]
        parts = []
        for index, item in enumerate(value):
            owner = (
                f"{prefix}.experts.{index}"
                if index < count
                else f"{prefix}.shared_experts"
            )
            if index > count:
                raise NotImplementedError(
                    "Multiple fused shared experts need an explicit layout"
                )
            if attr == "w13_weight":
                if item.shape[0] % 2:
                    raise ValueError(
                        "FP8 gate/up weight or scale cannot be split equally"
                    )
                half = item.shape[0] // 2
                parts += [
                    (owner + ".gate_proj." + suffix, item[:half]),
                    (owner + ".up_proj." + suffix, item[half:]),
                ]
            else:
                parts.append((owner + ".down_proj." + suffix, item))
        return parts
    # Native generic gate/up splitting is valid after canonicalizing transpose.
    return original(adapter, name, value)


def commit_staging(adapter):
    """Convert canonical wire FN into runtime FNUZ, updating weights and scale."""
    import torch

    plan = runtime_plan(adapter)
    if plan is None or not plan["staged"]:
        return
    from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz

    for name, (module, attr, transpose) in plan["modules"].items():
        destination = getattr(module, attr)
        if destination.dtype != torch.float8_e4m3fnuz:
            continue
        if name not in plan["staged"] or name + "_scale_inv" not in plan["staged"]:
            raise RuntimeError(f"Incomplete FP8 weight/scale staging pair: {name}")
        weight, scales, _ = normalize_e4m3fn_to_e4m3fnuz(
            plan["staged"][name].view(torch.float8_e4m3fn),
            plan["staged"][name + "_scale_inv"],
        )
        scale_destination = getattr(module, attr + "_scale_inv")
        (destination.T if transpose else destination).copy_(weight)
        (scale_destination.T if transpose else scale_destination).copy_(scales)


@contextmanager
def refresh_context(adapter):
    """Native MLA post_load expects checkpoint orientation, before GEMM packing."""
    import torch

    plan = runtime_plan(adapter)
    if plan is None:
        yield
        return
    saved = []
    refresh = getattr(
        plan["model"].post_load_weights, "__func__", plan["model"].post_load_weights
    )
    normalizes = getattr(refresh, "__globals__", {}).get("_is_fp8_fnuz", False)
    try:
        for name, (module, attr, transpose) in plan["modules"].items():
            if not name.endswith(".kv_b_proj.weight"):
                continue
            weight, scale = module.weight, module.weight_scale_inv
            saved.append((module, weight, scale))
            value, scales = (weight.T, scale.T) if transpose else (weight, scale)
            if normalizes and value.dtype == torch.float8_e4m3fnuz:
                value, scales = value.view(torch.float8_e4m3fn), scales * 0.5
            # SGLang may expose a ModelWeightParameter subclass.  Passing that
            # object directly to Parameter fails when its detach() returns a
            # plain Tensor; detach first while retaining the same storage.
            module.weight = torch.nn.Parameter(value.detach(), requires_grad=False)
            module.weight_scale_inv = torch.nn.Parameter(
                scales.detach(), requires_grad=False
            )
            module.weight_scale_inv.format_ue8m0 = False
        yield
    finally:
        for module, weight, scale in saved:
            module.weight, module.weight_scale_inv = weight, scale
