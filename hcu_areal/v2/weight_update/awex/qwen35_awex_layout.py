# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 dense FSDP -> SGLang AWEX canonical weight layout.

No device imports at startup. Logic applies only to Qwen3.5 and is tested against
8 real metadata traces for 4x FSDP and 4x SGLang tensor parallelism.
"""

from __future__ import annotations
import re

_TRUE = {"1", "true", "yes", "on"}


def enabled(v):
    return str(v).strip().lower() in _TRUE


def is_qwen35(adapter, role):
    engine = getattr(adapter, "_engine", None) if role == "train" else None
    model = getattr(engine, "model", None) if engine is not None else None
    cfg = getattr(engine, "model_config", None) if engine is not None else None
    if role == "infer":
        get_model = getattr(adapter, "_get_model", None)
        model = get_model() if callable(get_model) else None
        cfg = getattr(model, "config", None)
    if cfg is None:
        cfg = getattr(model, "config", None)
    vals = [getattr(cfg, "model_type", ""), type(model).__name__]
    vals += list(getattr(cfg, "architectures", ()) or ())
    sub = getattr(cfg, "text_config", None)
    if sub is not None:
        vals.append(getattr(sub, "model_type", ""))
    return any("qwen3_5" in str(v).lower() or "qwen3.5" in str(v).lower() for v in vals)


def text_config(cfg):
    sub = getattr(cfg, "text_config", None)
    return sub if sub is not None else cfg


def key_value_dims(cfg):
    cfg = text_config(cfg)
    return (
        int(cfg.linear_num_key_heads) * int(cfg.linear_key_head_dim),
        int(cfg.linear_num_value_heads) * int(cfg.linear_value_head_dim),
    )


def canonical_name(name):
    if name.startswith("model.model."):
        name = name.replace("model.model.", "model.", 1)
    if name.startswith("model.language_model."):
        name = name.replace("model.language_model.", "model.", 1)
    if name.startswith("model.visual."):
        name = name.replace("model.visual.", "visual.", 1)
    name = re.sub(r"(model\.layers\.\d+)\.self_attn\.", r"\1.", name)
    return name


def segments_for_training(name, shape, cfg):
    """Return [(target_name, row_start, row_count)] for a complete global parameter.

    Each target is a contiguous portion of the original dim-0 tensor. Shards
    crossing a segment boundary are clipped without gathering the full model.
    """
    name = canonical_name(name)
    if ".linear_attn.in_proj_qkv.weight" in name:
        k, v = key_value_dims(cfg)
        sizes = (k, k, v)
        names = (
            name.replace("in_proj_qkv", "in_proj_q"),
            name.replace("in_proj_qkv", "in_proj_k"),
            name.replace("in_proj_qkv", "in_proj_v"),
        )
    elif ".linear_attn.conv1d.weight" in name:
        k, v = key_value_dims(cfg)
        sizes = (k, k, v)
        names = (
            name.replace("conv1d", "conv1d_q"),
            name.replace("conv1d", "conv1d_k"),
            name.replace("conv1d", "conv1d_v"),
        )
    elif re.search(r"^visual\.blocks\.\d+\.attn\.qkv\.(weight|bias)$", name):
        if shape[0] % 3:
            raise ValueError(f"Vision QKV not divisible by 3: {name} {shape}")
        sizes = (shape[0] // 3,) * 3
        names = (name.replace(".qkv.", f".{part}_proj.") for part in ("q", "k", "v"))
    else:
        return [(name, 0, shape[0] if shape else 1)]
    if sum(sizes) != shape[0]:
        raise ValueError(
            f"Qwen3.5 segment geometry mismatch: {name} {shape} expected rows={sum(sizes)}"
        )
    result = []
    start = 0
    for new_name, count in zip(names, sizes):
        result.append((new_name, start, count))
        start += count
    return result


def slice_segment(local_tensor, source_offset, source_shape, start, size):
    """Create a native view and logical offset for overlapping 1D row intervals.

    Unlike repacking/all-gather, each segment view remains backed by the
    original FSDP shard storage. Returns None if this rank owns no rows.
    """
    if not source_shape or source_offset[0] < 0:
        raise ValueError("Invalid shard geometry")
    local_start = int(source_offset[0])
    local_end = local_start + int(local_tensor.shape[0])
    begin, end = max(local_start, start), min(local_end, start + size)
    if end <= begin:
        return None
    view = local_tensor.narrow(0, begin - local_start, end - begin)
    logical_offset = (begin - start, *source_offset[1:])
    logical_shape = (size, *source_shape[1:])
    if any(
        int(a) + int(b) > int(c)
        for a, b, c in zip(logical_offset, view.shape, logical_shape)
    ):
        raise AssertionError("Converted source shard escaped logical target")
    return view, logical_offset, logical_shape


def inference_parts(name, tensor, cfg, tp_size):
    """Return canonical `(name, tensor_view)` pairs; None means use legacy splitter.

    The SGLang model loader uses rank-local layouts:
      GDN qkvz: [Q, K, V, Z]; ba: [B, A]; conv1d: [Q,K,V]
      gated full attention: [Q-with-gate,K,V]
      vision attention: [Q,K,V], equal thirds
    """
    name = canonical_name(name)
    cfg_text = text_config(cfg)
    if re.search(r"^visual\.blocks\.\d+\.attn\.qkv_proj\.(weight|bias)$", name):
        if tensor.shape[0] % 3:
            raise ValueError(
                f"Vision QKV local rows not divisible by 3: {name} {tuple(tensor.shape)}"
            )
        chunks = tensor.chunk(3, dim=0)
        return [
            (name.replace(".qkv_proj.", f".{s}_proj."), t)
            for s, t in zip("qkv", chunks)
        ]
    if ".linear_attn.in_proj_qkvz." in name:
        k, v = key_value_dims(cfg)
        if k % tp_size or v % tp_size:
            raise ValueError(f"GDN qkvz not divisible by tp_size={tp_size}")
        sizes = (k // tp_size, k // tp_size, v // tp_size, v // tp_size)
        if sum(sizes) != tensor.shape[0]:
            raise ValueError(
                f"GDN qkvz rows mismatch {name}: actual={tensor.shape[0]}, expected={sizes}"
            )
        chunks = tensor.split(sizes, dim=0)
        return [
            (name.replace("in_proj_qkvz", f"in_proj_{s}"), t)
            for s, t in zip(("q", "k", "v", "z"), chunks)
        ]
    if ".linear_attn.in_proj_ba." in name:
        heads = int(cfg_text.linear_num_value_heads)
        if heads % tp_size:
            raise ValueError(f"GDN ba not divisible by tp_size={tp_size}")
        sz = heads // tp_size
        if tensor.shape[0] != sz * 2:
            raise ValueError(
                f"GDN ba rows mismatch {name}: expected={sz * 2}, got={tensor.shape[0]}"
            )
        chunks = tensor.split((sz, sz), dim=0)
        return [
            (name.replace("in_proj_ba", f"in_proj_{s}"), t)
            for s, t in zip(("b", "a"), chunks)
        ]
    if ".linear_attn.conv1d.weight" in name:
        k, v = key_value_dims(cfg)
        if k % tp_size or v % tp_size:
            raise ValueError(f"GDN conv not divisible by tp_size={tp_size}")
        sizes = (k // tp_size, k // tp_size, v // tp_size)
        if sum(sizes) != tensor.shape[0]:
            raise ValueError(
                f"GDN conv rows mismatch {name}: actual={tensor.shape[0]}, expected={sizes}"
            )
        chunks = tensor.split(sizes, dim=0)
        return [
            (name.replace("conv1d", f"conv1d_{s}"), t)
            for s, t in zip(("q", "k", "v"), chunks)
        ]
    if re.search(r"^model\.layers\.\d+\.qkv_proj\.weight$", name):
        nh = int(cfg_text.num_attention_heads)
        nk = int(cfg_text.num_key_value_heads)
        hd = int(getattr(cfg_text, "head_dim", None) or cfg_text.hidden_size // nh)
        gate = 2 if getattr(cfg_text, "attn_output_gate", True) else 1
        if nh % tp_size or nk % tp_size:
            raise ValueError("Gated full attention heads not divisible by TP")
        sizes = (gate * nh // tp_size * hd, nk // tp_size * hd, nk // tp_size * hd)
        if sum(sizes) != tensor.shape[0]:
            raise ValueError(
                f"Gated QKV rows mismatch {name}: actual={tensor.shape[0]}, expected={sizes}"
            )
        chunks = tensor.split(sizes, dim=0)
        return [
            (name.replace("qkv_proj", f"{s}_proj"), t) for s, t in zip("qkv", chunks)
        ]
    return None


def infer_strategy_override(name, tp_size):
    """Return (kind, axis, parts) for corrected layouts, or None for original.

    kind is 'tp' or 'none', resolved to real Awex enum by runtime code.
    """
    name = canonical_name(name)
    if name.startswith("visual.patch_embed.proj."):
        return "none", 0, 1
    if re.search(r"^visual\.blocks\.\d+\.(attn\.proj|mlp\.linear_fc2)\.bias$", name):
        return "none", 0, 1
    if name == "visual.merger.linear_fc2.bias":
        return "none", 0, 1
    if re.search(r"^visual\.blocks\.\d+\.(attn\.proj|mlp\.linear_fc2)\.weight$", name):
        return "tp", 1, tp_size
    if name == "visual.merger.linear_fc2.weight":
        return "tp", 1, tp_size
    if ".linear_attn." in name and re.search(
        r"(in_proj_[qkvba]\.weight|conv1d_[qkv]\.weight)$", name
    ):
        return "tp", 0, tp_size
    return None
