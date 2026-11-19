# SPDX-License-Identifier: Apache-2.0

"""Gemma3-4B multimodal, FSDP DP4 -> SGLang TP4 layout rules.

Only the Gemma3ForConditionalGeneration model is targeted.  Rules are based on
observed 883/883 rank metadata *names* and the SGLang gemma3_mm / siglip
implementation.  Shape and dtype are checked before NCCL by the Feature.

No GPU modules are imported at interpreter startup.
"""

from __future__ import annotations

import os
import re


def is_gemma3(adapter, role: str) -> bool:
    if os.getenv("HCU_AREAL_DISABLE_GEMMA3_AWEX", "").lower() in (
        "1",
        "true",
        "on",
        "yes",
    ):
        return False
    if role == "train":
        engine = getattr(adapter, "_engine", None)
        config = getattr(engine, "model_config", None)
    elif role == "infer":
        model = adapter._get_model()
        config = getattr(model, "config", None)
    else:
        raise ValueError(f"Unknown AWEX role: {role}")
    if config is None:
        return False
    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type != "gemma3":
        return False
    # Gemma3 multimodal uses model_type="gemma3"; text-only is "gemma3_text",
    # Gemma3n is "gemma3n". Do not depend on optional `architectures` being set.
    return True


def gemma3_train_to_infer_name(name: str) -> str:
    """Map an HF/FSDP parameter key into the SGLang unfused key namespace."""
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model.") :]
    if name.startswith("model.vision_tower."):
        mapped = name[len("model.") :]
        return mapped.replace(".self_attn.out_proj.", ".self_attn.proj.")
    if name.startswith("model.multi_modal_projector."):
        return name[len("model.") :]
    return name


_VISION_QKV = re.compile(
    r"^vision_tower\.vision_model\.encoder\.layers\.\d+\.self_attn\.qkv_proj\.(weight|bias)$"
)


def gemma3_inference_parts(name, tensor):
    """Vision QKV is always Q/K/V in equal thirds, not the TEXT head ratio.

    Returns views into the original fused parameter. Other parameters defer to
    the AReaL _unfuse_params implementation (e.g. language_model QKV & gate/up).
    """
    if not _VISION_QKV.fullmatch(name):
        return None
    if tensor.ndim < 1 or int(tensor.shape[0]) % 3:
        raise ValueError(
            f"Gemma3 Vision QKV not divisible by 3: {name}, {tuple(tensor.shape)}"
        )
    parts = tensor.chunk(3, dim=0)
    return [
        (name.replace(".qkv_proj.", f".{q}_proj."), view)
        for q, view in zip(("q", "k", "v"), parts)
    ]


_ROW_VISION = re.compile(
    r"^vision_tower\.vision_model\.encoder\.layers\.\d+\."
    r"(?:self_attn\.proj|mlp\.fc2)\.(weight|bias)$"
)


class Gemma3ShardingProxy:
    """Correct SGLang Gemma3 rank-local tensor layouts for TP4.

    Qwen3.5 uses its own proxy. All unrelated models return the original proxy.
    """

    def __init__(self, base, tp_size: int):
        self._base = base
        self.tp_size = int(tp_size)
        if self.tp_size < 1:
            raise ValueError("Gemma3 TP size must be positive")

    def get_sharding_strategy(self, name, **kwargs):
        from awex.sharding.param_sharding import ShardingType

        no = (ShardingType.NO_SHARDING, 0, 1)
        if name == "language_model.model.embed_tokens.weight":
            # Gemma3TextScaledWordEmbedding is torch.nn.Embedding, not vocab-TP.
            return no
        if name == "multi_modal_projector.mm_input_projection_weight":
            # Plain nn.Parameter in Gemma3MultiModalProjector, replicated.
            return no
        if name.startswith("multi_modal_projector.mm_soft_emb_norm."):
            return no
        if name.startswith("vision_tower.vision_model.embeddings.patch_embedding."):
            # Plain Conv2dLayer; weight/bias replicated, not TP.
            return no
        if name.startswith("vision_tower.vision_model.post_layernorm."):
            return no
        m = _ROW_VISION.fullmatch(name)
        if m:
            if m.group(1) == "bias":
                return no
            if self.tp_size > 1:
                return (ShardingType.TP_SHARDING, 1, self.tp_size)
            return no
        # Vision fc1 and q/k/v are column-parallel; language model uses
        # ordinary column/row rules. Position embedding is vocab-TP.
        return self._base.get_sharding_strategy(name, **kwargs)

    def __getattr__(self, name):
        return getattr(self._base, name)


def assert_gemma3_transfer_metadata(infer_meta, train_meta):
    """Fail closed if names, global shapes or wire dtypes do not match.

    AWEX otherwise builds an overlapping-region plan which can silently omit
    ranges when metadata claims an incorrect global size. A separate check for
    full shard/replica coverage is deliberately NOT inferred from merged meta
    unless that merge's semantics have been validated on bw67.
    """

    def to_index(entries, role):
        index = {}
        for item in entries:
            if item.name in index:
                raise ValueError(f"Gemma3 AWEX duplicate {role} name: {item.name}")
            index[item.name] = item
        return index

    inf = to_index(infer_meta, "infer")
    trn = to_index(train_meta, "train")
    missing = sorted(set(trn) - set(inf))
    extra = sorted(set(inf) - set(trn))
    bad_shapes = [
        (name, tuple(trn[name].global_shape), tuple(inf[name].global_shape))
        for name in sorted(set(trn) & set(inf))
        if tuple(trn[name].global_shape) != tuple(inf[name].global_shape)
    ]
    bad_dtypes = [
        (name, str(trn[name].dtype), str(inf[name].dtype))
        for name in sorted(set(trn) & set(inf))
        if str(trn[name].dtype) != str(inf[name].dtype)
    ]
    if missing or extra or bad_shapes or bad_dtypes:
        raise ValueError(
            "Gemma3 AWEX metadata preflight FAILED: "
            f"train={len(trn)} infer={len(inf)} "
            f"missing={len(missing)} {missing[:8]} "
            f"extra={len(extra)} {extra[:8]} "
            f"shape_mismatch={len(bad_shapes)} {bad_shapes[:8]} "
            f"dtype_mismatch={len(bad_dtypes)} {bad_dtypes[:8]}"
        )
    return len(trn)
