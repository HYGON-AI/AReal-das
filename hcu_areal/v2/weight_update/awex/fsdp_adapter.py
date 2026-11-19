# SPDX-License-Identifier: Apache-2.0

"""HCU adaptations for areal.v2.weight_update.awex.fsdp_adapter."""

from __future__ import annotations

import functools
from dataclasses import replace
from math import prod

from .qwen35_awex_layout import (
    is_qwen35,
    canonical_name,
    segments_for_training,
    slice_segment,
)
from .gemma3_awex_layout import is_gemma3, gemma3_train_to_infer_name
from .qwen3_vl_layout import (
    is_qwen3_vl_type,
    is_qwen3_vl_vision_qkv,
    normalize_qwen3_vl_train_name,
    pack_vision_qkv_rank_major,
)
from .model_support import _qwen35_report, check_group_preflight


def _wire_dtype(adapter, tensor):
    """Match the exported compute dtype without requiring an AReaL patch."""
    return (
        adapter._engine._compute_dtype() if tensor.is_floating_point() else tensor.dtype
    )


def _train_iter(adapter):
    """Yield (name, view, global_offset, logical_global_shape, base_shard) per rank.

    Special-case slicing is performed on the *local* FSDP shard, not the
    globally gathered parameter. No hidden synchronizations or materialization.
    """
    from torch.distributed.tensor import DTensor

    engine = adapter._engine
    rank_info = adapter._build_rank_info()
    cfg = engine.model_config
    for raw_name, parameter in engine.model.named_parameters():
        name = canonical_name(raw_name)
        if adapter._tie_word_embeddings and name == "lm_head.weight":
            continue
        data = parameter.data
        if isinstance(data, DTensor):
            view = data._local_tensor
            meta = adapter._extract_dtensor_shard_meta(name, data, rank_info)
            global_shape = tuple(data.shape)
        else:
            view = data
            meta = adapter._extract_plain_shard_meta(name, data, rank_info)
            global_shape = tuple(data.shape)
        source_offset = tuple(meta.global_offset)
        if len(global_shape) == 0:
            raise ValueError(f"Scalar Qwen3.5 weights are not yet supported: {name}")
        for cname, start, length in segments_for_training(name, global_shape, cfg):
            part = slice_segment(view, source_offset, global_shape, start, length)
            if part is None:
                continue
            part_view, part_offset, part_shape = part
            yield cname, part_view, part_offset, part_shape, meta


def _qwen35_train_metadata_factory(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        if not is_qwen35(self, "train"):
            return original(self, *args, **kwargs)
        if args or kwargs:
            raise TypeError("Qwen3.5 get_weight_metadata unexpected arguments")
        import torch
        from awex.meta.weight_meta import ParameterMeta, ParameterReplicaMeta

        result = []
        for cname, view, offset, gshape, source_meta in _train_iter(self):
            dtype = (
                torch.float32
                if cname.endswith(".linear_attn.A_log")
                else _wire_dtype(self, view)
            )
            shard = replace(
                source_meta,
                name=cname,
                shape=tuple(view.shape),
                numel=int(view.numel()),
                global_offset=tuple(offset),
                dtype=dtype,
            )
            result.append(
                ParameterMeta(
                    name=cname,
                    global_numel=prod(gshape),
                    global_shape=tuple(gshape),
                    dtype=dtype,
                    shards=[shard],
                    replicas=[ParameterReplicaMeta(shards=[shard])],
                )
            )
        _qwen35_report("train", self, result)
        return result

    return wrapper


def _qwen35_train_params_factory(original):
    @functools.wraps(original)
    def wrapper(self, required_names=None):
        if not is_qwen35(self, "train"):
            return original(self, required_names)
        import torch

        required = set(required_names) if required_names else None
        result = {}
        for name, tensor, offset, gshape, meta in _train_iter(self):
            if required is not None and name not in required:
                continue
            # A_log is FP32 in SGLang, unlike default FSDP BF16 export.
            result[name] = (
                tensor.to(torch.float32)
                if name.endswith(".linear_attn.A_log")
                else self._engine._cast_to_compute_dtype(tensor)
            )
        return result

    return wrapper


def _gemma3_train_name_factory(original):
    @functools.wraps(original)
    def wrapper(self, name):
        converted = original(self, name)
        if not is_gemma3(self, "train"):
            return converted
        return gemma3_train_to_infer_name(converted)

    return wrapper


def _is_train(adapter):
    return is_qwen3_vl_type(getattr(adapter._engine.model_config, "model_type", None))


def _repack(adapter, tensor):
    import torch
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Shard

    tp_size = getattr(adapter, "_qwen3_vl_vision_tp_size", None)
    if tp_size is None:
        raise RuntimeError(
            "Qwen3-VL vision TP is unknown; initialize the AWEX transfer plan before exporting visual QKV tensors"
        )
    with torch.no_grad():
        if isinstance(tensor, DTensor):
            if any(isinstance(p, Shard) and p.dim != 0 for p in tensor.placements):
                raise NotImplementedError(
                    "Qwen3-VL vision QKV converter requires FSDP sharding along dim 0"
                )
            offset = adapter._compute_dtensor_offset(tensor)[0]
            shape = tuple(tensor._local_tensor.shape)
            packed = pack_vision_qkv_rank_major(tensor.full_tensor(), tp_size)
            local = packed.narrow(0, offset, shape[0]).contiguous()
            if tuple(local.shape) != shape:
                raise RuntimeError(
                    f"Qwen3-VL vision QKV local shape changed: {tuple(local.shape)} != {shape}"
                )
            return local
        return pack_vision_qkv_rank_major(tensor, tp_size)


def _qwen3vl_train_name_factory(original):
    @functools.wraps(original)
    def name(self, raw_name):
        if _is_train(self):
            return normalize_qwen3_vl_train_name(raw_name)
        return original(self, raw_name)

    return name


def _compute_dtype_metadata_factory(original):
    qwen35_active = "qwen35.awex.train.meta" in getattr(
        original, "__hcu_areal_patch_ids__", ()
    )

    @functools.wraps(original)
    def metadata(self, *args, **kwargs):
        values = original(self, *args, **kwargs)
        # The Qwen3.5 feature preserves FP32 A_log, including its metadata.
        if qwen35_active and is_qwen35(self, "train"):
            return values
        for value in values:
            if value.dtype.is_floating_point:
                dtype = self._engine._compute_dtype()
                value.dtype = dtype
                for shard in value.shards:
                    shard.dtype = dtype
                for replica in value.replicas:
                    for shard in replica.shards:
                        shard.dtype = dtype
        return values

    return metadata


def _compute_dtype_params_factory(original):
    qwen35_active = "qwen35.awex.train.params" in getattr(
        original, "__hcu_areal_patch_ids__", ()
    )

    @functools.wraps(original)
    def params(self, required_names=None):
        values = original(self, required_names)
        if qwen35_active and is_qwen35(self, "train"):
            return values
        return {
            name: self._engine._cast_to_compute_dtype(value)
            for name, value in values.items()
        }

    return params


def _qwen3vl_train_params_factory(original):
    @functools.wraps(original)
    def params(self, required_names=None):
        values = original(self, required_names)
        if _is_train(self):
            for raw_name, parameter in self._engine.model.named_parameters():
                name = self._to_hf_name(raw_name)
                if name in values and is_qwen3_vl_vision_qkv(name):
                    values[name] = self._engine._cast_to_compute_dtype(
                        _repack(self, parameter.data)
                    )
        return values

    return params


def _qwen3vl_train_init_factory(original):
    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        if _is_train(self):
            pair = args[0] if args else kwargs.get("pair_name")
            url = args[5] if len(args) > 5 else kwargs.get("kv_store_url")
            from areal.v2.weight_update.awex import fetch_kv_metadata

            infer_meta, _ = fetch_kv_metadata(url, pair)
            sizes = {
                int(shard.num_shards)
                for meta in infer_meta
                if is_qwen3_vl_vision_qkv(meta.name)
                for shard in meta.shards
            }
            if len(sizes) != 1:
                raise RuntimeError(
                    f"Cannot determine Qwen3-VL inference vision TP from AWEX metadata: {sizes}"
                )
            self._qwen3_vl_vision_tp_size = sizes.pop()
        return original(self, *args, **kwargs)

    return initialize


def _qwen3vl_train_teardown_factory(original):
    @functools.wraps(original)
    def teardown(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if _is_train(self):
            self._qwen3_vl_vision_tp_size = None
        return result

    return teardown


def _group_preflight_factory(original, role="train", models=None):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        check_group_preflight(self, args, kwargs, role, models)
        return original(self, *args, **kwargs)

    return wrapper
