# SPDX-License-Identifier: Apache-2.0

"""HCU adaptations for areal.v2.weight_update.awex.megatron_adapter."""

from __future__ import annotations

import functools
import sys

from .glm5_awex_layout import (
    is_glm5,
    glm5_mapping_registry,
    glm5_convert_gathered,
    glm5_wire_tensor,
)
from .deepseek_awex_layout import is_deepseek, deepseek_convert_gathered
from .qwen35_awex_layout import (
    canonical_name,
    qwen35_convert_gathered,
    qwen35_mapping_registry,
    text_config,
)
from .model_support import (
    _qwen35_report,
    glm5_metadata_report,
    glm5_group_preflight,
    deepseek_metadata_report,
    deepseek_group_preflight,
)


def _dense_sync_release_grad_buffers_factory(original):
    """Reuse Megatron's grad-buffer release around a dense AWEX transfer."""

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        if self._dte_config.enabled:
            # Upstream already performs this lifecycle for DTE updates.
            return original(self, *args, **kwargs)

        models = getattr(self._engine, "model", None)
        models = models if isinstance(models, (list, tuple)) else [models]
        if not models or any(
            model is None
            or not callable(getattr(model, "offload_grad_buffers", None))
            or not callable(getattr(model, "restore_grad_buffers", None))
            for model in models
        ):
            raise RuntimeError(
                "Dense AWEX grad-buffer release requires Megatron DDP models"
            )

        import torch

        torch.cuda.synchronize()
        try:
            self._release_grad_buffers_for_separation_sync()
            print(
                "[hcu_areal] Megatron gradient buffers released for dense AWEX sync",
                file=sys.stderr,
                flush=True,
            )
            return original(self, *args, **kwargs)
        finally:
            try:
                torch.cuda.synchronize()
            finally:
                self._restore_grad_buffers_after_separation_sync()

    return wrapper


def _glm5_train_iter_factory(original):
    @functools.wraps(original)
    def wrapper(self, theta_by_id=None, consume_overrides=False):
        if not is_glm5(self, "train"):
            yield from original(self, theta_by_id, consume_overrides)
            return
        from areal.engine.megatron_utils.megatron import (
            all_gather_param,
            get_named_parameters,
        )

        engine = self._engine
        if getattr(engine, "quantization_config", None) or getattr(
            engine.config, "use_lora", False
        ):
            raise NotImplementedError(
                "GLM AWEX patch currently supports unquantized full-model training"
            )
        overrides = theta_by_id if theta_by_id is not None else {}
        experts = getattr(engine.tf_config, "num_moe_experts", None)
        registry = glm5_mapping_registry(engine.hf_config)
        for name, param in get_named_parameters(engine.model, experts):
            src = overrides.get(id(param), param)
            if src is not param:
                for attr in (
                    "tensor_model_parallel",
                    "partition_dim",
                    "partition_stride",
                ):
                    if hasattr(param, attr):
                        setattr(src, attr, getattr(param, attr))
            gathered = all_gather_param(
                name,
                src,
                fp8_direct_convert=False,
                quantization_config=None,
                duplicated_param_names=engine._duplicated_param_names,
                gated_linear_unit=bool(engine.tf_config.gated_linear_unit),
            )
            for hf_name, tensor in glm5_convert_gathered(name, gathered, registry):
                if (
                    getattr(engine.hf_config, "tie_word_embeddings", False)
                    and hf_name == "lm_head.weight"
                ):
                    continue
                yield hf_name, glm5_wire_tensor(hf_name, tensor.detach())
            if consume_overrides:
                overrides.pop(id(param), None)

    return wrapper


def _has_active_mtp(engine):
    """Check the built model, not the stale pre-provider HF-derived config."""
    mcore = getattr(engine, "mcore_config", None)
    enabled = getattr(mcore, "enable_mtp", None)
    if enabled is True:
        return True
    for model in getattr(engine, "model", None) or ():
        seen = set()
        while model is not None and id(model) not in seen:
            seen.add(id(model))
            if (
                getattr(model, "mtp_process", False)
                or getattr(model, "mtp", None) is not None
            ):
                return True
            model = getattr(model, "module", None)
    # When the engine does not expose the effective setting, fail closed.
    return enabled is not False and bool(getattr(engine.tf_config, "mtp_num_layers", 0))


def _is_qwen35_megatron(adapter):
    return adapter._engine.hf_config.model_type in {"qwen3_5", "qwen3_5_moe"}


def _qwen35_train_iter(adapter, theta_by_id, consume_overrides):
    from areal.engine.megatron_utils.megatron import (
        all_gather_param,
        get_named_parameters,
        remove_padding,
    )
    from megatron.core import parallel_state as mpu

    engine = adapter._engine
    if engine.quantization_config or engine.config.use_lora or _has_active_mtp(engine):
        raise NotImplementedError(
            "Qwen3.5 AWEX requires unquantized full-model training with MTP disabled"
        )
    hf_config = engine.hf_config
    config = engine.model[0].config
    vocab_size = text_config(hf_config).vocab_size
    vision_heads = hf_config.vision_config.num_heads
    tp_size = mpu.get_tensor_model_parallel_world_size()
    registry = qwen35_mapping_registry(hf_config.model_type)
    overrides = theta_by_id if theta_by_id is not None else {}
    for name, param in get_named_parameters(engine.model, config.num_moe_experts):
        src = overrides.get(id(param), param)
        if src is not param:
            for attr in ("tensor_model_parallel", "partition_dim", "partition_stride"):
                if hasattr(param, attr):
                    setattr(src, attr, getattr(param, attr))
        gathered = all_gather_param(
            name,
            src,
            duplicated_param_names=engine._duplicated_param_names,
            gated_linear_unit=".language_model." in name,
        )
        gathered = remove_padding(name, gathered, vocab_size)
        for hf_name, tensor in qwen35_convert_gathered(
            name, gathered, registry, config, vision_heads, tp_size
        ):
            if hf_name == "lm_head.weight" and hf_config.tie_word_embeddings:
                continue
            if hf_name.endswith(".linear_attn.A_log"):
                tensor = tensor.float()
            yield canonical_name(hf_name), tensor.detach()
        if consume_overrides:
            overrides.pop(id(param), None)


def deepseek_train_iter(adapter, theta_by_id=None, consume_overrides=False):
    import torch
    from areal.engine.megatron_utils.megatron import (
        all_gather_param,
        get_named_parameters,
    )

    engine = adapter._engine
    if getattr(engine.config, "use_lora", False):
        raise NotImplementedError("DeepSeek AWEX patch requires full-model training")
    from hcu_areal.v2.weight_update.awex.deepseek_fp8 import (
        target_config,
        validate_quant_config,
        source_scale_map,
        compute_source,
        export_weight,
    )

    quantization = target_config(engine)
    source_quantization = validate_quant_config(
        getattr(engine, "quantization_config", None)
    )
    source_scales = source_scale_map(engine.model)
    scale_ids = {id(scale) for scale in source_scales.values()}
    if _has_active_mtp(engine):
        raise NotImplementedError(
            "DeepSeek AWEX patch requires active MTP training to be disabled"
        )
    overrides = theta_by_id if theta_by_id is not None else {}
    experts = getattr(engine.tf_config, "num_moe_experts", None)
    for name, param in get_named_parameters(engine.model, experts):
        if id(param) in scale_ids:
            continue  # Export scales from the gathered logical weights, not as model weights.
        src = overrides.get(id(param), param)
        if src is not param:
            for attr in ("tensor_model_parallel", "partition_dim", "partition_stride"):
                if hasattr(param, attr):
                    setattr(src, attr, getattr(param, attr))
        src = compute_source(src, source_quantization, source_scales)
        gathered = all_gather_param(
            name,
            src,
            fp8_direct_convert=False,
            quantization_config=source_quantization,
            duplicated_param_names=engine._duplicated_param_names,
            gated_linear_unit=bool(engine.tf_config.gated_linear_unit),
        )
        for hf_name, tensor in deepseek_convert_gathered(name, gathered):
            if (
                getattr(engine.hf_config, "tie_word_embeddings", False)
                and hf_name == "lm_head.weight"
            ):
                continue
            if tensor.dtype not in (torch.bfloat16, torch.float32):
                raise NotImplementedError(
                    f"DeepSeek BF16 AWEX unsupported dtype: {hf_name} {tensor.dtype}"
                )
            vocab = getattr(engine.hf_config, "vocab_size", None)
            if vocab and hf_name in ("model.embed_tokens.weight", "lm_head.weight"):
                if tensor.shape[0] < vocab:
                    raise ValueError(
                        f"DeepSeek vocabulary tensor is too short: {hf_name}"
                    )
                tensor = tensor[:vocab]
            # R1 has no DSA indexer: retain each parameter's original precision,
            # including the FP32 expert correction bias.
            yield from export_weight(hf_name, tensor.detach(), quantization)
        if consume_overrides:
            overrides.pop(id(param), None)


def _enabled(models, name):
    return models is None or name in models


def _train_iter_factory(original, models=None):
    glm = _glm5_train_iter_factory(original)

    @functools.wraps(original)
    def wrapper(self, theta_by_id=None, consume_overrides=False):
        if _enabled(models, "qwen35") and _is_qwen35_megatron(self):
            yield from _qwen35_train_iter(self, theta_by_id, consume_overrides)
        elif _enabled(models, "deepseek") and is_deepseek(self, "train"):
            yield from deepseek_train_iter(self, theta_by_id, consume_overrides)
        elif _enabled(models, "glm5"):
            yield from glm(self, theta_by_id, consume_overrides)
        else:
            yield from original(self, theta_by_id, consume_overrides)

    return wrapper


def _train_metadata_factory(original, models=None):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if _enabled(models, "qwen35") and _is_qwen35_megatron(self):
            _qwen35_report("train", self, result)
        if _enabled(models, "glm5"):
            glm5_metadata_report("train", self, result)
        if _enabled(models, "deepseek"):
            deepseek_metadata_report("train", self, result)
        return result

    return wrapper


def _train_preflight_factory(original, models=None):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        if _enabled(models, "deepseek") and is_deepseek(self, "train"):
            deepseek_group_preflight(self, args, kwargs)
        if _enabled(models, "glm5") and is_glm5(self, "train"):
            glm5_group_preflight(self, args, kwargs)
        return original(self, *args, **kwargs)

    return wrapper
