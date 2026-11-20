# SPDX-License-Identifier: Apache-2.0

"""Install GLM DSA compatibility from the external HCU feature layer."""

from __future__ import annotations

import importlib
import inspect
import math
import sys
from functools import wraps
from typing import Any


MODULE = "areal.engine.megatron_engine"
_HCU_DSA_MODEL_TYPES = frozenset({"glm_moe_dsa"})
_HCU_DSA_PATCHED = False
_OPTIMIZER_CPU_OFFLOAD_ENV = "HCU_AREAL_OPTIMIZER_CPU_OFFLOAD_FRACTION"
_OPTIMIZER_PIN_CPU_GRADS_ENV = "HCU_AREAL_OPTIMIZER_PIN_CPU_GRADS"


def enable_core_dev_compat():
    """Restore the checkpoint factory expected by the pinned AReaL release."""
    from megatron.core.dist_checkpointing import serialization

    if not hasattr(serialization, "get_default_save_sharded_strategy"):
        serialization.get_default_save_sharded_strategy = (
            serialization.TorchDistSaveShardedStrategy
        )


def _core_dev_optimizer_factory(original):
    import torch
    from megatron.core import utils
    from megatron.core.optimizer import clip_grads

    # The image's fused L2 reduction faults on the V4 gradient buffers.
    clip_grads.multi_tensor_applier = utils.local_multi_tensor_applier
    clip_grads.l2_norm_impl = utils.local_multi_tensor_l2_norm
    clip_grads.multi_tensor_scale_impl = utils.local_multi_tensor_scale
    clip_grads.multi_tensor_scale_tensor_impl = None

    # Core dev takes views of native FP32 parameters while constructing the
    # distributed optimizer. Those views must be leaf tensors for CPU offload.
    return torch.no_grad()(original)


def optimizer_cpu_offload_fraction(env):
    """Parse the opt-in Megatron optimizer offload setting."""
    value = str(env.get(_OPTIMIZER_CPU_OFFLOAD_ENV, "")).strip()
    if not value:
        return None
    try:
        fraction = float(value)
    except ValueError as exc:
        raise ValueError(f"{_OPTIMIZER_CPU_OFFLOAD_ENV} must be in (0, 1]") from exc
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError(f"{_OPTIMIZER_CPU_OFFLOAD_ENV} must be in (0, 1]")
    return fraction


def optimizer_pin_cpu_grads(env):
    """Parse an optional override for optimizer gradient pinning."""
    value = str(env.get(_OPTIMIZER_PIN_CPU_GRADS_ENV, "")).strip().lower()
    if not value:
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{_OPTIMIZER_PIN_CPU_GRADS_ENV} must be a boolean")


def _optimizer_cpu_offload_config_factory(original, fraction, pin_cpu_grads=None):
    """Pass offload options through AReaL's Megatron config boundary."""

    @wraps(original)
    def optimizer_config(*args, **kwargs):
        expected = {
            "optimizer_cpu_offload": True,
            "optimizer_offload_fraction": fraction,
        }
        if pin_cpu_grads is not None:
            expected["pin_cpu_grads"] = pin_cpu_grads
        for name, value in expected.items():
            if name in kwargs and kwargs[name] != value:
                raise ValueError(f"Conflicting Megatron optimizer option {name}")
        kwargs.update(expected)
        config = original(*args, **kwargs)
        print(
            f"[hcu_areal] Megatron optimizer CPU offload enabled: "
            f"fraction={fraction}, pin_cpu_grads={pin_cpu_grads}",
            file=sys.stderr,
            flush=True,
        )
        return config

    return optimizer_config


def requires_hcu_dsa_patches(model_type: str | None) -> bool:
    """Return whether a model architecture requires HCU DSA patches."""
    return model_type in _HCU_DSA_MODEL_TYPES


def apply_hcu_dsa_patches() -> None:
    """Install HCU Megatron DSA patches once in the current process."""
    global _HCU_DSA_PATCHED
    if _HCU_DSA_PATCHED:
        return

    try:
        dsa_feature_module = importlib.import_module(
            "hcu_megatron.features_manager.transformer.dsa_feature"
        )
        patch_utils_module = importlib.import_module("hcu_megatron.patch_utils")
    except ModuleNotFoundError as exc:
        if exc.name == "hcu_megatron":
            raise RuntimeError(
                "HCU DSA support for model_type='glm_moe_dsa' requires the "
                "optional hcu_megatron package. Install the HCU environment "
                "described in requirements-hcu.txt before launching this model."
            ) from exc
        raise

    patches_manager: Any = patch_utils_module.MegatronPatchesManager
    dsa = importlib.import_module(
        "megatron.core.transformer.experimental_attention_variant.dsa"
    )
    if "up_v_weight" in inspect.signature(dsa.DSAttention.forward).parameters:
        # New Core handles packed and absorbed MLA layouts itself. The HYGON
        # legacy forward/indexer wrappers predate that interface and unpack
        # tensors differently; retain only its HCU Hadamard implementation.
        hcu_dsa = importlib.import_module(
            "hcu_megatron.core.transformer.experimental_attention_variant.dsa"
        )
        patches_manager.register_patch(
            "megatron.core.transformer.experimental_attention_variant.dsa.rotate_activation",
            hcu_dsa.rotate_activation,
        )
    else:
        dsa_feature: Any = dsa_feature_module.DSAFeature()
        dsa_feature.register_patches(patches_manager, args=None)
    patches_manager.apply_patches()
    _HCU_DSA_PATCHED = True


def _configure_pipeline_factory(original):
    @wraps(original)
    def wrapper(parallel_strategy, hf_config, tf_config):
        result = original(parallel_strategy, hf_config, tf_config)
        if requires_hcu_dsa_patches(getattr(hf_config, "model_type", None)):
            apply_hcu_dsa_patches()
        return result

    return wrapper
