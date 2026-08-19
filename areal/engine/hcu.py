# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Optional HCU-specific Megatron integration."""

from __future__ import annotations

import importlib
from typing import Any

_HCU_DSA_MODEL_TYPES = frozenset({"glm_moe_dsa"})
_HCU_DSA_PATCHED = False


def requires_hcu_dsa_patches(model_type: str | None) -> bool:
    """Return whether a model architecture requires HCU DSA patches."""
    return model_type in _HCU_DSA_MODEL_TYPES


def apply_hcu_dsa_patches() -> None:
    """Apply the optional HCU DSA compatibility patches exactly once.

    Imports are intentionally deferred so CUDA, NPU, and non-DSA Megatron users
    can import and use AReaL without installing the HCU package.
    """
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

    dsa_feature: Any = dsa_feature_module.DSAFeature()
    patches_manager: Any = patch_utils_module.MegatronPatchesManager
    dsa_feature.register_patches(patches_manager, args=None)
    patches_manager.apply_patches()
    _HCU_DSA_PATCHED = True