# SPDX-License-Identifier: Apache-2.0

"""Fallback for environments without Megatron fused weight gradients."""

from __future__ import annotations

import functools
import importlib
import inspect


MODULE = "areal.models.mcore.registry"


class _BridgeProxy:
    def __init__(self, bridge, logger):
        self._bridge = bridge
        self._logger = logger

    def __getattr__(self, name):
        return getattr(self._bridge, name)

    def to_megatron_provider(self, *args, **kwargs):
        provider = self._bridge.to_megatron_provider(*args, **kwargs)
        if (
            getattr(provider, "experimental_attention_variant", None) == "dsv4_hybrid"
            and provider.dsa_indexer_loss_coeff is None
        ):
            # The packed CSA path expects a numeric auxiliary-loss coefficient.
            provider.dsa_indexer_loss_coeff = 0.0
        try:
            importlib.import_module("fused_weight_gradient_mlp_cuda")
        except ImportError:
            provider.gradient_accumulation_fusion = False
            self._logger.warning(
                "fused_weight_gradient_mlp_cuda is unavailable; "
                "disabling gradient_accumulation_fusion."
            )
        return provider


def _make_model_factory(original):
    signature = inspect.signature(original)
    logger = original.__globals__["logger"]

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        bridge = bound.arguments.get("bridge")
        if (
            bridge is not None
            and bound.arguments.get("bridge_type") == "megatron-bridge"
        ):
            bound.arguments["bridge"] = _BridgeProxy(bridge, logger)
        return original(*bound.args, **bound.kwargs)

    return wrapper
