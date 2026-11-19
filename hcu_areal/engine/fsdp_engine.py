# SPDX-License-Identifier: Apache-2.0

"""Model creation and weight-name compatibility for the AReaL FSDP engine."""

from functools import wraps

MODULE = "areal.engine.fsdp_engine"


def _gemma3_xccl_names_factory(original):
    """Restore SGLang's SigLIP namespace for flattened Transformers models."""

    @wraps(original)
    def get_model_name_parameters(self, meta):
        config = getattr(self, "model_config", None)
        allocation = getattr(meta, "gen_allocation", None)
        enabled = (
            getattr(self, "is_vision_model", False)
            and getattr(config, "model_type", None) == "gemma3"
            and getattr(allocation, "backend", None) == "sglang"
        )
        for name, param in original(self, meta):
            if (
                enabled
                and name.startswith("vision_tower.")
                and not name.startswith("vision_tower.vision_model.")
            ):
                name = "vision_tower.vision_model." + name[len("vision_tower.") :]
            yield name, param

    return get_model_name_parameters


def _create_device_model_factory(original):
    @wraps(original)
    def create_device_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        from areal.engine.core.model import is_qwen3_vl_model

        if self.is_vision_model and is_qwen3_vl_model(self.model_config.model_type):
            from hcu_areal.models.transformers.qwen3_vl import (
                patch_qwen3_vl_vision_text_packing,
            )

            patch_qwen3_vl_vision_text_packing(self.model)
        return result

    return create_device_model
