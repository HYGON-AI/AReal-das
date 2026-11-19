# SPDX-License-Identifier: Apache-2.0

"""Apply engine-specific HCU patches at the RPC engine import boundary."""

from functools import wraps


def _engine_import_factory(original):
    @wraps(original)
    def import_engine(path):
        if isinstance(path, str):
            if path.startswith("areal.engine.fsdp_engine.") or path.startswith(
                "areal.engine.FSDP"
            ):
                from hcu_areal.areal_adaptor import apply_training_patches

                apply_training_patches("fsdp")
            elif path.startswith("areal.engine.megatron_engine.") or path.startswith(
                "areal.engine.Megatron"
            ):
                from hcu_areal.areal_adaptor import apply_training_patches

                apply_training_patches("megatron")
        return original(path)

    return import_engine
