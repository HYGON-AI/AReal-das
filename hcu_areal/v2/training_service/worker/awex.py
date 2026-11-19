# SPDX-License-Identifier: Apache-2.0

"""Create the weight-update adapter for the selected training engine."""


def _create_training_adapter(engine):
    # Import only the selected adapter. Importing Megatron while serving FSDP
    # would bind its model factory before the Megatron worker prepares it.
    from areal.engine.fsdp_engine import FSDPEngine

    if isinstance(engine, FSDPEngine):
        from areal.v2.weight_update.awex.fsdp_adapter import AwexFSDPAdapter

        return AwexFSDPAdapter(engine)

    from areal.engine.megatron_engine import MegatronEngine

    if isinstance(engine, MegatronEngine):
        from areal.v2.weight_update.awex.megatron_adapter import AwexMegatronAdapter

        return AwexMegatronAdapter(engine)

    raise TypeError(
        f"Unsupported engine type for weight update: {type(engine).__name__}"
    )
