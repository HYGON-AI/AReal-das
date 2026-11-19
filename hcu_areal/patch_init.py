# SPDX-License-Identifier: Apache-2.0

"""AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"""

import os as _os
from typing import TYPE_CHECKING as _TYPE_CHECKING

# A staged entry links to the upstream package without copying its files.
# Resolve that link for imports so tools such as Hydra see the same real path
# for a module's __file__ and its source code.
_upstream_package = _os.path.join(_os.path.dirname(__file__), "_upstream")
if _os.path.islink(_upstream_package):
    __path__[:] = [_os.path.realpath(_upstream_package)]  # noqa: F821

_hcu_enabled = _os.environ.get("HCU_AREAL_ENABLE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
if _hcu_enabled:
    # Do this before AReaL imports torch or discovers its platform.
    from hcu_areal.infra.platforms import (
        prepare_process_environment as _prepare_hcu_environment,
    )

    _prepare_hcu_environment()
    import torch as _torch

    if _torch.version.hip is not None:
        # AReaL's infra imports and configures loggers during initialization.
        # Preserve their identities before those imports establish references.
        from hcu_areal.areal_adaptor import apply_modules as _apply_hcu_modules

        _apply_hcu_modules("areal.utils.logging")

from .version import __version__  # noqa

from .infra import (  # noqa: E402
    RolloutController,
    StalenessManager,
    TrainController,
    WorkflowExecutor,
    current_platform,
    workflow_context,
)


if _TYPE_CHECKING:
    from .trainer import DPOTrainer, PPOTrainer, RWTrainer, SFTTrainer


def __getattr__(name: str):
    if name in ("DPOTrainer", "PPOTrainer", "RWTrainer", "SFTTrainer"):
        from .trainer import DPOTrainer, PPOTrainer, RWTrainer, SFTTrainer

        _map = {
            "DPOTrainer": DPOTrainer,
            "PPOTrainer": PPOTrainer,
            "RWTrainer": RWTrainer,
            "SFTTrainer": SFTTrainer,
        }
        globals().update(_map)
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DPOTrainer",
    "PPOTrainer",
    "RolloutController",
    "RWTrainer",
    "SFTTrainer",
    "StalenessManager",
    "TrainController",
    "WorkflowExecutor",
    "current_platform",
    "workflow_context",
]

if _hcu_enabled:
    # The opt-in identifies the intended HCU environment; the HIP build check
    # avoids installing device-specific patches in a CPU/CUDA installation.
    if _torch.version.hip is not None:
        from hcu_areal.areal_adaptor import (
            apply_package_patches as _apply_hcu_package_patches,
        )

        _apply_hcu_package_patches()
