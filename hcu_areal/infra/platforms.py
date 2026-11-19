# SPDX-License-Identifier: Apache-2.0

"""HCU device visibility policy at an explicitly started process boundary."""

import os


def prepare_process_environment(env=None):
    """Clear requested variables and normalize visibility before torch loads."""
    env = os.environ if env is None else env
    for name in env.get("HCU_AREAL_UNSET_ENV", "").split(","):
        if name:
            env.pop(name, None)
    if str(env.get("HCU_AREAL_ENABLE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        normalize_device_visibility(env)


def normalize_device_visibility(env=None):
    env = os.environ if env is None else env
    rocr = env.get("ROCR_VISIBLE_DEVICES")
    hip = env.get("HIP_VISIBLE_DEVICES")
    cuda = env.get("CUDA_VISIBLE_DEVICES")
    # Empty values are explicit no-device restrictions, not missing values.
    if rocr is not None and (hip is not None or cuda is not None):
        raise ValueError(
            "Do not set ROCR_VISIBLE_DEVICES together with HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES"
        )
    if hip is not None and cuda is not None and hip != cuda:
        raise ValueError("HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES must match")
    if rocr is not None:
        env["CUDA_VISIBLE_DEVICES"] = env.pop("ROCR_VISIBLE_DEVICES")
    elif hip is not None:
        env["CUDA_VISIBLE_DEVICES"] = env.pop("HIP_VISIBLE_DEVICES")
