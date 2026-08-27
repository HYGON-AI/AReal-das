# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import types

from torch import nn


def patch_qwen_vl_visual_forward_for_packed_text(vision_model: nn.Module) -> None:
    """Prevent AReaL packed-text metadata from entering a Qwen-VL vision tower."""
    if getattr(vision_model, "_areal_packed_text_kwargs_filtered", False):
        return

    original_forward = vision_model.forward

    def patched_visual_forward(self, hidden_states, grid_thw, **kwargs):
        # nhb: These fields describe packed text tokens. Qwen-VL vision must
        # derive its own visual sequence lengths from image_grid_thw.
        for key in (
            "cu_seqlens",
            "max_seqlen",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "max_length_q",
            "max_length_k",
        ):
            kwargs.pop(key, None)

        return original_forward(hidden_states, grid_thw, **kwargs)

    vision_model.forward = types.MethodType(patched_visual_forward, vision_model)
    vision_model._areal_packed_text_kwargs_filtered = True