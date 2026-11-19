# SPDX-License-Identifier: Apache-2.0

"""Keep Qwen3-VL vision boundaries separate from text packing metadata."""


def _remove_text_packing_from_vision(module, args, kwargs):
    for key in (
        "cu_seqlens",
        "cu_seq_lens_q",
        "cu_seq_lens_k",
        "max_seqlen",
        "max_length_q",
        "max_length_k",
    ):
        kwargs.pop(key, None)
    return args, kwargs


def patch_qwen3_vl_vision_text_packing(model):
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    if hasattr(modeling_qwen3_vl, "get_vision_cu_seqlens"):
        model.model.visual.register_forward_pre_hook(
            _remove_text_packing_from_vision, with_kwargs=True
        )
