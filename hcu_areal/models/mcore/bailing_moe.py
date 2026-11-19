# SPDX-License-Identifier: Apache-2.0

"""Forward newer Megatron RoPE keywords in AReaL's Bailing MLA patch."""

from functools import wraps
import inspect

MODULE = "areal.models.mcore.bailing_moe"


def _patch_factory(original, manager):
    import megatron.core.models.common.embeddings.rope_utils as rope_utils
    from megatron.core import parallel_state
    import torch

    previous = rope_utils.apply_rotary_pos_emb
    if previous.__module__ != MODULE or not previous.__qualname__.startswith(
        "_patch_mla_thd_rope_for_cp.<locals>."
    ):
        raise RuntimeError(
            "Unexpected Bailing MLA RoPE patch; cannot safely recover original"
        )
    underlying = inspect.getclosurevars(previous).nonlocals.get("_original")
    if not callable(underlying):
        raise RuntimeError("Bailing MLA RoPE original function is unavailable")

    @wraps(underlying)
    def apply_rotary_pos_emb(
        t, freqs, config, cu_seqlens=None, mscale=1.0, cp_group=None, **kwargs
    ):
        if cp_group is None:
            cp_group = parallel_state.get_context_parallel_group()
        if (
            not config.apply_rope_fusion
            and cu_seqlens is not None
            and cp_group is not None
            and cp_group.size() > 1
            and freqs.size(0) >= 2
        ):
            max_needed = cu_seqlens[-1].item()
            if freqs.size(0) < max_needed:
                unit = freqs[1:2]
                positions = torch.arange(
                    max_needed, device=freqs.device, dtype=freqs.dtype
                )
                positions = positions.view(-1, *([1] * (unit.dim() - 1)))
                freqs = positions * unit
        return underlying(t, freqs, config, cu_seqlens, mscale, cp_group, **kwargs)

    # AReaL installs its patch while the module is imported, before the HCU
    # after-import hook can run. Replace the same references, then restore them
    # if the manager is uninstalled in a test.
    import sys

    modules = [rope_utils]
    for name in (
        "megatron.core.transformer.multi_latent_attention",
        "megatron.core.models.common.embeddings.rotary_pos_embedding",
    ):
        module = sys.modules.get(name)
        if (
            module is not None
            and getattr(module, "apply_rotary_pos_emb", None) is previous
        ):
            modules.append(module)
    for module in modules:
        module.apply_rotary_pos_emb = apply_rotary_pos_emb

    def cleanup():
        for module in modules:
            if getattr(module, "apply_rotary_pos_emb", None) is apply_rotary_pos_emb:
                module.apply_rotary_pos_emb = previous

    manager.add_cleanup(cleanup)

    @wraps(original)
    def patch_once():
        # The AReaL module already applied its patch during import.
        return None

    return patch_once
