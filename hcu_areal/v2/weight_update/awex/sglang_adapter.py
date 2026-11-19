# SPDX-License-Identifier: Apache-2.0

"""HCU adaptations for areal.v2.weight_update.awex.sglang_adapter."""

from __future__ import annotations

import functools

from .qwen35_awex_layout import (
    is_qwen35,
    canonical_name,
    inference_parts,
    infer_strategy_override,
)
from .qwen3_vl_layout import (
    is_qwen3_vl_type,
    is_qwen3_vl_vision_qkv,
    normalize_qwen3_vl_visual_infer_name,
    qwen3_vl_vision_sharding_dim,
)
from .glm5_awex_layout import is_glm5
from .deepseek_awex_layout import is_deepseek
from .model_support import (
    _qwen35_report,
    glm5_metadata_report,
    deepseek_metadata_report,
    check_group_preflight,
)


def _model_context_factory(original):
    @functools.wraps(original)
    def model_context(self):
        result = original(self)
        ps = getattr(self._scheduler, "ps", None)
        if ps is None:
            return result
        for name in (
            "tp_rank",
            "tp_size",
            "pp_rank",
            "pp_size",
            "dp_size",
            "attn_tp_rank",
            "attn_tp_size",
            "attn_dp_rank",
        ):
            result[name] = int(getattr(ps, name))
        upstream = original.__globals__
        result["local_rank"] = int(upstream["os"].environ.get("LOCAL_RANK", ps.gpu_id))
        dist = upstream["dist"]
        if not (dist.is_available() and dist.is_initialized()):
            result["global_rank"] = int(ps.pp_rank * ps.tp_size + ps.tp_rank)
        return result

    return model_context


def _infer_parts_factory(original, models=None):
    @functools.wraps(original)
    def wrapper(self, name, tensor):
        from hcu_areal.v2.weight_update.awex.deepseek_awex_layout import (
            is_deepseek,
            deepseek_inference_parts,
            is_deepseek_runtime_kv_scale,
        )

        if (models is None or "deepseek" in models) and is_deepseek(self, "infer"):
            if is_deepseek_runtime_kv_scale(name):
                return []
            cfg = validate_inference_model(self).config
            from hcu_areal.v2.weight_update.awex.deepseek_fp8 import (
                inference_parts as fp8_parts,
            )

            converted = fp8_parts(self, name, tensor, original)
            if converted is not None:
                return converted
            converted = deepseek_inference_parts(name, tensor, cfg)
            return converted if converted is not None else original(self, name, tensor)
        from hcu_areal.v2.weight_update.awex.glm5_awex_layout import (
            is_glm5,
            glm5_inference_parts,
        )

        if (models is None or "glm5" in models) and is_glm5(self, "infer"):
            converted = glm5_inference_parts(name, tensor, self._get_model().config)
            return converted if converted is not None else original(self, name, tensor)
        # Share the existing, already-tested inference hook with Gemma3.
        # Keep the Qwen3.5 execution path unchanged and model-guarded.
        from hcu_areal.v2.weight_update.awex.gemma3_awex_layout import (
            is_gemma3,
            gemma3_inference_parts,
        )

        if (models is None or "gemma3" in models) and is_gemma3(self, "infer"):
            converted = gemma3_inference_parts(name, tensor)
            return converted if converted is not None else original(self, name, tensor)
        if (models is not None and "qwen35" not in models) or not is_qwen35(
            self, "infer"
        ):
            return original(self, name, tensor)
        ctx = self._get_model_context()
        cfg = self._get_model().config
        converted = inference_parts(name, tensor, cfg, int(ctx["tp_size"]))
        if converted is not None:
            return converted
        return [
            (canonical_name(key), value) for key, value in original(self, name, tensor)
        ]

    return wrapper


class _StrategyProxy:
    def __init__(self, base, tp_size):
        self._base = base
        self._tp_size = tp_size

    def get_sharding_strategy(self, name, **kwargs):
        chosen = infer_strategy_override(name, self._tp_size)
        if chosen is None:
            return self._base.get_sharding_strategy(name, **kwargs)
        from awex.sharding.param_sharding import ShardingType

        kind, axis, parts = chosen
        return (
            ShardingType.TP_SHARDING if kind == "tp" else ShardingType.NO_SHARDING,
            axis,
            parts,
        )

    def __getattr__(self, k):
        return getattr(self._base, k)


def _infer_strategy_factory(original, models=None):
    @functools.wraps(original)
    def wrapper(self, rank_info):
        base = original(self, rank_info)
        from hcu_areal.v2.weight_update.awex.deepseek_awex_layout import (
            is_deepseek,
            DeepseekShardingProxy,
        )

        if (models is None or "deepseek" in models) and is_deepseek(self, "infer"):
            validate_inference_model(self)
            return DeepseekShardingProxy(base)
        from hcu_areal.v2.weight_update.awex.glm5_awex_layout import (
            is_glm5,
            Glm5ShardingProxy,
        )

        if (models is None or "glm5" in models) and is_glm5(self, "infer"):
            return Glm5ShardingProxy(base)
        # Model-specific sharding logic lives in its own Feature module.
        # Do not change any of the validated Qwen3.5 branches.
        from hcu_areal.v2.weight_update.awex.gemma3_awex_layout import (
            is_gemma3,
            Gemma3ShardingProxy,
        )

        if (models is None or "gemma3" in models) and is_gemma3(self, "infer"):
            return Gemma3ShardingProxy(base, int(rank_info.tp_size))
        if (models is not None and "qwen35" not in models) or not is_qwen35(
            self, "infer"
        ):
            return base
        return _StrategyProxy(base, int(rank_info.tp_size))

    return wrapper


def _infer_metadata_factory(original, models=None):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        if models is None or "qwen35" in models:
            _qwen35_report("infer", self, value)
        if models is None or "glm5" in models:
            glm5_metadata_report("infer", self, value)
        if models is None or "deepseek" in models:
            deepseek_metadata_report("infer", self, value)
        return value

    return wrapper


def _is_infer(adapter):
    return is_qwen3_vl_type(getattr(adapter._get_model().config, "model_type", None))


class _VisionStrategy:
    def __init__(self, base, rank_info, server_args):
        self._base, self._rank_info, self._server_args = base, rank_info, server_args

    def __getattr__(self, name):
        return getattr(self._base, name)

    def get_sharding_strategy(self, name, **kwargs):
        if not name.startswith("model.visual."):
            return self._base.get_sharding_strategy(name, **kwargs)
        from awex.sharding.param_sharding import ShardingType

        axis = qwen3_vl_vision_sharding_dim(name)
        tp = int(self._rank_info.attn_tp_size)
        if axis is None or tp == 1:
            result = ShardingType.NO_SHARDING, 0, 1
        else:
            kind = (
                ShardingType.DP_TP_SHARDING
                if self._server_args.enable_dp_attention
                else ShardingType.TP_SHARDING
            )
            result = kind, axis, tp
        if is_qwen3_vl_vision_qkv(name) and (result[1] != 0 or int(result[2]) != tp):
            raise RuntimeError(
                f"Invalid visual QKV sharding strategy for {name}: {result}"
            )
        return result


def _qwen3vl_infer_parts_factory(original):
    @functools.wraps(original)
    def parts(self, name, tensor):
        if _is_infer(self) and name.startswith("visual."):
            return [(normalize_qwen3_vl_visual_infer_name(name), tensor)]
        return original(self, name, tensor)

    return parts


def _qwen3vl_infer_strategy_factory(original):
    @functools.wraps(original)
    def strategy(self, rank_info):
        result = original(self, rank_info)
        if _is_infer(self):
            return _VisionStrategy(result, rank_info, self._scheduler.server_args)
        return result

    return strategy


def _qwen3vl_infer_metadata_factory(original):
    @functools.wraps(original)
    def metadata(self, *args, **kwargs):
        if _is_infer(self) and getattr(
            self._scheduler.server_args, "mm_enable_dp_encoder", False
        ):
            raise NotImplementedError(
                "Qwen3-VL AWEX visual QKV repacking requires mm_enable_dp_encoder=False"
            )
        return original(self, *args, **kwargs)

    return metadata


def validate_inference_model(adapter):
    model = adapter._get_model()
    if getattr(model, "quant_config", None) is not None:
        from hcu_areal.v2.weight_update.awex.deepseek_fp8 import runtime_plan

        runtime_plan(adapter)
    architectures = getattr(model.config, "architectures", ()) or ()
    if (
        any("NextN" in name for name in architectures)
        or "NextN" in type(model).__name__
    ):
        raise NotImplementedError(
            "DeepSeek AWEX patch does not yet support active MTP/NextN inference"
        )
    return model


def _glm5_mla_refresh(adapter):
    model = adapter._get_model()
    refresh = getattr(model, "post_load_weights", None)
    if not callable(refresh):
        raise RuntimeError(
            "GLM AWEX requires SGLang post_load_weights for MLA cache refresh"
        )
    if getattr(model, "quant_config", None) is not None:
        raise NotImplementedError(
            "GLM AWEX patch currently requires unquantized inference weights"
        )

    def run():
        import torch
        from areal.infra.platforms import current_platform

        with torch.no_grad():
            refresh()
        current_platform.synchronize()

    return run


def _glm5_refresh_factory(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        if not is_glm5(self, "infer"):
            return original(self, *args, **kwargs)
        refresh = _glm5_mla_refresh(self)
        result = original(self, *args, **kwargs)
        # AWEX writes parameters directly, bypassing SGLang's load_weights.
        # Rebuild absorbed MLA w_kc/w_vc from the updated kv_b_proj before ACK.
        refresh()
        return result

    return wrapper


def _glm5_colocate_refresh_factory(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        if not is_glm5(self, "infer"):
            return original(self, *args, **kwargs)
        refresh = _glm5_mla_refresh(self)
        client = self._colocate_http_client
        completed = False

        # The original colocate method publishes completion internally. Refresh
        # immediately before that publication, while IPC tensors remain alive,
        # rather than after the trainer has already observed the ACK.
        class CompletionClient:
            def put(proxy, url, *put_args, **put_kwargs):
                nonlocal completed
                if "/colocate_done_rank" in str(url):
                    refresh()
                    completed = True
                return client.put(url, *put_args, **put_kwargs)

            def __getattr__(proxy, name):
                return getattr(client, name)

        self._colocate_http_client = CompletionClient()
        try:
            result = original(self, *args, **kwargs)
            if not completed:
                raise RuntimeError(
                    "GLM colocate update did not publish its completion event"
                )
            return result
        finally:
            self._colocate_http_client = client

    return wrapper


def _deepseek_mla_refresh(adapter):
    model = validate_inference_model(adapter)
    refresh = getattr(model, "post_load_weights", None)
    if not callable(refresh):
        raise RuntimeError(
            "DeepSeek AWEX requires SGLang post_load_weights for MLA cache refresh"
        )

    def run():
        import torch
        from areal.infra.platforms import current_platform
        from hcu_areal.v2.weight_update.awex.deepseek_fp8 import (
            refresh_context,
            commit_staging,
        )

        commit_staging(adapter)
        with torch.no_grad(), refresh_context(adapter):
            refresh()
        current_platform.synchronize()

    return run


def _deepseek_refresh_factory(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        if not is_deepseek(self, "infer"):
            return original(self, *args, **kwargs)
        refresh = _deepseek_mla_refresh(self)
        result = original(self, *args, **kwargs)
        refresh()
        return result

    return wrapper


def _deepseek_colocate_refresh_factory(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        if not is_deepseek(self, "infer"):
            return original(self, *args, **kwargs)
        refresh = _deepseek_mla_refresh(self)
        client = self._colocate_http_client
        completed = False

        class CompletionClient:
            def put(proxy, url, *put_args, **put_kwargs):
                nonlocal completed
                if "/colocate_done_rank" in str(url):
                    refresh()
                    completed = True
                return client.put(url, *put_args, **put_kwargs)

            def __getattr__(proxy, name):
                return getattr(client, name)

        self._colocate_http_client = CompletionClient()
        try:
            result = original(self, *args, **kwargs)
            if not completed:
                raise RuntimeError(
                    "DeepSeek colocate update did not publish its completion event"
                )
            return result
        finally:
            self._colocate_http_client = client

    return wrapper


def _enabled(models, name):
    return models is None or name in models


def _refresh_factory(original, models=None):
    glm = _glm5_refresh_factory(original)
    deepseek = _deepseek_refresh_factory(original)

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        if _enabled(models, "deepseek") and is_deepseek(self, "infer"):
            return deepseek(self, *args, **kwargs)
        if _enabled(models, "glm5") and is_glm5(self, "infer"):
            return glm(self, *args, **kwargs)
        return original(self, *args, **kwargs)

    return wrapper


def _colocate_refresh_factory(original, models=None):
    glm = _glm5_colocate_refresh_factory(original)
    deepseek = _deepseek_colocate_refresh_factory(original)

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        if _enabled(models, "deepseek") and is_deepseek(self, "infer"):
            return deepseek(self, *args, **kwargs)
        if _enabled(models, "glm5") and is_glm5(self, "infer"):
            return glm(self, *args, **kwargs)
        return original(self, *args, **kwargs)

    return wrapper


def _group_preflight_factory(original, role="train", models=None):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        check_group_preflight(self, args, kwargs, role, models)
        return original(self, *args, **kwargs)

    return wrapper
