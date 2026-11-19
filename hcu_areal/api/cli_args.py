# SPDX-License-Identifier: Apache-2.0

"""Extend official v2.1.0 SGLang structured configuration for HCU."""

from __future__ import annotations

import dataclasses
import functools
import json
import os


MODULE = "areal.api.cli_args"
_FIELDS = {
    "nsa_prefill_backend": (str | None, None),
    "nsa_decode_backend": (str | None, None),
    "max_total_tokens": (int | None, None),
    "fp8_gemm_backend": (
        str,
        dataclasses.field(
            default="auto",
            metadata={
                "help": "SGLang blockwise FP8 GEMM backend. Passed as --fp8-gemm-backend.",
                "choices": [
                    "auto",
                    "deep_gemm",
                    "flashinfer_trtllm",
                    "flashinfer_cutlass",
                    "flashinfer_deepgemm",
                    "cutlass",
                    "triton",
                    "aiter",
                ],
            },
        ),
    ),
    "page_size": (int | None, 64),
}


def _extend_dataclass(config, manager):
    existing = config.__dataclass_fields__
    missing = {key: val for key, val in _FIELDS.items() if key not in existing}
    if not missing:
        return
    if len(missing) != len(_FIELDS):
        raise RuntimeError(
            "SGLangConfig has a partial HCU field set; refusing to patch"
        )
    annotations = dict(config.__annotations__)
    old_fields = config.__dataclass_fields__
    methods = {
        name: config.__dict__[name] for name in ("__init__", "__repr__", "__eq__")
    }

    def cleanup():
        config.__annotations__ = annotations
        config.__dataclass_fields__ = old_fields
        for name, value in methods.items():
            setattr(config, name, value)
        for name in missing:
            if name in config.__dict__:
                delattr(config, name)

    manager.add_cleanup(cleanup)
    config.__annotations__ = dict(annotations)
    for name, (annotation, default) in missing.items():
        config.__annotations__[name] = annotation
        setattr(config, name, default)
    for name in methods:
        delattr(config, name)
    dataclasses.dataclass(config)


def _build_args_factory(original, manager):
    namespace = original.__globals__
    _extend_dataclass(namespace["SGLangConfig"], manager)

    @functools.wraps(original)
    def build_args(
        sglang_config,
        tp_size,
        base_gpu_id,
        host=None,
        port=None,
        dist_init_addr=None,
        n_nodes=1,
        node_rank=0,
        pp_size=1,
    ):
        attention_backend = sglang_config.attention_backend
        if (
            sglang_config.enable_deterministic_inference
            and attention_backend is not None
            and attention_backend.lower()
            not in namespace["_SGLANG_DETERMINISTIC_ATTENTION_BACKENDS"]
        ):
            namespace["logger"].warning(
                "SGLang deterministic inference is only documented for attention "
                "backends %s; configured attention_backend=%r may be non-deterministic.",
                sorted(namespace["_SGLANG_DETERMINISTIC_ATTENTION_BACKENDS"]),
                attention_backend,
            )
        args = namespace["conf_as_dict"](sglang_config)
        if sglang_config.enable_multithread_load:
            args["model_loader_extra_config"] = json.dumps(
                {"enable_multithread_load": sglang_config.enable_multithread_load},
                separators=(",", ":"),
            )
        args.pop("enable_multithread_load", None)
        args = dict(
            tokenizer_path=sglang_config.model_path,
            tokenizer_mode="auto",
            trust_remote_code=True,
            is_embedding=False,
            tp_size=tp_size,
            base_gpu_id=base_gpu_id,
            nnodes=n_nodes,
            node_rank=node_rank,
            dist_init_addr=dist_init_addr,
            **args,
        )
        if pp_size > 1:
            args["pp_size"] = pp_size
        if host is not None:
            args["host"] = host
        if port is not None:
            args["port"] = port
        if os.getenv("AREAL_SKIP_SGLANG_VERSION_CHECK", "1") != "1":
            if not namespace["pkg_version"].is_version_greater_or_equal(
                "sglang", "0.5.10.post1"
            ):
                raise RuntimeError("Needs sglang>=0.5.10.post1 to run the code.")
        return args

    return build_args
