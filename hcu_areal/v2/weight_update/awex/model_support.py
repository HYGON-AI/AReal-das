# SPDX-License-Identifier: Apache-2.0

"""Shared AWEX metadata, preflight and model-specific helper rules."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from .qwen35_awex_layout import is_qwen35
from .glm5_awex_layout import is_glm5, assert_glm5_transfer_metadata
from .deepseek_awex_layout import is_deepseek, assert_deepseek_transfer_metadata
from .deepseek_v4_layout import is_deepseek_v4
from .gemma3_awex_layout import is_gemma3, assert_gemma3_transfer_metadata


def _qwen35_report(role, adapter, meta):
    if not is_qwen35(adapter, role):
        return
    trace = os.environ.get("HCU_AREAL_PATCH_TRACE_DIR", "").strip()
    print(
        f"[hcu_areal pid={os.getpid()}] qwen35 {role}: canonical={len(meta)}",
        file=sys.stderr,
        flush=True,
    )
    if not trace:
        return
    base = Path(trace)
    base.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "unknown"))

    def summarized(m):
        return {
            "name": m.name,
            "global_shape": list(m.global_shape),
            "dtype": str(m.dtype),
            "shards": [
                {
                    "name": s.name,
                    "shape": list(s.shape),
                    "global_offset": list(s.global_offset),
                    "sharding_type": str(s.sharding_type),
                    "sharding_dim": s.sharding_dim,
                    "num_shards": s.num_shards,
                    "tp_rank": s.tp_rank,
                    "global_rank": s.global_rank,
                }
                for s in m.shards
            ],
        }

    file = base / f"qwen35-v3-{role}-pid{os.getpid()}-rank{rank}.json"
    file.write_text(
        json.dumps(
            {
                "role": role,
                "pid": os.getpid(),
                "rank": rank,
                "metadata": [summarized(m) for m in meta],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[hcu_areal pid={os.getpid()}] metadata written: {file}",
        file=sys.stderr,
        flush=True,
    )


def glm5_metadata_report(role, adapter, metadata):
    if not is_glm5(adapter, role):
        return
    trace = os.getenv("HCU_AREAL_PATCH_TRACE_DIR", "").strip()
    if not trace:
        return
    path = Path(trace)
    path.mkdir(parents=True, exist_ok=True)
    rows = [
        dict(
            name=m.name,
            global_shape=list(m.global_shape),
            dtype=str(m.dtype),
            shards=[
                dict(
                    shape=list(s.shape),
                    global_offset=list(s.global_offset),
                    sharding_type=str(s.sharding_type),
                    sharding_dim=s.sharding_dim,
                    num_shards=s.num_shards,
                    global_rank=s.global_rank,
                )
                for s in m.shards
            ],
        )
        for m in metadata
    ]
    (path / f"glm5-{role}-pid{os.getpid()}.json").write_text(
        json.dumps(dict(role=role, metadata=rows), indent=2), encoding="utf-8"
    )


def glm5_group_preflight(adapter, args, kwargs):
    pair_name = args[0] if args else kwargs.get("pair_name")
    kv_store_url = args[5] if len(args) > 5 else kwargs.get("kv_store_url")
    if not pair_name or not kv_store_url:
        raise TypeError("GLM AWEX preflight requires pair_name and kv_store_url")
    from areal.v2.weight_update.awex import fetch_kv_metadata

    inf, trn = fetch_kv_metadata(kv_store_url, pair_name)
    count = assert_glm5_transfer_metadata(inf, trn)
    print(
        f"[hcu_areal pid={os.getpid()}] GLM AWEX metadata PASS: train/infer={count}",
        file=sys.stderr,
        flush=True,
    )


def deepseek_metadata_report(role, adapter, metadata):
    if not (is_deepseek(adapter, role) or is_deepseek_v4(adapter, role)):
        return
    trace = os.getenv("HCU_AREAL_PATCH_TRACE_DIR", "").strip()
    if not trace:
        return
    path = Path(trace)
    path.mkdir(parents=True, exist_ok=True)
    rows = [
        dict(
            name=m.name,
            global_shape=list(m.global_shape),
            dtype=str(m.dtype),
            shards=[
                dict(
                    shape=list(s.shape),
                    global_offset=list(s.global_offset),
                    sharding_type=str(s.sharding_type),
                    sharding_dim=s.sharding_dim,
                    num_shards=s.num_shards,
                    global_rank=s.global_rank,
                )
                for s in m.shards
            ],
        )
        for m in metadata
    ]
    (path / f"deepseek-{role}-pid{os.getpid()}.json").write_text(
        json.dumps(dict(role=role, metadata=rows), indent=2), encoding="utf-8"
    )


def deepseek_group_preflight(adapter, args, kwargs):
    pair_name = args[0] if args else kwargs.get("pair_name")
    kv_store_url = args[5] if len(args) > 5 else kwargs.get("kv_store_url")
    if not pair_name or not kv_store_url:
        raise TypeError("DeepSeek AWEX preflight requires pair_name and kv_store_url")
    from areal.v2.weight_update.awex import fetch_kv_metadata

    inf, trn = fetch_kv_metadata(kv_store_url, pair_name)
    count = assert_deepseek_transfer_metadata(inf, trn)
    print(
        f"[hcu_areal pid={os.getpid()}] DeepSeek AWEX metadata PASS: train/infer={count}",
        file=sys.stderr,
        flush=True,
    )


def check_group_preflight(adapter, args, kwargs, role, models=None):
    """Check metadata before the caller's adapter establishes its group."""
    if (models is None or "deepseek" in models) and (
        is_deepseek(adapter, role) or is_deepseek_v4(adapter, role)
    ):
        deepseek_group_preflight(adapter, args, kwargs)
        return
    if (models is None or "glm5" in models) and is_glm5(adapter, role):
        glm5_group_preflight(adapter, args, kwargs)
        return
    if (models is not None and "gemma3" not in models) or not is_gemma3(adapter, role):
        return
    pair_name = args[0] if args else kwargs.get("pair_name")
    kv_store_url = args[5] if len(args) > 5 else kwargs.get("kv_store_url")
    if not pair_name or not kv_store_url:
        raise TypeError("Gemma3 AWEX preflight requires pair_name, kv_store_url")
    from areal.v2.weight_update.awex import fetch_kv_metadata

    infer, train = fetch_kv_metadata(kv_store_url, pair_name)
    count = assert_gemma3_transfer_metadata(infer, train)
    print(
        f"[hcu_areal pid={os.getpid()}] gemma3 AWEX {role} metadata PASS: "
        f"train/infer={count}, global shapes and dtypes match",
        file=sys.stderr,
        flush=True,
    )
