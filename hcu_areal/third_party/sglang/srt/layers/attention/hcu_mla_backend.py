# SPDX-License-Identifier: Apache-2.0

"""Bridge the missing KV-pool attributes in SGLang-DAS 453edca."""

from __future__ import annotations

import functools


MODULE = "sglang.srt.layers.attention.hcu_mla_backend"


def _init_factory(original):
    @functools.wraps(original)
    def initialize(self, model_runner, *args, **kwargs):
        original(self, model_runner, *args, **kwargs)
        for name in ("req_to_token_pool", "token_to_kv_pool"):
            if not hasattr(self, name):
                setattr(self, name, getattr(model_runner, name))

    return initialize
