from types import SimpleNamespace

import pytest

from hcu_areal.engine.megatron_engine import (
    optimizer_cpu_offload_fraction,
    optimizer_pin_cpu_grads,
    requires_hcu_dsa_patches,
)
from hcu_areal.third_party.sglang.srt.layers.attention.hcu_mla_backend import (
    _init_factory,
)
from hcu_areal.v2.inference_service.sglang.launch_server import prepare_triton_kv_layout


def test_optimizer_offload_and_dsa_are_explicit():
    assert optimizer_cpu_offload_fraction({}) is None
    assert (
        optimizer_cpu_offload_fraction(
            {"HCU_AREAL_OPTIMIZER_CPU_OFFLOAD_FRACTION": "0.5"}
        )
        == 0.5
    )
    assert (
        optimizer_pin_cpu_grads({"HCU_AREAL_OPTIMIZER_PIN_CPU_GRADS": "false"}) is False
    )
    with pytest.raises(ValueError):
        optimizer_cpu_offload_fraction(
            {"HCU_AREAL_OPTIMIZER_CPU_OFFLOAD_FRACTION": "2"}
        )
    assert requires_hcu_dsa_patches("glm_moe_dsa")
    assert not requires_hcu_dsa_patches("deepseek_v3")


def test_sglang_mla_preserves_existing_pools():
    existing = object()
    runner = SimpleNamespace(req_to_token_pool=object(), token_to_kv_pool=object())
    backend = SimpleNamespace(req_to_token_pool=existing)
    _init_factory(lambda self, model_runner: None)(backend, runner)
    assert backend.req_to_token_pool is existing
    assert backend.token_to_kv_pool is runner.token_to_kv_pool


def test_triton_layout_is_limited_to_verified_sglang_version():
    env = {"HCU_AREAL_ENABLE": "1"}
    prepare_triton_kv_layout(
        ["--attention-backend", "triton"], env, sglang_version="0.5.18+g453edca"
    )
    assert env["SGLANG_KV_LAYOUT_HCU_FA"] == "0"
    other = {"HCU_AREAL_ENABLE": "1"}
    prepare_triton_kv_layout(
        ["--attention-backend", "triton"], other, sglang_version="0.5.18+g5c859f"
    )
    assert "SGLANG_KV_LAYOUT_HCU_FA" not in other
