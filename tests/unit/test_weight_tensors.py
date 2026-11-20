from types import SimpleNamespace

import pytest
import torch

from hcu_areal.v2.weight_update.awex.deepseek_awex_layout import (
    deepseek_inference_parts,
)
from hcu_areal.v2.weight_update.awex.deepseek_fp8 import (
    dequantize_blocks,
    export_weight,
    validate_quant_config,
)
from hcu_areal.v2.weight_update.awex.gemma3_awex_layout import gemma3_inference_parts
from hcu_areal.v2.weight_update.awex.glm5_awex_layout import glm5_inference_parts
from hcu_areal.v2.weight_update.awex.qwen35_awex_layout import (
    _split_gdn_tp_rows,
    inference_parts,
)
from hcu_areal.v2.weight_update.awex.qwen3_vl_layout import pack_vision_qkv_rank_major


def test_qwen3vl_vision_qkv_repacking_by_rank():
    source = torch.arange(12)
    assert pack_vision_qkv_rank_major(source, 2).tolist() == [
        0,
        1,
        4,
        5,
        8,
        9,
        2,
        3,
        6,
        7,
        10,
        11,
    ]
    with pytest.raises(ValueError, match="padded/dummy heads"):
        pack_vision_qkv_rank_major(torch.arange(15), 2)


@pytest.mark.parametrize("split", [deepseek_inference_parts, glm5_inference_parts])
def test_fused_mla_parts_are_writable_views(split):
    config = SimpleNamespace(
        q_lora_rank=2, kv_lora_rank=1, qk_rope_head_dim=1, hidden_size=2
    )
    name = "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight"
    source = torch.arange(8).reshape(4, 2)
    parts = split(name, source, config)
    assert [name.rsplit(".", 2)[-2] for name, _ in parts] == [
        "q_a_proj",
        "kv_a_proj_with_mqa",
    ]
    parts[1][1].fill_(-1)
    assert source.tolist() == [[0, 1], [2, 3], [-1, -1], [-1, -1]]


def test_gemma3_and_qwen35_vision_parts():
    gemma = "vision_tower.vision_model.encoder.layers.0.self_attn.qkv_proj.weight"
    qwen = "visual.blocks.0.attn.qkv_proj.weight"
    source = torch.arange(6)
    for parts in (
        gemma3_inference_parts(gemma, source),
        inference_parts(qwen, source, SimpleNamespace(), 1),
    ):
        assert [tensor.tolist() for _, tensor in parts] == [[0, 1], [2, 3], [4, 5]]
        assert [name.rsplit(".", 2)[-2] for name, _ in parts] == [
            "q_proj",
            "k_proj",
            "v_proj",
        ]


def test_qwen35_megatron_gdn_splits_tp_rows_in_projection_order():
    # Each rank stores [Q, K, V, Z, B, A], with wider V/Z than Q/K.
    rank_major = torch.arange(16).reshape(16, 1)
    parts = _split_gdn_tp_rows(rank_major, (2, 2, 4, 4, 2, 2), tp_size=2)
    assert [part.flatten().tolist() for part in parts] == [
        [0, 8],
        [1, 9],
        [2, 3, 10, 11],
        [4, 5, 12, 13],
        [6, 14],
        [7, 15],
    ]


def test_deepseek_fp8_export_has_independent_block_scales():
    config = validate_quant_config(
        {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [2, 2],
        }
    )
    name = "model.layers.0.self_attn.q_b_proj.weight"
    source = torch.tensor([[1.0, 2.0, 40.0, 80.0], [3.0, 4.0, 120.0, 160.0]])
    result = export_weight(name, source, config)
    assert [key for key, _ in result] == [name, name + "_scale_inv"]
    payload, scale = result[0][1], result[1][1]
    assert payload.dtype == torch.uint8 and tuple(scale.shape) == (1, 2)
    torch.testing.assert_close(scale, torch.tensor([[4.0 / 448.0, 160.0 / 448.0]]))
    restored = dequantize_blocks(
        payload.view(torch.float8_e4m3fn), scale, [2, 2], dtype=torch.float32
    )
    torch.testing.assert_close(restored, source, rtol=0.06, atol=0.1)
