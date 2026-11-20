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


@pytest.mark.parametrize("suffix, rows", [("weight", 1024), ("weight_scale_inv", 8)])
def test_deepseek_v4_fused_qkv_exposes_writable_transposed_views(suffix, rows):
    from hcu_areal.v2.weight_update.awex.deepseek_v4_layout import inference_parts

    name = "model.layers.0.self_attn.wqkv_a." + suffix
    source = torch.zeros((4, rows + rows // 2))
    parts = dict(inference_parts(name, source))

    assert parts[name.replace("wqkv_a", "wq_a")].shape == (rows, 4)
    parts[name.replace("wqkv_a", "wkv")].fill_(3)
    assert source[:, :rows].eq(0).all()
    assert source[:, rows:].eq(3).all()


def test_deepseek_v4_expert_scales_and_overlap_ape():
    from hcu_areal.v2.weight_update.awex.deepseek_v4_layout import (
        export_parts,
        inference_parts,
    )

    prefix = "model.layers.2.mlp.experts"
    scale = torch.zeros((2, 4, 3))
    parts = dict(inference_parts(prefix + ".w13_weight_scale_inv", scale))
    parts[prefix + ".1.up_proj.weight_scale_inv"].fill_(2)
    assert scale[1, :2].eq(0).all()
    assert scale[1, 2:].eq(2).all()

    ape = torch.arange(8).reshape(2, 4)
    [(name, value)] = export_parts(
        "layers.2.attn.compressor.ape", ape, [0, 0, 4, 128], 129280
    )
    assert name == "model.layers.2.self_attn.compressor.ape"
    assert value.dtype == torch.float32
    assert value.tolist() == [[0, 1, 4, 5], [2, 3, 6, 7]]


@pytest.mark.parametrize("suffix", ["weight", "weight_scale_inv"])
def test_deepseek_v4_square_output_projection_uses_transposed_runtime_layout(suffix):
    from hcu_areal.v2.weight_update.awex.deepseek_v4_layout import inference_parts

    name = "model.layers.0.self_attn.wo_b." + suffix
    storage = torch.tensor([[1, 2], [3, 4]])
    [(actual_name, view)] = inference_parts(name, storage)

    assert actual_name == name
    assert view.tolist() == [[1, 3], [2, 4]]
    view[0, 1] = 5
    assert storage[1, 0] == 5
