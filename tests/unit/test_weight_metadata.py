from types import SimpleNamespace

import pytest

from hcu_areal.v2.weight_update.awex.deepseek_awex_layout import (
    assert_deepseek_transfer_metadata,
)
from hcu_areal.v2.weight_update.awex.gemma3_awex_layout import (
    assert_gemma3_transfer_metadata,
    gemma3_train_to_infer_name,
)
from hcu_areal.v2.weight_update.awex.glm5_awex_layout import (
    assert_glm5_transfer_metadata,
)
from hcu_areal.v2.weight_update.awex.qwen35_awex_layout import (
    canonical_name,
    segments_for_training,
)
from hcu_areal.v2.weight_update.awex.qwen3_vl_layout import (
    normalize_qwen3_vl_train_name,
    normalize_qwen3_vl_visual_infer_name,
)


@pytest.mark.parametrize(
    "check",
    [
        assert_deepseek_transfer_metadata,
        assert_glm5_transfer_metadata,
        assert_gemma3_transfer_metadata,
    ],
)
def test_transfer_metadata_matches_names_shapes_and_dtype(check):
    item = SimpleNamespace(name="weight", global_shape=(4, 8), dtype="bf16")
    assert check([item], [item]) == 1
    for changed in (
        SimpleNamespace(name="other", global_shape=(4, 8), dtype="bf16"),
        SimpleNamespace(name="weight", global_shape=(8, 4), dtype="bf16"),
        SimpleNamespace(name="weight", global_shape=(4, 8), dtype="fp32"),
    ):
        with pytest.raises(ValueError):
            check([changed], [item])


def test_deepseek_transfer_metadata_matches_inference_replicas():
    item = SimpleNamespace(name="weight", global_shape=(4, 8), dtype="bf16")
    replica = SimpleNamespace(name="weight", global_shape=(4, 8), dtype="bf16")
    assert assert_deepseek_transfer_metadata([item, replica], [item]) == 1

    for changed in (
        SimpleNamespace(name="weight", global_shape=(8, 4), dtype="bf16"),
        SimpleNamespace(name="weight", global_shape=(4, 8), dtype="fp32"),
    ):
        with pytest.raises(ValueError, match="conflicting infer metadata"):
            assert_deepseek_transfer_metadata([changed, replica], [item])
    with pytest.raises(ValueError, match="duplicate train name"):
        assert_deepseek_transfer_metadata([item], [item, replica])


def test_model_weight_names_are_normalized():
    assert (
        gemma3_train_to_infer_name("model.language_model.layers.0.weight")
        == "language_model.model.layers.0.weight"
    )
    assert (
        normalize_qwen3_vl_train_name("model.language_model.layers.0.weight")
        == "model.layers.0.weight"
    )
    assert (
        normalize_qwen3_vl_visual_infer_name("visual.blocks.0.attn.qkv_proj.weight")
        == "model.visual.blocks.0.attn.qkv.weight"
    )
    assert (
        canonical_name("model.language_model.layers.0.self_attn.q_proj.weight")
        == "model.layers.0.q_proj.weight"
    )


def test_qwen35_training_qkv_segments():
    config = SimpleNamespace(
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=2,
        linear_value_head_dim=3,
    )
    name = "model.layers.0.linear_attn.in_proj_qkv.weight"
    segments = segments_for_training(name, (14, 3), config)
    assert [(n.rsplit(".", 2)[-2], offset, size) for n, offset, size in segments] == [
        ("in_proj_q", 0, 4),
        ("in_proj_k", 4, 4),
        ("in_proj_v", 8, 6),
    ]
