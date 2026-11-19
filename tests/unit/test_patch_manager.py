from types import ModuleType, SimpleNamespace

import pytest

from hcu_areal.areal_adaptor import enabled_awex_models, patch_features
from hcu_areal.patch_utils import AReaLPatchesManager
from hcu_areal.v2.weight_update.awex.sglang_adapter import _infer_parts_factory


def test_patch_composition_is_idempotent_and_reversible():
    module = ModuleType("example")
    module.run = lambda: "original"
    manager = AReaLPatchesManager()
    for name in ("first", "second"):
        manager.register_patch(
            "example.run",
            lambda old, name=name: lambda: name + ":" + old(),
            module_name="example",
            patch_id=name,
            apply_wrapper=True,
            compose=name == "second",
        )
    assert manager.apply_module(module) == 2
    assert module.run() == "second:first:original"
    assert manager.apply_module(module) == 0
    manager.remove_patches()
    assert module.run() == "original"


def test_patch_rejects_conflicting_targets():
    manager = AReaLPatchesManager()
    manager.register_patch(
        "example.run", lambda: None, module_name="example", patch_id="first"
    )
    with pytest.raises(ValueError, match="Conflicting target"):
        manager.register_patch(
            "example.run", lambda: None, module_name="example", patch_id="second"
        )


def test_model_switches_do_not_change_other_models():
    env = {"HCU_AREAL_DISABLE_DEEPSEEK_AWEX": "1"}
    assert "deepseek" not in enabled_awex_models(env)
    assert "glm5" in enabled_awex_models(env)
    original = object()
    adapter = SimpleNamespace(
        _get_model=lambda: SimpleNamespace(
            config=SimpleNamespace(model_type="deepseek_v3")
        )
    )
    infer = _infer_parts_factory(
        lambda self, name, tensor: [(name, tensor)], enabled_awex_models(env)
    )
    assert infer(adapter, "weight", original) == [("weight", original)]
    patches = [
        p
        for p in patch_features(env)._registered.values()
        if p.target_path.endswith("AwexSGLangAdapter._unfuse_params")
    ]
    assert [(p.patch_id, p.compose) for p in patches] == [
        ("qwen35.awex.infer.unfuse", False),
        ("qwen3vl.infer.parts", True),
    ]
