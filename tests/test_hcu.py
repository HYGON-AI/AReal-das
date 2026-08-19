# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import pytest

import areal.engine.hcu as hcu


def setup_function() -> None:
    hcu._HCU_DSA_PATCHED = False


def test_requires_hcu_dsa_patches_only_for_glm_moe_dsa() -> None:
    assert hcu.requires_hcu_dsa_patches("glm_moe_dsa")
    assert not hcu.requires_hcu_dsa_patches("qwen3")
    assert not hcu.requires_hcu_dsa_patches(None)


def test_apply_hcu_dsa_patches_reports_missing_optional_dependency(monkeypatch) -> None:
    def missing_hcu(_: str):
        raise ModuleNotFoundError(name="hcu_megatron")

    monkeypatch.setattr(hcu.importlib, "import_module", missing_hcu)

    with pytest.raises(RuntimeError, match="optional hcu_megatron package"):
        hcu.apply_hcu_dsa_patches()


def test_apply_hcu_dsa_patches_applies_once(monkeypatch) -> None:
    calls: list[object] = []

    class Feature:
        def register_patches(self, manager, *, args) -> None:
            calls.append((manager, args))

    class PatchesManager:
        @classmethod
        def apply_patches(cls) -> None:
            calls.append("applied")

    modules = {
        "hcu_megatron.features_manager.transformer.dsa_feature": SimpleNamespace(
            DSAFeature=Feature
        ),
        "hcu_megatron.patch_utils": SimpleNamespace(
            MegatronPatchesManager=PatchesManager
        ),
    }
    monkeypatch.setattr(hcu.importlib, "import_module", modules.__getitem__)

    hcu.apply_hcu_dsa_patches()
    hcu.apply_hcu_dsa_patches()

    assert calls == [(PatchesManager, None), "applied"]