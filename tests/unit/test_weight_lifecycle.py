from types import SimpleNamespace

import pytest
import torch

from hcu_areal.v2.weight_update.awex import megatron_adapter, sglang_adapter


@pytest.mark.parametrize(
    "model,refresh",
    [("glm_moe_dsa", "_glm5_mla_refresh"), ("deepseek_v3", "_deepseek_mla_refresh")],
)
def test_mla_refresh_only_after_successful_update(monkeypatch, model, refresh):
    events = []
    adapter = SimpleNamespace(
        _get_model=lambda: SimpleNamespace(config=SimpleNamespace(model_type=model))
    )
    monkeypatch.setattr(
        sglang_adapter, refresh, lambda adapter: lambda: events.append("refresh")
    )
    update = sglang_adapter._refresh_factory(lambda self: events.append("update"))
    update(adapter)
    assert events == ["update", "refresh"]
    events.clear()

    def fail(self):
        events.append("update")
        raise RuntimeError("transfer failed")

    with pytest.raises(RuntimeError, match="transfer failed"):
        sglang_adapter._refresh_factory(fail)(adapter)
    assert events == ["update"]


@pytest.mark.parametrize(
    "model,refresh",
    [("glm_moe_dsa", "_glm5_mla_refresh"), ("deepseek_v3", "_deepseek_mla_refresh")],
)
def test_colocate_refresh_precedes_completion_and_restores_client(
    monkeypatch, model, refresh
):
    events = []
    client = SimpleNamespace(put=lambda url: events.append("ack"))
    adapter = SimpleNamespace(
        _get_model=lambda: SimpleNamespace(config=SimpleNamespace(model_type=model)),
        _colocate_http_client=client,
    )
    monkeypatch.setattr(
        sglang_adapter, refresh, lambda adapter: lambda: events.append("refresh")
    )

    def update(self):
        events.append("update")
        self._colocate_http_client.put("/colocate_done_rank/0")

    sglang_adapter._colocate_refresh_factory(update)(adapter)
    assert events == ["update", "refresh", "ack"]
    assert adapter._colocate_http_client is client


def test_dense_sync_restores_gradient_buffers_after_failure(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    model = SimpleNamespace(
        offload_grad_buffers=lambda **kwargs: None,
        restore_grad_buffers=lambda **kwargs: None,
    )
    adapter = SimpleNamespace(
        _engine=SimpleNamespace(model=[model]),
        _dte_config=SimpleNamespace(enabled=False),
        _release_grad_buffers_for_separation_sync=lambda: events.append("release"),
        _restore_grad_buffers_after_separation_sync=lambda: events.append("restore"),
    )

    def fail(self):
        events.append("transfer")
        raise RuntimeError("transfer failed")

    with pytest.raises(RuntimeError, match="transfer failed"):
        megatron_adapter._dense_sync_release_grad_buffers_factory(fail)(adapter)
    assert events == ["release", "transfer", "restore"]
