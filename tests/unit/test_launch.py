import hashlib
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

from hcu_areal import areal_adaptor, launch
from hcu_areal.infra.platforms import prepare_process_environment
from hcu_areal.infra.scheduler.ray import (
    _prepare_worker_specs_factory,
    launcher_runtime_options,
)
from types import SimpleNamespace


def test_package_overlay_preserves_upstream(tmp_path, monkeypatch):
    source = tmp_path / "3rdparty/AReaL/areal"
    source.mkdir(parents=True)
    original = b"# upstream\n"
    (source / "__init__.py").write_bytes(original)
    (source / "module.py").write_text("value = 1\n")
    monkeypatch.setattr(
        launch, "_UPSTREAM_INIT_SHA256", hashlib.sha256(original).hexdigest()
    )
    launch._package_fingerprint.cache_clear()
    overlay = launch.install_package_init(
        tmp_path, cache_dir=tmp_path / "cache", env={}
    )
    assert (source / "__init__.py").read_bytes() == original
    assert (overlay / "areal/__init__.py").read_bytes() == Path(
        launch.__file__
    ).with_name("patch_init.py").read_bytes()
    assert (overlay / "areal/_upstream").resolve() == source.resolve()
    assert (
        launch.install_package_init(tmp_path, cache_dir=tmp_path / "cache", env={})
        == overlay
    )


def test_role_is_prepared_before_service_starts(monkeypatch):
    events = []
    monkeypatch.setenv("HCU_AREAL_ENABLE", "0")
    monkeypatch.setattr(
        launch, "prepare_process_environment", lambda: events.append("environment")
    )
    monkeypatch.setattr(
        areal_adaptor,
        "apply_process_role",
        lambda module: events.append(("patch", module)),
    )
    monkeypatch.setattr(
        launch.runpy,
        "run_module",
        lambda module, **kwargs: events.append(("run", module)),
    )
    monkeypatch.setattr(sys, "argv", ["pytest"])
    module = "areal.v2.training_service.guard"
    launch.run_explicit_module(module, ["--port", "1234"])
    assert events == ["environment", ("patch", module), ("run", module)]


def test_device_visibility_and_ray_environment():
    env = {
        "HCU_AREAL_ENABLE": "1",
        "HIP_VISIBLE_DEVICES": "2",
        "HCU_AREAL_UNSET_ENV": "OLD",
        "OLD": "x",
    }
    prepare_process_environment(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert "HIP_VISIBLE_DEVICES" not in env and "OLD" not in env
    options = launcher_runtime_options(
        {"HCU_AREAL_ENABLE": "1", "PYTHONPATH": "/repo", "CUDA_VISIBLE_DEVICES": "2"}
    )
    assert options == {
        "runtime_env": {"env_vars": {"HCU_AREAL_ENABLE": "1", "PYTHONPATH": "/repo"}}
    }


def test_ray_explicit_worker_environment_wins():
    spec = SimpleNamespace(env_vars={"TMS_INIT_ENABLE": "0"})

    def original(scheduler, role, count, specs):
        specs[0].env_vars.update(TMS_INIT_ENABLE="1", LD_PRELOAD="tms.so")
        return specs

    result = _prepare_worker_specs_factory(original)(object(), "actor", 1, [spec])
    assert result[0].env_vars == {"TMS_INIT_ENABLE": "0", "LD_PRELOAD": "tms.so"}


def test_owned_ray_is_released_on_driver_failure(tmp_path, monkeypatch):
    owner = Mock()
    monkeypatch.setattr(launch, "runtime_environment", lambda runtime: {})
    monkeypatch.setattr(launch, "install_package_init", lambda **kwargs: None)
    monkeypatch.setattr(
        launch, "start_owned_ray", lambda config, env: (owner, "127.0.0.1:1234")
    )
    stop = Mock()
    monkeypatch.setattr(launch, "stop_owned_ray", stop)
    monkeypatch.setattr(
        launch.subprocess, "Popen", Mock(side_effect=RuntimeError("driver failed"))
    )
    config = {
        "runtime": {
            "driver_log": str(tmp_path / "driver.log"),
            "entrypoint": "driver.py",
            "ray": {"mode": "auto"},
        },
        "training": {},
    }
    with pytest.raises(RuntimeError, match="driver failed"):
        launch.run_training(config)
    stop.assert_called_once_with(owner)
