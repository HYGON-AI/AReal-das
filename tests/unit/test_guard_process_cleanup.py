import sys
import time

import psutil
import pytest

pytest.importorskip("areal")
from areal.infra.rpc.guard import app as guard
from hcu_areal.infra.utils import proc
from hcu_areal.v2.inference_service.controller import _kill_forked_service_factory
from unittest.mock import Mock


def test_guard_stops_forked_service_within_controller_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(
        guard,
        "run_with_streaming_logs",
        proc._run_with_streaming_logs_factory(guard.run_with_streaming_logs),
    )
    monkeypatch.setattr(
        guard,
        "kill_process_tree",
        proc._kill_process_tree_factory(guard.kill_process_tree),
    )
    state = guard.GuardState()
    state.fileroot = str(tmp_path)
    client = guard.create_app(state).test_client()
    child_file = tmp_path / "child.pid"
    script = """
import signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c',
    'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(0.2)
open(sys.argv[1], 'w').write(str(child.pid))
time.sleep(60)
"""
    response = client.post(
        "/fork",
        json={
            "role": "inf-server",
            "worker_index": 0,
            "raw_cmd": [sys.executable, "-c", script, str(child_file)],
        },
    )
    assert response.status_code == 200
    parent = psutil.Process(response.get_json()["pid"])
    child = None
    try:
        deadline = time.monotonic() + 10
        while not child_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child_file.exists()
        child = psutil.Process(int(child_file.read_text()))
        started = time.monotonic()
        response = client.post(
            "/kill_forked_worker", json={"role": "inf-server", "worker_index": 0}
        )
        assert response.status_code == 200
        assert time.monotonic() - started < 9
        assert not parent.is_running() or parent.status() == psutil.STATUS_ZOMBIE
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        assert not state.forked_children and not state.forked_children_map
    finally:
        for process in (child, parent):
            if process is not None:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass


def test_inference_controller_waits_for_graceful_service_shutdown():
    from areal.v2.inference_service.controller.controller import RolloutControllerV2

    client = Mock()
    client.post.return_value.status_code = 200
    controller = Mock(_sync_client=client)
    kill = _kill_forked_service_factory(RolloutControllerV2._kill_forked_service)
    kill(controller, "http://guard", "inf-server", 0)
    client.post.assert_called_once_with(
        "http://guard/kill_forked_worker",
        json={"role": "inf-server", "worker_index": 0},
        timeout=45,
    )
