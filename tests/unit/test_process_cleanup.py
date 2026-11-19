import subprocess
import sys
import time
import weakref

import psutil
import pytest

from hcu_areal.infra.utils import proc


@pytest.mark.parametrize("ignore_term", [False, True])
def test_managed_cleanup_stops_parent_and_stubborn_child(tmp_path, ignore_term):
    child_file = tmp_path / "child.pid"
    script = """
import signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c',
    'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])
signal.signal(signal.SIGTERM, signal.SIG_IGN if sys.argv[2] == '1' else signal.SIG_DFL)
time.sleep(0.2)
open(sys.argv[1], 'w').write(str(child.pid))
time.sleep(60)
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", script, str(child_file), str(int(ignore_term))]
    )
    child = None
    try:
        deadline = time.monotonic() + 10
        while not child_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child_file.exists()
        child = psutil.Process(int(child_file.read_text()))

        class Thread:
            def join(self, timeout):
                pass

        proc._STREAMED_PROCESSES[parent.pid] = (weakref.ref(parent), Thread())
        started = time.monotonic()
        proc._managed_kill_process_tree(
            lambda *args, **kwargs: pytest.fail("unexpected fallback"),
            parent.pid,
            timeout=3,
        )
        assert time.monotonic() - started < 9
        assert parent.poll() is not None
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        proc._STREAMED_PROCESSES.pop(parent.pid, None)
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if child is not None:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass


def test_inference_service_gets_time_to_finish_shutdown(tmp_path, monkeypatch):
    ready = tmp_path / "ready"
    finished = tmp_path / "finished"
    marker = "areal.v2.inference_service.sglang.launch_server"
    script = """
import signal, sys, time
def stop(*args):
    time.sleep(1)
    open(sys.argv[2], 'w').write('closed')
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
open(sys.argv[1], 'w').write('ready')
time.sleep(60)
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", script, str(ready), str(finished), marker]
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists()
        monkeypatch.setitem(proc._INFERENCE_GRACE, marker, 3)

        class Thread:
            def join(self, timeout):
                pass

        proc._STREAMED_PROCESSES[parent.pid] = (weakref.ref(parent), Thread())
        proc._managed_kill_process_tree(
            lambda *a, **kw: pytest.fail("unexpected fallback"), parent.pid, timeout=0.1
        )
        assert finished.read_text() == "closed"
        assert parent.poll() == 0
    finally:
        proc._STREAMED_PROCESSES.pop(parent.pid, None)
        if parent.poll() is None:
            parent.kill()
            parent.wait()
