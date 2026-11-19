# SPDX-License-Identifier: Apache-2.0

"""Select HCU entrypoints and clean up AReaL-owned subprocesses."""

from functools import partial
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import weakref


_STREAMED_PROCESSES = {}
_STREAMED_LOCK = threading.Lock()
_INFERENCE_GRACE = {
    "areal.v2.inference_service.gateway": 15,
    "areal.v2.inference_service.sglang.launch_server": 30,
}

_HCU_MODULES = frozenset(
    {
        "areal.infra.rpc.rpc_server",
        "areal.v2.training_service.guard",
        "areal.v2.training_service.worker",
        "areal.v2.inference_service.guard",
        "areal.v2.inference_service.sglang.launch_server",
        "areal.v2.weight_update.gateway",
    }
)


def hcu_command(command):
    """Rewrite only known Python module commands; preserve custom commands."""
    if (
        isinstance(command, (list, tuple))
        and len(command) >= 3
        and Path(command[0]).name.startswith("python")
        and command[1] == "-m"
        and command[2] in _HCU_MODULES
    ):
        return [
            command[0],
            "-m",
            "hcu_areal.launch",
            "--module",
            command[2],
            *command[3:],
        ]
    return command


def _managed_run_with_streaming_logs(
    original,
    cmd,
    log_file,
    merged_log,
    role,
    *,
    env=None,
    env_vars_in_cmd=None,
    stdout=None,
):
    command = hcu_command(cmd)
    if (
        not isinstance(command, (list, tuple))
        or not command
        or not Path(command[0]).name.startswith("python")
        or (stdout is not None and not hasattr(stdout, "write"))
    ):
        return original(
            command,
            log_file,
            merged_log,
            role,
            env=env,
            env_vars_in_cmd=env_vars_in_cmd,
            stdout=stdout,
        )

    child_env = dict(os.environ if env is None else env)
    child_env.update({str(k): str(v) for k, v in (env_vars_in_cmd or {}).items()})
    stdbuf = shutil.which("stdbuf") if "LD_PRELOAD" not in child_env else None
    actual_command = [stdbuf, "-oL", *command] if stdbuf else list(command)
    role_log = open(log_file, "a", encoding="utf-8", buffering=1)
    try:
        merged = open(merged_log, "a", encoding="utf-8", buffering=1)
    except Exception:
        role_log.close()
        raise
    try:
        process = subprocess.Popen(
            actual_command,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except Exception:
        role_log.close()
        merged.close()
        raise
    output = stdout or sys.stdout
    prefix = f"[{role}]".ljust(original.__globals__.get("LOG_PREFIX_WIDTH", 10))

    def stream():
        try:
            for line in process.stdout:
                role_log.write(line)
                merged.write(prefix + line)
                try:
                    try:
                        output.write(line)
                    except TypeError:
                        output.write(line.encode("utf-8", errors="replace"))
                    output.flush()
                except (BrokenPipeError, OSError, ValueError):
                    pass
        finally:
            process.stdout.close()
            role_log.close()
            merged.close()
            if process.poll() is not None:
                with _STREAMED_LOCK:
                    _STREAMED_PROCESSES.pop(process.pid, None)

    thread = threading.Thread(target=stream, name=f"hcu-log-{process.pid}", daemon=True)
    with _STREAMED_LOCK:
        _STREAMED_PROCESSES[process.pid] = (weakref.ref(process), thread)
    thread.start()
    return process


def _managed_kill_process_tree(
    original,
    parent_pid=None,
    timeout=5,
    include_parent=True,
    skip_pid=None,
    graceful=True,
):
    with _STREAMED_LOCK:
        entry = _STREAMED_PROCESSES.pop(parent_pid, None)
    if entry and graceful and include_parent and skip_pid is None:
        process, thread = entry[0](), entry[1]
        if process is not None and process.poll() is None:
            import psutil

            try:
                parent = psutil.Process(parent_pid)
                descendants = parent.children(recursive=True)
                parent.terminate()
                command = (
                    process.args if isinstance(process.args, (list, tuple)) else ()
                )
                grace = max(
                    [timeout]
                    + [
                        _INFERENCE_GRACE[arg]
                        for arg in command
                        if arg in _INFERENCE_GRACE
                    ]
                )
                _, alive_parent = psutil.wait_procs([parent], timeout=grace)
                if alive_parent:
                    try:
                        descendants.extend(
                            child
                            for child in parent.children(recursive=True)
                            if child not in descendants
                        )
                    except psutil.NoSuchProcess:
                        alive_parent = []
                remaining = []
                for child in descendants:
                    try:
                        if (
                            child.is_running()
                            and child.status() != psutil.STATUS_ZOMBIE
                        ):
                            child.terminate()
                            remaining.append(child)
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(remaining + alive_parent, timeout=1)
                logger = original.__globals__.get("logger")
                if alive and logger is not None:
                    logger.warning(
                        "Force killing managed processes: %s", [p.pid for p in alive]
                    )
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(alive, timeout=2)
                alive = [
                    p
                    for p in alive
                    if p.is_running() and p.status() != psutil.STATUS_ZOMBIE
                ]
                if alive:
                    raise RuntimeError(
                        f"Failed to stop processes: {[p.pid for p in alive]}"
                    )
                thread.join(timeout=1)
                return
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass
    result = original(
        parent_pid,
        timeout=timeout,
        include_parent=include_parent,
        skip_pid=skip_pid,
        graceful=graceful,
    )
    if entry:
        entry[1].join(timeout=1)
    return result


def _run_with_streaming_logs_factory(original):
    # Module-level callables and an importable original can be serialized by
    # Ray by reference; a nested closure would capture the process lock.
    return partial(_managed_run_with_streaming_logs, original)


def _kill_process_tree_factory(original):
    return partial(_managed_kill_process_tree, original)
