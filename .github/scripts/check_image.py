# ci diagnostic (read-only) - transient roster probe v2, reverted after the run
import os as _os, subprocess as _sp

def _run(c, t=60):
    print("--- CMD: " + c, flush=True)
    try:
        p = _sp.Popen(c, shell=True, stdout=_sp.PIPE, stderr=_sp.STDOUT)
        out = p.communicate(timeout=t)[0]
        if out:
            print(out.decode("utf-8", "replace"), flush=True)
        print("--- rc=" + str(p.returncode), flush=True)
    except Exception as e:
        print("--- ERR " + repr(e), flush=True)

def _ci_probe():
    print("=== C141628-P2-I141628-ROSTER-PROBE-BEGIN ===", flush=True)
    _run("hostname; id; nproc; free -g | head -2; date -u; echo RUNNER_NAME=$RUNNER_NAME")
    _run("ls -l /dev/kfd /dev/mkfd /dev/dri 2>&1 | head -20")
    _run("/opt/hyhal/bin/hy-smi --showproductname 2>&1 | head -40")
    _run("/opt/hyhal/bin/hy-smi --showuniqueid 2>&1 | head -40")
    _run("/opt/hyhal/bin/hy-smi --showmeminfo vram 2>&1 | head -60")
    _run("/opt/hyhal/bin/hy-smi --showdriverversion 2>&1 | head -8")
    _run("lspci 2>/dev/null | grep -iE 'hygon|haiguang|dcu|nvidia' | head -20")
    print("=== C141628-P2-I141628-ROSTER-PROBE-END ===", flush=True)

_ci_probe()

"""Check that a built image can import its HCU runtime and access a device."""

from importlib.metadata import PackageNotFoundError, version

import areal
import sglang
import torch
from packaging.version import Version

try:
    sglang_version = version("sglang")
except PackageNotFoundError:
    sglang_version = sglang.__version__

print("torch:", torch.__version__, "hip=", torch.version.hip)
print("areal:", areal.__file__)
print("sglang:", sglang.__file__, sglang_version)
assert torch.version.hip is not None, "HCU PyTorch is required"
assert torch.cuda.device_count() > 0, "No HCU device is visible"
assert torch.cuda.is_available(), "HCU runtime is unavailable"
assert torch.ones(1, device="cuda").item() == 1
assert Version(sglang_version) >= Version("0.5.12")
