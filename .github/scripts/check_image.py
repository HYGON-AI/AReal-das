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
