# Install AReaL on HCU

This distribution targets Hygon Compute Units (HCU).

## Requirements

- Python 3.11
- A DTK release compatible with the HCU device and driver
- HCU-enabled PyTorch, Triton, Transformer Engine, Flash Attention, Ray, and SGLang

`requirements-hcu.txt` lists direct Python dependencies. Install the HCU software stack supplied for the matching DTK version before using it.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-hcu.txt
```

## Configure source locations

```bash
export AREAL_HOME=/path/to/AReaL
export SGLANG_ROOT=/path/to/sglang
export MEGATRON_HOME=/path/to/hcu_megatron
export MEGATRON_BRIDGE_HOME="${MEGATRON_HOME}/Megatron-Bridge"
export PYTHONPATH="${AREAL_HOME}:${SGLANG_ROOT}/python:${MEGATRON_HOME}/Megatron-LM:${MEGATRON_BRIDGE_HOME}/src:${PYTHONPATH:-}"
```

Only `glm_moe_dsa` requires `hcu_megatron`; other supported models use conditional imports.

Continue with the [HCU quickstart](quickstart.md). See [HCU_README](../../../README.md) for compatibility notes and multi-node launch.
