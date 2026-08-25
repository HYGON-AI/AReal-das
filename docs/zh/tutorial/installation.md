# AReaL HCU 安装

此发行版适配海光计算单元（HCU）。

## 前置条件

- Python 3.11
- 与 HCU 设备、驱动匹配的 DTK 版本
- HCU 版本的 PyTorch、Triton、Transformer Engine、Flash Attention、Ray 与 SGLang

`requirements-hcu.txt` 仅列出直接 Python 依赖。请先安装与 DTK 版本匹配的 HCU 软件栈。

## 安装

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-hcu.txt
```

## 配置源码路径

```bash
export AREAL_HOME=/path/to/AReaL
export SGLANG_ROOT=/path/to/sglang
export MEGATRON_HOME=/path/to/hcu_megatron
export MEGATRON_BRIDGE_HOME="${MEGATRON_HOME}/Megatron-Bridge"
export PYTHONPATH="${AREAL_HOME}:${SGLANG_ROOT}/python:${MEGATRON_HOME}/Megatron-LM:${MEGATRON_BRIDGE_HOME}/src:${PYTHONPATH:-}"
```

仅 `glm_moe_dsa` 需要 `hcu_megatron`，其他支持模型使用条件导入。

请继续阅读 [HCU 快速开始](quickstart.md)。兼容性说明和多节点启动流程见 [HCU_README](../../../README.md)。
