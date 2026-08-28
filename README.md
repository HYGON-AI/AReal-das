# AReaL-das

AReaL-das 是基于 [AReaL](https://github.com/areal-project/AReaL) 的 HCU 平台适配版本，用于在海光计算单元（HCU）上使用 Ray、SGLang、FSDP 和 Megatron 进行强化学习训练。

## 特性

- 支持 HCU 平台训练与 rollout
- 支持 Ray 单节点和多节点资源调度
- 支持 SGLang rollout 后端
- 支持 FSDP 和 Megatron Actor 训练后端
- 提供 Qwen、GLM 的 HCU GRPO 示例

## 目录结构

~~~text
AReaL-das/
├── areal/                    # AReaL 源码与 HCU 适配
├── hcu_example/              # HCU 环境配置与启动脚本
├── docs/                     # 面向 HCU 的项目文档
├── requirements-hcu.txt      # HCU 直接 Python 依赖
├── LICENSE                   # Apache License 2.0
└── README.md
~~~

## 环境要求

- Linux
- Python 3.11
- 与 HCU 设备和驱动匹配的 DTK 版本
- HCU 版本的 PyTorch、SGLang、Ray；需要时还包括 Megatron-LM 与 Megatron-Bridge

> HCU 驱动、运行时，以及定制的 PyTorch、SGLang、Megatron 等软件包由对应 HCU 软件栈单独提供。requirements-hcu.txt 仅列出直接 Python 依赖；底层 HCU 软件栈由对应 DTK 发行包提供。

## 安装

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-hcu.txt
python -m pip install -e . --no-deps
~~~

> `--no-deps` 可避免 pip 覆盖已安装的 HCU PyTorch、SGLang 和 Megatron 软件栈。

## 配置外部组件路径

请按实际环境设置以下路径：

~~~bash
export AREAL_HOME=/path/to/AReaL-das
export SGLANG_ROOT=/path/to/sglang
export MEGATRON_HOME=/path/to/hcu_megatron
export MEGATRON_BRIDGE_HOME="${MEGATRON_HOME}/Megatron-Bridge"
export PYTHONPATH="${AREAL_HOME}:${SGLANG_ROOT}/python:${MEGATRON_HOME}/Megatron-LM:${MEGATRON_BRIDGE_HOME}/src:${PYTHONPATH:-}"
~~~

仅 glm_moe_dsa 模型路径需要 hcu_megatron。

## 快速开始

HCU 示例脚本、环境变量说明、模型与后端支持矩阵、单节点与多节点启动流程均位于 [hcu_example](hcu_example/README.md)。

首次运行前，请先按 [GRPO 示例说明](hcu_example/grpo/README.md) 配置 HCU 软件栈、模型路径和 Ray 环境；再使用该目录中的 `run.sh` 选择模型与 Actor 后端。示例中提供 Qwen、Qwen3-VL、GLM 的 FSDP/Megatron + SGLang 启动脚本。

## 支持的 HCU 示例

| 模型 | Actor 后端 | Rollout 后端 | 启动脚本 |
| --- | --- | --- | --- |
| Qwen3-8B | FSDP 或 Megatron | SGLang | hcu_example/grpo/run_qwen3_8b_*_sglang.sh |
| Qwen3-VL-4B | FSDP | SGLang（多模态） | hcu_example/grpo/run_qwen3_vl_4b_fsdp_sglang.sh |
| Qwen3-30B-A3B（4 层） | Megatron | SGLang | hcu_example/grpo/run_qwen3_30b_a3b_4layers_megatron_sglang.sh |
| Qwen3-1.7B | FSDP 或 Megatron | SGLang | hcu_example/grpo/run_qwen3_1_7b_*_sglang.sh |
| Qwen2.5-0.5B | FSDP 或 Megatron | SGLang | hcu_example/grpo/run_qwen2_5_0_5b_*_sglang.sh |
| GLM-5（4 层） | Megatron | SGLang | hcu_example/grpo/run_glm5_4layers_megatron_sglang.sh |

环境配置和启动细节请参阅 [HCU 示例](hcu_example/grpo/README.md) 与 [HCU 安装文档](docs/zh/tutorial/installation.md)。

## 依赖说明

requirements-hcu.txt 仅包含项目直接 Python 依赖。以下底层组件必须按 HCU 软件栈和 DTK 版本预先安装，不包含在 requirements-hcu.txt 中：

- HCU 驱动和运行时
- HCU 版本的 PyTorch、TorchVision 与 Triton
- SGLang、sglang-kernel 与相关 HCU kernel
- Flash Attention 与 Transformer Engine
- Megatron-LM 与 Megatron-Bridge（仅 Megatron 示例需要）

## 上游项目与版权

AReaL-das 基于 [areal-project/AReaL](https://github.com/areal-project/AReaL) 进行 HCU 平台适配开发。

- 上游仓库：https://github.com/areal-project/AReaL
- 上游基线：v1.0.4
- 上游许可证：Apache License 2.0

本仓库保留上游源文件中的原始版权声明和许可证条款。HCU 平台适配、HCU 示例脚本和 HCU 文档由 Hygon Information Technology Co., Ltd. 于 2026 年修改，并同样采用 Apache License 2.0：

~~~text
Copyright (c) 2026 Hygon Information Technology Co., Ltd.
~~~

完整的上游来源、版权和许可证保留说明见 [NOTICE](NOTICE)。第三方依赖以各自发行包和许可证文件中的条款为准。

## 开源许可

本项目采用 [Apache License 2.0](LICENSE)。再分发派生作品时，请保留上游的版权和许可证声明。第三方组件说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
