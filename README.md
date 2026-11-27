# AReaL-das

AReaL-das 是基于 [AReaL](https://github.com/areal-project/AReaL)
的HCU生态的强化学习训练套件，旨在为HCU生态合作伙伴提供强化学习训练解决方案。

## 快速开始

在 HCU 宿主机上拉取光源社区 Harbor 发布的 AReaL-das v2.1.0 镜像。

```bash
docker pull harbor.sourcefind.cn:5443/hcu/admin/base/custom:areal-das-v2.1.0-ubuntu22.04-dtk26.04-py3.12-torch2.10-20261119
```

从环境准备到两步训练，参见 [单机 Qwen3 GRPO 快速上手](docs/quickstart.md)。该示例使用
Qwen3-8B、GSM8K、[Megatron-LM-das](https://github.com/HYGON-AI/Megatron-LM-das) 训练和
[SGLang-das](https://github.com/HYGON-AI/sglang-das) 推理，训练使用 4 张卡、推理使用 4 张卡。

其他模型配置、启动方式和验证范围见 [HCU 示例](examples/README.md)；适配代码及开发测试见
[HCU 适配说明](hcu_areal/README.md)。

## 自行配置环境

当前参考镜像使用 Ubuntu 22.04、DTK 26.04-rc4 和 Python
3.12.7。自行配置时，可在[光源社区 DAS PyPI](https://pypi.sourcefind.cn/) 查找适配包，按 DTK、Python 和 PyTorch
版本选择对应的 wheel。下表记录镜像内的实际构建版本，包源是否提供该构建请以页面查询结果为准。通用 Python 包版本见
[requirements.txt](requirements.txt)。

| 用途                 | Python 包            | 镜像中的版本                                           |
| -------------------- | -------------------- | ------------------------------------------------------ |
| 张量计算与视觉       | `torch`              | `2.10.0+das.opt1.dtk2604.2609081206.g28498c`           |
| 张量计算与视觉       | `torchvision`        | `0.25.0+das.opt1.dtk2604.torch2100.2609081417.gdcd044` |
| 训练与算子           | `transformer_engine` | `2.13.0+das185.dtk2604.torch2100.2608131458.g06a38a`   |
| 训练与算子           | `triton`             | `3.6.0+das.opt1.dtk2604.torch2100.2609081433.g65af71`  |
| 训练与算子           | `flash_attn`         | `2.8.3+das.opt1.dtk2604.torch2100.2609081541.ge8ddf7`  |
| 训练与算子           | `flash_mla`          | `1.2.0+das.opt1.dtk2604.torch2100.2609081603.gc33d94`  |
| 推理                 | `sglang`             | `0.5.18+g453edca`                                      |
| 推理                 | `sglang-kernel`      | `0.4.6.post1`                                          |
| 推理                 | `vllm`               | `0.21.0+dtk2604.torch2100.2609082054.g8c979d`          |
| 训练到推理的权重同步 | `awex`               | `0.8.1`                                                |

## 目录结构

```text
AReaL-das/
├── 3rdparty/                 # 固定的 AReaL 和 Megatron DAS 子模块
├── hcu_areal/                # HCU 运行时适配
├── examples/                 # HCU 示例配置与使用说明
├── scripts/                  # 四层检查点准备脚本
├── tests/                    # HCU 测试与验证
├── docs/                     # 文档目录
├── requirements.txt          # Python 依赖
├── LICENSE                   # Apache-2.0 许可证
└── README.md
```

## 模型列表

推理后端使用 SGLang-DAS，训练算法为 GRPO。✅ 两步短测已验证（点击查看启动方式） | 🚧 开发中 | 🔍 暂未验证

| 模型              |               FSDP 训练               |                    Megatron 训练                    | 说明                                                               |
| ----------------- | :-----------------------------------: | :-------------------------------------------------: | ------------------------------------------------------------------ |
| Qwen2.5-1.5B      | [✅](examples/README.md#qwen25-fsdp)  |      [✅](examples/README.md#qwen25-megatron)       |                                                                    |
| Qwen3-8B          |  [✅](examples/README.md#qwen3-fsdp)  |       [✅](examples/README.md#qwen3-megatron)       |                                                                    |
| Qwen3-30B-A3B     |                  🔍                   |     [✅](examples/README.md#qwen3-moe-megatron)     | 单机 4+4 卡短训已验证，双机待验证                                  |
| Qwen3-VL-4B       | [✅](examples/README.md#qwen3vl-fsdp) |                         🔍                          | 真实图片短测，存在重分词提示                                       |
| Qwen3.5-4B        | [✅](examples/README.md#qwen35-fsdp)  |                         🔍                          | 审计镜像 4 卡两步流程已验证，两步梯度均为 0                        |
| Qwen3.5-35B-A3B   |                  🔍                   |    [✅](examples/README.md#qwen35-moe-megatron)     | 完整检查点，4+2 卡两步已验证，示例默认 8 卡                        |
| Qwen3.6-35B-A3B   |                  🔍                   |    [✅](examples/README.md#qwen36-moe-megatron)     | 完整检查点，4+2 卡两步已验证，示例默认 8 卡                        |
| Qwen3.8-27B       |                  🔍                   |      [✅](examples/README.md#qwen38-megatron)       | 完整 BF16 检查点，4+2 卡两步已验证，示例默认 8 卡                  |
| Gemma3-4B         | [✅](examples/README.md#gemma3-fsdp)  |                         🔍                          | 真实图片短测，仍有重分词风险                                       |
| DeepSeek R1       |                  🔍                   |     [🔍](examples/README.md#deepseek-megatron)      | 四层 FP8 检查点已验证可生成，端到端训练待验证                      |
| DeepSeek‑V4‑Flash |                  🔍                   | [✅](examples/README.md#deepseek-v4-flash-megatron) | 四层，默认 8 卡，4+2 卡流程已验证，两步梯度均为 0，需挂载 Core dev |
| GLM-5             |                  🔍                   |       [✅](examples/README.md#glm5-megatron)        | 四层 BF16 检查点，单机 4+4 卡两步已验证，双机待验证                |

短测结果的镜像版本、并行配置和验证范围见各模型示例。GLM-5、DeepSeek‑V4‑Flash 和 DeepSeek‑R1
的权重准备步骤见[四层模型检查点](docs/four-layer-checkpoints.md)。

## 上游项目与版权

AReaL-das 基于 [areal-project/AReaL](https://github.com/areal-project/AReaL) 进行 HCU
平台适配开发。

- 上游仓库：https://github.com/areal-project/AReaL
- 上游基线：v2.1.0
- 上游 Commit：ecc8b0e4dfb4e3965f67121a5c87866a07c390ae
- 上游许可证：Apache-2.0

本仓库保留上游源文件中的原始版权声明和许可证条款。Modified by Hygon Information Technology Co., Ltd.

HCU 平台适配、HCU 示例脚本和 HCU 文档由 Hygon Information Technology Co., Ltd. 于 2026 年修改，并同样采用
Apache-2.0：

```text
Copyright (c) 2026 Hygon Information Technology Co., Ltd.
```

完整的上游来源、版权和许可证保留说明见 [NOTICE](NOTICE)。第三方依赖以各自发行包和许可证文件中的条款为准。

## 开源许可

本项目采用 [Apache License 2.0](LICENSE)。再分发派生作品时，请保留上游的版权和许可证声明。第三方组件说明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
