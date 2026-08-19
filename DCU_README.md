# AReaL DCU 使用说明

本文档介绍 AReaL 在 DCU 环境中的安装、配置和训练方法。上游项目的
完整功能说明请参阅[官方 README](./README.md)。

## 环境要求

当前示例环境基于以下软件组合：

- Python 3.11.15
- DTK 2604
- PyTorch、Triton、Transformer Engine 和 Flash Attention 的 DCU 定制版本
- Ray 单节点或多节点调度
- SGLang 推理后端
- FSDP 或 Megatron 训练后端

`requirements-dcu.txt` 是已验证环境的软件包快照。带有 DCU 定制版本号的
软件包通常无法从官方 PyPI 获取，需要先配置组织内部的 DCU Python 软件源或
准备对应 wheel。

## 安装依赖

建议创建独立的 Python 3.11 虚拟环境：

```bash
python3.11 -m venv /opt/areal-venv-py31115
source /opt/areal-venv-py31115/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-dcu.txt
```

安装完成后检查关键组件：

```bash
python - <<'PY'
import ray
import torch
import triton

print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("accelerator available:", torch.cuda.is_available())
print("accelerator count:", torch.cuda.device_count())
print("triton:", triton.__version__)
print("ray:", ray.__version__)
PY
```

## 源码配置

AReaL 的 DCU 环境使用以下适配源码：

- [SGLang DCU 源码](https://developer.sourcefind.cn/codes/OpenDAS/sglang)
- [DCU Megatron-LM/Megatron-Bridge 源码](http://42.228.13.241:10068/dcutoolkit/deeplearing/dcu_megatron)

AReaL、SGLang、Megatron-LM 和 Megatron-Bridge 使用源码运行时，可按实际安装
位置设置：

```bash
export AREAL_HOME=/home/AReaL-1.0.4
export SGLANG_HOME=/home/sglang_v0.5.12
export MEGATRON_HOME=/home/dcu_megatron/Megatron-LM
export MEGATRON_BRIDGE_HOME=/home/dcu_megatron/Megatron-Bridge

export PYTHONPATH="${AREAL_HOME}:${MEGATRON_BRIDGE_HOME}/src:${MEGATRON_HOME}:${SGLANG_HOME}/python:${PYTHONPATH:-}"
cd "${AREAL_HOME}"
```

验证源码是否来自预期目录：

```bash
python - <<'PY'
import areal
import sglang
from megatron import core

print("areal:", areal.__file__)
print("sglang:", sglang.__file__)
print("megatron.core:", core.__file__)
PY
```

## 训练示例

### 单节点训练示例

下面以 `Qwen3-8B + FSDP + SGLang + 单节点 8 卡` 为例。
单节点情况下，可以直接让 `run.sh` 负责启动 Ray Head 和训练。可以更换 `fsdp` 为 `megatron`，使用 Megatron 作为训练后端。
```bash
bash run.sh \
  --model=qwen3_8b \
  --backend=fsdp \
  --restart-ray
```

### 多节点训练示例

下面以 `Qwen3-30B-A3B + Megatron + SGLang + 多节点 Ray` 为例。对于 Qwen3-MoE 等大规模 MoE 模型，建议优先使用 Megatron 的 TP、PP、EP 等并行能力，不使用 FSDP 作为默认训练方案。

假设当前有两个节点：

```text
Head:
10.16.1.48
8 DCU

Worker:
10.16.1.61
8 DCU
```

所有节点必须能够访问相同的 `AReaL 源码`、`SGLang 源码`、`Megatron 源码` 和 `模型权重目录`。

**在 Head 节点启动 Ray**

在 `10.16.1.48` 执行：

```bash
source /opt/areal-venv-py31115/bin/activate
cd "${AREAL_HOME}/dcu_example2/grpo"
bash run.sh \
  --ray-head \
  --model=qwen3_30b_a3b_4layers \
  --backend=megatron \
  --ray-address=10.16.1.48:6379
```
该命令只负责启动 Ray Head，不启动训练。

**Worker 节点加入 Ray**

在 `10.16.1.61` 执行：
```bash
source /opt/areal-venv-py31115/bin/activate
cd "${AREAL_HOME}/dcu_example2/grpo"
bash run.sh \
  --ray-worker \
  --model=qwen3_30b_a3b_4layers \
  --backend=megatron \
  --ray-address=10.16.1.48:6379 \
  --worker-ip=10.16.1.61
```

如果还有第三个节点，例如 `10.16.1.62`，则在该节点执行：

```bash
bash run.sh \
  --ray-worker \
  --model=qwen3_30b_a3b_4layers \
  --backend=megatron \
  --ray-address=10.16.1.48:6379 \
  --worker-ip=10.16.1.62
```
**检查 Ray 集群**

回到 Head 节点执行：
```bash
ray status --address=10.16.1.48:6379
```
例如两个 8 卡节点，应看到总计 16 张 GPU；三个 8 卡节点则应为 24 张 GPU。在 Ray 节点数和 GPU 数量不正确时，不应启动训练。

**启动多节点训练**

Ray 集群建立完成后，只需要在 Head 节点执行训练命令：

```bash
bash run.sh \
  --model=qwen3_30b_a3b_4layers \
  --backend=megatron \
  --ray-address=10.16.1.48:6379
```
## 常用训练配置

后端字符串采用 `d<DP>p<PP>t<TP>` 格式，例如：

```bash
ACTOR_BACKEND=megatron:d1p1t4
ROLLOUT_BACKEND=sglang:d1p1t4
```

这表示 Actor 和 Rollout 分别使用 4 路张量并行。采用 separation 放置策略时，
两者总计需要 8 张 GPU。模型路径、批大小、生成数量和并行度均应在脚本顶部
集中修改。

对于显存紧张的 Megatron BF16 训练，可使用内存占用更低的优化器状态：

```bash
actor.optimizer.type=adam_bf16
```

## Torch Profiler

示例补丁支持通过环境变量抓取指定 Actor rank 的一次 PPO 更新：

```bash
export AREAL_TORCH_PROF_DIR=/home/areal_runs/profiles/${TRIAL_NAME}
export AREAL_TORCH_PROF_RANK=0
export AREAL_TORCH_PROF_UPDATE=2
```

`AREAL_TORCH_PROF_UPDATE=2` 表示跳过两次预热，抓取第三次 PPO 更新。未设置
`AREAL_TORCH_PROF_DIR` 时不会启动 Torch Profiler。生成的 JSON 可以使用
Perfetto 或 `chrome://tracing` 打开。

## 常见问题

### `ModuleNotFoundError: No module named 'areal'`

确认仓库根目录已加入 `PYTHONPATH`，并且脚本执行前已切换到 `${AREAL_HOME}`。

### Ray 显示的 GPU 数量不正确

启动 Ray 前检查 `CUDA_VISIBLE_DEVICES`、`HIP_VISIBLE_DEVICES` 和
`ROCR_VISIBLE_DEVICES`。由 Ray 管理设备分配时，通常应先清除外部遗留的设备
可见性变量，再通过 `ray start --num-gpus=<数量>` 注册资源。

### 多节点只能看到一个节点

确认工作节点使用正确的 Head IP 加入集群，并检查容器网络、端口、防火墙以及
共享目录在两个节点中的挂载路径是否一致。

### 依赖版本无法从 PyPI 安装

DCU 环境依赖定制的 PyTorch、Triton、Transformer Engine、Flash Attention、
SGLang Kernel 等软件包。请使用与 DTK 版本匹配的内部软件源或 wheel，不要用
官方同名包直接覆盖已验证版本。
