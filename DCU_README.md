# AReaL DCU 使用说明

本文档介绍 AReaL 在 DCU/HCU 环境中的安装、配置和训练方法。上游项目的
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

## 示例脚本

DCU 示例脚本位于 [`dcu_example`](./dcu_example/)：

| 脚本 | 用途 |
| --- | --- |
| `run_areal_qwen_grpo_rocm.sh` | 使用本地调度器运行 Qwen GRPO |
| `run_areal_qwen_grpo_ray.sh` | 使用单节点 Ray 运行 Qwen GRPO |
| `run_areal_ray_2nodes_qwen.sh` | 双节点 Qwen Ray 训练模板 |
| `run_areal_ray_2nodes.sh` | 通用双节点 Ray 启停与训练脚本 |

脚本移动到子目录后仍需从仓库根目录解析相对路径。脚本开头应包含：

```bash
# nhb: Resolve the repository root when this script is under dcu_example.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AREAL_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${AREAL_HOME}"
export PYTHONPATH="${AREAL_HOME}:${PYTHONPATH:-}"
```

### 单节点 Ray

```bash
source /opt/areal-venv-py31115/bin/activate
cd /home/AReaL-1.0.4
bash dcu_example/run_areal_qwen_grpo_ray.sh
```

启动后确认 Ray 识别的 GPU 数量与脚本中的 `N_GPUS_PER_NODE` 一致：

```bash
ray status
```

### 双节点 Ray

双节点必须能够访问相同的模型、AReaL 源码和共享实验目录。先在主节点启动
Ray Head：

```bash
cd /home/AReaL-1.0.4
bash dcu_example/run_areal_ray_2nodes_qwen.sh head
```

随后在工作节点加入集群，其中 `<HEAD_IP>` 为主节点可被工作节点访问的 IP：

```bash
cd /home/AReaL-1.0.4
bash dcu_example/run_areal_ray_2nodes_qwen.sh worker <HEAD_IP>
```

回到主节点确认两个节点均已加入，然后启动训练：

```bash
ray status
bash dcu_example/run_areal_ray_2nodes_qwen.sh train <HEAD_IP>
```

例如每个节点提供 8 张卡时，`ray status` 应显示总计 16 张 GPU。多节点配置还应
将以下目录设置为所有节点均可读写的共享存储：

```bash
cluster.fileroot=/shared/areal/experiments
cluster.name_resolve.nfs_record_root=/shared/areal/name_resolve
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
