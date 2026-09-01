# AReaL-das HCU 用户指南

本指南说明如何使用 HCU 基础镜像安装并运行 AReaL-das 示例。镜像中的驱动、HCU 运行时以及 HCU 版 PyTorch/SGLang 由镜像提供。

## 1. 使用 SGLang 基础镜像

请先从[光源社区](https://developer.sourcefind.cn/servicelist/detail?post_id=1abf923f-5a33-11f1-9e57-0242ac150003)获取当前可用的
HCU SGLang 基础镜像名称和标签：

```bash
docker pull REPOSITORY:TAG
```

将 `REPOSITORY:TAG` 替换为页面提供的实际镜像地址。镜像版本应与主机 HCU 驱动兼容。

## 2. 创建 HCU 容器

以下命令按单节点示例给出，请根据实际目录和镜像名称调整：

```bash
docker run -it \
  --name areal-das-hcu \
  --shm-size=64G \
  --device=/dev/kfd \
  --device=/dev/mkfd \
  --device=/dev/dri \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 \
  --ipc=host \
  --network=host \
  --workdir=/workspace \
  --privileged \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v <host-workspace>:/workspace \
  REPOSITORY:TAG \
  /bin/bash
```

容器启动后，确认 `/dev/kfd`、`/dev/dri` 和 HCU 版 PyTorch/SGLang 均可见。不要在容器内重新安装会覆盖 HCU 版本的
CUDA/ROCm 二进制包。

## 3. 获取代码和安装 AReaL

将本仓库挂载到容器的 `/workspace`，或在容器中获取源码：

```bash
cd /workspace/AReaL-das
python -m pip install -r requirements-hcu.txt
python -m pip install -e . --no-deps
```

`--no-deps` 用于保留基础镜像提供的 HCU 版 PyTorch、SGLang 和 Megatron 组件。

## 4. 运行快速示例

完整的单节点 Qwen3-8B 命令请参阅：

- [中文快速开始](../docs/zh/tutorial/quickstart.md)
- [AReaL-das 示例说明](README.md)

运行前需要准备模型目录（例如 `/model/qwen3/Qwen3-8B`）、Python 虚拟环境、`hcu_megatron` 和 SGLang 源码路径。

版权：Copyright (c) 2026 Hygon Information Technology Co., Ltd.
