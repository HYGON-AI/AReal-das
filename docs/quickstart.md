# 单机 Qwen3 GRPO 快速上手

本指南使用 Qwen3-8B、GSM8K、Megatron 和 SGLang，在一台 HCU 机器上运行两步 GRPO。训练使用 4 张卡，推理使用 4 张卡。

## 1. 准备环境

在 HCU 宿主机上从光源社区 Harbor 拉取镜像。

```bash
docker pull harbor.sourcefind.cn:5443/hcu/admin/base/custom:areal-das-v2.1.0-ubuntu22.04-dtk26.04-py3.12-torch2.10-20261119
```

## 2. 获取代码

在宿主机获取项目，并初始化固定版本的 AReaL、Megatron-LM-das 和 Megatron-bridge-das 子模块：

```bash
git clone --recurse-submodules https://github.com/HYGON-AI/AReal-das.git
cd AReal-das
git submodule update --init --recursive
git submodule status
```

## 3. 创建容器

在宿主机执行下面的 Docker 命令。

将尖括号及其中的内容替换为实际的容器名称、宿主机仓库路径、模型路径和输出目录。建议先创建输出目录，模型以只读方式挂载；冒号右侧的容器内路径与后续命令保持一致：

```bash
docker run -it --init --name "<容器名称>" --shm-size=128g \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  --ipc=host --network=host \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "<宿主机仓库路径>:/workspace/AReal-das:ro" \
  -v "<宿主机Qwen3-8B模型路径>:/models/Qwen3-8B:ro" \
  -v "<宿主机输出目录>:/workspace/runs" \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --entrypoint /bin/bash -w /workspace/AReal-das \
  "harbor.sourcefind.cn:5443/hcu/admin/base/custom:areal-das-v2.1.0-ubuntu22.04-dtk26.04-py3.12-torch2.10-20261119"
```

## 4. 使用现有配置

使用仓库中的 `examples/configs/gsm8k_qwen3_megatron.yaml`。该配置默认单机 8 卡：4 张用于 Megatron 训练，4 张用于
SGLang 推理。

本例需要容器已分配全部 8 张卡。参考镜像在一些宿主机上需要从挂载的 `/opt/hyhal/share/smi` 导入驱动的 `amdsmi` Python 模块，用于
HCU 设备和平台识别。下面的 `export` 将该目录加入搜索路径，同时保留原来的 `PYTHONPATH`。如果容器已能从正确位置导入驱动绑定并识别
HCU，可以省略这行。部分旧驱动将模块放在 `/opt/hyhal/share/amd_smi`，请按宿主机驱动布局替换。

在容器内运行以下命令，按实际设备数向 Ray 声明 8
张卡。缓存和临时目录使用环境默认位置；命令指定模型和两步训练。配置默认关闭定期保存模型和定期评估。示例默认自动创建本次专用的 Ray，结束时由启动器清理：

```bash
export PYTHONPATH="/opt/hyhal/share/smi${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m hcu_areal.launch --config examples/configs/gsm8k_qwen3_megatron.yaml \
  --set 'training.actor.path=/models/Qwen3-8B' \
  --set 'runtime.ray.num_gpus=8' \
  --set 'training.total_train_steps=2'
```

## 5. 查看结果

本例的主要训练日志位于容器内 `/workspace/runs/logs/gsm8k_qwen3_megatron/train.log`，训练和推理进程的详细日志位于
`/workspace/runs/logs/` 下，按实验名称和角色分目录或文件保存。

其他模型、配置和多机说明见 [HCU 示例](../examples/README.md)；适配实现见 [HCU 适配代码](../hcu_areal/README.md)。
