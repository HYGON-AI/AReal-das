# HCU 强化学习示例

使用一份 YAML 指定 HCU 运行环境、日志、缓存、Ray 和 AReaL 训练配置。启动入口位于 `hcu_areal`，本目录保存 HCU
示例配置，训练脚本和基础配置使用 `3rdparty/AReaL/examples/`；上游 AReaL 位于 `3rdparty/AReaL`，启动器只复制 HCU 的
`__init__.py` 并链接其余上游文件，不修改子模块。

首次使用可以从 [单机 Qwen3 GRPO 快速上手](../docs/quickstart.md) 开始，使用 Megatron + SGLang
按步骤准备镜像和模型并运行两步训练。

## 启动前准备

### 在宿主机准备代码和容器

按[快速上手](../docs/quickstart.md)拉取镜像。在宿主机的仓库目录初始化子模块，并创建用于保留日志的宿主机目录。

```bash
cd "<宿主机仓库路径>"
git submodule update --init --recursive
mkdir -p "<宿主机输出目录>"
```

在宿主机启动容器。Docker 的 `-v` 参数中，冒号左侧是宿主机路径，右侧是容器内路径。将尖括号及其中内容替换成实际值。 `<容器内设备编号>`
填写本次分配并映射进容器的逻辑卡号，例如 8 卡时填写 `0,1,2,3,4,5,6,7`。

```bash
docker run -it --init --name "<容器名称>" --shm-size=128g \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  --ipc=host --network=host \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "<宿主机仓库路径>:/workspace/AReal-das:ro" \
  -v "<宿主机模型目录>:/models/<模型目录名>:ro" \
  -v "<宿主机输出目录>:/workspace/runs" \
  -e "CUDA_VISIBLE_DEVICES=<容器内设备编号>" \
  --entrypoint /bin/bash -w /workspace/AReal-das \
  "<镜像地址:标签>"
```

### 在容器内运行示例

以下各模型的启动命令都在容器内执行，当前目录为 `/workspace/AReal-das`。配置路径 `examples/configs/...`
相对于该目录。命令中的模型路径使用挂载后的容器内路径，例如 `/models/Qwen3-8B`。参考镜像如需通过挂载的驱动目录导入
`amdsmi`，请先按[快速上手的容器环境说明](../docs/quickstart.md#4-%E4%BD%BF%E7%94%A8%E7%8E%B0%E6%9C%89%E9%85%8D%E7%BD%AE)设置
`PYTHONPATH`。

单机示例默认自动启动本次专用的 Ray，完成后关闭。启动前检查训练、推理及其他角色的 GPU、CPU 和内存需求，确保 Ray
的可用资源满足调度；否则任务可能一直等待资源。YAML 未指定每个 worker 的 CPU 数量时，上游默认每卡申请 8 个 CPU。实验输出和名称发现使用容器内的
`/workspace/runs`，启动前须确认对应的宿主机目录可写。缓存和临时目录使用环境默认位置，临时目录应支持动态库加载，不应挂载为 `noexec`。Qwen3 MoE
和 GLM-5 双机配置默认连接已有 Ray 集群，需提供 Ray 地址；多机时 `/workspace/runs` 须为各节点可访问的同一个共享目录。

以下配置默认关闭定期保存 checkpoint 和定期评估。每个示例说明中的步数和卡数以当前 YAML 为准，不代表该默认配置已经完成实际训练验证。

双机需先启动 Ray 集群，具体步骤见下方 [Qwen3-30B 双机启动](#qwen3-moe-multinode)和
[GLM-5 双机启动](#glm5-multinode)。不要停止其他任务使用的 Ray 集群。更多容器准备说明见[快速上手](../docs/quickstart.md)。

<a id="qwen25-fsdp"></a>

## Qwen2.5-1.5B：FSDP + SGLang

模型目录填写 `<容器内Qwen2.5-1.5B 模型目录>`。YAML 声明单机 8 卡，使用 2 张 FSDP 训练卡和 2 张 SGLang 推理卡，最多训练 10 步。

```bash
python3 -m hcu_areal.launch --config examples/configs/gsm8k_qwen25_fsdp.yaml \
  --set "training.actor.path=<容器内Qwen2.5-1.5B 模型目录>"
```

<a id="qwen25-megatron"></a>

## Qwen2.5-1.5B：Megatron + SGLang

模型目录填写 `<容器内Qwen2.5-1.5B 模型目录>`。YAML 声明单机 8 卡，使用 2 张 Megatron 训练卡和 2 张 SGLang 推理卡，最多训练
10 步。

```bash
python3 -m hcu_areal.launch --config examples/configs/gsm8k_qwen25_megatron.yaml \
  --set "training.actor.path=<容器内Qwen2.5-1.5B 模型目录>"
```

<a id="qwen3-fsdp"></a>

## Qwen3-8B：FSDP + SGLang

模型目录填写 `<容器内Qwen3-8B 模型目录>`，单机 4 张 FSDP 训练卡和 4 张 SGLang 推理卡，最多训练 10 步。

```bash
python3 -m hcu_areal.launch --config examples/configs/gsm8k_qwen3_fsdp.yaml \
  --set "training.actor.path=<容器内Qwen3-8B 模型目录>"
```

<a id="qwen3-megatron"></a>

## Qwen3-8B：Megatron + SGLang

模型目录填写 `<容器内Qwen3-8B 模型目录>`，单机 4 张 Megatron 训练卡和 4 张 SGLang 推理卡，最多训练 10
步。[快速上手](../docs/quickstart.md)介绍如何覆盖步数，运行两步短训。

```bash
python3 -m hcu_areal.launch --config examples/configs/gsm8k_qwen3_megatron.yaml \
  --set "training.actor.path=<容器内Qwen3-8B 模型目录>"
```

<a id="qwen35-fsdp"></a>

## Qwen3.5-4B：FSDP + SGLang

模型目录填写 `<容器内Qwen3.5-4B 模型目录>`，单机 2 张 FSDP 训练卡和 2 张 SGLang 推理卡，最多训练 2 步，使用 BF16 KV
cache。临时目录需要允许加载动态库，挂载为 `noexec` 的目录不适用于此用途。此前在 SGLang-DAS `5c859f` 上的两步短测还需要对两个 KDA
模块做临时兼容修改；运行环境应包含相应修复，**仅修改 YAML 不能解决原镜像中的 `boltops` 导入错误**。

```bash
python3 -m hcu_areal.launch --config examples/configs/gsm8k_qwen35_fsdp.yaml \
  --set "training.actor.path=<容器内Qwen3.5-4B 模型目录>"
```

<a id="qwen3-moe-megatron"></a>

## Qwen3-30B-A3B：Megatron + SGLang

### 单机启动

单机配置使用 **4 张 Megatron 训练卡 + 4 张 SGLang 推理卡**，最多训练 2 步，并启用优化器 CPU 卸载。运行前准备 8 张显存充足的 HCU
和足够的主机内存；此前一次类似测试的主机内存峰值约 509 GiB。

将尖括号及其中的内容替换为实际模型路径；命令会为单机运行自动启动独立 Ray：

```bash
python3 -m hcu_areal.launch --config examples/configs/gsm8k_qwen3_moe_megatron_single_node.yaml \
  --set "training.actor.path=<容器内Qwen3-30B-A3B 模型目录>"
```

此前短测的退出阶段曾出现推理服务清理超时记录，应检查退出状态及日志。

<a id="qwen3-moe-multinode"></a>

### 双机启动

双机配置默认每机 8 卡、每个 Ray 节点声明 128 个 CPU，最多训练 10
步。两台机器需使用相同镜像和代码版本及路径，并在相同容器内路径下可读地访问模型；`/workspace/runs` 须挂载到同一个可写共享目录。两台宿主机分别运行前面的
Docker 命令，使用各自的宿主机路径和相同的容器挂载路径。确认每台机器的设备和 CPU 满足配置，端口可用，并将命令中的尖括号及内容替换为实际值。如果需要调整 Ray
声明的 CPU 数，在各节点启动命令中增加 `--set "runtime.ray.num_cpus=<实际可用 CPU 数>"`，并确保各角色的资源申请能够调度。

在 Head 节点的容器内启动 Ray。

```bash
python3 -m hcu_areal.launch --config examples/configs/gsm8k_qwen3_moe_megatron.yaml \
  --action ray-head \
  --set "runtime.ray.address=<Head 节点 IP>:<端口>" \
  --set "runtime.ray.node_ip=<Head 节点 IP>"
```

在 Worker 节点的容器内加入集群，地址和端口与 Head 一致。

```bash
python3 -m hcu_areal.launch --config examples/configs/gsm8k_qwen3_moe_megatron.yaml \
  --action ray-worker \
  --set "runtime.ray.address=<Head 节点 IP>:<端口>" \
  --set "runtime.ray.node_ip=<Worker 节点 IP>"
```

在 Head 节点的容器内检查资源并启动训练。

```bash
ray status --address="<Head 节点 IP>:<端口>"

python3 -m hcu_areal.launch --config examples/configs/gsm8k_qwen3_moe_megatron.yaml \
  --set "runtime.ray.address=<Head 节点 IP>:<端口>" \
  --set "training.actor.path=<容器内Qwen3-30B-A3B 模型目录>"
```

训练结束后按需要清理本次创建的 Ray 集群；不要停止其他任务的集群。

<a id="qwen3vl-fsdp"></a>

## Qwen3-VL-4B：FSDP + SGLang

模型目录填写 `<容器内Qwen3-VL-4B 模型目录>`，单机 4 张训练卡和 4 张推理卡，最多训练 10 步。

```bash
python3 -m hcu_areal.launch --config examples/configs/geometry3k_qwen3vl_fsdp.yaml \
  --set "training.actor.path=<容器内Qwen3-VL-4B 模型目录>"
```

<a id="gemma3-fsdp"></a>

## Gemma3-4B：FSDP + SGLang

模型目录填写 `<容器内Gemma3-4B 模型目录>`；YAML 使用 4 张 FSDP 训练卡和 2 张 SGLang 推理卡，最多训练 2
步。此前短测日志中有图片重分词提示，其对 token 对齐和训练数值的影响尚未验证。

```bash
python3 -m hcu_areal.launch --config examples/configs/geometry3k_gemma3_fsdp.yaml \
  --set "training.actor.path=<容器内Gemma3-4B 模型目录>"
```

<a id="deepseek-megatron"></a>

## DeepSeek R1：Megatron + SGLang

面向原版完整 DeepSeek R1 的检查点格式为官方 FP8 E4M3，使用 128×128 分块和配套的 `weight_scale_inv`。训练端由
Megatron-Bridge-das 子模块读取权重并反量化为 BF16。完整官方 FP8 检查点的端到端训练尚待验证。

下方命令用于复现已验证的四层 BF16 短测。该检查点为裁剪后的验证用权重。配置使用 4 张训练卡和 4 张推理卡，运行两步，采用 Triton attention、BF16
KV cache 和优化器 CPU 卸载。

```bash
python3 -m hcu_areal.launch --config examples/configs/gsm8k_deepseek_megatron.yaml \
  --set "training.actor.path=<容器内DeepSeek R1 四层 BF16 检查点目录>"
```

<a id="glm5-megatron"></a>

## GLM-5（4层）：Megatron + SGLang

使用已准备好的 GLM-5 四层 BF16 检查点；YAML 不会自动将完整模型裁剪为四层。

### 单机启动

单机使用 4 张 Megatron 训练卡和 4 张 SGLang 推理卡，启用优化器 CPU 卸载，自动启动 Ray，运行两步。若当前镜像默认设置
`CUDA_VISIBLE_DEVICES=0`，且本次容器已分配并映射全部 8 卡，在 `docker run` 中加入
`-e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`。

```bash
python3 -m hcu_areal.launch --config examples/configs/gsm8k_glm5_megatron_single_node.yaml \
  --set "training.actor.path=<容器内GLM-5 四层 BF16 检查点目录>"
```

此前四层检查点在 SGLang-DAS `5c859f` 镜像上以此并行度完成两步短训和两次权重同步。

<a id="glm5-multinode"></a>

### 双机启动

双机配置每机 8 卡，使用 8 张 Megatron 训练卡和 8 张 SGLang 推理卡，最多训练 10 步。该配置尚待双机实测。

在两台宿主机上分别运行前面的 Docker 命令。两台机器使用相同的镜像、代码版本和容器内代码路径，模型在两台机器的相同容器内路径下可读；将 `/workspace/runs`
挂载到两台机器共同访问的可写共享目录。确认两台机器的设备、CPU、内存和网络满足任务需求，Head 节点地址及端口可从 Worker 访问。YAML 为每个 Ray 节点声明
8 卡和 128 个 CPU；如需调整 CPU 数，在两条 Ray 启动命令中分别增加
`--set "runtime.ray.num_cpus=<实际可用 CPU 数>"`，并确保所有角色的资源申请能够调度。

将尖括号及其中的内容替换为实际值。在 Head 节点的容器内启动 Ray。

```bash
python3 -m hcu_areal.launch --config examples/configs/gsm8k_glm5_megatron.yaml \
  --action ray-head \
  --set "runtime.ray.address=<Head 节点 IP>:<端口>" \
  --set "runtime.ray.node_ip=<Head 节点 IP>"
```

在 Worker 节点的容器内加入集群，使用相同的 Head 地址和端口。

```bash
python3 -m hcu_areal.launch --config examples/configs/gsm8k_glm5_megatron.yaml \
  --action ray-worker \
  --set "runtime.ray.address=<Head 节点 IP>:<端口>" \
  --set "runtime.ray.node_ip=<Worker 节点 IP>"
```

在 Head 节点的容器内确认两台机器及所需资源已加入，再启动训练。

```bash
ray status --address="<Head 节点 IP>:<端口>"

python3 -m hcu_areal.launch --config examples/configs/gsm8k_glm5_megatron.yaml \
  --set "runtime.ray.address=<Head 节点 IP>:<端口>" \
  --set "training.actor.path=<容器内GLM-5 四层 BF16 检查点目录>"
```

训练结束后，查看 `/workspace/runs/logs/gsm8k_glm5_megatron/train.log` 和命令退出状态；按需要清理本次创建的 Ray
集群，不要停止其他任务的集群。

## 查看结果

容器内的主要训练日志默认位于 `/workspace/runs/logs/<配置文件名>/train.log`，将 `<配置文件名>` 替换为不带 `.yaml`
的配置文件名。训练和推理角色的详细日志位于 `/workspace/runs/logs/<用户>/<实验名>/<运行名>/`，包括各角色日志和
`merged.log`；实验名和运行名分别由 YAML 的 `training.experiment_name` 和 `training.trial_name` 指定。

按前文挂载方式，也可在宿主机的 `<宿主机输出目录>/logs/` 下查看相同文件。

查看训练和采样是否完成、权重同步及梯度记录，并检查命令退出状态。梯度全为 0 不能证明参数发生了有效更新；短训通过也不代表精度或收敛已经验证。
