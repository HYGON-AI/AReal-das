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

单机使用 2 张 FSDP 训练卡和 2 张 SGLang 推理卡，最多训练 2 步，KV cache 使用 BF16。临时目录需要允许加载动态库。

在容器内按[快速上手](../docs/quickstart.md#4-%E4%BD%BF%E7%94%A8%E7%8E%B0%E6%9C%89%E9%85%8D%E7%BD%AE)设置驱动模块搜索路径，确认
4 张卡可见，并显式向 Ray 声明卡数。

```bash
python3 -m hcu_areal.launch --config examples/configs/gsm8k_qwen35_fsdp.yaml \
  --set "training.actor.path=<容器内 Qwen3.5-4B 模型目录>" \
  --set 'runtime.ray.num_gpus=4'
```

SGLang `0.5.18+g453edca` 审计镜像使用上述配置完成两步训练和两次权重同步，两步梯度均为 0。当前公开发布镜像的同配置短训待验证。

<a id="qwen35-moe-megatron"></a>

## Qwen3.5-35B-A3B：Megatron + SGLang

默认单机 8 卡，4 张 Megatron 训练卡和 4 张 SGLang 推理卡（两个 TP2 副本）。最多训练 2 步，通过 v2 AWEX 同步权重。配置关闭
MTP，训练优化器全量卸载到宿主机内存。完整检查点已用公开镜像完成 4+2 卡两步短训和两次权重同步。

在容器内执行，填入对应的模型挂载路径。

```bash
python3 -m hcu_areal.launch --config examples/configs/gsm8k_qwen35_moe_megatron.yaml \
  --set "training.actor.path=<容器内 Qwen3.5-35B-A3B 模型目录>"
```

<a id="qwen36-moe-megatron"></a>

## Qwen3.6-35B-A3B：Megatron + SGLang

默认单机 8 卡，4 张 Megatron 训练卡和 4 张 SGLang 推理卡（两个 TP2 副本）。最多训练 2 步，通过 v2 AWEX 同步权重。配置关闭
MTP，训练优化器全量卸载到宿主机内存。完整检查点已用公开镜像完成 4+2 卡两步短训和两次权重同步。

在容器内执行，填入对应的模型挂载路径。

```bash
python3 -m hcu_areal.launch --config examples/configs/gsm8k_qwen36_moe_megatron.yaml \
  --set "training.actor.path=<容器内 Qwen3.6-35B-A3B 模型目录>"
```

<a id="qwen38-megatron"></a>

## Qwen3.8-27B：Megatron + SGLang

使用完整 BF16 检查点。默认单机 8 卡，其中 4 张卡运行 Megatron（TP2 × PP2），4 张卡运行 SGLang（两个 TP2 副本）。配置最多训练 2
步，关闭 MTP，训练优化器全量卸载到宿主机内存，通过 v2 AWEX 同步权重。4+2 卡实测完成两步训练及两次权重同步，两步梯度范数均非零。

在容器内执行，填入模型挂载路径。

```bash
python3 -m hcu_areal.launch --config examples/configs/gsm8k_qwen38_megatron.yaml \
  --set "training.actor.path=<容器内 Qwen3.8-27B BF16 模型目录>"
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
步。该配置尚待双机实测。两台机器需使用相同镜像和代码版本及路径，并在相同容器内路径下可读地访问模型；`/workspace/runs`
须挂载到同一个可写共享目录。两台宿主机分别运行前面的 Docker 命令，使用各自的宿主机路径和相同的容器挂载路径。确认每台机器的设备和 CPU
满足配置，端口可用，并将命令中的尖括号及内容替换为实际值。如果需要调整 Ray 声明的 CPU 数，在各节点启动命令中增加
`--set "runtime.ray.num_cpus=<实际可用 CPU 数>"`，并确保各角色的资源申请能够调度。

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

<a id="deepseek-v4-flash-megatron"></a>

## DeepSeek‑V4‑Flash 四层：Megatron + SGLang

在宿主机按[四层模型检查点](../docs/four-layer-checkpoints.md#deepseek-v4-flash)准备模型目录，或使用已有的四层检查点。

Megatron 训练需要支持 `dsv4_hybrid` 的 Core。下载
[`43e45e13074f36da68b34c7a189cd997a01d1b41`](https://github.com/NVIDIA/Megatron-LM/commit/43e45e13074f36da68b34c7a189cd997a01d1b41)
并解压至 `<宿主机 Core dev 目录>`。沿用[启动前准备](#%E5%90%AF%E5%8A%A8%E5%89%8D%E5%87%86%E5%A4%87)，在宿主机用
8 张卡启动容器。将通用 `docker run` 命令的模型挂载替换为以下两项。

```bash
-v "<宿主机 Core dev 目录>:/workspace/AReal-das/3rdparty/Megatron-LM-das/3rdparty/Megatron-LM:ro" \
-v "<宿主机四层模型目录>:/models/DeepSeek-V4-Flash-4layers:ro"
```

如果四层目录通过绝对符号链接引用原始分片，还需给同一容器追加 `-v "<宿主机原始模型绝对路径>:<宿主机原始模型绝对路径>:ro"`，以解析链接。

配置最多训练 2 步。默认单机 8 卡，使用 4 张 Megatron 训练卡（TP1、EP4）和 4 张 SGLang 推理卡（两个 TP2 副本）。 SGLang
使用镜像的 FP4 专家转 FP8 路径，并通过 v2 AWEX 同步权重。在容器内的 `/workspace/AReal-das` 执行以下命令。

```bash
python3 -m hcu_areal.launch --config examples/configs/gsm8k_deepseek_v4_flash_megatron.yaml \
  --set 'training.actor.path=/models/DeepSeek-V4-Flash-4layers'
```

公开镜像与上述 Core dev 搭配，在 4+2 卡配置下已完成两步 GRPO 和两次 AWEX 同步。测试中两步梯度均为 0，结果覆盖采样、训练和同步流程。

<a id="deepseek-megatron"></a>

## DeepSeek R1：Megatron + SGLang

使用官方 FP8 权重。检查点采用 E4M3、128 × 128 分块和配套的
`weight_scale_inv`。按[四层模型检查点](../docs/four-layer-checkpoints.md#deepseek-r1)准备四层 FP8
权重，并按该文档只读挂载四层目录和原始权重目录。

仓库提供单机 4 张 Megatron 训练卡和 4 张 SGLang 推理卡的两步试跑配置。四层 FP8 检查点的生成已验证，四层和完整官方 FP8 检查点的端到端 GRPO
均待验证。在容器内执行下面的命令。

```bash
python3 -m hcu_areal.launch --config examples/configs/gsm8k_deepseek_megatron.yaml \
  --set "training.actor.path=<容器内 DeepSeek-R1 四层 FP8 检查点目录>" \
  --set 'runtime.ray.num_gpus=8'
```

<a id="glm5-megatron"></a>

## GLM-5（4层）：Megatron + SGLang

在宿主机按[四层模型检查点](../docs/four-layer-checkpoints.md#glm-5-bf16)准备 GLM-5 四层 BF16
检查点。原始分片通过符号链接使用时，按该文档同时挂载原始目录。

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

## 复现 4+2 卡短训

Qwen3.5‑35B‑A3B、Qwen3.6‑35B‑A3B、Qwen3.8‑27B 和 DeepSeek‑V4‑Flash 四层的端到端短训使用 4 张训练卡和 2
张推理卡。使用快速上手中的公开镜像。启动容器时将 `CUDA_VISIBLE_DEVICES` 设为实际映射的 6 张卡，例如
`0,1,2,3,4,5`。DeepSeek‑V4‑Flash 还需要按该示例准备 Core dev 和模型挂载。

在容器内选择上文对应的模型配置，运行下列命令。两个卡数设置分别用于 Ray 和 AReaL，推理端使用单个 TP2 副本。

```bash
python3 -m hcu_areal.launch --config "examples/configs/<对应模型的 YAML 文件名>" \
  --set "training.actor.path=<容器内模型目录>" \
  --set 'runtime.ray.num_gpus=6' \
  --set 'training.cluster.n_gpus_per_node=6' \
  --set 'training.rollout.backend=sglang:d1p1t2'
```

## 查看结果

容器内的主要训练日志默认位于 `/workspace/runs/logs/<配置文件名>/train.log`，将 `<配置文件名>` 替换为不带 `.yaml`
的配置文件名。训练和推理角色的详细日志位于 `/workspace/runs/logs/<用户>/<实验名>/<运行名>/`，包括各角色日志和
`merged.log`；实验名和运行名分别由 YAML 的 `training.experiment_name` 和 `training.trial_name` 指定。

按前文挂载方式，也可在宿主机的 `<宿主机输出目录>/logs/` 下查看相同文件。

查看训练和采样是否完成、权重同步及梯度记录，并检查命令退出状态。梯度全为 0 不能证明参数发生了有效更新；短训通过也不代表精度或收敛已经验证。
