# AReaL v1.0.4 HCU Example 使用说明

本目录提供 AReaL在 HCU 环境下运行 GRPO 训练的示例脚本，训练侧支持 Megatron 和 PyTorch FSDP2，推理侧统一使用 SGLang。

---

## 1. 目录结构

推荐将本目录放在 AReaL 仓库根目录下，例如：

```text
<AREAL_HOME>/
└── hcu_example/
    ├── README.md
    ├── env.yaml
    ├── common/
    │   └── common_env.sh
    ├── scripts/
    │   ├── check_env.sh
    │   ├── cleanup_areal.sh
    │   ├── ray_status.sh
    │   ├── start_ray.sh
    │   ├── start_ray_worker.sh
    │   └── stop_ray.sh
    └── grpo/
        ├── README.md
        ├── common.sh
        ├── run.sh
        ├── run_qwen2_5_0_5b_fsdp_sglang.sh
        ├── run_qwen2_5_0_5b_megatron_sglang.sh
        ├── run_qwen3_1_7b_fsdp_sglang.sh
        ├── run_qwen3_1_7b_megatron_sglang.sh
        ├── run_qwen3_8b_fsdp_sglang.sh
        ├── run_qwen3_vl_4b_fsdp_sglang.sh
        ├── run_qwen3_8b_megatron_sglang.sh
        ├── run_qwen3_30b_a3b_4layers_megatron_sglang.sh
        ├── run_qwen3_5_2b_megatron_sglang.sh
        └── run_glm5_4layers_megatron_sglang.sh
```

其中 `grpo/run.sh` 是推荐的统一入口，`run.sh` 负责发现训练脚本、启动/检查 Ray 并最终调用对应 launcher。

---

## 2. 运行前需要准备什么

运行训练之前，所有参与 Ray 集群的节点都应使用相同版本的 Python、AReaL、SGLang、Megatron-LM 和 Megatron-Bridge。多节点情况下，模型目录、实验输出目录以及 name-resolve 目录也必须在所有节点上使用相同的路径；如果这些目录来自 NFS 或其他共享文件系统，应先确认各节点都能正常读写。

建议至少准备以下路径：

```text
<VENV_PATH>               Python 3.11 虚拟环境
<AREAL_HOME>               HCU_AReaL 源码根目录
<MEGATRON_HOME>            hcu_megatron 根目录
<SGLANG_ROOT>              SGLang 源码根目录
<MODEL_PATH>               模型权重目录
<SHARED_RUNTIME_ROOT>      多节点共享运行目录
```
---

## 3. `common_env.sh` 与源码路径

公共环境由：

```text
hcu_example/common/common_env.sh
```

统一设置。Ray head、Ray worker、AReaL Actor 和 SGLang rollout 都会继承 Ray daemon 启动时的环境，所以源码路径必须在启动 Ray **之前** 配置正确。最常见的问题不是训练脚本参数本身，而是 Ray worker 仍然导入了旧目录下的 AReaL、Megatron 或 SGLang。

使用命令前可以导出环境变量替代 `common_env.sh` 中的默认值，例如：

```bash
export AREAL_HOME="<YOUR_AREAL_HOME>"
export MEGATRON_HOME="<YOUR_MEGATRON_HOME>"
export SGLANG_ROOT="<YOUR_SGLANG_ROOT>"
bash run.sh \
  --model=qwen3_8b \
  --backend=fsdp \
  --restart-ray
```

当前 `common_env.sh` 中最重要的变量包括：

| 变量 | 作用 |
| --- | --- |
| `AREAL_HOME` | AReaL 仓库根目录 |
| `AREAL_ROOT` | AReaL 源码根目录，通常与 `AREAL_HOME` 相同 |
| `VENV` / `VENV_PATH` | Python 虚拟环境 |
| `PYTHON_BIN` | Ray worker 和训练使用的 Python |
| `MEGATRON_HOME` / `MEGATRON_ROOT` | HCU Megatron 源码根目录 |
| `SGLANG_ROOT` | SGLang 仓库根目录 |
| `SGLANG_HOME` | SGLang Python 源码目录，一般为 `<SGLANG_ROOT>/python` |
| `PYTHONPATH` | AReaL、Megatron、Megatron-Bridge 和 SGLang 的源码搜索路径 |
---

## 4. AReaL 运行目录与重要环境变量

AReaL 训练过程中会产生训练日志、Ray 协调信息、name-resolve 记录以及编译缓存。`common_env.sh` 将这些内容分开管理，主要变量如下。

| 变量 | 说明 |
| --- | --- |
| `AREAL_RUNS_ROOT` | 正式训练日志和实验输出根目录 |
| `AREAL_RUNTIME_ROOT` | 运行时协调目录 |
| `AREAL_CACHE_ROOT` | Torch/Triton 等缓存根目录 |
| `FILER_ROOT` | AReaL `cluster.fileroot` |
| `NAME_RESOLVE_ROOT` | AReaL `cluster.name_resolve.nfs_record_root` |
| `LOG_ROOT` | 日志根目录 |
| `RAY_TMPDIR` | Ray 临时目录，推荐使用节点本地 `/dev/shm` |
| `AREAL_ENV_PROFILE` | 模型环境 profile，例如 `qwen`、`qwen35`、`glm5` |

单节点时这些目录可以位于本地磁盘；多节点时 `FILER_ROOT` 和 `NAME_RESOLVE_ROOT` 必须在所有节点上可见。推荐显式设置一个共享目录，例如：

```bash
export AREAL_RUNTIME_ROOT="<SHARED_RUNTIME_ROOT>"
export FILER_ROOT="${AREAL_RUNTIME_ROOT}/experiments"
export NAME_RESOLVE_ROOT="${AREAL_RUNTIME_ROOT}/name_resolve"
```

不要因为不同容器里都存在 `/workspace` 就默认它们是同一块存储。如果底层并非共享文件系统，多节点 name-resolve 和实验协调会出现异常。

---

## 5. 模型脚本可覆盖的常用训练变量

每个 `run_<model>_<backend>_sglang.sh` 都提供默认参数，并允许通过环境变量覆盖一部分配置。不同模型会有额外变量，但日常最常用的是下面这些。

| 变量 | 说明 |
| --- | --- |
| `MODEL_PATH` | 模型目录 |
| `TOKENIZER_PATH` | tokenizer 目录，通常与模型目录相同 |
| `N_NODES` | 训练期望的 Ray 节点数 |
| `N_GPUS_PER_NODE` | 每节点注册的 HCU 数 |
| `ACTOR_BACKEND` | Actor 并行拓扑 |
| `ROLLOUT_BACKEND` | SGLang rollout 并行拓扑 |
| `TRAIN_BATCH_SIZE` | train dataloader batch |
| `VALID_BATCH_SIZE` | validation batch |
| `N_SAMPLES` | 每个 prompt 的 rollout 数量 |
| `MAX_NEW_TOKENS` | rollout 最大生成 token 数 |
| `TOTAL_TRAIN_STEPS` | 总训练 step 数 |
| `ACTOR_LR` | Actor learning rate |
| `ACTOR_MAX_TOKENS_PER_MB` / `ACTOR_MB_TOKENS` | Actor micro-batch token 容量 |
| `SGLANG_MEM_FRACTION_STATIC` | SGLang static memory fraction |
| `SGLANG_CONTEXT_LENGTH` | SGLang context length |
| `SGLANG_CHUNKED_PREFILL_SIZE` | chunked prefill 大小 |
| `SGLANG_PAGE_SIZE` | KV cache page size |
| `SGLANG_ATTENTION_BACKEND` | SGLang attention backend |
| `CLEAN_BEFORE_TRAIN` | 训练前是否清理本节点旧 AReaL/SGLang 进程 |

例如临时覆盖一个模型的训练步数和模型路径，可以直接：

```bash
MODEL_PATH="<MODEL_PATH>" \
TOKENIZER_PATH="<TOKENIZER_PATH>" \
TOTAL_TRAIN_STEPS=20 \
MAX_NEW_TOKENS=1024 \
bash run.sh \
  --model=<MODEL_KEY> \
  --backend=<BACKEND> \
  --ray-address=<HEAD_IP>:<RAY_PORT>
```

也可以使用 `run.sh` 提供的 `--model-path` 和 `--tokenizer-path` 参数。命令行覆盖更适合临时测试，长期使用的模型默认值建议仍写在对应模型 launcher 中。

---

## 6. 当前模型与后端

实际结果始终以 `bash run.sh --list` 为准，因为 `run.sh` 会扫描当前目录中真正存在的 launcher。

| 模型 key | FSDP2 | Megatron | 备注 |
| --- | --- | --- | --- |
| `qwen2_5_0_5b` | 支持 | 支持 | Dense |
| `qwen3_1_7b` | 支持 | 支持 | Dense |
| `qwen3_8b` | 支持 | 支持 | Dense，推荐作为 FSDP 验证模型 |
| `qwen3_30b_a3b_4layers` | 不提供 | 支持 | MoE，使用 Megatron |
| `glm5_4layers` | 不提供 | 支持 | MoE/MLA/DSA，使用 Megatron |

对于 Qwen3-30B-A3B 和 GLM-5，后续如果重新引入 PyTorch-native MoE 训练方案，应单独评估模型架构、expert 权重布局、HCU kernel 和 SGLang 在线权重更新，不建议恢复之前的 FSDP launcher 后直接用于正式训练。

---

# 7. `run.sh` 完整用法

进入统一入口目录：

```bash
cd <AREAL_HOME>/hcu_example/grpo
```

首先可以执行：

```bash
bash run.sh --help
```

当前主要功能分为四类：模型查询、训练启动、单节点 Ray 生命周期管理以及多节点 Ray 生命周期管理。

## 7.1 查询所有模型

```bash
bash run.sh --list
```

该命令根据当前目录里的 launcher 动态输出模型和可用 Actor backend。删除某个 launcher 后，重新执行 `--list` 即可确认结果。

## 7.2 搜索模型

```bash
bash run.sh --search=<PATTERN>
```

例如：

```bash
bash run.sh --search=qwen3
```

搜索采用 substring 匹配，适合不知道完整模型 key 时快速查找。

## 7.3 查询某个模型的 backend

```bash
bash run.sh \
  --model=<MODEL_KEY> \
  --backends
```

例如：

```bash
bash run.sh \
  --model=qwen3_8b \
  --backends
```

如果模型同时存在 FSDP 和 Megatron launcher，会同时显示两个 backend；如果只剩一个 launcher，训练时通常可以由 `run.sh` 自动推断 backend，但正式命令仍建议显式填写 `--backend`，便于日志和脚本复现。

## 7.4 查看模型配置

查看指定 backend：

```bash
bash run.sh \
  --model=<MODEL_KEY> \
  --backend=<fsdp|megatron> \
  --info
```

如果省略 `--backend`：

```bash
bash run.sh \
  --model=<MODEL_KEY> \
  --info
```

则会显示该模型当前所有 backend 的配置。`--info` 主要用于检查 launcher、环境 profile、模型路径、Actor/Rollout backend、节点数、每节点 GPU 数以及 batch 等默认值。修改模型 launcher 后建议先执行一次 `--info`，再真正启动训练。

## 7.5 FSDP 静态检查

```bash
bash run.sh --check-fsdp
```

该命令只审计当前仍存在的 `run_*_fsdp_sglang.sh`。 它会检查shell 语法、DP batch 条件和GPU budget。

## 7.6 Dry run

```bash
bash run.sh \
  --model=<MODEL_KEY> \
  --backend=<BACKEND> \
  --dry-run
```

该命令会解析 model、backend、launcher、profile、节点数、GPU 数、Ray 地址以及模型覆盖路径，然后退出，不进入正式训练。

## 7.7 指定模型和 tokenizer

```bash
bash run.sh \
  --model=<MODEL_KEY> \
  --backend=<BACKEND> \
  --model-path=<MODEL_PATH> \
  --tokenizer-path=<TOKENIZER_PATH> \
  --ray-address=<HEAD_IP>:<RAY_PORT>
```

## 7.8 覆盖节点数量

```bash
bash run.sh \
  --model=<MODEL_KEY> \
  --backend=<BACKEND> \
  --nodes=<N_NODES> \
  --gpus-per-node=<N_GPUS_PER_NODE> \
  --ray-address=<HEAD_IP>:<RAY_PORT>
```

`--nodes` 和 `--gpus-per-node` 只覆盖 cluster 资源期望值，**不会自动修改** `ACTOR_BACKEND`、`ROLLOUT_BACKEND`、TP、PP、DP 或 EP。修改节点数时必须确认模型脚本中的并行拓扑仍然能放进新的 GPU budget。

## 7.9 关闭训练前 cleanup

默认 launcher 会在训练前清理本节点遗留的 AReaL/SGLang 子进程，但不会停止 Ray。排查问题或希望保留已有进程时可以：

```bash
bash run.sh \
  --model=<MODEL_KEY> \
  --backend=<BACKEND> \
  --no-cleanup \
  --ray-address=<HEAD_IP>:<RAY_PORT>
```

一般不建议长期关闭 cleanup，因为上一次异常退出留下的 SGLang scheduler 或 model worker 可能继续占用 HCU 显存。

---

# 10. 单节点训练完整示例：Qwen3-8B FSDP

下面用 Qwen3-8B 演示一次完整的单节点训练流程。假设当前节点有 8 张 HCU，模型目录为 `<QWEN3_8B_MODEL_PATH>`，AReaL 示例目录为 `<AREAL_HOME>/hcu_example/grpo`。

首先进入环境并查看模型：

```bash
source <VENV_PATH>/bin/activate
cd <AREAL_HOME>/hcu_example/grpo

bash run.sh --model=qwen3_8b --backends
bash run.sh --model=qwen3_8b --backend=fsdp --info
```

如果模型不在 launcher 的默认目录，可以通过命令行覆盖。第一次运行建议先做一个短 smoke test，减少生成长度和训练步数，只验证 Ray、FSDP、SGLang、GRPO forward/backward 和在线权重更新主链路。

```bash
TOTAL_TRAIN_STEPS=2 \
MAX_NEW_TOKENS=128 \
bash run.sh \
  --model=qwen3_8b \
  --backend=fsdp \
  --model-path=<QWEN3_8B_MODEL_PATH> \
  --tokenizer-path=<QWEN3_8B_MODEL_PATH> \
  --restart-ray
```

`--restart-ray` 只适用于单节点任务。它会使用当前模型对应的 `AREAL_ENV_PROFILE` 重新启动本机 Ray head，并让新的 Ray daemon/worker 继承正确的 Python、AReaL、Megatron 和 SGLang 环境。启动后 `run.sh` 会继续执行 Qwen3-8B FSDP launcher。

如果 Ray 已经由其他命令正确启动，则不需要 `--restart-ray`，直接指定地址即可：

```bash
TOTAL_TRAIN_STEPS=2 \
MAX_NEW_TOKENS=128 \
bash run.sh \
  --model=qwen3_8b \
  --backend=fsdp \
  --model-path=<QWEN3_8B_MODEL_PATH> \
  --ray-address=<HEAD_IP>:<RAY_PORT>
```

FSDP Data Parallel 对 batch 有额外约束。若 Actor 为 `fsdp:d4p1t1`，`TRAIN_BATCH_SIZE` 至少应为 4 且能够被 4 整除。具体值以当前 launcher 为准，修改 FSDP DP 后要同步检查 batch，并运行：

```bash
bash run.sh --check-fsdp
```

---

# 11. 单节点训练：切换到 Megatron

同一个 Dense 模型如果同时提供 Megatron launcher，只需更换 backend：

```bash
bash run.sh \
  --model=qwen3_8b \
  --backend=megatron \
  --model-path=<QWEN3_8B_MODEL_PATH> \
  --restart-ray
```

因此 `run.sh` 的作用不是把一个 launcher 在运行时“转换成”另一个 backend，而是在当前模型已经存在的 launcher 中选择正确脚本。FSDP 和 Megatron 的 Actor 并行、micro-batch、optimizer 和权重更新配置仍然保留在各自独立的模型脚本里。

---

# 12. 多节点训练完整示例：Qwen3-30B-A3B Megatron

Qwen3-30B-A3B 当前只保留 Megatron + SGLang。下面假设训练需要两个 8 卡节点，其中一台作为 Ray head，另一台作为 worker。实际 IP 使用 `<HEAD_IP>` 和 `<WORKER_IP>` 代替，Ray 端口使用 `<RAY_PORT>`。

多节点开始之前，两个节点都必须确认以下内容一致：Python venv、AReaL 路径、Megatron 路径、SGLang 路径、模型路径和共享运行目录。尤其不要在 head 使用一个模型路径、worker 使用另一个路径。

## 12.1 Head 节点启动 Ray

在 head 节点执行：

```bash
source <VENV_PATH>/bin/activate
cd <AREAL_HOME>/hcu_example/grpo

bash run.sh \
  --ray-head \
  --model=qwen3_30b_a3b_4layers \
  --backend=megatron \
  --ray-address=<HEAD_IP>:<RAY_PORT>
```

`--ray-head` 只负责当前物理节点的 Ray 生命周期管理，执行完成后退出，不会启动训练。如果模型默认配置为多节点，命令结束时会提示 worker 的加入方式。

## 12.2 Worker 节点加入集群

在 worker 节点执行：

```bash
source <VENV_PATH>/bin/activate
cd <AREAL_HOME>/hcu_example/grpo

bash run.sh \
  --ray-worker \
  --model=qwen3_30b_a3b_4layers \
  --backend=megatron \
  --ray-address=<HEAD_IP>:<RAY_PORT> \
  --worker-ip=<WORKER_IP>
```

`--worker-ip` 必须是当前 worker 节点自己的可通信 IP。`run.sh` 会检查该 IP 是否属于本机，避免误把其他节点地址作为 worker bind address。

如果需要第三个或更多 worker，在每个新增物理节点上重复 `--ray-worker` 操作即可。`run.sh` 不会通过 SSH 自动操作另一台服务器，所以 head、worker 命令必须分别在对应节点执行。

## 12.3 检查 Ray 集群

所有 worker 加入后，在 head 上执行：

```bash
bash run.sh \
  --ray-status \
  --model=qwen3_30b_a3b_4layers \
  --backend=megatron \
  --ray-address=<HEAD_IP>:<RAY_PORT>
```

model-aware `--ray-status` 会根据 launcher 的 `N_NODES` 和 `N_GPUS_PER_NODE` 检查 alive node 数、总 GPU 数和每节点 GPU 数。只有资源符合预期后才建议启动训练。

也可以直接查看 Ray：

```bash
ray status --address=<HEAD_IP>:<RAY_PORT>
```

## 12.4 只在 Head 启动一次训练

Ray 集群建立完成后，只在 head 节点启动一份 training driver：

```bash
TOTAL_TRAIN_STEPS=2 \
MAX_NEW_TOKENS=128 \
bash run.sh \
  --model=qwen3_30b_a3b_4layers \
  --backend=megatron \
  --model-path=<QWEN3_MOE_MODEL_PATH> \
  --tokenizer-path=<QWEN3_MOE_MODEL_PATH> \
  --ray-address=<HEAD_IP>:<RAY_PORT>
```

不要在 worker 上重复执行训练命令。Ray scheduler 会根据 Actor 和 Rollout 的 placement group 自动在集群中创建 worker。

多节点模型不要使用 `--restart-ray`。该参数只会管理当前物理节点，无法替你重启其他服务器上的 Ray worker。多节点标准流程始终是：

```text
Head:   --ray-head
Worker: --ray-worker
Head:   --ray-status
Head:   正式训练
```

---

# 13. 不带模型时管理 Ray

`run.sh` 的 Ray-only 模式可以通过 `--profile` 在没有 `--model` 时运行。该功能适合先建立通用 Ray 集群，再决定具体训练模型。

启动 head：

```bash
bash run.sh \
  --ray-head \
  --profile=<qwen|qwen35|glm5|base> \
  --head-ip=<HEAD_IP> \
  --ray-address=<HEAD_IP>:<RAY_PORT>
```

加入 worker：

```bash
bash run.sh \
  --ray-worker \
  --profile=<qwen|qwen35|glm5|base> \
  --ray-address=<HEAD_IP>:<RAY_PORT> \
  --worker-ip=<WORKER_IP>
```

查看集群：

```bash
bash run.sh \
  --ray-status \
  --profile=<qwen|qwen35|glm5|base> \
  --ray-address=<HEAD_IP>:<RAY_PORT>
```

如果已经知道最终要跑哪个模型，更推荐使用 `--model=<MODEL_KEY>`，因为 `run.sh` 可以直接从模型 launcher 自动推导正确的环境 profile 和资源默认值。

---

# 14. Ray 端口与网络要求

默认 Ray head/GCS port 为 6379，但文档和脚本建议统一使用 `<RAY_PORT>` 表示实际配置。公共环境还定义了 worker、object manager 和 node manager 的固定端口范围。

| 变量 | 作用 |
| --- | --- |
| `RAY_PORT` | Ray head/GCS port |
| `RAY_MIN_WORKER_PORT` | worker port range 起点 |
| `RAY_MAX_WORKER_PORT` | worker port range 终点 |
| `RAY_OBJECT_MANAGER_PORT` | object manager port |
| `RAY_NODE_MANAGER_PORT` | node manager port |
| `RAY_TMPDIR` | Ray 本地临时目录 |

多节点之间必须允许这些端口双向通信。容器环境还应确认 `--network=host` 或等价网络配置能够让节点之间直接访问 `<HEAD_IP>:<RAY_PORT>`。

---

# 15. 模型脚本中的 Actor / Rollout backend

常见 Dense FSDP 配置：

```text
ACTOR_BACKEND=fsdp:d4p1t1
ROLLOUT_BACKEND=sglang:d1p1t4
```

常见 Megatron 配置：

```text
ACTOR_BACKEND=megatron:d1p2t4
ROLLOUT_BACKEND=sglang:d1p1t8
```

MoE 模型还可能使用 heterogeneous backend，例如：

```text
megatron:(attn:d1p2t4|ffn:d1p2t1e4)
```

这里的 `d/p/t/e/c` 分别对应 Data Parallel、Pipeline Parallel、Tensor Parallel、Expert Parallel 和 Context Parallel。修改 `N_NODES` 并不会自动修改这些并行维度，因此调整资源时必须同时重新计算 Actor world size、Rollout world size 和总 GPU budget。

---

# 16. 日志在哪里

默认训练日志位于：

```text
<AREAL_RUNS_ROOT>/<experiment>-<trial>-<timestamp>/
```

通常至少包含：

```text
train.log
runtime_env.txt
```

`train.log` 记录 AReaL、FSDP/Megatron、SGLang、Ray worker 等训练输出；`runtime_env.txt` 用于记录关键 Python 和源码路径，排查多节点环境不一致时非常重要。

如果训练在 `grpo_prepare_run()` 之前就失败，完整错误可能只输出在终端，还没有机会写入 `train.log`。因此分析启动失败时要同时保留终端输出。

---

# 17. 环境检查

在正式训练前，可以直接运行：

```bash
cd <AREAL_HOME>/hcu_example

AREAL_ENV_PROFILE=<qwen|qwen35|glm5|base> \
bash scripts/check_env.sh
```

至少应确认 Python、AReaL、Megatron、SGLang 均来自预期源码路径，并且 PyTorch 可以看到正确数量的 HCU。

也可以手工检查：

```bash
<VENV_PATH>/bin/python - <<'PY'
import torch
import areal
import sglang

print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("accelerator count:", torch.cuda.device_count())
print("areal:", areal.__file__)
print("sglang:", sglang.__file__)
PY
```

如果 Ray worker 的 import 路径和 driver 不一致，优先重新启动整个 Ray 集群，而不是只重新执行 training driver。

---

# 18. 常见问题

## 18.1 `ModuleNotFoundError: No module named 'areal'`

先检查 `PYTHON_BIN` 和 `PYTHONPATH` 是否来自当前 HCU 环境。Ray worker 使用的是 Ray daemon 启动时继承的环境，因此即使当前 shell 已经能 `import areal`，旧 Ray worker 仍可能继续使用错误路径。修改环境后停止并重新启动 Ray。

## 18.2 Ray 只能看到一个节点

确认 worker 使用 `<HEAD_IP>:<RAY_PORT>` 加入了正确的 head，并检查容器网络、防火墙、Ray 端口以及节点 IP。之后使用：

```bash
bash run.sh \
  --ray-status \
  --model=<MODEL_KEY> \
  --backend=<BACKEND> \
  --ray-address=<HEAD_IP>:<RAY_PORT>
```

核对 alive node 和 GPU 总数。

## 18.3 Ray GPU 数量不正确

启动 Ray 前检查：

```bash
echo "${CUDA_VISIBLE_DEVICES:-}"
echo "${HIP_VISIBLE_DEVICES:-}"
echo "${ROCR_VISIBLE_DEVICES:-}"
```

Ray 管理设备分配时，公共环境通常会清除外部遗留的设备 mask，然后由 `ray start --num-gpus=<N>` 注册资源。不要让旧的可见性变量把 8 卡节点限制成更少设备。

## 18.4 `Failed to connect to GCS`

这通常表示 `<HEAD_IP>:<RAY_PORT>` 上没有可用 Ray head、head 已退出或网络不可达。先执行：

```bash
ray status --address=<HEAD_IP>:<RAY_PORT>
```

确认 Ray 本身正常，再启动 AReaL。

## 18.5 SGLang `ConnectionRefusedError`

`Cannot connect to host <IP>:<PORT>` 通常只是 SGLang server 已退出后的后续错误，不应只根据最后一条 ConnectionRefused 定位。应向前搜索：

```text
OutOfMemory
HSA VMFault
SIGABRT
RuntimeError
Scheduler crashed
```

找到第一个真正导致 scheduler/model worker 退出的异常。

## 18.6 FSDP batch 不满足 DP

如果 Dense FSDP Actor 使用：

```text
fsdp:d<DP>p1t1
```

应满足：

```text
TRAIN_BATCH_SIZE >= DP
TRAIN_BATCH_SIZE % DP == 0
```

`grpo/common.sh` 中的通用 FSDP batch 检查会在模型加载前进行校验。这个逻辑与 Qwen3-MoE/GLM-5 无关，即使删除它们的 FSDP launcher 也应保留。