# GRPO 模型脚本说明

本目录只负责 GRPO 模型配置和统一训练入口。

- 公共 Python/HCU/Megatron/SGLang 环境：`../common/common_env.sh`
- Ray head/worker 生命周期：`../scripts/`
- GRPO 公共启动流程：`common.sh`
- 模型统一入口：`run.sh`
- 模型特有配置：`run_<model>_<train_backend>_<rollout_backend>.sh`

## 当前模型

| 模型 key                | Backend  | Actor            | Rollout              | 默认资源             | profile | 状态            |
| ----------------------- | -------- | ---------------- | -------------------- | -------------------- | ------- | --------------- |
| `qwen2_5_0_5b`          | FSDP2    | FSDP DP2         | SGLang TP2           | 1×8 HCU（实际使用4） | qwen    | 新增/待实机验证 |
| `qwen2_5_0_5b`          | Megatron | Megatron TP2     | SGLang TP2           | 1×8 HCU（实际使用4） | qwen    | 新增/待实机验证 |
| `qwen3_1_7b`            | FSDP2    | FSDP DP4         | SGLang TP4           | 1×8 HCU              | qwen    | 已迁移          |
| `qwen3_1_7b`            | Megatron | Megatron TP4     | SGLang TP4           | 1×8 HCU              | qwen    | 已迁移          |
| `qwen3_8b`              | FSDP2    | FSDP DP4         | SGLang TP4           | 1×8 HCU              | qwen    | 已运行          |
| `qwen3_8b`              | Megatron | Megatron TP4     | SGLang TP4           | 1×8 HCU              | qwen    | 已运行          |
| `qwen3_vl_4b`           | FSDP2    | FSDP DP4         | SGLang TP4（多模态） | 1×8 HCU              | qwen    | Geometry3K 示例 |
| `qwen3_30b_a3b_4layers` | Megatron | Megatron TP8     | SGLang TP8           | 2×8 HCU              | qwen    | 已运行基线      |
| `glm5_4layers`          | Megatron | Megatron TP8/EP8 | SGLang TP8           | 2×8 HCU              | glm5    | custom          |

查看：

```bash
bash run.sh --list
```

运行：

```bash
bash run.sh --model=<name> --model-path=<path-to-model> --ray-address=<head-node-ip>:6379
```

## 参数职责

所有模型脚本保持：

```text
CLUSTER_CONFIG
DATA_CONFIG
ACTOR_CONFIG
ROLLOUT_CONFIG
SGLANG_CONFIG
TRAINER_CONFIG
```

不要把模型特有的 TP/EP/recompute/SGLang backend 参数下沉到 `common.sh`。

## 新增模型必须做的事情

1. 创建 `run_<model>_<backend>_sglang.sh`，其中 `<backend>` 为 `fsdp` 或 `megatron`。
1. 对照 AReaL v1.0.4 `cli_args.py` 和对应官方 YAML，确认每个 Hydra key 存在。
1. 选择正确的 `AREAL_ENV_PROFILE`。
1. 设计合理的 Actor/Rollout TP/DP/PP/EP，而不是机械复制其他模型。
1. 不要修改 `run.sh` 注册模型；`run.sh` 会根据 launcher 文件名自动发现模型和 backend。
1. 更新本 README 的模型表。
1. 执行 `bash -n run_<model>_<backend>_sglang.sh`，并使用
   `bash run.sh --list`、`bash run.sh --model=<model> --backends` 和
   `bash run.sh --model=<model> --backend=<backend> --info` 检查发现结果。
1. 对 FSDP launcher 额外执行 `bash run.sh --check-fsdp`。
1. 首次只做 10~20 step smoke test。
1. 如果环境/profile/source 路径变化，重启 Ray 后再测试。

详细步骤见上一级 `README.md`。

### 单节点自动重建 Ray

单节点调试时可使用：

```bash
bash run.sh --model=qwen3_1_7b_megatron_sglang --restart-ray
```

或显式指定本机 head 地址：

```bash
bash run.sh \
  --model=qwen3_1_7b_megatron_sglang \
  --ray-address=<head-node-ip>:6379 \
  --restart-ray
```

该参数会在训练前停止当前节点旧 Ray、按模型 profile 创建新 head 并验证 worker 环境。仅支持 `N_NODES=1`；多节点请继续手动管理所有节点的
Ray。
