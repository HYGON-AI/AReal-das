# GRPO 模型脚本说明

本目录只负责 GRPO 模型配置和统一训练入口。

- 公共 Python/HCU/Megatron/SGLang 环境：`../common/common_env.sh`
- Ray head/worker 生命周期：`../scripts/`
- GRPO 公共启动流程：`common.sh`
- 模型统一入口：`run.sh`
- 模型特有配置：`run_<family>_<variant>_<actor_backend>_<rollout_backend>.sh`

## Launcher 命名

当前 launcher 不把参数量写入文件名。文件名表达模型架构、Actor backend 和 Rollout backend：

```text
run_qwen2_5_dense_fsdp_sglang.sh
run_qwen2_5_dense_megatron_sglang.sh
run_qwen3_dense_fsdp_sglang.sh
run_qwen3_dense_megatron_sglang.sh
run_qwen3_vl_fsdp_sglang.sh
run_qwen3_moe_megatron_sglang.sh
run_qwen3_5_dense_fsdp_sglang.sh
run_glm5_moe_megatron_sglang.sh
run_gemma3_vl_fsdp_sglang.sh
```

字段含义：

```text
family         = qwen2_5 / qwen3 / qwen3_5 / glm5 / gemma3
variant        = dense / moe / vl / vl_moe
actor backend  = fsdp / megatron
rollout        = sglang / vllm
```

`_sglang` 必须保留。未来增加 vLLM 时，使用同一命名槽位，例如：

```text
run_qwen3_dense_fsdp_vllm.sh
run_qwen3_vl_fsdp_vllm.sh
```

每个 launcher 顶部包含 `HCU_LAUNCHER_*` 元数据，`run.sh` 查询时只读取这些赋值，不会 source launcher，也不会触发 HCU
环境。新增 launcher 不需要修改中央注册文件。

## 当前模型

| Family    | Variant | Actor                   | Rollout    | 默认资源 | env profile | 备注                               |
| --------- | ------- | ----------------------- | ---------- | -------- | ----------- | ---------------------------------- |
| `qwen2_5` | dense   | FSDP DP2 / Megatron TP2 | SGLang TP2 | 1×8 HCU  | qwen        | 文本 Dense                         |
| `qwen3`   | dense   | FSDP DP4 / Megatron TP4 | SGLang TP4 | 1×8 HCU  | qwen        | Qwen3 Dense，不按参数量拆 launcher |
| `qwen3`   | vl      | FSDP DP4                | SGLang TP4 | 1×8 HCU  | qwen        | Geometry3K 多模态                  |
| `qwen3`   | moe     | Megatron TP/PP/EP       | SGLang TP8 | 2×8 HCU  | qwen        | MoE，独立拓扑                      |
| `qwen3_5` | dense   | FSDP DP4                | SGLang TP4 | 1×8 HCU  | qwen35      | Qwen3.5 Dense，fa3 + fp8 KV cache  |
| `glm5`    | moe     | Megatron TP/EP          | SGLang TP8 | 2×8 HCU  | glm5        | MLA/DSA/custom                     |
| `gemma3`  | vl      | FSDP DP4                | SGLang TP4 | 1×8 HCU  | gemma3      | 多模态，无 speculative/MTP 路径    |

同一个 Qwen3 Dense launcher 可以接收 Qwen3-1.7B 或 Qwen3-8B；具体权重由 `--model-path` 指定。`run.sh`
不读取模型目录，权重与 launcher 是否匹配在训练加载阶段暴露。

## 统一入口

列出所有 launcher：

```bash
bash run.sh --list
```

按 family、variant、Actor backend、Rollout backend 或旧 key 搜索：

```bash
bash run.sh --search=qwen3
```

Qwen3 Dense + FSDP + SGLang：

```bash
bash run.sh \
  --model=qwen3 \
  --variant=dense \
  --backend=fsdp \
  --rollout=sglang \
  --model-path=/model/qwen3/Qwen3-8B \
  --dry-run
```

Qwen3-1.7B 使用同一个 launcher，只替换 `--model-path`。不需要把 `1_7b` 或 `8b` 写进 launcher 文件名。

Qwen3 MoE：

```bash
bash run.sh \
  --model=qwen3 \
  --variant=moe \
  --backend=megatron \
  --rollout=sglang \
  --model-path=/model/qwen3/Qwen3-MoE \
  --dry-run
```

如果 family/variant/backend/rollout 不能唯一确定 launcher，`run.sh` 会报错并列出候选，不会静默选择错误的 Actor 或
Rollout backend。

Dense、MoE 和 VL 由 `--variant` 显式指定，`run.sh` 不做任何推断。选择只依据命令行参数和 launcher 顶部的
`HCU_LAUNCHER_*` 元数据；`--model-path` 仅透传给 launcher，不参与选择。

## 旧命令兼容

旧的 `run.sh` 模型 key 仍然可以使用：

```bash
bash run.sh --model=qwen3_8b --backend=fsdp --rollout=sglang --info
bash run.sh --model=qwen3_8b_fsdp_sglang --dry-run
```

这些 legacy key 会统一解析到新的无参数量 launcher：

```text
qwen2_5_0_5b           -> qwen2_5_dense_fsdp_sglang / megatron_sglang
qwen3_1_7b             -> qwen3_dense_fsdp_sglang / megatron_sglang
qwen3_8b               -> qwen3_dense_fsdp_sglang / megatron_sglang
qwen3_vl_4b            -> qwen3_vl_fsdp_sglang
qwen3_30b_a3b_4layers  -> qwen3_moe_megatron_sglang
glm5_4layers           -> glm5_moe_megatron_sglang
```

`--profile` 仍然表示 Ray/AReaL 运行环境
profile（`qwen`、`qwen35`、`glm5`、`gemma3`、`base`），不是模型规模。参数量只属于 `--model-path` 指向的模型目录，不属于
launcher identity。

## 新增模型必须做的事情

1. 创建 `run_<family>_<variant>_<actor_backend>_<rollout_backend>.sh`。
1. 在 launcher 顶部填写
   `HCU_LAUNCHER_FAMILY`、`HCU_LAUNCHER_VARIANT`、`HCU_LAUNCHER_ACTOR_BACKEND`、`HCU_LAUNCHER_ROLLOUT_BACKEND`
   和 `HCU_LAUNCHER_PROFILE`。
1. 不要把参数量、层数或具体 checkpoint 名称写入 launcher 文件名；如果拓扑确实不同，应增加有意义的 variant，而不是增加 `8b`、`30b`
   等字段。
1. 对照 AReaL v1.0.4 `cli_args.py` 和对应官方 YAML，确认每个 Hydra key 存在。
1. 设计正确的 Actor/Rollout TP/DP/PP/EP，保留模型特有的 SGLang/VLLM 参数在对应 launcher 内。
1. 不修改 `run.sh` 注册模型；`run.sh` 会扫描 rollout-aware launcher 文件自动发现。
1. 执行 `bash -n`、`bash run.sh --list`、`--info`、`--backends` 和 `--check-fsdp`。
1. 首次只做 10~20 step smoke test；如果环境/profile/source 路径变化，重启 Ray 后再测试。

## 单节点自动重建 Ray

```bash
bash run.sh \
  --model=qwen3 \
  --variant=dense \
  --backend=fsdp \
  --rollout=sglang \
  --model-path=/model/qwen3/Qwen3-8B \
  --restart-ray
```

`--restart-ray` 只支持 `N_NODES=1`，会在训练前重建当前节点 Ray。多节点必须分别执行
`--ray-head`、`--ray-worker`，最后只在 head 节点启动训练。`--dry-run` 不会启动或重启 Ray。
