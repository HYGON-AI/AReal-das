# HCU 适配代码

`hcu_areal` 为官方 AReaL v2.1.0 提供 HCU 运行时适配。训练示例及 YAML 配置见
[examples](../examples/README.md)。

## 代码结构

- `launch.py`：读取统一 YAML，准备设备环境，并显式启动训练脚本和服务进程。
- `patch_init.py`：保留 AReaL v2.1.0 包入口的原有导出，并在启用 HCU 的 HIP 进程中激活公共补丁；启动器只复制这个
  `areal/__init__.py`，其余上游内容通过链接访问。
- `areal_adaptor.py`：集中登记补丁、模型开关和共享 AWEX 分派；各进程在自身入口只应用相关补丁。
- `patch_utils.py`：补丁注册、应用和回滚。
- `api/`、`infra/`、`engine/`、`models/`、`trainer/`、`workflow/`、`v2/`：按 AReaL 上游组件组织具体适配实现。
- `v2/weight_update/awex/fsdp_adapter.py`、`megatron_adapter.py`、`sglang_adapter.py`：分别对应
  AReaL 的三个 AWEX adapter；模型布局及公共预检位于同目录的辅助文件。
- `third_party/`：对 Megatron-Bridge 等第三方源码的适配，按其实际模块路径命名。
- `tests/unit/`：HCU 适配单元测试。
- `.github/scripts/`：CI 镜像检查、单元测试和模型冒烟入口。

原 `features_manager/` 中的登记逻辑已合并至 `areal_adaptor.py`。共享 AWEX 适配按模型开关统一分派，Gemma3
单独启用时也会登记所需的推理适配。旧自动启动目录和全局延迟导入机制已移除；模型和部署环境仍需分别验证。

共享钩子的诊断标识按启用模型选择：推理适配优先使用 Qwen3.5、GLM5、DeepSeek、Gemma3 的标识；Megatron 权重更新优先使用 GLM5，否则使用
DeepSeek。标识用于定位注册项，实际分派还会检查对应模型开关。

项目使用 `3rdparty/AReaL`、`3rdparty/Megatron-LM-das` 和 `3rdparty/Megatron-bridge-das`
子模块；SGLang 由镜像提供。克隆后先运行 `git submodule update --init --recursive`。启动器在
`runtime.cache_dir` 指定的目录下生成 AReaL 入口目录；未指定时使用 `AREAL_CACHE_DIR` 或系统临时目录。启动器只复制
`__init__.py`，缓存入口中的一个符号链接指向上游包目录，其他模块从真实的上游路径导入，不修改子模块；源入口不符合官方 v2.1.0 时会报错。

启动器会把缓存入口和项目内 Megatron、Bridge 加入各进程的 Python 路径，同时保留镜像中的
SGLang。多机运行时各节点需要使用相同的代码路径、镜像和缓存路径；对于已有 Ray 集群，缓存入口必须位于所有节点可访问的位置，或者先在每个节点执行
`python -m hcu_areal.launch --install-init` 并使用相同的 `AREAL_CACHE_DIR`。

通过启动器设置 `HCU_AREAL_ENABLE=1` 后，导入缓存入口中 `areal` 的 HIP 进程会应用公共补丁，不会预先导入 Megatron 或
SGLang。直接导入未经处理的子模块不会启用 HCU 入口。训练与推理后端所需补丁仍在明确的组件边界按需安装；自行启动训练脚本时使用
`python -m hcu_areal.launch --driver 脚本路径 脚本参数`。已启动的 Python 进程不会自动更新补丁。

AWEX 训练和推理元数据诊断仅在设置 `HCU_AREAL_PATCH_TRACE_DIR` 时登记；这不影响模型权重转换和传输前检查。

controller、gateway 和 Gemma3 视觉工作流采用对应路径中的显式适配实现；升级 AReaL
时需重新审阅这些适配。模型适配按目标模型分发，具体精度与布局边界见专项验证材料。

DeepSeek R1 的官方检查点使用 FP8 E4M3、128×128 分块和 `weight_scale_inv`。训练端加载交给固定版本的
Megatron-Bridge-das，按 scale 将检查点反量化为 BF16。AWEX 权重同步由相应训练和推理适配器处理。

DeepSeek BF16 MLA 使用 SGLang-DAS 自带的 HCU 算子。数值检查使用的 SGLang-DAS 提交为
`453edca64008bdaf162bb42dcebda14bd1b4927a`；该提交仍需在后端初始化时补齐 KV cache 引用。依赖旧版 DCU 补丁的
SGLang 镜像不在此路径的验证范围内。

## Megatron 优化器 CPU 卸载

默认不启用。可在启动配置的 `runtime.env` 中设置 `HCU_AREAL_OPTIMIZER_CPU_OFFLOAD_FRACTION`，取值为
`(0, 1]`，将该比例传给 Megatron 的优化器 CPU 卸载配置。按参数张量分配时，实际比例可能不完全相同。可选设置
`HCU_AREAL_OPTIMIZER_PIN_CPU_GRADS: 'false'`，关闭 CPU 梯度缓冲的锁页；不设置时沿用 Megatron
默认值。该开关必须与卸载比例同时设置。

CPU 卸载使用 Megatron 的混合设备优化器，CPU AdamW 状态为 FP32，即使训练配置中的优化器名为
`adam_bf16`。锁页设置可能影响传输效率；完整模型的显存、主机内存和训练数值需要在目标环境中验证。

可选设置 `HCU_AREAL_MEGATRON_AWEX_RELEASE_GRADS: '1'`，在优化器更新后的常规 Megatron AWEX 权重同步期间暂时释放
DDP 梯度缓冲，同步结束后重新分配并清零。此设置不备份梯度，适用于同步后下一轮会重新计算梯度的训练流程；DTE
路径沿用其已有的缓冲管理。默认不启用，具体模型仍需验证同步及后续训练。

## 验证

```bash
export PYTHONPATH="$(python3 -m hcu_areal.launch --pythonpath)"
python3 -m pytest -q tests/unit
```

单元测试使用 CPU 张量，不验证完整模型或 HCU 通信。测试范围及硬件冒烟入口见 [tests/README.md](../tests/README.md)。
