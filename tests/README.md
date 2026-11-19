# HCU 适配测试

`unit/` 覆盖补丁应用与回滚、进程启动和 Ray 环境、版本兼容，以及模型权重名称、布局与 FP8 数值。测试使用小型 CPU 张量，不需要模型权重或 HCU
设备；运行环境需要 Python 3.11/3.12、pytest、psutil、PyYAML 和支持 float8 的 PyTorch。

在仓库根目录运行：

```bash
export PYTHONPATH="$(python3 -m hcu_areal.launch --pythonpath)"
python3 -m pytest -q -rs tests/unit
```

完整运行 Guard 测试还需要可导入的 AReaL 子模块及其依赖。使用项目镜像并初始化子模块，检查 pytest 的跳过原因。

PR 的 CPU 测试入口是 `.github/scripts/run_hcu_unit_tests.sh`，CI 在配置的 HCU
镜像中运行且不暴露设备。多卡、真实模型和通信链路不由这些单元测试证明。模型短训按 PR 中可选的 `CI_MODELS:` 选择；未填写时运行 Qwen3-4B 两步基线。可用
case 及资源要求见 `.github/workflows/README.md`。
