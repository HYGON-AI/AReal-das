# AReaL-das 贡献指南

感谢你关注 AReaL-das。该项目是 AReaL 的 HCU 平台适配版本，欢迎提交 HCU 适配、稳定性修复、测试、文档和示例改进。

## 可接受的贡献范围

- HCU 训练、推理与分布式运行适配
- HCU 环境下的性能、稳定性和兼容性修复
- HCU 示例、测试和文档完善
- 与上游 AReaL 同步时所需的兼容性改动

新增公共接口、依赖、模型支持或较大重构前，请先通过 Issue 或 Draft PR 说明目的、影响范围与验证计划。

## 开发环境

请使用与目标 HCU 环境匹配的软件栈：

- Linux 与 Python 3.11
- 匹配设备和驱动的 DTK
- HCU 版本的 PyTorch、SGLang、Ray；按需安装 Megatron-LM 与 Megatron-Bridge

安装项目依赖：

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-hcu.txt
~~~

HCU 专用包可能无法从公共 PyPI 获取，请使用与 DTK 版本匹配的发行包或软件源。具体路径和 Qwen3-8B 冒烟测试请参阅 [README](README.md)。

## 提交改动

1. 从最新主分支创建描述清晰的分支。
2. 保持改动范围聚焦；不要混入格式化无关文件或生成产物。
3. 不要提交模型权重、训练日志、checkpoint、内部 IP、内部地址、账号、令牌、密钥或密码。
4. 修改源码时，保留已有的 SPDX 和上游版权声明；新增海光修改应包含：

~~~text
Copyright (c) 2026 Hygon Information Technology Co., Ltd.
~~~

5. 同步或修改上游 AReaL 代码时，必须保留 Apache-2.0 许可证和原始版权声明。

## 提交前检查

提交前至少执行：

~~~bash
git diff --check
pre-commit run --all-files
~~~

并运行与改动相关的测试。例如：

~~~bash
pytest -q tests/test_hcu.py
~~~

涉及 HCU 运行路径、启动脚本或模型适配的改动，应至少完成一次可复现验证，并在 PR 中提供：

- HCU 型号与卡数
- DTK、Python、PyTorch、SGLang 和 Megatron 版本
- 模型名称与训练/rollout 后端
- 实际执行命令
- 验证结果或关键日志摘要

请勿把短冒烟测试结果表述为完整训练性能结论。

## Pull Request 要求

PR 描述应包括：

- 改动目的和影响范围
- 是否影响公开 API、配置或依赖
- 已执行的检查和测试
- HCU 验证环境与结果（如适用）
- 兼容性、已知限制和回退方式（如适用）

提交信息建议使用清晰的类型前缀，例如：

~~~text
feat(hcu): add HCU rollout compatibility check
fix(grpo): handle tokenizer path validation
docs: update HCU quickstart
~~~

## 文档与测试

功能改动应同步更新中文文档、示例或测试。若改动影响安装、依赖、环境变量、模型支持范围或启动命令，必须更新根目录 README 和相关 HCU 文档。

## 许可证

AReaL-das 采用 [Apache License 2.0](LICENSE)。贡献代码即表示你有权按该许可证提交，并同意保留本项目与上游的版权和许可证声明。
