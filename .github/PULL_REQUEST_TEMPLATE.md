## 改动说明

<!-- 简要说明改动目的、影响范围与兼容性。 -->

## 改动类型

- [ ] HCU 适配或兼容性修复
- [ ] 训练/rollout 示例
- [ ] 文档更新
- [ ] 测试或 CI
- [ ] 性能优化
- [ ] 其他

## 验证情况

<!-- HCU 相关改动请填写；纯文档改动可填写“不适用”。 -->

- HCU 型号与卡数：
- DTK / Python / PyTorch / SGLang / Megatron 版本：
- 模型与 Actor/Rollout 后端：
- 执行命令：
- 验证结果或关键日志：

## CI 模型短训

<!-- 可选：留空时运行 qwen3-4b-megatron 基线；填写已登记的 case 名称则只运行所填 case，多个名称用英文逗号分隔。 -->

<!-- 当前可用 case：qwen3-4b-megatron。完整清单及新增方式见 .github/model-cases.yaml 和 .github/workflows/README.md。 -->

<!-- 使用已有 case 无需修改清单；新增 case 时，需随 PR 提交清单项和训练配置，并填写模型在共享根目录下的相对路径。 -->

CI_MODELS:

## 提交前检查

- [ ] 已执行 `git diff --check`
- [ ] 已执行相关 Python 和 Shell 语法检查
- [ ] 已运行与改动相关的测试或 HCU 冒烟验证
- [ ] 未提交模型权重、日志、checkpoint、内部地址、账号或密钥
- [ ] 已保留上游 Apache-2.0 版权与许可证声明
- [ ] 已更新 README、HCU 示例或文档（如适用）

## 其他说明

<!-- 记录已知限制、回退方式或需要审阅者关注的事项。 -->
