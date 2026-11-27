# 准备四层模型检查点

本仓库的 GLM-5 和 DeepSeek‑V4‑Flash 示例使用四层检查点。DeepSeek‑R1 也可从官方 FP8 权重准备四层检查点，用于后续验证。脚本保留前四个
Transformer 层、这些层内的全部专家以及词嵌入、输出头等全局权重，关闭 MTP。四层模型用于训练和权重同步冒烟。

在仓库根目录执行下载和切分命令。将尖括号及其中的内容替换为实际路径。已有符合要求的原始检查点时可跳过下载。输出目录必须尚未存在，原始权重以只读方式使用。

## GLM-5 BF16

使用 78 层的官方
[zai-org/GLM-5 BF16](https://huggingface.co/zai-org/GLM-5/tree/c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2)。以下命令使用
Hugging Face CLI `hf` 仅下载四层模型所需的十个分片和元数据。

```bash
hf download zai-org/GLM-5 \
  --revision c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2 \
  --local-dir "<原始模型目录>" \
  config.json model.safetensors.index.json tokenizer.json tokenizer_config.json \
  chat_template.jinja generation_config.json \
  model-{00001,00002,00038,00039,00075,00076,00077,00078,00079,00282}-of-00282.safetensors
```

[GLM-5 脚本](../scripts/prepare_glm5_4layers.py)使用 PyTorch 和 `safetensors`，可在项目镜像的 CPU
环境运行。脚本链接只含保留张量的分片，逐分片重写同时含有其他层的文件。保留的张量沿用原始 dtype，输出目录额外占约 4.3 GiB。

```bash
python3 scripts/prepare_glm5_4layers.py \
  "<原始模型目录>" "<四层模型目录>"
```

<a id="deepseek-v4-flash"></a>

## DeepSeek‑V4‑Flash

使用官方
[deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1)
的 43 层检查点。以下命令下载原始分片命名下所需的六个分片以及 tokenizer 和模型配置。

```bash
hf download deepseek-ai/DeepSeek-V4-Flash \
  --revision 60d8d70770c6776ff598c94bb586a859a38244f1 \
  --local-dir "<原始模型目录>" \
  config.json model.safetensors.index.json tokenizer.json tokenizer_config.json \
  generation_config.json encoding/encoding_dsv4.py \
  model-{00001,00002,00003,00004,00005,00045}-of-00046.safetensors
```

[DeepSeek‑V4‑Flash 脚本](../scripts/prepare_deepseek_v4_flash_4layers.py)仅使用 Python
标准库。它保留原始混合精度权重，生成四层配置、权重索引和分片链接。如果源 tokenizer 缺少聊天模板，脚本会为短训补充文本对话模板。

```bash
python3 scripts/prepare_deepseek_v4_flash_4layers.py \
  "<原始模型目录>" "<四层模型目录>"
```

<a id="deepseek-r1"></a>

## DeepSeek‑R1 FP8

使用 61 层的官方
[deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1/tree/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad)。脚本保留
128 × 128 分块的 E4M3 权重及对应的 `weight_scale_inv`，关闭 MTP。以下命令只下载所需的五个分片。

```bash
hf download deepseek-ai/DeepSeek-R1 \
  --revision 56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad \
  --local-dir "<原始模型目录>" \
  config.json model.safetensors.index.json tokenizer.json tokenizer_config.json \
  generation_config.json configuration_deepseek.py modeling_deepseek.py \
  model-{00001,00002,00003,00004,00160}-of-000163.safetensors
```

[DeepSeek‑R1 脚本](../scripts/prepare_deepseek_r1_4layers.py)使用 PyTorch 和
`safetensors`，可在项目镜像的 CPU 环境运行。它链接只含保留张量的分片，重写混合分片。

```bash
python3 scripts/prepare_deepseek_r1_4layers.py \
  "<原始模型目录>" "<四层模型目录>"
```

四层 FP8 检查点的生成和张量已验证，端到端 GRPO 待验证。

## 挂载并启动

运行 Docker 时加入下面两项。冒号左侧是宿主机路径，右侧是容器内路径。四层目录中的符号链接指向原始分片，所以第一项两侧都填原始模型在宿主机上解析符号链接后的绝对路径。

```bash
-v "<宿主机原始模型绝对路径>:<宿主机原始模型绝对路径>:ro" \
-v "<宿主机四层模型目录>:/models/<四层模型目录名>:ro"
```

训练命令的 `training.actor.path` 填容器内的 `/models/<四层模型目录名>`。

[GLM-5 示例](../examples/README.md#glm5-megatron)和
[DeepSeek‑V4‑Flash 示例](../examples/README.md#deepseek-v4-flash-megatron)提供启动命令。DeepSeek‑V4‑Flash
还需要按示例挂载指定版本的 Megatron Core
dev。[DeepSeek‑R1 示例](../examples/README.md#deepseek-megatron)提供四层 FP8 试跑命令，端到端 GRPO 待验证。
