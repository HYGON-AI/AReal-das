# HCU 快速开始：单节点 Qwen3-8B

在单个 8 卡 HCU 节点上，以 Qwen3-8B、FSDP 和 SGLang 运行两步 GRPO 冒烟测试。

```bash
source /path/to/venv/bin/activate
export VENV="${VIRTUAL_ENV}"
export PYTHON_BIN="${VENV}/bin/python"
export AREAL_HOME=/path/to/AReaL
export MEGATRON_HOME=/path/to/hcu_megatron
export SGLANG_ROOT=/path/to/sglang
export MODEL_PATH=/model/qwen3/Qwen3-8B
export TOKENIZER_PATH="${MODEL_PATH}"

test -x "${PYTHON_BIN}"
test -d "${MEGATRON_HOME}/Megatron-LM"
test -d "${SGLANG_ROOT}/python/sglang"
test -f "${MODEL_PATH}/config.json"

cd "${AREAL_HOME}/hcu_example/grpo"
bash run.sh --model=qwen3_8b --backend=fsdp --model-path="${MODEL_PATH}" --tokenizer-path="${TOKENIZER_PATH}" --gpus-per-node=8 --info

TOTAL_TRAIN_STEPS=2 MAX_NEW_TOKENS=256 N_SAMPLES=1 \
bash run.sh --model=qwen3_8b --backend=fsdp \
  --model-path="${MODEL_PATH}" --tokenizer-path="${TOKENIZER_PATH}" \
  --gpus-per-node=8 --restart-ray
```

该命令启动或重启单节点 Ray。冒烟参数不应用于正式训练；检查点是否保存由示例配置决定。
