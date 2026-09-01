# HCU Quickstart: Qwen3-8B on One Node

Run a two-step GRPO smoke test with Qwen3-8B, FSDP, and SGLang on one 8-HCU node.

## Container installation

Use an HCU base image that includes the HCU runtime, HCU PyTorch, and SGLang. Obtain the
current image name and tag from the
[SourceFind community](https://developer.sourcefind.cn/servicelist/detail?post_id=1abf923f-5a33-11f1-9e57-0242ac150003):

```bash
docker pull REPOSITORY:TAG
```

Start the container with HCU devices and mounted source/model directories:

```bash
docker run -it --name areal-das-hcu --shm-size=64G \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --ipc=host --network=host \
  --workdir=/workspace --privileged \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v <host-workspace>:/workspace \
  REPOSITORY:TAG /bin/bash
```

Inside the container, install AReaL without replacing HCU binary packages:

```bash
python -m pip install -r requirements-hcu.txt
python -m pip install -e . --no-deps
```

See [`hcu_example/user_guide.md`](../../../hcu_example/user_guide.md) for the full
container setup.

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

This starts or restarts single-node Ray. Do not use smoke-test settings for production
training. Checkpoint saving depends on the example configuration.
