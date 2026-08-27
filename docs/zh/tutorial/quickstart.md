# HCU 快速开始：单节点 Qwen3-8B

在单个 8 卡 HCU 节点上，以 Qwen3-8B、FSDP 和 SGLang 运行两步 GRPO 冒烟测试。

## 镜像安装路径

推荐使用包含 HCU 运行时、HCU 版 PyTorch 和 SGLang 的基础镜像。请从[光源社区](https://developer.sourcefind.cn/servicelist/detail?post_id=1abf923f-5a33-11f1-9e57-0242ac150003)获取实际镜像名称和标签：

```bash
docker pull REPOSITORY:TAG
```

启动容器时需要透传 HCU 设备，并将 AReaL、模型和外部源码目录挂载进去：

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
进入容器后，在仓库根目录执行：

```bash
python -m pip install -r requirements-hcu.txt
python -m pip install -e . --no-deps
```

`--no-deps` 可避免覆盖基础镜像中的 HCU 版 PyTorch、SGLang 和 Megatron。更多容器准备说明见 [`hcu_example/user_guide.md`](../../../hcu_example/user_guide.md)。

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