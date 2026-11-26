# SPDX-License-Identifier: Apache-2.0
"""Run a two-step HCU smoke test for a single-node GSM8K configuration."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

STEPS = 2
BASELINE = "qwen3-4b-megatron"
GSM8K_CONFIG = "3rdparty/AReaL/examples/math/gsm8k_grpo.yaml"
SMOKE_ENV_CONFIG = Path(__file__).resolve().parents[1] / "model-smoke-env.yaml"


def check_model_and_devices(model_dir, case_name, gpus):
    import torch

    model = json.loads((model_dir / "config.json").read_text())
    if not any(model_dir.glob("*.safetensors")):
        raise ValueError("Model weights are missing")
    if case_name == BASELINE and (
        model.get("model_type") != "qwen3"
        or model.get("hidden_size") != 2560
        or model.get("num_hidden_layers") != 36
        or model.get("quantization_config") is not None
        or model.get("torch_dtype", model.get("dtype")) != "bfloat16"
    ):
        raise ValueError("The baseline requires unquantized BF16 Qwen3-4B weights")
    count = torch.cuda.device_count()
    print(
        f"torch={torch.__version__}, hip={torch.version.hip}, devices={count}",
        flush=True,
    )
    if torch.version.hip is None or count < gpus:
        raise ValueError(f"Smoke training requires HCU PyTorch and {gpus} visible HCUs")
    architectures = {
        torch.cuda.get_device_properties(index).gcnArchName.partition(":")[0]
        for index in range(gpus)
    }
    print(f"HCU architectures: {', '.join(sorted(architectures))}", flush=True)
    return architectures


def make_dataset(path):
    from datasets import Dataset, DatasetDict

    if path.exists():
        shutil.rmtree(path)
    pairs = [(1, 1), (2, 3), (3, 4), (4, 5), (5, 2), (6, 1), (7, 3), (8, 2)]
    records = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"What is {a} + {b}? Put your final answer within \\boxed{{}}.",
                }
            ],
            "answer": f"#### {a + b}",
        }
        for a, b in pairs
    ]
    dataset = Dataset.from_list(records)
    DatasetDict({"train": dataset, "test": dataset.select(range(4))}).save_to_disk(
        str(path)
    )


def prepare_config(config, model_dir, scratch, cache, train_log, gpus, ray_temp_dir):
    runtime, training = config["runtime"], config["training"]
    cluster = training["cluster"]
    if runtime["base_config"] != GSM8K_CONFIG or cluster.get("n_nodes", 1) != 1:
        raise ValueError("Smoke training currently supports single-node GSM8K examples")

    runtime.update(
        cache_dir=str(cache), temp_dir=str(scratch / "tmp"), driver_log=str(train_log)
    )
    runtime["ray"].update(mode="auto", num_gpus=gpus, temp_dir=ray_temp_dir)
    runtime.setdefault("env", {}).update(
        HF_DATASETS_OFFLINE="1",
        HF_HUB_OFFLINE="1",
        WANDB_MODE="disabled",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    training["actor"]["path"] = str(model_dir)
    base_training = yaml.safe_load(Path(GSM8K_CONFIG).read_text())
    specs = training["actor"].setdefault(
        "scheduling_spec", base_training["actor"]["scheduling_spec"]
    )
    for spec in specs:
        spec.setdefault("cpu", 2)
    cluster.update(fileroot=str(scratch / "experiments"), n_gpus_per_node=gpus)
    cluster.setdefault("name_resolve", {})["nfs_record_root"] = str(
        scratch / "name_resolve"
    )
    training.setdefault("gconfig", {}).update(max_new_tokens=256, n_samples=2)
    training.update(total_train_epochs=1, total_train_steps=STEPS)
    for name in ("train_dataset", "valid_dataset"):
        training.setdefault(name, {})["path"] = str(scratch / "math-smoke-dataset")
    for name in ("evaluator", "recover", "saver"):
        training.setdefault(name, {}).update(
            freq_steps=None, freq_epochs=None, freq_secs=None
        )
    training["recover"]["mode"] = "disabled"


def apply_smoke_env(config, case_name, architectures):
    version_file = Path("/opt/dtk/.info/version-dev")
    if not version_file.is_file():
        return
    dtk_version = ".".join(version_file.read_text().strip().split(".")[:2])
    profiles = yaml.safe_load(SMOKE_ENV_CONFIG.read_text())
    overrides = {}
    for profile in profiles.get(dtk_version, ()):
        architecture = profile.get("architecture")
        cases = profile.get("cases")
        if (architecture is None or architecture in architectures) and (
            cases is None or case_name in cases
        ):
            overrides.update(profile["env"])
    if overrides:
        config["runtime"]["env"].update(overrides)
        print(
            f"CI environment overrides for DTK {dtk_version}: {overrides}", flush=True
        )


def check_log(log):
    steps = re.findall(r"Train step (\d+)/\d+ done\.", log)
    if steps != [str(step) for step in range(1, STEPS + 1)]:
        raise ValueError(
            f"Expected exactly {STEPS} completed training steps, got {steps}"
        )
    if "rollout/accepted" not in log:
        raise ValueError("Missing rollout acceptance metric")
    for step in range(1, STEPS + 1):
        if f"Weight update v{step} completed (ok," not in log:
            raise ValueError(f"Missing successful weight update v{step}")
    for metric in ("timeperf/update_weights", "ppo_actor/update/grad_norm"):
        if sum(metric in line for line in log.splitlines()) < STEPS:
            raise ValueError(f"Missing {metric} metrics for the two training steps")
    if "Error killing forked service" in log:
        print("::warning::A service did not stop within its shutdown timeout")


def main():
    case_name = os.environ["CI_MODEL_CASE"]
    gpus = int(os.environ["CI_MODEL_GPUS"])
    model_dir = Path(os.environ["MODEL_PATH"])
    scratch = Path(os.environ["AREAL_RUNTIME_ROOT"])
    runs = Path(os.environ["AREAL_RUNS_ROOT"])
    cache = Path(os.environ["AREAL_CACHE_ROOT"])
    marker = Path(os.environ["CI_SUCCESS_MARKER"])
    train_log = runs / "train.log"
    os.umask(0)
    marker.unlink(missing_ok=True)
    for directory in (runs, cache, scratch / "tmp", scratch / "name_resolve"):
        directory.mkdir(parents=True, exist_ok=True)
    train_log.unlink(missing_ok=True)
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        os.environ.pop(key, None)
    if case_name == BASELINE:
        os.environ["SGLANG_ROCM_USE_AITER_MOE"] = "0"
    os.environ.update(
        HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1", WANDB_MODE="disabled"
    )

    try:
        config = yaml.safe_load(Path(os.environ["CI_EXAMPLE_CONFIG"]).read_text())
        architectures = check_model_and_devices(model_dir, case_name, gpus)
        # Ray appends a session directory and socket name to this path.
        with tempfile.TemporaryDirectory(prefix="ray-ci-", dir="/tmp") as ray_temp_dir:
            prepare_config(
                config, model_dir, scratch, cache, train_log, gpus, ray_temp_dir
            )
            apply_smoke_env(config, case_name, architectures)
            make_dataset(scratch / "math-smoke-dataset")
            launch_config = scratch / "model-smoke-launch.yaml"
            launch_config.write_text(yaml.safe_dump(config, sort_keys=False))
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "hcu_areal.launch",
                    "--config",
                    str(launch_config),
                ],
                check=True,
            )
        check_log(train_log.read_text(errors="replace"))
        marker.write_text(f"{case_name} completed {STEPS} training steps.\n")
    finally:
        subprocess.run(
            ["chmod", "-R", "a+rwX", str(runs), str(scratch), str(cache)], check=False
        )


if __name__ == "__main__":
    main()
