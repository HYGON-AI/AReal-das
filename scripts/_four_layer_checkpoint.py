# SPDX-License-Identifier: Apache-2.0

"""Keep the first four layers of an indexed Hugging Face checkpoint."""

import json
from pathlib import Path
from shutil import copy2

from safetensors import safe_open
from safetensors.torch import save_file

LAYER_PREFIXES = tuple(f"model.layers.{layer}." for layer in range(4))
GLOBAL_WEIGHTS = {"lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"}


def write_four_layer_checkpoint(source: Path, target: Path, config, extra_files=()):
    weight_map = json.loads((source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    weights = {
        name: shard
        for name, shard in weight_map.items()
        if name in GLOBAL_WEIGHTS or name.startswith(LAYER_PREFIXES)
    }
    if not GLOBAL_WEIGHTS <= weights.keys() or not all(
        any(name.startswith(prefix) for name in weights) for prefix in LAYER_PREFIXES
    ):
        raise ValueError(
            "Checkpoint index is missing global weights or one of the first four layers"
        )

    # Validate the referenced files before creating the output directory.
    shards = {}
    for name, shard in weights.items():
        shards.setdefault(shard, []).append(name)
    for shard in shards:
        (source / shard).resolve(strict=True)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (source / name).resolve(strict=True)

    target.mkdir(parents=True, exist_ok=False)
    config.update(num_hidden_layers=4, num_nextn_predict_layers=0)
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        *extra_files,
    ):
        if (source / name).is_file():
            copy2(source / name, target / name)

    for shard, names in sorted(shards.items()):
        original = source / shard
        output = target / shard
        with safe_open(original, framework="pt", device="cpu") as weights_file:
            if set(names) == set(weights_file.keys()):
                output.symlink_to(original.resolve())
            else:
                # Mixed shards must omit the unselected tensors even for readers
                # that iterate all tensors in a referenced shard.
                tensors = {name: weights_file.get_tensor(name) for name in names}
                save_file(tensors, output, metadata=weights_file.metadata())
                del tensors
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights}, indent=2) + "\n"
    )
