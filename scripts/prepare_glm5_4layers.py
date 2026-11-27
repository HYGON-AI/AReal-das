# SPDX-License-Identifier: Apache-2.0

"""Prepare a four-layer GLM-5 checkpoint from the official BF16 release."""

import argparse
import json
from pathlib import Path

from _four_layer_checkpoint import write_four_layer_checkpoint


def prepare(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    config = json.loads((source / "config.json").read_text())
    if (
        config["model_type"] != "glm_moe_dsa"
        or config["num_hidden_layers"] != 78
        or config.get("dtype", config.get("torch_dtype")) != "bfloat16"
        or config.get("quantization_config")
    ):
        raise ValueError("Expected the official 78-layer GLM-5 BF16 checkpoint")

    write_four_layer_checkpoint(source, target, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original GLM-5 BF16 checkpoint")
    parser.add_argument("target", type=Path, help="New four-layer checkpoint directory")
    args = parser.parse_args()
    prepare(args.source, args.target)
