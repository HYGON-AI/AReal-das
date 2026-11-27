# SPDX-License-Identifier: Apache-2.0

"""Prepare a four-layer DeepSeek-R1 checkpoint from the official block-FP8 release."""

import argparse
import json
from pathlib import Path

from _four_layer_checkpoint import write_four_layer_checkpoint


def prepare(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    config = json.loads((source / "config.json").read_text())
    quantization = config.get("quantization_config") or {}
    if (
        config["model_type"] != "deepseek_v3"
        or config["num_hidden_layers"] != 61
        or quantization.get("quant_method") != "fp8"
        or quantization.get("fmt") != "e4m3"
        or quantization.get("weight_block_size") != [128, 128]
    ):
        raise ValueError(
            "Expected a 61-layer DeepSeek-R1 checkpoint with 128×128 block FP8"
        )

    write_four_layer_checkpoint(
        source,
        target,
        config,
        extra_files=("configuration_deepseek.py", "modeling_deepseek.py"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original DeepSeek-R1 FP8 checkpoint")
    parser.add_argument("target", type=Path, help="New four-layer checkpoint directory")
    args = parser.parse_args()
    prepare(args.source, args.target)
