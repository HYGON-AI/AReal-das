# SPDX-License-Identifier: Apache-2.0

"""Create a four-layer view of the official DeepSeek-V4-Flash checkpoint."""

import argparse
import json
from pathlib import Path


# Match the checkpoint's encoding/encoding_dsv4.py for text-only chat mode.
CHAT_TEMPLATE = """
{{- bos_token -}}
{%- for message in messages -%}
  {%- if message.role == 'system' -%}
    {{- message.content -}}
  {%- elif message.role in ('user', 'developer') -%}
    {{- '<｜User｜>' + message.content -}}
    {%- if (loop.last and add_generation_prompt)
           or (not loop.last and messages[loop.index].role == 'assistant') -%}
      {{- '<｜Assistant｜></think>' -}}
    {%- endif -%}
  {%- elif message.role == 'assistant' -%}
    {{- message.content + eos_token -}}
  {%- else -%}
    {{- raise_exception('This four-layer example only supports text chat.') -}}
  {%- endif -%}
{%- endfor -%}
"""


def prepare(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    config = json.loads((source / "config.json").read_text())
    if config["model_type"] != "deepseek_v4" or config["num_hidden_layers"] != 43:
        raise ValueError("Expected the official 43-layer DeepSeek-V4-Flash checkpoint")

    config["num_hidden_layers"] = 4
    config["num_nextn_predict_layers"] = 0
    config["compress_ratios"] = config["compress_ratios"][:4]
    index = json.loads((source / "model.safetensors.index.json").read_text())
    if "embed.weight" not in index["weight_map"]:
        raise ValueError("Expected the original DeepSeek-V4-Flash shard layout")
    prefixes = (
        "embed.",
        "head.",
        "norm.",
        "hc_head_",
        *(f"layers.{i}." for i in range(4)),
    )
    weights = {
        name: shard
        for name, shard in index["weight_map"].items()
        if name.startswith(prefixes)
    }
    shards = sorted(set(weights.values()))
    source_files = {
        name: (source / name).resolve(strict=True)
        for name in (*shards, "tokenizer.json")
    }
    tokenizer = json.loads((source / "tokenizer_config.json").read_text())

    target.mkdir(parents=True, exist_ok=False)
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights}, indent=2) + "\n"
    )
    if not tokenizer.get("chat_template"):
        tokenizer["chat_template"] = CHAT_TEMPLATE
    (target / "tokenizer_config.json").write_text(
        json.dumps(tokenizer, indent=2, ensure_ascii=False) + "\n"
    )
    for name, file in source_files.items():
        (target / name).symlink_to(file)
    for name in ("generation_config.json", "encoding"):
        if (source / name).exists():
            (target / name).symlink_to((source / name).resolve(strict=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original 43-layer checkpoint")
    parser.add_argument("target", type=Path, help="New four-layer checkpoint directory")
    args = parser.parse_args()
    prepare(args.source, args.target)
