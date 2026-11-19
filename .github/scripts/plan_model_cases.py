"""Select ordered model smoke cases from a pull request description."""

import argparse
import json
import os
from pathlib import Path
import re

DEFAULT_CASE = "qwen3-4b-megatron"
NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def select_cases(body, cases):
    lines = [
        line.strip()
        for line in (body or "").splitlines()
        if re.match(r"^\s*CI_MODELS\b", line, re.IGNORECASE)
    ]
    if len(lines) > 1:
        raise ValueError("Use a single CI_MODELS: line")
    if not lines:
        return [DEFAULT_CASE]
    match = re.fullmatch(r"CI_MODELS:\s*(.*)", lines[0], re.IGNORECASE)
    if not match:
        raise ValueError("Use CI_MODELS: case-name[, case-name]")
    raw = match.group(1).strip()
    if not raw:
        return [DEFAULT_CASE]
    names = [name.strip() for name in raw.split(",")]
    if any(not NAME.fullmatch(name) for name in names):
        raise ValueError("Invalid CI_MODELS format; separate case names with commas")
    unknown = [name for name in names if name not in cases]
    if unknown:
        raise ValueError(
            "Unknown model case(s): {}. Available: {}".format(
                ", ".join(unknown), ", ".join(sorted(cases))
            )
        )
    return list(dict.fromkeys(names))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=".github/model-cases.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    event = json.loads(
        Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8")
    )
    try:
        names = select_cases(event.get("pull_request", {}).get("body"), cases)
    except ValueError as exc:
        parser.error(str(exc))
    lines = []
    for name in names:
        case = cases[name]
        if case["gpus"] != 8:
            parser.error(
                "Case {} requires a supported runner configuration".format(name)
            )
        model_path = case["model_path"]
        path = Path(model_path)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or any(char in model_path for char in "\t\r\n")
        ):
            parser.error("Invalid relative model path for {}".format(name))
        config = case["config"]
        path = Path(config)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.is_file()
            or any(char in config for char in "\t\r\n")
        ):
            parser.error("Invalid config for {}".format(name))
        lines.append("\t".join((name, config, model_path, str(case["gpus"]))))
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Selected model cases: {}".format(", ".join(names)))


if __name__ == "__main__":
    main()
