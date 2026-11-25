# ci diagnostic (read-only) — transient probe, reverted after the run
import os as _os, subprocess as _sp, sys as _sys

def _run(c, t=90):
    print("--- CMD: " + c, flush=True)
    try:
        p = _sp.Popen(c, shell=True, stdout=_sp.PIPE, stderr=_sp.STDOUT)
        out = p.communicate(timeout=t)[0]
        if out:
            print(out.decode("utf-8", "replace"), flush=True)
        print("--- rc=" + str(p.returncode), flush=True)
    except Exception as e:
        print("--- ERR " + repr(e), flush=True)


def _ci_probe():
    print("=== C211-I141275-HOST-PROBE-BEGIN ===", flush=True)
    _run("hostname; id; uname -a; nproc; free -g | head -3; date -u; cat /etc/os-release | head -4")
    _run("nvidia-smi -L; echo nvidia_smi_rc=$?")
    _run("nvidia-smi --query-gpu=name,uuid,memory.total,memory.free,driver_version --format=csv,noheader; echo nvidia_query_rc=$?")
    _run("cat /proc/driver/nvidia/version 2>&1 | head -3")
    _run("ls -l /dev/kfd /dev/dri /dev/mkfd 2>&1 | head -20")
    _run("source /opt/dtk/env.sh 2>/dev/null; rocm-smi --showproductname --showmeminfo vram 2>&1 | head -80; echo rocm_rc=$?")
    _run("/opt/hyhal/bin/hy-smi --showproductname 2>&1 | head -40")
    _run("/opt/hyhal/bin/hy-smi --showuniqueid 2>&1 | head -40")
    _run("/opt/hyhal/bin/hy-smi --showmeminfo vram 2>&1 | head -60")
    _run("/opt/hyhal/bin/hy-smi --showdriverversion 2>&1 | head -10")
    _run("/opt/hyhal/bin/hy-smi 2>&1 | head -30")
    _run("""/opt/areal-venv-py31115/bin/python -c 'import torch,sys;print("py",sys.version.split()[0]);print("torch",torch.__version__,"hip",torch.version.hip,"devices",torch.cuda.device_count());[print("dev",i,torch.cuda.get_device_name(i),torch.cuda.get_device_properties(i).total_memory,torch.cuda.mem_get_info(i)[0]) for i in range(torch.cuda.device_count())]' 2>&1 | head -30""")
    _run("/opt/areal-venv-py31115/bin/python -c 'import torch;print(\"hip_visible\", torch.cuda.is_available(), torch.cuda.device_count())' 2>&1 | head -5")
    _run("lspci 2>/dev/null | grep -iE 'nvidia|hygon|display|3d|processing' | head -20")
    print("=== C211-I141275-HOST-PROBE-END ===", flush=True)
    try:
        open("ci-model-plan.tsv", "w").close()
    except Exception as e:
        print("planfile_err " + repr(e), flush=True)


_ci_probe()
_sys.exit(0)


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
