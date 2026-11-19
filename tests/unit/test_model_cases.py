import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/plan_model_cases.py"
spec = importlib.util.spec_from_file_location("plan_model_cases", SCRIPT)
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)
CASES = {"qwen3-4b-megatron": {}, "gemma3": {}}


@pytest.mark.parametrize(
    "body,expected",
    [
        (None, ["qwen3-4b-megatron"]),
        ("description\nCI_MODELS:   ", ["qwen3-4b-megatron"]),
        ("CI_MODELS: gemma3", ["gemma3"]),
        (
            "CI_MODELS: gemma3, qwen3-4b-megatron, gemma3",
            ["gemma3", "qwen3-4b-megatron"],
        ),
    ],
)
def test_case_selection(body, expected):
    assert planner.select_cases(body, CASES) == expected


@pytest.mark.parametrize(
    "body",
    [
        "CI_MODELS: missing",
        "CI_MODELS gemma3",
        "CI_MODELS: gemma3,",
        "CI_MODELS: gemma3\nCI_MODELS: gemma3",
    ],
)
def test_invalid_selections_fail(body):
    with pytest.raises(ValueError):
        planner.select_cases(body, CASES)


def test_planner_outputs_runnable_default_from_event(tmp_path):
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"body": ""}}))
    output = tmp_path / "plan.tsv"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        cwd=ROOT,
        env=dict(os.environ, GITHUB_EVENT_PATH=str(event)),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text().strip().split("\t") == [
        "qwen3-4b-megatron",
        "examples/configs/gsm8k_qwen3_megatron.yaml",
        "qwen3/Qwen3-4B",
        "8",
    ]


@pytest.mark.parametrize("first_fails", [False, True])
def test_workflow_runs_all_cases_when_container_reads_stdin(tmp_path, first_fails):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/hcu-pr-model-smoke.yml").read_text()
    )
    steps = workflow["jobs"]["model-smoke"]["steps"]
    script = next(
        step["run"]
        for step in steps
        if step.get("name") == "Run selected model smoke cases"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text("""#!/bin/bash
[[ "$1" == stop ]] && exit 0
cat > /dev/null
for arg in "$@"; do
  case "$arg" in
    hcu-smoke-*-first) case_name=first ;;
    hcu-smoke-*-second) case_name=second ;;
  esac
done
printf "%s\n" "$@" | grep ":/model:ro$" >> model-mounts
[[ "$*" == *"exec python3 .github/scripts/run_model_smoke.py"* ]] || exit 2
echo "$case_name" >> docker-calls
if [[ "$case_name" == first && "$FIRST_FAILS" == 1 ]]; then exit 1; fi
mkdir -p "ci-artifacts/$case_name"
echo ok > "ci-artifacts/$case_name/smoke-success"
""")
    docker.chmod(0o755)
    for model in ("model-a", "model-b"):
        (tmp_path / model).mkdir()
        (tmp_path / model / "config.json").write_text("{}")
    (tmp_path / "ci-model-plan.tsv").write_text(
        "first\tconfig\tmodel-a\t8\nsecond\tconfig\tmodel-b\t8\n"
    )
    env = dict(
        os.environ,
        PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
        AREAL_CI_MODELS_ROOT_V2=str(tmp_path),
        FIRST_FAILS="1" if first_fails else "0",
        GITHUB_RUN_ID="1",
        GITHUB_RUN_ATTEMPT="1",
        GITHUB_WORKSPACE=str(tmp_path),
        AREAL_CI_IMAGE="test",
        AREAL_CI_VENV="/venv",
        AREAL_CI_SGLANG_ROOT="/sglang",
    )
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert (tmp_path / "docker-calls").read_text().splitlines() == ["first", "second"]
    assert (tmp_path / "model-mounts").read_text().splitlines() == [
        f"{tmp_path / model}:/model:ro" for model in ("model-a", "model-b")
    ]
    assert result.returncode == int(first_fails), result.stderr
