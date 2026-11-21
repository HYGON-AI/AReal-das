# HCU CI workflows

These workflows validate the public AReaL-das source on self-hosted runners from the
`ci-general` group. Model weights remain local to eligible HCU runners and are never
downloaded or exposed by a workflow.

HCU patch tests live under `tests/unit/`; hardware training checks use the model smoke
workflow. CI entrypoint scripts live under `.github/scripts/`. Upstream tests belong to
the AReaL submodule.

Both the CPU unit workflow and the model smoke workflow run for every opened,
synchronized, reopened or ready-for-review pull request, including draft pull requests.
The repository ruleset file lists the source, CPU unit and model smoke checks as
required. Apply the ruleset in GitHub to enforce it on the default branch. The model
smoke workflow also reruns when the PR description is edited. Add a line such as
`CI_MODELS: qwen3-4b-megatron` to select cases; comma-separated cases run sequentially
without adding the default. If the line is absent or empty, `qwen3-4b-megatron` runs.
Unknown or malformed names fail the check. The CPU unit workflow always runs
independently.

`.github/model-cases.yaml` is the case registry (JSON syntax, valid YAML). Each entry
names an example configuration, a model path relative to the shared model root, and its
GPU count. The shared Python runner supports single-node, eight-HCU GSM8K examples. Each
new case references the example YAML and a model directory under the shared root.
The registry uses full checkpoints for the following cases.

| Case                      | Checkpoint      |
| ------------------------- | --------------- |
| `qwen3-4b-megatron`       | Qwen3-4B        |
| `qwen35-35b-a3b-megatron` | Qwen3.5-35B-A3B |
| `qwen36-35b-a3b-megatron` | Qwen3.6-35B-A3B |
| `qwen38-27b-megatron`     | Qwen3.8-27B     |

## `hcu-source-checks.yml`

Runs public source-level checks on `ci-general` runners: Python compilation and shell
syntax checks. It runs `pre-commit run --all-files` using the root hook configuration
and checks `requirements.txt` for internal HCU build pins. It is intended to catch
syntax errors early and does not expose HCU devices to the test container.

## `hcu-image-test.yml`

On pull requests this job is path-filtered to submodules, HCU code, examples,
requirements.txt, the image check script and this workflow. All three Docker invocations
explicitly use /bin/bash as their entrypoint.

Runs an HCU runtime smoke check with the existing `HCU_BASE_IMAGE_V2` on a self-hosted
runner from the `ci-general` group with labels `self-hosted` and `ci`. The checked-out
repository is mounted read-only; this workflow does not build an image. It checks that:

- the image imports AReaL-das and SGLang;
- the image uses HCU PyTorch and exposes at least one HCU device;
- the installed SGLang distribution is version 0.5.12 or later;
- the checked-out Python source compiles, the HCU test scripts pass shell syntax checks,
  and the HCU Megatron adapter imports.

The image must already provide a compatible Python environment and HCU runtime.

### Required repository configuration

Create the following GitHub Actions repository variable before enabling the workflow:

| Variable            | Required value                                                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------------------ |
| `HCU_BASE_IMAGE_V2` | HCU image reference available on the runner. For a Docker save archive, use the tag restored by docker load. |
| `HCU_CI_PYTHON`     | Optional Python executable for source checks and CPU tests; defaults to `python` in the image.               |

The runner must have Docker access and the HCU device files `/dev/kfd`, `/dev/dri`, and
`/dev/mkfd`. The base image must already provide the HCU-specific binary stack,
including HCU PyTorch, SGLang, and its kernels. Set the shared model root in the
`AREAL_CI_MODELS_ROOT_V2` repository variable. Keep registry credentials and access
tokens out of repository variables and workflow files.

### `hcu-pr-model-smoke.yml`

Runs the selected model smoke cases on an eligible 8-HCU runner from the `ci-general`
group. The default case runs two-step Qwen3-4B GRPO (Megatron actor + SGLang rollout).
It uses `pull_request_target` so fork pull requests can use the target repository's CI
variables, then explicitly checks out the pull request merge commit. HCU workflows run
automatically only during the pull request stage; merging to `main` does not run them
again. Maintainers can still start them manually with `workflow_dispatch`.

Configure the image, shared archive, and model directory as repository variables. The
`_V2` variables are separate from the variables used by the legacy workflows. Each case
records a relative `model_path`. The CI resolves it under the shared root and mounts the
model directory read-only at `/model`. The root and model directories must be readable
on every eligible `ci-general` runner.

| Repository variable         | Purpose                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------ |
| `HCU_BASE_IMAGE_V2`         | AReaL HCU runtime image tag                                                          |
| `HCU_BASE_IMAGE_ARCHIVE_V2` | Shared image archive loaded when the selected runner does not have the image         |
| `AREAL_CI_MODELS_ROOT_V2`   | Shared root containing model directories, with `qwen3/Qwen3-4B` for the default case |

The shared Python smoke runner (`.github/scripts/run_model_smoke.py`) uses
`/opt/areal-venv-py31115` when present and otherwise uses the Python environment
provided by the image. SGLang is expected in the image.

It mounts the checked-out PR source read-write and the model directory read-only, uses
`/dev/kfd`, `/dev/dri`, `/dev/mkfd`, and the host HCU management library, then launches
`python3 -m hcu_areal.launch` with a generated config for two GRPO steps, 256 maximum
new tokens and two samples per prompt. The job succeeds only when the training log
contains exactly two completed steps, confirms step 2, and records rollout acceptance,
successful versioned weight updates, plus weight-update timing and gradient norm for
each step. A zero gradient still passes this smoke check and does not demonstrate a
parameter update. The generated test configuration defaults to two CPUs for each actor
worker without an explicit CPU request and sets the CPU thread count to two, allowing
the training, rollout and evaluation roles to be scheduled on a 32-CPU runner. Because
Qwen3-4B is a dense model, the workflow disables SGLang's optional AITER MoE backend. It
generates a small deterministic math dataset in the job workspace, so the smoke test
does not depend on Hugging Face network access or a node-local dataset cache. Logs and
runtime files are retained for 14 days when GitHub's artifact service is reachable; an
artifact service outage does not replace the training result. Service shutdown timeouts
are surfaced as warnings; inspect the retained logs and runner state when they occur.
The runner checks device utilization, memory and device processes before launch; this
check does not reserve the devices against other jobs. Use a dedicated or externally
scheduled runner to avoid competing allocations.

Before checkout, every HCU workflow uses the CI image to restore runner ownership of the
reused workspace. Python bytecode writes are disabled inside the root model container so
subsequent checkouts can clean the workspace without permission errors.

### Scope of the current smoke checks

The model smoke workflow uses the `ci-general` runner group. Every matching runner must
provide eight HCUs and access to the shared image archive and model directory. Because
`pull_request_target` checks out and executes the pull request merge commit, repository
administrators must treat this workflow as trusted-runner execution of contributor code.
The HCU workflows explicitly opt in to this checkout behavior required by
`actions/checkout@v6`. They are not public online-service tests and do not publish a
container image.

## Organization quality gate

Pull requests targeting `main` also trigger `quality-gate.yml` on opening,
synchronization, reopening and marking ready for review. Its `Checks` job calls the
shared HYGON-AI PR quality workflow at v2.0.3 independently of the HCU jobs.
