# HCU CI workflows

These workflows validate the public AReaL-das source on self-hosted runners from the `ci-general` group. Model weights remain local to eligible HCU runners and are never downloaded or exposed by a workflow.

## `hcu-source-checks.yml`

Runs public source-level checks on `ci-general` runners: pre-commit, Python compilation, and shell syntax checks. It is intended to catch formatting and syntax errors early and does not expose HCU devices to the test container.

## `hcu-image-test.yml`

Builds `docker/Dockerfile.hcu` and runs an HCU runtime smoke check on a self-hosted runner from the `ci-general` group with labels `self-hosted` and `ci`. It checks that:

- the image imports AReaL-das and SGLang;
- the image uses HCU PyTorch and exposes at least one HCU device;
- the installed SGLang distribution is version 0.5.12 or later;
- HCU GRPO launch scripts pass shell syntax, launcher discovery, and FSDP static audit checks.

The workflow uploads build and check logs as an artifact retained for 14 days.

### Required repository configuration

Create the following GitHub Actions repository variable before enabling the workflow:

| Variable | Required value |
| --- | --- |
| `HCU_BASE_IMAGE` | Official HCU SGLang base image pinned by digest, for example `registry.example.com/hcu/sglang@sha256:<64-hex-digits>` |

The runner must have Docker access and the HCU device files `/dev/kfd`, `/dev/dri`, and `/dev/mkfd`. The base image must already provide the HCU-specific binary stack, including HCU PyTorch, SGLang, and its kernels. Do not put registry credentials, model paths, access tokens, or data paths into repository variables or workflow files; configure registry credentials only on the self-hosted runner.

### `hcu-pr-model-smoke.yml`

Runs a two-step Qwen3-8B GRPO smoke test (FSDP actor + SGLang rollout) on an eligible 8-HCU runner from the `ci-general` group. It runs for non-draft pull requests from branches in this repository and for pushes to `main`. Pull requests from forks are deliberately skipped: a public fork must not run arbitrary code in the private HCU, image, and model environment.

Every `ci-general` runner eligible for this job must define these local-only environment variables; never add their values to repository variables, workflow files, or logs.

| Runner environment variable | Purpose |
| --- | --- |
| `AREAL_CI_IMAGE` | Locally available AReaL HCU runtime image tag |
| `AREAL_CI_MODEL_PATH` | Local Qwen3-8B model directory containing `config.json` |
| `AREAL_CI_VENV` | Python virtual environment inside the image |
| `AREAL_CI_MEGATRON_HOME` | Megatron checkout inside the image |
| `AREAL_CI_SGLANG_ROOT` | SGLang checkout inside the image |

It mounts the checked-out PR source read-write and the model directory read-only, uses `/dev/kfd` and `/dev/dri`, then launches two GRPO steps with `MAX_NEW_TOKENS=256` and `N_SAMPLES=1`. Logs and runtime files are retained for 14 days.

### Scope of the current smoke checks

The model smoke workflow uses the `ci-general` runner group. Every matching runner must provide eight HCUs, the required local image and model, and all `AREAL_CI_*` environment variables. Do not remove the same-repository PR guard. It is not a public online-service test and does not publish a container image.
