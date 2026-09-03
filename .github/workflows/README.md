# HCU CI workflows

These workflows validate the public AReaL-das source on self-hosted runners from the
`ci-general` group. Model weights remain local to eligible HCU runners and are never
downloaded or exposed by a workflow.

## `hcu-source-checks.yml`

Runs public source-level checks on `ci-general` runners: pre-commit, Python compilation,
and shell syntax checks. It is intended to catch formatting and syntax errors early and
does not expose HCU devices to the test container.

## `hcu-image-test.yml`

Builds `docker/Dockerfile.hcu` and runs an HCU runtime smoke check on a self-hosted
runner from the `ci-general` group with labels `self-hosted` and `ci`. It checks that:

- the image imports AReaL-das and SGLang;
- the image uses HCU PyTorch and exposes at least one HCU device;
- the installed SGLang distribution is version 0.5.12 or later;
- HCU GRPO launch scripts pass shell syntax, launcher discovery, and FSDP static audit
  checks.

The workflow uploads build and check logs as an artifact retained for 14 days.

### Required repository configuration

Create the following GitHub Actions repository variable before enabling the workflow:

| Variable         | Required value                                                                                                        |
| ---------------- | --------------------------------------------------------------------------------------------------------------------- |
| `HCU_BASE_IMAGE` | Official HCU SGLang base image pinned by digest, for example `registry.example.com/hcu/sglang@sha256:<64-hex-digits>` |

The runner must have Docker access and the HCU device files `/dev/kfd`, `/dev/dri`, and
`/dev/mkfd`. The base image must already provide the HCU-specific binary stack,
including HCU PyTorch, SGLang, and its kernels. Do not put registry credentials, model
paths, access tokens, or data paths into repository variables or workflow files;
configure registry credentials only on the self-hosted runner.

### `hcu-pr-model-smoke.yml`

Runs a two-step Qwen3-8B GRPO smoke test (FSDP actor + SGLang rollout) on an eligible
8-HCU runner from the `ci-general` group. It uses `pull_request_target` so fork pull
requests can use the target repository's CI variables, then explicitly checks out the
pull request merge commit. It also runs for pushes to `main`.

Configure the image, shared archive, and model directory as repository variables. The
model directory must be readable at the same path on every eligible `ci-general` runner.

| Repository variable      | Purpose                                                                      |
| ------------------------ | ---------------------------------------------------------------------------- |
| `HCU_BASE_IMAGE`         | AReaL HCU runtime image tag                                                  |
| `HCU_BASE_IMAGE_ARCHIVE` | Shared image archive loaded when the selected runner does not have the image |
| `AREAL_CI_MODEL_PATH`    | Shared Qwen3-8B model directory containing `config.json`                     |

The image provides `/opt/areal-venv-py31115`, `/opt/hcu_megatron`, and `/opt/sglang`;
these container paths are set directly by the workflow.

It mounts the checked-out PR source read-write and the model directory read-only, uses
`/dev/kfd`, `/dev/dri`, `/dev/mkfd`, and the host HCU management library, then launches
two GRPO steps with `MAX_NEW_TOKENS=256` and `N_SAMPLES=1`. The job succeeds only when
the training log contains exactly two completed steps and confirms that step 2
completed. Because Qwen3-8B is a dense model, the workflow disables SGLang's optional
AITER MoE backend. It generates a small deterministic math dataset in the job workspace,
so the smoke test does not depend on Hugging Face network access or a node-local dataset
cache. Logs and runtime files are retained for 14 days when GitHub's artifact service is
reachable; an artifact service outage does not replace the training result.

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
