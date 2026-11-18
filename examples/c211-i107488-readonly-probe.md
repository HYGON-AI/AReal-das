# c211-i107488 read-only CI diagnostic

This file exists only to trigger the pull_request_target `examples/**` path filter of
`.github/workflows/hcu-image-test.yml` so the CI log records the HCU runner's own
read-only device inventory (`/opt/hyhal/bin/hy-smi`, `/dev/kfd`, `/dev/dri`, `docker version`).
No build inputs, source files or configuration are modified. This branch/PR will be
closed immediately after the diagnostic run.
