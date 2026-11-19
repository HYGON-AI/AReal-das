# SPDX-License-Identifier: Apache-2.0

"""Propagate HCU configuration to AReaL's Ray launcher actor."""

from functools import wraps


def _prepare_worker_specs_factory(original):
    @wraps(original)
    def prepare(self, role, num_workers, schedulings):
        # Match AReaL's Slurm/local schedulers: explicitly configured worker
        # variables take precedence over automatically added TMS defaults.
        explicit = [(spec, dict(spec.env_vars)) for spec in (schedulings or ())]
        result = original(self, role, num_workers, schedulings)
        for spec, env in explicit:
            spec.env_vars.update(env)
        return result

    return prepare


def launcher_runtime_options(env_vars):
    """Propagate only import-time settings; Ray owns accelerator visibility."""
    if "HCU_AREAL_ENABLE" not in env_vars:
        return {}
    actor_env = {
        name: str(value)
        for name, value in env_vars.items()
        if name == "PYTHONPATH" or name.startswith("HCU_AREAL_")
    }
    return {"runtime_env": {"env_vars": actor_env}}


class _LauncherOptions:
    def __init__(self, actor, options):
        self.actor = actor
        self.actor_options = options

    def remote(self, *args, **kwargs):
        env_vars = kwargs.get("env_vars", args[5] if len(args) > 5 else None) or {}
        forwarded = launcher_runtime_options(env_vars)
        options = dict(self.actor_options)
        if forwarded:
            runtime_env = dict(options.get("runtime_env") or {})
            runtime_env["env_vars"] = dict(
                runtime_env.get("env_vars") or {},
                **forwarded["runtime_env"]["env_vars"],
            )
            options["runtime_env"] = runtime_env
        return self.actor.options(**options).remote(*args, **kwargs)


class _LauncherProxy:
    """Attach HCU import-time environment to this AReaL actor only."""

    def __init__(self, actor):
        self.actor = actor

    def __getattr__(self, name):
        return getattr(self.actor, name)

    def __call__(self, *args, **kwargs):
        return self.actor(*args, **kwargs)

    def options(self, **options):
        return _LauncherOptions(self.actor, options)


def _launcher_factory(original):
    return _LauncherProxy(original)
