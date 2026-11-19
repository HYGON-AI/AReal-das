# SPDX-License-Identifier: Apache-2.0

"""Launch an official AReaL example from a single HCU runtime configuration."""

import argparse
import copy
from contextlib import redirect_stdout
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import select
import signal
import shutil
import subprocess
import sys
import tempfile
import time

import yaml


ROOT = Path(__file__).resolve().parent.parent
_UPSTREAM_INIT_SHA256 = (
    "9077835b57ada25c10eadb447bd2fa2efad47e282d466bafea678af00ba1f56e"
)


@lru_cache(maxsize=16)
def _package_fingerprint(root):
    """Identify the checked-out AReaL source and the copied HCU entry."""
    source = Path(root) / "3rdparty" / "AReaL" / "areal"
    template = Path(__file__).with_name("patch_init.py")
    init = source / "__init__.py"
    if hashlib.sha256(init.read_bytes()).hexdigest() != _UPSTREAM_INIT_SHA256:
        raise RuntimeError(f"AReaL package initializer has unexpected content: {init}")
    # Links in the cached entry point refer to this particular checkout.
    digest = hashlib.sha256(b"hcu-areal-package-link-v2\0")
    digest.update(str(Path(root).resolve()).encode("utf-8") + b"\0")
    digest.update(template.read_bytes())
    for path in sorted(source.rglob("*")):
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        ):
            digest.update(str(path.relative_to(source)).encode("utf-8") + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def areal_package_root(cache_dir=None, *, root=None, env=None):
    """Return a cache location shared by the driver and its worker processes."""
    root = Path(ROOT if root is None else root)
    env = os.environ if env is None else env
    if env.get("HCU_AREAL_PACKAGE_ROOT"):
        return Path(env["HCU_AREAL_PACKAGE_ROOT"])
    cache = (
        cache_dir
        or env.get("AREAL_CACHE_DIR")
        or Path(tempfile.gettempdir()) / f"hcu-areal-{os.getuid()}"
    )
    return Path(cache) / "hcu-areal" / _package_fingerprint(str(root))


def install_package_init(root=None, *, cache_dir=None, env=None):
    """Prepare an entry overlay without changing the upstream submodule."""
    root = Path(ROOT if root is None else root)
    source = root / "3rdparty" / "AReaL" / "areal"
    fingerprint = _package_fingerprint(str(root))
    package_root = areal_package_root(cache_dir, root=root, env=env)
    template = Path(__file__).with_name("patch_init.py")
    replacement = template.read_bytes()
    marker = package_root / ".hcu-areal-source"
    target = package_root / "areal" / "__init__.py"
    source_link = package_root / "areal" / "_upstream"

    def verify():
        if not marker.is_file() or marker.read_text().strip() != fingerprint:
            raise RuntimeError(
                f"AReaL package cache has unexpected source: {package_root}"
            )
        if (
            target.is_symlink()
            or not target.is_file()
            or target.read_bytes() != replacement
        ):
            raise RuntimeError(
                f"AReaL package cache has unexpected initializer: {target}"
            )
        if not source_link.is_symlink() or source_link.resolve() != source.resolve():
            raise RuntimeError(
                f"AReaL package cache has unexpected source link: {source_link}"
            )

    if package_root.exists():
        verify()
        return package_root
    package_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".hcu-areal-", dir=package_root.parent))
    try:
        (temporary / "areal").mkdir()
        # The temporary and final directories have the same depth, so the
        # relative link remains valid after the atomic rename.
        relative = os.path.relpath(source, package_root / "areal")
        (temporary / "areal" / "_upstream").symlink_to(
            relative, target_is_directory=True
        )
        (temporary / "areal" / "__init__.py").write_bytes(replacement)
        (temporary / ".hcu-areal-source").write_text(fingerprint + "\n")
        try:
            os.rename(temporary, package_root)
        except OSError:
            if not package_root.is_dir():
                raise
        verify()
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return package_root


def merge(base, overlay):
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path):
    with open(str(path), encoding="utf-8") as stream:
        result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise ValueError("YAML must contain a mapping: {}".format(path))
    return result


def set_override(config, expression):
    if "=" not in expression:
        raise ValueError("Overrides must use KEY=VALUE: {}".format(expression))
    key, raw = expression.split("=", 1)
    parts = key.split(".")
    if (
        not parts
        or parts[0] not in ("runtime", "training")
        or any(not p for p in parts)
    ):
        raise ValueError(
            "Override must begin with runtime. or training.: {}".format(key)
        )
    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise ValueError("Unknown override section: {}".format(key))
        node = node[part]
    if parts[-1] not in node:
        raise ValueError("Unknown override field: {}".format(key))
    node[parts[-1]] = yaml.safe_load(raw)


def cache_environment(cache):
    if not cache:
        return {}
    cache = str(cache)
    return {
        "AREAL_CACHE_DIR": cache,
        "HF_HOME": os.path.join(cache, "huggingface"),
        "HF_DATASETS_CACHE": os.path.join(cache, "huggingface", "datasets"),
        "XDG_CACHE_HOME": os.path.join(cache, "xdg"),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(cache, "torchinductor"),
        "TORCH_EXTENSIONS_DIR": os.path.join(cache, "torch_extensions"),
    }


def temporary_environment(path):
    if not path:
        return {}
    return {"TMPDIR": str(path), "TEMP": str(path), "TMP": str(path)}


_IMAGE_MEGATRON_ROOTS = (
    "/opt/hcu-stack/Megatron-LM-das",
    "/opt/hcu-stack/Megatron-bridge-das",
    "/opt/megatron",
)


def _is_image_megatron_path(path):
    return isinstance(path, str) and any(
        path == old or path.startswith(old + os.sep) for old in _IMAGE_MEGATRON_ROOTS
    )


def _pythonpath(inherited, package_root=None):
    package_root = areal_package_root() if package_root is None else Path(package_root)
    megatron = ROOT / "3rdparty" / "Megatron-LM-das"
    bridge = ROOT / "3rdparty" / "Megatron-bridge-das"
    paths = [
        str(ROOT),
        str(package_root),
        str(ROOT / "3rdparty" / "AReaL"),
        str(megatron),
        str(megatron / "3rdparty" / "Megatron-LM"),
        str(bridge),
        str(bridge / "hcu_megatron_bridge" / "src"),
        str(bridge / "3rdparty" / "Megatron-Bridge" / "src"),
    ]
    # The candidate image contains older copies of these same packages. Keep
    # SGLang and other inherited paths while avoiding mixed Megatron sources.
    for path in str(inherited or "").split(os.pathsep):
        if path and not _is_image_megatron_path(path):
            paths.append(path)
    return os.pathsep.join(dict.fromkeys(paths))


def _activate_package():
    """Use the staged package and pinned dependencies in this process and children."""
    package_root = install_package_init()
    os.environ["HCU_AREAL_PACKAGE_ROOT"] = str(package_root)
    pythonpath = _pythonpath(os.environ.get("PYTHONPATH"), package_root)
    os.environ["PYTHONPATH"] = pythonpath
    paths = pythonpath.split(os.pathsep)
    sys.path[:] = paths + [
        path
        for path in sys.path
        if path not in paths and not _is_image_megatron_path(path)
    ]


def prepare_process_environment():
    """Apply the HCU device policy before this process imports its role."""
    from hcu_areal.infra.platforms import prepare_process_environment as prepare

    prepare()


def run_explicit_module(module, args):
    """Start one known AReaL service after applying its role-specific patches."""
    from hcu_areal.infra.utils.proc import _HCU_MODULES

    if module not in _HCU_MODULES:
        raise ValueError(f"Unsupported HCU service module: {module}")
    prepare_process_environment()
    if os.environ.get("HCU_AREAL_ENABLE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        _activate_package()
    if module == "areal.v2.inference_service.sglang.launch_server":
        from hcu_areal.v2.inference_service.sglang.launch_server import (
            prepare_triton_kv_layout,
        )

        prepare_triton_kv_layout(args)
    from hcu_areal import areal_adaptor as adaptor

    adaptor.apply_process_role(module)
    sys.argv = [module, *args]
    if module == "areal.v2.inference_service.sglang.launch_server":
        from hcu_areal.v2.inference_service.sglang.launch_server import main

        return main()
    runpy.run_module(module, run_name="__main__", alter_sys=True)


def run_explicit_driver(entrypoint, args):
    prepare_process_environment()
    if os.environ.get("HCU_AREAL_ENABLE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        _activate_package()
    from hcu_areal.areal_adaptor import apply_driver_patches

    apply_driver_patches()
    sys.argv = [entrypoint, *args]
    sys.path.insert(0, str(Path(entrypoint).resolve().parent))
    runpy.run_path(entrypoint, run_name="__main__")


def prepare(config_path, overrides=()):
    config = load_yaml(config_path)
    if set(config) != {"runtime", "training"}:
        raise ValueError(
            "The config must contain exactly runtime and training sections"
        )
    runtime, training = config["runtime"], config["training"]
    if not isinstance(runtime, dict) or not isinstance(training, dict):
        raise ValueError("runtime and training must be mappings")
    base_path = ROOT / runtime.get("base_config", "")
    if not base_path.is_file():
        raise ValueError("Official base config not found: {}".format(base_path))
    config["training"] = merge(load_yaml(base_path), training)
    for override in overrides:
        set_override(config, override)
    runtime = config["runtime"]
    entry = ROOT / runtime.get("entrypoint", "")
    if not entry.is_file():
        raise ValueError("Official training entrypoint not found: {}".format(entry))
    ray = runtime.get("ray", {})
    mode = ray.get("mode", "existing")
    if mode not in ("existing", "auto", "local"):
        raise ValueError("runtime.ray.mode must be existing, auto or local")
    expected_scheduler = "local" if mode == "local" else "ray"
    if config["training"].get("scheduler", {}).get("type") != expected_scheduler:
        raise ValueError(
            "scheduler.type must be {} for ray.mode={}".format(expected_scheduler, mode)
        )
    if mode == "auto" and config["training"].get("cluster", {}).get("n_nodes") != 1:
        raise ValueError("runtime.ray.mode=auto requires training.cluster.n_nodes=1")
    backend = config["training"].get("actor", {}).get("backend", "")
    match = re.search(r"^fsdp:.*d(\d+)", str(backend))
    batch = config["training"].get("train_dataset", {}).get("batch_size")
    if match and isinstance(batch, int):
        dp = int(match.group(1))
        if batch < dp or batch % dp:
            raise ValueError(
                "train_dataset.batch_size must be a positive multiple of FSDP DP size {}".format(
                    dp
                )
            )
    # Explicitly forward the model environment to each scheduled worker. A Ray
    # daemon may have been started from a different shell. Interpolated rollout
    # specs inherit the actor spec; independently configured specs need both.
    model_env = {str(k): str(v) for k, v in runtime.get("env", {}).items()}
    model_env["HCU_AREAL_ENABLE"] = "1"
    model_env.update(cache_environment(runtime.get("cache_dir")))
    model_env.update(temporary_environment(runtime.get("temp_dir")))
    unset_env = runtime.get("unset_env", [])
    if not isinstance(unset_env, list) or any(
        not isinstance(k, str) or not k.isidentifier() for k in unset_env
    ):
        raise ValueError(
            "runtime.unset_env must be a list of environment variable names"
        )
    if set(unset_env).intersection(model_env):
        raise ValueError(
            "runtime.env and runtime.unset_env cannot contain the same name"
        )
    package_env = {
        key: value
        for key, value in os.environ.items()
        if key not in unset_env and key != "HCU_AREAL_PACKAGE_ROOT"
    }
    package_env.update(model_env)
    model_env["HCU_AREAL_PACKAGE_ROOT"] = str(
        areal_package_root(runtime.get("cache_dir"), env=package_env)
    )
    # A local worker inherits the driver environment. Inserting those values
    # into SchedulingSpec also exposes them to AReaL's comma-separated parser,
    # which cannot represent values containing commas or equals signs.
    if mode == "local":
        return config
    inherited_path = model_env.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    if "PYTHONPATH" in unset_env:
        inherited_path = ""
    model_env["PYTHONPATH"] = _pythonpath(
        inherited_path, model_env["HCU_AREAL_PACKAGE_ROOT"]
    )
    # PYTHONPATH was rebuilt above; removing it again during package startup
    # would prevent child processes from finding the HCU package.
    worker_unset_env = [name for name in unset_env if name != "PYTHONPATH"]
    if worker_unset_env:
        model_env["HCU_AREAL_UNSET_ENV"] = ",".join(worker_unset_env)
    for role in ("actor", "rollout"):
        section = config["training"].get(role, {})
        specs = section.get("scheduling_spec") if isinstance(section, dict) else None
        if isinstance(specs, list):
            for spec in specs:
                if isinstance(spec, dict):
                    existing = spec.setdefault("env_vars", {})
                    if not isinstance(existing, dict):
                        raise ValueError(
                            "{}.scheduling_spec env_vars must be a mapping".format(role)
                        )
                    if (
                        existing.get(
                            "HCU_AREAL_PACKAGE_ROOT",
                            model_env["HCU_AREAL_PACKAGE_ROOT"],
                        )
                        != model_env["HCU_AREAL_PACKAGE_ROOT"]
                    ):
                        raise ValueError(
                            "{}.scheduling_spec cannot override HCU_AREAL_PACKAGE_ROOT".format(
                                role
                            )
                        )
                    spec["env_vars"] = dict(model_env, **existing)
                    if "PYTHONPATH" in existing:
                        spec["env_vars"]["PYTHONPATH"] = _pythonpath(
                            existing["PYTHONPATH"], model_env["HCU_AREAL_PACKAGE_ROOT"]
                        )
    return config


def runtime_environment(runtime, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    for name in runtime.get("unset_env", []):
        env.pop(name, None)
    for name, value in runtime.get("env", {}).items():
        env[name] = str(value)
    if "HCU_AREAL_PACKAGE_ROOT" not in runtime.get("env", {}):
        env.pop("HCU_AREAL_PACKAGE_ROOT", None)
    worker_unset_env = [
        name for name in runtime.get("unset_env", []) if name != "PYTHONPATH"
    ]
    if worker_unset_env:
        env["HCU_AREAL_UNSET_ENV"] = ",".join(worker_unset_env)
    else:
        env.pop("HCU_AREAL_UNSET_ENV", None)
    env.update(cache_environment(runtime.get("cache_dir")))
    env.update(temporary_environment(runtime.get("temp_dir")))
    env["HCU_AREAL_ENABLE"] = "1"
    env["HCU_AREAL_PACKAGE_ROOT"] = str(
        areal_package_root(runtime.get("cache_dir"), env=env)
    )
    env["PYTHONPATH"] = _pythonpath(
        env.get("PYTHONPATH"), env["HCU_AREAL_PACKAGE_ROOT"]
    )
    ray = runtime.get("ray", {})
    if ray.get("mode", "existing") == "existing":
        if not ray.get("address"):
            raise ValueError(
                "runtime.ray.address is required for an existing Ray cluster"
            )
        env["RAY_ADDRESS"] = str(ray["address"])
    else:
        env.pop("RAY_ADDRESS", None)
    return env


def ray_command(action, ray):
    if not ray.get("address"):
        raise ValueError("runtime.ray.address is required")
    command = ["ray", "start"]
    if action == "ray-head":
        port = str(ray["address"]).rsplit(":", 1)[-1]
        command.extend(["--head", "--port=" + port])
    else:
        command.append("--address=" + str(ray["address"]))
    if ray.get("node_ip"):
        command.append("--node-ip-address=" + str(ray["node_ip"]))
    if ray.get("temp_dir"):
        command.append("--temp-dir=" + str(ray["temp_dir"]))
    for key, option in (("num_gpus", "--num-gpus"), ("num_cpus", "--num-cpus")):
        if ray.get(key) is not None:
            command.append(option + "=" + str(ray[key]))
    return command


def run_ray_owner(options):
    """Own a new Ray cluster until the launcher closes our stdin pipe."""
    import ray

    kwargs = {"address": "local", "include_dashboard": False}
    for key in ("num_cpus", "num_gpus"):
        if options.get(key) is not None:
            kwargs[key] = options[key]
    if options.get("temp_dir"):
        kwargs["_temp_dir"] = str(options["temp_dir"])
    try:
        with redirect_stdout(sys.stderr):
            context = ray.init(**kwargs)
        address = context.address_info["address"]
        print("HCU_RAY_ADDRESS=" + json.dumps(address), flush=True)
        sys.stdin.read()
    finally:
        ray.shutdown()


def start_owned_ray(ray_config, env):
    owner_env = dict(env)
    owner_env.pop("RAY_ADDRESS", None)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hcu_areal.launch",
            "--ray-owner",
            json.dumps(ray_config),
        ],
        cwd=str(ROOT),
        env=owner_env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1,
    )
    try:
        deadline = time.monotonic() + 180
        output = b""
        prefix = b"HCU_RAY_ADDRESS="
        address = None
        while address is None:
            ready, _, _ = select.select(
                [process.stdout], [], [], max(0, deadline - time.monotonic())
            )
            if not ready:
                raise ValueError("Timed out starting the private Ray cluster")
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                raise ValueError(
                    "Private Ray cluster failed to start; inspect the Ray output above"
                )
            output += chunk
            while b"\n" in output:
                line, output = output.split(b"\n", 1)
                if line.startswith(prefix):
                    try:
                        address = json.loads(line[len(prefix) :].decode("utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        raise ValueError(
                            "Private Ray cluster returned an invalid address"
                        )
                    break
                print(line.decode("utf-8", "replace"), file=sys.stderr)
            if len(output) > 65536:
                raise ValueError(
                    "Private Ray cluster produced an invalid startup response"
                )
        if not isinstance(address, str) or not address:
            raise ValueError("Private Ray cluster returned an invalid address")
        return process, address
    except BaseException:
        stop_owned_ray(process)
        raise


def stop_owned_ray(process):
    if process.stdin:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    finally:
        if process.stdout:
            process.stdout.close()


def run_training(config):
    runtime = config["runtime"]
    env = runtime_environment(runtime)
    if runtime.get("ray", {}).get("mode", "existing") == "existing":
        # AReaL can fall back to starting a local Ray instance. Explicitly
        # check the requested cluster first so a typo cannot do that silently.
        address = str(runtime["ray"]["address"])
        try:
            result = subprocess.run(
                ["ray", "status", "--address=" + address],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            raise ValueError("Timed out checking Ray cluster {}".format(address))
        if result.returncode:
            raise ValueError(
                "Ray cluster {} is unavailable: {}".format(
                    address, result.stdout[-1500:]
                )
            )
    install_package_init(env=env)
    logfile = Path(runtime["driver_log"])
    logfile.parent.mkdir(parents=True, exist_ok=True)
    if runtime.get("cache_dir"):
        Path(runtime["cache_dir"]).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="hcu-training-", encoding="utf-8", delete=False
    ) as stream:
        config_file = stream.name
        yaml.safe_dump(
            config["training"], stream, default_flow_style=False, allow_unicode=True
        )
    command = [
        sys.executable,
        "-m",
        "hcu_areal.launch",
        "--driver",
        str(ROOT / runtime["entrypoint"]),
        "--config",
        config_file,
    ]
    owner = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def terminate(_signum, _frame):
        raise SystemExit(128 + signal.SIGTERM)

    signal.signal(signal.SIGTERM, terminate)
    try:
        if runtime.get("ray", {}).get("mode", "existing") == "auto":
            owner, address = start_owned_ray(runtime["ray"], env)
            env["RAY_ADDRESS"] = address
            print("Started private Ray cluster at {}".format(address), flush=True)
        with logfile.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            try:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
                return process.wait()
            except KeyboardInterrupt:
                process.send_signal(signal.SIGINT)
                return process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            finally:
                process.stdout.close()
    finally:
        try:
            if owner is not None:
                stop_owned_ray(owner)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            os.unlink(config_file)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--install-init"]:
        print(install_package_init())
        return 0
    if argv == ["--pythonpath"]:
        print(_pythonpath(os.environ.get("PYTHONPATH"), install_package_init()))
        return 0
    if len(argv) == 2 and argv[0] == "--ray-owner":
        run_ray_owner(json.loads(argv[1]))
        return 0
    if len(argv) >= 2 and argv[0] == "--module":
        run_explicit_module(argv[1], argv[2:])
        return 0
    if len(argv) >= 2 and argv[0] == "--driver":
        run_explicit_driver(argv[1], argv[2:])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set", action="append", default=[], metavar="SECTION.KEY=VALUE"
    )
    parser.add_argument(
        "--action",
        choices=("train", "render", "ray-head", "ray-worker"),
        default="train",
    )
    args = parser.parse_args(argv)
    try:
        config = prepare(Path(args.config), args.set)
        if args.action == "render":
            yaml.safe_dump(
                config, sys.stdout, default_flow_style=False, allow_unicode=True
            )
            return 0
        if args.action in ("ray-head", "ray-worker"):
            if config["runtime"]["ray"].get("mode", "existing") != "existing":
                raise ValueError("Ray start actions require runtime.ray.mode=existing")
            env = runtime_environment(config["runtime"])
            install_package_init(env=env)
            env.pop("RAY_ADDRESS", None)
            return subprocess.call(
                ray_command(args.action, config["runtime"]["ray"]), env=env
            )
        return run_training(config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print("HCU launch error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
