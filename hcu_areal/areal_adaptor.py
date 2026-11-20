# SPDX-License-Identifier: Apache-2.0

"""Register HCU adaptations without importing AReaL or device libraries.

Patch implementations live under the matching AReaL or third-party paths.
The order here is significant: Qwen3-VL wraps the shared AWEX adaptations.
"""

from __future__ import annotations

import os
import threading

from .api import cli_args
from .engine import fsdp_engine, megatron_engine, sglang_remote
from .infra.rpc.guard import engine_blueprint
from .infra.scheduler import ray as ray_scheduler
from .infra.utils import proc
from .models.mcore import bailing_moe, registry
from .patch_utils import AReaLPatchesManager
from .third_party.sglang.srt.layers.attention import hcu_mla_backend
from .trainer import rl_trainer
from .utils import logging as areal_logging
from .v2.inference_service.sglang import launch_server as sglang_launch_server
from .v2.inference_service.sglang import pp_bridge, scheduler as sglang_scheduler
from .v2.inference_service import controller as inference_controller
from .v2.training_service.controller import controller
from .v2.training_service.worker import awex as worker_awex
from .v2.training_service.worker import engine as worker_engine
from .v2.weight_update import nccl_group
from .v2.weight_update.awex import fsdp_adapter, megatron_adapter, sglang_adapter
from .v2.weight_update.controller import controller as weight_controller
from .v2.weight_update.gateway import app as gateway_app
from .workflow import vision_rlvr


_TRUE = {"1", "true", "yes", "on"}
_FSDP_AWEX = "areal.v2.weight_update.awex.fsdp_adapter"
_MEGATRON_AWEX = "areal.v2.weight_update.awex.megatron_adapter"
_SGLANG_AWEX = "areal.v2.weight_update.awex.sglang_adapter"
_EXPLICIT_MANAGER = None
_EXPLICIT_LOCK = threading.RLock()


def _enabled(env, disable_key, *, strip=True):
    value = str(env.get(disable_key, ""))
    # Gemma3's existing switch intentionally does not strip whitespace.
    return (value.strip() if strip else value).lower() not in _TRUE


def enabled_awex_models(env):
    """Models enabled for the shared Megatron/SGLang AWEX dispatch."""
    switches = (
        ("qwen35", "HCU_AREAL_DISABLE_QWEN35_AWEX", True),
        ("gemma3", "HCU_AREAL_DISABLE_GEMMA3_AWEX", False),
        ("glm5", "HCU_AREAL_DISABLE_GLM5_AWEX", True),
        ("deepseek", "HCU_AREAL_DISABLE_DEEPSEEK_AWEX", True),
    )
    return frozenset(
        name for name, key, strip in switches if _enabled(env, key, strip=strip)
    )


def _add(
    manager, module, target, replacement, patch_id, *, wrapper=True, compose=False
):
    manager.register_patch(
        f"{module}.{target}",
        replacement,
        module_name=module,
        patch_id=patch_id,
        apply_wrapper=wrapper,
        compose=compose,
    )


def _register_shared_awex(manager, models, *, trace=False):
    if not models:
        return
    infer_owner = next(
        name for name in ("qwen35", "glm5", "deepseek", "gemma3") if name in models
    )
    infer_patches = [
        ("_unfuse_params", sglang_adapter._infer_parts_factory, "unfuse"),
        (
            "_build_sharding_strategy",
            sglang_adapter._infer_strategy_factory,
            "sharding",
        ),
    ]
    if trace:
        infer_patches.append(
            ("get_weight_metadata", sglang_adapter._infer_metadata_factory, "meta")
        )
    for method, factory, suffix in infer_patches:
        _add(
            manager,
            _SGLANG_AWEX,
            f"AwexSGLangAdapter.{method}",
            lambda original, factory=factory: factory(original, models),
            f"{infer_owner}.awex.infer.{suffix}",
        )

    preflight_owner = next(
        (name for name in ("gemma3", "glm5", "deepseek") if name in models), None
    )
    if preflight_owner is not None:
        _add(
            manager,
            _SGLANG_AWEX,
            "AwexSGLangAdapter.init_weight_update_group",
            lambda original: sglang_adapter._group_preflight_factory(
                original, "infer", models
            ),
            f"{preflight_owner}.awex.infer.preflight",
        )

    megatron_models = models.intersection({"glm5", "deepseek", "qwen35"})
    if not megatron_models:
        return
    owner = next(
        name for name in ("glm5", "deepseek", "qwen35") if name in megatron_models
    )
    megatron_patches = [
        (
            _MEGATRON_AWEX,
            "AwexMegatronAdapter._iter_hf_params",
            megatron_adapter._train_iter_factory,
            "train.params",
        ),
        (
            _MEGATRON_AWEX,
            "AwexMegatronAdapter.init_weight_update_group",
            megatron_adapter._train_preflight_factory,
            "train.preflight",
        ),
        (
            _SGLANG_AWEX,
            "AwexSGLangAdapter.execute_weight_update",
            sglang_adapter._refresh_factory,
            "infer.refresh",
        ),
        (
            _SGLANG_AWEX,
            "AwexSGLangAdapter.execute_colocate_weight_update",
            sglang_adapter._colocate_refresh_factory,
            "infer.colocate_refresh",
        ),
    ]
    if trace:
        megatron_patches.insert(
            1,
            (
                _MEGATRON_AWEX,
                "AwexMegatronAdapter.get_weight_metadata",
                megatron_adapter._train_metadata_factory,
                "train.meta",
            ),
        )
    for module, method, factory, suffix in megatron_patches:
        _add(
            manager,
            module,
            method,
            lambda original, factory=factory: factory(original, megatron_models),
            f"{owner}.awex.{suffix}",
        )


def patch_features(env=None) -> AReaLPatchesManager:
    """Build the ordered patch list for the enabled HCU adaptations."""
    env = os.environ if env is None else env
    manager = AReaLPatchesManager()
    models = enabled_awex_models(env)

    _add(
        manager,
        areal_logging.MODULE,
        "getLogger",
        areal_logging._get_logger_factory,
        "areal.logging.preserve_manager",
    )

    # FSDP training, followed by the AWEX dispatch shared across models.
    if "qwen35" in models:
        _add(
            manager,
            _FSDP_AWEX,
            "AwexFSDPAdapter.get_weight_metadata",
            fsdp_adapter._qwen35_train_metadata_factory,
            "qwen35.awex.train.meta",
        )
        _add(
            manager,
            _FSDP_AWEX,
            "AwexFSDPAdapter.get_local_shard_parameters",
            fsdp_adapter._qwen35_train_params_factory,
            "qwen35.awex.train.params",
        )
    # FSDP exports floating weights in the compute dtype for all models. Keep
    # this behavior independent of the Qwen3-VL layout switch.
    _add(
        manager,
        _FSDP_AWEX,
        "AwexFSDPAdapter.get_weight_metadata",
        fsdp_adapter._compute_dtype_metadata_factory,
        "fsdp.awex.compute_dtype.meta",
        compose=True,
    )
    _add(
        manager,
        _FSDP_AWEX,
        "AwexFSDPAdapter.get_local_shard_parameters",
        fsdp_adapter._compute_dtype_params_factory,
        "fsdp.awex.compute_dtype.params",
        compose=True,
    )
    if "gemma3" in models:
        _add(
            manager,
            _FSDP_AWEX,
            "AwexFSDPAdapter._to_hf_name",
            fsdp_adapter._gemma3_train_name_factory,
            "gemma3.awex.train.names",
        )
        _add(
            manager,
            _FSDP_AWEX,
            "AwexFSDPAdapter.init_weight_update_group",
            lambda original: fsdp_adapter._group_preflight_factory(
                original, "train", frozenset({"gemma3"})
            ),
            "gemma3.awex.train.preflight",
        )
    _register_shared_awex(
        manager,
        models,
        trace=bool(str(env.get("HCU_AREAL_PATCH_TRACE_DIR", "")).strip()),
    )
    if (
        str(env.get("HCU_AREAL_MEGATRON_AWEX_RELEASE_GRADS", "")).strip().lower()
        in _TRUE
    ):
        _add(
            manager,
            _MEGATRON_AWEX,
            "AwexMegatronAdapter.execute_weight_update",
            megatron_adapter._dense_sync_release_grad_buffers_factory,
            "megatron.awex.release_grads",
        )

    # SGLang-DAS 453edca is missing the pools used by its model forward context.
    _add(
        manager,
        hcu_mla_backend.MODULE,
        "HCUMLABackend.__init__",
        hcu_mla_backend._init_factory,
        "sglang.mla.pool_compat",
    )

    # AReaL configuration, engine, scheduler and lifecycle adaptations.
    if env.get("HCU_AREAL_MCORE_DEV_COMPAT") == "1":
        _add(
            manager,
            megatron_engine.MODULE,
            "MegatronEngine._create_optimizer",
            megatron_engine._core_dev_optimizer_factory,
            "megatron.core_dev.optimizer",
        )
    offload_fraction = megatron_engine.optimizer_cpu_offload_fraction(env)
    pin_cpu_grads = megatron_engine.optimizer_pin_cpu_grads(env)
    if pin_cpu_grads is not None and offload_fraction is None:
        raise ValueError(
            "HCU_AREAL_OPTIMIZER_PIN_CPU_GRADS requires optimizer CPU offload"
        )
    if offload_fraction is not None:
        _add(
            manager,
            megatron_engine.MODULE,
            "MCoreOptimizerConfig",
            lambda original: megatron_engine._optimizer_cpu_offload_config_factory(
                original, offload_fraction, pin_cpu_grads
            ),
            "megatron.optimizer.cpu_offload",
        )
    if _enabled(env, "HCU_AREAL_DISABLE_GLM_DSA"):
        _add(
            manager,
            megatron_engine.MODULE,
            "configure_pipeline_layer_splits",
            megatron_engine._configure_pipeline_factory,
            "glm.dsa.pipeline_config",
        )
    if _enabled(env, "HCU_AREAL_DISABLE_NCCL_COMPAT"):
        _add(
            manager,
            nccl_group.MODULE,
            "init_custom_process_group",
            nccl_group.init_custom_process_group,
            "nccl.group.version_compat",
            wrapper=False,
        )
    if _enabled(env, "HCU_AREAL_DISABLE_EXIT_DIAGNOSTICS"):
        _add(
            manager,
            rl_trainer.MODULE,
            "PPOTrainer.close",
            rl_trainer._close_factory,
            "trainer.exit.diagnostics",
        )
    if _enabled(env, "HCU_AREAL_DISABLE_MCORE_FUSION_FALLBACK"):
        _add(
            manager,
            registry.MODULE,
            "make_mcore_model",
            registry._make_model_factory,
            "mcore.fused_gradient_fallback",
        )
    if _enabled(env, "HCU_AREAL_DISABLE_FSDP_COMPAT"):
        _add(
            manager,
            fsdp_engine.MODULE,
            "FSDPEngine._create_device_model",
            fsdp_engine._create_device_model_factory,
            "fsdp.qwen3vl.vision_packing",
        )
        _add(
            manager,
            fsdp_engine.MODULE,
            "FSDPEngine._get_model_name_parameters",
            fsdp_engine._gemma3_xccl_names_factory,
            "gemma3.xccl.vision_names",
        )
    if _enabled(env, "HCU_AREAL_DISABLE_BAILING_ROPE"):
        _add(
            manager,
            bailing_moe.MODULE,
            "_patch_mla_thd_rope_for_cp",
            lambda original: bailing_moe._patch_factory(original, manager),
            "bailing.mla.rope_kwargs",
        )
    if _enabled(env, "HCU_AREAL_DISABLE_RAY_LAUNCHER"):
        _add(
            manager,
            "areal.infra.scheduler.ray",
            "RayWorkerProcessLauncher",
            ray_scheduler._launcher_factory,
            "ray.launcher.environment",
        )
        _add(
            manager,
            "areal.infra.scheduler.ray",
            "RayScheduler._prepare_worker_specs",
            ray_scheduler._prepare_worker_specs_factory,
            "ray.worker_env.precedence",
        )
    if _enabled(env, "HCU_AREAL_DISABLE_SGLANG_CONFIG"):
        _add(
            manager,
            cli_args.MODULE,
            "SGLangConfig.build_args",
            lambda original: cli_args._build_args_factory(original, manager),
            "sglang.config.hcu",
        )
    if _enabled(env, "HCU_AREAL_DISABLE_SGLANG_SCHEDULER"):
        _add(
            manager,
            sglang_scheduler.MODULE,
            "areal_run_scheduler_process",
            sglang_scheduler.areal_run_scheduler_process,
            "sglang.scheduler.api_compat",
            wrapper=False,
        )
        for method, factory in (
            ("__init__", sglang_scheduler._bridge_init_factory),
            ("awex_report_weight_meta", sglang_scheduler._bridge_report_factory),
            ("awex_get_parameters", sglang_scheduler._bridge_get_parameters_factory),
        ):
            _add(
                manager,
                sglang_scheduler.MODULE,
                f"AwexSchedulerBridge.{method}",
                factory,
                f"sglang.scheduler.bridge.{method}",
            )
        _add(
            manager,
            pp_bridge.MODULE,
            "PPSchedulerBridge.bind",
            pp_bridge._bind_factory,
            "sglang.scheduler.pp_bridge",
        )
        _add(
            manager,
            _SGLANG_AWEX,
            "AwexSGLangAdapter._get_model_context",
            sglang_adapter._model_context_factory,
            "sglang.scheduler.model_context",
        )
    _add(
        manager,
        sglang_launch_server.MODULE,
        "areal_launch_server",
        sglang_launch_server.areal_launch_server,
        "sglang.server.api_compat",
        wrapper=False,
    )

    # These wrappers compose with, and must be registered after, shared AWEX.
    if _enabled(env, "HCU_AREAL_DISABLE_QWEN3VL_AWEX"):
        for module, target, factory, patch_id in (
            (
                _FSDP_AWEX,
                "AwexFSDPAdapter._to_hf_name",
                fsdp_adapter._qwen3vl_train_name_factory,
                "qwen3vl.train.name",
            ),
            (
                _FSDP_AWEX,
                "AwexFSDPAdapter.get_local_shard_parameters",
                fsdp_adapter._qwen3vl_train_params_factory,
                "qwen3vl.train.params",
            ),
            (
                _FSDP_AWEX,
                "AwexFSDPAdapter.init_weight_update_group",
                fsdp_adapter._qwen3vl_train_init_factory,
                "qwen3vl.train.init",
            ),
            (
                _FSDP_AWEX,
                "AwexFSDPAdapter.teardown_weight_update_group",
                fsdp_adapter._qwen3vl_train_teardown_factory,
                "qwen3vl.train.teardown",
            ),
            (
                _SGLANG_AWEX,
                "AwexSGLangAdapter._unfuse_params",
                sglang_adapter._qwen3vl_infer_parts_factory,
                "qwen3vl.infer.parts",
            ),
            (
                _SGLANG_AWEX,
                "AwexSGLangAdapter._build_sharding_strategy",
                sglang_adapter._qwen3vl_infer_strategy_factory,
                "qwen3vl.infer.strategy",
            ),
            (
                _SGLANG_AWEX,
                "AwexSGLangAdapter.get_weight_metadata",
                sglang_adapter._qwen3vl_infer_metadata_factory,
                "qwen3vl.infer.metadata",
            ),
        ):
            _add(manager, module, target, factory, patch_id, compose=True)

    # Controller and vision adaptations are explicit same-path implementations.
    if _enabled(env, "HCU_AREAL_DISABLE_TRAIN_CONTROLLER"):
        for name, replacement in (
            ("connect_engine", controller.connect_engine),
            ("update_weights", controller.update_weights),
            ("_cleanup_runtime_state", controller._cleanup_runtime_state),
        ):
            _add(
                manager,
                controller.MODULE,
                f"GatewayTrainController.{name}",
                replacement,
                f"train.controller.{name}",
                wrapper=False,
            )
    if _enabled(env, "HCU_AREAL_DISABLE_WEIGHT_GATEWAY"):
        _add(
            manager,
            gateway_app.MODULE,
            "create_app",
            lambda original: gateway_app._factory(original, manager),
            "weight.gateway.v210",
        )
    if _enabled(env, "HCU_AREAL_DISABLE_GEMMA3_VISION"):
        _add(
            manager,
            vision_rlvr.MODULE,
            "VisionRLVRWorkflow.arun_episode",
            vision_rlvr.arun_episode,
            "gemma3.vision.token_types",
            wrapper=False,
        )

    # Select the actual training backend where the worker imports its engine.
    # These are explicit consumer boundaries, not a general import hook.
    if manager.module_names.intersection(
        {
            _FSDP_AWEX,
            _MEGATRON_AWEX,
            fsdp_engine.MODULE,
            megatron_engine.MODULE,
            registry.MODULE,
            bailing_moe.MODULE,
        }
    ):
        for module, factory, patch_id in (
            (
                "areal.infra.rpc.guard.engine_blueprint",
                engine_blueprint._engine_import_factory,
                "train.engine_import.rpc",
            ),
            (
                "areal.v2.training_service.worker.engine",
                worker_engine._engine_import_factory,
                "train.engine_import.v2",
            ),
        ):
            _add(manager, module, "import_from_string", factory, patch_id)

    if not _enabled(env, "HCU_AREAL_DISABLE_SGLANG_SCHEDULER"):
        _add(
            manager,
            sglang_scheduler.MODULE,
            "areal_run_scheduler_process",
            sglang_scheduler.upstream_run_scheduler_process,
            "sglang.scheduler.process_entry",
            wrapper=False,
        )
    for module, patch_id in (
        ("areal.infra.scheduler.ray", "launch.ray_subprocess"),
        ("areal.infra.scheduler.local", "launch.local_subprocess"),
        ("areal.infra.rpc.guard.app", "launch.guard_subprocess"),
    ):
        _add(
            manager,
            module,
            "run_with_streaming_logs",
            proc._run_with_streaming_logs_factory,
            patch_id,
        )
        _add(
            manager,
            module,
            "kill_process_tree",
            proc._kill_process_tree_factory,
            f"{patch_id}.cleanup",
        )
    _add(
        manager,
        "areal.v2.inference_service.controller.controller",
        "RolloutControllerV2._kill_forked_service",
        inference_controller._kill_forked_service_factory,
        "inference.service.graceful_cleanup",
    )
    _add(
        manager,
        "areal.v2.weight_update.controller.controller",
        "WeightUpdateController.initialize",
        weight_controller.initialize,
        "launch.weight_gateway",
        wrapper=False,
    )
    _add(
        manager,
        "areal.engine.sglang_remote",
        "SGLangBackend.launch_server",
        sglang_remote.launch_server,
        "launch.sglang_server",
        wrapper=False,
    )
    _add(
        manager,
        "areal.v2.training_service.worker.awex",
        "_create_training_adapter",
        worker_awex._create_training_adapter,
        "train.awex.adapter_boundary",
        wrapper=False,
    )
    return manager


def apply_modules(*module_names):
    """Apply registered patches to explicitly named, role-relevant modules.

    Callers must select modules at the process or component boundary. This
    function neither monitors imports nor imports modules for disabled patches.
    """
    if os.environ.get("HCU_AREAL_ENABLE", "0").strip().lower() not in _TRUE:
        return
    import torch

    if torch.version.hip is None:
        return
    global _EXPLICIT_MANAGER
    import importlib

    with _EXPLICIT_LOCK:
        if _EXPLICIT_MANAGER is None:
            _EXPLICIT_MANAGER = patch_features()
        for name in module_names:
            if name in _EXPLICIT_MANAGER.module_names:
                _EXPLICIT_MANAGER.apply_module(importlib.import_module(name))


def apply_package_patches():
    """Patch light modules already imported by AReaL's package initializer.

    Heavy training and SGLang modules remain deferred to the component that
    actually uses them, including in Ray actors without an accelerator.
    """
    apply_modules(
        "areal.api.cli_args", "areal.infra.scheduler.ray", "areal.infra.scheduler.local"
    )


def apply_driver_patches():
    """Apply driver-only patches before the training script consumes them."""
    if os.getenv("HCU_AREAL_MCORE_DEV_COMPAT") == "1":
        megatron_engine.enable_core_dev_compat()
    apply_modules(
        "areal.api.cli_args",
        "areal.infra.scheduler.ray",
        "areal.infra.scheduler.local",
        "areal.engine.sglang_remote",
        "areal.v2.inference_service.controller.controller",
        "areal.v2.training_service.controller.controller",
        "areal.v2.weight_update.controller.controller",
        "areal.trainer.rl_trainer",
        "areal.workflow.vision_rlvr",
    )


def apply_guard_patches():
    """Prepare a guard to start its own HCU worker processes."""
    apply_modules("areal.api.cli_args", "areal.infra.rpc.guard.app")


def apply_sglang_server_patches():
    """Select the HCU scheduler function before the server starts it."""
    apply_modules(
        "areal.api.cli_args",
        "areal.v2.inference_service.sglang.scheduler",
        "areal.v2.inference_service.sglang.launch_server",
    )


def apply_gateway_patches():
    apply_modules("areal.v2.weight_update.gateway.app")


def apply_worker_patches(version):
    """Prepare the explicit engine-import boundary for one worker type."""
    if version == "rpc":
        module = "areal.infra.rpc.guard.engine_blueprint"
        apply_modules("areal.workflow.vision_rlvr")
    elif version == "v2":
        module = "areal.v2.training_service.worker.engine"
        apply_modules("areal.v2.training_service.worker.awex")
    else:
        raise ValueError(f"Unsupported HCU worker type: {version}")
    apply_modules("areal.api.cli_args", "areal.engine.sglang_remote", module)


def apply_process_role(module):
    """Apply only the adaptations required by a known AReaL service."""
    if module == "areal.infra.rpc.rpc_server":
        apply_guard_patches()
        apply_worker_patches("rpc")
    elif module == "areal.v2.training_service.worker":
        apply_worker_patches("v2")
    elif module in (
        "areal.v2.training_service.guard",
        "areal.v2.inference_service.guard",
    ):
        apply_guard_patches()
    elif module == "areal.v2.inference_service.sglang.launch_server":
        apply_sglang_server_patches()
    elif module == "areal.v2.weight_update.gateway":
        apply_gateway_patches()
    else:
        raise ValueError(f"Unsupported HCU service module: {module}")


def apply_training_patches(backend):
    """Apply training patches only after the worker's backend is known."""
    apply_modules("areal.v2.weight_update.nccl_group")
    if backend == "fsdp":
        apply_modules("areal.engine.fsdp_engine", _FSDP_AWEX)
    elif backend == "megatron":
        if os.getenv("HCU_AREAL_MCORE_DEV_COMPAT") == "1":
            megatron_engine.enable_core_dev_compat()
        # The engine imports make_mcore_model by value; patch its defining
        # module before importing the engine, not by scanning other modules.
        apply_modules(
            "areal.models.mcore.bailing_moe",
            "areal.models.mcore.registry",
            "areal.engine.megatron_engine",
            _MEGATRON_AWEX,
        )
    else:
        raise ValueError(f"Unsupported HCU training backend: {backend}")


def apply_sglang_patches():
    """Apply AWEX compatibility in the SGLang scheduler process."""
    apply_modules("areal.v2.weight_update.nccl_group", _SGLANG_AWEX)
