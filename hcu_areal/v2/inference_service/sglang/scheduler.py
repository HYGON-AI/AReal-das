# SPDX-License-Identifier: Apache-2.0

"""SGLang scheduler process API compatibility for AReaL v2.1.0."""

from __future__ import annotations

import logging
import os
import signal
import functools


MODULE = "areal.v2.inference_service.sglang.scheduler"


def _bridge_init_factory(original):
    @functools.wraps(original)
    def initialize(self, scheduler):
        ps = getattr(scheduler, "ps", None)
        if ps is None:
            return original(self, scheduler)
        upstream = original.__globals__
        self._scheduler, self._adapter, self._result_push = scheduler, None, None
        result_ipc = upstream["os"].environ.get(upstream["RESULT_IPC_ENV"])
        if result_ipc and ps.tp_rank == 0 and (ps.dp_rank is None or ps.dp_rank == 0):
            zmq = upstream["zmq"]
            self._result_push = zmq.Context(1).socket(zmq.PUSH)
            self._result_push.connect(result_ipc)

    return initialize


def _bridge_report_factory(original):
    @functools.wraps(original)
    def report(self):
        scheduler = self._scheduler
        ps = getattr(scheduler, "ps", None)
        if ps is None:
            return original(self)
        local_meta = self._require_adapter().get_weight_metadata()
        if ps.tp_size > 1:
            gathered = [[] for _ in range(ps.tp_size)]
            original.__globals__["dist"].all_gather_object(
                gathered, local_meta, group=scheduler.tp_cpu_group
            )
            local_meta = [item for rank_meta in gathered for item in rank_meta]
        self._push_result(original.__globals__["serialize_value"](local_meta))

    return report


def _bridge_get_parameters_factory(original):
    @functools.wraps(original)
    def get_parameters(self, save_path, names=None):
        ps = getattr(self._scheduler, "ps", None)
        if ps is None:
            return original(self, save_path, names)
        adapter = self._require_adapter()
        if ps.tp_rank == 0:
            adapter.save_parameters(save_path, names)

    return get_parameters


def prepare_scheduler_process(server_args, *, v2_awex=True):
    """Prepare the spawned scheduler before it loads its model backend."""
    from hcu_areal.launch import prepare_process_environment
    from hcu_areal.areal_adaptor import apply_modules, apply_sglang_patches

    prepare_process_environment()
    if v2_awex:
        apply_sglang_patches()
    backends = (
        getattr(server_args, "attention_backend", None),
        getattr(server_args, "prefill_attention_backend", None),
        getattr(server_args, "decode_attention_backend", None),
    )
    if "hcu_mla" in backends:
        apply_modules("sglang.srt.layers.attention.hcu_mla_backend")


def _close_sglang_process_groups():
    """Release SGLang's groups after its scheduler exits gracefully."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    try:
        destroy_model_parallel()
    finally:
        destroy_distributed_environment()
    logging.getLogger(MODULE).info("Closed SGLang distributed process groups")


def upstream_run_scheduler_process(*args, **kwargs):
    """Run the upstream scheduler API while retaining the HCU process entry."""
    server_args = args[0] if args else kwargs["server_args"]
    prepare_scheduler_process(server_args)
    from areal.v2.inference_service.sglang.scheduler import areal_run_scheduler_process

    return areal_run_scheduler_process(*args, **kwargs)


def areal_run_scheduler_process(
    server_args,
    port_args,
    gpu_id,
    tp_rank,
    attn_cp_rank,
    moe_dp_rank,
    moe_ep_rank,
    pp_rank,
    dp_rank,
    pipe_writer,
    display_tp_rank=None,
    display_dp_rank=None,
    display_moe_ep_rank=None,
):
    # Spawn starts a new interpreter, so apply this scheduler's own patches.
    prepare_scheduler_process(server_args)
    from hcu_areal.areal_adaptor import apply_modules

    apply_modules(MODULE, "areal.v2.inference_service.sglang.pp_bridge")

    from importlib import import_module

    native = import_module("sglang.srt.managers.scheduler")
    if hasattr(native, "publish") and hasattr(native, "get_observability"):
        return _run_scheduler_0518(
            native,
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            attn_cp_rank,
            moe_dp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
            pipe_writer,
            display_tp_rank,
            display_dp_rank,
            display_moe_ep_rank,
        )
    if any(
        value is not None
        for value in (display_tp_rank, display_dp_rank, display_moe_ep_rank)
    ):
        raise RuntimeError(
            "The installed SGLang scheduler does not accept display ranks"
        )

    import psutil
    from sglang.srt.environ import envs
    from sglang.srt.managers.scheduler import Scheduler, configure_scheduler_process
    from sglang.srt.observability.trace import (
        process_tracing_init,
        trace_set_thread_info,
    )
    from sglang.srt.utils import (
        get_bool_env_var,
        kill_itself_when_parent_died,
        set_gpu_proc_affinity,
    )
    from sglang.srt.utils.numa_utils import (
        get_numa_node_if_available,
        numa_bind_to_node,
    )
    from sglang.utils import get_exception_traceback
    from areal.v2.inference_service.sglang.pp_bridge import PPSchedulerBridge
    from areal.v2.inference_service.sglang.scheduler import AwexSchedulerBridge

    logger = logging.getLogger(MODULE)
    dp_rank = configure_scheduler_process(
        server_args,
        gpu_id,
        tp_rank,
        attn_cp_rank,
        moe_dp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
    )
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, gpu_id
        )
    numa_node = get_numa_node_if_available(server_args, gpu_id)
    if numa_node is not None and not envs.SGLANG_NUMA_BIND_V2.get():
        numa_bind_to_node(numa_node)
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        thread_label = "Scheduler"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
        trace_set_thread_info(thread_label, tp_rank, dp_rank)
    try:
        scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            attn_cp_rank,
            moe_dp_rank,
            dp_rank,
        )
        AwexSchedulerBridge(scheduler).bind()
        PPSchedulerBridge(scheduler, server_args).bind()
        pipe_writer.send(scheduler.get_init_info())
        scheduler.run_event_loop()
    except Exception:
        logger.error("Scheduler hit an exception: %s", get_exception_traceback())
        parent_process.send_signal(signal.SIGQUIT)


def _run_scheduler_0518(
    native,
    server_args,
    port_args,
    gpu_id,
    tp_rank,
    attn_cp_rank,
    moe_dp_rank,
    moe_ep_rank,
    pp_rank,
    dp_rank,
    pipe_writer,
    display_tp_rank,
    display_dp_rank,
    display_moe_ep_rank,
):
    """AReaL's bridge binding within the 0.5.18 scheduler lifecycle."""
    from areal.v2.inference_service.sglang.scheduler import AwexSchedulerBridge
    from areal.v2.inference_service.sglang.pp_bridge import PPSchedulerBridge

    native.load_plugins()
    native.publish(server_args, role="scheduler")
    dp_rank = native.configure_scheduler_process(
        server_args,
        gpu_id,
        tp_rank,
        attn_cp_rank,
        moe_dp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
        display_tp_rank=display_tp_rank,
        display_dp_rank=display_dp_rank,
        display_moe_ep_rank=display_moe_ep_rank,
    )
    parent = native.psutil.Process().parent()
    observability = native.get_observability()
    if observability.enable_trace:
        native.process_tracing_init(
            observability.otlp_traces_endpoint,
            "sglang",
            trace_modules=observability.trace_modules,
        )
        mode = native.get_disagg().disaggregation_mode
        label = {"prefill": "Prefill Scheduler", "decode": "Decode Scheduler"}.get(
            mode, "Scheduler"
        )
        native.trace_set_thread_info(label, tp_rank, dp_rank, pp_rank)

    scheduler = None
    try:
        scheduler = native.Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            attn_cp_rank,
            moe_dp_rank,
            dp_rank,
        )
        AwexSchedulerBridge(scheduler).bind()
        PPSchedulerBridge(scheduler, server_args).bind()
        pipe_writer.send(scheduler.get_init_info())
        scheduler.run_event_loop()
    except Exception:
        logging.getLogger(MODULE).error(
            "Scheduler hit an exception: %s", native.get_exception_traceback()
        )
        parent.send_signal(signal.SIGQUIT)
        if native.envs.SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION.get():
            try:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            except OSError:
                pass
    finally:
        if scheduler is not None:
            scheduler.metrics_reporter._shutdown_fpm()
            if scheduler.gracefully_exit:
                try:
                    scheduler.release_host_resources()
                finally:
                    _close_sglang_process_groups()
