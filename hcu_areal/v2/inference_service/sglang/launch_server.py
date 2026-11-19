# SPDX-License-Identifier: Apache-2.0

"""AReaL's SGLang service entry, adapted to the newer Engine return value."""

from __future__ import annotations

import os
import sys


MODULE = "areal.v2.inference_service.sglang.launch_server"


def is_sglang_453(sglang_version=None):
    if sglang_version is None:
        from importlib.metadata import version

        sglang_version = version("sglang")
    revision = str(sglang_version).lower()
    return revision.startswith("0.5.18") and "g453edc" in revision


def prepare_triton_kv_layout(argv, env=None, *, sglang_version=None):
    """Select the verified KV layout for the 453edc Triton implementation."""
    env = os.environ if env is None else env
    if env.get("HCU_AREAL_ENABLE", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    names = {
        "--attention-backend",
        "--prefill-attention-backend",
        "--decode-attention-backend",
    }
    selected = {}
    for index, arg in enumerate(argv):
        name, separator, value = arg.partition("=")
        if name not in names:
            continue
        if not separator:
            if index + 1 >= len(argv):
                continue  # The upstream CLI will report the missing argument.
            value = argv[index + 1]
        selected[name] = value.strip().lower()
    default = selected.get("--attention-backend")
    backends = {
        selected.get("--prefill-attention-backend", default),
        selected.get("--decode-attention-backend", default),
    } - {None}
    if "triton" not in backends:
        return
    if not is_sglang_453(sglang_version):
        return
    if backends != {"triton"}:
        raise RuntimeError(
            "SGLang 453edc mixed Triton/other attention backends are not validated"
        )
    configured = env.get("SGLANG_KV_LAYOUT_HCU_FA")
    if configured is None:
        env["SGLANG_KV_LAYOUT_HCU_FA"] = "0"
    elif configured.strip().lower() not in {"0", "false", "no", "off"}:
        raise ValueError("SGLang 453edc Triton requires SGLANG_KV_LAYOUT_HCU_FA=0")


def _register_hcu_mla_choice():
    """Expose SGLang's registered HCU MLA backend to its CLI parser."""
    if os.environ.get("HCU_AREAL_ENABLE", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    if not any(arg == "hcu_mla" or arg.endswith("=hcu_mla") for arg in sys.argv[1:]):
        return
    from sglang.srt.server_args import (
        ATTENTION_BACKEND_CHOICES,
        add_attention_backend_choices,
    )

    if "hcu_mla" not in ATTENTION_BACKEND_CHOICES:
        add_attention_backend_choices(["hcu_mla"])


def areal_launch_server(server_args):
    from sglang.srt.entrypoints.engine import Engine, init_tokenizer_manager
    from sglang.srt.entrypoints.http_server import (
        _execute_server_warmup,
        _setup_and_run_http_server,
        app,
    )
    from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
    from areal.v2.inference_service.sglang.awex import register_awex_endpoints
    from areal.v2.inference_service.sglang.rpc_proxy import RpcProxy
    from areal.v2.inference_service.sglang.scheduler import (
        areal_run_scheduler_process,
        create_result_ipc,
    )

    result_ipc = create_result_ipc()
    result = Engine._launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=init_tokenizer_manager,
        run_scheduler_process_func=areal_run_scheduler_process,
        run_detokenizer_process_func=run_detokenizer_process,
    )
    if len(result) not in (5, 6):
        raise RuntimeError(
            f"Unexpected SGLang Engine subprocess result: {len(result)} items"
        )
    tokenizer, template, ports, scheduler_result, watchdog = result[:5]
    daemons = result[5] if len(result) == 6 else []
    proxy = None
    try:
        if tokenizer is None:
            return
        proxy = RpcProxy(ports, result_ipc)
        register_awex_endpoints(app, proxy)
        _setup_and_run_http_server(
            server_args,
            tokenizer,
            template,
            ports,
            scheduler_result.scheduler_infos,
            watchdog,
            execute_warmup_func=_execute_server_warmup,
        )
    finally:
        try:
            if proxy is not None:
                proxy.close()
        finally:
            if daemons:
                Engine._terminate_weight_cache_daemons(daemons)


def main():
    """Call the patched AReaL function, rather than a second __main__ copy."""
    from areal.v2.inference_service.sglang.launch_server import areal_launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    from sglang.srt.utils.common import suppress_noisy_warnings

    suppress_noisy_warnings()
    _register_hcu_mla_choice()
    server_args = prepare_server_args(sys.argv[1:])
    try:
        areal_launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
