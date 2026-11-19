# SPDX-License-Identifier: Apache-2.0

"""Launch a v1 SGLang backend through the explicit HCU service entrypoint."""

from __future__ import annotations


def launch_server(self, server_args):
    from areal.engine import sglang_remote as upstream
    from hcu_areal.infra.utils.proc import hcu_command

    awex_meta_addr = server_args.pop("awex_meta_server_addr", None)
    awex_colocate = server_args.pop("awex_colocate_mode", False)
    awex_gpus_per_server = server_args.pop("_awex_gpus_per_server", None)
    if (
        awex_colocate
        or awex_meta_addr
        or awex_gpus_per_server is not None
        or upstream.os.environ.get("AWEX_META_SERVER_ADDR")
    ):
        raise ValueError("v1 AWEX colocate is unsupported by the HCU SGLang launcher")
    cmd = upstream.SGLangConfig.build_cmd_from_args(server_args)
    env = self.build_server_env(upstream.os.environ)
    return upstream.subprocess.Popen(
        hcu_command(cmd),
        env=env,
        stdout=upstream.sys.stdout,
        stderr=upstream.sys.stdout,
    )
