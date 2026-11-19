# SPDX-License-Identifier: Apache-2.0

"""Start the AReaL weight gateway through its explicit HCU entrypoint."""

from __future__ import annotations


def initialize(self):
    from areal.v2.weight_update.controller import controller as upstream

    cfg = self.config
    port = cfg.port or upstream.find_free_ports(1)[0]
    cmd = [
        upstream.sys.executable,
        "-m",
        "hcu_areal.launch",
        "--module",
        "areal.v2.weight_update.gateway",
        "--host",
        cfg.host,
        "--port",
        str(port),
        "--admin-api-key",
        cfg.admin_api_key,
        "--init-timeout",
        str(cfg.init_timeout_s),
        "--update-timeout",
        str(cfg.update_timeout_s),
        "--log-level",
        cfg.log_level,
    ]
    self._gateway_proc = upstream.subprocess.Popen(
        cmd, stdout=upstream.sys.stdout, stderr=upstream.sys.stdout
    )
    self._gateway_url = f"http://{cfg.host}:{port}"
    self._session = upstream.httpx.Client()
    self._session.headers["Authorization"] = f"Bearer {cfg.admin_api_key}"
    self._wait_for_health()
    upstream.logger.info("Gateway ready at %s", self._gateway_url)
