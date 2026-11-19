# SPDX-License-Identifier: Apache-2.0

"""HCU-compatible methods for the AReaL v2 training controller."""

from __future__ import annotations

MODULE = "areal.v2.training_service.controller.controller"


def connect_engine(self, rollout, meta):
    """Connect the weight gateway with configured timeouts and cleanup."""
    import requests

    from areal.v2.inference_service.controller.controller import RolloutControllerV2
    from areal.v2.training_service.controller import controller as upstream
    from areal.v2.weight_update.controller.config import WeightUpdateControllerConfig
    from areal.v2.weight_update.controller.controller import WeightUpdateController

    self._ensure_initialized()
    if not isinstance(rollout, RolloutControllerV2):
        raise TypeError(
            "GatewayTrainController requires RolloutControllerV2, "
            f"got {type(rollout).__name__}. "
            "Ensure _version='v2' is set on InferenceEngineConfig."
        )
    self.rollout = rollout
    if meta.type not in ("awex", "disk"):
        raise ValueError(
            "GatewayTrainController supports 'awex' or 'disk' weight "
            f"updates, got '{meta.type}'"
        )

    ctrl = WeightUpdateController(
        WeightUpdateControllerConfig(
            host=upstream.gethostip(),
            admin_api_key=self.config.admin_api_key,
            log_level=self.config.log_level,
            setup_timeout=self.config.setup_timeout,
            request_timeout=self.config.request_timeout,
        )
    )
    self._weight_update_ctrl = ctrl
    try:
        ctrl.initialize()
    except BaseException:
        try:
            ctrl.destroy()
        finally:
            self._weight_update_ctrl = None
        raise

    inference_urls = rollout.inference_worker_urls
    pair_name = f"{self._role}-rollout"
    if meta.type == "awex":
        response = requests.post(
            f"{rollout.inference_guard_addrs[0]}/alloc_ports",
            json={"count": 1},
            timeout=30,
        )
        response.raise_for_status()
        port_data = response.json()
        ctrl.connect(
            pair_name=pair_name,
            train_worker_urls=self._worker_addrs,
            inference_worker_urls=inference_urls,
            mode="awex",
            nccl_master_addr=port_data["host"],
            nccl_master_port=port_data["ports"][0],
        )
    else:
        ctrl.connect(
            pair_name=pair_name,
            train_worker_urls=self._worker_addrs,
            inference_worker_urls=inference_urls,
            mode="disk",
            save_path=meta.path or "",
            use_lora=meta.use_lora,
            lora_name=meta.lora_name,
            lora_keep_versions=meta.lora_keep_versions,
        )
    self._weight_update_ctrl = ctrl
    upstream.logger.info(
        "WeightUpdateController connected (pair=%s, train=%d, inf=%d)",
        pair_name,
        len(self._worker_addrs),
        len(inference_urls),
    )


def update_weights(self, meta):
    """Keep rollout paused when the weight gateway reports failure."""
    from areal.v2.training_service.controller import controller as upstream

    if self._weight_update_ctrl is None or self.rollout is None:
        raise RuntimeError("connect_engine() must be called before update_weights()")
    self.rollout.pause_generation()
    assert meta.version is not None and meta.version > 0, (
        f"meta.version must be a positive integer, got {meta.version}"
    )
    result = self._weight_update_ctrl.update_weights(version=meta.version)
    if result.status != "ok":
        raise RuntimeError(
            f"Weight update v{meta.version} failed after {result.duration_ms:.0f}ms: {result.error!r}"
        )
    self.rollout.continue_generation()
    upstream.logger.info(
        "Weight update v%d completed (%s, %.0fms)",
        meta.version,
        result.status,
        result.duration_ms,
    )


def _cleanup_runtime_state(self):
    """Close the gateway and controller resources in their required order."""
    from areal.v2.training_service.controller import controller as upstream

    if self._router_addr and self._model_addr:
        try:
            import requests

            requests.post(
                f"{self._router_addr}/unregister",
                json={"model_addr": self._model_addr},
                headers={"Authorization": f"Bearer {self.config.admin_api_key}"},
                timeout=10,
            )
        except Exception:
            upstream.logger.error(
                "Failed to unregister model: %s", upstream.traceback.format_exc()
            )

    ctrl = self._weight_update_ctrl
    self._weight_update_ctrl = None
    if ctrl is not None:
        try:
            ctrl.destroy()
        except Exception:
            upstream.logger.exception("Failed to destroy weight update controller")
    self._graceful_shutdown_workers()

    for guard_addr, role, worker_index in reversed(self._forked_services):
        try:
            self._kill_forked_service(guard_addr, role, worker_index)
        except Exception:
            upstream.logger.error(
                "Error killing %s/%d: %s",
                role,
                worker_index,
                upstream.traceback.format_exc(),
            )
    self._forked_services.clear()
    for role in reversed(self._service_roles):
        try:
            self.scheduler.delete_workers(role=role)
            upstream.logger.info("Workers deleted for role: %s", role)
        except Exception:
            upstream.logger.error(
                "Error deleting workers for %s: %s",
                role,
                upstream.traceback.format_exc(),
            )
    self._service_roles.clear()
    self._worker_addrs.clear()
    self._router_addr = ""
    self._gateway_addr = ""
    self._model_addr = ""
    self.api_key = None

    if self._async_client is not None:
        upstream.logger.info("[EXIT] actor async HTTP client close begin")
        client = self._async_client

        async def close_client():
            await upstream.asyncio.wait_for(client.aclose(), timeout=10.0)

        try:
            upstream.run_async_task(close_client)
        except Exception:
            upstream.logger.exception("[EXIT] actor async HTTP client close failed")
        finally:
            self._async_client = None
            self._async_client_loop = None
        upstream.logger.info("[EXIT] actor async HTTP client close end")

    import torch.distributed as dist

    if self._own_process_group:
        try:
            if dist.is_initialized():
                upstream.logger.info(
                    "[EXIT] controller Gloo process group destroy begin"
                )
                dist.destroy_process_group()
                upstream.logger.info("[EXIT] controller Gloo process group destroy end")
        except Exception:
            upstream.logger.error(
                "Failed to destroy process group: %s", upstream.traceback.format_exc()
            )
        finally:
            self._own_process_group = False
