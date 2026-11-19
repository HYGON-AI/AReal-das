# SPDX-License-Identifier: Apache-2.0

"""Allow inference services enough time to finish graceful shutdown."""

from functools import wraps


def _kill_forked_service_factory(original):
    @wraps(original)
    def kill(self, guard_addr, role, worker_index):
        if role not in ("gateway", "inf-server"):
            return original(self, guard_addr, role, worker_index)
        httpx = original.__globals__["httpx"]
        logger = original.__globals__["logger"]
        try:
            response = self._sync_client.post(
                f"{guard_addr}/kill_forked_worker",
                json={"role": role, "worker_index": worker_index},
                timeout=45,
            )
            if response.status_code == 200:
                logger.info("Killed forked service %s/%d", role, worker_index)
            else:
                logger.warning(
                    "Failed to kill forked service %s/%d: %s",
                    role,
                    worker_index,
                    response.text,
                )
        except httpx.HTTPError as exc:
            logger.error(
                "Error killing forked service %s/%d: %s", role, worker_index, exc
            )

    return kill
