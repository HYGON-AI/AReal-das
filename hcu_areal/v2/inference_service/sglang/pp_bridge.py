# SPDX-License-Identifier: Apache-2.0

"""AReaL PP weight-update routing for SGLang's separated WeightUpdater."""

from __future__ import annotations

import functools


MODULE = "areal.v2.inference_service.sglang.pp_bridge"


def _bind_factory(original):
    @functools.wraps(original)
    def bind(self):
        scheduler = self._scheduler
        ps = getattr(scheduler, "ps", None)
        if ps is None or self._pp_size <= 1:
            return original(self)

        worker = scheduler.tp_worker
        updater = worker.model_runner.weight_updater
        groups = updater._model_update_group
        extract_rank = original.__globals__["_extract_pp_rank_from_group_name"]
        original_init = worker.init_weights_update_group
        original_update = worker.update_weights_from_distributed
        original_destroy = worker.destroy_weights_update_group

        def init_group(request):
            target = extract_rank(request.group_name)
            if target is not None and target != ps.pp_rank:
                groups[request.group_name] = None
                return (
                    True,
                    f"Skipped group for PP rank {target}; local PP rank is {ps.pp_rank}.",
                )
            return original_init(request)

        def update(request):
            group_name = request.group_name
            if group_name in groups and groups[group_name] is None:
                return True, f"Skipped weight update for group {group_name}."
            return original_update(request)

        def destroy(request):
            group_name = request.group_name
            if group_name in groups and groups[group_name] is None:
                groups.pop(group_name)
                return True, f"Skipped group destruction for {group_name}."
            return original_destroy(request)

        worker.init_weights_update_group = init_group
        worker.update_weights_from_distributed = update
        worker.destroy_weights_update_group = destroy

    return bind
