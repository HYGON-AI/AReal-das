# SPDX-License-Identifier: Apache-2.0

"""PyTorch process-group version compatibility for official AReaL v2.1.0."""

from __future__ import annotations

import os


MODULE = "areal.v2.weight_update.nccl_group"


def init_custom_process_group(
    backend=None,
    init_method=None,
    timeout=None,
    world_size=-1,
    rank=-1,
    store=None,
    group_name=None,
    pg_options=None,
):
    """Create an additional group using the PyTorch version's option name."""
    import torch
    from packaging.version import Version
    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
        rendezvous,
    )

    assert (store is None) or (init_method is None), (
        "Cannot specify both init_method and store."
    )
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"
    backend = Backend(backend) if backend else Backend("undefined")
    if timeout is None:
        timeout = default_pg_timeout
    if store is None:
        iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(iterator)
        store.set_timeout(timeout)
        store = PrefixStore(group_name, store)

    option_name = (
        "backend_options"
        if Version(torch.__version__.split("+")[0]) >= Version("2.6.0")
        else "pg_options"
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{option_name: pg_options},
        timeout=timeout,
    )
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg
