# SPDX-License-Identifier: Apache-2.0

"""Weight gateway compatibility without rewriting upstream source."""

from __future__ import annotations

import asyncio
import contextvars
import os
import time
from functools import wraps

MODULE = "areal.v2.weight_update.gateway.app"


async def _post_once(session, url, timeout_s, json_data=None):
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with session.post(url, json=json_data, timeout=timeout) as response:
        response.raise_for_status()


async def _colocate_transfer(upstream, store, pair, version, session, timeout):
    async def post_all(urls, path, payload):
        await asyncio.gather(
            *[
                upstream._post(
                    session, f"{url}/awex/{path}", timeout, json_data=payload
                )
                for url in urls
            ]
        )

    await post_all(pair.train_worker_urls, "release_memory", {"tags": ["optimizer"]})
    await post_all(pair.inference_worker_urls, "resume_memory", {"tags": ["weights"]})
    await post_all(
        pair.train_worker_urls + pair.inference_worker_urls,
        "execute_colocate_weight_update",
        {"version": version},
    )
    await post_all(pair.train_worker_urls, "release_memory", {"tags": ["weights"]})
    await post_all(pair.inference_worker_urls, "resume_memory", {"tags": ["kv_cache"]})
    for rank in range(pair.train_world_size):
        transfer_rank = pair.inference_world_size + rank
        store.delete(pair.pair_name, f"colocate_weights_rank{transfer_rank}_{version}")
        store.delete(pair.pair_name, f"colocate_done_rank{transfer_rank}_{version}")


async def _disk_transfer(upstream, pair, version, session, timeout):
    from areal.api.io_struct import SaveLoadMeta, get_versioned_lora_name
    from areal.infra.rpc.serialization import serialize_value

    save_path = os.path.join(pair.save_path, f"weight_update_v{version}")
    save_meta = SaveLoadMeta(path=save_path, weight_format="hf", with_optim=False)
    payload = {"args": serialize_value([save_meta]), "kwargs": serialize_value({})}
    await asyncio.gather(
        *[
            upstream._post_json(session, f"{url}/save", timeout, json_data=payload)
            for url in pair.train_worker_urls
        ]
    )
    if not pair.use_lora:
        await asyncio.gather(
            *[
                upstream._post_json(
                    session,
                    f"{url}/update_weights_from_disk",
                    timeout,
                    json_data={"model_path": save_path, "abort_all_requests": True},
                )
                for url in pair.inference_worker_urls
            ]
        )
        return

    lora_name = get_versioned_lora_name(pair.lora_name, version)
    await asyncio.gather(
        *[
            upstream._post_json(
                session,
                f"{url}/load_lora_adapter",
                timeout,
                json_data={"lora_name": lora_name, "lora_path": save_path},
            )
            for url in pair.inference_worker_urls
        ]
    )
    keep = pair.lora_keep_versions
    if keep > 0 and version - keep >= 0:
        stale_name = get_versioned_lora_name(pair.lora_name, version - keep)

        async def unload(url):
            try:
                await upstream._post_json(
                    session,
                    f"{url}/unload_lora_adapter",
                    timeout,
                    json_data={"lora_name": stale_name},
                )
            except Exception as exc:
                upstream.logger.warning(
                    "unload_lora_adapter(%s) on %s failed (best-effort): %s",
                    stale_name,
                    url,
                    exc,
                )

        await asyncio.gather(*[unload(url) for url in pair.inference_worker_urls])


def _replace_post_route(app, path, endpoint, request_type, response_type):
    """Replace one public FastAPI route while retaining its position and name."""
    for index, route in enumerate(app.router.routes):
        if route.path == path and "POST" in (route.methods or set()):
            endpoint.__annotations__ = {
                "request": request_type[0],
                "body": request_type[1],
                "return": response_type,
            }
            app.router.add_api_route(
                path,
                endpoint,
                methods=["POST"],
                name=route.name,
                response_model=route.response_model,
                status_code=route.status_code,
                include_in_schema=route.include_in_schema,
            )
            app.router.routes[index] = app.router.routes.pop()
            return
    raise RuntimeError(f"AReaL gateway is missing POST {path}")


def _factory(original, manager):
    """Adapt the two affected routes on the app created by AReaL v2.1."""

    @wraps(original)
    def create_app(config=None):
        from areal.v2.weight_update.gateway import app as upstream

        application = original(config)
        store = application.state.kv_store
        registry = application.state.registry
        config = application.state.config
        merge_infer = contextvars.ContextVar("hcu_merge_infer_metadata", default=False)
        put = store.put

        def put_metadata(pair_name, key, value):
            if key == "infer_params_meta" and merge_infer.get():
                value = upstream._merge_training_meta_by_name(value)
            return put(pair_name, key, value)

        store.put = put_metadata
        connect_route = next(
            (
                route
                for route in application.router.routes
                if route.path == "/connect" and "POST" in (route.methods or set())
            ),
            None,
        )
        if connect_route is None:
            raise RuntimeError("AReaL gateway is missing POST /connect")
        original_connect = connect_route.endpoint

        async def connect(request, body):
            merge = (
                not body.colocate
                and body.mode != "disk"
                and len(body.inference_worker_urls) == 1
            )
            token = merge_infer.set(merge)
            try:
                return await original_connect(request, body)
            finally:
                merge_infer.reset(token)

        async def update_weights(request, body):
            upstream.require_admin_key(request, config.admin_api_key)
            pair = registry.get_by_name(body.pair_name)
            if pair is None:
                return upstream.JSONResponse(
                    status_code=404,
                    content={"error": f"Pair '{body.pair_name}' not found"},
                )
            session = request.app.state.http_session
            timeout = config.update_timeout_s
            start = time.monotonic()
            try:
                if pair.colocate:
                    await _colocate_transfer(
                        upstream, store, pair, body.version, session, timeout
                    )
                elif pair.mode == "disk":
                    await _disk_transfer(upstream, pair, body.version, session, timeout)
                else:
                    await asyncio.gather(
                        *[
                            _post_once(
                                session,
                                f"{url}/awex/update_weights",
                                timeout,
                                json_data={"version": body.version},
                            )
                            for url in pair.train_worker_urls
                            + pair.inference_worker_urls
                        ]
                    )
            except Exception as exc:
                duration = (time.monotonic() - start) * 1000
                upstream.logger.error(
                    "Weight update failed for pair '%s': %r", pair.pair_name, exc
                )
                return upstream.WeightUpdateResult(
                    status="error",
                    version=body.version,
                    duration_ms=duration,
                    error=f"{type(exc).__name__}: {exc!r}",
                )
            duration = (time.monotonic() - start) * 1000
            pair.last_version = body.version
            upstream.logger.info(
                "Weight update completed for pair '%s' v%d (%.1fms)",
                pair.pair_name,
                body.version,
                duration,
            )
            return upstream.WeightUpdateResult(
                status="ok", version=body.version, duration_ms=duration
            )

        _replace_post_route(
            application,
            "/connect",
            connect,
            (upstream.Request, upstream.ConnectRequest),
            upstream.ConnectResponse,
        )
        _replace_post_route(
            application,
            "/update_weights",
            update_weights,
            (upstream.Request, upstream.UpdateWeightsRequest),
            upstream.WeightUpdateResult,
        )
        return application

    return create_app
