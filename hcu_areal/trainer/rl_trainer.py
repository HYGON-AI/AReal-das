# SPDX-License-Identifier: Apache-2.0

"""Trainer shutdown diagnostics for the AReaL v2.1.0 close path."""

from __future__ import annotations

import faulthandler
import functools
import threading


MODULE = "areal.trainer.rl_trainer"


def _close_factory(original):
    logger = original.__globals__["logger"]
    perf_tracer = original.__globals__["perf_tracer"]

    @functools.wraps(original)
    def close(self):
        faulthandler.enable(all_threads=True)
        faulthandler.dump_traceback_later(60.0, repeat=True)
        logger.info("[EXIT] trainer.close begin; stack dump every 60 seconds")
        # Construction can fail before the saver, rollout, or other fields
        # exist. Clean up every component that was actually created without
        # replacing the original initialization error with AttributeError.
        partial = getattr(self, "saver", None) is None
        for name, method in (
            ("saver", "finalize"),
            ("_train_rdataset", "close"),
            ("_valid_rdataset", "close"),
            ("data_controller", "destroy"),
            ("stats_logger", "close"),
            ("eval_rollout", "destroy"),
            ("rollout", "destroy"),
            ("teacher", "destroy"),
            ("ref", "destroy"),
            ("critic", "destroy"),
            ("actor", "destroy"),
        ):
            component = getattr(self, name, None)
            if component is None:
                continue
            if name == "actor":
                logger.info("[EXIT] actor.destroy begin")
            try:
                getattr(component, method)()
            except Exception:
                if not partial:
                    raise
                logger.exception("[EXIT] failed to clean up %s", name)
            if name == "actor":
                logger.info("[EXIT] actor.destroy end")
        logger.info("[EXIT] perf tracer save begin")
        perf_tracer.save(force=True)
        logger.info("[EXIT] perf tracer save end")
        for thread in threading.enumerate():
            logger.info(
                "[EXIT] remaining thread: name=%s ident=%s daemon=%s alive=%s",
                thread.name,
                thread.ident,
                thread.daemon,
                thread.is_alive(),
            )
        logger.info("[EXIT] trainer.close returned")

    return close
