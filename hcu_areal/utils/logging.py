# SPDX-License-Identifier: Apache-2.0

"""Preserve Python logger identities while configuring AReaL logging."""

from functools import wraps


MODULE = "areal.utils.logging"


def _get_logger_factory(original):
    upstream = original.__globals__

    @wraps(original)
    def get_logger(name=None, type_=None, level=upstream["LOGLEVEL"]):
        # AReaL's original implementation replaces Logger.manager on every
        # call, orphaning loggers held by torch.compile and other libraries.
        # Keep AReaL's configuration and file-handler behavior without that reset.
        logging = upstream["logging"]
        config = upstream["log_config"]
        for logger_config in config["loggers"].values():
            if logger_config.get("handlers"):
                logger_config.setdefault("propagate", False)
        logging.config.dictConfig(config)
        if name is None:
            name = "plain"
        if type_ is None:
            type_ = "plain"
        assert type_ in ("plain", "benchmark", "colored", "system")
        if name not in config["loggers"]:
            config["loggers"][name] = {
                "handlers": [f"{type_}Handler"],
                "level": level,
                "propagate": False,
            }
            logging.config.dictConfig(config)
        logger = logging.getLogger(name)
        with upstream["_loggers_lock"]:
            upstream["_created_loggers"][name] = logger
            for handler in upstream["_file_handlers"]:
                if handler not in logger.handlers:
                    logger.addHandler(handler)
        return logger

    return get_logger
