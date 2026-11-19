# SPDX-License-Identifier: Apache-2.0

"""V2 worker uses the same explicit engine import boundary as the RPC worker."""

from hcu_areal.infra.rpc.guard.engine_blueprint import (
    _engine_import_factory as _engine_import_factory,
)
