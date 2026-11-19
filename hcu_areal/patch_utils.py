# SPDX-License-Identifier: Apache-2.0

"""Megatron-LM-das-style runtime patch registration, application and rollback.

The feature registration stage is import-light. A target module is patched
*after* it has been imported, allowing Ray/SGLang child interpreters to
install the feature without importing HCU libraries at interpreter startup.
"""

from __future__ import annotations
import inspect
import os
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Any


@dataclass(frozen=True)
class Patch:
    target_path: str
    module_name: str
    new_func_or_cls: Callable
    patch_id: str
    apply_wrapper: bool = False
    compose: bool = False

    @property
    def owner_and_attr(self):
        if not self.target_path.startswith(self.module_name + "."):
            raise ValueError(f"Target is not within module: {self.target_path}")
        rest = self.target_path[len(self.module_name) + 1 :]
        parts = rest.split(".")
        if any(not name.isidentifier() for name in parts):
            raise ValueError(f"Invalid patch target: {self.target_path}")
        return parts[:-1], parts[-1]


class AReaLPatchesManager:
    """Register, apply and remove patches (analogous to MegatronPatchesManager)."""

    def __init__(self):
        self._registered: dict[str, Patch] = {}
        self._applied: list[tuple[object, str, Any, Any, str]] = []
        self._cleanups: list[Callable] = []

    def add_cleanup(self, callback: Callable):
        """Register rollback for state installed while applying a wrapper."""
        self._cleanups.append(callback)

    @property
    def module_names(self) -> frozenset[str]:
        return frozenset(p.module_name for p in self._registered.values())

    @property
    def patch_ids(self) -> tuple[str, ...]:
        return tuple(self._registered)

    def register_patch(
        self,
        orig_func_or_cls_name: str,
        new_func_or_cls: Callable,
        *,
        module_name: str,
        patch_id: str,
        apply_wrapper: bool = False,
        compose: bool = False,
    ):
        if patch_id in self._registered:
            raise ValueError(f"Patch already registered: {patch_id}")
        if compose and not apply_wrapper:
            raise ValueError("Composed patches must wrap the previous implementation")
        if not compose and any(
            p.target_path == orig_func_or_cls_name for p in self._registered.values()
        ):
            raise ValueError(f"Conflicting target: {orig_func_or_cls_name}")
        if not callable(new_func_or_cls):
            raise TypeError(f"Patch replacement is not callable: {patch_id}")
        entry = Patch(
            orig_func_or_cls_name,
            module_name,
            new_func_or_cls,
            patch_id,
            apply_wrapper,
            compose,
        )
        entry.owner_and_attr  # validate without importing original module
        self._registered[patch_id] = entry

    def register_cls_funcs(
        self, module_name: str, cls_name: str, functions: list[Callable]
    ):
        """Megatron-style convenience method for direct class-method replacement."""
        for fn in functions:
            self.register_patch(
                f"{module_name}.{cls_name}.{fn.__name__}",
                fn,
                module_name=module_name,
                patch_id=f"{cls_name}.{fn.__name__}",
            )

    def apply_module(self, module: ModuleType) -> int:
        applied = 0
        for entry in self._registered.values():
            if entry.module_name != module.__name__:
                continue
            parents, name = entry.owner_and_attr
            owner: object = module
            for child in parents:
                if not hasattr(owner, child):
                    raise RuntimeError(f"Missing patch parent: {entry.target_path}")
                owner = getattr(owner, child)
            try:
                descriptor = inspect.getattr_static(owner, name)
            except AttributeError as exc:
                raise RuntimeError(
                    f"Missing patch target: {entry.target_path}"
                ) from exc
            kind = (
                type(descriptor)
                if isinstance(descriptor, (staticmethod, classmethod))
                else None
            )
            original = descriptor.__func__ if kind is not None else descriptor
            marker = getattr(original, "__hcu_areal_patch_ids__", frozenset())
            if entry.patch_id in marker:
                continue
            replacement = (
                entry.new_func_or_cls(original)
                if entry.apply_wrapper
                else entry.new_func_or_cls
            )
            if not callable(replacement):
                raise TypeError(f"Invalid replacement for: {entry.target_path}")
            # Mark the raw function, not the descriptor, so repeated install is safe.
            try:
                replacement.__hcu_areal_patch_ids__ = frozenset(marker) | {
                    entry.patch_id
                }
            except Exception as exc:
                raise TypeError(
                    f"Replacement does not support patch marker: {entry.target_path}"
                ) from exc
            wrapped = kind(replacement) if kind is not None else replacement
            setattr(owner, name, wrapped)
            self._applied.append((owner, name, descriptor, wrapped, entry.patch_id))
            applied += 1
            print(
                f"[hcu_areal pid={os.getpid()}] applied {entry.patch_id}",
                file=sys.stderr,
                flush=True,
            )
        return applied

    def remove_patches(self):
        """LIFO rollback; safe in tests, not during live training."""
        for owner, name, original, wrapped, patch_id in reversed(self._applied):
            if inspect.getattr_static(owner, name) is wrapped:
                setattr(owner, name, original)
        self._applied.clear()
        for cleanup in reversed(self._cleanups):
            cleanup()
        self._cleanups.clear()
