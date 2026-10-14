"""Whether an object is mjlab's, and what its config sets, without importing mjlab."""

from __future__ import annotations

import dataclasses
from typing import Any


def is_from_mjlab(obj: Any) -> bool:
    """Whether any class in *obj*'s MRO comes from ``mjlab``: a task's subclass of an
    mjlab config reports its own module, so the leaf class alone is not enough."""
    return any(
        (getattr(klass, "__module__", "") or "").startswith("mjlab")
        for klass in type(obj).__mro__
    )


def field_default(f: dataclasses.Field[Any]) -> Any:
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return dataclasses.MISSING


def differs(value: Any, default: Any) -> bool:
    try:
        return bool(value != default)
    except Exception:  # noqa: BLE001 (an ambiguous comparison is not the default)
        return value is not default


__all__ = ["differs", "field_default", "is_from_mjlab"]
