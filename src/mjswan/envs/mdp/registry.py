"""What an author's ``register_observation`` / ``register_termination`` /
``register_event`` put in place of a term's function."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve(
    func: Any, term_name: str | None, registry: Mapping[str, Any], binding: type
) -> Any:
    """*func*, or its registered override: by function name, else by term name, which
    also covers a closure, registrable only by its dict key."""
    if isinstance(func, binding):
        return func
    name = getattr(func, "__name__", None)
    if name and name in registry:
        return registry[name]
    if term_name and term_name in registry:
        return registry[term_name]
    return func
