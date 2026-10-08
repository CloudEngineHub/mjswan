"""Observation terms the runtime evaluates itself, so there is nothing to trace.

Named by function: mjlab's ``generated_commands`` reads a command the browser may own
outright (a UI command), which a trace env has no term for. It becomes a marker entry
naming the command to read.
"""

from __future__ import annotations

from typing import Any, Callable

NATIVE_OBSERVATION_FUNCS: dict[str, str] = {
    "generated_commands": "command",
}


def _native_observation_kind(func: Callable[..., Any]) -> str | None:
    return NATIVE_OBSERVATION_FUNCS.get(getattr(func, "__name__", ""))


def native_observation_entry(
    name: str, func: Callable[..., Any], params: dict[str, Any]
) -> dict[str, Any] | None:
    """The ``native`` marker for an observation the runtime holds, else ``None``.

    The caller adds ``size`` (and, when fusing, the graph ``input`` name) since the two
    paths resolve widths differently.
    """
    kind = _native_observation_kind(func)
    if kind is None:
        return None
    return {"name": name, "native": kind, "command_name": params["command_name"]}
