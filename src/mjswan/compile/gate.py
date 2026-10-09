"""Env sets a draw selects, kept whole and gated, so a body narrowing to them traces.

mjlab narrows a write to the envs a draw selected (``env_ids[mask]``,
``mask.nonzero()``) and skips an empty set (``if len(ids) > 0``), but a trace would
freeze the set its example took. Under :class:`GatedIndexing` such a set keeps every
env and carries the mask: a read through it reads every env, a write through it is
``where(mask, new, old)``, and an entity write through it carries the mask to the
browser as its gate. Its ``len`` is the env count, so an emptiness guard always takes
its branch, which the gate turns into a no-op for an empty set.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_leaves, tree_map


class GatedIds(torch.Tensor):
    """Every env id, and the mask of those the body selected (see :func:`gated`)."""

    gate: torch.Tensor
    column: bool
    """Still ``nonzero()``'s ``(n, 1)`` shape, until the body flattens it."""

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        # Outside GatedIndexing it is a plain tensor; nothing would carry the gate.
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **(kwargs or {}))


def gated(ids: torch.Tensor, gate: torch.Tensor, *, column: bool = False) -> GatedIds:
    out = ids.as_subclass(GatedIds)
    out.gate = gate
    out.column = column
    return out


def _plain(value: Any) -> Any:
    return value.as_subclass(torch.Tensor) if isinstance(value, GatedIds) else value


def _unsupported(use: str) -> ValueError:
    return ValueError(
        f"A term body {use} an env set a draw selected. The tracer keeps such a set "
        "whole and gates every read and write through it, which covers indexing a "
        "tensor's first axis with it and passing it as `env_ids`; register a "
        "trace-friendly override for anything else."
    )


def _is_mask_over(ids: torch.Tensor, index: Any) -> bool:
    return (
        isinstance(index, torch.Tensor)
        and not isinstance(index, GatedIds)
        and index.dtype == torch.bool
        and index.dim() == 1
        and ids.dim() == 1
        and len(index) == len(ids)
    )


def _is_ids(tensor: torch.Tensor) -> bool:
    return not tensor.is_floating_point() and tensor.dtype != torch.bool


def _split(index: Any) -> tuple[Any, torch.Tensor | None]:
    """*index* with a :class:`GatedIds` on the env axis made plain, and its gate."""
    first = index[0] if isinstance(index, tuple) and index else index
    rest = index[1:] if isinstance(index, tuple) else ()
    if any(isinstance(i, GatedIds) for i in rest):
        raise _unsupported("indexed an axis other than the first with")
    if not isinstance(first, GatedIds):
        return index, None
    if first.column:
        raise _unsupported("indexed with nonzero()'s (n, 1) result of")
    plain = _plain(first)
    return ((plain, *rest) if isinstance(index, tuple) else plain), first.gate


def _where(gate: torch.Tensor, new: Any, old: torch.Tensor) -> torch.Tensor:
    """``new`` where *gate* selects the env, else ``old``; *gate* runs down axis 0."""
    mask = gate.reshape(gate.shape[0], *([1] * (old.dim() - 1)))
    new = torch.as_tensor(new, dtype=old.dtype, device=old.device).expand_as(old)
    if old.dtype == torch.bool:
        # ONNX Runtime's Where has no bool kernel.
        return torch.where(mask, new.long(), old.long()).bool()
    return torch.where(mask, new, old)


#: What a body may ask of a gated set besides indexing with it: its size.
_SHAPE_QUERIES = frozenset({"__len__", "__get__", "size", "dim", "ndimension", "numel"})
_FLATTENS = frozenset({"flatten", "squeeze", "view", "reshape"})


class GatedIndexing(TorchFunctionMode):
    """Turns draw-selected env sets into :class:`GatedIds` for the ``with`` block."""

    def __torch_function__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        name = getattr(func, "__name__", "")
        if func is torch.Tensor.__getitem__:
            base, index = args
            if isinstance(base, GatedIds):
                if base.column or not _is_mask_over(base, index):
                    raise _unsupported("indexed by anything but a mask")
                return gated(_plain(base), base.gate & index)
            if _is_mask_over(base, index) and _is_ids(base):
                return gated(base, index)
            plain, gate = _split(index)
            if gate is not None:
                return func(base, plain)
        elif func is torch.Tensor.__setitem__:
            base, index, value = args
            plain, gate = _split(index)
            if gate is not None:
                return func(base, plain, _where(gate, _plain(value), base[plain]))
        elif func in (torch.Tensor.nonzero, torch.nonzero):
            mask = args[0]
            if mask.dtype == torch.bool and mask.dim() == 1:
                ids = torch.arange(len(mask), device=mask.device)
                if kwargs.get("as_tuple"):
                    return (gated(ids, mask),)
                return gated(ids, mask, column=True)
        elif name in _FLATTENS and isinstance(args[0], GatedIds):
            flat = func(*tree_map(_plain, args), **kwargs)
            if flat.dim() != 1:
                raise _unsupported(f"reshaped to {tuple(flat.shape)}")
            return gated(flat, args[0].gate)
        if name not in _SHAPE_QUERIES and any(
            isinstance(leaf, GatedIds) for leaf in tree_leaves((args, kwargs))
        ):
            raise _unsupported(f"passed to {name or func}()")
        return func(*args, **kwargs)


__all__ = ["GatedIds", "GatedIndexing", "gated"]
