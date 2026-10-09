"""Build-time RNG spy/replay, so a traced graph can take ``rand`` as an explicit input.

Nothing here is the runtime's seeded PRNG. Recording mjlab's real draws and replaying
those exact values into the graph is what lets the parity harness compare the two
without them diverging on randomness alone.

The runtime draws each ``rand`` element uniformly in its ``[low, high]``, so a sampler
whose output is not uniform is carried as the uniforms behind it, and the graph maps
them: ``log(x)`` for a log-uniform draw, a Box-Muller pair for a Gaussian one, and the
value itself, floored in the graph, for an integer one.

The spy patches the module globals a draw is looked up in (the term function's own,
and mjlab's DR distributions'), since mjlab binds the name at import time and patching
the source module would not reach it. ``torch.randint`` and the in-place samplers
(``Tensor.uniform_``, ``Tensor.normal_``) are torch functions, caught by a
:class:`~torch.overrides.TorchFunctionMode` instead.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
from torch.overrides import TorchFunctionMode

#: mjlab RNG helpers a term may import, by how ``rand`` carries their draws.
_SAMPLERS = {
    "sample_uniform": "uniform",
    "sample_log_uniform": "log_uniform",
    "sample_gaussian": "gaussian",
}
#: ``torch.Tensor`` methods that draw in place, likewise.
_IN_PLACE_SAMPLERS = {
    torch.Tensor.uniform_: "uniform",
    torch.Tensor.normal_: "gaussian",
}

Make = Callable[[str, Callable[..., Any]], Callable[..., Any]]


def _rng_namespaces(funcs: tuple[Callable[..., Any], ...]) -> list[dict[str, Any]]:
    """The globals a term's draws are looked up in: its functions' own, and mjlab's DR
    distributions', whose lambdas call the samplers by name."""
    spaces: list[dict[str, Any]] = []
    try:
        from mjlab.envs.mdp.dr import _types

        candidates = [*(getattr(f, "__globals__", {}) for f in funcs), vars(_types)]
    except ImportError:
        candidates = [getattr(f, "__globals__", {}) for f in funcs]
    for space in candidates:
        if all(space is not seen for seen in spaces):
            spaces.append(space)
    return spaces


def _in_place(make: Make, kind: str, real: Callable[..., Any]) -> Callable[..., Any]:
    """``Tensor.<sampler>_`` as the stand-in *make* builds for a free function: called
    as ``(lower, upper, size)``, its draw lands in the tensor."""

    def method(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        lower = float(kwargs.pop("from", kwargs.pop("mean", args[0] if args else 0.0)))
        upper = float(
            kwargs.pop("to", kwargs.pop("std", args[1] if len(args) > 1 else 1))
        )
        draw = make(kind, lambda *_: real(tensor, lower, upper, **kwargs))
        out = draw(lower, upper, tuple(tensor.shape))
        return out if out is tensor else tensor.copy_(out)

    return method


class _TorchDraws(TorchFunctionMode):
    """Hands ``torch.randint`` and the in-place samplers to *make*'s stand-ins."""

    def __init__(self, make: Make):
        super().__init__()
        self._make = make

    def __torch_function__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        if func is torch.randint:
            return self._make("randint", func)(*args, **kwargs)
        kind = _IN_PLACE_SAMPLERS.get(func)
        if kind is not None:
            return _in_place(self._make, kind, func)(*args, **kwargs)
        return func(*args, **kwargs)


class _Patches:
    """Swaps the samplers for stand-ins until :meth:`remove`."""

    def __init__(self, *funcs: Callable[..., Any]):
        self._spaces = _rng_namespaces(funcs)
        self._saved: list[tuple[dict[str, Any], str, Any]] = []
        self._mode: _TorchDraws | None = None

    def install(self, make: Make) -> None:
        for space in self._spaces:
            for name, kind in _SAMPLERS.items():
                real = space.get(name)
                if real is not None:
                    self._saved.append((space, name, real))
                    space[name] = make(kind, real)
        self._mode = _TorchDraws(make)
        self._mode.__enter__()

    def remove(self) -> None:
        for space, name, real in reversed(self._saved):
            space[name] = real
        self._saved.clear()
        if self._mode is not None:
            self._mode.__exit__(None, None, None)
            self._mode = None


def _randint_args(*args: Any, **kwargs: Any) -> tuple[int, int, Any]:
    """``(low, high, size)`` from either of ``torch.randint``'s signatures."""
    positional = list(args)
    size = kwargs["size"] if "size" in kwargs else positional.pop()
    if "high" in kwargs:
        high = kwargs["high"]
        low = kwargs.get("low", positional[0] if positional else 0)
    elif len(positional) >= 2:
        low, high = positional[:2]
    else:
        low, high = kwargs.get("low", 0), positional[0]
    return int(low), int(high), size


def _sampler_args(args: Any, kwargs: Any) -> tuple[Any, Any, Any]:
    """``(lower, upper, size)`` of a sampler call, ``mean``/``std`` for a Gaussian."""
    lower = kwargs.get("lower", kwargs.get("mean", args[0] if args else 0.0))
    upper = kwargs.get("upper", kwargs.get("std", args[1] if len(args) > 1 else 1.0))
    size = kwargs.get("size", args[2] if len(args) > 2 else None)
    return lower, upper, size


def _element_bounds(shape: torch.Size, *args: Any, **kwargs: Any) -> torch.Tensor:
    """``(numel, 2)`` of the ``[low, high]`` behind each element of one draw.

    Bounds are scalars or broadcast against ``size``, so they are broadcast the same way
    here and flattened in the draw's own element order.
    """
    lower, upper, _size = _sampler_args(args, kwargs)
    columns = [
        torch.broadcast_to(torch.as_tensor(bound, dtype=torch.float32).cpu(), shape)
        for bound in (lower, upper)
    ]
    return torch.stack([c.reshape(-1) for c in columns], dim=1)


def _at_lower_bound(
    kind: str, out: torch.Tensor, args: Any, kwargs: Any
) -> torch.Tensor:
    """*out* as the draw every ``rand`` element at its lower bound gives: the low end
    of a uniform, log-uniform or integer range, a Gaussian's mean."""
    if kind == "randint":
        low, high, _size = _randint_args(*args, **kwargs)
        bounds = _element_bounds(out.shape, low, high)
    else:
        bounds = _element_bounds(out.shape, *args, **kwargs)
    return bounds[:, 0].reshape(out.shape).to(out.dtype)


class DrawRecorder:
    """Records the values a term's RNG calls return, in call order.

    Wraps a single term invocation on the live env: the helpers still return mjlab's
    real draw, so the reference rollout is unaffected. With *at_lower_bounds* every draw
    is its range's low end instead, which a ``draw <= p`` selection always passes.
    """

    def __init__(self, *funcs: Callable[..., Any], at_lower_bounds: bool = False):
        self._patches = _Patches(*funcs)
        self._at_lower_bounds = at_lower_bounds
        self._draws: list[torch.Tensor] = []
        self._bounds: list[torch.Tensor] = []

    def __enter__(self) -> DrawRecorder:
        self._patches.install(self._make_spy)
        return self

    def __exit__(self, *exc: object) -> None:
        self._patches.remove()

    def _make_spy(self, kind: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def spy(*args: Any, **kwargs: Any) -> Any:
            out = real(*args, **kwargs)
            if isinstance(out, torch.Tensor):
                if self._at_lower_bounds:
                    out = _at_lower_bound(kind, out, args, kwargs)
                self._record(kind, out, args, kwargs)
            return out

        return spy

    def _record(self, kind: str, out: torch.Tensor, args: Any, kwargs: Any) -> None:
        values = out.detach().cpu().to(torch.float32).reshape(-1)
        if kind == "randint":
            low, high, _size = _randint_args(*args, **kwargs)
            self._append(values, _element_bounds(out.shape, low, high))
        elif kind == "uniform":
            self._append(values, _element_bounds(out.shape, *args, **kwargs))
        elif kind == "log_uniform":
            bounds = _element_bounds(out.shape, *args, **kwargs)
            self._append(values.log(), bounds.log())
        else:
            # The uniforms behind a Gaussian draw: `|z|` sets u1, its sign u2.
            mean, std = _element_bounds(out.shape, *args, **kwargs).unbind(dim=1)
            z = torch.where(std > 0, (values - mean) / std.clamp_min(1e-30), 0.0)
            unit = _element_bounds(out.shape, 0.0, 1.0)
            self._append(1.0 - torch.exp(-0.5 * z * z), unit)
            self._append(torch.where(z < 0, 0.5, 0.0), unit)

    def _append(self, values: torch.Tensor, bounds: torch.Tensor) -> None:
        self._draws.append(values.clone())
        self._bounds.append(bounds)

    @property
    def rand_dim(self) -> int:
        return int(sum(d.numel() for d in self._draws))

    @property
    def rand_vector(self) -> torch.Tensor:
        """Flat ``rand`` tensor: every draw concatenated in call order."""
        if not self._draws:
            return torch.zeros(0)
        return torch.cat(self._draws)

    @property
    def rand_ranges(self) -> list[list[float]]:
        """Per-element ``[low, high]`` of the flat ``rand`` vector, in draw order.

        The graph takes each draw as an input and so remembers no bounds, making these
        the only record of them: without them the runtime would draw [0, 1) and turn an
        empty ``pose_range`` into a metre of teleport per reset.
        """
        if not self._bounds:
            return []
        stacked = torch.cat(self._bounds)
        return [[float(low), float(high)] for low, high in stacked.tolist()]


class ReplayRng:
    """Serves recorded draws back to a term as it runs, in call order.

    Installed in place of the term's RNG helpers during tracing, which is what turns
    ``rand`` into an explicit graph input. Each draw maps its slice of ``rand`` the way
    :class:`DrawRecorder` stored it.
    """

    def __init__(
        self,
        func: Callable[..., Any] | tuple[Callable[..., Any], ...],
        rand: torch.Tensor,
    ):
        self._patches = _Patches(*(func if isinstance(func, tuple) else (func,)))
        self._rand = rand.reshape(-1)
        self._offset = 0

    def __enter__(self) -> ReplayRng:
        self._patches.install(self._make_consumer)
        return self

    def __exit__(self, *exc: object) -> None:
        self._patches.remove()

    def _take(self, n: int) -> torch.Tensor:
        values = self._rand[self._offset : self._offset + n]
        self._offset += n
        return values

    def _make_consumer(
        self, kind: str, _real: Callable[..., Any]
    ) -> Callable[..., Any]:
        if kind == "randint":

            def randint(*args: Any, **kwargs: Any) -> torch.Tensor:
                low, high, size = _randint_args(*args, **kwargs)
                n = int(torch.Size(size).numel())
                picks = torch.floor(self._take(n)).clamp(low, high - 1)
                return picks.to(torch.long).reshape(size)

            return randint

        def consume(*args: Any, **kwargs: Any) -> torch.Tensor:
            lower, upper, size = _sampler_args(args, kwargs)
            if isinstance(size, int):
                size = (size,)
            if kind == "gaussian":
                mean = torch.as_tensor(lower, dtype=torch.float32)
                std = torch.as_tensor(upper, dtype=torch.float32)
                # mjlab's `torch.normal(mean, std)` takes their shape over `size`.
                if not isinstance(lower, float):
                    size = torch.broadcast_shapes(mean.shape, std.shape)
                n = int(torch.Size(size).numel())
                u1, u2 = self._take(n), self._take(n)
                z = torch.sqrt(-2.0 * torch.log1p(-u1)) * torch.cos(2.0 * math.pi * u2)
                return mean + std * z.reshape(size)
            n = int(torch.Size(size).numel())
            if kind == "uniform":
                return self._take(n).reshape(size)
            return torch.exp(self._take(n)).reshape(size)

        return consume
