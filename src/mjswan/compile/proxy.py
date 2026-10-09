"""Stand-ins for live mjlab objects, shared by the discovery and replay passes.

A proxy subclasses the real object's class, so the ``isinstance`` checks in term bodies
keep passing, and replaces only what a pass needs to see: a sensor's ``.data``, a
command's tensor attributes, ``env.sim``. A raw sim field travels as a
:class:`_FieldProxy`, a tensor subclass whose indexing the tracer watches.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Collection

import torch

from .slot import _TERM_ENV_READS, UnsupportedEnvRead

#: Torch calls that read a tensor's shape, never its values: a proxy treats them as
#: no read at all.
_SHAPE_QUERIES = frozenset({"__get__", "size", "dim", "ndimension", "numel", "__len__"})


def _traces_through(data: Any, name: str, reader_fields: Collection[str]) -> bool:
    """Whether ``data.<name>`` is a property to run against the sim proxy.

    Only a property can be: a plain tensor field has no math to trace, so it stays a
    slot of its own.
    """
    if name in reader_fields:
        return False
    return isinstance(getattr(type(data), name, None), property)


def _with_sim_data(data: Any, sim: Any) -> Any:
    """A shallow copy of a real ``EntityData`` with ``data`` swapped for ``sim``.

    ``EntityData`` is a plain dataclass and ``data`` an ordinary field, so its
    properties run unchanged against the proxy; ``indexing``, ``model`` and the tensor
    fields (``encoder_bias``, ``gravity_vec_w``) stay real and bake as constants.
    """
    replay = copy.copy(data)
    object.__setattr__(replay, "data", sim)
    return replay


def _class_proxy(real: Any, overrides: dict[str, Any]) -> Any:
    """A stand-in for a live mjlab object that still satisfies ``isinstance`` checks.

    Terms assert on concrete classes (``builtin_sensor`` on ``BuiltinSensor``, say),
    so subclass the real object's class and share its ``__dict__``, replacing only
    what ``overrides`` names.
    """
    cls = type(real)
    proxy_cls = type(f"_Proxy{cls.__name__}", (cls,), overrides)
    proxy = object.__new__(proxy_cls)
    proxy.__dict__ = real.__dict__
    return proxy


def _sensor_proxy(real: Any, get_data: Callable[[], Any]) -> Any:
    """A sensor stand-in whose ``.data`` comes from ``get_data``."""
    return _class_proxy(real, {"data": property(lambda _self: get_data())})


def _command_proxy(real: Any, on_tensor: Callable[[str, Any], Any]) -> Any:
    """A command-term stand-in routing every tensor attribute through ``on_tensor``.

    A command's state lives in plain instance attributes, so ``__getattr__`` never
    fires and ``__getattribute__`` is the only hook that sees the read.
    """

    def __getattribute__(self: Any, attr: str) -> Any:  # noqa: N807
        value = object.__getattribute__(self, attr)
        if isinstance(value, torch.Tensor):
            return on_tensor(attr, value)
        return value

    return _class_proxy(real, {"__getattribute__": __getattribute__})


def action_term_window(manager: Any, name: str) -> tuple[int, int]:
    """Where action term *name*'s slice of the policy's action vector starts, and its
    width, as ``ActionManager.process_action`` splits the vector in config order.

    Raises for a name the manager does not hold, as mjlab's ``get_term`` does.
    """
    names = list(manager.active_terms)
    offset = 0
    for term_name, dim in zip(names, manager.action_term_dim, strict=True):
        if term_name == name:
            return offset, int(dim)
        offset += int(dim)
    raise ValueError(
        f"Term read action term {name!r}, which the scene does not define. "
        f"Available: {', '.join(names) if names else '(none)'}."
    )


def _forward_action_attr(manager: Any, name: str) -> Any:
    """``manager.<name>`` for a non-tensor (a width, the term names); raise for a tensor
    the runtime does not hold, which would otherwise bake as a constant."""
    value = getattr(manager, name)
    if isinstance(value, torch.Tensor):
        raise UnsupportedEnvRead(f"action_manager.{name}", _TERM_ENV_READS)
    return value


def _action_term_proxy(real: Any, raw_action: Callable[[], torch.Tensor]) -> Any:
    """An action-term stand-in serving ``raw_action`` from ``raw_action()``; any other
    tensor it holds (``processed_actions``, say) raises, as the manager's do."""

    def __getattribute__(self: Any, attr: str) -> Any:  # noqa: N807
        if attr == "raw_action":
            return raw_action()
        value = object.__getattribute__(self, attr)
        if isinstance(value, torch.Tensor):
            raise UnsupportedEnvRead(
                f"action_manager.get_term(...).{attr}", _TERM_ENV_READS
            )
        return value

    return _class_proxy(real, {"__getattribute__": __getattribute__})


def entity_static(name: str, value: Any) -> Any:
    """An entity attribute that is neither a tensor nor a scalar, as a term may use it.

    Static structure (``indexing``, name lists, methods) passes through; an actuator
    refuses a ``set_*`` call, which would change Python-side state the browser does not
    run, mutating the trace env and capturing nothing.
    """
    if name == "actuators":
        return [_actuator_guard(actuator) for actuator in value]
    return value


def _actuator_guard(real: Any) -> Any:
    def __getattribute__(self: Any, attr: str) -> Any:  # noqa: N807
        value = object.__getattribute__(self, attr)
        if attr.startswith("set_") and callable(value):
            raise ValueError(
                f"Event term calls {type(real).__name__}.{attr}(), which sets actuator "
                "state in Python that the browser does not run, so the change would be "
                "lost. The browser takes PD gains from the policy's action config."
            )
        return value

    return _class_proxy(real, {"__getattribute__": __getattribute__})


def _is_sensor(scene: Any, name: str) -> bool:
    """Whether ``scene[name]`` resolves to a sensor rather than an entity."""
    sensors = getattr(scene, "sensors", None)
    return bool(sensors) and name in sensors


class _FieldProxy(torch.Tensor):
    """A raw sim field with ``__getitem__`` in the tracer's hands; every other torch
    function sees the plain tensor. Never constructed directly: ``as_subclass``."""


def _plain(value: Any) -> Any:
    return value.as_subclass(torch.Tensor) if isinstance(value, _FieldProxy) else value


class _SimStandIn:
    """``env.sim`` for a term: ``.data`` is the pass's sim proxy, and nothing else.

    The rest of ``Simulation`` (``mj_model``, ``forward()``) would act on the real env
    mid-trace, so it raises.
    """

    def __init__(self, data: Any):
        object.__setattr__(self, "data", data)

    def __getattr__(self, name: str) -> Any:
        raise UnsupportedEnvRead(f"sim.{name}", _TERM_ENV_READS)
