"""Trace a task-defined ``ActionTerm``'s ``process_actions`` and ``reset`` to ONNX.

ADR 0009. The term's state, every tensor attribute either body writes, is promoted to
explicit graph I/O, as a stateful command's is (ADR 0005 §3)::

    process(raw, slots..., prev_<f>..., [rand]) -> (next_<f>..., processed, [processed_from])
    reset(slots..., prev_<f>..., [rand])        -> (next_<f>...)

The browser holds the state, runs ``process`` once per control step after the policy and
``reset`` in the reset chain, and applies ``processed`` natively each substep: what the
term's ``apply_actions`` does is not traced but named, by its mjlab base class or its
binding. Before they ship, both graphs run against the term's own bodies over random
actions, resets and episode clocks.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Literal

import numpy as np
import torch
from torch import nn

from .command import _restore_state, _snapshot_state
from .export import (
    _classify_slots,
    _const_values,
    _export_onnx,
    _input_rows,
    _narrow_inputs,
    _narrow_slots,
    _prepare_single_env_export,
    _register_consts,
)
from .gate import GatedIndexing, MaskedRows
from .record import _RecordingEnv
from .replay import _ReplayEnv
from .rng import DrawRecorder, ReplayRng
from .slot import READER_FIELDS, SlotKey, _slot_input_name, read_slot

Body = Literal["reset", "process"]

#: The bodies a draw may sit in, up the term's class hierarchy.
_BODIES = ("process_actions", "reset")

#: Plain values a body could keep as Python state, which a graph cannot carry.
_SCALARS = (bool, int, float, str, type(None))

#: State dtypes the runtime holds natively; any other crosses as float32.
_NATIVE_STATE_DTYPES = (torch.float32, torch.bool)


class UntraceableAction(ValueError):
    """An action term whose body the tracer cannot turn into a graph that matches it."""


def _bodies(term: Any) -> tuple[Callable[..., Any], ...]:
    """Each class's own ``process_actions`` and ``reset``: a ``super()`` call runs in
    the base's module, whose globals a draw is looked up in too."""
    return tuple(
        body
        for cls in type(term).__mro__
        for name in _BODIES
        if (body := vars(cls).get(name)) is not None
    )


def _tensor_versions(term: Any) -> dict[str, tuple[int, int]]:
    """Each tensor attribute's identity and in-place version: a write changes one."""
    return {
        k: (id(v), v._version)
        for k, v in vars(term).items()
        if isinstance(v, torch.Tensor)
    }


def _scalars(term: Any) -> dict[str, Any]:
    return {k: v for k, v in vars(term).items() if isinstance(v, _SCALARS)}


class _GuardedEntity:
    """``self._entity`` for a body: data reads go to *data_source*, other attributes to
    the real entity, and a call that would change the sim is refused."""

    def __init__(self, data_source: Any, real: Any, term_name: str):
        object.__setattr__(self, "_source", data_source)
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_term", term_name)

    def __getattr__(self, name: str) -> Any:
        if name == "data":
            return self._source.data
        if name.startswith(("write_", "set_")) or name in ("reset", "update"):
            raise UntraceableAction(
                f"Action term {self._term!r} calls entity.{name}() in process_actions "
                "or reset. Targets reach the sim through apply_actions, which the "
                "browser runs natively; a body that writes the sim cannot be traced."
            )
        return getattr(self._real, name)


class _Swap:
    """Point a term's ``_env`` and its entity attributes at stand-ins for one pass."""

    def __init__(self, term: Any, env: Any, entity: Any, entity_attrs: list[str]):
        self._term = term
        self._env = env
        self._entity = entity
        self._attrs = entity_attrs

    def __enter__(self) -> None:
        self._saved = {a: getattr(self._term, a) for a in ["_env", *self._attrs]}
        self._term._env = self._env
        for a in self._attrs:
            setattr(self._term, a, self._entity)

    def __exit__(self, *exc: object) -> None:
        for a, v in self._saved.items():
            setattr(self._term, a, v)


def _entity_attrs(term: Any, name: str) -> tuple[str, list[str]]:
    """The entity the term drives, and every attribute holding it.

    Refuses an attribute holding another entity or a manager's term: the term cached it
    at init, so its reads would bypass the tracer and bake as constants.
    """
    env = term._env
    entity_name = term.cfg.entity_name
    entity = env.scene[entity_name]
    attrs: list[str] = []
    foreign: list[str] = []
    for attr, value in vars(term).items():
        if value is entity:
            attrs.append(attr)
        elif attr not in ("_env", "cfg") and _is_live_object(value):
            foreign.append(attr)
    if foreign:
        raise UntraceableAction(
            f"Action term {name!r} keeps {foreign}, a live mjlab object other than its "
            f"entity {entity_name!r}, from its __init__. Its reads would bypass the "
            "tracer and freeze at their build-time values; read it through self._env "
            "in the body instead, via mjswan.register_action(..., ActionBinding("
            "trace_override=...))."
        )
    return entity_name, attrs


def _is_live_object(value: Any) -> bool:
    """An entity, sensor or manager term: something whose state changes under the term."""
    if isinstance(value, (torch.Tensor, *_SCALARS, list, tuple, dict)):
        return False
    try:
        from mjlab.managers.manager_base import ManagerTermBase
    except ImportError:
        ManagerTermBase = ()  # noqa: N806
    if isinstance(value, ManagerTermBase):
        return True
    return hasattr(value, "data") and (
        hasattr(value, "write_joint_state_to_sim") or hasattr(value, "update")
    )


@dataclass
class _Pass:
    """What one body read, wrote and drew during discovery."""

    log: list[tuple[SlotKey, Any]]
    recorder: _RecordingEnv
    written: set[str]
    draws: DrawRecorder


def _discover(
    term: Any,
    name: str,
    body: Body,
    raw: torch.Tensor,
    *,
    entity_name: str,
    entity_attrs: list[str],
    readers: Collection[str],
) -> _Pass:
    """Run *body* against recording proxies, every draw at its range's low end."""
    recorder = _RecordingEnv(term._env, readers)
    entity = _GuardedEntity(
        recorder.scene[entity_name], term._env.scene[entity_name], name
    )
    tensors, scalars = _tensor_versions(term), _scalars(term)
    with (
        _Swap(term, recorder, entity, entity_attrs),
        DrawRecorder(*_bodies(term), at_lower_bounds=True) as draws,
        GatedIndexing(),
        MaskedRows(),
    ):
        _run_body(term, body, raw)
    after = _tensor_versions(term)
    if created := sorted(set(after) - set(tensors)):
        raise UntraceableAction(
            f"Action term {name!r} creates {created} in {_body_name(body)}. Create its "
            "state in __init__, so the build knows its starting value."
        )
    changed = sorted(
        k
        for k, v in _scalars(term).items()
        if k not in entity_attrs and scalars.get(k, v) != v
    )
    if changed:
        raise UntraceableAction(
            f"Action term {name!r} changes {changed} in {_body_name(body)}: Python "
            "state a graph cannot carry, so the browser would keep its build-time "
            "value. Keep it in a tensor instead, via mjswan.register_action(..., "
            "ActionBinding(trace_override=...))."
        )
    written = {k for k, v in after.items() if tensors.get(k) != v}
    return _Pass(list(recorder._log), recorder, written, draws)  # noqa: SLF001


def _body_name(body: Body) -> str:
    return "process_actions" if body == "process" else "reset"


def _run_body(term: Any, body: Body, raw: torch.Tensor | None) -> None:
    if body == "reset":
        term.reset(torch.arange(term.num_envs))
    else:
        term.process_actions(raw)


def _processed(term: Any) -> torch.Tensor:
    """The term's processed action, which mjlab's ``BaseAction`` keeps in
    ``_processed_actions`` and ``apply_actions`` writes to the sim."""
    value = getattr(term, "_processed_actions", None)
    if not isinstance(value, torch.Tensor):
        raise UntraceableAction(
            f"{type(term).__name__} keeps no `_processed_actions`, which the browser "
            "applies: derive it from mjlab's BaseAction."
        )
    return value


def _state_example(value: torch.Tensor) -> torch.Tensor:
    """A state field as the runtime feeds it: float32, or bool."""
    return value if value.dtype in _NATIVE_STATE_DTYPES else value.float()


class _ActionModule(nn.Module):
    """One body of an action term as a pure function of its slots and state."""

    def __init__(
        self,
        term: Any,
        body: Body,
        *,
        state_fields: list[str],
        state_dtypes: list[torch.dtype],
        entity_name: str,
        entity_attrs: list[str],
        dynamic_keys: list[SlotKey],
        constants: dict[SlotKey, torch.Tensor],
        sensors: dict[str, Any],
        commands: dict[str, Any],
        sim_rows: Any,
        readers: Collection[str],
        with_rand: bool,
        ramp_from: bool,
    ):
        super().__init__()
        self._term = term
        self._body: Body = body
        self._state_fields = state_fields
        self._state_dtypes = state_dtypes
        self._entity_name = entity_name
        self._entity_attrs = entity_attrs
        self._dynamic_keys = dynamic_keys
        self._sensors = sensors
        self._commands = commands
        self._sim_rows = sim_rows
        self._readers = readers
        self._with_rand = with_rand
        self._ramp_from = ramp_from
        self._real_env = term._env
        self._const_buffers = _register_consts(self, constants)

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        args_list = list(args)
        raw = args_list.pop(0) if self._body == "process" else None
        dynamic = args_list[: len(self._dynamic_keys)]
        state = args_list[len(self._dynamic_keys) :][: len(self._state_fields)]
        rand = args_list[-1] if self._with_rand else None

        slots: dict[SlotKey, torch.Tensor] = dict(zip(self._dynamic_keys, dynamic))
        _narrow_slots(slots, self._sim_rows)
        slots.update(_const_values(self, self._const_buffers))
        env = _ReplayEnv(
            slots,
            self._sensors,
            self._commands,
            real_env=self._real_env,
            reader_fields=self._readers,
        )
        entity = _GuardedEntity(
            env.scene[self._entity_name],
            self._real_env.scene[self._entity_name],
            self._entity_name,
        )
        with _Swap(self._term, env, entity, self._entity_attrs):
            for name, dtype, value in zip(
                self._state_fields, self._state_dtypes, state
            ):
                setattr(self._term, name, value.to(dtype))
            draws = (
                ReplayRng(*_bodies(self._term), rand=rand)
                if rand is not None
                else contextlib.nullcontext()
            )
            with draws, GatedIndexing(), MaskedRows():
                start = (
                    _processed(self._term).clone()
                    if self._body == "process" and self._ramp_from
                    else None
                )
                _run_body(self._term, self._body, raw)
            outputs = [
                _state_example(getattr(self._term, f)).clone()
                for f in self._state_fields
            ]
            if self._body == "process":
                outputs.append(_processed(self._term).float().clone())
                if start is not None:
                    outputs.append(start.float())
        return tuple(outputs)


@dataclass
class ActionGraph:
    """One traced body: its graph, the slots it reads and the draws it takes."""

    onnx_bytes: bytes
    input_slots: list[SlotKey]
    input_names: list[str]
    output_names: list[str]
    rand_dim: int
    rand_ranges: list[list[float]]
    input_shapes: list[list[int]] = field(default_factory=list)
    input_rows: list[list[int] | None] = field(default_factory=list)


@dataclass
class ActionExport:
    """A traced action term: its graphs, its state and the joints it drives."""

    name: str
    process: ActionGraph
    reset: ActionGraph | None
    state_fields: list[dict[str, Any]]
    """Per state field: ``{name, shape, dtype, init}``, as ``cfg.build(env)`` left it."""
    target_names: list[str]
    """The joints, entity-prefixed, in the order ``processed`` lists them."""
    ramp_from: bool


def _export_body(
    term: Any,
    name: str,
    body: Body,
    found: _Pass,
    *,
    state_fields: list[str],
    entity_name: str,
    entity_attrs: list[str],
    readers: Collection[str],
    raw: torch.Tensor,
    ramp_from: bool,
    opset: int,
) -> ActionGraph:
    dynamic: dict[SlotKey, torch.Tensor] = {}
    constants: dict[SlotKey, torch.Tensor] = {}
    _classify_slots(found.log, dynamic, constants)
    sim_rows = _narrow_inputs(dynamic, [found.recorder._sim])  # noqa: SLF001
    dynamic_keys = sorted(dynamic)
    input_names = [_slot_input_name(k) for k in dynamic_keys]
    state_values = [getattr(term, f) for f in state_fields]
    with_rand = found.draws.rand_dim > 0

    module = _ActionModule(
        term,
        body,
        state_fields=state_fields,
        state_dtypes=[v.dtype for v in state_values],
        entity_name=entity_name,
        entity_attrs=entity_attrs,
        dynamic_keys=dynamic_keys,
        constants=constants,
        sensors=dict(found.recorder._sensors),  # noqa: SLF001
        commands=dict(found.recorder._commands),  # noqa: SLF001
        sim_rows=sim_rows,
        readers=readers,
        with_rand=with_rand,
        ramp_from=ramp_from,
    ).eval()

    prev_names = [f"prev_{f}" for f in state_fields]
    output_names = [f"next_{f}" for f in state_fields]
    example: list[torch.Tensor] = []
    names: list[str] = []
    if body == "process":
        example.append(raw)
        names.append("raw")
        output_names += ["processed", *(["processed_from"] if ramp_from else [])]
    example += [dynamic[k] for k in dynamic_keys]
    example += [_state_example(v) for v in state_values]
    names += input_names + prev_names
    if with_rand:
        example.append(found.draws.rand_vector)
        names.append("rand")

    snapshot = _snapshot_state(term)
    try:
        onnx_bytes = _export_onnx(
            module,
            tuple(example),
            input_names=names,
            output_names=output_names,
            # `rand` keeps its traced length: one flat draw vector, not rows.
            batch_axis=[n for n in names if n != "rand"] + output_names,
            opset=opset,
        )
    finally:
        _restore_state(term, snapshot)
    return ActionGraph(
        onnx_bytes=onnx_bytes,
        input_slots=dynamic_keys,
        input_names=input_names,
        output_names=output_names,
        rand_dim=found.draws.rand_dim,
        rand_ranges=found.draws.rand_ranges,
        input_shapes=[list(dynamic[k].shape) for k in dynamic_keys],
        input_rows=_input_rows(dynamic_keys, sim_rows),
    )


def trace_action_term(
    term: Any,
    *,
    name: str,
    ramp_from: bool = False,
    opset: int = 17,
    reader_fields: Collection[str] | None = None,
) -> ActionExport:
    """Trace a built mjlab action term's ``process_actions`` and ``reset`` to ONNX.

    *term* comes from ``cfg.build(env)`` on a single-env trace env and is left as it was
    found. *ramp_from* adds the ``processed_from`` output an interpolating apply starts
    from. Raises :class:`UntraceableAction` for a body the browser could not reproduce.
    """
    _prepare_single_env_export(term.num_envs)
    readers = READER_FIELDS if reader_fields is None else frozenset(reader_fields)
    entity_name, entity_attrs = _entity_attrs(term, name)
    targets = [f"{entity_name}/{n}" for n in term.target_names]
    if term.action_dim != len(targets):
        raise UntraceableAction(
            f"Action term {name!r} takes {term.action_dim} action value(s) but drives "
            f"{len(targets)} joint(s). The browser feeds a traced term one action per "
            "joint it drives."
        )

    generator = torch.Generator().manual_seed(0)
    raw = torch.rand((term.num_envs, term.action_dim), generator=generator) * 2 - 1
    snapshot = _snapshot_state(term)
    try:
        passes = {
            body: _discover(
                term,
                name,
                body,
                raw,
                entity_name=entity_name,
                entity_attrs=entity_attrs,
                readers=readers,
            )
            for body in ("reset", "process")
        }
    finally:
        _restore_state(term, snapshot)
    state_fields = sorted(passes["reset"].written | passes["process"].written)

    graphs = {
        body: _export_body(
            term,
            name,
            body,
            passes[body],
            state_fields=state_fields,
            entity_name=entity_name,
            entity_attrs=entity_attrs,
            readers=readers,
            raw=raw,
            ramp_from=ramp_from,
            opset=opset,
        )
        for body in ("reset", "process")
        # A reset that writes no state is mjlab's no-op for this term: no graph.
        if body == "process" or passes["reset"].written
    }
    export = ActionExport(
        name=name,
        process=graphs["process"],
        reset=graphs.get("reset"),
        state_fields=[_state_spec(term, f) for f in state_fields],
        target_names=targets,
        ramp_from=ramp_from,
    )
    _verify(export, term)
    return export


def _state_spec(term: Any, name: str) -> dict[str, Any]:
    """``{name, shape, dtype, init}``, with ``init`` as plain JSON values."""
    value = _state_example(getattr(term, name)).detach()
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype).replace("torch.", ""),
        "init": [
            bool(v) if value.dtype == torch.bool else v
            for v in value.reshape(-1).tolist()
        ],
    }


GraphRunner = Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]


def reference_runner(graph: ActionGraph) -> GraphRunner:
    """*graph* run by ONNX's reference evaluator, which needs no runtime."""
    import onnx
    from onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(onnx.load_from_string(graph.onnx_bytes))
    declared = set(evaluator.input_names)

    def run(feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outs = evaluator.run(
            graph.output_names, {k: v for k, v in feeds.items() if k in declared}
        )
        return {n: np.asarray(o) for n, o in zip(graph.output_names, outs)}

    return run


def compare_body(
    term: Any,
    export: ActionExport,
    body: Body,
    run_graph: GraphRunner,
    *,
    raw: torch.Tensor | None = None,
    rand: torch.Tensor | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> str | None:
    """One body through its graph and through mjlab, from the term's current state;
    what disagreed, or ``None``. Leaves the term in mjlab's next state."""
    graph = export.process if body == "process" else export.reset
    assert graph is not None
    env = term._env
    feeds: dict[str, np.ndarray] = {
        f"prev_{spec['name']}": _state_example(getattr(term, spec["name"]))
        .detach()
        .cpu()
        .numpy()
        for spec in export.state_fields
    }
    for key, input_name, rows in zip(
        graph.input_slots, graph.input_names, graph.input_rows
    ):
        feeds[input_name] = read_slot(env, key, rows).detach().cpu().numpy()
    if raw is not None:
        feeds["raw"] = raw.detach().cpu().numpy().astype(np.float32)
    if rand is not None:
        feeds["rand"] = rand.detach().cpu().numpy().astype(np.float32)
    start = _processed(term).clone() if body == "process" else None
    outputs = run_graph(feeds)

    draws = (
        ReplayRng(*_bodies(term), rand=rand)
        if rand is not None
        else contextlib.nullcontext()
    )
    with draws:
        _run_body(term, body, raw)

    def differs(label: str, got: np.ndarray, want: torch.Tensor) -> str | None:
        expected = _state_example(want).detach().cpu().numpy().astype(np.float32)
        actual = np.asarray(got, dtype=np.float32).reshape(expected.shape)
        if np.allclose(actual, expected, atol=atol, rtol=rtol):
            return None
        return f"{label}: graph {actual.tolist()}, mjlab {expected.tolist()}"

    for spec in export.state_fields:
        note = differs(
            spec["name"], outputs[f"next_{spec['name']}"], getattr(term, spec["name"])
        )
        if note:
            return note
    if body == "process":
        if note := differs("processed", outputs["processed"], _processed(term)):
            return note
        if export.ramp_from and start is not None:
            return differs("processed_from", outputs["processed_from"], start)
    return None


#: Episode clocks the check walks through: both sides of any hold up to 20 s at 50 Hz.
_EPISODE_STEPS = (0, 1, 2, 3, 10, 49, 50, 51, 100, 1000)


def _probe(graph: ActionGraph, generator: torch.Generator) -> torch.Tensor | None:
    if graph.rand_dim == 0:
        return None
    ranges = torch.tensor(graph.rand_ranges, dtype=torch.float32).reshape(-1, 2)
    low, high = ranges[:, 0], ranges[:, 1]
    return low + (high - low) * torch.rand(len(low), generator=generator)


def _verify(export: ActionExport, term: Any) -> None:
    """Refuse a graph that disagrees with the term's own body.

    Runs reset, then a process step at each of :data:`_EPISODE_STEPS`, then a reset and
    a few more steps, chaining state through mjlab's bodies, with random actions in
    [-1.5, 1.5] and draws inside their ranges. A branch the trace took one way and the
    body takes another (a Python ``if`` on a tensor, a counter) shows up here.
    """
    env = term._env
    process = reference_runner(export.process)
    reset = reference_runner(export.reset) if export.reset is not None else None
    generator = torch.Generator().manual_seed(1)
    snapshot = _snapshot_state(term)
    found = getattr(env, "episode_length_buf", None)
    clock = found if isinstance(found, torch.Tensor) else None
    saved_clock = clock.clone() if clock is not None else None

    def step(episode: int) -> str | None:
        if clock is not None:
            clock[:] = episode
        raw = (
            torch.rand((term.num_envs, term.action_dim), generator=generator) * 3 - 1.5
        )
        rand = _probe(export.process, generator)
        return compare_body(term, export, "process", process, raw=raw, rand=rand)

    def restart() -> str | None:
        if reset is None or export.reset is None:
            _run_body(term, "reset", None)
            return None
        rand = _probe(export.reset, generator)
        return compare_body(term, export, "reset", reset, rand=rand)

    try:
        plan: list[tuple[str, Callable[[], str | None]]] = [("reset", restart)]
        plan += [(f"process at step {s}", lambda s=s: step(s)) for s in _EPISODE_STEPS]
        plan += [("reset", restart)]
        plan += [(f"process at step {s}", lambda s=s: step(s)) for s in (0, 1, 2)]
        for label, run in plan:
            if note := run():
                raise UntraceableAction(
                    f"Action term {export.name!r} traced to a graph that disagrees with "
                    f"its own body ({label}: {note}). A branch the trace cannot follow, "
                    "a Python `if` on a tensor or a counter, is the likely cause; supply "
                    "a trace-friendly body via mjswan.register_action(..., "
                    "ActionBinding(trace_override=...))."
                )
    finally:
        _restore_state(term, snapshot)
        if clock is not None and saved_clock is not None:
            clock[:] = saved_clock


__all__ = [
    "ActionExport",
    "ActionGraph",
    "UntraceableAction",
    "compare_body",
    "reference_runner",
    "trace_action_term",
]
