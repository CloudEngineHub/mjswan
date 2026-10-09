"""Trace a stateful ``CommandTerm`` to ONNX.

A command's hidden state is promoted to explicit graph I/O, so ``_resample_command`` +
``_update_command`` trace as one pure function::

    forward(prev_state..., resample_mask, [reset_mask], rand)
        -> (next_state..., entity_write?)

with the runtime holding ``state`` across frames and owning the resample timer. What a
term's ``reset`` adds to mjlab's ``CommandTerm.reset`` runs between the two, gated by
``reset_mask``, since it happens on an episode reset alone.

mjlab's bodies narrow to the envs a draw selected; :mod:`.gate` keeps those sets whole
and gated, and every draw is traced at its range's low end so each guarded branch is
taken. Before it ships, the graph is run against the term's own body on draws that take
each selection both ways.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from unittest import mock

import numpy as np
import torch
from torch import nn

from .export import (
    _classify_tagged,
    _const_values,
    _export_onnx,
    _prepare_single_env_export,
    _register_consts,
)
from .gate import GatedIndexing, _where
from .record import (
    _WRITE_FIELDS,
    WriteCaptures,
    _EventCaptureEnv,
    _EvRecEntity,
    _flatten_captures,
    entity_write_target,
)
from .replay import _EventReplayEnv, _EvReplayEntity
from .rng import DrawRecorder, ReplayRng
from .slot import _COMMAND_NS, SlotKey, TaggedKey, _slot_input_name

_ENTITY_WRITE_METHODS = {
    "write_joint_state_to_sim": "joint_state",
    "write_root_link_pose_to_sim": "root_pose",
    "write_root_link_velocity_to_sim": "root_velocity",
}


def _entity_attrs(term: Any) -> list[str]:
    """Names of ``term`` attributes that are entities (read state from / write to)."""
    return [
        attr
        for attr, value in vars(term).items()
        if hasattr(value, "data")
        and any(hasattr(type(value), m) for m in _ENTITY_WRITE_METHODS)
    ]


class _RecordCommand:
    """Swap a command's entity attrs + ``_env`` to recording proxies, so its reads are
    logged and its writes captured without mutating the sim.

    Single-entity commands only: all entity attrs are keyed by ``cfg.entity_name``.
    """

    def __init__(
        self, term: Any, entity_attr_names: list[str], entity_name: str | None
    ):
        self.term = term
        self._attrs = entity_attr_names
        self._entity_name = entity_name
        self.log: list[tuple[TaggedKey, Any]] = []
        self.captures: WriteCaptures = {}

    def __enter__(self) -> _RecordCommand:
        self._orig = {a: getattr(self.term, a) for a in self._attrs}
        self._orig_env = getattr(self.term, "_env", None)
        for a in self._attrs:
            setattr(
                self.term,
                a,
                _EvRecEntity(self._orig[a], self._entity_name, self.log, self.captures),
            )
        if self._orig_env is not None:
            self.term._env = _EventCaptureEnv(
                self._orig_env, self.log, self.captures, commands=True
            )
        return self

    def __exit__(self, *exc: object) -> None:
        for a, v in self._orig.items():
            setattr(self.term, a, v)
        if self._orig_env is not None:
            self.term._env = self._orig_env


def _slot_example(key: SlotKey, value: torch.Tensor) -> torch.Tensor:
    """The graph input for a slot. The browser serves another command's state as
    float32, so a flag crosses as 0 or 1 and is cast back inside the graph."""
    return value.float() if key[0] == _COMMAND_NS else value


def _snapshot_state(term: Any) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().clone()
        for k, v in vars(term).items()
        if isinstance(v, torch.Tensor)
    }


def _restore_state(term: Any, snap: dict[str, torch.Tensor]) -> None:
    for k, v in snap.items():
        setattr(term, k, v.clone())


def _gate_state(
    term: Any, fields: list[str], mask: torch.Tensor, before: dict[str, torch.Tensor]
) -> None:
    for f in fields:
        setattr(term, f, _where(mask, getattr(term, f), before[f]))


def _gate_reset_writes(
    captures: WriteCaptures, before: WriteCaptures, mask: torch.Tensor
) -> None:
    """Gate each write the reset made by *mask*: it lands on an episode reset alone."""
    for key, values in captures.items():
        if before.get(key) is values:
            continue
        fields = _WRITE_FIELDS[key[1]]
        own = (
            values[len(fields)] if len(values) > len(fields) else torch.ones(len(mask))
        )
        captures[key] = (*values[: len(fields)], own * mask.to(torch.float32))


def _reset_extra(term: Any) -> Callable[[torch.Tensor], None] | None:
    """What the term's ``reset`` adds to mjlab's ``CommandTerm.reset``, or ``None``.

    The base clears metrics, restarts the timer and resamples, which the graph and the
    runtime already do, so it is stubbed out while the addition runs.
    """
    try:
        from mjlab.managers.command_manager import CommandTerm
    except ImportError:
        return None
    if not isinstance(term, CommandTerm) or type(term).reset is CommandTerm.reset:
        return None

    def extra(env_ids: torch.Tensor) -> None:
        with mock.patch.object(CommandTerm, "reset", lambda _self, _env_ids: {}):
            type(term).reset(term, env_ids)

    return extra


def _step(
    term: Any,
    state_fields: list[str],
    captures: WriteCaptures,
    masks: tuple[torch.Tensor, ...],
    reset_extra: Callable[[torch.Tensor], None] | None,
) -> None:
    """Resample, the reset's addition, update: each pass runs this, so they agree."""
    env_ids = torch.arange(term.num_envs)
    prev = {f: getattr(term, f).clone() for f in state_fields}
    term._resample_command(env_ids)
    _gate_state(term, state_fields, masks[0], prev)
    if reset_extra is not None:
        resampled = {f: getattr(term, f).clone() for f in state_fields}
        before = dict(captures)
        reset_extra(env_ids)
        _gate_state(term, state_fields, masks[1], resampled)
        _gate_reset_writes(captures, before, masks[1])
    term._update_command(None)


def _draw_funcs(term: Any) -> tuple[Callable[..., Any], ...]:
    """The bodies whose module globals a draw is looked up in."""
    reset = getattr(type(term), "reset", None)
    return (term._resample_command, *([reset] if reset is not None else []))


class _CommandModule(nn.Module):
    """Traces a CommandTerm's resample+update as a pure function.

    ``forward(*dynamic_slots, *prev_state, *masks, rand)``: state is injected and read
    back, the resample is gated by ``resample_mask`` (and the reset's addition by
    ``reset_mask``), ``_update_command`` always runs, and any ``entity_write`` is
    captured.

    ``resample_mask`` gates the state fields only: a resample's writes are a fresh draw
    every call and valid only when it ran, which ``OnnxCommand.step`` enforces. The
    reset's writes carry ``reset_mask`` in their gate.
    """

    def __init__(
        self,
        term: Any,
        state_fields: list[str],
        entity_attr_names: list[str],
        entity_name: str | None,
        *,
        dynamic_keys: list[SlotKey],
        dynamic_dtypes: list[torch.dtype],
        tensor_consts: dict[TaggedKey, torch.Tensor],
        scalar_consts: dict[TaggedKey, Any],
        reset_extra: Callable[[torch.Tensor], None] | None = None,
    ):
        super().__init__()
        self._term = term
        self._state_fields = state_fields
        self._entity_attr_names = entity_attr_names
        self._entity_name = entity_name
        self._dynamic_keys = dynamic_keys
        self._dynamic_dtypes = dynamic_dtypes
        self._scalar_consts = scalar_consts
        self._reset_extra = reset_extra
        self._const_buffers = _register_consts(self, tensor_consts)

    def forward(self, *args: torch.Tensor):
        n_dyn = len(self._dynamic_keys)
        n_state = len(self._state_fields)
        dynamic = args[:n_dyn]
        state_inputs = args[n_dyn : n_dyn + n_state]
        masks = args[n_dyn + n_state : -1]
        rand = args[-1]

        served: dict[TaggedKey, Any] = dict(self._scalar_consts)
        served.update(_const_values(self, self._const_buffers))
        for key, dtype, tensor in zip(
            self._dynamic_keys, self._dynamic_dtypes, dynamic
        ):
            # Another command's state is served under its slot key, as in _ReplayEnv.
            served[key if key[0] == _COMMAND_NS else ("data", *key)] = tensor.to(dtype)

        captures: WriteCaptures = {}
        orig = {a: getattr(self._term, a) for a in self._entity_attr_names}
        orig_env = getattr(self._term, "_env", None)
        for a in self._entity_attr_names:
            setattr(self._term, a, _EvReplayEntity(self._entity_name, served, captures))
        if orig_env is not None:
            # `real_env` is the env being swapped out, not the term: `num_envs`
            # forwards to `_env`, so the term would forward to itself.
            self._term._env = _EventReplayEnv(
                served, captures, real_env=orig_env, commands=True
            )
        try:
            for field_name, value in zip(self._state_fields, state_inputs):
                setattr(self._term, field_name, value)
            with ReplayRng(*_draw_funcs(self._term), rand=rand), GatedIndexing():
                _step(
                    self._term, self._state_fields, captures, masks, self._reset_extra
                )
            outputs = [getattr(self._term, f) for f in self._state_fields]
            _, write_tensors = _flatten_captures(captures)
            return tuple(outputs) + tuple(write_tensors)
        finally:
            for a, v in orig.items():
                setattr(self._term, a, v)
            if orig_env is not None:
                self._term._env = orig_env


@dataclass
class CommandExport:
    """The result of tracing one command term body to ONNX."""

    name: str
    onnx_bytes: bytes
    state_fields: list[dict[str, Any]]
    """Per state field: ``{name, shape, dtype}``, written to the manifest entry."""
    command_field: str
    input_slots: list[SlotKey]
    input_names: list[str]
    rand_dim: int
    rand_ranges: list[list[float]]
    """Per-element ``[low, high]`` the runtime draws ``rand`` from."""
    output_names: list[str]
    write_targets: list[dict[str, Any]]
    reference_rand: torch.Tensor
    input_shapes: list[list[int]] = field(default_factory=list)
    """Traced shape of each input slot, parallel to ``input_slots``."""


def trace_command_term(
    term: Any,
    state_fields: list[str],
    *,
    name: str,
    command_field: str,
    opset: int = 17,
) -> CommandExport:
    """Trace a stateful CommandTerm to ONNX.

    Promotes ``state_fields`` to explicit graph I/O and threads randomness through
    ``rand``, then checks the graph against the term's own body (:func:`_verify`).
    """
    entity_attr_names = _entity_attrs(term)
    entity_name = getattr(getattr(term, "cfg", None), "entity_name", None)
    reset_extra = _reset_extra(term)
    snap = _snapshot_state(term)
    state_example = tuple(getattr(term, f).detach().clone() for f in state_fields)
    masks = (torch.ones(term.num_envs, dtype=torch.bool),) * (2 if reset_extra else 1)

    with _RecordCommand(term, entity_attr_names, entity_name) as rec_env:
        # Every draw at its low end passes every `draw <= p` selection, so both passes
        # take the branches a selection guards; the gates make them no-ops when empty.
        with (
            DrawRecorder(*_draw_funcs(term), at_lower_bounds=True) as rec,
            GatedIndexing(),
        ):
            _step(term, state_fields, rec_env.captures, masks, reset_extra)
        log = list(rec_env.log)
        captures = dict(rec_env.captures)
    ref_rand = rec.rand_vector
    _restore_state(term, snap)

    output_write_names, _ = _flatten_captures(captures)
    write_targets = [
        entity_write_target(key, values, entity_name)
        for key, values in captures.items()
    ]

    dynamic, tensor_consts, scalar_consts = _classify_tagged(log)

    dynamic_keys = sorted(dynamic)
    dynamic_dtypes = [dynamic[k].dtype for k in dynamic_keys]
    dyn_names = [_slot_input_name(k) for k in dynamic_keys]
    prev_names = [f"prev_{f}" for f in state_fields]
    mask_names = ["resample_mask", "reset_mask"][: len(masks)]

    example = (
        *(_slot_example(k, dynamic[k]) for k in dynamic_keys),
        *state_example,
        *masks,
        ref_rand,
    )
    input_names = [*dyn_names, *prev_names, *mask_names, "rand"]
    output_names = [f"next_{f}" for f in state_fields] + output_write_names

    module = _CommandModule(
        term,
        state_fields,
        entity_attr_names,
        entity_name,
        dynamic_keys=dynamic_keys,
        dynamic_dtypes=dynamic_dtypes,
        tensor_consts=tensor_consts,
        scalar_consts=scalar_consts,
        reset_extra=reset_extra,
    ).eval()
    _prepare_single_env_export(term.num_envs)
    # `rand` keeps its traced length: it is one flat draw vector, not a batch of rows.
    onnx_bytes = _export_onnx(
        module,
        example,
        input_names=input_names,
        output_names=output_names,
        batch_axis=[*dyn_names, *prev_names, *mask_names, *output_names],
        opset=opset,
    )
    _restore_state(term, snap)

    # Initial values, as `cfg.build(env)` left them. Without them the runtime
    # zero-fills, which starts a counter or a held previous value wrong.
    state_specs = [
        {
            "name": f,
            "shape": list(getattr(term, f).shape),
            "dtype": str(getattr(term, f).dtype).replace("torch.", ""),
            "init": [
                # Plain JSON values; the reader rebuilds the typed array from `dtype`.
                bool(v) if getattr(term, f).dtype == torch.bool else v
                for v in getattr(term, f).detach().reshape(-1).tolist()
            ],
        }
        for f in state_fields
    ]

    export = CommandExport(
        name=name,
        onnx_bytes=onnx_bytes,
        state_fields=state_specs,
        command_field=command_field,
        input_slots=dynamic_keys,
        input_names=dyn_names,
        rand_dim=rec.rand_dim,
        rand_ranges=rec.rand_ranges,
        output_names=output_names,
        write_targets=write_targets,
        reference_rand=ref_rand.detach(),
        input_shapes=[list(dynamic[k].shape) for k in dynamic_keys],
    )
    _verify(export, term, state_fields)
    return export


GraphRunner = Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]


def live_slot_values(term: Any, export: CommandExport) -> dict[str, np.ndarray]:
    """Each dynamic slot's live value off the term's env, by graph input name."""
    attrs = _entity_attrs(term)
    entity: Any = getattr(term, attrs[0]) if attrs else None
    out = {}
    for in_name, (namespace, fld) in zip(export.input_names, export.input_slots):
        if namespace == _COMMAND_NS:
            command, _, attr = fld.partition(".")
            manager = term._env.command_manager
            value = (
                manager.get_command(command)
                if attr == "command"
                else getattr(manager.get_term(command), attr)
            ).float()
        else:
            value = getattr(entity.data, fld)
        out[in_name] = value.detach().cpu().numpy()
    return out


def compare_step(
    term: Any,
    export: CommandExport,
    state_fields: list[str],
    run_graph: GraphRunner,
    rand: torch.Tensor,
    *,
    resample: bool,
    reset: bool,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> tuple[float, str | None]:
    """One step through the graph and through the term's own body, from the term's
    current state; the largest difference, and what disagreed (``None`` if nothing).

    The body runs as mjlab runs it: no gating, and *rand* served through
    :class:`ReplayRng`. Leaves the term in the body's next state.
    """
    reset_extra = _reset_extra(term)
    feeds = {f"prev_{f}": getattr(term, f).detach().cpu().numpy() for f in state_fields}
    feeds.update(live_slot_values(term, export))
    feeds["resample_mask"] = np.full(term.num_envs, resample)
    feeds["reset_mask"] = np.full(term.num_envs, reset)
    feeds["rand"] = rand.detach().cpu().numpy().astype(np.float32)
    outputs = run_graph(feeds)

    env_ids = torch.arange(term.num_envs)
    entity_name = getattr(getattr(term, "cfg", None), "entity_name", None)
    with _RecordCommand(term, _entity_attrs(term), entity_name) as rec:
        with ReplayRng(*_draw_funcs(term), rand=rand):
            if resample:
                term._resample_command(env_ids)
                if reset and reset_extra is not None:
                    reset_extra(env_ids)
            term._update_command(None)
    worst = 0.0

    def differs(label: str, got: np.ndarray, want: Any) -> str | None:
        nonlocal worst
        want = np.asarray(want.detach().cpu().numpy(), dtype=np.float32)
        got = np.asarray(got, dtype=np.float32).reshape(want.shape)
        worst = max(worst, float(np.max(np.abs(got - want))) if want.size else 0.0)
        if np.allclose(got, want, atol=atol, rtol=rtol):
            return None
        return f"{label}: graph {got.tolist()}, mjlab {want.tolist()}"

    for f in state_fields:
        note = differs(f, outputs[f"next_{f}"], getattr(term, f))
        if note:
            return worst, note
    for target in export.write_targets:
        key = (target["entity"], target["kind"])
        gate = outputs[target["gate"]].reshape(-1)[0] if "gate" in target else 1.0
        applied = resample and gate > 0.5
        if applied != (key in rec.captures):
            return worst, (
                f"{target['kind']} write: the graph {'' if applied else 'does not '}"
                f"apply it, mjlab {'does' if key in rec.captures else 'does not'}"
            )
        for output, value in zip(target["outputs"], rec.captures.get(key, ())):
            note = differs(f"{target['kind']} write", outputs[output], value)
            if note:
                return worst, note
    return worst, None


def _probe_draws(
    export: CommandExport, has_reset: bool
) -> list[tuple[torch.Tensor, bool, bool]]:
    """``(rand, resample, reset)`` for :func:`_verify`: every draw at its low end (each
    selection taken), just under its high end (each skipped), then mixed."""
    ranges = torch.tensor(export.rand_ranges, dtype=torch.float32).reshape(-1, 2)
    low, span = ranges[:, 0], ranges[:, 1] - ranges[:, 0]
    generator = torch.Generator().manual_seed(0)
    fractions = [
        torch.zeros(len(low)),
        torch.full((len(low),), 0.999),
        *(torch.rand(len(low), generator=generator) for _ in range(6)),
    ]
    return [
        (low + span * f, i % 4 != 3, has_reset and i % 2 == 0)
        for i, f in enumerate(fractions)
    ]


def reference_runner(export: CommandExport) -> GraphRunner:
    """*export*'s graph, run by ONNX's reference evaluator, which needs no runtime."""
    import onnx
    from onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(onnx.load_from_string(export.onnx_bytes))
    declared = set(evaluator.input_names)

    def run(feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outs = evaluator.run(
            export.output_names, {k: v for k, v in feeds.items() if k in declared}
        )
        return {n: np.asarray(o) for n, o in zip(export.output_names, outs)}

    return run


def _verify(export: CommandExport, term: Any, state_fields: list[str]) -> None:
    """Refuse a graph that disagrees with the term's own body.

    The trace took each guarded branch; one whose body is not a no-op when its guard
    fails would ship a command mjlab never issues, so the graph is run against the
    body on :func:`_probe_draws`, chaining state from step to step.
    """
    run = reference_runner(export)
    snap = _snapshot_state(term)
    try:
        for rand, resample, reset in _probe_draws(
            export, _reset_extra(term) is not None
        ):
            _, note = compare_step(
                term, export, state_fields, run, rand, resample=resample, reset=reset
            )
            if note is not None:
                raise ValueError(
                    f"Command term {export.name!r} traced to a graph that disagrees "
                    f"with its own body ({note}; draws {rand.tolist()}, "
                    f"resample={resample}, reset={reset}). A branch the trace cannot "
                    "follow is the likely cause; supply a trace-friendly body with "
                    "mjswan.register_command(..., CommandBinding(trace_override=...))."
                )
    finally:
        _restore_state(term, snap)
