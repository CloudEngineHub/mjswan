"""Record mjlab's viser GUI as control-panel descriptors.

Running mjlab's own ``create_gui`` (a command term's) and ``create_scene_gui`` (the
viewer's Scene section) against a recording stand-in makes mjlab's declaration the
control panel's only definition, so its labels and ranges cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


@dataclass
class _Handle:
    """One recorded control, doubling as the handle ``create_gui`` keeps.

    mjlab stores handles to read ``.value`` later and its ``on_update`` callbacks
    assign ``.min``/``.max``; plain attributes cover both. Callbacks are never
    replayed — they only re-derive what the recording already holds.
    """

    kind: str
    label: str
    value: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    icon: str | None = None
    hint: str | None = None

    def on_update(self, fn: Any) -> Any:
        return fn

    def on_click(self, fn: Any) -> Any:
        return fn


@dataclass
class _Folder:
    """Nests what is recorded inside it under one node of ``_GuiRecorder.tree``."""

    label: str
    recorder: _GuiRecorder

    def __enter__(self) -> _Folder:
        node: dict[str, Any] = {"type": "folder", "label": self.label, "children": []}
        self.recorder._level.append(node)
        self.recorder._stack.append(node["children"])
        return self

    def __exit__(self, *_: Any) -> None:
        self.recorder._stack.pop()


@dataclass
class _GuiRecorder:
    """``server.gui``, recording in call order.

    Only the ``add_*`` mjlab's command terms use — an unknown one raises
    ``AttributeError`` rather than silently dropping a control the viewer shows.
    """

    recorded: list[_Handle] = field(default_factory=list)
    """Every control in call order, folders flattened away."""
    tree: list[dict[str, Any]] = field(default_factory=list)
    """The same controls under the folders they were declared in."""
    _stack: list[list[dict[str, Any]]] = field(default_factory=list)

    @property
    def _level(self) -> list[dict[str, Any]]:
        return self._stack[-1] if self._stack else self.tree

    def _record(self, handle: _Handle) -> _Handle:
        self.recorded.append(handle)
        node = {
            k: v
            for k, v in {
                "type": handle.kind,
                "label": handle.label,
                "default": handle.value,
                "min": handle.min,
                "max": handle.max,
                "step": handle.step,
                "icon": handle.icon,
                "hint": handle.hint,
            }.items()
            if v is not None
        }
        self._level.append(node)
        return handle

    def add_folder(self, label: str, **_: Any) -> _Folder:
        return _Folder(label, self)

    def add_slot(self, name: str) -> None:
        """Mark where the browser fills in controls only it knows."""
        self._level.append({"type": "slot", "name": name})

    def add_checkbox(
        self, label: str, initial_value: bool = False, hint: str | None = None, **_: Any
    ) -> _Handle:
        return self._record(
            _Handle("checkbox", label, value=bool(initial_value), hint=hint)
        )

    def add_slider(
        self,
        label: str,
        min: float | None = None,
        max: float | None = None,
        step: float | None = None,
        initial_value: float | None = None,
        *,
        hint: str | None = None,
        **_: Any,
    ) -> _Handle:
        return self._record(
            _Handle(
                "slider",
                label,
                value=initial_value,
                min=min,
                max=max,
                step=step,
                hint=hint,
            )
        )

    def add_button(self, label: str, icon: Any = None, **_: Any) -> _Handle:
        # viser's `Icon` members are plain tabler names (`Icon.SQUARE_X` == "square-x").
        return self._record(
            _Handle("button", label, icon=str(icon) if icon is not None else None)
        )


@dataclass
class _ServerRecorder:
    gui: _GuiRecorder = field(default_factory=_GuiRecorder)

    def on_client_connect(self, fn: Any) -> Any:
        return fn


def _slug(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def to_ui_descriptor(handles: list[_Handle]) -> dict[str, Any] | None:
    """Recorded controls -> a ``commands.<term>.ui`` descriptor, ``None`` if empty.

    mjlab identifies handles by variable reference, so the conventions here are
    structural rather than label-based:

    - The first checkbox is named ``enabled``, which is what ``OnnxCommand.isUiEnabled``
      looks for.
    - A one-sided slider is a "Max <label>" companion rescaling the next axis, never an
      axis itself — an axis straddles zero.
    - Order is the contract: axis sliders map onto the command vector positionally.
    """
    inputs: list[dict[str, Any]] = []
    enable_name: str | None = None
    companion: dict[str, Any] | None = None

    for handle in handles:
        if handle.kind == "checkbox":
            name = "enabled" if enable_name is None else _slug(handle.label)
            enable_name = enable_name or name
            inputs.append(
                {
                    "type": "checkbox",
                    "name": name,
                    "label": handle.label,
                    "default": bool(handle.value),
                }
            )
        elif handle.kind == "slider":
            if handle.min is not None and handle.min >= 0:
                companion = {
                    "min": handle.min,
                    "max": handle.max,
                    "step": handle.step,
                    "default": handle.value,
                }
                continue
            entry: dict[str, Any] = {
                "type": "slider",
                "name": _slug(handle.label),
                "label": handle.label,
                "min": handle.min,
                "max": handle.max,
                "step": handle.step,
                "default": handle.value,
            }
            if enable_name is not None:
                entry["enabled_when"] = enable_name
            if companion is not None:
                # No `label`: the browser synthesizes `Max <label>`, matching mjlab.
                entry["adjustable_range"] = companion
                companion = None
            inputs.append(entry)
        elif handle.kind == "button":
            entry = {
                "type": "button",
                "name": _slug(handle.label),
                "label": handle.label,
            }
            if handle.icon is not None:
                entry["icon"] = handle.icon
            inputs.append(entry)

    return {"inputs": inputs} if inputs else None


def record_gui(term: Any, name: str) -> dict[str, Any] | None:
    """A built term's ``create_gui`` recorded as a UI descriptor.

    ``None`` when it declares no controls — ``create_gui`` is a base-class no-op, so an
    empty recording is the only signal that a term does not override it.
    """
    server = _ServerRecorder()
    term.create_gui(name, server, lambda: 0)
    return to_ui_descriptor(server.gui.recorded)


DEBUG_VIS_SLOT = "debug_vis"
"""Where the Debug Viz folder lists its drawings, which depend on the loaded policy."""


def record_scene_gui() -> list[dict[str, Any]] | None:
    """mjlab's play viewer's Scene section as a tree, or ``None`` without mjlab.

    It runs on a bare scene holding only the state ``create_scene_gui`` reads, Debug Viz
    on as ``ViserPlayViewer`` sets it. Its per-drawing checkboxes become a slot.
    """
    try:
        import numpy as np
        from mjlab.viewer.viser.scene import MjlabViserScene
    except ImportError:
        return None

    server = _ServerRecorder()
    # `Any`: it holds stand-ins, not the server and model its annotations name.
    scene: Any = object.__new__(MjlabViserScene)
    scene.server = server
    scene.num_envs = 1
    scene.env_idx = 0
    scene.camera_tracking_enabled = True
    scene.show_all_envs = False
    scene.debug_visualization_enabled = True
    # Read only to place the camera, which the browser's viewer config already does.
    scene.mj_model = SimpleNamespace(
        stat=SimpleNamespace(center=np.zeros(3), extent=1.0)
    )
    with server.gui.add_folder("Scene"):
        scene.create_scene_gui(
            debug_viz_extra_gui=lambda: server.gui.add_slot(DEBUG_VIS_SLOT)
        )
    return server.gui.tree


__all__ = ["DEBUG_VIS_SLOT", "record_gui", "record_scene_gui", "to_ui_descriptor"]
