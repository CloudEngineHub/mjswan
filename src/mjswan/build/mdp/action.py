"""Action terms: nothing to trace, only the authored config to merge over."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any

from ...envs.mdp.actions import CommandJointPositionActionCfg

if TYPE_CHECKING:
    from ...managers.action_manager import ActionTermCfg


def serialize_actions(
    actions: Mapping[str, ActionTermCfg],
    authored: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    commands: Collection[str] = (),
    widths: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """The MDP entry's ``actions`` block, keyed by term name.

    Each term's ``to_dict()`` is merged field-wise over what the ``config_path`` sidecar
    authored for the same term, where a motor robot's PD gains live, so a scene can
    tweak the offset without restating them.

    Raises ``ValueError`` if a command term's command is missing or the wrong width.
    """
    for name, cfg in actions.items():
        if not isinstance(cfg, CommandJointPositionActionCfg):
            continue
        if cfg.command_name not in commands:
            raise ValueError(
                f"Action term {name!r} reads command {cfg.command_name!r}, which the "
                f"MDP does not have (its commands: {sorted(commands)}). Add it to the "
                "policy's `commands`."
            )
        width = (widths or {}).get(cfg.command_name)
        if width is not None and width != len(cfg.actuator_names):
            raise ValueError(
                f"Action term {name!r} drives {len(cfg.actuator_names)} joint(s), but "
                f"command {cfg.command_name!r} is {width} wide: one column per joint."
            )
    authored = authored or {}
    return {
        name: {**authored.get(name, {}), **cfg.to_dict()}
        for name, cfg in actions.items()
    }


__all__ = ["serialize_actions"]
