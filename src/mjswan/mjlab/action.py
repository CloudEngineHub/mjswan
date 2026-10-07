"""mjlab action terms as mjswan's.

Actions stay a fixed, native, non-traced set (ADR 0005 §7), matched by class name. The
regex scale/offset patterns, and the PD gains mjlab keeps on the actuator config, are
resolved against the policy's joint names, since the browser looks joints up by exact
name.
"""

from __future__ import annotations

import copy
import dataclasses
import re
import warnings
from collections.abc import Mapping
from typing import Any

from ..envs.mdp import actions as _actions_module
from ..envs.mdp.actions.actions import (
    JointPositionActionCfg,
    JointVelocityActionCfg,
    ReferenceJointPositionActionCfg,
)
from ..managers.action_manager import ActionTermCfg as MjswanActionTermCfg
from .detect import is_from_mjlab

_ACTION_CLASS_ALIASES: dict[str, str] = {
    # myosuite's muscle action cfg sits outside mjlab's hierarchy; translate it.
    "MyoMuscleActivationActionCfg": "MuscleActivationActionCfg",
}


def _mjswan_action_class(term: Any) -> type[MjswanActionTermCfg] | None:
    class_name = _ACTION_CLASS_ALIASES.get(type(term).__name__, type(term).__name__)
    cls = getattr(_actions_module, class_name, None)
    if isinstance(cls, type) and issubclass(cls, MjswanActionTermCfg):
        return cls
    return None


def _has_mjswan_action(term: Any) -> bool:
    """Whether *term*'s class name maps to an mjswan action, mjlab bases or not."""
    return _mjswan_action_class(term) is not None


def _adapt_action_cfg(term: Any) -> MjswanActionTermCfg | None:
    """Convert a single mjlab ``ActionTermCfg`` to mjswan.

    The class is looked up by name on ``mjswan.envs.mdp.actions`` and every field it
    shares with *term* is copied. ``None`` if no mjswan equivalent exists.
    """
    class_name = _ACTION_CLASS_ALIASES.get(type(term).__name__, type(term).__name__)
    mjswan_cls = _mjswan_action_class(term)

    if mjswan_cls is None:
        warnings.warn(
            f"mjlab action type '{class_name}' has no mjswan equivalent. "
            f"It will be skipped.",
            category=RuntimeWarning,
            stacklevel=3,
        )
        return None

    kwargs: dict[str, Any] = {}
    entity_name = getattr(term, "entity_name", None)
    for f in dataclasses.fields(mjswan_cls):
        if f.name == "unsupported_reason":
            continue
        val = getattr(term, f.name, dataclasses.MISSING)
        if val is not dataclasses.MISSING:
            kwargs[f.name] = val

    # Prefixed with the entity name, as mjlab does, to match policy_joint_names. An
    # alternation is grouped, or the prefix would bind to its first branch only.
    if entity_name and "actuator_names" in kwargs:
        raw = kwargs["actuator_names"]
        if isinstance(raw, (list, tuple)):
            kwargs["actuator_names"] = tuple(
                f"{entity_name}/(?:{n})" if "|" in n else f"{entity_name}/{n}"
                for n in raw
            )

    return mjswan_cls(**kwargs)


def adapt_actions(
    actions: Mapping[str, Any] | None,
) -> Mapping[str, MjswanActionTermCfg] | None:
    """Adapt action configs, converting mjlab types if detected."""
    if actions is None:
        return None
    result: dict[str, MjswanActionTermCfg] = {}
    for key, term in actions.items():
        if term is None:
            continue
        if isinstance(term, MjswanActionTermCfg):
            result[key] = term
        elif is_from_mjlab(term) or _has_mjswan_action(term):
            adapted = _adapt_action_cfg(term)
            if adapted is not None:
                result[key] = adapted
        else:
            # Copied: `resolve_action_scales` rewrites `scale` in place, and this is
            # the object a live mjlab env config holds.
            result[key] = copy.copy(term)
    return result


def resolve_action_scales(
    actions: Mapping[str, MjswanActionTermCfg] | None,
    joint_names: list[str],
) -> None:
    """Expand regex-keyed scale/offset dicts in action configs to literal joint names.

    mjlab keys per-joint values by regex (``{".*_hip_joint": 0.37}``) and the browser
    looks them up by exact name, so patterns are expanded against *joint_names*, the
    policy's joints prefixed with the entity name (``"robot/left_hip_joint"``).
    Mutates each term in place.
    """
    if not actions or not joint_names:
        return

    for term in actions.values():
        scale = getattr(term, "scale", None)
        if isinstance(scale, dict):
            setattr(term, "scale", _expand_patterns(scale, joint_names))
        offset = getattr(term, "offset", None)
        if isinstance(offset, dict):
            setattr(term, "offset", _expand_patterns(offset, joint_names))


def _expand_patterns(value: Any, joint_names: list[str]) -> Any:
    """A ``{pattern: value}`` mapping keyed by the joint names it matches."""
    if not isinstance(value, dict):
        return value
    resolved: dict[str, float] = {}
    for pattern, val in value.items():
        try:
            regex = re.compile(pattern)
        except re.error:
            resolved[pattern] = val
            continue
        for joint_name in joint_names:
            bare = joint_name.split("/")[-1] if "/" in joint_name else joint_name
            if regex.fullmatch(bare) or regex.fullmatch(joint_name):
                resolved[joint_name] = val
    return resolved if resolved else value


#: This family computes its PD in torch; builtin position actuators bake the gains
#: into the model and need nothing here.
_PYTHON_PD_ACTUATOR = "IdealPdActuatorCfg"


def _entity_cfg(env_cfg: Any | None, term: MjswanActionTermCfg) -> Any | None:
    entities = getattr(getattr(env_cfg, "scene", None), "entities", None) or {}
    return entities.get(getattr(term, "entity_name", "robot"))


def resolve_pd_gains(
    actions: Mapping[str, MjswanActionTermCfg] | None,
    joint_names: list[str],
    env_cfg: Any | None,
) -> None:
    """Fill a term's ``stiffness``/``damping`` from the entity's actuators, in place.

    The browser runs the PD for a ``biastype=none`` actuator itself and reads the gains
    off the *action* term, where mjlab keeps them on the *actuator* config. A velocity
    term takes ``damping`` only, and a term that sets either gain is left alone.
    """
    if not actions or not joint_names:
        return

    for term in actions.values():
        velocity = isinstance(term, JointVelocityActionCfg)
        if not velocity and not isinstance(
            term, (JointPositionActionCfg, ReferenceJointPositionActionCfg)
        ):
            continue
        if getattr(term, "stiffness", None) is not None or term.damping is not None:
            continue
        entity = _entity_cfg(env_cfg, term)
        actuators = [
            cfg
            for cfg in getattr(getattr(entity, "articulation", None), "actuators", ())
            if any(base.__name__ == _PYTHON_PD_ACTUATOR for base in type(cfg).__mro__)
        ]
        gains = {
            field: _expand_patterns(
                {
                    pattern: float(getattr(cfg, field))
                    for cfg in actuators
                    for pattern in cfg.target_names_expr
                },
                joint_names,
            )
            for field in ("stiffness", "damping")
        }
        if not gains["stiffness"]:
            continue
        if not velocity:
            term.stiffness = gains["stiffness"]
        term.damping = gains["damping"]


def resolve_default_joint_vel(
    actions: Mapping[str, MjswanActionTermCfg] | None,
    joint_names: list[str],
    env_cfg: Any | None,
) -> None:
    """Make a velocity term's ``offset`` the entity's default joint velocity.

    mjlab's ``use_default_offset`` replaces the offset with ``init_state.joint_vel``,
    which the browser has no copy of, so it is baked in here by exact joint name.
    """
    if not actions or not joint_names:
        return

    for term in actions.values():
        if not isinstance(term, JointVelocityActionCfg) or not term.use_default_offset:
            continue
        init_state = getattr(_entity_cfg(env_cfg, term), "init_state", None)
        joint_vel = getattr(init_state, "joint_vel", None)
        if not isinstance(joint_vel, dict):
            continue
        targets = [re.compile(pattern) for pattern in term.actuator_names]
        offset: dict[str, float] = {}
        for name in joint_names:
            if not any(target.fullmatch(name) for target in targets):
                continue
            bare = name.split("/")[-1]
            # First match wins, as in mjlab's `resolve_expr`.
            value = next((v for p, v in joint_vel.items() if re.match(p, bare)), 0.0)
            if value:
                offset[name] = float(value)
        term.offset = offset or 0.0


__all__ = [
    "adapt_actions",
    "resolve_action_scales",
    "resolve_default_joint_vel",
    "resolve_pd_gains",
]
