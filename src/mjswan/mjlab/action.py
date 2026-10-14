"""mjlab action terms as mjswan's.

Actions stay a fixed, native, non-traced set (ADR 0005 §7), matched by class name. The
regex scale/offset patterns, and the PD gains mjlab keeps on the actuator config, are
resolved against the policy's joint names, or a command term's own, since the browser
looks joints up by exact name.
"""

from __future__ import annotations

import copy
import dataclasses
import re
import reprlib
import warnings
from collections.abc import Mapping
from typing import Any

from ..envs.mdp import actions as _actions_module
from ..envs.mdp.actions.actions import (
    CommandJointPositionActionCfg,
    JointPositionActionCfg,
    JointVelocityActionCfg,
    ReferenceJointPositionActionCfg,
)
from ..managers.action_manager import ActionTermCfg as MjswanActionTermCfg
from .detect import differs, field_default, is_from_mjlab

_ACTION_CLASS_ALIASES: dict[str, str] = {
    # myosuite's muscle action cfg sits outside mjlab's hierarchy; translate it.
    "MyoMuscleActivationActionCfg": "MuscleActivationActionCfg",
}

#: Set by each mjlab action class in ``__post_init__`` (a tendon action's is ``TENDON``),
#: so the class name the mapping goes by already carries it.
_SET_BY_CLASS = frozenset({"transmission_type"})


def _set_fields(cfg: Any) -> dict[str, Any]:
    """*cfg*'s dataclass fields that hold other than their default, as a field without
    one always does. A duck-typed cfg has no fields to walk."""
    if not dataclasses.is_dataclass(cfg):
        return {}
    found: dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name, dataclasses.MISSING)
        default = field_default(f)
        if default is dataclasses.MISSING or differs(value, default):
            found[f.name] = value
    return found


def _listing(fields: Mapping[str, Any]) -> str:
    return ", ".join(f"{name}={reprlib.repr(value)}" for name, value in fields.items())


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
    shares with *term* is copied; a field set on *term* that the class lacks warns, since
    the browser ignores it. ``None`` if no mjswan equivalent exists.
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

    carried = {f.name for f in dataclasses.fields(mjswan_cls)} | _SET_BY_CLASS
    if dropped := {k: v for k, v in _set_fields(term).items() if k not in carried}:
        warnings.warn(
            f"mjlab action type '{type(term).__name__}' sets {_listing(dropped)}, which "
            f"mjswan's {mjswan_cls.__name__} does not have and the browser ignores.",
            category=RuntimeWarning,
            stacklevel=3,
        )

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
    policy's joints or a command term's own, prefixed with the entity name
    (``"robot/left_hip_joint"``). Mutates each term in place.
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


def _pd_actuators(env_cfg: Any | None, term: MjswanActionTermCfg) -> list[Any]:
    entity = _entity_cfg(env_cfg, term)
    return [
        cfg
        for cfg in getattr(getattr(entity, "articulation", None), "actuators", ())
        if any(base.__name__ == _PYTHON_PD_ACTUATOR for base in type(cfg).__mro__)
    ]


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
            term,
            (
                JointPositionActionCfg,
                ReferenceJointPositionActionCfg,
                CommandJointPositionActionCfg,
            ),
        ):
            continue
        if getattr(term, "stiffness", None) is not None or term.damping is not None:
            continue
        actuators = _pd_actuators(env_cfg, term)
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


def _unrun_fields(cfg: Any) -> dict[str, Any]:
    """What an ideal-PD-family *cfg* sets beyond that PD: a subclass's own fields (a
    torque-speed curve, friction) and a command delay."""
    pd_fields = {
        f.name
        for base in type(cfg).__mro__
        if base.__name__ == _PYTHON_PD_ACTUATOR and dataclasses.is_dataclass(base)
        for f in dataclasses.fields(base)
    }
    delayed = getattr(cfg, "delay_max_lag", 0) > 0
    return {
        name: value
        for name, value in _set_fields(cfg).items()
        if name not in pd_fields or (delayed and name.startswith("delay_"))
    }


def warn_unrun_actuator_fields(
    actions: Mapping[str, MjswanActionTermCfg] | None, env_cfg: Any | None
) -> None:
    """Warn once per ideal-PD actuator config of the entities *actions* drive that does
    more than its PD within the effort limit, which is all the browser runs."""
    actuators = {
        id(cfg): cfg
        for term in (actions or {}).values()
        for cfg in _pd_actuators(env_cfg, term)
    }
    for cfg in actuators.values():
        if unrun := _unrun_fields(cfg):
            warnings.warn(
                f"mjlab actuator '{type(cfg).__name__}' on "
                f"{reprlib.repr(cfg.target_names_expr)} sets {_listing(unrun)}, which "
                "the browser does not run: it runs only the PD within the effort limit.",
                category=RuntimeWarning,
                stacklevel=2,
            )


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
    "warn_unrun_actuator_fields",
]
