"""The Debug Viz drawings mjlab makes beside the command terms: sensors and rewards.

mjlab draws them from the live env every frame (``update_visualizers``), so the build
restates what to draw as data, read off the trace env.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

from .sensor import raycast_sensor_descriptor


def _body_name(env: Any, body_id: int) -> str:
    return env.sim.mj_model.body(int(body_id)).name


def _upright_entry(func: Any, env: Any) -> dict[str, Any]:
    """mjlab's ``upright`` reward: terrain normal and body up, from its own params."""
    asset_cfg = func._asset_cfg
    entity = env.scene[asset_cfg.name]
    indexing = entity.indexing
    body_ids = asset_cfg.body_ids
    body = (
        _body_name(env, indexing.body_ids[body_ids[0]])
        if isinstance(body_ids, list) and body_ids
        else None
    )
    sensors = func._terrain_sensor_names
    return {
        "kind": "upright",
        "root_body": _body_name(env, indexing.root_body_id),
        "body": body,
        "terrain_sensors": list(sensors) if sensors is not None else None,
    }


def _reward_entry(name: str, func: Any, env: Any) -> dict[str, Any] | None:
    try:
        from mjlab.tasks.velocity.mdp.rewards import upright
    except ImportError:
        upright = None
    if upright is not None and isinstance(func, upright):
        return _upright_entry(func, env)
    warnings.warn(
        f"Reward term {name!r} ({type(func).__name__}) has a debug drawing mjswan does "
        "not know, so the browser shows nothing where mjlab's viewer draws.",
        category=RuntimeWarning,
        stacklevel=3,
    )
    return None


def debug_vis_entry(env: Any) -> dict[str, Any] | None:
    """The sensors and reward terms mjlab's Debug Viz lists, or None if there are none.

    A sensor is listed as mjlab's viewer lists it: a raycast sensor with ``debug_vis``.
    One a reward drawing reads is shipped too, unlisted.
    """
    if env is None:
        return None
    reward_manager = getattr(env, "reward_manager", None)
    rewards: dict[str, Any] = {}
    if reward_manager is not None:
        for name, func in reward_manager.get_visualizable_terms():
            entry = _reward_entry(name, func, env)
            if entry is not None:
                rewards[name] = entry
    read = {s for r in rewards.values() for s in r.get("terrain_sensors") or ()}

    sensors: dict[str, Any] = {}
    for name, sensor in (getattr(env.scene, "sensors", None) or {}).items():
        descriptor = raycast_sensor_descriptor(env, name)
        if descriptor is None:
            continue
        listed = bool(sensor.cfg.debug_vis)
        if not listed and name not in read:
            continue
        sensors[name] = {
            **descriptor,
            "debug_vis": listed,
            "viz": dataclasses.asdict(sensor.cfg.viz),
        }

    if not sensors and not rewards:
        return None
    return {
        # mjlab sizes the sensor drawing by it (`DebugVisualizer.meansize`).
        "meansize": float(env.sim.mj_model.stat.meansize),
        "sensors": sensors,
        "rewards": rewards,
    }


__all__ = ["debug_vis_entry"]
