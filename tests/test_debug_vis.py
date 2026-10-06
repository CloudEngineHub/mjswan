"""The sensor and reward drawings mjlab's Debug Viz lists, as the build ships them.

Layer: L1 against stand-in envs; the `slow` tests pin it against mjlab's own tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import mjswan  # noqa: F401  (sets MUJOCO_GL before anything imports mujoco's renderer)
from mjswan.build.mdp import debug_vis_entry


@dataclass
class _Viz:
    hit_color: tuple = (0.0, 1.0, 0.0, 0.8)
    show_rays: bool = False


class _Tensor:
    def __init__(self, rows):
        self.rows = rows

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.rows


def _raycast(debug_vis: bool):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            debug_vis=debug_vis,
            viz=_Viz(),
            ray_alignment="yaw",
            max_distance=5.0,
            exclude_parent_body=True,
            include_geom_groups=(0,),
        ),
        _local_offsets=_Tensor([[0.0, 0.0, 0.0]]),
        _local_directions=_Tensor([[0.0, 0.0, -1.0]]),
        _frame_infos=[("body", 1, None)],
    )


def _env(sensors, rewards=()):
    names = ["world", "robot/trunk", "robot/torso"]
    named = lambda i: SimpleNamespace(name=names[i])  # noqa: E731
    model = SimpleNamespace(
        body=named, site=named, geom=named, stat=SimpleNamespace(meansize=0.03)
    )
    entity = SimpleNamespace(indexing=SimpleNamespace(root_body_id=1, body_ids=[1, 2]))
    scene = type("Scene", (), {"sensors": sensors, "__getitem__": lambda _, k: entity})
    return SimpleNamespace(
        sim=SimpleNamespace(mj_model=model),
        scene=scene(),
        reward_manager=SimpleNamespace(get_visualizable_terms=lambda: list(rewards)),
    )


def test_lists_a_raycast_sensor_with_its_viz_and_meansize():
    entry = debug_vis_entry(_env({"scan": _raycast(True)}))
    assert entry is not None
    assert entry["meansize"] == 0.03
    scan = entry["sensors"]["scan"]
    assert scan["debug_vis"] is True
    assert scan["viz"] == {"hit_color": (0.0, 1.0, 0.0, 0.8), "show_rays": False}
    assert scan["frames"] == [{"type": "body", "name": "robot/trunk"}]


def test_nothing_to_draw_ships_nothing():
    assert debug_vis_entry(_env({"scan": _raycast(False)})) is None
    assert debug_vis_entry(None) is None


def test_upright_ships_its_bodies_and_the_sensors_it_reads():
    pytest.importorskip("mjlab")
    from mjlab.tasks.velocity.mdp.rewards import upright

    term = upright.__new__(upright)
    term._asset_cfg = SimpleNamespace(name="robot", body_ids=[1])
    term._terrain_sensor_names = ("scan",)
    entry = debug_vis_entry(_env({"scan": _raycast(False)}, [("upright", term)]))
    assert entry is not None
    assert entry["rewards"]["upright"] == {
        "kind": "upright",
        "root_body": "robot/trunk",
        "body": "robot/torso",
        "terrain_sensors": ["scan"],
    }
    # Read by the reward, so shipped, but not listed: its own debug_vis is off.
    assert entry["sensors"]["scan"]["debug_vis"] is False


def test_an_unknown_reward_drawing_warns():
    class lean_reward:
        def debug_vis(self, visualizer):
            pass

    with pytest.warns(RuntimeWarning, match="lean_reward"):
        entry = debug_vis_entry(_env({}, [("lean", lean_reward())]))
    assert entry is None


@pytest.mark.slow
@pytest.mark.mjlab
@pytest.mark.parametrize(
    ("task", "sensors", "terrain_sensors"),
    [
        (
            "Mjlab-Velocity-Rough-Unitree-Go1",
            {"terrain_scan", "foot_height_scan"},
            ["terrain_scan"],
        ),
        ("Mjlab-Velocity-Flat-Unitree-G1", {"foot_height_scan"}, None),
    ],
)
def test_mjlabs_velocity_tasks(task, sensors, terrain_sensors):
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    from mjswan.mjlab.env import build_mjlab_env

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    env = build_mjlab_env(cfg)
    try:
        entry = debug_vis_entry(env)
    finally:
        env.close()
    assert entry is not None
    assert {n for n, s in entry["sensors"].items() if s["debug_vis"]} == sensors
    assert entry["rewards"]["upright"]["terrain_sensors"] == terrain_sensors
