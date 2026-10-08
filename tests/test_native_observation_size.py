"""Observation widths a plain scene's trace env cannot supply.

A trace env built by `build_single_entity_trace_env` (a plain `add_scene()` scene) has
no action terms and no command manager, yet a fused graph has to fix every input's
width at export time. A traced `last_action` takes the policy's action width from the
stand-in action manager the build installs; a native `generated_commands` takes the
command's width from its UI inputs.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _trace_env():
    """A minimal trace env: entity data, but no actions and no commands."""

    class _Data:
        def __init__(self):
            self.root_link_ang_vel_b = torch.tensor([[0.0, 0.1, 0.2]])

    class _Scene:
        def __init__(self):
            self.sensors = {}
            self._entities = {"robot": type("E", (), {"data": _Data()})()}

        def __getitem__(self, name):
            return self._entities[name]

    class _ActionManager:
        # mjlab sizes this by `total_action_dim`, which is 0 with no action terms.
        action = torch.zeros((1, 0))
        total_action_dim = 0

    class _CommandManager:
        def get_command(self, name):
            return None  # mjlab's NullCommandManager.

    class _Env:
        num_envs = 1
        device = "cpu"

        def __init__(self):
            self.scene = _Scene()
            self.action_manager = _ActionManager()
            self.command_manager = _CommandManager()

    return _Env()


def _specs(native_size_command=None):
    from mjlab.envs.mdp import observations as obs_fns

    from mjswan.compile.group import GroupTermSpec

    return [
        GroupTermSpec("base_ang_vel", obs_fns.base_ang_vel, {}),
        GroupTermSpec("last_action", obs_fns.last_action, {}),
        GroupTermSpec(
            "velocity_cmd",
            obs_fns.generated_commands,
            {"command_name": "velocity"},
            native_size=native_size_command,
        ),
    ]


def test_the_policy_width_fixes_the_action_slot():
    pytest.importorskip("mjlab")
    from mjswan.compile.group import trace_observation_group
    from mjswan.compile.slot import slots_json
    from mjswan.mjlab.env import policy_actions

    env = _trace_env()
    with policy_actions(env, 29):
        export = trace_observation_group(
            _specs(native_size_command=3), env, name="policy"
        )

    assert export.layout == [
        {"name": "base_ang_vel", "size": 3},
        {"name": "last_action", "size": 29},
        {"name": "velocity_cmd", "size": 3},
    ]
    assert {"action": "action", "input": "action__action", "shape": [1, 29]} in (
        slots_json(export)
    )
    assert [native["name"] for native in export.native_inputs] == ["velocity_cmd"]
    # The stand-in lasts only as long as the trace.
    assert env.action_manager.action.shape == (1, 0)


def test_an_action_width_neither_side_knows_fails_the_build():
    """Not silently zero-wide: that shortens the policy's input vector."""
    pytest.importorskip("mjlab")
    from mjswan.compile.group import trace_observation_group

    with pytest.raises(ValueError, match="policy_num_actions"):
        trace_observation_group(
            _specs(native_size_command=3), _trace_env(), name="policy"
        )


def test_a_command_width_neither_side_knows_fails_the_build():
    pytest.importorskip("mjlab")
    from mjswan.compile.group import trace_observation_group
    from mjswan.mjlab.env import policy_actions

    env = _trace_env()
    with policy_actions(env, 29), pytest.raises(ValueError, match="velocity_cmd"):
        trace_observation_group(_specs(), env, name="policy")


def test_an_env_with_action_terms_keeps_its_own():
    """An mjlab task's env knows its width; a stand-in would hide its terms."""
    from mjswan.mjlab.env import policy_actions

    env = _trace_env()
    own = type("_Own", (), {"action": torch.zeros(1, 4), "total_action_dim": 4})()
    env.action_manager = own
    with policy_actions(env, 29):
        assert env.action_manager is own


def test_native_command_sizes_reads_the_ui_inputs():
    from mjswan.build.mdp import native_command_sizes
    from mjswan.envs.mdp.commands import ui_command, velocity_command
    from mjswan.managers.command_manager import Button

    sizes = native_command_sizes(
        {
            "velocity": velocity_command(),
            # Buttons carry no value, so this command contributes no width.
            "buttons": ui_command([Button(name="go", label="Go")]),
        }
    )

    assert sizes == {"velocity": 3}
