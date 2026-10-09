"""Input widths a plain scene's trace env cannot supply.

A trace env built by `build_single_entity_trace_env` (a plain `add_scene()` scene) has
no action terms and no command manager, yet a graph has to fix every input's width at
export time. While tracing an MDP the build stands in for both: an action manager of the
policy's width, and a zero command of each width it knows.
"""

from __future__ import annotations

from types import SimpleNamespace

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
        # mjlab's NullCommandManager.
        def get_command(self, name):
            return None

        def get_term(self, name):
            return None

    class _Env:
        num_envs = 1
        device = "cpu"

        def __init__(self):
            self.scene = _Scene()
            self.action_manager = _ActionManager()
            self.command_manager = _CommandManager()

    return _Env()


def _specs():
    from mjlab.envs.mdp import observations as obs_fns

    from mjswan.compile.group import GroupTermSpec

    return [
        GroupTermSpec("base_ang_vel", obs_fns.base_ang_vel, {}),
        GroupTermSpec("last_action", obs_fns.last_action, {}),
        GroupTermSpec(
            "velocity_cmd", obs_fns.generated_commands, {"command_name": "velocity"}
        ),
    ]


def test_the_stand_ins_fix_the_slot_widths():
    pytest.importorskip("mjlab")
    from mjswan.compile.group import trace_observation_group
    from mjswan.compile.slot import slots_json
    from mjswan.mjlab.env import mdp_commands, policy_actions

    env = _trace_env()
    with policy_actions(env, 29), mdp_commands(env, {"velocity": 3}):
        export = trace_observation_group(_specs(), env, name="policy")

    assert export.layout == [
        {"name": "base_ang_vel", "size": 3},
        {"name": "last_action", "size": 29},
        {"name": "velocity_cmd", "size": 3},
    ]
    slots = slots_json(export)
    assert {"action": "action", "input": "action__action", "shape": [1, 29]} in slots
    assert {
        "command": "velocity",
        "field": "command",
        "input": "command__velocity_command",
        "shape": [1, 3],
    } in slots
    # Both last only as long as the trace.
    assert env.action_manager.action.shape == (1, 0)
    assert env.command_manager.get_command("velocity") is None


def test_an_action_width_neither_side_knows_fails_the_build():
    """Not silently zero-wide: that shortens the policy's input vector."""
    pytest.importorskip("mjlab")
    from mjswan.compile.group import trace_observation_group
    from mjswan.mjlab.env import mdp_commands

    env = _trace_env()
    with (
        mdp_commands(env, {"velocity": 3}),
        pytest.raises(ValueError, match="policy_num_actions"),
    ):
        trace_observation_group(_specs(), env, name="policy")


def test_a_command_width_neither_side_knows_fails_the_build():
    """Named, rather than mjlab's bare `assert command is not None`."""
    pytest.importorskip("mjlab")
    from mjswan.compile.group import trace_observation_group
    from mjswan.mjlab.env import policy_actions

    env = _trace_env()
    with policy_actions(env, 29), pytest.raises(ValueError, match="'velocity'"):
        trace_observation_group(_specs(), env, name="policy")


def test_an_env_with_action_terms_keeps_its_own():
    """An mjlab task's env knows its width; a stand-in would hide its terms."""
    from mjswan.mjlab.env import policy_actions

    env = _trace_env()
    own = SimpleNamespace(action=torch.zeros(1, 4), total_action_dim=4)
    env.action_manager = own
    with policy_actions(env, 29):
        assert env.action_manager is own


def test_an_env_with_the_command_keeps_it():
    from mjswan.mjlab.env import TraceCommandManager, mdp_commands

    env = _trace_env()
    own = SimpleNamespace(command=torch.ones(1, 3))
    env.command_manager = TraceCommandManager({"velocity": own})
    with mdp_commands(env, {"velocity": 3, "height": 1}):
        assert env.command_manager.get_term("velocity") is own
        assert env.command_manager.get_command("height").shape == (1, 1)


def test_command_widths_come_from_the_trace_or_the_ui_inputs():
    from mjswan.build.mdp import command_widths
    from mjswan.envs.mdp.commands import ui_command, velocity_command
    from mjswan.managers.command_manager import Button, CommandTermConfig

    widths = command_widths(
        {
            "velocity": velocity_command(),
            # Buttons carry no value, so this command contributes no width.
            "buttons": ui_command([Button(name="go", label="Go")]),
            "twist": CommandTermConfig(term_name="OnnxCommand"),
        },
        {
            "twist": {
                "command_field": "vel_command_b",
                "state_fields": [
                    {"name": "heading_target", "shape": [1]},
                    {"name": "vel_command_b", "shape": [1, 3]},
                ],
            }
        },
    )

    assert widths == {"velocity": 3, "twist": 3}
