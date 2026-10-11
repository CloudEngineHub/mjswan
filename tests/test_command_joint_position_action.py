"""``CommandJointPositionActionCfg``: serialization, and the command it must find."""

from __future__ import annotations

import json

import pytest

from mjswan import Builder, SliderConfig, ui_command
from mjswan.envs.mdp.actions import CommandJointPositionActionCfg


def test_to_dict():
    cfg = CommandJointPositionActionCfg(
        command_name="arm",
        actuator_names=("robot/.*_elbow_joint",),
        offset=0.1,
        stiffness=40.0,
        clip={".*": (-1.0, 1.0)},
    )
    assert cfg.unsupported_reason is None
    assert cfg.to_dict() == {
        "type": "joint_position_command",
        "offset": 0.1,
        "actuator_names": ["robot/.*_elbow_joint"],
        "command_name": "arm",
        "stiffness": 40.0,
        "clip": {".*": [-1.0, 1.0]},
    }


def _build(tmp_path, model, policy, commands) -> dict:
    builder = Builder()
    scene = builder.add_project(name="P").add_scene(
        control_dt=0.02, name="S", model=model
    )
    scene.add_policy(
        name="Policy",
        policy=policy,
        commands=commands,
        actions={
            "arm": CommandJointPositionActionCfg(
                command_name="arm", actuator_names=("elbow",)
            )
        },
    )
    out = tmp_path / "out"
    builder._save_web(out)
    return json.loads((out / "manifest.json").read_text())


def test_reaches_the_mdp_entry(tmp_path, minimal_model, minimal_onnx):
    slider = SliderConfig(name="elbow", label="Elbow", range=(-1.0, 1.0))
    manifest = _build(
        tmp_path, minimal_model, minimal_onnx, {"arm": ui_command([slider])}
    )
    mdp = manifest["projects"][0]["scenes"][0]["mdps"][0]
    assert mdp["actions"]["arm"] == {
        "type": "joint_position_command",
        "actuator_names": ["elbow"],
        "command_name": "arm",
    }


def test_a_missing_command_fails_the_build(tmp_path, minimal_model, minimal_onnx):
    with pytest.raises(ValueError, match="reads command 'arm'"):
        _build(tmp_path, minimal_model, minimal_onnx, {})
