"""A motion's ``frame_fields`` and ``group``: per-frame arrays a traced graph reads off the
tracking command, and the heading the motion selector lists a motion under.

Layer: L1 (a build with the Node step mocked; no tracing).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from mjswan.build.asset import check_frame_fields
from mjswan.builder import Builder
from mjswan.envs.mdp.commands import TRACKING_STATE_FIELDS
from mjswan.managers.command_manager import CommandTermConfig
from mjswan.motion import MotionConfig

TRACKING_COMMAND_TS = (
    Path(__file__).resolve().parents[1]
    / "src/mjswan/template/src/core/command/TrackingCommand.ts"
)

_MOTION = {"anchor_body_name": "pelvis", "body_names": ("pelvis",)}


def _clip(path: Path, frames: int = 4, **arrays: np.ndarray) -> str:
    np.savez(path, joint_pos=np.zeros((frames, 1), np.float32), **arrays)
    return str(path)


def _policy(minimal_model, minimal_onnx, tmp_path: Path, slots=()):
    """A policy on a ``TrackingCommand`` named ``motion``, its one observation group
    reading ``slots`` off it."""
    config_path = tmp_path / "policy.json"
    slot = [{"command": "motion", "field": f, "input": f"command__{f}"} for f in slots]
    group = {"input_slots": slot}
    config_path.write_text(json.dumps({"observations": {"actor": group}}))
    builder = Builder()
    scene = builder.add_project(name="P").add_scene(
        control_dt=0.02, name="S", model=minimal_model
    )
    policy = scene.add_policy(
        name="Tracker",
        policy=minimal_onnx,
        config_path=str(config_path),
        commands={
            "motion": CommandTermConfig(
                term_name="TrackingCommand", params=dict(_MOTION)
            )
        },
    )
    return builder, policy


class TestSerialization:
    def test_omitted_when_unset(self):
        entry = MotionConfig(name="Walk", **_MOTION).to_dict("assets/walk.npz")
        assert "frame_fields" not in entry
        assert "group" not in entry

    def test_written_when_set(self):
        entry = MotionConfig(
            name="Walk", frame_fields=("latent",), group="Walk forward", **_MOTION
        ).to_dict("assets/walk.npz")
        assert entry["frame_fields"] == ["latent"]
        assert entry["group"] == "Walk forward"

    def test_add_motion_reaches_the_manifest(
        self, tmp_path, minimal_model, minimal_onnx, build_manifest
    ):
        builder, policy = _policy(minimal_model, minimal_onnx, tmp_path)
        latent = np.zeros((4, 32), np.float32)
        policy.add_motion(
            name="Walk",
            source=_clip(tmp_path / "walk.npz", latent=latent),
            frame_fields=["latent"],
            group="Walk forward",
            **_MOTION,
        )
        policy.add_motion(
            name="Idle", source=_clip(tmp_path / "idle.npz", latent=latent), **_MOTION
        )
        manifest = build_manifest(builder, tmp_path / "out")
        motions = manifest["projects"][0]["scenes"][0]["policies"][0]["motions"]
        assert motions[0]["frame_fields"] == ["latent"]
        assert motions[0]["group"] == "Walk forward"
        assert "frame_fields" not in motions[1] and "group" not in motions[1]


class TestClipValidation:
    def test_the_build_checks_each_clip(
        self, tmp_path, minimal_model, minimal_onnx, build_manifest
    ):
        builder, policy = _policy(minimal_model, minimal_onnx, tmp_path)
        policy.add_motion(
            name="Walk",
            source=_clip(tmp_path / "walk.npz"),
            frame_fields=["latent"],
            **_MOTION,
        )
        with pytest.raises(ValueError, match="'latent' is not an array in its .npz"):
            build_manifest(builder, tmp_path / "out")

    def test_one_row_per_frame_is_required(self, tmp_path):
        source = _clip(tmp_path / "walk.npz", latent=np.zeros((3, 32), np.float32))
        motion = MotionConfig(name="Walk", source=source, frame_fields=("latent",))
        with pytest.raises(ValueError, match=r"shape \(3, 32\).*one row per frame, 4"):
            check_frame_fields(motion)

    def test_a_native_field_name_is_refused(self, tmp_path):
        source = _clip(tmp_path / "walk.npz", body_pos_w=np.zeros((4, 3), np.float32))
        motion = MotionConfig(name="Walk", source=source, frame_fields=("body_pos_w",))
        with pytest.raises(ValueError, match="'body_pos_w' is a field the tracking"):
            check_frame_fields(motion)

    def test_a_qpos_clip_counts_its_qpos_frames(self, tmp_path):
        path = tmp_path / "walk.npz"
        np.savez(path, qpos=np.zeros((5, 7)), latent=np.zeros((5, 2)))
        check_frame_fields(
            MotionConfig(
                name="Walk",
                data=path.read_bytes(),
                clip_format="qpos",
                frame_fields=("latent",),
            )
        )


class TestGraphReads:
    """A graph reading a field ``TrackingCommand`` cannot serve fails the build."""

    def test_a_field_every_motion_lists_builds(
        self, tmp_path, minimal_model, minimal_onnx, build_manifest
    ):
        builder, policy = _policy(minimal_model, minimal_onnx, tmp_path, ("latent",))
        for name in ("Walk", "Run"):
            policy.add_motion(
                name=name,
                source=_clip(
                    tmp_path / f"{name}.npz", latent=np.zeros((4, 2), np.float32)
                ),
                frame_fields=["latent"],
                **_MOTION,
            )
        build_manifest(builder, tmp_path / "out")

    def test_native_fields_need_no_frame_fields(
        self, tmp_path, minimal_model, minimal_onnx, build_manifest
    ):
        builder, policy = _policy(
            minimal_model, minimal_onnx, tmp_path, ("anchor_pos_w", "command")
        )
        policy.add_motion(name="Walk", source=_clip(tmp_path / "w.npz"), **_MOTION)
        build_manifest(builder, tmp_path / "out")

    def test_a_motion_lacking_the_field_is_named(
        self, tmp_path, minimal_model, minimal_onnx, build_manifest
    ):
        builder, policy = _policy(minimal_model, minimal_onnx, tmp_path, ("latent",))
        policy.add_motion(
            name="Walk",
            source=_clip(tmp_path / "w.npz", latent=np.zeros((4, 2), np.float32)),
            frame_fields=["latent"],
            **_MOTION,
        )
        policy.add_motion(name="Run", source=_clip(tmp_path / "r.npz"), **_MOTION)
        with pytest.raises(
            ValueError, match=r"field 'latent'.*motions \['Run'\] do not list it"
        ):
            build_manifest(builder, tmp_path / "out")

    def test_a_policy_with_no_motion_is_refused(
        self, tmp_path, minimal_model, minimal_onnx, build_manifest
    ):
        builder, _ = _policy(minimal_model, minimal_onnx, tmp_path, ("latent",))
        with pytest.raises(ValueError, match="no motion does list it"):
            build_manifest(builder, tmp_path / "out")


def test_native_fields_match_tracking_command():
    """``TRACKING_STATE_FIELDS`` is ``TrackingCommand.getStateField``'s own cases."""
    source = TRACKING_COMMAND_TS.read_text()
    body = source[source.index("getStateField(field: string)") :]
    body = body[: body.index("default:")]
    assert set(re.findall(r"case '(\w+)':", body)) == TRACKING_STATE_FIELDS
