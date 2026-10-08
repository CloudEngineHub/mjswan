"""A traced command body reading another command's state through ``env.command_manager``.

The read becomes a ``{command, field}`` input slot, which the browser serves from that
command's state as it serves an observation's.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("onnxruntime")

from mjswan.compile import run_command_parity, trace_command_term  # noqa: E402
from mjswan.compile.slot import _COMMAND_NS, slots_json  # noqa: E402


class _Leader:
    """Stands in for the command the traced one reads."""

    def __init__(self) -> None:
        self.vel = torch.tensor([[0.4, -0.2, 1.0]])
        self.gain = torch.tensor([[2.0]])
        self.is_standing = torch.tensor([False])


class _Manager:
    def __init__(self, leader: _Leader) -> None:
        self.leader = leader

    def get_term(self, name: str) -> Any:
        assert name == "leader"
        return self.leader

    def get_command(self, name: str) -> torch.Tensor:
        return self.get_term(name).vel


class _Follower:
    """A command integrating the leader's speed, scaled by one of its fields."""

    def __init__(self, env: Any, read_flag: bool = False) -> None:
        self._env = env
        self.num_envs = 1
        self.device = "cpu"
        self.cfg = SimpleNamespace(entity_name=None)
        self.level = torch.zeros(1, 1)
        self._read_flag = read_flag

    def _resample_command(self, env_ids: Any) -> None:
        del env_ids
        self.level = torch.zeros_like(self.level)

    def _update_command(self, env_ids: Any) -> None:
        del env_ids
        manager = self._env.command_manager
        speed = manager.get_command("leader")[:, :2].norm(dim=1, keepdim=True)
        self.level = self.level + speed * manager.get_term("leader").gain
        if self._read_flag:
            standing = manager.get_term("leader").is_standing
            self.level = torch.where(standing, torch.zeros_like(self.level), self.level)


def _follower(read_flag: bool = False) -> tuple[_Follower, _Leader]:
    leader = _Leader()
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        cfg=None,
        scene=SimpleNamespace(entities={}),
        command_manager=_Manager(leader),
    )
    return _Follower(env, read_flag), leader


def test_command_read_becomes_an_input_slot() -> None:
    term, _ = _follower()
    export = trace_command_term(term, ["level"], name="follower", command_field="level")

    assert export.input_slots == [
        (_COMMAND_NS, "leader.command"),
        (_COMMAND_NS, "leader.gain"),
    ]
    assert [(s["command"], s["field"]) for s in slots_json(export)] == [
        ("leader", "command"),
        ("leader", "gain"),
    ]

    import onnxruntime as ort

    session = ort.InferenceSession(
        export.onnx_bytes, providers=["CPUExecutionProvider"]
    )
    vel = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
    gain = np.array([[0.5]], dtype=np.float32)
    feeds = dict(zip(export.input_names, (vel, gain)))
    feeds["prev_level"] = np.array([[1.0]], dtype=np.float32)
    feeds["resample_mask"] = np.zeros(1, dtype=bool)
    names = {i.name for i in session.get_inputs()}
    feeds = {k: v for k, v in feeds.items() if k in names}
    (level,) = session.run(["next_level"], feeds)
    # Values unlike the trace-time ones, so a baked constant would show.
    np.testing.assert_allclose(level, [[1.0 + 5.0 * 0.5]])


def test_command_read_passes_parity() -> None:
    term, leader = _follower()
    leader.vel = torch.tensor([[-0.3, 0.1, 0.0]])
    report = run_command_parity(term, ["level"], name="follower", command_field="level")
    assert report.passed, report


def test_command_read_of_a_flag_is_refused() -> None:
    term, _ = _follower(read_flag=True)
    with pytest.raises(ValueError, match="leader.is_standing"):
        trace_command_term(term, ["level"], name="follower", command_field="level")
