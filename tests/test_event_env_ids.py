"""The ``env_ids`` an event term is traced with: what mjlab's ``EventManager.apply``
hands it in that mode."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
# In this module's globals, where the tracer's draw recorder finds it.
sample_uniform = pytest.importorskip("mjlab.utils.lab_api.math").sample_uniform

from mjswan.compile import trace_event_term  # noqa: E402


class _Data:
    def __init__(self):
        self.root_link_vel_w = torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.1]])


class _Entity:
    def __init__(self):
        self.data = _Data()

    def write_root_link_velocity_to_sim(self, velocity, env_ids=None):
        pass


class _Scene(dict):
    def __getitem__(self, name):
        return self.setdefault(name, _Entity())


class _Env:
    def __init__(self):
        self.scene = _Scene()
        self.num_envs = 1
        self.device = "cpu"


def push(env, env_ids):
    """A push written for the ids mjlab passes, with no ``None`` case."""
    asset = env.scene["robot"]
    vel_w = asset.data.root_link_vel_w[env_ids]
    vel_w[:, :2] += sample_uniform(-0.3, 0.3, (len(env_ids), 2), device=env.device)
    asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


@pytest.mark.parametrize("mode", ["interval", "reset", "manual"])
def test_a_term_indexing_its_env_ids_traces_to_a_graph_that_runs(mode):
    export = trace_event_term(push, {}, _Env(), name="push_robot", mode=mode)

    session = ort.InferenceSession(export.onnx_bytes)
    feeds = {
        "robot__root_link_vel_w": np.array(
            [[0.2, 0.0, 0.0, 0.0, 0.0, 0.1]], np.float32
        ),
        "rand": np.array([0.3, -0.1], np.float32),
    }
    (velocity,) = session.run(None, feeds)
    assert velocity.reshape(-1) == pytest.approx([0.5, -0.1, 0.0, 0.0, 0.0, 0.1])


@pytest.mark.parametrize(
    ("mode", "is_global_time", "expected"),
    [
        ("startup", False, None),
        ("interval", True, None),
        ("interval", False, [0]),
        ("reset", False, [0]),
        ("manual", False, [0]),
    ],
)
def test_the_term_gets_the_env_ids_mjlab_would_pass(mode, is_global_time, expected):
    seen = []

    def record(env, env_ids):
        seen.append(env_ids)
        asset = env.scene["robot"]
        asset.write_root_link_velocity_to_sim(asset.data.root_link_vel_w)

    trace_event_term(
        record, {}, _Env(), name="record", mode=mode, is_global_time=is_global_time
    )

    assert seen
    for env_ids in seen:
        assert (env_ids if env_ids is None else env_ids.tolist()) == expected
