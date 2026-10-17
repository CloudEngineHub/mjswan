"""A traced graph ships only if it computes what its term's body computes (#168): each
tracer runs its module against the graph on perturbed inputs, and refuses a read the body
went on without."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mjswan.compile.export import DivergentGraph  # noqa: E402
from mjswan.compile.term import trace_term  # noqa: E402


def _env():
    class _Data:
        def __init__(self):
            self.root_link_pos_w = torch.tensor([[0.1, -0.2, 0.4]])

    class _Entity:
        def __init__(self):
            self.data = _Data()

        def write_root_link_velocity_to_sim(self, velocity, env_ids=None):
            pass

    class _Scene:
        def __init__(self):
            self.sensors = {}
            self._entities = {"robot": _Entity()}

        def __getitem__(self, name):
            return self._entities[name]

    class _Env:
        def __init__(self):
            self.scene = _Scene()
            self.num_envs = 1
            self.device = "cpu"

    return _Env()


def _pos(env):
    return env.scene["robot"].data.root_link_pos_w


def scaled_by_a_read_value(env):
    return _pos(env) * _pos(env)[0, 2].item()


def branching_on_a_tensor(env):
    return _pos(env) if bool((_pos(env)[:, 2] > 0).any()) else -_pos(env)


def offset_by_a_rebuilt_tensor(env):
    return _pos(env) + torch.tensor(_pos(env)[0, 0].item())


@pytest.mark.parametrize(
    ("body", "flagged"),
    [
        (scaled_by_a_read_value, "Python number"),
        (branching_on_a_tensor, "Python boolean"),
        (offset_by_a_rebuilt_tensor, "Python number"),
    ],
)
def test_a_value_the_body_works_out_in_python_fails_the_trace(body, flagged):
    with pytest.raises(DivergentGraph, match="on perturbed inputs") as excinfo:
        trace_term(body, {}, _env(), name="pos")
    # The tracer's warning names the line to fix.
    assert f"{Path(__file__).name}:" in str(excinfo.value)
    assert flagged in str(excinfo.value)


def test_a_counter_fails_on_the_inputs_it_was_traced_on():
    calls = []

    def counted(env):
        calls.append(None)
        return _pos(env) * len(calls)

    with pytest.raises(DivergentGraph, match="on its trace inputs"):
        trace_term(counted, {}, _env(), name="counted")


def test_the_same_branch_as_a_tensor_op_traces():
    def branching_with_where(env):
        return torch.where(_pos(env) > 0, _pos(env), -_pos(env))

    export = trace_term(branching_with_where, {}, _env(), name="pos")
    assert export.input_slots == [("robot", "root_link_pos_w")]


@pytest.mark.parametrize(
    "body",
    [
        lambda env: _pos(env) * getattr(env, "scale", 1.0),
        lambda env: _pos(env) * (env.scale if hasattr(env, "scale") else 1.0),
    ],
    ids=["getattr", "hasattr"],
)
def test_a_probe_cannot_hide_a_read_the_tracer_refuses(body):
    env = _env()
    env.scale = 2.0
    with pytest.raises(DivergentGraph, match="read `env.scale`"):
        trace_term(body, {}, env, name="scaled")


def test_a_probe_for_what_the_env_lacks_traces():
    """mjlab's env has no `scale` either, so both take the default."""
    export = trace_term(
        lambda env: _pos(env) * getattr(env, "scale", 1.0), {}, _env(), name="pos"
    )
    assert export.input_slots == [("robot", "root_link_pos_w")]


def test_any_over_a_count_traces():
    """The export of `any` casts to an integer, so a count must stay a whole number."""
    env = _env()
    env.scene["robot"].data.contact_count = torch.zeros(1, 4)

    def touching(env):
        return torch.any(env.scene["robot"].data.contact_count, dim=-1)

    export = trace_term(touching, {}, env, name="touching")
    assert export.input_slots == [("robot", "contact_count")]


def test_a_fused_group_names_the_term_that_disagrees():
    from mjswan.compile.group import GroupTermSpec, trace_observation_group

    specs = [
        GroupTermSpec("pos", _pos, {}),
        GroupTermSpec("scaled", scaled_by_a_read_value, {}),
    ]
    with pytest.raises(DivergentGraph, match="group 'policy'.*term 'scaled'"):
        trace_observation_group(specs, _env(), name="policy")


def test_a_fused_termination_names_its_lane():
    from mjswan.compile.group import GroupTermSpec, trace_termination_group

    def too_low(env):
        return _pos(env)[:, 2] < 0.2

    def below_its_start(env):
        return _pos(env)[:, 2] < _pos(env)[0, 2].item() - 0.1

    specs = [
        GroupTermSpec("too_low", too_low, {}),
        GroupTermSpec("below_its_start", below_its_start, {}),
    ]
    with pytest.raises(DivergentGraph, match="term 'below_its_start'"):
        trace_termination_group(specs, _env(), name="terminations")


def push_when_drawn_forward(env, env_ids):
    asset = env.scene["robot"]
    velocity = torch.zeros(len(env_ids), 6)
    push = torch.empty(len(env_ids), 2).uniform_(-0.3, 0.3)
    if push[0, 0] > 0:
        velocity[:, :2] += push
    asset.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)


def test_an_event_branching_on_its_draw_fails_the_trace():
    from mjswan.compile import trace_event_term

    with pytest.raises(DivergentGraph, match="Event term 'push'"):
        trace_event_term(push_when_drawn_forward, {}, _env(), name="push", mode="reset")
