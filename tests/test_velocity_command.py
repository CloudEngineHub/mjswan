"""mjlab's own `UniformVelocityCommand`, traced, against the same term run eagerly.

Layer: L1 (no env build: the term is constructed against a stand-in env, and the graph
is run by ONNX's reference evaluator).

The graph has to agree with mjlab's body on every branch a draw sends it down: heading,
world-frame, standing and forward-only envs, any mix of them, and on an episode reset
the `init_velocity_prob` start. Getting one wrong is silent: a well-formed command of
the right width that mjlab would not have issued.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg  # noqa: E402

from mjswan.compile import trace_command_term  # noqa: E402
from mjswan.compile.command import compare_step, reference_runner  # noqa: E402
from mjswan.managers.command_manager import _custom_registry  # noqa: E402

STATE_FIELDS = list(_custom_registry["UniformVelocityCommandCfg"].state_fields or [])
FLAGS = ["is_heading_env", "is_standing_env", "is_world_env", "is_forward_env"]


class _Robot:
    """`data.heading_w`, and the write methods that mark an attribute as an entity."""

    def __init__(self):
        self.data = SimpleNamespace(heading_w=torch.tensor([0.7]))

    def write_root_link_velocity_to_sim(self, *args, **kwargs):
        raise AssertionError("the tracer captures writes")

    write_root_link_velocity_b_to_sim = write_root_link_velocity_to_sim


class _FakeEnv:
    """Enough env for `CommandTerm.__init__` and the tracer. No scene is compiled, so
    these tests stay out of the `slow` tier."""

    def __init__(self):
        self.scene = {"robot": _Robot()}
        self.command_manager = SimpleNamespace(
            get_command=lambda name: None, get_term=lambda name: None
        )
        self.num_envs = 1
        self.device = "cpu"


def _cfg(**overrides) -> UniformVelocityCommandCfg:
    """mjlab's own velocity-task shape: heading tracking, standing and forward envs."""
    params = dict(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.1,
        rel_heading_envs=0.3,
        rel_forward_envs=0.2,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
    )
    params.update(overrides)
    return UniformVelocityCommandCfg(**params)


#: Every selection a coin flip, so a few dozen steps take each branch both ways.
_MIXED = dict(
    rel_standing_envs=0.5,
    rel_heading_envs=0.5,
    rel_world_envs=0.5,
    rel_forward_envs=0.5,
    init_velocity_prob=0.5,
)
_NO_HEADING = dict(
    heading_command=False,
    ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-0.5, 0.5)
    ),
)


def _traced(**overrides):
    term = _cfg(**overrides).build(_FakeEnv())
    export = trace_command_term(
        term, STATE_FIELDS, name="twist", command_field="vel_command_b"
    )
    return term, export


def test_the_binding_traces_mjlabs_own_body():
    binding = _custom_registry["UniformVelocityCommandCfg"]
    assert binding.is_onnx_traced
    assert binding.trace_override is None
    assert binding.command_field == "vel_command_b"


@pytest.mark.parametrize(
    "overrides",
    [{}, _MIXED, {**_MIXED, **_NO_HEADING}],
    ids=["task", "every-branch", "no-heading"],
)
def test_the_graph_matches_mjlab_step_after_step(overrides):
    """Resamples, episode resets and plain updates, chained, at varying headings."""
    term, export = _traced(**overrides)
    run = reference_runner(export)
    ranges = torch.tensor(export.rand_ranges).reshape(-1, 2)
    generator = torch.Generator().manual_seed(0)
    seen = {flag: set() for flag in FLAGS}
    for step in range(48):
        term.robot.data.heading_w = torch.rand(1, generator=generator) * 6 - 3
        resample = step % 4 != 3
        draw = torch.rand(len(ranges), generator=generator)
        rand = ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * draw
        _, note = compare_step(
            term,
            export,
            STATE_FIELDS,
            run,
            rand,
            resample=resample,
            reset=resample and step % 2 == 0,
        )
        assert note is None, f"step {step}: {note}"
        for flag in FLAGS:
            seen[flag].add(bool(getattr(term, flag)))
    if overrides is _MIXED:
        # Otherwise a branch could have matched by never running.
        assert all(values == {False, True} for values in seen.values()), seen


def test_init_velocity_prob_starts_an_env_moving_on_a_reset_alone():
    """mjlab writes the commanded planar velocity, in the body frame, before the update
    that would zero a standing env's command."""
    term, export = _traced(init_velocity_prob=0.5)
    (target,) = export.write_targets
    assert (target["kind"], target["entity"]) == ("root_velocity_b", "robot")
    run = reference_runner(export)
    ranges = torch.tensor(export.rand_ranges).reshape(-1, 2)
    feeds = {f"prev_{f}": getattr(term, f).numpy() for f in STATE_FIELDS}
    feeds["robot__heading_w"] = np.array([0.0], dtype=np.float32)
    feeds["resample_mask"] = np.array([True])
    # Every draw at its low end: the init draw selects, and so does standing.
    feeds["rand"] = ranges[:, 0].numpy()

    on_reset = run({**feeds, "reset_mask": np.array([True])})
    on_timer = run({**feeds, "reset_mask": np.array([False])})

    assert on_reset[target["gate"]].tolist() == [1.0]
    assert on_timer[target["gate"]].tolist() == [0.0]
    velocity = on_reset[target["outputs"][0]].reshape(-1)
    # Forward-only takes |x| = 1 and zeroes the rest; standing has not run yet.
    assert velocity.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert on_reset["next_vel_command_b"].reshape(-1).tolist() == [0.0, 0.0, 0.0]


def test_a_body_the_trace_cannot_follow_fails_the_build():
    """A guarded branch that is not a no-op when its guard fails would ship a command
    mjlab never issues; the build runs the graph against the body and refuses it."""

    class _Counter:
        num_envs = 1
        cfg = SimpleNamespace(entity_name=None)

        def __init__(self):
            self._env = _FakeEnv()
            self.flag = torch.zeros(1, dtype=torch.bool)
            self.count = torch.zeros(1)

        def _resample_command(self, env_ids):
            r = torch.empty(len(env_ids))
            self.flag[env_ids] = r.uniform_(0.0, 1.0) <= 0.5

        def _update_command(self, env_ids=None):
            if self.flag.any():
                self.count = self.count + 1.0

    with pytest.raises(ValueError, match="disagrees with its own body"):
        trace_command_term(
            _Counter(), ["flag", "count"], name="counter", command_field="count"
        )


#: Assigned by every `_update_command` before anything reads it.
_DERIVED = {"heading_error"}


def test_the_body_keeps_no_state_the_binding_does_not_carry():
    """A tensor the body carries across frames that `state_fields` omits would restart
    from its build-time value every browser step."""
    term = _cfg(**_MIXED).build(_FakeEnv())
    before = {
        name: value.clone()
        for name, value in vars(term).items()
        if isinstance(value, torch.Tensor)
    }
    torch.manual_seed(0)
    term._resample_command(torch.arange(1))
    term._update_command(None)

    changed = {
        name
        for name, value in vars(term).items()
        if isinstance(value, torch.Tensor)
        and (name not in before or not torch.equal(value, before[name]))
    }
    assert changed - _DERIVED <= set(STATE_FIELDS), sorted(changed - set(STATE_FIELDS))
