"""A task's own action term traced to ONNX (ADR 0009).

Layer: L3 (builds a small mjlab env on the CPU).

The trace has to reproduce the term's own ``process_actions`` and ``reset``: the state
it promotes to graph I/O, the env reads it threads as slots, and the branches it bakes.
The build runs each graph against the body before shipping it; these pin what that
check is for, and what the tracer refuses instead of shipping a frozen copy.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("mjlab")

import action_term_fixture as fx  # noqa: E402

from mjswan.build.mdp.action import (  # noqa: E402
    inherited_apply,
    serialize_actions,
)
from mjswan.compile.action import (  # noqa: E402
    UntraceableAction,
    compare_body,
    trace_action_term,
)
from mjswan.envs.mdp.actions import TracedActionCfg  # noqa: E402
from mjswan.managers.action_manager import ActionBinding  # noqa: E402


@pytest.fixture(scope="module")
def env():
    """One env for every term: each test builds its own term on it."""
    env = fx.arm_env({"joint_pos": fx.on_arm(fx.DeltaActionCfg)})
    yield env
    env.close()


def _ort(graph):
    import onnxruntime as ort

    session = ort.InferenceSession(graph.onnx_bytes)
    names = {i.name for i in session.get_inputs()}

    def run(feeds):
        outs = session.run(
            graph.output_names, {k: v for k, v in feeds.items() if k in names}
        )
        return dict(zip(graph.output_names, outs))

    return run


def _state_names(export) -> list[str]:
    return [spec["name"] for spec in export.state_fields]


def test_a_delta_term_carries_its_target_and_restarts_it_at_the_measured_pose(env):
    term = fx.on_arm(fx.DeltaActionCfg).build(env)
    export = trace_action_term(term, name="joint_pos", ramp_from=True)

    assert "_target" in _state_names(export)
    assert export.target_names == ["robot/j1", "robot/j2", "robot/j3"]
    # The soft limits are constants; only the reset reads the live joint positions.
    assert export.process.input_names == []
    assert export.reset is not None
    assert export.reset.input_names == ["robot__joint_pos"]
    assert export.process.output_names[-2:] == ["processed", "processed_from"]


def test_the_delta_graph_matches_the_body_in_onnx_runtime(env):
    term = fx.on_arm(fx.DeltaActionCfg).build(env)
    export = trace_action_term(term, name="joint_pos", ramp_from=True)
    run_process, run_reset = _ort(export.process), _ort(export.reset)
    generator = torch.Generator().manual_seed(3)

    assert compare_body(term, export, "reset", run_reset) is None
    for _ in range(12):
        raw = torch.rand((1, 3), generator=generator) * 4 - 2
        assert compare_body(term, export, "process", run_process, raw=raw) is None


def test_the_ramp_starts_where_the_term_stood_before_the_step(env):
    term = fx.on_arm(fx.DeltaActionCfg).build(env)
    export = trace_action_term(term, name="joint_pos", ramp_from=True)
    run = _ort(export.process)
    feeds = {
        f"prev_{s['name']}": np.asarray(s["init"], np.float32).reshape(s["shape"])
        for s in export.state_fields
    }
    feeds["raw"] = np.ones((1, 3), np.float32)
    outputs = run(feeds)
    np.testing.assert_allclose(
        outputs["processed_from"], feeds["prev__processed_actions"]
    )
    # The home pose is inside the soft limits, so a full action is one whole delta.
    np.testing.assert_allclose(
        outputs["processed"] - outputs["processed_from"], 1 / 24, rtol=1e-5
    )


def test_a_held_term_reads_the_episode_clock_and_zeroes_the_action_while_holding(env):
    term = fx.on_arm(fx.HeldActionCfg, scale=0.5).build(env)
    export = trace_action_term(term, name="joint_pos")
    assert export.process.input_names == ["env__episode_length_buf"]
    run = _ort(export.process)

    def processed(step: int) -> np.ndarray:
        feeds = {
            f"prev_{s['name']}": np.asarray(s["init"], np.float32).reshape(s["shape"])
            for s in export.state_fields
        }
        feeds["raw"] = np.ones((1, 3), np.float32)
        feeds["env__episode_length_buf"] = np.asarray([step], np.float32)
        return run(feeds)["processed"]

    default = term._offset.numpy()
    # 50 steps of 20 ms is the 1 s hold: the trace kept both sides of mjlab's branch.
    np.testing.assert_allclose(processed(49), default, atol=1e-6)
    np.testing.assert_allclose(processed(50), default + 0.5, atol=1e-6)


def test_an_ema_term_gets_mjlabs_apply_without_a_binding(env):
    term = fx.on_arm(fx.EmaActionCfg).build(env)
    export = trace_action_term(term, name="joint_pos")
    assert "_ema" in _state_names(export)
    assert inherited_apply(term, unchanged=True) == "joint_position"


def test_a_term_overriding_apply_needs_a_binding_to_say_what_it_does(env):
    held = fx.on_arm(fx.HeldActionCfg).build(env)
    assert inherited_apply(held, unchanged=True) is None
    # With a binding, the mjlab base it subclasses names the apply.
    assert inherited_apply(held, unchanged=False) == "joint_position"
    delta = fx.on_arm(fx.DeltaActionCfg).build(env)
    assert inherited_apply(delta, unchanged=False) is None


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (fx.on_arm(fx.CountingActionCfg), r"changes \['_steps'\]"),
        (fx.on_arm(fx.WritingActionCfg), r"set_joint_position_target"),
    ],
    ids=["python-counter", "writes-the-sim"],
)
def test_what_a_graph_cannot_carry_is_refused(env, cfg, match):
    with pytest.raises(UntraceableAction, match=match):
        trace_action_term(cfg.build(env), name="joint_pos")


def test_a_trace_leaves_the_term_as_it_found_it(env):
    term = fx.on_arm(fx.DeltaActionCfg).build(env)
    before = {
        k: v.clone() for k, v in vars(term).items() if isinstance(v, torch.Tensor)
    }
    trace_action_term(term, name="joint_pos", ramp_from=True)
    for name, value in before.items():
        assert torch.equal(getattr(term, name), value), name


def test_the_build_writes_both_graphs_and_names_the_apply(env, tmp_path):
    entry = serialize_actions(
        {
            "joint_pos": TracedActionCfg(
                mjlab_cfg=fx.on_arm(fx.DeltaActionCfg),
                binding=ActionBinding(
                    apply="joint_position",
                    interpolate=lambda cfg: cfg.interpolate_decimation,
                ),
            )
        },
        env=env,
        out_dir=tmp_path,
    )["joint_pos"]

    assert entry["type"] == "joint_position"
    assert entry["interpolate"] is True
    assert entry["actuator_names"] == ["robot/j1", "robot/j2", "robot/j3"]
    assert entry["process"]["ramp_from"] is True
    assert (tmp_path / entry["process"]["onnx"]).is_file()
    assert (tmp_path / entry["reset"]["onnx"]).is_file()
    assert entry["reset"]["input_slots"] == [
        {
            "entity": "robot",
            "field": "joint_pos",
            "input": "robot__joint_pos",
            "shape": [1, 3],
        }
    ]
    init = {f["name"]: f["init"] for f in entry["process"]["state_fields"]}
    np.testing.assert_allclose(init["_target"], list(fx.HOME.values()), atol=1e-6)


def test_the_build_refuses_an_overridden_apply_without_a_binding(env, tmp_path):
    cfg = TracedActionCfg(mjlab_cfg=fx.on_arm(fx.HeldActionCfg))
    with pytest.raises(ValueError, match="overrides apply_actions"):
        serialize_actions({"joint_pos": cfg}, env=env, out_dir=tmp_path)


def test_without_a_trace_env_the_build_says_how_to_supply_one(tmp_path):
    cfg = TracedActionCfg(mjlab_cfg=fx.on_arm(fx.DeltaActionCfg))
    with pytest.raises(ValueError, match="set_trace_env"):
        serialize_actions({"joint_pos": cfg}, out_dir=tmp_path)


def test_an_observation_reads_a_traced_terms_state_as_a_slot(env, tmp_path):
    from mjswan.build.mdp import serialize_observation_group
    from mjswan.compile.slot import traced_action_state
    from mjswan.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )

    def target(env, action_name="joint_pos"):
        return env.action_manager.get_term(action_name)._target

    def processed(env, action_name="joint_pos"):
        return env.action_manager.get_term(action_name)._processed_actions

    group = ObservationGroupCfg(
        terms={
            "target": ObservationTermCfg(func=target),
            "processed": ObservationTermCfg(func=processed),
        }
    )
    with traced_action_state({"joint_pos": ["_target", "_processed_actions"]}):
        entry = serialize_observation_group(group, env, tmp_path, "policy")

    assert [(s["action_term"], s["field"]) for s in entry["input_slots"]] == [
        ("joint_pos", "_processed_actions"),
        ("joint_pos", "_target"),
    ]
