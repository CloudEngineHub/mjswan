"""Dump the action-parity fixture for ``actionParity.test.ts``: task-defined action terms
stepped in mjlab, with what each physics substep measured and actuated.

The terms are ``action_term_fixture``'s: in-hand rotation's delta with its substep ramp,
luwu's held action, and an EMA term on mjlab's own apply behind an actuator command
delay. Each goes through the adapter and the Builder's own ``serialize_actions``, so
the test loads the entry and the graphs that really ship.

**States are replayed, not co-simulated**, as in ``dump_rollout_fixture.py``: the test
feeds each substep's joint positions from here and compares the ``ctrl`` it writes.

Regenerate with::

    MUJOCO_GL=disable .venv/bin/python tests/dump_action_fixture.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "disable")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "src/mjswan/template/src/core/action/__tests__/fixtures/action"

STEPS = 14
#: The episode restarts before this step, so the reset graph and the delay's reset run.
RESET_AT = 9
SEED = 0


def _flat(value: Any) -> list[float]:
    return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]


def _cases() -> dict[str, tuple[Any, Any, int]]:
    """Per case: the task's mjlab cfg, the binding it registers, the command delay."""
    import action_term_fixture as fx

    from mjswan import ActionBinding

    return {
        "ramp": (
            fx.on_arm(fx.DeltaActionCfg),
            ActionBinding(
                apply="joint_position",
                interpolate=lambda cfg: cfg.interpolate_decimation,
            ),
            0,
        ),
        # 5 * float32(0.02) rounds to just under float32(0.1): mjlab still holds at step 5.
        "hold": (
            fx.on_arm(fx.HeldActionCfg, hold_duration_s=0.1, scale=0.5),
            ActionBinding(hold_s=lambda cfg: cfg.hold_duration_s),
            0,
        ),
        "delay": (fx.on_arm(fx.EmaActionCfg, scale=0.5), None, 2),
    }


def _dump_case(cfg: Any, binding: Any, delay_lag: int, out_dir: Path) -> dict[str, Any]:
    import action_term_fixture as fx
    import torch

    from mjswan.build.mdp import serialize_actions
    from mjswan.managers.action_manager import _action_registry, register_action
    from mjswan.mjlab import adapt_actions, resolve_actuator_delays

    env = fx.arm_env({"joint_pos": cfg}, delay_lag=delay_lag)
    joints = [f"robot/{name}" for name in env.scene["robot"].joint_names]
    if binding is not None:
        register_action(type(cfg).__name__, binding)
    try:
        actions = adapt_actions({"joint_pos": cfg}) or {}
        resolve_actuator_delays(actions, joints, env.cfg)
        # The build looks the binding up, so it stays registered until then.
        entry = serialize_actions(actions, env=env, out_dir=out_dir)["joint_pos"]
    finally:
        _action_registry.pop(type(cfg).__name__, None)
    # The browser starts the term at `init`; the env's own term starts at its reset.
    term = env.action_manager.get_term("joint_pos")
    for spec in entry["process"]["state_fields"]:
        if _flat(getattr(term, spec["name"])) != spec["init"]:
            raise RuntimeError(f"{spec['name']}: init is not the env's starting state")

    substeps: list[dict[str, list[float]]] = []
    physics_step = env.sim.step

    def recording_step() -> None:
        substeps.append(
            {"qpos": _flat(env.sim.data.qpos[0]), "ctrl": _flat(env.sim.data.ctrl[0])}
        )
        physics_step()

    env.sim.step = recording_step
    generator = torch.Generator().manual_seed(SEED)
    steps: list[dict[str, Any]] = []
    for index in range(STEPS):
        step: dict[str, Any] = {}
        if index == RESET_AT:
            with contextlib.redirect_stdout(io.StringIO()):
                env.reset()
            step["reset_joint_pos"] = _flat(env.scene["robot"].data.joint_pos[0])
        # Past [-1, 1], so the delta's own clip matters.
        raw = torch.rand((1, len(joints)), generator=generator) * 3 - 1.5
        step["raw"] = _flat(raw)
        step["episode_length"] = int(env.episode_length_buf[0])
        substeps.clear()
        env.step(raw)
        step["substeps"] = list(substeps)
        steps.append(step)

    payload = {
        "entry": entry,
        "joints": joints,
        "decimation": env.cfg.decimation,
        "step_dt": env.step_dt,
        "steps": steps,
    }
    env.close()
    return payload


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    fixture = {
        name: _dump_case(cfg, binding, lag, OUT_DIR / name)
        for name, (cfg, binding, lag) in _cases().items()
    }
    index = OUT_DIR / "action.json"
    index.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    for name, payload in fixture.items():
        graphs = sorted(
            p.relative_to(OUT_DIR).as_posix() for p in (OUT_DIR / name).rglob("*.onnx")
        )
        print(f"{name}: {len(payload['steps'])} steps, graphs: {', '.join(graphs)}")
    print(f"wrote {index} ({index.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
