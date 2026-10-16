"""Task-defined action terms shaped like the ones ADR 0009 targets, on a 3-joint arm.

``DeltaAction`` is in-hand-rotation-mjlab's ``JointPositionDeltaAction``, trimmed;
``HeldAction`` is luwu_mjlab's ``HeldJointPositionAction``; ``EmaAction`` smooths the
target the way wuji's EMA terms do, inheriting mjlab's apply. The rest each break one
rule the tracer enforces.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import mujoco
import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import (
    BaseAction,
    BaseActionCfg,
    JointPositionAction,
)
from mjlab.scene import SceneCfg

ARM = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="l1" pos="0 0 0.2">
        <joint name="j1" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" size="0.03 0.1"/>
        <body name="l2" pos="0 0 0.2">
          <joint name="j2" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" size="0.03 0.1"/>
          <body name="l3" pos="0 0 0.2">
            <joint name="j3" type="hinge" axis="1 0 0" range="-0.5 0.5"/>
            <geom type="capsule" size="0.03 0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

HOME = {"j1": 0.2, "j2": -0.3, "j3": 0.1}


@dataclass(kw_only=True)
class DeltaActionCfg(BaseActionCfg):
    delta: float = 1 / 24
    interpolate_decimation: bool = True

    def __post_init__(self):
        self.transmission_type = TransmissionType.JOINT

    def build(self, env):
        return DeltaAction(self, env)


class DeltaAction(BaseAction):
    """``target += clip(raw) * delta`` within the soft limits, started at the measured
    position, ramped over the substeps."""

    def __init__(self, cfg, env):
        super().__init__(cfg=cfg, env=env)
        self._target = torch.zeros_like(self._raw_actions)
        self._prev_target = torch.zeros_like(self._raw_actions)
        self._substep_counter = 0
        self._decimation = env.cfg.decimation
        self._initialize_target(slice(None))

    def _clamp(self, value, env_ids):
        limits = self._entity.data.soft_joint_pos_limits[env_ids][:, self._target_ids]
        return torch.clamp(value, min=limits[..., 0], max=limits[..., 1])

    def _initialize_target(self, env_ids):
        current = self._entity.data.joint_pos[env_ids][:, self._target_ids]
        self._target[env_ids] = self._clamp(current, env_ids)
        self._processed_actions[env_ids] = self._target[env_ids]
        self._prev_target[env_ids] = self._target[env_ids]

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._prev_target[:] = self._target
        self._substep_counter = 0
        delta = torch.clamp(self._raw_actions, -1.0, 1.0) * self.cfg.delta
        self._target = self._clamp(self._target + delta, slice(None))
        self._processed_actions = self._target

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._initialize_target(slice(None) if env_ids is None else env_ids)

    def apply_actions(self):
        alpha = (self._substep_counter + 1) / self._decimation
        target = (1 - alpha) * self._prev_target + alpha * self._processed_actions
        self._substep_counter += 1
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


@dataclass(kw_only=True)
class HeldActionCfg(JointPositionActionCfg):
    hold_duration_s: float = 1.0

    def build(self, env):
        return HeldAction(self, env)


class HeldAction(JointPositionAction):
    """Each joint held where the reset left it for ``hold_duration_s``."""

    def process_actions(self, actions):
        super().process_actions(actions)
        elapsed = self._env.episode_length_buf * self._env.step_dt
        hold_mask = elapsed < self.cfg.hold_duration_s
        if not hold_mask.any():
            return
        self._raw_actions[hold_mask] = 0.0
        self._processed_actions[hold_mask] = (
            self._raw_actions[hold_mask] * self._scale + self._offset[hold_mask]
        )

    def apply_actions(self):
        elapsed = self._env.episode_length_buf * self._env.step_dt
        if (elapsed < self.cfg.hold_duration_s).any():
            target = self._entity.data.joint_pos[:, self._target_ids].clone()
        else:
            bias = self._entity.data.encoder_bias[:, self._target_ids]
            target = self._processed_actions - bias
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


@dataclass(kw_only=True)
class EmaActionCfg(JointPositionActionCfg):
    alpha: float = 0.3

    def build(self, env):
        return EmaAction(self, env)


class EmaAction(JointPositionAction):
    """mjlab's target, smoothed; resets to the default pose."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ema = self._offset.clone()

    def process_actions(self, actions):
        super().process_actions(actions)
        alpha = self.cfg.alpha
        self._ema = alpha * self._processed_actions + (1 - alpha) * self._ema
        self._processed_actions = self._ema.clone()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._ema[env_ids] = self._offset[env_ids]


@dataclass(kw_only=True)
class CountingActionCfg(JointPositionActionCfg):
    def build(self, env):
        return CountingAction(self, env)


class CountingAction(JointPositionAction):
    """Counts steps in a Python int, which a graph cannot carry."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._steps = 0

    def process_actions(self, actions):
        super().process_actions(actions)
        self._steps += 1


@dataclass(kw_only=True)
class WritingActionCfg(JointPositionActionCfg):
    def build(self, env):
        return WritingAction(self, env)


class WritingAction(JointPositionAction):
    """Sets the sim's target in process_actions, which belongs in apply_actions."""

    def process_actions(self, actions):
        super().process_actions(actions)
        self._entity.set_joint_position_target(self._processed_actions)


@dataclass(kw_only=True)
class StatelessDeltaActionCfg(JointPositionActionCfg):
    def build(self, env):
        return StatelessDeltaAction(self, env)


class StatelessDeltaAction(JointPositionAction):
    """Overrides apply_actions without a binding naming what it does."""

    def apply_actions(self):
        super().apply_actions()


def arm_env(
    actions: dict, *, decimation: int = 4, delay_lag: int = 0
) -> ManagerBasedRlEnv:
    """The arm on position actuators at 200 Hz, its terms *actions*, reset once.

    *delay_lag* delays every actuator's command by that many physics steps.
    """
    entity = EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_string(ARM),
        init_state=EntityCfg.InitialStateCfg(joint_pos=HOME),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*",),
                    stiffness=20.0,
                    damping=1.0,
                    effort_limit=10.0,
                    delay_min_lag=delay_lag,
                    delay_max_lag=delay_lag,
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
    )
    cfg = ManagerBasedRlEnvCfg(
        decimation=decimation,
        scene=SceneCfg(num_envs=1, entities={"robot": entity}),
        actions=actions,
        episode_length_s=20.0,
    )
    cfg.sim.mujoco.timestep = 0.005
    with contextlib.redirect_stdout(io.StringIO()):
        env = ManagerBasedRlEnv(cfg, device="cpu")
        env.reset()
    return env


def on_arm(cls, **kwargs):
    """A *cls* config driving every joint of the arm."""
    return cls(entity_name="robot", actuator_names=(".*",), **kwargs)
