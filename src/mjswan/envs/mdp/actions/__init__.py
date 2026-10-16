"""Action configuration classes for mjswan.

Mirrors ``mjlab.envs.mdp.actions`` so that the following import pattern
translates directly::

    # mjlab
    from mjlab.envs.mdp.actions import JointPositionActionCfg

    # mjswan (identical)
    from mjswan.envs.mdp.actions import JointPositionActionCfg
"""

from .actions import (
    ActionTermCfg,
    BaseActionCfg,
    CommandJointPositionActionCfg,
    JointEffortActionCfg,
    JointPositionActionCfg,
    JointVelocityActionCfg,
    MuscleActivationActionCfg,
    ReferenceJointPositionActionCfg,
    RelativeJointPositionActionCfg,
    SiteEffortActionCfg,
    TendonEffortActionCfg,
    TendonLengthActionCfg,
    TendonVelocityActionCfg,
    TracedActionCfg,
)

__all__ = [
    "ActionTermCfg",
    "BaseActionCfg",
    "JointPositionActionCfg",
    "JointVelocityActionCfg",
    "ReferenceJointPositionActionCfg",
    "RelativeJointPositionActionCfg",
    "CommandJointPositionActionCfg",
    "JointEffortActionCfg",
    "MuscleActivationActionCfg",
    "TendonLengthActionCfg",
    "TendonVelocityActionCfg",
    "TendonEffortActionCfg",
    "SiteEffortActionCfg",
    "TracedActionCfg",
]
