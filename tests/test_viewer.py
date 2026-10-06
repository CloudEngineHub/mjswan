"""ViewerConfig's spherical pose is MuJoCo's free camera's, so mjlab's values carry over."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mjswan.mjlab.task import adapt_viewer_config
from mjswan.viewer import ViewerConfig

_MODEL = mujoco.MjModel.from_xml_string(
    "<mujoco><worldbody><geom size='0.1'/></worldbody></mujoco>"
)


def _mujoco_camera_position(lookat, distance, elevation, azimuth) -> np.ndarray:
    """Where ``mjv_updateScene`` puts the free camera: midway between its two eyes."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance, camera.elevation, camera.azimuth = distance, elevation, azimuth
    scene = mujoco.MjvScene(_MODEL, maxgeom=10)
    mujoco.mjv_updateScene(
        _MODEL,
        mujoco.MjData(_MODEL),
        mujoco.MjvOption(),
        mujoco.MjvPerturb(),
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    return (scene.camera[0].pos + scene.camera[1].pos) / 2


@pytest.mark.parametrize(
    ("elevation", "azimuth"), [(-10.0, 90.0), (-30.0, -135.0), (20.0, 170.0)]
)
def test_from_position_inverts_mujocos_free_camera(elevation, azimuth):
    lookat = (0.5, -1.0, 0.3)
    position = _mujoco_camera_position(lookat, 1.5, elevation, azimuth)

    config = ViewerConfig.from_position(tuple(position), lookat)

    assert config.distance == pytest.approx(1.5)
    assert config.elevation == pytest.approx(elevation)
    assert config.azimuth == pytest.approx(azimuth)


def test_an_mjlab_viewer_config_keeps_its_azimuth():
    mjlab_viewer = SimpleNamespace(distance=1.5, elevation=-10.0, azimuth=90.0)
    adapted = adapt_viewer_config(mjlab_viewer)
    assert adapted is not None
    assert (adapted.elevation, adapted.azimuth) == (-10.0, 90.0)
