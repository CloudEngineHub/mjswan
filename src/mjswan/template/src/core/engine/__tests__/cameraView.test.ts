/**
 * The spherical camera pose, read as MuJoCo's free camera reads it: azimuth and elevation
 * are the direction the camera looks, so an mjlab `ViewerConfig` opens on the same side.
 */
import { describe, expect, it } from 'vitest';

import { threeToMjcCoordinate } from '../../scene/coordinate';
import { cameraAngles, computeCameraPosition } from '../viewer_config';

/** Where MuJoCo puts the camera, in MuJoCo coordinates. */
function mujocoPosition(distance: number, elevation: number, azimuth: number): number[] {
  return threeToMjcCoordinate(computeCameraPosition([0, 0, 0], distance, elevation, azimuth)).toArray();
}

describe('computeCameraPosition', () => {
  it('places the camera where MuJoCo puts its free camera', () => {
    // mjv_updateScene with Go1's mjlab view (distance 1.5, elevation -10, azimuth 90).
    const [x, y, z] = mujocoPosition(1.5, -10, 90);
    expect(x).toBeCloseTo(0, 6);
    expect(y).toBeCloseTo(-1.477, 3);
    expect(z).toBeCloseTo(0.26, 3);
  });

  it('looks along +x at azimuth 0, from behind a robot that faces it', () => {
    const [x, y] = mujocoPosition(2, 0, 0);
    expect(x).toBeCloseTo(-2, 6);
    expect(y).toBeCloseTo(0, 6);
  });

  it('offsets from the look-at point', () => {
    const position = threeToMjcCoordinate(computeCameraPosition([1, 2, 3], 2, 0, 0));
    expect(position.toArray()).toEqual([expect.closeTo(-1), expect.closeTo(2), expect.closeTo(3)]);
  });
});

describe('cameraAngles', () => {
  it('reads back the pose computeCameraPosition placed', () => {
    for (const [elevation, azimuth] of [
      [-10, 90],
      [-30, -135],
      [20, 170],
      [-45, 0],
    ]) {
      const offset = threeToMjcCoordinate(computeCameraPosition([0, 0, 0], 3, elevation, azimuth));
      const angles = cameraAngles(offset);
      expect(angles.distance).toBeCloseTo(3, 6);
      expect(angles.elevation).toBeCloseTo(elevation, 6);
      expect(angles.azimuth).toBeCloseTo(azimuth, 6);
    }
  });
});
