/**
 * `RaycastSensor` against the real MuJoCo WASM.
 *
 * mjlab's height scan is the one sensor mjswan has to *recompute* rather than read
 * — its data is ray hits, not a `sensordata` window — so the numbers are only as
 * good as this file's reproduction of `RayCastSensor._compute_data`. A fake model
 * would only prove the arithmetic agrees with itself; these load the actual WASM
 * and cast into a scene whose geometry gives every distance a known answer.
 */
import { beforeAll, describe, expect, it } from 'vitest';

import { RaycastSensor, isRaycastField, type RaycastSensorDescriptor } from '../raycast';

type MainModule = import('mujoco').MainModule;

/** Flat floor at z=0 with a 0.5 m block centred at x=1, and a robot body at z=1. */
const SCENE = `<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"/>
    <geom name="block" type="box" size="0.25 0.25 0.25" pos="1 0 0.25"/>
    <body name="robot/pelvis" pos="0 0 1">
      <joint name="free" type="free"/>
      <geom name="torso" type="sphere" size="0.1"/>
      <site name="robot/imu" pos="0 0 0.2"/>
    </body>
  </worldbody>
</mujoco>`;

/** A 3-ray line along +x at 1 m spacing, pointing straight down. */
function descriptor(over: Partial<RaycastSensorDescriptor> = {}): RaycastSensorDescriptor {
  return {
    kind: 'raycast',
    local_offsets: [
      [0, 0, 0],
      [1, 0, 0],
      [4.9, 0, 0],
    ],
    local_directions: [
      [0, 0, -1],
      [0, 0, -1],
      [0, 0, -1],
    ],
    frames: [{ type: 'body', name: 'robot/pelvis' }],
    ray_alignment: 'yaw',
    max_distance: 5.0,
    exclude_parent_body: true,
    ...over,
  };
}

let mujoco: MainModule;
let mjModel: import('mujoco').MjModel;
let mjData: import('mujoco').MjData;

beforeAll(async () => {
  const load = (await import('mujoco')).default;
  mujoco = await load();
  mjModel = (mujoco as unknown as { MjModel: { from_xml_string(s: string): never } })
    .MjModel.from_xml_string(SCENE);
  mjData = new (mujoco as unknown as { MjData: new (m: unknown) => never }).MjData(
    mjModel,
  );
  mujoco.mj_forward(mjModel, mjData);
});

describe('RaycastSensor', () => {
  it('measures distance to the surface under each ray', () => {
    const sensor = new RaycastSensor(mujoco, descriptor());
    const distances = sensor.read('distances', mjModel, mjData)!;
    // Frame at z=1: ray 0 hits the floor (1.0), ray 1 the block top (0.5), ray 2 floor.
    expect(Array.from(distances)).toEqual([1, 0.5, 1]);
  });

  it('excludes the frame’s own body, so a ray cannot hit the robot', () => {
    // The torso sphere sits on ray 0's origin, so without the exclusion the reading is
    // 0.1 rather than the floor's 1.0.
    const included = new RaycastSensor(mujoco, descriptor({ exclude_parent_body: false }));
    expect(included.read('distances', mjModel, mjData)![0]).toBeCloseTo(0.1, 6);
    const excluded = new RaycastSensor(mujoco, descriptor());
    expect(excluded.read('distances', mjModel, mjData)![0]).toBeCloseTo(1.0, 6);
  });

  it('reports a miss as -1, including a hit beyond max_distance', () => {
    // Off the edge of the 5 m plane: nothing to hit at all.
    const offEdge = new RaycastSensor(
      mujoco,
      descriptor({ local_offsets: [[9, 0, 0]], local_directions: [[0, 0, -1]] }),
    );
    expect(offEdge.read('distances', mjModel, mjData)![0]).toBe(-1);

    // A real hit at 1.0 beyond a 0.5 m reach: mjlab folds it into the same -1.
    const shortRange = new RaycastSensor(mujoco, descriptor({ max_distance: 0.5 }));
    expect(Array.from(shortRange.read('distances', mjModel, mjData)!)).toEqual([
      -1, 0.5, -1,
    ]);
  });

  it('collapses a missed ray’s hit position onto its origin', () => {
    const sensor = new RaycastSensor(mujoco, descriptor());
    const hits = Array.from(sensor.read('hit_pos_w', mjModel, mjData)!);
    // Hits land on the surface...
    expect(hits.slice(0, 3)).toEqual([0, 0, 0]);
    expect(hits.slice(3, 6)).toEqual([1, 0, 0.5]);

    const missed = new RaycastSensor(
      mujoco,
      descriptor({ local_offsets: [[9, 0, 0]], local_directions: [[0, 0, -1]] }),
    );
    // ...and a miss reports the origin, not a point at infinity.
    expect(Array.from(missed.read('hit_pos_w', mjModel, mjData)!)).toEqual([9, 0, 1]);
  });

  it('serves the frame pose fields', () => {
    const sensor = new RaycastSensor(mujoco, descriptor());
    expect(Array.from(sensor.read('frame_pos_w', mjModel, mjData)!)).toEqual([0, 0, 1]);
    expect(Array.from(sensor.read('pos_w', mjModel, mjData)!)).toEqual([0, 0, 1]);
    const quat = Array.from(sensor.read('quat_w', mjModel, mjData)!);
    expect(quat[0]).toBeCloseTo(1, 6); // identity orientation
  });

  it('rotates the pattern with the frame under `base`, but only by yaw under `yaw`', () => {
    // Pitch the body 90° about +y so its local +x points down: `base` tips the pattern
    // with it, `yaw` keeps the grid level — the reason a height map uses yaw.
    const q = Math.SQRT1_2;
    mjData.qpos[3] = q;
    mjData.qpos[4] = 0;
    mjData.qpos[5] = q;
    mjData.qpos[6] = 0;
    mujoco.mj_forward(mjModel, mjData);
    try {
      const yawed = new RaycastSensor(mujoco, descriptor());
      // Level grid: ray 1 is still 1 m along world x, over the block.
      expect(Array.from(yawed.read('distances', mjModel, mjData)!)).toEqual([1, 0.5, 1]);

      const based = new RaycastSensor(mujoco, descriptor({ ray_alignment: 'base' }));
      // Tipped: offsets run downward and directions along -x, unlike the level answer.
      expect(Array.from(based.read('distances', mjModel, mjData)!)).not.toEqual([
        1, 0.5, 1,
      ]);
    } finally {
      mjData.qpos[3] = 1;
      mjData.qpos[5] = 0;
      mujoco.mj_forward(mjModel, mjData);
    }
  });

  it('reports unavailable when the model has no such frame', () => {
    const sensor = new RaycastSensor(mujoco, descriptor({
      frames: [{ type: 'body', name: 'not_here' }],
    }));
    expect(sensor.read('distances', mjModel, mjData)).toBeNull();
  });

  it('casts only into the geom groups the sensor allows', () => {
    // mjlab's terrain scan is `include_geom_groups=(0,)`. Without it the rays stop on the
    // robot's own legs (group 3), which reads as a step in the ground under the robot.
    const scene = `<mujoco><worldbody>
      <geom name="floor" type="plane" size="5 5 0.1"/>
      <geom name="leg" type="box" size="0.2 0.2 0.2" pos="0 0 0.5" group="3"/>
      <body name="robot/pelvis" pos="0 0 1"><joint type="free"/>
        <geom type="sphere" size="0.1"/>
      </body>
    </worldbody></mujoco>`;
    const model = (mujoco as unknown as { MjModel: { from_xml_string(s: string): never } })
      .MjModel.from_xml_string(scene);
    const data = new (mujoco as unknown as { MjData: new (m: unknown) => never }).MjData(
      model,
    );
    mujoco.mj_forward(model, data);
    const down = descriptor({ local_offsets: [[0, 0, 0]], local_directions: [[0, 0, -1]] });

    expect(new RaycastSensor(mujoco, down).read('distances', model, data)![0]).toBeCloseTo(0.3, 3);
    const filtered = { ...down, include_geom_groups: [0] };
    expect(new RaycastSensor(mujoco, filtered).read('distances', model, data)![0]).toBeCloseTo(1.0, 3);
  });

  it('knows which fields it can serve', () => {
    expect(isRaycastField('distances')).toBe(true);
    expect(isRaycastField('hit_pos_w')).toBe(true);
    expect(isRaycastField('normals_w')).toBe(true);
    expect(isRaycastField('geom_ids')).toBe(false);
  });

  it('reads the surface normal at each hit, and zeros for a miss', () => {
    // A slope tilted 20 degrees about y: its normal leans toward +x.
    const tilted = (mujoco as unknown as { MjModel: { from_xml_string(s: string): never } })
      .MjModel.from_xml_string(`<mujoco><worldbody>
        <geom type="box" size="2 2 0.1" pos="0 0 -0.1" euler="0 20 0"/>
        <body name="robot/pelvis" pos="0 0 1"><joint type="free"/><geom size="0.05"/></body>
      </worldbody></mujoco>`);
    const data = new (mujoco as unknown as { MjData: new (m: unknown) => never }).MjData(tilted);
    mujoco.mj_forward(tilted, data);
    const sensor = new RaycastSensor(
      mujoco,
      descriptor({ local_offsets: [[0, 0, 0], [9, 0, 0]], local_directions: [[0, 0, -1], [0, 0, -1]] }),
    );
    const normals = Array.from(sensor.read('normals_w', tilted, data)!);
    const angle = (20 * Math.PI) / 180;
    expect(normals[0]).toBeCloseTo(Math.sin(angle), 5);
    expect(normals[1]).toBeCloseTo(0, 5);
    expect(normals[2]).toBeCloseTo(Math.cos(angle), 5);
    expect(normals.slice(3)).toEqual([0, 0, 0]);
    sensor.dispose();
  });
});
