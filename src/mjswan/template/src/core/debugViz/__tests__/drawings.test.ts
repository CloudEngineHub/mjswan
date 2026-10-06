/**
 * The sensor and reward drawings against the real MuJoCo WASM: what mjlab's
 * `RayCastSensor.debug_vis` and `upright.debug_vis` would queue for the same state.
 */
import * as THREE from 'three';
import { beforeAll, describe, expect, it } from 'vitest';

import { RaycastSensor, type RaycastSensorDescriptor } from '../../onnx/raycast';
import {
  type DebugVisSensorConfig,
  RaycastDrawing,
  UprightDrawing,
  fitTerrainNormal,
  subsampleIndices,
} from '../drawings';

type MainModule = import('mujoco').MainModule;
type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

let mujoco: MainModule;

beforeAll(async () => {
  mujoco = await (await import('mujoco')).default();
});

function compile(xml: string): { model: MjModel; data: MjData } {
  const model = (mujoco as unknown as { MjModel: { from_xml_string(s: string): MjModel } })
    .MjModel.from_xml_string(xml);
  const data = new (mujoco as unknown as { MjData: new (m: MjModel) => MjData }).MjData(model);
  mujoco.mj_forward(model, data);
  return { model, data };
}

/** Floor at z=0, a 0.5 m block at x=1, a robot body at z=1. */
const SCENE = `<mujoco><worldbody>
  <geom type="plane" size="5 5 0.1"/>
  <geom type="box" size="0.25 0.25 0.25" pos="1 0 0.25"/>
  <body name="robot/pelvis" pos="0 0 1"><joint type="free"/><geom size="0.1"/></body>
</worldbody></mujoco>`;

const RAYS: RaycastSensorDescriptor = {
  kind: 'raycast',
  // Floor, block top, and one off the plane's edge.
  local_offsets: [
    [0, 0, 0],
    [1, 0, 0],
    [9, 0, 0],
  ],
  local_directions: [
    [0, 0, -1],
    [0, 0, -1],
    [0, 0, -1],
  ],
  frames: [{ type: 'body', name: 'robot/pelvis' }],
  ray_alignment: 'yaw',
  max_distance: 5,
  exclude_parent_body: true,
};

function sensorConfig(viz: Partial<DebugVisSensorConfig['viz']> = {}): DebugVisSensorConfig {
  return {
    ...RAYS,
    debug_vis: true,
    viz: {
      hit_color: [0, 1, 0, 0.8],
      miss_color: [1, 0, 0, 0.4],
      hit_sphere_color: [0, 1, 1, 1],
      hit_sphere_radius: 0.5,
      show_rays: false,
      show_normals: false,
      normal_color: [1, 1, 0, 1],
      normal_length: 5,
      ...viz,
    },
  };
}

function counts(root: THREE.Object3D, name: string): { arrows: number; spheres: number } {
  const arrows = root.getObjectByName(`${name}-arrows`)!.children[0] as THREE.InstancedMesh;
  const spheres = root.getObjectByName(`${name}-hits`) as THREE.InstancedMesh | undefined;
  return { arrows: arrows.count, spheres: spheres?.count ?? 0 };
}

describe('fitTerrainNormal', () => {
  const slope = (points: number[][]) => points.flat();

  it("recovers a slope's upward normal, whichever way the fit's vector points", () => {
    // z = 0.5 x: normal ∝ (-0.5, 0, 1).
    const points = slope([
      [0, 0, 0],
      [1, 0, 0.5],
      [0, 1, 0],
      [1, 1, 0.5],
      [2, -1, 1],
    ]);
    const normal = fitTerrainNormal(points, [true, true, true, true, true]);
    const norm = Math.hypot(0.5, 1);
    expect(normal[0]).toBeCloseTo(-0.5 / norm, 6);
    expect(normal[1]).toBeCloseTo(0, 6);
    expect(normal[2]).toBeCloseTo(1 / norm, 6);
  });

  it('falls back to world up for fewer than three valid points, or collinear ones', () => {
    const points = slope([
      [0, 0, 0],
      [1, 0, 0.5],
      [0, 1, 0],
    ]);
    expect(fitTerrainNormal(points, [true, true, false])).toEqual([0, 0, 1]);
    const line = slope([
      [0, 0, 0],
      [1, 0, 0.5],
      [2, 0, 1],
    ]);
    expect(fitTerrainNormal(line, [true, true, true])).toEqual([0, 0, 1]);
  });
});

describe('subsampleIndices', () => {
  it("is torch's linspace(0, N-1, 32).long(), and every index below 32 rays", () => {
    expect(subsampleIndices(200)).toEqual([
      0, 6, 12, 19, 25, 32, 38, 44, 51, 57, 64, 70, 77, 83, 89, 96, 102, 109, 115, 121, 128,
      134, 141, 147, 154, 160, 166, 173, 179, 186, 192, 199,
    ]);
    expect(subsampleIndices(11)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  });
});

describe('RaycastDrawing', () => {
  it('marks each hit with a sphere sized by meansize, and draws nothing hidden', () => {
    const { model, data } = compile(SCENE);
    const root = new THREE.Group();
    const drawing = new RaycastDrawing('scan', sensorConfig(), new RaycastSensor(mujoco, RAYS), 0.2, root);
    drawing.update(true, model, data);
    // Spheres only by default, as mjlab's VizCfg has show_rays off.
    expect(counts(root, 'mjswan-sensor-scan')).toEqual({ arrows: 0, spheres: 2 });
    const spheres = root.getObjectByName('mjswan-sensor-scan-hits') as THREE.InstancedMesh;
    const matrix = new THREE.Matrix4();
    spheres.getMatrixAt(1, matrix);
    const position = new THREE.Vector3().setFromMatrixPosition(matrix);
    // The block top at (1, 0, 0.5), in three's y-up frame; radius 0.5 * meansize.
    expect(position.toArray().map((v) => +v.toFixed(6))).toEqual([1, 0.5, 0]);
    expect(new THREE.Vector3().setFromMatrixScale(matrix).x).toBeCloseTo(0.1, 6);

    drawing.update(false, model, data);
    expect(counts(root, 'mjswan-sensor-scan')).toEqual({ arrows: 0, spheres: 0 });
  });

  it('adds a ray per cast and a normal per hit when the task asks for them', () => {
    const { model, data } = compile(SCENE);
    const root = new THREE.Group();
    const config = sensorConfig({ show_rays: true, show_normals: true });
    const drawing = new RaycastDrawing('scan', config, new RaycastSensor(mujoco, RAYS), 0.2, root);
    drawing.update(true, model, data);
    expect(counts(root, 'mjswan-sensor-scan')).toEqual({ arrows: 3 + 2, spheres: 2 });
    drawing.dispose();
    expect(root.children).toHaveLength(0);
  });
});

describe('UprightDrawing', () => {
  const SLOPE = `<mujoco><worldbody>
    <geom type="box" size="3 3 0.1" pos="0 0 -0.1" euler="0 20 0"/>
    <body name="robot/trunk" pos="0 0 1"><joint type="free"/><geom size="0.05"/></body>
  </worldbody></mujoco>`;
  const GRID: RaycastSensorDescriptor = {
    ...RAYS,
    frames: [{ type: 'body', name: 'robot/trunk' }],
    local_offsets: [-0.5, 0, 0.5].flatMap((x) => [-0.5, 0, 0.5].map((y) => [x, y, 0])),
    local_directions: Array.from({ length: 9 }, () => [0, 0, -1]),
  };

  /** Each arrow's direction, back in MuJoCo coordinates (three's +y is MuJoCo's +z). */
  function arrowDirections(root: THREE.Object3D): THREE.Vector3[] {
    const shafts = root.getObjectByName('mjswan-reward-upright-arrows')!
      .children[0] as THREE.InstancedMesh;
    const matrix = new THREE.Matrix4();
    return Array.from({ length: shafts.count }, (_, i) => {
      shafts.getMatrixAt(i, matrix);
      const d = new THREE.Vector3(0, 1, 0).applyMatrix4(new THREE.Matrix4().extractRotation(matrix));
      return new THREE.Vector3(d.x, -d.z, d.y);
    });
  }

  it('draws the fitted slope normal and the body up over the root', () => {
    const { model, data } = compile(SLOPE);
    const root = new THREE.Group();
    const drawing = new UprightDrawing(
      'upright',
      { kind: 'upright', root_body: 'robot/trunk', body: null, terrain_sensors: ['scan'] },
      [new RaycastSensor(mujoco, GRID)],
      root,
    );
    drawing.update(true, model, data);
    const [terrain, bodyUp] = arrowDirections(root);
    const angle = (20 * Math.PI) / 180;
    expect(terrain.x).toBeCloseTo(Math.sin(angle), 4);
    expect(terrain.z).toBeCloseTo(Math.cos(angle), 4);
    expect(bodyUp.toArray().map((v) => +v.toFixed(6))).toEqual([0, 0, 1]);
  });

  it('draws nothing for a task that rates tilt against world up', () => {
    const { model, data } = compile(SLOPE);
    const root = new THREE.Group();
    const drawing = new UprightDrawing(
      'upright',
      { kind: 'upright', root_body: 'robot/trunk', body: null, terrain_sensors: null },
      [],
      root,
    );
    drawing.update(true, model, data);
    expect(arrowDirections(root)).toHaveLength(0);
  });
});
