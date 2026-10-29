/**
 * What lies under a point of the canvas (ADR 0011), by MuJoCo's own ray: the answer the
 * simulation would give, an infinite plane and a height field included, and nothing that
 * is only drawn (the mode gizmos, the preview, a skin, a splat).
 */
import * as THREE from 'three';

import { threeToMjcCoordinate } from '../scene/coordinate';
import { geomGroupMask } from '../onnx/raycast';

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;
type MainModule = import('mujoco').MainModule;
type DoubleBuffer = import('mujoco').DoubleBuffer;
type IntBuffer = import('mujoco').IntBuffer;

/** The groups the scene draws (`geom_group < 3`): a ray does not land on what no one sees. */
const DRAWN_GROUPS = geomGroupMask([0, 1, 2]);

/** A surface hit, in MuJoCo's world axes. */
export interface PickHit {
  point: [number, number, number];
  /** The surface's normal, turned to face the camera. */
  normal: [number, number, number];
  geom: number;
  body: number;
}

/** A canvas's place on the page, as `getBoundingClientRect` gives it. */
export interface CanvasRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export class SurfacePicker {
  private readonly raycaster = new THREE.Raycaster();
  private readonly ndc = new THREE.Vector2();
  /** `mj_ray`'s outputs: the binding writes back only into its own buffers. */
  private geomId: IntBuffer | null = null;
  private normal: DoubleBuffer | null = null;

  constructor(private readonly mujoco: MainModule) {}

  /** `clientX` and `clientY` are CSS pixels, as a `PointerEvent` gives them. */
  pick(
    mjModel: MjModel,
    mjData: MjData,
    camera: THREE.Camera,
    rect: CanvasRect,
    clientX: number,
    clientY: number
  ): PickHit | null {
    if (!rect.width || !rect.height) return null;
    this.ndc.set(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );
    this.raycaster.setFromCamera(this.ndc, camera);
    const origin = threeToMjcCoordinate(this.raycaster.ray.origin);
    const direction = threeToMjcCoordinate(this.raycaster.ray.direction);

    this.geomId ??= new this.mujoco.IntBuffer(1);
    this.normal ??= new this.mujoco.DoubleBuffer(3);
    const distance = this.mujoco.mj_ray(
      mjModel,
      mjData,
      [origin.x, origin.y, origin.z],
      [direction.x, direction.y, direction.z],
      DRAWN_GROUPS as unknown as number[],
      true,
      -1,
      this.geomId,
      this.normal
    );
    if (distance < 0) return null;

    const geom = this.geomId.GetView()[0] as number;
    const n = this.normal.GetView() as ArrayLike<number>;
    const facing = n[0] * direction.x + n[1] * direction.y + n[2] * direction.z > 0 ? -1 : 1;
    return {
      point: [
        origin.x + direction.x * distance,
        origin.y + direction.y * distance,
        origin.z + direction.z * distance,
      ],
      normal: [n[0] * facing, n[1] * facing, n[2] * facing],
      geom,
      body: mjModel.geom_bodyid[geom],
    };
  }

  dispose(): void {
    this.geomId?.delete();
    this.normal?.delete();
    this.geomId = null;
    this.normal = null;
  }
}
