/** The Debug Viz drawings that are not command terms: raycast sensors and the `upright` reward. */

import * as THREE from 'three';

import type { RaycastSensor, RaycastSensorDescriptor } from '../onnx/raycast';
import { ArrowBatch, SphereBatch, type Rgba } from './primitives';

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

/** mjlab's `RayCastSensorCfg.VizCfg`. */
export interface RaycastVizConfig {
  hit_color: Rgba;
  miss_color: Rgba;
  hit_sphere_color: Rgba;
  /** In units of the model's `meansize`, as is `normal_length`. */
  hit_sphere_radius: number;
  show_rays: boolean;
  show_normals: boolean;
  normal_color: Rgba;
  normal_length: number;
}

export interface DebugVisSensorConfig extends RaycastSensorDescriptor {
  /** Listed in Debug Viz; a sensor only a reward drawing reads is shipped unlisted. */
  debug_vis: boolean;
  viz: RaycastVizConfig;
}

/** mjlab's `upright` reward term: it rates `body`, or `root_body` when null. */
export interface UprightDrawingConfig {
  kind: 'upright';
  root_body: string;
  body: string | null;
  /** Null when the task rates tilt against world up, which mjlab does not draw. */
  terrain_sensors: string[] | null;
}

/** The MDP entry's `debug_vis` block. */
export interface DebugVisConfig {
  meansize: number;
  sensors?: Record<string, DebugVisSensorConfig>;
  rewards?: Record<string, UprightDrawingConfig>;
}

export interface Drawing {
  /** The mjlab term's `_debug_vis_enabled`. */
  enabled: boolean;
  update(visible: boolean, mjModel: MjModel, mjData: MjData): void;
  dispose(): void;
}

/** mjlab's `RayCastSensor.debug_vis`. */
export class RaycastDrawing implements Drawing {
  enabled = true;
  private readonly arrows: ArrowBatch;
  private readonly spheres: SphereBatch;

  constructor(
    name: string,
    private readonly config: DebugVisSensorConfig,
    private readonly caster: RaycastSensor,
    private readonly meansize: number,
    root: THREE.Object3D,
  ) {
    const rays = config.frames.length * config.local_offsets.length;
    this.arrows = new ArrowBatch(rays * 2, `mjswan-sensor-${name}-arrows`);
    this.spheres = new SphereBatch(rays, `mjswan-sensor-${name}-hits`, config.viz.hit_sphere_color[3]);
    root.add(this.arrows.object, this.spheres.object);
  }

  update(visible: boolean, mjModel: MjModel, mjData: MjData): void {
    this.arrows.clear();
    this.spheres.clear();
    const viz = this.config.viz;
    const trace = visible ? this.caster.trace(mjModel, mjData, viz.show_normals) : null;
    if (trace) {
      const rayWidth = 0.1 * this.meansize;
      const sphereRadius = viz.hit_sphere_radius * this.meansize;
      const normalLength = viz.normal_length * this.meansize;
      const missExtent = Math.min(0.5, this.config.max_distance * 0.05);
      for (let i = 0; i < trace.distances.length; i++) {
        const origin = trace.origins.subarray(i * 3, i * 3 + 3);
        const hit = trace.distances[i] >= 0;
        const end = hit
          ? trace.hitPos.subarray(i * 3, i * 3 + 3)
          : [0, 1, 2].map((k) => origin[k] + trace.directions[i * 3 + k] * missExtent);
        if (viz.show_rays) {
          this.arrows.add(origin, end, hit ? viz.hit_color : viz.miss_color, rayWidth);
        }
        if (!hit) continue;
        this.spheres.add(end, sphereRadius, viz.hit_sphere_color);
        if (trace.normals) {
          const tip = [0, 1, 2].map((k) => end[k] + trace.normals![i * 3 + k] * normalLength);
          this.arrows.add(end, tip, viz.normal_color, rayWidth);
        }
      }
    }
    this.arrows.commit();
    this.spheres.commit();
  }

  dispose(): void {
    this.arrows.dispose();
    this.spheres.dispose();
  }
}

/** Eigenvalues (ascending) and unit eigenvectors of a symmetric 3x3, by Jacobi rotations. */
function symmetricEigen3(m: number[][]): { values: number[]; vectors: number[][] } {
  const a = m.map((row) => row.slice());
  const v = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let sweep = 0; sweep < 50; sweep++) {
    const off = a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2;
    if (off < 1e-30) break;
    for (const [p, q] of [
      [0, 1],
      [0, 2],
      [1, 2],
    ]) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[k][p];
        const akq = a[k][q];
        a[k][p] = c * akp - s * akq;
        a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p][k];
        const aqk = a[q][k];
        a[p][k] = c * apk - s * aqk;
        a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k][p];
        const vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq;
        v[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const order = [0, 1, 2].sort((i, j) => a[i][i] - a[j][j]);
  return {
    values: order.map((i) => a[i][i]),
    vectors: order.map((i) => [v[0][i], v[1][i], v[2][i]]),
  };
}

const FLOAT32_EPS = 1.1920928955078125e-7;

/**
 * mjlab's `fit_terrain_normal`: the plane normal of the valid points, oriented up, or
 * world up when there are fewer than three or they do not span a plane.
 */
export function fitTerrainNormal(points: ArrayLike<number>, valid: ArrayLike<boolean>): number[] {
  const up = [0, 0, 1];
  const n = valid.length;
  let count = 0;
  const centroid = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    if (!valid[i]) continue;
    count += 1;
    for (let k = 0; k < 3; k++) centroid[k] += points[i * 3 + k];
  }
  if (count < 3) return up;
  for (let k = 0; k < 3; k++) centroid[k] /= count;
  const cov = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < n; i++) {
    if (!valid[i]) continue;
    const d = [0, 1, 2].map((k) => points[i * 3 + k] - centroid[k]);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) cov[r][c] += d[r] * d[c];
  }
  if (!cov.flat().every(Number.isFinite)) return up;
  const { values, vectors } = symmetricEigen3(cov);
  const planeLike = values[0] / Math.max(values[1], FLOAT32_EPS) < 0.1;
  const hasSpread = values[1] > Math.max(values[2], FLOAT32_EPS) * 1e-6;
  if (!planeLike || !hasSpread) return up;
  const normal = vectors[0];
  const norm = Math.max(Math.hypot(normal[0], normal[1], normal[2]), 1e-8);
  const sign = normal[2] < 0 ? -1 : 1;
  return normal.map((x) => (sign * x) / norm);
}

/** mjlab's `terrain_normal_from_sensors` subsampling: `linspace(0, N-1, 32).long()`. */
export function subsampleIndices(total: number, maxPoints = 32): number[] {
  if (total <= maxPoints) return Array.from({ length: total }, (_, i) => i);
  return Array.from({ length: maxPoints }, (_, i) =>
    Math.trunc((i * (total - 1)) / (maxPoints - 1)),
  );
}

function findBody(mjModel: MjModel, wanted: string): number {
  const bare = wanted.slice(wanted.lastIndexOf('/') + 1);
  let fallback = -1;
  for (let b = 0; b < mjModel.nbody; b++) {
    const name = mjModel.body(b).name;
    if (name === wanted) return b;
    if (fallback < 0 && (name === bare || name.endsWith(`/${bare}`))) fallback = b;
  }
  return fallback;
}

const TERRAIN_NORMAL_COLOR: Rgba = [0.8, 0.2, 0.8, 0.8];
const BODY_UP_COLOR: Rgba = [1.0, 0.5, 0.0, 0.8];

/** mjlab's `upright.debug_vis`: terrain normal and body up. */
export class UprightDrawing implements Drawing {
  enabled = true;
  private readonly arrows: ArrowBatch;
  private ids: { model: MjModel; root: number; body: number } | null = null;

  constructor(
    name: string,
    private readonly config: UprightDrawingConfig,
    private readonly sensors: RaycastSensor[],
    root: THREE.Object3D,
  ) {
    this.arrows = new ArrowBatch(2, `mjswan-reward-${name}-arrows`);
    root.add(this.arrows.object);
  }

  private bodies(mjModel: MjModel): { root: number; body: number } {
    if (this.ids?.model !== mjModel) {
      const root = findBody(mjModel, this.config.root_body);
      const body = this.config.body ? findBody(mjModel, this.config.body) : root;
      this.ids = { model: mjModel, root, body };
    }
    return this.ids;
  }

  update(visible: boolean, mjModel: MjModel, mjData: MjData): void {
    this.arrows.clear();
    const { root, body } = this.bodies(mjModel);
    if (visible && this.config.terrain_sensors && root >= 0 && body >= 0) {
      const points: number[] = [];
      const valid: boolean[] = [];
      for (const sensor of this.sensors) {
        const trace = sensor.trace(mjModel, mjData);
        if (!trace) continue;
        for (const i of subsampleIndices(trace.distances.length)) {
          points.push(trace.hitPos[i * 3], trace.hitPos[i * 3 + 1], trace.hitPos[i * 3 + 2]);
          valid.push(trace.distances[i] >= 0);
        }
      }
      const normal = fitTerrainNormal(points, valid);
      const q = mjData.xquat;
      const [w, x, y, z] = [q[body * 4], q[body * 4 + 1], q[body * 4 + 2], q[body * 4 + 3]];
      // The body's z-axis in world coordinates: `quat_apply(quat, [0, 0, 1])`.
      const bodyUp = [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)];
      const pos = mjData.xpos.subarray(root * 3, root * 3 + 3);
      const origin = [pos[0], pos[1] + 0.3, pos[2]];
      const scale = 0.25;
      const tip = (v: number[]) => origin.map((o, k) => o + v[k] * scale);
      this.arrows.add(origin, tip(normal), TERRAIN_NORMAL_COLOR, 0.01);
      this.arrows.add(origin, tip(bodyUp), BODY_UP_COLOR, 0.01);
    }
    this.arrows.commit();
  }

  dispose(): void {
    this.arrows.dispose();
  }
}
