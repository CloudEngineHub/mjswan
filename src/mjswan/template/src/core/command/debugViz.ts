/**
 * The browser half of mjlab's `_debug_vis_impl`: the build ships the primitives that
 * method would have drawn, and this evaluates them for every command alike.
 *
 * Vectors are evaluated in MuJoCo coordinates, frame transform included, and converted
 * to three's only at the end — the arithmetic stays mjlab's.
 */

import * as THREE from 'three';
import type { SlotReader } from '../onnx/session';
import { mjcToThreeCoordinate } from '../scene/coordinate';
import { unitHead, unitShaft, unitSphere, viserMaterial } from '../debugViz/primitives';

/** A 3-vector source: exactly one of `const` / `state` / `field`. */
export interface VizVector {
  const?: number[];
  /** A `state_fields` entry on the owning term. */
  state?: string;
  /** An entity data field (with `entity`), read through the slot reader. */
  entity?: string;
  field?: string;
  /** Which source component feeds each output axis; `null` zeroes that axis. */
  components?: (number | null)[];
  scale?: number;
}

/** The entity frame a primitive's vectors are expressed in; absent means world. */
export interface VizFrame {
  entity: string;
  pos_field: string;
  quat_field: string;
}

export interface VizPrimitive {
  shape: 'sphere' | 'arrow';
  /** RGBA, each in [0, 1]. */
  color: [number, number, number, number];
  radius?: number;
  /** Arrow shaft width, as mjlab's `add_arrow`. */
  width?: number;
  frame?: VizFrame;
  origin: VizVector;
  /** Arrow only: the tip's offset from `origin`, in frame coordinates. */
  vector?: VizVector;
}

/** Reads one of the owning term's state fields. */
export type StateReader = (field: string) => Float32Array | null;

const UP = new THREE.Vector3(0, 1, 0);
const HEAD_FRACTION = 0.2;

function resolve(
  vec: VizVector,
  state: StateReader,
  readSlot: SlotReader | undefined,
): Float32Array | null {
  let source: ArrayLike<number> | null = null;
  if (vec.const) source = vec.const;
  else if (vec.state) source = state(vec.state);
  else if (vec.field) source = readSlot?.({ entity: vec.entity ?? null, field: vec.field }) ?? null;
  if (!source) return null;

  const scale = vec.scale ?? 1;
  const out = new Float32Array(3);
  for (let axis = 0; axis < 3; axis++) {
    const index = vec.components ? vec.components[axis] : axis;
    if (index === null || index === undefined) continue;
    out[axis] = (source[index] ?? 0) * scale;
  }
  return out;
}

/** Frame-local (or world) MuJoCo coordinates → a three.js world position. */
function toWorld(
  local: Float32Array,
  frame: VizFrame | undefined,
  readSlot: SlotReader | undefined,
): THREE.Vector3 | null {
  if (!frame) return mjcToThreeCoordinate(local);
  const pos = readSlot?.({ entity: frame.entity, field: frame.pos_field });
  const quat = readSlot?.({ entity: frame.entity, field: frame.quat_field });
  if (!pos || !quat) return null;
  // MuJoCo packs a quaternion (w, x, y, z); three takes (x, y, z, w).
  const rotated = new THREE.Vector3(local[0], local[1], local[2]).applyQuaternion(
    new THREE.Quaternion(quat[1], quat[2], quat[3], quat[0]),
  );
  return mjcToThreeCoordinate([rotated.x + pos[0], rotated.y + pos[1], rotated.z + pos[2]]);
}

/**
 * Shaft + head, unit-height with their base at the origin, so a frame sets only
 * `scale.y` and one quaternion. Not `ArrowHelper`: its shaft is a 1px line, so mjlab's
 * `width` would be ignored. Opaque, as mjlab's viser scene draws every arrow.
 */
function makeArrow(primitive: VizPrimitive): THREE.Group {
  const width = primitive.width ?? 0.015;
  const material = viserMaterial(primitive.color);
  const shaft = new THREE.Mesh(unitShaft().scale(width, 1, width), material);
  const head = new THREE.Mesh(unitHead().scale(width, 1, width), material);
  const group = new THREE.Group();
  group.add(shaft, head);
  return group;
}

/** viser draws a sphere at its color's own alpha, unlike an arrow. */
function makeSphere(primitive: VizPrimitive): THREE.Mesh {
  const radius = primitive.radius ?? 0.03;
  return new THREE.Mesh(
    unitSphere().scale(radius, radius, radius),
    viserMaterial(primitive.color, primitive.color[3] ?? 1),
  );
}

/** One term's debug drawing: built once, repositioned per frame. */
export class CommandDebugVisuals {
  private readonly objects: THREE.Object3D[];

  constructor(
    termName: string,
    private readonly primitives: readonly VizPrimitive[],
    private readonly root: THREE.Object3D,
  ) {
    this.objects = primitives.map((primitive, index) => {
      const object = primitive.shape === 'arrow' ? makeArrow(primitive) : makeSphere(primitive);
      object.name = `mjswan-command-${termName}-viz-${index}`;
      object.visible = false;
      root.add(object);
      return object;
    });
  }

  /** Place each primitive from this frame's state; hide any whose source is missing. */
  update(visible: boolean, state: StateReader, readSlot?: SlotReader): void {
    for (let i = 0; i < this.primitives.length; i++) {
      const primitive = this.primitives[i];
      const object = this.objects[i];
      object.visible = false;
      if (!visible) continue;

      const origin = resolve(primitive.origin, state, readSlot);
      if (!origin) continue;
      const start = toWorld(origin, primitive.frame, readSlot);
      if (!start) continue;

      if (primitive.shape !== 'arrow') {
        object.position.copy(start);
        object.visible = true;
        continue;
      }

      const vector = primitive.vector && resolve(primitive.vector, state, readSlot);
      if (!vector) continue;
      const tip = Float32Array.from([
        origin[0] + vector[0],
        origin[1] + vector[1],
        origin[2] + vector[2],
      ]);
      const end = toWorld(tip, primitive.frame, readSlot);
      if (!end) continue;
      const direction = end.clone().sub(start);
      const length = direction.length();
      // No direction to point it in.
      if (length < 1e-6) continue;

      const [shaft, head] = object.children;
      shaft.scale.set(1, length * (1 - HEAD_FRACTION), 1);
      head.scale.set(1, length * HEAD_FRACTION, 1);
      head.position.y = length * (1 - HEAD_FRACTION);
      object.quaternion.setFromUnitVectors(UP, direction.divideScalar(length));
      object.position.copy(start);
      object.visible = true;
    }
  }

  dispose(): void {
    for (const object of this.objects) {
      this.root.remove(object);
      object.traverse(child => {
        if (!(child instanceof THREE.Mesh)) return;
        child.geometry.dispose();
        const material = child.material;
        if (Array.isArray(material)) material.forEach(entry => entry.dispose());
        else material.dispose();
      });
    }
  }
}
