/**
 * A geom drawn over the scene and never simulated: what a host shows where a geom would
 * go before it adds one (ADR 0011). Parented to the three.js scene, as the mode gizmos
 * are, so a scene load keeps it, `frameCamera` never frames it, and the pointer's
 * raycast skips it.
 */
import * as THREE from 'three';

import { mjcToThreeCoordinate } from './coordinate';
import { primitiveGeometry, primitiveScale, type PrimitiveType } from './primitive';
import { getQuaternion, reflectanceParams } from './scene';

/** One primitive geom, in MuJoCo's world axes and units. */
export interface PreviewInput {
  type: PrimitiveType;
  /** MuJoCo's `size` for the type. */
  size: ArrayLike<number>;
  pos: ArrayLike<number>;
  /** (w, x, y, z); the identity when absent. */
  quat?: ArrayLike<number>;
  /** 0 to 1 each; MuJoCo's geom grey when absent. */
  rgba?: ArrayLike<number>;
}

const GEOM_GREY = [0.5, 0.5, 0.5, 1];
const IDENTITY = [1, 0, 0, 0];

export class GeomPreview {
  private mesh: THREE.Mesh<THREE.BufferGeometry, THREE.MeshPhysicalMaterial> | null = null;
  /** The type and size its geometry was built for: moving it alone rebuilds nothing. */
  private shape = '';

  constructor(private readonly scene: THREE.Scene) {}

  /** `specularRatio` is the loaded scene's (`lightSpecularRatio`), so it lights as a geom there. */
  show(input: PreviewInput, specularRatio = 1): void {
    const shape = `${input.type} ${Array.from(input.size).join(' ')}`;
    if (!this.mesh) {
      const material = new THREE.MeshPhysicalMaterial(reflectanceParams(null, -1, specularRatio));
      this.mesh = new THREE.Mesh(primitiveGeometry(input.type, input.size), material);
      this.mesh.castShadow = true;
      this.mesh.userData.interactionGizmo = true;
      this.scene.add(this.mesh);
    } else if (shape !== this.shape) {
      this.mesh.geometry.dispose();
      this.mesh.geometry = primitiveGeometry(input.type, input.size);
    }
    this.shape = shape;

    const [r, g, b, a] = Array.from(input.rgba ?? GEOM_GREY);
    const material = this.mesh.material;
    material.color.setRGB(r, g, b);
    material.opacity = a;
    if (material.transparent !== a < 1) {
      material.transparent = a < 1;
      material.needsUpdate = true;
    }
    this.mesh.position.copy(mjcToThreeCoordinate(input.pos));
    getQuaternion(Float32Array.from(input.quat ?? IDENTITY), 0, this.mesh.quaternion);
    primitiveScale(input.type, input.size, this.mesh.scale);
  }

  hide(): void {
    if (!this.mesh) return;
    this.scene.remove(this.mesh);
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
    this.mesh = null;
    this.shape = '';
  }
}
