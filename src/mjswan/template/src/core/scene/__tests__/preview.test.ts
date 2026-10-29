/** The geom a host previews: drawn in MuJoCo's axes, never in the way of a ray, and cheap to move. */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { GeomPreview } from '../preview';

const shown = (scene: THREE.Scene) =>
  scene.children.filter((child): child is THREE.Mesh<THREE.BufferGeometry, THREE.MeshPhysicalMaterial> =>
    child instanceof THREE.Mesh
  );

describe('GeomPreview', () => {
  it("draws one geom at MuJoCo's position and turn, where the pointer's raycast skips it", () => {
    const scene = new THREE.Scene();
    const preview = new GeomPreview(scene);
    // A quarter turn about MuJoCo's z, which is three.js's y.
    const half = Math.SQRT1_2;
    preview.show({ type: 'box', size: [0.1, 0.2, 0.3], pos: [1, 2, 3], quat: [half, 0, 0, half] });
    const [mesh] = shown(scene);
    expect(shown(scene)).toHaveLength(1);
    expect(mesh.position.toArray()).toEqual([1, 3, -2]);
    const x = new THREE.Vector3(1, 0, 0).applyQuaternion(mesh.quaternion);
    // MuJoCo's x turned to its y, which is three.js's -z.
    expect(x.toArray().map((v) => Number(v.toFixed(6)))).toEqual([0, 0, -1]);
    expect(mesh.userData.interactionGizmo).toBe(true);
    expect(mesh.castShadow).toBe(true);
  });

  it("is MuJoCo's grey unless coloured, and see-through only below full opacity", () => {
    const scene = new THREE.Scene();
    const preview = new GeomPreview(scene);
    preview.show({ type: 'sphere', size: [0.1], pos: [0, 0, 0] });
    const [mesh] = shown(scene);
    expect(mesh.material.color.toArray()).toEqual([0.5, 0.5, 0.5]);
    expect(mesh.material.transparent).toBe(false);
    preview.show({ type: 'sphere', size: [0.1], pos: [0, 0, 0], rgba: [0.85, 0.4, 0.3, 0.6] });
    expect(mesh.material.color.toArray().map((v) => Number(v.toFixed(6)))).toEqual([0.85, 0.4, 0.3]);
    expect(mesh.material.opacity).toBe(0.6);
    expect(mesh.material.transparent).toBe(true);
  });

  it('moves without building anything, and rebuilds only for a new shape', () => {
    const scene = new THREE.Scene();
    const preview = new GeomPreview(scene);
    preview.show({ type: 'capsule', size: [0.05, 0.2], pos: [0, 0, 0] });
    const [mesh] = shown(scene);
    const first = mesh.geometry;
    preview.show({ type: 'capsule', size: [0.05, 0.2], pos: [0.5, 0, 0] });
    expect(mesh.geometry).toBe(first);
    preview.show({ type: 'capsule', size: [0.05, 0.4], pos: [0.5, 0, 0] });
    expect(mesh.geometry).not.toBe(first);
    expect(shown(scene)).toEqual([mesh]);
  });

  it('scales an ellipsoid to its radii', () => {
    const scene = new THREE.Scene();
    new GeomPreview(scene).show({ type: 'ellipsoid', size: [0.1, 0.2, 0.3], pos: [0, 0, 0] });
    expect(shown(scene)[0].scale.toArray()).toEqual([0.1, 0.3, 0.2]);
  });

  it('leaves the scene as it found it when hidden, and can show again', () => {
    const scene = new THREE.Scene();
    const preview = new GeomPreview(scene);
    preview.hide();
    preview.show({ type: 'box', size: [0.1, 0.1, 0.1], pos: [0, 0, 0] });
    preview.hide();
    expect(scene.children).toEqual([]);
    preview.show({ type: 'box', size: [0.1, 0.1, 0.1], pos: [0, 0, 0] });
    expect(shown(scene)).toHaveLength(1);
  });
});
