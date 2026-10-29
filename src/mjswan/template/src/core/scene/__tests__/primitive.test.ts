/** A primitive geom as three.js draws it: MuJoCo's half-sizes, with its z as three.js's y. */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { primitiveGeometry, primitiveScale } from '../primitive';

/** The geometry's extent along x, y and z, as drawn. */
function extent(geometry: THREE.BufferGeometry, scale = new THREE.Vector3(1, 1, 1)): number[] {
  geometry.computeBoundingBox();
  const size = geometry.boundingBox!.getSize(new THREE.Vector3()).multiply(scale);
  return size.toArray().map((v) => Number(v.toFixed(6)));
}

describe('primitiveGeometry', () => {
  it("lays MuJoCo's half-sizes out with its z as three.js's y", () => {
    expect(extent(primitiveGeometry('box', [0.1, 0.2, 0.3]))).toEqual([0.2, 0.6, 0.4]);
    expect(extent(primitiveGeometry('cylinder', [0.1, 0.5]))).toEqual([0.2, 1, 0.2]);
    // The half-length, plus a radius at each end.
    expect(extent(primitiveGeometry('capsule', [0.1, 0.5]))).toEqual([0.2, 1.2, 0.2]);
    expect(extent(primitiveGeometry('sphere', [0.25]))).toEqual([0.5, 0.5, 0.5]);
  });

  it('lays a finite plane flat, facing up', () => {
    const plane = primitiveGeometry('plane', [2, 3, 0.1]);
    expect(extent(plane)).toEqual([4, 0, 6]);
    const normal = new THREE.Vector3().fromBufferAttribute(plane.getAttribute('normal'), 0);
    expect(normal.toArray().map((v) => Math.round(v))).toEqual([0, 1, 0]);
  });

  it('scales a unit sphere to an ellipsoid, and nothing else', () => {
    const scale = primitiveScale('ellipsoid', [0.1, 0.2, 0.3], new THREE.Vector3());
    expect(extent(primitiveGeometry('ellipsoid', [0.1, 0.2, 0.3]), scale)).toEqual([0.2, 0.6, 0.4]);
    expect(primitiveScale('box', [0.1, 0.2, 0.3], new THREE.Vector3()).toArray()).toEqual([1, 1, 1]);
  });
});
