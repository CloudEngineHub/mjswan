/**
 * A box's cube texture, face by face, as MuJoCo draws it: `fileright` to `fileback` fill
 * the faces +X, -X, +Y, -Y, +Z, -Z, sampled with OpenGL's cube-map lookup, while three
 * builds the box in its own frame, where MuJoCo's (x, y, z) is (x, -z, y).
 */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { cubeFace, mapCubeTextureOnBox } from '../cubeMap';

type Vec3 = [number, number, number];

/** The texture coordinate of the box's vertex at MuJoCo point `at` on the face along `normal`. */
function uvAt(geometry: THREE.BufferGeometry, at: Vec3, normal: Vec3): [number, number] {
  const position = geometry.getAttribute('position');
  const normals = geometry.getAttribute('normal');
  const uv = geometry.getAttribute('uv');
  const near = (a: Vec3, b: Vec3) => a.every((value, i) => Math.abs(value - b[i]) < 1e-6);
  // Float32 positions over float64 sizes land a hair off the corner; `|| 0` drops a -0.
  const round = (value: number) => Math.round(value * 1e6) / 1e6 || 0;
  for (let v = 0; v < position.count; v++) {
    const p: Vec3 = [position.getX(v), -position.getZ(v), position.getY(v)];
    const n: Vec3 = [normals.getX(v), -normals.getZ(v), normals.getY(v)];
    if (near(p, at) && near(n, normal)) return [round(uv.getX(v)), round(uv.getY(v))];
  }
  throw new Error(`no vertex at ${at} facing ${normal}`);
}

describe('cubeFace', () => {
  it('numbers the faces in MuJoCo order', () => {
    const normals: Vec3[] = [
      [1, 0, 0],
      [-1, 0, 0],
      [0, 1, 0],
      [0, -1, 0],
      [0, 0, 1],
      [0, 0, -1],
    ];
    expect(normals.map(cubeFace)).toEqual([0, 1, 2, 3, 4, 5]);
  });
});

describe('mapCubeTextureOnBox', () => {
  const size: Vec3 = [0.05, 0.03, 0.02];
  const box = () => new THREE.BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2);

  it("draws on each of three's groups the image of the axis it faces", () => {
    // three's groups face its own +x, -x, +y, -y, +z, -z.
    expect(mapCubeTextureOnBox(box(), size)).toEqual([0, 1, 4, 5, 3, 2]);
  });

  it('lays each image out as an OpenGL cube map, its first texel at the face corner', () => {
    const geometry = box();
    mapCubeTextureOnBox(geometry, size);
    const [x, y, z] = size;
    expect(uvAt(geometry, [x, y, z], [1, 0, 0])).toEqual([0, 0]);
    expect(uvAt(geometry, [-x, y, -z], [-1, 0, 0])).toEqual([0, 0]);
    expect(uvAt(geometry, [-x, y, -z], [0, 1, 0])).toEqual([0, 0]);
    expect(uvAt(geometry, [-x, -y, z], [0, -1, 0])).toEqual([0, 0]);
    expect(uvAt(geometry, [-x, y, z], [0, 0, 1])).toEqual([0, 0]);
    expect(uvAt(geometry, [x, y, -z], [0, 0, -1])).toEqual([0, 0]);
    expect(uvAt(geometry, [x, -y, -z], [1, 0, 0])).toEqual([1, 1]);
  });
});
