import type * as THREE from 'three';

type Vec3 = readonly [number, number, number];

/**
 * Each face's (sc, tc) in OpenGL's cube-map lookup, which MuJoCo samples a cube texture
 * with, for a point `r` of the unit cube. Faces in MuJoCo's order, +X, -X, +Y, -Y, +Z, -Z,
 * which `fileright` to `fileback` fill.
 */
const FACE_COORDS: readonly ((r: Vec3) => [number, number])[] = [
  r => [-r[2], -r[1]],
  r => [r[2], -r[1]],
  r => [r[0], r[2]],
  r => [r[0], -r[2]],
  r => [r[0], -r[1]],
  r => [-r[0], -r[1]],
];

/** The cube-texture face, 0 to 5 in MuJoCo's order, that geom-local normal `n` faces. */
export function cubeFace(n: Vec3): number {
  const axis = [0, 1, 2].reduce((a, b) => (Math.abs(n[b]) > Math.abs(n[a]) ? b : a));
  return 2 * axis + (n[axis] < 0 ? 1 : 0);
}

/**
 * Point a box's texture coordinates at a cube texture's faces as MuJoCo lays them out,
 * and return the face each of its groups draws. `geometry` is the `BoxGeometry` of a box
 * with half-sizes `size`, in three's frame: MuJoCo's (x, y, z) is its (x, -z, y).
 */
export function mapCubeTextureOnBox(
  geometry: THREE.BufferGeometry,
  size: ArrayLike<number>,
): number[] {
  const position = geometry.getAttribute('position');
  const normal = geometry.getAttribute('normal');
  const uv = geometry.getAttribute('uv');
  const faceAt = (v: number): number =>
    cubeFace([normal.getX(v), -normal.getZ(v), normal.getY(v)]);
  for (let v = 0; v < position.count; v++) {
    const r: Vec3 = [
      position.getX(v) / size[0],
      -position.getZ(v) / size[1],
      position.getY(v) / size[2],
    ];
    const [sc, tc] = FACE_COORDS[faceAt(v)](r);
    uv.setXY(v, (sc + 1) / 2, (tc + 1) / 2);
  }
  uv.needsUpdate = true;
  const index = geometry.getIndex();
  return geometry.groups.map(group => faceAt(index ? index.getX(group.start) : group.start));
}
