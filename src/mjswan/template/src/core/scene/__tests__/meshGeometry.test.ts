/**
 * A mesh's three geometry, built off the real MuJoCo WASM model.
 *
 * The model's `mesh_vert` is a view of the arrays MuJoCo collides against: turned Y-up
 * in place for three, every mesh collider would turn 90 degrees about its own x.
 */
import { beforeAll, expect, it } from 'vitest';

import type { MjModel } from 'mujoco';
import { createMeshGeometry } from '../scene';

/** An off-centre tetrahedron, so the turn moves every vertex. */
const SCENE = `<mujoco>
  <asset>
    <mesh name="tet" vertex="0 0 0  0.1 0 0  0 0.2 0  0 0 0.3"/>
  </asset>
  <worldbody>
    <body name="link" pos="0 0 1">
      <freejoint/>
      <geom type="mesh" mesh="tet"/>
    </body>
  </worldbody>
</mujoco>`;

let mjModel: MjModel;

beforeAll(async () => {
  const load = (await import('mujoco')).default;
  const mujoco = await load();
  mjModel = (mujoco as unknown as { MjModel: { from_xml_string(s: string): MjModel } })
    .MjModel.from_xml_string(SCENE);
});

it('leaves the vertices MuJoCo collides against as compiled', () => {
  const compiled = Array.from(mjModel.mesh_vert);
  const normals = Array.from(mjModel.mesh_normal);
  createMeshGeometry(mjModel, 0);
  expect(Array.from(mjModel.mesh_vert)).toEqual(compiled);
  expect(Array.from(mjModel.mesh_normal)).toEqual(normals);
});

it("draws each vertex Y-up, MuJoCo's (x, y, z) at three's (x, z, -y)", () => {
  const vert = Array.from(mjModel.mesh_vert as Float32Array);
  const expected = new Set<string>();
  for (let v = 0; v < vert.length; v += 3) {
    expected.add([vert[v], vert[v + 2], -vert[v + 1]].map((x) => x.toFixed(6)).join());
  }
  const position = createMeshGeometry(mjModel, 0).getAttribute('position');
  for (let i = 0; i < position.count; i++) {
    const drawn = [position.getX(i), position.getY(i), position.getZ(i)];
    expect(expected).toContain(drawn.map((x) => x.toFixed(6)).join());
  }
});
