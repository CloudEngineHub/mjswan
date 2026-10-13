import type * as THREE from 'three';

/**
 * Has `material` sample `cube`, in MuJoCo's face order, at the geom-local point over `size`,
 * as MuJoCo does. The geometry is in three's frame, where MuJoCo's (x, y, z) is (x, -z, y).
 */
export function sampleCubeTexture(
  material: THREE.Material,
  cube: THREE.CubeTexture,
  size: readonly number[],
): void {
  material.onBeforeCompile = shader => {
    shader.uniforms.mjCube = { value: cube };
    shader.uniforms.mjCubeSize = { value: size };
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nuniform vec3 mjCubeSize;\nvarying vec3 vMjCubeDir;')
      .replace(
        '#include <begin_vertex>',
        '#include <begin_vertex>\nvMjCubeDir = vec3(position.x, -position.z, position.y) / mjCubeSize;',
      );
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform samplerCube mjCube;\nvarying vec3 vMjCubeDir;')
      .replace('#include <map_fragment>', 'diffuseColor *= textureCube(mjCube, vMjCubeDir);');
  };
  material.addEventListener('dispose', () => cube.dispose());
}
