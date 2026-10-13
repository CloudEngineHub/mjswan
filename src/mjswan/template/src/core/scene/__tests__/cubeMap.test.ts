/** A box's cube texture, sampled in three's physical shader as MuJoCo samples it. */
import * as THREE from 'three';
import { expect, it } from 'vitest';

import { sampleCubeTexture } from '../cubeMap';

it('samples the cube at the geom-local point over the size', () => {
  const material = new THREE.MeshPhysicalMaterial();
  const cube = new THREE.CubeTexture();
  sampleCubeTexture(material, cube, [0.05, 0.03, 0.02]);
  const shader = {
    uniforms: {} as Record<string, THREE.IUniform>,
    vertexShader: THREE.ShaderLib.physical.vertexShader,
    fragmentShader: THREE.ShaderLib.physical.fragmentShader,
  };
  material.onBeforeCompile(
    shader as THREE.WebGLProgramParametersWithUniforms,
    {} as THREE.WebGLRenderer,
  );

  expect(shader.uniforms.mjCube.value).toBe(cube);
  expect(shader.uniforms.mjCubeSize.value).toEqual([0.05, 0.03, 0.02]);
  expect(shader.vertexShader).toContain('varying vec3 vMjCubeDir;');
  expect(shader.vertexShader).toContain(
    'vMjCubeDir = vec3(position.x, -position.z, position.y) / mjCubeSize;',
  );
  expect(shader.fragmentShader).toContain('uniform samplerCube mjCube;');
  expect(shader.fragmentShader).toContain('diffuseColor *= textureCube(mjCube, vMjCubeDir);');
});
