/**
 * `motion` stays a native command (a clip lookup is data, not math), so the tracking task's
 * traced observations and terminations read their `{command: "motion", field}` slots off
 * `TrackingCommand` itself. This pins the one part of that which can be silently wrong: the
 * re-anchoring frame, mjlab's `update_relative_body_poses`.
 */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { TrackingCommand, reanchorBodyPositions } from '../TrackingCommand';
import type { CommandConfigEntry, CommandTermContext } from '../types';

function close(actual: Float32Array, expected: number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) expect(actual[i]).toBeCloseTo(expected[i], 6);
}

describe('reanchorBodyPositions', () => {
  it('takes x/y from the robot anchor, z from the reference, and rotates by the yaw between them', () => {
    const s = Math.SQRT1_2; // 90 deg about +z
    const relative = reanchorBodyPositions(
      // Two reference bodies: one 1 m ahead of the anchor, one 1 m above it.
      Float32Array.from([1, 0, 1, 0, 0, 2]),
      [0, 0, 1],
      [1, 0, 0, 0],
      [5, 7, 2],
      [s, 0, 0, s],
    );
    // Anchor lands at (5, 7, 1) — the robot's x/y, the reference's z, so a reader
    // that took the robot's z would put the whole skeleton 1 m too high.
    // The +x offset rotates onto +y; the +z offset is untouched by yaw.
    close(relative, [5, 8, 1, 5, 7, 2]);
  });

  it('is the identity when the robot sits exactly on the reference anchor', () => {
    const bodies = Float32Array.from([1, 2, 3, -1, 0, 0.5]);
    close(reanchorBodyPositions(bodies, [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [1, 0, 0, 0]), [
      1, 2, 3, -1, 0, 0.5,
    ]);
  });
});

/**
 * The `ref_*` look-ahead window, which mjlab's `MotionCommand` has no equivalent of. A
 * wrong window is silent — the policy runs, tracking the wrong part of the clip — so the
 * offset→frame mapping, edge clamping and not-ready fallback are all pinned. No model
 * needed: the reference buffers are the command's own state.
 */
function trackingCommand(timeSteps: number[], frames: number): TrackingCommand {
  const config = {
    name: 'TrackingCommand',
    time_steps: timeSteps,
  } as unknown as CommandConfigEntry;
  const context = {
    mujoco: {},
    mjModel: null,
    mjData: null,
    scene: new THREE.Scene(),
  } as unknown as CommandTermContext;
  const term = new TrackingCommand('motion', config, context);
  term.refLen = frames;
  term.nJoints = 2;
  // Frame i is identifiable in every field: position (i, 0, 0), joints (i, -i).
  term.refRootPos = Array.from({ length: frames }, (_, i) => Float32Array.from([i, 0, 0]));
  term.refRootQuat = Array.from({ length: frames }, () => Float32Array.from([1, 0, 0, 0]));
  term.refJointPos = Array.from({ length: frames }, (_, i) => Float32Array.from([i, -i]));
  // `isReady()` also needs a selected motion; the window never reads its contents.
  (term as unknown as { selectedMotion: unknown }).selectedMotion = {};
  return term;
}

describe('TrackingCommand command slot', () => {
  it('serves `command` as getCommand(): what `get_command("motion")` reads in mjlab', () => {
    const term = trackingCommand([0], 4);
    (term as unknown as { selectedMotion: unknown }).selectedMotion = {
      jointVel: Array.from({ length: 4 }, (_, i) => Float32Array.from([10 * i, -10 * i])),
    };
    term.refIdx = 2;
    // Joint positions, then joint velocities.
    close(term.getStateField('command')!, [2, -2, 20, -20]);
  });
});

describe('TrackingCommand ref window', () => {
  it('samples each field at every time_steps offset, in order', () => {
    const term = trackingCommand([0, 2, -1], 10);
    term.refIdx = 5;
    // Offsets 0/+2/-1 of frame 5 -> frames 5, 7, 4.
    close(term.getStateField('ref_root_pos_w')!, [5, 0, 0, 7, 0, 0, 4, 0, 0]);
    close(term.getStateField('ref_joint_pos')!, [5, -5, 7, -7, 4, -4]);
  });

  it('clamps a window running off either end rather than wrapping', () => {
    const term = trackingCommand([-4, 0, 4], 3);
    term.refIdx = 0;
    // Wrapping would read frame 2 for the -4 offset; clamping repeats frame 0.
    close(term.getStateField('ref_root_pos_w')!, [0, 0, 0, 0, 0, 0, 2, 0, 0]);
  });

  it('reports readiness, and falls back to finite values before a clip loads', () => {
    const term = trackingCommand([0, 1], 2);
    (term as unknown as { selectedMotion: unknown }).selectedMotion = null;
    close(term.getStateField('is_ready')!, [0]);
    close(term.getStateField('ref_root_pos_w')!, [0, 0, 0, 0, 0, 0]);
    // Identity quats, not zeros: the term normalizes these, and a zero quat is NaN.
    close(term.getStateField('ref_root_quat_w')!, [1, 0, 0, 0, 1, 0, 0, 0]);

    (term as unknown as { selectedMotion: unknown }).selectedMotion = {};
    close(term.getStateField('is_ready')!, [1]);
  });

  it('defaults to the current frame alone when no time_steps are configured', () => {
    const config = { name: 'TrackingCommand' } as unknown as CommandConfigEntry;
    const context = {
      mujoco: {},
      mjModel: null,
      mjData: null,
      scene: new THREE.Scene(),
    } as unknown as CommandTermContext;
    const term = new TrackingCommand('motion', config, context);
    term.refLen = 4;
    term.refIdx = 2;
    term.refRootPos = Array.from({ length: 4 }, (_, i) => Float32Array.from([i, 0, 0]));
    (term as unknown as { selectedMotion: unknown }).selectedMotion = {};
    close(term.getStateField('ref_root_pos_w')!, [2, 0, 0]);
  });
});

/** mjlab's `_debug_vis_impl` in `"ghost"` mode. */
describe('TrackingCommand ghost', () => {
  function ghostCommand(config: Record<string, unknown> = {}): { term: TrackingCommand; scene: THREE.Scene } {
    const scene = new THREE.Scene();
    const link = new THREE.Group();
    link.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial()));
    const context = {
      mujoco: { MjData: class {} },
      // Body 1 hangs off a joint, so the ghost clones it.
      mjModel: { nbody: 2, body_jntnum: [0, 1], body_parentid: [0, 0] },
      mjData: null,
      scene,
      bodies: { 1: link },
    } as unknown as CommandTermContext;
    const term = new TrackingCommand(
      'motion',
      { name: 'TrackingCommand', ...config } as unknown as CommandConfigEntry,
      context,
    );
    (term as unknown as { selectedMotion: unknown }).selectedMotion = {};
    term.refLen = 1;
    return { term, scene };
  }

  const ghostOf = (scene: THREE.Scene) => scene.getObjectByName('Tracking Ghost')!;

  it('draws only while Debug Viz and its own checkbox are both on', () => {
    const { term, scene } = ghostCommand({ debug_vis: true });
    term.updateDebugVisuals(true);
    expect(ghostOf(scene).visible).toBe(true);

    term.updateDebugVisuals(false);
    expect(ghostOf(scene).visible).toBe(false);

    term.setDebugVisEnabled(false);
    term.updateDebugVisuals(true);
    expect(ghostOf(scene).visible).toBe(false);
    expect(term.debugVisEnabled()).toBe(false);
  });

  it('stays hidden with no motion selected', () => {
    const { term, scene } = ghostCommand();
    (term as unknown as { selectedMotion: unknown }).selectedMotion = null;
    term.updateDebugVisuals(true);
    expect(ghostOf(scene).visible).toBe(false);
  });

  it('builds no ghost and offers no checkbox when the task leaves debug_vis off', () => {
    const { term, scene } = ghostCommand({ debug_vis: false });
    expect(term.debugVisEnabled()).toBeNull();
    expect(scene.getObjectByName('Tracking Ghost')).toBeUndefined();
  });

  it('draws for a document written before debug_vis was carried', () => {
    expect(ghostCommand().term.debugVisEnabled()).toBe(true);
  });

  it("paints the ghost in the task's ghost_color at viser's ghost opacity", () => {
    const { scene } = ghostCommand({ ghost_color: [1, 0, 0, 0.25] });
    const mesh = ghostOf(scene).getObjectByProperty('type', 'Mesh') as THREE.Mesh;
    const material = mesh.material as THREE.MeshStandardMaterial;
    expect(material.color.toArray()).toEqual([1, 0, 0]);
    // The color's alpha only marks visual geoms in mjlab; `add_ghost_mesh` draws at 0.5.
    expect(material.opacity).toBe(0.5);
    expect(mesh.castShadow).toBe(false);
  });
});

/** mjlab's `viz.mode="frames"`: reference and robot frames for each body and the anchor. */
describe('TrackingCommand frames', () => {
  it('draws three axes for each of the four frames of one body, and builds no ghost', () => {
    const scene = new THREE.Scene();
    const context = {
      mujoco: { MjData: class {} },
      mjModel: {
        nbody: 2,
        body: (i: number) => ({ name: ['world', 'robot/pelvis'][i] }),
        body_jntnum: [0, 1],
        body_parentid: [0, 0],
      },
      mjData: {
        xpos: Float64Array.from([0, 0, 0, 0.2, 0, 0.9]),
        xquat: Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0]),
      },
      scene,
      bodies: { 1: new THREE.Group() },
    } as unknown as CommandTermContext;
    const motion = { name: 'clip', body_names: ['pelvis'], anchor_body_name: 'pelvis' };
    const term = new TrackingCommand(
      'motion',
      { name: 'TrackingCommand', viz_mode: 'frames', motions: [motion] } as unknown as CommandConfigEntry,
      context,
    );
    const shafts = () =>
      (scene.getObjectByName('Tracking Frames')!.children[0] as THREE.InstancedMesh).count;
    expect(scene.getObjectByName('Tracking Ghost')).toBeUndefined();
    expect(term.debugVisEnabled()).toBe(true);

    term.updateDebugVisuals(true);
    expect(shafts()).toBe(0); // No clip yet.

    const internals = term as unknown as Record<string, unknown>;
    internals.selectedMotion = motion;
    internals.refBodyPosW = [Float32Array.from([0, 0, 1])];
    internals.refBodyQuatW = [Float32Array.from([1, 0, 0, 0])];
    term.refLen = 1;
    term.updateDebugVisuals(true);
    expect(shafts()).toBe(3 * 4);
    term.updateDebugVisuals(false);
    expect(shafts()).toBe(0);
  });
});
