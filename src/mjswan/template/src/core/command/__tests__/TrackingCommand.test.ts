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
    // mjlab's `MotionCommand.command`: the joint positions, then the joint velocities.
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
