/**
 * A traced action term's native half (ADR 0009): its state carried across control steps,
 * its reset graph, the slice of the action it is fed, and what an observation reads off
 * it. The graph's math is checked Python-side against the term's own body; the session
 * here is a fake, so these run with no ORT.
 */
import { describe, expect, it } from 'vitest';

import { SeededRng } from '../../rng';
import type { OnnxSession, OnnxTensorLike } from '../../onnx/session';
import type { ResolvedActionTerm } from '../applyAction';
import { ActionManager } from '../ActionManager';
import { OnnxActionProcess, type OnnxActionProcessConfig } from '../OnnxActionProcess';

/** Records feeds and answers with `respond`. */
class FakeSession implements OnnxSession {
  readonly calls: Array<Record<string, number[]>> = [];

  constructor(
    private readonly respond: (feeds: Record<string, OnnxTensorLike>) => Record<string, OnnxTensorLike>,
    readonly inputNames?: string[],
  ) {}

  run(feeds: Record<string, OnnxTensorLike>): Promise<Record<string, OnnxTensorLike>> {
    this.calls.push(
      Object.fromEntries(
        Object.entries(feeds).map(([k, v]) => [k, Array.from(v.data as ArrayLike<number>, Number)]),
      ),
    );
    return Promise.resolve(this.respond(feeds));
  }
}

function tensor(values: number[]): OnnxTensorLike {
  return { data: Float32Array.from(values), dims: [1, values.length] };
}

function read(t: OnnxTensorLike): number[] {
  return Array.from(t.data as ArrayLike<number>, Number);
}

/** inhand's delta term, as its graph computes it: target += clip(raw) / 24. */
const DELTA_CFG: OnnxActionProcessConfig = {
  onnx: 'mdp/a/action/joint_pos.onnx',
  state_fields: [{ name: '_target', shape: [1, 2], dtype: 'float32', init: [0.1, 0.2] }],
  input_slots: [{ entity: 'robot', field: 'joint_pos', input: 'robot__joint_pos', shape: [1, 2] }],
  ramp_from: true,
};

function deltaSession(): FakeSession {
  return new FakeSession((feeds) => {
    const prev = read(feeds.prev__target);
    const raw = read(feeds.raw);
    const next = prev.map((p, i) => p + Math.max(-1, Math.min(1, raw[i])) / 24);
    return {
      next__target: tensor(next),
      processed: tensor(next),
      processed_from: tensor(prev),
    };
  });
}

/** A reset graph that puts the target at the measured joint position. */
function resetSession(): FakeSession {
  return new FakeSession((feeds) => ({ next__target: tensor(read(feeds.robot__joint_pos)) }));
}

function tracedTerm(over: Partial<ResolvedActionTerm> = {}): ResolvedActionTerm {
  return {
    controlType: 'joint_position',
    ctrlAdr: [0, 1],
    qposAdr: [0, 1],
    qvelAdr: [0, 1],
    actionIndices: [2, 0],
    actionScale: new Float32Array(2),
    actionOffset: new Float32Array(2),
    defaultJointPos: new Float32Array(2),
    encoderBias: new Float32Array(2),
    positionActuator: [true, true],
    kp: new Float32Array(2),
    kd: new Float32Array(2),
    muscleMode: 'sigmoid',
    muscleCtrlLo: new Float32Array(0),
    muscleCtrlHi: new Float32Array(0),
    clipLo: new Float32Array(2).fill(-Infinity),
    clipHi: new Float32Array(2).fill(Infinity),
    processed: new Float32Array(2),
    active: false,
    rampFrom: new Float32Array(2),
    holdS: null,
    holding: false,
    delays: [],
    ...over,
  };
}

const JOINT_POS = [0.4, -0.3];
const readSlot = () => Float32Array.from(JOINT_POS);
const CONTEXT = { episodeLength: 0, stepDt: 0.05 };

describe('OnnxActionProcess', () => {
  it('feeds its slice of the action and its state, and carries the next state over', async () => {
    const session = deltaSession();
    const processor = new OnnxActionProcess('joint_pos', DELTA_CFG, null, {
      process: session,
      rng: new SeededRng(1),
      readSlot,
    });
    const term = tracedTerm();
    const manager = new ActionManager([{ name: 'joint_pos', term, traced: processor }]);

    await manager.process(Float32Array.from([0.5, 9, 1.0]), CONTEXT);
    // `actionIndices` [2, 0] gather the term's slice in its own target order.
    expect(session.calls[0].raw).toEqual([1.0, 0.5]);
    expect(session.calls[0].prev__target.map((v) => Number(v.toFixed(6)))).toEqual([0.1, 0.2]);
    expect(term.active).toBe(true);
    expect(Array.from(term.processed).map((v) => Number(v.toFixed(6)))).toEqual([
      Number((0.1 + 1 / 24).toFixed(6)),
      Number((0.2 + 0.5 / 24).toFixed(6)),
    ]);
    // The ramp starts from where the term stood before this step.
    expect(Array.from(term.rampFrom!).map((v) => Number(v.toFixed(6)))).toEqual([0.1, 0.2]);

    await manager.process(Float32Array.from([0, 0, 0]), CONTEXT);
    expect(session.calls[1].prev__target).toEqual(Array.from(term.processed));
  });

  it('runs its reset graph at a reset, and only that one', async () => {
    const process = deltaSession();
    const reset = resetSession();
    const processor = new OnnxActionProcess(
      'joint_pos',
      DELTA_CFG,
      { onnx: 'mdp/a/action/joint_pos_reset.onnx', input_slots: DELTA_CFG.input_slots },
      { process, reset, rng: new SeededRng(1), readSlot },
    );
    const manager = new ActionManager([{ name: 'joint_pos', term: tracedTerm(), traced: processor }]);

    await manager.reset();
    expect(process.calls).toHaveLength(0);
    expect(reset.calls).toHaveLength(1);
    // An observation of the term's internals sees the reset's state.
    const target = Array.from(manager.getStateField('joint_pos', '_target') ?? []);
    expect(target.map((v) => Number(v.toFixed(6)))).toEqual(JOINT_POS);
  });

  it('keeps the previous targets when a slot cannot be read', async () => {
    const session = deltaSession();
    const processor = new OnnxActionProcess('joint_pos', DELTA_CFG, null, {
      process: session,
      rng: new SeededRng(1),
      readSlot: () => null,
    });
    const term = tracedTerm({ processed: Float32Array.from([0.7, 0.8]), active: true });
    const manager = new ActionManager([{ name: 'joint_pos', term, traced: processor }]);
    const warn = console.warn;
    console.warn = () => {};
    try {
      await manager.process(Float32Array.from([1, 1, 1]), CONTEXT);
    } finally {
      console.warn = warn;
    }
    expect(session.calls).toHaveLength(0);
    expect(term.active).toBe(false);
    expect(Array.from(term.processed).map((v) => Number(v.toFixed(6)))).toEqual([0.7, 0.8]);
  });

  it('draws `rand` from the seeded generator when the graph takes one', async () => {
    const session = new FakeSession(() => ({ processed: tensor([0, 0]) }));
    const processor = new OnnxActionProcess(
      'noisy',
      { onnx: 'x.onnx', state_fields: [], rand_dim: 3, rand_ranges: [[0, 1], [0, 1], [0, 1]] },
      null,
      { process: session, rng: new SeededRng(7), readSlot },
    );
    await processor.process(new Float32Array(2), new Float32Array(2), null);
    expect(session.calls[0].rand).toHaveLength(3);
    expect(session.calls[0].rand.every((v) => v >= 0 && v < 1)).toBe(true);
  });
});

describe('ActionManager', () => {
  it('serves a native term\'s processed action under either of mjlab\'s names', () => {
    const term = tracedTerm({ processed: Float32Array.from([1, 2]) });
    const manager = new ActionManager([{ name: 'joint_pos', term }]);
    expect(Array.from(manager.getStateField('joint_pos', 'processed_actions')!)).toEqual([1, 2]);
    expect(Array.from(manager.getStateField('joint_pos', '_processed_actions')!)).toEqual([1, 2]);
    expect(manager.getStateField('joint_pos', '_target')).toBeNull();
    expect(manager.getStateField('other', 'processed_actions')).toBeNull();
  });

  it('decides a traced term\'s hold on the same clock as a native one', async () => {
    const term = tracedTerm({ holdS: 1.0 });
    const processor = new OnnxActionProcess('held', DELTA_CFG, null, {
      process: deltaSession(),
      rng: new SeededRng(1),
      readSlot,
    });
    const manager = new ActionManager([{ name: 'held', term, traced: processor }]);
    await manager.process(new Float32Array(3), { episodeLength: 19, stepDt: 0.05 });
    expect(term.holding).toBe(true);
    await manager.process(new Float32Array(3), { episodeLength: 20, stepDt: 0.05 });
    expect(term.holding).toBe(false);
  });
});
