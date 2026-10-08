/**
 * Rollout parity: the whole browser MDP chain against mjlab, over N steps — the only
 * test that runs the pieces composed. Fixture state → real `SlotReader` → real ORT
 * session over the Builder's graph bytes → real `FusedObservation` → group vector, and
 * the same through `TerminationManager` to a verdict.
 *
 * Catches what nothing narrower does: a layout offset, a slot fed under a name the graph
 * does not declare, a lane read from the wrong column, `time_out` counted as a
 * termination.
 *
 * **States are replayed, not co-simulated** — mjlab integrates with `mujoco_warp` and the
 * browser with MuJoCo's WASM, so a free-running comparison would measure MuJoCo against
 * itself. The fixture carries mjlab's state per step plus its own vector and verdicts at
 * that state.
 *
 * Regenerate: `MUJOCO_GL=disable .venv/bin/python tests/dump_rollout_fixture.py`
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { beforeAll, describe, expect, it } from 'vitest';

import fixture from './fixtures/rollout/rollout.json';
import { CommandManager } from '../../command/CommandManager';
import type {
  CommandConfigEntry,
  CommandTerm,
  CommandTermConstructor,
  CommandTermContext,
  CommandsConfig,
} from '../../command/types';
import type { FusedObservationConfig } from '../../observation/FusedObservation';
import { createOnnxSession, type OnnxSession, type SlotReader } from '../../onnx/session';
import { createSlotReader, type SlotReaderContext } from '../../onnx/slotReader';
import { isFusedTerminationConfig } from '../../termination/FusedTermination';
import { TerminationManager } from '../../termination/TerminationManager';
import { PolicyRunner } from '../../policy/PolicyRunner';
import type {
  PolicyConfig,
  PolicyRunnerContext,
  PolicyState,
  TerminationConfigEntry,
} from '../../policy/types';

const FIXTURES = join(__dirname, 'fixtures/rollout');

interface Step {
  action: number[];
  data: Record<string, number[]>;
  /** mjlab's `episode_length_buf` at this state. */
  episode_length: number;
  native: Record<string, number[]>;
  obs: number[];
  terminations: Record<string, boolean>;
}

interface TaskFixture {
  group: FusedObservationConfig & { fused: string };
  terminations: Record<string, TerminationConfigEntry>;
  model: Record<string, number | number[]>;
  encoder_bias: Record<string, number>;
  num_actions: number;
  steps: Step[];
}

const TASKS = fixture as unknown as Record<string, TaskFixture>;

/** Load a graph the fixture dumper wrote, as the bytes a bundle would deliver. */
async function sessionFor(taskId: string, ref: string): Promise<OnnxSession> {
  const buf = readFileSync(join(FIXTURES, taskId, ref));
  return createOnnxSession(
    buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer,
  );
}

/** One step's `mjModel`/`mjData` view — mjlab's arrays, so its vector is the answer. */
function contextFor(task: TaskFixture, step: Step): SlotReaderContext {
  const { names, ...rest } = task.model;
  const mjModel = { ...rest, names: Uint8Array.from(names as number[]).buffer };
  const mjData: Record<string, Float64Array> = {};
  for (const [key, values] of Object.entries(step.data)) {
    mjData[key] = Float64Array.from(values);
  }
  return { mjModel, mjData, episodeLength: step.episode_length } as unknown as SlotReaderContext;
}

/** Reads one native input's fixture value for the current step. */
function nativeReader(task: TaskFixture, step: () => Step, kind: string): () => Float32Array {
  const entry = (task.group.native_inputs ?? []).find(native => native.native === kind);
  return () => Float32Array.from(entry ? (step().native[entry.input] ?? []) : []);
}

/** A command term serving one fixture value, registered like any plugin term. */
function fixtureCommandClass(read: () => Float32Array): CommandTermConstructor {
  return class FixtureCommand implements CommandTerm {
    constructor(
      _termName: string,
      _config: CommandConfigEntry,
      _context: CommandTermContext,
    ) {}

    getCommand(): Float32Array {
      return read();
    }
  };
}

/**
 * The real `PolicyRunner` and `CommandManager`, wired as `runtime.ts` wires them, so the
 * Action→Observation and Command→Observation paths run for real: the fixture's numbers
 * reach the graph through `setLastActions`/`getLastActions` and through a term registered
 * under the config's own name.
 *
 * The runner builds the fused observation itself, so the layout and width bookkeeping
 * around the graph is under test too.
 */
async function harnessFor(
  taskId: string,
  task: TaskFixture,
  step: () => Step,
  readSlot: SlotReader,
): Promise<PolicyRunner> {
  const commandManager = new CommandManager();
  // Under the build's own names, through the real registry, so both `getCommand` and
  // `FusedObservation`'s name binding run against a real manager.
  const commands: CommandsConfig = {};
  for (const native of task.group.native_inputs ?? []) {
    if (native.native === 'command' && native.command_name) {
      commands[native.command_name] = { name: 'FixtureCommand' };
    }
  }
  commandManager.initialize(commands, {} as unknown as CommandTermContext, {
    FixtureCommand: fixtureCommandClass(nativeReader(task, step, 'command')),
  });

  const sessions = new Map<string, OnnxSession>([
    [task.group.fused, await sessionFor(taskId, task.group.fused)],
  ]);
  const runner = new PolicyRunner(
    {
      policy_num_actions: task.num_actions,
      observations: { policy: task.group },
    } as unknown as PolicyConfig,
    {
      onnxSessions: { get: (path: string) => sessions.get(path) } as never,
      readOnnxSlot: readSlot,
    },
  );
  await runner.init({
    mujoco: null,
    mjModel: null,
    mjData: null,
    commandManager,
  } as unknown as PolicyRunnerContext);
  return runner;
}

/** float32 through the graph, so compare at float32 resolution. */
function expectClose(actual: ArrayLike<number>, expected: number[], label: string): void {
  expect(actual.length, `${label}: width`).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const tolerance = Math.max(1e-4, Math.abs(expected[i]) * 1e-4);
    expect(Math.abs(actual[i] - expected[i]), `${label}[${i}]`).toBeLessThan(tolerance);
  }
}

const EMPTY_STATE = {} as PolicyState;

describe.each(Object.keys(TASKS))('rollout parity vs mjlab — %s', taskId => {
  const task = TASKS[taskId];
  let current: Step = task.steps[0];
  let runner: PolicyRunner;
  let terminations: TerminationManager;

  const readSlot: SlotReader = slot =>
    createSlotReader(() => contextFor(task, current), {
      jointBias: name => task.encoder_bias[name] ?? 0,
    })(slot);

  /** mjlab's `action_manager.action` for the current step, stored as the runtime does. */
  const readActions = nativeReader(task, () => current, 'prev_action');

  /** Advance to one fixture step, pushing its action through the real runner. */
  const seek = (step: Step): void => {
    current = step;
    runner.setLastActions(readActions());
  };

  beforeAll(async () => {
    runner = await harnessFor(taskId, task, () => current, readSlot);

    // Keyed as the config references them, so the manager resolves them as the runtime does.
    const sessions = new Map<string, OnnxSession>();
    for (const entry of Object.values(task.terminations)) {
      for (const ref of [
        (entry as { onnx?: string }).onnx,
        (entry as { fused?: string }).fused,
      ]) {
        if (ref) sessions.set(ref, await sessionFor(taskId, ref));
      }
    }
    terminations = new TerminationManager(task.terminations, {}, runner, {
      onnxSessions: { get: (path: string) => sessions.get(path) } as never,
      readOnnxSlot: readSlot,
    });
  }, 120_000);

  it('reproduces mjlab’s observation vector at every step', async () => {
    for (const [index, step] of task.steps.entries()) {
      seek(step);
      // Through the runner, so the group layout and native reads are in the path.
      const vector = await runner.collectObservations(EMPTY_STATE);
      expectClose(vector, step.obs, `step ${index}`);
    }
    // A graph silently returning zeros would pass against a zero fixture.
    const first = task.steps[0].obs;
    const last = task.steps[task.steps.length - 1].obs;
    expect(first.some((v, i) => Math.abs(v - last[i]) > 1e-6)).toBe(true);
  }, 120_000);

  it('keeps the native inputs that make the manager wiring observable', () => {
    // Guards the harness: those two paths are only exercised by a task whose group has native
    // inputs, so a regenerated fixture that lost them would silently stop testing them.
    const natives = task.group.native_inputs ?? [];
    const command = natives.find(native => native.native === 'command');
    if (command) {
      expect(command.command_name, 'a command input must name its term').toBeTruthy();
      // Found by name in the real manager, which is what `getCommand` resolves.
      expect(runner.getContext()?.commandManager?.termNames()).toContain(command.command_name);
    }
    if (natives.some(native => native.native === 'prev_action')) {
      expect(runner.getNumActions()).toBe(task.num_actions);
      expect(readActions().length).toBe(task.num_actions);
    }
    // Cartpole has neither; assert that, so this reads as a property, not a skipped check.
    if (natives.length === 0) expect(taskId).toBe('Mjlab-Cartpole-Balance');
  });

  it('reproduces every termination verdict at every step', async () => {
    for (const [index, step] of task.steps.entries()) {
      seek(step);
      const result = await terminations.evaluate(EMPTY_STATE);

      const expected = Object.entries(step.terminations)
        .filter(([, fired]) => fired)
        .map(([name]) => name);
      expect(result.reasons.slice().sort(), `step ${index} reasons`).toEqual(expected.sort());
      expect(result.done, `step ${index} done`).toBe(expected.length > 0);
    }
  }, 120_000);

  it('keeps a rollout that could tell a right verdict from a constant one', () => {
    // Without a firing state the comparison above would pass for a graph hardwired to
    // `false`. The dumper tilts the root through an orientation limit and sets the
    // episode counter either side of the horizon for this.
    const names = Object.entries(task.terminations).flatMap(([name, entry]) =>
      isFusedTerminationConfig(entry) ? entry.lanes.map(lane => lane.name) : [name],
    );
    expect(names.length).toBeGreaterThan(0);
    for (const name of names) {
      const verdicts = task.steps.map(step => step.terminations[name]);
      expect(verdicts, `${name} never fires`).toContain(true);
      expect(verdicts, `${name} never clears`).toContain(false);
    }
  });

  it('covers the whole group, not a prefix of it', () => {
    // The layout splices terms into the vector, and a truncated one still passes above.
    const layout = task.group.layout ?? [];
    const width = layout.reduce((n, term) => n + term.size, 0);
    expect(width).toBe(task.group.size);
    expect(task.steps[0].obs.length).toBe(task.group.size);
    expect(layout.length).toBeGreaterThan(0);
  });
});
