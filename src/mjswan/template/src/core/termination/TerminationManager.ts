import { TerminationBase, type TerminationConfig } from './TerminationBase';
import type { TerminationConstructor } from './terminations';
import {
  FusedLane,
  FusedTermination,
  isFusedTerminationConfig,
  type FusedTerminationConfig,
} from './FusedTermination';
import { OnnxTermination, type OnnxTerminationConfig } from './OnnxTermination';
import { TimeOutTermination, type TimeOutTerminationConfig } from './TimeOutTermination';
import type { OnnxSessionCache, SlotReader } from '../onnx/session';
import type { PolicyState, TerminationConfigEntry } from '../policy/types';
import type { PolicyRunner } from '../policy/PolicyRunner';

export type TerminationResult = {
  done: boolean;
  terminated: boolean;
  truncated: boolean;
  reasons: string[];
};

/** Absent for a policy whose terminations are all plugin terms. */
export type TerminationManagerDeps = {
  onnxSessions?: OnnxSessionCache;
  readOnnxSlot?: SlotReader;
  /** mjlab's `episode_length_buf`, for a pre-format-4 `time_out`. */
  episodeLength?: () => number;
  /** The control period, for a pre-format-4 `time_out`'s step limit. */
  stepDt?: number;
};

/** Whether an entry names a traced-ONNX termination. */
function isOnnxEntry(entry: TerminationConfigEntry): entry is OnnxTerminationConfig {
  return typeof (entry as { onnx?: unknown }).onnx === 'string';
}

/** A pre-format-4 `time_out` marker, matched on `native` being present: its text is prose. */
function isNativeTimeOutEntry(
  entry: TerminationConfigEntry,
): entry is TimeOutTerminationConfig {
  return typeof (entry as { native?: unknown }).native === 'string';
}

export class TerminationManager {
  private terms: { name: string; term: TerminationBase; isTimeOut: boolean }[] = [];
  /** Every graph, fused or not, run before any term is read. */
  private graphs: (OnnxTermination | FusedTermination)[] = [];

  constructor(
    config: Record<string, TerminationConfigEntry>,
    registry: Record<string, TerminationConstructor>,
    runner: PolicyRunner,
    deps: TerminationManagerDeps = {},
  ) {
    for (const [name, entry] of Object.entries(config)) {
      if (isFusedTerminationConfig(entry)) {
        this.addFusedGroup(entry, runner, deps);
        continue;
      }
      if (isOnnxEntry(entry)) {
        const term = this.buildOnnxTermination(name, entry, runner, deps);
        if (term) {
          this.graphs.push(term);
          this.terms.push({ name, term, isTimeOut: entry.time_out ?? false });
        }
        continue;
      }
      if (isNativeTimeOutEntry(entry)) {
        this.terms.push({
          name,
          term: new TimeOutTermination(
            runner,
            { ...entry, name },
            deps.episodeLength ?? (() => 0),
            deps.stepDt ?? 0,
          ),
          isTimeOut: entry.time_out ?? false,
        });
        continue;
      }
      const TermClass = registry[entry.name];
      if (!TermClass) {
        // Throws like the observation and command registries: continuing would run the
        // episode without a reset condition it is configured to have.
        throw new Error(`Unknown termination type: ${entry.name}`);
      }
      const termConfig: TerminationConfig = {
        name: entry.name,
        params: entry.params,
        time_out: entry.time_out,
      };
      this.terms.push({
        name,
        term: new TermClass(runner, termConfig),
        isTimeOut: entry.time_out ?? false,
      });
    }
  }

  /**
   * Expand a fused graph into one entry per lane, so the logic below need not know about
   * fusion. Skips the group on missing deps: lost reset conditions beat a dead scene.
   */
  private addFusedGroup(
    entry: FusedTerminationConfig,
    runner: PolicyRunner,
    deps: TerminationManagerDeps,
  ): void {
    const session = deps.onnxSessions?.get(entry.fused);
    const readSlot = deps.readOnnxSlot;
    if (!session || !readSlot) {
      console.warn(
        `[TerminationManager] the fused graph "${entry.fused}" needs a session and ` +
          'a slot reader; skipping every term it covers.',
      );
      return;
    }
    const group = new FusedTermination(entry, { session, readSlot });
    this.graphs.push(group);
    entry.lanes.forEach((lane, index) => {
      this.terms.push({
        name: lane.name,
        term: new FusedLane(runner, { name: lane.name }, group, index),
        isTimeOut: lane.time_out ?? false,
      });
    });
  }

  /**
   * Build a traced-ONNX termination, or warn and skip: unlike an observation, dropping one
   * loses a reset condition rather than reshaping the policy's input vector.
   */
  private buildOnnxTermination(
    name: string,
    entry: OnnxTerminationConfig,
    runner: PolicyRunner,
    deps: TerminationManagerDeps,
  ): OnnxTermination | null {
    const session = deps.onnxSessions?.get(entry.onnx);
    const readSlot = deps.readOnnxSlot;
    if (!session || !readSlot) {
      console.warn(
        `[TerminationManager] "${name}" needs the ONNX session "${entry.onnx}" and a ` +
          'slot reader; skipping.',
      );
      return null;
    }
    return new OnnxTermination(runner, { ...entry, name }, { session, readSlot });
  }

  /**
   * OR-reduce every term to one verdict, `time_out` split out. Every graph is awaited
   * first, so a verdict is this step's, as in mjlab.
   */
  async evaluate(state: PolicyState): Promise<TerminationResult> {
    await Promise.all(this.graphs.map((graph) => graph.step()));
    let terminated = false;
    let truncated = false;
    const reasons: string[] = [];

    for (const { name, term, isTimeOut } of this.terms) {
      if (term.evaluate(state)) {
        reasons.push(name);
        if (isTimeOut) {
          truncated = true;
        } else {
          terminated = true;
        }
      }
    }

    return {
      done: terminated || truncated,
      terminated,
      truncated,
      reasons,
    };
  }

  reset(): void {
    for (const graph of this.graphs) graph.reset();
    for (const { term } of this.terms) {
      term.reset?.();
    }
  }

  get size(): number {
    return this.terms.length;
  }
}
