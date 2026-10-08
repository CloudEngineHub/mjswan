/**
 * One graph for several termination terms: the same trade as observation fusion, since
 * the fixed per-`ort.run()` cost does not shrink with the graph.
 *
 * The output is a bool *lane* per term rather than one verdict: the manager reports
 * which term fired and splits `time_out` from real terminations, both of which OR-ing
 * inside the graph would throw away. Lanes wear the single-term interface
 * (`FusedLane`), and the manager runs the graph once per evaluation, before any lane
 * is read.
 */

import { TerminationBase, type TerminationConfig } from './TerminationBase';
import type { OnnxInputSlot, OnnxSession, OnnxTensorLike, SlotReader } from '../onnx/session';
import { buildFeeds } from '../onnx/session';
import type { PolicyRunner } from '../policy/PolicyRunner';
import type { PolicyState } from '../policy/types';

export interface FusedTerminationLane {
  name: string;
  /** Whether this lane is a truncation rather than a failure. */
  time_out?: boolean;
}

export interface FusedTerminationConfig {
  /** Path to the group's graph; its presence is what marks the entry fused. */
  fused: string;
  input_slots?: OnnxInputSlot[];
  lanes: FusedTerminationLane[];
}

export interface FusedTerminationDeps {
  session: OnnxSession;
  readSlot: SlotReader;
}

/** Whether a terminations-config entry is the fused graph rather than one term. */
export function isFusedTerminationConfig(entry: unknown): entry is FusedTerminationConfig {
  return (
    typeof entry === 'object' &&
    entry !== null &&
    typeof (entry as { fused?: unknown }).fused === 'string' &&
    Array.isArray((entry as { lanes?: unknown }).lanes)
  );
}

function isTruthy(data: OnnxTensorLike['data'], lane: number): boolean {
  const value = data[lane];
  return typeof value === 'bigint' ? value !== 0n : Boolean(value);
}

export class FusedTermination {
  private verdicts: boolean[];

  constructor(
    private readonly config: FusedTerminationConfig,
    private readonly deps: FusedTerminationDeps,
  ) {
    this.verdicts = config.lanes.map(() => false);
  }

  get lanes(): FusedTerminationLane[] {
    return this.config.lanes;
  }

  verdict(lane: number): boolean {
    return this.verdicts[lane] ?? false;
  }

  reset(): void {
    this.verdicts = this.config.lanes.map(() => false);
  }

  /** Run the graph once and latch every lane; as in `OnnxTermination`, a failure holds them. */
  async step(): Promise<void> {
    try {
      const { feeds, missing } = buildFeeds(this.config.input_slots, this.deps.readSlot);
      if (missing) {
        console.warn(
          `[FusedTermination] could not read slot ${missing}; holding the previous verdicts.`,
        );
        return;
      }
      const outputs = await this.deps.session.run(feeds);
      const first = Object.values(outputs)[0];
      if (!first) {
        console.warn('[FusedTermination] the graph produced no output.');
        return;
      }
      this.verdicts = this.config.lanes.map((_, lane) => isTruthy(first.data, lane));
    } catch (error) {
      console.warn('[FusedTermination] graph failed:', error);
    }
  }
}

/** One lane of a fused graph, read-only: the manager runs the shared graph. */
export class FusedLane extends TerminationBase {
  constructor(
    runner: PolicyRunner,
    config: TerminationConfig,
    private readonly group: FusedTermination,
    private readonly lane: number,
  ) {
    super(runner, config);
  }

  evaluate(_state: PolicyState): boolean {
    return this.group.verdict(this.lane);
  }
}
