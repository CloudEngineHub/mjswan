/**
 * A termination term whose body is a traced ONNX graph; one generic class covers every
 * one, since the graph and its input slots are data in `policy.json`.
 *
 * The output is ORT's `bool` dtype (a `Uint8Array` of 0/1), and any non-zero element
 * means done, mjlab's per-env semantics at N=1. The manager awaits `step()` before it
 * reads `evaluate()`.
 */

import { TerminationBase, type TerminationConfig } from './TerminationBase';
import type { OnnxInputSlot, OnnxSession, SlotReader } from '../onnx/session';
import { buildFeeds } from '../onnx/session';
import type { PolicyRunner } from '../policy/PolicyRunner';
import type { PolicyState } from '../policy/types';

export interface OnnxTerminationConfig extends TerminationConfig {
  onnx: string;
  input_slots?: OnnxInputSlot[];
}

export interface OnnxTerminationDeps {
  session: OnnxSession;
  readSlot: SlotReader;
}

export class OnnxTermination extends TerminationBase {
  private readonly onnxConfig: OnnxTerminationConfig;
  private readonly deps: OnnxTerminationDeps;
  private done = false;

  constructor(
    runner: PolicyRunner,
    config: OnnxTerminationConfig,
    deps: OnnxTerminationDeps,
  ) {
    super(runner, config);
    this.onnxConfig = config;
    this.deps = deps;
  }

  evaluate(_state: PolicyState): boolean {
    return this.done;
  }

  reset(): void {
    this.done = false;
  }

  /**
   * Run the graph and latch its verdict. A failure holds the previous one and warns: a
   * verdict stuck at `false` is an episode that never ends.
   */
  async step(): Promise<void> {
    try {
      const { feeds, missing } = buildFeeds(this.onnxConfig.input_slots, this.deps.readSlot);
      if (missing) {
        console.warn(
          `[OnnxTermination] "${this.config.name}" could not read slot ${missing}; ` +
            'holding the previous verdict.',
        );
        return;
      }
      const outputs = await this.deps.session.run(feeds);
      const first = Object.values(outputs)[0];
      if (!first) {
        console.warn(`[OnnxTermination] "${this.config.name}" produced no output.`);
        return;
      }
      this.done = isAnyTruthy(first.data);
    } catch (error) {
      console.warn(`[OnnxTermination] "${this.config.name}" failed:`, error);
    }
  }
}

function isAnyTruthy(data: Float32Array | BigInt64Array | Uint8Array): boolean {
  for (let i = 0; i < data.length; i++) {
    if (Number(data[i]) !== 0) return true;
  }
  return false;
}
