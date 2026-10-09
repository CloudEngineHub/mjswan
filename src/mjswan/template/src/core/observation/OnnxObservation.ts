/**
 * An observation term whose body is a traced ONNX graph. One generic class covers every
 * traced observation: the graph, its input slots and the clip/scale metadata are all
 * data in `policy.json`.
 *
 * `size` comes from the build because `PolicyRunner` needs each term's width
 * synchronously at load, while inference is async.
 *
 * Unlike `OnnxCommand`/`OnnxEvent`, which skip a frame rather than queue, `compute()`
 * returns a promise the group awaits — a stale observation degrades control directly.
 */

import { ObservationBase, type ObservationConfig } from './ObservationBase';
import { assertCommandTermBound } from './NativeObservation';
import {
  applyObservationPipeline,
  conformToSize,
  type ObservationClip,
  type ObservationScale,
} from './pipeline';
import type { OnnxInputSlot, OnnxSession, SlotReader } from '../onnx/session';
import { buildFeeds, toFloat32 } from '../onnx/session';
import type { PolicyRunner } from '../policy/PolicyRunner';
import type { PolicyState } from '../policy/types';

export interface OnnxObservationConfig extends ObservationConfig {
  onnx: string;
  /** Output width, from the build (see class docs). */
  size: number;
  input_slots?: OnnxInputSlot[];
  scale?: ObservationScale;
  clip?: ObservationClip;
}

export interface OnnxObservationDeps {
  session: OnnxSession;
  readSlot: SlotReader;
}

/** Whether a config entry names a traced-ONNX observation. */
export function isOnnxObservationConfig(
  entry: ObservationConfig,
): entry is OnnxObservationConfig {
  return typeof (entry as { onnx?: unknown }).onnx === 'string';
}

export class OnnxObservation extends ObservationBase<OnnxObservationConfig> {
  private readonly deps: OnnxObservationDeps;
  /** Last completed value, served if a later frame's inference fails. */
  private last: Float32Array;

  constructor(
    runner: PolicyRunner,
    config: OnnxObservationConfig,
    deps: OnnxObservationDeps,
  ) {
    super(runner, config);
    for (const slot of config.input_slots ?? []) {
      if (slot.command) assertCommandTermBound(runner, config.name, slot.command);
    }
    this.deps = deps;
    this.last = new Float32Array(config.size);
  }

  get size(): number {
    return this.config.size;
  }

  async compute(_state: PolicyState): Promise<Float32Array> {
    const { feeds, missing } = buildFeeds(this.config.input_slots, this.deps.readSlot);
    if (missing) {
      // An unsupplied slot means absent state: serve the last good value, not zeros.
      console.warn(
        `[OnnxObservation] "${this.config.name}" could not read slot ${missing}; ` +
          'reusing the previous value.',
      );
      return this.last;
    }

    const outputs = await this.deps.session.run(feeds);
    const first = Object.values(outputs)[0];
    if (!first) {
      console.warn(`[OnnxObservation] "${this.config.name}" produced no output.`);
      return this.last;
    }
    const raw = toFloat32(first.data);
    this.last = applyObservationPipeline(conformToSize(raw, this.config.size), this.config);
    return this.last;
  }
}
