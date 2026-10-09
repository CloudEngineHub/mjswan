import type { OnnxInputSlot } from '../onnx/session';
import type { ModelFieldDefaults } from './modelWrite';

export type EventConfig = {
  name: string;
  params?: Record<string, unknown>;
  /** Set for an ONNX-backed event; without `mode`/`onnx` it is a reset-only plugin term. */
  mode?: 'startup' | 'reset' | 'interval' | 'manual';
  /** Set by the build for a term it could not trace; `reason` says why. */
  native?: boolean;
  reason?: string;
  onnx?: string;
  rand_dim?: number;
  input_slots?: OnnxInputSlot[];
  write_targets?: unknown[];
  rand_ranges?: Array<[number, number]>;
  interval_range_s?: [number, number];
  is_global_time?: boolean;
  min_step_count_between_reset?: number;
  /** Control-panel text: a `manual` term's button, an `interval` term's arm checkbox. */
  label?: string;
  /** `mode="manual"` only: the `mode="interval"` term whose armed schedule greys it out. */
  disabled_when?: string;
};

export type EventContext = {
  mjModel: import('mujoco').MjModel | null;
  mjData: import('mujoco').MjData | null;
  /** Only a model write needs it, to call `mj_setConst`. */
  mujoco?: import('mujoco').MainModule | null;
  /** The model-lifetime snapshot a model write records the compiled value in, which the
   * runtime restores before an MDP switch. */
  modelDefaults?: ModelFieldDefaults | null;
  terrainData?: TerrainData | null;
};

export type TerrainData = {
  flat_patches?: Record<string, number[][]>;
};

export abstract class EventBase {
  protected config: EventConfig;

  constructor(config: EventConfig) {
    this.config = config;
  }

  /** Called on every episode reset. */
  abstract onReset(context: EventContext): void;
}

export type EventConstructor = new (config: EventConfig) => EventBase;
