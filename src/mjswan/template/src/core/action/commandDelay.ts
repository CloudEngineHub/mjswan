/**
 * mjlab's actuator command delay (`ActuatorCfg.delay_*`) for one env: each physics step
 * the actuator acts on the target issued `lag` steps ago, with `lag` drawn from
 * `[min_lag, max_lag]`. A port of `DelayBuffer` over `CircularBuffer` at batch size 1.
 */

/** One delay setting and the term targets it covers, as the build resolved them. */
export interface DelaySpec {
  /** Positions in the term's target list. */
  indices: number[];
  minLag: number;
  maxLag: number;
  /** Chance of keeping the current lag when it would be redrawn. */
  holdProb: number;
  /** Physics steps between redraws; 0 redraws every step. */
  updatePeriod: number;
  /** Start the redraw schedule at a random phase, as mjlab staggers its envs. */
  perEnvPhase: boolean;
}

/** The draws a delay makes; `SeededRng` provides them. */
export interface DelayRng {
  next(): number;
}

/** The manifest's form of a {@link DelaySpec}, with joints by name. */
export interface DelayConfigEntry {
  joints: string[];
  min_lag?: number;
  max_lag?: number;
  hold_prob?: number;
  update_period?: number;
  per_env_phase?: boolean;
}

export class CommandDelay {
  private readonly history: Float32Array[];
  private pointer = 0;
  private lag = 0;
  private stepCount = 0;
  private phase = 0;

  constructor(
    readonly spec: DelaySpec,
    private readonly rng: DelayRng,
  ) {
    const length = Math.max(spec.maxLag, 0) + 1;
    this.history = Array.from({ length }, () => new Float32Array(spec.indices.length));
    this.reset();
  }

  /**
   * Start over, as mjlab's entity reset does at an episode's start. It zeroes the targets
   * and writes them once before the next step, so the history starts as that zero frame
   * and the lag schedule has taken its first step.
   */
  reset(): void {
    for (const frame of this.history) frame.fill(0);
    this.lag = 0;
    this.stepCount = 0;
    const { updatePeriod, perEnvPhase } = this.spec;
    this.phase = updatePeriod > 0 && perEnvPhase ? this.randint(0, updatePeriod - 1) : 0;
    this.advance();
  }

  /** One physics step: push `values` at `spec.indices`, then replace them with the delayed ones. */
  apply(values: Float32Array): void {
    const { indices } = this.spec;
    const size = this.history.length;
    this.pointer = (this.pointer + 1) % size;
    const frame = this.history[this.pointer];
    for (let i = 0; i < indices.length; i++) frame[i] = values[indices[i]];
    this.advance();
    const source = this.history[(this.pointer - this.lag + size) % size];
    for (let i = 0; i < indices.length; i++) values[indices[i]] = source[i];
  }

  /** mjlab's `_update_lags`: one step of the redraw schedule. */
  private advance(): void {
    const { minLag, maxLag, holdProb, updatePeriod } = this.spec;
    let redraw = updatePeriod > 0 ? (this.stepCount + this.phase) % updatePeriod === 0 : true;
    // mjlab draws the candidate every step, redraw or not, so the stream stays aligned.
    const candidate = this.randint(minLag, maxLag);
    if (holdProb > 0) redraw = this.rng.next() >= holdProb && redraw;
    if (redraw) this.lag = candidate;
    this.stepCount += 1;
  }

  private randint(low: number, high: number): number {
    return low + Math.floor(this.rng.next() * (high - low + 1));
  }
}

/**
 * Each delay entry's joints resolved to positions among `targetNames`. An entry whose
 * joints are all elsewhere is dropped; an unknown joint warns, since its command would
 * then arrive undelayed.
 */
export function resolveDelays(
  entries: readonly DelayConfigEntry[] | undefined,
  targetNames: readonly string[],
  termKey: string,
): DelaySpec[] {
  const specs: DelaySpec[] = [];
  for (const entry of entries ?? []) {
    const indices: number[] = [];
    for (const joint of entry.joints ?? []) {
      const index = targetNames.indexOf(joint);
      if (index >= 0) indices.push(index);
      else console.warn(`[ActionManager] Action term "${termKey}": delay names "${joint}", which it does not drive.`);
    }
    if (indices.length === 0) continue;
    const minLag = Math.max(0, Math.floor(entry.min_lag ?? 0));
    specs.push({
      indices,
      minLag,
      maxLag: Math.max(minLag, Math.floor(entry.max_lag ?? minLag)),
      holdProb: entry.hold_prob ?? 0,
      updatePeriod: Math.max(0, Math.floor(entry.update_period ?? 0)),
      perEnvPhase: entry.per_env_phase ?? true,
    });
  }
  return specs;
}
