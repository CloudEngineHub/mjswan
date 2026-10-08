/**
 * `time_out` as a document before format 4 carries it: a native marker, not a graph.
 * Counts steps against `ceil(episode_length_s / step_dt)` as mjlab does, since summed
 * time drifts.
 *
 * A *truncation*, not a failure: the manager keeps the two apart.
 */

import { TerminationBase, type TerminationConfig } from './TerminationBase';
import type { PolicyRunner } from '../policy/PolicyRunner';
import type { PolicyState } from '../policy/types';

export interface TimeOutTerminationConfig extends TerminationConfig {
  episode_length_s?: number;
}

export class TimeOutTermination extends TerminationBase {
  /** mjlab's `max_episode_length`. */
  private readonly maxEpisodeLength: number;

  constructor(
    runner: PolicyRunner,
    config: TimeOutTerminationConfig,
    private readonly episodeLength: () => number,
    stepDt: number,
  ) {
    super(runner, config);
    // No finite horizon never times out; nor does a missing value, which would fire always.
    const declared = config.episode_length_s ?? 0;
    this.maxEpisodeLength =
      declared > 0 && stepDt > 0 ? Math.ceil(declared / stepDt) : Number.POSITIVE_INFINITY;
  }

  evaluate(_state: PolicyState): boolean {
    return this.episodeLength() >= this.maxEpisodeLength;
  }
}
