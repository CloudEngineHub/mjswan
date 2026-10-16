/**
 * The action terms of one policy, run as mjlab's `ActionManager` runs them: `process`
 * once per control step, `apply` once per substep, `reset` at an episode's start.
 *
 * A native term processes in `processAction`; a traced one (ADR 0009) in its
 * `OnnxActionProcess`, which this awaits. Either way `applyAction` takes it from there.
 */

import { inHold, processAction, type ProcessContext, type ResolvedActionTerm } from './applyAction';
import type { OnnxActionProcess } from './OnnxActionProcess';

/** One term and, for a traced one, the graph that processes it. */
export interface ManagedActionTerm {
  name: string;
  term: ResolvedActionTerm;
  traced?: OnnxActionProcess | null;
}

/** `processed_actions` by either name mjlab gives it: the property or its backing field. */
const PROCESSED_FIELDS = new Set(['processed_actions', '_processed_actions']);

export class ActionManager {
  readonly terms: ResolvedActionTerm[];
  private readonly byName: Map<string, ManagedActionTerm>;

  constructor(private readonly entries: readonly ManagedActionTerm[]) {
    this.terms = entries.map((entry) => entry.term);
    this.byName = new Map(entries.map((entry) => [entry.name, entry]));
  }

  /** Once per control step, after the policy: every term's `processed`, in config order. */
  async process(actions: Float32Array, context: ProcessContext): Promise<void> {
    for (const { term, traced } of this.entries) {
      if (!traced) {
        processAction(term, actions, context);
        continue;
      }
      term.holding = inHold(term.holdS, context);
      const raw = new Float32Array(term.actionIndices.length);
      for (let i = 0; i < raw.length; i++) raw[i] = actions[term.actionIndices[i]] ?? 0;
      term.active = await traced.process(raw, term.processed, term.rampFrom);
    }
  }

  /** mjlab's `action_manager.reset`: each traced term's own `reset`, in config order. */
  async reset(): Promise<void> {
    for (const { traced } of this.entries) await traced?.reset();
  }

  /** mjlab's actuator reset, which comes with the sim state's: the command delays start over. */
  resetActuators(): void {
    for (const term of this.terms) for (const delay of term.delays) delay.reset();
  }

  /**
   * What an observation reads off a term (`get_term(name).<field>`): a traced term's
   * state field, else any term's processed action.
   */
  getStateField(name: string, field: string): Float32Array | null {
    const entry = this.byName.get(name);
    if (!entry) return null;
    const state = entry.traced?.getStateField(field);
    if (state) return state;
    return PROCESSED_FIELDS.has(field) ? Float32Array.from(entry.term.processed) : null;
  }
}
