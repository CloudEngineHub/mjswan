/**
 * When the step loop runs its next control step: one period after the last deadline, not
 * after the step that just ran, so a timer that wakes late (Chrome clamps nested timers to
 * 4 ms) shortens the next wait instead of stretching every step.
 */

/** Further behind than this (a hidden tab, a long stall), the loop drops the backlog. */
const MAX_BEHIND_MS = 250;

export function nextDeadline(previous: number, now: number, period: number): number {
  const next = previous + period;
  return now - next > MAX_BEHIND_MS ? now : next;
}
