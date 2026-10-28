import { describe, expect, it } from 'vitest';

import { nextDeadline } from '../stepDeadline';

describe('nextDeadline', () => {
  it('holds 200 Hz under timers that clamp to 4 ms and wake late', () => {
    const period = 5;
    const compute = 1.3;
    const sleep = (ms: number) => Math.max(ms, 4) + 0.6;
    let now = 0;
    let deadline = 0;
    for (let step = 0; step < 1000; step++) {
      now += compute;
      deadline = nextDeadline(deadline, now, period);
      if (deadline > now) now += sleep(deadline - now);
    }
    expect(1000 / (now / 1000)).toBeCloseTo(200, 0);
  });

  it('drops the backlog once far behind', () => {
    expect(nextDeadline(0, 1000, 5)).toBe(1000);
    expect(nextDeadline(0, 100, 5)).toBe(5);
  });
});
