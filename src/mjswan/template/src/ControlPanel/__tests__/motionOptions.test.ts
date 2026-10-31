import { describe, expect, it } from 'vitest';

import { motionOptions } from '../motionOptions';

const option = (name: string) => ({ value: name, label: name });

describe('motionOptions', () => {
  it('lists motions flat, in order, when none has a group', () => {
    expect(motionOptions([{ name: 'b' }, { name: 'a' }])).toEqual([option('b'), option('a')]);
  });

  it('groups motions under their headings in order of first appearance', () => {
    const motions = [
      { name: 'walk 1', group: 'Walk forward' },
      { name: 'back 1', group: 'Walk backward' },
      { name: 'walk 2', group: 'Walk forward' },
    ];
    expect(motionOptions(motions)).toEqual([
      { group: 'Walk forward', items: [option('walk 1'), option('walk 2')] },
      { group: 'Walk backward', items: [option('back 1')] },
    ]);
  });

  it('keeps ungrouped motions as plain options ahead of the groups', () => {
    const motions = [{ name: 'walk', group: 'Walk forward' }, { name: 'idle' }];
    expect(motionOptions(motions)).toEqual([
      option('idle'),
      { group: 'Walk forward', items: [option('walk')] },
    ]);
  });
});
