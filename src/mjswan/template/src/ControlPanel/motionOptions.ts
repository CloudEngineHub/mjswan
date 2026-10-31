import type { ComboboxData } from '@mantine/core';

import type { SelectOption } from './ControlPanel';

/**
 * The motion selector's data: ungrouped motions as plain options, then each `group`
 * heading in order of first appearance. Ungrouped ones go first so none reads as part
 * of the group above it.
 */
export function motionOptions(motions: ReadonlyArray<{ name: string; group?: string }>): ComboboxData {
  const ungrouped: SelectOption[] = [];
  const groups = new Map<string, SelectOption[]>();
  for (const { name, group } of motions) {
    const option = { value: name, label: name };
    if (group === undefined) {
      ungrouped.push(option);
    } else {
      groups.set(group, [...(groups.get(group) ?? []), option]);
    }
  }
  return [...ungrouped, ...Array.from(groups, ([group, items]) => ({ group, items }))];
}
