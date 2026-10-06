import { Checkbox } from '@mantine/core';

import type { DebugVisDescriptor } from '../engine';
import type { ViewerGuiNode } from '../manifest';
import { CommandSection } from './CommandSection';
import { LabeledInput } from './LabeledInput';
import { SliderRow } from './SliderRow';

/** The slot mjlab's Debug Viz folder lists its drawings in (`mjlab/gui.py`). */
export const DEBUG_VIS_SLOT = 'debug_vis';

/** Wires a recorded control to the engine. */
export interface ViewerGuiBinding {
  value: number | boolean;
  onChange: (value: number | boolean) => void;
}

/** A recorded control's key: its folders, then its label (`Scene/Camera/FOV (°)`). */
export function viewerGuiPath(folders: string[], label: string): string {
  return [...folders, label].join('/');
}

/**
 * mjlab's viewer GUI as recorded. A control without a binding (mjlab's "All envs": the
 * browser runs one env) shows its default, disabled, so the section still reads as mjlab's.
 */
export function ViewerGuiSection({
  nodes,
  bindings,
  debugVis,
  onDebugVisChange,
  folders = [],
}: {
  nodes: ViewerGuiNode[];
  bindings: Record<string, ViewerGuiBinding>;
  debugVis: DebugVisDescriptor[];
  onDebugVisChange: (id: string, enabled: boolean) => void;
  folders?: string[];
}) {
  return (
    <>
      {nodes.map((node, index) => {
        if (node.type === 'folder') {
          return (
            <CommandSection key={`${node.label}-${index}`} label={node.label} expandByDefault={true}>
              <ViewerGuiSection
                nodes={node.children}
                bindings={bindings}
                debugVis={debugVis}
                onDebugVisChange={onDebugVisChange}
                folders={[...folders, node.label]}
              />
            </CommandSection>
          );
        }
        if (node.type === 'slot') {
          if (node.name !== DEBUG_VIS_SLOT) return null;
          return debugVis.map((entry) => (
            <LabeledInput key={entry.id} id={`debugvis:${entry.id}`} label={entry.label}>
              <Checkbox
                id={`debugvis:${entry.id}`}
                checked={entry.enabled}
                onChange={(event) => onDebugVisChange(entry.id, event.currentTarget.checked)}
                size="xs"
              />
            </LabeledInput>
          ));
        }
        const path = viewerGuiPath(folders, node.label);
        const binding = bindings[path];
        const id = `viewer:${path}`;
        if (node.type === 'checkbox') {
          return (
            <LabeledInput key={id} id={id} label={node.label}>
              <Checkbox
                id={id}
                title={node.hint}
                checked={binding ? Boolean(binding.value) : Boolean(node.default)}
                onChange={(event) => binding?.onChange(event.currentTarget.checked)}
                disabled={!binding}
                size="xs"
              />
            </LabeledInput>
          );
        }
        return (
          <SliderRow
            key={id}
            id={id}
            label={node.label}
            value={binding ? Number(binding.value) : (node.default ?? 0)}
            min={node.min ?? 0}
            max={node.max ?? 1}
            step={node.step}
            onChange={(value) => binding?.onChange(value)}
            disabled={!binding}
          />
        );
      })}
    </>
  );
}
