/** Query-parameter view state, so a copied link reopens the same view (guides/embedding.md). */

export const PROJECT_PARAM = 'project';
export const SCENE_PARAM = 'scene';
export const POLICY_PARAM = 'policy';
export const PANEL_PARAM = 'panel';
export const VIZ_PARAM = 'viz';

export interface UrlState {
  /** Id of the active entry, or null to leave the parameter off. */
  project: string | null;
  scene: string | null;
  policy: string | null;
  panel: boolean;
  /** Debug Viz "Enabled". */
  viz: boolean;
}

/** Rewrite a query string to carry `state`, leaving unrelated parameters alone. */
export function applyUrlState(search: string, state: UrlState): string {
  const params = new URLSearchParams(search);
  const set = (key: string, value: string | null) =>
    value === null ? params.delete(key) : params.set(key, value);
  set(PROJECT_PARAM, state.project);
  set(SCENE_PARAM, state.scene);
  set(POLICY_PARAM, state.policy);
  // Booleans are written only when off, so a default view keeps a bare URL.
  set(PANEL_PARAM, state.panel ? null : '0');
  set(VIZ_PARAM, state.viz ? null : '0');
  return params.toString();
}
