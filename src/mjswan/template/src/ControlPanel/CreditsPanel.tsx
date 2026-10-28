import { useEffect, useState, type ReactNode } from 'react';
import { Anchor, Badge, Box, Button, Group, Stack, Text } from '@mantine/core';
import type { CreditEntry, LicenseEntry } from '../manifest';
import { MJSWAN_VERSION } from '../Version';

const BASE = import.meta.env.BASE_URL || '/';

// `install_spa` puts the engine's LICENSE and THIRD_PARTY_LICENSES.md at every build's root.
const ENGINE: LicenseEntry = {
  spdx: 'Apache-2.0',
  license: async () => {
    const response = await fetch(`${BASE}LICENSE`);
    if (!response.ok) throw new Error(`LICENSE: ${response.status}`);
    return response.text();
  },
};

/** The Credits tab's scene part: the scene in view, its skins, the work. */
export interface SceneCredits {
  scene: string;
  project: string;
  credits: CreditEntry[];
  work?: LicenseEntry;
  /** The scene's skins in menu order. */
  skins: Array<{ id: string; name: string }>;
  /** The skin on screen, by id. */
  skin: string | null;
}

/** A VRM's meta is the author's text: link it only when it is a web address. */
function webLink(value: string | undefined) {
  if (!value) return null;
  return /^https?:\/\//i.test(value) ? (
    <Anchor href={value} target="_blank" rel="noopener noreferrer" inherit>
      {value}
    </Anchor>
  ) : (
    value
  );
}

/** A license file's text, fetched once the row shows it. */
function useText(load: (() => Promise<string>) | undefined, wanted: boolean) {
  const [state, setState] = useState<{ load?: () => Promise<string>; text?: string; failed?: boolean }>({});
  // A row React reuses for another scene's file must not keep showing the old text.
  const own = state.load === load ? state : {};
  const done = own.text !== undefined || !!own.failed;
  useEffect(() => {
    if (!load || !wanted || done) return;
    let live = true;
    load().then(
      (text) => live && setState({ load, text }),
      () => live && setState({ load, failed: true }),
    );
    return () => {
      live = false;
    };
  }, [load, wanted, done]);
  return own;
}

/** A NOTICE or LICENSE as written, in monospace so a hard-wrapped text keeps its shape. */
function TextBlock({ load, name, wanted, long }: {
  load?: () => Promise<string>;
  name: string;
  wanted: boolean;
  /** A license text: smaller, and scrolled past a few paragraphs. */
  long?: boolean;
}) {
  const { text, failed } = useText(load, wanted);
  if (!load || !wanted) return null;
  return (
    <Box
      component="pre"
      m={0}
      ff="monospace"
      fz={long ? '0.7rem' : 'xs'}
      bg="var(--mantine-color-default-hover)"
      style={{
        padding: '0.5em 0.65em',
        whiteSpace: 'pre-wrap',
        overflowWrap: 'anywhere',
        borderRadius: 'var(--mantine-radius-sm)',
        maxHeight: long ? '14em' : undefined,
        overflowY: 'auto',
      }}
    >
      {failed ? `${name} could not be loaded.` : (text ?? 'Loading…')}
    </Box>
  );
}

function LicenseBadges({ spdx, restriction }: { spdx?: string; restriction?: string }) {
  return (
    <>
      {spdx && (
        <Badge variant="outline" color="gray" radius="xs" size="sm" tt="none" ff="monospace">
          {spdx === 'LicenseRef-custom' ? 'Custom' : spdx}
        </Badge>
      )}
      {restriction && (
        <Badge variant="light" color="yellow" radius="xs" size="sm" tt="none">
          {restriction}
        </Badge>
      )}
    </>
  );
}

function CreditRow({ title, mono, entry, suffix, onScreen, vrm, children }: {
  title: string;
  mono: boolean;
  entry: LicenseEntry;
  /** `.<component>`, or empty for a bare `LICENSE`; names the file in a load error. */
  suffix: string;
  onScreen?: boolean;
  vrm?: CreditEntry['vrm'];
  children?: ReactNode;
}) {
  const [full, setFull] = useState(false);
  return (
    <Stack gap="0.35em" py="0.6em" style={{ borderBottom: '1px solid var(--mantine-color-default-border)' }}>
      <Group gap="0.4em" wrap="wrap">
        <Text size="sm" ff={mono ? 'monospace' : undefined} fw={mono ? 400 : 500}>
          {title}
        </Text>
        <LicenseBadges spdx={entry.spdx} restriction={entry.restriction} />
        {vrm && (
          <Badge variant="light" color="indigo" radius="xs" size="sm" tt="none">
            VRM
          </Badge>
        )}
        {onScreen && (
          <Badge variant="light" color="teal" radius="xs" size="sm" tt="none">
            On screen
          </Badge>
        )}
        {entry.license && (
          <Button variant="subtle" size="compact-xs" ml="auto" onClick={() => setFull((v) => !v)}>
            {full ? 'Hide license' : 'License'}
          </Button>
        )}
      </Group>
      {entry.copyright && (
        <Text size="xs" c="dimmed" style={{ overflowWrap: 'anywhere' }}>
          {entry.copyright}
        </Text>
      )}
      {vrm && (
        <Text size="xs" c="dimmed" style={{ overflowWrap: 'anywhere' }}>
          {vrm.authors.length > 0 && <>by {vrm.authors.join(', ')} · </>}
          License: {webLink(vrm.license) ?? 'not stated'}
          {vrm.otherLicenseUrl && <> · Other terms: {webLink(vrm.otherLicenseUrl)}</>}
        </Text>
      )}
      {children}
      <TextBlock load={entry.notice} name={`NOTICE${suffix}`} wanted />
      <TextBlock load={entry.license} name={`LICENSE${suffix}`} wanted={full} long />
    </Stack>
  );
}

function CreditGroup({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Box>
      <Text size="xs" c="dimmed" tt="uppercase" fw={500} pt="sm" pb="0.2em" style={{ letterSpacing: '0.05em' }}>
        {label}
      </Text>
      {children}
    </Box>
  );
}

/** The Credits tab: per component its license, its NOTICE in full, its text on demand. */
export function CreditsPanel({ scene, project, credits, work, skins, skin }: SceneCredits) {
  const components = credits.filter((c) => c.skin === undefined);
  const menuOrder = new Map(skins.map((s, i) => [s.id, i]));
  // Menu order, the skin on screen first.
  const rank = (c: CreditEntry) => (c.skin === skin ? -1 : (menuOrder.get(c.skin!) ?? skins.length));
  const skinCredits = credits.filter((c) => c.skin !== undefined).sort((a, b) => rank(a) - rank(b));
  return (
    <Stack gap={0}>
      {components.length > 0 && (
        <CreditGroup label={scene}>
          {components.map((c) => (
            <CreditRow key={c.component} title={c.component} mono entry={c} suffix={`.${c.component}`} />
          ))}
        </CreditGroup>
      )}
      {skinCredits.length > 0 && (
        <CreditGroup label="Skins">
          {skinCredits.map((c) => (
            <CreditRow
              key={c.component}
              title={skins.find((s) => s.id === c.skin)?.name ?? c.component}
              mono={false}
              entry={c}
              suffix={`.${c.component}`}
              onScreen={c.skin === skin}
              vrm={c.vrm}
            />
          ))}
        </CreditGroup>
      )}
      {work && (
        <CreditGroup label={project}>
          <CreditRow title={project} mono={false} entry={work} suffix="" />
        </CreditGroup>
      )}
      <CreditGroup label="mjswan">
        <CreditRow title={`mjswan ${MJSWAN_VERSION}`} mono={false} entry={ENGINE} suffix="">
          <Anchor href={`${BASE}THIRD_PARTY_LICENSES.md`} target="_blank" size="xs">
            Third-party licenses
          </Anchor>
        </CreditRow>
      </CreditGroup>
    </Stack>
  );
}
