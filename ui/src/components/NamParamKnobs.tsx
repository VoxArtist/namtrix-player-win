import { useCallback } from 'react';
import type { NamParamSpec } from '../types/chain';
import { KnobControl } from './KnobControl';
import { linearScale } from './knobScale';
import { KNOB_SIZE_SECONDARY, WHITE, MUTED, HIGHLIGHT } from './theme';

/**
 * The knob row for a parametric NAM model (ConcatWaveNet / HyperWaveNet /
 * ConcatLSTM). Snapshot models declare no specs and render nothing, so the
 * caller can mount this unconditionally.
 *
 * Values are in the model's own units -- the numbers printed on the real
 * faceplate -- while KnobControl works in normalized 0..1, so each knob
 * carries a linearScale over its spec range and converts on the way out.
 *
 * `switch` params are a discrete choice, not a range: a knob would invite
 * fractional values that native rejects outright, so they render as a
 * cycling button showing the current option's name.
 */

/** Width of one knob cell. KnobControl's label is `white-space: nowrap` with
    no width of its own, so without a fixed cell the labels of adjacent knobs
    overlap each other. Wide enough for the 36px knob plus breathing room. */
const CELL_WIDTH = 76;

/** Longest label that fits a cell before it starts colliding. */
const LABEL_MAX_CHARS = 11;

/**
 * Fit a model's knob name into a cell.
 *
 * Amp knob names carry qualifiers the faceplate has room for and this cell
 * does not -- "Volume I (High Treb)" is a real example. Drop the parenthetical
 * first, since it disambiguates rather than names, and only then truncate.
 * The full name always stays reachable as the cell's tooltip.
 */
const shortLabel = (name: string): string => {
  const withoutQualifier = name.replace(/\s*\([^)]*\)\s*$/, '').trim() || name;
  return withoutQualifier.length > LABEL_MAX_CHARS
    ? `${withoutQualifier.slice(0, LABEL_MAX_CHARS - 1)}\u2026`
    : withoutQualifier;
};

/** Decimals for a knob's readout, from how wide its range is. A 0..10 gain
    wants one decimal; a 0..1 blend would read as a flat "0" / "1" without
    two. Purely cosmetic -- the value sent to native is never rounded here. */
const decimalsFor = (min: number, max: number): number => {
  const span = Math.abs(max - min);
  if (span >= 20) return 0;
  if (span >= 2) return 1;
  return 2;
};

interface NamParamKnobsProps {
  specs: NamParamSpec[];
  /** Positionally matched to `specs`; a short array falls back to defaults. */
  values: number[];
  /** (index, value in the spec's own units) */
  onChange: (index: number, value: number) => void;
  onDragStateChange?: (dragging: boolean) => void;
}

export function NamParamKnobs({
  specs,
  values,
  onChange,
  onDragStateChange,
}: NamParamKnobsProps) {
  const valueAt = useCallback(
    (index: number) => {
      const v = values[index];
      return Number.isFinite(v) ? v : specs[index].default;
    },
    [values, specs]
  );

  if (specs.length === 0) return null;

  return (
    <div
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'flex-end',
        justifyContent: 'center',
        gap: '4rem',
      }}
    >
      {specs.map((spec, index) => {
        const value = valueAt(index);

        if (spec.type === 'switch') {
          const names = spec.enumNames ?? [];
          // Guard the index: a preset saved against a model with more options
          // can land past the end of this model's list.
          const current = Math.min(Math.max(Math.round(value), 0), Math.max(names.length - 1, 0));
          const label = names[current] ?? String(current);
          return (
            <div
              key={spec.name}
              title={spec.name}
              style={{
                width: `${CELL_WIDTH}rem`,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 4,
              }}
            >
              <button
                type="button"
                onClick={() => names.length > 0 && onChange(index, (current + 1) % names.length)}
                title={`${spec.name}: ${label}`}
                style={{
                  maxWidth: '100%',
                  padding: '4px 8px',
                  background: 'transparent',
                  color: WHITE,
                  border: `1px solid ${HIGHLIGHT}`,
                  borderRadius: 4,
                  fontSize: 11,
                  cursor: names.length > 1 ? 'pointer' : 'default',
                }}
              >
                {label}
              </button>
              <span
                style={{
                  fontSize: 11,
                  color: MUTED,
                  maxWidth: '100%',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {shortLabel(spec.name)}
              </span>
            </div>
          );
        }

        const span = spec.max - spec.min;
        // A degenerate range (min == max) would divide by zero in the
        // normalization; the knob is meaningless anyway, so pin it.
        const norm = span === 0 ? 0 : (value - spec.min) / span;
        return (
          <div
            key={spec.name}
            title={spec.name}
            style={{
              width: `${CELL_WIDTH}rem`,
              display: 'flex',
              justifyContent: 'center',
              // The label is nowrap and centered; the cell clips whatever a
              // long name still overflows rather than letting it run into the
              // neighbouring knob.
              overflow: 'hidden',
            }}
          >
            <KnobControl
              label={shortLabel(spec.name)}
              value={Math.min(Math.max(norm, 0), 1)}
              onChange={(n) => onChange(index, spec.min + n * span)}
              onDragStateChange={onDragStateChange}
              size={KNOB_SIZE_SECONDARY}
              labelBottom
              thumb="secondary"
              scale={linearScale(spec.min, spec.max, '', decimalsFor(spec.min, spec.max))}
              defaultValue={span === 0 ? 0 : (spec.default - spec.min) / span}
            />
          </div>
        );
      })}
    </div>
  );
}
