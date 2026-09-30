import React from 'react';

/**
 * Marks a block whose model has runtime knobs (a parametric capture) rather
 * than being a snapshot of one fixed setting.
 *
 * The glyph is a Lissajous curve -- the shape a pair of parameters traces out
 * -- which says "parametric" without needing a word, and stays legible as a
 * single woven line at badge size where a letterform would not.
 */

/** Frequency ratio of the two axes. 3:4 gives the woven square that reads as
    a parametric plot; simpler ratios collapse into a figure-eight. */
const X_FREQ = 3;
const Y_FREQ = 4;
/** Enough segments for a smooth curve, few enough to keep the path short. */
const SEGMENTS = 96;
const VIEW = 24;
const RADIUS = 10;

/** Built once at module load rather than inlined: 96 points of path data is
    noise in the source, and this costs one loop at startup. */
const LISSAJOUS_PATH = (() => {
  const points: string[] = [];
  for (let i = 0; i <= SEGMENTS; i++) {
    const t = (2 * Math.PI * i) / SEGMENTS;
    const x = VIEW / 2 + RADIUS * Math.sin(X_FREQ * t);
    const y = VIEW / 2 + RADIUS * Math.sin(Y_FREQ * t);
    points.push(`${x.toFixed(2)} ${y.toFixed(2)}`);
  }
  return `M ${points.join(' L ')}`;
})();

export const ParametricBadge: React.FC<{
  /** Rendered size in design px. */
  size?: number;
  /** Tooltip; defaults to a plain description of what the mark means. */
  title?: string;
}> = ({ size = 14, title = 'Parametric model: has knobs of its own' }) => (
  <svg
    viewBox={`0 0 ${VIEW} ${VIEW}`}
    width={`${size}rem`}
    height={`${size}rem`}
    fill="none"
    aria-hidden
    style={{ display: 'block', pointerEvents: 'none' }}
  >
    <title>{title}</title>
    <path
      d={LISSAJOUS_PATH}
      stroke="#ffffff"
      strokeWidth={1.4}
      strokeLinejoin="round"
      strokeLinecap="round"
      // Transparent so it reads as a watermark over the artwork rather than
      // competing with the tile's own controls.
      opacity={0.45}
    />
  </svg>
);
