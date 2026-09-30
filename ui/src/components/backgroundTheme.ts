import { useSyncExternalStore } from 'react';

/**
 * User-chosen background gradient for the plugin window.
 *
 * A per-machine UI preference, not plugin state: it lives in webview
 * localStorage like the hint-bar toggle, so presets, undo and the DAW session
 * never carry it (see helpText.ts for the same pattern).
 *
 * The gradient runs corner to corner -- colour 1 at the top-left, colour 2 at
 * the bottom-right -- so both stops sit where the user picked them rather than
 * at the midpoints a plain vertical fade would give. It is composited over the
 * background plate as a tint, not painted instead of it.
 */

/** Mid-grey is soft-light's no-op point, so the shipped default leaves the
    metal plate exactly as photographed. Anything the user picks tints it. */
export const DEFAULT_BG_START = '#808080';
export const DEFAULT_BG_END = '#808080';

const START_KEY = 't3k.bgStart';
const END_KEY = 't3k.bgEnd';

/** Guards what goes into a CSS gradient: only #rgb / #rrggbb, so a corrupt or
    hand-edited storage value can't inject arbitrary CSS. */
const isHexColor = (value: string): boolean => /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(value);

const read = (key: string, fallback: string): string => {
  try {
    const stored = localStorage.getItem(key);
    return stored && isHexColor(stored) ? stored : fallback;
  } catch {
    return fallback;
  }
};

let startColor = read(START_KEY, DEFAULT_BG_START);
let endColor = read(END_KEY, DEFAULT_BG_END);

// useSyncExternalStore compares snapshots by identity, so the snapshot object
// is rebuilt only when a colour actually changes. Returning a fresh object on
// every read would re-render the whole tree on every store poll.
let snapshot = { start: startColor, end: endColor };
const listeners = new Set<() => void>();

const emit = () => {
  snapshot = { start: startColor, end: endColor };
  listeners.forEach((listener) => listener());
};

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

const write = (key: string, value: string) => {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Storage unavailable (private mode, cleared site data): the colour still
    // applies for this session, it just won't survive a relaunch.
  }
};

export const setBackgroundColors = (start: string, end: string) => {
  const nextStart = isHexColor(start) ? start : startColor;
  const nextEnd = isHexColor(end) ? end : endColor;
  if (nextStart === startColor && nextEnd === endColor) return;
  startColor = nextStart;
  endColor = nextEnd;
  write(START_KEY, startColor);
  write(END_KEY, endColor);
  emit();
};

export const resetBackgroundColors = () =>
  setBackgroundColors(DEFAULT_BG_START, DEFAULT_BG_END);

export const useBackgroundColors = () => useSyncExternalStore(subscribe, () => snapshot);

/** The CSS value for the window background. */
export const backgroundGradient = (start: string, end: string): string =>
  `linear-gradient(to bottom right, ${start} 0%, ${end} 100%)`;

/**
 * True when a colour is light enough that white chrome would disappear on it.
 *
 * Uses relative luminance rather than a plain RGB average: the eye is far more
 * sensitive to green than to blue, so a saturated yellow reads as bright while
 * a saturated blue of the same average does not. The 0.55 line is where white
 * text stops holding contrast in practice.
 */
export const isLightColor = (hex: string): boolean => {
  const value = hex.replace('#', '');
  const full =
    value.length === 3
      ? value
          .split('')
          .map((c) => c + c)
          .join('')
      : value;
  if (full.length !== 6) return false;
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16) / 255);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55;
};
