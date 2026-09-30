import React from 'react';
import { rem } from '../hooks/useUiScale';
import { Undo2, Redo2 } from './icons';
import { AccountMenu } from './AccountMenu';
import { IconButton } from './IconButton';
import { PresetBar } from './PresetBar';
import { StereoModeToggle } from './StereoModeToggle';
import { HELP } from './helpText';
import { BORDER, FONT_MONO, MUTED, WHITE } from './theme';
import { NAMTRIX_VERSION } from '../version';
import { isLightColor, useBackgroundColors } from './backgroundTheme';
import type { usePresets } from '../hooks/usePresets';
import type { ActivePreset } from '../types/chain';
import type { User } from '../types/tone';

type PresetStore = ReturnType<typeof usePresets>;

// Lucide has no tuning fork, so this mimics its 24x24 stroke style.
const TuningForkIcon: React.FC<{ size?: number }> = ({ size = 18 }) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    style={{ width: rem(size), height: rem(size) }}
  >
    <path d="M8 3v7a4 4 0 0 0 8 0V3" />
    <line x1="12" y1="14" x2="12" y2="21" />
  </svg>
);

interface PluginHeaderProps {
  presetStore: PresetStore;
  activePreset: ActivePreset | null;
  /** Greys out the preset bar's New button (see PresetBar). */
  atDefault: boolean;
  onReset: () => void;
  stereoEnabled: boolean;
  onStereoToggle: (enabled: boolean) => void;
  showTuner: boolean;
  onToggleTuner: (show: boolean) => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  user: User | null;
  authenticated: boolean;
  onOpenSettings: () => void;
  onLogin: () => void;
  onLogout: () => void;
}

/**
 * Full-width top bar: logo, preset controls, stereo toggle, tuner, undo/redo
 * and the account menu. Memoized because Plugin re-renders on every chain
 * poll tick while nothing up here changes.
 */
export const PluginHeader = React.memo(function PluginHeader({
  presetStore,
  activePreset,
  atDefault,
  onReset,
  stereoEnabled,
  onStereoToggle,
  showTuner,
  onToggleTuner,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  user,
  authenticated,
  onOpenSettings,
  onLogin,
  onLogout,
}: PluginHeaderProps) {
  // The wordmark and its type flip to black when the background behind them
  // is light enough that white would wash out.
  const onLight = isLightColor(useBackgroundColors().start);

  return (
    <div
      style={{
        width: '100%',
        height: '64rem',
        flexShrink: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        // Transparent so the window gradient reaches the top-left corner,
        // where the first colour stop is anchored. The bottom border still
        // separates the bar from the content.
        backgroundColor: 'transparent',
        padding: '0 24rem',
        boxSizing: 'border-box',
        borderBottom: BORDER,
      }}
    >
      {/* NAMTRIX Player wordmark. Sized by height, not width: the art is a
        wide 2.5:1 marquee and the header is a fixed 64rem, so pinning the
        height is what keeps it inside the bar whatever the art's dimensions.
        No longer a link -- it used to open tone3000.com, which is not where
        this build comes from. */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10rem' }}>
        {/* Two cuts of the same wordmark. The header sits at the top-left,
          which is where the gradient's first stop is, so that colour decides
          which one keeps its contrast. */}
        <img
          src={onLight ? '/namtrix-logo-dark.png' : '/namtrix-logo.png'}
          alt="NAMTRIX Player"
          style={{ height: '30rem', width: 'auto', display: 'block' }}
        />
        {/* Product name beside the wordmark: the logo art is the NAMTRIX
          marquee alone, so what this build *is* gets said in type. Stacked
          over the build version, which is bumped on every edit so a stale
          WebView bundle is immediately obvious. */}
        <span
          style={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            lineHeight: 1.15,
            whiteSpace: 'nowrap',
          }}
        >
          <span
            style={{
              fontFamily: FONT_MONO,
              fontSize: '13rem',
              fontWeight: 400,
              color: onLight ? '#000000' : WHITE,
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
            }}
          >
            Parametric Player
          </span>
          <span
            style={{
              fontFamily: FONT_MONO,
              fontSize: '10rem',
              fontWeight: 400,
              color: onLight ? 'rgba(0, 0, 0, 0.55)' : MUTED,
            }}
          >
            {NAMTRIX_VERSION}
          </span>
        </span>
      </div>
      {/* 40px between header items; tight pairs (undo/redo) group inside. */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '40rem' }}>
        <PresetBar
          active={activePreset}
          presets={presetStore.presets}
          atDefault={atDefault}
          onSave={presetStore.actions.save}
          onLoad={presetStore.actions.load}
          onRename={presetStore.actions.rename}
          onDelete={presetStore.actions.remove}
          onMove={presetStore.actions.move}
          onReset={onReset}
        />
        <StereoModeToggle stereoEnabled={stereoEnabled} onToggle={onStereoToggle} />
        <IconButton
          onClick={() => onToggleTuner(!showTuner)}
          help={HELP.tuner}
          active={showTuner}
          fillWhenActive
          size={28}
        >
          <TuningForkIcon size={18} />
        </IconButton>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16rem' }}>
          <IconButton onClick={onUndo} disabled={!canUndo} help={HELP.undo} size={28}>
            <Undo2 size={18} />
          </IconButton>
          <IconButton onClick={onRedo} disabled={!canRedo} help={HELP.redo} size={28}>
            <Redo2 size={18} />
          </IconButton>
        </div>
        <AccountMenu
          user={user}
          authenticated={authenticated}
          onOpenSettings={onOpenSettings}
          onLogin={onLogin}
          onLogout={onLogout}
        />
      </div>
    </div>
  );
});
