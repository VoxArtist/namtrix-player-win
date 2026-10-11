// NAMTRIX Player's own branding, kept out of Brand.h so upstream's artwork
// and ours never collide in a merge.
#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace t3k::ui {

struct Namtrix {
  // Build version, shown beside the wordmark. Bumped on every edit to this
  // fork so it is always clear, at a glance, whether the running plugin is
  // the build you just made.
  static constexpr const char* kVersion = "v0.18";

  // The wordmark, white on transparency, and its inverted cut for light
  // backgrounds.
  static const juce::Image& wordmark();
  static const juce::Image& wordmarkDark();
  // Brushed-metal plate behind the whole window.
  static const juce::Image& backgroundPlate();

  // Fill `area` with the plate, scaled to cover and centred. Falls back to a
  // flat fill if the image is missing, so a decode failure can't leave the
  // window transparent.
  static void drawBackground(juce::Graphics& g, juce::Rectangle<int> area, juce::Colour fallback);
};

}  // namespace t3k::ui
