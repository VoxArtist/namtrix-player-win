#include "Namtrix.h"

#include "UiBinaryData.h"

namespace t3k::ui {
namespace {

// Decoded once and kept: these are drawn on every repaint, and JUCE's PNG/JPEG
// decoders are far too slow to run per frame.
const juce::Image& decode(const char* data, int size) {
  static std::map<const char*, juce::Image> cache;
  auto it = cache.find(data);
  if (it == cache.end())
    it = cache.emplace(data, juce::ImageFileFormat::loadFrom(data, static_cast<size_t>(size))).first;
  return it->second;
}

}  // namespace

const juce::Image& Namtrix::wordmark() {
  return decode(UiBinaryData::namtrixlogo_png, UiBinaryData::namtrixlogo_pngSize);
}

const juce::Image& Namtrix::wordmarkDark() {
  return decode(UiBinaryData::namtrixlogodark_png, UiBinaryData::namtrixlogodark_pngSize);
}

const juce::Image& Namtrix::backgroundPlate() {
  return decode(UiBinaryData::bgmetal_jpg, UiBinaryData::bgmetal_jpgSize);
}

void Namtrix::drawBackground(juce::Graphics& g, juce::Rectangle<int> area, juce::Colour fallback) {
  const auto& plate = backgroundPlate();
  if (!plate.isValid() || area.isEmpty()) {
    g.fillAll(fallback);
    return;
  }

  // Fill behind the plate as well: the plate is scaled to cover, so it can
  // never leave a gap, but a decode that half-succeeded shouldn't show through
  // to whatever was on screen before.
  g.fillAll(fallback);

  // Cover, not stretch: the plate is a photograph of brushed metal and
  // distorting it would bend the grain. The larger of the two ratios fills
  // both axes and crops the overflow.
  const float scale = juce::jmax(static_cast<float>(area.getWidth()) / plate.getWidth(),
                                 static_cast<float>(area.getHeight()) / plate.getHeight());
  const int w = juce::roundToInt(plate.getWidth() * scale);
  const int h = juce::roundToInt(plate.getHeight() * scale);
  g.drawImage(plate, area.getX() + (area.getWidth() - w) / 2, area.getY() + (area.getHeight() - h) / 2,
              w, h, 0, 0, plate.getWidth(), plate.getHeight());
}

}  // namespace t3k::ui
