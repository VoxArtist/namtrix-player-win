#include "NamParamKnobs.h"

#include <cmath>

#include "core/Fonts.h"
#include "core/Theme.h"

namespace t3k::ui {
namespace {

// Air kept on each side of a label, so neighbours never touch even when both
// fill their cells.
constexpr int kLabelInset = 3;

// Standard amp-faceplate abbreviations, tried before any truncation: a reader
// recognises "MASTER VOL" instantly, where "MASTER V…" makes them guess.
juce::String abbreviateWord(const juce::String& token) {
  static const std::pair<const char*, const char*> kWords[] = {
      {"VOLUME", "VOL"},  {"BALANCE", "BAL"}, {"PRESENCE", "PRES"}, {"MIDDLE", "MID"},
      {"TREBLE", "TREB"}, {"RESONANCE", "RES"}, {"MASTER", "MSTR"}, {"CHANNEL", "CH"},
  };
  for (const auto& [word, shortForm] : kWords) {
    if (!token.equalsIgnoreCase(word))
      continue;
    const juce::String upper(shortForm);
    // Keep the original's casing: "VOLUME" -> "VOL", "Volume" -> "Vol".
    return token == token.toUpperCase() ? upper
                                        : upper.substring(0, 1) + upper.substring(1).toLowerCase();
  }
  return token;
}

struct Fitted {
  juce::String text;
  bool truncated = false;
};

// The most complete version of a knob name that fits `room` pixels at `px`,
// measured in the font the Knob draws its label with.
//
// Order: the name as written minus a trailing parenthetical (it disambiguates
// rather than names: "Volume I (High Treb)"); then words abbreviated one at a
// time from the right, since the leading word usually carries the identity
// ("LEAD VOLUME" -> "LEAD VOL", never "LD VOLUME"); then truncation marked
// with an ellipsis. Measured, not counted: "MIDDLE" and "MASTER" have the same
// length but not the same width.
Fitted fitLabel(const juce::String& name, int room, float px) {
  const auto font = Fonts::sans(px);
  const auto fits = [&](const juce::String& text) {
    return juce::GlyphArrangement::getStringWidthInt(font, text) <= room;
  };

  juce::String base = name.trim();
  const int open = base.lastIndexOfChar('(');
  if (open > 0 && base.endsWithChar(')'))
    base = base.substring(0, open).trim();
  if (base.isEmpty())
    base = name.trim();
  if (fits(base))
    return {base};

  juce::StringArray tokens;
  tokens.addTokens(base, " ", "");
  for (int i = tokens.size() - 1; i >= 0; --i) {
    tokens.set(i, abbreviateWord(tokens[i]));
    const auto candidate = tokens.joinIntoString(" ");
    if (fits(candidate))
      return {candidate};
  }

  const juce::String ellipsis = juce::String::charToString(0x2026);
  juce::String cut = tokens.joinIntoString(" ");
  while (cut.length() > 1 && !fits(cut + ellipsis))
    cut = cut.dropLastCharacters(1).trimEnd();
  return {cut + ellipsis, true};
}

// One label size for the whole row: the largest at which every name fits
// without truncation. Truncation is the failure worth avoiding -- on a
// Savage, "LEAD VOLUME" cut to "LEAD..." sits beside the LEAD knob and the two
// read as the same control. Mixed sizes within a row would look accidental,
// so the row shrinks together, down to a floor still legible at a glance.
float rowLabelPx(const std::vector<NamParamSpec>& specs, int room) {
  static constexpr float kSizes[] = {14.0f, 13.0f, 12.0f, 11.0f};
  for (const float px : kSizes) {
    bool allFit = true;
    for (const auto& spec : specs)
      if (fitLabel(spec.name, room, px).truncated) {
        allFit = false;
        break;
      }
    if (allFit)
      return px;
  }
  return kSizes[std::size(kSizes) - 1];
}

// Decimals for a knob's readout, from how wide its range is. A 0..10 gain
// wants one; a 0..1 blend would read as a flat "0" / "1" without two.
int decimalsFor(float min, float max) {
  const float span = std::abs(max - min);
  if (span >= 20.0f)
    return 0;
  return span >= 2.0f ? 1 : 2;
}

bool sameControls(const std::vector<NamParamSpec>& a, const std::vector<NamParamSpec>& b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].name != b[i].name || a[i].min != b[i].min || a[i].max != b[i].max ||
        a[i].type != b[i].type || a[i].enumNames != b[i].enumNames)
      return false;
  return true;
}

}  // namespace

NamParamKnobs::NamParamKnobs(ChainStore& chain, std::string blockId)
    : chain_(chain), blockId_(std::move(blockId)) {}

void NamParamKnobs::setModel(const std::vector<NamParamSpec>& specs,
                             const std::vector<float>& values) {
  values_ = values;
  if (sameControls(specs, specs_)) {
    pushValues();
    return;
  }
  specs_ = specs;
  // Width unknown until the owner lays us out; build at the widest cell and
  // let resized() refit if the real width calls for narrower ones.
  build(builtCellWidth_ > 0 ? builtCellWidth_ : kCellWidth);
  resized();
}

int NamParamKnobs::cellWidthFor(int width) const {
  const int count = static_cast<int>(specs_.size());
  if (count == 0 || width <= 0)
    return kCellWidth;
  return juce::jlimit(kMinCellWidth, kCellWidth, width / count);
}

int NamParamKnobs::perRowFor(int width) const {
  return juce::jmax(1, width / cellWidthFor(width));
}

int NamParamKnobs::preferredHeight(int width) const {
  const int count = static_cast<int>(specs_.size());
  if (count == 0)
    return 0;
  const int perRow = perRowFor(width);
  const int rows = (count + perRow - 1) / perRow;
  return rows * Knob::heightFor(theme::kKnobSizeSecondary) + (rows - 1) * kRowGap;
}

void NamParamKnobs::build(int cellWidth) {
  entries_.clear();
  builtCellWidth_ = cellWidth;
  const int labelRoom = cellWidth - 2 * kLabelInset;
  const float labelPx = rowLabelPx(specs_, labelRoom);

  for (size_t i = 0; i < specs_.size(); ++i) {
    const auto& spec = specs_[i];
    auto entry = std::make_unique<Entry>();
    entry->spec = spec;

    // A degenerate range would divide by zero in the normalisation; the knob
    // is meaningless anyway, so pin it rather than special-case every use.
    const float span = spec.max - spec.min;
    const float safeSpan = span == 0.0f ? 1.0f : span;

    Knob::Options options;
    options.label = fitLabel(spec.name, labelRoom, labelPx).text;
    options.labelPx = labelPx;
    options.size = theme::kKnobSizeSecondary;
    options.thumb = Knob::Thumb::secondary;
    options.defaultValue = (spec.defaultValue - spec.min) / safeSpan;

    if (spec.isSwitch() && spec.enumNames.size() > 1) {
      // A switch is a discrete choice: detented, with the option's name as the
      // readout. The detents are correctness, not feel -- native rejects a
      // fractional switch index outright.
      const auto names = spec.enumNames;
      const double last = static_cast<double>(names.size() - 1);
      auto scale = std::make_unique<KnobScale>();
      scale->toDisplay = [last](double n) { return n * last; };
      scale->fromDisplay = [last](double d) { return last > 0 ? d / last : 0.0; };
      scale->format = [names, last](double n) {
        return names[juce::jlimit(0, names.size() - 1, static_cast<int>(std::lround(n * last)))];
      };
      scale->editText = [last](double n) { return juce::String(std::lround(n * last)); };
      options.steps = names.size();
      options.scale = scale.get();
      entry->scale = std::move(scale);
    } else {
      entry->scale = std::make_unique<KnobScale>(
          scales::linear(spec.min, spec.max, {}, decimalsFor(spec.min, spec.max)));
      options.scale = entry->scale.get();
    }

    entry->knob = std::make_unique<Knob>(options);
    const int index = static_cast<int>(i);
    const float min = spec.min;
    Entry* raw = entry.get();
    entry->knob->onChange = [this, index, min, safeSpan](float normalised) {
      const float value = min + normalised * safeSpan;
      if (static_cast<size_t>(index) < values_.size())
        values_[static_cast<size_t>(index)] = value;
      chain_.setBlockNamParam(blockId_, index, value);
    };
    // Hold the value locally while dragging so a resync landing mid-gesture
    // cannot snap the knob back to its pre-drag position.
    entry->knob->onDragStateChange = [raw](bool dragging) { raw->dragging = dragging; };

    addAndMakeVisible(*entry->knob);
    entries_.push_back(std::move(entry));
  }
  pushValues();
}

void NamParamKnobs::pushValues() {
  for (size_t i = 0; i < entries_.size(); ++i) {
    auto& entry = *entries_[i];
    if (entry.dragging)
      continue;
    const float value = i < values_.size() ? values_[i] : entry.spec.defaultValue;
    const float span = entry.spec.max - entry.spec.min;
    entry.knob->setValue(span == 0.0f ? 0.0f : (value - entry.spec.min) / span);
  }
}

bool NamParamKnobs::anyDragging() const {
  for (const auto& entry : entries_)
    if (entry->dragging)
      return true;
  return false;
}

void NamParamKnobs::resized() {
  if (specs_.empty())
    return;

  // Labels were fitted to the width the knobs were built for. If the real
  // width calls for different cells, rebuild so the labels match -- but never
  // mid-drag, which would destroy the knob under the pointer. The card has a
  // fixed width, so in practice this happens once, on first layout.
  const int cellWidth = cellWidthFor(getWidth());
  if (cellWidth != builtCellWidth_ && !anyDragging())
    build(cellWidth);

  const int perRow = perRowFor(getWidth());
  const int rowHeight = Knob::heightFor(theme::kKnobSizeSecondary);
  const int count = static_cast<int>(entries_.size());
  for (int i = 0; i < count; ++i) {
    const int row = i / perRow;
    const int column = i % perRow;
    // Centre each row, so a short last row sits under the middle of the one
    // above rather than hugging the left edge.
    const int inThisRow = juce::jmin(perRow, count - row * perRow);
    const int x = (getWidth() - inThisRow * builtCellWidth_) / 2 + column * builtCellWidth_;
    const int y = row * (rowHeight + kRowGap);
    // The whole cell, not just the 36px face: Knob centres its face in
    // whatever width it is given and draws the label across all of it. Sized
    // to the face alone, any label wider than 36px would be clipped to it.
    entries_[static_cast<size_t>(i)]->knob->setBounds(x, y, builtCellWidth_, rowHeight);
  }
}

}  // namespace t3k::ui
