#include "NamParamKnobs.h"

#include <cmath>

#include "core/Theme.h"

namespace t3k::ui {
namespace {

// Longest label that fits a cell before neighbouring names start colliding.
constexpr int kLabelMaxChars = 11;

// Fit a model's knob name into a cell.
//
// Amp knob names carry qualifiers a real faceplate has room for and this cell
// does not -- "Volume I (High Treb)" is a real example from a Plexi capture.
// Drop the parenthetical first, since it disambiguates rather than names, and
// only then truncate.
juce::String shortLabel(const juce::String& name) {
  juce::String trimmed = name;
  const int open = name.lastIndexOfChar('(');
  if (open > 0 && name.trimEnd().endsWithChar(')'))
    trimmed = name.substring(0, open).trim();
  if (trimmed.isEmpty())
    trimmed = name;
  return trimmed.length() > kLabelMaxChars
             ? trimmed.substring(0, kLabelMaxChars - 1) + juce::String::charToString(0x2026)
             : trimmed;
}

// Decimals for a knob's readout, from how wide its range is. A 0..10 gain
// wants one; a 0..1 blend would read as a flat "0" / "1" without two.
int decimalsFor(float min, float max) {
  const float span = std::abs(max - min);
  if (span >= 20.0f)
    return 0;
  return span >= 2.0f ? 1 : 2;
}

}  // namespace

NamParamKnobs::NamParamKnobs(ChainStore& chain, std::string blockId)
    : chain_(chain), blockId_(std::move(blockId)) {}

void NamParamKnobs::setModel(const std::vector<NamParamSpec>& specs,
                             const std::vector<float>& values) {
  // Compare by what the UI is built from: a different set of controls needs
  // new knobs, the same set only needs new values. Rebuilding on every resync
  // would destroy the knob under the user's pointer mid-drag.
  bool sameModel = specs.size() == specs_.size();
  for (size_t i = 0; sameModel && i < specs.size(); ++i)
    sameModel = specs[i].name == specs_[i].name && specs[i].min == specs_[i].min &&
                specs[i].max == specs_[i].max && specs[i].type == specs_[i].type;

  if (!sameModel) {
    rebuild(specs, values);
    return;
  }
  pushValues(values);
}

void NamParamKnobs::rebuild(const std::vector<NamParamSpec>& specs,
                            const std::vector<float>& values) {
  knobs_.clear();
  specs_ = specs;

  for (size_t i = 0; i < specs_.size(); ++i) {
    const auto& spec = specs_[i];
    auto entry = std::make_unique<Entry>();
    entry->spec = spec;

    const float span = spec.max - spec.min;
    // A degenerate range would divide by zero in the normalisation; the knob
    // is meaningless anyway, so pin it rather than special-case every use.
    const float safeSpan = span == 0.0f ? 1.0f : span;

    Knob::Options options;
    options.label = shortLabel(spec.name);
    options.size = theme::kKnobSizeSecondary;
    options.thumb = Knob::Thumb::secondary;
    options.defaultValue = (spec.defaultValue - spec.min) / safeSpan;

    if (spec.isSwitch() && spec.enumNames.size() > 1) {
      // A switch is a discrete choice, so the knob is detented and its readout
      // shows the option's name. Native rejects a fractional switch index
      // outright, so the detents are correctness, not just feel.
      const auto names = spec.enumNames;
      const double last = static_cast<double>(names.size() - 1);
      auto scale = std::make_unique<KnobScale>();
      scale->toDisplay = [last](double n) { return n * last; };
      scale->fromDisplay = [last](double d) { return last > 0 ? d / last : 0.0; };
      scale->format = [names, last](double n) {
        const int index = juce::jlimit(0, names.size() - 1, static_cast<int>(std::lround(n * last)));
        return names[index];
      };
      scale->editText = [last](double n) { return juce::String(std::lround(n * last)); };
      options.steps = names.size();
      options.scale = scale.get();
      entry->scale = std::move(scale);
    } else {
      auto scale = std::make_unique<KnobScale>(
          scales::linear(spec.min, spec.max, {}, decimalsFor(spec.min, spec.max)));
      options.scale = scale.get();
      entry->scale = std::move(scale);
    }

    entry->knob = std::make_unique<Knob>(options);

    const int index = static_cast<int>(i);
    const float min = spec.min;
    Entry* raw = entry.get();
    entry->knob->onChange = [this, index, min, safeSpan](float normalised) {
      chain_.setBlockNamParam(blockId_, index, min + normalised * safeSpan);
    };
    // Hold the value locally while dragging so a chain resync landing
    // mid-gesture cannot snap the knob back to its pre-drag position.
    entry->knob->onDragStateChange = [raw](bool dragging) { raw->dragging = dragging; };

    addAndMakeVisible(*entry->knob);
    knobs_.push_back(std::move(entry));
  }

  pushValues(values);
  resized();
}

void NamParamKnobs::pushValues(const std::vector<float>& values) {
  for (size_t i = 0; i < knobs_.size(); ++i) {
    auto& entry = *knobs_[i];
    if (entry.dragging)
      continue;
    const float value = i < values.size() ? values[i] : entry.spec.defaultValue;
    const float span = entry.spec.max - entry.spec.min;
    entry.knob->setValue(span == 0.0f ? 0.0f : (value - entry.spec.min) / span);
  }
}

int NamParamKnobs::columns(int width) const {
  return juce::jmax(1, width / kCellWidth);
}

int NamParamKnobs::preferredHeight(int width) const {
  if (knobs_.empty())
    return 0;
  const int perRow = columns(width);
  const int rows = (static_cast<int>(knobs_.size()) + perRow - 1) / perRow;
  const int rowHeight = Knob::heightFor(theme::kKnobSizeSecondary);
  return rows * rowHeight + (rows - 1) * kRowGap;
}

void NamParamKnobs::resized() {
  if (knobs_.empty())
    return;
  const int perRow = columns(getWidth());
  const int rowHeight = Knob::heightFor(theme::kKnobSizeSecondary);
  // Centre each row's cells in the available width so a short last row sits
  // under the middle of the one above rather than hugging the left edge.
  const int count = static_cast<int>(knobs_.size());
  for (int i = 0; i < count; ++i) {
    const int row = i / perRow;
    const int column = i % perRow;
    const int inThisRow = juce::jmin(perRow, count - row * perRow);
    const int rowWidth = inThisRow * kCellWidth;
    const int x = (getWidth() - rowWidth) / 2 + column * kCellWidth;
    const int y = row * (rowHeight + kRowGap);
    auto& knob = *knobs_[static_cast<size_t>(i)]->knob;
    knob.setBounds(x + (kCellWidth - knob.knobSize()) / 2, y, knob.knobSize(), rowHeight);
  }
}

}  // namespace t3k::ui
