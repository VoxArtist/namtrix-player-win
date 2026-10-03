// The knob row for a parametric NAM model: one knob per control the model
// declares for itself (ConcatWaveNet / HyperWaveNet / ConcatLSTM). Snapshot
// models declare none and the row builds empty, so owners can hold one
// unconditionally and ask it for its height.
//
// Values are in each knob's own units -- the numbers printed on the real
// faceplate -- while Knob works in normalised 0..1, so every knob carries a
// linear scale over its spec range and converts on the way out.
#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include <memory>
#include <string>
#include <vector>

#include "core/KnobScale.h"
#include "model/ChainState.h"
#include "services/ChainStore.h"
#include "widgets/Knob.h"

namespace t3k::ui {

class NamParamKnobs : public juce::Component {
public:
  // Cell width bounds. Cells narrow to keep every knob on one row, because the
  // card is pinned at its Figma height and a second row would crowd out the
  // tone's title. Below the floor a label has no usable room, so the row
  // wraps instead.
  static constexpr int kCellWidth = 76;
  static constexpr int kMinCellWidth = 56;
  static constexpr int kRowGap = 4;

  NamParamKnobs(ChainStore& chain, std::string blockId);

  // Rebuild for a new model, or follow a chain resync for the same one.
  // Knobs are rebuilt only when the set of controls changes; a resync for the
  // same model just pushes values, so it cannot fight a drag.
  void setModel(const std::vector<NamParamSpec>& specs, const std::vector<float>& values);

  bool isEmpty() const { return specs_.empty(); }
  int preferredHeight(int width) const;

  void resized() override;

private:
  struct Entry {
    NamParamSpec spec;
    std::unique_ptr<Knob> knob;
    // Knob::Options holds a bare pointer to its scale, so the scale has to
    // outlive the knob.
    std::unique_ptr<KnobScale> scale;
    bool dragging = false;
  };

  // Labels are fitted to a cell width, and a Knob's label is fixed at
  // construction, so the knobs are (re)built for a specific width.
  void build(int cellWidth);
  void pushValues();
  bool anyDragging() const;
  int cellWidthFor(int width) const;
  int perRowFor(int width) const;

  ChainStore& chain_;
  std::string blockId_;
  std::vector<NamParamSpec> specs_;
  std::vector<float> values_;
  std::vector<std::unique_ptr<Entry>> entries_;
  int builtCellWidth_ = 0;
};

}  // namespace t3k::ui
