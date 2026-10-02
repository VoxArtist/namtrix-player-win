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
  // Wide enough for a secondary knob plus the breathing room its label needs:
  // labels are centred and single-line, so without a fixed cell the names of
  // adjacent knobs collide.
  static constexpr int kCellWidth = 76;
  static constexpr int kRowGap = 4;

  NamParamKnobs(ChainStore& chain, std::string blockId);

  // Rebuild for a new model, or follow a chain resync for the same one.
  // Knobs are rebuilt only when the specs actually change; a resync that
  // carries the same model just pushes values, so it cannot fight a drag.
  void setModel(const std::vector<NamParamSpec>& specs, const std::vector<float>& values);

  bool isEmpty() const { return knobs_.empty(); }
  // Height needed for `count` knobs at the given width, so the owner can size
  // the row before it is populated.
  int preferredHeight(int width) const;

  void resized() override;

private:
  struct Entry {
    NamParamSpec spec;
    std::unique_ptr<Knob> knob;
    // Owned here: Knob::Options holds a bare pointer to its scale, so the
    // scale has to outlive the knob.
    std::unique_ptr<KnobScale> scale;
    bool dragging = false;
  };

  void rebuild(const std::vector<NamParamSpec>& specs, const std::vector<float>& values);
  void pushValues(const std::vector<float>& values);
  int columns(int width) const;

  ChainStore& chain_;
  std::string blockId_;
  std::vector<NamParamSpec> specs_;
  std::vector<std::unique_ptr<Entry>> knobs_;
};

}  // namespace t3k::ui
