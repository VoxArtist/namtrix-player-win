#include "PluginHeader.h"

#include "core/Namtrix.h"
#include "core/Brand.h"
#include "core/CustomIcons.h"
#include "core/Design.h"
#include "core/Help.h"
#include "core/Paint.h"
#include "core/Theme.h"
#include "widgets/Clickable.h"

namespace t3k::ui {

namespace {
constexpr int kPadX = 24;
constexpr int kGroupGap = 40;  // between header items
constexpr int kPairGap = 16;   // tight pairs (undo/redo)
constexpr int kLogoWidth = 160;
constexpr int kLogoHeight = 24;  // 160 * 32 / 210, rounded like the browser
}  // namespace

// The NAMTRIX wordmark, with the product name and build version set beside
// it. Not a link: the stock header opened tone3000.com, which is not where
// this build comes from.
class PluginHeader::LogoLink : public Clickable {
public:
  LogoLink() : Clickable({}) {
    setTitle("NAMTRIX Parametric Player");
    setInterceptsMouseClicks(false, false);
  }
  void paintButton(juce::Graphics& g, bool, bool) override {
    const auto bounds = getLocalBounds();
    const auto& art = Namtrix::wordmark();
    int textX = bounds.getX();
    if (art.isValid()) {
      // Sized by height so the art's aspect decides its width; the type then
      // starts wherever the wordmark actually ends.
      const int h = juce::jmin(bounds.getHeight(), kWordmarkHeight);
      const int w = juce::roundToInt(h * art.getWidth() / static_cast<float>(art.getHeight()));
      const int y = bounds.getY() + (bounds.getHeight() - h) / 2;
      g.drawImage(art, bounds.getX(), y, w, h, 0, 0, art.getWidth(), art.getHeight());
      textX = bounds.getX() + w + kWordmarkGap;
    }

    // Product name over build version, both left-aligned on the wordmark's
    // trailing edge and centred as a pair against it.
    const int textW = bounds.getRight() - textX;
    if (textW <= 0)
      return;
    const int nameH = 15, versionH = 12;
    const int top = bounds.getY() + (bounds.getHeight() - (nameH + versionH)) / 2;
    g.setColour(theme::kWhite);
    g.setFont(Fonts::mono(11.0f));
    g.drawText("PARAMETRIC PLAYER", textX, top, textW, nameH, juce::Justification::centredLeft);
    g.setColour(theme::kMuted);
    g.setFont(Fonts::mono(10.0f));
    g.drawText(Namtrix::kVersion, textX, top + nameH, textW, versionH,
               juce::Justification::centredLeft);
  }

private:
  static constexpr int kWordmarkHeight = 22;
  static constexpr int kWordmarkGap = 10;
};

PluginHeader::PluginHeader(Services& services)
    : services_(services),
      logo_(std::make_unique<LogoLink>()),
      presetBar_(services),
      tuner_(custom_icons::kTuningFork, 28, 18) {
  addAndMakeVisible(*logo_);
  addAndMakeVisible(presetBar_);

  stereo_.onToggle = [this](bool stereo) {
    if (onStereoToggle) onStereoToggle(stereo);
  };
  addAndMakeVisible(stereo_);

  tuner_.setHelpText(help::text(help::Key::tuner));
  tuner_.setFillWhenActive(true);
  tuner_.onClick = [this] {
    if (onToggleTuner) onToggleTuner(!tunerShown_);
  };
  addAndMakeVisible(tuner_);

  undo_.setHelpText(help::text(help::Key::undo));
  undo_.onClick = [this] {
    if (onUndo) onUndo();
  };
  redo_.setHelpText(help::text(help::Key::redo));
  redo_.onClick = [this] {
    if (onRedo) onRedo();
  };
  addAndMakeVisible(undo_);
  addAndMakeVisible(redo_);

  account_.onOpenSettings = [this] {
    if (onOpenSettings) onOpenSettings();
  };
  account_.onLogin = [this] {
    if (onLogin) onLogin();
  };
  account_.onLogout = [this] {
    if (onLogout) onLogout();
  };
  addAndMakeVisible(account_);

  services_.chain.addListener(this);
  services_.session.addListener(this);
  sessionChanged();
  chainChanged(services_.chain.state());
  setTunerShown(false);
  setSize(design::kWidth, kHeight);
}

PluginHeader::~PluginHeader() {
  services_.session.removeListener(this);
  services_.chain.removeListener(this);
}

void PluginHeader::sessionChanged() {
  const auto& session = services_.session;
  account_.setAuthenticated(session.authenticated());
  const auto user = session.user();
  const auto url = user ? user->avatarUrl : juce::String();
  if (url == avatarUrl_ && (url.isNotEmpty() || !session.authenticated())) return;
  avatarUrl_ = url;
  account_.setAvatar(services_.images, url);
}

void PluginHeader::setTunerShown(bool shown) {
  tunerShown_ = shown;
  // Lit + HIGHLIGHT fill while the tuner is up; the icon itself stays white.
  tuner_.setActive(true);
  tuner_.setFillWhenActive(shown);
}

void PluginHeader::chainChanged(const ChainState& state) {
  undo_.setEnabled(state.canUndo);
  redo_.setEnabled(state.canRedo);
  stereo_.setStereoEnabled(state.stereoEnabled);
}

void PluginHeader::paint(juce::Graphics& g) {
  g.fillAll(theme::kBlack);
  paint::hairlineH(g, 0, static_cast<float>(getWidth()), static_cast<float>(getHeight() - 1),
                   theme::kBorder);
}

void PluginHeader::resized() {
  // The 1px bottom border is inside the 64px box; items centre in the 63px
  // content row like the flex container did.
  auto area = getLocalBounds().withTrimmedBottom(1).reduced(kPadX, 0);
  // 31.5 in the browser; rounding up puts even-height boxes where the
  // browser's anti-aliased edges land.
  const int cy = (area.getHeight() + 1) / 2;
  auto place = [&](juce::Component& c, int right) {
    c.setBounds(right - c.getWidth(), cy - c.getHeight() / 2, c.getWidth(), c.getHeight());
    return c.getX();
  };

  logo_->setBounds(area.getX(), cy - kLogoHeight / 2, kLogoWidth, kLogoHeight);

  // Right group, laid out from the right edge: account · undo/redo · tuner ·
  // stereo · presets, 40px apart (16px inside the undo/redo pair).
  int x = place(account_, area.getRight()) - kGroupGap;
  x = place(redo_, x) - kPairGap;
  x = place(undo_, x) - kGroupGap;
  x = place(tuner_, x) - kGroupGap;
  x = place(stereo_, x) - kGroupGap;
  place(presetBar_, x);
}

}  // namespace t3k::ui
