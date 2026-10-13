// Local file loading (loadLocalTone for drops, loadLocalTonePath for the
// tile menus' file pickers): load-time validation (NAM files must be A2),
// and that accepted files ride the normal load pipeline as `local` blocks:
// background load from the content-addressed stash (no network), IR mix
// defaults by kernel length, folder loads as one multi-model tone whose
// full model list survives switches.
//
// Note: both entry points write their stash into the real app-data
// LocalModels folder. The names are content hashes, so repeated runs reuse
// the same few small files (and the week-based GC clears them eventually).

#include "chain_test_helpers.h"
#include "NAM/get_dsp.h"
#include "json.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <functional>

#if !JUCE_WINDOWS
#include <sys/stat.h>  // chmod, for the unwritable-directory heal test
#include <unistd.h>    // geteuid: root ignores permission bits, so that leg skips
#endif

namespace {

// fileEntry / testFileEntry / filesOf (the { name, data } entries the UI
// ships) live in chain_test_helpers.h.

// First tone block of the (mono) chain, or void when none.
juce::var firstToneBlock(TONE3000Processor& proc) {
  const juce::var state = proc.getChainState(-1);
  if (const auto* lane = state["chain"].getArray())
    for (const auto& item : *lane)
      if (item["kind"].toString() == "tone")
        return item;
  return {};
}

}  // namespace

TEST(LocalLoadTest, A2NamFileLoadsAsLocalBlock) {
  TONE3000Processor proc;
  const juce::var res =
      proc.loadLocalTone("a2-amp-test", filesOf({testFileEntry("a2-amp-test.nam")}));
  EXPECT_TRUE(res["error"].isVoid()) << res["error"].toString().toStdString();
  ASSERT_TRUE(res["blockId"].toString().isNotEmpty());
  ASSERT_TRUE(waitForChainLoaded(proc));

  const juce::var block = firstToneBlock(proc);
  EXPECT_EQ(block["blockId"].toString(), res["blockId"].toString());
  EXPECT_TRUE(static_cast<bool>(block["tone"]["local"]));
  EXPECT_EQ(block["tone"]["format"].toString(), juce::String("nam"));
  EXPECT_EQ(block["tone"]["title"].toString(), juce::String("a2-amp-test"));
  // Model names drop the extension too.
  ASSERT_EQ(block["tone"]["models"].size(), 1);
  EXPECT_EQ(block["tone"]["models"][0]["name"].toString(), juce::String("a2-amp-test"));
}

TEST(LocalLoadTest, FolderLoadsAsOneTonePerFileModelsSurviveSwitch) {
  TONE3000Processor proc;
  const juce::var res = proc.loadLocalTone(
      "My Pack", filesOf({testFileEntry("a2-amp-test.nam"), testFileEntry("a2-amp-cab-test.nam")}));
  EXPECT_TRUE(res["error"].isVoid()) << res["error"].toString().toStdString();
  ASSERT_TRUE(waitForChainLoaded(proc));

  juce::var block = firstToneBlock(proc);
  EXPECT_EQ(block["tone"]["title"].toString(), juce::String("My Pack"));
  ASSERT_EQ(block["tone"]["models"].size(), 2);
  // Local summary models carry their stash URL: it's what the picker's
  // switch call sends back down.
  const juce::var second = block["tone"]["models"][1];
  ASSERT_TRUE(second["model_url"].toString().startsWith("file://"));

  ASSERT_TRUE(proc.switchModel(block["blockId"].toString().toStdString(),
                               static_cast<int>(second["id"]), second));
  ASSERT_TRUE(waitForChainLoaded(proc));

  // The switch keeps the whole local model list (catalog tones would prune
  // to the picked model here).
  block = firstToneBlock(proc);
  EXPECT_EQ(static_cast<int>(block["activeModelId"]), static_cast<int>(second["id"]));
  EXPECT_EQ(block["tone"]["models"].size(), 2);
}

TEST(LocalLoadTest, DropOnExistingToneBlockSwapsInPlace) {
  TONE3000Processor proc;
  const juce::var first =
      proc.loadLocalTone("a2-amp-test", filesOf({testFileEntry("a2-amp-test.nam")}));
  ASSERT_TRUE(first["blockId"].toString().isNotEmpty());
  ASSERT_TRUE(waitForChainLoaded(proc));
  const juce::String blockId = first["blockId"].toString();

  const juce::var swapped = proc.loadLocalTone(
      "cab-ir-test", filesOf({testFileEntry("cab-ir-test.wav")}), blockId.toStdString());
  EXPECT_TRUE(swapped["error"].isVoid()) << swapped["error"].toString().toStdString();
  EXPECT_EQ(swapped["blockId"].toString(), blockId);
  ASSERT_TRUE(waitForChainLoaded(proc));

  const juce::var block = firstToneBlock(proc);
  EXPECT_EQ(block["blockId"].toString(), blockId);
  EXPECT_EQ(block["tone"]["title"].toString(), juce::String("cab-ir-test"));
  EXPECT_EQ(block["tone"]["format"].toString(), juce::String("ir"));

  // Still a single tone block (didn't insert a second).
  const juce::var state = proc.getChainState(-1);
  int tones = 0;
  if (const auto* lane = state["chain"].getArray())
    for (const auto& item : *lane)
      if (item["kind"].toString() == "tone")
        ++tones;
  EXPECT_EQ(tones, 1);
}

TEST(LocalLoadTest, IrMixDefaultsFollowKernelLength) {
  // Short (cab) IR: fully wet by default.
  {
    TONE3000Processor proc;
    proc.loadLocalTone("cab-ir-test", filesOf({testFileEntry("cab-ir-test.wav")}));
    ASSERT_TRUE(waitForChainLoaded(proc));
    const juce::var block = firstToneBlock(proc);
    EXPECT_FALSE(static_cast<bool>(block["irLong"]));
    EXPECT_FLOAT_EQ(static_cast<float>(block["params"]["mix"]), 1.0f);
  }
  // Long (reverb) IR: half wet by default, same as a Select-flow load.
  {
    TONE3000Processor proc;
    proc.loadLocalTone("reverb-ir-mono-test", filesOf({testFileEntry("reverb-ir-mono-test.wav")}));
    ASSERT_TRUE(waitForChainLoaded(proc));
    const juce::var block = firstToneBlock(proc);
    EXPECT_TRUE(static_cast<bool>(block["irLong"]));
    EXPECT_FLOAT_EQ(static_cast<float>(block["params"]["mix"]), 0.5f);
  }
}

// Parametric (knob-conditioned) NAM: the drop-time gate admits the
// architecture, the loaded engine surfaces its knob specs on the block, and
// setBlockNamParam moves a knob and is reflected back in the chain state.
// Snapshot models must keep surfacing no specs at all, which is what the UI
// keys off to decide whether to draw a knob row.
TEST(LocalLoadTest, ParametricNamExposesKnobsAndAcceptsEdits) {
  TONE3000Processor proc;
  const juce::var res = proc.loadLocalTone(
      "parametric-concat-test", filesOf({testFileEntry("parametric-concat-test.nam")}));
  EXPECT_TRUE(res["error"].isVoid()) << res["error"].toString().toStdString();
  const juce::String blockId = res["blockId"].toString();
  ASSERT_TRUE(blockId.isNotEmpty());
  ASSERT_TRUE(waitForChainLoaded(proc));

  const juce::var block = firstToneBlock(proc);
  const juce::var specsVar = block["namParamSpecs"];
  const auto* specs = specsVar.getArray();
  ASSERT_NE(specs, nullptr) << "parametric model surfaced no knob specs";
  ASSERT_EQ(specs->size(), 2);

  // Spec 0: a continuous gain over its faceplate range, not a normalized 0..1.
  EXPECT_EQ((*specs)[0]["name"].toString(), juce::String("gain"));
  EXPECT_EQ((*specs)[0]["type"].toString(), juce::String("continuous"));
  EXPECT_DOUBLE_EQ(static_cast<double>((*specs)[0]["min"]), 0.0);
  EXPECT_DOUBLE_EQ(static_cast<double>((*specs)[0]["max"]), 10.0);

  // Spec 1: a switch, carrying the option names the UI labels itself with.
  EXPECT_EQ((*specs)[1]["name"].toString(), juce::String("mode"));
  EXPECT_EQ((*specs)[1]["type"].toString(), juce::String("switch"));
  const juce::var enumNamesVar = (*specs)[1]["enumNames"];
  const auto* enumNames = enumNamesVar.getArray();
  ASSERT_NE(enumNames, nullptr);
  ASSERT_EQ(enumNames->size(), 3);
  EXPECT_EQ((*enumNames)[0].toString(), juce::String("clean"));
  EXPECT_EQ((*enumNames)[2].toString(), juce::String("lead"));

  // Knob values, read through a helper: getArray() hands back a pointer into
  // the var it is called on, so the chain of temporaries in
  // firstToneBlock(proc)["params"]["namParamValues"] would dangle. Each var is
  // held in a named local instead.
  const auto knobValues = [](TONE3000Processor& p) {
    const juce::var block = firstToneBlock(p);
    const juce::var params = block["params"];
    const juce::var values = params["namParamValues"];
    std::vector<double> out;
    if (const auto* array = values.getArray())
      for (const auto& v : *array)
        out.push_back(static_cast<double>(v));
    return out;
  };

  // Values start at the model's own defaults.
  const std::vector<double> initial = knobValues(proc);
  ASSERT_EQ(initial.size(), 2u);
  EXPECT_DOUBLE_EQ(initial[0], 5.0);

  // A knob edit lands and reads back.
  EXPECT_TRUE(proc.setBlockNamParam(blockId.toStdString(), 0, 8.25));
  const std::vector<double> moved = knobValues(proc);
  ASSERT_EQ(moved.size(), 2u);
  EXPECT_NEAR(moved[0], 8.25, 1e-4);

  // Out-of-range values clamp to the spec rather than being rejected, and a
  // switch rounds to a whole index (native throws on a fractional one).
  EXPECT_TRUE(proc.setBlockNamParam(blockId.toStdString(), 0, 99.0));
  EXPECT_TRUE(proc.setBlockNamParam(blockId.toStdString(), 1, 1.6));
  const std::vector<double> clamped = knobValues(proc);
  ASSERT_EQ(clamped.size(), 2u);
  EXPECT_DOUBLE_EQ(clamped[0], 10.0);
  EXPECT_DOUBLE_EQ(clamped[1], 2.0);

  // An index past the end is refused, not clamped: it would silently move
  // the wrong knob.
  EXPECT_FALSE(proc.setBlockNamParam(blockId.toStdString(), 7, 1.0));
}

// A2's layout at a width the fast engine wasn't compiled for (it only exists
// at 3 and 8 channels). The gate admits it and it runs on the generic engine.
// The loosening is width only: a network that breaks any other A2 rule is
// still refused at drop time.
TEST(LocalLoadTest, WideA2ShapedNamLoadsOnTheGenericEngine) {
  {
    TONE3000Processor proc;
    const juce::var res =
        proc.loadLocalTone("a2-wide-16ch-test", filesOf({testFileEntry("a2-wide-16ch-test.nam")}));
    EXPECT_TRUE(res["error"].isVoid()) << res["error"].toString().toStdString();
    ASSERT_TRUE(waitForChainLoaded(proc));
    const juce::var block = firstToneBlock(proc);
    EXPECT_TRUE(static_cast<bool>(block["loaded"]));
    EXPECT_FALSE(static_cast<bool>(block["loadFailed"]));
  }

  // Same file with one dilation changed: no longer A2-shaped at any width.
  juce::var broken = juce::JSON::parse(testFile("a2-wide-16ch-test.nam"));
  auto* layer = broken["config"]["layers"][0].getDynamicObject();
  ASSERT_NE(layer, nullptr);
  juce::Array<juce::var> dilations = *layer->getProperty("dilations").getArray();
  dilations.set(0, static_cast<int>(dilations[0]) + 1);
  layer->setProperty("dilations", dilations);
  const juce::String brokenJson = juce::JSON::toString(broken);

  TONE3000Processor proc;
  const juce::var res = proc.loadLocalTone(
      "not-a2", filesOf({fileEntry("not-a2.nam", juce::Base64::toBase64(
                                                     brokenJson.toRawUTF8(),
                                                     brokenJson.getNumBytesAsUTF8()))}));
  EXPECT_EQ(res["error"].toString(),
            juce::String("Only A2 and parametric NAM files are supported"));
}

TEST(LocalLoadTest, SnapshotNamExposesNoKnobs) {
  TONE3000Processor proc;
  const juce::var res =
      proc.loadLocalTone("a2-amp-test", filesOf({testFileEntry("a2-amp-test.nam")}));
  EXPECT_TRUE(res["error"].isVoid()) << res["error"].toString().toStdString();
  ASSERT_TRUE(waitForChainLoaded(proc));

  const juce::var block = firstToneBlock(proc);
  EXPECT_TRUE(block["namParamSpecs"].isVoid());
  EXPECT_TRUE(block["params"]["namParamValues"].isVoid());
  EXPECT_FALSE(proc.setBlockNamParam(res["blockId"].toString().toStdString(), 0, 1.0));
}

// The tile's glyph: a local tone carries the catalog `gear` id when native
// could infer one from the file, and no `gear` at all otherwise (the UI then
// keeps its generic file icon). Folders take the first file's answer.
TEST(LocalLoadTest, GearIsInferredFromNamMetadataAndIrLength) {
  const auto gearOf = [](const juce::String& title, const juce::Array<juce::var>& files) {
    TONE3000Processor proc;
    const juce::var res = proc.loadLocalTone(title, filesOf(files));
    EXPECT_TRUE(res["error"].isVoid()) << res["error"].toString().toStdString();
    EXPECT_TRUE(waitForChainLoaded(proc));
    return firstToneBlock(proc)["tone"]["gear"];
  };

  // metadata.gear_type spellings map onto the catalog ids.
  EXPECT_EQ(gearOf("amp", {testFileEntry("a2-am-test-2.nam")}).toString(), juce::String("amp"));
  EXPECT_EQ(gearOf("amp cab", {testFileEntry("a2-amp-cab-test.nam")}).toString(),
            juce::String("amp-cab"));
  // No metadata: no gear (not an amp by default).
  EXPECT_TRUE(gearOf("plain", {testFileEntry("a2-amp-test.nam")}).isVoid());

  // Unknown free text stays unknown: the real A2 file with a made-up type.
  juce::var odd = juce::JSON::parse(testFile("a2-amp-test.nam"));
  juce::DynamicObject::Ptr metadata = new juce::DynamicObject();
  metadata->setProperty("gear_type", "synth");
  odd.getDynamicObject()->setProperty("metadata", juce::var(metadata.get()));
  const juce::String oddJson = juce::JSON::toString(odd);
  EXPECT_TRUE(gearOf("odd", {fileEntry("odd.nam", juce::Base64::toBase64(
                                                       oddJson.toRawUTF8(),
                                                       oddJson.getNumBytesAsUTF8()))})
                  .isVoid());

  // IRs: cab-length kernels are cabs, reverbs stay generic.
  EXPECT_EQ(gearOf("cab", {testFileEntry("cab-ir-test.wav")}).toString(), juce::String("cab"));
  EXPECT_TRUE(gearOf("verb", {testFileEntry("reverb-ir-mono-test.wav")}).isVoid());

  // Folder: the first file decides.
  EXPECT_EQ(gearOf("pack", {testFileEntry("a2-amp-cab-test.nam"), testFileEntry("a2-am-test-2.nam")})
                .toString(),
            juce::String("amp-cab"));
}

TEST(LocalLoadTest, RejectsBadFilesAndSkipsThemInFolders) {
  TONE3000Processor proc;

  // Valid JSON, wrong architecture: rejected at drop time (never reaches
  // the background loader, whose failure UI would suggest retrying).
  const juce::String lstm = R"({"version":"0.5.4","architecture":"LSTM","config":{}})";
  const juce::String lstm64 = juce::Base64::toBase64(lstm.toRawUTF8(), lstm.getNumBytesAsUTF8());
  juce::var res = proc.loadLocalTone("model", filesOf({fileEntry("model.nam", lstm64)}));
  EXPECT_EQ(res["error"].toString(),
            juce::String("Only A2 and parametric NAM files are supported"));

  res = proc.loadLocalTone("bad", filesOf({fileEntry("bad.wav", juce::Base64::toBase64("x", 1))}));
  EXPECT_EQ(res["error"].toString(), juce::String("Not a valid WAV file"));

  res = proc.loadLocalTone("notes",
                           filesOf({fileEntry("notes.txt", juce::Base64::toBase64("hi", 2))}));
  EXPECT_EQ(res["error"].toString(), juce::String("Only .nam and .wav files are supported"));

  // A rejected drop must not leave a block behind.
  EXPECT_TRUE(firstToneBlock(proc).isVoid());

  // A folder with one bad file still loads the good ones.
  res = proc.loadLocalTone(
      "Mixed Pack", filesOf({fileEntry("model.nam", lstm64), testFileEntry("a2-amp-test.nam")}));
  EXPECT_TRUE(res["error"].isVoid()) << res["error"].toString().toStdString();
  ASSERT_TRUE(waitForChainLoaded(proc));
  EXPECT_EQ(firstToneBlock(proc)["tone"]["models"].size(), 1);
}

// loadLocalTonePath: the tile menus' Load File / Load Folder pickers. Same
// pipeline as loadLocalTone but fed from disk paths (native FileChooser
// results); the folder rules (majority extension, natural order, title from
// the folder name) live natively here instead of in the web UI.

TEST(LocalLoadTest, PathLoadsSingleFileAndSwapsInPlace) {
  TONE3000Processor proc;
  const juce::var res = proc.loadLocalTonePath(testFile("a2-amp-test.nam"));
  EXPECT_TRUE(res["error"].isVoid()) << res["error"].toString().toStdString();
  ASSERT_TRUE(res["blockId"].toString().isNotEmpty());
  ASSERT_TRUE(waitForChainLoaded(proc));

  juce::var block = firstToneBlock(proc);
  EXPECT_TRUE(static_cast<bool>(block["tone"]["local"]));
  EXPECT_EQ(block["tone"]["format"].toString(), juce::String("nam"));
  EXPECT_EQ(block["tone"]["title"].toString(), juce::String("a2-amp-test"));

  // Targeting an existing tone block replaces in place, like a drop on a tile.
  const juce::String blockId = res["blockId"].toString();
  const juce::var swapped =
      proc.loadLocalTonePath(testFile("cab-ir-test.wav"), blockId.toStdString());
  EXPECT_EQ(swapped["blockId"].toString(), blockId);
  ASSERT_TRUE(waitForChainLoaded(proc));
  block = firstToneBlock(proc);
  EXPECT_EQ(block["blockId"].toString(), blockId);
  EXPECT_EQ(block["tone"]["format"].toString(), juce::String("ir"));
  EXPECT_EQ(block["tone"]["title"].toString(), juce::String("cab-ir-test"));
}

TEST(LocalLoadTest, PathLoadsFolderMajorityExtensionInNaturalOrder) {
  // Scratch folder: three distinct .nam files whose natural order differs
  // from lexicographic ("amp 10" sorts before "amp 2" there), one of them in
  // a subfolder (recursion), plus a minority .wav and a stray .txt (ignored).
  const juce::File dir =
      juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("t3k-pack-test");
  dir.deleteRecursively();
  ASSERT_TRUE(dir.getChildFile("More").createDirectory());
  ASSERT_TRUE(testFile("a2-amp-test.nam").copyFileTo(dir.getChildFile("amp 2.nam")));
  ASSERT_TRUE(testFile("a2-amp-cab-test.nam").copyFileTo(dir.getChildFile("amp 10.nam")));
  ASSERT_TRUE(
      testFile("a2-am-test-2.nam").copyFileTo(dir.getChildFile("More").getChildFile("amp 1.nam")));
  ASSERT_TRUE(testFile("cab-ir-test.wav").copyFileTo(dir.getChildFile("cab.wav")));
  ASSERT_TRUE(dir.getChildFile("notes.txt").replaceWithText("hi"));

  TONE3000Processor proc;
  const juce::var res = proc.loadLocalTonePath(dir);
  EXPECT_TRUE(res["error"].isVoid()) << res["error"].toString().toStdString();
  ASSERT_TRUE(waitForChainLoaded(proc));

  const juce::var block = firstToneBlock(proc);
  EXPECT_EQ(block["tone"]["title"].toString(), juce::String("t3k-pack-test"));
  EXPECT_EQ(block["tone"]["format"].toString(), juce::String("nam"));
  ASSERT_EQ(block["tone"]["models"].size(), 3)
      << "models: " << juce::JSON::toString(block["tone"]["models"]).toStdString();
  EXPECT_EQ(block["tone"]["models"][0]["name"].toString(), juce::String("amp 1"));
  EXPECT_EQ(block["tone"]["models"][1]["name"].toString(), juce::String("amp 2"));
  EXPECT_EQ(block["tone"]["models"][2]["name"].toString(), juce::String("amp 10"));

  dir.deleteRecursively();
}

TEST(LocalLoadTest, PathRejectsBadInputs) {
  TONE3000Processor proc;

  juce::var res = proc.loadLocalTonePath(juce::File("/nonexistent-t3k/missing.nam"));
  EXPECT_EQ(res["error"].toString(), juce::String("Couldn't read the file"));

  const juce::File dir = juce::File::getSpecialLocation(juce::File::tempDirectory)
                             .getChildFile("t3k-pack-reject-test");
  dir.deleteRecursively();
  ASSERT_TRUE(dir.createDirectory().wasOk());
  ASSERT_TRUE(dir.getChildFile("notes.txt").replaceWithText("hi"));

  res = proc.loadLocalTonePath(dir.getChildFile("notes.txt"));
  EXPECT_EQ(res["error"].toString(), juce::String("Only .nam and .wav files are supported"));

  // A folder whose only contents are unsupported types has nothing to load.
  res = proc.loadLocalTonePath(dir);
  EXPECT_EQ(res["error"].toString(), juce::String("No .nam or .wav files in the folder"));

  // Rejected loads must not leave a block behind.
  EXPECT_TRUE(firstToneBlock(proc).isVoid());
  dir.deleteRecursively();
}

// The stash path a block persists as its model_url is absolute, and the tone
// JSON carrying it lives in presets, DAW/app state and undo snapshots. On iOS
// the app data container's UUID rotates on every reinstall or app update, so
// those paths name a container that is gone. Resolution therefore re-roots the
// (content-hashed) file name under the current stash folder.
TEST(LocalLoadTest, StashUrlResolvesUnderTheCurrentRoot) {
  const juce::File tmp = juce::File::getSpecialLocation(juce::File::tempDirectory);
  const juce::File root = tmp.getChildFile("t3k-stash-" + juce::Uuid().toString());
  const juce::File live = root.getChildFile("deadbeef-4096.nam");
  ASSERT_TRUE(root.createDirectory());
  ASSERT_TRUE(live.replaceWithText("model bytes"));

  // What a preset saved under a previous container holds.
  const juce::String staleUrl =
      juce::URL(tmp.getChildFile("Containers")
                    .getChildFile(juce::Uuid().toString())
                    .getChildFile("LocalModels")
                    .getChildFile("deadbeef-4096.nam"))
          .toString(false);
  EXPECT_EQ(TONE3000Processor::resolveLocalModelFile(root, staleUrl), live);

  // A path that still resolves comes back untouched: every desktop case.
  EXPECT_EQ(TONE3000Processor::resolveLocalModelFile(root, juce::URL(live).toString(false)), live);

  // Nothing to re-root for a catalog URL.
  EXPECT_EQ(TONE3000Processor::resolveLocalModelFile(root, "https://test.invalid/model.nam"),
            juce::File());

  // Bytes that are gone everywhere still read as missing (the caller reports
  // the failure), not as some other file.
  EXPECT_FALSE(
      TONE3000Processor::resolveLocalModelFile(root, staleUrl.replace("deadbeef", "0badc0de"))
          .existsAsFile());

  root.deleteRecursively();
}

// The iOS document picker hands back URLs, whose last path component is
// percent-escaped. Names must come back exactly as the desktop path derives
// them from juce::File, or a file called "Deluxe Reverb 2.nam" loads (and
// stashes, and titles its tile) as "Deluxe%20Reverb%202".
TEST(LocalLoadTest, UrlFileNameIsPercentDecoded) {
  const juce::File dir =
      juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("t3k-name");
  const juce::File spaced = dir.getChildFile("Deluxe Reverb 2.nam");
  EXPECT_EQ(TONE3000Processor::localFileNameFromUrl(juce::URL(spaced)), "Deluxe Reverb 2.nam");

  const juce::File accented = dir.getChildFile("Amplificador Válvulas.wav");
  EXPECT_EQ(TONE3000Processor::localFileNameFromUrl(juce::URL(accented)),
            juce::String::fromUTF8("Amplificador V\xc3\xa1lvulas.wav"));

  // A non-file URL has no local file to ask, so the escapes come off the path.
  EXPECT_EQ(TONE3000Processor::localFileNameFromUrl(
                juce::URL("https://test.invalid/x/Deluxe%20Reverb%202.nam")),
            "Deluxe Reverb 2.nam");
}

// The iOS picker path. Compiled on every platform (the DSP suite does not
// build for iOS), fed file:// URLs here, security-scoped ones on the iPad; the
// bytes come through juce::URL::createInputStream either way. Multi-select is
// the folder route on iOS, so it must sort, filter and aggregate errors the
// way loadLocalTonePath does for a folder.
TEST(LocalLoadTest, UrlsLoadMultiSelectInNaturalOrderSkippingBadFiles) {
  const juce::File dir =
      juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("t3k-url-test");
  dir.deleteRecursively();
  ASSERT_TRUE(dir.createDirectory().wasOk());
  ASSERT_TRUE(testFile("a2-amp-test.nam").copyFileTo(dir.getChildFile("amp 2.nam")));
  ASSERT_TRUE(testFile("a2-amp-cab-test.nam").copyFileTo(dir.getChildFile("amp 10.nam")));
  ASSERT_TRUE(testFile("a2-am-test-2.nam").copyFileTo(dir.getChildFile("amp 1.nam")));
  ASSERT_TRUE(dir.getChildFile("notes.txt").replaceWithText("hi"));

  // Picker order is arbitrary; hand them over scrambled, with a wrong type and
  // a file that does not exist mixed in.
  TONE3000Processor proc;
  const juce::var res = proc.loadLocalToneUrls({
      juce::URL(dir.getChildFile("amp 10.nam")),
      juce::URL(dir.getChildFile("notes.txt")),
      juce::URL(dir.getChildFile("amp 2.nam")),
      juce::URL(dir.getChildFile("missing.nam")),
      juce::URL(dir.getChildFile("amp 1.nam")),
  });
  EXPECT_TRUE(res["error"].isVoid()) << res["error"].toString().toStdString();
  ASSERT_TRUE(waitForChainLoaded(proc));

  const juce::var block = firstToneBlock(proc);
  EXPECT_EQ(block["tone"]["title"].toString(), juce::String("5 files"));
  EXPECT_EQ(block["tone"]["format"].toString(), juce::String("nam"));
  ASSERT_EQ(block["tone"]["models"].size(), 3)
      << "models: " << juce::JSON::toString(block["tone"]["models"]).toStdString();
  EXPECT_EQ(block["tone"]["models"][0]["name"].toString(), juce::String("amp 1"));
  EXPECT_EQ(block["tone"]["models"][1]["name"].toString(), juce::String("amp 2"));
  EXPECT_EQ(block["tone"]["models"][2]["name"].toString(), juce::String("amp 10"));

  dir.deleteRecursively();
}

TEST(LocalLoadTest, UrlsSingleFileTitlesFromNameAndRejectBadInputs) {
  TONE3000Processor proc;

  // One file: the title is its name without the extension, as on desktop.
  juce::var res = proc.loadLocalToneUrls({juce::URL(testFile("a2-amp-test.nam"))});
  EXPECT_TRUE(res["error"].isVoid()) << res["error"].toString().toStdString();
  ASSERT_TRUE(waitForChainLoaded(proc));
  EXPECT_EQ(firstToneBlock(proc)["tone"]["title"].toString(), juce::String("a2-amp-test"));

  TONE3000Processor rejecting;
  res = rejecting.loadLocalToneUrls({});
  EXPECT_EQ(res["error"].toString(), juce::String("Nothing to load"));

  res = rejecting.loadLocalToneUrls({juce::URL(juce::File("/nonexistent-t3k/missing.nam"))});
  EXPECT_EQ(res["error"].toString(), juce::String("Couldn't read the file"));

  const juce::File dir =
      juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("t3k-url-reject");
  dir.deleteRecursively();
  ASSERT_TRUE(dir.createDirectory().wasOk());
  ASSERT_TRUE(dir.getChildFile("notes.txt").replaceWithText("hi"));
  res = rejecting.loadLocalToneUrls({juce::URL(dir.getChildFile("notes.txt"))});
  EXPECT_EQ(res["error"].toString(), juce::String("Only .nam and .wav files are supported"));

  // The folder cap (300 files) applies to a multi-select too. Nothing is read
  // before the cap fires, so the URLs need not exist.
  juce::Array<juce::URL> many;
  for (int i = 0; i < 301; ++i)
    many.add(juce::URL(dir.getChildFile("amp " + juce::String(i) + ".nam")));
  res = rejecting.loadLocalToneUrls(many);
  EXPECT_EQ(res["error"].toString(), juce::String("Too many files (max 300)"));

  EXPECT_TRUE(firstToneBlock(rejecting).isVoid());
  dir.deleteRecursively();
}

// ensureWritableDir heals the app-data wound behind github issue #76: a
// folder left root-owned (sudo'd install script, restored backup) fails
// every stash and settings write while reads keep working. The plugin can't
// chown it back; it renames the broken node aside (the parent belongs to
// the user) and starts fresh, deleting nothing.
TEST(LocalLoadTest, EnsureWritableDirCreatesHealsAndPreservesEvidence) {
  const juce::File tmp = juce::File::getSpecialLocation(juce::File::tempDirectory)
                             .getChildFile("t3k-ensure-" + juce::Uuid().toString());
  ASSERT_TRUE(tmp.createDirectory().wasOk());

  // Missing: created outright.
  const juce::File fresh = tmp.getChildFile("data");
  EXPECT_TRUE(TONE3000Processor::ensureWritableDir(fresh));
  EXPECT_TRUE(fresh.isDirectory());

  // Healthy: untouched, contents kept, nothing moved aside.
  ASSERT_TRUE(fresh.getChildFile("keep.bin").replaceWithText("bytes"));
  EXPECT_TRUE(TONE3000Processor::ensureWritableDir(fresh));
  EXPECT_TRUE(fresh.getChildFile("keep.bin").existsAsFile());
  EXPECT_EQ(tmp.findChildFiles(juce::File::findFilesAndDirectories, false).size(), 1);

  // A file squatting on the path: moved aside with its bytes, a directory
  // takes its place.
  const juce::File squat = tmp.getChildFile("squat");
  ASSERT_TRUE(squat.replaceWithText("old"));
  EXPECT_TRUE(TONE3000Processor::ensureWritableDir(squat));
  EXPECT_TRUE(squat.isDirectory());
  bool asideKeptBytes = false;
  for (const auto& sibling : tmp.findChildFiles(juce::File::findFiles, false))
    if (sibling.getFileName().startsWith("squat") && sibling.loadFileAsString() == "old")
      asideKeptBytes = true;
  EXPECT_TRUE(asideKeptBytes);

#if !JUCE_WINDOWS
  // The restored-backup shape: a directory the user still owns, write bits
  // stripped. Healed in place by putting the mode back, so its contents
  // never move. The root-owned variant takes the rename-aside path instead;
  // a test can't stage that unprivileged, and root sails past permission
  // bits anyway, so this leg is unprivileged-only.
  if (geteuid() != 0) {
    const juce::File locked = tmp.getChildFile("locked");
    ASSERT_TRUE(locked.createDirectory().wasOk());
    ASSERT_TRUE(locked.getChildFile("old.t3kpreset").replaceWithText("preset"));
    ASSERT_EQ(::chmod(locked.getFullPathName().toRawUTF8(), 0555), 0);
    ASSERT_FALSE(locked.hasWriteAccess());

    EXPECT_TRUE(TONE3000Processor::ensureWritableDir(locked));
    EXPECT_TRUE(locked.isDirectory());
    EXPECT_TRUE(locked.hasWriteAccess());
    EXPECT_TRUE(locked.getChildFile("old.t3kpreset").existsAsFile());
    // Only the owner got the write bit back: the folder holds the user's
    // sign-in tokens, so 0555 must heal to 0755, never 0777.
    struct stat healed {};
    ASSERT_EQ(::stat(locked.getFullPathName().toRawUTF8(), &healed), 0);
    EXPECT_EQ(healed.st_mode & 0777, 0755u);
    // Nothing moved aside: the fix happened in place.
    for (const auto& sibling : tmp.findChildFiles(juce::File::findDirectories, false))
      EXPECT_TRUE(sibling == locked || !sibling.getFileName().startsWith("locked"));
  }
#endif

  EXPECT_TRUE(tmp.deleteRecursively());
}

// Artwork for local models: the image embedded in the .nam's metadata.image
// (a data URL, as NAMTRIX writes it) or, failing that, a same-named image
// beside a model. Either is copied into the NAMTRIX Player artwork cache by
// content hash and shipped as the tone's image; a malformed one never stops
// the model loading.

namespace {

juce::File artworkCacheDirForTest() {
  juce::File base = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory);
#if JUCE_MAC
  base = base.getChildFile("Application Support");
#endif
  return base.getChildFile("NAMTRIX Player").getChildFile("Artwork");
}

// A solid square, encoded. Colours are per test: tests run in parallel
// processes and share the cache, so one that deletes its file must own it.
juce::MemoryBlock encodedSquare(juce::Colour colour, bool png, int side = 64) {
  juce::Image image(juce::Image::RGB, side, side, true);
  image.clear(image.getBounds(), colour);
  juce::MemoryOutputStream out;
  if (png)
    juce::PNGImageFormat().writeImageToStream(image, out);
  else
    juce::JPEGImageFormat().writeImageToStream(image, out);
  return out.getMemoryBlock();
}

std::string dataUrlOf(const juce::MemoryBlock& bytes, const char* mime) {
  return std::string("data:") + mime + ";base64," +
         juce::Base64::toBase64(bytes.getData(), bytes.getSize()).toStdString();
}

// a2-amp-test.nam with metadata.image set to `image` (any JSON value).
std::string namWithImage(const nlohmann::json& image) {
  juce::MemoryBlock bytes;
  EXPECT_TRUE(testFile("a2-amp-test.nam").loadFileAsData(bytes));
  const auto* begin = static_cast<const char*>(bytes.getData());
  nlohmann::json config = nlohmann::json::parse(begin, begin + bytes.getSize());
  config["metadata"]["image"] = image;
  return config.dump();
}

juce::var namEntry(const juce::String& name, const std::string& json) {
  return fileEntry(name, juce::Base64::toBase64(json.data(), json.size()));
}

// The tone's shipped image URL (what the block card and tile draw), or "".
juce::String artworkOf(TONE3000Processor& proc) {
  const juce::var images = firstToneBlock(proc)["tone"]["images"];
  return images.isArray() && images.size() > 0 ? images[0].toString() : juce::String();
}

juce::File freshDir(const juce::String& name) {
  auto dir = juce::File::getSpecialLocation(juce::File::tempDirectory)
                 .getChildFile("namtrix-artwork-test")
                 .getChildFile(name);
  dir.deleteRecursively();
  dir.createDirectory();
  return dir;
}

void writeBytes(const juce::File& file, const std::string& bytes) {
  ASSERT_TRUE(file.replaceWithData(bytes.data(), bytes.size()));
}

void writeBytes(const juce::File& file, const juce::MemoryBlock& bytes) {
  ASSERT_TRUE(file.replaceWithData(bytes.getData(), bytes.getSize()));
}

}  // namespace

TEST(LocalArtworkTest, EmbeddedJpegIsCachedOnceAndShown) {
  const auto jpeg = encodedSquare(juce::Colour(0xffd03020), false);
  const std::string nam = namWithImage(dataUrlOf(jpeg, "image/jpeg"));

  juce::File cached;
  {
    TONE3000Processor proc;
    const juce::var res = proc.loadLocalTone("embedded", filesOf({namEntry("embedded.nam", nam)}));
    EXPECT_TRUE(res["error"].isVoid()) << res["error"].toString().toStdString();
    ASSERT_TRUE(waitForChainLoaded(proc));

    const juce::String url = artworkOf(proc);
    ASSERT_TRUE(url.isNotEmpty()) << "embedded image was not picked up";
    cached = juce::URL(url).getLocalFile();
    EXPECT_EQ(cached.getParentDirectory(), artworkCacheDirForTest());
    EXPECT_EQ(cached.getFileExtension(), juce::String(".jpg"));
    juce::MemoryBlock onDisk;
    ASSERT_TRUE(cached.loadFileAsData(onDisk));
    EXPECT_EQ(onDisk, jpeg) << "the cache holds the embedded bytes as they are";
  }

  // Written only when missing: a second load of the same image reuses the
  // file untouched.
  const juce::Time old(2020, 0, 1, 0, 0);
  ASSERT_TRUE(cached.setLastModificationTime(old));
  TONE3000Processor proc;
  proc.loadLocalTone("embedded", filesOf({namEntry("embedded.nam", nam)}));
  ASSERT_TRUE(waitForChainLoaded(proc));
  EXPECT_EQ(juce::URL(artworkOf(proc)).getLocalFile(), cached);
  EXPECT_EQ(cached.getLastModificationTime(), old);
}

TEST(LocalArtworkTest, EmbeddedPngIsCachedAndShown) {
  const auto png = encodedSquare(juce::Colour(0xff20b040), true);
  TONE3000Processor proc;
  proc.loadLocalTone("png", filesOf({namEntry("png.nam", namWithImage(dataUrlOf(png, "image/png")))}));
  ASSERT_TRUE(waitForChainLoaded(proc));

  const juce::File cached = juce::URL(artworkOf(proc)).getLocalFile();
  ASSERT_TRUE(cached.existsAsFile());
  EXPECT_EQ(cached.getFileExtension(), juce::String(".png"));
  const juce::Image image = juce::ImageFileFormat::loadFrom(cached);
  ASSERT_TRUE(image.isValid());
  EXPECT_EQ(image.getWidth(), 64);
}

TEST(LocalArtworkTest, SameNamedImageBesideTheModelIsUsed) {
  {
    // Single file; uppercase extension on disk for the case-insensitive match.
    const auto dir = freshDir("sidecar");
    ASSERT_TRUE(testFile("a2-amp-test.nam").copyFileTo(dir.getChildFile("plexi.nam")));
    writeBytes(dir.getChildFile("plexi.JPG"), encodedSquare(juce::Colour(0xff3050d0), false));

    TONE3000Processor proc;
    const juce::var res = proc.loadLocalTonePath(dir.getChildFile("plexi.nam"), {});
    EXPECT_TRUE(res["error"].isVoid()) << res["error"].toString().toStdString();
    ASSERT_TRUE(waitForChainLoaded(proc));

    const juce::File cached = juce::URL(artworkOf(proc)).getLocalFile();
    ASSERT_TRUE(cached.existsAsFile()) << "sidecar image was not picked up";
    // Copied, not referenced in place: the model's folder can move.
    EXPECT_EQ(cached.getParentDirectory(), artworkCacheDirForTest());
    EXPECT_TRUE(juce::ImageFileFormat::loadFrom(cached).isValid());
    dir.deleteRecursively();
  }

  {
    // Folder: the first model *with* an image lends it, not just the first model.
    const auto dir = freshDir("sidecar-folder");
    ASSERT_TRUE(testFile("a2-amp-test.nam").copyFileTo(dir.getChildFile("amp 1.nam")));
    ASSERT_TRUE(testFile("a2-amp-cab-test.nam").copyFileTo(dir.getChildFile("amp 2.nam")));
    const auto png = encodedSquare(juce::Colour(0xff30a0a0), true);
    writeBytes(dir.getChildFile("amp 2.png"), png);

    TONE3000Processor proc;
    proc.loadLocalTonePath(dir, {});
    ASSERT_TRUE(waitForChainLoaded(proc));
    juce::MemoryBlock onDisk;
    ASSERT_TRUE(juce::URL(artworkOf(proc)).getLocalFile().loadFileAsData(onDisk));
    EXPECT_EQ(onDisk, png);
    dir.deleteRecursively();
  }

  {
    const auto dir = freshDir("no-sidecar");
    ASSERT_TRUE(testFile("a2-amp-test.nam").copyFileTo(dir.getChildFile("plexi.nam")));
    TONE3000Processor proc;
    proc.loadLocalTonePath(dir.getChildFile("plexi.nam"), {});
    ASSERT_TRUE(waitForChainLoaded(proc));
    EXPECT_TRUE(artworkOf(proc).isEmpty());
    dir.deleteRecursively();
  }
}

TEST(LocalArtworkTest, EmbeddedImageWinsOverSidecar) {
  const auto embedded = encodedSquare(juce::Colour(0xffd03020), false);
  const auto sidecar = encodedSquare(juce::Colour(0xff8040c0), true);

  {
    // Same model: both present.
    const auto dir = freshDir("both");
    writeBytes(dir.getChildFile("amp.nam"), namWithImage(dataUrlOf(embedded, "image/jpeg")));
    writeBytes(dir.getChildFile("amp.png"), sidecar);
    TONE3000Processor proc;
    proc.loadLocalTonePath(dir.getChildFile("amp.nam"), {});
    ASSERT_TRUE(waitForChainLoaded(proc));
    juce::MemoryBlock onDisk;
    ASSERT_TRUE(juce::URL(artworkOf(proc)).getLocalFile().loadFileAsData(onDisk));
    EXPECT_EQ(onDisk, embedded);
    dir.deleteRecursively();
  }

  {
    // Folder: a later model's embedded image beats an earlier model's sidecar.
    const auto dir = freshDir("both-folder");
    ASSERT_TRUE(testFile("a2-amp-test.nam").copyFileTo(dir.getChildFile("a.nam")));
    writeBytes(dir.getChildFile("a.png"), sidecar);
    writeBytes(dir.getChildFile("b.nam"), namWithImage(dataUrlOf(embedded, "image/jpeg")));
    TONE3000Processor proc;
    proc.loadLocalTonePath(dir, {});
    ASSERT_TRUE(waitForChainLoaded(proc));
    juce::MemoryBlock onDisk;
    ASSERT_TRUE(juce::URL(artworkOf(proc)).getLocalFile().loadFileAsData(onDisk));
    EXPECT_EQ(onDisk, embedded);
    dir.deleteRecursively();
  }
}

TEST(LocalArtworkTest, MalformedArtworkStillLoadsTheModel) {
  const juce::String notAnImage = juce::Base64::toBase64("hello, not an image", 19);
  const std::vector<std::pair<const char*, nlohmann::json>> cases = {
      {"empty string", ""},
      {"not a data URL", "https://example.com/art.jpg"},
      {"no comma", "data:image/jpeg;base64"},
      {"unsupported type", "data:image/gif;base64,R0lGODlhAQABAAAAACw="},
      {"not base64-encoded", "data:image/jpeg,rawbytes"},
      {"bad base64", "data:image/jpeg;base64,!!!not base64!!!"},
      {"not an image", "data:image/jpeg;base64," + notAnImage.toStdString()},
      {"over the size limit", "data:image/jpeg;base64," + std::string(3 * 1024 * 1024, 'A')},
      {"a number", 42},
      {"an object", nlohmann::json{{"url", "art.jpg"}}},
  };

  for (const auto& [label, image] : cases) {
    SCOPED_TRACE(label);
    TONE3000Processor proc;
    const juce::var res = proc.loadLocalTone("bad", filesOf({namEntry("bad.nam", namWithImage(image))}));
    EXPECT_TRUE(res["error"].isVoid()) << res["error"].toString().toStdString();
    ASSERT_TRUE(waitForChainLoaded(proc));
    const juce::var block = firstToneBlock(proc);
    EXPECT_TRUE(static_cast<bool>(block["loaded"]));
    EXPECT_FALSE(static_cast<bool>(block["loadFailed"]));
    EXPECT_TRUE(artworkOf(proc).isEmpty());
  }
}

TEST(LocalArtworkTest, MissingCachedArtworkIsRederivedOnRestore) {
  // A colour no other test uses: this one deletes its cached file.
  const auto jpeg = encodedSquare(juce::Colour(0xff10e0f0), false);
  const std::string nam = namWithImage(dataUrlOf(jpeg, "image/jpeg"));

  juce::MemoryBlock saved;
  juce::File cached;
  {
    TONE3000Processor proc;
    proc.loadLocalTone("restore", filesOf({namEntry("restore.nam", nam)}));
    ASSERT_TRUE(waitForChainLoaded(proc));
    cached = juce::URL(artworkOf(proc)).getLocalFile();
    ASSERT_TRUE(cached.existsAsFile());
    proc.getStateInformation(saved);
  }

  ASSERT_TRUE(cached.deleteFile());  // cleared cache

  TONE3000Processor restored;
  restored.setStateInformation(saved.getData(), static_cast<int>(saved.getSize()));
  ASSERT_TRUE(waitForChainLoaded(restored));
  const juce::String url = artworkOf(restored);
  ASSERT_TRUE(url.isNotEmpty()) << "artwork was not re-derived from the model";
  EXPECT_EQ(juce::URL(url).getLocalFile(), cached);
  juce::MemoryBlock onDisk;
  ASSERT_TRUE(cached.loadFileAsData(onDisk));
  EXPECT_EQ(onDisk, jpeg);
}

TEST(LocalArtworkTest, NoArtworkToRederiveLeavesThePlaceholder) {
  // A sidecar can't be re-derived (the original folder isn't known), so a
  // restored chain whose copy is gone ships no image instead of a dead URL.
  const auto dir = freshDir("sidecar-gone");
  ASSERT_TRUE(testFile("a2-amp-test.nam").copyFileTo(dir.getChildFile("plexi.nam")));
  writeBytes(dir.getChildFile("plexi.png"), encodedSquare(juce::Colour(0xfff0e010), true));

  juce::MemoryBlock saved;
  juce::File cached;
  {
    TONE3000Processor proc;
    proc.loadLocalTonePath(dir.getChildFile("plexi.nam"), {});
    ASSERT_TRUE(waitForChainLoaded(proc));
    cached = juce::URL(artworkOf(proc)).getLocalFile();
    ASSERT_TRUE(cached.existsAsFile());
    proc.getStateInformation(saved);
  }
  dir.deleteRecursively();
  ASSERT_TRUE(cached.deleteFile());

  TONE3000Processor restored;
  restored.setStateInformation(saved.getData(), static_cast<int>(saved.getSize()));
  ASSERT_TRUE(waitForChainLoaded(restored));
  EXPECT_TRUE(static_cast<bool>(firstToneBlock(restored)["loaded"]));
  EXPECT_TRUE(artworkOf(restored).isEmpty());
}

TEST(LocalArtworkTest, EmbeddedImageDoesNotChangeTheSound) {
  // test/files/a2-amp-test-artwork.nam is a2-amp-test.nam with a 512 px
  // JPEG added as metadata.image and nothing else changed.
  const auto engineFor = [](const char* name) {
    juce::MemoryBlock bytes;
    EXPECT_TRUE(testFile(name).loadFileAsData(bytes));
    const auto* begin = static_cast<const char*>(bytes.getData());
    const nlohmann::json config = nlohmann::json::parse(begin, begin + bytes.getSize());
    std::vector<std::unique_ptr<nam::DSP>> instances;
    instances.push_back(nam::get_dsp(config));
    auto engine = std::make_unique<NamEngine>(std::move(instances), 1, 1);
    engine->prepare(512);
    return engine;
  };
  auto plain = engineFor("a2-amp-test.nam");
  auto withArt = engineFor("a2-amp-test-artwork.nam");

  const auto in = makeSine(512 * 40, 220.0, 0.25f);
  juce::AudioBuffer<float> a(2, 512), b(2, 512);
  for (size_t off = 0; off < in.size(); off += 512) {
    for (int ch = 0; ch < 2; ++ch) {
      a.copyFrom(ch, 0, in.data() + off, 512);
      b.copyFrom(ch, 0, in.data() + off, 512);
    }
    plain->process(a);
    withArt->process(b);
    for (int ch = 0; ch < 2; ++ch)
      ASSERT_EQ(std::memcmp(a.getReadPointer(ch), b.getReadPointer(ch), 512 * sizeof(float)), 0)
          << "output differs at sample " << off;
  }

  // And it shows its artwork when loaded from disk.
  TONE3000Processor proc;
  proc.loadLocalTonePath(testFile("a2-amp-test-artwork.nam"), {});
  ASSERT_TRUE(waitForChainLoaded(proc));
  const juce::Image art = juce::ImageFileFormat::loadFrom(juce::URL(artworkOf(proc)).getLocalFile());
  ASSERT_TRUE(art.isValid());
  EXPECT_EQ(art.getWidth(), 512);
  EXPECT_EQ(art.getHeight(), 512);
}

// A photo dropped on a model loaded from disk is written into that .nam as
// its artwork (embedLocalArtwork): a 512 px centre-square JPEG data URL in
// metadata.image, with every other byte of the file left as it was.

namespace {

// A landscape photo: red centre square with blue bands left and right, so a
// correct centre crop comes out all red.
juce::File writePhoto(const juce::File& file, juce::Colour centre, int w = 1200, int h = 800) {
  juce::Image image(juce::Image::RGB, w, h, true);
  image.clear(image.getBounds(), juce::Colours::blue);
  image.clear({(w - h) / 2, 0, h, h}, centre);
  juce::FileOutputStream out(file);
  EXPECT_TRUE(out.openedOk());
  out.setPosition(0);
  out.truncate();
  juce::PNGImageFormat().writeImageToStream(image, out);
  return file;
}

std::string readText(const juce::File& file) {
  juce::MemoryBlock bytes;
  EXPECT_TRUE(file.loadFileAsData(bytes));
  return std::string(static_cast<const char*>(bytes.getData()), bytes.getSize());
}

// The decoded metadata.image of a .nam on disk (invalid when absent).
juce::Image embeddedImageOf(const juce::File& nam) {
  const auto config = nlohmann::json::parse(readText(nam));
  const auto url = config.at("metadata").value("image", std::string());
  const std::string prefix = "data:image/jpeg;base64,";
  if (url.rfind(prefix, 0) != 0)
    return {};
  juce::MemoryOutputStream jpeg;
  EXPECT_TRUE(juce::Base64::convertFromBase64(jpeg, url.substr(prefix.size())));
  return juce::ImageFileFormat::loadFrom(jpeg.getData(), jpeg.getDataSize());
}

bool isMostly(juce::Colour pixel, juce::Colour want) {
  return std::abs(pixel.getRed() - want.getRed()) < 24 &&
         std::abs(pixel.getGreen() - want.getGreen()) < 24 &&
         std::abs(pixel.getBlue() - want.getBlue()) < 24;
}

// Loads `nam` from disk into a fresh slot; the new block's id.
std::string loadFromDisk(TONE3000Processor& proc, const juce::File& nam) {
  const juce::var res = proc.loadLocalTonePath(nam, {});
  EXPECT_TRUE(res["error"].isVoid()) << res["error"].toString().toStdString();
  EXPECT_TRUE(waitForChainLoaded(proc));
  return res["blockId"].toString().toStdString();
}

}  // namespace

TEST(LocalArtworkTest, DroppedPhotoIsSavedIntoTheNamFile) {
  const auto dir = freshDir("drop-photo");
  const juce::File nam = dir.getChildFile("plexi.nam");
  ASSERT_TRUE(testFile("a2-amp-test.nam").copyFileTo(nam));
  const std::string original = readText(nam);
  const juce::File photo = writePhoto(dir.getChildFile("photo.png"), juce::Colours::red);

  juce::MemoryBlock saved;
  juce::File cached;
  {
    TONE3000Processor proc;
    const std::string blockId = loadFromDisk(proc, nam);
    const juce::var res = proc.embedLocalArtwork(blockId, photo);
    ASSERT_TRUE(res["error"].isVoid()) << res["error"].toString().toStdString();
    EXPECT_EQ(res["savedTo"].toString(), juce::String("plexi.nam"));

    // In the file: a 512 px square, cropped from the photo's centre.
    const juce::Image embedded = embeddedImageOf(nam);
    ASSERT_TRUE(embedded.isValid());
    EXPECT_EQ(embedded.getWidth(), 512);
    EXPECT_EQ(embedded.getHeight(), 512);
    EXPECT_TRUE(isMostly(embedded.getPixelAt(4, 4), juce::Colours::red));
    EXPECT_TRUE(isMostly(embedded.getPixelAt(507, 256), juce::Colours::red));

    // Every other byte as it was: the edit is one inserted member.
    std::string edited = readText(nam);
    const size_t at = edited.find("\"image\": ");
    ASSERT_NE(at, std::string::npos);
    const size_t end = edited.find("\", ", at) + 3;
    edited.erase(at, end - at);
    EXPECT_TRUE(edited == original) << "the rest of the file changed";

    // On the tile now, from the artwork cache.
    cached = juce::URL(artworkOf(proc)).getLocalFile();
    ASSERT_TRUE(cached.existsAsFile());
    EXPECT_EQ(cached.getParentDirectory(), artworkCacheDirForTest());
    EXPECT_TRUE(static_cast<bool>(firstToneBlock(proc)["loaded"]));
    proc.getStateInformation(saved);
  }

  {
    // Loading the file again finds it embedded.
    TONE3000Processor proc;
    loadFromDisk(proc, nam);
    EXPECT_EQ(juce::URL(artworkOf(proc)).getLocalFile(), cached);
  }

  // The saved state carries the new file's bytes: with the cache cleared,
  // a restore re-derives the photo from them.
  ASSERT_TRUE(cached.deleteFile());
  TONE3000Processor restored;
  restored.setStateInformation(saved.getData(), static_cast<int>(saved.getSize()));
  ASSERT_TRUE(waitForChainLoaded(restored));
  EXPECT_EQ(juce::URL(artworkOf(restored)).getLocalFile(), cached);
  EXPECT_TRUE(cached.existsAsFile());
  dir.deleteRecursively();
}

TEST(LocalArtworkTest, SecondPhotoReplacesTheFirst) {
  const auto dir = freshDir("replace-photo");
  const juce::File nam = dir.getChildFile("amp.nam");
  ASSERT_TRUE(testFile("a2-amp-test-artwork.nam").copyFileTo(nam));  // starts with artwork

  TONE3000Processor proc;
  const std::string blockId = loadFromDisk(proc, nam);
  for (const auto colour : {juce::Colours::green, juce::Colours::yellow}) {
    const juce::var res =
        proc.embedLocalArtwork(blockId, writePhoto(dir.getChildFile("photo.png"), colour, 300, 200));
    ASSERT_TRUE(res["error"].isVoid()) << res["error"].toString().toStdString();
    const juce::Image embedded = embeddedImageOf(nam);
    ASSERT_TRUE(embedded.isValid());
    EXPECT_EQ(embedded.getWidth(), 200) << "a small photo isn't upscaled";
    EXPECT_TRUE(isMostly(embedded.getPixelAt(100, 100), colour));
  }
  const std::string text = readText(nam);
  size_t count = 0;
  for (size_t at = text.find("\"image\""); at != std::string::npos; at = text.find("\"image\"", at + 1))
    ++count;
  EXPECT_EQ(count, 1u);
  dir.deleteRecursively();
}

TEST(LocalArtworkTest, DroppedPhotoWorksWithoutExistingMetadata) {
  // Files whose metadata is absent, null or empty get one with the image.
  juce::MemoryBlock bytes;
  ASSERT_TRUE(testFile("a2-amp-test.nam").loadFileAsData(bytes));
  const auto* begin = static_cast<const char*>(bytes.getData());
  const auto base = nlohmann::json::parse(begin, begin + bytes.getSize());

  const std::vector<std::pair<const char*, std::function<void(nlohmann::json&)>>> variants = {
      {"absent", [](nlohmann::json& c) { c.erase("metadata"); }},
      {"null", [](nlohmann::json& c) { c["metadata"] = nullptr; }},
      {"empty", [](nlohmann::json& c) { c["metadata"] = nlohmann::json::object(); }},
  };
  for (const auto& [label, change] : variants) {
    SCOPED_TRACE(label);
    const auto dir = freshDir(juce::String("metadata-") + label);
    nlohmann::json config = base;
    change(config);
    const juce::File nam = dir.getChildFile("amp.nam");
    writeBytes(nam, config.dump(1));

    TONE3000Processor proc;
    const std::string blockId = loadFromDisk(proc, nam);
    const juce::var res =
        proc.embedLocalArtwork(blockId, writePhoto(dir.getChildFile("p.png"), juce::Colours::red));
    ASSERT_TRUE(res["error"].isVoid()) << res["error"].toString().toStdString();
    EXPECT_TRUE(embeddedImageOf(nam).isValid());
    // Still the same model.
    auto after = nlohmann::json::parse(readText(nam));
    after.erase("metadata");
    config.erase("metadata");
    EXPECT_TRUE(after == config);
    dir.deleteRecursively();
  }
}

TEST(LocalArtworkTest, DroppedPhotoDoesNotChangeTheSound) {
  const auto dir = freshDir("drop-sound");
  const juce::File nam = dir.getChildFile("amp.nam");
  ASSERT_TRUE(testFile("a2-amp-test.nam").copyFileTo(nam));
  {
    TONE3000Processor proc;
    const std::string blockId = loadFromDisk(proc, nam);
    ASSERT_TRUE(proc.embedLocalArtwork(blockId, writePhoto(dir.getChildFile("p.png"), juce::Colours::red))["error"].isVoid());
  }

  const auto engineFor = [](const juce::File& file) {
    juce::MemoryBlock bytes;
    EXPECT_TRUE(file.loadFileAsData(bytes));
    const auto* begin = static_cast<const char*>(bytes.getData());
    std::vector<std::unique_ptr<nam::DSP>> instances;
    instances.push_back(nam::get_dsp(nlohmann::json::parse(begin, begin + bytes.getSize())));
    auto engine = std::make_unique<NamEngine>(std::move(instances), 1, 1);
    engine->prepare(512);
    return engine;
  };
  auto plain = engineFor(testFile("a2-amp-test.nam"));
  auto withArt = engineFor(nam);
  const auto in = makeSine(512 * 20, 330.0, 0.25f);
  juce::AudioBuffer<float> a(2, 512), b(2, 512);
  for (size_t off = 0; off < in.size(); off += 512) {
    for (int ch = 0; ch < 2; ++ch) {
      a.copyFrom(ch, 0, in.data() + off, 512);
      b.copyFrom(ch, 0, in.data() + off, 512);
    }
    plain->process(a);
    withArt->process(b);
    for (int ch = 0; ch < 2; ++ch)
      ASSERT_EQ(std::memcmp(a.getReadPointer(ch), b.getReadPointer(ch), 512 * sizeof(float)), 0);
  }
  dir.deleteRecursively();
}

TEST(LocalArtworkTest, DroppedPhotoRefusalsLeaveTheFileAlone) {
  const auto dir = freshDir("drop-refusals");
  const juce::File nam = dir.getChildFile("amp.nam");
  ASSERT_TRUE(testFile("a2-amp-test.nam").copyFileTo(nam));
  const juce::File photo = writePhoto(dir.getChildFile("p.png"), juce::Colours::red);
  const auto errorOf = [](const juce::var& res) { return res["error"].toString(); };

  TONE3000Processor proc;
  const std::string blockId = loadFromDisk(proc, nam);
  const std::string original = readText(nam);

  {
    SCOPED_TRACE("not an image");
    const juce::File junk = dir.getChildFile("junk.png");
    writeBytes(junk, std::string("not a png"));
    EXPECT_EQ(errorOf(proc.embedLocalArtwork(blockId, junk)), juce::String("Couldn't read that photo"));
  }
  {
    SCOPED_TRACE("unsupported type");
    const juce::File gif = dir.getChildFile("p.gif");
    ASSERT_TRUE(photo.copyFileTo(gif));
    EXPECT_EQ(errorOf(proc.embedLocalArtwork(blockId, gif)), juce::String("Drop a .jpg or .png photo"));
  }
  {
    SCOPED_TRACE("no such block");
    EXPECT_EQ(errorOf(proc.embedLocalArtwork("nope", photo)), juce::String("Drop the photo on a model"));
  }
#if !JUCE_WINDOWS
  if (geteuid() != 0) {
    SCOPED_TRACE("read-only file");
    ASSERT_EQ(chmod(nam.getFullPathName().toRawUTF8(), 0444), 0);
    EXPECT_EQ(errorOf(proc.embedLocalArtwork(blockId, photo)), juce::String("amp.nam is read-only"));
    ASSERT_EQ(chmod(nam.getFullPathName().toRawUTF8(), 0644), 0);
  }
#endif
  EXPECT_TRUE(readText(nam) == original);

  {
    SCOPED_TRACE("file changed since it was loaded");
    const std::string changed = original + "\n";
    writeBytes(nam, changed);
    EXPECT_TRUE(errorOf(proc.embedLocalArtwork(blockId, photo)).contains("changed since it was loaded"));
    EXPECT_TRUE(readText(nam) == changed);
  }
  {
    SCOPED_TRACE("model gone from disk");
    ASSERT_TRUE(nam.deleteFile());
    EXPECT_TRUE(errorOf(proc.embedLocalArtwork(blockId, photo)).startsWith("Can't find amp.nam on disk"));
  }
  {
    SCOPED_TRACE("loaded as bytes: no file to write");
    TONE3000Processor other;
    const juce::var res = other.loadLocalTone("a2-amp-test", filesOf({testFileEntry("a2-amp-test.nam")}));
    ASSERT_TRUE(waitForChainLoaded(other));
    EXPECT_TRUE(errorOf(other.embedLocalArtwork(res["blockId"].toString().toStdString(), photo))
                    .startsWith("Can't find"));
  }
  {
    SCOPED_TRACE("an IR");
    TONE3000Processor other;
    const juce::var res = other.loadLocalTonePath(testFile("cab-ir-test.wav"), {});
    ASSERT_TRUE(waitForChainLoaded(other));
    EXPECT_EQ(errorOf(other.embedLocalArtwork(res["blockId"].toString().toStdString(), photo)),
              juce::String("Artwork can only be added to .nam models"));
  }
  dir.deleteRecursively();
}
