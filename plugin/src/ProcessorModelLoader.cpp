#include "Processor.h"
#include "json.hpp"
#include "NAM/wavenet/a2_fast.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#if !JUCE_WINDOWS
#include <sys/stat.h>
#endif

// #####################################
// MODEL LOADING HELPERS
// #####################################

// IR block sizing constants.
// TONE3000 IR tones cover two very different species: cab IRs (tens of
// milliseconds) and convolution-reverb IRs (whole seconds). ONE length
// cutoff (kShortIrMaxSeconds) splits every IR into short or long, and that
// split drives everything downstream:
//   short: uniform zero-latency engine, -18 dB output pad, 100% default mix
//   long:  non-uniform engine,           no output pad,     50% default mix
// The engine choice is always by length (a CPU decision, inaudible). The
// audible pair (pad, default mix) is by length only when the tone's catalog
// gear doesn't already say what the IR is; see irIsLongFor.
namespace {

// Hard cap on loaded IR length. Bounds memory and engine-build time for
// arbitrary downloads while comfortably covering any published reverb IR
// (the longest cathedral tails run ~8 s).
constexpr double kMaxIrSeconds = 10.0;

// The short/long cutoff: cab IRs top out around 0.5 s, reverbs start well
// above 1 s; nothing meaningful lives at the boundary. IRs always convolve
// at the base rate (see ChainBlock::irBaseRateIsland), so the sample
// threshold is a constant.
constexpr double kShortIrMaxSeconds = 1.0;
constexpr int kShortIrMaxBaseSamples = static_cast<int>(kShortIrMaxSeconds * kChainBaseSampleRate);

// The cab-like / reverb-like verdict for a loaded IR (ChainBlock::irIsLong),
// from the tone's catalog gear and the kernel-length fallback. The gear tag
// wins where it is unambiguous: a "cab" is a cab however much room tail or
// trailing noise floor the file carries, and a "space" is a reverb however
// short (github issue #89: a 389 ms chamber, and a cab whose fade-out dipped
// under the trim floor just inside the cutoff, both used to flip on length
// alone and land 18 dB and a mix default apart from their siblings). Every
// other gear ("pedal", "experimental", untagged local files) genuinely mixes
// both species, so there the length decides, as before.
bool irIsLongFor(const juce::String& gear, bool longByLength) {
  if (gear.equalsIgnoreCase("cab")) return false;
  if (gear.equalsIgnoreCase("space")) return true;
  return longByLength;
}

// Long IRs use JUCE's two-stage non-uniform engine (still zero latency):
// the first kIrNonUniformHeadSamples convolve in partitions of the
// convolver's prepared block size (irConvolverBlockSizeFor, ChainBlock.h),
// the tail in partitions of this size, so per-callback CPU stops scaling
// linearly with tail length. Bigger head = more per-callback FFT work;
// smaller = chunkier tail batches. 8192 (~170 ms) is a conventional
// reverb head size.
constexpr int kIrNonUniformHeadSamples = 8192;

// NAM phase-interleaved oversampling eligibility (see NamEngine.h).
// Phase interleaving is exact only for architectures whose temporal structure
// is pure (dilated) convolution: splitting the oversampled stream into factor
// phases and running one native-rate instance per phase computes the same
// math as one dilation-scaled instance at the high rate, and the temporal
// images a dilated kernel produces above the base Nyquist are removed by the
// chain's decimation filter. Recurrent models (LSTM) update state on
// consecutive samples and can't be phase-split; they get a single instance
// running time-scaled at the full chain rate instead.
bool namConfigIsPhaseSafe(const nlohmann::json& modelJson) {
  const std::string architecture = modelJson.value("architecture", "");
  if (architecture == "WaveNet" || architecture == "ConvNet" || architecture == "Linear")
    return true;
  if (architecture == "SlimmableContainer") {
    // Safe only when every submodel is (containers switch tiers at runtime).
    if (!modelJson.contains("config") || !modelJson["config"].contains("submodels"))
      return false;
    const auto& submodels = modelJson["config"]["submodels"];
    if (!submodels.is_array() || submodels.empty())
      return false;
    for (const auto& entry : submodels) {
      if (!entry.contains("model") || !namConfigIsPhaseSafe(entry["model"]))
        return false;
    }
    return true;
  }
  return false;  // LSTM and anything unknown
}

// A2's layout at a width the fast engine was not compiled for (it is a
// template instantiated only at 3 and 8 channels). These are still A2 in every
// respect that matters to the runtime -- same layers, kernels, dilations,
// activations and head -- so they are admitted and simply run on the generic
// WaveNet engine, which is slower but exact.
//
// is_a2_shape is not loosened to say this: it also SELECTS the fast engine,
// and a match at 16 channels would try to build a fast model that doesn't
// exist. Instead the question is put to it exactly: would this config pass if
// it were 8 wide? Only channels and bottleneck depend on width in that check,
// so normalising those two leaves every other A2 rule enforced unchanged.
bool namConfigIsWideA2(const nlohmann::json& modelJson) {
  if (modelJson.value("architecture", "") != "WaveNet" || !modelJson.contains("config"))
    return false;
  nlohmann::json config = modelJson["config"];
  auto layers = config.find("layers");
  if (layers == config.end() || !layers->is_array() || layers->size() != 1 ||
      !(*layers)[0].is_object())
    return false;
  auto& layerArray = (*layers)[0];
  const int width = layerArray.value("channels", 0);
  // A2 requires channels == bottleneck; keep that rule rather than paper over it.
  if (width <= 0 || width != layerArray.value("bottleneck", 0))
    return false;
  layerArray["channels"] = 8;
  layerArray["bottleneck"] = 8;
  int fastWidth = 0;
  return nam::wavenet::a2_fast::is_a2_shape(config, &fastWidth);
}

// Gate for drop-loaded local NAM files: the catalog only serves
// architecture-2 models (the runtime is tuned around them: 48 kHz training
// rate, the slimmable tiers, the fast path), so local files must be A2-shaped
// too -- at the fast engine's widths, or wider on the generic engine.
// Same container shape as namConfigIsPhaseSafe: a bare A2 WaveNet, or a
// SlimmableContainer whose every submodel is one.
bool namConfigIsA2(const nlohmann::json& modelJson) {
  const std::string architecture = modelJson.value("architecture", "");
  if (architecture == "WaveNet") {
    int channels = 0;
    return modelJson.contains("config") &&
           (nam::wavenet::a2_fast::is_a2_shape(modelJson["config"], &channels) ||
            namConfigIsWideA2(modelJson));
  }
  if (architecture == "SlimmableContainer") {
    if (!modelJson.contains("config") || !modelJson["config"].contains("submodels"))
      return false;
    const auto& submodels = modelJson["config"]["submodels"];
    if (!submodels.is_array() || submodels.empty())
      return false;
    for (const auto& entry : submodels) {
      if (!entry.contains("model") || !namConfigIsA2(entry["model"]))
        return false;
    }
    return true;
  }
  return false;
}

// Parametric (knob-conditioned) architectures from the NeuralAmpModelerCore
// parametric fork. These are not A2-shaped -- their conditioning path is a
// FiLM/concat stage the A2 fast path doesn't model -- so they fail
// namConfigIsA2 and need admitting on their own terms.
//
// Deliberately NOT added to namConfigIsPhaseSafe: phase interleaving is exact
// only for pure dilated convolution, and neither the concat conditioner (which
// carries its own ramp state across a block) nor HyperWaveNet (which
// regenerates weights per parameter change) is that. They fall through to the
// single-instance, time-scaled path, which is the same defensive route LSTM
// takes -- correct, just without the oversampling win.
bool namConfigIsParametric(const nlohmann::json& modelJson) {
  const std::string architecture = modelJson.value("architecture", "");
  return architecture == "ConcatWaveNet" || architecture == "HyperWaveNet" ||
         architecture == "ConcatLSTM";
}

// Stash folder for drop-loaded local models. The block's toneJson persists
// the stash path as its model_url, so a cache-lost reload (undo after
// remove, undo across a tone swap) re-reads this copy even after the user's
// original file moved. Content-addressed names dedupe re-drops of the same
// file; stale entries age out (see cleanLocalModelStash). Same app-data
// root as PresetManager. Persisted URLs are resolved back to this folder by
// resolveLocalModelFile, which is what survives the iOS container rotating.
juce::File localModelsDir() {
  juce::File base = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory);
#if JUCE_MAC
  base = base.getChildFile("Application Support");
#endif
  return base.getChildFile("TONE3000").getChildFile("LocalModels");
}

// FNV-1a over the file bytes: stable across sessions and platforms without
// pulling in juce_cryptography. The size suffix in the stash name backs up
// the (already negligible) collision odds.
juce::uint64 fnv1a64(const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  juce::uint64 hash = 1469598103934665603ULL;
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

// Caps for local file loads, the same whether the bytes arrive as a
// base64 array (the DSP tests) or straight from disk (the UI's drops and
// pickers, plugin/ui/services/LocalFiles).
constexpr juce::int64 kMaxLocalFileBytes = 50 * 1024 * 1024;
constexpr int kMaxFolderModels = 300;

// Best-effort catalog gear id ("amp", "amp-cab", "cab", "pedal", "outboard")
// for a local NAM file, from the trainer-written `metadata.gear_type`. The
// field is free text, so only the common spellings map; anything else (or
// no metadata at all) is "" and the UI keeps its generic file glyph.
juce::String localGearFromNamMetadata(const nlohmann::json& config) {
  const auto metadata = config.find("metadata");
  if (metadata == config.end() || !metadata->is_object())
    return {};
  const auto gearType = metadata->find("gear_type");
  if (gearType == metadata->end() || !gearType->is_string())
    return {};

  const juce::String type = juce::String(gearType->get<std::string>()).trim().toLowerCase();
  if (type == "amp" || type == "pedal_amp" || type == "preamp") return "amp";
  if (type == "amp_cab" || type == "amp_pedal_cab" || type == "amp-cab") return "amp-cab";
  if (type == "studio" || type == "outboard") return "outboard";
  if (type == "pedal") return "pedal";
  if (type == "cab") return "cab";
  return {};
}

// Same for a local IR: a kernel on the cab side of the short/long cutoff is
// a cab; anything longer (reverbs, rooms) stays generic.
juce::String localGearFromIr(const juce::AudioFormatReader& reader) {
  if (reader.sampleRate <= 0.0)
    return {};
  const double seconds = static_cast<double>(reader.lengthInSamples) / reader.sampleRate;
  return seconds <= kShortIrMaxSeconds ? "cab" : juce::String();
}

// Artwork for local models: the image NAMTRIX embeds in the .nam itself
// (metadata.image, a "data:image/jpeg;base64,..." or image/png data URL), or
// an image beside the model with the same base name ("M4 - Marshall.nam" +
// "M4 - Marshall.jpg"). Either way the bytes are copied into this cache,
// named by their content hash, and the tone's images array points at the
// copy with a file:// URL. The editor's ImageLoader reads file URLs directly
// (juce::URL streams local files), decoding and sizing off the message thread
// like any catalog artwork, so makeToneSummary and the UI need nothing new.
//
// Unlike the model stash, nothing here is swept: presets and DAW state point
// at these files by path. If one goes missing anyway (cleared cache, another
// Mac), the block's next model load re-derives it; see localArtworkMissing in
// ProcessorChain.cpp.
juce::File artworkCacheDir() {
  juce::File base = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory);
#if JUCE_MAC
  base = base.getChildFile("Application Support");
#endif
  return base.getChildFile("NAMTRIX Player").getChildFile("Artwork");
}

// Artwork is decoration, so anything past these is ignored rather than
// decoded for a 192 px tile. Embedded images are NAMTRIX's ~60 KB squares;
// sidecars can be camera photos, hence the roomier cap.
constexpr size_t kMaxEmbeddedArtworkBytes = 2 * 1024 * 1024;
constexpr juce::int64 kMaxSidecarArtworkBytes = 8 * 1024 * 1024;

// ".jpg" or ".png" when the bytes decode as that format, else nullptr. Sniffed
// from the bytes rather than trusted from a MIME type or file extension, and
// decoded once in full: a truncated or corrupt image must fall back to the
// placeholder here, not reach the UI as a URL that never draws.
const char* decodableArtworkExtension(const void* data, size_t size) {
  juce::MemoryInputStream in(data, size, false);
  const char* extension = nullptr;
  if (juce::JPEGImageFormat().canUnderstand(in)) {
    extension = ".jpg";
  } else {
    in.setPosition(0);
    if (juce::PNGImageFormat().canUnderstand(in))
      extension = ".png";
  }
  if (extension == nullptr || !juce::ImageFileFormat::loadFrom(data, size).isValid())
    return nullptr;
  return extension;
}

// Copy image bytes into the artwork cache (written only when that content
// isn't there yet) and return its file:// URL, or "" when they don't decode
// or can't be stored.
juce::String cacheArtworkBytes(const void* data, size_t size) {
  const char* extension = decodableArtworkExtension(data, size);
  if (extension == nullptr)
    return {};
  const juce::uint64 hash = fnv1a64(data, size);
  const juce::File cached = artworkCacheDir().getChildFile(
      juce::String::toHexString(static_cast<juce::int64>(hash)) + "-" +
      juce::String(static_cast<juce::int64>(size)) + extension);
  if (!cached.existsAsFile() &&
      !(TONE3000Processor::ensureWritableDir(cached.getParentDirectory()) &&
        cached.replaceWithData(data, size)))
    return {};
  return juce::URL(cached).toString(false);
}

// The model's embedded artwork, cached; "" when it has none or it's unusable.
// Never throws: a bad image must not stop the model loading.
juce::String cacheEmbeddedArtwork(const nlohmann::json& config) {
  try {
    const auto metadata = config.find("metadata");
    if (metadata == config.end() || !metadata->is_object())
      return {};
    const auto image = metadata->find("image");
    if (image == metadata->end() || !image->is_string())
      return {};
    const auto& dataUrl = image->get_ref<const std::string&>();
    if (dataUrl.empty())
      return {};

    auto reject = [](const juce::String& why) {
      juce::Logger::writeToLog("[LocalLoad] Embedded artwork ignored: " + why);
      return juce::String();
    };

    const size_t comma = dataUrl.find(',');
    if (comma == std::string::npos || comma > 64)
      return reject("not a data URL");
    const juce::String header =
        juce::String(dataUrl.data(), comma).trim().toLowerCase().removeCharacters(" ");
    if (header != "data:image/jpeg;base64" && header != "data:image/jpg;base64" &&
        header != "data:image/png;base64")
      return reject("not a base64 JPEG or PNG data URL");

    // Base64 is 4 chars per 3 bytes; checked before decoding so an oversized
    // payload costs nothing. The slack covers line breaks some encoders add.
    const size_t payloadChars = dataUrl.size() - comma - 1;
    if (payloadChars > kMaxEmbeddedArtworkBytes / 3 * 4 + 4096)
      return reject("too large");
    const juce::String payload =
        juce::String(dataUrl.data() + comma + 1, payloadChars).removeCharacters(" \t\r\n");
    juce::MemoryOutputStream decoded;
    if (!juce::Base64::convertFromBase64(decoded, payload) || decoded.getDataSize() == 0)
      return reject("bad base64");
    if (decoded.getDataSize() > kMaxEmbeddedArtworkBytes)
      return reject("too large");

    const juce::String url = cacheArtworkBytes(decoded.getData(), decoded.getDataSize());
    if (url.isEmpty())
      return reject("doesn't decode as an image");
    return url;
  } catch (const std::exception&) {
    return {};
  }
}

// The same-named image beside a model, cached; "" when there is none.
juce::String cacheSidecarArtwork(const juce::File& modelFile) {
  // Preference order when several exist.
  static const char* kImageExtensions[] = {".jpg", ".jpeg", ".png"};

  const juce::File directory = modelFile.getParentDirectory();
  const juce::String stem = modelFile.getFileNameWithoutExtension();
  // Exact names first, since that is by far the common case and avoids
  // listing the directory; then one case-insensitive pass ("plexi.PNG").
  juce::Array<juce::File> candidates;
  for (const char* extension : kImageExtensions)
    if (const juce::File exact = directory.getChildFile(stem + extension); exact.existsAsFile())
      candidates.add(exact);
  if (candidates.isEmpty()) {
    const auto siblings = directory.findChildFiles(juce::File::findFiles, false);
    for (const char* extension : kImageExtensions)
      for (const auto& sibling : siblings)
        if (sibling.getFileName().equalsIgnoreCase(stem + extension))
          candidates.add(sibling);
  }

  for (const auto& candidate : candidates) {
    if (candidate.getSize() <= 0)
      continue;
    if (candidate.getSize() > kMaxSidecarArtworkBytes) {
      juce::Logger::writeToLog("[LocalLoad] Artwork too large, ignoring: " + candidate.getFileName());
      continue;
    }
    juce::MemoryBlock bytes;
    if (!candidate.loadFileAsData(bytes) || bytes.getSize() == 0)
      continue;
    if (const juce::String url = cacheArtworkBytes(bytes.getData(), bytes.getSize());
        url.isNotEmpty()) {
      juce::Logger::writeToLog("[LocalLoad] Artwork: " + candidate.getFileName());
      return url;
    }
  }
  return {};
}

// The first sidecar among a load's models, in load order; "" when none has one.
juce::String cacheSidecarArtwork(const juce::Array<juce::File>& modelFiles) {
  for (const auto& file : modelFiles)
    if (const juce::String url = cacheSidecarArtwork(file); url.isNotEmpty())
      return url;
  return {};
}

// One local file's bytes: validate and stash a content-addressed copy.
// Validation happens here, at load time, instead of letting a bad file
// reach the background loader: its failure surfaces as a retry badge, which
// is the wrong affordance for a file that can never load. Returns the model
// object { id, name, model_url[, gear] } for the synthetic tone, or void
// with `error` set to a user-facing message.
//
// `artworkUrl`, when given and still empty, receives the model's embedded
// artwork (see cacheEmbeddedArtwork), read from the JSON validation already
// parsed, so a folder load ends up with the first model's that has one.
juce::var stashLocalBytes(const juce::String& filename, juce::MemoryOutputStream& decoded,
                          juce::String& error, juce::String* artworkUrl = nullptr) {
  auto fail = [&](const juce::String& message) {
    juce::Logger::writeToLog("[LocalLoad] " + filename + ": " + message);
    error = message;
    return juce::var();
  };

  const juce::String extension = filename.fromLastOccurrenceOf(".", false, false).toLowerCase();
  const bool isNam = extension == "nam";
  if (!isNam && extension != "wav")
    return fail("Only .nam and .wav files are supported");

  // Inferred while the file is open for validation anyway (see
  // localGearFromNamMetadata / localGearFromIr); "" when unknown.
  juce::String gear;
  juce::String artwork;
  if (isNam) {
    try {
      const auto* bytes = static_cast<const char*>(decoded.getData());
      const nlohmann::json config = nlohmann::json::parse(bytes, bytes + decoded.getDataSize());
      if (!namConfigIsA2(config) && !namConfigIsParametric(config))
        return fail("Only A2 and parametric NAM files are supported");
      gear = localGearFromNamMetadata(config);
      if (artworkUrl != nullptr && artworkUrl->isEmpty())
        artwork = cacheEmbeddedArtwork(config);
    } catch (const std::exception&) {
      return fail("Not a valid NAM file");
    }
  } else {
    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(
        std::make_unique<juce::MemoryInputStream>(decoded.getData(), decoded.getDataSize(),
                                                  false)));
    if (reader == nullptr || reader->lengthInSamples <= 0)
      return fail("Not a valid WAV file");
    gear = localGearFromIr(*reader);
  }

  const juce::uint64 hash = fnv1a64(decoded.getData(), decoded.getDataSize());
  const juce::File stash = localModelsDir().getChildFile(
      juce::String::toHexString(static_cast<juce::int64>(hash)) + "-" +
      juce::String(static_cast<juce::int64>(decoded.getDataSize())) + "." + extension);
  if (!stash.existsAsFile()) {
    // The stash folder can exist without being writable (root-owned after a
    // sudo'd install script or a restored backup; github issue #76 saw every
    // drop fail here while validation kept passing). ensureWritableDir heals
    // what it can; a folder that stays unwritable gets its own message so
    // the report names the disease, not the symptom.
    if (!TONE3000Processor::ensureWritableDir(stash.getParentDirectory()))
      return fail("TONE3000's data folder isn't writable");
    if (!stash.replaceWithData(decoded.getData(), decoded.getDataSize()))
      return fail("Couldn't store the dropped file");
  } else {
    // Re-drop of known bytes: refresh the GC liveness stamp (see
    // cleanLocalModelStash).
    stash.setLastModificationTime(juce::Time::getCurrentTime());
  }

  // Positive, content-stable model id. It's only ever compared within its
  // own block (cache key, activeModelId), so collisions across different
  // files would be harmless anyway.
  juce::DynamicObject::Ptr model = new juce::DynamicObject();
  model->setProperty("id", static_cast<int>(hash % 0x7ffffffe) + 1);
  model->setProperty("name", filename.upToLastOccurrenceOf(".", false, false));
  model->setProperty("model_url", juce::URL(stash).toString(false));
  if (gear.isNotEmpty())
    model->setProperty("gear", gear);
  // Only once the model itself made it: a file that fails to stash can't
  // lend the tone its artwork.
  if (artwork.isNotEmpty())
    *artworkUrl = artwork;
  return juce::var(model.get());
}

// A file shipped as { name, data } with base64 bytes. The DSP tests feed
// files this way (it predates the on-disk path below, from when the UI
// could only hand over bytes); the UI itself always has a path.
juce::var stashLocalFile(const juce::String& filename, const juce::String& base64Data,
                         juce::String& error, juce::String* artworkUrl) {
  juce::MemoryOutputStream decoded;
  if (!juce::Base64::convertFromBase64(decoded, base64Data) || decoded.getDataSize() == 0) {
    juce::Logger::writeToLog("[LocalLoad] " + filename + ": Couldn't read the dropped file");
    error = "Couldn't read the dropped file";
    return {};
  }
  return stashLocalBytes(filename, decoded, error, artworkUrl);
}

// A file native already has on disk (the tile menus' file picker flow;
// no base64 round-trip).
juce::var stashLocalFileFromDisk(const juce::File& file, juce::String& error,
                                 juce::String* artworkUrl) {
  juce::MemoryOutputStream bytes;
  juce::FileInputStream in(file);
  if (!in.openedOk() || bytes.writeFromInputStream(in, -1) <= 0) {
    juce::Logger::writeToLog("[LocalLoad] " + file.getFullPathName() + ": Couldn't read the file");
    error = "Couldn't read the file";
    return {};
  }
  juce::var model = stashLocalBytes(file.getFileName(), bytes, error, artworkUrl);
  // Where the model came from, so a photo dropped on it later can be saved
  // into the file itself (embedLocalArtwork).
  if (auto* object = model.getDynamicObject())
    object->setProperty("source_path", file.getFullPathName());
  return model;
}

// A file the OS document picker handed us as a security-scoped URL.
//
// iOS is the reason this exists, and iOS is the only caller (see
// LocalFiles::pick); it is compiled everywhere so the DSP suite, which does
// not build for iOS, can exercise the same code the iPad runs. Everything the
// picker returns from the Files app lives outside the app sandbox (an iCloud /
// file-provider container), and
// the app is only allowed to touch it through the security scope JUCE's
// FileChooser opened and stored as a bookmark. Reading the raw path with a
// FileInputStream, which is what stashLocalFileFromDisk does and what every
// desktop build correctly does, is refused by the sandbox and surfaces as
// "Couldn't read the file". juce::URL::createInputStream goes through the
// bookmark and the scope, so it reads the same bytes the user actually picked.
juce::var stashLocalFileFromUrl(const juce::URL& url, juce::String& error,
                                juce::String* artworkUrl) {
  const juce::String filename = TONE3000Processor::localFileNameFromUrl(url);
  juce::MemoryOutputStream bytes;
  const auto in = url.createInputStream(
      juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress));
  if (in == nullptr || bytes.writeFromInputStream(*in, -1) <= 0) {
    juce::Logger::writeToLog("[LocalLoad] " + filename + ": Couldn't read the file");
    error = "Couldn't read the file";
    return {};
  }
  return stashLocalBytes(filename, bytes, error, artworkUrl);
}

// Writing artwork into a .nam (embedLocalArtwork). The file is edited as
// text, not re-serialized: a parse/dump round trip would reformat every
// weight (and nlohmann::json would reorder the keys), turning a one-key edit
// into a rewrite of someone's model file. So a minimal scanner finds the
// top-level "metadata" object and the "image" value inside it, and only
// that span changes. The result is then parsed and checked against the
// original before anything is written (sameApartFromImage).

constexpr size_t kNpos = std::string::npos;

size_t skipJsonSpace(const std::string& s, size_t i) {
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
    ++i;
  return i;
}

// Each skip returns the offset just past what it skipped, or kNpos.
size_t skipJsonString(const std::string& s, size_t i) {
  if (i >= s.size() || s[i] != '"')
    return kNpos;
  for (++i; i < s.size(); ++i) {
    if (s[i] == '\\')
      ++i;
    else if (s[i] == '"')
      return i + 1;
  }
  return kNpos;
}

size_t skipJsonValue(const std::string& s, size_t i) {
  if (i >= s.size())
    return kNpos;
  if (s[i] == '"')
    return skipJsonString(s, i);
  if (s[i] == '{' || s[i] == '[') {
    int depth = 0;
    while (i < s.size()) {
      const char c = s[i];
      if (c == '"') {
        i = skipJsonString(s, i);
        if (i == kNpos)
          return kNpos;
        continue;
      }
      if (c == '{' || c == '[')
        ++depth;
      else if ((c == '}' || c == ']') && --depth == 0)
        return i + 1;
      ++i;
    }
    return kNpos;
  }
  // Number, true, false or null: up to the next delimiter.
  const size_t start = i;
  while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' && s[i] != ' ' &&
         s[i] != '\t' && s[i] != '\n' && s[i] != '\r')
    ++i;
  return i > start ? i : kNpos;
}

struct JsonMember {
  std::string key;  // raw, between the quotes
  size_t valueStart = 0;
  size_t valueEnd = 0;
};

// The members of the object whose '{' is at `open`; nullopt when malformed.
std::optional<std::vector<JsonMember>> jsonMembers(const std::string& s, size_t open) {
  std::vector<JsonMember> members;
  size_t i = skipJsonSpace(s, open + 1);
  if (i < s.size() && s[i] == '}')
    return members;
  while (i < s.size()) {
    const size_t keyStart = skipJsonSpace(s, i);
    const size_t keyEnd = skipJsonString(s, keyStart);
    if (keyEnd == kNpos)
      return std::nullopt;
    i = skipJsonSpace(s, keyEnd);
    if (i >= s.size() || s[i] != ':')
      return std::nullopt;
    JsonMember member;
    member.key = s.substr(keyStart + 1, keyEnd - keyStart - 2);
    member.valueStart = skipJsonSpace(s, i + 1);
    member.valueEnd = skipJsonValue(s, member.valueStart);
    if (member.valueEnd == kNpos)
      return std::nullopt;
    members.push_back(member);
    i = skipJsonSpace(s, member.valueEnd);
    if (i < s.size() && s[i] == ',') {
      ++i;
      continue;
    }
    if (i < s.size() && s[i] == '}')
      return members;
    return std::nullopt;
  }
  return std::nullopt;
}

// `json` with metadata.image set to `imageValue` (JSON text): replaced when
// present, added otherwise, creating "metadata" when the file has none (or
// has null there). nullopt when the text isn't a JSON object.
std::optional<std::string> withMetadataImage(const std::string& json,
                                             const std::string& imageValue) {
  const size_t root = skipJsonSpace(json, 0);
  if (root >= json.size() || json[root] != '{')
    return std::nullopt;
  const auto members = jsonMembers(json, root);
  if (!members)
    return std::nullopt;

  for (const auto& member : *members) {
    if (member.key != "metadata")
      continue;
    if (json[member.valueStart] != '{')
      return json.substr(0, member.valueStart) + "{\"image\": " + imageValue + "}" +
             json.substr(member.valueEnd);
    const auto fields = jsonMembers(json, member.valueStart);
    if (!fields)
      return std::nullopt;
    for (const auto& field : *fields)
      if (field.key == "image")
        return json.substr(0, field.valueStart) + imageValue + json.substr(field.valueEnd);
    const std::string entry = "\"image\": " + imageValue + (fields->empty() ? "" : ", ");
    return json.substr(0, member.valueStart + 1) + entry + json.substr(member.valueStart + 1);
  }

  const std::string entry =
      "\"metadata\": {\"image\": " + imageValue + "}" + (members->empty() ? "" : ", ");
  return json.substr(0, root + 1) + entry + json.substr(root + 1);
}

// The edit's safety check: `after` carries exactly `dataUrl` as
// metadata.image and is otherwise the same model as `before` (a metadata
// object holding only an old image counts the same as none).
bool sameApartFromImage(nlohmann::json before, nlohmann::json after, const std::string& dataUrl) {
  if (!after.contains("metadata") || !after["metadata"].is_object() ||
      after["metadata"].value("image", std::string()) != dataUrl)
    return false;
  const auto detachMetadata = [](nlohmann::json& config) {
    nlohmann::json metadata = nlohmann::json::object();
    if (config.contains("metadata") && config["metadata"].is_object()) {
      metadata = config["metadata"];
      metadata.erase("image");
    }
    config.erase("metadata");
    return metadata;
  };
  return detachMetadata(before) == detachMetadata(after) && before == after;
}

// A dropped photo as NAMTRIX artwork: the centre square, at most
// kEmbeddedArtworkSide px, as a JPEG. Halving steps before the final resize
// keep a camera-sized photo from aliasing on the way down to 512.
constexpr int kEmbeddedArtworkSide = 512;

juce::MemoryBlock artworkJpegFrom(const juce::Image& photo) {
  const int side = std::min(photo.getWidth(), photo.getHeight());
  juce::Image current = photo.getClippedImage(
      {(photo.getWidth() - side) / 2, (photo.getHeight() - side) / 2, side, side});
  const int target = std::min(side, kEmbeddedArtworkSide);
  while (current.getWidth() / 2 >= target)
    current = current.rescaled(current.getWidth() / 2, current.getHeight() / 2,
                               juce::Graphics::highResamplingQuality);
  if (current.getWidth() != target)
    current = current.rescaled(target, target, juce::Graphics::highResamplingQuality);

  // JPEG has no alpha: a transparent PNG lands on black, like the tile behind it.
  juce::Image rgb(juce::Image::RGB, target, target, true);
  {
    juce::Graphics g(rgb);
    g.fillAll(juce::Colours::black);
    g.drawImageAt(current, 0, 0);
  }
  juce::MemoryOutputStream out;
  juce::JPEGImageFormat jpeg;
  jpeg.setQuality(0.85f);
  jpeg.writeImageToStream(rgb, out);
  return out.getMemoryBlock();
}

// Shared user-facing error result for the local-load entry points.
juce::var localToneError(const juce::String& title, const juce::String& message) {
  juce::Logger::writeToLog("[LocalLoad] " + title + ": " + message);
  juce::DynamicObject::Ptr err = new juce::DynamicObject();
  err->setProperty("error", message);
  return juce::var(err.get());
}

}  // namespace

float TONE3000Processor::computeIrNormalizationGain(const juce::File& irFile,
                                                    size_t maxIrFileSamples) {
  juce::AudioFormatManager formatManager;
  formatManager.registerBasicFormats();
  std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(irFile));

  if (!reader || reader->lengthInSamples <= 0 || reader->numChannels <= 0) {
    return 1.0f;
  }

  const juce::int64 totalSamples =
      std::min<juce::int64>(reader->lengthInSamples, static_cast<juce::int64>(maxIrFileSamples));
  juce::AudioBuffer<float> tmp(static_cast<int>(reader->numChannels),
                               static_cast<int>(totalSamples));
  reader->read(&tmp, 0, static_cast<int>(totalSamples), 0, true, true);

  double sumSquares = 0.0;
  juce::int64 count = 0;
  for (int ch = 0; ch < tmp.getNumChannels(); ++ch) {
    const float* data = tmp.getReadPointer(ch);
    for (int i = 0; i < tmp.getNumSamples(); ++i) {
      const float sample = data[i];
      sumSquares += static_cast<double>(sample) * static_cast<double>(sample);
      ++count;
    }
  }

  if (count <= 0) {
    return 1.0f;
  }

  // The convolver resamples the IR to the base rate (IR convolution always
  // runs there, see ChainBlock::irBaseRateIsland) and, with Normalise::no,
  // scales it by fileRate/baseRate to preserve the filter's magnitude
  // response; the net effect on kernel *energy* is one factor of
  // fileRate/baseRate. Fold that in so the gain matches what the engine
  // actually convolves regardless of the file's sample rate (a no-op for
  // 48 kHz files).
  const double rateScale =
      reader->sampleRate > 0.0 ? reader->sampleRate / kChainBaseSampleRate : 1.0;
  const double l2norm = std::sqrt(sumSquares * rateScale);
  if (!std::isfinite(l2norm) || l2norm <= 0.0) {
    return 1.0f;
  }

  // Attenuation-only unit-energy normalization: the wet path leaves the
  // block at roughly the dry level for cabs and reverbs alike. The 1.0
  // ceiling never boosts a quiet IR; the floor is a safety net against
  // absurdly hot files: -48 dB rather than -24 because a dense
  // multi-second reverb IR peaked at 0 dBFS legitimately carries 25-35 dB
  // more energy than a cab hit.
  const double linear = 1.0 / l2norm;
  const double linearClamped =
      juce::jlimit(juce::Decibels::decibelsToGain(-48.0), 1.0, linear);

  return static_cast<float>(linearClamped);
}

juce::var TONE3000Processor::loadLocalTone(const juce::String& title, const juce::var& files,
                                           const std::string& targetInsertId) {
  const auto* fileArray = files.getArray();
  if (fileArray == nullptr || fileArray->isEmpty())
    return localToneError(title, "Nothing to load");

  // A folder with some bad files still loads the good ones; only when
  // nothing survives does the first file's error surface (which for a
  // single-file drop is exactly that file's error).
  juce::Array<juce::var> models;
  juce::String firstError;
  juce::String artworkUrl;  // embedded only: bytes carry no folder to look beside
  for (const auto& file : *fileArray) {
    juce::String error;
    const juce::var model =
        stashLocalFile(file["name"].toString(), file["data"].toString(), error, &artworkUrl);
    if (!model.isObject()) {
      if (firstError.isEmpty())
        firstError = error;
      continue;
    }
    models.add(model);
  }

  return finishLocalToneLoad(title, models, firstError, fileArray->size(), targetInsertId,
                             artworkUrl);
}

juce::var TONE3000Processor::loadLocalTonePath(const juce::File& source,
                                               const std::string& targetInsertId) {
  if (source.isDirectory()) {
    // Folder rules mirror the UI's folder drop (useToneLoadFlow.ts): every
    // .nam/.wav under the folder (subfolders included), the majority
    // extension decides NAM vs IR, everything else is ignored. Extension
    // matching is by lowercased name, not findChildFiles wildcards, which
    // are case-sensitive on Linux.
    const juce::String title = source.getFileName();
    juce::Array<juce::File> nams, wavs;
    for (const auto& file : source.findChildFiles(juce::File::findFiles, true)) {
      const juce::String extension = file.getFileExtension().toLowerCase();
      if (extension == ".nam")
        nams.add(file);
      else if (extension == ".wav")
        wavs.add(file);
    }
    juce::Array<juce::File>& picked = nams.size() >= wavs.size() ? nams : wavs;

    if (picked.isEmpty())
      return localToneError(title, "No .nam or .wav files in the folder");
    if (picked.size() > kMaxFolderModels)
      return localToneError(title,
                            "Folder has too many files (max " + juce::String(kMaxFolderModels) +
                                ")");
    for (const auto& file : picked)
      if (file.getSize() > kMaxLocalFileBytes)
        return localToneError(title, "A file is too large");

    // Listing order is filesystem-dependent; natural name order keeps the
    // model list stable ("amp 2" before "amp 10"), like the UI's drop path.
    std::sort(picked.begin(), picked.end(), [](const juce::File& a, const juce::File& b) {
      return a.getFileName().compareNatural(b.getFileName()) < 0;
    });

    juce::Array<juce::var> models;
    juce::String firstError;
    juce::String artworkUrl;
    for (const auto& file : picked) {
      juce::String error;
      const juce::var model = stashLocalFileFromDisk(file, error, &artworkUrl);
      if (!model.isObject()) {
        if (firstError.isEmpty())
          firstError = error;
        continue;
      }
      models.add(model);
    }

    // Embedded artwork wins; otherwise the first model with an image beside it.
    if (artworkUrl.isEmpty() && !models.isEmpty())
      artworkUrl = cacheSidecarArtwork(picked);
    return finishLocalToneLoad(title, models, firstError, picked.size(), targetInsertId,
                               artworkUrl);
  }

  const juce::String title = source.getFileNameWithoutExtension();
  const juce::String extension = source.getFileExtension().toLowerCase();
  if (extension != ".nam" && extension != ".wav")
    return localToneError(title, "Only .nam and .wav files are supported");
  if (source.getSize() > kMaxLocalFileBytes)
    return localToneError(title, "File is too large");

  juce::String error;
  juce::String artworkUrl;
  const juce::var model = stashLocalFileFromDisk(source, error, &artworkUrl);
  if (!model.isObject())
    return localToneError(title, error);

  if (artworkUrl.isEmpty())
    artworkUrl = cacheSidecarArtwork(source);
  return finishLocalToneLoad(title, {model}, {}, 1, targetInsertId, artworkUrl);
}

juce::var TONE3000Processor::loadLocalToneUrls(const juce::Array<juce::URL>& sources,
                                               const std::string& targetInsertId) {
  // Multi-select stands in for the folder route on iOS: the document picker
  // can hand back a folder URL, but a security-scoped directory cannot be
  // enumerated through juce::URL (there is no listing API behind the
  // bookmark), so "Load Folder" asks for the files themselves instead. See
  // LocalFiles::pick. Not gated on JUCE_IOS so the DSP suite can run it;
  // the editor only reaches it on iOS.
  if (sources.isEmpty())
    return localToneError("Load Files", "Nothing to load");

  const juce::String title =
      sources.size() == 1
          ? localFileNameFromUrl(sources.getReference(0)).upToLastOccurrenceOf(".", false, false)
          : juce::String(sources.size()) + " files";

  if (sources.size() > kMaxFolderModels)
    return localToneError(
        title, "Too many files (max " + juce::String(kMaxFolderModels) + ")");

  // Natural name order, matching the folder and drop paths, so the model list
  // is stable regardless of the order the picker reports.
  juce::Array<juce::URL> picked(sources);
  std::sort(picked.begin(), picked.end(), [](const juce::URL& a, const juce::URL& b) {
    return localFileNameFromUrl(a).compareNatural(localFileNameFromUrl(b)) < 0;
  });

  juce::Array<juce::var> models;
  juce::String firstError;
  // No extension check here: stashLocalBytes rejects anything but .nam and
  // .wav by name before it looks at the bytes, with the same message.
  // Embedded artwork only: a security-scoped URL can't be listed for a
  // sidecar beside it (see stashLocalFileFromUrl).
  juce::String artworkUrl;
  for (const auto& url : picked) {
    juce::String error;
    const juce::var model = stashLocalFileFromUrl(url, error, &artworkUrl);
    if (!model.isObject()) {
      if (firstError.isEmpty())
        firstError = error;
      continue;
    }
    models.add(model);
  }

  if (models.isEmpty())
    return localToneError(title, firstError.isEmpty() ? "Couldn't read the file" : firstError);

  return finishLocalToneLoad(title, models, firstError, picked.size(), targetInsertId,
                             artworkUrl);
}

juce::var TONE3000Processor::finishLocalToneLoad(const juce::String& title,
                                                 const juce::Array<juce::var>& stashedModels,
                                                 const juce::String& firstError, int fileCount,
                                                 const std::string& targetInsertId,
                                                 const juce::String& imageUrl) {
  // Identical bytes under two names would collide on the content-derived
  // id (cache key, picker selection); the first name wins.
  juce::Array<juce::var> models;
  for (const auto& model : stashedModels) {
    const int modelId = model["id"];
    const bool duplicate =
        std::any_of(models.begin(), models.end(),
                    [modelId](const juce::var& m) { return static_cast<int>(m["id"]) == modelId; });
    if (!duplicate)
      models.add(model);
  }

  if (models.isEmpty())
    return localToneError(title, firstError.isNotEmpty() ? firstError : "No loadable files");

  // Every model in a local tone shares one format (folder loads keep only
  // the majority extension); the first stash URL names it.
  const bool isNam = models.getReference(0)["model_url"].toString().endsWithIgnoreCase(".nam");

  juce::DynamicObject::Ptr tone = new juce::DynamicObject();
  tone->setProperty("id", 0);
  tone->setProperty("local", true);
  tone->setProperty("title", title);
  tone->setProperty("format", isNam ? "nam" : "ir");
  tone->setProperty("models", models);
  // Artwork embedded in a model or found beside one (cacheEmbeddedArtwork,
  // cacheSidecarArtwork). Same shape as a catalog tone's images array, so it
  // reaches the tile and the detail card through the usual path.
  if (imageUrl.isNotEmpty())
    tone->setProperty("images", juce::Array<juce::var>{juce::var(imageUrl)});

  // Same catalog `gear` field a TONE3000 tone carries, inferred from the
  // first file (see stashLocalBytes); the tile draws that gear's glyph
  // instead of the generic file icon. Absent when nothing could be inferred.
  const juce::String gear = models.getReference(0)["gear"].toString();
  if (gear.isNotEmpty())
    tone->setProperty("gear", gear);

  const juce::String toneJson = juce::JSON::toString(juce::var(tone.get()));

  // A load targeting an existing tone tile replaces in place (same block id
  // + params). Insert-slot ids (and missing/stale ids) still go through
  // loadTone and consume/create a slot.
  if (!targetInsertId.empty() && swapTone(targetInsertId, toneJson)) {
    juce::Logger::writeToLog("[LocalLoad] Swapped '" + title + "' into block " +
                             juce::String(targetInsertId) + " (" + juce::String(models.size()) +
                             " of " + juce::String(fileCount) + " file(s))");
    juce::DynamicObject::Ptr ok = new juce::DynamicObject();
    ok->setProperty("blockId", juce::String(targetInsertId));
    return juce::var(ok.get());
  }

  const std::string blockId = loadTone(toneJson, targetInsertId);
  if (blockId.empty())
    return localToneError(title, "Couldn't add the block");

  juce::Logger::writeToLog("[LocalLoad] Loaded '" + title + "' into block " +
                           juce::String(blockId) + " (" + juce::String(models.size()) + " of " +
                           juce::String(fileCount) + " file(s))");
  juce::DynamicObject::Ptr ok = new juce::DynamicObject();
  ok->setProperty("blockId", juce::String(blockId));
  return juce::var(ok.get());
}

juce::var TONE3000Processor::embedLocalArtwork(const std::string& blockId,
                                               const juce::File& image) {
  const auto fail = [](const juce::String& message) {
    juce::Logger::writeToLog("[Artwork] " + message);
    juce::DynamicObject::Ptr result = new juce::DynamicObject();
    result->setProperty("error", message);
    return juce::var(result.get());
  };

  // 1. The block's active model: its original file, and the bytes it plays.
  juce::String sourcePath, modelName;
  int modelId = 0;
  std::vector<uint8_t> loadedBytes;
  {
    juce::ScopedLock lock(chainMutex);
    ChainBlock* block = findBlockById(blockId);
    if (block == nullptr || block->type == ChainBlockType::INSERT)
      return fail("Drop the photo on a model");
    if (!static_cast<bool>(block->toneVar["local"]))
      return fail("Artwork can only be added to models loaded from disk");
    if (block->toneVar["format"].toString() != "nam")
      return fail("Artwork can only be added to .nam models");
    modelId = block->activeModelId;
    if (const auto* models = block->toneVar["models"].getArray())
      for (const auto& model : *models)
        if (static_cast<int>(model["id"]) == modelId) {
          sourcePath = model["source_path"].toString();
          modelName = model["name"].toString();
        }
    if (const auto cached = block->modelCache.find(modelId); cached != block->modelCache.end())
      loadedBytes = cached->second;
  }
  if (loadedBytes.empty())
    return fail("Wait for the model to finish loading");
  // Chains from before source_path was recorded, and drops that arrived as
  // bytes, don't know where the file lives.
  if (sourcePath.isEmpty() || !juce::File::isAbsolutePath(sourcePath) ||
      !juce::File(sourcePath).existsAsFile())
    return fail("Can't find " + modelName +
                ".nam on disk. Load it from its file again, then drop the photo.");
  const juce::File source(sourcePath);
  if (!source.hasWriteAccess())
    return fail(source.getFileName() + " is read-only");

  // 2. The photo, as NAMTRIX artwork.
  const juce::String extension = image.getFileExtension().toLowerCase();
  if (extension != ".jpg" && extension != ".jpeg" && extension != ".png")
    return fail("Drop a .jpg or .png photo");
  if (image.getSize() > kMaxLocalFileBytes)
    return fail("That photo is too large");
  const juce::Image photo = juce::ImageFileFormat::loadFrom(image);
  if (!photo.isValid())
    return fail("Couldn't read that photo");
  const juce::MemoryBlock jpeg = artworkJpegFrom(photo);
  const std::string dataUrl =
      "data:image/jpeg;base64," +
      juce::Base64::toBase64(jpeg.getData(), jpeg.getSize()).toStdString();

  // 3. The edit, made only to the model the block is playing: a file
  // retrained or replaced since it was loaded is left alone.
  juce::MemoryBlock onDisk;
  if (!source.loadFileAsData(onDisk))
    return fail("Couldn't read " + source.getFileName());
  if (onDisk.getSize() != loadedBytes.size() ||
      std::memcmp(onDisk.getData(), loadedBytes.data(), loadedBytes.size()) != 0)
    return fail(source.getFileName() +
                " changed since it was loaded. Load it again, then drop the photo.");
  const std::string original(static_cast<const char*>(onDisk.getData()), onDisk.getSize());
  const auto edited = withMetadataImage(original, nlohmann::json(dataUrl).dump());
  try {
    if (!edited || !sameApartFromImage(nlohmann::json::parse(original),
                                       nlohmann::json::parse(*edited), dataUrl))
      return fail("Couldn't add the photo to " + source.getFileName());
  } catch (const std::exception&) {
    return fail("Couldn't add the photo to " + source.getFileName());
  }

  // 4. Written whole or not at all (replaceWithData goes through a temporary
  // file next to it and renames it over the original).
  if (!source.replaceWithData(edited->data(), edited->size()))
    return fail("Couldn't write to " + source.getFileName());
  juce::Logger::writeToLog("[Artwork] Saved artwork into " + source.getFullPathName());

  // 5. The block follows: its stash copy and cached bytes become the new file
  // (same weights, so the engine keeps playing untouched), which is what
  // presets and DAW state embed from now on, and the tile shows the photo.
  juce::MemoryOutputStream newBytes;
  newBytes.write(edited->data(), edited->size());
  juce::String stashError;
  const juce::var stashed = stashLocalBytes(source.getFileName(), newBytes, stashError);
  const juce::String artworkUrl = cacheArtworkBytes(jpeg.getData(), jpeg.getSize());
  {
    juce::ScopedLock lock(chainMutex);
    ChainBlock* block = findBlockById(blockId);
    if (block != nullptr && block->activeModelId == modelId &&
        static_cast<bool>(block->toneVar["local"])) {
      juce::var tone = juce::JSON::parse(block->toneJson);  // a copy: history shares toneVar
      if (auto* object = tone.getDynamicObject()) {
        if (stashed.isObject())
          if (const auto* models = object->getProperty("models").getArray())
            for (const auto& model : *models)
              if (auto* entry = model.getDynamicObject();
                  entry != nullptr && static_cast<int>(entry->getProperty("id")) == modelId)
                entry->setProperty("model_url", stashed["model_url"]);
        if (artworkUrl.isNotEmpty())
          object->setProperty("images", juce::Array<juce::var>{juce::var(artworkUrl)});
        setToneOnBlock(*block, block->toneId, juce::JSON::toString(tone), tone);
        if (stashed.isObject())
          block->modelCache[modelId] = std::vector<uint8_t>(edited->begin(), edited->end());
        bumpChainRevision();
      }
    }
  }

  juce::DynamicObject::Ptr result = new juce::DynamicObject();
  result->setProperty("savedTo", source.getFileName());
  return juce::var(result.get());
}

void TONE3000Processor::cleanLocalModelStash() {
  // The stash only has to outlive whatever references it *by path*: running
  // sessions' undo histories and not-yet-reloaded chains. Presets and DAW
  // state embed the bytes and re-heal the stash on load (see
  // refreshLocalStashCopy), and every use re-stamps the file's mtime, so a
  // week of no use means nothing alive points at it. Runs once per process,
  // off the constructor's thread (it does directory IO).
  static std::once_flag once;
  std::call_once(once, [] {
    juce::Thread::launch([] {
      const juce::Time cutoff = juce::Time::getCurrentTime() - juce::RelativeTime::days(7);
      for (const auto& file : localModelsDir().findChildFiles(juce::File::findFiles, false)) {
        if (file.getLastModificationTime() < cutoff && file.deleteFile())
          juce::Logger::writeToLog("[LocalLoad] Stash GC removed " + file.getFileName());
      }
    });
  });
}

void TONE3000Processor::cleanLeakedIrTempFiles() {
  // Runs once per process, off the constructor's thread (it does directory
  // IO), same pattern as cleanLocalModelStash.
  static std::once_flag once;
  std::call_once(once, [] {
    juce::Thread::launch([] {
      const int removed =
          sweepLeakedIrTempFiles(juce::File::getSpecialLocation(juce::File::tempDirectory));
      if (removed > 0)
        juce::Logger::writeToLog("[ModelLoader] Swept " + juce::String(removed) +
                                 " leaked IR temp file(s)");
    });
  });
}

bool TONE3000Processor::ensureWritableDir(const juce::File& dir) {
  if (dir.isDirectory() && dir.hasWriteAccess())
    return true;

  // A folder the user still owns with stripped write bits (restored backup)
  // is fixable in place with a chmod, which keeps its contents where they
  // are. chmod is owner-only, so this quietly does nothing to a root-owned
  // folder, which falls through to the rename below. The owner's write bit
  // only: juce::File::setReadOnly(false) would also grant group and world
  // (0555 -> 0777), and the folder holds the user's sign-in tokens.
  const auto restoreOwnerWrite = [&dir] {
#if JUCE_WINDOWS
    return dir.setReadOnly(false);
#else
    struct stat st {};
    const juce::String path = dir.getFullPathName();
    return ::stat(path.toRawUTF8(), &st) == 0 &&
           ::chmod(path.toRawUTF8(), (st.st_mode & 07777) | S_IWUSR) == 0;
#endif
  };
  if (dir.isDirectory() && restoreOwnerWrite() && dir.hasWriteAccess()) {
    juce::Logger::writeToLog("[AppData] Restored write permission on " + dir.getFullPathName());
    return true;
  }

  if (dir.exists()) {
    // What chmod can't fix: a file squatting on the path, or a directory
    // owned by someone else (root, after a sudo'd install script). chown-ing
    // it back needs privileges we don't have, but the parent belongs to the
    // user, so *renaming* the broken node aside works (though older macOS,
    // seen on the macos-14 CI runner, refuses to rename an unwritable
    // directory; current macOS allows it). Nothing is deleted: the sibling
    // keeps whatever is inside for manual recovery, and the log names it.
    const juce::File aside =
        dir.getSiblingFile(dir.getFileName() + ".unwritable").getNonexistentSibling();
    if (dir.moveFileTo(aside))
      juce::Logger::writeToLog("[AppData] Moved unusable " + dir.getFullPathName() +
                               " aside to " + aside.getFileName());
    else
      juce::Logger::writeToLog("[AppData] " + dir.getFullPathName() +
                               " is not writable and couldn't be moved aside");
  }

  const juce::Result created = dir.createDirectory();
  if (created.failed())
    juce::Logger::writeToLog("[AppData] Couldn't create " + dir.getFullPathName() + ": " +
                             created.getErrorMessage());
  return dir.isDirectory() && dir.hasWriteAccess();
}

int TONE3000Processor::sweepLeakedIrTempFiles(const juce::File& tempDir) {
  // Builds through v0.0.2 wrote one "<uuid>_ir.wav" per IR engine build into
  // the OS temp dir and never deleted it (github issue #25: hundreds of MB
  // after a session of preset browsing). The loader now scopes its temp file
  // to the engine build (see prepareBlockModelOffThread), so anything
  // matching the pattern is that legacy bloat or a crash leftover. Nothing
  // ever reads these files after the build, so deleting them is safe even
  // under a running older build; the hour of grace covers the only live
  // window, a concurrent instance mid-build, whose file is seconds old.
  const juce::Time cutoff = juce::Time::getCurrentTime() - juce::RelativeTime::hours(1);
  int removed = 0;
  for (const auto& file : tempDir.findChildFiles(juce::File::findFiles, false, "*_ir.wav"))
    if (file.getLastModificationTime() < cutoff && file.deleteFile())
      ++removed;
  return removed;
}

juce::String TONE3000Processor::localFileNameFromUrl(const juce::URL& url) {
  // A file:// URL knows its own local file, which carries the decoded name;
  // anything else only has the escaped path component to unescape.
  if (url.isLocalFile())
    return url.getLocalFile().getFileName();
  return juce::URL::removeEscapeChars(url.getFileName());
}

juce::File TONE3000Processor::resolveLocalModelFile(const juce::File& stashRoot,
                                                    const juce::String& modelUrl) {
  const juce::URL url(modelUrl);
  if (!url.isLocalFile())
    return {};

  const juce::File stored = url.getLocalFile();
  if (stored.existsAsFile())
    return stored;

  // The stored path is gone. On iOS that is the normal case after a
  // reinstall or an app update: the data container's UUID rotates, so every
  // absolute path a preset, an undo snapshot or the saved app state baked in
  // points at a container that no longer exists, and the block came back as
  // "Download failed / Retry". Stash names are content hashes in one flat
  // folder, so re-rooting the name is exact, not a guess. On desktop the
  // root never moves, so a path this machine wrote still exists and comes
  // back untouched above. The one desktop case this does change is a preset
  // or saved state carrying a stash URL from another machine or account:
  // that path is gone here too, and the block now re-roots it into the local
  // stash (and refreshLocalStashCopy writes the embedded bytes there) instead
  // of reading from the embedded cache alone. Same bytes, one more file on
  // disk. Not gated on JUCE_IOS because that case is a repair, not a
  // regression.
  const juce::String name = stored.getFileName();
  if (name.isEmpty() || name == "." || name == "..")
    return stored;
  return stashRoot.getChildFile(name);
}

void TONE3000Processor::refreshLocalStashCopy(const juce::String& modelUrl,
                                              const std::vector<uint8_t>& bytes) {
  const juce::File stash = resolveLocalModelFile(localModelsDir(), modelUrl);
  if (stash == juce::File() || !stash.isAChildOf(localModelsDir()))
    return;

  if (stash.existsAsFile()) {
    // In use, so keep the GC away (mtime is its liveness signal).
    stash.setLastModificationTime(juce::Time::getCurrentTime());
    return;
  }

  // The bytes came from an embedded cache but the stash copy is gone (GC'd,
  // or a preset from another machine): put it back so paths that need the
  // file (undo of a remove, retry) keep working.
  if (ensureWritableDir(stash.getParentDirectory()) &&
      stash.replaceWithData(bytes.data(), bytes.size()))
    juce::Logger::writeToLog("[LocalLoad] Restored stash copy " + stash.getFileName());
  else
    juce::Logger::writeToLog("[LocalLoad] Couldn't restore stash copy " + stash.getFileName());
}

std::vector<uint8_t> TONE3000Processor::fetchModelFromUrl(const juce::String& modelUrl) {
  juce::URL url(modelUrl);

  // Local-file models (drag-and-drop loads) resolve to their stash copy:
  // no network, no auth. See loadLocalTone.
  if (url.isLocalFile()) {
    // Through resolveLocalModelFile, not url.getLocalFile(): a path persisted
    // before an iOS reinstall names a container that is gone.
    const juce::File stash = resolveLocalModelFile(localModelsDir(), modelUrl);
    juce::MemoryBlock data;
    if (!stash.loadFileAsData(data) || data.getSize() == 0) {
      juce::Logger::writeToLog("[ModelLoader] Local model file missing or unreadable: " +
                               modelUrl);
      return {};
    }
    // In use, so keep the GC away (mtime is its liveness signal, see
    // cleanLocalModelStash).
    stash.setLastModificationTime(juce::Time::getCurrentTime());
    const auto* bytes = static_cast<const uint8_t*>(data.getData());
    return std::vector<uint8_t>(bytes, bytes + data.getSize());
  }

  // The TONE3000 API requires a Bearer token on `model_url` requests;
  // attach the latest token the UI handed us, if any. Anonymous fetches still
  // work for legacy public URLs, so we degrade gracefully when no token is set.
  // Chain the option builders since `InputStreamOptions` has no copy-assign.
  const juce::String token = getAccessToken();
  const juce::String extraHeaders =
      token.isNotEmpty() ? juce::String("Authorization: Bearer ") + token
                         : juce::String();
  if (token.isEmpty()) {
    // Happens when a restore-time load misses the embedded cache before the
    // UI has pushed the auth token; the API rejects anonymous model fetches.
    juce::Logger::writeToLog("[ModelLoader] Fetching model without auth token (may be rejected)");
  }

  auto options =
      juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
          .withConnectionTimeoutMs(30000)
          .withExtraHeaders(extraHeaders);

  std::unique_ptr<juce::InputStream> stream(url.createInputStream(options));

  if (!stream) {
    juce::Logger::writeToLog("[ModelLoader] Failed to open stream for model URL (network down or "
                             "unreachable): " + modelUrl);
    return {};
  }

  juce::MemoryBlock memoryBlock;
  const size_t blockSize = 8192;
  char buffer[blockSize];

  // The connection timeout above only covers the connect; a stalled response
  // body would otherwise pin a loader thread forever (an eternal "loading"
  // block in the UI). Bound the whole download instead.
  const juce::uint32 readDeadline = juce::Time::getMillisecondCounter() + 120000;

  while (!stream->isExhausted()) {
    if (juce::Time::getMillisecondCounter() > readDeadline) {
      juce::Logger::writeToLog("[ModelLoader] Model download timed out: " + modelUrl);
      return {};
    }
    int bytesRead = stream->read(buffer, blockSize);
    if (bytesRead > 0) {
      memoryBlock.append(buffer, bytesRead);
    } else {
      break;
    }
  }

  if (memoryBlock.getSize() == 0) {
    juce::Logger::writeToLog("[ModelLoader] Downloaded 0 bytes from model URL: " + modelUrl);
    return {};
  }

  DBG("Successfully downloaded " << memoryBlock.getSize() << " bytes");

  std::vector<uint8_t> result(memoryBlock.getSize());
  std::memcpy(result.data(), memoryBlock.getData(), memoryBlock.getSize());

  return result;
}


int TONE3000Processor::chainBaseBlockSize() const noexcept {
  // Before prepareToPlay has reported the real host config (state-restore
  // loads race it at launch), assume a conservatively large host block so
  // engines prepared early can absorb whatever the device settles on.
  // prepareChain / applyPreparedModelToChainBlock re-prepare on drift.
  const int knownHostBlock = juce::jmax(maxBlockSize, getBlockSize());
  const int hostBlockSize = knownHostBlock > 0 ? knownHostBlock : 4096;
  const double sr = hostSampleRate > 0.0 ? hostSampleRate : kChainBaseSampleRate;
  // Upsampling hosts below 48k (e.g. 44.1k) yield MORE frames per boundary
  // callback than the host block size; never go below the host size either,
  // since the direct path (48k host) hands host-sized blocks straight through.
  const int baseFrames =
      static_cast<int>(std::ceil(static_cast<double>(hostBlockSize) * kChainBaseSampleRate / sr));
  return juce::jmax(hostBlockSize, baseFrames);
}

int TONE3000Processor::chainDomainBlockSize() const noexcept {
  // The oversampler hands the chain stage exactly factor× the base frames.
  return chainBaseBlockSize() * chainOversampleFactor.load(std::memory_order_relaxed);
}

TONE3000Processor::PreparedBlockModel TONE3000Processor::prepareBlockModelOffThread(
    ChainBlockType type,
    const std::vector<uint8_t>& modelData,
    const juce::String& filename,
    double namSlimSize,
    const std::vector<float>& namParamValues,
    bool rederiveArtwork) {
  PreparedBlockModel out;

  if (modelData.empty()) {
    DBG("Cannot prepare empty model data");
    return out;
  }

  juce::Logger::writeToLog("[ModelLoader] Preparing " +
                           juce::String(type == ChainBlockType::NAM ? "NAM" : "IR") + " model: " +
                           filename + " (" + juce::String((juce::int64)modelData.size()) + " bytes)");

  const int domainBlockSize = chainDomainBlockSize();
  out.preparedBlockSize = domainBlockSize;

  try {
    if (type == ChainBlockType::NAM) {
      // Parse the .nam JSON directly from the downloaded bytes. Never round-trip
      // through a temp file here: nam::get_dsp(std::filesystem::path) built from
      // a JUCE UTF-8 string mis-decodes non-ASCII characters (model names, user
      // temp dirs) on Windows and the load fails silently.
      const nlohmann::json config =
          nlohmann::json::parse(modelData.begin(), modelData.end());

      // From the JSON this prepare parses anyway; never fails the load.
      if (rederiveArtwork)
        out.artworkUrl = cacheEmbeddedArtwork(config);

      // Oversampling: phase-safe architectures get one native-rate instance
      // per phase (see NamEngine.h); anything else runs a single instance
      // time-scaled at the full chain rate. Dual mono doubles the set: one
      // independent voice per channel (voice-major layout). Both counts are
      // read once here and baked into the engine; the apply path re-queues
      // the build if either requirement moved while it was in flight.
      const int oversampleFactor = chainOversampleFactor.load();
      const int voices = wantedNamVoices();
      const bool phaseSafe = oversampleFactor > 1 && namConfigIsPhaseSafe(config);
      const int phaseCount = phaseSafe ? oversampleFactor : 1;
      const int instanceCount = phaseCount * voices;
      if (oversampleFactor > 1 && !phaseSafe) {
        juce::Logger::writeToLog(
            "[ModelLoader] NAM architecture '" +
            juce::String(config.value("architecture", std::string("?"))) +
            "' can't be phase-interleaved; running time-scaled at ×" +
            juce::String(oversampleFactor));
      }

      std::vector<std::unique_ptr<nam::DSP>> instances;
      instances.reserve(static_cast<size_t>(instanceCount));
      for (int i = 0; i < instanceCount; ++i) {
        std::unique_ptr<nam::DSP> rawDsp = nam::get_dsp(config);
        if (!rawDsp) {
          juce::Logger::writeToLog("[ModelLoader] Failed to load NAM model: null DSP returned");
          return out;
        }
        if (rawDsp->NumInputChannels() != 1) {
          throw std::runtime_error("NAM model must have 1 input channel, but has " +
                                   std::to_string(rawDsp->NumInputChannels()));
        }
        if (rawDsp->NumOutputChannels() != 1) {
          throw std::runtime_error("NAM model must have 1 output channel, but has " +
                                   std::to_string(rawDsp->NumOutputChannels()));
        }
        instances.push_back(std::move(rawDsp));
      }

      auto engine = std::make_unique<NamEngine>(std::move(instances), oversampleFactor, voices);

      // The chain domain runs everything at the chain rate. A2 models are
      // all trained at 48k; anything else is rare enough that we just run it
      // anyway and note it in the log (a slight pitch/tone shift beats
      // per-block resampling machinery for a case that ~never happens).
      if (std::abs(engine->getModelSampleRate() - kChainBaseSampleRate) > 0.1) {
        juce::Logger::writeToLog(
            "[ModelLoader] NAM model reports " + juce::String(engine->getModelSampleRate()) +
            " Hz; the chain runs at " + juce::String(chainSampleRate()) + " Hz regardless");
      }

      // The block's A2 size (no-op for non-slimmable models); prepare()
      // applies it.
      engine->setSlimmableSize(namSlimSize);

      // Restored knob positions go on BEFORE prepare(): prepare() resets the
      // model, and IParametricControl specifies that a reset settles on the
      // committed vector instead of ramping into it. Committing after would
      // make every session restore audibly glide from the model's defaults to
      // the user's saved knobs. A length mismatch (preset saved against a
      // different model) is rejected by setParamValues, leaving the model's
      // own defaults -- the right fallback, and better than a wrong tone.
      if (!namParamValues.empty() && engine->isParametric()) {
        if (!engine->setParamValues(namParamValues)) {
          juce::Logger::writeToLog(
              "[ModelLoader] Stored knob values don't fit this model (" +
              juce::String(namParamValues.size()) + " vs " +
              juce::String(engine->getParamSpecs().size()) + "); using model defaults");
        }
      }

      engine->prepare(domainBlockSize);

      out.namEngine = std::move(engine);
      juce::Logger::writeToLog(
          "[ModelLoader] NAM model prepared, model sample rate: " +
          juce::String(out.namEngine->getModelSampleRate()) +
          (phaseSafe ? " (" + juce::String(phaseCount) + " phase instances)" : "") +
          (voices > 1 ? " (" + juce::String(voices) + " voices)" : ""));

      out.success = true;
    } else {
      // IRs go through the JUCE convolution/format-reader API, which wants a
      // file. The TemporaryFile's random leaf name (plus the right extension)
      // keeps the model's display name (which may contain characters that are
      // illegal in file names) out of the path, and its destructor deletes
      // the file on every exit path (success, early return, exception).
      // Nothing needs the file after this scope: loadImpulseResponse copies
      // the bytes into memory when prepare() drains the convolver's queue,
      // and later re-prepares rebuild from that copy, never from the file.
      // (Files leaked here by older builds are swept at startup, see
      // cleanLeakedIrTempFiles.)
      const juce::TemporaryFile scopedIrFile("_ir.wav");
      const juce::File& tempFile = scopedIrFile.getFile();

      if (!tempFile.replaceWithData(modelData.data(), modelData.size())) {
        juce::Logger::writeToLog("[ModelLoader] Failed to create temporary IR file: " +
                                 tempFile.getFullPathName());
        return out;
      }

      juce::AudioFormatManager formatManager;
      formatManager.registerBasicFormats();
      std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(tempFile));

      if (!reader) {
        juce::Logger::writeToLog("[ModelLoader] Failed to read IR file: " + filename);
        return out;
      }

      // The load cap is a time bound, not a fixed sample count; truncating
      // a multi-second reverb IR audibly chops its decay. It is passed to
      // JUCE in *file-rate* samples (truncation happens before the convolver
      // resamples to the base rate).
      const double fileSampleRate =
          reader->sampleRate > 0.0 ? reader->sampleRate : kChainBaseSampleRate;
      const auto maxIrFileSamples = static_cast<size_t>(kMaxIrSeconds * fileSampleRate);
      const juce::int64 fileSamplesToLoad = std::min<juce::int64>(
          reader->lengthInSamples, static_cast<juce::int64>(maxIrFileSamples));
      // Upper bound on the engine's kernel length at the base rate; the
      // authoritative (trimmed) length is read back off the built engine
      // below; a cab IR padded with trailing silence must still classify
      // as short for the level logic.
      const int irLengthUpperBound = static_cast<int>(std::llround(
          static_cast<double>(fileSamplesToLoad) * kChainBaseSampleRate / fileSampleRate));
      const int irNumChannels = juce::jlimit(1, 2, static_cast<int>(reader->numChannels));

      // Engine by the short/long cutoff. The engine must be constructed
      // before the trimmed size is known, so this one decision uses the
      // pre-trim upper bound; a silence-padded cab merely lands on the
      // non-uniform engine, a CPU choice, not an audible one. All audible
      // logic (output pad, default mix) uses the trimmed length below.
      const bool engineLongIr = irLengthUpperBound > kShortIrMaxBaseSamples;
      auto makeConvolver = [engineLongIr] {
        return engineLongIr ? std::make_unique<juce::dsp::Convolution>(
                                  juce::dsp::Convolution::NonUniform{kIrNonUniformHeadSamples})
                            : std::make_unique<juce::dsp::Convolution>();
      };

      // IR convolution always runs at the base rate: when the chain is
      // oversampled, the block's island (ChainBlock::irBaseRateIsland) hands
      // the convolver base-rate frames, and the RT path never passes more
      // than kIrConvolverMaxBlockSize of them per call. So the spec is
      // factor-independent: base rate, base block size capped at that.
      const int irBlockSize = irConvolverBlockSize();
      out.preparedIrBlockSize = irBlockSize;
      juce::dsp::ProcessSpec spec{kChainBaseSampleRate, static_cast<juce::uint32>(irBlockSize),
                                  2};

      // Load *before* prepare: prepare() drains the convolver's background
      // message queue synchronously, so the engine (FFT segmentation and
      // all) is fully built right here on the loader thread; the block can
      // never go live with its IR still initialising in the background.
      //
      // "Fully built" is not "fully installed", though: juce::dsp::Convolution
      // runs its own loader thread, and when that thread wins the race for
      // the pending-engine slot, prepare() finds it empty and the engine is
      // instead installed lazily on the *first process() call*, behind a
      // ~50 ms internal crossfade from dry (CrossoverMixer). Live, that dry
      // half-window is the un-cabbed signal bleeding through right as the
      // block's own fade-in ends. So after prepare(), run silence through
      // the convolver until that window has provably elapsed: the install
      // and its crossfade happen here on the loader thread, and the engine
      // goes live deterministically full-wet with silent state.
      auto elapseConvolverInstallFade = [&spec](juce::dsp::Convolution& convolver) {
        const int chunk = static_cast<int>(spec.maximumBlockSize);
        juce::AudioBuffer<float> silence(static_cast<int>(spec.numChannels), chunk);
        silence.clear();
        // 3× JUCE's 0.05 s install fade: margin over exactness, it's cheap.
        const int warmupSamples = static_cast<int>(kChainBaseSampleRate * 0.15);
        for (int done = 0; done < warmupSamples; done += chunk) {
          juce::dsp::AudioBlock<float> blockRef(silence);
          convolver.process(juce::dsp::ProcessContextReplacing<float>(blockRef));
        }
      };

      // Mono fallback convolver: IR channel 0 applied to every audio channel.
      auto convolverMono = makeConvolver();
      convolverMono->loadImpulseResponse(tempFile, juce::dsp::Convolution::Stereo::no,
                                         juce::dsp::Convolution::Trim::yes, maxIrFileSamples,
                                         juce::dsp::Convolution::Normalise::no);
      convolverMono->prepare(spec);
      elapseConvolverInstallFade(*convolverMono);
      out.convolverMono = std::move(convolverMono);

      // True-stereo convolver: only meaningful when the file actually has 2 channels.
      if (irNumChannels > 1) {
        auto convolverStereo = makeConvolver();
        convolverStereo->loadImpulseResponse(tempFile, juce::dsp::Convolution::Stereo::yes,
                                             juce::dsp::Convolution::Trim::yes, maxIrFileSamples,
                                             juce::dsp::Convolution::Normalise::no);
        convolverStereo->prepare(spec);
        elapseConvolverInstallFade(*convolverStereo);
        out.convolverStereo = std::move(convolverStereo);
      }

      // The engine was built synchronously above, so it can report the real
      // (trimmed + resampled) kernel length, the basis for the length verdict
      // and the host tail report. Fall back to the pre-trim bound
      // defensively. (The block's final cab/reverb classification also
      // weighs the tone's gear, which only the apply step knows; see
      // irIsLongFor.)
      const int engineIrSamples = out.convolverMono->getCurrentIRSize();
      const int irLengthBaseSamples = engineIrSamples > 0 ? engineIrSamples : irLengthUpperBound;

      out.irNumChannels = irNumChannels;
      out.irLengthBaseSamples = irLengthBaseSamples;
      out.irIsLongByLength = irLengthBaseSamples > kShortIrMaxBaseSamples;
      out.irNormalizationGainLinear = computeIrNormalizationGain(tempFile, maxIrFileSamples);

      juce::Logger::writeToLog(
          "[ModelLoader] IR prepared: " + juce::String(irNumChannels) + " ch, " +
          juce::String(irLengthBaseSamples / kChainBaseSampleRate, 2) + " s (" +
          (out.irIsLongByLength ? "long" : "short") + ", norm " +
          juce::String(juce::Decibels::gainToDecibels(out.irNormalizationGainLinear), 1) + " dB)");
      out.success = true;
    }
  } catch (const std::exception& e) {
    juce::Logger::writeToLog("[ModelLoader] Error preparing model '" + filename +
                             "': " + e.what());
    out.success = false;
    out.namEngine.reset();
    out.convolverMono.reset();
    out.convolverStereo.reset();
  }

  return out;
}

void TONE3000Processor::requestSwapFadeAndWait(const std::string& blockId, bool muteWetOnly) {
  // No callbacks running → nothing audible; the change can apply directly
  // (and a click gesture on the message thread shouldn't stall on the
  // timeout below).
  if (!isAudioActive())
    return;

  {
    juce::ScopedLock lock(chainMutex);
    ChainBlock* block = findBlockById(blockId);
    if (block == nullptr)
      return;
    // Nothing audible to fade: fresh block, failed/unloaded block, powered
    // off, or no engine yet. The change can splice in directly.
    const bool audible = block->loaded && block->enabled &&
                         (block->namEngine != nullptr || block->convolverMono != nullptr);
    if (!audible)
      return;
    block->swapMuteWet.store(muteWetOnly);
    block->swapFadeDone.store(false);
    block->swapFadePending.store(true);
  }

  // The audio thread completes the fade within a few callbacks; the deadline
  // is a backstop for audio stopping mid-wait. Every path past this point
  // clears swapFadePending, so a timed-out fade can't leave the block
  // bypassed.
  const juce::uint32 deadline = juce::Time::getMillisecondCounter() + 250;
  while (juce::Time::getMillisecondCounter() < deadline) {
    juce::Thread::sleep(2);
    juce::ScopedLock lock(chainMutex);
    ChainBlock* block = findBlockById(blockId);
    if (block == nullptr || block->swapFadeDone.load())
      return;
  }
}

void TONE3000Processor::requestChainEditFadeAndWait() {
  if (!isAudioActive()) {
    // No callbacks → nothing to fade; the edit applies directly. But still
    // arm the mute with the gain snapped to its landed value: a device can
    // start *mid-edit* (launch-time state restore racing the audio device
    // open), and callbacks arriving then must come up silent and take the
    // wait-free skip in processBlock, not blast the half-restored chain or
    // block behind the splice's lock hold. Touching the smoother here is
    // safe: no callbacks are running.
    chainEditFadeGain.setCurrentAndTargetValue(0.0f);
    chainEditFadeDone.store(true);
    chainEditFadePending.store(true);
    return;
  }

  chainEditFadeDone.store(false);
  chainEditFadePending.store(true);

  // The caller (ChainEditFade) clears chainEditFadePending after its edit,
  // so a timed-out fade can't leave the chain muted.
  const juce::uint32 deadline = juce::Time::getMillisecondCounter() + 250;
  while (juce::Time::getMillisecondCounter() < deadline) {
    juce::Thread::sleep(2);
    if (chainEditFadeDone.load())
      return;
  }
}

void TONE3000Processor::releaseChainEditFadeWhenLoadsSettle() {
  // The deadline is a *stall* detector, not a total budget: every completed
  // load pushes it out again (see the job below), so a heavy preset that
  // rebuilds many engines holds the mute as long as loads keep landing. Only
  // when nothing has finished for this long (wedged network fetch, starved
  // pool) does the chain glide back in early; any block still loading then
  // passes through until it lands, its own splice-in fade covering the entry.
  constexpr juce::uint32 kStallMs = 2000;
  chainEditFadeHoldDeadlineMs.store(juce::Time::getMillisecondCounter() + kStallMs);

  // Arm the mute even when requestChainEditFadeAndWait skipped it (audio
  // idle): if callbacks resume mid-reload the chain comes up muted instead
  // of blasting dry input, and the waiter below bounds the hold either way.
  chainEditFadeDone.store(false);
  chainEditFadePending.store(true);

  // Generation captured *after* the restore queued its loads (and after the
  // ChainEditFade ctor bumped it): a newer fade session invalidates this
  // waiter, whoever owns that session then owns the release.
  const int generation = chainEditFadeHoldGeneration.load();

  // Whether this restore queued any loads is decided here, synchronously:
  // a fast cached load could land before the job's first scan, and its
  // splice-in entry still needs the grace period below.
  bool loadsQueued = false;
  {
    juce::ScopedLock lock(chainMutex);
    for (const auto& l : lanes)
      for (const auto& b : l)
        if (b->modelLoading)
          loadsQueued = true;
  }

  struct ReleaseJob : public juce::ThreadPoolJob {
    TONE3000Processor& processor;
    const int generation;
    const bool loadsQueued;

    ReleaseJob(TONE3000Processor& p, int gen, bool loads)
        : ThreadPoolJob("Chain Edit Fade Release"), processor(p), generation(gen),
          loadsQueued(loads) {}

    JobStatus runJob() override {
      constexpr juce::uint32 kStallMs = 2000;
      int lastLoadingCount = std::numeric_limits<int>::max();
      for (;;) {
        if (shouldExit())
          return jobHasFinished;  // pool teardown; audio is gone anyway
        {
          // Decision under chainMutex: restores queue loads and flip
          // modelLoading under this lock, so the scan can't interleave a
          // half-applied restore.
          juce::ScopedLock lock(processor.chainMutex);
          if (processor.chainEditFadeHoldGeneration.load() != generation)
            return jobHasFinished;  // a newer fade session owns the release

          int loadingCount = 0;
          for (const auto& l : processor.lanes)
            for (const auto& b : l)
              if (b->modelLoading)
                ++loadingCount;

          // Progress restarts the stall clock: each landed load buys the
          // remaining ones another window, so the hold scales with the
          // preset instead of cutting off mid-rebuild.
          if (loadingCount < lastLoadingCount && loadingCount > 0)
            processor.chainEditFadeHoldDeadlineMs.store(juce::Time::getMillisecondCounter() +
                                                        kStallMs);
          lastLoadingCount = loadingCount;

          if (loadingCount == 0 || juce::Time::getMillisecondCounter() >=
                                       processor.chainEditFadeHoldDeadlineMs.load())
            break;
        }
        juce::Thread::sleep(20);
      }

      // A freshly applied block enters through its own ~25 ms splice-in fade
      // *from dry* (wetFadeGain, the continuity-preserving entry for live
      // loads). Let that finish under the mute, or a partial-dry sliver of
      // the last block's entry overlaps the glide-in. Skipped when this
      // restore queued nothing; a settings-only restore releases
      // immediately.
      if (loadsQueued)
        juce::Thread::sleep(60);

      juce::ScopedLock lock(processor.chainMutex);
      if (processor.chainEditFadeHoldGeneration.load() == generation)
        processor.chainEditFadePending.store(false);
      return jobHasFinished;
    }
  };

  loadingThreadPool.addJob(new ReleaseJob(*this, generation, loadsQueued), true);
}

void TONE3000Processor::applyPreparedModelToChainBlock(ChainBlock& block, ChainBlockType newType,
                                                       PreparedBlockModel& prepared) {
  // Loaded state is part of what getChainState reports, so any outcome here
  // must wake up the UI's revision-gated poll.
  bumpChainRevision();

  // Whatever the outcome, this load attempt is over. Clearing the fade flag
  // lets the audio thread ramp the wet mix back up when there is something
  // to hear. Remember the fade's shape first; it decides how the new
  // engine fades in below.
  const bool muteWetSwap = block.swapFadePending.load() && block.swapMuteWet.load();
  block.modelLoading = false;
  block.swapFadePending.store(false);
  block.swapFadeDone.store(false);
  block.swapMuteWet.store(false);

  if (!prepared.success) {
    // Corrupt/unreadable model: surface the retry UI. The UI already shows
    // the new tone/model, so the block drops out of processing to match
    // (the caller's pre-apply fade already glided it to bypass); its old
    // engines are swapped out by the next successful load.
    block.loaded = false;
    block.loadFailed = true;
    return;
  }

  // An oversampling or input-mode change can race an in-flight load: the
  // engine was built with the old factor's phase count or the old mode's
  // voice count, and neither can be re-prepared into the new one. Drop it
  // and re-queue; the rebuild reads the settled requirements and reuses the
  // block's in-memory model cache (no network).
  if (newType == ChainBlockType::NAM && prepared.namEngine != nullptr) {
    const int liveFactor = chainOversampleFactor.load();
    const int liveVoices = wantedNamVoices();
    if (prepared.namEngine->getOversampleFactor() != liveFactor ||
        prepared.namEngine->getVoiceCount() != liveVoices) {
      juce::Logger::writeToLog(
          "[ModelLoader] Chain requirements changed during prepare (×" +
          juce::String(prepared.namEngine->getOversampleFactor()) + " -> ×" +
          juce::String(liveFactor) + ", " + juce::String(prepared.namEngine->getVoiceCount()) +
          " -> " + juce::String(liveVoices) + " voices); re-queuing block " +
          juce::String(block.id));
      block.modelLoading = true;
      queueActiveModelLoad(block);
      return;
    }
  }

  // A restore-time prepare can race prepareToPlay: the engine may have been
  // sized off a stale (or defaulted) host config, and prepareChain can't have
  // covered it (the engine wasn't on the block yet). Re-prepare before it
  // goes live; feeding an engine more frames than it was prepared for is what
  // used to silently kill blocks on relaunch ("Processing failed").
  const int requiredBlockSize = chainDomainBlockSize();
  if (prepared.namEngine != nullptr && prepared.preparedBlockSize < requiredBlockSize) {
    juce::Logger::writeToLog("[ModelLoader] Chain domain drifted during prepare (block " +
                             juce::String(prepared.preparedBlockSize) + " -> " +
                             juce::String(requiredBlockSize) + "); re-preparing");
    prepared.namEngine->prepare(requiredBlockSize);
  }
  // Convolvers track the capped base block (irConvolverBlockSizeFor). Unlike
  // NAM, a *shrink* matters too: a load that ran before prepareToPlay saw the
  // 4096 default and built a 256-partition engine, which on a host that then
  // reports 64 would cost more per callback than it needs to. Re-prepare on
  // any difference so the convolver always matches the live host block.
  // (prepare() rebuilds the FFT engine under chainMutex, heavier for
  // reverb-length IRs, but this only fires on that startup race, when audio
  // has barely started.)
  const int requiredIrBlockSize = irConvolverBlockSize();
  if (prepared.convolverMono != nullptr && prepared.preparedIrBlockSize != requiredIrBlockSize) {
    juce::Logger::writeToLog("[ModelLoader] Host block drifted during IR prepare (convolver " +
                             juce::String(prepared.preparedIrBlockSize) + " -> " +
                             juce::String(requiredIrBlockSize) + "); re-preparing");
    const juce::dsp::ProcessSpec spec{kChainBaseSampleRate,
                                      static_cast<juce::uint32>(requiredIrBlockSize), 2};
    prepared.convolverMono->prepare(spec);  // keeps the loaded impulse
    if (prepared.convolverStereo != nullptr)
      prepared.convolverStereo->prepare(spec);
  }

  // Engines are *swapped*, not reset: the block's previous engines end up in
  // `prepared`, and the caller destroys them after releasing chainMutex;
  // NAM graph / convolution teardown must never run while the audio thread
  // can be blocked on the lock.
  if (newType == ChainBlockType::NAM && prepared.namEngine != nullptr) {
    block.type = newType;  // may differ while a tone swap was in flight
    std::swap(block.namEngine, prepared.namEngine);            // new engine in, old out
    std::swap(block.convolverMono, prepared.convolverMono);    // null in, any old IR out
    std::swap(block.convolverStereo, prepared.convolverStereo);
    block.irNumChannels = 1;
    block.irLengthBaseSamples = 0;
    block.irIsLong = false;

    // Re-assert the block's A2 size in case it changed while this engine
    // was downloading/preparing (a no-op retier when it didn't).
    block.namEngine->setSlimmableSize(block.namSlimSize);

    // Same for the knobs: the user can turn them while a swap is in flight.
    // This runs post-reset so it ramps rather than settles, which is right --
    // it is a live edit, not a restore. Sync the block's copy either way, so
    // a snapshot model clears stale values and a parametric one records the
    // defaults it actually started from.
    if (block.namEngine->isParametric() && !block.namParamValues.empty())
      block.namEngine->setParamValues(block.namParamValues);
    block.namParamValues = block.namEngine->isParametric()
                               ? block.namEngine->getParamValues()
                               : std::vector<float>{};

    block.namNormalizationSmoother.reset(chainSampleRate(), 0.05f);
    block.namNormalizationSmoother.setCurrentAndTargetValue(1.0f);
    block.loaded = true;
    block.loadFailed = false;

  } else if (newType == ChainBlockType::IR && prepared.convolverMono != nullptr) {
    block.type = newType;
    std::swap(block.namEngine, prepared.namEngine);            // null in, any old NAM out
    std::swap(block.convolverMono, prepared.convolverMono);
    std::swap(block.convolverStereo, prepared.convolverStereo);
    block.irNumChannels = prepared.irNumChannels;
    block.irLengthBaseSamples = prepared.irLengthBaseSamples;

    // Cab-like or reverb-like: the tone's gear where it is decisive, the
    // kernel length otherwise (see irIsLongFor). toneVar is already the
    // tone this engine belongs to; a swap sets it before queueing the load.
    const juce::String gear = block.toneVar["gear"].toString();
    block.irIsLong = irIsLongFor(gear, prepared.irIsLongByLength);
    if (block.irIsLong != prepared.irIsLongByLength)
      juce::Logger::writeToLog("[ModelLoader] IR classified " +
                               juce::String(block.irIsLong ? "long" : "short") + " by gear '" +
                               gear + "' (length said " +
                               (prepared.irIsLongByLength ? "long" : "short") + ")");

    // The base-rate island around the convolvers: blocks added mid-session
    // were never seen by prepareChain, so (re)prepare it here with the same
    // capacity/factor arguments as the chain-wide oversampler. Cheap (three
    // small work buffers), and the swap-fade already has the wet path silent.
    block.irBaseRateIsland.prepare(chainOversampleFactor.load(),
                                   juce::jmax(1, chainBaseBlockSize()));

    // Fresh blocks (Select-flow loads) default their mix by IR length: long
    // IRs are reverbs/effects meant to be blended (half wet), short cab IRs
    // replace the signal (fully wet). Length is only known here, after the
    // download, so loadTone arms this one-shot flag instead of guessing
    // from tone metadata. Swaps/restores keep the user's mix.
    if (block.applyDefaultMixOnLoad)
      block.mixNormalized = block.irIsLong ? 0.5f : 1.0f;

    block.irNormalizationGainLinear = prepared.irNormalizationGainLinear;
    block.irNormalizationSmoother.reset(chainSampleRate(), 0.05f);
    block.irNormalizationSmoother.setCurrentAndTargetValue(block.irNormalizationGainLinear);

    block.loaded = true;
    block.loadFailed = false;
  } else {
    DBG("Prepared model type/engine mismatch; dropping block from processing");
    block.loaded = false;
    block.loadFailed = true;
    return;
  }

  // The set of live IR engines changed; keep the host-facing tail length in
  // sync (caller holds chainMutex through this whole apply).
  refreshIrTailLength();

  // First successful load done; later loads on this block (model switches,
  // tone swaps) must keep the user's mix.
  block.applyDefaultMixOnLoad = false;

  // Per-block smoothers: prepareChain only covers blocks that existed at
  // prepareToPlay, so (re)arm them here for blocks added mid-session; a
  // never-reset LinearSmoothedValue jumps instantly (zipper noise on the
  // first knob drag). Snapping to target is inaudible: the wet path is at
  // bypass right now (fade below / fresh block).
  block.inputGainSmoother.reset(chainSampleRate(), 0.05f);
  block.outputGainSmoother.reset(chainSampleRate(), 0.05f);
  block.mixSmoother.reset(chainSampleRate(), 0.05f);
  block.mixSmoother.setCurrentAndTargetValue(block.mixNormalized);

  // Splice-in fade: the new engine enters from silence instead of jumping
  // in mid-waveform, mirroring how the outgoing one left (see
  // requestSwapFadeAndWait / ChainBlock.h). A wet-mute swap fades back
  // through the wet-only mute gain; the dry share of the user's mix held
  // steady the whole time, so the un-processed input is never exposed. A
  // bypass-shaped entry (fresh block, or the fade never engaged) crossfades
  // in from dry via wetFadeGain, exactly as before.
  block.swapWetMuteGain.reset(chainSampleRate(), kWetFadeSeconds);
  if (muteWetSwap) {
    // wetFadeGain stayed at 1 through the whole swap (the audio thread owns
    // it and it wasn't part of this fade); leave it alone so a concurrent
    // power-off glide keeps its course.
    block.swapWetMuteGain.setCurrentAndTargetValue(0.0f);
  } else {
    block.swapWetMuteGain.setCurrentAndTargetValue(1.0f);
    block.wetFadeGain.reset(chainSampleRate(), kWetFadeSeconds);
    block.wetFadeGain.setCurrentAndTargetValue(0.0f);
  }
}

