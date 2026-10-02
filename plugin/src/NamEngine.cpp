#include "NamEngine.h"
#include "NAM/slimmable.h"
#include "RtWorkerPool.h"
#include <atomic>
#include <stdexcept>

NamEngine::NamEngine(std::vector<std::unique_ptr<nam::DSP>> modelInstances, int factor,
                     int voiceCount)
    : instances(std::move(modelInstances)), oversampleFactor(factor), voices(voiceCount) {
  if (instances.empty())
    throw std::invalid_argument("NamEngine needs at least one model instance");
  for (const auto& instance : instances) {
    if (instance == nullptr)
      throw std::invalid_argument("NAM model cannot be null");
  }
  if (factor < 1)
    throw std::invalid_argument("NamEngine oversampling factor must be >= 1");
  if (voices < 1 || voices > kMaxVoices)
    throw std::invalid_argument("NamEngine voice count must be 1 or 2");
  const auto count = static_cast<int>(instances.size());
  if (count % voices != 0)
    throw std::invalid_argument("NamEngine instance count must be a multiple of the voice count");
  phases = count / voices;
  if (phases != 1 && phases != factor)
    throw std::invalid_argument("NamEngine phases per voice must be 1 or the oversampling factor");
  // The flat voice × phase fork publishes one job per instance.
  if (count > RtWorkerPool::kMaxJobs)
    throw std::invalid_argument("NamEngine instance count exceeds the worker pool's job limit");

  // Resolve the parametric interface once: dynamic_cast allocates nothing but
  // walks the RTTI graph, which is not something to do per audio block. Every
  // phase is the same architecture, so either all of them are parametric or
  // none are; a partial result would leave phases out of lockstep, so treat it
  // as non-parametric rather than driving only some of them.
  {
    std::vector<nam::IParametricControl*> controls;
    controls.reserve(instances.size());
    for (const auto& instance : instances) {
      if (auto* pc = dynamic_cast<nam::IParametricControl*>(instance.get()))
        controls.push_back(pc);
    }
    if (controls.size() == instances.size() && !controls.empty()) {
      paramControls = std::move(controls);
      paramSpecs = paramControls.front()->GetParamSpecs();
      const auto committed = paramControls.front()->GetParams();
      paramValues.assign(committed.begin(), committed.end());
    }
  }

  // Informational only; the chain domain runs the model at the chain rate
  // regardless (the loader logs a warning on mismatch).
  try {
    modelSampleRate = primary().GetExpectedSampleRate();
  } catch (...) {
    modelSampleRate = kChainBaseSampleRate;
  }
}

void NamEngine::prepare(int newMaxBlockSize) {
  maxBlockSize = newMaxBlockSize;

  const auto slotCount = instances.size();
  // +1: a carried phase offset can hand one phase an extra frame when a
  // defensive slice isn't divisible by the phase count.
  const size_t perPhaseCapacity = static_cast<size_t>(maxBlockSize) / static_cast<size_t>(phases) + 1;
  slotInputs.assign(slotCount, std::vector<double>(perPhaseCapacity));
  slotOutputs.assign(slotCount, std::vector<double>(perPhaseCapacity));
  phaseFrames.assign(static_cast<size_t>(phases), 0);
  phaseOffset = 0;

  for (auto& instance : instances) {
    instance->ResetAndPrewarm(instanceSampleRate(), static_cast<int>(perPhaseCapacity));

    // SlimmableContainer / SlimmableWavenet: after max buffer size is set on the
    // root DSP, SetSlimmableSize re-selects the active tier and ResetAndPrewarms
    // it (see NAM render tools and NeuralAmpModelerCore tests). Without this,
    // container models can stay on a stale or uninitialized sub-path.
    if (auto* slimmable = dynamic_cast<nam::SlimmableModel*>(instance.get()))
      slimmable->SetSlimmableSize(requestedSlimmableSize);
  }

  isPrepared = true;
}

void NamEngine::setSlimmableSize(double val) {
  requestedSlimmableSize = juce::jlimit(0.0, 1.0, val);
  if (!isPrepared)
    return;
  for (auto& instance : instances) {
    if (auto* slimmable = dynamic_cast<nam::SlimmableModel*>(instance.get()))
      slimmable->SetSlimmableSize(requestedSlimmableSize);
  }
}

bool NamEngine::setParamValues(const std::vector<float>& values) {
  if (paramControls.empty())
    return false;
  if (values.size() != paramSpecs.size())
    return false;

  // Clamp to each spec's range, and round switches: the conditioner rejects a
  // non-integral or out-of-range switch index by throwing, and this runs from
  // the message thread where a throw would take down the editor. A UI slider
  // landing on 1.9999 is the expected case, not a defensive one.
  std::vector<float> sanitized(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    const auto& spec = paramSpecs[i];
    float v = juce::jlimit(spec.min, spec.max, values[i]);
    if (spec.type == "switch") {
      v = std::round(v);
      // num_inputs() is the enum width; the highest valid index is one below.
      v = juce::jlimit(0.0f, static_cast<float>(spec.num_inputs() - 1), v);
    }
    sanitized[i] = v;
  }

  for (auto* control : paramControls) {
    try {
      control->SetParams(sanitized);
    } catch (const std::exception& e) {
      juce::Logger::writeToLog(juce::String("NamEngine::setParamValues rejected: ") + e.what());
      return false;
    }
  }

  paramValues = std::move(sanitized);
  return true;
}

void NamEngine::process(juce::AudioBuffer<float>& buffer, RtWorkerPool* pool, bool dualMono) {
  if (!isPrepared || maxBlockSize < 1) {
    throw std::runtime_error("NamEngine must be prepared before processing");
  }

  const int numSamples = buffer.getNumSamples();
  const int numChannels = buffer.getNumChannels();

  // Which channels get their own voice this call: channel 0 always; channel
  // 1 only in dual mono, on a two-voice engine, with a stereo buffer. Every
  // other combination is the classic mono path (voice 0, then fan out), so a
  // single-voice engine is unaffected by the flag and a two-voice engine
  // never processes a mirrored channel twice for nothing.
  const int activeVoices = (dualMono && voices > 1 && numChannels > 1) ? 2 : 1;
  const int jobCount = activeVoices * phases;

  // NAM models are mono: each active voice processes its own channel.
  // Buffers larger than the prepared size are run in prepared-size slices
  // (models stream statefully, so slicing is exact); throwing here would
  // permanently disable the block when a startup prepare races the host's
  // actual block size.
  for (int offset = 0; offset < numSamples; offset += maxBlockSize) {
    const int chunk = juce::jmin(maxBlockSize, numSamples - offset);

    // Deinterleave each active channel into its voice's per-phase streams
    // (single phase = one stream = a straight copy). This doubles as the
    // float → double conversion. Both voices see the same phase walk, so
    // phaseFrames is computed once from voice 0.
    std::fill(phaseFrames.begin(), phaseFrames.end(), 0);
    for (int v = 0; v < activeVoices; ++v) {
      const float* channel = buffer.getReadPointer(v) + offset;
      int phase = phaseOffset;
      int frames[RtWorkerPool::kMaxJobs] = {};
      for (int i = 0; i < chunk; ++i) {
        slotInputs[static_cast<size_t>(slotIndex(v, phase))][static_cast<size_t>(frames[phase]++)] =
            static_cast<double>(channel[i]);
        if (++phase == phases)
          phase = 0;
      }
      if (v == 0)
        for (int p = 0; p < phases; ++p)
          phaseFrames[static_cast<size_t>(p)] = frames[p];
    }

    if (pool != nullptr && jobCount > 1) {
      // Fork every (voice, phase) job across the pool (see the header
      // comment): each job touches only its own instance and buffers, so
      // any schedule produces the same bits as the serial loop below.
      // Exceptions can't cross threads, so a job traps its own and the
      // joiner rethrows once all jobs are done; the caller's RT failure
      // handling (disable the block, log off-thread) works exactly as for
      // a serial throw.
      struct SlotJob {
        nam::DSP* model;
        NAM_SAMPLE* input;
        NAM_SAMPLE* output;
        int frames;
        std::atomic<bool>* failed;
      };
      std::atomic<bool> jobFailed{false};
      jassert(jobCount <= RtWorkerPool::kMaxJobs);  // enforced by the constructor
      SlotJob jobs[RtWorkerPool::kMaxJobs];
      void* ctxs[RtWorkerPool::kMaxJobs];
      for (int v = 0; v < activeVoices; ++v) {
        for (int p = 0; p < phases; ++p) {
          const int job = slotIndex(v, p);
          const auto slot = static_cast<size_t>(job);
          jobs[job] = {instances[slot].get(), slotInputs[slot].data(), slotOutputs[slot].data(),
                       phaseFrames[static_cast<size_t>(p)], &jobFailed};
          ctxs[job] = &jobs[job];
        }
      }
      pool->forkJoin(
          [](void* ctx) {
            auto& j = *static_cast<SlotJob*>(ctx);
            if (j.frames <= 0)
              return;
            try {
              NAM_SAMPLE* inputPtrs[] = {j.input};
              NAM_SAMPLE* outputPtrs[] = {j.output};
              j.model->process(inputPtrs, outputPtrs, j.frames);
            } catch (...) {
              j.failed->store(true, std::memory_order_release);
            }
          },
          ctxs, jobCount);
      if (jobFailed.load(std::memory_order_acquire))
        throw std::runtime_error("NAM phase processing failed");
    } else {
      for (int v = 0; v < activeVoices; ++v) {
        for (int p = 0; p < phases; ++p) {
          const int frames = phaseFrames[static_cast<size_t>(p)];
          if (frames <= 0)
            continue;
          const auto slot = static_cast<size_t>(slotIndex(v, p));
          NAM_SAMPLE* inputPtrs[] = {slotInputs[slot].data()};
          NAM_SAMPLE* outputPtrs[] = {slotOutputs[slot].data()};
          instances[slot]->process(inputPtrs, outputPtrs, frames);
        }
      }
    }

    // Reinterleave each voice back onto its channel (and convert to float).
    for (int v = 0; v < activeVoices; ++v) {
      float* channel = buffer.getWritePointer(v) + offset;
      int phase = phaseOffset;
      int frames[RtWorkerPool::kMaxJobs] = {};
      for (int i = 0; i < chunk; ++i) {
        channel[i] = static_cast<float>(
            slotOutputs[static_cast<size_t>(slotIndex(v, phase))][static_cast<size_t>(frames[phase]++)]);
        if (++phase == phases)
          phase = 0;
      }
    }

    phaseOffset = (phaseOffset + chunk) % phases;
  }

  // Mono path: fan voice 0's result out to the other channels. In dual mono
  // channel 1 already carries voice 1's output and is left alone.
  for (int ch = activeVoices; ch < numChannels; ++ch) {
    buffer.copyFrom(ch, 0, buffer, 0, 0, numSamples);
  }
}
