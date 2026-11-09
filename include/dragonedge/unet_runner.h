#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "dragonedge/arena.h"
#include "dragonedge/backend.h"
#include "dragonedge/manifest.h"
#include "dragonedge/memory_plan.h"
#include "dragonedge/sampler.h"
#include "dragonedge/streamer.h"

namespace dragonedge {

class MemAccounting;

constexpr const char* kLatent = "latent";

struct ArenaLayout {
  std::vector<TensorSpec> tensors;
  std::vector<LiveRange> ranges;
  MemoryPlan plan;
};

ArenaLayout plan_unet_arena(const Manifest& m, uint64_t alignment = 4096);

struct StepStats {
  int index = 0;
  double begin_ms = 0;
  double end_ms = 0;
  double shell_down_ms = 0;
  double core_ms = 0;
  double shell_up_ms = 0;
  CorePassStats core;
};

class UNetRunner {
 public:
  UNetRunner(const Manifest& m, Backend& backend, int slots, MemAccounting* mem = nullptr);
  ~UNetRunner();

  void load_shell();
  void unload_shell();
  StepStats run_full_step(int index);

  const LoadTiming& shell_down_timing() const;
  const LoadTiming& shell_up_timing() const;

 private:
  const Manifest& m_;
  Backend& backend_;
  MemAccounting* mem_;
  CoreStreamer streamer_;
  std::unique_ptr<LoadedPiece> shell_down_;
  std::unique_ptr<LoadedPiece> shell_up_;
};

struct Conditioning {
  std::vector<float> ehs;
  std::vector<float> pooled;
  float time_ids[6] = {0, 0, 0, 0, 0, 0};
};

Conditioning zero_conditioning(const Manifest& m);

struct DenoiseResult {
  std::vector<float> latent;
  std::vector<StepStats> steps;
};

DenoiseResult denoise(UNetRunner& runner, Arena& arena, const Manifest& m, const EulerSchedule& sched,
                      const Conditioning& cond, const std::vector<float>& initial_noise);

}
