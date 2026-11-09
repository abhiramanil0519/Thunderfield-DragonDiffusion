#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

#include "dragonedge/arena.h"
#include "dragonedge/backend.h"

namespace dragonedge {

struct MockOptions {
  double load_mb_per_s = 0;
  double exec_ms = 0;
  bool hold_weights = true;
  bool read_context_files = false;
};

class MockBackend : public Backend {
 public:
  MockBackend(Arena& arena, MockOptions opts);

  std::string name() const override { return "mock"; }
  std::unique_ptr<LoadedPiece> load(const PieceSpec& spec) override;
  void execute(LoadedPiece& piece) override;
  void unload(std::unique_ptr<LoadedPiece> piece) override;

  int resident() const { return resident_; }
  int max_resident_core() const { return max_resident_core_; }

  using InputFn = std::function<const uint8_t*(const std::string&, uint64_t*)>;
  using OutputFn = std::function<uint8_t*(const std::string&, uint64_t*)>;
  static void compute(const PieceSpec& spec, const InputFn& in, const OutputFn& out);

 private:
  Arena& arena_;
  MockOptions opts_;
  std::atomic<int> resident_{0};
  std::atomic<int> resident_core_{0};
  std::atomic<int> max_resident_core_{0};
};

}
