#include "mock_backend.h"

#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#include "dragonedge/clock.h"
#include "dragonedge/file_io.h"
#include "dragonedge/mem_accounting.h"

namespace dragonedge {

namespace {

class MockPiece : public LoadedPiece {
 public:
  using LoadedPiece::LoadedPiece;
  std::vector<uint8_t> weights;
};

uint64_t fnv1a(const void* data, uint64_t n, uint64_t h = 1469598103934665603ull) {
  const auto* p = static_cast<const uint8_t*>(data);
  for (uint64_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

uint64_t splitmix(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

void sleep_ms(double ms) {
  if (ms > 0) std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int64_t>(ms * 1000)));
}

}

MockBackend::MockBackend(Arena& arena, MockOptions opts) : arena_(arena), opts_(opts) {}

void MockBackend::compute(const PieceSpec& spec, const InputFn& in, const OutputFn& out) {
  uint64_t h = fnv1a(spec.name.data(), spec.name.size());
  for (const auto& name : spec.inputs) {
    uint64_t n = 0;
    const uint8_t* p = in(name, &n);
    h = fnv1a(name.data(), name.size(), h);
    h = fnv1a(p, n, h);
  }
  for (const auto& name : spec.outputs) {
    uint64_t n = 0;
    uint8_t* p = out(name, &n);
    uint64_t s = fnv1a(name.data(), name.size(), h);
    auto* v = reinterpret_cast<uint16_t*>(p);
    for (uint64_t i = 0; i < n / 2; ++i) {
      float x = static_cast<float>(splitmix(s) >> 40) * (2.f / 16777216.f) - 1.f;
      v[i] = f32_to_f16(x);
    }
  }
}

std::unique_ptr<LoadedPiece> MockBackend::load(const PieceSpec& spec) {
  const double t0 = now_ms();
  auto piece = std::make_unique<MockPiece>(spec);
  uint64_t bytes = spec.est_weight_bytes;
  if (opts_.read_context_files && file_exists(spec.context_path)) {
    MappedFile f(spec.context_path);
    f.touch();
    bytes = f.size();
  }
  piece->timing.file_bytes = bytes;
  const double t1 = now_ms();
  if (opts_.hold_weights) {
    piece->weights.resize(bytes);
    for (uint64_t off = 0; off < bytes; off += 4096) piece->weights[off] = 1;
    npu_tracked_bytes() += static_cast<int64_t>(bytes);
  }
  if (opts_.load_mb_per_s > 0) sleep_ms(bytes / (opts_.load_mb_per_s * 1048576.0) * 1000.0 - (now_ms() - t1));
  for (const auto& n : spec.inputs) arena_.bind_encoding(n, Encoding{DType::F16}, spec.name);
  for (const auto& n : spec.outputs) arena_.bind_encoding(n, Encoding{DType::F16}, spec.name);

  ++resident_;
  if (spec.stage == Stage::Core) {
    int now = ++resident_core_;
    int prev = max_resident_core_.load();
    while (now > prev && !max_resident_core_.compare_exchange_weak(prev, now)) {
    }
  }
  piece->timing.read_ms = t1 - t0;
  piece->timing.create_ms = now_ms() - t1;
  piece->timing.total_ms = now_ms() - t0;
  return piece;
}

void MockBackend::execute(LoadedPiece& piece) {
  compute(
      *piece.spec,
      [&](const std::string& n, uint64_t* bytes) -> const uint8_t* {
        const Slot& s = arena_.slot(n);
        *bytes = s.bytes;
        return s.data;
      },
      [&](const std::string& n, uint64_t* bytes) -> uint8_t* {
        Slot& s = arena_.slot(n);
        *bytes = s.bytes;
        return s.data;
      });
  sleep_ms(opts_.exec_ms);
}

void MockBackend::unload(std::unique_ptr<LoadedPiece> piece) {
  auto* p = static_cast<MockPiece*>(piece.get());
  if (opts_.hold_weights) npu_tracked_bytes() -= static_cast<int64_t>(p->weights.size());
  if (p->spec->stage == Stage::Core) --resident_core_;
  --resident_;
  piece.reset();
}

}
