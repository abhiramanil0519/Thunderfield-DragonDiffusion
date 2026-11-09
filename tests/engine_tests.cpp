#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "dragonedge/arena.h"
#include "dragonedge/clock.h"
#include "dragonedge/json.h"
#include "dragonedge/manifest.h"
#include "dragonedge/memory_plan.h"
#include "mock_backend.h"
#include "dragonedge/sampler.h"
#include "dragonedge/streamer.h"
#include "dragonedge/unet_runner.h"

using namespace dragonedge;

namespace {

struct TestCase {
  const char* name;
  std::function<void()> fn;
};

std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}

struct Register {
  Register(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};

struct Failure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

#define TEST(name)                                   \
  static void name();                                \
  static Register register_##name(#name, name);      \
  static void name()

#define CHECK(cond)                                                                                  \
  do {                                                                                               \
    if (!(cond)) throw Failure(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #cond); \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                                              \
  do {                                                                                                     \
    double a_ = (a), b_ = (b);                                                                             \
    if (std::fabs(a_ - b_) > (tol))                                                                        \
      throw Failure(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #a " = " +              \
                    std::to_string(a_) + ", expected " + std::to_string(b_));                              \
  } while (0)

#define CHECK_THROWS(expr)            \
  do {                                \
    bool threw_ = false;              \
    try {                             \
      expr;                           \
    } catch (const Failure&) {        \
      throw;                          \
    } catch (const std::exception&) { \
      threw_ = true;                  \
    }                                 \
    CHECK(threw_ && "expected throw"); \
  } while (0)

const std::string kManifest = DRAGON_TEST_MANIFEST;

}

TEST(json_parses_documents) {
  auto v = json::Value::parse(R"({"a": [1, -2.5e1, true, null], "b": {"c": "x\"y\u00e9\n"}, "d": 18446744})");
  CHECK(v.at("a").as_array().size() == 4);
  CHECK(v.at("a").as_array()[0].as_int() == 1);
  CHECK_NEAR(v.at("a").as_array()[1].as_number(), -25.0, 0);
  CHECK(v.at("a").as_array()[2].as_bool());
  CHECK(v.at("a").as_array()[3].is_null());
  CHECK(v.at("b").at("c").as_string() == "x\"y\xC3\xA9\n");
  CHECK(v.at("d").as_uint() == 18446744u);
  CHECK(v.find("zz") == nullptr);
  CHECK_THROWS(json::Value::parse("{\"a\": }"));
  CHECK_THROWS(json::Value::parse("[1, 2"));
  CHECK_THROWS(json::Value::parse("{} x"));
  CHECK_THROWS(v.at("b").as_array());
  CHECK(json::quote("a\"b\n") == "\"a\\\"b\\n\"");
}

TEST(fp16_conversion) {
  CHECK(f32_to_f16(1.0f) == 0x3C00);
  CHECK(f32_to_f16(-2.0f) == 0xC000);
  CHECK(f32_to_f16(65504.f) == 0x7BFF);
  CHECK(f32_to_f16(65520.f) == 0x7C00);
  CHECK(f32_to_f16(0.1f) == 0x2E66);
  CHECK(f32_to_f16(5.9604645e-8f) == 0x0001);
  CHECK(f32_to_f16(2.98e-8f) == 0x0000);
  CHECK(f32_to_f16(6.1035156e-5f) == 0x0400);
  CHECK(f32_to_f16(1.0f + 1.0f / 2048) == 0x3C00);
  CHECK(f32_to_f16(1.0f + 3.0f / 2048) == 0x3C02);
  for (uint32_t h = 0; h < 0x10000; ++h) {
    const bool nan = (h & 0x7C00) == 0x7C00 && (h & 0x3FF);
    if (nan) continue;
    CHECK(f32_to_f16(f16_to_f32(static_cast<uint16_t>(h))) == h);
  }
}

TEST(fixed_point_encoding) {
  Encoding e{DType::UFixed16, 0.001f, -30000};
  float in[3] = {0.f, 1.5f, -29.0f};
  uint16_t q[3];
  float out[3];
  encode_f32(in, q, 3, e);
  decode_f32(q, out, 3, e);
  CHECK(q[0] == 30000);
  CHECK_NEAR(out[1], 1.5, 0.001);
  CHECK_NEAR(out[2], -29.0, 0.001);
  float big = 1e9f;
  encode_f32(&big, q, 1, e);
  CHECK(q[0] == 65535);
}

TEST(memory_plan_reuses_dead_tensors) {
  std::vector<LiveRange> r = {{"a", 1000, 0, 1}, {"b", 1000, 2, 3}, {"c", 500, 1, 2}};
  MemoryPlan p = plan_memory(r, 256);
  check_plan(p);
  CHECK(p.find("a")->offset == p.find("b")->offset);
  CHECK(p.total_bytes == 1024 + 512);
  CHECK(p.peak_live_bytes == 1024 + 512);
  CHECK(p.unshared_bytes == 1024 * 2 + 512);
  CHECK_THROWS(plan_memory({{"x", 1, 0, 0}, {"x", 1, 1, 1}}, 64));
  CHECK_THROWS(plan_memory({{"x", 1, 2, 1}}, 64));
  CHECK_THROWS(plan_memory({{"x", 1, 0, 0}}, 100));
}

TEST(memory_plan_random_stress) {
  std::mt19937 rng(7);
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<LiveRange> r;
    int n = 1 + static_cast<int>(rng() % 40);
    for (int i = 0; i < n; ++i) {
      int a = static_cast<int>(rng() % 30), b = a + static_cast<int>(rng() % 10);
      r.push_back({"t" + std::to_string(i), 1 + rng() % 100000, a, b});
    }
    MemoryPlan p = plan_memory(r, 64);
    check_plan(p);
    CHECK(p.total_bytes >= p.peak_live_bytes);
    CHECK(p.total_bytes <= p.unshared_bytes);
  }
  MemoryPlan bad = plan_memory({{"a", 100, 0, 1}, {"b", 100, 1, 2}}, 64);
  bad.placements[1].offset = bad.placements[0].offset;
  CHECK_THROWS(check_plan(bad));
}

TEST(euler_trailing_matches_sdxl_lightning) {
  EulerSchedule s = euler_trailing(4);
  CHECK(s.timesteps == (std::vector<float>{999, 749, 499, 249}));
  CHECK(s.sigmas.size() == 5);
  CHECK_NEAR(s.sigmas[0], 14.614641229333646, 1e-4);
  CHECK_NEAR(s.sigmas[1], 4.081729363724309, 1e-5);
  CHECK_NEAR(s.sigmas[2], 1.612886194303876, 1e-5);
  CHECK_NEAR(s.sigmas[3], 0.6932048996386885, 1e-6);
  CHECK(s.sigmas[4] == 0.f);
  CHECK_NEAR(s.init_noise_sigma, s.sigmas[0], 0);
}

TEST(timestep_embedding_matches_diffusers) {
  TimeProjSpec spec{320, true, 0.f};
  std::vector<float> e(320);
  timestep_embedding(999.f, spec, e.data());
  CHECK_NEAR(e[0], 0.9996498529808264, 2e-4);
  CHECK_NEAR(e[160], -0.026460752737064126, 2e-4);
  CHECK_NEAR(e[1], 0.8026775155873899, 2e-4);
  CHECK_NEAR(e[161], 0.596413284535527, 2e-4);
  CHECK_NEAR(e[80], -0.8444698188786249, 2e-4);
  CHECK_NEAR(e[240], -0.5356031413305027, 2e-4);
  CHECK_NEAR(e[159], 0.9944063455342584, 2e-4);
  CHECK_NEAR(e[319], 0.10562206190565039, 2e-4);
  std::vector<float> pooled(1280, 0.5f);
  const float ids[6] = {1024, 1024, 0, 0, 1024, 1024};
  std::vector<float> add = sdxl_add_proj(pooled, ids, TimeProjSpec{256, true, 0.f});
  CHECK(add.size() == 2816);
  CHECK(add[0] == 0.5f);
  CHECK_NEAR(add[1280 + 2 * 256], 1.0, 0);
}

TEST(manifest_loads_and_validates) {
  Manifest m = Manifest::load(kManifest);
  CHECK(m.resolution == 1024 && m.latent == 128);
  CHECK(m.pieces.front().name == "shell_down" && m.pieces.back().name == "shell_up");
  CHECK(m.stage(Stage::Core).size() >= 2);
  CHECK(m.tensor("core_out").shape == (std::vector<int64_t>{1, 1280, 64, 64}));
  CHECK(m.tensor("ehs").shape == (std::vector<int64_t>{1, 77, 2048}));
  CHECK(m.time_proj.dim == 320 && m.add_time_proj.dim == 256 && m.pooled_dim == 1280);
  uint64_t skip_bytes = 0;
  for (const auto& t : m.tensors)
    if (t.role == "skip") skip_bytes += t.bytes();
  CHECK(skip_bytes > 45u << 20 && skip_bytes < 55u << 20);
}

TEST(unet_arena_plan_is_valid_and_compact) {
  Manifest m = Manifest::load(kManifest);
  ArenaLayout l = plan_unet_arena(m);
  check_plan(l.plan);
  CHECK(l.plan.total_bytes < l.plan.unshared_bytes);
  CHECK(l.plan.total_bytes <= l.plan.peak_live_bytes + (8u << 20));
  CHECK(l.plan.total_bytes < 100u << 20);
  const Placement* core_out = l.plan.find("core_out");
  CHECK(core_out && core_out->first == 0 && core_out->last == static_cast<int>(m.pieces.size()) + 1);
  std::printf("    arena %.1f MB planned, %.1f MB live at peak, %.1f MB without reuse\n",
              l.plan.total_bytes / 1048576.0, l.plan.peak_live_bytes / 1048576.0, l.plan.unshared_bytes / 1048576.0);
}

TEST(streamer_bounds_residency) {
  Manifest m = Manifest::load(kManifest);
  ArenaLayout l = plan_unet_arena(m);
  HostAllocator alloc;
  Arena arena(l.tensors, l.plan, alloc);
  for (int slots = 1; slots <= 3; ++slots) {
    MockOptions o;
    o.hold_weights = false;
    o.exec_ms = 2;
    MockBackend backend(arena, o);
    CoreStreamer s(backend, m.stage(Stage::Core), slots);
    CorePassStats st = s.run_pass();
    CHECK(st.max_slots_used == slots);
    CHECK(backend.max_resident_core() == slots);
    CHECK(backend.resident() == 0);
    CHECK(st.shards.size() == m.stage(Stage::Core).size());
    for (size_t k = 0; k + 1 < st.shards.size(); ++k) CHECK(st.shards[k].exec_end <= st.shards[k + 1].exec_begin);
    if (slots >= 2) {
      bool overlapped = false;
      for (size_t k = 0; k + 1 < st.shards.size(); ++k)
        overlapped |= st.shards[k + 1].load_begin < st.shards[k].exec_end;
      CHECK(overlapped);
    }
  }
}

TEST(streamer_hides_first_load_behind_overlap) {
  Manifest m = Manifest::load(kManifest);
  ArenaLayout l = plan_unet_arena(m);
  HostAllocator alloc;
  Arena arena(l.tensors, l.plan, alloc);
  MockOptions o;
  o.hold_weights = false;
  o.load_mb_per_s = 20000;
  MockBackend backend(arena, o);
  CoreStreamer s(backend, m.stage(Stage::Core), 2);
  double overlap_end = 0;
  CorePassStats st = s.run_pass([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    overlap_end = now_ms();
  });
  CHECK(st.shards[0].load_end <= overlap_end);
  CHECK(st.shards[0].stall_ms < 2.0);
  CHECK(backend.max_resident_core() <= 2);
}

namespace {

class FailingBackend : public Backend {
 public:
  FailingBackend(Arena& a, std::string fail_on) : inner_(a, MockOptions{0, 0, false, false}), fail_on_(fail_on) {}
  std::string name() const override { return "failing"; }
  std::unique_ptr<LoadedPiece> load(const PieceSpec& s) override {
    if (s.name == fail_on_) throw std::runtime_error("injected load failure");
    return inner_.load(s);
  }
  void execute(LoadedPiece& p) override { inner_.execute(p); }
  void unload(std::unique_ptr<LoadedPiece> p) override { inner_.unload(std::move(p)); }
  int resident() const { return inner_.resident(); }

 private:
  MockBackend inner_;
  std::string fail_on_;
};

}

namespace {

class TightMemoryBackend : public Backend {
 public:
  explicit TightMemoryBackend(Arena& a) : inner_(a, MockOptions{0, 1, false, false}) {}
  std::string name() const override { return "tight"; }
  std::unique_ptr<LoadedPiece> load(const PieceSpec& s) override {
    if (s.stage == Stage::Core && inner_.resident() > 0) {
      ++refused_;
      throw std::runtime_error("QNN_COMMON_ERROR_MEM_ALLOC (simulated: only one shard fits)");
    }
    return inner_.load(s);
  }
  void execute(LoadedPiece& p) override { inner_.execute(p); }
  void unload(std::unique_ptr<LoadedPiece> p) override { inner_.unload(std::move(p)); }
  int refused() const { return refused_; }
  int max_resident_core() const { return inner_.max_resident_core(); }

 private:
  MockBackend inner_;
  std::atomic<int> refused_{0};
};

}

TEST(streamer_downgrades_to_one_slot_when_memory_is_short) {
  Manifest m = Manifest::load(kManifest);
  ArenaLayout l = plan_unet_arena(m);
  HostAllocator alloc;
  Arena arena(l.tensors, l.plan, alloc);
  TightMemoryBackend backend(arena);
  CoreStreamer s(backend, m.stage(Stage::Core), 2);
  CorePassStats st = s.run_pass();
  CHECK(st.shards.size() == m.stage(Stage::Core).size());
  CHECK(s.slots() == 1);
  CHECK(backend.refused() == 1);
  CHECK(backend.max_resident_core() == 1);
  CorePassStats again = s.run_pass();
  CHECK(again.max_slots_used == 1);
  CHECK(backend.refused() == 1);
}

TEST(streamer_propagates_load_failure) {
  Manifest m = Manifest::load(kManifest);
  ArenaLayout l = plan_unet_arena(m);
  HostAllocator alloc;
  Arena arena(l.tensors, l.plan, alloc);
  auto core = m.stage(Stage::Core);
  FailingBackend backend(arena, core[core.size() / 2]->name);
  CoreStreamer s(backend, core, 2);
  CHECK_THROWS(s.run_pass());
  CHECK(backend.resident() == 0);
}

TEST(denoise_matches_unshared_reference) {
  Manifest m = Manifest::load(kManifest);
  ArenaLayout l = plan_unet_arena(m);
  HostAllocator alloc;
  Arena arena(l.tensors, l.plan, alloc);
  MockOptions o;
  o.hold_weights = false;
  MockBackend backend(arena, o);
  UNetRunner runner(m, backend, 2);
  EulerSchedule sched = euler_trailing(4);
  Conditioning cond = zero_conditioning(m);
  for (size_t i = 0; i < cond.ehs.size(); ++i) cond.ehs[i] = std::sin(0.001f * i);
  for (size_t i = 0; i < cond.pooled.size(); ++i) cond.pooled[i] = std::cos(0.01f * i);
  const size_t n = m.tensor("sample").numel();
  std::vector<float> noise(n);
  fill_normal(noise.data(), n, 42, 1.f);
  DenoiseResult got = denoise(runner, arena, m, sched, cond, noise);
  runner.unload_shell();

  std::map<std::string, std::vector<uint8_t>> buf;
  for (const auto& t : m.tensors) buf[t.name].assign(t.bytes(), 0);
  auto put = [&](const std::string& name, const std::vector<float>& v) {
    encode_f32(v.data(), buf[name].data(), v.size(), Encoding{DType::F16});
  };
  put("ehs", cond.ehs);
  put("add_proj", sdxl_add_proj(cond.pooled, cond.time_ids, m.add_time_proj));
  std::vector<float> latent(n), scratch(n), tproj(m.time_proj.dim);
  for (size_t i = 0; i < n; ++i) latent[i] = noise[i] * sched.init_noise_sigma;
  for (size_t s = 0; s < sched.timesteps.size(); ++s) {
    scale_model_input(latent.data(), scratch.data(), n, sched.sigmas[s]);
    put("sample", scratch);
    timestep_embedding(sched.timesteps[s], m.time_proj, tproj.data());
    put("t_proj", tproj);
    for (const auto& p : m.pieces)
      MockBackend::compute(
          p,
          [&](const std::string& name, uint64_t* bytes) -> const uint8_t* {
            *bytes = buf[name].size();
            return buf[name].data();
          },
          [&](const std::string& name, uint64_t* bytes) -> uint8_t* {
            *bytes = buf[name].size();
            return buf[name].data();
          });
    decode_f32(buf["eps"].data(), scratch.data(), n, Encoding{DType::F16});
    euler_step(latent.data(), scratch.data(), n, sched.sigmas[s], sched.sigmas[s + 1]);
  }
  CHECK(got.latent.size() == n);
  CHECK(std::memcmp(got.latent.data(), latent.data(), n * 4) == 0);
  CHECK(got.steps.size() == 4);
  for (const auto& st : got.steps) CHECK(st.core.max_slots_used == 2);
}

int main() {
  int failed = 0;
  for (const auto& t : registry()) {
    try {
      t.fn();
      std::printf("[ ok ] %s\n", t.name);
    } catch (const std::exception& e) {
      ++failed;
      std::printf("[FAIL] %s\n    %s\n", t.name, e.what());
    }
  }
  std::printf("%zu tests, %d failed\n", registry().size(), failed);
  return failed ? 1 : 0;
}
