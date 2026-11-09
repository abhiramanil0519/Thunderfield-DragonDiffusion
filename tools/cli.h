#pragma once

#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dragonedge/arena.h"
#include "dragonedge/clock.h"
#include "dragonedge/log.h"
#include "dragonedge/manifest.h"
#include "dragonedge/memory_plan.h"
#include "dragonedge/png.h"
#include "dragonedge/qnn_backend.h"
#include "dragonedge/unet_runner.h"

namespace dragonedge::cli {

class Args {
 public:
  Args(int argc, char** argv, std::string usage) : usage_(std::move(usage)) {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "-h" || a == "--help") {
        std::fputs(usage_.c_str(), stdout);
        std::exit(0);
      }
      if (a.rfind("--", 0) != 0) fail("unexpected argument " + a);
      std::string key = a.substr(2);
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) values_[key] = argv[++i];
      else values_[key] = "1";
    }
  }

  std::string str(const std::string& key, const std::string& def = "") const {
    used_[key] = true;
    auto it = values_.find(key);
    return it == values_.end() ? def : it->second;
  }
  double num(const std::string& key, double def) const {
    std::string s = str(key);
    return s.empty() ? def : std::atof(s.c_str());
  }
  bool flag(const std::string& key) const { return !str(key).empty(); }
  void finish() const {
    for (const auto& [k, v] : values_)
      if (!used_.count(k)) fail("unknown option --" + k);
  }

 private:
  [[noreturn]] void fail(const std::string& msg) const {
    std::fprintf(stderr, "error: %s\n\n%s", msg.c_str(), usage_.c_str());
    std::exit(2);
  }
  std::string usage_;
  std::map<std::string, std::string> values_;
  mutable std::map<std::string, bool> used_;
};

constexpr const char* kCommonUsage = R"(common:
  --manifest PATH      manifest.json (default: manifest.json)
  --contexts DIR       directory holding <piece>.bin (default: the manifest's directory)
  --log PATH           also append the log to PATH
  --verbose            debug-level log (per-tensor encodings, QNN debug output)
  --qnn-backend-lib P  (default: libQnnHtp.so)     --qnn-system-lib P (default: libQnnSystem.so)
  --qnn-log-level N    QNN's own log level 1..5 (default: 2 = warnings)
  --no-burst           leave the NPU in its default power mode
  --no-shared-spill-fill   give every context its own spill-fill buffer
)";

inline double mb(uint64_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); }

inline std::vector<float> read_f32(const std::string& path, size_t expect) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<float> v(expect);
  size_t n = std::fread(v.data(), 4, expect, f);
  std::fclose(f);
  if (n != expect) throw std::runtime_error(path + ": expected " + std::to_string(expect) + " float32 values");
  return v;
}

struct Runtime {
  Manifest manifest;
  ArenaLayout layout;
  std::unique_ptr<RpcMemAllocator> allocator;
  std::unique_ptr<Arena> arena;
  std::unique_ptr<QnnBackend> backend;
};

inline void setup_logging(const Args& args) {
  if (args.flag("verbose")) log_set_level(LogLevel::Debug);
  std::string path = args.str("log");
  if (!path.empty()) log_set_file(path);
}

inline QnnOptions qnn_options(const Args& args) {
  QnnOptions o;
  o.backend_lib = args.str("qnn-backend-lib", o.backend_lib);
  o.system_lib = args.str("qnn-system-lib", o.system_lib);
  o.share_spill_fill = !args.flag("no-shared-spill-fill");
  o.max_spill_fill_override = static_cast<uint64_t>(args.num("spill-fill-bytes", 0));
  o.burst = !args.flag("no-burst");
  o.log_level = static_cast<int>(args.num("qnn-log-level", 2));
  return o;
}

inline std::unique_ptr<Runtime> open_runtime(const Args& args) {
  auto rt = std::make_unique<Runtime>();
  const std::string path = args.str("manifest", "manifest.json");
  rt->manifest = Manifest::load(path, args.str("contexts"));
  DE_INFO("engine", "manifest %s: %d px, %zu core shards", path.c_str(), rt->manifest.resolution,
          rt->manifest.stage(Stage::Core).size());
  rt->layout = plan_unet_arena(rt->manifest);
  DE_INFO("engine", "arena plan: %.2f MB (%.2f MB live at peak, %.2f MB without reuse)",
          mb(rt->layout.plan.total_bytes), mb(rt->layout.plan.peak_live_bytes), mb(rt->layout.plan.unshared_bytes));
  rt->allocator = std::make_unique<RpcMemAllocator>();
  rt->arena = std::make_unique<Arena>(rt->layout.tensors, rt->layout.plan, *rt->allocator);
  rt->backend = std::make_unique<QnnBackend>(*rt->arena, rt->manifest, qnn_options(args));
  return rt;
}

inline void decode_to_png(const std::string& decoder_path, const std::vector<float>& latent, int latent_hw,
                          const std::string& png_path, const QnnOptions& opts) {
  Manifest dm;
  dm.path = decoder_path;
  TensorSpec in{"latent", {1, 4, latent_hw, latent_hw}, 2, "external"};
  TensorSpec out{"image", {1, 3, latent_hw * 8, latent_hw * 8}, 2, "output"};
  dm.tensors = {in, out};
  PieceSpec piece;
  piece.name = "taesdxl_decoder";
  piece.stage = Stage::Core;
  piece.context_path = decoder_path;
  piece.inputs = {"latent"};
  piece.outputs = {"image"};
  dm.pieces = {piece};
  MemoryPlan plan = plan_memory({{"latent", in.bytes(), 0, 1}, {"image", out.bytes(), 0, 1}}, 4096);
  RpcMemAllocator alloc;
  Arena arena(dm.tensors, plan, alloc);
  QnnOptions o = opts;
  o.share_spill_fill = false;
  QnnBackend backend(arena, dm, o);
  auto p = backend.load(dm.pieces[0]);
  DE_INFO("decode", "decoder loaded in %.1f ms (read %.1f, create %.1f, register %.1f)", p->timing.total_ms,
          p->timing.read_ms, p->timing.create_ms, p->timing.register_ms);
  arena.write_f32("latent", latent.data(), latent.size());
  const double t0 = now_ms();
  backend.execute(*p);
  const double exec_ms = now_ms() - t0;
  const int hw = latent_hw * 8;
  std::vector<float> img(out.numel());
  arena.read_f32("image", img.data(), img.size());
  backend.unload(std::move(p));
  std::vector<uint8_t> rgb(static_cast<size_t>(hw) * hw * 3);
  const size_t plane = static_cast<size_t>(hw) * hw;
  for (size_t i = 0; i < plane; ++i)
    for (int c = 0; c < 3; ++c) {
      float v = img[c * plane + i];
      v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
      rgb[i * 3 + c] = static_cast<uint8_t>(v * 255.f + 0.5f);
    }
  write_png_rgb8(png_path, hw, hw, rgb.data());
  DE_INFO("decode", "TAESDXL decode on the NPU: %.1f ms -> %s (%dx%d)", exec_ms, png_path.c_str(), hw, hw);
}

}
