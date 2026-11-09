#include <algorithm>
#include <cstdio>
#include <exception>
#include <sstream>

#include "cli.h"
#include "dragonedge/clock.h"
#include "dragonedge/file_io.h"
#include "dragonedge/json.h"
#include "dragonedge/mem_accounting.h"
#include "dragonedge/streamer.h"
#include "dragonedge/unet_runner.h"

using namespace dragonedge;

static const char* kUsage = R"(shard_bench: per-shard load / execute / free cost on the NPU.

  --pieces LIST        comma-separated names, or all | core | shell (default: all)
  --reps N             executes per piece (default: 3)
  --no-cold            skip cold loads (page cache evicted first)
  --stream             also time full core passes with 1 slot and 2 slots
  --report PATH        JSON report (default: bench_report.json)

Per piece it reports: file size, spill-fill size from the binary, cold and warm
load time split into read / create / register, execute time, free time, and the
memory a resident piece costs (RSS, DMA-BUF, MemAvailable deltas).
)";

namespace {

struct Delta {
  int64_t rss = 0;
  int64_t dmabuf = 0;
  int64_t npu_tracked = 0;
  int64_t avail_drop = 0;
};

Delta diff(const MemSnapshot& a, const MemSnapshot& b) {
  Delta d;
  d.rss = static_cast<int64_t>(b.rss) - static_cast<int64_t>(a.rss);
  d.dmabuf = static_cast<int64_t>(b.dmabuf_fd_bytes) - static_cast<int64_t>(a.dmabuf_fd_bytes);
  d.npu_tracked = static_cast<int64_t>(b.npu_tracked) - static_cast<int64_t>(a.npu_tracked);
  d.avail_drop = static_cast<int64_t>(a.mem_available) - static_cast<int64_t>(b.mem_available);
  return d;
}

double smb(int64_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); }

struct Row {
  std::string name;
  uint64_t file_bytes = 0;
  uint64_t spill_fill = 0;
  LoadTiming cold, warm;
  bool has_cold = false;
  double exec_ms = 0;
  double exec_min_ms = 0;
  double free_ms = 0;
  Delta resident;
  Delta residual;
};

std::vector<const PieceSpec*> select(const Manifest& m, const std::string& which) {
  std::vector<const PieceSpec*> out;
  if (which == "all") {
    for (const auto& p : m.pieces) out.push_back(&p);
  } else if (which == "core") {
    out = m.stage(Stage::Core);
  } else if (which == "shell") {
    out = {&m.pieces.front(), &m.pieces.back()};
  } else {
    std::stringstream ss(which);
    std::string name;
    while (std::getline(ss, name, ',')) out.push_back(&m.piece(name));
  }
  return out;
}

std::string timing_json(const LoadTiming& t) {
  std::ostringstream o;
  o << "{\"total_ms\":" << t.total_ms << ",\"read_ms\":" << t.read_ms << ",\"create_ms\":" << t.create_ms
    << ",\"register_ms\":" << t.register_ms << "}";
  return o.str();
}

std::string delta_json(const Delta& d) {
  std::ostringstream o;
  o << "{\"rss\":" << d.rss << ",\"dmabuf\":" << d.dmabuf << ",\"npu_tracked\":" << d.npu_tracked
    << ",\"avail_drop\":" << d.avail_drop << "}";
  return o.str();
}

}

int main(int argc, char** argv) {
  try {
    cli::Args args(argc, argv, std::string(kUsage) + cli::kCommonUsage);
    cli::setup_logging(args);
    const std::string which = args.str("pieces", "all");
    const int reps = static_cast<int>(args.num("reps", 3));
    const bool cold = !args.flag("no-cold");
    const bool stream = args.flag("stream");
    const std::string report_path = args.str("report", "bench_report.json");
    auto rt = cli::open_runtime(args);
    args.finish();
    const Manifest& m = rt->manifest;
    QnnBackend* backend = rt->backend.get();
    const std::vector<const PieceSpec*> pieces = select(m, which);

    std::vector<std::string> missing;
    for (const auto& p : m.pieces)
      if (!file_exists(p.context_path)) missing.push_back(p.context_path);
    if (!missing.empty()) {
      for (const auto& f : missing) DE_ERROR("bench", "missing context binary %s", f.c_str());
      throw std::runtime_error(std::to_string(missing.size()) + " context binaries missing; copy them next to the manifest or pass --contexts");
    }

    std::ostringstream report;
    const QnnDeviceInfo& d = backend->device_info();
    report << "{\"backend\":\"qnn-htp\",\"device\":{\"soc_model\":" << d.soc_model << ",\"arch\":" << d.arch
           << ",\"vtcm_mb\":" << d.vtcm_size << "},\"max_spill_fill\":" << backend->max_spill_fill();

    const PieceSpec& anchor_spec = m.pieces.front();
    std::unique_ptr<LoadedPiece> anchor = backend->load(anchor_spec);
    DE_INFO("bench", "anchor %s pinned in %.1f ms (owns the shared spill-fill buffer)", anchor_spec.name.c_str(),
            anchor->timing.total_ms);

    std::vector<Row> rows;
    for (const PieceSpec* spec : pieces) {
      if (spec == &anchor_spec) continue;
      Row r;
      r.name = spec->name;
      r.spill_fill = backend->spill_fill(spec->name);
      if (cold) {
        evict_page_cache(spec->context_path);
        auto p = backend->load(*spec);
        r.cold = p->timing;
        r.has_cold = true;
        backend->unload(std::move(p));
      }

      prefetch_page_cache(spec->context_path);
      MappedFile(spec->context_path).touch();
      MemSnapshot before = MemAccounting::snapshot();
      auto p = backend->load(*spec);
      MemSnapshot loaded = MemAccounting::snapshot();
      r.warm = p->timing;
      r.file_bytes = p->timing.file_bytes;
      r.resident = diff(before, loaded);

      double total = 0;
      r.exec_min_ms = 1e30;
      for (int i = 0; i < reps; ++i) {
        const double t0 = now_ms();
        backend->execute(*p);
        const double dt = now_ms() - t0;
        total += dt;
        r.exec_min_ms = std::min(r.exec_min_ms, dt);
      }
      r.exec_ms = reps > 0 ? total / reps : 0;
      if (reps == 0) r.exec_min_ms = 0;

      const double f0 = now_ms();
      backend->unload(std::move(p));
      r.free_ms = now_ms() - f0;
      r.residual = diff(before, MemAccounting::snapshot());

      DE_INFO("bench",
              "%-10s file %6.1f MB spill %5.2f MB | cold %7.1f ms (read %6.1f create %7.1f reg %5.1f) | warm %7.1f ms | "
              "exec %7.2f ms (min %7.2f) | free %5.1f ms | resident: rss %+6.1f dmabuf %+6.1f avail %+6.1f MB",
              r.name.c_str(), cli::mb(r.file_bytes), cli::mb(r.spill_fill), r.cold.total_ms, r.cold.read_ms,
              r.cold.create_ms, r.cold.register_ms, r.warm.total_ms, r.exec_ms, r.exec_min_ms, r.free_ms,
              smb(r.resident.rss), smb(r.resident.dmabuf), smb(r.resident.avail_drop));
      rows.push_back(r);
    }

    report << ",\"anchor\":" << json::quote(anchor_spec.name) << ",\"anchor_load\":" << timing_json(anchor->timing)
           << ",\"pieces\":[";
    for (size_t i = 0; i < rows.size(); ++i) {
      const Row& r = rows[i];
      report << (i ? "," : "") << "{\"name\":" << json::quote(r.name) << ",\"file_bytes\":" << r.file_bytes
             << ",\"spill_fill_bytes\":" << r.spill_fill << ",\"cold\":" << (r.has_cold ? timing_json(r.cold) : "null")
             << ",\"warm\":" << timing_json(r.warm) << ",\"exec_ms_mean\":" << r.exec_ms
             << ",\"exec_ms_min\":" << r.exec_min_ms << ",\"free_ms\":" << r.free_ms
             << ",\"resident\":" << delta_json(r.resident) << ",\"after_free\":" << delta_json(r.residual) << "}";
    }
    report << "]";

    if (stream) {
      std::unique_ptr<LoadedPiece> up = backend->load(m.pieces.back());
      report << ",\"stream\":[";
      for (int slots = 1; slots <= 2; ++slots) {
        for (const PieceSpec* c : m.stage(Stage::Core)) prefetch_page_cache(c->context_path);
        MemAccounting mem(2);
        mem.start();
        CoreStreamer streamer(*backend, m.stage(Stage::Core), slots, &mem);
        CorePassStats s = streamer.run_pass();
        mem.stop();
        MemPeaks pk = mem.peaks();
        DE_INFO("bench", "core pass with %d slot(s): %.1f ms (exec %.1f, stall %.1f, max %d resident) | peak rss %.1f MB, "
                "npu %.1f MB, avail drop %.1f MB",
                slots, s.end_ms - s.begin_ms, s.exec_ms, s.stall_ms, s.max_slots_used, cli::mb(pk.cpu_peak()),
                cli::mb(pk.npu_peak()), cli::mb(pk.system_drop()));
        report << (slots > 1 ? "," : "") << "{\"slots\":" << slots << ",\"pass_ms\":" << s.end_ms - s.begin_ms
               << ",\"exec_ms\":" << s.exec_ms << ",\"stall_ms\":" << s.stall_ms << ",\"memory\":" << mem.to_json()
               << "}";
      }
      report << "]";
      backend->unload(std::move(up));
    }
    backend->unload(std::move(anchor));
    report << "}";

    std::string s = report.str();
    write_file(report_path, s.data(), s.size());
    DE_INFO("bench", "report written to %s", report_path.c_str());
    return 0;
  } catch (const std::exception& e) {
    DE_ERROR("bench", "%s", e.what());
    return 1;
  }
}
