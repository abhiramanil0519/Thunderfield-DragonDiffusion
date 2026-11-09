#include <cstdio>
#include <exception>
#include <sstream>

#include "cli.h"
#include "dragonedge/file_io.h"
#include "dragonedge/json.h"
#include "dragonedge/mem_accounting.h"
#include "dragonedge/sampler.h"
#include "dragonedge/unet_runner.h"

using namespace dragonedge;

static const char* kUsage = R"(run_unet: run the SDXL-Lightning UNet on the NPU (all steps full) and report memory.

  --ehs PATH             prompt embeddings, float32 [1,77,2048]   (required)
  --pooled PATH          pooled text embeddings, float32 [1,1280] (required)
  --slots N              core streaming slots (default: 2)
  --steps N              denoising steps (default: 4)
  --seed N               initial-noise seed (default: 0)
  --noise PATH           initial noise, float32 NCHW (overrides --seed)
  --out-latent PATH      final latent as float32 (default: latent.raw)
  --out-image PATH       decoded image (default: image.png)
  --decoder PATH         TAESDXL decoder context (default: <contexts>/taesdxl_decoder.bin)
  --report PATH          JSON report (default: run_report.json)
  --sample-ms N          memory sampler period (default: 5)
)";

int main(int argc, char** argv) {
  try {
    cli::Args args(argc, argv, std::string(kUsage) + cli::kCommonUsage);
    cli::setup_logging(args);
    const int slots = static_cast<int>(args.num("slots", 2));
    const int steps = static_cast<int>(args.num("steps", 4));
    const uint64_t seed = static_cast<uint64_t>(args.num("seed", 0));
    const std::string noise_path = args.str("noise");
    const std::string ehs_path = args.str("ehs");
    const std::string pooled_path = args.str("pooled");
    const std::string out_latent = args.str("out-latent", "latent.raw");
    const std::string out_image = args.str("out-image", "image.png");
    std::string decoder_path = args.str("decoder");
    const std::string report_path = args.str("report", "run_report.json");
    MemAccounting mem(static_cast<int>(args.num("sample-ms", 5)));
    if (ehs_path.empty() || pooled_path.empty()) throw std::runtime_error("--ehs and --pooled are required");

    mem.start();
    auto rt = cli::open_runtime(args);
    args.finish();
    mem.mark("backend ready");
    const Manifest& m = rt->manifest;

    Conditioning cond = zero_conditioning(m);
    cond.ehs = cli::read_f32(ehs_path, cond.ehs.size());
    cond.pooled = cli::read_f32(pooled_path, cond.pooled.size());
    const size_t n = m.tensor("sample").numel();
    std::vector<float> noise(n);
    if (!noise_path.empty()) noise = cli::read_f32(noise_path, n);
    else fill_normal(noise.data(), n, seed, 1.f);

    UNetRunner runner(m, *rt->backend, slots, &mem);
    EulerSchedule sched = euler_trailing(steps);
    DE_INFO("engine", "denoising %d steps (timesteps %.0f..%.0f), %d core slots", steps, sched.timesteps.front(),
            sched.timesteps.back(), slots);
    DenoiseResult res = denoise(runner, *rt->arena, m, sched, cond, noise);
    runner.unload_shell();
    write_file(out_latent, res.latent.data(), res.latent.size() * 4);
    DE_INFO("result", "latent written to %s", out_latent.c_str());

    if (decoder_path.empty()) {
      const std::string& first = m.pieces.front().context_path;
      decoder_path = first.substr(0, first.find_last_of("/\\") + 1) + "taesdxl_decoder.bin";
    }
    const QnnOptions opts = cli::qnn_options(args);
    rt->backend.reset();
    mem.mark("unet released");
    if (file_exists(decoder_path)) {
      cli::decode_to_png(decoder_path, res.latent, m.latent, out_image, opts);
      mem.mark("decoded");
    } else {
      DE_WARN("decode", "no decoder context at %s; skipping image decode", decoder_path.c_str());
    }
    mem.stop();

    double total = 0;
    for (const auto& s : res.steps) total += s.end_ms - s.begin_ms;
    MemPeaks p = mem.peaks();
    DE_INFO("result", "UNet total %.1f ms over %d steps (%.1f ms/step)", total, steps, total / steps);
    DE_INFO("result", "CPU side   : peak RSS %.1f MB (%.1f MB above baseline), kernel high-water %.1f MB",
            cli::mb(p.cpu_peak()), cli::mb(p.cpu_peak_delta()), cli::mb(p.kernel_peak_rss));
    DE_INFO("result", "NPU shared : peak %.1f MB (dmabuf fds %.1f MB, rpcmem tracked %.1f MB)", cli::mb(p.npu_peak()),
            cli::mb(p.peak_dmabuf_fd_bytes), cli::mb(p.peak_npu_tracked));
    DE_INFO("result", "System     : MemAvailable dropped %.1f MB", cli::mb(p.system_drop()));

    std::ostringstream o;
    o << "{\"backend\":\"qnn-htp\",\"slots\":" << slots << ",\"arena_bytes\":" << rt->layout.plan.total_bytes
      << ",\"unet_total_ms\":" << total << ",\"steps\":[";
    for (size_t i = 0; i < res.steps.size(); ++i) {
      const auto& s = res.steps[i];
      o << (i ? "," : "") << "{\"index\":" << s.index << ",\"timestep\":" << sched.timesteps[s.index]
        << ",\"total_ms\":" << s.end_ms - s.begin_ms << ",\"shell_down_ms\":" << s.shell_down_ms
        << ",\"core_ms\":" << s.core_ms << ",\"shell_up_ms\":" << s.shell_up_ms << ",\"stall_ms\":" << s.core.stall_ms
        << ",\"shards\":[";
      for (size_t k = 0; k < s.core.shards.size(); ++k) {
        const auto& e = s.core.shards[k];
        o << (k ? "," : "") << "{\"name\":" << json::quote(e.name) << ",\"load_ms\":" << e.load_end - e.load_begin
          << ",\"read_ms\":" << e.timing.read_ms << ",\"create_ms\":" << e.timing.create_ms
          << ",\"register_ms\":" << e.timing.register_ms << ",\"exec_ms\":" << e.exec_end - e.exec_begin
          << ",\"free_ms\":" << e.unload_end - e.exec_end << ",\"stall_ms\":" << e.stall_ms << "}";
      }
      o << "]}";
    }
    o << "],\"memory\":" << mem.to_json() << "}";
    std::string s = o.str();
    write_file(report_path, s.data(), s.size());
    DE_INFO("result", "report written to %s", report_path.c_str());
    return 0;
  } catch (const std::exception& e) {
    DE_ERROR("run_unet", "%s", e.what());
    return 1;
  }
}
