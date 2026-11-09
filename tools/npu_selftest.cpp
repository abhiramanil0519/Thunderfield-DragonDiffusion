#include <cstdio>
#include <exception>
#include <string>

#include "cli.h"
#include "dragonedge/mem_accounting.h"

using namespace dragonedge;

static const char* kUsage = R"(npu_selftest: check the DragonEdge NPU path on this phone.

Opens the QNN HTP backend, reports the Hexagon architecture and VTCM size,
allocates the planned activation block in rpcmem, builds a 16-bit quantized
ElementWiseAdd graph on the device, finalizes it on the HTP, runs it on
shared buffers (no copies), checks every output element against the CPU,
and registers every planned tensor as a shared buffer.

  --mb N               size of each self-test tensor in MB (default: 16)
  --reps N             timed executions (default: 20)
)";

int main(int argc, char** argv) {
  try {
    cli::Args args(argc, argv, std::string(kUsage) + cli::kCommonUsage);
    cli::setup_logging(args);
    const double size_mb = args.num("mb", 16);
    const int reps = static_cast<int>(args.num("reps", 20));
    MemSnapshot before = MemAccounting::snapshot();
    auto rt = cli::open_runtime(args);
    args.finish();

    const QnnDeviceInfo& d = rt->backend->device_info();
    const uint64_t elems = static_cast<uint64_t>(size_mb * 1048576.0 / 2.0);
    DE_INFO("selftest", "running uint16 ElementWiseAdd on the HTP: 3 x %.1f MB tensors, %d timed runs", size_mb, reps);
    QnnSelfTest r = rt->backend->self_test(elems, reps);
    MemSnapshot after = MemAccounting::snapshot();

    DE_INFO("selftest", "rpcmem alloc %.2f ms | graph build %.2f ms | HTP finalize %.1f ms | memRegister %.2f ms",
            r.alloc_ms, r.build_ms, r.finalize_ms, r.register_ms);
    DE_INFO("selftest", "first execute %.2f ms | steady execute %.3f ms | %.2f GB/s through shared buffers",
            r.first_exec_ms, r.exec_ms, r.gb_per_s);
    DE_INFO("selftest", "result check over %llu elements: max |npu - cpu| = %.6f -> %s",
            static_cast<unsigned long long>(r.elems), r.max_abs_err, r.pass ? "PASS" : "FAIL");
    DE_INFO("selftest", "registered all %u planned arena tensors as shared buffers in %.2f ms", r.arena_slots,
            r.arena_register_ms);

    std::printf("\nDragonEdge NPU self-test\n");
    std::printf("  device        : soc_model %u, Hexagon v%u, VTCM %llu MB%s\n", d.soc_model, d.arch,
                static_cast<unsigned long long>(d.vtcm_size), d.valid ? "" : " (platform info unavailable)");
    std::printf("  arena         : %.2f MB rpcmem, %u tensors registered\n", cli::mb(rt->arena->size()), r.arena_slots);
    std::printf("  HTP finalize  : %.1f ms\n", r.finalize_ms);
    std::printf("  execute       : %.3f ms steady, %.2f GB/s\n", r.exec_ms, r.gb_per_s);
    std::printf("  correctness   : max abs error %.6f over %llu elements -> %s\n", r.max_abs_err,
                static_cast<unsigned long long>(r.elems), r.pass ? "PASS" : "FAIL");
    std::printf("  process RSS   : %.1f MB -> %.1f MB, dmabuf %.1f MB -> %.1f MB\n", cli::mb(before.rss), cli::mb(after.rss),
                cli::mb(before.dmabuf_fd_bytes), cli::mb(after.dmabuf_fd_bytes));
    return r.pass ? 0 : 1;
  } catch (const std::exception& e) {
    DE_ERROR("selftest", "%s", e.what());
    return 1;
  }
}
