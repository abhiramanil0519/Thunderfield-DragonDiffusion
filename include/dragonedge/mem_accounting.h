#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace dragonedge {

std::atomic<int64_t>& npu_tracked_bytes();

struct MemSnapshot {
  double t_ms = 0;
  uint64_t rss = 0;
  uint64_t rss_anon = 0;
  uint64_t rss_file = 0;
  uint64_t rss_shmem = 0;
  uint64_t kernel_peak_rss = 0;
  uint64_t npu_tracked = 0;
  uint64_t dmabuf_fd_bytes = 0;
  uint64_t mem_available = 0;
};

struct MemMark {
  std::string label;
  MemSnapshot snap;
};

struct MemPeaks {
  MemSnapshot baseline;
  uint64_t peak_rss = 0;
  uint64_t peak_rss_anon = 0;
  uint64_t kernel_peak_rss = 0;
  uint64_t peak_npu_tracked = 0;
  uint64_t peak_dmabuf_fd_bytes = 0;
  uint64_t min_mem_available = 0;
  uint64_t samples = 0;

  uint64_t cpu_peak() const { return peak_rss; }
  uint64_t cpu_peak_delta() const { return peak_rss > baseline.rss ? peak_rss - baseline.rss : 0; }
  uint64_t npu_peak() const { return peak_dmabuf_fd_bytes > peak_npu_tracked ? peak_dmabuf_fd_bytes : peak_npu_tracked; }
  uint64_t system_drop() const {
    return baseline.mem_available > min_mem_available ? baseline.mem_available - min_mem_available : 0;
  }
};

class MemAccounting {
 public:
  explicit MemAccounting(int period_ms = 5);
  ~MemAccounting();
  MemAccounting(const MemAccounting&) = delete;
  MemAccounting& operator=(const MemAccounting&) = delete;

  static MemSnapshot snapshot();

  void start();
  void stop();
  MemSnapshot mark(const std::string& label);
  MemPeaks peaks() const;
  std::vector<MemMark> marks() const;
  std::string to_json() const;

 private:
  void absorb(const MemSnapshot& s);

  int period_ms_;
  mutable std::mutex mu_;
  MemPeaks peaks_;
  std::vector<MemMark> marks_;
  std::atomic<bool> running_{false};
  std::thread sampler_;
};

}
