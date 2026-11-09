#include "dragonedge/mem_accounting.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <sstream>

#include "dragonedge/clock.h"
#include "dragonedge/json.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <dirent.h>
#include <unistd.h>
#endif

namespace dragonedge {

std::atomic<int64_t>& npu_tracked_bytes() {
  static std::atomic<int64_t> bytes{0};
  return bytes;
}

#if defined(__linux__)
static uint64_t kb_field(const char* text, const char* key) {
  const char* p = std::strstr(text, key);
  if (!p) return 0;
  p += std::strlen(key);
  unsigned long long v = 0;
  if (std::sscanf(p, " %llu", &v) != 1) return 0;
  return static_cast<uint64_t>(v) * 1024;
}

static std::string slurp(const char* path) {
  FILE* f = std::fopen(path, "r");
  if (!f) return {};
  std::string out;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
  std::fclose(f);
  return out;
}

static uint64_t dmabuf_fd_bytes() {
  DIR* d = opendir("/proc/self/fd");
  if (!d) return 0;
  std::set<unsigned long long> seen;
  uint64_t total = 0;
  while (dirent* e = readdir(d)) {
    if (e->d_name[0] == '.') continue;
    char link[64], target[256];
    std::snprintf(link, sizeof(link), "/proc/self/fd/%s", e->d_name);
    ssize_t n = readlink(link, target, sizeof(target) - 1);
    if (n <= 0) continue;
    target[n] = 0;
    if (!std::strstr(target, "dmabuf")) continue;
    char info_path[64];
    std::snprintf(info_path, sizeof(info_path), "/proc/self/fdinfo/%s", e->d_name);
    std::string info = slurp(info_path);
    const char* sz = std::strstr(info.c_str(), "size:");
    const char* ino = std::strstr(info.c_str(), "ino:");
    unsigned long long size = 0, inode = 0;
    if (!sz || std::sscanf(sz + 5, " %llu", &size) != 1) continue;
    if (ino) std::sscanf(ino + 4, " %llu", &inode);
    if (inode && !seen.insert(inode).second) continue;
    total += size;
  }
  closedir(d);
  return total;
}
#endif

MemSnapshot MemAccounting::snapshot() {
  MemSnapshot s;
  s.t_ms = now_ms();
  s.npu_tracked = static_cast<uint64_t>(std::max<int64_t>(0, npu_tracked_bytes().load()));
#if defined(__linux__)
  std::string status = slurp("/proc/self/status");
  s.rss = kb_field(status.c_str(), "VmRSS:");
  s.kernel_peak_rss = kb_field(status.c_str(), "VmHWM:");
  s.rss_anon = kb_field(status.c_str(), "RssAnon:");
  s.rss_file = kb_field(status.c_str(), "RssFile:");
  s.rss_shmem = kb_field(status.c_str(), "RssShmem:");
  std::string meminfo = slurp("/proc/meminfo");
  s.mem_available = kb_field(meminfo.c_str(), "MemAvailable:");
  s.dmabuf_fd_bytes = dmabuf_fd_bytes();
#elif defined(_WIN32)
  PROCESS_MEMORY_COUNTERS_EX pmc{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
    s.rss = pmc.WorkingSetSize;
    s.kernel_peak_rss = pmc.PeakWorkingSetSize;
    s.rss_anon = pmc.PrivateUsage;
  }
  MEMORYSTATUSEX ms{};
  ms.dwLength = sizeof(ms);
  if (GlobalMemoryStatusEx(&ms)) s.mem_available = ms.ullAvailPhys;
#endif
  return s;
}

MemAccounting::MemAccounting(int period_ms) : period_ms_(period_ms) {}

MemAccounting::~MemAccounting() { stop(); }

void MemAccounting::absorb(const MemSnapshot& s) {
  peaks_.peak_rss = std::max(peaks_.peak_rss, s.rss);
  peaks_.peak_rss_anon = std::max(peaks_.peak_rss_anon, s.rss_anon);
  peaks_.kernel_peak_rss = std::max(peaks_.kernel_peak_rss, s.kernel_peak_rss);
  peaks_.peak_npu_tracked = std::max(peaks_.peak_npu_tracked, s.npu_tracked);
  peaks_.peak_dmabuf_fd_bytes = std::max(peaks_.peak_dmabuf_fd_bytes, s.dmabuf_fd_bytes);
  peaks_.min_mem_available = std::min(peaks_.min_mem_available, s.mem_available);
  ++peaks_.samples;
}

void MemAccounting::start() {
  stop();
  MemSnapshot base = snapshot();
  {
    std::lock_guard<std::mutex> l(mu_);
    peaks_ = MemPeaks{};
    peaks_.baseline = base;
    peaks_.min_mem_available = std::numeric_limits<uint64_t>::max();
    marks_.clear();
    absorb(base);
    marks_.push_back({"baseline", base});
  }
  running_ = true;
  sampler_ = std::thread([this] {
    while (running_) {
      MemSnapshot s = snapshot();
      {
        std::lock_guard<std::mutex> l(mu_);
        absorb(s);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(period_ms_));
    }
  });
}

void MemAccounting::stop() {
  if (!running_.exchange(false)) return;
  if (sampler_.joinable()) sampler_.join();
  mark("end");
}

MemSnapshot MemAccounting::mark(const std::string& label) {
  MemSnapshot s = snapshot();
  std::lock_guard<std::mutex> l(mu_);
  absorb(s);
  marks_.push_back({label, s});
  return s;
}

MemPeaks MemAccounting::peaks() const {
  std::lock_guard<std::mutex> l(mu_);
  return peaks_;
}

std::vector<MemMark> MemAccounting::marks() const {
  std::lock_guard<std::mutex> l(mu_);
  return marks_;
}

static std::string snap_json(const MemSnapshot& s) {
  std::ostringstream o;
  o << "{\"t_ms\":" << s.t_ms << ",\"rss\":" << s.rss << ",\"rss_anon\":" << s.rss_anon
    << ",\"rss_file\":" << s.rss_file << ",\"rss_shmem\":" << s.rss_shmem << ",\"kernel_peak_rss\":" << s.kernel_peak_rss
    << ",\"npu_tracked\":" << s.npu_tracked << ",\"dmabuf_fd_bytes\":" << s.dmabuf_fd_bytes
    << ",\"mem_available\":" << s.mem_available << "}";
  return o.str();
}

std::string MemAccounting::to_json() const {
  MemPeaks p = peaks();
  std::vector<MemMark> ms = marks();
  std::ostringstream o;
  o << "{\"cpu_peak_rss\":" << p.cpu_peak() << ",\"cpu_peak_rss_delta\":" << p.cpu_peak_delta()
    << ",\"cpu_peak_rss_anon\":" << p.peak_rss_anon << ",\"kernel_peak_rss\":" << p.kernel_peak_rss
    << ",\"npu_peak\":" << p.npu_peak() << ",\"npu_peak_tracked\":" << p.peak_npu_tracked
    << ",\"npu_peak_dmabuf_fd\":" << p.peak_dmabuf_fd_bytes << ",\"system_available_drop\":" << p.system_drop()
    << ",\"samples\":" << p.samples << ",\"baseline\":" << snap_json(p.baseline) << ",\"marks\":[";
  for (size_t i = 0; i < ms.size(); ++i)
    o << (i ? "," : "") << "{\"label\":" << json::quote(ms[i].label) << ",\"snap\":" << snap_json(ms[i].snap) << "}";
  o << "]}";
  return o.str();
}

}
