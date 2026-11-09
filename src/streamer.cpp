#include "dragonedge/streamer.h"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "dragonedge/clock.h"
#include "dragonedge/log.h"
#include "dragonedge/mem_accounting.h"

namespace dragonedge {

CoreStreamer::CoreStreamer(Backend& backend, std::vector<const PieceSpec*> shards, int slots, MemAccounting* mem)
    : backend_(backend), shards_(std::move(shards)), slots_(slots), mem_(mem) {
  if (slots_ < 1) throw std::invalid_argument("streamer: need at least one slot");
  if (shards_.empty()) throw std::invalid_argument("streamer: no shards");
}

CorePassStats CoreStreamer::run_pass(const std::function<void()>& overlap) {
  const size_t n = shards_.size();
  CorePassStats stats;
  stats.shards.resize(n);
  stats.begin_ms = now_ms();

  std::vector<std::unique_ptr<LoadedPiece>> ready(n);
  std::mutex mu;
  std::condition_variable cv;
  int free_slots = slots_;
  int used = 0;
  size_t loaded = 0;
  bool abort = false;
  std::exception_ptr load_error;

  std::thread loader([&] {
    for (size_t k = 0; k < n; ++k) {
      {
        std::unique_lock<std::mutex> l(mu);
        cv.wait(l, [&] { return free_slots > 0 || abort; });
        if (abort) return;
        --free_slots;
        stats.max_slots_used = std::max(stats.max_slots_used, ++used);
      }
      ShardEvent& ev = stats.shards[k];
      ev.name = shards_[k]->name;
      ev.load_begin = now_ms();
      std::unique_ptr<LoadedPiece> piece;
      try {
        piece = backend_.load(*shards_[k]);
      } catch (const std::exception& first) {
        bool retry = false;
        {
          std::unique_lock<std::mutex> l(mu);
          if (used > 1) {
            DE_WARN("core", "%s: load failed with %d shards resident (%s); retrying alone and continuing with 1 slot",
                    ev.name.c_str(), used, first.what());
            cv.wait(l, [&] { return used == 1 || abort; });
            if (abort) return;
            free_slots = 0;
            slots_ = 1;
            retry = true;
          }
        }
        std::exception_ptr retry_error;
        if (retry) {
          try {
            ev.load_begin = now_ms();
            piece = backend_.load(*shards_[k]);
          } catch (...) {
            retry_error = std::current_exception();
            retry = false;
          }
        }
        if (!retry) {
          std::lock_guard<std::mutex> l(mu);
          load_error = retry_error ? retry_error : std::current_exception();
          abort = true;
          cv.notify_all();
          return;
        }
      } catch (...) {
        std::lock_guard<std::mutex> l(mu);
        load_error = std::current_exception();
        abort = true;
        cv.notify_all();
        return;
      }
      ev.load_end = now_ms();
      ev.timing = piece->timing;
      if (mem_) {
        MemSnapshot s = mem_->mark("loaded " + ev.name);
        ev.dmabuf_loaded = s.dmabuf_fd_bytes;
        ev.rss_loaded = s.rss;
      }
      {
        std::lock_guard<std::mutex> l(mu);
        ready[k] = std::move(piece);
        loaded = k + 1;
      }
      cv.notify_all();
    }
  });

  auto shutdown = [&] {
    {
      std::lock_guard<std::mutex> l(mu);
      abort = true;
    }
    cv.notify_all();
    loader.join();
    for (auto& p : ready)
      if (p) backend_.unload(std::move(p));
  };

  try {
    if (overlap) overlap();
    for (size_t k = 0; k < n; ++k) {
      std::unique_ptr<LoadedPiece> piece;
      const double wait0 = now_ms();
      {
        std::unique_lock<std::mutex> l(mu);
        cv.wait(l, [&] { return loaded > k || load_error; });
        if (loaded <= k) break;
        piece = std::move(ready[k]);
      }
      ShardEvent& ev = stats.shards[k];
      ev.stall_ms = now_ms() - wait0;
      stats.stall_ms += ev.stall_ms;
      ev.exec_begin = now_ms();
      backend_.execute(*piece);
      ev.exec_end = now_ms();
      stats.exec_ms += ev.exec_end - ev.exec_begin;
      backend_.unload(std::move(piece));
      ev.unload_end = now_ms();
      if (mem_) mem_->mark("freed " + ev.name);
      DE_INFO("core", "%s load %7.1f ms (read %6.1f create %7.1f register %5.1f) | stall %6.1f | exec %7.1f | free %5.1f ms"
              " | at load: dmabuf %.0f MB rss %.0f MB",
              ev.name.c_str(), ev.load_end - ev.load_begin, ev.timing.read_ms, ev.timing.create_ms,
              ev.timing.register_ms, ev.stall_ms, ev.exec_end - ev.exec_begin, ev.unload_end - ev.exec_end,
              ev.dmabuf_loaded / 1048576.0, ev.rss_loaded / 1048576.0);
      {
        std::lock_guard<std::mutex> l(mu);
        ++free_slots;
        --used;
      }
      cv.notify_all();
    }
  } catch (...) {
    shutdown();
    throw;
  }
  shutdown();
  if (load_error) std::rethrow_exception(load_error);
  stats.end_ms = now_ms();
  return stats;
}

}
