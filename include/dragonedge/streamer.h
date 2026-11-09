#pragma once

#include <functional>
#include <string>
#include <vector>

#include "dragonedge/backend.h"
#include "dragonedge/manifest.h"

namespace dragonedge {

class MemAccounting;

struct ShardEvent {
  std::string name;
  double load_begin = 0;
  double load_end = 0;
  double exec_begin = 0;
  double exec_end = 0;
  double unload_end = 0;
  double stall_ms = 0;
  uint64_t dmabuf_loaded = 0;
  uint64_t rss_loaded = 0;
  LoadTiming timing;
};

struct CorePassStats {
  double begin_ms = 0;
  double end_ms = 0;
  double stall_ms = 0;
  double exec_ms = 0;
  int max_slots_used = 0;
  std::vector<ShardEvent> shards;
};

class CoreStreamer {
 public:
  CoreStreamer(Backend& backend, std::vector<const PieceSpec*> shards, int slots = 2, MemAccounting* mem = nullptr);

  CorePassStats run_pass(const std::function<void()>& overlap = {});
  int slots() const { return slots_; }

 private:
  Backend& backend_;
  std::vector<const PieceSpec*> shards_;
  int slots_;
  MemAccounting* mem_;
};

}
